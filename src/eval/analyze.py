"""Decompose the illusion cell and compare configs with paired tests.

    python -m src.eval.analyze --dataset hotpot \
        --runs bm25_hotpot_dev hybrid_ircot_hotpot_dev hybrid_ircot_rerank_hotpot_dev \
        --closed-book closed_book_hotpot_dev --data-file data/processed/hotpotqa_dev.jsonl

Every run must be scored on the same questions (same dataset, same --n, same
order), which is how run_qa works, so rows join on question id. The trace
files are needed too: the buckets below look at which sentences were in
context, not just how many.

The illusion cell (answer right, gold evidence incomplete) gets split into
buckets, assigned in this order, each naming a different reason the answer
could still be right:

    parametric   the closed-book run got it right too, so the evidence was
                 never needed
    yesno        a yes/no answer; right half the time by chance whatever the
                 context, so it says little about evidence use
    shortcut     the gold sentence that contains the answer span was
                 retrieved, the rest of the chain (the bridge fact) was not:
                 the question plus the answer-bearing sentence sufficed
    paragraph    every gold paragraph reached the model, just not every
                 annotated gold sentence, and the answer sentence was not
                 among them: a neighbouring sentence likely carried the fact
    partial      some gold in context, at least one gold paragraph never
                 made it, no answer-bearing sentence
    none         no gold sentence in context and closed-book wrong: the
                 answer came from non-gold context or a guess

Writes results/<dataset>_decomposition.csv, results/<dataset>_paired.csv and
results/<dataset>_decomposition.png.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.answer_metrics import normalize
from src.eval.run_qa import paragraph_recall
from src.eval.stats import mcnemar, paired_bootstrap, wilson_ci

BUCKETS = ("parametric", "yesno", "shortcut", "paragraph", "partial", "none")
COLORS = {"parametric": "#7f7f7f", "yesno": "#c7c7c7", "shortcut": "#ff7f0e",
          "paragraph": "#9ecae1", "partial": "#1f77b4", "none": "#d62728"}


def load_data(path):
    """id -> {gold, answer, text: {chunk_id: text}}"""
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = {"gold": set(d["gold_chunk_ids"]), "answer": d["answer"],
                            "aliases": d.get("answer_aliases", []),
                            "text": {c["chunk_id"]: c["text"] for c in d["chunks"]}}
    return out


def load_trace(results_dir, name):
    out = {}
    p = Path(results_dir) / f"{name}_trace.jsonl"
    if not p.exists():
        raise SystemExit(f"{name}: no trace file, re-run it with the current run_qa")
    with open(p) as f:
        for line in f:
            t = json.loads(line)
            out[t["id"]] = {c for h in t["hops"] for c in h["retrieved"]}
    return out


def answer_in(context_ids, data, gold_side):
    """Does a normalized gold answer (or alias) occur in a retrieved sentence
    on the gold (True) or non-gold (False) side? None for yes/no answers."""
    answers = [normalize(a) for a in [data["answer"]] + list(data["aliases"]) if a]
    if any(a in ("yes", "no") for a in answers):
        return None
    for c in context_ids:
        if (c in data["gold"]) != gold_side:
            continue
        s = normalize(data["text"].get(c, ""))
        if any(a and a in s for a in answers):
            return True
    return False


def load_run(results_dir, name, data):
    df = pd.read_csv(Path(results_dir) / f"{name}_qa.csv").set_index("id")
    if "gold_recall" not in df:
        raise SystemExit(f"{name}: no gold_recall column, re-run it with the current run_qa")
    trace = load_trace(results_dir, name)
    df = df.loc[[i for i in df.index if i in trace and i in data]]
    df["para_recall"] = [paragraph_recall(trace[i], data[i]["gold"])[2] for i in df.index]
    df["ans_in_gold_ctx"] = [answer_in(trace[i], data[i], True) for i in df.index]
    df["ans_in_nongold_ctx"] = [answer_in(trace[i], data[i], False) for i in df.index]
    return df


def bucket(row):
    if not (row.em == 1 and row.retrieval_correct == 0):
        return ""
    if row.cb_em == 1:
        return "parametric"
    if row.answer_type == "yesno":
        return "yesno"
    if row.ans_in_gold_ctx:
        return "shortcut"
    if row.para_recall >= 1.0:
        return "paragraph"
    if row.gold_recall > 0:
        return "partial"
    return "none"


def decompose(df, name):
    n = len(df)
    ill = df[df.bucket != ""]
    k = len(ill)
    lo, hi = wilson_ci(k, n)
    out = {"run": name, "n": n,
           "em": round(df.em.mean(), 4), "f1": round(df.f1.mean(), 4),
           "closed_book_em": round(df.cb_em.mean(), 4),
           "retrieval_correct": round(df.retrieval_correct.mean(), 4),
           "para_complete": round((df.para_recall >= 1.0).mean(), 4),
           "illusion": round(k / n, 4), "illusion_lo": round(lo, 4), "illusion_hi": round(hi, 4),
           "illusion_share_of_correct": round(k / max(1, int(df.em.sum())), 4)}
    for b in BUCKETS:
        sub = ill[ill.bucket == b]
        out[b] = len(sub)
        out[f"{b}_rate"] = round(len(sub) / n, 4)
    # the strict residual: right answer, not from memory, not yes/no, no
    # answer-bearing gold sentence in context
    out["unexplained"] = int(sum(out[b] for b in ("paragraph", "partial", "none")))
    out["unexplained_rate"] = round(out["unexplained"] / n, 4)
    # em conditional on how much gold was in context
    for lab, mask in (("em_given_all_gold", df.gold_recall >= 1.0),
                      ("em_given_all_para", (df.para_recall >= 1.0) & (df.gold_recall < 1.0)),
                      ("em_given_partial_gold", (df.para_recall < 1.0) & (df.gold_recall > 0)),
                      ("em_given_no_gold", df.gold_recall <= 0)):
        out[lab] = round(df[mask].em.mean(), 4) if mask.any() else float("nan")
        out[lab.replace("em_given", "n")] = int(mask.sum())
    return out


def paired(runs, a, b):
    da, db = runs[a], runs[b]
    rows = []
    for metric in ("em", "retrieval_correct", "illusion", "unexplained"):
        x, y = da[metric].tolist(), db[metric].tolist()
        only_a, only_b, p = mcnemar(x, y)
        diff, lo, hi = paired_bootstrap(x, y)
        rows.append({"a": a, "b": b, "metric": metric, "mean_a": round(sum(x) / len(x), 4),
                     "mean_b": round(sum(y) / len(y), 4), "diff": round(diff, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "only_a": only_a, "only_b": only_b, "mcnemar_p": round(p, 5)})
    x, y = da.f1.tolist(), db.f1.tolist()
    diff, lo, hi = paired_bootstrap(x, y)
    rows.append({"a": a, "b": b, "metric": "f1", "mean_a": round(sum(x) / len(x), 4),
                 "mean_b": round(sum(y) / len(y), 4), "diff": round(diff, 4),
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                 "only_a": "", "only_b": "", "mcnemar_p": ""})
    return rows


def plot(table, out_path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [r["run"] for r in table]
    bottom = [0.0] * len(table)
    for b in BUCKETS:
        vals = [r[f"{b}_rate"] for r in table]
        if not any(vals):
            continue
        ax.bar(names, vals, bottom=bottom, label=b, color=COLORS[b])
        bottom = [x + y for x, y in zip(bottom, vals)]
    for i, r in enumerate(table):
        ax.errorbar(i, r["illusion"], yerr=[[r["illusion"] - r["illusion_lo"]],
                                            [r["illusion_hi"] - r["illusion"]]],
                    fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("share of questions: answer right, gold evidence incomplete")
    ax.set_title(f"what the illusion cell is made of ({dataset})")
    ax.legend(title="bucket", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label for the output files")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--closed-book", required=True)
    ap.add_argument("--data-file", required=True, help="processed jsonl the runs were scored on")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    data = load_data(args.data_file)
    cb = pd.read_csv(Path(args.results_dir) / f"{args.closed_book}_qa.csv").set_index("id")
    runs = {}
    for name in args.runs:
        df = load_run(args.results_dir, name, data)
        df = df.loc[df.index.intersection(cb.index)]
        df["cb_em"] = cb.em.reindex(df.index)
        df["illusion"] = ((df.em == 1) & (df.retrieval_correct == 0)).astype(int)
        df["bucket"] = df.apply(bucket, axis=1)
        df["unexplained"] = df.bucket.isin(("paragraph", "partial", "none")).astype(int)
        runs[name] = df
    ids = set.intersection(*(set(df.index) for df in runs.values()))
    for name in runs:
        runs[name] = runs[name].loc[sorted(ids)]
    print(f"{len(ids)} questions common to all runs")

    table = [decompose(runs[name], name) for name in args.runs]
    out = Path(args.results_dir)
    pd.DataFrame(table).to_csv(out / f"{args.dataset}_decomposition.csv", index=False)

    pairs = []
    for i in range(len(args.runs)):
        for j in range(i + 1, len(args.runs)):
            pairs += paired(runs, args.runs[i], args.runs[j])
    pd.DataFrame(pairs).to_csv(out / f"{args.dataset}_paired.csv", index=False)
    plot(table, out / f"{args.dataset}_decomposition.png", args.dataset)

    cols = ["run", "em", "closed_book_em", "retrieval_correct", "illusion", *BUCKETS,
            "unexplained", "em_given_all_gold", "em_given_partial_gold", "em_given_no_gold"]
    print(pd.DataFrame(table)[cols].to_string(index=False))
    if pairs:
        print()
        print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
