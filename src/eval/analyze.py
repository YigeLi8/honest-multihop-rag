"""Decompose the illusion cell and compare configs with paired tests.

    python -m src.eval.analyze --dataset hotpot \
        --runs bm25_hotpot_dev hybrid_ircot_hotpot_dev hybrid_ircot_rerank_hotpot_dev \
        --closed-book closed_book_hotpot_dev

Every run must be scored on the same questions (same dataset, same --n, same
order), which is how run_qa works, so rows join on question id.

The illusion cell (answer right, gold evidence incomplete) gets split into
buckets that each name a different reason the answer could be right:

    parametric   the closed-book run got it right too, so the evidence was
                 never needed
    paragraph    every gold paragraph reached the model, just not every
                 annotated gold sentence: the strict sentence-level label
                 calls this incomplete, a neighbouring sentence of the same
                 paragraph may well carry the fact
    partial      some gold in context but at least one gold paragraph never
                 made it (the model saw part of the chain)
    none         no gold sentence in context and closed-book wrong: the
                 answer came from non-gold context or a guess

Yes/no answers are reported inside each bucket because a wrong-evidence yes/no
is right half the time by chance.

Writes results/<dataset>_decomposition.csv, results/<dataset>_paired.csv and
results/<dataset>_decomposition.png.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.stats import mcnemar, paired_bootstrap, wilson_ci


def load_run(results_dir, name, gold_by_id=None):
    df = pd.read_csv(Path(results_dir) / f"{name}_qa.csv").set_index("id")
    if "gold_recall" not in df:
        raise SystemExit(f"{name}: no gold_recall column, re-run it with the current run_qa")
    if "para_recall" not in df:
        # older runs of the current harness: rebuild it from the trace file
        from src.eval.run_qa import paragraph_recall
        trace = Path(results_dir) / f"{name}_trace.jsonl"
        if not trace.exists() or gold_by_id is None:
            raise SystemExit(f"{name}: no para_recall column and no trace to rebuild it from")
        rec = {}
        with open(trace) as f:
            for line in f:
                t = json.loads(line)
                got = {c for h in t["hops"] for c in h["retrieved"]}
                rec[t["id"]] = paragraph_recall(got, gold_by_id.get(t["id"], []))[2]
        df["para_recall"] = df.index.map(rec)
    return df


def load_gold(data_file):
    out = {}
    with open(data_file) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = d["gold_chunk_ids"]
    return out


def bucket(row):
    if not (row.em == 1 and row.retrieval_correct == 0):
        return ""
    if row.cb_em == 1:
        return "parametric"
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
           "illusion": round(k / n, 4), "illusion_lo": round(lo, 4), "illusion_hi": round(hi, 4),
           "illusion_share_of_correct": round(k / max(1, int(df.em.sum())), 4)}
    out["para_complete"] = round((df.para_recall >= 1.0).mean(), 4)
    for b in ("parametric", "paragraph", "partial", "none"):
        sub = ill[ill.bucket == b]
        out[b] = len(sub)
        out[f"{b}_rate"] = round(len(sub) / n, 4)
        out[f"{b}_yesno"] = int((sub.answer_type == "yesno").sum())
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
    for metric in ("em", "retrieval_correct", "illusion"):
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

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["run"] for r in table]
    bottom = [0.0] * len(table)
    for b, color in (("parametric", "#7f7f7f"), ("paragraph", "#9ecae1"),
                     ("partial", "#1f77b4"), ("none", "#d62728")):
        vals = [r[f"{b}_rate"] for r in table]
        ax.bar(names, vals, bottom=bottom, label=b, color=color)
        bottom = [x + y for x, y in zip(bottom, vals)]
    for i, r in enumerate(table):
        ax.errorbar(i, r["illusion"], yerr=[[r["illusion"] - r["illusion_lo"]],
                                            [r["illusion_hi"] - r["illusion"]]],
                    fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("share of questions: answer right, gold evidence incomplete")
    ax.set_title(f"what the illusion cell is made of ({dataset})")
    ax.legend(title="bucket")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label for the output files")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--closed-book", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-file", default=None,
                    help="processed jsonl, only needed to rebuild para_recall for older runs")
    args = ap.parse_args()

    gold = load_gold(args.data_file) if args.data_file else None
    cb = load_run(args.results_dir, args.closed_book, gold)
    runs = {}
    for name in args.runs:
        df = load_run(args.results_dir, name, gold)
        df = df.loc[df.index.intersection(cb.index)]
        df["cb_em"] = cb.em.reindex(df.index)
        df["illusion"] = ((df.em == 1) & (df.retrieval_correct == 0)).astype(int)
        df["bucket"] = df.apply(bucket, axis=1)
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

    cols = ["run", "em", "closed_book_em", "retrieval_correct", "para_complete", "illusion",
            "parametric", "paragraph", "partial", "none", "em_given_all_gold",
            "em_given_all_para", "em_given_partial_gold", "em_given_no_gold"]
    print(pd.DataFrame(table)[cols].to_string(index=False))
    print()
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
