"""Stage 5c gate: what the reader is handed, not what is retrieved.
(docs/plan.md, Part B)

The intervention arms (src/eval/interventions.py) and the ircot k=20 run say
that answers are lost after retrieval: on hotpot questions whose ircot
context already held every gold sentence, the gold sentences alone answer
better than the whole context (+0.113 EM, paired). Before a memory decides
which retrieved chunks to hand over, ask whether anything short of the
oracle recovers part of that, at the same context size:

    full      the run's own answer context, rebuilt from its trace (the
              union of the hops' hits in first-seen order, as the pipeline
              builds it) and answered again; greedy decoding, so it should
              match the run's answers, which is checked
    gold      the gold chunks inside that context, in context order (oracle)
    rerank3   the cross-encoder's top 3 of the context against the question
    rerank5   its top 5 (the reranker's top_n in configs/base.yaml)
    learned3  a chunk filter fit on gold labels of the other half of the
              questions (seeded halves, both ways): logistic over gold-free
              features of the chunk in this trajectory, top 3 kept
    random3   3 chunks drawn at random from the context (seeded)

Every kept set is listed in context order and answered with the one answer
prompt, so two arms over a question differ only in which chunks are there.
Three chunks is the mean gold size rounded up (2.4 on the hotpot fold).
The learned filter is the "filter trained on the same gold" baseline a
memory has to beat; it is fit once per half, not online. With --train-trace
it is instead fit once on another fold's contexts (the exploration fold) and
applied to these questions, so nothing is fit on the fold being scored. Its labels are
gold, its features never are (src/memory/features.py's rule).

    python -m src.memory.reader_context run --config configs/hybrid_ircot_k20.yaml \\
        --trace results/hybrid_ircot_k20_hotpot_dev_trace.jsonl \\
        --ids results/folds/hotpot_explore.txt --name reader_ctx_hotpot_explore [--dry-run]
    python -m src.memory.reader_context run --config configs/hybrid_ircot_k20.yaml \
        --trace results/hybrid_ircot_k20_hotpot_confirm_trace.jsonl --ids results/folds/hotpot_confirm.txt \
        --train-trace results/hybrid_ircot_k20_hotpot_dev_trace.jsonl --train-ids results/folds/hotpot_explore.txt \
        --name reader_ctx_hotpot_confirm
    python -m src.memory.reader_context summarize --name reader_ctx_hotpot_explore \\
        --run-qa results/hybrid_ircot_k20_hotpot_dev_qa.csv
"""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

ARMS = ("full", "gold", "rerank3", "rerank5", "learned3", "random3")
FEATURES = ("ce_score", "ce_rank", "ctx_pos", "first_hop", "first_rank", "n_hops_hit",
            "title_in_question", "question_overlap", "n_ctx")
_WORD = re.compile(r"[a-z0-9]+")


def answer_context(trace):
    """Chunk ids in the order the pipeline put them in the answer prompt, with
    where each was first seen and how many hops returned it."""
    seen = {}
    for h in trace["hops"]:
        for rank, cid in enumerate(h["retrieved"], 1):
            if cid not in seen:
                seen[cid] = {"first_hop": h["hop"], "first_rank": rank, "n_hops_hit": 0}
            seen[cid]["n_hops_hit"] += 1
    return seen


def chunk_features(question, chunks, info, ce_scores):
    """One gold-free feature row per chunk of the context, in context order."""
    q = set(_WORD.findall(question.lower()))
    order = sorted(range(len(chunks)), key=lambda i: -ce_scores[i])
    ce_rank = {i: r for r, i in enumerate(order, 1)}
    rows = []
    for i, c in enumerate(chunks):
        words = set(_WORD.findall(c.text.lower()))
        title = c.title or c.chunk_id.rsplit("::", 1)[0]
        d = info[c.chunk_id]
        rows.append({"ce_score": float(ce_scores[i]), "ce_rank": ce_rank[i], "ctx_pos": i,
                     "first_hop": d["first_hop"], "first_rank": d["first_rank"],
                     "n_hops_hit": d["n_hops_hit"],
                     "title_in_question": int(bool(title) and title.lower() in question.lower()),
                     "question_overlap": len(q & words) / max(1, len(q)),
                     "n_ctx": len(chunks)})
    return rows


def keep_top(scores, n):
    """Indices of the n highest scores, returned in context order."""
    top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:n]
    return sorted(top)


def _fit(items):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[r[f] for f in FEATURES] for it in items for r in it["feats"]], float)
    y = np.array([g for it in items for g in it["is_gold"]])
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def _score(clf, items):
    import numpy as np
    return {it["id"]: clf.predict_proba(np.array([[r[f] for f in FEATURES] for r in it["feats"]],
                                                 float))[:, 1].tolist() for it in items}


def learned_scores(items, seed=13, train_items=None):
    """Held-out probability that a chunk is gold: fit on train_items when
    given, else questions split in halves by a seeded shuffle and a logistic
    fit on one half scores the other."""
    if train_items is not None:
        if {it["id"] for it in train_items} & {it["id"] for it in items}:
            raise SystemExit("train and scored questions overlap")
        return _score(_fit(train_items), items)
    qids = sorted({it["id"] for it in items})
    random.Random(seed).shuffle(qids)
    half = set(qids[: len(qids) // 2])
    out = {}
    for train_half in (True, False):
        tr = [it for it in items if (it["id"] in half) == train_half]
        te = [it for it in items if (it["id"] in half) != train_half]
        out.update(_score(_fit(tr), te))
    return out


def select(items, learned, seed=13):
    """arm -> kept indices into the context, per question."""
    rng = random.Random(seed)
    sel = {}
    for it in items:
        n = len(it["chunks"])
        ce = [r["ce_score"] for r in it["feats"]]
        s = {"full": list(range(n)),
             "rerank3": keep_top(ce, 3), "rerank5": keep_top(ce, 5),
             "learned3": keep_top(learned[it["id"]], 3),
             "random3": sorted(rng.sample(range(n), min(3, n)))}
        g = [i for i, x in enumerate(it["is_gold"]) if x]
        if g:
            s["gold"] = g
        sel[it["id"]] = s
    return sel


def build(cfg, examples, traces):
    """Context, features and gold flags per question; cross-encoder on MPS."""
    from src.retrieval.rerank import Reranker
    rr = Reranker(cfg)
    rr._load()
    items = []
    for ex in examples:
        tr = traces[ex.id]
        info = answer_context(tr)
        by_id = {}
        for c in ex.chunks:
            by_id.setdefault(c.chunk_id, c)
        chunks = [by_id[cid] for cid in info]
        ce = rr.model.predict([(ex.question, c.text) for c in chunks]).tolist()
        gold = set(ex.gold_chunk_ids)
        items.append({"id": ex.id, "ex": ex, "chunks": chunks, "answer_pred": tr["answer_pred"],
                      "feats": chunk_features(ex.question, chunks, info, ce),
                      "is_gold": [int(c.chunk_id in gold) for c in chunks]})
    return items


def cmd_run(args):
    from src.config import load_config
    from src.eval.run_qa import best_em_f1, load_examples, read_ids
    from src.pipeline.multihop import PROMPT, first_line

    cfg = load_config(args.config)
    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    with open(args.trace) as f:
        traces = {t["id"]: t for t in map(json.loads, f)}
    ids = read_ids(args.ids) if args.ids else list(traces)
    examples = load_examples(data_file, None, ids)
    items = build(cfg, examples, traces)
    train_items = None
    if args.train_trace:
        with open(args.train_trace) as f:
            tr_traces = {t["id"]: t for t in map(json.loads, f)}
        tr_ids = read_ids(args.train_ids) if args.train_ids else list(tr_traces)
        train_items = build(cfg, load_examples(data_file, None, tr_ids), tr_traces)
        print(f"chunk filter fit on {len(train_items)} questions from {args.train_trace}")
    sel = select(items, learned_scores(items, train_items=train_items))
    print(f"{len(items)} questions from {data_file}; context {sum(len(i['chunks']) for i in items) / len(items):.1f} "
          f"chunks per question, gold {sum(sum(i['is_gold']) for i in items) / len(items):.2f}")
    for arm in ARMS:
        have = [it for it in items if arm in sel[it["id"]]]
        n_gold = [sum(it["is_gold"]) for it in have]
        kept_all = [int(sum(it["is_gold"][j] for j in sel[it["id"]][arm]) == g)
                    for it, g in zip(have, n_gold)]
        size = sum(len(sel[it["id"]][arm]) for it in have) / max(1, len(have))
        print(f"  {arm:<9} n={len(have)}  chunks {size:.1f}  keeps every gold chunk of the context "
              f"{sum(kept_all) / max(1, len(have)):.3f}")
    if args.dry_run:
        return

    from src.pipeline.generate import Generator
    gen = Generator(cfg)
    out_dir = Path(cfg.paths.results_dir)
    rows = []
    with open(out_dir / f"{args.name}_trace.jsonl", "w") as tf:
        for k, it in enumerate(items):
            ex = it["ex"]
            for arm in ARMS:
                if arm not in sel[it["id"]]:
                    continue
                kept = [it["chunks"][j] for j in sel[it["id"]][arm]]
                ctx = "\n".join(f"- {c.text}" for c in kept)
                t0 = time.perf_counter()
                text, m = gen.generate(PROMPT.format(context=ctx, question=ex.question))
                pred = first_line(text)
                em_, f1_ = best_em_f1(pred, ex)
                g = sum(it["is_gold"])
                kg = sum(it["is_gold"][j] for j in sel[it["id"]][arm])
                rows.append({"id": ex.id, "arm": arm, "answer_gold": ex.answer, "answer_pred": pred,
                             "em": em_, "f1": round(f1_, 4), "n_ctx": len(kept),
                             "n_gold_ctx": g, "n_gold_kept": kg,
                             "all_gold_ctx": int(g == len(set(ex.gold_chunk_ids)) and g > 0),
                             "same_as_run": int(pred == it["answer_pred"]),
                             "prompt_tokens": getattr(m, "prompt_tokens", 0),
                             "latency_s": round(time.perf_counter() - t0, 3)})
                tf.write(json.dumps({"id": ex.id, "arm": arm,
                                     "context": [c.chunk_id for c in kept], "raw_output": text}) + "\n")
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(items)}", flush=True)
    out = out_dir / f"{args.name}_qa.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"csv: {out}")


def summarize_frame(qa):
    """Per arm EM, and every arm paired against full (McNemar exact, paired
    bootstrap, Holm across the arms), on all questions and on the questions
    whose context held every gold chunk."""
    import pandas as pd
    from src.eval.stats import holm, mcnemar, paired_bootstrap

    wide = qa.pivot(index="id", columns="arm", values="em")
    allg = qa[qa.arm == "full"].set_index("id").all_gold_ctx.reindex(wide.index)
    out = []
    for subset, idx in (("all", wide.index), ("all_gold_in_context", wide.index[allg == 1])):
        rows = []
        for arm in ARMS:
            if arm == "full" or arm not in wide:
                continue
            both = wide.loc[idx, ["full", arm]].dropna()
            x, y = both["full"].astype(int).tolist(), both[arm].astype(int).tolist()
            only_full, only_arm, p = mcnemar(x, y)
            diff, lo, hi = paired_bootstrap(x, y)
            rows.append({"subset": subset, "arm": arm, "n": len(x),
                         "em_full": round(sum(x) / len(x), 4), "em_arm": round(sum(y) / len(y), 4),
                         "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "only_arm": only_arm, "only_full": only_full, "mcnemar_p": p})
        for r, ph in zip(rows, holm([r["mcnemar_p"] for r in rows])):
            r["mcnemar_p"] = round(r["mcnemar_p"], 6)
            r["holm_p"] = round(ph, 6)
        out += rows
    return pd.DataFrame(out)


def cmd_summarize(args):
    import pandas as pd

    res = Path(args.results_dir)
    qa = pd.read_csv(res / f"{args.name}_qa.csv", dtype={"id": str})
    full = qa[qa.arm == "full"]
    print(f"full arm reproduces the run's answer on {full.same_as_run.mean():.3f} of {len(full)} questions")
    if args.run_qa:
        run = pd.read_csv(args.run_qa, dtype={"id": str}).set_index("id")
        print(f"run EM on these ids {run.loc[full.id, 'em'].mean():.4f}, full arm {full.em.mean():.4f}")
    per_arm = qa.groupby("arm").agg(n=("em", "size"), em=("em", "mean"), f1=("f1", "mean"),
                                    n_ctx=("n_ctx", "mean"), prompt_tokens=("prompt_tokens", "mean"))
    per_arm = per_arm.reindex([a for a in ARMS if a in per_arm.index]).round(4)
    paired = summarize_frame(qa)
    per_arm.to_csv(res / f"{args.name}_summary.csv")
    paired.to_csv(res / f"{args.name}_paired.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(per_arm.to_string())
        print()
        print(paired.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, help="the run's config: dataset, reranker, generator")
    r.add_argument("--trace", required=True, help="the ircot run's trace.jsonl")
    r.add_argument("--ids", default=None)
    r.add_argument("--name", required=True)
    r.add_argument("--train-trace", default=None, help="fit the chunk filter on this run's contexts instead")
    r.add_argument("--train-ids", default=None)
    r.add_argument("--dry-run", action="store_true", help="build contexts and selections, no generation")
    s = sub.add_parser("summarize")
    s.add_argument("--name", required=True)
    s.add_argument("--run-qa", default=None, help="the ircot run's qa.csv, to check the full arm")
    s.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    main()
