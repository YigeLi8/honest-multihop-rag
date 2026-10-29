"""Intervention arms: hand the generator a context built from the gold labels
instead of from a retriever, and see what it answers.

    python -m src.eval.interventions run --config configs/bm25.yaml \
        --ids results/folds/hotpot_explore.txt --name interventions_hotpot_explore
    python -m src.eval.interventions summarize --name interventions_hotpot_explore \
        --data-file data/processed/hotpotqa_dev.jsonl \
        --closed-book closed_book_hotpot_dev --bm25-run bm25_hotpot_dev

The decomposition (analyze.py) says what was in context when an answer was
right without all the gold. These arms ask the converse on the same
questions: given exactly this evidence, does the model get it right?

    gold         every gold chunk (sentence; paragraph on musique), nothing else
    answer_only  gold minus the bridge: only the gold chunks that contain the
                 answer string, i.e. the shortcut bucket's situation made clean
    bridge_only  gold minus the answer chunk: only the gold chunks that do not
                 contain it. A right answer here comes from memory or from
                 inference over the bridge, not from the answer's own sentence
    distractors  the config's top-k non-gold chunks by bm25 on the question,
                 taken from paragraphs that hold no gold chunk and with no text
                 equal to a gold chunk's (musique repeats paragraphs under
                 other ids). No gold paragraph at all

"Contains the answer" is analyze.answer_in's rule (normalized gold answer or
alias as a substring of the normalized text), so answer_only is exactly what
the shortcut bucket calls an answer-bearing gold sentence. answer_only and
bridge_only exist only where the answer is a span (not yes/no), at least one
gold chunk contains it and at least one does not; gold and distractors run on
every question. Every arm lists its chunks in pool order, so two arms over the
same question differ only in which chunks are there, and the prompt is the one
answer prompt every config uses. Gold builds the input here on purpose: these
are interventions, not a retriever.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from src.eval.answer_metrics import normalize
from src.eval.run_qa import answer_type, best_em_f1, load_examples, paragraph_of, read_ids
from src.eval.stats import holm, mcnemar, paired_bootstrap, wilson_ci
from src.pipeline.multihop import PROMPT, first_line

ARMS = ("gold", "answer_only", "bridge_only", "distractors")


def answer_strings(ex):
    """Normalized gold answer and aliases; None for a yes/no question."""
    answers = [normalize(a) for a in [ex.answer] + list(ex.answer_aliases) if a]
    if any(a in ("yes", "no") for a in answers):
        return None
    return [a for a in answers if a]


def contains_answer(text, answers):
    s = normalize(text)
    return any(a in s for a in answers)


def build_contexts(ex, k, ranker=None):
    """arm -> chunks in pool order, for the arms defined on this question.
    ranker(question, chunks, k) picks the distractors; without one the first k
    candidates in pool order are taken (tests only)."""
    gold = set(ex.gold_chunk_ids)
    seen, pool = set(), []
    for c in ex.chunks:                 # some 2wiki pools hold a paragraph twice
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            pool.append(c)
    gold_chunks = [c for c in pool if c.chunk_id in gold]
    out = {}
    if gold_chunks:
        out["gold"] = gold_chunks
    answers = answer_strings(ex)
    if answers:
        ans = [c for c in gold_chunks if contains_answer(c.text, answers)]
        rest = [c for c in gold_chunks if not contains_answer(c.text, answers)]
        if ans and rest:
            out["answer_only"] = ans
            out["bridge_only"] = rest
    gold_paras = {paragraph_of(c.chunk_id) for c in gold_chunks}
    gold_text = {normalize(c.text) for c in gold_chunks}
    cands = [c for c in pool if paragraph_of(c.chunk_id) not in gold_paras
             and normalize(c.text) not in gold_text]
    if cands:
        picked = ranker(ex.question, cands, k) if ranker else cands[:k]
        keep = {c.chunk_id for c in picked}
        out["distractors"] = [c for c in cands if c.chunk_id in keep]
    return out


def bm25_ranker(cfg):
    from src.retrieval.bm25 import Bm25Retriever
    r = Bm25Retriever(cfg)

    def rank(question, chunks, k):
        r.index(chunks)
        return [h.chunk for h in r.retrieve(question, k)]
    return rank


def run_arms(ex, contexts, generator):
    """One generation per defined arm. Returns (csv rows, trace records)."""
    answers = answer_strings(ex)
    gold = set(ex.gold_chunk_ids)
    rows, traces = [], []
    for arm in ARMS:
        if arm not in contexts:
            continue
        chunks = contexts[arm]
        ctx = "\n".join(f"- {c.text}" for c in chunks)
        t0 = time.perf_counter()
        text, m = generator.generate(PROMPT.format(context=ctx, question=ex.question))
        pred = first_line(text)
        em_, f1_ = best_em_f1(pred, ex)
        in_ctx = "" if answers is None else int(any(contains_answer(c.text, answers)
                                                    for c in chunks))
        rows.append({"id": ex.id, "arm": arm, "qtype": ex.qtype,
                     "answer_type": answer_type(ex.answer), "answer_gold": ex.answer,
                     "answer_pred": pred, "em": em_, "f1": round(f1_, 4),
                     "n_ctx": len(chunks), "n_gold_in_ctx": sum(c.chunk_id in gold for c in chunks),
                     "answer_in_ctx": in_ctx,
                     "prompt_tokens": getattr(m, "prompt_tokens", 0),
                     "latency_s": round(time.perf_counter() - t0, 3)})
        traces.append({"id": ex.id, "arm": arm, "context": [c.chunk_id for c in chunks],
                       "answer_pred": pred, "raw_output": text})
    return rows, traces


def cmd_run(args):
    from src.config import load_config
    from src.pipeline.generate import Generator

    cfg = load_config(args.config)
    cfg.generation.backend = "mlx"
    if args.model:
        cfg.generation.model = args.model
    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    examples = load_examples(data_file, args.n, read_ids(args.ids) if args.ids else None)
    k = cfg.retrieval.top_k
    rank = bm25_ranker(cfg)
    contexts = [build_contexts(ex, k, rank) for ex in examples]
    counts = {a: sum(a in c for c in contexts) for a in ARMS}
    print(f"{len(examples)} questions from {data_file}, distractor k={k}; "
          f"arms defined: {counts}, {sum(counts.values())} generations")
    if args.dry_run:
        return

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(exist_ok=True)
    gen = Generator(cfg)
    rows = []
    with open(out_dir / f"{args.name}_trace.jsonl", "w") as tf:
        for i, (ex, ctxs) in enumerate(zip(examples, contexts)):
            r, t = run_arms(ex, ctxs, gen)
            rows += r
            for rec in t:
                tf.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(examples)}", flush=True)
    out = out_dir / f"{args.name}_qa.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"csv: {out}")


# pairs compared on the questions where both are defined; closed_book is the
# closed-book run on the same ids
PAIRS = (("closed_book", "gold"), ("closed_book", "distractors"),
         ("closed_book", "bridge_only"), ("gold", "answer_only"),
         ("gold", "bridge_only"), ("bridge_only", "answer_only"))


def summarize_frames(qa, cb_em, buckets=None):
    """qa: the arms csv; cb_em: id -> closed-book em; buckets: id -> bm25
    bucket label ('' outside the cell). Returns (per-arm, paired, by-bucket)."""
    wide = qa.pivot(index="id", columns="arm", values="em")
    wide["closed_book"] = pd.Series(cb_em).reindex(wide.index)
    per_arm = []
    for arm in ARMS:
        if arm not in wide:
            continue
        sub = qa[qa.arm == arm]
        n, k = len(sub), int(sub.em.sum())
        lo, hi = wilson_ci(k, n)
        ain = pd.to_numeric(sub.answer_in_ctx, errors="coerce")
        per_arm.append({"arm": arm, "n": n, "em": round(k / n, 4), "em_lo": round(lo, 4),
                        "em_hi": round(hi, 4), "f1": round(sub.f1.mean(), 4),
                        "closed_book_em_same_ids": round(wide.loc[sub.id, "closed_book"].mean(), 4),
                        "answer_in_ctx": round(ain.mean(), 4) if ain.notna().any() else float("nan"),
                        "n_ctx_mean": round(sub.n_ctx.mean(), 2),
                        "prompt_tokens_mean": round(sub.prompt_tokens.mean(), 1)})
    paired = []
    for a, b in PAIRS:
        if a not in wide or b not in wide:
            continue
        both = wide[[a, b]].dropna()
        x, y = both[a].astype(int).tolist(), both[b].astype(int).tolist()
        only_a, only_b, p = mcnemar(x, y)
        diff, lo, hi = paired_bootstrap(x, y)
        paired.append({"a": a, "b": b, "n": len(x), "em_a": round(sum(x) / len(x), 4),
                       "em_b": round(sum(y) / len(y), 4), "diff": round(diff, 4),
                       "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                       "only_a": only_a, "only_b": only_b, "mcnemar_p": p})
    for r, ph in zip(paired, holm([r["mcnemar_p"] for r in paired])):
        r["mcnemar_p"] = round(r["mcnemar_p"], 6)
        r["holm_p"] = round(ph, 6)
    by_bucket = []
    if buckets is not None:
        lab = pd.Series(buckets).reindex(wide.index).fillna("(not in run)")
        for b in sorted(set(lab)):
            ids = lab.index[lab == b]
            row = {"bm25_bucket": b, "n": len(ids),
                   "closed_book_em": round(wide.loc[ids, "closed_book"].mean(), 4)}
            for arm in ARMS:
                if arm in wide:
                    v = wide.loc[ids, arm].dropna()
                    row[f"{arm}_n"] = len(v)
                    row[f"{arm}_em"] = round(v.mean(), 4) if len(v) else float("nan")
            by_bucket.append(row)
    return pd.DataFrame(per_arm), pd.DataFrame(paired), pd.DataFrame(by_bucket)


def cmd_summarize(args):
    from src.eval.analyze import bucket, load_data, load_run

    res = Path(args.results_dir)
    qa = pd.read_csv(res / f"{args.name}_qa.csv", dtype={"id": str})
    cb = pd.read_csv(res / f"{args.closed_book}_qa.csv", dtype={"id": str}).set_index("id")
    missing = set(qa.id) - set(cb.index)
    if missing:
        raise SystemExit(f"{len(missing)} ids have no closed-book row, e.g. {sorted(missing)[:3]}")
    buckets = None
    if args.bm25_run:
        data = load_data(args.data_file)
        df = load_run(args.results_dir, args.bm25_run, data)
        df = df.loc[df.index.intersection(cb.index)]
        df["cb_em"] = cb.em.reindex(df.index)
        # outside the cell, keep "answer wrong" apart from "right with all gold"
        buckets = {i: bucket(r) or ("(right, all gold)" if r.em == 1 else "(wrong)")
                   for i, r in df.iterrows()}
    per_arm, paired, by_bucket = summarize_frames(qa, cb.em.to_dict(), buckets)
    per_arm.to_csv(res / f"{args.name}_summary.csv", index=False)
    paired.to_csv(res / f"{args.name}_paired.csv", index=False)
    if len(by_bucket):
        by_bucket.to_csv(res / f"{args.name}_by_bucket.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(per_arm.to_string(index=False))
        print()
        print(paired.to_string(index=False))
        if len(by_bucket):
            print()
            print(by_bucket.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, help="dataset, data dir and distractor k come from here")
    r.add_argument("--ids", default=None)
    r.add_argument("--n", type=int, default=None)
    r.add_argument("--name", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--dry-run", action="store_true", help="build the contexts, count arms, no model")
    s = sub.add_parser("summarize")
    s.add_argument("--name", required=True)
    s.add_argument("--closed-book", required=True)
    s.add_argument("--bm25-run", default=None, help="run whose illusion buckets split the table")
    s.add_argument("--data-file", default=None)
    s.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        if args.bm25_run and not args.data_file:
            ap.error("--bm25-run needs --data-file")
        cmd_summarize(args)


if __name__ == "__main__":
    main()
