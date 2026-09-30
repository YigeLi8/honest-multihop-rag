"""End-to-end QA eval: retrieve, generate, score, build the 2x2.

    python -m src.eval.run_qa --config configs/bm25.yaml --backend mlx --n 200

Writes results/<run>_qa.csv (one row per question), results/<run>_trace.jsonl
(what was retrieved at every hop plus the raw completions, so later audits
don't need a re-run) and results/<run>_2x2.png. retrieval_correct means all
gold sentences were in the context handed to the model; gold_recall is the
fraction that were, which separates "partly there" from "not there at all".
"""
import argparse
import csv
import json
from pathlib import Path

from src.config import load_config
from src.eval.answer_metrics import em, f1, normalize
from src.eval.decoupling import build_table, plot_table
from src.eval.per_hop_precision import plot_per_hop, score
from src.eval.stats import wilson_ci
from src.pipeline.multihop import MultiHopPipeline
from src.types import example_from_json


def read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_examples(path, n=None, ids=None):
    """First n records, or exactly the ids listed (in the list's order)."""
    if ids:
        want = set(ids)
        by_id = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in want:
                    by_id[d["id"]] = example_from_json(d)
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids not in {path}, e.g. {missing[:3]}")
        return [by_id[i] for i in ids]
    out = []
    with open(path) as f:
        for line in f:
            out.append(example_from_json(json.loads(line)))
            if n and len(out) >= n:
                break
    return out


def answer_type(gold):
    return "yesno" if normalize(gold) in ("yes", "no") else "span"


def best_em_f1(pred, ex):
    """Max over the gold answer and its aliases (musique lists aliases; the
    hotpot and 2wiki records have none, so this is plain em/f1 there)."""
    golds = [ex.answer] + [a for a in ex.answer_aliases if a]
    return max(em(pred, g) for g in golds), max(f1(pred, g) for g in golds)


def paragraph_of(chunk_id):
    """Chunk ids are "{title}::{sent_idx}" (hotpot, 2wiki) or "p{idx}::{title}"
    (musique, paragraph chunks). Either way the paragraph is the prefix."""
    return chunk_id.rsplit("::", 1)[0] if chunk_id.startswith("p") and "::" in chunk_id \
        and chunk_id.split("::", 1)[0][1:].isdigit() else chunk_id.rsplit("::", 1)[0]


def paragraph_recall(got_ids, gold_ids):
    """Fraction of gold paragraphs with at least one retrieved sentence. The
    sentence-level gold is strict; a neighbouring sentence of the same
    paragraph often carries the same fact, and this tells those cases apart
    from a paragraph that never reached the model."""
    gold_paras = {paragraph_of(c) for c in gold_ids}
    got_paras = {paragraph_of(c) for c in got_ids}
    if not gold_paras:
        return 0, 0, 0.0
    hit = len(gold_paras & got_paras)
    return len(gold_paras), hit, hit / len(gold_paras)


def score_row(ex, res):
    got = {rc.chunk.chunk_id for h in res.hops for rc in h.retrieved}
    gold = set(ex.gold_chunk_ids)
    hit = len(gold & got)
    n_para, para_hit, para_rec = paragraph_recall(got, gold)
    em_, f1_ = best_em_f1(res.answer, ex)
    return {
        "id": ex.id,
        "qtype": ex.qtype,
        "level": ex.level,
        "answer_type": answer_type(ex.answer),
        "answer_gold": ex.answer,
        "answer_pred": res.answer,
        "em": em_,
        "f1": round(f1_, 4),
        "retrieval_correct": int(gold <= got),
        "n_gold": len(gold),
        "n_gold_hit": hit,
        "gold_recall": round(hit / len(gold), 4) if gold else 0.0,
        "n_gold_para": n_para,
        "n_gold_para_hit": para_hit,
        "para_recall": round(para_rec, 4),
        "hops": len(res.hops),
        "stop_reason": res.stop_reason,
        "n_context": len(got),
        "reasoning_tokens": res.reasoning_tokens,
        "latency_s": round(res.latency_s, 3),
        "tokens_per_s": round(res.tokens_per_s, 1),
        "peak_memory_mb": round(res.peak_memory_mb, 1),
    }


def trace_record(ex, res):
    return {
        "id": ex.id,
        "answer_pred": res.answer,
        "stop_reason": res.stop_reason,
        "hops": [{"hop": h.hop, "query": h.query,
                  "retrieved": [rc.chunk.chunk_id for rc in h.retrieved]}
                 for h in res.hops],
        "raw_outputs": res.raw_outputs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ids", default=None, help="fold file with one question id per line")
    ap.add_argument("--backend", default=None, help="override generation backend")
    ap.add_argument("--model", default=None, help="override generation model")
    ap.add_argument("--name", default=None, help="override run name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.backend:
        cfg.generation.backend = args.backend
    if args.model:
        cfg.generation.model = args.model
    if args.name:
        cfg.run.name = args.name

    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    examples = load_examples(data_file, args.n, read_ids(args.ids) if args.ids else None)

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(exist_ok=True)
    trace_path = out_dir / f"{cfg.run.name}_trace.jsonl"

    pipe = MultiHopPipeline(cfg)
    rows = []
    results = []
    with open(trace_path, "w") as tf:
        for i, ex in enumerate(examples):
            res = pipe.run_example(ex)
            results.append(res)
            rows.append(score_row(ex, res))
            tf.write(json.dumps(trace_record(ex, res)) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(examples)}")

    out = out_dir / f"{cfg.run.name}_qa.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    em_hits = sum(r["em"] for r in rows)
    mean_f1 = sum(r["f1"] for r in rows) / n
    table = build_table((bool(r["em"]), bool(r["retrieval_correct"])) for r in rows)
    png = out_dir / f"{cfg.run.name}_2x2.png"
    plot_table(table, png)

    em_lo, em_hi = wilson_ci(em_hits, n)
    il_lo, il_hi = wilson_ci(table.ac_rw, n)
    print(f"\n{cfg.run.name}: n={n}  EM={em_hits / n:.4f} "
          f"(95% CI {em_lo:.3f}-{em_hi:.3f})  F1={mean_f1:.4f}")
    print(f"{table}  illusion CI {il_lo:.3f}-{il_hi:.3f}")

    if cfg.pipeline.mode != "closed_book":
        hop_stats = score(results, {ex.id: ex for ex in examples})
        for hop, s in hop_stats.items():
            aligned = (f" gold[hop]-seen={s['aligned_recall']:.3f}"
                       if "aligned_recall" in s else "")
            print(f"hop {hop}: precision={s['precision']:.3f} recall={s['recall']:.3f} "
                  f"cumulative={s['cumulative_recall']:.3f}{aligned} "
                  f"n={s['n']} retrieved/hop={s['retrieved_mean']:.1f} "
                  f"(var {s['retrieved_var']:.1f})")
        with open(out_dir / f"{cfg.run.name}_per_hop.json", "w") as f:
            json.dump(hop_stats, f, indent=1)
        if len(hop_stats) > 1:
            plot_per_hop(hop_stats, out_dir / f"{cfg.run.name}_per_hop.png")

    print(f"csv: {out}\ntrace: {trace_path}\nplot: {png}")


if __name__ == "__main__":
    main()
