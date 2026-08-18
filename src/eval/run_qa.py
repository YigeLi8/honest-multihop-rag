"""End-to-end QA eval: retrieve, generate, score, build the 2x2.

    python -m src.eval.run_qa --config configs/bm25.yaml --backend mlx --n 200

Writes results/<run>_qa.csv and results/<run>_2x2.png. retrieval_correct means
all gold sentences were in the context handed to the model.
"""
import argparse
import csv
import json
from pathlib import Path

from src.config import load_config
from src.eval.answer_metrics import em, f1
from src.eval.decoupling import build_table, plot_table
from src.eval.per_hop_precision import plot_per_hop, score
from src.eval.stats import wilson_ci
from src.pipeline.multihop import MultiHopPipeline
from src.types import example_from_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--backend", default=None, help="override generation backend")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.backend:
        cfg.generation.backend = args.backend

    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(example_from_json(json.loads(line)))
            if args.n and len(examples) >= args.n:
                break

    pipe = MultiHopPipeline(cfg)
    rows = []
    results = []
    for i, ex in enumerate(examples):
        res = pipe.run_example(ex)
        results.append(res)
        got = {rc.chunk.chunk_id for h in res.hops for rc in h.retrieved}
        retr_ok = set(ex.gold_chunk_ids) <= got
        rows.append({
            "id": ex.id,
            "answer_gold": ex.answer,
            "answer_pred": res.answer,
            "em": em(res.answer, ex.answer),
            "f1": round(f1(res.answer, ex.answer), 4),
            "retrieval_correct": int(retr_ok),
            "hops": len(res.hops),
            "n_context": len(got),
            "reasoning_tokens": res.reasoning_tokens,
            "latency_s": round(res.latency_s, 3),
            "tokens_per_s": round(res.tokens_per_s, 1),
            "peak_memory_mb": round(res.peak_memory_mb, 1),
        })
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(examples)}")

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(exist_ok=True)
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

    hop_stats = score(results)
    for hop, s in hop_stats.items():
        print(f"hop {hop}: precision={s['precision']:.3f} recall={s['recall']:.3f} "
              f"n={s['n']} retrieved/hop={s['retrieved_mean']:.1f} "
              f"(var {s['retrieved_var']:.1f})")
    if len(hop_stats) > 1:
        plot_per_hop(hop_stats, out_dir / f"{cfg.run.name}_per_hop.png")

    print(f"csv: {out}\nplot: {png}")


if __name__ == "__main__":
    main()
