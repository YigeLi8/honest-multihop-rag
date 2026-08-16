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
    for i, ex in enumerate(examples):
        res = pipe.run_example(ex)
        got = {rc.chunk.chunk_id for h in res.hops for rc in h.retrieved}
        retr_ok = set(ex.gold_chunk_ids) <= got
        rows.append({
            "id": ex.id,
            "answer_gold": ex.answer,
            "answer_pred": res.answer,
            "em": em(res.answer, ex.answer),
            "f1": round(f1(res.answer, ex.answer), 4),
            "retrieval_correct": int(retr_ok),
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
    mean_em = sum(r["em"] for r in rows) / n
    mean_f1 = sum(r["f1"] for r in rows) / n
    table = build_table((bool(r["em"]), bool(r["retrieval_correct"])) for r in rows)
    png = out_dir / f"{cfg.run.name}_2x2.png"
    plot_table(table, png)

    print(f"\n{cfg.run.name}: n={n}  EM={mean_em:.4f}  F1={mean_f1:.4f}")
    print(table)
    print(f"csv: {out}\nplot: {png}")


if __name__ == "__main__":
    main()
