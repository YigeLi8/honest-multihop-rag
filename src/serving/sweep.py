"""Serving sweep on Metal: quant x top_k x rerank, single-hop so points are
comparable. Each point runs the qa pipeline over a fixed subset and records
answer quality next to throughput; rows go to the csv as they finish so a
killed run keeps its partial results.

Batch size / concurrency needs a backend with continuous batching (llama.cpp
server); that axis comes later. All numbers are hardware-specific.

    python -m src.serving.sweep --config configs/serving_sweep.yaml --n 100
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import yaml

from src.config import load_config
from src.eval.answer_metrics import em, f1
from src.pipeline.multihop import MultiHopPipeline
from src.types import example_from_json

QUANT_MODELS = {
    "4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "8bit": "mlx-community/Qwen2.5-7B-Instruct-8bit",
}


def expand_grid(axes):
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def run_point(config_path, point, examples):
    cfg = load_config(config_path)
    cfg.pipeline.mode = "single_hop"
    cfg.generation.backend = "mlx"
    cfg.generation.model = QUANT_MODELS[point["quant"]]
    cfg.retrieval.top_k = point["top_k"]
    cfg.rerank.enabled = point["rerank"]

    pipe = MultiHopPipeline(cfg)
    ems, f1s, rcs, tps, lat = [], [], [], [], []
    peak = 0.0
    for ex in examples:
        res = pipe.run_example(ex)
        got = {rc.chunk.chunk_id for h in res.hops for rc in h.retrieved}
        ems.append(em(res.answer, ex.answer))
        f1s.append(f1(res.answer, ex.answer))
        rcs.append(set(ex.gold_chunk_ids) <= got)
        tps.append(res.tokens_per_s)
        lat.append(res.latency_s)
        peak = max(peak, res.peak_memory_mb)
    n = len(examples)
    return {**point, "n": n,
            "em": round(sum(ems) / n, 4), "f1": round(sum(f1s) / n, 4),
            "retrieval_correct": round(sum(rcs) / n, 4),
            "tokens_per_s": round(sum(tps) / n, 1),
            "latency_s": round(sum(lat) / n, 3),
            "peak_memory_mb": round(peak, 1)}


def plot_pareto(rows, out_path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r["tokens_per_s"] for r in rows], [r["em"] for r in rows])
    for r in rows:
        tag = f"{r['quant']},k={r['top_k']}" + (",rr" if r["rerank"] else "")
        ax.annotate(tag, (r["tokens_per_s"], r["em"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    # frontier: walking from fastest to slowest, keep points that raise em
    front, best = [], -1.0
    for r in sorted(rows, key=lambda r: -r["tokens_per_s"]):
        if r["em"] > best:
            front.append(r)
            best = r["em"]
    front.sort(key=lambda r: r["tokens_per_s"])
    ax.plot([r["tokens_per_s"] for r in front], [r["em"] for r in front],
            "--", color="gray")
    ax.set_xlabel("generation throughput (tokens/s)")
    ax.set_ylabel("answer EM")
    ax.set_title(f"EM vs throughput, Qwen2.5-7B ({label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--label", default="apple-silicon-metal",
                    help="hardware label recorded with the results")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with open(args.config) as f:
        axes = yaml.safe_load(f).get("sweep", {})
    # 4bit points first so each model only loads once
    points = sorted(expand_grid(axes), key=lambda p: p["quant"])
    print(f"{len(points)} sweep points, n={args.n} each")

    examples = []
    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    with open(data_file) as f:
        for line in f:
            examples.append(example_from_json(json.loads(line)))
            if len(examples) >= args.n:
                break

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "sweep_metal.csv"
    rows = []
    for i, point in enumerate(points):
        print(f"[{i + 1}/{len(points)}] {point}")
        row = run_point(args.config, point, examples)
        row["device"] = args.label
        rows.append(row)
        with open(out_csv, "w", newline="") as f:   # rewrite so partials survive
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"    em={row['em']} tok/s={row['tokens_per_s']} peak={row['peak_memory_mb']}mb")

    plot_pareto(rows, out_dir / "pareto.png", args.label)
    print(f"\ncsv: {out_csv}\nplot: {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main()
