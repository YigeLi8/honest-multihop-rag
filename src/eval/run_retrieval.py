"""Single-hop retrieval eval against the gold chunk ids.

No generation involved, just: does the retriever surface the gold sentences.
Writes a per-question csv to results/ and prints the aggregate.

    python -m src.eval.run_retrieval --config configs/bm25.yaml
    python -m src.eval.run_retrieval --config configs/dense_bge.yaml --n 500
"""
import argparse
import csv
import json
from pathlib import Path

from src.config import load_config
from src.eval.per_hop_precision import precision_recall
from src.retrieval.base import get_retriever
from src.types import Chunk


def load_examples(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=None, help="cap the number of questions")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_file = Path(cfg.paths.data_dir) / f"{cfg.dataset.name}_dev.jsonl"
    examples = load_examples(data_file, args.n or cfg.dataset.subset_size)
    retriever = get_retriever(cfg)
    k = cfg.retrieval.top_k

    rows = []
    for i, ex in enumerate(examples):
        chunks = [Chunk(chunk_id=c["chunk_id"], text=c["text"],
                        title=c["title"], sent_idx=c["sent_idx"])
                  for c in ex["chunks"]]
        retriever.index(chunks)
        hits = retriever.retrieve(ex["question"], k)
        got = [h.chunk.chunk_id for h in hits]
        gold = ex["gold_chunk_ids"]
        p, r = precision_recall(got, gold)
        rows.append({"id": ex["id"], "level": ex.get("level"), "qtype": ex.get("qtype"),
                     "n_chunks": len(chunks), "n_gold": len(gold),
                     "precision": round(p, 4), "recall": round(r, 4),
                     "retrieval_correct": int(set(gold) <= set(got))})
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(examples)}")

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{cfg.run.name}_retrieval.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mp = sum(r["precision"] for r in rows) / n
    mr = sum(r["recall"] for r in rows) / n
    rc = sum(r["retrieval_correct"] for r in rows) / n
    print(f"\n{cfg.run.name}: n={n}, top_k={k}")
    print(f"precision@{k}={mp:.4f}  recall@{k}={mr:.4f}  all-gold-in-top-{k}={rc:.4f}")
    print(f"per-question csv: {out}")


if __name__ == "__main__":
    main()
