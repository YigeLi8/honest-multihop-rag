"""MuSiQue (answerable dev, 2,417 q) -> data/processed/musique_dev.jsonl.

Raw files are musique_ans_v1.0_dev.jsonl / musique_full_v1.0_dev.jsonl from
the StonyBrookNLP release (mirrored on the hf hub as bdsaglam/musique). Each
question comes with 20 paragraphs flagged is_supporting, and a
question_decomposition listing the hops in order: sub-question, answer and
the index of the paragraph that supports it. So unlike hotpot the gold is
paragraph-level, exact (no heuristic mapping) and ordered by hop.

Chunks are paragraphs here, chunk_id = "p{idx}::{title}". The reasoning_path
carries one entry per hop with its gold chunk, which is what real per-hop
precision needs. qtype is the composition shape encoded in the id
(2hop, 3hop1, 3hop2, 4hop1, 4hop2, 4hop3).
"""
import argparse
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw" / "musique"
PROCESSED = Path(__file__).resolve().parent / "processed"


def build_record(raw):
    chunks = [
        {"chunk_id": f"p{p['idx']}::{p['title']}", "title": p["title"],
         "sent_idx": p["idx"], "text": p["paragraph_text"]}
        for p in raw["paragraphs"]
    ]
    by_idx = {p["idx"]: f"p{p['idx']}::{p['title']}" for p in raw["paragraphs"]}
    path = []
    for step in raw["question_decomposition"]:
        idx = step.get("paragraph_support_idx")
        path.append({"sub_question": step["question"], "answer": step["answer"],
                     "gold_chunk_id": by_idx.get(idx) if idx is not None else None})
    gold = [s["gold_chunk_id"] for s in path if s["gold_chunk_id"]]
    flagged = [c["chunk_id"] for c, p in zip(chunks, raw["paragraphs"]) if p["is_supporting"]]
    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "answer_aliases": raw.get("answer_aliases", []),
        "answerable": raw.get("answerable", True),
        "hops": len(raw["question_decomposition"]),
        "qtype": raw["id"].split("__")[0],
        "chunks": chunks,
        "gold_chunk_ids": gold if gold else flagged,
        "gold_supporting_facts": [{"title": c.split("::", 1)[1], "sent_idx": -1} for c in gold],
        "reasoning_path": path,
    }


def prepare(full=False, split="dev"):
    src = RAW_DIR / f"musique_{'full' if full else 'ans'}_v1.0_{split}.jsonl"
    dest = PROCESSED / f"musique_{split}.jsonl"
    n, n_mismatch = 0, 0
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(src) as f, open(dest, "w") as out:
        for line in f:
            raw = json.loads(line)
            rec = build_record(raw)
            flagged = {c["chunk_id"] for c, p in zip(rec["chunks"], raw["paragraphs"])
                       if p["is_supporting"]}
            if rec["answerable"] and set(rec["gold_chunk_ids"]) != flagged:
                n_mismatch += 1
            out.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} records to {dest}")
    print(f"decomposition gold vs is_supporting flags disagree on {n_mismatch} answerable questions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="use the full split (answerable + unanswerable) instead of answerable only")
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    args = p.parse_args()
    prepare(full=args.full, split=args.split)
