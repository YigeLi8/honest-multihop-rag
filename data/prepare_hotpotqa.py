"""HotpotQA distractor dev -> data/processed/hotpotqa_dev.jsonl.

Chunks are sentences (same granularity as the supporting-fact annotations),
chunk_id = "{title}::{sent_idx}". Raw file is the official
hotpot_dev_distractor_v1.json: a list of dicts with _id, question, answer,
type, level, supporting_facts = [[title, sent_idx], ...] and
context = [[title, [sentence, ...]], ...].
"""
import argparse
import json
import random
from pathlib import Path

from src.eval.gold_mapping import MappingReport, map_facts

HERE = Path(__file__).resolve().parent
RAW = {"dev": HERE / "raw" / "hotpot_dev_distractor_v1.json",
       "train": HERE / "raw" / "hotpot_train_v1.1.json"}
OUT = {"dev": HERE / "processed" / "hotpotqa_dev.jsonl",
       "train": HERE / "processed" / "hotpotqa_train.jsonl"}


def build_record(raw):
    sents_by_title = {title: sents for title, sents in raw["context"]}
    chunks = [
        {"chunk_id": f"{title}::{i}", "title": title, "sent_idx": i, "text": sent}
        for title, sents in raw["context"]
        for i, sent in enumerate(sents)
    ]
    facts = [(t, i) for t, i in raw["supporting_facts"]]
    gold, unmapped = map_facts(facts, sents_by_title)
    record = {
        "id": raw["_id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "hops": 2,
        "qtype": raw.get("type"),
        "level": raw.get("level"),
        "chunks": chunks,
        "gold_chunk_ids": gold,
        "gold_supporting_facts": [{"title": t, "sent_idx": i} for t, i in facts],
    }
    return record, unmapped


def prepare(subset_size=None, seed=13, split="dev"):
    raw, out = RAW[split], OUT[split]
    with open(raw) as f:
        data = json.load(f)
    if subset_size:
        data = random.Random(seed).sample(data, subset_size)

    report = MappingReport()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for raw in data:
            record, unmapped = build_record(raw)
            report.add(record["gold_chunk_ids"], unmapped)
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(data)} records to {out}")
    print(f"gold mapping: {report}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="subset size, default the full split")
    p.add_argument("--split", default="dev", choices=["dev", "train"],
                   help="train needs hotpot_train_v1.1.json in data/raw (the selector arm trains on it)")
    args = p.parse_args()
    prepare(subset_size=args.n, split=args.split)
