"""2WikiMultihopQA dev -> data/processed/2wiki_dev.jsonl.

Raw file is dev.json from the data_ids_april7 release (fixed sentence
segmentation). Same layout as hotpot: _id, question, answer,
supporting_facts = [[title, sent_idx], ...], context = [[title, [sent, ...]]].
On top of that, `evidences` holds (subject, relation, object) triples in
reasoning order, which is what hotpot lacks: real per-hop gold. The
supporting_facts come ordered along the same chain, so reasoning_path is just
their title sequence.
"""
import argparse
import json
import random
from pathlib import Path

from src.eval.gold_mapping import MappingReport, map_facts

RAW = Path(__file__).resolve().parent / "raw" / "2wiki" / "dev.json"
OUT = Path(__file__).resolve().parent / "processed" / "2wiki_dev.jsonl"


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
        "hops": len(facts),
        "qtype": raw.get("type"),
        "chunks": chunks,
        "gold_chunk_ids": gold,
        "gold_supporting_facts": [{"title": t, "sent_idx": i} for t, i in facts],
        "evidence_triples": [
            {"subject": s, "relation": r, "object": o} for s, r, o in raw.get("evidences", [])
        ],
        "reasoning_path": [t for t, _ in facts],
    }
    return record, unmapped


def prepare(subset_size=None, seed=13):
    with open(RAW) as f:
        data = json.load(f)
    if subset_size:
        data = random.Random(seed).sample(data, subset_size)

    report = MappingReport()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for raw in data:
            record, unmapped = build_record(raw)
            report.add(record["gold_chunk_ids"], unmapped)
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(data)} records to {OUT}")
    print(f"gold mapping: {report}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="subset size, default full dev")
    args = p.parse_args()
    prepare(subset_size=args.n)
