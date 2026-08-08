"""HotpotQA distractor dev -> data/processed/hotpotqa_dev.jsonl.

Chunks are sentences (same granularity as the supporting-fact annotations),
chunk_id = "{title}::{sent_idx}".
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "processed" / "hotpotqa_dev.jsonl"


def prepare(split="dev", subset_size=None):
    raise NotImplementedError


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prepare()
