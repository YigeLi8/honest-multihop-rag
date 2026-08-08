"""2WikiMultihopQA dev -> data/processed/2wiki_dev.jsonl.

Same schema as hotpotqa plus evidence_triples and reasoning_path (gives the
per-hop gold ordering, and feeds the graph arm).
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "processed" / "2wiki_dev.jsonl"


def prepare(split="dev", subset_size=None):
    raise NotImplementedError


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prepare()
