"""BEIR subsets (nfcorpus, scifact, fiqa) for the nDCG@10 sanity check."""
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent / "processed"
SUBSETS = ["nfcorpus", "scifact", "fiqa"]


def prepare(subset):
    raise NotImplementedError


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for s in SUBSETS:
        prepare(s)
