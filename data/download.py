"""Download raw datasets into data/raw/."""
import argparse
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"


def download(name):
    # hotpotqa: datasets.load_dataset("hotpot_qa", "distractor"), dev split
    # 2wiki: check the HF mirror first, official release is a zip
    # beir: nfcorpus/scifact/fiqa corpora + qrels
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "beir"])
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        download(name)
