"""Download raw datasets into data/raw/."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

# from hotpotqa.github.io
HOTPOT_DEV = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
BEIR_SUBSETS = ["nfcorpus", "scifact", "fiqa"]


def fetch(url, dest):
    if dest.exists():
        print(f"{dest.name} already downloaded")
        return dest
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def download_hotpotqa():
    fetch(HOTPOT_DEV, RAW / "hotpot_dev_distractor_v1.json")


def download_2wiki():
    # the official zip sits behind a dropbox link in the Alab-NII/2wikimultihop
    # readme and the link rotates; grab it manually and unzip into data/raw/2wiki/
    target = RAW / "2wiki"
    if target.exists():
        print("2wiki already in place")
    else:
        print("2wiki: get the data zip from the Alab-NII/2wikimultihop readme "
              "and unzip into data/raw/2wiki/ (need dev.json)")


def download_beir():
    for name in BEIR_SUBSETS:
        z = fetch(BEIR_URL.format(name), RAW / f"{name}.zip")
        out = RAW / name
        if not out.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(RAW)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "beir"])
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        {"hotpotqa": download_hotpotqa, "2wiki": download_2wiki, "beir": download_beir}[name]()
