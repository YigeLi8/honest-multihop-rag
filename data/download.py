"""Download raw datasets into data/raw/."""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

# from hotpotqa.github.io; the host times out now and then, hf is the fallback
HOTPOT_DEV = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
BEIR_SUBSETS = ["nfcorpus", "scifact", "fiqa"]


def fetch(url, dest, timeout=30):
    if dest.exists():
        print(f"{dest.name} already downloaded")
        return dest
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    return dest


def hotpot_from_hf(dest):
    # the hf mirror stores supporting_facts/context as dicts of parallel lists;
    # convert back to the official layout so prepare_hotpotqa doesn't care
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    for r in ds:
        sf = r["supporting_facts"]
        ctx = r["context"]
        rows.append({
            "_id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "type": r["type"],
            "level": r["level"],
            "supporting_facts": [[t, i] for t, i in zip(sf["title"], sf["sent_id"])],
            "context": [[t, s] for t, s in zip(ctx["title"], ctx["sentences"])],
        })
    with open(dest, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} records from the hf mirror")


def download_hotpotqa():
    dest = RAW / "hotpot_dev_distractor_v1.json"
    if dest.exists():
        print(f"{dest.name} already downloaded")
        return
    try:
        fetch(HOTPOT_DEV, dest)
    except Exception as e:
        print(f"official host not responding ({e}), using the huggingface mirror")
        hotpot_from_hf(dest)


def download_2wiki():
    # the official zip sits behind a dropbox link in the Alab-NII/2wikimultihop
    # readme and the link rotates; grab it manually and unzip into data/raw/2wiki/
    target = RAW / "2wiki"
    if target.exists():
        print("2wiki already in place")
    else:
        print("2wiki: get the data zip from the Alab-NII/2wikimultihop readme "
              "and unzip into data/raw/2wiki/ (need dev.json)")


def download_musique():
    # StonyBrookNLP release, mirrored on the hf hub (answerable + full, dev + train)
    from huggingface_hub import hf_hub_download
    target = RAW / "musique"
    target.mkdir(parents=True, exist_ok=True)
    for fn in ["musique_ans_v1.0_dev.jsonl", "musique_full_v1.0_dev.jsonl",
               "musique_ans_v1.0_train.jsonl"]:
        dest = target / fn
        if dest.exists():
            print(f"{fn} already downloaded")
            continue
        shutil.copy(hf_hub_download("bdsaglam/musique", fn, repo_type="dataset"), dest)
        print(f"wrote {dest.name}")


def download_beir():
    for name in BEIR_SUBSETS:
        z = fetch(BEIR_URL.format(name), RAW / f"{name}.zip")
        out = RAW / name
        if not out.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(RAW)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique", "beir"])
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        {"hotpotqa": download_hotpotqa, "2wiki": download_2wiki,
         "musique": download_musique, "beir": download_beir}[name]()
