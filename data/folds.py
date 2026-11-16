"""Fixed question folds, written once and committed.

The first 500 questions of each dev file are what I looked at while building
the harness and choosing knobs (exploration). Anything I want to claim is
tested once on a disjoint confirmation fold with criteria written down
before the run. Ids are listed in evaluation order.

    python -m data.folds

    hotpot_explore   hotpotqa dev positions 0-499
    hotpot_confirm   positions 500-1499
    2wiki_explore    2wiki dev positions 0-499
    2wiki_confirm    positions 500-1499
    musique_explore  musique dev (shuffled at prep) positions 0-499
    musique_confirm  1000 ids stratified by composition shape, seed 13,
                     drawn from positions 500 onward

    python -m data.folds --train-hard

    hotpot_train_hard  500 hotpotqa train questions of level hard (the dev
                     set is all hard), seed 17, none from the first 20,000
                     train records (the answer guard's wording rule was
                     written on those) or from hotpot_train_mem (the
                     experience memory was built on it). Fresh questions
                     for choosing agent changes once the exploration fold
                     has been read for one (docs/plan.md, Stage 6b)
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent / "processed"
OUT = Path(__file__).resolve().parent.parent / "results" / "folds"


def ids_and_types(name):
    ids, types = [], []
    with open(PROCESSED / f"{name}_dev.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["id"])
            types.append(d.get("qtype") or "")
    return ids, types


def stratified(ids, types, n, seed):
    by_type = defaultdict(list)
    for i, t in zip(ids, types):
        by_type[t].append(i)
    rng = random.Random(seed)
    total = len(ids)
    picked = []
    for t in sorted(by_type):
        pool = by_type[t]
        k = round(n * len(pool) / total)
        picked += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picked)
    return picked[:n]


def write(name, ids):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.txt", "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{name}: {len(ids)} ids")


def main(seed=13):
    for ds, label in (("hotpotqa", "hotpot"), ("2wiki", "2wiki"), ("musique", "musique")):
        ids, types = ids_and_types(ds)
        write(f"{label}_explore", ids[:500])
        if label == "musique":
            write("musique_confirm", stratified(ids[500:], types[500:], 1000, seed))
        else:
            write(f"{label}_confirm", ids[500:1500])


def train_hard(n=500, seed=17, skip=20000):
    """Ids of hard train questions outside the first `skip` records and the
    memory's train fold, in a seeded order. Reads only the head of each
    record (the train file is 0.9 GB)."""
    mem = set((OUT / "hotpot_train_mem.txt").read_text().split())
    pool = []
    with open(PROCESSED / "hotpotqa_train.jsonl") as f:
        for k, line in enumerate(f):
            if k < skip:
                continue
            head = line[:4000]
            qid = re.search(r'"id": "([^"]+)"', head).group(1)
            if re.search(r'"level": "hard"', head) and qid not in mem:
                pool.append(qid)
    return random.Random(seed).sample(pool, n)


if __name__ == "__main__":
    if "--train-hard" in sys.argv:
        write("hotpot_train_hard", train_hard())
    else:
        main()
