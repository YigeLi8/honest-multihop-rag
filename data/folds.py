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
"""
import json
import random
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


if __name__ == "__main__":
    main()
