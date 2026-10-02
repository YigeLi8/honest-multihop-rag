"""Stage 1b: look-alike pairs. Do questions that look alike get the same
retrieval outcome? (docs/plan.md, Part B)

A similarity memory assumes they do: it keys past experience on the
question's surface and applies it to the nearest new question. The tables
here measure that assumption directly, before any memory is built.

Natural pairs: each question's nearest other question by cosine over the
question text (tf-idf, or the dense retriever's question embedding), deduped,
binned by similarity. Per band and arm, how often the two questions of a
pair share the arm's outcome (both hit or both miss), and how often the
best arm is the same, beside the same rates over random pairs. If the
nearest-neighbour agreement is no higher than random-pair agreement, the
surface carries nothing about the outcome and a surface-keyed memory has no
signal to find; the plan expects this to be sparse on hotpot and says to
report it as such.

Constructed twins (data/make_twins.py): a twin keeps the question and
changes the pool (distractor), or keeps the pool and changes one entity
mention (alias). Per kind and arm, how often the twin's outcome differs
from the original's, which way, and whether the best arm moved. A
distractor twin that flips outcomes is a pair a surface-keyed memory cannot
tell apart and a boundary over pool features could; an alias twin that
keeps its outcomes is a pair the surface key misses and the boundary
should not.

Outcomes are retrieval-level and come from the hop-0 records of an arms run
(src.memory.arms.arm_table). Nothing is fit: these are descriptive tables
with Wilson intervals, no model.

    python -m src.memory.lookalike natural --records results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset hotpot [--dense-neighbours] [--ids ...]
    python -m src.memory.lookalike twins --records ... --twin-records results/memory/bm25_arms_hotpot_twins_experience.jsonl \\
        --dataset hotpot

Writes results/<dataset>_lookalike_pairs.csv and _lookalike_summary.csv, or
results/<dataset>_twins.csv.
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.stats import mcnemar, wilson_ci
from src.memory.arms import (arm_table, arms_of, hits, question_similarity, question_vectors,
                             read_ids, read_records)

BANDS: tuple[tuple[float, float], ...] = ((0.6, 1.01), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2))
TWIN_SEP = "::twin:"


def outcome_matrix(table: pd.DataFrame) -> np.ndarray:
    """questions x arms, binary hits, arms in arms_of order."""
    return np.stack([hits(table, a).to_numpy() for a in arms_of(table)], axis=1)


def best_arm(outcomes: np.ndarray, arms: Sequence[str]) -> list[str]:
    """The best arm per question: 'all' when every arm hits, 'none' when
    none does, else the hitting arms joined by '+' in arms order. Two
    questions have the same best arm when this label is the same."""
    labels = []
    for row in outcomes:
        on = [a for a, h in zip(arms, row) if h]
        labels.append("all" if len(on) == len(arms) else "none" if not on else "+".join(on))
    return labels


def natural_pairs(table: pd.DataFrame, vectors: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Each question's nearest other question, deduped to unordered pairs,
    with the similarity, both outcomes per arm and whether they agree."""
    n = len(table)
    text = table["question"].fillna("").tolist()
    everyone = np.arange(n)
    sims = question_similarity(text, everyone, everyone, vectors)
    np.fill_diagonal(sims, -np.inf)
    nearest = np.argmax(sims, axis=1)
    outcomes = outcome_matrix(table)
    arms = arms_of(table)
    best = best_arm(outcomes, arms)
    seen, rows = set(), []
    for i in range(n):
        j = int(nearest[i])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        a, b = key
        row = {"id_a": table["id"].iloc[a], "id_b": table["id"].iloc[b],
               "similarity": round(float(sims[a, b]), 4),
               "question_a": text[a], "question_b": text[b],
               "best_a": best[a], "best_b": best[b], "best_same": int(best[a] == best[b])}
        for k, arm in enumerate(arms):
            row[f"{arm}_a"], row[f"{arm}_b"] = int(outcomes[a, k]), int(outcomes[b, k])
            row[f"{arm}_same"] = int(outcomes[a, k] == outcomes[b, k])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def random_pairs(table: pd.DataFrame, n_pairs: int, seed: int = 13) -> pd.DataFrame:
    """n_pairs random unordered pairs of distinct questions, scored like
    natural_pairs (no similarity column)."""
    rng = np.random.RandomState(seed)
    n = len(table)
    outcomes = outcome_matrix(table)
    arms = arms_of(table)
    best = best_arm(outcomes, arms)
    rows = []
    for _ in range(n_pairs):
        a, b = rng.choice(n, 2, replace=False)
        row = {"best_same": int(best[a] == best[b])}
        for k, arm in enumerate(arms):
            row[f"{arm}_same"] = int(outcomes[a, k] == outcomes[b, k])
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_rows(pairs: pd.DataFrame, arms: Sequence[str], label: str) -> list[dict]:
    rows = []
    for what in [f"{a}_same" for a in arms] + ["best_same"]:
        k, n = int(pairs[what].sum()), len(pairs)
        lo, hi = wilson_ci(k, n)
        rows.append({"pairs": label, "n": n, "agreement": what[:-len("_same")],
                     "rate": round(k / n, 4) if n else float("nan"),
                     "lo": round(lo, 4), "hi": round(hi, 4)})
    return rows


def pair_summary(pairs: pd.DataFrame, table: pd.DataFrame,
                 bands: Sequence[tuple[float, float]] = BANDS, seed: int = 13,
                 n_random: int = 5000) -> pd.DataFrame:
    """Agreement rates per similarity band of the nearest-neighbour pairs,
    beside the same rates over random pairs."""
    arms = arms_of(table)
    rows = []
    for lo, hi in bands:
        band = pairs[(pairs["similarity"] >= lo) & (pairs["similarity"] < hi)]
        if len(band):
            rows += agreement_rows(band, arms, f"nearest[{lo:.1f},{min(hi, 1.0):.1f})")
    rows += agreement_rows(pairs, arms, "nearest_all")
    rows += agreement_rows(random_pairs(table, n_random, seed), arms, "random")
    return pd.DataFrame(rows)


def twin_of(twin_id: str) -> tuple[str, str]:
    """('<original id>', '<kind>') from '<original id>::twin:<kind>'."""
    if TWIN_SEP not in twin_id:
        raise ValueError(f"{twin_id!r} is not a twin id")
    original, kind = twin_id.rsplit(TWIN_SEP, 1)
    return original, kind


def twin_table(originals: pd.DataFrame, twins: pd.DataFrame) -> pd.DataFrame:
    """Per twin kind and arm: how the twin's outcome differs from its
    original's. hit_to_miss and miss_to_hit count flips, mcnemar_p tests
    their asymmetry, changed is the share of pairs whose outcome differs,
    best_moved the share whose best arm differs. The 'any' arm row counts a
    pair as changed when any arm's outcome differs."""
    arms = arms_of(originals)
    if arms_of(twins) != arms:
        raise ValueError(f"twins carry arms {arms_of(twins)}, originals {arms}")
    at = {i: n for n, i in enumerate(originals["id"])}
    o_out, t_out = outcome_matrix(originals), outcome_matrix(twins)
    o_best = best_arm(o_out, arms)
    t_best = best_arm(t_out, arms)
    kinds = {}
    for n, tid in enumerate(twins["id"]):
        original, kind = twin_of(tid)
        if original in at:
            kinds.setdefault(kind, []).append((at[original], n))
    rows = []
    for kind, pairs in sorted(kinds.items()):
        oi = np.array([p[0] for p in pairs])
        ti = np.array([p[1] for p in pairs])
        any_changed = np.zeros(len(pairs), dtype=bool)
        for k, arm in enumerate(arms):
            x, y = o_out[oi, k], t_out[ti, k]
            hit_to_miss, miss_to_hit, p = mcnemar(x.tolist(), y.tolist())
            changed = x != y
            any_changed |= changed
            rows.append({"kind": kind, "arm": arm, "n": len(pairs),
                         "original_rate": round(float(x.mean()), 4),
                         "twin_rate": round(float(y.mean()), 4),
                         "hit_to_miss": hit_to_miss, "miss_to_hit": miss_to_hit,
                         "mcnemar_p": round(p, 4), "changed": round(float(changed.mean()), 4)})
        moved = np.array([o_best[a] != t_best[b] for a, b in pairs])
        rows.append({"kind": kind, "arm": "any", "n": len(pairs),
                     "original_rate": float("nan"), "twin_rate": float("nan"),
                     "hit_to_miss": int(sum(r["hit_to_miss"] for r in rows if r["kind"] == kind)),
                     "miss_to_hit": int(sum(r["miss_to_hit"] for r in rows if r["kind"] == kind)),
                     "mcnemar_p": float("nan"), "changed": round(float(any_changed.mean()), 4),
                     "best_moved": round(float(moved.mean()), 4)})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["natural", "twins"])
    ap.add_argument("--records", required=True, help="experience jsonl of the arms run")
    ap.add_argument("--twin-records", default=None, help="experience jsonl of the twins run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outcome", default="all_gold_in_topk")
    ap.add_argument("--ids", default=None, help="fold file; only these originals")
    ap.add_argument("--dense-neighbours", action="store_true",
                    help="nearest by bge-small question embedding instead of tf-idf")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)

    header, records = read_records(args.records)
    ids = read_ids(args.ids) if args.ids else None
    table = arm_table(records, outcome=args.outcome, ids=ids)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.records}: run {header['run']}, {len(table)} questions, arms {arms_of(table)}")

    if args.what == "natural":
        vectors = question_vectors(table["question"].fillna("").tolist()) \
            if args.dense_neighbours else None
        pairs = natural_pairs(table, vectors)
        summary = pair_summary(pairs, table, seed=args.seed)
        tag = "_dense" if args.dense_neighbours else ""
        pairs.to_csv(out / f"{args.dataset}_lookalike{tag}_pairs.csv", index=False)
        summary.to_csv(out / f"{args.dataset}_lookalike{tag}_summary.csv", index=False)
        print(f"{len(pairs)} nearest-neighbour pairs")
        print(summary.to_string(index=False))
        return

    if not args.twin_records:
        raise SystemExit("twins needs --twin-records")
    _, twin_records = read_records(args.twin_records)
    twins = arm_table(twin_records, outcome=args.outcome)
    result = twin_table(table, twins)
    result.to_csv(out / f"{args.dataset}_twins.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
