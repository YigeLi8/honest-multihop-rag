"""Stage 3: failure typing by rule, and what recovered each failure.
(docs/plan.md, Part B)

The failure-memory baseline stores past retrieval failures with a type and
the arm that recovered them. This module is the typing half: a fixed,
ordered set of rules over one hop-0 record of the primary arm. The type is
an outcome label, computed from the gold ids like every other outcome, so it
is never a feature; the applicability half (deciding, gold-free, that a new
question is heading for the same failure) is the memory's job, not this
file's.

Rules, first match wins, on the primary arm's top-k against the gold chunks:

  none          every gold chunk is in the top-k
  near_miss     every missed gold chunk sits within 2k of the arm's full-pool
                ranking (gold_ranks): a deeper list would have it
  named_miss    a missed gold chunk's paragraph is named in the query; the
                arm had the name and still missed the chunk (on sentence
                pools often another sentence of the same paragraph is in)
  bridge_miss   at least one gold chunk was hit and every missed one is in a
                paragraph the query does not name: the first hop worked and
                the hop through it did not
  total_miss    no gold chunk in the top-k and none of them named

Hypothesis the table tests: failure types differ in what recovers them (the
share recovered by dense, by hybrid, by neither). If every type is
recovered at the same rate by the same arm, the type adds nothing over
"bm25 missed" and the failure memory collapses into the similarity memory.

The 50-case audit file is for checking the rules by hand: each row shows the
query, the type, the missed gold chunks with their full-pool rank and the
top-k, with an empty audit column to fill in. The plan item is not ticked
until that audit is done and its disagreement rate recorded.

    python -m src.memory.failures --records results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset hotpot [--ids ...] [--audit 50]

Writes results/<dataset>_failure_types.csv and, with --audit, _failure_audit.csv.
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.memory.arms import arm_name, read_ids, read_records
from src.memory.features import normalise_title, tokens

FAILURE_TYPES: tuple[str, ...] = ("none", "near_miss", "named_miss", "bridge_miss", "total_miss")
NEAR_FACTOR = 2    # near_miss: every missed gold within NEAR_FACTOR * k of the full ranking


def paragraph_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0]


def named_in(query: str, title: str) -> bool:
    """Is the paragraph title (without its disambiguator) in the query, as
    a token string? The same test as the query_title_mentions feature."""
    name = normalise_title(title)
    return bool(name) and f" {name} " in f" {' '.join(tokens(query))} "


def failure_type(record: dict) -> str:
    """The rule-based type of the primary arm's hop-0 outcome."""
    out = record["hop_outcome"]
    if out["all_gold_in_topk"]:
        return "none"
    hit = set(out["gold_hit_ids"])
    ranks = out.get("gold_ranks") or {}
    gold = list(ranks) if ranks else list(hit)
    if not ranks:
        # without the full ranking the missed ids are unknown by id; the
        # backfill and live runs always carry it, the stub fixture does not
        missed_titles = []
    else:
        missed = [g for g in gold if g not in hit]
        k = record["strategy"].get("top_k") or len(record["retrieved"])
        if missed and all(ranks[g] is not None and ranks[g][0] <= NEAR_FACTOR * k for g in missed):
            return "near_miss"
        missed_titles = [paragraph_of(g) for g in missed]
    query = record["query"]
    if any(named_in(query, t) for t in missed_titles):
        return "named_miss"
    if hit:
        return "bridge_miss"
    return "total_miss"


def recovered_by(record: dict) -> str:
    """The shadow arms whose top-k had every gold chunk, joined by '+', or
    'none'."""
    primary = arm_name(record["strategy"])
    arms = [a for a, block in record["shadow"].items()
            if a != primary and block.get("all_gold_in_topk")]
    return "+".join(arms) if arms else "none"


def typed_rows(records: Sequence[dict], ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per hop-0 record: id, query, failure_type, recovered_by and
    one column per shadow arm with its hit."""
    want = None if ids is None else set(ids)
    rows = []
    for r in records:
        if r["hop"] != 0 or (want is not None and r["id"] not in want):
            continue
        primary = arm_name(r["strategy"])
        row = {"id": r["id"], "qtype": r["qtype"], "query": r["query"],
               "failure_type": failure_type(r), "recovered_by": recovered_by(r)}
        for a, block in r["shadow"].items():
            if a != primary:
                row[f"hit:{a}"] = int(bool(block.get("all_gold_in_topk")))
        rows.append(row)
    if not rows:
        raise SystemExit("no hop-0 records")
    return pd.DataFrame(rows)


def failure_table(typed: pd.DataFrame) -> pd.DataFrame:
    """Per failure type: count, share of all questions, and the share of
    that type recovered by each shadow arm, by any and by none."""
    arms = [c[len("hit:"):] for c in typed.columns if c.startswith("hit:")]
    rows = []
    for ftype in FAILURE_TYPES + ("all_failures",):
        sub = typed[typed["failure_type"] != "none"] if ftype == "all_failures" \
            else typed[typed["failure_type"] == ftype]
        row = {"failure_type": ftype, "n": len(sub),
               "share": round(len(sub) / len(typed), 4) if len(typed) else float("nan")}
        for a in arms:
            row[f"recovered_by_{a}"] = round(float(sub[f"hit:{a}"].mean()), 4) if len(sub) else float("nan")
        if arms:
            any_ = sub[[f"hit:{a}" for a in arms]].max(axis=1) if len(sub) else pd.Series(dtype=float)
            row["recovered_by_any"] = round(float(any_.mean()), 4) if len(sub) else float("nan")
            row["recovered_by_none"] = round(1 - row["recovered_by_any"], 4) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def audit_sample(records: Sequence[dict], n: int = 50, seed: int = 13,
                 ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """n failures drawn at random (seeded), stratified by nothing, with what
    a reader needs to check the type by hand."""
    want = None if ids is None else set(ids)
    failures = [r for r in records if r["hop"] == 0 and not r["hop_outcome"]["all_gold_in_topk"]
                and (want is None or r["id"] in want)]
    rng = np.random.RandomState(seed)
    picked = [failures[i] for i in sorted(rng.choice(len(failures), min(n, len(failures)),
                                                     replace=False))]
    rows = []
    for r in picked:
        out = r["hop_outcome"]
        hit = set(out["gold_hit_ids"])
        ranks = out.get("gold_ranks") or {}
        missed = {g: (ranks[g][0] if ranks.get(g) else None) for g in ranks if g not in hit}
        rows.append({"id": r["id"], "query": r["query"], "failure_type": failure_type(r),
                     "recovered_by": recovered_by(r),
                     "gold_hit": " | ".join(sorted(hit)),
                     "gold_missed_rank": " | ".join(f"{g} @{rank}" for g, rank in missed.items()),
                     "topk": " | ".join(h["chunk_id"] for h in r["retrieved"]),
                     "audit_ok": "", "audit_note": ""})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", default=None, help="fold file; only these questions")
    ap.add_argument("--audit", type=int, default=0, help="write this many failures for a hand audit")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    header, records = read_records(args.records)
    ids = read_ids(args.ids) if args.ids else None
    typed = typed_rows(records, ids)
    table = failure_table(typed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{args.dataset}_failure_types.csv", index=False)
    print(f"{args.records}: run {header['run']}, {len(typed)} questions")
    print(table.to_string(index=False))
    cross = pd.crosstab(typed["failure_type"], typed["recovered_by"])
    print("\nrecovered_by per type (counts)")
    print(cross.to_string())
    if args.audit:
        audit = audit_sample(records, args.audit, args.seed, ids)
        audit.to_csv(out / f"{args.dataset}_failure_audit.csv", index=False)
        print(f"\naudit sample: {len(audit)} rows in {out / f'{args.dataset}_failure_audit.csv'}")


if __name__ == "__main__":
    main()
