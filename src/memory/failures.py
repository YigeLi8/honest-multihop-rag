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

One label per failure from ordered rules is exposed to the order (paper log,
2026-11-11): a failure that is both a near miss and a named miss is typed by
whichever rule is tested first, and the type shares, and the recovery rates
per type, can move with the order. Two checks go with the table, so that
the reading "near misses want the fused list, named and bridge misses want
dense" is shown to survive it: the co-occurrence of the rules' conditions
over failures (rule_flags: near, named and partial hold independently of
each other), and the type shares, the recovery rate per type and the type x
recovering-arm chi-square under every order of the three tests
(order_table). The default order is near, named, partial; bridge_miss and
total_miss are the two ends of the partial test and never move relative to
each other.

    python -m src.memory.failures --records results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset hotpot [--ids ...] [--audit 50]

Writes results/<dataset>_failure_types.csv, _failure_cooccurrence.csv,
_failure_orders.csv and, with --audit, _failure_audit.csv.
"""
import argparse
import re
from itertools import combinations, permutations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.memory.arms import arm_name, read_ids, read_records
from src.memory.features import normalise_title, tokens

FAILURE_TYPES: tuple[str, ...] = ("none", "near_miss", "named_miss", "bridge_miss", "total_miss")
NEAR_FACTOR = 2    # near_miss: every missed gold within NEAR_FACTOR * k of the full ranking


_MUSIQUE_PARAGRAPH = re.compile(r"p\d+")


def paragraph_of(chunk_id: str) -> str:
    """The paragraph title a chunk id names. hotpot and 2wiki ids are
    `Title::sent_idx`; musique ids are `p{idx}::{title}` (data/prepare_musique),
    where the title is the second part. Until 5 Oct this took the first part
    on every dataset, so on musique the named-miss rule could never fire."""
    head, _, tail = chunk_id.partition("::")
    if tail and _MUSIQUE_PARAGRAPH.fullmatch(head):
        return tail
    return chunk_id.rsplit("::", 1)[0]


def named_in(query: str, title: str) -> bool:
    """Is the paragraph title (without its disambiguator) in the query, as
    a token string? The same test as the query_title_mentions feature."""
    name = normalise_title(title)
    return bool(name) and f" {name} " in f" {' '.join(tokens(query))} "


RULES: tuple[str, ...] = ("near", "named", "partial")
DEFAULT_ORDER: tuple[str, ...] = RULES


def rule_flags(record: dict) -> Optional[dict]:
    """The three conditions of the rules, each tested on its own: near (every
    missed gold chunk within NEAR_FACTOR * k of the full-pool ranking), named
    (a missed gold paragraph is named in the query), partial (some gold chunk
    was hit). None when the record is not a failure."""
    out = record["hop_outcome"]
    if out["all_gold_in_topk"]:
        return None
    hit = set(out["gold_hit_ids"])
    ranks = out.get("gold_ranks") or {}
    if not ranks:
        # without the full ranking the missed ids are unknown by id; the
        # backfill and live runs always carry it, the stub fixture does not
        return {"near": False, "named": False, "partial": bool(hit)}
    missed = [g for g in ranks if g not in hit]
    k = record["strategy"].get("top_k") or len(record["retrieved"])
    near = bool(missed) and all(ranks[g] is not None and ranks[g][0] <= NEAR_FACTOR * k
                                for g in missed)
    named = any(named_in(record["query"], paragraph_of(g)) for g in missed)
    return {"near": near, "named": named, "partial": bool(hit)}


def type_from_flags(flags: Optional[dict], order: Sequence[str] = DEFAULT_ORDER) -> str:
    """The type the ordered rules give: the first test in `order` that holds
    names it (near -> near_miss, named -> named_miss, partial -> bridge_miss);
    when none holds it is total_miss."""
    if flags is None:
        return "none"
    if set(order) != set(RULES) or len(order) != len(RULES):
        raise ValueError(f"order must be a permutation of {RULES}, not {tuple(order)}")
    for rule in order:
        if flags[rule]:
            return {"near": "near_miss", "named": "named_miss", "partial": "bridge_miss"}[rule]
    return "total_miss"


def failure_type(record: dict, order: Sequence[str] = DEFAULT_ORDER) -> str:
    """The rule-based type of the primary arm's hop-0 outcome."""
    return type_from_flags(rule_flags(record), order)


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
        flags = rule_flags(r)
        row = {"id": r["id"], "qtype": r["qtype"], "query": r["query"],
               "failure_type": type_from_flags(flags), "recovered_by": recovered_by(r)}
        row.update({f"rule:{k}": int(v) for k, v in (flags or {}).items()})
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


def cooccurrence(typed: pd.DataFrame) -> pd.DataFrame:
    """Over the failures: how often each rule's condition holds, and how
    often each pair and all three hold together. A failure that holds more
    than one condition is one whose type depends on the rule order."""
    fails = typed[typed["failure_type"] != "none"]
    n = len(fails)
    cols = {r: fails[f"rule:{r}"].astype(bool) for r in RULES if f"rule:{r}" in fails}
    rows = []
    for label, mask in ([(r, cols[r]) for r in cols]
                        + [(f"{a}&{b}", cols[a] & cols[b]) for a, b in combinations(cols, 2)]
                        + ([("near&named&partial", cols["near"] & cols["named"] & cols["partial"])]
                           if len(cols) == 3 else [])):
        rows.append({"rules": label, "n": int(mask.sum()),
                     "share_of_failures": round(float(mask.mean()), 4) if n else float("nan")})
    held = sum(cols[r].astype(int) for r in cols) if cols else pd.Series(dtype=int)
    rows.append({"rules": "more_than_one", "n": int((held > 1).sum()),
                 "share_of_failures": round(float((held > 1).mean()), 4) if n else float("nan")})
    return pd.DataFrame(rows)


def order_table(typed: pd.DataFrame) -> pd.DataFrame:
    """Per order of the three tests: the type shares among failures, the
    share of each type recovered by any shadow arm and by each arm, and the
    chi-square p of type x recovering arm. The hypothesis the Stage 3 table
    rests on (types differ in what recovers them) has to hold under every
    order, or it is a property of the order."""
    from scipy.stats import chi2_contingency
    fails = typed[typed["failure_type"] != "none"].copy()
    arms = [c[len("hit:"):] for c in typed.columns if c.startswith("hit:")]
    rows = []
    for order in permutations(RULES):
        flags = [{r: bool(row[f"rule:{r}"]) for r in RULES} for _, row in fails.iterrows()]
        types = pd.Series([type_from_flags(f, order) for f in flags], index=fails.index)
        row = {"order": ">".join(order), "n_failures": len(fails)}
        for t in FAILURE_TYPES[1:]:
            sub = fails[types == t]
            row[f"share_{t}"] = round(len(sub) / len(fails), 4) if len(fails) else float("nan")
            for a in arms:
                row[f"{t}_by_{a}"] = round(float(sub[f"hit:{a}"].mean()), 4) if len(sub) else float("nan")
            if arms:
                any_ = sub[[f"hit:{a}" for a in arms]].max(axis=1)
                row[f"{t}_by_any"] = round(float(any_.mean()), 4) if len(sub) else float("nan")
        cross = pd.crosstab(types, fails["recovered_by"])
        if cross.shape[0] > 1 and cross.shape[1] > 1:
            chi, p, dof, _ = chi2_contingency(cross)
            row.update({"chi2": round(float(chi), 2), "dof": int(dof), "p": float(f"{p:.3g}")})
        else:
            row.update({"chi2": float("nan"), "dof": 0, "p": float("nan")})
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
    co = cooccurrence(typed)
    co.to_csv(out / f"{args.dataset}_failure_cooccurrence.csv", index=False)
    print("\nrule conditions over failures")
    print(co.to_string(index=False))
    orders = order_table(typed)
    orders.to_csv(out / f"{args.dataset}_failure_orders.csv", index=False)
    print("\ntype shares and recovery under every rule order")
    print(orders.to_string(index=False))
    if args.audit:
        audit = audit_sample(records, args.audit, args.seed, ids)
        audit.to_csv(out / f"{args.dataset}_failure_audit.csv", index=False)
        print(f"\naudit sample: {len(audit)} rows in {out / f'{args.dataset}_failure_audit.csv'}")


if __name__ == "__main__":
    main()
