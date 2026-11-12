"""Stage 5, induced drift: the stream switches dataset mid-way.
(docs/plan.md, Part B, Stage 5, "baselines on the same stream: ... induced
drift (switch dataset mid-stream)")

Hypothesis under test: a memory whose applicability boundaries revise as
contradicting experiences arrive should lose less, and recover sooner, when
the questions change regime than the same memory with fixed regions
(utility), and the shared router is the row that says whether the lesson
structure has anything to do with it. The stream is the decide-then-reveal
protocol of src.memory.stream run over the questions of one dataset in a
seeded order followed by the questions of another, so the same policies,
the same features (the 19 frozen process features; the dataset's question
type one-hots are left out, since on a mixed stream they would announce
the dataset and the point is whether the regime change is noticed from the
process features and the revealed outcomes) and the same default arm on
both sides. Nothing tells a policy that the switch happened.

What each policy is scored on, per seeded order:

  before          its rate on the first segment; the same number it would
                  get on that dataset alone, because nothing from the
                  second segment can reach a decision made before it
  after           its rate on the second segment with whatever it carried
                  over the switch (the stale run)
  cold            its rate on the same second-segment questions in the same
                  order, started fresh at the switch (the cold run): what
                  the memory would have done had it forgotten everything
  transfer        after - cold: what carrying the first dataset's memory
                  into the second is worth, negative when it hurts. The
                  fixed arms and the probe have transfer 0 by construction
  window          the same three on the first `window` questions after the
                  switch, where a memory that adapts and one that does not
                  differ most; the rest of the segment is `late`
  revisions       for the lesson memories: lessons alive at the switch, how
                  many of those retire inside the window, and the revision
                  rate (logged operator steps per question) before the
                  switch, inside the window and after it

Every rate is paired per order: the stale run against the cold run on the
same questions (McNemar, the largest p over orders), and the stale run
against the fixed default on the second segment. Both directions of the
switch are run, since a regime change from an easy default to a hard one
is not the reverse of the other.

    python -m src.memory.drift --records-a results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset-a hotpot --records-b results/memory/bm25_arms_musique_dev_experience.jsonl \\
        --dataset-b musique [--default bm25] [--window 200] [--n-per-segment N]

Writes results/<a>_to_<b>_drift.csv and results/<b>_to_<a>_drift.csv, one
row per policy, and the first boundary memory's revision log on the first
order of each direction as results/<a>_to_<b>_drift_revisions.csv.
"""
import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.stats import mcnemar
from src.memory.arms import arm_table, arms_of, read_records
from src.memory.stream import (OPERATORS, FixedArm, Stream, StreamResult, best_fixed_arm,
                               default_policies, make_stream, rule_policies, run_stream)

SEGMENT = "segment"


def blank_qtype(table: pd.DataFrame) -> pd.DataFrame:
    """The table with the question type removed, so that make_stream builds
    one constant qtype column rather than one per dataset."""
    out = table.copy()
    out["qtype"] = ""
    return out


def drift_orders(n_a: int, n_b: int, seeds: Sequence[int],
                 n_per_segment: Optional[int] = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per seed, a seeded order over the first table's rows and one over the
    second's, each cut to n_per_segment (default: the smaller table) so the
    two segments are the same length."""
    n = min(n_a, n_b) if n_per_segment is None else n_per_segment
    if n > min(n_a, n_b):
        raise ValueError(f"n_per_segment {n} exceeds a table of {min(n_a, n_b)} questions")
    orders = []
    for s in seeds:
        rng = np.random.RandomState(s)
        orders.append((rng.permutation(n_a)[:n], rng.permutation(n_b)[:n]))
    return orders


def joined_table(table_a: pd.DataFrame, table_b: pd.DataFrame) -> pd.DataFrame:
    """Both tables as one, with a segment column (0 for the first, 1 for the
    second) and the question types blanked. The arms must match."""
    if arms_of(table_a) != arms_of(table_b):
        raise ValueError(f"the arms differ: {arms_of(table_a)} against {arms_of(table_b)}")
    a, b = blank_qtype(table_a), blank_qtype(table_b)
    a[SEGMENT], b[SEGMENT] = 0, 1
    cols = [c for c in a.columns if c in b.columns]
    return pd.concat([a[cols], b[cols]], ignore_index=True)


def drift_streams(joined: pd.DataFrame, stage: str, order_a: np.ndarray, order_b: np.ndarray,
                  seed: int) -> tuple[Stream, Stream]:
    """The stale stream (segment 0 in order_a, then segment 1 in order_b) and
    the cold stream (segment 1 alone, in the same order)."""
    n_a = int((joined[SEGMENT] == 0).sum())
    stale = make_stream(joined, stage, np.concatenate([order_a, n_a + order_b]), seed=seed)
    cold_table = joined[joined[SEGMENT] == 1].reset_index(drop=True)
    cold = make_stream(cold_table, stage, order_b, seed=seed)
    return stale, cold


def segment_rates(H: np.ndarray, result: StreamResult, lo: int, hi: int, d: int) -> dict:
    """The policy's rate, cost and the conditioned rates over stream
    positions [lo, hi): H is the hit matrix in stream order."""
    hit = (result.hit[lo:hi] >= 1).astype(int)
    Hs = H[lo:hi]
    other = np.delete(Hs, d, axis=1).max(axis=1)
    recoverable = (Hs[:, d] == 0) & (other == 1)
    fired = result.fired[lo:hi]
    ok = result.lesson_arm_ok[lo:hi]

    def rate(num, den):
        return float(num.sum() / den.sum()) if den.sum() else math.nan

    return {"rate": float(hit.mean()) if len(hit) else math.nan,
            "cost": float(result.cost[lo:hi].mean()) if len(hit) else math.nan,
            "gain": float(hit.mean() - Hs[:, d].mean()) if len(hit) else math.nan,
            "fired": float(fired.mean()) if len(hit) else math.nan,
            "false_application": rate(fired & ~ok, fired),
            "harm": rate(result.harm[lo:hi], fired),
            "repeated_failure": rate(recoverable & (hit == 0), recoverable)}


def revision_counts(log: list, switch: int, window: int, n_after: int) -> dict:
    """From a lesson memory's log: lessons alive at the switch, how many of
    them retire (or are split away) inside the window, and the revision rate
    per question before the switch, in the window and after it."""
    if not log:
        return {}
    created, gone = set(), set()
    for e in log:
        if e["t"] >= switch:
            break
        if e["op"] == "create":
            created.add(e["lesson"])
        elif e["op"] in ("retire", "split"):
            gone.add(e["lesson"])
    alive = created - gone
    retired_in_window = {e["lesson"] for e in log if e["lesson"] in alive
                         and e["op"] in ("retire", "split") and switch <= e["t"] < switch + window}
    retired_after = {e["lesson"] for e in log if e["lesson"] in alive
                     and e["op"] in ("retire", "split") and e["t"] >= switch}
    steps = lambda lo, hi: sum(1 for e in log if lo <= e["t"] < hi and e["op"] != "create")  # noqa: E731
    w = min(window, n_after)
    return {"alive_at_switch": len(alive),
            "alive_retired_window": len(retired_in_window),
            "alive_retired_after": len(retired_after),
            "revisions_per_q_before": steps(0, switch) / switch if switch else math.nan,
            "revisions_per_q_window": steps(switch, switch + w) / w if w else math.nan,
            "revisions_per_q_late": (steps(switch + w, switch + n_after) / (n_after - w)
                                     if n_after > w else math.nan)}


def drift_rules() -> list[tuple]:
    """The Stage 4 static conditions (stream.bm25_rules) written without the
    log's top-k, since the two segments may use different depths: a thin list
    is one with fewer scored hits than hits returned, which is the same rule
    whenever the pool holds at least k chunks. These are the no-persistence
    control on the drift stream: nothing is kept across questions, so a
    regime change cannot reach them."""
    return [
        ("thin_list", lambda f: f["f:n_scored_hits"] < f["f:n_hits"], "hybrid"),
        ("flat_scores", lambda f: f["f:score_entropy"] > 0.95 and f["f:score_ratio"] > 0.9,
         "hybrid"),
        ("no_title", lambda f: f["f:query_title_mentions"] == 0, "dense"),
    ]


def evaluate_drift(table_a: pd.DataFrame, table_b: pd.DataFrame, default: str,
                   seeds: Sequence[int] = (13, 17, 19, 23, 29),
                   policies: Optional[Sequence[tuple]] = None, window: int = 200,
                   n_per_segment: Optional[int] = None,
                   rules: Optional[Sequence[tuple]] = None) -> tuple[pd.DataFrame, list[dict]]:
    """Every policy on every seeded drift order, stale and cold. Returns the
    summary table (means over orders, the paired tests as the largest p)
    and the first boundary memory's log on the first order."""
    joined = joined_table(table_a, table_b)
    arms = arms_of(joined)
    if default not in arms:
        raise ValueError(f"default {default!r} is not one of the arms {arms}")
    d = arms.index(default)
    best_a, best_b = best_fixed_arm(table_a, default), best_fixed_arm(table_b, default)
    orders = drift_orders(len(table_a), len(table_b), seeds, n_per_segment)
    pols = [(f"fixed:{a}", "query", lambda s, dd, arm=a: FixedArm(s, dd, arm)) for a in arms] \
        + list(policies if policies is not None else default_policies(default))
    if rules:
        pols += rule_policies({"primary_hits": rules})
    rows, first_log = [], []
    cache: dict = {}
    for label, stage, factory in pols:
        if stage in ("primary_hits", "history") and default != arms[0]:
            continue   # the primary-hits features belong to the logs' primary arm
        per_order = []
        for (oa, ob), seed in zip(orders, seeds):
            key = (stage, seed)
            if key not in cache:
                cache[key] = drift_streams(joined, stage, oa, ob, seed)
            stale_s, cold_s = cache[key]
            stale = run_stream(stale_s, factory(stale_s, default))
            cold = run_stream(cold_s, factory(cold_s, default))
            switch, n = len(oa), len(ob)
            H = stale_s.H[stale_s.order]
            w = min(window, n)
            after = segment_rates(H, stale, switch, switch + n, d)
            after_cold = segment_rates(cold_s.H[cold_s.order], cold, 0, n, d)
            win = segment_rates(H, stale, switch, switch + w, d)
            win_cold = segment_rates(cold_s.H[cold_s.order], cold, 0, w, d)
            late = segment_rates(H, stale, switch + w, switch + n, d)
            late_cold = segment_rates(cold_s.H[cold_s.order], cold, w, n, d)
            sh, ch = (stale.hit[switch:] >= 1).astype(int), (cold.hit >= 1).astype(int)
            only_cold, only_stale, p = mcnemar(ch.tolist(), sh.tolist())
            _, _, p_win = mcnemar(ch[:w].tolist(), sh[:w].tolist())
            _, _, p_default = mcnemar(H[switch:, d].tolist(), sh.tolist())
            row = {"before_rate": segment_rates(H, stale, 0, switch, d)["rate"],
                   "before_gain": segment_rates(H, stale, 0, switch, d)["gain"],
                   "after_rate": after["rate"], "after_cost": after["cost"],
                   "after_gain": after["gain"], "cold_rate": after_cold["rate"],
                   "cold_cost": after_cold["cost"], "transfer": after["rate"] - after_cold["rate"],
                   "only_stale": only_stale, "only_cold": only_cold, "p_vs_cold": p,
                   "p_vs_default": p_default,
                   "window_rate": win["rate"], "window_cold": win_cold["rate"],
                   "window_transfer": win["rate"] - win_cold["rate"], "p_window_vs_cold": p_win,
                   "late_rate": late["rate"], "late_cold": late_cold["rate"],
                   "late_transfer": late["rate"] - late_cold["rate"],
                   "after_fired": after["fired"], "cold_fired": after_cold["fired"],
                   "window_fired": win["fired"], "window_cold_fired": win_cold["fired"],
                   "after_false_application": after["false_application"],
                   "cold_false_application": after_cold["false_application"],
                   "window_false_application": win["false_application"],
                   "window_cold_false_application": win_cold["false_application"],
                   "after_harm": after["harm"], "cold_harm": after_cold["harm"],
                   "after_repeated_failure": after["repeated_failure"],
                   "cold_repeated_failure": after_cold["repeated_failure"],
                   **revision_counts(stale.log, switch, window, n),
                   **{f"op_{o}": stale.state.get(f"op_{o}", math.nan) for o in OPERATORS}}
            per_order.append(row)
            if stale.log and label.startswith("boundary") and not first_log:
                first_log = [{"policy": label, "switch": switch, **e} for e in stale.log]
        df = pd.DataFrame(per_order)
        out = {"policy": label, "stage": stage, "default": default, "n_orders": len(df),
               "best_fixed_a": best_a, "best_fixed_b": best_b, "window": window}
        for c in df.columns:
            if c.startswith("p_"):
                out[f"{c}_max"] = round(float(df[c].max()), 4)
            elif c.startswith("only_"):
                out[c] = round(float(df[c].mean()), 1)
            elif c.startswith("op_") or c.startswith("alive"):
                out[c] = round(float(df[c].mean()), 1)
            else:
                out[c] = round(float(df[c].mean()), 4)
        for c in ("transfer", "window_transfer"):
            out[f"{c}_sd"] = round(float(df[c].std(ddof=0)), 4)
        rows.append(out)
    return pd.DataFrame(rows), first_log


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-a", required=True, help="experience jsonl of the first dataset")
    ap.add_argument("--dataset-a", required=True)
    ap.add_argument("--records-b", required=True, help="experience jsonl of the second dataset")
    ap.add_argument("--dataset-b", required=True)
    ap.add_argument("--outcome", default="all_gold_in_topk")
    ap.add_argument("--default", default=None, help="the default arm (the logs' primary)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 19, 23, 29])
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--n-per-segment", type=int, default=None)
    ap.add_argument("--one-direction", action="store_true", help="a to b only")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    tables = {}
    for name, path in ((args.dataset_a, args.records_a), (args.dataset_b, args.records_b)):
        _, records = read_records(path)
        tables[name] = arm_table(records, outcome=args.outcome)
        print(f"{path}: {len(tables[name])} questions, arms {arms_of(tables[name])}")
    default = args.default or arms_of(tables[args.dataset_a])[0]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    directions = [(args.dataset_a, args.dataset_b)]
    if not args.one_direction:
        directions.append((args.dataset_b, args.dataset_a))
    for a, b in directions:
        summary, log = evaluate_drift(tables[a], tables[b], default, seeds=args.seeds,
                                      window=args.window, n_per_segment=args.n_per_segment,
                                      rules=drift_rules())
        summary.insert(0, "direction", f"{a}->{b}")
        summary.to_csv(out / f"{a}_to_{b}_drift.csv", index=False)
        pd.DataFrame(log).to_csv(out / f"{a}_to_{b}_drift_revisions.csv", index=False)
        print(f"\n{a} -> {b}")
        print(summary[["policy", "stage", "before_rate", "after_rate", "cold_rate", "transfer",
                       "p_vs_cold_max", "window_rate", "window_cold", "window_transfer",
                       "p_window_vs_cold_max", "after_cost", "cold_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
