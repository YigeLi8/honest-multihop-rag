"""Stage 1b: is there anything for a memory to learn? (docs/plan.md, Part B)

Reads the experience log of a single-hop run that carried the other
first-stage retrievers as shadow arms (configs/*_arms_*.yaml) and answers
three questions about retrieval-level outcomes before any memory exists:

1. Discordance. How often do the arms disagree on a question? If one arm
   dominates and the discordant set is small, there is no routing decision
   to learn and the memory study stops at that table.
2. Ceiling. Every arm is scored on every question, so the best arm per
   question is known, the oracle rate is exact and each fixed arm's regret
   against it is exact. The ceiling bounds what any memory can gain.
3. Predictability. How well can an arm's hit, and which of two arms hits
   where they disagree, be predicted from the frozen features, held out, at
   each feature stage (size control, qtype, query, pool, primary hits,
   probe)? A high AUC from the query or pool stage alone means a fixed rule
   or one shared router already has the information and a per-lesson
   boundary has nothing to add; the plan says to report that, not tune. The
   size control (hop, pool size, number of hits) is the MemSafe check: how
   much of any AUC is the trivial predictors. qtype is the 2wiki template
   control.

Hypothesis under test: the best arm varies by question (oracle above the
best single arm, paired) and that variation is only partly predictable from
the arm-free features (held-out AUC well below 1 at the query and pool
stages). Both must hold for Stages 2-5 to have a problem to work on.

Nothing is fit and scored on the same questions: the questions are split in
halves by a seeded permutation and every model is fit on one half and scored
on the other, both ways, over several seeds; or fit on one fold file and
scored on another. Outcomes are retrieval-level only. Only hop-0 records are
used, where a shadow arm's outcome is a question-level outcome (at later hops
it depends on the primary arm's trajectory).

    python -m src.memory.arms --records results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset hotpot [--ids results/folds/hotpot_explore.txt] \\
        [--train-ids results/folds/hotpot_explore.txt --eval-ids results/folds/hotpot_confirm.txt]

Writes results/<dataset>_arms_discordance.csv, _ceiling.csv, _predictability.csv
and _routed.csv (what a shared router over the same features gets, held out,
and what nearest past questions get: the similarity-memory baseline).
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.stats import mcnemar, paired_bootstrap
from src.memory.features import FEATURE_STAGE, FEATURES, PROBE_FEATURES

OUTCOMES = ("all_gold_in_topk", "hop_para_recall", "aligned_hit")

# the feature stages, cumulative in this order (FEATURE_STAGE names what each
# feature needs; a stage holds every feature available at or before it)
STAGES = ("query", "pool", "history", "primary_hits")
SIZE_CONTROL = ("hop", "n_pool_chunks", "n_hits")

# fit nothing with fewer questions than this in either half, or fewer
# positives / negatives than this in the half being fit
MIN_FIT = 20
MIN_CLASS = 5


def read_records(path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines or not lines[0].get("header"):
        raise SystemExit(f"{path}: not an experience log (no header line)")
    return lines[0], lines[1:]


def arm_name(strategy: dict) -> str:
    """bm25 | dense | hybrid, with +rerank when a reranker cut the list."""
    name = strategy.get("method", "")
    return f"{name}+rerank" if strategy.get("rerank_enabled") else name


def arm_table(records: Sequence[dict], outcome: str = "all_gold_in_topk",
              ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per question from its hop-0 record: the outcome of the primary
    arm and of every shadow arm (arm:<name> columns, floats), the features
    (f:<name>) and the probe features (probe:<arm>:<name>).

    outcome   a key of arm_outcome (OUTCOMES); aligned_hit is None without an
              ordered gold path and such questions are dropped for it
    ids       keep these questions, in this order
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
    want = None if ids is None else {i: n for n, i in enumerate(ids)}
    rows = []
    for r in records:
        if r["hop"] != 0 or (want is not None and r["id"] not in want):
            continue
        primary = arm_name(r["strategy"])
        values = {primary: r["hop_outcome"].get(outcome)}
        for arm, block in r["shadow"].items():
            if arm != primary:        # the same ranking twice says nothing
                values[arm] = block.get(outcome)
        if any(v is None for v in values.values()):
            continue
        row = {"id": r["id"], "qtype": r["qtype"], "question": r["query"],
               "n_gold": r["hop_outcome"]["n_gold"]}
        row.update({f"arm:{a}": float(v) for a, v in values.items()})
        row.update({f"f:{k}": float(v) for k, v in r["features"].items()})
        for arm, probe in r["probe_features"].items():
            row.update({f"probe:{arm}:{k}": float(probe[k]) for k in PROBE_FEATURES})
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no hop-0 records with every arm scored")
    if want is not None:
        table = table.sort_values("id", key=lambda s: s.map(want)).reset_index(drop=True)
    arms = arm_columns(table)
    if table[arms].isna().any().any():
        raise SystemExit("an arm is missing on some questions; every record needs the same arms")
    return table


def arm_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith("arm:")]


def arms_of(table: pd.DataFrame) -> list[str]:
    return [c[len("arm:"):] for c in arm_columns(table)]


def hits(table: pd.DataFrame, arm: str) -> pd.Series:
    """The arm's outcome as a hit: 1 when the outcome is complete (all gold
    in top-k, every gold paragraph, the aligned chunk), else 0. The binary
    analyses (discordance, predictability) read this; ceiling keeps the
    graded value."""
    return (table[f"arm:{arm}"] >= 1.0).astype(int)


def discordance(table: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement of the arms: both hit, only one, neither, and the
    exact McNemar p on the discordant pairs."""
    rows = []
    for a, b in combinations(arms_of(table), 2):
        x, y = hits(table, a), hits(table, b)
        only_a, only_b, p = mcnemar(x.tolist(), y.tolist())
        rows.append({"arm_a": a, "arm_b": b, "n": len(table),
                     "both": int((x & y).sum()), "only_a": only_a, "only_b": only_b,
                     "neither": int((~x.astype(bool) & ~y.astype(bool)).sum()),
                     "discordant": round((only_a + only_b) / len(table), 4),
                     "mcnemar_p": round(p, 4)})
    return pd.DataFrame(rows)


def ceiling(table: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    """Each arm's rate, the oracle (best arm per question) and the regret of
    every fixed arm against it, with a paired bootstrap interval on the
    oracle's gain over the best single arm. all_arms is the rate at which
    every arm hits, the questions no routing decision can lose."""
    arms = arms_of(table)
    scores = {a: table[f"arm:{a}"].tolist() for a in arms}
    oracle = table[arm_columns(table)].max(axis=1).tolist()
    every = table[arm_columns(table)].min(axis=1).tolist()
    best = max(arms, key=lambda a: sum(scores[a]))
    rows = []
    for a in arms:
        gain, lo, hi = paired_bootstrap(scores[a], oracle, seed=seed)
        rows.append({"arm": a, "n": len(table), "rate": round(float(np.mean(scores[a])), 4),
                     "regret": round(gain, 4), "regret_lo": round(lo, 4),
                     "regret_hi": round(hi, 4), "best_single": a == best})
    rows.append({"arm": "oracle", "n": len(table), "rate": round(float(np.mean(oracle)), 4),
                 "regret": 0.0, "regret_lo": 0.0, "regret_hi": 0.0, "best_single": False})
    rows.append({"arm": "all_arms", "n": len(table), "rate": round(float(np.mean(every)), 4),
                 "regret": round(float(np.mean(oracle) - np.mean(every)), 4),
                 "regret_lo": math.nan, "regret_hi": math.nan, "best_single": False})
    return pd.DataFrame(rows)


def stage_columns(table: pd.DataFrame, stage: str) -> list[str]:
    """The feature columns a model at this stage may read.

    size          the size control: hop, pool size, number of hits
    qtype         one-hot question type (the 2wiki template control)
    query ... primary_hits   every feature available at or before that stage
    probe         primary_hits plus the probe features of every shadow arm
    """
    if stage == "size":
        return [f"f:{n}" for n in SIZE_CONTROL]
    if stage == "qtype":
        return [c for c in table.columns if c.startswith("qtype:")]
    if stage == "probe":
        return stage_columns(table, "primary_hits") + \
            [c for c in table.columns if c.startswith("probe:")]
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    allowed = set(STAGES[:STAGES.index(stage) + 1])
    return [f"f:{n}" for n in FEATURES if FEATURE_STAGE[n] in allowed]


def with_qtype_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for q in sorted(set(out["qtype"].fillna(""))):
        out[f"qtype:{q or 'none'}"] = (out["qtype"].fillna("") == q).astype(float)
    return out


def seeded_halves(n: int, seeds: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit index, score index) pairs: for each seed one permutation cut in
    two, used both ways."""
    splits = []
    for seed in seeds:
        perm = np.random.RandomState(seed).permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        splits += [(a, b), (b, a)]
    return splits


def fold_splits(ids: Sequence[str], train_ids: Sequence[str],
                eval_ids: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One (fit, score) pair from two fold files. They must not overlap."""
    if set(train_ids) & set(eval_ids):
        raise ValueError("train and eval folds overlap")
    at = {i: n for n, i in enumerate(ids)}
    fit = np.array([at[i] for i in train_ids if i in at], dtype=int)
    score = np.array([at[i] for i in eval_ids if i in at], dtype=int)
    return [(fit, score)]


def fit_predict(X: np.ndarray, y: np.ndarray, fit: np.ndarray,
                score: np.ndarray) -> Optional[np.ndarray]:
    """P(y = 1) on the score rows from a standardised logistic regression fit
    on the fit rows, or None when the fit half is too small or one-class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    yf = y[fit]
    if len(fit) < MIN_FIT or min(yf.sum(), len(yf) - yf.sum()) < MIN_CLASS:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X[fit], yf)
    return model.predict_proba(X[score])[:, 1]


def held_out_auc(X: np.ndarray, y: np.ndarray,
                 splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, int]:
    """Mean and sd of the ROC AUC over the (fit, score) splits, and how many
    splits could be scored. A split whose fit half is too small or
    one-class, or whose score half is one-class, is skipped (AUC is
    undefined there)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for fit, score in splits:
        ys = y[score]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        prob = fit_predict(X, y, fit, score)
        if prob is not None:
            aucs.append(roc_auc_score(ys, prob))
    if not aucs:
        return math.nan, math.nan, 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def routed_rate(table: pd.DataFrame, seeds: Sequence[int] = (13, 17, 19, 23, 29),
                splits: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
                stages: Sequence[str] = ("size", "qtype", "query", "pool", "primary_hits",
                                         "probe")) -> pd.DataFrame:
    """What a shared router over the frozen features gets: per stage, one
    hit model per arm is fit on the fit half, the arm with the highest
    predicted hit probability is chosen on the score half, and the chosen
    arm's actual outcome is averaged there. Compared on the same score half
    with the best single arm (chosen on the fit half) and the oracle, so the
    gain column is the share of the ceiling a router at this stage recovers.
    AUC says whether the features carry signal; this says what the signal
    buys at retrieval level.
    """
    table = with_qtype_columns(table)
    arms = arms_of(table)
    hit = {a: hits(table, a).to_numpy(dtype=float) for a in arms}
    oracle = np.max(np.stack([hit[a] for a in arms]), axis=0)
    sub_splits = splits if splits is not None else seeded_halves(len(table), seeds)
    rows = []
    for stage in stages:
        cols = stage_columns(table, stage)
        if not cols:
            continue
        X = table[cols].to_numpy(dtype=float)
        routed, single, ceiling_ = [], [], []
        for fit, score in sub_splits:
            probs = [fit_predict(X, hit[a], fit, score) for a in arms]
            if any(p is None for p in probs):
                continue
            chosen = np.argmax(np.stack(probs), axis=0)
            actual = np.stack([hit[a][score] for a in arms])
            routed.append(float(actual[chosen, np.arange(len(score))].mean()))
            best = max(arms, key=lambda a: hit[a][fit].mean())
            single.append(float(hit[best][score].mean()))
            ceiling_.append(float(oracle[score].mean()))
        if not routed:
            continue
        gains = np.array(routed) - np.array(single)
        rows.append({"stage": stage, "n_features": len(cols), "routed": round(np.mean(routed), 4),
                     "best_single": round(np.mean(single), 4),
                     "oracle": round(np.mean(ceiling_), 4),
                     "gain": round(float(gains.mean()), 4), "gain_sd": round(float(gains.std()), 4),
                     "n_splits": len(routed)})
    return pd.DataFrame(rows)


def predictability(table: pd.DataFrame, seeds: Sequence[int] = (13, 17, 19, 23, 29),
                   splits: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
                   stages: Sequence[str] = ("size", "qtype", "query", "pool", "primary_hits",
                                            "probe")) -> pd.DataFrame:
    """Held-out AUC per target and feature stage.

    Targets: each arm's hit over all questions (can a miss be seen coming),
    and for each pair of arms, which one hits on the questions where they
    disagree (is the routing decision predictable). The baseline is 0.5.
    """
    table = with_qtype_columns(table)
    arms = arms_of(table)
    targets = [(f"hit:{a}", table.index.to_numpy(), hits(table, a).to_numpy(dtype=float))
               for a in arms]
    for a, b in combinations(arms, 2):
        x, y = hits(table, a).to_numpy(), hits(table, b).to_numpy()
        keep = np.flatnonzero(x != y)
        targets.append((f"{a}>{b}", keep, (x[keep] > y[keep]).astype(float)))
    rows = []
    for target, idx, y in targets:
        sub_splits = splits if splits is not None else seeded_halves(len(idx), seeds)
        if splits is not None:
            # restrict the fold split to the questions this target keeps
            pos = {q: n for n, q in enumerate(idx)}
            sub_splits = [(np.array([pos[i] for i in fit if i in pos], dtype=int),
                           np.array([pos[i] for i in score if i in pos], dtype=int))
                          for fit, score in splits]
        for stage in stages:
            cols = stage_columns(table, stage)
            if not cols:
                continue
            X = table.loc[idx, cols].to_numpy(dtype=float)
            mean, sd, n_splits = held_out_auc(X, y, sub_splits)
            rows.append({"target": target, "stage": stage, "n": len(idx),
                         "n_pos": int(y.sum()), "n_features": len(cols),
                         "auc": round(mean, 4) if n_splits else math.nan,
                         "auc_sd": round(sd, 4) if n_splits else math.nan,
                         "n_splits": n_splits})
    return pd.DataFrame(rows)


def neighbour_routed_rate(table: pd.DataFrame, ks: Sequence[int] = (5, 20),
                          seeds: Sequence[int] = (13, 17, 19, 23, 29),
                          splits: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
                          ) -> pd.DataFrame:
    """The Stage 2 baseline at retrieval level: similarity memory over past
    questions. For a held-out question, the k nearest fit-half questions by
    tf-idf cosine over the question text vote with their own outcomes, one
    vote per arm, and the arm with the highest neighbour hit rate is chosen
    (ties go to the arm that is better on the fit half). Scored like
    routed_rate, so the two tables say whether look-alike questions carry
    what the process features do not. The question text is the only input;
    outcomes enter only as the neighbours' votes, from the fit half.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    arms = arms_of(table)
    hit = {a: hits(table, a).to_numpy(dtype=float) for a in arms}
    oracle = np.max(np.stack([hit[a] for a in arms]), axis=0)
    text = table["question"].fillna("").tolist()
    sub_splits = splits if splits is not None else seeded_halves(len(table), seeds)
    rows = []
    for k in ks:
        routed, single, ceiling_ = [], [], []
        for fit, score in sub_splits:
            if len(fit) < max(MIN_FIT, k):
                continue
            vec = TfidfVectorizer(sublinear_tf=True).fit([text[i] for i in fit])
            sims = (vec.transform([text[i] for i in score]) @ vec.transform([text[i] for i in fit]).T).toarray()
            nearest = np.argsort(-sims, axis=1)[:, :k]            # positions within fit
            order = sorted(arms, key=lambda a: -hit[a][fit].mean())  # tie-break: fit-half best
            votes = np.stack([hit[a][fit][nearest].mean(axis=1) for a in order])
            chosen = np.argmax(votes, axis=0)                     # first max wins the tie
            actual = np.stack([hit[a][score] for a in order])
            routed.append(float(actual[chosen, np.arange(len(score))].mean()))
            single.append(float(hit[order[0]][score].mean()))
            ceiling_.append(float(oracle[score].mean()))
        if not routed:
            continue
        gains = np.array(routed) - np.array(single)
        rows.append({"stage": f"question_knn{k}", "n_features": k, "routed": round(np.mean(routed), 4),
                     "best_single": round(np.mean(single), 4),
                     "oracle": round(np.mean(ceiling_), 4),
                     "gain": round(float(gains.mean()), 4), "gain_sd": round(float(gains.std()), 4),
                     "n_splits": len(routed)})
    return pd.DataFrame(rows)


def read_ids(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="experience jsonl with shadow arms")
    ap.add_argument("--dataset", required=True, help="label for the output files")
    ap.add_argument("--outcome", default="all_gold_in_topk", choices=OUTCOMES)
    ap.add_argument("--ids", default=None, help="fold file; only these questions")
    ap.add_argument("--train-ids", default=None, help="fit on this fold ...")
    ap.add_argument("--eval-ids", default=None, help="... and score on this one")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 19, 23, 29])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)

    header, records = read_records(args.records)
    ids = read_ids(args.ids) if args.ids else None
    table = arm_table(records, outcome=args.outcome, ids=ids)
    print(f"{args.records}: run {header['run']}, {len(table)} questions, arms {arms_of(table)}, "
          f"outcome {args.outcome}")

    splits = None
    if (args.train_ids is None) != (args.eval_ids is None):
        raise SystemExit("--train-ids and --eval-ids go together")
    if args.train_ids:
        splits = fold_splits(table["id"].tolist(), read_ids(args.train_ids),
                             read_ids(args.eval_ids))
        print(f"fit on {len(splits[0][0])} questions, scored on {len(splits[0][1])}")
    else:
        print(f"seeded halves, {len(args.seeds)} seeds, both directions")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"discordance": discordance(table), "ceiling": ceiling(table, seed=args.seeds[0]),
              "predictability": predictability(table, seeds=args.seeds, splits=splits),
              "routed": pd.concat([routed_rate(table, seeds=args.seeds, splits=splits),
                                   neighbour_routed_rate(table, seeds=args.seeds, splits=splits)],
                                  ignore_index=True)}
    for name, df in tables.items():
        path = out / f"{args.dataset}_arms_{name}.csv"
        df.to_csv(path, index=False)
        print(f"\n{name} ({path})")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
