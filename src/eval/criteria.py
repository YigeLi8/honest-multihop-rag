"""The answer-correctness ladder: EM, F1 >= 0.5 and capped containment, each
beside the same retrieval column, for runs that are already scored.

    python -m src.eval.criteria --dataset hotpot \
        --runs bm25_hotpot_dev hybrid_ircot_hotpot_dev hybrid_ircot_rerank_hotpot_dev
    python -m src.eval.criteria --dataset musique --runs bm25_musique_dev \
        --data-file data/processed/musique_dev.jsonl

Reads results/<run>_qa.csv, nothing is re-run. em and f1 are taken as the run
scored them (with aliases where the run had them); containment is computed
here from answer_pred against answer_gold and the aliases in --data-file, so a
musique run reported without its data file is containment against the gold
string alone, and the output says so.

For every run and criterion: the answer-correct rate, the 2x2 against
retrieval_correct with the illusion cell (answer right, gold evidence
incomplete) and its Wilson interval, and the paired change against EM on the
same questions (questions the criterion adds, questions it drops, McNemar).
The point is to see whether the illusion cell is an artefact of strict string
matching: if a looser criterion moves the answer rate but not the cell, it is
not. Writes results/<dataset>_criteria.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.answer_metrics import CRITERIA, correct_by
from src.eval.stats import mcnemar, wilson_ci


def load_aliases(path):
    """id -> answer aliases, from a processed data file (musique has them)."""
    out = {}
    if not path:
        return out
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = [a for a in d.get("answer_aliases", []) if a]
    return out


def score_criteria(df, aliases=None):
    """Add one boolean column per criterion to a qa dataframe.

    em and f1_50 come from the run's own em and f1 columns; contain is
    computed from answer_pred, answer_gold and the aliases of the id.
    """
    aliases = aliases or {}
    out = df.copy()
    out["em"] = out["em"].astype(float) >= CRITERIA["em"][1]
    out["f1_50"] = out["f1"].astype(float) >= CRITERIA["f1_50"][1]
    out["contain"] = [
        correct_by(str(pred), [str(gold)] + aliases.get(qid, []), "contain")
        for qid, pred, gold in zip(out["id"], out["answer_pred"], out["answer_gold"])]
    return out


def criteria_rows(run, df):
    """One row per criterion for one scored run."""
    rows = []
    retr = df["retrieval_correct"].astype(int).astype(bool)
    em = df["em"].astype(bool)
    n = len(df)
    for crit in CRITERIA:
        ok = df[crit].astype(bool)
        cell = int((ok & ~retr).sum())
        lo, hi = wilson_ci(cell, n)
        only_em, only_crit, p = mcnemar(em.astype(int).tolist(), ok.astype(int).tolist())
        rows.append({
            "run": run, "criterion": crit, "n": n,
            "answer_correct": round(ok.mean(), 4),
            "ans_ok_retr_ok": int((ok & retr).sum()),
            "ans_ok_retr_wrong": cell,
            "ans_wrong_retr_ok": int((~ok & retr).sum()),
            "ans_wrong_retr_wrong": int((~ok & ~retr).sum()),
            "illusion_rate": round(cell / n, 4) if n else 0.0,
            "illusion_lo": round(lo, 4), "illusion_hi": round(hi, 4),
            # the cell as a share of the answers the criterion accepts
            "illusion_share_of_correct": round(cell / ok.sum(), 4) if ok.sum() else 0.0,
            "vs_em_added": only_crit, "vs_em_dropped": only_em,
            "vs_em_mcnemar_p": round(p, 4),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label for the output file")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-file", default=None, help="processed jsonl, for answer aliases")
    args = ap.parse_args(argv)

    aliases = load_aliases(args.data_file)
    results = Path(args.results_dir)
    rows = []
    for run in args.runs:
        df = pd.read_csv(results / f"{run}_qa.csv", keep_default_na=False)
        rows += criteria_rows(run, score_criteria(df, aliases))
    table = pd.DataFrame(rows)
    out = results / f"{args.dataset}_criteria.csv"
    table.to_csv(out, index=False)

    if not aliases:
        print("containment against the gold string only (no --data-file, so no aliases)")
    cols = ["run", "criterion", "n", "answer_correct", "ans_ok_retr_wrong", "illusion_rate",
            "illusion_lo", "illusion_hi", "illusion_share_of_correct", "vs_em_added",
            "vs_em_dropped", "vs_em_mcnemar_p"]
    print(table[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
