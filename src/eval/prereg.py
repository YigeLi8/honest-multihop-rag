"""The preregistered tests on the confirmation folds, one row per rule.

    python -m src.eval.prereg

docs/preregistration.md fixes the comparisons, the tests and the pass rules;
this only computes them from the scored confirmation runs (results/*_confirm
csvs and traces, nothing is re-run) and writes results/confirm_prereg.csv.
H1, H3a and H4 are McNemar tests, Holm-adjusted together. H2 and H3b are
equivalence rules on the paired-bootstrap interval of the EM difference. H5
and H6 are descriptive, with the rule stated in the row. Questions are put in
sorted-id order before the seeded bootstrap, and the rules are applied to the
unrounded numbers; the csv is rounded.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.eval.analyze import BUCKETS, bucket, bucket_alt, load_data, load_run
from src.eval.criteria import score_criteria
from src.eval.stats import holm, mcnemar, paired_bootstrap

MARGIN = 0.05      # equivalence margin on EM (H2, H3b)
BAND = 0.04        # H6: the cell's share of accepted answers, against its EM value
RESIDUAL = 0.03    # H5: paragraph + partial + none, as a share of questions

DATASETS = {
    "hotpot": ("hotpotqa", ["bm25_hotpot_confirm", "bm25_k13_hotpot_confirm",
                            "hybrid_ircot_hotpot_confirm", "hybrid_ircot_rerank_hotpot_confirm"]),
    "2wiki": ("2wiki", ["bm25_2wiki_confirm", "bm25_k13_2wiki_confirm"]),
    "musique": ("musique", ["bm25_musique_confirm", "bm25_k7_musique_confirm"]),
}


def load(results_dir, data_dir, dataset):
    stem, names = DATASETS[dataset]
    data = load_data(Path(data_dir) / f"{stem}_dev.jsonl")
    cb = pd.read_csv(Path(results_dir) / f"closed_book_{dataset}_confirm_qa.csv").set_index("id")
    runs = {}
    for name in names:
        df = load_run(results_dir, name, data)
        df = df.loc[df.index.intersection(cb.index)]
        df["cb_em"] = cb.em.reindex(df.index)
        df["illusion"] = ((df.em == 1) & (df.retrieval_correct == 0)).astype(int)
        df["bucket"] = df.apply(bucket, axis=1)
        df["bucket_alt"] = df.apply(bucket_alt, axis=1)
        runs[name] = df
    ids = sorted(set.intersection(*(set(df.index) for df in runs.values())))
    return {name: df.loc[ids] for name, df in runs.items()}


def compare(runs, a, b, metric):
    x, y = runs[a][metric].astype(int).tolist(), runs[b][metric].astype(int).tolist()
    only_a, only_b, p = mcnemar(x, y)
    diff, lo, hi = paired_bootstrap(x, y)
    return {"a": a, "b": b, "metric": metric, "n": len(x),
            "value_a": round(sum(x) / len(x), 4), "value_b": round(sum(y) / len(y), 4),
            "diff": diff, "ci_lo": lo, "ci_hi": hi,
            "only_a": only_a, "only_b": only_b, "mcnemar_p": p}


def equivalence(row):
    lo, hi = row["ci_lo"], row["ci_hi"]
    if -MARGIN < lo and hi < MARGIN:
        return "pass" if lo <= 0 <= hi else "pass, small difference"
    if hi < -MARGIN or lo > MARGIN:
        return "fail, em differs"
    return "fail, equivalence not shown"


def counts(df, col):
    cell = df[df[col] != ""]
    return {b: int((cell[col] == b).sum()) for b in BUCKETS}, len(cell)


def composition(runs, dataset):
    """H5 rows for the bm25 cell of one dataset."""
    name = DATASETS[dataset][1][0]
    df = runs[name]
    n = len(df)
    prim, k = counts(df, "bucket")
    alt, _ = counts(df, "bucket_alt")
    residual = sum(prim[b] for b in ("paragraph", "partial", "none")) / n
    base = {"a": name, "n": n}
    rows = []
    detail = (f"cell {k}; primary " + " ".join(f"{b}={prim[b]}" for b in BUCKETS)
              + "; alternative " + " ".join(f"{b}={alt[b]}" for b in BUCKETS))
    top_alt = max(alt, key=alt.get)
    if dataset == "hotpot":
        diff, lo, hi = paired_bootstrap((df.bucket == "parametric").astype(int).tolist(),
                                        (df.bucket == "shortcut").astype(int).tolist())
        ranked = sorted(prim, key=prim.get, reverse=True)[:2] == ["shortcut", "parametric"]
        ok = ranked and lo > 0 and top_alt == "shortcut"
        rows.append({**base, "hypothesis": "H5 hotpot", "metric": "shortcut - parametric, share of n",
                     "diff": diff, "ci_lo": lo, "ci_hi": hi,
                     "rule": "shortcut largest, parametric second, ci excludes 0; "
                             "shortcut largest under the alternative order",
                     "verdict": "pass" if ok else "fail", "detail": detail})
    elif dataset == "2wiki":
        share = (prim["parametric"] + prim["shortcut"]) / k if k else 0.0
        ok = share >= 0.75 and top_alt == "shortcut"
        rows.append({**base, "hypothesis": "H5 2wiki", "metric": "parametric + shortcut, share of the cell",
                     "value_a": round(share, 4),
                     "rule": ">= 0.75 of the cell; shortcut largest under the alternative order",
                     "verdict": "pass" if ok else "fail", "detail": detail})
    else:
        ok = max(prim, key=prim.get) == "shortcut" and top_alt == "shortcut"
        rows.append({**base, "hypothesis": "H5 musique", "metric": "parametric, share of the cell",
                     "value_a": round(prim["parametric"] / k, 4) if k else 0.0,
                     "rule": "shortcut largest under both orders; lowest parametric share of "
                             "the three datasets (compare the H5 rows)",
                     "verdict": "pass" if ok else "fail", "detail": detail})
    rows.append({**base, "hypothesis": f"H5 {dataset} residual", "metric": "unexplained",
                 "value_a": round(residual, 4), "rule": f"<= {RESIDUAL}",
                 "verdict": "pass" if residual <= RESIDUAL else "fail", "detail": ""})
    return rows, (prim["parametric"] / k if k else 0.0), (alt["parametric"] / k if k else 0.0)


def criteria_rows(runs, names, a=None, b=None):
    """H6: the cell's share of accepted answers under each criterion, and the
    H1 comparison repeated under the two looser ones."""
    rows, scored = [], {}
    for name in names:
        df = score_criteria(runs[name].reset_index())
        retr = df.retrieval_correct.astype(int).astype(bool)
        share = {c: float((df[c] & ~retr).sum() / max(1, df[c].sum())) for c in ("em", "f1_50", "contain")}
        scored[name] = {c: (df[c] & ~retr).astype(int).tolist() for c in share}
        worst = max(abs(share[c] - share["em"]) for c in ("f1_50", "contain"))
        rows.append({"hypothesis": "H6 share", "a": name, "n": len(df), "metric": "cell / accepted answers",
                     "value_a": round(share["em"], 4), "diff": round(worst, 4),
                     "rule": f"f1_50 and contain shares within {BAND} of the em share",
                     "verdict": "pass" if worst <= BAND else "criterion-sensitive",
                     "detail": " ".join(f"{c}={share[c]:.4f}" for c in share)})
    if a and b:
        for c in ("f1_50", "contain"):
            only_a, only_b, p = mcnemar(scored[a][c], scored[b][c])
            va, vb = sum(scored[a][c]) / len(scored[a][c]), sum(scored[b][c]) / len(scored[b][c])
            rows.append({"hypothesis": f"H6 direction ({c})", "a": a, "b": b, "n": len(scored[a][c]),
                         "metric": f"the cell under {c}", "value_a": round(va, 4), "value_b": round(vb, 4),
                         "diff": round(vb - va, 4), "only_a": only_a, "only_b": only_b, "mcnemar_p": p,
                         "rule": "lower under b, p < 0.05 unadjusted",
                         "verdict": "pass" if vb < va and p < 0.05 else "fail", "detail": ""})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args(argv)

    hp = load(args.results_dir, args.data_dir, "hotpot")
    bm25, k13, ircot, rerank = DATASETS["hotpot"][1]
    rows = []

    primary = [("H1", bm25, ircot, "illusion"), ("H3a", ircot, rerank, "retrieval_correct"),
               ("H4", k13, ircot, "illusion")]
    tests = [compare(hp, a, b, m) for _, a, b, m in primary]
    for (h, *_), t, p_adj in zip(primary, tests, holm([t["mcnemar_p"] for t in tests])):
        ok = t["diff"] < 0 and p_adj < 0.05 and t["ci_hi"] < 0
        rows.append({"hypothesis": h, **t, "holm_p": p_adj,
                     "rule": "lower under b, holm p < 0.05, ci excludes 0",
                     "verdict": "pass" if ok else "fail", "detail": ""})
    for h, a, b in (("H2", k13, ircot), ("H3b", ircot, rerank)):
        t = compare(hp, a, b, "em")
        t["mcnemar_p"] = None      # an interval rule, no p-value (the paired csv has the descriptive one)
        rows.append({"hypothesis": h, **t, "rule": f"95% ci of the em difference inside +-{MARGIN}",
                     "verdict": equivalence(t), "detail": ""})

    shares, shares_alt = {}, {}
    for dataset in DATASETS:
        runs = hp if dataset == "hotpot" else load(args.results_dir, args.data_dir, dataset)
        comp, shares[dataset], shares_alt[dataset] = composition(runs, dataset)
        rows += comp
        if dataset == "hotpot":
            rows += criteria_rows(runs, DATASETS["hotpot"][1], bm25, ircot)
        elif dataset == "2wiki":
            rows += criteria_rows(runs, DATASETS["2wiki"][1][:1])
    lowest = min(shares, key=shares.get)
    for r in rows:
        if r["hypothesis"] == "H5 musique":
            r["detail"] += ("; parametric share of the cell " + " ".join(f"{d}={s:.3f}" for d, s in shares.items())
                            + "; under the alternative order " + " ".join(f"{d}={s:.3f}" for d, s in shares_alt.items()))
            if lowest != "musique":
                r["verdict"] = "fail"

    cols = ["hypothesis", "a", "b", "metric", "n", "value_a", "value_b", "diff", "ci_lo", "ci_hi",
            "only_a", "only_b", "mcnemar_p", "holm_p", "rule", "verdict", "detail"]
    table = pd.DataFrame(rows).reindex(columns=cols)
    shown = ["value_a", "value_b", "diff", "ci_lo", "ci_hi"]
    table[shown] = table[shown].astype(float).round(4)
    out = Path(args.results_dir) / "confirm_prereg.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(table.drop(columns=["rule"]).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
