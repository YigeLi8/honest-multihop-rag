"""Redraw the per-run plots from committed files, without a model.

    python scripts/plots.py [--results results] [--runs RUN ...]

For every run with a results/<run>_qa.csv (or the --runs given): the 2x2 of
answer correct against retrieval correct (<run>_2x2.png), from the qa csv the
same way run_qa draws it, and, where results/<run>_per_hop.json has more than
one hop, the per-hop precision and recall plot (<run>_per_hop.png). The serving
sweep's pareto plot is redrawn from results/sweep_metal.csv. The cell's
decomposition plots come from src.eval.analyze. Interventions and reader
context files share the _qa.csv suffix but hold one row per arm, so they get no
2x2.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.decoupling import build_table, plot_table
from src.eval.per_hop_precision import plot_per_hop

NOT_RUNS = ("interventions_", "reader_ctx_")


def runs_in(results):
    return sorted(p.name[:-len("_qa.csv")] for p in results.glob("*_qa.csv")
                  if not p.name.startswith(NOT_RUNS))


def two_by_two(results, run):
    with open(results / f"{run}_qa.csv") as f:
        rows = list(csv.DictReader(f))
    table = build_table((float(r["em"]) > 0, int(float(r["retrieval_correct"])) > 0)
                        for r in rows)
    plot_table(table, results / f"{run}_2x2.png")
    return table


def per_hop(results, run):
    path = results / f"{run}_per_hop.json"
    if not path.exists():
        return False
    with open(path) as f:
        stats = {int(h): s for h, s in json.load(f).items()}
    if len(stats) < 2:
        return False
    plot_per_hop(stats, results / f"{run}_per_hop.png")
    return True


def pareto(results):
    path = results / "sweep_metal.csv"
    if not path.exists():
        return False
    from src.serving.sweep import plot_pareto
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["em"] = float(r["em"])
        r["tokens_per_s"] = float(r["tokens_per_s"])
        r["rerank"] = r["rerank"] == "True"
    plot_pareto(rows, results / "pareto.png", rows[0].get("device") or "")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args(argv)
    results = Path(args.results)
    runs = args.runs or runs_in(results)
    n_hop = 0
    for run in runs:
        t = two_by_two(results, run)
        n_hop += per_hop(results, run)
        print(f"{run}: n {t.n}, answer right / retrieval wrong {t.ac_rw}")
    done = pareto(results)
    print(f"{len(runs)} 2x2 plots, {n_hop} per-hop plots, pareto {'yes' if done else 'no'}")


if __name__ == "__main__":
    main()
