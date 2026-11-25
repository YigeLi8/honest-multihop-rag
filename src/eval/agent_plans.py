"""The main agent's plans against the outcome, read from committed runs with
no model (docs/plan.md, Stage 6b, From the traces).

    python -m src.eval.agent_plans \
        --run orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore \
        --base orchestrator_hotpot_explore --out agent_plans_hotpot_explore

The planner has not changed since 4 Nov, and the runs say so: a question's
plan is the same in every round, so the 4 Nov agent (--base) and the agent as
it stands (--run) can be read side by side on the same plans. The script
checks this and stops if a plan differs. Writes:

  results/<out>_shape.csv    question type by plan shape (one lookup,
                             parallel, chain): EM now and on 4 Nov on the same
                             questions, won and lost between them, the wrong
                             answers by place (agent_losses.PLACES), how often
                             some lookup came back with the gold, and how
                             often the last lookup restates the question
  results/<out>_source.csv   bridge questions, by shape and by where the
                             final answer came from (SOURCE): who made the
                             last hop, a lookup or the final read
  results/<out>_restate.csv  bridge questions, by shape and by how much of
                             the question the last lookup asks (RESTATE)

SOURCE, as agent_echoes.echo_of: last (the final answer equals the last
lookup's answer by EM), earlier (an earlier lookup's), none (no lookup's: the
read answered with something no lookup gave), yesno.

RESTATE, gold-free: the share of the question's content words (normalized,
stopwords and question words out) that the last lookup, as asked with #k
filled in, also has. whole at 0.75 or more, part under it. A lookup asking
the whole question in other words ("Who sang it?" for the question) counts as
part, and a lookup that copies the question's names but asks only the first
hop can reach whole, so it is a word test of wording, not of what was asked.

Gold is read only to score and place. Shapes are chosen by the planner on
different questions, so EM across shapes is not a comparison; the 4 Nov to
now change within a shape is paired on identical questions and plans.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from src.eval.agent_echoes import ECHO, echo_of, sub_right
from src.eval.agent_losses import PLACES, load_gold, load_run, place_of, write
from src.eval.agent_traces import lookups, plan_shape, plan_step, words

SHAPES = ("one", "parallel", "chain")
RESTATE = ("whole", "part")
WHOLE = 0.75
_QWORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is",
           "are", "was", "were", "do", "does", "did", "name", "that", "this", "has", "have"}


def coverage(lookup, question):
    """Share of the question's content words the lookup also has."""
    q = words(question) - _QWORDS
    return len(q & words(lookup)) / len(q) if q else 0.0


def restate_of(trace):
    """RESTATE of the last lookup as asked (gold-free)."""
    subs = lookups(trace)
    if not subs:
        return "part"
    q = plan_step(trace).get("question", "")
    return "whole" if coverage(subs[-1]["sub_question"], q) >= WHOLE else "part"


def facts(rows, traces, gold, base_rows):
    """{id: dict of the per-question facts the tables count}."""
    out = {}
    for i, r in rows.items():
        t = traces[i]
        golds, texts = gold[i]
        out[i] = {"qtype": r["qtype"], "shape": plan_shape(plan_step(t)["plan"]),
                  "right": bool(float(r["em"])), "base": bool(float(base_rows[i]["em"])),
                  "place": place_of(t, golds, texts), "sub_gold": sub_right(t, golds),
                  "source": echo_of(r["answer_pred"], t), "restate": restate_of(t),
                  "n_lookups": len(lookups(t))}
    return out


def _count(qs, f):
    n = len(qs)
    right = sum(f[i]["right"] for i in qs)
    row = {"n": n, "right": right, "em": round(right / n, 4) if n else "",
           "em_base": round(sum(f[i]["base"] for i in qs) / n, 4) if n else "",
           "won": sum(f[i]["right"] and not f[i]["base"] for i in qs),
           "lost": sum(f[i]["base"] and not f[i]["right"] for i in qs)}
    for p in PLACES:
        row[f"wrong_{p}"] = sum(not f[i]["right"] and f[i]["place"] == p for i in qs)
    row["sub_gold"] = sum(f[i]["sub_gold"] for i in qs)
    row["sub_gold_wrong"] = sum(f[i]["sub_gold"] and not f[i]["right"] for i in qs)
    return row


def tables(f):
    ids = list(f)
    shape = []
    for qt in sorted({f[i]["qtype"] for i in ids}) + ["all"]:
        for sh in SHAPES + ("all",):
            qs = [i for i in ids if qt in ("all", f[i]["qtype"]) and sh in ("all", f[i]["shape"])]
            if not qs:
                continue
            row = {"qtype": qt, "shape": sh}
            row.update(_count(qs, f))
            row["restate_whole"] = sum(f[i]["restate"] == "whole" for i in qs)
            row["lookups"] = round(sum(f[i]["n_lookups"] for i in qs) / len(qs), 3)
            shape.append(row)

    bridge = [i for i in ids if f[i]["qtype"] == "bridge"]
    source = []
    for sh in SHAPES + ("all",):
        for src in ECHO + ("all",):
            qs = [i for i in bridge if sh in ("all", f[i]["shape"]) and src in ("all", f[i]["source"])]
            if qs:
                source.append({"shape": sh, "source": src, **_count(qs, f)})

    restate = []
    for sh in SHAPES + ("all",):
        for rs in RESTATE:
            qs = [i for i in bridge if sh in ("all", f[i]["shape"]) and f[i]["restate"] == rs]
            if not qs:
                continue
            row = {"shape": sh, "restate": rs, **_count(qs, f)}
            c = Counter(f[i]["source"] for i in qs)
            row.update({f"source_{k}": c[k] for k in ECHO})
            restate.append(row)
    return shape, source, restate


def same_plans(traces, base_traces):
    """Ids whose plan differs between the two runs (should be none)."""
    return [i for i in traces if plan_step(traces[i])["plan"] != plan_step(base_traces[i])["plan"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", default=["data/processed/hotpotqa_dev.jsonl"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    rows, traces = load_run(args.results, args.run)
    base_rows, base_traces = load_run(args.results, args.base)
    assert set(rows) == set(base_rows), "the two runs must hold the same questions"
    differ = same_plans(traces, base_traces)
    if differ:
        sys.exit(f"{len(differ)} plans differ between {args.run} and {args.base}, e.g. {differ[0]}")
    gold = load_gold(args.data, rows)
    f = facts(rows, traces, gold, base_rows)
    out = Path(args.results) / args.out
    for name, t in zip(("shape", "source", "restate"), tables(f)):
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
