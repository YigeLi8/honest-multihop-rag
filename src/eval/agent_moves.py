"""Where a change moved the wrong answers between two runs on the same
questions, read from their traces with no model (docs/plan.md, Stage 6b,
From the traces).

    python -m src.eval.agent_moves \
        --base orchestrator_guard_wide_unsup_hotpot_explore \
        --arm orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore \
        --out agent_moves_dev4_hotpot_explore

Every question gets an outcome in each run: right (EM), or the place of
agent_losses (in_context, dropped, not_retrieved, yesno_gold). Writes:

  results/<out>_moves.csv   outcome in base x outcome in arm, with how many
                            are bridge and comparison questions
  results/<out>_moved.csv   one row per question that was dropped in base and
                            is not in arm: its outcome in arm, how the
                            sentence holding the gold string came to be read
                            (HOW), both replies, whether they are the same
                            answer, what the arm's reply was when wrong
                            (agent_reads.READS) and whether some lookup came
                            back with the gold
  results/<out>_kind.csv    the moved ones that are still wrong with the
                            string read, by kind of reply and HOW

HOW, for a question whose gold string is in what the arm read: the first
sentence read that holds it was

  added      put in the notebook from a lookup's answer (pipeline.
             keep_answer_sentence, the step's "added")
  thin       not in the notebook, read because the notebook was thin and
             everything retrieved followed it ("+thin")
  fallback   read because nothing was kept ("+fallback")
  kept       kept by a keep step

What was read is agent_reads.read_ids: the notebook; with "+thin" the
notebook and then everything else retrieved; with "+fallback" or no
notebook, everything retrieved. The
string test is agent_losses' exact one. Gold is read only to place and score.
"""
import argparse
import csv
import sys
from pathlib import Path

from src.eval.agent_losses import PLACES, holds, load_gold, load_run, place_of, write
from src.eval.agent_reads import READS, load_questions, lookup_echo, read_ids, read_kind
from src.eval.agent_traces import load_titles
from src.eval.answer_metrics import em

OUTCOMES = ("right",) + PLACES
HOW = ("added", "thin", "fallback", "kept")


def outcome(row, trace, golds, texts):
    return "right" if float(row["em"]) else place_of(trace, golds, texts)


def added_ids(trace):
    return {c for s in trace.get("steps", []) for c in s.get("added", [])}


def how_read(trace, golds, texts):
    """How the first sentence read holding a gold string got there (one of
    HOW), or None when no sentence read holds one."""
    first = next((c for c in read_ids(trace) if any(holds(texts.get(c, ""), g) for g in golds)),
                 None)
    if first is None:
        return None
    stop = trace.get("stop_reason") or ""
    nb = trace.get("notebook") or []
    if not nb or "+fallback" in stop:
        return "fallback"
    if first not in nb:
        return "thin"
    return "added" if first in added_ids(trace) else "kept"


def tables(base, arm, gold, titles, questions):
    (brows, btr), (arows, atr) = base, arm
    assert set(brows) == set(arows), "the two runs must hold the same question ids"
    ids = list(brows)
    ob = {i: outcome(brows[i], btr[i], *gold[i]) for i in ids}
    oa = {i: outcome(arows[i], atr[i], *gold[i]) for i in ids}

    moves = []
    for f in OUTCOMES:
        for t in OUTCOMES:
            qs = [i for i in ids if ob[i] == f and oa[i] == t]
            if qs:
                moves.append({"base": f, "arm": t, "n": len(qs),
                              "bridge": sum(brows[i]["qtype"] == "bridge" for i in qs),
                              "comparison": sum(brows[i]["qtype"] == "comparison" for i in qs)})

    moved = []
    for i in ids:
        if ob[i] != "dropped" or oa[i] == "dropped":
            continue
        golds, texts = gold[i]
        bp, ap = brows[i]["answer_pred"], arows[i]["answer_pred"]
        kind = ""
        if oa[i] == "in_context":
            kind = read_kind(ap, golds, questions[i], titles[i], atr[i], texts, arows[i]["qtype"])
        moved.append({"id": i, "qtype": arows[i]["qtype"], "arm": oa[i],
                      "how": how_read(atr[i], golds, texts) or "", "same_reply": int(em(bp, ap)),
                      "kind": kind, "lookup_right": int(lookup_echo(ap, golds, atr[i])[1]),
                      "base_pred": bp, "arm_pred": ap, "gold": golds[0]})

    still = [m for m in moved if m["arm"] == "in_context"]
    kinds = []
    for k in READS:
        qs = [m for m in still if m["kind"] == k]
        if qs:
            row = {"kind": k, "n": len(qs), "same_reply": sum(m["same_reply"] for m in qs),
                   "lookup_right": sum(m["lookup_right"] for m in qs)}
            row.update({h: sum(m["how"] == h for m in qs) for h in HOW})
            kinds.append(row)
    total = {"kind": "all", "n": len(still)}
    if kinds:
        total.update({c: sum(r[c] for r in kinds) for c in kinds[0] if c not in ("kind", "n")})
    kinds.append(total)
    return moves, moved, kinds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", default=["data/processed/hotpotqa_dev.jsonl"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    base = load_run(args.results, args.base)
    arm = load_run(args.results, args.arm)
    gold = load_gold(args.data, base[0])
    titles = load_titles(args.data, base[0])
    questions = load_questions(args.data, base[0])
    out = Path(args.results) / args.out
    for name, t in zip(("moves", "moved", "kind"), tables(base, arm, gold, titles, questions)):
        if not t:
            continue
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
