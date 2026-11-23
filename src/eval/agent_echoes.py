"""Final answers that repeat a lookup's answer, and whether the sentences the
final read was given held the gold when the repeated answer was wrong
(docs/plan.md, Stage 6b, From the traces).

    python -m src.eval.agent_echoes --out agent_echoes_hotpot_explore \
        --runs orchestrator_hotpot_explore orchestrator_notes_hotpot_explore \
        --compare orchestrator_hotpot_explore=orchestrator_notes_hotpot_explore

B1 and B5 (the lookups and their answers written into the final read) lost
mostly where the final answer repeats a lookup's answer. This splits such
answers by whether the record was in front of the final read: the gold
answer string in a sentence the final read was given (agent_losses.place_of
"in_context") against not. A repeated wrong answer with the gold read is a
read that took the lookup's answer over the notebook; one without it is a
lookup failure passed on, which no final read could have fixed from what it
was given.

Each final answer is put in one ECHO class:

  yesno     the reply is yes or no (comparisons answer so whatever the
            lookups said; not counted as a repeat)
  last      equals (EM after normalization) the last lookup's answer
  earlier   equals an earlier lookup's answer and not the last one's
  none      equals no lookup's answer

Writes:

  results/<out>_echo.csv    one row per run and ECHO class: n, right, EM,
                            the wrong ones by place, and how many of the
                            wrong ones had some lookup answer equal to the
                            gold
  results/<out>_paired.csv  per --compare base=arm, the questions the arm
                            lost and won, by the arm's ECHO class and the
                            place of the gold string in what the arm's
                            final read was given
  results/<out>_adopt.csv   per --compare base=arm, two exposures (ADOPT):
                            how often the arm's read took a lookup's answer
                            over a notebook the base read right, and how
                            often it took a right last lookup's answer the
                            base read missed

ADOPT, for arms whose final read is handed the lookups' answers (B1, B5)
and a base that is the same agent without them (the runs match up to the
final read, so both reads were given the same notebook):

  wrong_last   the base was right, the gold string is in what the arm read,
               and the last lookup's answer is neither the gold nor yes/no;
               adopted when the arm's reply equals a lookup's answer that is
               not the gold
  right_last   the base was wrong and the last lookup's answer is the gold;
               adopted when the arm is right

with, for each, how many of the exposed and of the adopted had the last
lookup's answer string in a sentence the arm read (a test that needs no
gold, the shape of B6's: whether it tells the two exposures apart).

The string test is agent_losses' exact one, so "in_context" undercounts
paraphrases. Gold is read only to place and score.
"""
import argparse
import csv
import sys
from pathlib import Path

from src.eval.agent_losses import PLACES, holds, load_gold, load_run, place_of, write
from src.eval.agent_reads import read_ids
from src.eval.agent_traces import lookups
from src.eval.answer_metrics import em, normalize

ECHO = ("yesno", "last", "earlier", "none")


def echo_of(pred, trace):
    """The ECHO class of a final answer."""
    if normalize(pred) in ("yes", "no"):
        return "yesno"
    subs = lookups(trace)
    if not subs or not normalize(pred):
        return "none"
    if em(subs[-1]["sub_answer"], pred):
        return "last"
    return "earlier" if any(em(s["sub_answer"], pred) for s in subs[:-1]) else "none"


def sub_right(trace, golds):
    return any(em(s["sub_answer"], g) for s in lookups(trace) for g in golds)


def facts(rows, traces, gold):
    """{id: (right, echo, place, a lookup answered the gold)}."""
    out = {}
    for i, r in rows.items():
        golds, texts = gold[i]
        out[i] = (bool(float(r["em"])), echo_of(r["answer_pred"], traces[i]),
                  place_of(traces[i], golds, texts), sub_right(traces[i], golds))
    return out


def echo_table(run, f):
    tab = []
    for e in ECHO:
        qs = [v for v in f.values() if v[1] == e]
        wrong = [v for v in qs if not v[0]]
        row = {"run": run, "echo": e, "n": len(qs), "right": len(qs) - len(wrong),
               "em": round((len(qs) - len(wrong)) / len(qs), 4) if qs else "",
               "wrong": len(wrong)}
        for p in PLACES:
            row[f"wrong_{p}"] = sum(v[2] == p for v in wrong)
        row["wrong_sub_right"] = sum(v[3] for v in wrong)
        tab.append(row)
    return tab


def paired_table(base, arm, fb, fa):
    """The arm's losses and wins against base, by the arm's echo class and
    the place of the gold string in what the arm read (for a win, always
    in_context or yesno_gold)."""
    tab = []
    for direction, keep in (("lost", lambda i: fb[i][0] and not fa[i][0]),
                            ("won", lambda i: fa[i][0] and not fb[i][0])):
        qs = [i for i in fa if keep(i)]
        for e in ECHO:
            sel = [i for i in qs if fa[i][1] == e]
            row = {"base": base, "arm": arm, "direction": direction, "echo": e, "n": len(sel)}
            for p in PLACES:
                row[p] = sum(fa[i][2] == p for i in sel)
            row["sub_right"] = sum(fa[i][3] for i in sel)
            row["base_echo_same"] = sum(fb[i][1] == e for i in sel)
            tab.append(row)
    return tab


def _is_gold(answer, golds):
    return any(em(answer, g) for g in golds)


def adopted_wrong(pred, trace, golds):
    """The reply equals a lookup's answer that is not the gold (yes/no aside)."""
    return (normalize(pred) not in ("yes", "no")
            and any(em(s["sub_answer"], pred) and not _is_gold(s["sub_answer"], golds)
                    for s in lookups(trace)))


def adopt_table(base, arm, loaded, gold, fb, fa):
    """The two ADOPT exposures of one base=arm pair."""
    (rb, _), (ra, ta) = loaded[base], loaded[arm]
    tab = []
    for kind in ("wrong_last", "right_last"):
        exposed = []
        for i in ra:
            golds = gold[i][0]
            subs = lookups(ta[i])
            if not subs:
                continue
            last = subs[-1]["sub_answer"]
            if kind == "wrong_last":
                ok = (fb[i][0] and fa[i][2] == "in_context" and not _is_gold(last, golds)
                      and normalize(last) not in ("yes", "no"))
            else:
                ok = not fb[i][0] and _is_gold(last, golds)
            if ok:
                exposed.append(i)
        if kind == "wrong_last":
            took = [i for i in exposed if adopted_wrong(ra[i]["answer_pred"], ta[i], gold[i][0])]
        else:
            took = [i for i in exposed if fa[i][0]]
        row = {"base": base, "arm": arm, "exposure": kind, "n": len(exposed),
               "adopted": len(took), "share": round(len(took) / len(exposed), 4) if exposed else ""}
        def supported(i):
            last, texts = lookups(ta[i])[-1]["sub_answer"], gold[i][1]
            return any(holds(texts[c], last) for c in read_ids(ta[i]) if c in texts)
        row["n_last_in_read"] = sum(map(supported, exposed))
        row["adopted_last_in_read"] = sum(map(supported, took))
        for qt in ("bridge", "comparison"):
            row[f"n_{qt}"] = sum(rb[i]["qtype"] == qt for i in exposed)
            row[f"adopted_{qt}"] = sum(rb[i]["qtype"] == qt for i in took)
        tab.append(row)
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--compare", action="append", default=[], help="base=arm")
    ap.add_argument("--data", nargs="+", default=["data/processed/hotpotqa_dev.jsonl"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    pairs = [c.split("=") for c in args.compare]
    names = list(dict.fromkeys(args.runs + [r for p in pairs for r in p]))
    loaded = {n: load_run(args.results, n) for n in names}
    ids = set().union(*(rows for rows, _ in loaded.values()))
    gold = load_gold(args.data, ids)
    f = {n: facts(*loaded[n], gold) for n in names}
    for base, arm in pairs:
        assert set(f[base]) == set(f[arm]), f"{base} and {arm} differ in ids"
    etab = [r for n in names for r in echo_table(n, f[n])]
    ptab = [r for base, arm in pairs for r in paired_table(base, arm, f[base], f[arm])]
    atab = [r for base, arm in pairs
            for r in adopt_table(base, arm, loaded, gold, f[base], f[arm])]
    out = Path(args.results) / args.out
    for name, t in (("echo", etab), ("paired", ptab), ("adopt", atab)):
        if not t:
            continue
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
