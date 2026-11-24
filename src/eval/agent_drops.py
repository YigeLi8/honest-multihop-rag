"""The answers that were retrieved and not kept, read from one main-agent run
with no model: what the subagent shown the answer sentence was asking, and
what it kept and answered instead (docs/plan.md, Stage 6b, From the traces).

    python -m src.eval.agent_drops \
        --run orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore \
        --out agent_drops_hotpot_explore

agent_traces already says which subagent was shown the sentence and what
broke the chain first; this reads the step that decided it. For every wrong
answer that agent_losses.place_of puts in dropped, the deciding lookup is the
last lookup whose keep steps were shown a retrieved sentence holding the gold
string (a holder). Writes:

  results/<out>_rows.csv    one row per dropped question: the first break
                            (agent_traces.CHAIN), ASKED, KEPT and FROM for its
                            deciding lookup, the holder's place, and the
                            lookup and its answer (no gold text)
  results/<out>_asked.csv   the same counted by ASKED, bridge and comparison
                            apart
  results/<out>_dangling.csv  plans with a dangling lookup (below), over every
                            question

ASKED, what the deciding lookup asked about, the first that holds:

  title_answer   a holder's paragraph has the gold answer for its title
                 ("Mr. Church::0"), so no lookup could name it before it was
                 answered; read with the hand counts, not with the others
  answer_para    it names the paragraph of a holder it was shown (every word
                 of the title, without a parenthesis, in the lookup as asked,
                 #k filled in): it asked about the entity whose sentence
                 answers, and kept or answered otherwise
  question_only  it names no holder's paragraph, and the holder's paragraph is
                 named in the question: a lookup about another part of the
                 question was shown it in passing
  other          neither: the lookup asked about something else (on a chain,
                 often an earlier lookup's wrong answer pasted in)

KEPT, what its keep steps kept: holder_para (a sentence of a holder's
paragraph, not the holder), other (only other paragraphs), nothing. A holder
itself cannot be kept, or the question would not be dropped.

A planned lookup is dangling (dangling, gold-free) when it is not the first,
has no #k, and points back in words: "this", "these" or "those", "that"
followed by a noun of a short list ("that building"), or a subject or object
pronoun ("he", "they"; not "his" or "her", which mostly refer within the
lookup, "his B.A."). Such a lookup is asked as written, without the answer
it points to. results/<out>_dangling.csv counts plans with one over every
question, by plan shape, with EM and the wrong answers by place. The word
list catches "this island" and misses "the island"; a pronoun can refer to
an entity named in the same lookup ("Where was Ann born and where did she
die?"), so it overcounts as well.

FROM, where its answer string is: added (the sentence B8 added for it,
pipeline.keep_answer_sentence), kept (a sentence it kept), shown (a sentence
its searches retrieved and it did not keep), nowhere (in none of them, or yes,
no or empty).

"names" is a word test and the title is gold-derived for the gold paragraphs
(read here only to place): a lookup that names the entity by another form
("JFK") counts as not naming it, and a lookup whose words happen to cover a
short title counts as naming it. Read by hand in docs/plan.md.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from src.eval.agent_losses import PLACES, holds, load_gold, load_run, place_of, write
from src.eval.agent_misses import names
from src.eval.agent_traces import _REF, first_break, holders, load_titles, lookups, plan_shape, plan_step, title_of
from src.eval.answer_metrics import normalize

ASKED = ("title_answer", "answer_para", "question_only", "other")
KEPT = ("holder_para", "other", "nothing")
FROM = ("added", "kept", "shown", "nowhere")

_BACK = re.compile(
    r"\b(this|these|those)\b"
    r"|\bthat (person|man|woman|building|island|city|town|film|movie|band|company|team|"
    r"country|state|place|player|actor|actress|show|series|album|song|book|group|"
    r"organization|university|school|river|area|event|game|one|character|author|singer)\b"
    r"|\b(he|she|him|they|them)\b", re.I)


def dangling(plan):
    """Indices (0-based) of planned lookups after the first that have no #k
    and point back in words."""
    return [k for k, p in enumerate(plan) if k and not _REF.search(p) and _BACK.search(p)]


def keep_steps(trace, k):
    """The keep steps (one per search) of lookup k, in order."""
    return [s for s in trace.get("steps", [])
            if s.get("role") == "subagent" and "shown" in s and s.get("sub") == k]


def deciding(trace, hold):
    """(index of the last lookup whose keep steps were shown a holder, the
    holders it was shown) or (None, set())."""
    for k in range(len(lookups(trace)) - 1, -1, -1):
        seen = {c for s in keep_steps(trace, k) for c in s["shown"]} & hold
        if seen:
            return k, seen
    return None, set()


def asked(lookup, question, seen, golds=()):
    """One of ASKED for a lookup shown the holders seen."""
    paras = {title_of(c) for c in seen}
    if any(names(g, t) and names(t, g) for t in paras for g in golds):
        return "title_answer"
    if any(names(lookup, t) for t in paras):
        return "answer_para"
    if any(names(question, t) for t in paras):
        return "question_only"
    return "other"


def kept_kind(trace, k, seen):
    """One of KEPT for lookup k."""
    kept = {c for s in keep_steps(trace, k) for c in s.get("kept", [])}
    if not kept:
        return "nothing"
    paras = {title_of(c) for c in seen}
    return "holder_para" if any(title_of(c) in paras for c in kept) else "other"


def answer_from(trace, k, texts):
    """One of FROM for lookup k's answer."""
    sub = lookups(trace)[k]
    a = sub.get("sub_answer", "")
    if normalize(a) in ("", "yes", "no"):
        return "nowhere"
    steps = keep_steps(trace, k)
    kept = [c for s in steps for c in s.get("kept", [])]
    shown = [c for s in steps for c in s["shown"]]

    def among(ids):
        return any(holds(texts.get(c, ""), a) for c in ids)
    if among(sub.get("added", [])):
        return "added"
    if among(kept):
        return "kept"
    return "shown" if among(shown) else "nowhere"


def holder_place(trace, k, seen, gold_titles):
    """(a holder shown to lookup k is in a gold paragraph, the first such
    holder opens its paragraph, its best position in a list lookup k was
    shown, from 1)."""
    gold = sorted(c for c in seen if title_of(c) in gold_titles)
    first = (gold or sorted(seen))[0]
    pos = min(s["shown"].index(c) + 1 for s in keep_steps(trace, k) for c in seen if c in s["shown"])
    return bool(gold), first.rsplit("::", 1)[1] == "0", pos


def tables(rows, traces, gold, titles):
    ids = list(rows)
    dropped = [i for i in ids if not float(rows[i]["em"])
               and place_of(traces[i], *gold[i]) == "dropped"]
    per = []
    for i in dropped:
        t, (golds, texts) = traces[i], gold[i]
        question = plan_step(t).get("question", "")
        subs = lookups(t)
        k, seen = deciding(t, holders(t, golds, texts))
        row = {"id": i, "qtype": rows[i]["qtype"], "shape": plan_shape(plan_step(t)["plan"]),
               "break": first_break(t, rows[i]["qtype"], titles[i], question)[0],
               "lookups": len(subs), "dangling": int(bool(dangling(plan_step(t)["plan"]))),
               "deciding": "" if k is None else k + 1}
        if k is None:
            row.update(asked="", kept="", answer_from="", is_last="", holder_gold="",
                       holder_first="", holder_pos="", lookup="", lookup_answer="")
        else:
            g, first, pos = holder_place(t, k, seen, titles[i])
            row.update(asked=asked(subs[k]["sub_question"], question, seen, golds),
                       kept=kept_kind(t, k, seen), answer_from=answer_from(t, k, texts),
                       is_last=int(k == len(subs) - 1), holder_gold=int(g),
                       holder_first=int(first), holder_pos=pos,
                       lookup=subs[k]["sub_question"], lookup_answer=subs[k]["sub_answer"])
        row["reply"] = rows[i].get("answer_pred", "")
        per.append(row)

    tab = []
    for qt in ("bridge", "comparison", "all"):
        qrows = [r for r in per if qt == "all" or r["qtype"] == qt]
        for a in ASKED + ("all",):
            rs = [r for r in qrows if a == "all" or r["asked"] == a]
            if not rs and a != "all":
                continue
            row = {"qtype": qt, "asked": a, "n": len(rs), "is_last": sum(r["is_last"] == 1 for r in rs)}
            for kk in KEPT:
                row[f"kept_{kk}"] = sum(r["kept"] == kk for r in rs)
            for f in FROM:
                row[f"from_{f}"] = sum(r["answer_from"] == f for r in rs)
            row["dangling"] = sum(r["dangling"] == 1 for r in rs)
            row["holder_gold"] = sum(r["holder_gold"] == 1 for r in rs)
            row["holder_first"] = sum(r["holder_first"] == 1 for r in rs)
            row["holder_pos_1"] = sum(r["holder_pos"] == 1 for r in rs)
            for b in ("off_path", "no_chain", "intact"):
                row[f"break_{b}"] = sum(r["break"] == b for r in rs)
            row["break_else"] = len(rs) - sum(row[f"break_{b}"] for b in ("off_path", "no_chain", "intact"))
            tab.append(row)

    place = {i: place_of(traces[i], *gold[i]) for i in ids}
    dtab = []
    for shape in ("parallel", "chain", "all"):
        for dg in (True, False):
            qs = [i for i in ids if (shape == "all" or plan_shape(plan_step(traces[i])["plan"]) == shape)
                  and bool(dangling(plan_step(traces[i])["plan"])) == dg]
            if not qs:
                continue
            row = {"shape": shape, "dangling": int(dg), "n": len(qs),
                   "bridge": sum(rows[i]["qtype"] == "bridge" for i in qs),
                   "em": round(sum(float(rows[i]["em"]) for i in qs) / len(qs), 4),
                   "wrong": sum(not float(rows[i]["em"]) for i in qs)}
            for p in PLACES:
                row[f"wrong_{p}"] = sum(not float(rows[i]["em"]) and place[i] == p for i in qs)
            dtab.append(row)
    return per, tab, dtab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", default=["data/processed/hotpotqa_dev.jsonl"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    rows, traces = load_run(args.results, args.run)
    gold = load_gold(args.data, rows)
    titles = load_titles(args.data, rows)
    out = Path(args.results) / args.out
    for name, t in zip(("rows", "asked", "dangling"), tables(rows, traces, gold, titles)):
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
