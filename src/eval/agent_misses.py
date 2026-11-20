"""The answers no search retrieved, and lookups that restate the question,
read from one main-agent run with no model (docs/plan.md, Stage 6b, From
the traces).

    python -m src.eval.agent_misses \
        --run orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore \
        --out agent_misses_hotpot_explore

Writes two tables:

  results/<out>_never.csv    the wrong answers whose gold string is in no
                             retrieved sentence (agent_losses.place_of says
                             not_retrieved), by MISS kind, with where the
                             paragraph that holds the answer was named
                             (NAMED) and whether the other gold paragraphs
                             were retrieved
  results/<out>_restate.csv  every question by plan shape and question type,
                             split by whether some planned lookup restates the
                             question (restates, below), with EM and the
                             wrong answers by place

MISS, the first that holds:

  title          the gold string is the title of a retrieved paragraph (the
                 sentence test reads sentences, so a title does not count)
  not_in_pool    no sentence of the question's pool holds the string either:
                 the answer is written otherwise, or split over sentences, so
                 no search could have retrieved it by this test
  para_partly    the paragraph of a pool sentence holding the string was
                 retrieved, but not that sentence
  para_unseen    no sentence of that paragraph was retrieved

For para_partly and para_unseen, the answer paragraph is the gold paragraph
holding the string when there is one, else the first pool paragraph that
does. NAMED says where its title was first named, in this order: in the
question itself, in a planned or asked lookup, in a search query, in a
lookup's answer only, or never. Also counted there: whether the
answer sentence is not its paragraph's first (sentence_not_first), and
whether a sentence next to it in the paragraph was retrieved
(neighbour_retrieved). A title is named when its words (without a
parenthesis such as "(film)") all occur in the text, in any order.

A lookup restates the question when it holds at least RESTATE of the
question's content words (normalized, without stopwords). That is a word
count and nothing more: a lookup that copies a two-hop question whole and one
that asks for the same thing in other words fall on different sides of it.
Gold (answers, gold_chunk_ids) is read only to place and score.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from src.eval.agent_losses import PLACES, holds, load_gold, load_run, place_of, write
from src.eval.agent_traces import _STOP, load_titles, lookups, plan_shape, plan_step, title_of
from src.eval.answer_metrics import normalize

MISS = ("title", "not_in_pool", "para_partly", "para_unseen")
NAMED = ("question", "lookup", "query", "answer_only", "never")
RESTATE = 0.8

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_QSTOP = _STOP | {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is",
                  "are", "was", "were", "did", "does", "do", "that", "this", "it", "its", "be",
                  "as", "or", "name", "has", "have", "had"}


def content_words(text):
    return set(normalize(text).split()) - _QSTOP


def names(text, title):
    """Every word of the title (without a trailing parenthesis) is in text."""
    want = set(normalize(_PAREN.sub("", title)).split())
    return bool(want) and want <= set(normalize(text).split())


def restates(lookup, question, share=RESTATE):
    """The lookup holds at least share of the question's content words."""
    q = content_words(question)
    return bool(q) and len(q & content_words(lookup)) / len(q) >= share


def answer_paragraph(golds, texts, gold_titles):
    """The title of the paragraph holding the answer string: a gold one first,
    else the first pool paragraph in pool order; None when no sentence holds
    it."""
    found = [title_of(c) for c, t in texts.items() if any(holds(t, g) for g in golds)]
    gold = [t for t in found if t in gold_titles]
    return (gold or found or [None])[0]


def miss_kind(trace, golds, texts, gold_titles):
    """(kind, answer paragraph title or None) for a question whose string no
    retrieved sentence holds; kind is one of MISS."""
    got = [c for h in trace["hops"] for c in h["retrieved"]]
    got_titles = {title_of(c) for c in got}
    if any(normalize(_PAREN.sub("", t)) == normalize(g) or normalize(t) == normalize(g)
           for t in got_titles for g in golds):
        return "title", None
    para = answer_paragraph(golds, texts, gold_titles)
    if para is None:
        return "not_in_pool", None
    return ("para_partly" if para in got_titles else "para_unseen"), para


def where_named(trace, title, question):
    """Where a paragraph's title was first named (one of NAMED)."""
    if names(question, title):
        return "question"
    subs = lookups(trace)
    if any(names(s.get("planned", ""), title) or names(s.get("sub_question", ""), title)
           for s in subs):
        return "lookup"
    if any(names(h.get("query", ""), title) for h in trace["hops"]):
        return "query"
    if any(names(s.get("sub_answer", ""), title) for s in subs):
        return "answer_only"
    return "never"


def other_gold_seen(trace, para, gold_titles):
    """Whether the gold paragraphs other than the answer's were retrieved: all,
    some, none, or n/a when there is no other."""
    others = [t for t in gold_titles if t != para]
    if not others:
        return "n/a"
    got = {title_of(c) for h in trace["hops"] for c in h["retrieved"]}
    seen = sum(t in got for t in others)
    return "all" if seen == len(others) else ("some" if seen else "none")


def answer_sentence(trace, golds, texts, para):
    """(the answer paragraph's first sentence holding the string is not its
    first sentence, a sentence next to it was retrieved)."""
    idx = sorted(int(c.rsplit("::", 1)[1]) for c, t in texts.items()
                 if title_of(c) == para and any(holds(t, g) for g in golds))
    got = {c for h in trace["hops"] for c in h["retrieved"]}
    j = idx[0]
    return j > 0, any(f"{para}::{k}" in got for k in (j - 1, j + 1))


def tables(rows, traces, gold, titles):
    ids = list(rows)
    place = {i: place_of(traces[i], *gold[i]) for i in ids}
    right = {i: bool(float(rows[i]["em"])) for i in ids}
    question = {i: plan_step(traces[i]).get("question", "") for i in ids}

    never = [i for i in ids if not right[i] and place[i] == "not_retrieved"]
    per = {}
    for i in never:
        kind, para = miss_kind(traces[i], gold[i][0], gold[i][1], titles[i])
        per[i] = (kind, para)
    ntab = []
    for kind in MISS:
        qs = [i for i in never if per[i][0] == kind]
        row = {"miss": kind, "n": len(qs),
               "bridge": sum(rows[i]["qtype"] == "bridge" for i in qs),
               "comparison": sum(rows[i]["qtype"] == "comparison" for i in qs)}
        located = kind in ("para_partly", "para_unseen")
        for w in NAMED:
            row[f"named_{w}"] = (sum(where_named(traces[i], per[i][1], question[i]) == w for i in qs)
                                 if located else "")
        for w in ("all", "some", "none", "n/a"):
            row[f"other_gold_{w}"] = (sum(other_gold_seen(traces[i], per[i][1], titles[i]) == w
                                          for i in qs) if located else "")
        row["answer_para_gold"] = sum(per[i][1] in titles[i] for i in qs) if located else ""
        sents = [answer_sentence(traces[i], gold[i][0], gold[i][1], per[i][1]) for i in qs] if located else []
        row["sentence_not_first"] = sum(a for a, _ in sents) if located else ""
        row["neighbour_retrieved"] = sum(b for _, b in sents) if located else ""
        for shape in ("one", "parallel", "chain"):
            row[f"plan_{shape}"] = sum(plan_shape(plan_step(traces[i])["plan"]) == shape for i in qs)
        ntab.append(row)
    total = {"miss": "all", "n": len(never)}
    for k in ntab[0]:
        if k not in ("miss", "n"):
            vals = [r[k] for r in ntab if r[k] != ""]
            total[k] = sum(vals)
    ntab.append(total)

    rtab = []
    for qt in sorted({rows[i]["qtype"] for i in ids}):
        for shape in ("one", "parallel", "chain", "all"):
            for rs in (True, False):
                qs = [i for i in ids if rows[i]["qtype"] == qt
                      and (shape == "all" or plan_shape(plan_step(traces[i])["plan"]) == shape)
                      and any(restates(p, question[i]) for p in plan_step(traces[i])["plan"]) == rs]
                if not qs:
                    continue
                row = {"qtype": qt, "shape": shape, "restates": int(rs), "n": len(qs),
                       "em": round(sum(right[i] for i in qs) / len(qs), 4),
                       "wrong": sum(not right[i] for i in qs)}
                for p in PLACES:
                    row[f"wrong_{p}"] = sum(not right[i] and place[i] == p for i in qs)
                row["lookups"] = round(sum(len(plan_step(traces[i])["plan"]) for i in qs) / len(qs), 2)
                rtab.append(row)
    return ntab, rtab


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
    for name, t in zip(("never", "restate"), tables(rows, traces, gold, titles)):
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
