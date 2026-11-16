"""What the main agent's traces say about its losses, without a model
(docs/plan.md, Stage 6b, From the traces).

    python -m src.eval.agent_traces --run orchestrator_guard_wide_hotpot_explore \
        --out agent_traces_hotpot_explore

Reads one main-agent run (results/<run>_{qa.csv,trace.jsonl}) and the gold
of its questions, and writes three tables:

  results/<out>_chain.csv    every question by where its answer string was
                             (agent_losses.place_of) and by the first thing
                             that broke its chain of lookups (CHAIN, below),
                             with right answers counted beside the wrong ones
  results/<out>_dropped.csv  for the wrong answers whose string was retrieved
                             and not kept: which subagent was shown the
                             sentence with the string, and whether a lookup
                             came back with the answer
  results/<out>_plan.csv     plan shape (one lookup, parallel, chain) against
                             the outcome, by question type
  results/<out>_keep.csv     every keep step that was shown a sentence with
                             the answer string, by the position of the first
                             such sentence in the list shown, and how often
                             the step kept one
  results/<out>_search2.csv  the second search of every lookup, by how it
                             relates to the lookup, for the never-retrieved
                             wrong answers against the rest

The first break of a chain, in this order (CHAIN):

  cut               the plan's reply listed more lookups than the cap kept
  non_answer        a lookup's answer that is not an answer ("no", "not
                    specified", "None", a yes/no reply to a lookup that does
                    not read as yes/no) was pasted into a later lookup by #k
  off_path          a lookup's answer pasted into a later lookup names no
                    paragraph the gold evidence is in (a gold-derived test,
                    read only here for scoring)
  last_non_answer   the last lookup came back with an answer that is not an
                    answer
  no_chain          a bridge question planned without any #k: one lookup, or
                    lookups that do not use each other's answers
  intact            none of the above

"off_path" is a word test (on_path): a pasted answer is on the path when it
shares a word, other than a stopword or a word of the question, with the
title of a gold paragraph. An acronym or another inflection of the title
counts as off the path, and a wrong answer sharing a word with a gold title
counts as on it, so off_path is approximate both ways; a sample of the calls
is read by hand in docs/plan.md.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.agent_losses import PLACES, holds, load_gold, load_run, place_of, write
from src.eval.answer_metrics import em, normalize

CHAIN = ("cut", "non_answer", "off_path", "last_non_answer", "no_chain", "intact")
SEARCH2 = ("none", "same", "narrower", "new_terms")

_REF = re.compile(r"#(\d+)")
_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+?)\s*$")
_AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have",
        "had", "will", "would", "should", "shall", "may", "might"}
_WH = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|name)\b", re.I)
_PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "this", "that"}
_REFUSAL = re.compile(
    r"\b(not (specified|provided|mentioned|stated|given|available|known|clear)"
    r"|no (information|answer|mention)"
    r"|does not (say|specify|provide|mention|state|contain|include)"
    r"|do not (say|specify|provide|mention|state|contain|include)"
    r"|cannot be (determined|answered)|unknown|insufficient)\b", re.I)


def looks_yesno(question):
    """The wording rule of the answer guard (src/pipeline/multihop.py), copied
    so that this module reads traces without importing the pipeline."""
    words = question.split()
    return (bool(words) and words[0].strip("\"'(").lower() in _AUX) or not _WH.search(question)


def is_non_answer(answer, lookup):
    """An answer that is not one: empty, a refusal, "none", or yes/no to a
    lookup that does not read as a yes/no question."""
    a = normalize(answer)
    if a in ("", "none", "n a", "unknown") or a in _PRONOUNS:
        return True
    if a in ("yes", "no"):
        return not looks_yesno(lookup)
    return bool(_REFUSAL.search(answer))


def lookups(trace):
    """The lookup steps in order: one per subagent, with its answer."""
    return [s for s in trace.get("steps", []) if s.get("role") == "subagent" and "sub_answer" in s]


def plan_step(trace):
    return next(s for s in trace["steps"] if s.get("role") == "planner")


def refs(planned):
    """The earlier lookups (0-based) a planned lookup refers to by #k."""
    return {int(k) - 1 for k in _REF.findall(planned)}


def plan_shape(plan):
    if len(plan) <= 1:
        return "one"
    return "chain" if any(refs(p) for p in plan) else "parallel"


def plan_cut(trace):
    """The plan's reply had more numbered lookups than were kept."""
    p = plan_step(trace)
    n = sum(bool(_LINE.match(line)) for line in p.get("reply", "").splitlines())
    return n > len(p["plan"])


def title_of(chunk_id):
    return chunk_id.rsplit("::", 1)[0]


_STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with", "from",
         "film", "band", "album", "song", "series", "tv", "season", "s"}


def words(text):
    return set(normalize(text).split()) - _STOP


def on_path(answer, titles, question=""):
    """A pasted answer names a paragraph the gold evidence is in: it shares a
    word with a gold paragraph's title that is not a stopword and not already
    in the question. Catches "Daniel Castellaneta" for Dan Castellaneta, misses an
    acronym ("IBF") or another inflection ("cranberries" for Cranberry)."""
    a = words(answer) - words(question)
    return any(a & words(t) for t in titles)


def first_break(trace, qtype, titles, question=""):
    """(kind, lookup index or None): the first thing that broke the chain,
    one of CHAIN."""
    if plan_cut(trace):
        return "cut", None
    subs = lookups(trace)
    for k, s in enumerate(subs):
        later = [t for t in subs[k + 1:] if k in refs(t["planned"])]
        if not later:
            continue
        if is_non_answer(s["sub_answer"], s["sub_question"]):
            return "non_answer", k
        if not on_path(s["sub_answer"], titles, question):
            return "off_path", k
    if subs and is_non_answer(subs[-1]["sub_answer"], subs[-1]["sub_question"]):
        return "last_non_answer", len(subs) - 1
    if qtype == "bridge" and plan_shape(plan_step(trace)["plan"]) != "chain":
        return "no_chain", None
    return "intact", None


def holders(trace, golds, texts):
    """Retrieved chunk ids whose sentence holds a gold answer string."""
    got = {c for h in trace["hops"] for c in h["retrieved"]}
    return {c for c in got if c in texts and any(holds(texts[c], g) for g in golds)}


def shown_to(trace, ids):
    """The subagents (0-based) whose keep step was shown one of ids."""
    return sorted({s["sub"] for s in trace.get("steps", [])
                   if "shown" in s and s.get("role") == "subagent" and set(s["shown"]) & ids})


POSITIONS = (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7+", 7, 99))


def keep_positions(trace, ids):
    """(position of the first sentence in ids in the list shown, kept one of
    ids) for every keep step that was shown one; positions from 1."""
    out = []
    for s in trace.get("steps", []):
        if "shown" in s and s.get("role") == "subagent":
            pos = [j + 1 for j, c in enumerate(s["shown"]) if c in ids]
            if pos:
                out.append((pos[0], bool(set(s.get("kept", [])) & ids)))
    return out


def dropped_row(trace, golds, texts):
    """For one dropped question: who was shown the sentence and what the
    lookups came back with."""
    subs = lookups(trace)
    last = len(subs) - 1
    who = shown_to(trace, holders(trace, golds, texts))
    right = [k for k, s in enumerate(subs) if any(em(s["sub_answer"], g) for g in golds)]
    if not who:
        where = "not_shown"
    elif last in who:
        where = "last_lookup"
    else:
        where = "earlier_lookup"
    return {"shown": where, "n_shown": len(who),
            "sub_right": "last" if last in right else ("earlier" if right else "none")}


def search2_kind(lookup, query2):
    """How a lookup's second search relates to the lookup: none, same words,
    a subset of them (narrower), or with words the lookup did not have."""
    if not query2:
        return "none"
    a, b = set(normalize(lookup).split()), set(normalize(query2).split())
    if b == a:
        return "same"
    return "narrower" if b <= a else "new_terms"


def second_searches(trace):
    """(lookup, second query or None) for every lookup of a trace."""
    by_sub = defaultdict(list)
    for s in trace.get("steps", []):
        if "shown" in s and s.get("role") == "subagent":
            by_sub[s["sub"]].append(s)
    out = []
    for k, s in enumerate(lookups(trace)):
        searches = by_sub.get(k, [])
        out.append((s["sub_question"], searches[1]["query"] if len(searches) > 1 else None))
    return out


def load_titles(data_files, ids):
    """{id: gold paragraph titles}, read from gold_chunk_ids (outcome only)."""
    want, out = set(ids), {}
    for data_file in data_files:
        with open(data_file) as f:
            for line in f:
                if line[:80].split('"')[3] in want:
                    d = json.loads(line)
                    out[d["id"]] = sorted({title_of(c) for c in d.get("gold_chunk_ids", [])})
    return out


def tables(rows, traces, gold, titles):
    ids = list(rows)
    place = {i: place_of(traces[i], *gold[i]) for i in ids}
    right = {i: bool(float(rows[i]["em"])) for i in ids}
    brk = {i: first_break(traces[i], rows[i]["qtype"], titles[i],
                          plan_step(traces[i]).get("question", ""))[0] for i in ids}

    chain = []
    for kind in CHAIN:
        qs = [i for i in ids if brk[i] == kind]
        row = {"break": kind, "n": len(qs), "right": sum(right[i] for i in qs),
               "wrong": sum(not right[i] for i in qs)}
        for p in PLACES:
            row[f"wrong_{p}"] = sum(not right[i] and place[i] == p for i in qs)
        row["em"] = round(row["right"] / len(qs), 4) if qs else ""
        chain.append(row)

    dropped = [i for i in ids if not right[i] and place[i] == "dropped"]
    drows = {i: dropped_row(traces[i], *gold[i]) for i in dropped}
    dtab = []
    for kind in CHAIN:
        qs = [i for i in dropped if brk[i] == kind]
        if not qs:
            continue
        row = {"break": kind, "n": len(qs),
               "comparison": sum(rows[i]["qtype"] == "comparison" for i in qs)}
        for w in ("last_lookup", "earlier_lookup", "not_shown"):
            row[f"shown_{w}"] = sum(drows[i]["shown"] == w for i in qs)
        row["shown_two_or_more"] = sum(drows[i]["n_shown"] > 1 for i in qs)
        for w in ("last", "earlier", "none"):
            row[f"sub_right_{w}"] = sum(drows[i]["sub_right"] == w for i in qs)
        dtab.append(row)
    total = {"break": "all", "n": len(dropped)}
    for k in dtab[0] if dtab else []:
        if k not in ("break", "n"):
            total[k] = sum(r[k] for r in dtab)
    dtab.append(total)

    ptab = []
    for qt in sorted({rows[i]["qtype"] for i in ids}):
        for shape in ("one", "parallel", "chain"):
            qs = [i for i in ids if rows[i]["qtype"] == qt
                  and plan_shape(plan_step(traces[i])["plan"]) == shape]
            if not qs:
                continue
            row = {"qtype": qt, "shape": shape, "n": len(qs),
                   "em": round(sum(right[i] for i in qs) / len(qs), 4),
                   "wrong": sum(not right[i] for i in qs)}
            for p in PLACES:
                row[f"wrong_{p}"] = sum(not right[i] and place[i] == p for i in qs)
            row["cut"] = sum(plan_cut(traces[i]) for i in qs)
            ptab.append(row)

    stab = []
    for group, keep in (("not_retrieved_wrong", lambda i: not right[i] and place[i] == "not_retrieved"),
                        ("other", lambda i: not (not right[i] and place[i] == "not_retrieved"))):
        qs = [i for i in ids if keep(i)]
        c = Counter(search2_kind(lk, q2) for i in qs for lk, q2 in second_searches(traces[i]))
        on = sum(bool(q2) and any(holds(q2, t) for t in titles[i])
                 for i in qs for _, q2 in second_searches(traces[i]))
        row = {"group": group, "questions": len(qs), "lookups": sum(c.values())}
        row.update({k: c[k] for k in SEARCH2})
        row["second_names_gold_title"] = on
        stab.append(row)
    ktab = []
    steps = [(i, pos, kept) for i in ids
             for pos, kept in keep_positions(traces[i], holders(traces[i], *gold[i]))]
    for name, lo, hi in POSITIONS:
        got = [(i, kept) for i, pos, kept in steps if lo <= pos <= hi]
        ktab.append({"position": name, "steps": len(got), "kept": sum(k for _, k in got),
                     "kept_share": round(sum(k for _, k in got) / len(got), 4) if got else "",
                     "steps_of_dropped_wrong": sum(not right[i] and place[i] == "dropped"
                                                   for i, _ in got)})
    return chain, dtab, ptab, ktab, stab


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
    for name, t in zip(("chain", "dropped", "plan", "keep", "search2"), tables(rows, traces, gold, titles)):
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
