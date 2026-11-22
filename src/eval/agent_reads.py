"""What the final read gave when the answer string was in what it read, without
a model (docs/plan.md, Stage 6b, From the traces).

    python -m src.eval.agent_reads --run orchestrator_guard_wide_unsup_hotpot_explore \
        --out agent_reads_hotpot_explore

Reads one run (results/<run>_{qa.csv,trace.jsonl}) and the gold of its
questions, takes the questions whose gold answer string was in a sentence the
answer was read from (agent_losses.place_of == in_context), and writes:

  results/<out>_kind.csv      the wrong ones by what the reply was (READS,
                              below), with how many repeat a lookup's answer,
                              how many had a lookup answer right, and on how
                              many the guard gave the final read a second read
  results/<out>_position.csv  all of them, right and wrong, by the position of
                              the first sentence holding the string in what was
                              read, by how many sentences and how many
                              paragraphs were read

What a wrong reply was, the first that holds, in this order (READS):

  yesno           yes or no
  inside_gold     a span inside the gold answer (agent_losses.reply_of)
  around_gold     a span around it
  gold_words      the reply's words are a subset of the gold's or the other way
                  round, not contiguous: "Larry Drake" for Larry Richard Drake,
                  "West Point Foundry" for West Point Iron and Cannon Foundry.
                  The gold under another form, which EM counts wrong
  picked_other    the reply is in the question, and so is the gold or the
                  question is a comparison: of two things the question names,
                  the other one (a comparison whose gold is written otherwise
                  than in the question, "712 5th Avenue", lands here too)
  in_question     the reply is in the question
  bridge_entity   the reply is the title of a gold paragraph (a parenthesis at
                  the end of the title dropped): an entity on the path given
                  for the answer, mostly the one the question goes through, or
                  the answer's own paragraph under a name the gold does not use
  other_title     the reply is the title of another retrieved paragraph
  same_sentence   the reply is in a sentence read that also holds the gold
                  string
  in_read         the reply is in another sentence read
  retrieved_only  the reply is in a retrieved sentence that was not read
  nowhere         in no retrieved sentence

Every test is the exact string test of agent_losses after the answer
normalization, so a reply written otherwise than in its sentence falls to
"nowhere", and bridge_entity needs the reply to be the title as written.
Gold (answers, gold paragraph titles) is read only to place and score.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.eval.agent_losses import holds, load_gold, load_run, place_of, reply_of, write
from src.eval.agent_traces import load_titles, lookups, title_of
from src.eval.answer_metrics import em, normalize

READS = ("yesno", "inside_gold", "around_gold", "gold_words", "picked_other", "in_question",
         "bridge_entity", "other_title", "same_sentence", "in_read", "retrieved_only", "nowhere")
POSITIONS = (("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 999))
SIZES = (("1-2", 1, 2), ("3-4", 3, 4), ("5+", 5, 999))

_STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with", "from"}
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def _words(text):
    return set(normalize(text).split()) - _STOP


def same_words(a, b):
    """One's words are a subset of the other's (stopwords aside)."""
    x, y = _words(a), _words(b)
    return bool(x) and bool(y) and (x <= y or y <= x)


def names_title(pred, titles):
    """The reply is one of the titles, with or without a parenthesis at the end."""
    p = normalize(pred)
    return bool(p) and any(p in (normalize(t), normalize(_PAREN.sub("", t))) for t in titles)


def read_ids(trace):
    """The sentences the final answer was read from, in order, as
    agent_losses.place_of reads them: the notebook; with "+thin" in
    stop_reason the notebook and then everything else retrieved; with
    "+fallback", or for a mode (or an agent) with no notebook, everything
    retrieved."""
    retrieved = list(dict.fromkeys(c for h in trace["hops"] for c in h["retrieved"]))
    nb = list(dict.fromkeys(trace.get("notebook") or []))
    stop = trace.get("stop_reason") or ""
    if not nb or "+fallback" in stop:
        return retrieved
    if "+thin" in stop:
        return nb + [c for c in retrieved if c not in nb]
    return nb


def retrieved_ids(trace):
    return {c for h in trace["hops"] for c in h["retrieved"]}


def read_kind(pred, golds, question, gold_titles, trace, texts, qtype=""):
    """What a wrong reply was, one of READS."""
    r = reply_of(pred, golds)
    if r in ("yesno", "inside_gold", "around_gold"):
        return r
    if r == "empty":
        return "nowhere"
    if any(same_words(pred, g) for g in golds):
        return "gold_words"
    if holds(question, pred):
        if qtype == "comparison" or any(holds(question, g) for g in golds):
            return "picked_other"
        return "in_question"
    if names_title(pred, gold_titles):
        return "bridge_entity"
    retrieved = retrieved_ids(trace)
    if names_title(pred, {title_of(c) for c in retrieved}):
        return "other_title"
    read = read_ids(trace)

    def among(ids):
        return any(holds(texts.get(c, ""), pred) for c in ids)
    if among([c for c in read if any(holds(texts.get(c, ""), g) for g in golds)]):
        return "same_sentence"
    if among(read):
        return "in_read"
    return "retrieved_only" if among(retrieved) else "nowhere"


def gold_position(trace, golds, texts):
    """1-based position of the first sentence read that holds a gold string,
    or None."""
    for k, c in enumerate(read_ids(trace)):
        if any(holds(texts.get(c, ""), g) for g in golds):
            return k + 1
    return None


def lookup_echo(pred, golds, trace):
    """(the reply equals a lookup's answer, a lookup's answer equals the gold)."""
    subs = lookups(trace)
    return (any(em(s["sub_answer"], pred) for s in subs),
            any(em(s["sub_answer"], g) for s in subs for g in golds))


def final_guarded(trace):
    """The guard gave the final read a second read (yes/no or unsupported)."""
    return any(g.get("where") == "final" for g in trace.get("guards", []))


def _bucket(x, buckets):
    return next(name for name, lo, hi in buckets if lo <= x <= hi)


def load_questions(data_files, ids):
    """{id: question text}."""
    want, out = set(ids), {}
    for data_file in data_files:
        with open(data_file) as f:
            for line in f:
                if line[:80].split('"')[3] in want:
                    d = json.loads(line)
                    out[d["id"]] = d["question"]
    return out


def tables(rows, traces, gold, titles, questions):
    ids = [i for i in rows if place_of(traces[i], *gold[i]) == "in_context"]
    right = {i: bool(float(rows[i]["em"])) for i in ids}
    wrong = [i for i in ids if not right[i]]
    kind, echo = {}, {}
    for i in wrong:
        kind[i] = read_kind(rows[i]["answer_pred"], gold[i][0], questions[i], titles[i],
                            traces[i], gold[i][1], rows[i]["qtype"])
        echo[i] = lookup_echo(rows[i]["answer_pred"], gold[i][0], traces[i])
    ktab = []
    for k in READS:
        qs = [i for i in wrong if kind[i] == k]
        ktab.append({"kind": k, "n": len(qs),
                     "bridge": sum(rows[i]["qtype"] == "bridge" for i in qs),
                     "comparison": sum(rows[i]["qtype"] == "comparison" for i in qs),
                     "repeats_lookup": sum(echo[i][0] for i in qs),
                     "lookup_right": sum(echo[i][1] for i in qs),
                     "guard_final": sum(final_guarded(traces[i]) for i in qs)})
    total = {"kind": "all", "n": len(wrong)}
    total.update({c: sum(r[c] for r in ktab) for c in ktab[0] if c not in ("kind", "n")})
    ktab.append(total)

    ptab = []
    feats = {
        "position": lambda i: _bucket(gold_position(traces[i], *gold[i]), POSITIONS),
        "sentences": lambda i: _bucket(len(read_ids(traces[i])), SIZES),
        "paragraphs": lambda i: str(min(len({title_of(c) for c in read_ids(traces[i])}), 4))
        .replace("4", "4+"),
    }
    for feat, f in feats.items():
        groups = Counter(f(i) for i in ids)
        for g in sorted(groups):
            qs = [i for i in ids if f(i) == g]
            ptab.append({"by": feat, "group": g, "n": len(qs), "right": sum(right[i] for i in qs),
                         "em": round(sum(right[i] for i in qs) / len(qs), 4),
                         "wrong_other_reply": sum(not right[i] and kind[i] not in
                                                  ("yesno", "inside_gold", "around_gold",
                                                   "gold_words") for i in qs)})
    return ktab, ptab


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
    questions = load_questions(args.data, rows)
    out = Path(args.results) / args.out
    for name, t in zip(("kind", "position"), tables(rows, traces, gold, titles, questions)):
        write(f"{out}_{name}.csv", t)
        print(f"== {name}")
        w = csv.DictWriter(sys.stdout, fieldnames=list(t[0]))
        w.writeheader()
        w.writerows(t)


if __name__ == "__main__":
    main()
