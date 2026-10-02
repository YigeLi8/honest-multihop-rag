"""Constructed twins for the look-alike pair study (docs/plan.md, Stage 1b).

A twin is a copy of a dev question with one thing changed, so that a pair
(original, twin) is a look-alike pair by construction: the retrieval
situation is the same except for the one change, and whether a lesson
learned on one should apply to the other is known from how the twin was
made. Two kinds, from a processed dev file:

  distractor   the question is left as it is and the pool gains up to
               --n-distractors sentences taken from other questions' pools
               that share a content token with the bridge title (a gold
               paragraph title the question does not name). Same surface,
               harder lexical situation: a memory keyed on the question text
               sees the same question, the retrievers see a different pool.
  alias        the pool is left as it is and one gold title the question
               names verbatim is replaced by an alias of it: the "known as"
               form from the paragraph's first sentence when there is one,
               else the surname of a two-word name. Different surface, same
               pool: a memory keyed on the question text sees a new question,
               the retrievers see almost the same situation (less so for
               bm25, which loses the title's tokens).

Only questions the rule applies to get a twin of that kind (--report says
how many). Twin ids are "<id>::twin:<kind>" and each record carries twin_of
and twin_kind; gold ids are unchanged, because the gold paragraphs are kept
in every twin. Gold fields are used here to build evaluation data, as the
intervention arms do; nothing here is a feature or a memory key.

    python -m data.make_twins --dataset hotpotqa [--n 1500] [--seed 13]
    -> data/processed/hotpotqa_twins_dev.jsonl

Evaluated by running the arms config on the twin file and joining with the
original run in src.memory.lookalike.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.memory.features import tokens

PROCESSED = Path(__file__).resolve().parent / "processed"

# tokens that carry no entity information; keeps the distractor pull on names
STOP = frozenset(
    "the a an of and in on at to for by with from or as is was were be been film "
    "band album song book novel series season episode new york city county state "
    "united states american british english university school college company".split())

_KNOWN_AS = re.compile(
    r"(?:also|commonly|better|often|professionally|popularly|simply|widely)?\s*known as\s+"
    r"(?:the\s+)?([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,4})")


_BIOGRAPHY = re.compile(r"\bborn\b|\(\d{4}\s*[-\u2013]\s*\d{4}\)")
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_CAP_STOP = frozenset("And The Of In On At For Or A An".split())


def plain_title(title):
    """The title without its trailing disambiguator, case kept."""
    return _TRAILING_PAREN.sub("", title).strip()


def content_tokens(title):
    """Alphabetic tokens of three letters or more that are not stop words;
    years and other numbers are left out."""
    return [t for t in tokens(title) if t not in STOP and len(t) > 2 and t.isalpha()]


def mention(question, title):
    """(start, length) of the title's mention in the question (the title
    without its trailing parenthetical, case-insensitive), or None."""
    name = plain_title(title)
    at = question.lower().find(name.lower()) if name else -1
    return (at, len(name)) if at >= 0 else None


def gold_titles(record):
    return list(dict.fromkeys(g.split("::")[0] for g in record["gold_chunk_ids"]))


def first_sentence(record, title):
    return next((c["text"] for c in record["chunks"]
                 if c["title"] == title and c.get("sent_idx", -1) == 0), "")


def alias_of(record, title):
    """(alias, rule) for a title the question names, or None. The surname
    rule only fires for a two-word name whose paragraph reads like a
    biography (a birth date or a life span in its first sentence)."""
    first = first_sentence(record, title)
    m = _KNOWN_AS.search(first)
    name = plain_title(title)
    if m:
        words = m.group(1).strip(" .").split()
        while words and words[-1] in _CAP_STOP:
            words.pop()
        alias = " ".join(words)
        if alias and alias.lower() != name.lower() and alias.lower() not in record["question"].lower():
            return alias, "known_as"
    words = name.split()
    if len(words) == 2 and all(w[0].isupper() and w.isalpha() for w in words) and _BIOGRAPHY.search(first):
        return words[1], "surname"
    return None


def alias_twin(record):
    """Replace the first gold title the question names with its alias."""
    q = record["question"]
    for title in gold_titles(record):
        found = mention(q, title)
        if found is None:
            continue
        at, length = found
        aliased = alias_of(record, title)
        if aliased is None:
            continue
        alias, rule = aliased
        twin = dict(record)
        twin["id"] = f"{record['id']}::twin:alias"
        twin["question"] = q[:at] + alias + q[at + length:]
        twin["twin_of"], twin["twin_kind"], twin["twin_rule"] = record["id"], "alias", rule
        twin["twin_change"] = {"title": title, "alias": alias}
        return twin
    return None


def bridge_titles(record):
    """Gold titles the question does not name: the paragraphs a retriever
    has to reach through the others."""
    return [t for t in gold_titles(record) if mention(record["question"], t) is None]


def distractor_twin(record, sentence_index, n_distractors, rng):
    """Add sentences from other pools that share a content token with a
    bridge title. sentence_index: token -> [(chunk dict, source id)]."""
    bridges = bridge_titles(record)
    if not bridges:
        return None
    have = {c["chunk_id"] for c in record["chunks"]}
    candidates, shared = {}, Counter()
    for title in bridges:
        for tok in set(content_tokens(title)):
            for chunk, source in sentence_index.get(tok, ()):
                if source != record["id"] and chunk["chunk_id"] not in have:
                    candidates[chunk["chunk_id"]] = chunk
                    shared[chunk["chunk_id"]] += 1
    if not candidates:
        return None
    # the sentences sharing the most bridge tokens first, random inside a tie
    order = sorted(candidates, key=lambda c: (-shared[c], rng.random()))
    picked = order[:n_distractors]
    twin = dict(record)
    twin["id"] = f"{record['id']}::twin:distractor"
    twin["chunks"] = list(record["chunks"]) + [dict(candidates[c]) for c in picked]
    twin["twin_of"], twin["twin_kind"], twin["twin_rule"] = record["id"], "distractor", "bridge_token"
    twin["twin_change"] = {"bridge_titles": bridges, "added": picked}
    return twin


def build_sentence_index(records):
    """token -> [(chunk, source id)] over every pool, content tokens only,
    for sentences that are gold nowhere (a distractor must not be evidence)."""
    index = {}
    for r in records:
        gold = set(r["gold_chunk_ids"])
        for c in r["chunks"]:
            if c["chunk_id"] in gold:
                continue
            for tok in set(content_tokens(c["text"])):
                index.setdefault(tok, []).append((c, r["id"]))
    return index


def make_twins(records, n_distractors=5, seed=13):
    rng = random.Random(seed)
    sentence_index = build_sentence_index(records)
    twins, counts = [], Counter()
    for r in records:
        for twin in (alias_twin(r), distractor_twin(r, sentence_index, n_distractors, rng)):
            if twin is not None:
                twins.append(twin)
                counts[(twin["twin_kind"], twin["twin_rule"])] += 1
    return twins, counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=None, help="first n questions of the dev file")
    ap.add_argument("--n-distractors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    src = PROCESSED / f"{args.dataset}_dev.jsonl"
    records = []
    with open(src) as f:
        for line in f:
            records.append(json.loads(line))
            if args.n and len(records) >= args.n:
                break
    twins, counts = make_twins(records, args.n_distractors, args.seed)
    out = Path(args.out) if args.out else PROCESSED / f"{args.dataset}_twins_dev.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in twins:
            f.write(json.dumps(t) + "\n")
    print(f"{len(records)} questions -> {len(twins)} twins in {out}")
    for (kind, rule), n in sorted(counts.items()):
        print(f"  {kind:11s} {rule:13s} {n}")


if __name__ == "__main__":
    main()
