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
  distractor_para   the harder version: up to --n-para-distractors sentences
               from other pools that share a content token with the question
               (so they score against it) and one with the text of a bridge
               paragraph (so they look like the gold the question has to
               reach). Same surface, and a pool built to compete with the
               bridge for the same top-k slot. --kinds picks which twins to
               build.
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

from src.memory.failures import _MUSIQUE_PARAGRAPH, paragraph_of
from src.memory.features import tokens

PROCESSED = Path(__file__).resolve().parent / "processed"

# tokens that carry no entity information; keeps the distractor pull on names
STOP = frozenset(
    "the a an of and in on at to for by with from or as is was were be been film "
    "band album song book novel series season episode new york city county state "
    "united states american british english university school college company".split())

# function words the first STOP list lets through; a sentence that shares only
# these with a paragraph does not look like it. Used by the bridge_paragraph
# twin only, so the bridge_token twin stays as it was reported
FUNCTION = frozenset(
    "also she her his he him its it they their them this that these those which who whom "
    "whose what when where has had have having not one two first after before into over "
    "than then there here only both all any some more most other such been being does did "
    "while during between under about".split())

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


def paragraph_tokens(text):
    """content_tokens without the function words: what makes a sentence
    score on a question or resemble a paragraph."""
    return [t for t in content_tokens(text) if t not in FUNCTION]


def mention(question, title):
    """(start, length) of the title's mention in the question (the title
    without its trailing parenthetical, case-insensitive), or None."""
    name = plain_title(title)
    at = question.lower().find(name.lower()) if name else -1
    return (at, len(name)) if at >= 0 else None


def gold_titles(record):
    """The gold paragraph titles in gold order (paragraph_of reads the
    title from either id form, so musique records get twins too)."""
    return list(dict.fromkeys(paragraph_of(g) for g in record["gold_chunk_ids"]))


def is_paragraph_pool(record):
    """musique pools are paragraphs, one chunk per paragraph with the pool
    position as sent_idx; hotpot and 2wiki pools are sentences."""
    return any(_MUSIQUE_PARAGRAPH.fullmatch(c["chunk_id"].partition("::")[0]) for c in record["chunks"][:1])


def first_sentence(record, title):
    """The first sentence of the titled paragraph: the sent_idx 0 chunk on
    sentence pools. On a paragraph pool sent_idx is the pool position, so
    there is no first sentence to read and the alias rules do not fire
    (until 5 Oct they read whatever paragraph sat at position 0, which gave
    four musique aliases, two of them wrong)."""
    if is_paragraph_pool(record):
        return ""
    return next((c["text"] for c in record["chunks"]
                 if c["title"] == title and c.get("sent_idx", -1) == 0), "")


def pool_contents(record):
    """Chunk ids and texts already in the pool. A distractor is skipped on
    either: on musique the same paragraph recurs across pools under
    different p{idx}:: ids, so an id check alone let a copy of the pool's
    own gold paragraph in as a distractor (54 of 484 twins of the first 500
    questions); on hotpot and 2wiki sentence ids are global and the text
    check changes nothing."""
    have = {c["chunk_id"] for c in record["chunks"]}
    have.update(c["text"] for c in record["chunks"])
    return have


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
    have = pool_contents(record)
    candidates, shared = {}, Counter()
    for title in bridges:
        for tok in sorted(set(content_tokens(title))):
            for chunk, source in sentence_index.get(tok, ()):
                if source != record["id"] and chunk["chunk_id"] not in have and chunk["text"] not in have:
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


def bridge_paragraph_twin(record, sentence_index, n_distractors, rng):
    """The harder distractor twin (plan, Stage 1b): sentences from other
    pools that share a content token with the question and one with the
    text of a bridge paragraph, the most shared tokens first.

    The bridge_token twin barely moves any arm (4.7% of pairs), because a
    sentence that shares a token with the bridge title alone rarely scores
    against the question. Here every added sentence scores on the question
    (it shares a query token, so bm25 ranks it) and looks like the bridge
    paragraph (it shares a token with the paragraph's text, so it competes
    with the gold for the same slot). Hypothesis: this is enough to flip an
    arm's outcome on a useful share of pairs while the question stays
    identical, which is the "same surface, different situation" pair the
    boundary study needs."""
    bridges = bridge_titles(record)
    if not bridges:
        return None
    have = pool_contents(record)
    q_tokens = set(paragraph_tokens(record["question"]))
    b_tokens = set()
    for c in record["chunks"]:
        if c["title"] in bridges:
            b_tokens.update(paragraph_tokens(c["text"]))
    b_tokens -= q_tokens        # tokens the question already carries do not make it look like the bridge
    candidates, shared = {}, {}
    for tok in sorted(q_tokens):
        for chunk, source in sentence_index.get(tok, ()):
            cid = chunk["chunk_id"]
            if source == record["id"] or cid in have or chunk["text"] in have or cid in candidates:
                continue
            s_tokens = set(paragraph_tokens(chunk["text"]))
            with_bridge = len(s_tokens & b_tokens)
            if with_bridge:
                candidates[cid] = chunk
                shared[cid] = (len(s_tokens & q_tokens), with_bridge)
    if not candidates:
        return None
    order = sorted(candidates, key=lambda c: (-sum(shared[c]), -shared[c][1], rng.random()))
    picked = order[:n_distractors]
    twin = dict(record)
    twin["id"] = f"{record['id']}::twin:distractor_para"
    twin["chunks"] = list(record["chunks"]) + [dict(candidates[c]) for c in picked]
    twin["twin_of"], twin["twin_kind"] = record["id"], "distractor_para"
    twin["twin_rule"] = "bridge_paragraph"
    twin["twin_change"] = {"bridge_titles": bridges, "added": picked,
                           "shared": {c: list(shared[c]) for c in picked}}
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
            for tok in sorted(set(content_tokens(c["text"]))):
                index.setdefault(tok, []).append((c, r["id"]))
    return index


KINDS = ("alias", "distractor", "distractor_para")


def make_twins(records, n_distractors=5, seed=13, kinds=KINDS, n_para_distractors=10):
    """One twin per record and kind in `kinds`, in record order. The
    distractor_para twin takes n_para_distractors sentences (more than the
    bridge_token twin, which is kept as it was for comparison)."""
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown twin kinds {sorted(unknown)}; choose from {KINDS}")
    rng = random.Random(seed)
    sentence_index = build_sentence_index(records)
    twins, counts = [], Counter()
    for r in records:
        made = []
        if "alias" in kinds:
            made.append(alias_twin(r))
        if "distractor" in kinds:
            made.append(distractor_twin(r, sentence_index, n_distractors, rng))
        if "distractor_para" in kinds:
            made.append(bridge_paragraph_twin(r, sentence_index, n_para_distractors, rng))
        for twin in made:
            if twin is not None:
                twins.append(twin)
                counts[(twin["twin_kind"], twin["twin_rule"])] += 1
    return twins, counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=None, help="first n questions of the dev file")
    ap.add_argument("--n-distractors", type=int, default=5, help="for the bridge_token twin")
    ap.add_argument("--n-para-distractors", type=int, default=10,
                    help="for the bridge_paragraph twin")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
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
    twins, counts = make_twins(records, args.n_distractors, args.seed, kinds=args.kinds,
                               n_para_distractors=args.n_para_distractors)
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
