"""Model-free features of one retrieval step.

Everything here is computed from what the retriever saw and what it returned:
the query string, the original question, the ranked hits, the candidate pool,
the hop index, the chunk ids of earlier hops and the previous hop's features.
Nothing comes from the dataset's labels. The features describe the situation
before its outcome is known, so whatever later conditions on them cannot peek.
compute_features never receives the example record, and tests/test_memory.py
reads this file as text and fails if a label field is named in it.

Features differ in what they cost. FEATURE_STAGE says what must have happened
before each one can be read: some need only the query, some the pool, some this
hop's retrieval. Comparisons with a second retriever (probe_features) cost that
retriever's run, so they are kept out of the feature dict and logged in a block
of their own: a model that reads them has paid for two retrievals, not one.

The schema is versioned and frozen: changing what a feature means, adding or
removing one bumps FEATURE_SCHEMA_VERSION, and records written under different
versions are not mixed. A test pins this file's hash to the version.
"""
import math
import re
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Union

from src.types import Chunk, RetrievedChunk

Number = Union[int, float]

FEATURE_SCHEMA_VERSION = 1

# name -> the earliest point at which the feature is available:
#   query         the query string, the original question and the hop index
#   pool          the above plus the question's candidate pool, nothing retrieved yet
#   history       the above plus the earlier hops of the same question
#   primary_hits  the above plus this hop's hits from the strategy in use
# The order of this mapping is part of the schema.
FEATURE_STAGE: Mapping[str, str] = MappingProxyType({
    "query_len_tokens": "query",        # tokens in the query
    "query_is_question": "query",       # 1 if the query is the original question verbatim, else 0;
                                        # equals hop == 0 in the current pipeline modes
    "hop": "query",                     # 0-based hop index
    "n_pool_chunks": "pool",            # chunks in this question's candidate pool
    "n_pool_paragraphs": "pool",        # distinct paragraphs those chunks come from
    "top1_score": "primary_hits",       # score of the first hit (0.0 without hits); raw, so on
                                        # the scale of the strategy that produced it
    "top2_score": "primary_hits",       # score of the second hit (0.0 without one);
                                        # top1_score - score_margin
    "score_margin": "primary_hits",     # top1_score - top2_score
    "score_ratio": "primary_hits",      # top2_score / top1_score when both are positive, else
                                        # 0.0; the scale-free version of the margin
    "score_entropy": "primary_hits",    # entropy of the positive hit scores, normalised to [0, 1]
    "unmatched_token_ratio": "pool",    # share of distinct query tokens found in no pool chunk
    "rare_token_ratio": "pool",         # share of distinct query tokens found in exactly one
    "query_title_mentions": "pool",     # distinct pool titles whose normalised form is in the query
    "query_cap_spans": "query",         # runs of two or more capitalised words in the query
    "top1_query_overlap": "primary_hits",   # token Jaccard between the query and the first hit
    "n_new_chunks": "primary_hits",     # hits that no earlier hop of this question returned
    "prev_top1_score": "history",       # top1_score of the previous hop (0.0 at hop 0)
    "n_hits": "primary_hits",           # number of hits returned; min(k, pool size)
    "n_scored_hits": "primary_hits",    # hits with a positive score (the rest are filler)
})

# A feature vector is [features[name] for name in FEATURES].
FEATURES: tuple[str, ...] = tuple(FEATURE_STAGE)

# Per shadow arm, from probe_features. Not in FEATURES: the set of arms is a
# property of the run, and each arm's pair costs one more retrieval.
PROBE_FEATURES: tuple[str, ...] = (
    "topk_jaccard",          # Jaccard of the two first-stage lists' chunk ids
    "top1_same",             # 1 if both lists start with the same chunk, else 0
)

_TOKEN = re.compile(r"[a-z0-9]+")
_WORD_OR_PUNCT = re.compile(r"\w[\w'’-]*|[^\w\s]")
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")

# Words that open a question and carry a capital only because they come first.
_LEADING: frozenset[str] = frozenset(
    "what which who whom whose when where why how "
    "is are was were do does did has have had can could will would should "
    "the a an this that these those both "
    "in on at of for from by with to between during after before".split())


def tokens(text: str) -> list[str]:
    """Lowercased runs of ascii letters and digits. Deliberately simple: no
    stemming, no stopword list, nothing that would need a model or a download."""
    return _TOKEN.findall(text.lower())


def _jaccard(a: Collection, b: Collection) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def score_entropy(scores: Sequence[float]) -> float:
    """Entropy of the positive scores treated as a distribution, divided by
    log(n) so that 1.0 means a flat list and 0.0 a single dominant hit.
    Non-positive scores are dropped (bm25 zeros, negative reranker logits);
    with fewer than two positive scores the entropy is 0.0."""
    pos = [s for s in scores if s > 0]
    if len(pos) < 2:
        return 0.0
    total = sum(pos)
    h = -sum((s / total) * math.log(s / total) for s in pos)
    return h / math.log(len(pos))


def token_rarity(query_tokens: Sequence[str], chunks: Sequence[Chunk]) -> tuple[float, float]:
    """(unmatched, rare): the shares of the distinct query tokens that occur in
    no chunk of the pool and in exactly one. They are kept apart because they
    pull in opposite directions for a lexical retriever: a token the pool never
    uses cannot be matched at all, a token only one chunk uses pins that chunk
    down."""
    distinct = set(query_tokens)
    if not distinct:
        return 0.0, 0.0
    df = dict.fromkeys(distinct, 0)
    for c in chunks:
        for t in distinct & set(tokens(c.text)):
            df[t] += 1
    counts = list(df.values())
    return counts.count(0) / len(distinct), counts.count(1) / len(distinct)


def normalise_title(title: str) -> str:
    """Title as a space-joined token string, with a trailing parenthetical
    dropped ("Ed Wood (film)" -> "ed wood"): questions name the entity, not
    the disambiguator."""
    return " ".join(tokens(_TRAILING_PAREN.sub("", title)))


def title_mentions(query_tokens: Sequence[str], chunks: Sequence[Chunk]) -> int:
    """Number of distinct pool titles whose normalised form occurs in the query
    as a contiguous token sequence. Titles are counted as the pool spells
    them, so "Ed Wood" and "Ed Wood (film)" in one pool are two mentions for a
    query naming Ed Wood: the query does not say which of them it means."""
    padded = " " + " ".join(query_tokens) + " "
    forms = (normalise_title(t) for t in {c.title for c in chunks if c.title})
    return sum(1 for form in forms if form and f" {form} " in padded)


def cap_spans(text: str) -> int:
    """Count maximal runs of two or more consecutive capitalised words.
    Punctuation ends a run, so "Derrickson, Ed Wood" holds one span. A question
    or function word at the very start ("Which", "The", "Are", ...) is skipped:
    its capital comes from its position, and "What American actor" names no
    two-word entity."""
    words = _WORD_OR_PUNCT.findall(text)
    if words and words[0].lower() in _LEADING:
        words = words[1:]
    spans, run = 0, 0
    for tok in words:
        if tok[0].isupper():
            run += 1
            continue
        spans += run >= 2
        run = 0
    return spans + (run >= 2)


def pool_paragraphs(chunks: Sequence[Chunk]) -> int:
    """Chunk ids are "{paragraph}::{suffix}" for every dataset in this repo,
    so the paragraph is the id up to the last separator."""
    return len({c.chunk_id.rsplit("::", 1)[0] for c in chunks})


def compute_features(query: str,
                     hits: Sequence[RetrievedChunk],
                     chunks: Sequence[Chunk],
                     hop: int,
                     question: str,
                     prev: Optional[Mapping[str, Number]] = None,
                     seen_ids: Collection[str] = (),
                     ) -> dict[str, Number]:
    """Features of one retrieval step, keyed in FEATURES order.

    query        the string sent to the retriever at this hop
    hits         what came back, ranked (after the reranker when there is one)
    chunks       the question's candidate pool
    hop          0-based hop index
    question     the original question
    prev         the features this function returned for the previous hop
    seen_ids     chunk ids returned at earlier hops of the same question

    Deterministic: no randomness, no model, no state kept between calls.
    Floats are rounded to 6 places so that a replay writes the same record.
    """
    q_tokens = tokens(query)
    scores = [float(h.score) for h in hits]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    ids = [h.chunk.chunk_id for h in hits]
    earlier = set(seen_ids)
    unmatched, rare = token_rarity(q_tokens, chunks)

    out: dict[str, Number] = {
        "query_len_tokens": len(q_tokens),
        "query_is_question": int(query == question),
        "hop": int(hop),
        "n_pool_chunks": len(chunks),
        "n_pool_paragraphs": pool_paragraphs(chunks),
        "top1_score": round(top1, 6),
        "top2_score": round(top2, 6),
        "score_margin": round(top1 - top2, 6),
        "score_ratio": round(top2 / top1, 6) if top1 > 0 and top2 > 0 else 0.0,
        "score_entropy": round(score_entropy(scores), 6),
        "unmatched_token_ratio": round(unmatched, 6),
        "rare_token_ratio": round(rare, 6),
        "query_title_mentions": title_mentions(q_tokens, chunks),
        "query_cap_spans": cap_spans(query),
        "top1_query_overlap": round(_jaccard(q_tokens, tokens(hits[0].chunk.text)), 6)
        if hits else 0.0,
        "n_new_chunks": sum(1 for i in ids if i not in earlier),
        "prev_top1_score": float(prev["top1_score"]) if prev else 0.0,
        "n_hits": len(hits),
        "n_scored_hits": sum(1 for s in scores if s > 0),
    }
    assert tuple(out) == FEATURES
    return out


def probe_features(first_stage: Sequence[RetrievedChunk],
                   shadow_hits: Optional[Mapping[str, Sequence[RetrievedChunk]]] = None,
                   ) -> dict[str, dict[str, Number]]:
    """{arm: {topk_jaccard, top1_same, depth}}: how far each shadow arm's list
    is from the primary retriever's.

    first_stage  the primary retriever's hits before any reranker, so that two
                 retrievers are compared and not a retriever with a reranker
    shadow_hits  {arm: hits} from the retrievers run beside it

    Both lists are cut to the shorter one's length (depth) first: an arm that
    returns the same ranking as the primary gets a Jaccard of 1.0 whatever the
    two depths are. Arms come out in sorted order.
    """
    ids = [h.chunk.chunk_id for h in first_stage]
    out: dict[str, dict[str, Number]] = {}
    for arm in sorted(shadow_hits or {}):
        other = [h.chunk.chunk_id for h in shadow_hits[arm]]
        depth = min(len(ids), len(other))
        a, b = ids[:depth], other[:depth]
        out[arm] = {
            "topk_jaccard": round(_jaccard(a, b), 6),
            "top1_same": int(depth > 0 and a[0] == b[0]),
            "depth": depth,
        }
    assert all(tuple(v)[:len(PROBE_FEATURES)] == PROBE_FEATURES for v in out.values())
    return out
