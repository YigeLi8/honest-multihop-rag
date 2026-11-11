"""Checks for the experience logging: features, logger, hooks, backfill.

No real data and no models: fake retrievers and generators, plus bm25 on a
handful of synthetic sentences for the end-to-end check.

Run from the repo root: python -m tests.test_memory (or pytest).
"""
import ast
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from src.memory.features import (FEATURE_SCHEMA_VERSION, FEATURE_STAGE, FEATURES,
                                 PROBE_FEATURES, cap_spans, compute_features,
                                 probe_features, title_mentions, tokens)
from src.types import Chunk, Example, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent
TIMING = ("latency_s", "tokens_per_s", "peak_memory_mb", "ttft_s")

# src/memory/features.py as frozen under each schema version
FEATURES_SHA256 = {1: "610852ebf19c481e37e281ef8e6839baca838a5e99b0be937675ba12bdee3acf"}

POOL = [
    Chunk("Book X::0", "Book X was written by Author Y.", "Book X", 0),
    Chunk("Book X::1", "It sold well in Europe.", "Book X", 1),
    Chunk("Author Y::0", "Author Y is a writer from Norway.", "Author Y", 0),
    Chunk("Author Y::1", "Author Y attended Some University.", "Author Y", 1),
    Chunk("Some University (Oslo)::0", "Some University is in Oslo.", "Some University (Oslo)", 0),
    Chunk("Distractor::0", "Nothing to see here.", "Distractor", 0),
]
QUESTION = "Which university did the author of Book X attend?"


def example(**kw):
    base = dict(id="q1", question=QUESTION, answer="Some University", hops=2,
                qtype="bridge", chunks=list(POOL),
                gold_chunk_ids=["Book X::0", "Author Y::1"],
                reasoning_path=[{"gold_chunk_id": "Book X::0"},
                                {"gold_chunk_id": "Author Y::1"}])
    base.update(kw)
    return Example(**base)


def hits_of(*pairs):
    by_id = {c.chunk_id: c for c in POOL}
    return [RetrievedChunk(chunk=by_id[cid], score=score, rank=r + 1)
            for r, (cid, score) in enumerate(pairs)]


class OverlapRetriever:
    """Ranks the pool by shared lowercase words with the query; ties keep pool order."""

    def __init__(self, reverse=False):
        self.reverse = reverse
        self.chunks = []
        self.queries = []
        self.depths = []

    def index(self, chunks):
        self.chunks = list(chunks)

    def retrieve(self, query, top_k):
        self.queries.append(query)
        self.depths.append(top_k)
        q = set(query.lower().replace("?", "").split())
        scored = [(len(q & set(c.text.lower().replace(".", "").split())), c)
                  for c in self.chunks]
        scored.sort(key=lambda t: t[0] if self.reverse else -t[0])
        return [RetrievedChunk(chunk=c, score=float(s), rank=r + 1)
                for r, (s, c) in enumerate(scored[:top_k])]


class SlowRetriever(OverlapRetriever):
    """Takes its time, to stand in for a shadow arm with an encoder."""

    def index(self, chunks):
        time.sleep(0.03)
        super().index(chunks)

    def retrieve(self, query, top_k):
        time.sleep(0.03)
        return super().retrieve(query, top_k)


class TailReranker:
    """Keeps the last top_n candidates, best last: the opposite of the first stage."""

    def __init__(self, top_n):
        self.top_n = top_n

    def rerank(self, query, candidates):
        kept = list(reversed(candidates))[:self.top_n]
        return [RetrievedChunk(chunk=c.chunk, score=float(self.top_n - i), rank=i + 1)
                for i, c in enumerate(kept)]


class ScriptedGen:
    """One SEARCH, then DONE, then the answer."""

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, prompt, max_tokens=None):
        from src.serving.runner_mlx import GenMetrics
        self.calls += 1
        self.prompts.append(prompt)
        m = GenMetrics(tokens_per_s=10.0, ttft_s=0.1, peak_memory_mb=1.0,
                       completion_tokens=5, prompt_tokens=20)
        if "gathering evidence" in prompt:
            return ("SEARCH: Author Y attended which university" if self.calls == 1
                    else "DONE"), m
        return "Some University\n", m


def stub_cfg(mode, store=None, top_k=3, log_deeper=0):
    cfg = SimpleNamespace(
        run=SimpleNamespace(name="stub"),
        dataset=SimpleNamespace(name="synthetic"),
        retrieval=SimpleNamespace(method="overlap", top_k=top_k),
        pipeline=SimpleNamespace(mode=mode, max_hops=3))
    if store is not None:
        cfg.memory = SimpleNamespace(enabled=True, mode="log", log_arms=[], store=str(store),
                                     log_deeper=log_deeper)
    return cfg


def stub_pipeline(mode, store=None, shadow=None, generate=True, top_k=3, rerank_top_n=None,
                  log_deeper=0):
    """A pipeline on fakes; with a store it also carries a logger."""
    from src.memory.experience import ExperienceLogger
    from src.pipeline.multihop import MultiHopPipeline
    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg = stub_cfg(mode, store, top_k, log_deeper)
    p.retriever = OverlapRetriever()
    p.reranker = None
    if rerank_top_n:
        p.cfg.rerank = SimpleNamespace(enabled=True, top_n=rerank_top_n)
        p.reranker = TailReranker(rerank_top_n)
    p.generator = ScriptedGen() if generate else None
    if store is not None:
        p.memory = ExperienceLogger(p.cfg, "stub")
        p.shadow = shadow or {}
    return p


def read_log(path):
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    return lines[0], lines[1:]


def comparable(ex, res):
    """What run_qa writes for a question, minus the wall-clock columns."""
    from src.eval.run_qa import score_row, trace_record
    row = {k: v for k, v in score_row(ex, res).items() if k not in TIMING}
    return json.dumps(trace_record(ex, res)), row


def test_feature_schema():
    hits = hits_of(("Book X::0", 3.0), ("Author Y::1", 3.0), ("Distractor::0", 0.0))
    a = compute_features(QUESTION, hits, POOL, 0, QUESTION)
    b = compute_features(QUESTION, hits, POOL, 0, QUESTION)
    assert a == b and json.dumps(a) == json.dumps(b)             # deterministic
    assert tuple(a) == FEATURES and FEATURE_SCHEMA_VERSION == 1
    assert isinstance(FEATURES, tuple)                            # frozen
    assert all(type(v) in (int, float) for v in a.values())
    # every feature says what it costs to read, in the same order
    assert tuple(FEATURE_STAGE) == FEATURES
    assert set(FEATURE_STAGE.values()) == {"query", "pool", "history", "primary_hits"}
    try:
        FEATURE_STAGE["hop"] = "pool"
        raise AssertionError("FEATURE_STAGE should be read-only")
    except TypeError:
        pass


def test_feature_schema_is_pinned():
    """A feature cannot change meaning without the version moving."""
    digest = hashlib.sha256((ROOT / "src" / "memory" / "features.py").read_bytes()).hexdigest()
    assert FEATURES_SHA256.get(FEATURE_SCHEMA_VERSION) == digest, (
        "src/memory/features.py changed. If a feature was added, removed or redefined, bump "
        f"FEATURE_SCHEMA_VERSION; then pin {digest} for that version in FEATURES_SHA256.")


def test_feature_values():
    hits = hits_of(("Book X::0", 3.0), ("Author Y::1", 3.0), ("Distractor::0", 0.0))
    f = compute_features(QUESTION, hits, POOL, 0, QUESTION)
    assert f["query_len_tokens"] == 9 and f["query_is_question"] == 1 and f["hop"] == 0
    assert f["n_pool_chunks"] == 6 and f["n_pool_paragraphs"] == 4
    assert f["top1_score"] == 3.0 and f["top2_score"] == 3.0 and f["score_margin"] == 0.0
    assert f["score_entropy"] == 1.0          # two equal positive scores, the zero is dropped
    assert f["score_ratio"] == 1.0
    # which, did, the, of, attend occur in no chunk ("attended" is another token),
    # book and x in one; university and author in more
    assert f["unmatched_token_ratio"] == round(5 / 9, 6)
    assert f["rare_token_ratio"] == round(2 / 9, 6)
    assert f["query_title_mentions"] == 1     # "Book X"; "Some University" is not in the query
    assert f["query_cap_spans"] == 1          # "Book X"; a lone capital does not count
    assert f["top1_query_overlap"] == round(3 / 13, 6)   # {book, x, author} over the union
    assert f["n_new_chunks"] == 3 and f["prev_top1_score"] == 0.0 and f["n_hits"] == 3
    assert f["n_scored_hits"] == 2            # the zero-score hit is filler

    query = "Author Y attended Some University, Oslo"
    hop1 = hits_of(("Author Y::1", 4.0), ("Some University (Oslo)::0", 1.0))
    g = compute_features(query, hop1, POOL, 1, QUESTION, prev=f,
                         seen_ids={"Book X::0", "Author Y::1", "Distractor::0"})
    assert g["query_is_question"] == 0 and g["hop"] == 1
    assert g["score_margin"] == 3.0 and 0.0 < g["score_entropy"] < 1.0
    assert g["score_ratio"] == 0.25
    assert g["query_title_mentions"] == 2     # "Author Y", and "Some University" without "(Oslo)"
    assert g["query_cap_spans"] == 2          # "Author Y", "Some University"; the comma ends it
    assert g["n_new_chunks"] == 1 and g["prev_top1_score"] == 3.0

    empty = compute_features("", [], POOL, 0, QUESTION)
    assert empty["n_hits"] == 0 and empty["top1_score"] == 0.0 and empty["score_entropy"] == 0.0
    assert empty["unmatched_token_ratio"] == 0.0 and empty["rare_token_ratio"] == 0.0
    assert empty["top1_query_overlap"] == 0.0 and empty["score_ratio"] == 0.0
    negative = compute_features(QUESTION, hits_of(("Book X::0", -1.0), ("Author Y::1", -2.0)),
                                POOL, 0, QUESTION)
    assert negative["score_ratio"] == 0.0 and negative["n_scored_hits"] == 0


def test_query_shape_features():
    # a question word in first position is capitalised by position, not as a name
    assert cap_spans("What American actor starred in it?") == 0
    assert cap_spans("Which University of Oslo alumni wrote Book X?") == 1
    assert cap_spans("The Oberoi family is part of which hotel company?") == 0
    assert cap_spans("Are Random House Tower and 888 7th Avenue both in New York?") == 2
    assert cap_spans("Ed Wood directed which film?") == 1      # a name in first position counts
    assert cap_spans("") == 0

    # titles are counted as the pool spells them: two entries that share a name are two
    pool = [Chunk("Ed Wood::0", "Ed Wood was a filmmaker.", "Ed Wood", 0),
            Chunk("Ed Wood (film)::0", "Ed Wood is a 1994 film.", "Ed Wood (film)", 0),
            Chunk("Ed Wood (film)::1", "Tim Burton directed it.", "Ed Wood (film)", 1),
            Chunk("Tim Burton::0", "Tim Burton is a director.", "Tim Burton", 0)]
    assert title_mentions(tokens("Was Ed Wood a director?"), pool) == 2
    assert title_mentions(tokens("Who is Tim Burton?"), pool) == 1
    assert title_mentions(tokens("Who directed Edward Woodward?"), pool) == 0


def test_probe_features():
    """Comparisons with a second retriever live apart from the features."""
    hits = hits_of(("Book X::0", 3.0), ("Author Y::1", 2.0))
    same = probe_features(hits, {"bm25": hits})
    assert same == {"bm25": {"topk_jaccard": 1.0, "top1_same": 1, "depth": 2}}
    assert tuple(same["bm25"])[:2] == PROBE_FEATURES
    other = hits_of(("Author Y::1", 0.9), ("Distractor::0", 0.1))
    f = probe_features(hits, {"dense": other, "bm25": hits})
    assert f["dense"] == {"topk_jaccard": round(1 / 3, 6), "top1_same": 0, "depth": 2}
    assert list(f) == ["bm25", "dense"]
    # lists of different length are compared at the shorter one's depth
    longer = hits_of(("Book X::0", 3.0), ("Author Y::1", 2.0), ("Distractor::0", 0.0))
    assert probe_features(hits, {"deep": longer})["deep"] == \
        {"topk_jaccard": 1.0, "top1_same": 1, "depth": 2}
    assert probe_features(hits) == {} and probe_features([], {"x": []})["x"]["top1_same"] == 0
    # and none of it reaches the feature dict
    assert not set(compute_features(QUESTION, hits, POOL, 0, QUESTION)) & \
        {f"{name}_{arm}" for name in PROBE_FEATURES for arm in ("bm25", "dense")}


def test_features_blind_to_gold():
    """Two examples that differ only in their labels get the same features."""
    from src.memory.experience import ExperienceLogger
    with tempfile.TemporaryDirectory() as tmp:
        recs = []
        for ex in (example(),
                   example(answer="something else", answer_aliases=["x"],
                           gold_chunk_ids=["Distractor::0"], reasoning_path=[],
                           gold_supporting_facts=[{"title": "Distractor", "sent_idx": 0}],
                           evidence_triples=[["a", "b", "c"]])):
            log = ExperienceLogger(stub_cfg("single_hop", tmp), "blind")
            hits = hits_of(("Book X::0", 3.0), ("Author Y::1", 2.0))
            log.on_hop(ex, 0, ex.question, hits, {"other": hits[:1]})
            recs.append(log.records()[0])
        assert recs[0].features == recs[1].features
        assert recs[0].probe_features == recs[1].probe_features
        assert recs[0].retrieved == recs[1].retrieved
        assert recs[0].hop_outcome != recs[1].hop_outcome       # outcomes do use gold


def test_features_source_has_no_gold():
    """The feature module must not name a label field, or import anything that
    could hand it one."""
    path = ROOT / "src" / "memory" / "features.py"
    text = path.read_text()
    for name in ("gold_chunk_ids", "reasoning_path", "evidence_triples",
                 "gold_supporting_facts", "is_supporting", "answer", "gold"):
        assert name not in text.lower(), f"{name} appears in {path}"
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"math", "re", "collections.abc", "types", "typing", "src.types"}, imported
    # and neither function is ever handed the example itself
    args = {f.name: [a.arg for a in f.args.args] for f in ast.walk(ast.parse(text))
            if isinstance(f, ast.FunctionDef)}
    assert args["compute_features"] == ["query", "hits", "chunks", "hop", "question", "prev",
                                        "seen_ids"]
    assert args["probe_features"] == ["first_stage", "shadow_hits"]


def test_deeper_window_and_features():
    """The probe stage of Stage 5b: ranks k+1..2k of the primary's full
    ranking and the gold-free features over them, by hand on the pool."""
    from dataclasses import asdict

    from src.eval.run_qa import score_row, trace_record
    from src.memory.backfill import replay
    from src.memory.deepen import DEEPER_FEATURES, deeper_features, deeper_window
    r = OverlapRetriever()
    r.index(POOL)
    full = r.retrieve(QUESTION, len(POOL))
    assert [h.chunk.chunk_id for h in full] == ["Book X::0", "Author Y::1", "Author Y::0",
                                                 "Some University (Oslo)::0", "Book X::1",
                                                 "Distractor::0"]
    window = deeper_window(full, 2, 2)
    assert [(h.chunk.chunk_id, h.score, h.rank) for h in window] == \
        [("Author Y::0", 1.0, 3), ("Some University (Oslo)::0", 1.0, 4)]
    assert deeper_window(full, 2, 0) == [] and deeper_window(full, 10, 2) == []

    # hop 0: the context is the top-2; both window titles are named in it
    # ("Author Y", "Some University"), one window paragraph is new, the gap
    # at the cut is (2 - 1) / (3 - 1), the mass below it 2 / 5
    f0 = deeper_features(full, 2, 2, full[:2], POOL, QUESTION, QUESTION)
    assert list(f0) == list(DEEPER_FEATURES)
    assert f0 == {"deep_n": 2, "deep_named_question": 0, "deep_named_query": 0,
                  "deep_named_context": 2, "deep_new_paragraphs": 1, "deep_gap_at_k": 0.5,
                  "deep_score_share": 0.4, "deep_max_query_overlap": round(1 / 13, 6)}
    # hop 1 on the follow-up query: the query names Author Y, the gap at the
    # cut is zero (a tie at ranks 2 and 3) and the pool chunks seen at hop 0
    # count as context
    query = "Author Y attended which university"
    full1 = r.retrieve(query, len(POOL))
    assert [h.chunk.chunk_id for h in full1[:4]] == ["Author Y::1", "Book X::0", "Author Y::0",
                                                      "Some University (Oslo)::0"]
    f1 = deeper_features(full1, 2, 2, full1[:2], POOL, QUESTION, query,
                         seen_ids={"Book X::0", "Author Y::1"})
    assert f1 == {"deep_n": 2, "deep_named_question": 0, "deep_named_query": 1,
                  "deep_named_context": 2, "deep_new_paragraphs": 1, "deep_gap_at_k": 0.0,
                  "deep_score_share": 0.5, "deep_max_query_overlap": 0.2}
    # a pool that ends at k has no window and every feature is zero
    assert deeper_features(full, 10, 2, full, POOL, QUESTION, QUESTION) == \
        dict.fromkeys(DEEPER_FEATURES, 0)

    with tempfile.TemporaryDirectory() as tmp:
        # logged only when memory.log_deeper is set, from the same full ranking
        # that gold_ranks reads, and blind to the labels
        recs = []
        for ex in (example(), example(gold_chunk_ids=["Distractor::0"], reasoning_path=[],
                                      answer="x")):
            p = stub_pipeline("multihop_ircot", store=tmp, top_k=2, log_deeper=2)
            res = p.run_example(ex)
            p.memory.on_outcome(ex, res, score_row(ex, res))
            recs.append(p.memory.records())
        first, second = recs[0]
        assert first.deeper == [{"chunk_id": "Author Y::0", "score": 1.0, "rank": 3},
                                {"chunk_id": "Some University (Oslo)::0", "score": 1.0, "rank": 4}]
        assert first.deeper_features == f0 and second.deeper_features == f1
        assert first.hop_outcome["gold_ranks"]["Author Y::1"] == [2, 2.0]
        assert [x.deeper for x in recs[1]] == [x.deeper for x in recs[0]]
        assert [x.deeper_features for x in recs[1]] == [x.deeper_features for x in recs[0]]
        assert recs[1][0].hop_outcome != first.hop_outcome
        header = p.memory.header()
        assert header["log_deeper"] == 2 and header["deeper_features"] == list(DEEPER_FEATURES)
        assert header["deeper_stage"] == "deeper"

        # the window is off by default, and a replay from the trace writes it back
        off = stub_pipeline("multihop_ircot", store=tmp, top_k=2)
        assert off.memory.log_deeper == 0
        res = off.run_example(example())
        assert all(x.deeper is None and x.deeper_features is None for x in off.memory.records())
        ex = example()
        live = stub_pipeline("multihop_ircot", store=tmp, top_k=2, log_deeper=2)
        res = live.run_example(ex)
        row = score_row(ex, res)
        live.memory.on_outcome(ex, res, row)
        again = stub_pipeline("multihop_ircot", store=tmp, top_k=2, log_deeper=2, generate=False)
        traces = [json.loads(json.dumps(trace_record(ex, res)))]
        rows = {ex.id: {k: str(v) for k, v in row.items()}}
        assert replay(again, [ex], traces, rows)[1] == []
        assert [asdict(x) for x in again.memory.records()] == \
            [asdict(x) for x in live.memory.records()]


def test_deeper_source_has_no_gold():
    """src/memory/deepen.py: no label field named, nothing imported that could
    hand it one, and the example never passed in."""
    path = ROOT / "src" / "memory" / "deepen.py"
    text = path.read_text()
    for name in ("gold_chunk_ids", "reasoning_path", "evidence_triples",
                 "gold_supporting_facts", "is_supporting", "answer", "gold"):
        assert name not in text.lower(), f"{name} appears in {path}"
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"collections.abc", "types", "typing", "src.types",
                        "src.memory.features"}, imported
    args = {f.name: [a.arg for a in f.args.args] for f in ast.walk(ast.parse(text))
            if isinstance(f, ast.FunctionDef)}
    assert args["deeper_features"] == ["full", "k", "mult", "hits", "chunks", "question",
                                       "query", "seen_ids"]


def test_logger_records():
    from src.eval.run_qa import score_row
    from src.memory.experience import OUTCOME_FIELDS
    with tempfile.TemporaryDirectory() as tmp:
        p = stub_pipeline("multihop_ircot", store=tmp, top_k=1)
        examples = [example(), example(id="q2", reasoning_path=[], qtype="comparison")]
        n_hops = 0
        for ex in examples:
            res = p.run_example(ex)
            p.generator.calls = 0
            n_hops += len(res.hops)
            p.memory.on_outcome(ex, res, score_row(ex, res))
        path = p.memory.close()
        assert path == Path(tmp) / "stub_experience.jsonl"

        header, records = read_log(path)
        assert header["header"] is True and header["run"] == "stub"
        assert header["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert header["config"]["memory"]["store"] == tmp
        assert header["config"]["retrieval"] == {"method": "overlap", "top_k": 1}
        assert {"git_sha", "git_dirty", "created", "platform"} <= set(header)
        assert header["record_schema_version"] == 2
        assert header["log_deeper"] == 0 and header["deeper_features"] == []
        assert header["features"] == list(FEATURES)
        assert header["feature_stage"] == dict(FEATURE_STAGE)
        assert header["probe_features"] == list(PROBE_FEATURES)
        assert header["features_sha256"] == FEATURES_SHA256[FEATURE_SCHEMA_VERSION]
        assert header["strategy"]["method"] == "overlap" and header["shadow_arms"] == {}

        assert n_hops == 4 and len(records) == n_hops       # one line per (question, hop)
        assert [(r["id"], r["hop"]) for r in records] == [("q1", 0), ("q1", 1),
                                                          ("q2", 0), ("q2", 1)]
        first, second = records[0], records[1]
        assert first["run"] == "stub" and first["dataset"] == "synthetic"
        assert first["qtype"] == "bridge" and first["query"] == QUESTION
        assert second["query"] == "Author Y attended which university"
        assert first["strategy"] == {"method": "overlap", "top_k": 1, "hybrid_alpha": None,
                                     "rerank_enabled": False, "rerank_top_n": None,
                                     "pipeline_mode": "multihop_ircot"}
        assert first["retrieved"] == [{"chunk_id": "Book X::0", "score": 3.0, "rank": 1}]
        assert first["feature_schema_version"] == 1
        assert list(first["features"]) == list(FEATURES)
        assert second["features"]["prev_top1_score"] == first["features"]["top1_score"]

        # hop 0 finds Book X::0, hop 1 adds Author Y::1; the other gold chunk
        # sits at rank 2 of the whole pool both times, just outside top_k = 1
        assert first["hop_outcome"] == {
            "hop_precision": 1.0, "hop_recall": 0.5, "hop_para_recall": 0.5,
            "cumulative_recall": 0.5, "all_gold_in_topk": False,
            "gold_hit_ids": ["Book X::0"], "new_gold": ["Book X::0"], "aligned_hit": True,
            "first_gold_rank": 1,
            "gold_ranks": {"Book X::0": [1, 3.0], "Author Y::1": [2, 2.0]},
            "full_prefix_same": True, "n_gold": 2, "n_gold_para": 2}
        assert second["hop_outcome"] == {
            "hop_precision": 1.0, "hop_recall": 0.5, "hop_para_recall": 0.5,
            "cumulative_recall": 1.0, "all_gold_in_topk": False,
            "gold_hit_ids": ["Author Y::1"], "new_gold": ["Author Y::1"], "aligned_hit": True,
            "first_gold_rank": 1,
            "gold_ranks": {"Book X::0": [2, 2.0], "Author Y::1": [1, 4.0]},
            "full_prefix_same": True, "n_gold": 2, "n_gold_para": 2}
        assert second["features"]["n_new_chunks"] == 1
        assert records[2]["hop_outcome"]["aligned_hit"] is None     # q2 has no ordered path

        for r in records:
            assert set(r["outcome"]) == set(OUTCOME_FIELDS) | {"failure_type"}
            assert r["outcome"]["em"] == 1.0 and r["outcome"]["retrieval_correct"] == 1
            assert r["outcome"]["hops"] == 2 and r["outcome"]["stop_reason"] == "done"
            assert r["outcome"]["failure_type"] == ""
            assert r["shadow"] == {} and r["probe_features"] == {} and r["first_stage"] is None
            assert r["deeper"] is None and r["deeper_features"] is None   # log_deeper is 0
            # everything that comes from a label sits in these three blocks, nowhere else
            assert set(r) - {"hop_outcome", "shadow", "outcome"} == {
                "run", "dataset", "id", "qtype", "hop", "query", "strategy", "retrieved",
                "first_stage", "feature_schema_version", "features", "probe_features",
                "deeper", "deeper_features"}


def test_logger_does_not_mutate():
    from src.eval.run_qa import score_row
    from src.memory.experience import ExperienceLogger
    from src.types import HopTrace, PipelineResult
    with tempfile.TemporaryDirectory() as tmp:
        log = ExperienceLogger(stub_cfg("single_hop", tmp), "stub")
        ex = example()
        hits = hits_of(("Book X::0", 3.0), ("Author Y::1", 2.0))
        shadow = {"other": hits_of(("Distractor::0", 1.0))}
        res = PipelineResult(example_id=ex.id, answer="Some University", hops=[
            HopTrace(hop=0, query=ex.question, retrieved=hits,
                     gold_chunk_ids=list(ex.gold_chunk_ids))])
        row = score_row(ex, res)
        before = copy.deepcopy((ex, hits, shadow, res, row))
        log.on_hop(ex, 0, ex.question, hits, shadow)
        log.on_outcome(ex, res, row)
        log.close()
        assert (ex, hits, shadow, res, row) == before


def test_memory_on_off_identical():
    """Same traces, answers, scored rows and prompts with the logger attached."""
    for mode in ("single_hop", "multihop_ircot"):
        with tempfile.TemporaryDirectory() as tmp:
            off = stub_pipeline(mode)
            on = stub_pipeline(mode, store=tmp, shadow={"rev": OverlapRetriever(reverse=True)})
            assert off.memory is None and on.memory is not None
            for ex in (example(), example(id="q2", question="Where is Some University?")):
                a, b = off.run_example(ex), on.run_example(ex)
                off.generator.calls = on.generator.calls = 0
                assert comparable(ex, a) == comparable(ex, b)
                assert a.answer == b.answer and a.stop_reason == b.stop_reason
                assert a.raw_outputs == b.raw_outputs
                assert [[(rc.chunk, rc.score, rc.rank) for rc in h.retrieved] for h in a.hops] \
                    == [[(rc.chunk, rc.score, rc.rank) for rc in h.retrieved] for h in b.hops]
            assert off.generator.prompts == on.generator.prompts
            # the logging path asks the primary retriever once more per hop, for
            # its ranking of the whole pool; the top-k calls are the same ones
            assert set(off.retriever.depths) == {3} and on.retriever.depths[::2] == \
                off.retriever.depths and set(on.retriever.depths[1::2]) == {len(POOL)}
            assert on.retriever.queries[::2] == off.retriever.queries
            assert len(on.memory.records()) == (2 if mode == "single_hop" else 4)


def test_reranked_run_compares_at_matched_depth():
    """top_k 5 into a reranker that keeps 2: the shadow arm is the same first
    stage, so it must not look like a different arm or a better one by depth."""
    with tempfile.TemporaryDirectory() as tmp:
        off = stub_pipeline("single_hop", top_k=5, rerank_top_n=2)
        on = stub_pipeline("single_hop", store=tmp, top_k=5, rerank_top_n=2,
                           shadow={"same": OverlapRetriever(), "rev": OverlapRetriever(True)})
        ex = example()
        a, b = off.run_example(ex), on.run_example(ex)
        assert comparable(ex, a) == comparable(ex, b)
        first_stage = ["Book X::0", "Author Y::1", "Author Y::0",
                       "Some University (Oslo)::0", "Book X::1"]
        rec = on.memory.records()[0]
        assert rec.strategy["rerank_enabled"] is True and rec.strategy["rerank_top_n"] == 2
        assert [h["chunk_id"] for h in rec.retrieved] == ["Book X::1", "Some University (Oslo)::0"]
        assert [h["chunk_id"] for h in rec.first_stage] == first_stage
        assert [h["rank"] for h in rec.first_stage] == [1, 2, 3, 4, 5]
        assert rec.features["n_hits"] == 2

        # retriever against retriever, before the reranker
        assert rec.probe_features["same"] == {"topk_jaccard": 1.0, "top1_same": 1, "depth": 5}
        assert rec.probe_features["rev"]["top1_same"] == 0

        # outcomes at the context size the primary handed on, the full list kept beside
        same = rec.shadow["same"]
        assert same["retrieved"] == first_stage and same["depth"] == 2
        assert same["gold_hit_ids"] == ["Book X::0", "Author Y::1"]
        assert same["all_gold_in_topk"] is True and same["hop_recall"] == 1.0
        assert rec.shadow["rev"]["depth"] == 2 and rec.shadow["rev"]["hop_recall"] == 0.0

        # the first stage had both gold chunks on top; the reranker dropped them
        assert rec.hop_outcome["hop_recall"] == 0.0 and rec.hop_outcome["first_gold_rank"] is None
        assert rec.hop_outcome["gold_ranks"] == {"Book X::0": [1, 3.0], "Author Y::1": [2, 2.0]}
        assert rec.hop_outcome["full_prefix_same"] is True


def test_shadow_arm_scored_like_primary():
    """Which gold an arm found is on the record, so a shadow arm that completes
    the chain is told apart from one that repeats what was already there."""
    from src.memory.experience import ExperienceLogger
    with tempfile.TemporaryDirectory() as tmp:
        log = ExperienceLogger(stub_cfg("multihop_ircot", tmp), "stub")
        ex = example()
        log.on_hop(ex, 0, ex.question, hits_of(("Book X::0", 3.0)))
        again = hits_of(("Book X::0", 2.0))
        other = hits_of(("Author Y::1", 0.8))
        full = hits_of(("Author Y::1", 0.8), ("Book X::0", 0.5), ("Distractor::0", 0.1))
        log.on_hop(ex, 1, "Author Y", again, {"other": other}, shadow_full={"other": full})
        rec = log.records()[1]
        assert rec.hop_outcome["hop_recall"] == 0.5 and rec.shadow["other"]["hop_recall"] == 0.5
        assert rec.hop_outcome["new_gold"] == [] and rec.hop_outcome["cumulative_recall"] == 0.5
        assert rec.hop_outcome["aligned_hit"] is False
        assert rec.shadow["other"]["new_gold"] == ["Author Y::1"]
        assert rec.shadow["other"]["gold_hit_ids"] == ["Author Y::1"]
        assert rec.shadow["other"]["cumulative_recall"] == 1.0
        assert rec.shadow["other"]["aligned_hit"] is True
        assert rec.shadow["other"]["gold_ranks"] == {"Book X::0": [2, 0.5], "Author Y::1": [1, 0.8]}
        assert rec.hop_outcome["n_gold"] == 2 and rec.hop_outcome["n_gold_para"] == 2
        assert "gold_ranks" not in rec.hop_outcome       # no full ranking was handed in
        # the shadow block and the primary's are the same measurements
        assert set(rec.hop_outcome) - {"n_gold", "n_gold_para"} == \
            set(rec.shadow["other"]) - {"retrieved", "depth", "gold_ranks", "full_prefix_same"}

        # a full-pool list that orders a tie differently from the top-k call is flagged
        log.on_hop(ex, 2, "Book X", hits_of(("Book X::0", 1.0), ("Book X::1", 1.0)),
                   primary_full=hits_of(("Book X::1", 1.0), ("Book X::0", 1.0)))
        assert log.records()[2].hop_outcome["full_prefix_same"] is False
        assert log.records()[2].hop_outcome["gold_ranks"] == {"Book X::0": [2, 1.0],
                                                             "Author Y::1": None}


def test_latency_excludes_logging():
    """A slow shadow arm does not show up in the question's latency."""
    with tempfile.TemporaryDirectory() as tmp:
        p = stub_pipeline("single_hop", store=tmp, shadow={"slow": SlowRetriever()},
                          generate=False)
        t0 = time.perf_counter()
        res = p.run_example(example())
        wall = time.perf_counter() - t0
        assert wall >= 0.09 and p._memory_s >= 0.09      # index, top-k and full-pool calls
        assert res.latency_s < 0.03 and abs(wall - p._memory_s - res.latency_s) < 0.01
        off = stub_pipeline("single_hop", generate=False)
        off.run_example(example())
        assert off._memory_s == 0.0


def test_shadow_never_changes_primary():
    from src.eval.run_qa import score_row
    with tempfile.TemporaryDirectory() as tmp:
        plain = stub_pipeline("single_hop", store=tmp, generate=False)
        rev = OverlapRetriever(reverse=True)
        shadowed = stub_pipeline("single_hop", store=tmp, shadow={"rev": rev}, generate=False)
        ex = example()
        a, b = plain.run_example(ex), shadowed.run_example(ex)
        ids = [rc.chunk.chunk_id for rc in a.hops[0].retrieved]
        assert ids == [rc.chunk.chunk_id for rc in b.hops[0].retrieved]
        assert [rc.score for rc in a.hops[0].retrieved] == \
               [rc.score for rc in b.hops[0].retrieved]

        # the shadow arm did run, on the same pool and query (at top_k, then for
        # its ranking of the whole pool), and only reached the log
        assert rev.chunks == ex.chunks and rev.queries == [ex.question] * 2
        assert rev.depths == [3, len(POOL)]
        rec = shadowed.memory.records()[0]
        shadow_ids = rec.shadow["rev"]["retrieved"]
        assert shadow_ids != ids and len(shadow_ids) == 3 and rec.shadow["rev"]["depth"] == 3
        assert rec.shadow["rev"]["all_gold_in_topk"] is False
        assert rec.shadow["rev"]["hop_recall"] == 0.0
        # both gold chunks are at the bottom of the reversed ranking
        assert rec.shadow["rev"]["gold_ranks"] == {"Book X::0": [6, 3.0], "Author Y::1": [5, 2.0]}
        assert [r["chunk_id"] for r in rec.retrieved] == ids
        assert set(rec.probe_features) == {"rev"} and list(rec.features) == list(FEATURES)
        assert plain.memory.records()[0].shadow == {}

        # nothing was generated, so there is no answer to score
        for p, res in ((plain, a), (shadowed, b)):
            p.memory.on_outcome(ex, res, score_row(ex, res))
            outcome = p.memory.records()[0].outcome
            assert outcome["em"] is None and outcome["f1"] is None
            assert outcome["retrieval_correct"] == 1 and outcome["gold_recall"] == 1.0


def test_backfill_replays_logger():
    """Replaying a multi-hop trace gives the records the live logger wrote."""
    from dataclasses import asdict

    from src.eval.run_qa import score_row, trace_record
    from src.memory.backfill import replay
    with tempfile.TemporaryDirectory() as tmp:
        live = stub_pipeline("multihop_ircot", store=tmp,
                             shadow={"rev": OverlapRetriever(reverse=True)})
        examples = [example(), example(id="q2", question="Where is Some University?")]
        traces, rows = [], {}
        for ex in examples:
            res = live.run_example(ex)
            live.generator.calls = 0
            row = score_row(ex, res)
            live.memory.on_outcome(ex, res, row)
            traces.append(json.loads(json.dumps(trace_record(ex, res))))
            rows[ex.id] = {k: str(v) for k, v in row.items()}    # as csv.DictReader reads it

        again = stub_pipeline("multihop_ircot", store=tmp, generate=False,
                              shadow={"rev": OverlapRetriever(reverse=True)})
        n_hops, mismatches, csv_disagree = replay(again, examples, traces, rows)
        assert n_hops == 4 and mismatches == [] and csv_disagree == 0
        assert [asdict(r) for r in again.memory.records()] == \
               [asdict(r) for r in live.memory.records()]

        # a trace that the retriever no longer reproduces is reported, hop by
        # hop, and that question gets no records
        traces[0]["hops"][1]["retrieved"].reverse()
        broken = stub_pipeline("multihop_ircot", store=tmp, generate=False)
        _, mismatches, _ = replay(broken, examples, traces, rows)
        assert [(m["id"], m["hop"]) for m in mismatches] == [("q1", 1)]
        assert [(r.id, r.hop) for r in broken.memory.records()] == [("q2", 0), ("q2", 1)]


def _tiny_run(tmp):
    """Five synthetic questions over one shared pool, a bm25 config with
    logging and a bm25 shadow arm, all inside tmp."""
    questions = [("t1", "Who wrote Book X?", "Author Y", ["Book X::0"]),
                 ("t2", "Where did Author Y study?", "Some University", ["Author Y::1"]),
                 ("t3", QUESTION, "Some University", ["Book X::0", "Author Y::1"]),
                 ("t4", "Is Some University in Oslo?", "yes", ["Some University (Oslo)::0"]),
                 ("t5", "Where is the writer Author Y from?", "Norway", ["Author Y::0"])]
    chunks = [{"chunk_id": c.chunk_id, "text": c.text, "title": c.title,
               "sent_idx": c.sent_idx} for c in POOL]
    data_file = Path(tmp) / "tiny_dev.jsonl"
    with open(data_file, "w") as f:
        for qid, q, a, gold in questions:
            f.write(json.dumps({"id": qid, "question": q, "answer": a, "hops": len(gold),
                                "qtype": "bridge", "chunks": chunks,
                                "gold_chunk_ids": gold}) + "\n")
    config = Path(tmp) / "tiny.yaml"
    config.write_text(
        "run: {name: tiny_live}\n"
        "dataset: {name: tiny}\n"
        "retrieval: {method: bm25, top_k: 3}\n"
        f"paths: {{data_dir: {json.dumps(tmp)}, results_dir: {json.dumps(tmp + '/results')}}}\n"
        f"memory: {{enabled: true, log_arms: [bm25], store: {json.dumps(tmp + '/memory')}}}\n")
    return config, data_file


def test_backfill_cli_matches_live_run():
    """run_qa with logging on, then the backfill CLI on its trace and csv."""
    from src.eval import run_qa
    from src.memory import backfill
    with tempfile.TemporaryDirectory() as tmp:
        config, data_file = _tiny_run(tmp)
        argv = sys.argv
        sys.argv = ["run_qa", "--config", str(config)]
        try:
            run_qa.main()
        finally:
            sys.argv = argv
        live_header, live = read_log(Path(tmp) / "memory" / "tiny_live_experience.jsonl")
        assert live_header["source"] == "live" and len(live) == 5
        assert live_header["config"]["memory"]["log_arms"] == ["bm25"]
        assert live_header["shadow_arms"] == {"bm25": {
            "method": "bm25", "top_k": 3, "hybrid_alpha": None, "rerank_enabled": False,
            "rerank_top_n": None, "pipeline_mode": "single_hop"}}
        assert live_header["data"] == {
            "file": str(data_file), "bytes": data_file.stat().st_size,
            "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
            "ids_file": None, "n": None, "n_questions": 5}
        for r in live:       # the bm25 shadow of a bm25 run is the run itself
            assert r["shadow"]["bm25"]["retrieved"] == [h["chunk_id"] for h in r["retrieved"]]
            assert r["probe_features"]["bm25"] == {"topk_jaccard": 1.0, "top1_same": 1, "depth": 3}
            assert r["shadow"]["bm25"]["gold_hit_ids"] == r["hop_outcome"]["gold_hit_ids"]
            assert r["shadow"]["bm25"]["gold_ranks"] == r["hop_outcome"]["gold_ranks"]
            assert set(r["hop_outcome"]["gold_ranks"]) == set(r["hop_outcome"]["gold_hit_ids"]) \
                | {g for g, at in r["hop_outcome"]["gold_ranks"].items() if at[0] > 3}
            assert r["outcome"]["hops"] == 1 and r["outcome"]["stop_reason"] == ""
            assert r["outcome"]["em"] is None and r["outcome"]["f1"] is None    # backend none

        args = ["--run", "tiny_live", "--config", str(config), "--data-file", str(data_file)]
        out = Path(tmp) / "backfill.jsonl"
        assert backfill.main(args + ["--out", str(out)]) == 0
        header, records = read_log(out)
        assert header["source"] == "backfill" and header["mismatching_hops"] == 0
        assert header["questions_dropped"] == 0
        assert header["outcome_source"] == str(Path(tmp) / "results" / "tiny_live_qa.csv")
        assert header["data"] == live_header["data"]
        assert records == live

        assert backfill.main(args + ["--out", str(out), "--n", "2"]) == 0
        assert read_log(out)[1] == live[:2]

        # tamper with the trace: the replay no longer matches and the CLI refuses
        trace = Path(tmp) / "results" / "tiny_live_trace.jsonl"
        lines = trace.read_text().splitlines()
        first = json.loads(lines[0])
        first["hops"][0]["retrieved"].reverse()
        trace.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
        out2 = Path(tmp) / "backfill2.jsonl"
        try:
            backfill.main(args + ["--out", str(out2)])
            raise AssertionError("a mismatching trace should exit non-zero")
        except SystemExit as e:
            assert e.code == 1
        assert not out2.exists()
        assert backfill.main(args + ["--out", str(out2), "--allow-mismatch"]) == 1
        header, records = read_log(out2)
        assert header["mismatching_hops"] == 1 and header["questions_dropped"] == 1
        assert records == live[1:]          # only the questions that reproduce


def test_failing_log_keeps_run_artefacts():
    """The experience log is written last: if it cannot be, the trace, the csv
    and the per-hop file of the run are already there."""
    from src.eval import run_qa
    with tempfile.TemporaryDirectory() as tmp:
        config, _ = _tiny_run(tmp)
        blocker = Path(tmp) / "not_a_dir"
        blocker.write_text("")
        config.write_text(config.read_text().replace(tmp + "/memory", str(blocker / "memory")))
        argv = sys.argv
        sys.argv = ["run_qa", "--config", str(config)]
        try:
            run_qa.main()
            raise AssertionError("a store under a regular file cannot be written")
        except OSError:
            pass
        finally:
            sys.argv = argv
        results = Path(tmp) / "results"
        for name in ("tiny_live_trace.jsonl", "tiny_live_qa.csv", "tiny_live_2x2.png",
                     "tiny_live_per_hop.json"):
            assert (results / name).stat().st_size > 0, name


def test_close_is_all_or_nothing():
    """A value json cannot write goes in as text; a failed write leaves the
    earlier log as it was and no temporary file."""
    import datetime

    from src.memory.experience import ExperienceLogger
    with tempfile.TemporaryDirectory() as tmp:
        cfg = stub_cfg("single_hop", tmp)
        cfg.run.started = datetime.date(2026, 9, 30)
        log = ExperienceLogger(cfg, "stub")
        ex = example()
        log.on_hop(ex, 0, ex.question, hits_of(("Book X::0", 3.0)))
        path = log.close()
        header, records = read_log(path)
        assert header["config"]["run"]["started"] == "2026-09-30" and len(records) == 1
        before = path.read_bytes()

        log.records()[0].outcome = {"bad": object()}
        try:
            log.close()
            raise AssertionError("an unserialisable record should fail")
        except TypeError:
            pass
        assert path.read_bytes() == before
        assert sorted(q.name for q in Path(tmp).iterdir()) == [path.name]


def test_config_memory_block():
    """Off by default in every config but the memlog one, which is bm25.yaml
    with a different run name and the switch on."""
    from src.config import as_dict, load_config
    from src.pipeline.multihop import MultiHopPipeline
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = load_config(path)
        arms = path.name.startswith("bm25_arms_")
        assert cfg.memory.enabled is (path.name == "bm25_memlog.yaml" or arms), path.name
        assert cfg.memory.mode == "log"
        assert cfg.memory.log_arms == (["dense", "hybrid"] if arms else []), path.name
        assert cfg.memory.store == "results/memory" and cfg.memory.seed == 13
        assert cfg.memory.log_deeper == 0, path.name      # the deeper window is opt-in
        if arms:        # retrieval only, bm25 first stage, nothing generated
            assert cfg.pipeline.mode == "single_hop" and cfg.retrieval.method == "bm25"
            assert cfg.generation.backend == "none" and not cfg.rerank.enabled

    plain, logged = as_dict(load_config(ROOT / "configs" / "bm25.yaml")), \
        as_dict(load_config(ROOT / "configs" / "bm25_memlog.yaml"))
    assert logged["run"]["name"] == "bm25_hotpot_dev_memlog"
    logged["run"]["name"] = plain["run"]["name"]
    logged["memory"]["enabled"] = False
    assert logged == plain

    cfg = load_config(ROOT / "configs" / "bm25.yaml")
    p = MultiHopPipeline(cfg)
    assert p.memory is None and p.shadow is None
    cfg.memory.mode = "none"
    cfg.memory.enabled = True
    assert MultiHopPipeline(cfg).memory is None                 # enabled but not logging
    cfg = load_config(ROOT / "configs" / "bm25_memlog.yaml")
    cfg.memory.log_arms = ["bm25"]
    p = MultiHopPipeline(cfg)
    assert p.memory is not None and p.memory.run == "bm25_hotpot_dev_memlog"
    assert list(p.shadow) == ["bm25"] and p.shadow["bm25"] is not p.retriever
    assert str(p.memory.path) == "results/memory/bm25_hotpot_dev_memlog_experience.jsonl"


def arms_log(tmp, outcomes):
    """A hop-0 log with a primary arm and two shadow arms whose hits are set
    by hand. outcomes: per question, {arm: hit (True) or miss (False)} for
    "primary", "other" and "third"; a hit retrieves both gold chunks, a miss
    retrieves none. Features vary with the question index."""
    from src.memory.experience import ExperienceLogger
    log = ExperienceLogger(stub_cfg("single_hop", tmp, top_k=2), "arms")
    hit = hits_of(("Book X::0", 3.0), ("Author Y::1", 2.0))
    miss = hits_of(("Distractor::0", 1.0), ("Book X::1", 0.5))
    for n, want in enumerate(outcomes):
        ex = example(id=f"q{n}", qtype="bridge" if n % 2 else "comparison",
                     question=QUESTION + " x" * n)      # query length grows with n
        log.on_hop(ex, 0, ex.question, hit if want["primary"] else miss,
                   {"other": hit if want["other"] else miss,
                    "third": hit if want["third"] else miss})
        log.on_outcome(ex, SimpleNamespace(hops=[None], raw_outputs=[], stop_reason=""),
                       {"em": None, "f1": None, "retrieval_correct": int(want["primary"]),
                        "gold_recall": None, "para_recall": None, "stop_reason": "",
                        "hops": 1})
    return log.close()


def test_arm_table_and_ceiling():
    from src.memory.arms import (arm_table, arms_of, ceiling, discordance, hits, read_records,
                                 stage_columns)
    outcomes = [dict(primary=True, other=True, third=False),
                dict(primary=True, other=False, third=False),
                dict(primary=False, other=True, third=False),
                dict(primary=False, other=True, third=True),
                dict(primary=False, other=False, third=False),
                dict(primary=False, other=False, third=True)]
    with tempfile.TemporaryDirectory() as tmp:
        header, records = read_records(arms_log(tmp, outcomes))
        assert header["run"] == "arms" and len(records) == 6
        table = arm_table(records)
        assert arms_of(table) == ["overlap", "other", "third"]     # primary first
        assert table["arm:overlap"].tolist() == [1, 1, 0, 0, 0, 0]
        assert table["arm:other"].tolist() == [1, 0, 1, 1, 0, 0]
        assert table["arm:third"].tolist() == [0, 0, 0, 1, 0, 1]
        assert table["f:query_len_tokens"].tolist() == [9, 10, 11, 12, 13, 14]
        assert table["question"].tolist()[1] == QUESTION + " x"
        assert table["probe:other:topk_jaccard"].tolist() == [1, 0, 0, 0, 1, 1]
        assert "f:hop" in stage_columns(table, "size") and len(stage_columns(table, "size")) == 3
        assert stage_columns(table, "query") == ["f:query_len_tokens", "f:query_is_question",
                                                 "f:hop", "f:query_cap_spans"]
        assert len(stage_columns(table, "pool")) == 9 and len(stage_columns(table, "history")) == 10
        assert len(stage_columns(table, "primary_hits")) == len(FEATURES)
        assert len(stage_columns(table, "probe")) == len(FEATURES) + 2 * len(PROBE_FEATURES)
        # ids select and order the rows; a secondary outcome works the same way
        sub = arm_table(records, ids=["q3", "q0"])
        assert sub["id"].tolist() == ["q3", "q0"]
        para = arm_table(records, outcome="hop_para_recall")      # a miss still has Book X::1
        assert para["arm:third"].tolist() == [0.5, 0.5, 0.5, 1, 0.5, 1]
        assert hits(para, "third").tolist() == [0, 0, 0, 1, 0, 1]   # graded value, binary hit
        assert discordance(para).set_index(["arm_a", "arm_b"]).loc[("overlap", "third")]["only_b"] == 2
        assert ceiling(para).set_index("arm").loc["all_arms"]["rate"] == 0.5

        d = discordance(table).set_index(["arm_a", "arm_b"])
        row = d.loc[("overlap", "other")]
        assert (row["both"], row["only_a"], row["only_b"], row["neither"]) == (1, 1, 2, 2)
        assert row["discordant"] == 0.5 and row["mcnemar_p"] == 1.0
        assert d.loc[("overlap", "third")]["only_b"] == 2 and len(d) == 3

        c = ceiling(table).set_index("arm")
        assert c.loc["overlap"]["rate"] == round(2 / 6, 4)
        assert c.loc["other"]["rate"] == 0.5 and bool(c.loc["other"]["best_single"])
        assert c.loc["oracle"]["rate"] == round(5 / 6, 4)        # q4 is lost by every arm
        assert c.loc["all_arms"]["rate"] == 0.0
        assert c.loc["other"]["regret"] == round(2 / 6, 4)        # q1 and q5 go to other arms
        assert c.loc["oracle"]["regret"] == 0.0
        assert c.loc["other"]["regret_lo"] <= c.loc["other"]["regret"] <= c.loc["other"]["regret_hi"]


def test_predictability_is_held_out():
    import numpy as np
    import pandas as pd
    from src.memory.arms import (fold_splits, held_out_auc, predictability, seeded_halves,
                                 with_qtype_columns)
    rng = np.random.RandomState(0)
    n = 400
    table = pd.DataFrame({f"f:{name}": rng.rand(n) for name in FEATURES})
    table["id"] = [f"q{i}" for i in range(n)]
    table["qtype"] = ["bridge" if i % 3 else "comparison" for i in range(n)]
    table["probe:b:topk_jaccard"] = rng.rand(n)
    table["probe:b:top1_same"] = rng.randint(0, 2, n).astype(float)
    # arm a hits iff the query is long (a query-stage feature); arm b at random
    table["arm:a"] = (table["f:query_len_tokens"] > 0.5).astype(float)
    table["arm:b"] = rng.randint(0, 2, n).astype(float)

    for fit, score in seeded_halves(n, [13]):
        assert len(fit) + len(score) == n and not set(fit) & set(score)
    a, b = seeded_halves(n, [13])
    assert set(a[0]) == set(b[1]) and set(a[1]) == set(b[0])     # both directions
    assert list(seeded_halves(n, [13])[0][0]) == list(seeded_halves(n, [13])[0][0])

    p = predictability(table, seeds=[13, 17]).set_index(["target", "stage"])
    assert p.loc[("hit:a", "query")]["auc"] == 1.0 and p.loc[("hit:a", "size")]["auc"] < 0.7
    assert p.loc[("hit:a", "probe")]["auc"] > 0.98             # later stages keep the feature
    assert abs(p.loc[("hit:b", "primary_hits")]["auc"] - 0.5) < 0.12
    assert p.loc[("hit:a", "qtype")]["n_features"] == 2
    assert p.loc[("a>b", "query")]["n"] == int((table["arm:a"] != table["arm:b"]).sum())
    assert p.loc[("a>b", "query")]["auc"] > 0.99 and p.loc[("a>b", "query")]["n_splits"] == 4

    # a fold split fits on one fold and scores on the other, once
    ids = table["id"].tolist()
    splits = fold_splits(ids, ids[:200], ids[200:])
    assert len(splits) == 1 and len(splits[0][0]) == 200 and not set(splits[0][0]) & set(splits[0][1])
    q = predictability(table, splits=splits).set_index(["target", "stage"])
    assert q.loc[("hit:a", "query")]["n_splits"] == 1 and q.loc[("hit:a", "query")]["auc"] > 0.99
    try:
        fold_splits(ids, ids[:200], ids[100:])
        raise AssertionError("overlapping folds should be refused")
    except ValueError:
        pass

    # a router over the stage that carries the signal gets a's hits where a
    # hits and b's elsewhere; the size control routes no better than the best arm
    from src.memory.arms import routed_rate
    r = routed_rate(table, seeds=[13, 17]).set_index("stage")
    assert r.loc["query"]["gain"] > 0.2 and r.loc["query"]["routed"] > 0.7
    assert abs(r.loc["size"]["gain"]) < 0.05 and r.loc["query"]["n_splits"] == 4
    assert r.loc["query"]["oracle"] >= r.loc["query"]["routed"] >= r.loc["query"]["best_single"] - 0.05
    # paired against the best single arm on the same held-out questions: the
    # planted signal clears zero in the pooled interval and on every split;
    # a router over noise is at or under the best arm, never above it
    # (gain_hi stays near zero); every question is scored once per seed
    assert r.loc["query"]["gain_lo"] > 0 and r.loc["query"]["p_vs_single_max"] < 0.05
    assert r.loc["size"]["gain_lo"] < 0 and r.loc["size"]["gain_hi"] < 0.03
    assert r.loc["size"]["p_vs_single_median"] > 0.05
    assert r.loc["query"]["n_questions"] == len(table)
    assert r.loc["query"]["gain_lo"] <= r.loc["query"]["gain"] <= r.loc["query"]["gain_hi"]
    one = routed_rate(table, splits=[(np.arange(len(table) // 2), np.arange(len(table) // 2, len(table)))])
    one = one.set_index("stage")
    assert one.loc["query"]["gain_sd"] == 0.0 and one.loc["query"]["n_splits"] == 1   # one split, no spread
    assert one.loc["query"]["n_questions"] == len(table) - len(table) // 2         # but a paired test
    assert one.loc["query"]["gain_lo"] > 0 and one.loc["query"]["p_vs_single_max"] < 0.05

    # similarity memory: when the arm that hits is a function of the question
    # text, the nearest past questions route it; random text does not
    from src.memory.arms import neighbour_routed_rate
    table["question"] = ["long query %03d" % i if table["arm:a"][i] else "short one %03d" % i
                         for i in range(n)]
    nn = neighbour_routed_rate(table, ks=[5], seeds=[13, 17]).set_index("stage")
    assert nn.loc["question_knn5"]["gain"] > 0.15 and nn.loc["question_knn5"]["n_splits"] == 4
    assert nn.loc["question_knn5"]["gain_lo"] > 0 and nn.loc["question_knn5"]["p_vs_single_max"] < 0.05
    table["question"] = ["word%d other%d" % (i * 7 % 11, i * 3 % 13) for i in range(n)]
    nn = neighbour_routed_rate(table, ks=[5], seeds=[13, 17]).set_index("stage")
    assert abs(nn.loc["question_knn5"]["gain"]) < 0.1
    assert neighbour_routed_rate(table.iloc[:10], ks=[5], seeds=[13]).empty   # too small to fit

    # dense neighbours: the same vote over question vectors; vectors that
    # separate the two kinds of question route them, random vectors do not
    from src.memory.arms import question_similarity, similarity_memory_tables
    kind = table["arm:a"].to_numpy()
    vectors = np.stack([kind * 3 + rng.rand(n) * 0.1, (1 - kind) * 3 + rng.rand(n) * 0.1], axis=1)
    dn = neighbour_routed_rate(table, ks=[5], seeds=[13, 17], vectors=vectors,
                               label="question_dense_knn").set_index("stage")
    assert dn.loc["question_dense_knn5"]["gain"] > 0.15
    dn = neighbour_routed_rate(table, ks=[5], seeds=[13], vectors=rng.rand(n, 4),
                               label="question_dense_knn").set_index("stage")
    assert abs(dn.loc["question_dense_knn5"]["gain"]) < 0.1
    fit, score = seeded_halves(n, [13])[0]
    sims = question_similarity(table["question"].tolist(), fit, score, vectors)
    assert sims.shape == (len(score), len(fit)) and sims.max() <= 1.0 + 1e-9
    try:
        neighbour_routed_rate(table, ks=[5], seeds=[13], vectors=vectors[:-1])
        raise AssertionError("a vector per question is required")
    except ValueError:
        pass

    # the discordant-only vote: only past questions on which the arms disagree
    # may vote. With b random, the informative half still carries a's signal
    table["question"] = ["long query %03d" % i if table["arm:a"][i] else "short one %03d" % i
                         for i in range(n)]
    dv = neighbour_routed_rate(table, ks=[5], seeds=[13, 17], discordant_only=True).set_index("stage")
    assert dv.loc["question_knn5_discordant"]["gain"] > 0.15
    # when every past question is concordant there is nothing to vote with
    table["arm:b"] = table["arm:a"]
    assert neighbour_routed_rate(table, ks=[5], seeds=[13], discordant_only=True).empty
    assert not neighbour_routed_rate(table, ks=[5], seeds=[13]).empty
    table["arm:b"] = rng.randint(0, 2, n).astype(float)
    every = similarity_memory_tables(table, seeds=[13], vectors=vectors, ks=[5])
    assert every["stage"].tolist() == ["question_knn5", "question_knn5_discordant",
                                       "question_dense_knn5", "question_dense_knn5_discordant"]
    assert similarity_memory_tables(table, seeds=[13], ks=[5])["stage"].tolist() == [
        "question_knn5", "question_knn5_discordant"]

    # too small, one-class: no AUC rather than a made-up one
    X, y = table[["f:hop"]].to_numpy(), np.zeros(n)
    assert held_out_auc(X, y, seeded_halves(n, [13]))[2] == 0
    assert held_out_auc(X[:10], table["arm:b"].to_numpy()[:10], seeded_halves(10, [13]))[2] == 0
    assert "qtype:bridge" in with_qtype_columns(table).columns


def test_arms_cli():
    from src.memory.arms import main
    outcomes = [dict(primary=i % 2 == 0, other=i % 3 == 0, third=i % 5 == 0) for i in range(60)]
    with tempfile.TemporaryDirectory() as tmp:
        path = arms_log(tmp, outcomes)
        ids = Path(tmp) / "ids.txt"
        ids.write_text("\n".join(f"q{i}" for i in range(0, 60, 2)) + "\n")
        main(["--records", str(path), "--dataset", "stub", "--out-dir", tmp, "--seeds", "13"])
        for name in ("discordance", "ceiling", "predictability", "routed"):
            assert (Path(tmp) / f"stub_arms_{name}.csv").exists()
        main(["--records", str(path), "--dataset", "half", "--out-dir", tmp, "--ids", str(ids)])
        import pandas as pd
        assert pd.read_csv(Path(tmp) / "half_arms_ceiling.csv")["n"].iloc[0] == 30


def test_make_twins():
    from data.make_twins import (alias_of, alias_twin, bridge_titles, distractor_twin, make_twins,
                                 mention)
    def record(id, question, chunks, gold):
        return {"id": id, "question": question, "answer": "x", "hops": 2,
                "chunks": [{"chunk_id": f"{t}::{i}", "title": t, "sent_idx": i, "text": s}
                           for t, i, s in chunks], "gold_chunk_ids": gold}
    scott = record("q0", "Were Scott Derrickson and Ed Wood of the same nationality?",
                   [("Scott Derrickson", 0, "Scott Derrickson (born July 16, 1966) is an American director."),
                    ("Ed Wood", 0, "Edward Davis Wood Jr. (1924 - 1978) was an American filmmaker."),
                    ("Big Stone", 0, "Big Stone is a lake.")],
                   ["Scott Derrickson::0", "Ed Wood::0"])
    shellite = record("q1", "Which component of Shellite has the formula X?",
                      [("Shellite (explosive)", 0, "Shellite, also known as Tridite, is an explosive."),
                       ("Shellite (explosive)", 1, "It was used in shells."),
                       ("Picric acid", 0, "Picric acid is a component of Shellite.")],
                      ["Shellite (explosive)::0", "Picric acid::0"])
    laleli = record("q2", "Are the Laleli Mosque and Esma Sultan Mansion in the same city?",
                    [("Laleli Mosque", 0, "The Laleli Mosque is an 18th-century mosque in Istanbul."),
                     ("Esma Sultan Mansion", 0, "The Esma Sultan Mansion is a mansion in Istanbul."),
                     ("Blue Mosque", 0, "The Blue Mosque is a mosque in Istanbul with Derrickson tiles."),
                     ("Acid Lake", 0, "Acid Lake is a lake whose water is picric."),
                     ("Acid Lake", 1, "Acid rain feeds it."),
                     ("Formula One", 0, "Formula One cars use a picric component.")],
                    ["Laleli Mosque::0", "Esma Sultan Mansion::0"])

    # mentions are found without the disambiguator, case-insensitive
    assert mention(shellite["question"], "Shellite (explosive)") == (19, 8)
    assert mention(scott["question"], "Big Stone") is None
    assert bridge_titles(shellite) == ["Picric acid"] and bridge_titles(laleli) == []
    # the surname rule needs a biography; a mosque does not get one
    assert alias_of(scott, "Scott Derrickson") == ("Derrickson", "surname")
    assert alias_of(laleli, "Laleli Mosque") is None
    assert alias_of(shellite, "Shellite (explosive)") == ("Tridite", "known_as")
    twin = alias_twin(shellite)
    assert twin["id"] == "q1::twin:alias" and twin["twin_of"] == "q1"
    assert twin["question"] == "Which component of Tridite has the formula X?"
    assert twin["gold_chunk_ids"] == shellite["gold_chunk_ids"]     # gold unchanged
    assert twin["chunks"] == shellite["chunks"]                     # pool unchanged
    assert alias_twin(laleli) is None
    assert shellite["question"].startswith("Which component of Shellite")   # original untouched

    twins, counts = make_twins([scott, shellite, laleli], n_distractors=2, seed=1)
    kinds = {t["id"]: t for t in twins}
    # q0 names both gold titles: no bridge, no distractor twin; q1 has one
    assert "q0::twin:distractor" not in kinds and "q1::twin:distractor" in kinds
    d = kinds["q1::twin:distractor"]
    added = d["twin_change"]["added"]
    assert d["twin_change"]["bridge_titles"] == ["Picric acid"]
    # the added sentences share a bridge token and come from other pools, never gold
    assert added and all(a not in {c["chunk_id"] for c in shellite["chunks"]} for a in added)
    assert all("acid" in next(c["text"] for c in d["chunks"] if c["chunk_id"] == a).lower()
               or "picric" in next(c["text"] for c in d["chunks"] if c["chunk_id"] == a).lower()
               for a in added)
    assert d["question"] == shellite["question"] and len(d["chunks"]) == 3 + len(added)
    assert counts[("alias", "surname")] == 1 and counts[("alias", "known_as")] == 1
    assert counts[("distractor", "bridge_token")] == 1    # q0 and q2 name every gold title
    assert added[0] == "Acid Lake::0"                      # shares two bridge tokens, not one
    # the same seed gives the same twins
    again, _ = make_twins([scott, shellite, laleli], n_distractors=2, seed=1)
    assert [t["id"] for t in again] == [t["id"] for t in twins]
    assert again[[t["id"] for t in again].index("q1::twin:distractor")]["twin_change"]["added"] == added

    # the harder distractor: a sentence must score on the question and look
    # like the bridge paragraph. "Acid Lake is a lake whose water is picric"
    # shares bridge tokens but no question token, so it is left out; the
    # Formula One sentence shares two question tokens and one bridge token
    p = kinds["q1::twin:distractor_para"]
    assert p["twin_rule"] == "bridge_paragraph" and p["twin_of"] == "q1"
    assert p["twin_change"]["added"] == ["Formula One::0"]
    assert p["twin_change"]["shared"]["Formula One::0"] == [2, 1]
    assert p["question"] == shellite["question"] and p["gold_chunk_ids"] == shellite["gold_chunk_ids"]
    assert "q0::twin:distractor_para" not in kinds and "q2::twin:distractor_para" not in kinds
    assert counts[("distractor_para", "bridge_paragraph")] == 1
    # musique-style ids (p{idx}::{title}): the gold titles are read from the
    # second part, so the bridge is found and the record gets its twins
    def musique_record(id, question, chunks, gold):
        return {"id": id, "question": question, "answer": "x", "hops": 2,
                "chunks": [{"chunk_id": f"p{n}::{t}", "title": t, "sent_idx": -1, "text": s}
                           for n, (t, s) in enumerate(chunks)], "gold_chunk_ids": gold}
    m_shellite = musique_record("m1", shellite["question"],
                                [(c["title"], c["text"]) for c in shellite["chunks"][::2]],
                                ["p0::Shellite (explosive)", "p1::Picric acid"])
    # m2's pool carries a copy of m1's gold Picric acid paragraph under its own
    # p{idx} id, as musique pools do: it scores on m1's question and looks like
    # the bridge, and it must not be added to m1's pool as a distractor
    m_laleli = musique_record("m2", laleli["question"],
                              [(c["title"], c["text"]) for c in laleli["chunks"]]
                              + [("Picric acid", "Picric acid is a component of Shellite.")],
                              ["p0::Laleli Mosque", "p1::Esma Sultan Mansion"])
    assert bridge_titles(m_shellite) == ["Picric acid"]
    m_twins, m_counts = make_twins([m_shellite, m_laleli], n_distractors=2, seed=1)
    m_kinds = {t["id"]: t for t in m_twins}
    assert "m1::twin:distractor_para" in m_kinds and "m1::twin:distractor" in m_kinds
    assert m_kinds["m1::twin:distractor_para"]["twin_change"]["added"] == ["p5::Formula One"]
    assert "p6::Picric acid" not in m_kinds["m1::twin:distractor"]["twin_change"]["added"]
    assert m_counts[("distractor_para", "bridge_paragraph")] == 1
    # a paragraph pool has no first sentence to read an alias from
    assert alias_twin(m_shellite) is None and not any(t["twin_kind"] == "alias" for t in m_twins)
    only_alias, c2 = make_twins([scott, shellite, laleli], kinds=("alias",))
    assert {t["twin_kind"] for t in only_alias} == {"alias"} and sum(c2.values()) == 2
    try:
        make_twins([scott], kinds=("nonsense",))
        assert False, "unknown kind accepted"
    except ValueError:
        pass


def test_lookalike_pairs_and_twins():
    import numpy as np
    import pandas as pd
    from src.memory.lookalike import (best_arm, discordant, natural_pairs, pair_summary,
                                      random_pairs, twin_of, twin_table)
    n = 40
    rng = np.random.RandomState(1)
    table = pd.DataFrame({"id": [f"q{i}" for i in range(n)]})
    # two families of questions: arm a hits the long family, arm b the short one
    fam = np.array([i % 2 for i in range(n)])
    table["question"] = ["long query about %s %03d" % ("rivers" if i % 4 < 2 else "rivers", i)
                         if fam[i] else "short one on %s %03d" % ("kings", i) for i in range(n)]
    table["qtype"] = ["long" if f else "short" for f in fam]
    table["arm:a"] = fam.astype(float)
    table["arm:b"] = (1 - fam).astype(float)
    assert best_arm(np.array([[1, 1], [0, 0], [1, 0]]), ["a", "b"]) == ["all", "none", "a"]

    pairs = natural_pairs(table)
    assert len(pairs) <= n and (pairs["similarity"].diff().dropna() <= 1e-9).all()
    assert set(pairs.columns) >= {"id_a", "id_b", "similarity", "a_same", "b_same", "best_same",
                                  "discordant_a", "discordant_b", "qtype_a", "qtype_b"}
    # the nearest neighbour is from the same family, so every pair agrees
    assert pairs["a_same"].all() and pairs["best_same"].all()
    assert (pairs["qtype_a"] == pairs["qtype_b"]).all()
    # one arm hits and one misses on every question, so every pair is discordant
    assert len(discordant(pairs)) == len(pairs)
    rnd = random_pairs(table, 400, seed=3)
    assert 0.35 < rnd["a_same"].mean() < 0.65 and len(rnd) == 400
    # type-matched: the neighbour is replaced by a question of its own type,
    # which here is the same family, so the control recovers the agreement
    # that the plain random draw lacks (the type explains all of it)
    matched = random_pairs(table, 400, seed=3, match=pairs)
    assert len(matched) == 400 and matched["a_same"].all() and matched["best_same"].all()
    assert (matched["qtype_a"] == matched["qtype_b"]).all()
    summary = pair_summary(pairs, table, n_random=400)
    nearest = summary[summary["pairs"] == "nearest_all"].set_index("agreement")
    random_ = summary[summary["pairs"] == "random"].set_index("agreement")
    typed = summary[summary["pairs"] == "random_type_matched"].set_index("agreement")
    assert nearest.loc["a"]["rate"] == 1.0 and random_.loc["a"]["rate"] < 0.7
    assert typed.loc["a"]["rate"] == 1.0 and typed.loc["best"]["rate"] == 1.0
    assert set(summary["pairs"]) >= {"nearest_discordant", "random_discordant",
                                     "random_type_matched_discordant"}
    assert (summary["lo"] <= summary["rate"]).all() and (summary["rate"] <= summary["hi"]).all()
    # with arm b hitting everywhere the long family is "all", not discordant,
    # and the discordant cut keeps only the short-family pairs
    both = table.copy()
    both["arm:b"] = 1.0
    d = discordant(natural_pairs(both))
    assert len(d) and (d["qtype_a"] == "short").all() and (d["qtype_b"] == "short").all()
    # no qtype column: the type-matched draw is the plain draw
    plain = random_pairs(table.drop(columns=["qtype"]), 200, seed=5, match=pairs)
    assert len(plain) == 200 and 0.3 < plain["a_same"].mean() < 0.7
    # vectors instead of text: the same families, the same answer
    vec = np.stack([fam * 2.0 + rng.rand(n) * 0.1, (1 - fam) * 2.0], axis=1)
    assert natural_pairs(table, vec)["a_same"].all()

    assert twin_of("abc::twin:alias") == ("abc", "alias")
    try:
        twin_of("abc")
        raise AssertionError("not a twin id")
    except ValueError:
        pass
    # twins: alias twins keep every outcome, distractor twins lose arm a on
    # the long family
    twins = pd.DataFrame({"id": [f"q{i}::twin:alias" for i in range(n)] +
                                [f"q{i}::twin:distractor" for i in range(n)]})
    twins["question"] = list(table["question"]) * 2
    twins["arm:a"] = list(table["arm:a"]) + [0.0] * n
    twins["arm:b"] = list(table["arm:b"]) * 2
    t = twin_table(table, twins).set_index(["kind", "arm"])
    assert t.loc[("alias", "a")]["changed"] == 0.0 and t.loc[("alias", "any")]["best_moved"] == 0.0
    assert t.loc[("distractor", "a")]["hit_to_miss"] == n // 2
    assert t.loc[("distractor", "a")]["miss_to_hit"] == 0 and t.loc[("distractor", "a")]["mcnemar_p"] < 0.001
    assert t.loc[("distractor", "b")]["changed"] == 0.0
    assert t.loc[("distractor", "any")]["changed"] == 0.5 and t.loc[("distractor", "any")]["best_moved"] == 0.5
    assert t.loc[("alias", "a")]["n"] == n
    # twins whose original is not in the table are left out; mismatched arms refused
    sub = twin_table(table.iloc[:10], twins).set_index(["kind", "arm"])
    assert sub.loc[("alias", "a")]["n"] == 10
    try:
        twin_table(table, twins.rename(columns={"arm:b": "arm:c"}))
        raise AssertionError("arms must match")
    except ValueError:
        pass


def test_failure_typing():
    import pandas as pd
    from src.memory.failures import (audit_sample, failure_table, failure_type, named_in,
                                     recovered_by, typed_rows)

    def record(id, query, topk, hit, ranks, other_hit=False, third_hit=False, k=2):
        gold = list(ranks)
        return {"id": id, "hop": 0, "qtype": "bridge", "query": query,
                "strategy": {"method": "bm25", "top_k": k, "rerank_enabled": False},
                "retrieved": [{"chunk_id": c, "score": 1.0, "rank": r + 1} for r, c in enumerate(topk)],
                "hop_outcome": {"all_gold_in_topk": set(gold) <= set(topk), "gold_hit_ids": hit,
                                "gold_ranks": ranks, "n_gold": len(gold)},
                "shadow": {"dense": {"all_gold_in_topk": other_hit},
                           "hybrid": {"all_gold_in_topk": third_hit},
                           "bm25": {"all_gold_in_topk": True}}}      # same arm as the primary: ignored

    q = "Which university did the author of Book X attend?"
    assert named_in(q, "Book X") and named_in(q, "Book X (novel)") and not named_in(q, "Author Y")
    assert not named_in("the bookish author", "Book")                 # token match, not substring
    full = record("ok", q, ["Book X::0", "Author Y::1"], ["Book X::0", "Author Y::1"],
                  {"Book X::0": [1, 3.0], "Author Y::1": [2, 2.0]})
    assert failure_type(full) == "none"
    near = record("near", q, ["Book X::0", "D::0"], ["Book X::0"],
                  {"Book X::0": [1, 3.0], "Author Y::1": [4, 1.0]})
    assert failure_type(near) == "near_miss"                           # rank 4 <= 2k
    deep = record("deep", q, ["Book X::0", "D::0"], ["Book X::0"],
                  {"Book X::0": [1, 3.0], "Author Y::1": [5, 1.0]}, other_hit=True)
    assert failure_type(deep) == "bridge_miss"                         # unnamed paragraph missed
    named = record("named", q, ["Author Y::1", "D::0"], ["Author Y::1"],
                   {"Book X::0": [9, 0.1], "Author Y::1": [1, 3.0]})
    assert failure_type(named) == "named_miss"                         # Book X is in the query
    nothing = record("nothing", q, ["D::0", "D::1"], [],
                     {"Book X::0": [9, 0.1], "Author Y::1": [7, 0.2]})
    assert failure_type(nothing) == "named_miss"                       # a named one is missed
    nothing2 = record("nothing2", "who wrote it", ["D::0", "D::1"], [],
                      {"Book X::0": [9, 0.1], "Author Y::1": [7, 0.2]}, third_hit=True)
    assert failure_type(nothing2) == "total_miss"
    unranked = record("unranked", "who wrote it", ["D::0", "D::1"], [],
                      {"Book X::0": None, "Author Y::1": [7, 0.2]})
    assert failure_type(unranked) == "total_miss"                      # None rank is never near
    assert recovered_by(deep) == "dense" and recovered_by(nothing2) == "hybrid"
    assert recovered_by(near) == "none"
    both = record("both", q, ["D::0", "D::1"], [], {"Book X::0": [9, 0.1]}, True, True)
    assert recovered_by(both) == "dense+hybrid"

    records = [full, near, deep, named, nothing, nothing2, unranked, both,
               dict(full, id="hop1", hop=1)]
    typed = typed_rows(records)
    assert len(typed) == 8 and "hit:bm25" not in typed.columns      # hop-1 and the primary's twin dropped
    assert typed.set_index("id").loc["deep"]["failure_type"] == "bridge_miss"
    table = failure_table(typed).set_index("failure_type")
    assert table.loc["none"]["n"] == 1 and table.loc["named_miss"]["n"] == 3
    assert table.loc["bridge_miss"]["recovered_by_dense"] == 1.0
    assert table.loc["total_miss"]["recovered_by_hybrid"] == 0.5     # nothing2 yes, unranked no
    assert table.loc["all_failures"]["n"] == 7
    assert abs(table.loc["all_failures"]["recovered_by_any"] - 3 / 7) < 1e-3
    assert abs(table.loc["all_failures"]["share"] - 7 / 8) < 1e-3
    sub = typed_rows(records, ids=["deep", "near"])
    assert sub["id"].tolist() == ["near", "deep"]                     # record order, filtered

    audit = audit_sample(records, n=3, seed=1)
    assert len(audit) == 3 and (audit["failure_type"] != "none").all()
    assert set(audit.columns) >= {"query", "gold_missed_rank", "topk", "audit_ok"}
    assert audit_sample(records, n=3, seed=1)["id"].tolist() == audit["id"].tolist()
    assert len(audit_sample(records, n=50)) == 7                       # all the failures there are
    # the stub record without gold_ranks still gets a type
    stub = dict(deep, hop_outcome={"all_gold_in_topk": False, "gold_hit_ids": ["Book X::0"], "n_gold": 2})
    assert failure_type(stub) == "bridge_miss"
    # the rules tested on their own, and the type under another order: a
    # failure that is both near and named is typed by whichever comes first
    from src.memory.failures import cooccurrence, order_table, rule_flags, type_from_flags
    assert rule_flags(full) is None and type_from_flags(None) == "none"
    assert rule_flags(near) == {"near": True, "named": False, "partial": True}
    assert rule_flags(named) == {"near": False, "named": True, "partial": True}
    assert rule_flags(nothing2) == {"near": False, "named": False, "partial": False}
    nearnamed = record("nearnamed", q, ["Author Y::1", "D::0"], ["Author Y::1"],
                       {"Book X::0": [3, 0.5], "Author Y::1": [1, 3.0]})
    assert rule_flags(nearnamed) == {"near": True, "named": True, "partial": True}
    assert failure_type(nearnamed) == "near_miss"
    assert failure_type(nearnamed, order=("named", "near", "partial")) == "named_miss"
    assert failure_type(nearnamed, order=("partial", "near", "named")) == "bridge_miss"
    assert failure_type(nothing2, order=("partial", "near", "named")) == "total_miss"
    try:
        failure_type(nearnamed, order=("near", "named"))
        raise AssertionError("order must be a permutation of the three rules")
    except ValueError:
        pass
    typed2 = typed_rows(records + [nearnamed])
    assert typed2.set_index("id").loc["nearnamed"]["rule:near"] == 1
    co = cooccurrence(typed2).set_index("rules")
    assert co.loc["near"]["n"] == 2 and co.loc["near&named"]["n"] == 1
    assert co.loc["more_than_one"]["n"] == 3 and co.loc["near&named&partial"]["n"] == 1
    orders = order_table(typed2).set_index("order")
    assert len(orders) == 6
    assert orders.loc["near>named>partial"]["share_near_miss"] == round(2 / 8, 4)
    assert orders.loc["named>near>partial"]["share_named_miss"] == round(4 / 8, 4)
    assert orders.loc["partial>near>named"]["share_bridge_miss"] == round(4 / 8, 4)
    assert (orders["share_total_miss"] == round(2 / 8, 4)).all()        # never moves
    assert orders.loc["near>named>partial"]["bridge_miss_by_dense"] == 1.0
    # musique ids carry the title after the paragraph index, so the named
    # rule reads it from there; hotpot and 2wiki ids keep the title first
    from src.memory.failures import paragraph_of
    assert paragraph_of("Book X::0") == "Book X" and paragraph_of("Book X (novel)::12") == "Book X (novel)"
    assert paragraph_of("p7::Book X") == "Book X" and paragraph_of("p12::Author Y") == "Author Y"
    assert paragraph_of("p7::Book X::0") == "Book X::0"    # a musique title with a colon pair is kept
    assert paragraph_of("p7") == "p7"
    musique = record("musique", q, ["p1::Author Y", "p2::D"], ["p1::Author Y"],
                     {"p0::Book X": [9, 0.1], "p1::Author Y": [1, 3.0]}, k=2)
    assert failure_type(musique) == "named_miss"                        # was bridge_miss before 5 Oct


def stream_table(n=400, seed=7):
    """An arm table (src.memory.arms.arm_table layout) with a boundary to
    learn: the primary arm misses when score_entropy is high; in that region
    "other" recovers the question when rare_token_ratio is low and "third"
    when it is high. Everything else is noise, so a lesson for "other" has a
    true boundary (entropy > 0.5 and rare < 0.5) and a lesson for "third"
    the complementary half. Every other feature is a constant or noise."""
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(seed)
    entropy = rng.rand(n)
    rare = rng.rand(n)
    primary = (entropy <= 0.5).astype(float)
    other = ((entropy > 0.5) & (rare < 0.5) | (primary == 1) & (rng.rand(n) < 0.5)).astype(float)
    third = ((entropy > 0.5) & (rare >= 0.5) | (primary == 1) & (rng.rand(n) < 0.5)).astype(float)
    table = pd.DataFrame({"id": [f"q{i}" for i in range(n)],
                          "qtype": ["bridge" if i % 2 else "comparison" for i in range(n)],
                          "question": [f"question {i}" for i in range(n)], "n_gold": 2,
                          "arm:primary": primary, "arm:other": other, "arm:third": third})
    for name in FEATURES:
        table[f"f:{name}"] = 0.0
    table["f:score_entropy"] = entropy
    table["f:rare_token_ratio"] = rare
    table["f:top1_score"] = rng.rand(n)          # noise
    table["f:n_hits"] = 2.0
    return table


def test_stream_protocol_and_baselines():
    import numpy as np
    from src.memory.stream import (AlwaysProbe, Decision, FixedArm, LessonMemory, OnlineKNN,
                                   OnlineRouter, Policy, StaticRule, boundary_metrics,
                                   evaluate, make_stream, run_stream, seeded_orders)
    table = stream_table()
    orders = seeded_orders(len(table), [13, 17])
    assert not np.array_equal(orders[0], orders[1]) and sorted(orders[0]) == list(range(len(table)))
    s = make_stream(table, "primary_hits", orders[0])
    assert s.arms == ["primary", "other", "third"] and s.X.shape == (400, len(FEATURES) + 2)
    assert "f:score_entropy" in s.columns and "qtype:bridge" in s.columns
    q = make_stream(table, "query", orders[0])
    assert "f:score_entropy" not in q.columns          # the query stage cannot see the hits

    # the fixed arms and the probe reproduce the ceiling table
    fixed = run_stream(s, FixedArm(s, "primary", "primary"))
    assert fixed.hit.mean() == table["arm:primary"].mean() and set(fixed.cost) == {1}
    probe = run_stream(s, AlwaysProbe(s, "primary"))
    assert probe.hit.mean() == table[["arm:primary", "arm:other", "arm:third"]].max(axis=1).mean()
    assert set(probe.cost) == {3} and set(probe.arm) == {"probe"}

    # decide-then-reveal: a policy never sees the current outcome before deciding
    class Peeker(Policy):
        stage = "primary_hits"
        def __init__(self, stream, default):
            super().__init__(stream, default)
            self.revealed = []
        def decide(self, x, t):
            assert len(self.revealed) == t          # exactly the earlier questions
            return Decision(self.default, cost=1)
        def reveal(self, x, h, decision, t):
            self.revealed.append(h)
    run_stream(s, Peeker(s, "primary"))

    # switching after the primary ran costs two arms; the primary itself one
    rules = [("high_entropy", lambda f: f["f:score_entropy"] > 0.5, "other")]
    rule = run_stream(s, StaticRule(s, "primary", "primary_hits", rules))
    fired = rule.fired
    assert fired.sum() > 100 and set(rule.cost[fired]) == {2} and set(rule.cost[~fired]) == {1}
    assert rule.hit.mean() > fixed.hit.mean()                 # the rule is half right
    veto = run_stream(s, StaticRule(s, "primary", "primary_hits", rules, veto=True, min_n=10))
    assert veto.fired.sum() < fired.sum()                     # the first min_n never fire
    m = boundary_metrics(s, rule, "primary")
    assert m["n_recoverable"] == int(((table["arm:primary"] == 0) &
                                      ((table["arm:other"] == 1) | (table["arm:third"] == 1))).sum())
    assert 0.3 < m["false_application"] < 0.7                 # "other" is right on half the region
    assert m["false_rejection"] == 0.0                        # every recoverable question fires
    assert abs(m["boundary_precision"] + m["false_application"] - 1) < 1e-9
    assert m["harm"] == 0.0                                   # primary never hits where it fires

    # the router and the knn learn the region at the primary_hits stage and
    # nothing at the query stage, where the features are constants
    router = run_stream(s, OnlineRouter(s, "primary", "primary_hits", warmup=50))
    assert router.hit[200:].mean() > 0.85
    knn = run_stream(s, OnlineKNN(s, "primary", "primary_hits", k=10, warmup=50))
    assert knn.hit[200:].mean() > 0.85
    blind = run_stream(q, OnlineRouter(q, "primary", "query", warmup=50))
    assert abs(blind.hit.mean() - fixed.hit.mean()) < 0.1

    # lessons: created only from recoverable failures, after the decision
    util = LessonMemory(s, "primary", "primary_hits", revise=False, radius=2.0, warmup=20)
    res_u = run_stream(s, util)
    assert util.lessons and all(e["op"] == "create" for e in util.log)
    assert all(l.arm in ("other", "third") for l in util.lessons)
    assert all(not l.retired for l in util.lessons)
    bound = LessonMemory(s, "primary", "primary_hits", revise=True, radius=2.0, warmup=20)
    res_b = run_stream(s, bound)
    ops = {e["op"] for e in bound.log}
    assert {"create", "narrow"} <= ops, ops
    assert all(set(e) >= {"t", "lesson", "op", "reason", "width_before", "width_after"}
               for e in bound.log)
    # a narrow step never widens the box and keeps every supporter inside
    for e in bound.log:
        if e["op"] == "narrow":
            assert e["width_after"] <= e["width_before"] + 1e-9
    for l in bound.lessons:
        assert all(np.all((p >= l.lo - 1e-9) & (p <= l.hi + 1e-9)) for p in l.support)
    assert res_b.state["op_narrow"] > 0 and res_b.state["n_lessons"] == len(bound.lessons)
    mb = boundary_metrics(s, res_b, "primary")
    assert mb["n_fired"] > 0 and mb["false_application"] <= m["false_application"]
    # the utility of a lesson never includes the question being decided:
    # on the first question inside a fresh lesson gain_n is still 0
    fresh = LessonMemory(s, "primary", "primary_hits", revise=False, radius=2.0, warmup=0, min_n=1)
    x0, h0 = s.X[s.order[0]], s.H[s.order[0]]
    fresh.reveal(x0, h0, Decision("primary"), 0)
    if fresh.lessons:
        assert fresh.lessons[0].gain_n == 0 and fresh.decide(x0, 1).lesson is None

    # the summary table: one row per policy, every rate inside [0, 1]
    summary, log = evaluate(table, "primary", seeds=[13, 17],
                            policies=[("utility", "primary_hits",
                                       lambda st, d: LessonMemory(st, d, "primary_hits", False,
                                                                  radius=2.0)),
                                      ("boundary", "primary_hits",
                                       lambda st, d: LessonMemory(st, d, "primary_hits", True,
                                                                  radius=2.0))],
                            rules={"primary_hits": rules})
    assert summary["policy"].tolist() == ["fixed:primary", "fixed:other", "fixed:third", "utility",
                                          "boundary", "rule:primary_hits",
                                          "rule:primary_hits+veto"]
    assert (summary["n_orders"] == 2).all()
    row = summary.set_index("policy")
    assert row.loc["fixed:primary", "gain"] == 0.0 and row.loc["fixed:primary", "mcnemar_p_max"] == 1.0
    assert row.loc["fixed:primary", "gain_half1"] == 0.0 and row.loc["fixed:primary", "gain_half2"] == 0.0
    # the learned policies gain more on the second half of the stream than on the first
    assert row.loc["boundary", "gain_half2"] > row.loc["boundary", "gain_half1"]
    assert row.loc["boundary", "op_create"] > 0 and row.loc["utility", "op_narrow"] == 0
    for c in ("rate", "false_application", "harm", "false_rejection", "repeated_failure",
              "boundary_precision", "boundary_recall"):
        vals = summary[c].dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), c
    assert log and log[0]["policy"] == "boundary" and log[0]["op"] == "create"


def test_twin_stream():
    import math
    import numpy as np
    import pandas as pd
    from src.memory.stream import (FixedArm, TextMemory, evaluate, make_stream, run_stream,
                                   seeded_orders, text_policies, twin_metrics, twin_pairs,
                                   with_twins)
    table = stream_table(n=200)
    rng = np.random.RandomState(3)
    # distractor twins: the same text, a harder pool: the primary flips on half
    dist = table.copy()
    dist["id"] = dist["id"] + "::twin:distractor_para"
    flip = rng.rand(len(dist)) < 0.5
    dist.loc[flip, "arm:primary"] = 1 - dist.loc[flip, "arm:primary"]
    dist["f:top1_score"] = rng.rand(len(dist))
    # alias twins: a new text, the same pool and outcomes
    alias = table.copy()
    alias["id"] = alias["id"] + "::twin:alias"
    alias["question"] = alias["question"] + " alias"
    stray = alias.iloc[:1].copy()
    stray["id"] = ["nobody::twin:alias"]                 # an original that is not in the table

    both = with_twins(table, dist)
    assert len(both) == 400 and both["id"].tolist()[:200] == table["id"].tolist()
    assert len(with_twins(table, pd.concat([alias, stray]))) == 400     # the stray twin is dropped
    pairs = twin_pairs(both)
    assert len(pairs) == 200
    assert all(both["id"][t] == both["id"][o] + "::twin:distractor_para" for t, o in pairs.items())
    try:
        with_twins(table, dist.rename(columns={"arm:third": "arm:fourth"}))
        assert False, "different arms accepted"
    except ValueError:
        pass

    order = seeded_orders(len(both), [13])[0]
    s = make_stream(both, "query", order)
    assert s.text[0] == "question 0" and s.text[200] == "question 0"
    # only the pairs whose original came first count; the fixed default's
    # hit on the twins is the default column
    fixed = run_stream(s, FixedArm(s, "primary", "primary"))
    m = twin_metrics(s, fixed, pairs, "primary")
    assert 50 < m["n_pairs"] < 150 and 0 < m["n_changed"] < m["n_pairs"]
    assert m["hit_unchanged"] == m["default_unchanged"] and m["hit_changed"] == m["default_changed"]
    assert m["same_decision_unchanged"] == 1.0 and m["same_decision_changed"] == 1.0
    # on an unchanged pair the repeated arm is the best arm
    assert m["repeat_unchanged"] == m["oracle_unchanged"]
    assert m["repeat_changed"] < m["oracle_changed"]
    # the exact-text memory carries the original's arm over to a distractor twin
    text = run_stream(s, TextMemory(s, "primary", 1.0))
    mt = twin_metrics(s, text, pairs, "primary")
    assert mt["hit_unchanged"] == mt["repeat_unchanged"] and mt["hit_changed"] == mt["repeat_changed"]
    assert mt["p_vs_repeat_changed"] == 1.0
    assert text.fired.sum() > 0 and set(text.cost) == {1}        # query stage: a switch costs 1
    # a twin met before its original is a new question to it
    first_twin = next(t for t, q in enumerate(order) if q >= 200 and
                      np.where(order == q - 200)[0][0] > t)
    assert text.arm[first_twin] == "primary" and not text.fired[first_twin]

    # alias twins: the exact key sees a new question, the Jaccard key does not
    both_a = with_twins(table, alias)
    pairs_a = twin_pairs(both_a)
    sa = make_stream(both_a, "query", seeded_orders(len(both_a), [13])[0])
    exact = twin_metrics(sa, run_stream(sa, TextMemory(sa, "primary", 1.0)), pairs_a, "primary")
    loose = twin_metrics(sa, run_stream(sa, TextMemory(sa, "primary", 0.6)), pairs_a, "primary")
    assert exact["n_changed"] == 0 and exact["hit_unchanged"] == exact["default_unchanged"]
    assert loose["hit_unchanged"] == loose["repeat_unchanged"] > exact["hit_unchanged"]
    assert math.isnan(exact["hit_changed"])

    summary, _ = evaluate(both, "primary", seeds=[13, 17], policies=text_policies(), pairs=pairs)
    assert summary["policy"].tolist() == ["fixed:primary", "fixed:other", "fixed:third",
                                          "text:exact", "text:0.6"]
    for c in ("n_pairs", "n_changed", "hit_changed", "repeat_changed", "same_decision_unchanged",
              "p_vs_repeat_changed"):
        assert c in summary.columns, c
    row = summary.set_index("policy")
    assert row.loc["text:exact", "hit_changed"] == row.loc["text:exact", "repeat_changed"]
    assert row.loc["fixed:primary", "hit_changed"] == row.loc["fixed:primary", "default_changed"]
    # without pairs the table has no twin columns
    plain, _ = evaluate(table, "primary", seeds=[13], policies=text_policies())
    assert "n_pairs" not in plain.columns


def test_stream_best_arm_pair_and_shuffled_controls():
    """The paired column against the best fixed arm, the paired test inside
    each utility / boundary pair, the shuffled-reveal control and the pooled
    twin test (docs/plan.md, Stage 5, the open items of 4 Oct)."""
    import math
    import numpy as np
    import pandas as pd
    from src.memory.stream import (FixedArm, LessonMemory, OnlineRouter, ShuffledReveal,
                                   TextMemory, best_fixed_arm, evaluate, make_stream, pair_of,
                                   pooled_twin_test, run_stream, seeded_orders, text_policies,
                                   twin_metrics, twin_pairs, twin_rows, with_twins)
    table = stream_table()
    # the best fixed arm is read off the table, ties to the default
    assert best_fixed_arm(table, "primary") in ("primary", "other", "third")
    rates = {a: table[f"arm:{a}"].mean() for a in ("primary", "other", "third")}
    assert rates[best_fixed_arm(table, "primary")] == max(rates.values())
    tied = table.copy()
    tied["arm:other"] = tied["arm:primary"]
    tied["arm:third"] = 0.0
    assert best_fixed_arm(tied, "primary") == "primary"

    # pair_of: the operators toggled, everything else the same
    assert pair_of("utility:query:r2") == "boundary:query:r2"
    assert pair_of("boundary:pool") == "utility:pool"
    assert pair_of("utility+lr:primary_hits:r2") == "boundary+lr:primary_hits:r2"
    assert pair_of("shuffled:boundary:query:r2") == "shuffled:utility:query:r2"
    assert pair_of("router:query") is None and pair_of("fixed:primary") is None

    # the shuffled control: the same decisions scored on the true outcomes,
    # the reveals a permutation of the outcome rows. On the planted table
    # the boundary memory learns the region unshuffled and nothing shuffled
    order = seeded_orders(len(table), [13])[0]
    s = make_stream(table, "primary_hits", order, seed=13)
    assert s.seed == 13
    plain = run_stream(s, LessonMemory(s, "primary", "primary_hits", revise=True, radius=2.0,
                                       warmup=20))
    ctrl = ShuffledReveal(s, "primary", LessonMemory(s, "primary", "primary_hits", revise=True,
                                                     radius=2.0, warmup=20))
    assert ctrl.name == "shuffled:boundary" and ctrl.stage == "primary_hits"
    assert sorted(ctrl.perm) == list(range(len(table)))
    shuf = run_stream(s, ctrl)
    fixed = run_stream(s, FixedArm(s, "primary", "primary"))
    assert plain.hit[200:].mean() > 0.8
    assert shuf.hit.mean() < plain.hit.mean() - 0.1
    # a policy that only learns the base rates is unaffected: the router at
    # the query stage, where the features are constants, lands in the same
    # place shuffled or not, since the marginal hit of every arm is unchanged
    q = make_stream(table, "query", order, seed=13)
    r_plain = run_stream(q, OnlineRouter(q, "primary", "query", warmup=50))
    r_shuf = run_stream(q, ShuffledReveal(q, "primary", OnlineRouter(q, "primary", "query",
                                                                      warmup=50)))
    assert abs(r_plain.hit.mean() - r_shuf.hit.mean()) < 0.05
    assert shuf.log is ctrl.inner.log and shuf.state == ctrl.inner.lesson_state()
    del fixed

    # the summary: the columns against the best fixed arm and inside each pair
    summary, _ = evaluate(table, "primary", seeds=[13, 17],
                          policies=[("utility:primary_hits:r2", "primary_hits",
                                     lambda st, d: LessonMemory(st, d, "primary_hits", False,
                                                                radius=2.0,
                                                                label="utility:primary_hits:r2")),
                                    ("boundary:primary_hits:r2", "primary_hits",
                                     lambda st, d: LessonMemory(st, d, "primary_hits", True,
                                                                radius=2.0,
                                                                label="boundary:primary_hits:r2"))],
                          shuffled=True)
    row = summary.set_index("policy")
    best = best_fixed_arm(table, "primary")
    assert (summary["best_fixed"] == best).all()
    assert row.loc[f"fixed:{best}", "gain_vs_best"] == 0.0
    assert row.loc[f"fixed:{best}", "p_vs_best_max"] == 1.0
    assert row.loc["fixed:primary", "gain_vs_best"] == -round(row.loc[f"fixed:{best}", "gain"], 4)
    assert row.loc["boundary:primary_hits:r2", "gain_vs_best"] > 0        # it learns the region
    assert row.loc["boundary:primary_hits:r2", "p_vs_best_max"] < 0.05
    # the pair test is symmetric
    assert row.loc["boundary:primary_hits:r2", "pair"] == "utility:primary_hits:r2"
    assert row.loc["utility:primary_hits:r2", "pair"] == "boundary:primary_hits:r2"
    assert row.loc["boundary:primary_hits:r2", "gain_vs_pair"] == \
        -row.loc["utility:primary_hits:r2", "gain_vs_pair"]
    assert row.loc["boundary:primary_hits:r2", "p_vs_pair_max"] == \
        row.loc["utility:primary_hits:r2", "p_vs_pair_max"]
    assert row.loc["boundary:primary_hits:r2", "only_policy_vs_pair"] == \
        row.loc["utility:primary_hits:r2", "only_pair"]
    assert row.loc["shuffled:router:query", "pair"] == "" and \
        math.isnan(row.loc["shuffled:router:query", "gain_vs_pair"])
    # the shuffled rows are there, one per control, and the shuffled boundary
    # memory is well under the unshuffled one
    shuffled = [p for p in summary["policy"] if p.startswith("shuffled:")]
    assert len(shuffled) == 12 and "shuffled:boundary:primary_hits:r2" in shuffled
    assert row.loc["shuffled:boundary:primary_hits:r2", "rate"] < \
        row.loc["boundary:primary_hits:r2", "rate"] - 0.1
    assert row.loc["shuffled:boundary:primary_hits:r2", "pair"] == "shuffled:utility:primary_hits:r2"
    # and each shuffled row is paired with the policy it shuffles
    assert abs(row.loc["shuffled:boundary:primary_hits:r2", "gain_vs_unshuffled"] -
               (row.loc["shuffled:boundary:primary_hits:r2", "rate"] -
                row.loc["boundary:primary_hits:r2", "rate"])) < 1e-3
    assert row.loc["shuffled:boundary:primary_hits:r2", "p_vs_unshuffled_max"] < 0.05
    assert math.isnan(row.loc["boundary:primary_hits:r2", "gain_vs_unshuffled"])
    assert math.isnan(row.loc["shuffled:router:query", "gain_vs_unshuffled"])   # router:query not run here
    plain_summary, _ = evaluate(table, "primary", seeds=[13], policies=[])
    assert not any(p.startswith("shuffled:") for p in plain_summary["policy"])

    # twins: repeat_best, best and the pooled test over distinct pairs
    small = stream_table(n=200)
    rng = np.random.RandomState(3)
    dist = small.copy()
    dist["id"] = dist["id"] + "::twin:distractor_para"
    flip = rng.rand(len(dist)) < 0.5
    dist.loc[flip, "arm:primary"] = 1 - dist.loc[flip, "arm:primary"]
    both = with_twins(small, dist)
    pairs = twin_pairs(both)
    best = best_fixed_arm(both, "primary")
    st = make_stream(both, "query", seeded_orders(len(both), [13])[0], seed=13)
    text = run_stream(st, TextMemory(st, "primary", 1.0))
    rows = twin_rows(st, text, pairs, "primary", best)
    assert set(rows.columns) >= {"twin", "changed", "repeat_best", "best"}
    assert rows["twin"].is_unique and (rows["twin"] >= 200).all()
    m = twin_metrics(st, text, pairs, "primary", best)
    # on an unchanged pair any arm that hit the original hits the twin, so
    # both repeat rules are the oracle there; the best column is the best
    # fixed arm's own hit
    assert m["repeat_best_unchanged"] == m["oracle_unchanged"] == m["repeat_unchanged"]
    assert m["repeat_best_changed"] <= m["oracle_changed"]
    assert m["best_unchanged"] == rows.loc[~rows["changed"], "best"].mean()
    # the pooled test: each distinct twin once, the text memory equals the
    # repeat arm exactly so its difference is zero with a zero-width interval
    pooled = pooled_twin_test(pd.concat([rows, rows], ignore_index=True))
    assert pooled["n_changed_distinct"] == int(rows["changed"].sum())
    assert pooled["n_unchanged_distinct"] + pooled["n_changed_distinct"] == len(rows)
    assert pooled["diff_vs_repeat_changed"] == 0.0
    assert pooled["diff_vs_repeat_changed_lo"] == 0.0 == pooled["diff_vs_repeat_changed_hi"]
    assert pooled["diff_vs_best_unchanged_lo"] <= pooled["diff_vs_best_unchanged"] <= \
        pooled["diff_vs_best_unchanged_hi"]
    empty = pooled_twin_test(rows.iloc[:0])
    assert empty["n_changed_distinct"] == 0 and math.isnan(empty["diff_vs_repeat_changed"])
    # over two orders the distinct count is at most the number of twins, and
    # a twin scored in both orders counts once
    summary, _ = evaluate(both, "primary", seeds=[13, 17], policies=text_policies(), pairs=pairs)
    row = summary.set_index("policy")
    assert (summary["n_changed_distinct"] <= int(dist["arm:primary"].ne(small["arm:primary"]).sum())).all()
    assert (summary["n_changed_distinct"] >= summary["n_changed"]).all()
    assert row.loc["text:exact", "diff_vs_repeat_changed"] == 0.0
    assert row.loc[f"fixed:{best}", "diff_vs_best_changed"] == 0.0
    assert "repeat_best_changed" in summary.columns and "best_changed" in summary.columns


def test_reader_context_selection():
    import pandas as pd
    from src.memory.reader_context import (answer_context, chunk_features, keep_top,
                                           select, summarize_frame)
    from src.types import Chunk

    tr = {"hops": [{"hop": 0, "retrieved": ["A::0", "B::0", "C::0"]},
                   {"hop": 1, "retrieved": ["B::0", "D::0"]}]}
    info = answer_context(tr)
    assert list(info) == ["A::0", "B::0", "C::0", "D::0"]        # first-seen order
    assert info["B::0"] == {"first_hop": 0, "first_rank": 2, "n_hops_hit": 2}
    assert info["D::0"]["first_hop"] == 1 and info["D::0"]["first_rank"] == 2
    chunks = [Chunk(c, t) for c, t in (("A::0", "alpha beta"), ("B::0", "gamma"),
                                      ("C::0", "beta delta"), ("D::0", "x"))]
    feats = chunk_features("Is A alpha?", chunks, info, [0.1, 0.9, 0.5, 0.2])
    assert [f["ce_rank"] for f in feats] == [4, 1, 2, 3]
    assert feats[0]["title_in_question"] == 1 and feats[1]["title_in_question"] == 0
    assert feats[0]["question_overlap"] > 0 and feats[3]["question_overlap"] == 0
    # gold flags never enter the features
    assert "is_gold" not in feats[0] and "gold" not in " ".join(feats[0])
    assert keep_top([0.1, 0.9, 0.5, 0.2], 2) == [1, 2]          # kept in context order
    assert keep_top([0.5, 0.5, 0.1], 1) == [0]                  # ties to the earlier chunk
    items = [{"id": "q", "chunks": chunks, "feats": feats, "is_gold": [1, 0, 0, 1]},
             {"id": "r", "chunks": chunks[:2], "feats": feats[:2], "is_gold": [0, 0]}]
    sel = select(items, {"q": [0.9, 0.1, 0.2, 0.8], "r": [0.3, 0.4]})
    assert sel["q"]["gold"] == [0, 3] and sel["q"]["learned3"] == [0, 2, 3]
    assert sel["q"]["rerank3"] == [1, 2, 3] and sel["q"]["full"] == [0, 1, 2, 3]
    assert "gold" not in sel["r"] and sel["r"]["rerank5"] == [0, 1]
    assert len(sel["q"]["random3"]) == 3 and len(sel["r"]["random3"]) == 2

    rows = []
    for qid, full_em, gold_em, allg in (("a", 0, 1, 1), ("b", 1, 1, 1), ("c", 0, 0, 0)):
        rows += [{"id": qid, "arm": "full", "em": full_em, "all_gold_ctx": allg},
                 {"id": qid, "arm": "gold", "em": gold_em, "all_gold_ctx": allg}]
    out = summarize_frame(pd.DataFrame(rows))
    a = out[(out.subset == "all") & (out.arm == "gold")].iloc[0]
    assert a.n == 3 and a.only_arm == 1 and a.only_full == 0
    g = out[(out.subset == "all_gold_in_context") & (out.arm == "gold")].iloc[0]
    assert g.n == 2 and g.em_full == 0.5 and g.em_arm == 1.0


def test_agent_episode():
    from src.agent.agent import parse_reply
    from src.eval.run_qa import score_row, trace_record
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics
    from src.types import Chunk, Example, RetrievedChunk

    assert parse_reply("KEEP: 2, 1, 9\nNEXT: SEARCH: who founded it", 3) == ([1, 0], "search", "who founded it")
    assert parse_reply("KEEP: none\nNEXT: ANSWER", 3) == ([], "answer", None)
    assert parse_reply("I think so", 3) == ([], "", None)

    pool = [Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0]) for c in
            ("A::0", "A::1", "B::0", "C::0")]

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS(); cfg.retrieval = NS()
    cfg.pipeline.mode = "agent"
    cfg.pipeline.max_hops = 3
    cfg.retrieval.top_k = 2

    class Retriever:
        def index(self, chunks):
            pass

        def retrieve(self, query, k):
            ids = ["B::0", "C::0"] if "founded" in query else ["A::0", "A::1"]
            return [RetrievedChunk(chunk=next(c for c in pool if c.chunk_id == i), score=1.0 / r,
                                   rank=r) for r, i in enumerate(ids, 1)]

    class Gen:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=4)
            if len(self.prompts) == 1:
                assert max_tokens == 64 and "[1] (A) text of A::0" in prompt
                assert "Lessons from earlier questions:\n- keep both entities" in prompt
                return "KEEP: 1\nNEXT: SEARCH who founded it", m
            if len(self.prompts) == 2:
                assert "- (A) text of A::0" in prompt.split("Notebook")[1]
                return "KEEP: 1\nNEXT: ANSWER", m
            assert "Answer the question using only the context" in prompt
            assert "text of A::0" in prompt and "text of B::0" in prompt
            assert "text of A::1" not in prompt and "text of C::0" not in prompt
            return "Bob\n", m

    class Lessons:
        def __init__(self):
            self.seen = []

        def lessons(self, step):
            self.seen.append(step)
            return ["keep both entities"] if step["hop"] == 0 else []

    ex = Example(id="q", question="who started the thing A made", answer="Bob", hops=2,
                 chunks=pool, gold_chunk_ids=["A::0", "B::0"])
    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg, p.retriever, p.reranker, p.generator = cfg, Retriever(), None, Gen()
    p.lessons = Lessons()
    res = p.run_example(ex)
    assert res.answer == "Bob" and res.stop_reason == "done"
    assert res.notebook == ["A::0", "B::0"] and len(res.hops) == 2
    assert [s["kept"] for s in res.steps] == [["A::0"], ["B::0"]]
    assert res.steps[0]["lessons"] == ["keep both entities"] and res.steps[1]["lessons"] == []
    assert p.lessons.seen[1]["notebook"] == ["A::0"] and "features" in res.steps[0]
    row = score_row(ex, res)
    assert row["em"] == 1 and row["notebook_all_gold"] == 1 and row["notebook_size"] == 2
    assert row["fallback"] == 0 and "steps" in trace_record(ex, res)

    # nothing kept: the answer is read from everything retrieved, and says so
    class Empty(Gen):
        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=4)
            if "Answer the question using only" in prompt:
                assert "text of A::1" in prompt
                return "Bob", m
            return "KEEP: none\nNEXT: ANSWER", m
    p.generator, p.lessons = Empty(), None
    res = p.run_example(ex)
    assert res.stop_reason == "done+fallback" and res.notebook == []
    assert score_row(ex, res)["fallback"] == 1

    # the other modes write the same row and trace as before
    cfg.pipeline.mode = "multihop_ircot"
    class Ircot(Gen):
        def generate(self, prompt, max_tokens=None):
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=4)
            return ("DONE" if "gathering evidence" in prompt else "Bob"), m
    p.generator = Ircot()
    res = p.run_example(ex)
    assert "notebook_size" not in score_row(ex, res) and "steps" not in trace_record(ex, res)


def test_orchestrator():
    from src.agent.orchestrator import fill_refs, parse_plan
    from src.eval.run_qa import score_row, trace_record
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics
    from src.types import Chunk, Example, RetrievedChunk

    assert parse_plan("Sure:\n1. Who directed X?\n2) Where was #1 born?\n3. a\n4. b", 3) == \
        ["Who directed X?", "Where was #1 born?", "a"]
    assert parse_plan("no list here", 3) == []
    assert fill_refs("Where was #1 born, and #3?", ["Ann"]) == "Where was Ann born, and #3?"

    pool = [Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0]) for c in
            ("Film::0", "Film::1", "Ann::0", "Ann::1")]

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS(); cfg.retrieval = NS()
    cfg.pipeline.mode = "orchestrator"
    cfg.pipeline.max_hops = 3
    cfg.pipeline.max_subquestions = 3
    cfg.pipeline.sub_max_hops = 2
    cfg.retrieval.top_k = 2

    class Retriever:
        def __init__(self):
            self.queries = []

        def index(self, chunks):
            pass

        def retrieve(self, query, k):
            self.queries.append(query)
            ids = ["Ann::0", "Ann::1"] if "Ann" in query else ["Film::0", "Film::1"]
            return [RetrievedChunk(chunk=next(c for c in pool if c.chunk_id == i), score=1.0 / r,
                                   rank=r) for r, i in enumerate(ids, 1)]

    class Gen:
        def __init__(self, plan):
            self.plan, self.prompts = plan, []

        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=3)
            if "Break the question" in prompt:
                assert max_tokens == 128 and "Question: where was the director born" in prompt
                return self.plan, m
            if "collecting evidence" in prompt:
                return "KEEP: 1\nNEXT: ANSWER", m
            assert "Answer the question using only the context" in prompt
            if "Question: Who directed the film?" in prompt:
                assert "text of Film::0" in prompt and "Ann::" not in prompt
                return "Ann", m
            if "Question: Where was Ann born?" in prompt:
                assert "text of Ann::0" in prompt and "Film::" not in prompt
                return "Paris", m
            assert "Question: where was the director born" in prompt
            assert "text of Film::0" in prompt and "text of Ann::0" in prompt
            assert "Film::1" not in prompt and "Ann::1" not in prompt
            return "Paris\n", m

    class Lessons:
        def __init__(self):
            self.roles = []

        def lessons(self, step):
            self.roles.append(step.get("role"))
            return []

    ex = Example(id="q", question="where was the director born", answer="Paris", hops=2,
                 chunks=pool, gold_chunk_ids=["Film::0", "Ann::0"])
    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg, p.retriever, p.reranker = cfg, Retriever(), None
    p.generator = Gen("1. Who directed the film?\n2. Where was #1 born?")
    p.lessons = Lessons()
    res = p.run_example(ex)
    assert res.answer == "Paris" and res.stop_reason == "subs=2"
    assert p.retriever.queries == ["Who directed the film?", "Where was Ann born?"]
    assert [h.hop for h in res.hops] == [0, 1] and res.notebook == ["Film::0", "Ann::0"]
    assert p.lessons.roles == ["planner", "subagent", "subagent"]
    subs = [s for s in res.steps if "sub_answer" in s]
    assert [s["sub_answer"] for s in subs] == ["Ann", "Paris"]
    assert subs[1]["planned"] == "Where was #1 born?" and subs[1]["sub_question"] == "Where was Ann born?"
    # planner, two keep steps, two sub-answers, the final answer
    assert len(p.generator.prompts) == 6 and res.gen_calls == 6
    row = score_row(ex, res)
    assert row["em"] == 1 and row["notebook_all_gold"] == 1 and row["fallback"] == 0
    assert trace_record(ex, res)["steps"][0]["plan"] == ["Who directed the film?", "Where was #1 born?"]

    # an unreadable plan runs the original question as the only sub-question
    class Flat(Gen):
        def generate(self, prompt, max_tokens=None):
            if "Answer the question using only" in prompt:
                return "Paris", GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0,
                                           completion_tokens=3)
            return super().generate(prompt, max_tokens)
    p.retriever, p.generator, p.lessons = Retriever(), Flat("I would look it up."), None
    res = p.run_example(ex)
    assert res.stop_reason == "subs=1+plan_fallback"
    assert p.retriever.queries == ["where was the director born"]
    assert not score_row(ex, res)["fallback"]


def test_experience_memory():
    import tempfile
    from pathlib import Path
    from src.agent.memory import (FALLBACK, PLANNER_FEATURES, ExperienceMemory, leaks,
                                  planner_features, score_steps)
    from src.memory.features import FEATURES
    from src.types import Chunk, Example, HopTrace, RetrievedChunk

    class NS:
        pass

    def F(v):
        return {f: float(v) for f in FEATURES}

    pool = {c: Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0])
            for c in ("G::0", "A::0", "X::0", "Y::0")}
    ex = Example(id="q", question="Who directed Night Orchard?", answer="Ann", hops=2,
                 chunks=list(pool.values()), gold_chunk_ids=["G::0"])

    def episode(x, shown, kept, notebook, planner=False):
        res = NS()
        res.steps = ([{"role": "planner", "question": ex.question, "plan": ["a", "b"]}]
                     if planner else [])
        res.steps.append({"hop": 0, "query": "q", "features": F(x), "shown": shown, "kept": kept})
        res.hops = [HopTrace(hop=0, query="q", retrieved=[RetrievedChunk(pool[c], 1.0, r)
                                                          for r, c in enumerate(shown, 1)])]
        res.notebook = notebook
        return res

    def step(x):
        return {"hop": 0, "features": F(x), "question": ex.question, "notebook": []}

    # error typing, gold only after the episode
    (role, key, x, failed, detail), = score_steps(ex, episode(0, ["G::0", "X::0", "Y::0"],
                                                              ["X::0", "Y::0"], ["X::0", "Y::0"]))
    assert role == "subagent" and key == 0 and failed == {"dropped": True, "noise": True}
    assert detail["dropped"] == ["G::0"] and len(x) == len(FEATURES)
    two = Example(id="q2", question=ex.question, answer="Ann", hops=2, chunks=ex.chunks,
                  gold_chunk_ids=["G::0", "A::0"])
    scored = score_steps(two, episode(0, ["G::0", "X::0"], ["G::0"], ["G::0"], planner=True))
    assert scored[0][0] == "planner" and scored[0][3] == {"unreached": True}
    assert scored[0][4]["missing_titles"] == ["A"] and len(scored[0][2]) == len(PLANNER_FEATURES)
    assert scored[1][3] == {"dropped": False, "noise": False}
    assert planner_features("Were Ann and Bo born in the same city?")[PLANNER_FEATURES.index("cue_same")] == 1

    assert leaks("Keep the sentence about Ann.", ex.question, ["Ann"])
    assert leaks("Check the 1990 release.", ex.question, ["G"])
    assert not leaks("Look up each film's director first.", ex.question, ["Night Orchard"])
    assert leaks("Find who directed the Orchard.", ex.question, [])

    # regions: create, probation, support, narrow, expand
    mem = ExperienceMemory(None, None, mode="region", learn=True, warmup=0, min_n=2, reflect=False)
    assert mem.lessons(step(0)) == []
    mem.on_episode(ex, episode(0, ["G::0", "X::0", "Y::0"], ["X::0", "Y::0"], ["X::0", "Y::0"]), {})
    assert [(l.id, l.type, l.source) for l in mem.entries] == \
        [("s1", "dropped", "fallback"), ("s2", "noise", "fallback")]
    s1 = mem.entries[0]
    assert mem.lessons(step(0.5)) == [FALLBACK["dropped"], FALLBACK["noise"]]   # on probation
    mem.on_episode(ex, episode(0.5, ["G::0", "X::0"], ["G::0"], ["G::0"]), {})
    assert (s1.shown, s1.ok_shown, len(s1.support)) == (1, 1, 2)
    assert mem.lessons(step(0.8)) == [FALLBACK["dropped"], FALLBACK["noise"]]
    mem.on_episode(ex, episode(0.8, ["G::0", "X::0"], ["X::0"], ["X::0"]), {})
    assert [e["op"] for e in mem.log] == ["create", "create", "narrow"]
    assert s1.hi[0] < 0.8 and s1.hi[1] == 1.0 and not s1.retired
    assert mem.lessons(step(0.8)) == [FALLBACK["noise"]]                       # cut off by the narrow
    mem.on_episode(ex, episode(0.8, ["G::0", "X::0"], ["X::0"], ["X::0"]), {})
    assert [e["op"] for e in mem.log][-1] == "expand" and len(mem.entries) == 2
    assert s1.hi[0] >= 0.8 and mem.base[("subagent", "dropped")] == [0, 2]

    # frozen copy shows the same; the baselines read the same store
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.json"
        mem.save(path)
        assert path.with_suffix(".log.jsonl").exists()
        frozen = ExperienceMemory(None, None, mode="region", learn=False, load=path)
        assert frozen.lessons(step(0.5)) == mem.lessons(step(0.5))
        frozen.on_episode(ex, episode(0.5, ["G::0"], [], []), {})
        assert [l.shown for l in frozen.entries] == [l.shown for l in mem.entries]
        shuf = ExperienceMemory(None, None, mode="shuffled", learn=False, load=path)
        assert sorted(l.advice for l in shuf.entries) == sorted(l.advice for l in mem.entries)
        assert [l.lo for l in shuf.entries] == [l.lo for l in mem.entries]
        far = step(100.0)
        assert ExperienceMemory(None, None, mode="region", learn=False, load=path).lessons(far) == []
        assert len(ExperienceMemory(None, None, mode="global", learn=False, load=path).lessons(far)) == 2
        flat = ExperienceMemory(None, None, mode="flat", learn=False, load=path)
        assert len(flat.lessons(far)) == 2                       # same question text
        assert flat.lessons(dict(far, question="zzz qqq")) == []
        assert ExperienceMemory(None, None, mode="none", learn=False, load=path).lessons(far) == []


def test_memory_in_the_loop():
    """The real memory inside the main agent, a fake model, run_qa's order of
    calls: run_example, score_row, on_episode."""
    from src.agent.memory import ExperienceMemory
    from src.eval.run_qa import score_row
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics
    from src.types import Chunk, Example, RetrievedChunk

    pool = [Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0])
            for c in ("Film::0", "Film::1", "Ann::0", "Ann::1")]

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS(); cfg.retrieval = NS()
    cfg.pipeline.mode = "orchestrator"
    cfg.pipeline.max_hops = 3
    cfg.pipeline.max_subquestions = 3
    cfg.pipeline.sub_max_hops = 1
    cfg.retrieval.top_k = 2

    class Retriever:
        def index(self, chunks):
            pass

        def retrieve(self, query, k):
            ids = ["Film::1", "Film::0"]        # never the Ann paragraph: planner fails
            return [RetrievedChunk(chunk=next(c for c in pool if c.chunk_id == i), score=1.0 / r,
                                   rank=r) for r, i in enumerate(ids, 1)]

    class Gen:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=3)
            if "improving an evidence-collecting assistant" in prompt:
                return "Look up each person the question names separately.", m
            if "Break the question" in prompt:
                return "1. Who directed the film?", m
            if "collecting evidence" in prompt:
                return "KEEP: 1\nNEXT: ANSWER", m     # keeps Film::1, drops gold Film::0
            return "Ann", m

    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg, p.retriever, p.reranker, p.generator = cfg, Retriever(), None, Gen()
    p.lessons = ExperienceMemory(cfg, p, mode="region", learn=True, warmup=0, min_n=5)
    for i in range(6):
        ex = Example(id=f"q{i}", question=f"Where was the director of film {i} born?",
                     answer="Paris", hops=2, chunks=pool, gold_chunk_ids=["Film::0", "Ann::0"])
        res = p.run_example(ex)
        row = score_row(ex, res)
        p.lessons.on_episode(ex, res, row)
    roles = {(l.role, l.type, l.source) for l in p.lessons.entries}
    assert ("planner", "unreached", "reflection") in roles
    assert ("subagent", "dropped", "reflection") in roles
    # created after episode 1, shown from episode 2 on, in both prompts
    shown = [q for q in p.generator.prompts if "Lessons from earlier questions:" in q]
    assert any("Break the question" in q for q in shown)
    assert any("collecting evidence" in q for q in shown)
    assert "Look up each person the question names separately." in shown[0]
    # reflection calls never count toward an episode's calls
    assert res.gen_calls == 4


def test_memory_v2():
    """Version 2: batched reflection with the generality check, held-out
    credit while building, proven lessons only when frozen."""
    import tempfile
    from pathlib import Path
    import numpy as np
    from src.agent.memory import FALLBACK, ExperienceMemory, score_steps, specific_words
    from src.memory.features import FEATURES
    from src.serving.runner_mlx import GenMetrics
    from src.types import Chunk, Example, HopTrace, RetrievedChunk

    class NS:
        pass

    texts = ["who founded the publisher of the magazine", "when was the band formed",
             "where was the singer born", "which magazine came out first"]
    assert specific_words("Keep results about the publisher.", texts) == ["publisher"]
    assert specific_words("Keep results naming the magazines.", texts) == []     # 2 of 4
    assert specific_words("Keep every result that names an entity the question asks about.",
                          texts) == []
    assert specific_words("Keep the singer.", ["the singer was born"]) == ["singer"]

    pool = {c: Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0])
            for c in ("G::0", "X::0", "Y::0")}

    def ex_(i, q="Who directed the film?"):
        return Example(id=i, question=q, answer="Ann", hops=2, chunks=list(pool.values()),
                       gold_chunk_ids=["G::0"])

    def episode(x, ok):
        res = NS()
        shown, kept = (["G::0", "X::0"], ["G::0"]) if ok else (["G::0", "X::0", "Y::0"],
                                                               ["X::0", "Y::0"])
        res.steps = [{"hop": 0, "query": "q", "features": {f: float(x) for f in FEATURES},
                      "shown": shown, "kept": kept}]
        res.hops = [HopTrace(hop=0, query="q", retrieved=[RetrievedChunk(pool[c], 1.0, r)
                                                          for r, c in enumerate(shown, 1)])]
        res.notebook = kept
        return res

    def step(x):
        return {"hop": 0, "features": {f: float(x) for f in FEATURES}, "question": "q",
                "notebook": []}

    # a lesson waits for failures on batch different questions
    mem = ExperienceMemory(None, None, mode="region", learn=True, warmup=0, min_n=2,
                           reflect=False, credit="heldout", batch=2, max_shown=1, seed=1)
    mem.on_episode(ex_("a"), episode(0.0, ok=False), {})
    mem.on_episode(ex_("a"), episode(0.1, ok=False), {})
    assert mem.entries == [] and len(mem._buffer[("subagent", "dropped")]) == 2
    mem.on_episode(ex_("b"), episode(0.2, ok=False), {})
    assert [(l.id, l.type) for l in mem.entries] == [("s1", "dropped"), ("s2", "noise")]
    s1, s2 = mem.entries
    assert [m[0] for m in s1.members] == ["b", "a"] and len(s1.support) == 2
    assert s1.lo[0] <= 0.1 and s1.hi[0] >= 0.2

    # building: each decision either holds every applicable lesson out or shows one
    saw = {"held": 0, "shown": 0}
    for i in range(40):
        advice = mem.lessons(step(0.15))
        if advice:
            assert len(advice) == 1 and len(mem.last_ids) == 1 and mem.last_held == []
            saw["shown"] += 1
        else:
            assert sorted(mem.last_held) == ["s1", "s2"] and mem.last_ids == []
            saw["held"] += 1
        mem.on_episode(ex_(f"r{i}"), episode(0.15, ok=bool(advice)), {})
    assert saw["held"] > 5 and saw["shown"] > 5
    assert s1.held + s2.held == 2 * saw["held"] and s1.shown + s2.shown == saw["shown"]
    assert s1.ok_held == 0 and s1.ok_shown == s1.shown            # success is the whole step
    assert mem.proven(s1) and mem.proven(s2) and len(mem.entries) == 2

    # frozen: proven lessons only, one per decision, nothing held out
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.json"
        mem.save(path)
        kw = dict(learn=False, load=path, credit="heldout", min_n=2, max_shown=1)
        frozen = ExperienceMemory(None, None, mode="region", **kw)
        assert len(frozen.lessons(step(0.15))) == 1 and frozen.last_held == []
        a, b = frozen.entries
        a.ok_shown = 0                                            # no better than held out
        assert frozen.lessons(step(0.15)) == [b.advice]
        b.held = 1                                                # too few held-out trials
        assert frozen.lessons(step(0.15)) == []
        assert len(ExperienceMemory(None, None, mode="global", **kw).lessons(step(100.0))) == 1
        assert ExperienceMemory(None, None, mode="region", **kw).lessons(step(100.0)) == []

    # retired once both counts reach min_n and it does no better than held out
    s2.ok_shown, s2.ok_held = 0, s2.held
    empty = NS()
    empty.steps, empty.hops, empty.notebook = [], [], []
    mem.on_episode(ex_("z"), empty, {})
    assert s2.retired and not s1.retired and mem.log[-1]["op"] == "retire"

    # batched reflection: a rule about one source step gets one retry, then
    # the fixed sentence; a leak goes straight to the fixed sentence
    class Gen:
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def generate(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            return self.replies.pop(0), GenMetrics(tokens_per_s=1.0, ttft_s=0,
                                                   peak_memory_mb=1.0, completion_tokens=1)

    qs = ["Who founded the publisher of the magazine?", "When was the band formed?",
          "Where was the singer born?", "Which album came out first?"]
    fails = []
    for i, q in enumerate(qs):
        e = ex_(f"m{i}", q)
        (_, _, x, _, detail), = score_steps(e, episode(0.3, ok=False))
        fails.append(ExperienceMemory._failure(x, e, detail))
    general = "Keep every result that states a fact about an entity the question names."
    for replies, source, advice in (
            (["Keep results about the publisher.", general], "reflection_retry", general),
            (["Keep results about the publisher.", "Keep the publisher."], "fallback",
             FALLBACK["dropped"]),
            (["Prefer results dated after 1990."], "fallback", FALLBACK["dropped"]),
            ([general], "reflection", general)):
        pipe = NS()
        pipe.generator = Gen(replies)
        m4 = ExperienceMemory(None, pipe, mode="region", learn=True, warmup=0, credit="heldout",
                              batch=4, generality=True)
        m4._create("subagent", "dropped", fails, np.ones(len(FEATURES)))
        l, = m4.entries
        assert (l.source, l.advice) == (source, advice), (replies, l.source, l.advice)
        assert [m[0] for m in l.members] == ["m0", "m1", "m2", "m3"]
        assert "4 steps, from different questions" in pipe.generator.prompts[0]
        assert all(q in pipe.generator.prompts[0] for q in qs)
        if len(pipe.generator.prompts) > 1:
            assert "too specific: it mentions publisher" in pipe.generator.prompts[1]
    # with a capitalised name in a source question, naming it is a leak
    fails[2]["ex_question"] = "Where was Ann born?"
    pipe = NS()
    pipe.generator = Gen(["Check where Ann was born."])
    m4 = ExperienceMemory(None, pipe, mode="region", learn=True, credit="heldout", batch=4,
                          generality=True)
    m4._create("subagent", "dropped", fails, np.ones(len(FEATURES)))
    assert m4.entries[0].source == "fallback" and len(pipe.generator.prompts) == 1


def test_memory_v2_in_the_loop():
    """Version 2 inside the main agent: held-out decisions are logged in the
    steps, and the counts add up to the decisions where lessons applied."""
    from src.agent.memory import ExperienceMemory
    from src.eval.run_qa import score_row, trace_record
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics
    from src.types import Chunk, Example, RetrievedChunk

    pool = [Chunk(chunk_id=c, text=f"text of {c}", title=c.split("::")[0])
            for c in ("Film::0", "Film::1", "Ann::0", "Ann::1")]

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS(); cfg.retrieval = NS()
    cfg.pipeline.mode = "orchestrator"
    cfg.pipeline.max_hops = 3
    cfg.pipeline.max_subquestions = 3
    cfg.pipeline.sub_max_hops = 1
    cfg.retrieval.top_k = 2

    class Retriever:
        def index(self, chunks):
            pass

        def retrieve(self, query, k):
            ids = ["Film::1", "Film::0"]
            return [RetrievedChunk(chunk=next(c for c in pool if c.chunk_id == i), score=1.0 / r,
                                   rank=r) for r, i in enumerate(ids, 1)]

    class Gen:
        def generate(self, prompt, max_tokens=None):
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=3)
            if "improving an evidence-collecting assistant" in prompt:
                return "Make one lookup for each entity the question names.", m
            if "Break the question" in prompt:
                return "1. Who directed the film?", m
            if "collecting evidence" in prompt:
                return "KEEP: 1\nNEXT: ANSWER", m
            return "Ann", m

    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg, p.retriever, p.reranker, p.generator = cfg, Retriever(), None, Gen()
    p.lessons = ExperienceMemory(cfg, p, mode="region", learn=True, warmup=0, min_n=20,
                                 credit="heldout", batch=2, max_shown=1, generality=True, seed=13)
    recs = []
    for i in range(10):
        ex = Example(id=f"q{i}", question=f"Where was the director of film {i} born?",
                     answer="Paris", hops=2, chunks=pool, gold_chunk_ids=["Film::0", "Ann::0"])
        res = p.run_example(ex)
        score_row(ex, res)
        recs.append(trace_record(ex, res))
        p.lessons.on_episode(ex, res, {})
    assert {(l.role, l.type) for l in p.lessons.entries} >= {("planner", "unreached"),
                                                            ("subagent", "dropped")}
    assert all(len(l.members) == 2 for l in p.lessons.entries)
    steps = [s for r in recs for s in r["steps"] if "reply" in s]
    held = [s for s in steps if s.get("held_ids")]
    shown = [s for s in steps if s.get("lesson_ids")]
    assert held and shown and not any(s.get("held_ids") and s.get("lesson_ids") for s in steps)
    assert all(len(s["lesson_ids"]) == 1 for s in shown)
    n_held = sum(len(s["held_ids"]) for s in held)
    assert sum(l.held for l in p.lessons.entries) == n_held
    assert sum(l.shown for l in p.lessons.entries) == len(shown)
    assert res.gen_calls == 4


if __name__ == "__main__":
    test_feature_schema()
    test_feature_schema_is_pinned()
    test_feature_values()
    test_query_shape_features()
    test_probe_features()
    test_features_blind_to_gold()
    test_features_source_has_no_gold()
    test_deeper_window_and_features()
    test_deeper_source_has_no_gold()
    test_logger_records()
    test_logger_does_not_mutate()
    test_memory_on_off_identical()
    test_reranked_run_compares_at_matched_depth()
    test_shadow_arm_scored_like_primary()
    test_latency_excludes_logging()
    test_shadow_never_changes_primary()
    test_backfill_replays_logger()
    test_backfill_cli_matches_live_run()
    test_failing_log_keeps_run_artefacts()
    test_close_is_all_or_nothing()
    test_config_memory_block()
    test_arm_table_and_ceiling()
    test_predictability_is_held_out()
    test_arms_cli()
    test_make_twins()
    test_lookalike_pairs_and_twins()
    test_failure_typing()
    test_stream_protocol_and_baselines()
    test_twin_stream()
    test_stream_best_arm_pair_and_shuffled_controls()
    test_reader_context_selection()
    test_agent_episode()
    test_orchestrator()
    test_experience_memory()
    test_memory_in_the_loop()
    test_memory_v2()
    test_memory_v2_in_the_loop()
    print("all good")
