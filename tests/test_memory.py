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


def stub_cfg(mode, store=None, top_k=3):
    cfg = SimpleNamespace(
        run=SimpleNamespace(name="stub"),
        dataset=SimpleNamespace(name="synthetic"),
        retrieval=SimpleNamespace(method="overlap", top_k=top_k),
        pipeline=SimpleNamespace(mode=mode, max_hops=3))
    if store is not None:
        cfg.memory = SimpleNamespace(enabled=True, mode="log", log_arms=[], store=str(store))
    return cfg


def stub_pipeline(mode, store=None, shadow=None, generate=True, top_k=3, rerank_top_n=None):
    """A pipeline on fakes; with a store it also carries a logger."""
    from src.memory.experience import ExperienceLogger
    from src.pipeline.multihop import MultiHopPipeline
    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg = stub_cfg(mode, store, top_k)
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
        assert header["record_schema_version"] == 1
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
            # everything that comes from a label sits in these three blocks, nowhere else
            assert set(r) - {"hop_outcome", "shadow", "outcome"} == {
                "run", "dataset", "id", "qtype", "hop", "query", "strategy", "retrieved",
                "first_stage", "feature_schema_version", "features", "probe_features"}


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

    # similarity memory: when the arm that hits is a function of the question
    # text, the nearest past questions route it; random text does not
    from src.memory.arms import neighbour_routed_rate
    table["question"] = ["long query %03d" % i if table["arm:a"][i] else "short one %03d" % i
                         for i in range(n)]
    nn = neighbour_routed_rate(table, ks=[5], seeds=[13, 17]).set_index("stage")
    assert nn.loc["question_knn5"]["gain"] > 0.15 and nn.loc["question_knn5"]["n_splits"] == 4
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
                     ("Acid Lake", 1, "Acid rain feeds it.")],
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


def test_lookalike_pairs_and_twins():
    import numpy as np
    import pandas as pd
    from src.memory.lookalike import (best_arm, natural_pairs, pair_summary, random_pairs,
                                      twin_of, twin_table)
    n = 40
    rng = np.random.RandomState(1)
    table = pd.DataFrame({"id": [f"q{i}" for i in range(n)]})
    # two families of questions: arm a hits the long family, arm b the short one
    fam = np.array([i % 2 for i in range(n)])
    table["question"] = ["long query about %s %03d" % ("rivers" if i % 4 < 2 else "rivers", i)
                         if fam[i] else "short one on %s %03d" % ("kings", i) for i in range(n)]
    table["arm:a"] = fam.astype(float)
    table["arm:b"] = (1 - fam).astype(float)
    assert best_arm(np.array([[1, 1], [0, 0], [1, 0]]), ["a", "b"]) == ["all", "none", "a"]

    pairs = natural_pairs(table)
    assert len(pairs) <= n and (pairs["similarity"].diff().dropna() <= 1e-9).all()
    assert set(pairs.columns) >= {"id_a", "id_b", "similarity", "a_same", "b_same", "best_same"}
    # the nearest neighbour is from the same family, so every pair agrees
    assert pairs["a_same"].all() and pairs["best_same"].all()
    rnd = random_pairs(table, 400, seed=3)
    assert 0.35 < rnd["a_same"].mean() < 0.65 and len(rnd) == 400
    summary = pair_summary(pairs, table, n_random=400)
    nearest = summary[summary["pairs"] == "nearest_all"].set_index("agreement")
    random_ = summary[summary["pairs"] == "random"].set_index("agreement")
    assert nearest.loc["a"]["rate"] == 1.0 and random_.loc["a"]["rate"] < 0.7
    assert (summary["lo"] <= summary["rate"]).all() and (summary["rate"] <= summary["hi"]).all()
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
    assert row.loc["boundary", "op_create"] > 0 and row.loc["utility", "op_narrow"] == 0
    for c in ("rate", "false_application", "harm", "false_rejection", "repeated_failure",
              "boundary_precision", "boundary_recall"):
        vals = summary[c].dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), c
    assert log and log[0]["policy"] == "boundary" and log[0]["op"] == "create"


if __name__ == "__main__":
    test_feature_schema()
    test_feature_schema_is_pinned()
    test_feature_values()
    test_query_shape_features()
    test_probe_features()
    test_features_blind_to_gold()
    test_features_source_has_no_gold()
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
    print("all good")
