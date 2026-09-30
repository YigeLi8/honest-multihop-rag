"""Checks that don't need the real data or models.

Run from the repo root: python -m tests.test_basic (or pytest).
"""
from data.prepare_hotpotqa import build_record
from src.eval.gold_mapping import map_facts
from src.retrieval.hybrid import HybridRetriever, minmax
from src.types import Chunk, RetrievedChunk

RAW = {
    "_id": "q1",
    "question": "Which university did the author of Book X attend?",
    "answer": "Some University",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": [["Book X", 0], ["Author Y", 1], ["Author Y", 9]],
    "context": [
        ["Book X", ["Book X was written by Author Y.", "It sold well."]],
        ["Author Y", ["Author Y is a writer.", "They attended Some University."]],
        ["Distractor", ["Nothing to see here."]],
    ],
}


def test_map_facts():
    sents = {t: s for t, s in RAW["context"]}
    mapped, unmapped = map_facts([(t, i) for t, i in RAW["supporting_facts"]], sents)
    assert mapped == ["Book X::0", "Author Y::1"]
    assert unmapped == [("Author Y", 9)]   # index past the paragraph end


def test_build_record():
    rec, unmapped = build_record(RAW)
    assert rec["id"] == "q1" and rec["hops"] == 2
    assert len(rec["chunks"]) == 5
    assert rec["chunks"][0]["chunk_id"] == "Book X::0"
    assert rec["gold_chunk_ids"] == ["Book X::0", "Author Y::1"]
    assert len(unmapped) == 1


def test_minmax():
    assert minmax([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert minmax([5.0, 5.0]) == [1.0, 1.0]


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k):
        return self.hits[:top_k]


def _hit(cid, score, rank):
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, text=cid), score=score, rank=rank)


def test_hybrid_fusion():
    h = HybridRetriever.__new__(HybridRetriever)

    class Cfg:
        pass
    h.cfg = Cfg(); h.cfg.retrieval = Cfg(); h.cfg.retrieval.hybrid = Cfg()
    h.cfg.retrieval.hybrid.alpha = 0.5
    h.bm25 = FakeRetriever([_hit("a", 10.0, 1), _hit("b", 5.0, 2), _hit("c", 0.0, 3)])
    h.dense = FakeRetriever([_hit("b", 0.9, 1), _hit("a", 0.8, 2), _hit("d", 0.1, 3)])

    out = h.retrieve("q", 3)
    # a: 0.5*1.0 + 0.5*0.875, b: 0.5*0.5 + 0.5*1.0, c and d both minmax to 0
    assert [r.chunk.chunk_id for r in out[:2]] == ["a", "b"]
    assert len(out) == 3 and [r.rank for r in out] == [1, 2, 3]


def test_answer_metrics():
    from src.eval.answer_metrics import em, f1
    assert em("The Beatles", "beatles") == 1.0
    assert em("yes", "no") == 0.0
    assert f1("Barack Obama", "Obama, Barack") == 1.0
    assert f1("yes", "no") == 0.0          # yes/no disagreement zeroes f1
    assert 0.0 < f1("the red car", "red bicycle") < 1.0


def test_precision_recall():
    from src.eval.per_hop_precision import precision_recall
    p, r = precision_recall(["a", "b", "c", "d"], ["a", "x"])
    assert p == 0.25 and r == 0.5
    assert precision_recall([], ["a"]) == (0.0, 0.0)


def test_decoupling_table():
    from src.eval.decoupling import build_table
    t = build_table([(True, True), (True, False), (True, False),
                     (False, True), (False, False)])
    assert (t.ac_rc, t.ac_rw, t.aw_rc, t.aw_rw) == (1, 2, 1, 1)
    assert t.n == 5 and abs(t.illusion_rate - 0.4) < 1e-9


def test_wilson_ci():
    from src.eval.stats import wilson_ci
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(0, 200)
    assert lo == 0.0 and hi < 0.02


def test_score_hops():
    from src.eval.per_hop_precision import score
    from src.types import Chunk, HopTrace, PipelineResult, RetrievedChunk

    def hit(cid):
        return RetrievedChunk(chunk=Chunk(chunk_id=cid, text=""), score=1.0, rank=1)

    res = PipelineResult(example_id="x", answer="", hops=[
        HopTrace(hop=0, query="q", retrieved=[hit("a"), hit("b")], gold_chunk_ids=["a"]),
        HopTrace(hop=1, query="q2", retrieved=[hit("c")], gold_chunk_ids=["c", "d"]),
    ])
    s = score([res])
    assert s[0]["precision"] == 0.5 and s[0]["recall"] == 1.0
    assert s[1]["recall"] == 0.5 and s[1]["retrieved_mean"] == 1.0


def test_parse_step():
    from src.pipeline.multihop import parse_step
    assert parse_step("thinking...\nSEARCH: who wrote book x") == (None, "who wrote book x")
    assert parse_step("ANSWER: yes") == ("yes", None)
    assert parse_step("no marker at all") == (None, None)


def test_ircot_loop():
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS(); cfg.retrieval = NS()
    cfg.pipeline.mode = "multihop_ircot"
    cfg.pipeline.max_hops = 3
    cfg.retrieval.top_k = 2

    class FakeIndexRetriever:
        def index(self, chunks):
            pass

        def retrieve(self, query, k):
            cid = "hop2::0" if "university" in query else "hop1::0"
            return [_hit(cid, 1.0, 1)]

    class FakeGen:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, max_tokens=None):
            self.calls += 1
            m = GenMetrics(tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0,
                           completion_tokens=5)
            if self.calls == 1:
                return "SEARCH: which university", m
            return "ANSWER: some university", m

    from src.types import Example
    ex = Example(id="q", question="where did the author study", answer="some university",
                 hops=2, gold_chunk_ids=["hop1::0", "hop2::0"])
    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg = cfg
    p.retriever = FakeIndexRetriever()
    p.reranker = None
    p.generator = FakeGen()

    res = p.run_example(ex)
    assert res.answer == "some university"
    assert len(res.hops) == 2                       # stopped once it answered
    assert res.hops[1].query == "which university"
    got = {rc.chunk.chunk_id for h in res.hops for rc in h.retrieved}
    assert got == {"hop1::0", "hop2::0"}            # both hops contributed
    assert res.reasoning_tokens == 10


def test_closed_book():
    from src.pipeline.multihop import MultiHopPipeline
    from src.serving.runner_mlx import GenMetrics
    from src.types import Example

    class NS:
        pass
    cfg = NS(); cfg.pipeline = NS()
    cfg.pipeline.mode = "closed_book"

    class FakeGen:
        def generate(self, prompt, max_tokens=None):
            assert "Context" not in prompt
            return "Paris\nbecause it is the capital", GenMetrics(
                tokens_per_s=10.0, ttft_s=0, peak_memory_mb=1.0, completion_tokens=3)

    p = MultiHopPipeline.__new__(MultiHopPipeline)
    p.cfg = cfg
    p.retriever = None
    p.reranker = None
    p.generator = FakeGen()
    ex = Example(id="q", question="capital of france", answer="Paris", hops=1,
                 gold_chunk_ids=["x::0"])
    res = p.run_example(ex)
    assert res.answer == "Paris"
    assert res.hops[0].retrieved == [] and res.hops[0].gold_chunk_ids == ["x::0"]
    assert res.raw_outputs == ["Paris\nbecause it is the capital"]


def test_score_row():
    from src.eval.run_qa import answer_type, score_row
    from src.types import Chunk, Example, HopTrace, PipelineResult, RetrievedChunk

    def hit(cid):
        return RetrievedChunk(chunk=Chunk(chunk_id=cid, text=""), score=1.0, rank=1)
    ex = Example(id="q", question="?", answer="yes", hops=2, qtype="comparison",
                 gold_chunk_ids=["a", "b"])
    res = PipelineResult(example_id="q", answer="Yes.", hops=[
        HopTrace(hop=0, query="?", retrieved=[hit("a"), hit("z")], gold_chunk_ids=["a", "b"])])
    row = score_row(ex, res)
    assert row["em"] == 1.0 and row["retrieval_correct"] == 0
    assert row["n_gold"] == 2 and row["n_gold_hit"] == 1 and row["gold_recall"] == 0.5
    assert row["answer_type"] == "yesno" and answer_type("Barack Obama") == "span"


if __name__ == "__main__":
    test_map_facts()
    test_build_record()
    test_minmax()
    test_hybrid_fusion()
    test_answer_metrics()
    test_precision_recall()
    test_decoupling_table()
    test_wilson_ci()
    test_score_hops()
    test_parse_step()
    test_ircot_loop()
    test_closed_book()
    test_score_row()
    print("all good")
