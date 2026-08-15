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


if __name__ == "__main__":
    test_map_facts()
    test_build_record()
    test_minmax()
    test_hybrid_fusion()
    test_answer_metrics()
    test_precision_recall()
    test_decoupling_table()
    print("all good")
