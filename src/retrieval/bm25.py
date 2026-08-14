"""BM25 over the per-question candidate sentences (bm25s backend; maybe
pyserini later if Lucene parity turns out to matter)."""
import bm25s

from src.types import RetrievedChunk


class Bm25Retriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bm25 = None
        self.chunks = []

    def index(self, chunks):
        self.chunks = list(chunks)
        tokens = bm25s.tokenize([c.text for c in self.chunks],
                                stopwords="en", show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)

    def retrieve(self, query, top_k):
        k = min(top_k, len(self.chunks))
        q = bm25s.tokenize(query, stopwords="en", show_progress=False)
        ids, scores = self.bm25.retrieve(q, k=k, show_progress=False)
        return [RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=r + 1)
                for r, (i, s) in enumerate(zip(ids[0], scores[0]))]
