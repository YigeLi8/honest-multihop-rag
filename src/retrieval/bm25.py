"""BM25 over the per-question candidate paragraphs (bm25s backend; maybe
pyserini later if Lucene parity turns out to matter)."""


class Bm25Retriever:
    def __init__(self, cfg):
        self.cfg = cfg

    def index(self, chunks):
        raise NotImplementedError

    def retrieve(self, query, top_k):
        raise NotImplementedError
