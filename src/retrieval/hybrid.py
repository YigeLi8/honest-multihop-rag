"""Score fusion: alpha * dense + (1 - alpha) * bm25, min-max normalized."""
from .bm25 import Bm25Retriever
from .dense import DenseRetriever


class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bm25 = Bm25Retriever(cfg)
        self.dense = DenseRetriever(cfg)

    def index(self, chunks):
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query, top_k):
        # pull a wider pool from both, normalize, blend, cut to top_k
        raise NotImplementedError
