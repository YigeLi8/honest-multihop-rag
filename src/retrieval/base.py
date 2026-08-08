def get_retriever(cfg):
    method = cfg.retrieval.method
    if method == "bm25":
        from .bm25 import Bm25Retriever
        return Bm25Retriever(cfg)
    if method == "dense":
        from .dense import DenseRetriever
        return DenseRetriever(cfg)
    if method == "hybrid":
        from .hybrid import HybridRetriever
        return HybridRetriever(cfg)
    raise ValueError(f"unknown retrieval method: {method}")
