"""IRCoT-style loop: retrieve, reason, retrieve again, answer.

single_hop mode is just the 1-hop special case, so the baseline and the
multi-hop runs share this code. Every hop gets logged as a HopTrace so per-hop
precision can be scored afterwards.
"""
from src.retrieval.base import get_retriever
from src.retrieval.rerank import Reranker


class MultiHopPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = get_retriever(cfg)
        self.reranker = Reranker(cfg) if cfg.rerank.enabled else None

    def run_example(self, ex):
        raise NotImplementedError

    def run(self, examples):
        return [self.run_example(ex) for ex in examples]
