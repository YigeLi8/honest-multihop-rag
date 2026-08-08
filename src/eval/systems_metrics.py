"""nDCG@10 for the BEIR sanity check, plus aggregation of serving metrics."""


def ndcg_at_k(ranked_ids, qrels, k=10):
    raise NotImplementedError


def aggregate(results, config_name, device):
    # means of tokens/s, ttft, latency, peak memory, reasoning length; keep n
    raise NotImplementedError
