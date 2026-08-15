"""Precision/recall of retrieved chunks against gold supporting facts."""


def precision_recall(retrieved_ids, gold_ids):
    if not retrieved_ids:
        return 0.0, 0.0
    got, gold = set(retrieved_ids), set(gold_ids)
    hit = len(got & gold)
    prec = hit / len(retrieved_ids)
    rec = hit / len(gold) if gold else 0.0
    return prec, rec


def score(results):
    """Per-hop aggregates over PipelineResults (multihop runs, day 3+)."""
    by_hop = {}
    for res in results:
        for h in res.hops:
            ids = [rc.chunk.chunk_id for rc in h.retrieved]
            by_hop.setdefault(h.hop, []).append(precision_recall(ids, h.gold_chunk_ids))
    return {
        hop: {
            "precision": sum(p for p, _ in prs) / len(prs),
            "recall": sum(r for _, r in prs) / len(prs),
            "n": len(prs),
        }
        for hop, prs in sorted(by_hop.items())
    }
