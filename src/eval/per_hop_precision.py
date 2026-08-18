"""Precision/recall of retrieved chunks against gold supporting facts."""


def _mean(xs):
    return sum(xs) / len(xs)


def precision_recall(retrieved_ids, gold_ids):
    if not retrieved_ids:
        return 0.0, 0.0
    got, gold = set(retrieved_ids), set(gold_ids)
    hit = len(got & gold)
    prec = hit / len(retrieved_ids)
    rec = hit / len(gold) if gold else 0.0
    return prec, rec


def score(results):
    """Per-hop aggregates over PipelineResults.

    Note: on hotpot the gold set isn't ordered by hop, so hop-level precision is
    against the full gold set; 2wiki's reasoning_path allows real per-hop gold.
    """
    by_hop = {}
    for res in results:
        for h in res.hops:
            ids = [rc.chunk.chunk_id for rc in h.retrieved]
            p, r = precision_recall(ids, h.gold_chunk_ids)
            by_hop.setdefault(h.hop, []).append((p, r, len(ids)))

    out = {}
    for hop, rows in sorted(by_hop.items()):
        ps = [p for p, _, _ in rows]
        rs = [r for _, r, _ in rows]
        ns = [float(n) for _, _, n in rows]
        m = _mean(ns)
        out[hop] = {
            "precision": _mean(ps),
            "recall": _mean(rs),
            "n": len(rows),
            "retrieved_mean": m,
            "retrieved_var": _mean([(x - m) ** 2 for x in ns]) if len(ns) > 1 else 0.0,
        }
    return out


def plot_per_hop(hop_stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hops = sorted(hop_stats)
    prec = [hop_stats[h]["precision"] for h in hops]
    rec = [hop_stats[h]["recall"] for h in hops]
    counts = [hop_stats[h]["n"] for h in hops]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hops, prec, "o-", label="precision")
    ax.plot(hops, rec, "s--", label="recall")
    for h, p, n in zip(hops, prec, counts):
        ax.annotate(f"n={n}", (h, p), fontsize=7, xytext=(3, -12),
                    textcoords="offset points")
    ax.set_xticks(hops)
    ax.set_xlabel("hop")
    ax.set_ylabel("retrieval quality vs gold facts")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
