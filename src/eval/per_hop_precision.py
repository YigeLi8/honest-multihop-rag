"""Precision/recall of retrieved chunks against gold supporting facts, per hop.

Two views, because the datasets differ in what "per hop" can mean:

* unordered gold (hotpot): each hop's retrieved set is scored against the full
  gold set, plus the cumulative recall after each hop, i.e. how much of the
  gold chain the model has seen by then.
* ordered gold (2wiki reasoning_path, musique decomposition): on top of the
  above, hop-aligned recall: was the h-th gold chunk in context by hop h. That
  is the closest thing to "did hop h retrieve what hop h needed" without
  assuming the model follows the annotated order.
"""


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def precision_recall(retrieved_ids, gold_ids):
    if not retrieved_ids:
        return 0.0, 0.0
    got, gold = set(retrieved_ids), set(gold_ids)
    hit = len(got & gold)
    prec = hit / len(retrieved_ids)
    rec = hit / len(gold) if gold else 0.0
    return prec, rec


def ordered_gold(ex):
    """Gold chunk ids in reasoning order, or [] when the dataset has no order."""
    path = getattr(ex, "reasoning_path", None) or []
    ids = [s.get("gold_chunk_id") for s in path if isinstance(s, dict)]
    return [i for i in ids if i]


def score(results, examples=None):
    """Per-hop aggregates over PipelineResults.

    examples: optional {example_id: Example}; when given and the examples carry
    an ordered reasoning_path, hop-aligned recall is reported as well.
    """
    by_hop = {}
    for res in results:
        ex = examples.get(res.example_id) if examples else None
        order = ordered_gold(ex) if ex is not None else []
        seen = set()
        for h in res.hops:
            ids = [rc.chunk.chunk_id for rc in h.retrieved]
            p, r = precision_recall(ids, h.gold_chunk_ids)
            seen |= set(ids)
            gold = set(h.gold_chunk_ids)
            cum = len(seen & gold) / len(gold) if gold else 0.0
            aligned = None
            if order and h.hop < len(order):
                aligned = float(order[h.hop] in seen)
            by_hop.setdefault(h.hop, []).append((p, r, cum, aligned, len(ids)))

    out = {}
    for hop, rows in sorted(by_hop.items()):
        ns = [float(n) for *_, n in rows]
        m = _mean(ns)
        aligned = [a for _, _, _, a, _ in rows if a is not None]
        out[hop] = {
            "precision": _mean([p for p, *_ in rows]),
            "recall": _mean([r for _, r, *_ in rows]),
            "cumulative_recall": _mean([c for _, _, c, *_ in rows]),
            "n": len(rows),
            "retrieved_mean": m,
            "retrieved_var": _mean([(x - m) ** 2 for x in ns]) if len(ns) > 1 else 0.0,
        }
        if aligned:
            out[hop]["aligned_recall"] = _mean(aligned)
            out[hop]["n_aligned"] = len(aligned)
    return out


def plot_per_hop(hop_stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hops = sorted(hop_stats)
    prec = [hop_stats[h]["precision"] for h in hops]
    rec = [hop_stats[h]["recall"] for h in hops]
    cum = [hop_stats[h]["cumulative_recall"] for h in hops]
    counts = [hop_stats[h]["n"] for h in hops]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hops, prec, "o-", label="precision (this hop)")
    ax.plot(hops, rec, "s--", label="recall (this hop)")
    ax.plot(hops, cum, "^:", label="cumulative recall")
    if all("aligned_recall" in hop_stats[h] for h in hops):
        ax.plot(hops, [hop_stats[h]["aligned_recall"] for h in hops], "d-.",
                label="h-th gold chunk seen by hop h")
    for h, p, n in zip(hops, prec, counts):
        ax.annotate(f"n={n}", (h, p), fontsize=7, xytext=(3, -12),
                    textcoords="offset points")
    ax.set_xticks(hops)
    ax.set_xlabel("hop")
    ax.set_ylabel("retrieval quality vs gold facts")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
