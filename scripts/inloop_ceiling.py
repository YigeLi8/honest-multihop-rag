"""Ceilings for in-loop actions, replayed from an ircot experience log.

Before a memory acts inside the retrieve -> reason -> retrieve loop, ask what
each action could recover at best if it were applied on every hop. Only the
actions whose effect is in the log can be replayed:

    k          the run as it was (union of gold hit over its hops)
    deep       every hop retrieves 2k from the primary retriever (gold_ranks)
    dense      every hop uses the dense shadow arm at k
    dense_deep every hop uses dense at 2k
    best_k     per hop, whichever logged arm hit more gold at k (oracle)
    deep_oracle  per hop, 2k only where 2k adds gold the hop missed at k
               (oracle): the most a memory that decides when to deepen can
               get, and the fewest extra chunks it needs for that

The hop queries stay the ones the primary trajectory produced, so this is a
replay with the trajectory fixed: a different context would have changed
later queries, the stop decision and the hop count, none of which is here.
For hybrid the full-pool ranking is fused over a different depth than the
top-k list (src/memory/experience.py), so "deep" on a hybrid run is
approximate; the bm25 and dense ranks are exact. Extra hops, query rewrites
and early stopping need generation and are not replayed.

Outcome per question: all gold chunks in the union of the hops' contexts.
Every action is paired against k on the same questions (McNemar exact,
paired bootstrap), Holm across the actions. Context cost is the mean number
of chunks handed over per question.

--gate asks whether "2k adds gold on this hop" is learnable from the frozen
features (src/memory/features.py stages query / pool / primary_hits, and the
size control hop / pool / hits): held-out AUC with the questions split in
halves by a seeded permutation, fit on one half and scored on the other,
both ways, 5 seeds. Then, at the cost of deepening 30% of the held-out hops,
the learned policy (deepen the 30% with the highest predicted probability)
against uniform k=13 and against deepening a random 30%, paired per split.

    python scripts/inloop_ceiling.py --records results/memory/hybrid_ircot_hotpot_dev_experience.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import holm, mcnemar, paired_bootstrap  # noqa: E402

ACTIONS = ("deep", "dense", "dense_deep", "best_k", "deep_oracle")


def within(ranks, depth):
    return {g for g, (rank, _score) in ranks.items() if rank is not None and rank <= depth}


def replay(records):
    """id -> {action: covered gold set, 'gold': all gold, 'cost': {action: chunks}}"""
    q = defaultdict(lambda: {"gold": set(), "cov": defaultdict(set), "cost": defaultdict(int), "pool": []})
    for r in records:
        k = r["strategy"]["top_k"]
        ho = r["hop_outcome"]
        d = q[r["id"]]
        d["gold"] |= set(ho["gold_ranks"])
        d["cov"]["k"] |= set(ho["gold_hit_ids"])
        deep = within(ho["gold_ranks"], 2 * k)
        d["cov"]["deep"] |= deep
        extra = deep - set(ho["gold_hit_ids"])
        d["cov"]["deep_oracle"] |= set(ho["gold_hit_ids"]) | extra
        d["pool"].append(r["features"]["n_pool_chunks"])
        dn = (r.get("shadow") or {}).get("dense")
        if dn is None:
            raise SystemExit("no dense shadow arm in this log")
        d["cov"]["dense"] |= set(dn["gold_hit_ids"])
        d["cov"]["dense_deep"] |= within(dn["gold_ranks"], 2 * k)
        arms = [set(ho["gold_hit_ids"])] + [set(s["gold_hit_ids"]) for s in r["shadow"].values()]
        d["cov"]["best_k"] |= max(arms, key=len)
        n = len(r["retrieved"])
        for a, c in (("k", n), ("deep", 2 * n), ("dense", n), ("dense_deep", 2 * n), ("best_k", n),
                     ("deep_oracle", 2 * n if extra else n)):
            d["cost"][a] += c
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    with open(args.records) as f:
        header = json.loads(f.readline())
        records = [json.loads(line) for line in f]
    if header["strategy"]["rerank_enabled"]:
        raise SystemExit("reranked run: top_k is the first-stage depth, 2k is not a deeper context")
    q = replay(records)
    ids = sorted(q)
    ok = {a: [int(q[i]["gold"] <= q[i]["cov"][a]) for i in ids] for a in ("k",) + ACTIONS}
    cost = {a: sum(q[i]["cost"][a] for i in ids) / len(ids) for a in ok}
    print(f"{header['run']}: {len(ids)} questions, {len(records)} hop records, "
          f"primary {header['strategy']['method']} k={header['strategy']['top_k']}")
    pool = [p for i in ids for p in q[i]["pool"]]
    k = header["strategy"]["top_k"]
    print(f"pool chunks per hop: mean {sum(pool) / len(pool):.1f}, "
          f"share of hops with pool <= 2k: {sum(p <= 2 * k for p in pool) / len(pool):.3f}")
    print(f"k           all gold {sum(ok['k']) / len(ids):.3f}  chunks/question {cost['k']:.1f}")
    rows = []
    for a in ACTIONS:
        only_k, only_a, p = mcnemar(ok["k"], ok[a])
        diff, lo, hi = paired_bootstrap(ok["k"], ok[a])
        rows.append((a, sum(ok[a]) / len(ids), diff, lo, hi, only_k, only_a, p))
    for row, ph in zip(rows, holm([r[-1] for r in rows])):
        a, rate, diff, lo, hi, only_k, only_a, p = row
        print(f"{a:<11} all gold {rate:.3f}  {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{only_a} vs {only_k}  holm p {ph:.2g}  chunks/question {cost[a]:.1f}")
    if args.gate:
        gate(header, records)


SIZE = ("hop", "n_pool_chunks", "n_hits")


def coverage(records, deep, k, uniform=None):
    """Per question all-gold and chunks, deepening the hops where deep is set
    (to 2k), or every hop at depth uniform."""
    gold, cov, chunks = defaultdict(set), defaultdict(set), defaultdict(int)
    for r, dm in zip(records, deep):
        ho = r["hop_outcome"]
        gold[r["id"]] |= set(ho["gold_ranks"])
        if uniform:
            depth = uniform
            cov[r["id"]] |= within(ho["gold_ranks"], depth)
        else:
            depth = 2 * k if dm else k
            cov[r["id"]] |= set(ho["gold_hit_ids"]) | (within(ho["gold_ranks"], depth) if dm else set())
        chunks[r["id"]] += min(depth, r["features"]["n_pool_chunks"])
    return gold, cov, chunks


def gate(header, records, budget=0.30, seeds=5):
    import ast

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = header["strategy"]["top_k"]
    stage = header["feature_stage"]
    stage = stage if isinstance(stage, dict) else ast.literal_eval(stage)
    names = list(records[0]["features"])
    y = np.array([int(bool(within(r["hop_outcome"]["gold_ranks"], 2 * k)
                           - set(r["hop_outcome"]["gold_hit_ids"]))) for r in records])
    qid = np.array([r["id"] for r in records])
    uq = sorted(set(qid))
    print(f"\nhops where 2k adds gold: {y.sum()} of {len(y)} ({y.mean():.3f})")
    sets = {"size control": list(SIZE)}
    for st, allowed in (("query", {"query"}), ("pool", {"query", "pool"}),
                        ("primary_hits", {"query", "pool", "primary_hits"})):
        sets[st] = [n for n in names if stage.get(n) in allowed]

    def splits():
        for seed in range(seeds):
            half = set(np.random.RandomState(seed).permutation(uq)[:len(uq) // 2])
            m = np.array([i in half for i in qid])
            yield seed, m, ~m
            yield seed, ~m, m

    def fit(X, tr):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])

    for st, cols in sets.items():
        X = np.array([[r["features"][n] for n in cols] for r in records], float)
        aucs = [roc_auc_score(y[te], fit(X, tr).predict_proba(X[te])[:, 1]) for _, tr, te in splits()]
        print(f"{st:<13} {len(cols):>2} features  held-out AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")

    X = np.array([[r["features"][n] for n in sets["primary_hits"]] for r in records], float)
    zero = np.zeros(len(records), bool)
    rate, cost, diffs = defaultdict(list), defaultdict(list), defaultdict(list)
    for seed, tr, te in splits():
        p = fit(X, tr).predict_proba(X)[:, 1]
        learned = (p >= np.quantile(p[te], 1 - budget)) & te
        idx = np.where(te)[0]
        rnd = zero.copy()
        rnd[np.random.RandomState(100 + seed).choice(idx, int(budget * len(idx)), replace=False)] = True
        te_ids = sorted(set(qid[te]))
        out = {}
        for name, (g, c, ch) in (("k", coverage(records, zero, k)),
                                 (f"uniform k={k + 3}", coverage(records, zero, k, k + 3)),
                                 ("random", coverage(records, rnd, k)),
                                 ("learned", coverage(records, learned, k))):
            out[name] = [int(g[i] <= c[i]) for i in te_ids]
            rate[name].append(np.mean(out[name]))
            cost[name].append(np.mean([ch[i] for i in te_ids]))
        for b in (f"uniform k={k + 3}", "random"):
            _, _, pv = mcnemar(out[b], out["learned"])
            diffs[b].append((np.mean(out["learned"]) - np.mean(out[b]), pv))
    print(f"\nmatched cost, deepen {budget:.0%} of held-out hops (primary_hits features), "
          f"{2 * seeds} half-splits of {len(uq) // 2} questions")
    for name in rate:
        print(f"{name:<13} all gold {np.mean(rate[name]):.3f}  chunks/question {np.mean(cost[name]):.1f}")
    for b, d in diffs.items():
        v = np.array([x for x, _ in d])
        ps = [p for _, p in d]
        print(f"learned - {b}: {v.mean():+.3f} +- {v.std():.3f}; per-split McNemar p "
              f"median {np.median(ps):.2g}, min {min(ps):.2g}")


if __name__ == "__main__":
    main()
