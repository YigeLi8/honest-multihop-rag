"""Small stats helpers: Wilson intervals for one proportion, and paired
comparisons for two runs scored on the same questions (the configs here are
always evaluated on identical question sets, so paired tests are the right
ones and much tighter than comparing two marginal intervals)."""
import math
import random


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion (k successes out of n)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap(a, b, n_boot=10000, seed=13):
    """Bootstrap CI on mean(b) - mean(a) over paired per-question scores.

    Resamples questions with replacement, keeping each question's (a, b) pair
    together. Returns (diff, lo, hi) at 95%.
    """
    assert len(a) == len(b) and a, "paired inputs must be the same non-zero length"
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return sum(diffs) / n, lo, hi


def mcnemar(a, b):
    """Exact McNemar test on paired binary outcomes.

    a, b: sequences of 0/1. Returns (n_only_a, n_only_b, p_value) where the
    p-value is the two-sided exact binomial probability on the discordant
    pairs. With no discordant pairs the runs are identical and p = 1.
    """
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return only_a, only_b, min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out
