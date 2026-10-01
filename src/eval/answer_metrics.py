"""Answer EM/F1 with the normalization the official hotpot eval uses."""
import re
import string
from collections import Counter


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    # official eval zeroes the score when one side is yes/no and they disagree
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pt), overlap / len(gt)
    return 2 * prec * rec / (prec + rec)


def contains(pred, gold, max_extra=4):
    """Capped containment: the normalized gold answer occurs as a contiguous
    token span of the normalized prediction, and the prediction carries at
    most max_extra tokens beyond it.

    EM misses "Mumbai, Maharashtra" for "Mumbai" and "Kansas Song (We're From
    Kansas)" for "Kansas Song": the answer is there with a qualifier. The cap
    is what keeps this from crediting a sentence that happens to mention the
    answer ("Edward G. Robinson played Dathan in The Ten Commandments." for
    "Dathan"): the ircot readers answer in sentences a third of the time, and
    without the cap containment would hand them ten extra points. yes/no gold
    answers are scored by EM alone ("no, it was yes" contains both).
    """
    p, g = normalize(pred).split(), normalize(gold).split()
    if not g or g in (["yes"], ["no"]):
        return float(p == g)
    if len(p) - len(g) > max_extra:
        return 0.0
    n = len(g)
    return float(any(p[i:i + n] == g for i in range(len(p) - n + 1)))


# The evidence-independent correctness criteria, strictest first. Each is a
# function of (pred, gold) in [0, 1]; a criterion is met when its best value
# over the gold answer and its aliases is at least its threshold.
CRITERIA = {
    "em": (em, 1.0),
    "f1_50": (f1, 0.5),
    "contain": (contains, 1.0),
}


def correct_by(pred, golds, criterion):
    """Is pred correct under one of CRITERIA, taking the best gold alias."""
    fn, threshold = CRITERIA[criterion]
    golds = [g for g in golds if g]
    return bool(golds) and max(fn(pred, g) for g in golds) >= threshold
