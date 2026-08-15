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
