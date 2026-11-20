"""Where a run's wrong answers are lost, and paired comparisons between runs
on the same questions (docs/plan.md, Stage 6b).

    python -m src.eval.agent_losses --out agent_dev_hotpot_explore \
        --runs hybrid_ircot_hotpot_dev orchestrator_hotpot_explore \
        --compare hybrid_ircot_hotpot_dev=orchestrator_hotpot_explore

Losses, one row per run (results/<out>_losses.csv). Every wrong answer (EM)
is put in one place, by where the gold answer string was:

    in_context      a sentence the answer was read from holds the string:
                    the notebook for the agents, everything retrieved for the
                    other modes and for an agent that kept nothing
    dropped         retrieved by some search, not among the kept sentences
    not_retrieved   in no retrieved sentence
    yesno_gold      the gold answer is yes or no, so there is no string

and, over the wrong answers to questions with a span answer, by what the
reply was: yes or no, a span inside the gold answer, a span around it, or
something else. For the main agent, also how many wrong answers had a
subagent's answer equal to the gold. The string test is exact after the
answer normalization, sentence by sentence: an answer the text only
paraphrases, or one split over two sentences, counts as not there, so
in_context is a lower bound.

Paired (results/<out>_paired.csv), one row per base=arm pair of --compare:
EM, F1, F1 >= 0.5 and capped containment differences with paired bootstrap
intervals, exact McNemar on EM with Holm within the --compare group, EM by
question type, calls and latency. When the arm is the main agent, also how
many of the questions it won had a subagent's answer equal to the gold, and
on how many of the ones it lost its final answer repeats a subagent's
answer. Gold is read here and nowhere in a run.

A run named a+b is runs a and b read as one, for a comparison pooled over
folds (their ids must not overlap); --data then takes every file the gold is
in:

    python -m src.eval.agent_losses --out agent_dev2_hotpot_pooled \
        --data data/processed/hotpotqa_dev.jsonl data/processed/hotpotqa_train.jsonl \
        --compare orchestrator_guard_hotpot_explore+orchestrator_guard_hotpot_train_hard=\
orchestrator_guard_wide_hotpot_explore+orchestrator_guard_wide_hotpot_train_hard
"""
import argparse
import csv
import json
from pathlib import Path

from src.eval.answer_metrics import correct_by, em, normalize
from src.eval.stats import holm, mcnemar, paired_bootstrap

PLACES = ("in_context", "dropped", "not_retrieved", "yesno_gold")
REPLIES = ("yesno", "inside_gold", "around_gold", "other", "empty")


def load_run(results_dir, run):
    """({id: qa row}, {id: trace record}) of a run, in file order; a+b is runs
    a and b one after the other."""
    results_dir, rows, traces = Path(results_dir), {}, {}
    for part in run.split("+"):
        with open(results_dir / f"{part}_qa.csv") as f:
            got = {r["id"]: r for r in csv.DictReader(f)}
        assert not set(got) & set(rows), f"{part} repeats ids of {run}"
        rows.update(got)
        with open(results_dir / f"{part}_trace.jsonl") as f:
            traces.update({d["id"]: d for d in map(json.loads, f)})
    return rows, traces


def load_gold(data_files, ids):
    """{id: (gold answers, {chunk_id: text})} for the ids asked, from one data
    file or several."""
    want, out = set(ids), {}
    for data_file in [data_files] if isinstance(data_files, (str, Path)) else data_files:
        with open(data_file) as f:
            for line in f:
                # records start {"id": "...", so the id is read without parsing the line
                if line[:80].split('"')[3] in want:
                    d = json.loads(line)
                    golds = [d["answer"]] + [a for a in d.get("answer_aliases", []) if a]
                    out[d["id"]] = (golds, {c["chunk_id"]: c["text"] for c in d["chunks"]})
    return out


def holds(text, answer):
    a = normalize(answer)
    return bool(a) and f" {a} " in f" {normalize(text)} "


def place_of(trace, golds, texts):
    """Where the gold answer string was for one question (one of PLACES). What
    was read is the notebook, or everything retrieved when the final read fell
    back to it ("+fallback" or, under pipeline.final_min_notebook, "+thin" in
    the main agent's stop_reason; an empty notebook in any run)."""
    if all(normalize(g) in ("yes", "no") for g in golds):
        return "yesno_gold"
    retrieved = {c for h in trace["hops"] for c in h["retrieved"]}
    stop = trace.get("stop_reason") or ""
    wide = "+thin" in stop or "+fallback" in stop
    read = retrieved if wide else set(trace.get("notebook") or retrieved)

    def among(ids):
        return any(holds(texts[c], g) for c in ids if c in texts for g in golds)
    if among(read):
        return "in_context"
    return "dropped" if among(retrieved) else "not_retrieved"


def reply_of(pred, golds):
    """What a wrong reply was (one of REPLIES)."""
    p = normalize(pred)
    if not p:
        return "empty"
    if p in ("yes", "no"):
        return "yesno"
    gs = [normalize(g) for g in golds if normalize(g)]
    if any(f" {p} " in f" {g} " for g in gs):
        return "inside_gold"
    if any(f" {g} " in f" {p} " for g in gs):
        return "around_gold"
    return "other"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def losses(run, rows, traces, gold):
    ids = list(rows)
    out = {"run": run, "n": len(ids), "em": mean(float(rows[i]["em"]) for i in ids),
           "f1": mean(float(rows[i]["f1"]) for i in ids)}
    for crit in ("f1_50", "contain"):
        out[crit] = mean(correct_by(rows[i]["answer_pred"], gold[i][0], crit) for i in ids)
    wrong = [i for i in ids if not float(rows[i]["em"])]
    out["wrong"] = len(wrong)
    place = {i: place_of(traces[i], *gold[i]) for i in ids}
    reply = {i: reply_of(rows[i]["answer_pred"], gold[i][0]) for i in wrong
             if place[i] != "yesno_gold"}
    for p in PLACES:
        out[p] = sum(place[i] == p for i in wrong)
    for r in REPLIES:
        out[f"reply_{r}"] = sum(k == r for k in reply.values())
    # the two that say what to work on: the string was read and the reply
    # was still yes/no or about something else
    for r in ("yesno", "other"):
        out[f"in_context_{r}"] = sum(place[i] == "in_context" and k == r for i, k in reply.items())
    # all questions by place, so a place's EM can be read off
    for p in PLACES:
        qs = [i for i in ids if place[i] == p]
        out[f"n_{p}"], out[f"em_{p}"] = len(qs), mean(float(rows[i]["em"]) for i in qs)
    subs = {i: [s["sub_answer"] for s in traces[i].get("steps", []) if "sub_answer" in s]
            for i in ids}
    if any(subs.values()):
        def right(a, i):
            return any(em(a, g) for g in gold[i][0])
        out["wrong_last_sub_right"] = sum(bool(subs[i]) and right(subs[i][-1], i) for i in wrong)
        out["wrong_any_sub_right"] = sum(any(right(a, i) for a in subs[i]) for i in wrong)
    for field in ("gen_calls", "latency_s", "notebook_size", "notebook_all_gold"):
        if field in rows[ids[0]]:
            out[field] = mean(float(rows[i][field]) for i in ids)
    if "guard_reads" in rows[ids[0]]:
        fired = [i for i in ids if int(rows[i]["guard_final"])]
        out["guard_reads"] = sum(int(rows[i]["guard_reads"]) for i in ids)
        out["guard_final"] = len(fired)
        out["guard_final_right"] = sum(float(rows[i]["em"]) > 0 for i in fired)
        # where the string was when the final answer got its second read
        for p in PLACES[:3]:
            out[f"guard_final_{p}"] = sum(place[i] == p for i in fired)
        # guard_unsupported runs: the second reads given to answers not in the context
        recs = {i: traces[i].get("guards") or [] for i in ids}
        if any("why" in g for rs in recs.values() for g in rs):
            unsup = [i for i in fired if any(g["where"] == "final" and g.get("why") == "unsupported"
                                             for g in recs[i])]
            out["guard_reads_unsupported"] = sum(g.get("why") == "unsupported"
                                                 for rs in recs.values() for g in rs)
            out["guard_final_unsupported"] = len(unsup)
            out["guard_final_unsupported_right"] = sum(float(rows[i]["em"]) > 0 for i in unsup)
    return out


def paired(base_run, arm_run, base, arm, gold, arm_traces=None):
    ids = list(base)
    assert set(arm) == set(ids), f"{arm_run} and {base_run} are not on the same questions"

    def col(rows, fn):
        return [float(fn(rows[i], i)) for i in ids]
    out = {"base": base_run, "arm": arm_run, "n": len(ids)}
    scores = {"em": lambda r, i: r["em"], "f1": lambda r, i: r["f1"],
              "f1_50": lambda r, i: correct_by(r["answer_pred"], gold[i][0], "f1_50"),
              "contain": lambda r, i: correct_by(r["answer_pred"], gold[i][0], "contain")}
    for name, fn in scores.items():
        a, b = col(base, fn), col(arm, fn)
        d, lo, hi = paired_bootstrap(a, b)
        out.update({f"{name}_base": mean(a), f"{name}_arm": mean(b),
                    f"d_{name}": d, f"d_{name}_lo": lo, f"d_{name}_hi": hi})
        if name == "em":
            out["lost"], out["won"], out["p"] = mcnemar(a, b)
    for qt in sorted({base[i]["qtype"] for i in ids}):
        qs = [i for i in ids if base[i]["qtype"] == qt]
        a = [float(base[i]["em"]) for i in qs]
        b = [float(arm[i]["em"]) for i in qs]
        d, lo, hi = paired_bootstrap(a, b)
        out.update({f"n_{qt}": len(qs), f"d_em_{qt}": d, f"d_em_{qt}_lo": lo,
                    f"d_em_{qt}_hi": hi, f"p_{qt}": mcnemar(a, b)[2]})
    for field in ("gen_calls", "latency_s"):
        out[f"{field}_base"] = mean(float(base[i][field]) for i in ids)
        out[f"{field}_arm"] = mean(float(arm[i][field]) for i in ids)
    subs = {i: [s["sub_answer"] for s in (arm_traces or {}).get(i, {}).get("steps", [])
                if "sub_answer" in s] for i in ids}
    if any(subs.values()):
        won = [i for i in ids if float(arm[i]["em"]) and not float(base[i]["em"])]
        lost = [i for i in ids if float(base[i]["em"]) and not float(arm[i]["em"])]
        out["won_sub_right"] = sum(any(em(a, g) for a in subs[i] for g in gold[i][0])
                                   for i in won)
        out["lost_is_sub"] = sum(any(normalize(a) == normalize(arm[i]["answer_pred"])
                                     for a in subs[i]) for i in lost)
    return out


def write(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="runs for the losses table")
    ap.add_argument("--compare", action="append", default=[],
                    help="base=arm1,arm2,...; Holm within each group")
    ap.add_argument("--out", required=True, help="results/<out>_{losses,paired}.csv")
    ap.add_argument("--data", nargs="+", default=["data/processed/hotpotqa_dev.jsonl"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    groups = [(g.split("=")[0], g.split("=")[1].split(",")) for g in args.compare]
    names = list(args.runs)
    for base, arms in groups:
        names += [r for r in [base] + arms if r not in names]
    runs = {r: load_run(args.results, r) for r in names}
    gold = load_gold(args.data, {i for rows, _ in runs.values() for i in rows})

    table = [losses(r, *runs[r], gold) for r in args.runs]
    if table:
        write(Path(args.results) / f"{args.out}_losses.csv", table)
        for t in table:
            print(f"{t['run']}: n {t['n']} em {t['em']:.3f} wrong {t['wrong']}: "
                  + " ".join(f"{p} {t[p]}" for p in PLACES)
                  + f" | yes/no reply {t['reply_yesno']} inside {t['reply_inside_gold']} "
                    f"around {t['reply_around_gold']} other {t['reply_other']}"
                  + (f" | sub right {t['wrong_any_sub_right']}" if "wrong_any_sub_right" in t else "")
                  + (f" | guard final {t['guard_final']} right {t['guard_final_right']}"
                     if "guard_final" in t else ""))
    pairs = []
    for base, arms in groups:
        group = [paired(base, a, runs[base][0], runs[a][0], gold, runs[a][1]) for a in arms]
        for row, ph in zip(group, holm([r["p"] for r in group])):
            row["p_holm"] = ph
            print(f"{row['arm']} - {row['base']}: em {row['em_arm']:.3f} vs {row['em_base']:.3f} "
                  f"d {row['d_em']:+.3f} [{row['d_em_lo']:+.3f}, {row['d_em_hi']:+.3f}] "
                  f"{row['lost']}/{row['won']} p {row['p']:.3f} holm {ph:.3f} | f1 {row['d_f1']:+.3f} "
                  f"contain {row['d_contain']:+.3f} | calls {row['gen_calls_arm']:.1f} vs "
                  f"{row['gen_calls_base']:.1f}")
        pairs += group
    if pairs:
        write(Path(args.results) / f"{args.out}_paired.csv", pairs)


if __name__ == "__main__":
    main()
