"""Stop audit of the ircot loop: at which hop did each gold sentence enter the
context, why did the loop stop, and was the answer right.

    python -m src.eval.stop_audit --dataset hotpot \
        --runs hybrid_ircot_hotpot_dev hybrid_ircot_hotpot_confirm \
        --data-file data/processed/hotpotqa_dev.jsonl

Reads results/<run>_trace.jsonl and results/<run>_qa.csv; nothing is re-run.
The loop in src/pipeline/multihop.py stops when the model writes the done
marker ("done"), writes no marker ("no_marker"), repeats a query
("repeated_query") or runs out of hops ("max_hops"). The question here is
whether those stops are informed: does the model say done once the gold is in
context and search on while it is not, or does the stop reason have nothing
to do with what has been retrieved. Three cuts of the same per-question
table answer it:

* arrival: for every gold sentence the hop at which it first entered the
  context (never, if it did not), and for the answer-bearing gold sentence
  (the one that contains the gold answer; none for yes/no questions) the
  same. hop_complete is the hop at which the last gold sentence arrived.
* the stop against the arrival: a stop is premature when the model itself
  stopped the loop (done, no_marker, repeated_query) with gold still
  missing, and it is late by the number of hops the loop went on after the
  gold was complete (surplus_hops). A max_hops stop with the gold complete
  since hop 0 is a loop that never noticed it was done.
* correctness within each cell: EM where the stop was premature is the part
  of the illusion cell the loop made; EM where the gold was complete at the
  stop is what the loop earned.

Writes results/<dataset>_stop_audit.csv (one row per question and run) and
results/<dataset>_stop_audit_summary.csv (one row per run and stop reason,
plus an "all" row per run). Descriptive, with Wilson intervals on the rates;
no model and no test, this is an audit of existing traces.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.interventions import answer_strings, contains_answer
from src.eval.stats import wilson_ci
from src.types import example_from_json

MODEL_STOPS = ("done", "no_marker", "repeated_query")
NEVER = -1


def load_examples(path):
    out = {}
    with open(path) as f:
        for line in f:
            ex = example_from_json(json.loads(line))
            out[ex.id] = ex
    return out


def read_traces(path):
    with open(path) as f:
        return {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}


def first_hops(trace, gold_ids):
    """gold chunk id -> hop at which it first appeared in the retrieved list,
    NEVER when it did not. Context is the union of every hop's list, so a
    chunk is in context from the hop it was first retrieved on."""
    first = {}
    for h in trace["hops"]:
        for cid in h["retrieved"]:
            if cid in gold_ids and cid not in first:
                first[cid] = h["hop"]
    return {g: first.get(g, NEVER) for g in gold_ids}


def answer_chunks(ex):
    """Gold chunks whose text contains the gold answer (or an alias); empty for
    a yes/no question, or when no gold sentence carries the answer string."""
    answers = answer_strings(ex)
    if not answers:
        return []
    gold = set(ex.gold_chunk_ids)
    return [c.chunk_id for c in ex.chunks if c.chunk_id in gold and contains_answer(c.text, answers)]


def audit_question(ex, trace, qa_row):
    gold = list(dict.fromkeys(ex.gold_chunk_ids))
    arrival = first_hops(trace, set(gold))
    hops = len(trace["hops"])
    stop = trace.get("stop_reason") or qa_row.get("stop_reason") or ""
    seen = [h for h in arrival.values() if h != NEVER]
    complete = len(seen) == len(gold) and bool(gold)
    hop_complete = max(seen) if complete else NEVER
    ans = answer_chunks(ex)
    ans_hops = [arrival[c] for c in ans if arrival[c] != NEVER]
    answer_hop = min(ans_hops) if ans_hops else NEVER
    model_stop = stop in MODEL_STOPS
    return {
        "id": ex.id,
        "qtype": ex.qtype,
        "stop_reason": stop,
        "hops": hops,
        "em": int(float(qa_row["em"]) >= 1.0),
        "retrieval_correct": int(qa_row["retrieval_correct"]),
        "n_gold": len(gold),
        "n_gold_seen": len(seen),
        "gold_first_hops": ";".join(f"{g}={arrival[g]}" for g in gold),
        "complete": int(complete),
        "hop_complete": hop_complete,
        "complete_at_hop0": int(complete and hop_complete == 0),
        "gold_after_hop0": int(any(h > 0 for h in seen)),
        "answer_sentence": int(bool(ans)),
        "answer_hop": answer_hop,
        "model_stop": int(model_stop),
        "premature": int(model_stop and not complete),
        "surplus_hops": hops - 1 - hop_complete if complete else NEVER,
        "late_gold": int(complete and hop_complete == hops - 1 and hops > 1),
    }


def audit_run(run, examples, results_dir="results"):
    results = Path(results_dir)
    traces = read_traces(results / f"{run}_trace.jsonl")
    qa = pd.read_csv(results / f"{run}_qa.csv", keep_default_na=False)
    rows = []
    for qa_row in qa.to_dict("records"):
        qid = qa_row["id"]
        if qid not in traces or qid not in examples:
            raise SystemExit(f"{run}: {qid} missing from the trace or the data file")
        row = audit_question(examples[qid], traces[qid], qa_row)
        if row["complete"] != row["retrieval_correct"]:
            raise SystemExit(f"{run}: {qid} gold arrival disagrees with retrieval_correct")
        rows.append({"run": run, **row})
    return pd.DataFrame(rows)


def _rate(col):
    k, n = int(col.sum()), len(col)
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    return round(k / n, 4) if n else float("nan"), round(lo, 4), round(hi, 4)


def summary_rows(audit):
    """Per run and stop reason (and 'all'): how many, how the loop ended
    against where the gold was, and EM inside each cell."""
    rows = []
    for run, by_run in audit.groupby("run", sort=False):
        groups = [("all", by_run)] + [(r, g) for r, g in by_run.groupby("stop_reason", sort=False)]
        for reason, g in groups:
            n = len(g)
            complete = g[g["complete"] == 1]
            premature = g[g["premature"] == 1]
            with_answer = g[g["answer_sentence"] == 1]
            em, em_lo, em_hi = _rate(g["em"])
            row = {
                "run": run, "stop_reason": reason, "n": n,
                "share": round(n / len(by_run), 4),
                "em": em, "em_lo": em_lo, "em_hi": em_hi,
                "retrieval_correct": _rate(g["complete"])[0],
                "mean_hops": round(g["hops"].mean(), 3),
                "complete_at_hop0": _rate(g["complete_at_hop0"])[0],
                "complete_later": round(((g["complete"] == 1) & (g["complete_at_hop0"] == 0)).mean(), 4),
                "never_complete": round((g["complete"] == 0).mean(), 4),
                "gold_after_hop0": _rate(g["gold_after_hop0"])[0],
                "premature": _rate(g["premature"])[0],
                "em_premature": _rate(premature["em"])[0] if len(premature) else float("nan"),
                "em_complete": _rate(complete["em"])[0] if len(complete) else float("nan"),
                "surplus_mean": round(complete["surplus_hops"].mean(), 3) if len(complete) else float("nan"),
                "surplus_any": round((complete["surplus_hops"] > 0).mean(), 4) if len(complete) else float("nan"),
                "late_gold": _rate(g["late_gold"])[0],
                "answer_never": round((with_answer["answer_hop"] == NEVER).mean(), 4)
                if len(with_answer) else float("nan"),
                "answer_at_hop0": round((with_answer["answer_hop"] == 0).mean(), 4)
                if len(with_answer) else float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label for the output files")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-file", required=True, help="processed jsonl with gold_chunk_ids")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)

    examples = load_examples(args.data_file)
    audit = pd.concat([audit_run(r, examples, args.results_dir) for r in args.runs],
                      ignore_index=True)
    summary = summary_rows(audit)
    out = Path(args.results_dir)
    audit.to_csv(out / f"{args.dataset}_stop_audit.csv", index=False)
    summary.to_csv(out / f"{args.dataset}_stop_audit_summary.csv", index=False)
    cols = ["run", "stop_reason", "n", "share", "em", "retrieval_correct", "mean_hops",
            "complete_at_hop0", "complete_later", "never_complete", "premature",
            "em_premature", "em_complete", "surplus_mean", "late_gold", "answer_never"]
    print(summary[cols].to_string(index=False))
    print(f"wrote {out / f'{args.dataset}_stop_audit.csv'} and the summary")


if __name__ == "__main__":
    main()
