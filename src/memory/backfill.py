"""Rebuild experience records from a finished run's trace, without generation.

    python -m src.memory.backfill --run bm25_hotpot_dev --config configs/bm25.yaml \
        --data-file data/processed/hotpotqa_dev.jsonl

For every question in results/<run>_trace.jsonl the retriever is rebuilt from
the config, the question's pool is indexed again and each hop's logged query
is replayed through the pipeline's own retrieval step, so the records are the
ones the live logger would have written. The recomputed chunk ids must equal
the trace at every hop: a mismatch means the config, the data file or the
retriever changed since the run, and the records would describe a retrieval
that never happened. Mismatches are counted and the exit code is non-zero
whenever there are any; without --allow-mismatch nothing is written, with it
the questions with a mismatching hop are left out and only the ones that
reproduce are written, and the exit code still says that some were.

The question-level outcome comes from results/<run>_qa.csv, joined by id.
Columns an older csv does not have (para_recall, stop_reason) are recomputed
from the replayed hits or read from the trace. The header is marked
source: backfill and its config is the yaml as it resolves now, so overrides
given on the original command line (--backend, --model) are not in it: the
generation block there does not describe the run, the csv named in
outcome_source does.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.config import load_config
from src.eval.run_qa import load_examples, read_ids, score_row
from src.pipeline.multihop import MultiHopPipeline
from src.types import Example, HopTrace, PipelineResult

# csv values are strings; these are the types score_row gives the same fields
CSV_TYPES = {"em": float, "f1": float, "retrieval_correct": int, "gold_recall": float,
             "para_recall": float, "stop_reason": str, "hops": int}


def read_trace(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_rows(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def outcome_row(ex: Example, res: PipelineResult, csv_row: Mapping[str, str]) -> dict:
    """The row run_qa scored for this question: recomputed from the replay,
    then overwritten by whatever the csv recorded at the time."""
    row = score_row(ex, res)
    for key, cast in CSV_TYPES.items():
        if key in csv_row and (csv_row[key] != "" or cast is str):
            row[key] = cast(csv_row[key])
    return row


def replay(pipe: MultiHopPipeline, examples: Sequence[Example], traces: Sequence[dict],
           rows: Mapping[str, Mapping[str, str]]) -> tuple[int, list[dict], int]:
    """Replay every traced hop through pipe (which carries the logger).

    A question with a mismatching hop gets no records: what the replay
    retrieved is not what the csv row scored, and the two must not be mixed.

    Returns (hops replayed, mismatching hops, questions whose csv
    retrieval_correct disagrees with the replay)."""
    n_hops, mismatches, csv_disagree = 0, [], 0
    for ex, t in zip(examples, traces):
        if t["id"] not in rows:
            raise SystemExit(f"{t['id']} is in the trace but not in the csv")
        pipe._index(ex)
        hops, reproduced = [], True
        for h in t["hops"]:
            hits = pipe._retrieve(h["query"], hop=h["hop"], ex=ex)
            got = [rc.chunk.chunk_id for rc in hits]
            n_hops += 1
            if got != h["retrieved"]:
                reproduced = False
                mismatches.append({"id": ex.id, "hop": h["hop"],
                                   "trace": h["retrieved"], "recomputed": got})
            hops.append(HopTrace(hop=h["hop"], query=h["query"], retrieved=hits,
                                 gold_chunk_ids=list(ex.gold_chunk_ids)))
        if not reproduced:
            pipe.memory.discard(ex.id)
            continue
        res = PipelineResult(example_id=ex.id, answer=t.get("answer_pred", ""), hops=hops,
                             raw_outputs=t.get("raw_outputs", []),
                             stop_reason=t.get("stop_reason", ""))
        replayed = score_row(ex, res)
        row = outcome_row(ex, res, rows[ex.id])
        csv_disagree += row["retrieval_correct"] != replayed["retrieval_correct"]
        pipe.memory.on_outcome(ex, res, row)
    return n_hops, mismatches, csv_disagree


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, as in results/<run>_trace.jsonl")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--ids", default=None, help="fold file; only these questions of the trace")
    ap.add_argument("--n", type=int, default=None, help="only the first n questions of the trace")
    ap.add_argument("--out", default=None,
                    help="where to write (default <memory.store>/<run>_experience.jsonl)")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write the questions that reproduce even if some hops do not")
    args = ap.parse_args(argv)

    cfg: Any = load_config(args.config)
    if cfg.pipeline.mode == "closed_book":
        raise SystemExit("closed_book runs retrieve nothing, there is nothing to backfill")
    cfg.run.name = args.run
    cfg.memory.enabled, cfg.memory.mode = True, "log"

    results_dir = Path(cfg.paths.results_dir)
    traces = read_trace(results_dir / f"{args.run}_trace.jsonl")
    if args.ids:
        want = set(read_ids(args.ids))
        traces = [t for t in traces if t["id"] in want]
    if args.n:
        traces = traces[:args.n]
    rows = read_rows(results_dir / f"{args.run}_qa.csv")
    examples = load_examples(args.data_file, ids=[t["id"] for t in traces])

    pipe = MultiHopPipeline(cfg)
    pipe.memory.source = "backfill"
    pipe.memory.describe_data(args.data_file, len(examples), ids_file=args.ids, n=args.n)
    if args.out:
        pipe.memory.path = Path(args.out)
    n_hops, mismatches, csv_disagree = replay(pipe, examples, traces, rows)

    print(f"{args.run}: {len(traces)} questions, {n_hops} hops replayed, "
          f"{len(mismatches)} hops differ from the trace")
    for m in mismatches[:5]:
        print(f"  {m['id']} hop {m['hop']}\n    trace      {m['trace']}\n"
              f"    recomputed {m['recomputed']}")
    if csv_disagree:
        print(f"warning: retrieval_correct in the csv differs from the replay on "
              f"{csv_disagree} questions (gold mapping or data file changed since the run?)")
    if mismatches and not args.allow_mismatch:
        print("not writing records; fix the config / data file or pass --allow-mismatch")
        raise SystemExit(1)

    pipe.memory.extra_header = {
        "mismatching_hops": len(mismatches),
        "questions_dropped": len({m["id"] for m in mismatches}),
        "outcome_source": str(results_dir / f"{args.run}_qa.csv"),
        "config_source": f"{args.config} as it resolves now, not as the run saw it",
    }
    path = pipe.memory.close()
    print(f"experience: {path} ({len(pipe.memory.records())} records)")
    return len(mismatches)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
