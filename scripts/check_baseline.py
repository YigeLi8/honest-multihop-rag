#!/usr/bin/env python3
"""Does experience logging leave a run untouched?

    python scripts/check_baseline.py --config configs/bm25.yaml --n 100 --backend none

Runs src.eval.run_qa twice on the same questions into a temporary results
directory: once with memory off, once with memory logging on and a bm25 shadow
arm. Then compares what the two runs wrote:

    trace    byte for byte
    csv      every column except the wall-clock ones (latency_s, tokens_per_s,
             peak_memory_mb, ttft_s)
    per_hop  byte for byte

The run names differ, so files are matched by kind and compared by content;
none of the three carries the run name. It also checks that the logging run
wrote one experience record per (question, hop) and the other run wrote none.

The timing columns are left out of the comparison. The pipeline keeps the time
spent in the logging path out of latency_s, but a logged run is still not the
place to take systems numbers from: use a run with memory off for those.

--backend none needs a single_hop config and runs retrieval only (no model,
no GPU). With a generation backend the answers are compared too, which is
only meaningful at temperature 0.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TIMING = ("latency_s", "tokens_per_s", "peak_memory_mb", "ttft_s")


def write_config(src, dst, name, results_dir, memory):
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("run", {})["name"] = name
    cfg.setdefault("paths", {})["results_dir"] = str(results_dir)
    cfg["memory"] = memory
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f)


def run_qa(config, args):
    cmd = [sys.executable, "-m", "src.eval.run_qa", "--config", str(config)]
    if args.ids:
        cmd += ["--ids", str(Path(args.ids).resolve())]
    else:
        cmd += ["--n", str(args.n)]
    if args.backend:
        cmd += ["--backend", args.backend]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-4000:], sep="\n")
        raise SystemExit(f"run_qa failed: {' '.join(cmd)}")


def csv_without_timing(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at row {i + 1}:\n      off: {x}\n      log: {y}"
    return f"lengths differ: {len(a)} vs {len(b)}"


def report(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def compare(results, off, log, store):
    ok = True

    a = (results / f"{off}_trace.jsonl").read_bytes()
    b = (results / f"{log}_trace.jsonl").read_bytes()
    n_questions = a.count(b"\n")
    ok &= report("trace", a == b, f"{n_questions} questions, byte for byte" if a == b else
                 first_difference(a.splitlines(), b.splitlines()))

    a = csv_without_timing(results / f"{off}_qa.csv")
    b = csv_without_timing(results / f"{log}_qa.csv")
    ok &= report("csv", a == b, f"{len(a)} rows, {len(a[0])} columns, timing dropped"
                 if a == b else first_difference(a, b))

    a, b = results / f"{off}_per_hop.json", results / f"{log}_per_hop.json"
    if a.exists() or b.exists():
        same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        ok &= report("per_hop", same, "byte for byte" if same else "files differ")
    else:
        report("per_hop", True, "not written by this pipeline mode")

    # the log itself: there for the logging run only, one record per traced hop
    with open(results / f"{log}_trace.jsonl") as f:
        traces = [json.loads(line) for line in f]
    hops = {(t["id"], h["hop"]): h["retrieved"] for t in traces for h in t["hops"]}
    log_path = store / f"{log}_experience.jsonl"
    if not log_path.exists() or (store / f"{off}_experience.jsonl").exists():
        return report("experience log", False, "missing, or written with memory off") and ok
    with open(log_path) as f:
        header, *records = [json.loads(line) for line in f]
    logged = {(r["id"], r["hop"]): [h["chunk_id"] for h in r["retrieved"]] for r in records}
    same = bool(header.get("header")) and logged == hops and len(records) == len(hops)
    ok &= report("experience log", same,
                 f"{len(records)} records for {len(hops)} traced hops"
                 + ("" if same else ", ids differ from the trace"))
    cfg = header["config"]
    if cfg["retrieval"]["method"] == "bm25" and not cfg["rerank"]["enabled"]:
        same = all(r["shadow"]["bm25"]["retrieved"] == logged[(r["id"], r["hop"])]
                   and r["probe_features"]["bm25"]["topk_jaccard"] == 1.0
                   and r["shadow"]["bm25"]["gold_hit_ids"] == r["hop_outcome"]["gold_hit_ids"]
                   for r in records)
        ok &= report("bm25 shadow == primary", same)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ids", default=None, help="fold file instead of the first --n")
    ap.add_argument("--backend", default=None, help="override generation backend")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="check_baseline_"))
    results, store = tmp / "results", tmp / "memory"
    results.mkdir()
    try:
        off, log = "check_memoff", "check_memlog"
        write_config(args.config, tmp / "off.yaml", off, results, {"enabled": False})
        write_config(args.config, tmp / "log.yaml", log, results,
                     {"enabled": True, "mode": "log", "log_arms": ["bm25"],
                      "store": str(store)})
        print(f"{args.config}: memory off ...")
        run_qa(tmp / "off.yaml", args)
        print(f"{args.config}: memory logging, shadow arm bm25 ...")
        run_qa(tmp / "log.yaml", args)
        ok = compare(results, off, log, store)
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("baseline unchanged by logging" if ok else "logging changed the run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
