#!/usr/bin/env python3
"""What the old sweep's tok/s was, on two corners re-run with the streaming runner.

    python scripts/serving_corner.py

results/sweep_metal.csv (16 Aug) was made with the first mlx runner, whose
tokens_per_s was answer tokens over the wall time of the whole generate call,
prefill included. With prompts of a few hundred tokens and ~5-token answers
that is mostly prefill time. The runner now reports ttft_s (prefill and the
first sampling step), prompt_tokens and the decode rate separately, so the
corners in results/serving_corner/ (same retrieval, same first 100 hotpot dev
questions, written by run_qa from the yaml next to them) can be read both ways.

Two things about the logged decode rate: mlx-lm counts the first token in the
numerator but starts its clock after it, so on an n-token answer the logged
tokens_per_s is n / (time of n-1 steps), EOS included in n; and the answers
are a handful of tokens. The per-step rate below is (n-1) / that time. The old
definition is rebuilt as (n-1) / (ttft + decode time): n-1 because the old
count had no EOS.

Every rate is a mean of per-question values, as the old sweep's was. latency_s
is the pipeline's: per-question indexing and retrieval, then generation; the
first question also pays the model loads, so it is left out of the split.
Single-call runs only: with ircot the token count is summed over calls and the
rate is the last call's.
"""
import csv
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
D = RESULTS / "serving_corner"
# name -> the sweep_metal.csv row it repeats (rerank off)
CORNERS = {"corner_8bit_k20": ("8bit", "20"), "corner_4bit_k5": ("4bit", "5")}
FIELDS = ["corner", "n", "em", "retrieval_correct", "prompt_tokens", "completion_tokens",
          "ttft_s", "decode_s", "other_s", "latency_s",
          "decode_tok_s_per_step", "decode_tok_s_logged", "old_definition_tok_s",
          "sweep_tok_s", "sweep_latency_s", "sweep_em", "sweep_retrieval_correct", "peak_memory_mb"]


def sweep_row(quant, top_k):
    with open(RESULTS / "sweep_metal.csv", newline="") as f:
        return next(r for r in csv.DictReader(f)
                    if (r["quant"], r["top_k"], r["rerank"]) == (quant, top_k, "False"))


def summarise(name):
    with open(D / f"{name}_qa.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(int(r["gen_calls"]) != 1 for r in rows):
        raise SystemExit(f"{name}: more than one generation call per question")
    col = lambda k, rs=rows: [float(r[k]) for r in rs]
    tps, ttft, ct = col("tokens_per_s"), col("ttft_s"), col("reasoning_tokens")
    decode = [c / p if p else 0.0 for c, p in zip(ct, tps)]            # s from the first token to the end
    step = [(c - 1) / d for c, d in zip(ct, decode) if c > 1 and d > 0]
    old = [(c - 1) / (t + d) for c, t, d in zip(ct, ttft, decode) if t + d > 0]
    warm = slice(1, None)                                              # without the model-load question
    lat = col("latency_s")[warm]
    old_row = sweep_row(*CORNERS[name])
    return {
        "corner": name, "n": len(rows),
        "em": round(st.mean(col("em")), 3), "retrieval_correct": round(st.mean(col("retrieval_correct")), 3),
        "prompt_tokens": round(st.mean(col("prompt_tokens")), 1), "completion_tokens": round(st.mean(ct), 2),
        "ttft_s": round(st.mean(ttft[warm]), 3), "decode_s": round(st.mean(decode[warm]), 3),
        "other_s": round(st.mean(lat) - st.mean(ttft[warm]) - st.mean(decode[warm]), 3),
        "latency_s": round(st.mean(lat), 3),
        "decode_tok_s_per_step": round(st.mean(step), 1), "decode_tok_s_logged": round(st.mean(tps), 1),
        "old_definition_tok_s": round(st.mean(old), 1),
        "sweep_tok_s": old_row["tokens_per_s"], "sweep_latency_s": old_row["latency_s"],
        "sweep_em": old_row["em"], "sweep_retrieval_correct": old_row["retrieval_correct"],
        "peak_memory_mb": round(max(col("peak_memory_mb")), 1),
    }


def main():
    out = [summarise(name) for name in CORNERS if (D / f"{name}_qa.csv").exists()]
    for r in out:
        print(r["corner"])
        for key in FIELDS[1:]:
            print(f"  {key:26s} {r[key]}")
    with open(D / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"csv: {D / 'summary.csv'}")


if __name__ == "__main__":
    main()
