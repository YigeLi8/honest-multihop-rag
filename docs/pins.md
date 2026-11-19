# Pins

What the committed results were made with, as far as it can be checked, and
what is still to be read off the Mac. `requirements.txt` stays the Mac's
install list; `requirements-linux-cpu.txt` is the frozen environment the
no-model stages were re-run in on linux (19 Nov). `results/manifest.csv`
says which command, config and fold each results file comes from
(`scripts/manifest.py`, below).

## Software

Generation, the agents' traces and every `*_qa.csv` / `*_trace.jsonl` /
`*_per_hop.json` come from the Mac (arm64, Python 3.12, Metal). The tables
written from them (the `analyses` stage of `scripts/run_all.sh`) and the
folds were re-made on linux from a fresh clone.

| package               | Mac (the runs)          | linux, no-model stages (19 Nov) |
|-----------------------|-------------------------|---------------------------------|
| python                | 3.12 (minor: to fill)   | 3.12.3                          |
| mlx                   | to fill                 | not installed                   |
| mlx-lm                | to fill                 | not installed                   |
| bm25s                 | 0.3.10 (requirements)   | 0.3.10 (0.3.16 tried, below)    |
| torch                 | to fill                 | 2.15.0 (cpu)                    |
| sentence-transformers | to fill                 | 6.3.0                           |
| transformers          | to fill                 | 5.23.0                          |
| faiss-cpu             | to fill                 | 1.16.0                          |
| numpy                 | to fill                 | 2.5.5                           |
| pandas                | to fill                 | 3.1.1                           |
| matplotlib            | to fill                 | 3.11.2                          |
| datasets              | to fill                 | 5.2.0                           |
| huggingface_hub       | to fill                 | 2.4.0                           |

To fill on the Mac, in the venv the runs use:

    pip freeze > requirements-mac.txt
    python -c "import platform; print(platform.python_version())"

bm25s orders tied scores differently across versions (0.3.10 on the Mac
against 0.3.11 / 0.3.12 earlier on linux), which moves retrieved lists and
so every downstream file; it is the one retrieval pin that matters. On
linux it is not enough: the retrieval stage rewrites
`results/bm25_hotpot_dev_retrieval.csv` with 16 of 7405 rows changed
(retrieval_correct is 1 on 6 of them, against 3 in the committed table) and `bm25_musique_dev_retrieval.csv`
with 1 of 2417, the same rows with bm25s 0.3.10 and 0.3.16, so the
difference is the platform's (numpy, the index build) and not the bm25s
version. Those two tables are the Mac's. Every other stage without the
model reads only committed files and `data/processed`, and with the linux
versions above rewrites its committed tables byte-identical (the 72 tables
of the analyses stage and the 5 folds, 19 Nov); the 4 decomposition plots
and the 105 redrawn by the plots stage differ in bytes, from the
matplotlib build.

## Models

| role                     | id                                    | hub revision on 19 Nov                   | last modified on the hub |
|--------------------------|---------------------------------------|------------------------------------------|--------------------------|
| generator (every run)    | mlx-community/Qwen2.5-7B-Instruct-4bit | c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed | 2024-11-06               |
| dense retriever          | BAAI/bge-small-en-v1.5                 | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | 2024-02-22               |
| reranker (rerank_bge)    | BAAI/bge-reranker-base                 | 2cfc18c9415c912f9d8155881c133215df768a70 | 2024-06-24               |

The revision column is what the hub serves today, not what the Mac
downloaded. None of the three has changed on the hub since before this
repository's first run, so they are probably the same, but only the Mac's
cache can say. To fill:

    for m in mlx-community--Qwen2.5-7B-Instruct-4bit BAAI--bge-small-en-v1.5 BAAI--bge-reranker-base; do
      echo "$m $(cat ~/.cache/huggingface/hub/models--$m/refs/main)"; done

Generation is greedy (temperature 0.0, max_tokens 512, `configs/base.yaml`);
the round 1 and 2 notes in `docs/plan.md` found reruns identical
completion for completion on the questions a change does not touch.

## Data

| file (data/raw)                      | source                                        | bytes       | sha256 (19 Nov download) |
|--------------------------------------|-----------------------------------------------|-------------|--------------------------|
| hotpot_dev_distractor_v1.json        | curtis.ml.cmu.edu/datasets/hotpot             | 46,818,692  | a9b4c5906587ac0d20eb83dcb29d0240e268e342c170f7f22988f59f13fa82c3 |
| hotpot_train_v1.1.json               | same host, hf mirror as fallback              | 566,426,227 | a07f9457cc4a87f921203972b09fb88284bc0514b8f0d4963b2d8be6775910c9 |
| musique/musique_ans_v1.0_dev.jsonl   | hf dataset bdsaglam/musique @ 22873a40        | 30,439,728  | 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b |
| musique/musique_full_v1.0_dev.jsonl  | same                                          | 59,422,562  | 8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a |
| musique/musique_ans_v1.0_train.jsonl | same                                          | 241,046,755 | 83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a |
| 2wiki/dev.json                       | data_ids_april7.zip, by hand                  | to fill     | to fill on the Mac       |

| file (data/processed, made here)    | bytes       | sha256 |
|-------------------------------------|-------------|--------|
| hotpotqa_dev.jsonl                  | 75,765,764  | 46c30a73704b2d8af6c89e926f18a9a1dddbe5245fa0abc4bb3116be71aa1fc9 |
| hotpotqa_train.jsonl                | 915,592,810 | 1b9fa623ed45ad17b2de36e757436e28a60f28f477ed4b39f8e0773a2c752bc3 |
| musique_dev.jsonl                   | 31,645,560  | 840a435326a570eab3e79671a5d246ab0b405b6244cbda11755c7a63f35a6c32 |

The processed files are shuffled with seed 13 and the folds drawn from them
(`data.folds`, seed 13; train_hard seed 17); the committed folds under
`results/folds/` are what every run read, and the data stage of
`run_all.sh` fails if it does not rewrite them byte-identical. To check the
Mac's copies against these: `shasum -a 256 data/raw/*.json data/raw/musique/*
data/processed/*.jsonl`.

## Which command made which file

    scripts/run_all.sh --no-model --record rec.tsv     # runs the stages without the model, logs every write
    python scripts/manifest.py --record rec.tsv         # -> results/manifest.csv

A file written by a stage without the model is credited to the command that
was seen writing it (`how = observed`). A [model] stage is not run here: its
commands are read from `scripts/run_all.sh --dry-run` and credited with the
files their run name gives (`how = by name`). Committed files that no
command in `run_all.sh` writes are listed with `how = not in run_all.sh`.
On the Mac, `scripts/run_all.sh --model-only --record rec_mac.tsv` turns the
model rows into observed ones.
