#!/usr/bin/env bash
# Everything the committed results come from, stage by stage, from a fresh clone.
#
#   scripts/run_all.sh                    every stage (the model stages need the Mac)
#   scripts/run_all.sh --no-model         only the stages that need no generation
#   scripts/run_all.sh --model-only       only the generation stages
#   scripts/run_all.sh --only analyses,plots
#   scripts/run_all.sh --smoke [...]      a few questions per stage, written to a
#                                         scratch results dir, never to results/
#   scripts/run_all.sh --dry-run [...]    print each python step (stage, kind, command),
#                                         run nothing
#   scripts/run_all.sh --record FILE [...] run, and append to FILE every file each
#                                         python step wrote (scripts/manifest.py reads it)
#
# Stages, in order. [model] stages generate with the local model (MLX, Metal)
# and write results/<run>_{qa.csv,trace.jsonl,per_hop.json,2x2.png,...}; the
# others only read data/processed and committed results/ files.
#
#   data           downloads, processed jsonl, folds                     no model
#   retrieval      retrieval-only evals (bm25, dense, hybrid)             no model
#   qa             Part A: retrieval + answer runs by config and fold     [model]
#   interventions  Part A: gold / answer-only / bridge-only / distractor  [model]
#   reader_ctx     Stage 5c: what the reader is handed                    [model]
#   agents         Stage 6 and 6b: the agents, the memory gates, rounds   [model]
#   analyses       decomposition, criteria, stop audit, prereg, agent     no model
#                  losses and traces; rewrites the committed tables in place
#   memory         Stage 1b-5: experience logs rebuilt from the committed no model
#                  traces (src.memory.backfill), then the memory tables
#   plots          2x2 and per-hop plots redrawn from committed files     no model
#
# 2wiki needs data_ids_april7.zip unpacked into data/raw/2wiki/ by hand; every
# 2wiki step is skipped when data/processed/2wiki_dev.jsonl is missing.
#
# After a full --no-model pass, `git status --short results/` lists the tables
# that did not come out byte-identical (expected: none but the plots, whose
# bytes depend on the matplotlib build).
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
DRY=0
RECORD=""
WANT_MODEL=1
WANT_CPU=1
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --no-model) WANT_MODEL=0 ;;
    --model-only) WANT_CPU=0 ;;
    --only) ONLY="$2"; shift ;;
    --dry-run) DRY=1 ;;
    --record) RECORD="$2"; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

PYBIN=${PY:-python}
CUR_STAGE=""; CUR_KIND=""
# Every python step goes through py(). --dry-run prints it; --record runs it and
# appends one line per file it wrote under results/ (or the smoke copy), data/
# processed or the folds: stage, kind, the command, the file (tab-separated).
py() {
  if [ "$DRY" = 1 ]; then
    printf '%s\t%s\t%s\t-\n' "$CUR_STAGE" "$CUR_KIND" "python $*"; return 0
  fi
  if [ -z "$RECORD" ]; then "$PYBIN" "$@"; return; fi
  local stamp; stamp=$(mktemp "${TMPDIR:-/tmp}/run_all_stamp.XXXXXX")
  sleep 1   # mtime granularity
  "$PYBIN" "$@"
  local f rel
  while IFS= read -r f; do
    rel=${f#"$RES"/}; [ "$rel" != "$f" ] && f="results/$rel"
    printf '%s\t%s\t%s\t%s\n' "$CUR_STAGE" "$CUR_KIND" "python $*" "$f" >> "$RECORD"
  done < <(find "$RES" data/processed -type f -newer "$stamp" ! -name 'smoke_*' 2>/dev/null | sort)
  rm -f "$stamp"
}
PY=py
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}   # scripts/*.py import src
SMOKE_N=${SMOKE_N:-5}
RES=results
if [ "$SMOKE" = 1 ] && [ "$DRY" = 0 ]; then
  RES=$(mktemp -d "${TMPDIR:-/tmp}/run_all_smoke.XXXXXX")
  # analyses read and write the same dir: give them a copy, never the originals
  cp -R results/. "$RES/"
  echo "smoke: results go to $RES"
fi
HAVE_2WIKI=0
[ -f data/processed/2wiki_dev.jsonl ] && HAVE_2WIKI=1
[ "$DRY" = 1 ] && HAVE_2WIKI=1   # list the 2wiki steps too
HOTPOT=data/processed/hotpotqa_dev.jsonl
HOTPOT_TRAIN=data/processed/hotpotqa_train.jsonl
MUSIQUE=data/processed/musique_dev.jsonl
WIKI=data/processed/2wiki_dev.jsonl
F=results/folds

stage() {  # stage NAME KIND(model|cpu): should this stage run?
  local name=$1 kind=$2
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then return 1; fi
  if [ "$kind" = model ] && [ "$WANT_MODEL" = 0 ]; then return 1; fi
  if [ "$kind" = cpu ] && [ "$WANT_CPU" = 0 ]; then return 1; fi
  CUR_STAGE=$name; CUR_KIND=$kind
  [ "$DRY" = 1 ] && return 0
  echo; echo "=== $name ($kind)"
}

# cfg NAME: the config path to run with. Under --smoke, a copy whose
# paths.results_dir is the scratch dir, so nothing is written to results/.
cfg() {
  if [ "$SMOKE" = 1 ]; then
    local tmp="$RES/smoke_cfg_$1.yaml"
    { cat "configs/$1.yaml"; printf '\npaths:\n  data_dir: data/processed\n  index_dir: data/processed/index\n  results_dir: %s\n' "$RES"; } > "$tmp"
    echo "$tmp"
  else
    echo "configs/$1.yaml"
  fi
}
NSMOKE=(); [ "$SMOKE" = 1 ] && NSMOKE=(--n "$SMOKE_N")
# ids FOLD: the fold file (under --smoke, its first SMOKE_N ids)
ids() {
  if [ "$SMOKE" = 1 ]; then
    head -n "$SMOKE_N" "$F/$1.txt" > "$RES/smoke_ids_$1.txt"; echo "$RES/smoke_ids_$1.txt"
  else
    echo "$F/$1.txt"
  fi
}

# qa CONFIG FOLD NAME [SPLIT]: one answer run on a committed fold
# (under --smoke the first SMOKE_N ids of it).
qa() {
  local extra=()
  [ -n "${4:-}" ] && extra+=(--split "$4")
  $PY -m src.eval.run_qa --config "$(cfg $1)" --ids "$(ids $2)" --name "$3" ${extra[@]+"${extra[@]}"}
}

losses() { $PY -m src.eval.agent_losses --results "$RES" "$@"; }

# ---------------------------------------------------------------- data
if stage data cpu; then
  $PY -m data.download --datasets hotpotqa hotpotqa_train musique
  $PY -m data.prepare_hotpotqa
  $PY -m data.prepare_hotpotqa --split train      # train_hard and the memory's train fold
  $PY -m data.prepare_musique
  if [ "$HAVE_2WIKI" = 1 ] || [ -d data/raw/2wiki ]; then
    $PY -m data.prepare_2wiki; HAVE_2WIKI=1
  else
    echo "2wiki: no data/raw/2wiki, skipped"
  fi
  # the folds are committed; these rewrite them and must leave results/folds unchanged
  $PY -m data.folds
  $PY -m data.folds --train-hard
  [ "$DRY" = 1 ] || git diff --exit-code --stat -- results/folds || { echo "folds differ from the committed ones" >&2; exit 1; }
fi

# ---------------------------------------------------------------- retrieval
if stage retrieval cpu; then
  # the committed per-question tables (results/bm25_<ds>_dev_retrieval.csv)
  for c in bm25 bm25_musique; do
    $PY -m src.eval.run_retrieval --config "$(cfg $c)" ${NSMOKE[@]+"${NSMOKE[@]}"}
  done
  [ "$HAVE_2WIKI" = 1 ] && $PY -m src.eval.run_retrieval --config "$(cfg bm25_2wiki)" ${NSMOKE[@]+"${NSMOKE[@]}"}
  # configs/{dense,hybrid}_*_retrieval.yaml and dense_bge.yaml run the same way;
  # their tables are not committed
fi

# ---------------------------------------------------------------- qa [model]
# Exploration runs keep their first names (*_dev: dev positions 0-499, the
# same ids in the same order as the explore fold); confirmation runs read the
# confirm folds, written before they ran (docs/preregistration.md).
if stage qa model; then
  $PY scripts/check_env.py
  for fold in explore confirm; do
    sfx=$([ $fold = explore ] && echo dev || echo confirm)
    qa closed_book      hotpot_$fold closed_book_hotpot_$sfx
    qa bm25             hotpot_$fold bm25_hotpot_$sfx
    qa bm25_k13         hotpot_$fold bm25_k13_hotpot_$sfx
    qa hybrid           hotpot_$fold hybrid_ircot_hotpot_$sfx
    qa rerank_bge       hotpot_$fold hybrid_ircot_rerank_hotpot_$sfx
    qa hybrid_ircot_k20 hotpot_$fold hybrid_ircot_k20_hotpot_$sfx
    qa closed_book_musique musique_$fold closed_book_musique_$sfx
    qa bm25_musique        musique_$fold bm25_musique_$sfx
  done
  qa bm25_k7_musique     musique_confirm bm25_k7_musique_confirm
  qa hybrid_musique      musique_confirm hybrid_musique_confirm
  qa hybrid_ircot_musique musique_confirm hybrid_ircot_musique_confirm
  if [ "$HAVE_2WIKI" = 1 ]; then
    for fold in explore confirm; do
      sfx=$([ $fold = explore ] && echo dev || echo confirm)
      qa closed_book_2wiki 2wiki_$fold closed_book_2wiki_$sfx
      qa bm25_2wiki        2wiki_$fold bm25_2wiki_$sfx
    done
    qa bm25_k13_2wiki 2wiki_confirm bm25_k13_2wiki_confirm
  fi
  # experience logs with the shadow arms (Stage 1b), dev positions 0-1499
  # (the musique one is the whole answerable dev set)
  for ds in hotpot musique; do
    n=(); [ "$SMOKE" = 1 ] && n=(--n "$SMOKE_N") || { [ $ds = hotpot ] && n=(--n 1500); }
    $PY -m src.eval.run_qa --config "$(cfg bm25_arms_$ds)" ${n[@]+"${n[@]}"}
  done
  [ "$HAVE_2WIKI" = 1 ] && $PY -m src.eval.run_qa --config "$(cfg bm25_arms_2wiki)" --n "$([ $SMOKE = 1 ] && echo $SMOKE_N || echo 1500)"
fi

# ---------------------------------------------------------------- interventions [model]
if stage interventions model; then
  for ds in hotpot musique 2wiki; do
    [ $ds = 2wiki ] && [ "$HAVE_2WIKI" = 0 ] && continue
    c=$([ $ds = hotpot ] && echo bm25 || echo bm25_$ds)
    for fold in explore confirm; do
      $PY -m src.eval.interventions run --config "$(cfg $c)" \
        --ids "$(ids ${ds}_$fold)" --name interventions_${ds}_$fold
    done
  done
fi

# ---------------------------------------------------------------- reader_ctx [model]
if stage reader_ctx model; then
  $PY -m src.memory.reader_context run --config "$(cfg hybrid_ircot_k20)" \
    --trace results/hybrid_ircot_k20_hotpot_dev_trace.jsonl \
    --ids "$(ids hotpot_explore)" --name reader_ctx_hotpot_explore
  $PY -m src.memory.reader_context run --config "$(cfg hybrid_ircot_k20)" \
    --trace results/hybrid_ircot_k20_hotpot_confirm_trace.jsonl --ids "$(ids hotpot_confirm)" \
    --train-trace results/hybrid_ircot_k20_hotpot_dev_trace.jsonl --train-ids $F/hotpot_explore.txt \
    --name reader_ctx_hotpot_confirm
  # A2's control: the single agent's contexts re-read (Stage 6)
  $PY -m src.memory.reader_context run --config "$(cfg agent)" \
    --trace results/agent_hotpot_explore_trace.jsonl \
    --ids "$(ids hotpot_explore)" --name reader_ctx_agent_hotpot_explore
fi

# ---------------------------------------------------------------- agents [model]
# Stage 6 (4 Nov agents and the memory gates) and Stage 6b (rounds 1-3 and
# ircot's answer read). Exploration and train_hard only: the confirmation run
# is added here when preregistration 3 is written.
if stage agents model; then
  qa agent        hotpot_explore agent_hotpot_explore
  qa orchestrator hotpot_explore orchestrator_hotpot_explore
  # memory gates A5 (mem) and A7 (mem2): build on the train fold, then read frozen
  for v in mem mem2; do
    qa orchestrator_${v}_build      hotpot_train_mem orchestrator_${v}_build_hotpot_train train
    qa orchestrator_${v}_build_flat hotpot_train_mem orchestrator_${v}_build_flat_hotpot_train train
    for arm in region global flat shuffled; do
      qa orchestrator_${v}_$arm hotpot_explore orchestrator_${v}_${arm}_hotpot_explore
    done
  done
  # Stage 6b round 1
  for c in orchestrator_notes orchestrator_guard orchestrator_notes_guard orchestrator_s1; do
    qa $c hotpot_explore ${c}_hotpot_explore
  done
  qa hybrid_guard hotpot_explore hybrid_ircot_guard_hotpot_explore
  # round 2, round 3 and ircot's answer read, on explore and train_hard
  for c in orchestrator_guard_wide orchestrator_guard_wide_unsup orchestrator_guard_wide_titles; do
    qa $c hotpot_explore ${c}_hotpot_explore
  done
  for c in hybrid_guard_unsup hybrid_guard_titles hybrid_guard_titles_unsup; do
    qa $c hotpot_explore hybrid_ircot_${c#hybrid_}_hotpot_explore
  done
  for c in orchestrator_guard orchestrator_guard_wide orchestrator_guard_chain \
           orchestrator_guard_wide_unsup orchestrator_guard_wide_titles; do
    qa $c hotpot_train_hard ${c}_hotpot_train_hard train
  done
  for c in hybrid_guard hybrid_guard_unsup hybrid_guard_titles hybrid_guard_titles_unsup; do
    qa $c hotpot_train_hard hybrid_ircot_${c#hybrid_}_hotpot_train_hard train
  done
fi

# ---------------------------------------------------------------- analyses
if stage analyses cpu; then
  A=(--results-dir "$RES")
  # Part A: the decomposition of the cell (and its plot), the criteria ladder,
  # the stop audit, the preregistered tests, the interventions
  for fold in dev confirm; do
    tag=$([ $fold = dev ] && echo hotpot || echo hotpot_confirm)
    $PY -m src.eval.analyze --dataset $tag ${A[@]+"${A[@]}"} --data-file $HOTPOT \
      --runs bm25_hotpot_$fold bm25_k13_hotpot_$fold hybrid_ircot_hotpot_$fold hybrid_ircot_rerank_hotpot_$fold \
      --closed-book closed_book_hotpot_$fold
    $PY -m src.eval.criteria --dataset $tag ${A[@]+"${A[@]}"} --data-file $HOTPOT \
      --runs closed_book_hotpot_$fold bm25_hotpot_$fold bm25_k13_hotpot_$fold hybrid_ircot_hotpot_$fold hybrid_ircot_rerank_hotpot_$fold
  done
  $PY -m src.eval.analyze --dataset musique ${A[@]+"${A[@]}"} --data-file $MUSIQUE \
    --runs bm25_musique_dev --closed-book closed_book_musique_dev
  # the committed musique dev ladder was made without --data-file: containment
  # there is against the gold string alone, not the aliases (the confirm one has them)
  $PY -m src.eval.criteria --dataset musique ${A[@]+"${A[@]}"} --runs closed_book_musique_dev bm25_musique_dev
  $PY -m src.eval.analyze --dataset musique_confirm ${A[@]+"${A[@]}"} --data-file $MUSIQUE \
    --runs bm25_musique_confirm bm25_k7_musique_confirm hybrid_musique_confirm hybrid_ircot_musique_confirm \
    --closed-book closed_book_musique_confirm
  $PY -m src.eval.criteria --dataset musique_confirm ${A[@]+"${A[@]}"} --data-file $MUSIQUE \
    --runs closed_book_musique_confirm bm25_musique_confirm bm25_k7_musique_confirm hybrid_musique_confirm hybrid_ircot_musique_confirm
  if [ "$HAVE_2WIKI" = 1 ]; then
    $PY -m src.eval.analyze --dataset 2wiki ${A[@]+"${A[@]}"} --data-file $WIKI \
      --runs bm25_2wiki_dev --closed-book closed_book_2wiki_dev
    $PY -m src.eval.criteria --dataset 2wiki ${A[@]+"${A[@]}"} --data-file $WIKI --runs closed_book_2wiki_dev bm25_2wiki_dev
    $PY -m src.eval.analyze --dataset 2wiki_confirm ${A[@]+"${A[@]}"} --data-file $WIKI \
      --runs bm25_2wiki_confirm bm25_k13_2wiki_confirm --closed-book closed_book_2wiki_confirm
    $PY -m src.eval.criteria --dataset 2wiki_confirm ${A[@]+"${A[@]}"} --data-file $WIKI \
      --runs closed_book_2wiki_confirm bm25_2wiki_confirm bm25_k13_2wiki_confirm
  else
    echo "2wiki tables: no data, skipped"
  fi
  $PY -m src.eval.stop_audit --dataset hotpot ${A[@]+"${A[@]}"} --data-file $HOTPOT \
    --runs hybrid_ircot_hotpot_dev hybrid_ircot_hotpot_confirm hybrid_ircot_rerank_hotpot_dev hybrid_ircot_rerank_hotpot_confirm
  $PY -m src.eval.stop_audit --dataset musique ${A[@]+"${A[@]}"} --data-file $MUSIQUE --runs hybrid_ircot_musique_confirm
  if [ "$HAVE_2WIKI" = 1 ]; then
    $PY -m src.eval.prereg ${A[@]+"${A[@]}"}
  else
    echo "prereg: reads the 2wiki confirmation runs' data, skipped"
  fi
  for ds in hotpot musique 2wiki; do
    [ $ds = 2wiki ] && [ "$HAVE_2WIKI" = 0 ] && continue
    df=$([ $ds = hotpot ] && echo $HOTPOT || echo data/processed/${ds}_dev.jsonl)
    for fold in explore confirm; do
      sfx=$([ $fold = explore ] && echo dev || echo confirm)
      $PY -m src.eval.interventions summarize --name interventions_${ds}_$fold ${A[@]+"${A[@]}"} \
        --data-file $df --closed-book closed_book_${ds}_$sfx --bm25-run bm25_${ds}_$sfx
    done
  done
  $PY -m src.memory.reader_context summarize --name reader_ctx_hotpot_explore ${A[@]+"${A[@]}"} \
    --run-qa $RES/hybrid_ircot_k20_hotpot_dev_qa.csv
  $PY -m src.memory.reader_context summarize --name reader_ctx_hotpot_confirm ${A[@]+"${A[@]}"} \
    --run-qa $RES/hybrid_ircot_k20_hotpot_confirm_qa.csv
  $PY -m src.memory.reader_context summarize --name reader_ctx_agent_hotpot_explore ${A[@]+"${A[@]}"}

  # Stage 6b: losses and paired tables, by round (docs/plan.md)
  E=_hotpot_explore; T=_hotpot_train_hard
  losses --out agent_dev_hotpot_explore \
    --runs hybrid_ircot_hotpot_dev hybrid_ircot_guard$E agent$E orchestrator$E orchestrator_notes$E \
           orchestrator_guard$E orchestrator_notes_guard$E orchestrator_s1$E \
    --compare hybrid_ircot_hotpot_dev=orchestrator$E \
    --compare orchestrator$E=orchestrator_notes$E,orchestrator_guard$E,orchestrator_notes_guard$E,orchestrator_s1$E \
    --compare hybrid_ircot_hotpot_dev=hybrid_ircot_guard$E \
    --compare hybrid_ircot_guard$E=orchestrator_guard$E,orchestrator_notes_guard$E \
    --compare orchestrator_guard$E=orchestrator_notes_guard$E \
    --compare orchestrator_notes$E=orchestrator_notes_guard$E
  losses --out agent_dev2_hotpot_explore --runs orchestrator_guard$E orchestrator_guard_wide$E \
    --compare orchestrator_guard$E=orchestrator_guard_wide$E
  losses --out agent_dev2_hotpot_train_hard --data $HOTPOT_TRAIN \
    --runs orchestrator_guard$T orchestrator_guard_wide$T orchestrator_guard_chain$T \
    --compare orchestrator_guard$T=orchestrator_guard_wide$T,orchestrator_guard_chain$T
  losses --out agent_dev2_hotpot_pooled --data $HOTPOT $HOTPOT_TRAIN \
    --compare orchestrator_guard$E+orchestrator_guard$T=orchestrator_guard_wide$E+orchestrator_guard_wide$T
  losses --out agent_dev3_hotpot_explore \
    --runs orchestrator_guard_wide$E orchestrator_guard_wide_unsup$E orchestrator_guard_wide_titles$E \
           hybrid_ircot_guard$E hybrid_ircot_guard_unsup$E hybrid_ircot_guard_titles$E \
    --compare hybrid_ircot_guard$E=orchestrator_guard_wide$E \
    --compare hybrid_ircot_guard_unsup$E=orchestrator_guard_wide_unsup$E \
    --compare hybrid_ircot_guard_titles$E=orchestrator_guard_wide_titles$E \
    --compare orchestrator_guard_wide$E=orchestrator_guard_wide_unsup$E,orchestrator_guard_wide_titles$E \
    --compare hybrid_ircot_guard$E=hybrid_ircot_guard_unsup$E,hybrid_ircot_guard_titles$E
  losses --out agent_dev3_hotpot_train_hard --data $HOTPOT_TRAIN \
    --runs orchestrator_guard_wide$T orchestrator_guard_wide_unsup$T orchestrator_guard_wide_titles$T \
    --compare orchestrator_guard_wide$T=orchestrator_guard_wide_unsup$T,orchestrator_guard_wide_titles$T
  losses --out agent_dev3_hotpot_pooled --data $HOTPOT $HOTPOT_TRAIN \
    --compare orchestrator_guard_wide$E+orchestrator_guard_wide$T=orchestrator_guard_wide_unsup$E+orchestrator_guard_wide_unsup$T,orchestrator_guard_wide_titles$E+orchestrator_guard_wide_titles$T
  losses --out agent_dev3_hotpot_explore_ircot_titles \
    --compare hybrid_ircot_guard_titles$E=orchestrator_guard_wide_unsup$E,orchestrator_guard_wide$E
  # ircot's answer read: each candidate against G, then the round 3 agent against each
  AG=orchestrator_guard_wide_unsup
  for fold in explore train_hard pooled; do
    case $fold in
      explore)    s() { echo "$1$E"; }; data=(--data $HOTPOT) ;;
      train_hard) s() { echo "$1$T"; }; data=(--data $HOTPOT_TRAIN) ;;
      pooled)     s() { echo "$1$E+$1$T"; }; data=(--data $HOTPOT $HOTPOT_TRAIN) ;;
    esac
    G=$(s hybrid_ircot_guard); GU=$(s hybrid_ircot_guard_unsup)
    GT=$(s hybrid_ircot_guard_titles); GTU=$(s hybrid_ircot_guard_titles_unsup)
    runs=(); [ $fold != pooled ] && runs=(--runs $G $GU $GT $GTU)
    losses --out ircot_read_hotpot_$fold ${data[@]+"${data[@]}"} ${runs[@]+"${runs[@]}"} \
      --compare $G=$GU,$GT,$GTU \
      --compare $G=$(s $AG) --compare $GU=$(s $AG) --compare $GT=$(s $AG) --compare $GTU=$(s $AG)
    losses --out ircot_read_hotpot_${fold}_b6_on_titles ${data[@]+"${data[@]}"} --compare $GT=$GTU
  done
  # From the traces (explore fold only)
  $PY -m src.eval.agent_traces --results "$RES" --run orchestrator_guard_wide$E --out agent_traces_hotpot_explore
  $PY -m src.eval.agent_reads --results "$RES" --run orchestrator_guard_wide_unsup$E --out agent_reads_hotpot_explore
  $PY -m src.eval.agent_reads --results "$RES" --run hybrid_ircot_guard_unsup$E --out agent_reads_ircot_hotpot_explore
  # the memory gates A5 and A7 (Stage 6)
  if [ "$SMOKE" = 0 ]; then
    $PY scripts/memory_eval.py
    $PY scripts/memory_eval.py --tag mem2
  fi
fi

# ---------------------------------------------------------------- memory
# The experience logs are not committed (.gitignore); backfill rebuilds them
# from the committed traces with the shadow arms re-run (bm25, dense on cpu or
# mps), and exits 1 if any hop differs from the trace.
if stage memory cpu; then
  MEM=$RES/memory
  mkdir -p "$MEM"
  $PY -m src.memory.backfill --run bm25_arms_hotpot_dev --config "$(cfg bm25_arms_hotpot)" \
    --data-file $HOTPOT ${NSMOKE[@]+"${NSMOKE[@]}"} --out $MEM/bm25_arms_hotpot_dev_experience.jsonl
  $PY -m src.memory.backfill --run bm25_arms_musique_dev --config "$(cfg bm25_arms_musique)" \
    --data-file $MUSIQUE ${NSMOKE[@]+"${NSMOKE[@]}"} --out $MEM/bm25_arms_musique_dev_experience.jsonl
  if [ "$SMOKE" = 0 ]; then
    for ds in hotpot musique; do
      rec=$MEM/bm25_arms_${ds}_dev_experience.jsonl
      $PY -m src.memory.arms --records $rec --dataset $ds --out-dir "$RES"
      $PY -m src.memory.failures --records $rec --dataset $ds --out-dir "$RES"
      $PY -m src.memory.lookalike natural --records $rec --dataset $ds --out-dir "$RES"
      $PY -m src.memory.stream --records $rec --dataset $ds --out-dir "$RES"
    done
    $PY -m src.memory.drift --records-a $MEM/bm25_arms_hotpot_dev_experience.jsonl --dataset-a hotpot \
      --records-b $MEM/bm25_arms_musique_dev_experience.jsonl --dataset-b musique --out-dir "$RES"
  fi
fi

# ---------------------------------------------------------------- plots
# The cell's decomposition plots come out of `analyses`; here the 2x2 and the
# per-hop plot of every committed run are redrawn from its qa.csv and
# per_hop.json, and the serving sweep's pareto plot from its csv.
if stage plots cpu; then
  $PY scripts/plots.py --results "$RES"
fi

[ "$DRY" = 1 ] || { echo; echo "done (results in $RES)"; }
