#!/usr/bin/env bash
# Evaluate a ladder of checkpoints and collect the curve.
#
# FOR DEBUGGING AND FOR THE CURVE. This scores checkpoints on the benchmarks'
# own test data, which is what you want when the question is "when did the model
# peak" and what you must not do when the question is "what number goes in the
# paper". Pick the reported checkpoint with --model_select_split, which never
# sees these scores. See docs/MODEL_SELECTION.md.
#
#   ./scripts/sweep_checkpoints.sh <run_dir> [TASKS] [N_GPUS]
#
#   ./scripts/sweep_checkpoints.sh out/cross8                       # 3 generative tasks, 1 GPU
#   ./scripts/sweep_checkpoints.sh out/cross8 "$ALL10" 8            # all ten, 8 GPUs
#
# EVAL USES ONE GPU. On a multi-GPU node the checkpoints are sharded across
# them and run at once, which is the difference between a two-hour sweep and a
# fifteen-hour one.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

ALL10="arc_challenge arc_easy boolq hellaswag openbookqa piqa winogrande gsm8k mbpp humaneval"

RUN_DIR="${1:?usage: sweep_checkpoints.sh <run_dir> [TASKS] [N_GPUS]}"
SWEEP_TASKS="${2:-gsm8k mbpp humaneval}"
NGPU="${3:-1}"

# A sweep wants the SHAPE of the curve, so one real description per task is
# enough and costs a sixth of the full protocol. The reported number needs all
# three plus the junk-description control -- set SWEEP_FULL=1 for that.
# eval_inlet records which protocol produced each file, so the two can never be
# put in one table by accident.
SWEEP_EVAL_ARGS="${SWEEP_EVAL_ARGS:---max-eval-descs 1 --skip-random-descs}"
[[ "${SWEEP_FULL:-0}" == "1" ]] && SWEEP_EVAL_ARGS=""

# WHICH= selects what to sweep. Start with `best`: four files, one per validation
# split, and if one of them is good enough the step ladder never has to be paid
# for. Fall back to `steps` when you need the curve.
#   best    the four --save_best_per_split checkpoints        (4 jobs per task)
#   steps   the --checkpoint_steps ladder                     (9 jobs per task)
#   all     both
WHICH="${WHICH:-steps}"

collect_ckpts() {
  case "$1" in
    best)  ls -1 "$RUN_DIR"/hypermod_inlet_best_*.pt 2>/dev/null ;;
    steps) ls -1 "$RUN_DIR"/hypermod_inlet_step*.pt 2>/dev/null \
             | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2 ;;
    all)   collect_ckpts best; collect_ckpts steps ;;
    *)     echo "WHICH must be best|steps|all, got $1" >&2; exit 1 ;;
  esac
}

# SWEEP_CKPTS overrides everything, for scoring specific files.
#
# Split on whitespace INCLUDING newlines. `read -r -a` reads one line and stops,
# so SWEEP_CKPTS="$(ls .../step{500,2000,8000}.pt)" silently swept the first file
# and dropped the rest -- no error, just two thirds of the answer missing.
if [[ -n "${SWEEP_CKPTS:-}" ]]; then
  # shellcheck disable=SC2206
  CKPTS=($SWEEP_CKPTS)
  # A path that does not exist must stop the sweep now, not after an engine build.
  missing=()
  for ck in "${CKPTS[@]}"; do [[ -f "$ck" ]] || missing+=("$ck"); done
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'SWEEP_CKPTS lists %d file(s) that do not exist:\n' "${#missing[@]}" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
else
  mapfile -t CKPTS < <(collect_ckpts "$WHICH")
fi

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no checkpoints matched WHICH=$WHICH in $RUN_DIR" >&2
  echo "Present:" >&2
  ls -1 "$RUN_DIR"/*.pt 2>/dev/null >&2 || echo "  (no .pt files at all)" >&2
  echo "The step ladder exists only if the run passed --checkpoint_steps." >&2
  exit 1
fi

echo "sweeping ${#CKPTS[@]} checkpoint(s) over ${NGPU} GPU(s)"
echo "  tasks: $SWEEP_TASKS"
printf '  %s\n' "${CKPTS[@]}"
echo

# One job per (checkpoint, task). Sharding by checkpoint alone would leave seven
# of eight GPUs idle whenever there is one checkpoint to score -- which is the
# case for the number that actually gets reported.
#
# Each job builds its own vLLM engine, and on a 7B that build is ~9 min against
# ~2.6 min per description variant. Engine builds, not variants, are what a
# sweep costs, and one job per (checkpoint, task) is the finest split that does
# not pay for extra ones.
JOBS=()
for ck in "${CKPTS[@]}"; do
  for t in $SWEEP_TASKS; do JOBS+=("$ck|$t"); done
done
echo "${#JOBS[@]} job(s) = ${#CKPTS[@]} checkpoint(s) x $(wc -w <<< "$SWEEP_TASKS") task(s)"
echo

LOGDIR="$INLET_OUTPUT_ROOT/sweep_logs"; mkdir -p "$LOGDIR"
fail=0
for ((i = 0; i < ${#JOBS[@]}; i += NGPU)); do
  pids=()
  for ((g = 0; g < NGPU && i + g < ${#JOBS[@]}; g++)); do
    IFS='|' read -r ck t <<< "${JOBS[i + g]}"
    step="$(basename "$ck" | sed 's/.*step\([0-9]*\)\.pt/\1/')"
    echo "  [gpu $g] step $step / $t"
    CUDA_VISIBLE_DEVICES="$g" TASKS="$t" \
      ${SWEEP_EVAL_ARGS:+EXTRA_EVAL_ARGS="$SWEEP_EVAL_ARGS"} \
      ./scripts/eval.sh "$ck" > "$LOGDIR/step${step}_${t}.log" 2>&1 &
    pids+=($!)
  done
  # Wait for the whole wave and report which member died, rather than letting
  # `set -e` kill the sweep on the first failure and lose the rest.
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
done

[[ $fail -gt 0 ]] && echo "WARNING: $fail eval(s) failed -- see $LOGDIR/" >&2

echo
"$PYBIN" -m inlet.sweep_report "$INLET_OUTPUT_ROOT/eval_results_inlet" \
    --plot "$INLET_OUTPUT_ROOT/sweep_curve.png"
