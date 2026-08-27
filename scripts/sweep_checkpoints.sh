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

mapfile -t CKPTS < <(ls -1 "$RUN_DIR"/hypermod_inlet_step*.pt 2>/dev/null \
                     | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no hypermod_inlet_step*.pt in $RUN_DIR" >&2
  echo "Those exist only if the run passed --checkpoint_steps. Present:" >&2
  ls -1 "$RUN_DIR"/*.pt 2>/dev/null >&2 || echo "  (no .pt files at all)" >&2
  exit 1
fi

echo "sweeping ${#CKPTS[@]} checkpoint(s) over ${NGPU} GPU(s)"
echo "  tasks: $SWEEP_TASKS"
printf '  %s\n' "${CKPTS[@]}"
echo

LOGDIR="$INLET_OUTPUT_ROOT/sweep_logs"; mkdir -p "$LOGDIR"
fail=0
for ((i = 0; i < ${#CKPTS[@]}; i += NGPU)); do
  pids=()
  for ((g = 0; g < NGPU && i + g < ${#CKPTS[@]}; g++)); do
    ck="${CKPTS[i + g]}"
    step="$(basename "$ck" | sed 's/.*step\([0-9]*\)\.pt/\1/')"
    echo "  [gpu $g] step $step"
    CUDA_VISIBLE_DEVICES="$g" TASKS="$SWEEP_TASKS" \
      ./scripts/eval.sh "$ck" > "$LOGDIR/step${step}.log" 2>&1 &
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
