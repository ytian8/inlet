#!/usr/bin/env bash
# Evaluate a ladder of checkpoints and collect the curve.
#
# FOR DEBUGGING. This scores checkpoints on the benchmarks' own test data, which
# is exactly what you want when the question is "when did the model peak?" and
# exactly what you must not do when the question is "what number goes in the
# paper". Pick the reported checkpoint with a rule that never sees these scores
# (--model_select_split); use this to understand the run.
#
#   ./scripts/sweep_checkpoints.sh <run_dir> [TASKS]
#
# Defaults to the three generative tasks, because they carry the whole signal:
# from step 4,000 to 130,000 the seven multiple-choice tasks moved by -0.73
# while the generative average moved by -10.96.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

RUN_DIR="${1:?usage: sweep_checkpoints.sh <run_dir> [TASKS]}"
SWEEP_TASKS="${2:-humaneval gsm8k mbpp}"

mapfile -t CKPTS < <(ls -1 "$RUN_DIR"/hypermod_inlet_step*.pt 2>/dev/null | sort -t_ -k4 -n)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no hypermod_inlet_step*.pt in $RUN_DIR" >&2
  echo "Those only exist if the run passed --checkpoint_steps. Available files:" >&2
  ls -1 "$RUN_DIR"/*.pt 2>/dev/null >&2 || true
  exit 1
fi

echo "sweeping ${#CKPTS[@]} checkpoint(s) over: $SWEEP_TASKS"
printf '  %s\n' "${CKPTS[@]}"
echo

for ck in "${CKPTS[@]}"; do
  step="$(basename "$ck" | sed 's/.*step\([0-9]*\)\.pt/\1/')"
  echo "=== step $step ==="
  TASKS="$SWEEP_TASKS" ./scripts/eval.sh "$ck"
done

echo
echo "collecting:"
"$PYBIN" -m inlet.sweep_report "$INLET_OUTPUT_ROOT/eval_results_inlet"
