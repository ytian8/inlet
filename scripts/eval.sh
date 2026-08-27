#!/usr/bin/env bash
# Score a checkpoint on the 10 benchmarks T2L's Table 2 reports.
#
#   ./scripts/eval.sh train_outputs/inlet/hyper_lora/full/hypermod_inlet.pt
#   TASKS="arc_challenge" ./scripts/eval.sh <ckpt>       # one task
#   ./scripts/eval.sh --zero-prompt                      # no ckpt: reproduce zero-shot
#
# T2L Table 2 reports "an average of three generated LoRAs, each with a
# different instance of task descriptions" -- so each task is evaluated once per
# description variant and averaged. eval_inlet.py loops the variants inside a
# single vLLM engine, which is why this is 30 evaluations and 10 engine builds.
#
# Needs vLLM, so it needs the LD_PRELOAD from the upstream env.sh (common.sh
# sources it). One GPU is enough; set CUDA_VISIBLE_DEVICES to pick it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
require_disk 10 "$INLET_ROOT" || exit 1   # eval materialises vLLM caches and result json

# The 10 benchmarks in T2L Table 2, in the paper's order.
TASKS="${TASKS:-arc_challenge arc_easy boolq gsm8k hellaswag openbookqa piqa winogrande humaneval mbpp}"
CKPT="${1:-}"
[[ "${CKPT:-}" == --* ]] && CKPT=""   # called with only flags, e.g. --zero-prompt
[[ -n "$CKPT" ]] && shift || true

OUT="${OUT:-$INLET_OUTPUT_ROOT/eval_results_inlet}"
mkdir -p "$OUT"
inlet_banner
echo " checkpoint  ${CKPT:-<none: zero-prompt gate>}"
echo " tasks       $TASKS"
echo " results     $OUT"
echo "--------------------------------------------------------------"

for t in $TASKS; do
  echo "=== $t ==="
  # EXTRA_EVAL_ARGS lets a sweep trim the description set without this script
  # having to know what it is trimming. Unset for a normal run, so the
  # reported protocol stays the default.
  # shellcheck disable=SC2086
  "$PYBIN" -m inlet.eval_inlet --task "$t" --out-dir "$OUT" ${EXTRA_EVAL_ARGS:-} \
    ${CKPT:+--checkpoint "$CKPT"} "$@"
done
