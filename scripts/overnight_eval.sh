#!/usr/bin/env bash
# Score the arms on the tasks that actually collapsed.
#
#   ./scripts/overnight_eval.sh control ews
#   TASKS="gsm8k humaneval" ./scripts/overnight_eval.sh control
#
# Why this exists rather than scripts/eval.sh: eval.sh runs one process per
# task, so a 2-task run pays two vLLM engine builds (~3 min each). eval_inlet
# takes `--tasks a b` and memoises the engine across them inside one process.
#
# `zero` is not an arm -- it is the frozen model down the identical eval path.
# It is the only number here that says whether a prompt helps or hurts at all,
# and on the previous run it was the number that proved the collapse was not an
# eval bug (humaneval 39.63 against a published zero-shot of 37.80).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
set +e

TASKS="${TASKS:-gsm8k humaneval}"
OUT="${INLET_OUTPUT_ROOT:-/root/outputs}"
RUNS="${OUT}/hyper_lora"
CKPT_NAME="${CKPT_NAME:-hypermod_inlet.pt}"
RESULTS="${OUT}/eval_results_inlet"
LOG="${OUT}/eval.log"
mkdir -p "$RESULTS"

# One description per task, and no junk-description controls: this pass is
# asking "did generation come back", not "does the description matter". The
# reported protocol averages three descriptions and adds the junk controls --
# do NOT put a number from this pass in the same table as one from that.
EXTRA=(--max-eval-descs 1 --skip-random-descs)

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "tasks: $TASKS   arms: $*   (single description, no junk controls)"

for arm in "$@"; do
  if [[ "$arm" == "zero" ]]; then
    say "EVAL zero-prompt (frozen model, identical path)"
    t0=$SECONDS
    "$PYBIN" -m inlet.eval_inlet --task "${TASKS%% *}" --tasks $TASKS \
        --zero-prompt "${EXTRA[@]}" --out-dir "$RESULTS" \
        > "$OUT/eval.zero.log" 2>&1
    rc=$?
  else
    ckpt="$RUNS/$arm/$CKPT_NAME"
    if [[ ! -f "$ckpt" ]]; then
      say "SKIP $arm -- no $ckpt"
      continue
    fi
    say "EVAL $arm  ($ckpt)"
    t0=$SECONDS
    "$PYBIN" -m inlet.eval_inlet --task "${TASKS%% *}" --tasks $TASKS \
        --checkpoint "$ckpt" "${EXTRA[@]}" --out-dir "$RESULTS" \
        > "$OUT/eval.$arm.log" 2>&1
    rc=$?
  fi
  dt=$(( SECONDS - t0 ))
  if (( rc == 0 )); then
    say "DONE $arm in $(( dt / 60 ))m$(( dt % 60 ))s"
  else
    say "FAIL $arm rc=$rc -- last 20 lines:"
    tail -20 "$OUT/eval.$arm.log" 2>/dev/null | tee -a "$LOG"
  fi
done

say "results json under $RESULTS"
ls -1 "$RESULTS" 2>/dev/null | tail -20 | tee -a "$LOG"
