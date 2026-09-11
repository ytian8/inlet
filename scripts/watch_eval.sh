#!/usr/bin/env bash
# Evaluate each checkpoint as it appears, alongside a training run that keeps
# going. Start it in its own tmux window after the training has started:
#
#   RUN=run3_m20 ./scripts/watch_eval.sh
#
# Then at any time, from anywhere:
#
#   python -m inlet.run_status --run run3_m20
#
# WHY THIS IS SAFE TO RUN NEXT TO TRAINING
#
# Training peaked at 14.58 GiB per rank (docs/RUN1.md §3), so on an 80 GB A100
# there is room for a vLLM engine at --gpu-memory-utilization 0.35 (28 GB):
# 14.58 + 28 = 43 GB. The direction of risk also matters -- training's caching
# allocator has already reserved what it needs, so an eval that cannot allocate
# fails ITSELF and leaves the training run alone. That is why the memory knob is
# turned down here rather than left at eval_inlet's 0.7 default, which would not
# fit.
#
# Cost: ~15 min per checkpoint against a 4.6-day run, so a few percent of wall
# clock. Two tasks only, deliberately: arc_challenge is the short-output family
# and humaneval the long-output one, which is the split that matters, and adding
# boolq or hellaswag would multiply the time by several.
#
# WHAT THESE NUMBERS ARE NOT
#
# One description, no junk-description control. They are a progress signal, not
# a result, and must never share a table with a reported-protocol number. The
# reported table is docs/RUN2.md.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
set +e                      # a failed eval must never take the watcher down

RUN="${RUN:?set RUN to the training run_name, e.g. RUN=run3_m20}"
TASKS="${TASKS:-arc_challenge humaneval}"
SCALES="${SCALES:-1.0,0.5}"
GPU_FRAC="${GPU_FRAC:-0.35}"
EVAL_GPU="${EVAL_GPU:-0}"
POLL_S="${POLL_S:-120}"
# Stop once the run's own final checkpoint exists and nothing new has appeared
# for this long. 0 = never stop; kill it by hand.
IDLE_STOP_S="${IDLE_STOP_S:-0}"

OUT="${INLET_OUTPUT_ROOT}"
RUNDIR="$OUT/hyper_lora/$RUN"
LOG="$OUT/${RUN}.watch.log"
mkdir -p "$OUT"

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "watching $RUNDIR"
say "tasks='$TASKS'  scales=$SCALES  gpu=$EVAL_GPU  mem_frac=$GPU_FRAC"
say "status at any time:  python -m inlet.run_status --run $RUN"

idle=0
while :; do
  found_new=0
  # Numeric sort so step 2000 is evaluated before step 16000 even though the
  # shell would order them the other way.
  for ck in $(ls -1 "$RUNDIR"/hypermod_inlet_step*.pt 2>/dev/null \
              | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2-); do
    step="$(basename "$ck" | sed 's/.*step\([0-9]*\)\.pt/\1/')"
    donefile="$OUT/.watch_done_${RUN}_$step"
    [[ -f "$donefile" ]] && continue

    # A checkpoint is written non-atomically. Wait for its size to stop moving
    # before reading it, or probe_prompt loads half a file and the step is
    # marked done with a garbage number.
    s1=$(stat -c %s "$ck" 2>/dev/null); sleep 10
    s2=$(stat -c %s "$ck" 2>/dev/null)
    [[ "$s1" != "$s2" || -z "$s1" ]] && { say "step $step still being written, skipping this round"; continue; }

    found_new=1; idle=0
    say "step $step -- probe"
    "$PYBIN" -m inlet.probe_prompt --checkpoint "$ck" --task arc_challenge \
        --out "$OUT/${RUN}_probe_step${step}.json" >> "$LOG" 2>&1

    say "step $step -- eval ($TASKS at $SCALES)"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
    TASKS="$TASKS" \
    EXTRA_EVAL_ARGS="--prompt-scales $SCALES --max-eval-descs 1 --skip-random-descs --gpu-memory-utilization $GPU_FRAC" \
        ./scripts/eval.sh "$ck" >> "$LOG" 2>&1
    rc=$?
    if (( rc != 0 )); then
      say "step $step eval rc=$rc -- leaving it un-done so the next round retries"
      say "  (if this is OOM, lower GPU_FRAC; training is unaffected either way)"
      tail -5 "$LOG"
      continue
    fi
    touch "$donefile"

    say "step $step done. current status:"
    "$PYBIN" -m inlet.run_status --run "$RUN" 2>&1 | tee -a "$LOG"
  done

  if (( ! found_new )); then
    idle=$((idle + POLL_S))
    if (( IDLE_STOP_S > 0 && idle >= IDLE_STOP_S )); then
      say "no new checkpoint for ${idle}s -- stopping"
      break
    fi
  fi
  sleep "$POLL_S"
done

say "final status:"
"$PYBIN" -m inlet.run_status --run "$RUN" 2>&1 | tee -a "$LOG"
