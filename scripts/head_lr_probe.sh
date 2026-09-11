#!/usr/bin/env bash
# Probe two --head_lr_mult values to step 4,000 each, then commit the winner to
# the full 147,500-step budget. Start it and leave; it takes its own decision.
#
#   ./scripts/head_lr_probe.sh
#   MULTS="20 60" BAR=0.15 ./scripts/head_lr_probe.sh
#   DRY=1 ./scripts/head_lr_probe.sh          # print the plan, run nothing
#
# WHY A PROBE AND NOT JUST THE LONG RUN
#
# docs/RUN3.md §1: the head that reads the task description trains at T2L's
# lr=2.5e-5 while per-task prompt tuning needed 3e-3 for vectors of the same
# shape. --head_lr_mult has existed the whole time and has never been set. If
# that diagnosis is right, cos(real,junk) separation moves early; if it is
# wrong, no amount of further training fixes it. Three hours answers it, so a
# 4.6-day run should not start until it has.
#
# A single value cannot answer it. "mult=20 did not separate" is either "the
# diagnosis is wrong" or "20 is too small", and those want opposite next steps.
# Two values separate them, which is the whole reason this script runs two.
#
# THE PROBE USES THE REAL SCHEDULE. Each probe starts a genuine
# --max_steps=147500 run and is killed once step 4,000 is on disk. Running a
# short --max_steps=4000 job instead would measure a different learning rate:
# warmup is 20% of max_steps, so step 4,000 sits at lr factor 0.99 in a short
# run and 0.136 in the real one. Nothing is wasted by killing it -- the first
# 4,000 steps are the same steps either way.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
set +e                      # a failing probe must not take the script down

MULTS="${MULTS:-20 60}"
PROBE_STEP="${PROBE_STEP:-4000}"
FULL_STEPS="${FULL_STEPS:-147500}"
# Run 1 (mult=1) reached 0.0516 at step 4,000 with ~9x this run's learning rate
# at that point. Clearing 0.15 is the bar; see docs/RUN3.md §3.
BAR="${BAR:-0.15}"
RUN1_REF="${RUN1_REF:-0.0516}"
NGPU="${NGPU:-2}"
# step 4,000 is ~3 h at the measured 2.67 s/step on 2x A100. Give it 5 before
# concluding the run died rather than that it is slow.
TIMEOUT_S="${TIMEOUT_S:-18000}"
CKPTS="1000,2000,4000,8000,16000,24000,32000,48000,64000,96000,128000,147500"

OUT="${INLET_OUTPUT_ROOT}"
LOG="$OUT/head_lr_probe.log"
mkdir -p "$OUT"

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# separation = mean(cos real-vs-real) - mean(cos real-vs-junk), which is the
# column docs/RUN3.md §1 tabulates for Run 1. probe_prompt writes both as dicts
# of pairwise values, so both have to be averaged before subtracting.
separation() {
  "$PYBIN" - "$1" <<'PY' 2>/dev/null || echo "NA"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    rr = list(d["cos_real_real"].values())
    rj = list(d["cos_real_junk"].values())
    print(f"{sum(rr)/len(rr) - sum(rj)/len(rj):.4f}")
except Exception:
    print("NA")
PY
}

# Returns the separation on stdout, or NA. Everything else goes to the log.
probe_one() {
  local mult="$1" name="probe_m${mult}"
  local dir="$OUT/hyper_lora/$name"
  local ckpt="$dir/hypermod_inlet_step${PROBE_STEP}.pt"
  local pj="$OUT/${name}_probe_step${PROBE_STEP}.json"

  if [[ -f "$pj" ]]; then
    say "probe mult=$mult already done, reusing $pj" >&2
    separation "$pj"; return
  fi

  say "probe mult=$mult -- training to step $PROBE_STEP (~3 h)" >&2
  # setsid so the trainer is its own process group: torchrun spawns children and
  # killing only the shell leaves them holding the GPUs.
  setsid ./scripts/train.sh "$NGPU" --run_name="$name" \
      --desc_slots=8 --cond=cross \
      --model_select_split=val/unseen \
      --head_lr_mult="$mult" \
      --max_steps="$FULL_STEPS" \
      --checkpoint_steps="$CKPTS" \
      --val_freq=4000 \
      --val_max_batches=15 \
      > "$OUT/${name}.train.log" 2>&1 &
  local pg=$!

  local waited=0
  while (( waited < TIMEOUT_S )); do
    [[ -f "$ckpt" ]] && break
    kill -0 "$pg" 2>/dev/null || { say "mult=$mult: trainer exited before step $PROBE_STEP" >&2; break; }
    sleep 60; waited=$((waited + 60))
  done

  kill -TERM -"$pg" 2>/dev/null
  sleep 20
  kill -KILL -"$pg" 2>/dev/null
  # torchrun can leave a rank holding memory for a few seconds after SIGKILL.
  sleep 30

  if [[ ! -f "$ckpt" ]]; then
    say "mult=$mult: NO step-$PROBE_STEP checkpoint. Last 15 log lines:" >&2
    tail -15 "$OUT/${name}.train.log" | tee -a "$LOG" >&2
    echo "NA"; return
  fi

  "$PYBIN" -m inlet.probe_prompt --checkpoint "$ckpt" \
      --task arc_challenge --out "$pj" >> "$LOG" 2>&1
  separation "$pj"
}

inlet_banner 2>/dev/null
say "head_lr_mult probe: mults='$MULTS'  bar=$BAR  (Run 1 at step $PROBE_STEP: $RUN1_REF)"

if [[ -n "${DRY:-}" ]]; then
  say "DRY: would probe $MULTS to step $PROBE_STEP, then run the winner to $FULL_STEPS"
  exit 0
fi

declare -A SEP
best_mult=""; best_sep=""
for m in $MULTS; do
  s="$(probe_one "$m" | tail -1)"
  SEP[$m]="$s"
  say "RESULT mult=$m  separation=$s   (bar $BAR, Run 1 $RUN1_REF)"
  if [[ "$s" != "NA" ]]; then
    if [[ -z "$best_sep" ]] || awk "BEGIN{exit !($s > $best_sep)}"; then
      best_sep="$s"; best_mult="$m"
    fi
  fi
done

say "=== probe summary ==="
{ echo "  mult=1 (Run 1, reference)  $RUN1_REF"
  for m in $MULTS; do echo "  mult=$m  ${SEP[$m]}"; done; } | tee -a "$LOG"

if [[ -z "$best_sep" ]]; then
  say "STOP: no probe produced a number. This is an infrastructure failure, not a result."
  exit 2
fi

if awk "BEGIN{exit !($best_sep <= $BAR)}"; then
  say "STOP -- and this is a RESULT, not a failure."
  say "  Best separation $best_sep (mult=$best_mult) did not clear the bar $BAR."
  say "  Raising the head learning rate 20-60x does not make the description"
  say "  path separate faster, so the amortization gap is not an optimization"
  say "  speed problem and a 4.6-day run at these settings will not fix it."
  say "  Do not start the long run. See docs/RUN3.md §3."
  exit 3
fi

# Deliberately NOT extrapolating past the largest tested value. If the response
# is still climbing at the top of the range the right next probe is higher than
# anything measured here, and guessing it while nobody is watching risks the
# whole 4.6 days on an untested number.
last_mult="${MULTS##* }"
if [[ "$best_mult" == "$last_mult" ]] && (( $(echo "$MULTS" | wc -w) > 1 )); then
  first_mult="${MULTS%% *}"
  say "NOTE: the largest mult tested ($best_mult) is also the best."
  say "      ${SEP[$first_mult]} at mult=$first_mult vs $best_sep at mult=$best_mult."
  say "      The response may still be rising; a higher mult was not tested."
  say "      Committing to $best_mult anyway -- re-probing costs another 3 h and"
  say "      this is the best value actually measured."
fi

say "COMMIT: full $FULL_STEPS-step run at --head_lr_mult=$best_mult (~4.6 days)"
NAME="run3_m${best_mult}"
setsid ./scripts/train.sh "$NGPU" --run_name="$NAME" \
    --desc_slots=8 --cond=cross \
    --model_select_split=val/unseen \
    --head_lr_mult="$best_mult" \
    --max_steps="$FULL_STEPS" \
    --checkpoint_steps="$CKPTS" \
    --val_freq=4000 \
    --val_max_batches=15 \
    > "$OUT/${NAME}.train.log" 2>&1 &
say "launched as $NAME, pgid $!, log $OUT/${NAME}.train.log"
say "probe checkpoints are kept under $OUT/hyper_lora/probe_m*/ -- delete them"
say "once the long run is past step 4,000 if disk is tight."
