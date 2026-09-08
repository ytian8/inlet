#!/usr/bin/env bash
# Five training arms, one GPU, sequential. See docs/OVERNIGHT.md for why each
# one exists; this file is only the mechanics.
#
#   ./scripts/overnight_arms.sh            # all arms, in priority order
#   ARMS="control ews" ./scripts/overnight_arms.sh
#   MAX_STEPS=2000 ./scripts/overnight_arms.sh
#
# Arms run in PRIORITY order and each writes its results before the next
# starts, so an interrupted sweep still yields a readable prefix. A run whose
# checkpoint already exists is skipped, which makes re-running the resume path.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
set +e   # one failing arm must not kill the sweep

# Sized from the smoke run on this box: ~100 training steps/2.5 min, and a
# 4-split validation at the stock max_batches=50 costs ~8 minutes -- so at
# max_steps=4000/val_freq=1000 one arm is ~2.3h and five are ~11.7h, most of it
# validation rather than training. 2000 steps with cheaper validation puts the
# five arms near 5.5h. The known optimum is at or before step 4000 and collapse
# shows up in the first few thousand steps, so 2000 still lands inside the
# region of interest -- and every arm gets the identical budget, which is what
# the comparison actually rests on. Report it as 2000, never as 4000.
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_FREQ="${VAL_FREQ:-500}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-15}"
CKPT_STEPS="${CKPT_STEPS:-500,1000,2000}"
L2="${L2:-0}"          # calibrated from the control arm's prompt_norm; 0 = skip that arm
OUT="${INLET_OUTPUT_ROOT:-/root/outputs}"
LOG="$OUT/arms.log"
mkdir -p "$OUT"

# The A/B substrate: every arm is cross-attention over 8 description slots,
# which is the configuration the reported inlet numbers came from. Only the
# one flag under test differs between an arm and `control`.
COMMON=(
  --desc_slots=8
  --cond=cross
  --model_select_split=val/unseen
  "--max_steps=$MAX_STEPS"
  "--val_freq=$VAL_FREQ"
  "--val_max_batches=$VAL_MAX_BATCHES"
  "--checkpoint_steps=$CKPT_STEPS"
)

# name -> the single flag that defines the arm, and the field it must land on.
#
# EVERY flag must be `--key=value`. Upstream's parser is not argparse: it does
#   other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] ...}
# so a bare `--freeze_head`, or `--equally_weight_sample False` as two tokens,
# raises IndexError before training starts. Verifying these against
# HfArgumentParser is NOT sufficient -- that is a different parser from the one
# train_inlet.py actually uses, and it accepts forms this one rejects.
declare -A ARM_FLAGS=(
  [control]=""
  [ews]="--equally_weight_sample=False"
  [l2]="--l2_reg_prompt=$L2"
  [striptask]="--strip_taskdef_in_training=True"
  [freezehead]="--freeze_head=True"
)
# field -> value the saved args.yaml must show for the arm to count as real.
declare -A ARM_EXPECT=(
  [control]=""
  [ews]="equally_weight_sample:false"
  [l2]="l2_reg_prompt:$L2"
  [striptask]="strip_taskdef_in_training:true"
  [freezehead]="freeze_head:true"
)
# Priority order, chosen so that a sweep cut short still answers the most
# valuable questions:
#   control     everything else is read against it
#   ews         does closing the 28.5x length tilt restore generation? the
#               single question with the most evidence behind it
#   striptask   does closing the redundant text channel make the description
#               matter? that is the paper's actual claim
#   freezehead  how much of the score is task-agnostic? reframes the rest
#   l2          the norm hypothesis -- last because its coefficient is a guess
#               calibrated from one number, and every other arm reports a
#               prompt_norm trajectory anyway, so the norm-vs-generation
#               relationship gets a scatter of points even if this never runs
ORDER=(control ews striptask freezehead l2)
ARMS="${ARMS:-${ORDER[*]}}"

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "arms: $ARMS   max_steps=$MAX_STEPS  val_freq=$VAL_FREQ  l2=$L2"
nvidia-smi --query-gpu=name --format=csv,noheader | tee -a "$LOG"

for arm in $ARMS; do
  flags="${ARM_FLAGS[$arm]-__missing__}"
  if [[ "$flags" == "__missing__" ]]; then
    say "SKIP $arm -- not a known arm (known: ${ORDER[*]})"
    continue
  fi
  if [[ "$arm" == "l2" && "$L2" == "0" ]]; then
    say "SKIP l2 -- L2 is 0, which is the control's setting, so the arm would be a duplicate."
    say "     Calibrate it from the control arm's prompt_norm and re-run with L2=<value>."
    continue
  fi

  run="$OUT/hyper_lora/$arm"
  if [[ -f "$run/hypermod_inlet.pt" ]]; then
    say "SKIP $arm -- $run/hypermod_inlet.pt already exists"
    continue
  fi

  say "START $arm   flags: ${flags:-<none, this is the control>}"
  t0=$SECONDS
  # shellcheck disable=SC2086
  ./scripts/train.sh 1 --run_name="$arm" "${COMMON[@]}" $flags \
      > "$OUT/$arm.train.log" 2>&1
  rc=$?
  dt=$(( SECONDS - t0 ))
  if (( rc == 0 )); then
    say "DONE  $arm in $(( dt / 60 ))m$(( dt % 60 ))s"
    # Did the flag actually land? The trainer writes args.yaml into the run dir
    # AFTER parsing, so it is the record of what really ran -- not what was
    # asked for. An arm whose flag was silently dropped trains fine, logs a
    # falling loss and produces a checkpoint that is a duplicate of control,
    # which is the one result this sweep cannot afford to report.
    want="${ARM_EXPECT[$arm]}"
    if [[ -n "$want" ]]; then
      field="${want%%:*}"; value="${want##*:}"
      got=$(grep -aiE "^${field}:" "$run/args.yaml" 2>/dev/null | head -1 | sed 's/^[^:]*: *//' | tr -d "'\"" | tr 'A-Z' 'a-z')
      # floats come back as 3000.0 where the flag said 3000
      got="${got%.0}"; value="${value%.0}"
      if [[ "$got" == "$value" ]]; then
        say "  verified: $field = $got"
      else
        say "  *** $arm IS NOT WHAT IT CLAIMS: args.yaml has $field='$got', expected '$value'."
        say "      Treat this arm as a duplicate of control, not as a null result."
      fi
    fi
  else
    say "FAIL  $arm rc=$rc after $(( dt / 60 ))m -- last 15 lines:"
    tail -15 "$OUT/$arm.train.log" | tee -a "$LOG"
  fi

  # The readouts, pulled out while they are fresh so an interrupted sweep still
  # leaves something readable.
  #
  # free_running_acc on val/generative (gsm8k) is the one that matters and it
  # costs nothing: the canary records it at EVERY validation, not only when the
  # gap is wide enough to warn. Teacher-forced per_token_acc sat at 0.80-0.82
  # for the whole 147,500-step run while humaneval was 4.47, so per_token_acc
  # on its own is not evidence of anything -- the gap between the two is.
  {
    echo "### $arm  rc=$rc  ${dt}s   flags: ${flags:-<control>}"
    for split in val/generative val/unseen val/benchmark; do
      line=$(grep -aE "\[step [0-9]+\] ${split}:" "$OUT/$arm.train.log" | tail -1)
      [[ -n "$line" ]] && echo "  ${split}: ${line#*: }"
    done
    echo "  varying_fraction: $(grep -aoE "varying_fraction=[0-9.e+-]+" "$OUT/$arm.train.log" | tail -4 | tr '\n' ' ')"
    echo "  prompt_norm:      $(grep -aoE "prompt_norm=[0-9.]+" "$OUT/$arm.train.log" | tail -4 | tr '\n' ' ')"
    echo "  free_running:     $(grep -aoE "free_running_acc=[0-9.]+" "$OUT/$arm.train.log" | tail -4 | tr '\n' ' ')"
    n=$(grep -ac "CANARY" "$OUT/$arm.train.log")
    echo "  CANARY warnings:  $n"
    (( n > 0 )) && grep -a "CANARY" "$OUT/$arm.train.log" | tail -2 | sed 's/^/    /'
    echo
  } >> "$OUT/readouts.txt"

  # Training writes to local disk on purpose -- a checkpoint written to a wedged
  # network mount comes back the right size and fails to load. But local disk
  # dies with the container, so copy each arm off as it finishes: an
  # interruption then costs the in-flight arm and nothing already earned.
  if [[ -n "${ARCHIVE_DIR:-}" ]]; then
    mkdir -p "$ARCHIVE_DIR"
    if rsync -a --exclude '*.tmp' "$run" "$OUT/$arm.train.log" "$OUT/readouts.txt" \
             "$ARCHIVE_DIR/" 2>>"$LOG"; then
      say "archived $arm -> $ARCHIVE_DIR"
    else
      say "WARNING: archiving $arm to $ARCHIVE_DIR failed; results are only on local disk"
    fi
  fi
done

say "sweep finished. readouts:"
cat "$OUT/readouts.txt" 2>/dev/null | tee -a "$LOG"
