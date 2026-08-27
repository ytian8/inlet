#!/usr/bin/env bash
# Ablation sweep: a queue of whole Inlet runs, one worker per GPU GROUP.
#
#   ./scripts/run_queue_inlet.sh 0        # worker using GPU 0
#   ./scripts/run_queue_inlet.sh 0,1      # worker using GPUs 0 and 1 (DDP within a job)
#
# Use this AFTER the headline run works, for the ablation table. Unlike the
# per-task prompt-tuning sweep (9 tasks x 3 seeds, embarrassingly parallel), one
# Inlet job is a whole training run over all 479 tasks and cannot be split
# further -- so the parallelism here is across CONFIGS, and DDP is what makes a
# single config fast.
#
# Each job is train-then-eval, so the first job to finish validates the whole
# chain instead of a defect surfacing hours in. Re-running is safe and is the
# resume path: a job whose results already exist is skipped.
#
# Several boxes, shared filesystem -> just launch workers everywhere; the
# flock'd cursor coordinates them. NOT shared -> give each box a disjoint slice:
#   NSHARD=3 SHARD=0 ./scripts/run_queue_inlet.sh 0,1     # box A
#   NSHARD=3 SHARD=1 ./scripts/run_queue_inlet.sh 0,1     # box B  ... etc
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
set +e   # a failing job must not kill the worker

GPUS="${1:?usage: run_queue_inlet.sh <gpu_id|gpu_id,gpu_id,...>}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

QDIR="$INLET_OUTPUT_ROOT/queue"
mkdir -p "$QDIR"
JOBS="$QDIR/jobs_inlet.txt"
CURSOR="$QDIR/.cursor.$SHARD"
LOCK="$QDIR/.cursor.$SHARD.lock"
LOG="$QDIR/queue.log"
RESULTS="$INLET_OUTPUT_ROOT/eval_results_inlet"
mkdir -p "$RESULTS"

if [[ ! -f "$JOBS" ]]; then
  "$PYBIN" - <<'PY' > "$JOBS"
SEEDS = [0, 1, 2]
# (n_virtual_tokens, head, extra flags)
CONFIGS = [
    (32, "per_slot", ""),               # headline, matches the 32-token upper bound
    (8,  "per_slot", ""),               # compressed budget
    (32, "per_slot", "--freeze_head"),  # base-only control: no description at all
    (32, "shared",   ""),               # head ablation
]
for m, head, extra in CONFIGS:
    for seed in SEEDS:
        # the base-only control has no description path to vary, so one seed
        if "--freeze_head" in extra and seed != 0:
            continue
        print(f"m{m}__{head}__seed{seed}", m, head, seed, extra)
PY
  echo 1 > "$CURSOR"
fi
[[ -f "$CURSOR" ]] || echo 1 > "$CURSOR"

log() { echo "[$(date '+%F %T')][gpu$GPUS/shard$SHARD] $*" | tee -a "$LOG"; }

next_job() {
  exec 9>"$LOCK"
  flock 9
  local i line
  i=$(cat "$CURSOR" 2>/dev/null || echo 1)
  while :; do
    line=$(sed -n "${i}p" "$JOBS")
    [[ -z "$line" ]] && break
    if (( (i - 1) % NSHARD == SHARD )); then break; fi
    i=$((i + 1))
  done
  echo $((i + 1)) > "$CURSOR"
  flock -u 9
  echo "$line"
}

log "worker up on $NGPU gpu(s). jobs=$JOBS shard=$SHARD/$NSHARD"
while :; do
  read -r TAG M HEAD SEED EXTRA <<< "$(next_job)"
  [[ -z "${TAG:-}" ]] && { log "queue drained"; break; }

  if compgen -G "$RESULTS/${TAG}__*.json" > /dev/null; then
    log "skip $TAG (results exist)"; continue
  fi

  log "START $TAG  m=$M head=$HEAD seed=$SEED $EXTRA"
  t0=$SECONDS
  JOBLOG="$QDIR/logs_${TAG}.txt"

  # shellcheck disable=SC2086
  if ! ./scripts/train.sh "$NGPU" --run_name="$TAG" --n_virtual_tokens="$M" \
        --hypernet_head="$HEAD" --seed="$SEED" $EXTRA >> "$JOBLOG" 2>&1; then
    log "FAIL(train) $TAG after $((SECONDS - t0))s -- see $JOBLOG"; continue
  fi

  CKPT="$INLET_OUTPUT_ROOT/hyper_lora/$TAG/hypermod_inlet.pt"
  if ! OUT="$RESULTS" ./scripts/eval.sh "$CKPT" >> "$JOBLOG" 2>&1; then
    log "FAIL(eval) $TAG after $((SECONDS - t0))s -- see $JOBLOG"; continue
  fi

  log "DONE $TAG in $((SECONDS - t0))s"
done
