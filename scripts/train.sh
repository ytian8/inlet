#!/usr/bin/env bash
# Inlet training. The ONLY thing that changes between machines is the GPU count.
#
#   ./scripts/train.sh 1  --run_name dbg --max_steps 200   # smoke, ~4 min
#   ./scripts/train.sh 2  --run_name full
#   ./scripts/train.sh 8  --run_name full
#   # 64 GPUs = 8 nodes x 8. GLOBAL_TASKS must stay divisible by
#   # TASKS_PER_RANK x world = 8 x 64 = 512, so the default 64 will NOT do:
#   GLOBAL_TASKS=512 NNODES=8 NODE_RANK=$i MASTER_ADDR=<node0-ip> \
#       ./scripts/train.sh 8 --run_name full
#
# First argument is the number of GPUs PER NODE. Everything after it is passed
# to inlet.train_inlet and overrides the recipe below.
#
# ---------------------------------------------------------------------------
# WHY 2 GPUs AND 8 GPUs GIVE THE SAME RESULT
#
# GLOBAL_TASKS (default 64) is the number of task descriptions per OPTIMIZER
# step, counted across all ranks. Gradient accumulation is derived from it:
#
#     grad_accum = GLOBAL_TASKS / (TASKS_PER_RANK * world)
#
# so 8 GPUs x 8 tasks x 1 accum and 2 GPUs x 8 tasks x 4 accum run the SAME
# optimization -- same batch, same step count, same LR schedule -- and differ
# only in wall time. That is what makes a 2-GPU rerun of an 8-GPU result a
# reproduction rather than a new experiment. It also means GLOBAL_TASKS must
# stay divisible by TASKS_PER_RANK * world; the trainer refuses to start rather
# than silently rounding.
#
# Hit OOM? Lower TASKS_PER_RANK, never GLOBAL_TASKS. grad_accum absorbs it and
# the optimization is unchanged.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
require_disk 20 "$INLET_ROOT" || exit 1   # training writes checkpoints and wandb logs

NGPU="${1:?usage: train.sh <n_gpus_per_node> [--flag=value ...]}"
shift || true
[[ "$NGPU" =~ ^[0-9]+$ && "$NGPU" -ge 1 ]] \
  || { echo "n_gpus must be a positive integer, got '$NGPU'" >&2; exit 1; }

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Derived from the port-less default so two jobs on one node do not collide with
# "Address already in use". Override for multi-node (all nodes must agree).
MASTER_PORT="${MASTER_PORT:-$(( 29500 + (${SLURM_JOB_ID:-$$} % 1000) ))}"
GLOBAL_TASKS="${GLOBAL_TASKS:-64}"
TASKS_PER_RANK="${TASKS_PER_RANK:-8}"

# --- multi-node guards -----------------------------------------------------
# Both of these fail as a SILENT HANG otherwise: torchrun's static rendezvous
# waits forever for peers that will never arrive, and the job burns its whole
# walltime allocation without printing anything.
if (( NNODES > 1 )); then
  if [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
    echo "ERROR: NNODES=$NNODES but MASTER_ADDR=$MASTER_ADDR (loopback)." >&2
    echo "       Every node would rendezvous with itself and hang. Set MASTER_ADDR" >&2
    echo "       to node 0's reachable address on ALL nodes." >&2
    exit 1
  fi
  if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "ERROR: NODE_RANK=$NODE_RANK must be an integer in [0, $NNODES)." >&2
    echo "       Two nodes sharing a NODE_RANK claim the same global ranks; the" >&2
    echo "       rendezvous then hangs or forms a partial group." >&2
    exit 1
  fi
fi
# INLET_OUTPUT_ROOT must be on a SHARED filesystem for NNODES>1: only global rank 0
# writes checkpoints, and the early-stop probe reads that directory on every rank.

WORLD=$(( NGPU * NNODES ))
DENOM=$(( TASKS_PER_RANK * WORLD ))
if (( GLOBAL_TASKS % DENOM != 0 )); then
  echo "ERROR: GLOBAL_TASKS=$GLOBAL_TASKS not divisible by TASKS_PER_RANK($TASKS_PER_RANK) x world($WORLD) = $DENOM." >&2
  echo "       Legal values here: $DENOM $((DENOM*2)) $((DENOM*4)) $((DENOM*8))" >&2
  exit 1
fi
ACCUM=$(( GLOBAL_TASKS / DENOM ))

# ---------------------------------------------------------------------------
# The recipe. Copied field for field from the upstream reference,
# scripts/train_t2l_mistral.sh, so that any difference in results is Inlet vs T2L
# and not two different training setups. Do not "tidy" these values.
#
#   --epochs=20000 is EPOCHS, not steps. At 479 tasks / 8 per batch there are
#   59 batches per epoch, so this is 1,180,000 single-GPU optimizer steps --
#   which is why T2L's README says ~5 days on one H100. The trainer divides by
#   (world x accum) so the DATA SEEN stays the same however many GPUs you have.
# ---------------------------------------------------------------------------
RECIPE=(
  --model_dir=mistralai/Mistral-7B-Instruct-v0.2
  --emb_model=Alibaba-NLP/gte-large-en-v1.5
  --warmup_frac=0.2
  --lr=2.5e-5
  --n_points_per_task=1
  --epochs=20000
  --n_descs_per_ds=128
  --n_train_ds=479
  --encoder_type=linear
  --label_smoothing=0.1
  --neftune_noise_alpha=5
  --weight_decay=1e-2
  # ---- Inlet's own, no T2L counterpart ----
  --n_virtual_tokens=32          # matches the per-task prompt-tuning upper bound exactly
  --hypernet_head=per_slot
  --lr_world_scale=sqrt
  # ---- set from the GPU count above ----
  "--n_tasks_per_batch=$TASKS_PER_RANK"
  "--global_tasks_per_step=$GLOBAL_TASKS"
)

# Last flag wins would be nice to rely on, but upstream's parser builds a dict
# and the ordering is not documented. So: anything the caller passes REPLACES
# the recipe entry with the same name, and only then do we append.
ARGS=()
for r in "${RECIPE[@]}"; do
  key="${r%%=*}"
  override=0
  for u in "$@"; do
    if [[ "${u%%=*}" == "$key" ]]; then override=1; break; fi
  done
  if (( override == 0 )); then ARGS+=("$r"); fi
done
if (( $# > 0 )); then ARGS+=("$@"); fi

inlet_banner
echo " gpus/node $NGPU   nodes $NNODES   world $WORLD"
echo " global batch $GLOBAL_TASKS tasks/step  =  $TASKS_PER_RANK per rank x $WORLD ranks x $ACCUM accum"
echo "--------------------------------------------------------------"

# torchrun, not `accelerate launch`: no config file to drift between clusters,
# and accelerate reads RANK/WORLD_SIZE/LOCAL_RANK straight out of the env that
# torchrun sets. Identical code path at NGPU=1.
# "$PYBIN" -m torch.distributed.run, not bare `torchrun`: the bare name resolves
# off PATH and can be a DIFFERENT interpreter's launcher than the one common.sh
# settled on. Going through PYBIN makes the interpreter unambiguous.
exec "$PYBIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NGPU" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m inlet.train_inlet "$CONFIG" "${ARGS[@]}"
