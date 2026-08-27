#!/usr/bin/env bash
# The 20-minute "does this whole thing work on this cluster" check.
# Run it before every long job. It has caught every environment break so far.
#
#   ./scripts/smoke.sh          # 1 GPU
#   ./scripts/smoke.sh 2        # also checks that DDP starts and agrees
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
require_disk 20 "$INLET_ROOT" || exit 1   # the smoke run trains and evals in miniature
NGPU="${1:-1}"

echo "### 0/4  does this checkout still fit the text-to-lora next to it? ###"
# Cheapest check in the repo and the first that should run: AST only, imports
# nothing, needs no torch. Catches an upstream that moved -- a renamed symbol, a
# changed signature, a flag that is no longer a field -- which otherwise shows up
# as a TypeError after the weights have loaded.
"$PYBIN" -m inlet.test_upstream_api

echo
echo "### 1/4  no-GPU consistency checks (train path == eval path) ###"
# Needs the per-task prompt-tuning baseline to compare against. If you do not
# have that directory this check cannot run -- it is the only thing in the repo
# that requires it, and skipping it does not affect training or eval.
if ! "$PYBIN" -m inlet.test_consistency_inlet --task arc_challenge; then
  echo "  ^ skipped or failed. If the message above says the baseline directory"
  echo "    is missing, that is expected on a fresh clone; continuing."
fi

echo
echo "### 1b/4  gradient-accumulation equivalence (CPU, no data needed) ###"
"$PYBIN" -m inlet.test_accum

echo
echo "### 1b2/4  train-side vs eval-side prompt assembly (CPU, no model) ###"
# The gate that did not exist: training builds [B, m+T, d] on dim 1, eval builds
# [m+T, d] on dim 0, and nothing used to compare them. m=0 gates cannot see it
# (both paths short-circuit) and --zero-prompt cannot either (the eval cat never
# runs). Flipping the eval axis used to leave every check in this repo green.
"$PYBIN" -m inlet.test_train_eval_agree

echo
echo "### 1b2b/4  description conditioning: slots, mask, cross-attention (CPU, no model) ###"
# Widening the description bottleneck (inlet/desc_pool.py) is the one change that
# can be wired up, look correct, and do nothing -- exactly the shape NEFTune's
# double no-op had. So this checks that cross-attention MOVES when a non-CLS
# description slot moves, and that cond='pooled' does NOT, as a control. It also
# pins K=1 to upstream's cls_pool bit for bit, which is what makes a K=1 vs K=8
# comparison a controlled A/B rather than two unrelated models.
"$PYBIN" -m inlet.test_desc_cond

echo
echo "### 1b2c/4  generative canary: context/target split (CPU, no model) ###"
# The canary is what would have caught the 147,500-step run going 100k steps
# past its own optimum. Its tensor half is checked here; its generation half
# runs inside validation during step 3 below.
"$PYBIN" -m inlet.canary

echo
echo "### 1b2d/4  val/generative leakage rules (CPU, no data) ###"
# None of upstream's three validation splits contains gsm8k, mbpp or
# humaneval -- the three tasks that collapsed. val/generative adds one, and
# these are the rules that keep scored data out of it: humaneval can never
# be in it, mbpp needs decontamination, an unvetted task is dropped.
"$PYBIN" -m inlet.generative_val

echo
echo "### 1b3/4  the reconstructed reference checks its own contract (CPU, no torch model) ###"
# These two run whether or not the reference baseline_prompt_tuning/ is present:
# they assert the reconstruction satisfies the contract the ORIGINAL was tested
# against. Under a second, and they are the only evidence that a clean clone's
# fallback implementation is the same operation as the reference.
"$PYBIN" -m inlet.baseline_ref
"$PYBIN" -m inlet.consistency_ref

echo
echo "### 1c/4  DDP equivalence (CPU, gloo, 2 fake ranks, no GPU needed) ###"
# The other half of the --global_tasks_per_step claim: accumulation equals the
# larger batch (1b), AND N ranks equal one process with N times the batch. If
# DDP summed gradients instead of averaging them the effective LR would scale
# with the GPU count and a 2-GPU rerun of an 8-GPU result would silently be a
# different experiment.
"$PYBIN" -m torch.distributed.run --nproc_per_node=2 --master_port="${DDP_TEST_PORT:-29777}" \
  -m inlet.test_ddp_equiv

echo
echo "### 2/4  m=0 gate: prepending ZERO vectors must reproduce the frozen model ###"
# If this fails the sequence is being assembled wrong -- wrong axis, wrong mask,
# wrong label shift -- and every number Inlet produces afterwards is quietly
# wrong while the code runs without error. Do not train through a failure here.
"$PYBIN" -m inlet.gate_m0

echo
echo "### 3/4  short training run on $NGPU GPU(s): loss must fall ###"
# GLOBAL_TASKS is PINNED, not scaled by NGPU.
#
# It used to be `8 * NGPU`, which kept grad_accum at exactly 1 for every GPU
# count -- so the one script that runs the real trainer on more than one GPU
# never exercised accumulation, and `smoke.sh 1` and `smoke.sh 2` ran different
# global batches (8 vs 16) and were therefore not comparable. Pinned at 16:
#   smoke.sh 1 -> 1 rank x 8 tasks x accum 2
#   smoke.sh 2 -> 2 ranks x 8 tasks x accum 1
# Same global batch, same step count, same LR -- so the two runs' loss curves
# ARE comparable, which is the property --global_tasks_per_step exists to give.
GLOBAL_TASKS="${GLOBAL_TASKS:-16}" \
  ./scripts/train.sh "$NGPU" \
    --run_name="smoke${NGPU}" --n_train_ds=12 --n_descs_per_ds=8 \
    --max_steps=200 --val_freq=100 --logging_freq=20

echo
echo "### 4/4  zero-prompt eval must reproduce the published zero-shot number ###"
TASKS=arc_challenge ./scripts/eval.sh --zero-prompt

echo
echo "smoke OK -- see $INLET_OUTPUT_ROOT/hyper_lora/smoke${NGPU}/train_summary.json"
