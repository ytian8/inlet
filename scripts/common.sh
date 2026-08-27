# Sourced by every script in this directory. Not executable on its own.
#
# One job: make `python -m inlet.<anything>` work, on any cluster, with the same
# environment on every rank.
set -euo pipefail

INLET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INLET_ROOT

# ---- where is the upstream text-to-lora checkout? --------------------------
# inlet/_env.py does the same search in Python; this is only for the messages and
# for anything shell-side that needs the path. Set T2L_ROOT to skip the search.
if [[ -z "${T2L_ROOT:-}" ]]; then
  for c in "$INLET_ROOT" "$INLET_ROOT/.." "$INLET_ROOT/../text-to-lora" \
           "$INLET_ROOT/../text-to-lora-main" "$INLET_ROOT/third_party/text-to-lora"; do
    if [[ -d "$c/src/hyper_llm_modulator" ]]; then T2L_ROOT="$(cd "$c" && pwd)"; break; fi
  done
fi
if [[ -z "${T2L_ROOT:-}" || ! -d "$T2L_ROOT/src/hyper_llm_modulator" ]]; then
  echo "ERROR: cannot find the text-to-lora checkout." >&2
  echo "  git clone https://github.com/SakanaAI/text-to-lora" >&2
  echo "  export T2L_ROOT=/abs/path/to/text-to-lora" >&2
  exit 1
fi
export T2L_ROOT

# src/fishfarm is a NESTED checkout -- the importable package is at
# src/fishfarm/fishfarm, so `src` alone is not enough.
export PYTHONPATH="$INLET_ROOT:$T2L_ROOT/src:$T2L_ROOT/src/fishfarm${PYTHONPATH:+:$PYTHONPATH}"

# Inlet's own outputs stay in THIS repo, so `git status` in the upstream checkout
# stays clean and a run is easy to find again.
export INLET_OUTPUT_ROOT="${INLET_OUTPUT_ROOT:-$INLET_ROOT/train_outputs}"
mkdir -p "$INLET_OUTPUT_ROOT"

# ---- environment facts, all of them load bearing ---------------------------
# piqa (ybisk/piqa) is a script-based HF dataset: datasets>=3 loads it but will
# not run its loader script without explicit consent. Exactly one of the ~510
# repos fails without this, and the error names neither piqa nor consent.
export HF_DATASETS_TRUST_REMOTE_CODE=1
# Tokenizer threads x DDP ranks x dataloader workers oversubscribes every core.
export TOKENIZERS_PARALLELISM=false
# FlashAttention2 is upstream's default. No environment used here has the
# pod has a flash_attn wheel for torch 2.9/cu13, so the default here is sdpa.
# Set INLET_NO_FLASH_ATTN=0 ONLY on a box where `import flash_attn` works -- and
# then never mix its numbers into a table with sdpa numbers (see docs).
export INLET_NO_FLASH_ATTN="${INLET_NO_FLASH_ATTN:-1}"
# Datasets are warmed once and then read offline, so a run can never stall on
# the hub six hours in. The HUB is a different matter: gte-large-en-v1.5 loads
# with trust_remote_code and resolves its modelling code through the hub, and
# HF_HUB_OFFLINE=1 turns that into
#     huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested
#     files in the disk cache and outgoing traffic has been disabled
# which reads like a missing download but is a disabled lookup. So: datasets
# offline by default, hub online by default. Set HF_HUB_OFFLINE=1 yourself only
# on a box you know has every artefact cached.
# ---------------------------------------------------------------------------
# WHERE THE 14 GB OF WEIGHTS LIVE. Pin this; do not let it default.
#
# Hugging Face defaults to $HOME/.cache/huggingface. Two things go wrong when
# you leave it there:
#   * $HOME is usually a small quota'd filesystem on a cluster, and the base
#     model plus the dataset cache is ~45 GB;
#   * if you ever download the model with HF_HOME pointed somewhere else -- say
#     you set it by hand once, in one shell, to get the download onto the big
#     volume -- then a later run in a shell WITHOUT that variable does not
#     reuse it. It silently starts a second 14 GB download into the default
#     location. That is how a volume with 13 GB free hits
#     "OSError: I/O error: Disk quota exceeded" thirty seconds into a job.
# One place, exported for every process. Override it before sourcing this file
# if your site keeps caches somewhere shared.
export HF_HOME="${HF_HOME:-$INLET_ROOT/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# torch cu13 wheels next to a vLLM built against cu12: env.sh in the upstream
# checkout LD_PRELOADs the cu12 cublas/cudart. Only eval needs vLLM.
if [[ -f "$T2L_ROOT/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$T2L_ROOT/env.sh" > /dev/null 2>&1 || true
fi

# Weights & Biases. The trainer calls accelerator.init_trackers(log_with="wandb")
# unconditionally, so without this a fresh box dies before the first step with
#     wandb.errors.errors.UsageError: No API key configured. Use `wandb login`.
# Offline mode writes the same run to ./wandb/ and needs no account. Set
# WANDB_MODE=online (and run `wandb login`) if you want the dashboard.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

CONFIG="${CONFIG:-configs/hyper_lora_decontam_lol_tasks.yaml}"
export CONFIG

# ---- the interpreter ------------------------------------------------------
# Activate the repo venv if there is one and the caller has not already done
# it. Without this, forgetting `source .venv/bin/activate` runs the scripts
# under the SYSTEM python, and the first symptom is
#     ModuleNotFoundError: No module named 'wandb'
# from inside torchrun -- which names neither the venv nor the real problem.
# Observed exactly that way on 2026-08-24.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV:-$INLET_ROOT/.venv}/bin/activate" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${VENV:-$INLET_ROOT/.venv}/bin/activate"
  echo "  (activated ${VENV:-$INLET_ROOT/.venv})" >&2
fi

# `python` does not exist on every distro; `python3` always does. Inside the
# venv both are present and identical.
PYBIN="${PYBIN:-$(command -v python || command -v python3)}"
[[ -n "$PYBIN" ]] || { echo "ERROR: no python interpreter on PATH" >&2; exit 1; }
export PYBIN

# Fail here, with a sentence that says what to do, rather than 300 lines deep
# inside torchrun. This is the cheapest possible check that the environment the
# scripts are about to use is the one setup_env.sh built.
if ! "$PYBIN" -c "import torch, wandb" > /dev/null 2>&1; then
  echo "ERROR: $PYBIN cannot import torch and wandb." >&2
  echo "  This is almost always the wrong interpreter -- the venv was never" >&2
  echo "  created, or PYBIN/VENV point somewhere else. Run:" >&2
  echo "      bash scripts/setup_env.sh" >&2
  echo "  or set VENV=/path/to/venv (or PYBIN=/path/to/python) and retry." >&2
  exit 1
fi

inlet_banner() {
  echo "--------------------------------------------------------------"
  echo " INLET_ROOT          $INLET_ROOT"
  echo " T2L_ROOT          $T2L_ROOT"
  echo " INLET_OUTPUT_ROOT   $INLET_OUTPUT_ROOT"
  echo " config            $CONFIG"
  echo " flash-attn off    $INLET_NO_FLASH_ATTN   (1 = use sdpa)"
  echo " visible GPUs      ${CUDA_VISIBLE_DEVICES:-<all>}"
  echo "--------------------------------------------------------------"
}

# Disk helpers live in their own file because setup_env.sh needs them BEFORE it
# has cloned text-to-lora -- and this file exits if T2L_ROOT is missing.
# shellcheck disable=SC1091
source "$INLET_ROOT/scripts/disk.sh"
