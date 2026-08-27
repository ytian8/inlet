#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Inlet environment setup. VERIFIED end to end on a clean 2x A100-80GB PCIe box
# (RunPod, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404) on
# 2026-08-24. Every install below is here because a run failed without it, and
# the comments say what the failure looked like -- because the failures do not
# name the missing thing.
#
#   bash scripts/setup_env.sh                  # everything
#   SKIP_VLLM=1 bash scripts/setup_env.sh      # DON'T -- see note at step 5
#   VENV=/path/to/venv bash scripts/setup_env.sh
#
# What it does NOT do, on purpose:
#   * log in to Hugging Face  -> not needed; both models are ungated. If you
#                                 do need it, you run `huggingface-cli login`
#   * download model weights  -> step 7 prints the command; run it yourself
#   * download datasets       -> that is scripts/warm_cache.sh (no GPU, 20-40m)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
INLET_ROOT="$PWD"

say() { printf '\n=== %s ===\n' "$*"; }

say "1/8  what is this machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null \
  || echo "  no nvidia-smi -- CPU box. Fine for warm_cache.sh and the unit tests."
python3 -c "import sys; print('python', sys.version.split()[0])"

# Disk, FIRST, before anything writes a byte. See the long note in disk.sh:
# filling the volume does not raise, it wedges the shell, and the box then looks
# idle while it keeps billing. A 2026-08-24 run died exactly this way at 97%.
#
# disk.sh, NOT common.sh: common.sh exits when text-to-lora is missing, and
# step 2 below is what clones it. Sourcing common.sh here would make a clean
# machine fail setup on line one.
# shellcheck disable=SC1091
source scripts/disk.sh
require_disk "${INLET_NEED_GB:-60}" "$INLET_ROOT" || exit 1

say "2/8  the upstream checkout"
# Inlet is an OVERLAY on SakanaAI/text-to-lora: the 479/31 decontaminated split,
# task metadata, the hierarchical sampler and the eval harness are imported
# from it, never copied. That is what makes a Inlet number comparable to a T2L
# number.
if [[ -z "${T2L_ROOT:-}" ]]; then
  for c in "$INLET_ROOT/.." "$INLET_ROOT/../text-to-lora" "$INLET_ROOT/../text-to-lora-main" \
           "$INLET_ROOT/third_party/text-to-lora"; do
    [[ -d "$c/src/hyper_llm_modulator" ]] && { T2L_ROOT="$(cd "$c" && pwd)"; break; }
  done
fi
if [[ -z "${T2L_ROOT:-}" ]]; then
  echo "  not found -- cloning into third_party/"
  mkdir -p third_party
  git clone --depth 1 --recurse-submodules \
      https://github.com/SakanaAI/text-to-lora third_party/text-to-lora
  T2L_ROOT="$INLET_ROOT/third_party/text-to-lora"
fi
echo "  T2L_ROOT=$T2L_ROOT"
# src/fishfarm is a NESTED checkout: the importable package is at
# src/fishfarm/fishfarm. Without --recurse-submodules it is an empty directory
# and eval dies on `import fishfarm`.
[[ -d "$T2L_ROOT/src/fishfarm/fishfarm" ]] || {
  echo "  fetching the fishfarm submodule"
  git -C "$T2L_ROOT" submodule update --init --recursive
}

# Does THIS checkout fit THAT one? Pure AST, stdlib only, so it runs here --
# before the venv exists and before 13 GB of wheels are downloaded. Upstream is
# a moving target this repo does not pin; when it moves, the failure otherwise
# arrives after the 14 GB of model weights have loaded.
T2L_ROOT="$T2L_ROOT" python3 -m inlet.test_upstream_api || {
  echo
  echo "  The upstream API check failed. Installing on top of this would waste"
  echo "  20 minutes and ~13 GB before hitting the same problem at runtime."
  exit 1
}

say "3/8  virtualenv"
VENV="${VENV:-$INLET_ROOT/.venv}"

# A venv on a NETWORK filesystem does not fail -- it hangs. Installing torch
# writes tens of thousands of small files, which is the worst case for a FUSE
# mount, and pip then sits in `request_wait_answer` with its I/O counters frozen
# and no output. It looks exactly like a slow download, so the instinct is to
# wait, and on a rented GPU that costs money by the hour.
#
# Measured 2026-08-25 on a RunPod 2xA100 whose /workspace is MooseFS over FUSE:
# pip moved ZERO bytes in 5 s and had been stuck ~25 min; the identical command
# with VENV on the container's own overlay disk wrote 741 MB in 6 s.
#
# The weights are fine on the network volume -- a few large files read
# sequentially is what it is good at. It is the venv that must be local.
_venv_parent="$(dirname "$VENV")"
_venv_fs="$(df -PT "$_venv_parent" 2>/dev/null | awk 'NR==2 {print $2}')"
case "$_venv_fs" in
  fuse* | nfs* | cifs* | smb* | 9p | lustre | ceph | glusterfs)
    cat >&2 <<EOF

  WARNING: $VENV would live on a ${_venv_fs} filesystem (network/FUSE).
  Installing torch there does not error -- it HANGS, with pip's I/O counters
  frozen and nothing printed. Put the venv on local disk instead:

      VENV=/root/venv-inlet bash scripts/setup_env.sh

  The model weights can stay on the network volume; set HF_HOME to a path there
  before downloading them. Only the venv needs to be local.

EOF
    if [[ "${INLET_ALLOW_NETWORK_VENV:-0}" != "1" ]]; then
      echo "  refusing to build a venv on ${_venv_fs}. Set INLET_ALLOW_NETWORK_VENV=1 to override." >&2
      exit 1
    fi
    echo "  INLET_ALLOW_NETWORK_VENV=1 -- continuing anyway" >&2
    ;;
esac

if [[ ! -d "$VENV" ]]; then
  # 3.10-3.12 only. On 3.13+ vllm 0.11.1 and datasets 3.5.1 have no wheels and
  # pip silently tries to build from source, which fails much later.
  PY="${PYTHON:-}"
  [[ -z "$PY" ]] && for c in python3.12 python3.11 python3.10 python3; do
    command -v "$c" > /dev/null && { PY="$c"; break; }
  done
  echo "  creating $VENV with $PY ($("$PY" -V))"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip

say "4/8  torch, matched to the driver"
CUDA_TAG="${CUDA_TAG:-}"
if [[ -z "$CUDA_TAG" ]] && command -v nvidia-smi > /dev/null; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if (( DRV >= 580 )); then CUDA_TAG=cu130; else CUDA_TAG=cu128; fi
fi
CUDA_TAG="${CUDA_TAG:-cu128}"
echo "  using $CUDA_TAG wheels"
# ---------------------------------------------------------------------------
# THE ONE THAT COSTS AN HOUR IF YOU SKIP IT
#
# torchvision must come from the SAME index as torch, in the SAME command.
# Inlet does not import it -- but vLLM depends on it, so it WILL be installed in
# step 5 whatever you do here, and if that copy was built against a different
# torch the ABI mismatch does not raise ImportError: the process dies with
#     terminate called after throwing an instance of 'std::bad_alloc'
#     Aborted (core dumped)
# on `import transformers`, naming nothing. Installing it here, matched, means
# step 5 finds the requirement already satisfied and never picks its own.
#
# torchaudio is not needed by anything and is left uninstalled.
# ---------------------------------------------------------------------------
pip install -q "torch==2.9.0" "torchvision==0.24.0" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
pip uninstall -qy torchaudio 2>/dev/null || true

say "5/8  everything else"
# vllm is NOT eval-only. hyper_llm_modulator.utils reaches vllm_eval through
# its import chain, so `python -m inlet.train_inlet` fails with
# `ModuleNotFoundError: No module named 'vllm'` before the first training step.
# SKIP_VLLM=1 therefore gives you a box that cannot train either. It exists
# only for a CPU box used to warm datasets.
grep -v '^torch==' requirements.txt | grep -v '^#' | grep -v '^$' \
  | { [[ "${SKIP_VLLM:-0}" == "1" ]] && grep -v '^vllm==' || cat; } > /tmp/inlet_req.txt
pip install -q -r /tmp/inlet_req.txt

# evalplus, separately and with --no-deps -- see the long comment in
# requirements.txt. PyPI evalplus renamed SUCCESS -> _SUCCESS and the vendored
# fishfarm imports the old name, so the commit below (the one upstream's
# uv.lock pins) is not optional.
EVALPLUS_REV=1895d2f6aa8895044a7cf69defc24bd57695e885
pip uninstall -qy evalplus 2>/dev/null || true
pip install -q --no-deps "git+https://github.com/evalplus/evalplus@$EVALPLUS_REV"

say "6/8  verify (this is the part that matters)"
python - <<'PY'
import importlib, sys
ok = True
import torch
print(f"  torch        {torch.__version__}  cuda={torch.version.cuda}  "
      f"gpus={torch.cuda.device_count()}")
for m, want in [("transformers", "4.57.6"), ("accelerate", "1.14.0"),
                ("datasets", "3.5.1"), ("peft", "0.20.0")]:
    got = importlib.import_module(m).__version__
    flag = "" if got == want else f"   <-- expected {want}"
    if got != want: ok = False
    print(f"  {m:12s} {got}{flag}")
try:
    import vllm; print(f"  vllm         {vllm.__version__}")
except ImportError:
    print("  vllm         MISSING -- training will fail too, not just eval"); ok = False
# torchvision: vLLM depends on it, so after step 5 it IS installed again and
# that is fine. Presence was never the hazard -- a torchvision built against a
# DIFFERENT torch is, because the ABI mismatch aborts the process with
# std::bad_alloc instead of raising ImportError. So probe the thing that
# actually breaks, in a subprocess, and only fail on that.
import importlib.util, subprocess
if importlib.util.find_spec("torchvision") is None:
    print("  torchvision  absent (fine -- vllm normally pulls it back in)")
else:
    probe = subprocess.run([sys.executable, "-c", "import torchvision, transformers"],
                           capture_output=True)
    # Read the version from metadata, NOT by importing it: if the ABI really is
    # mismatched, importing torchvision here would abort this process too and
    # you would never see the diagnosis.
    import importlib.metadata as _md
    tv_ver = _md.version("torchvision")
    if probe.returncode == 0:
        print(f"  torchvision  {tv_ver} (vllm needs it, and it "
              f"imports alongside transformers -- do NOT uninstall it)")
    else:
        print(f"  torchvision  {tv_ver} ABI MISMATCH against "
              f"torch {torch.__version__}")
        print("               `import torchvision, transformers` died with "
              f"returncode {probe.returncode} -- this is the std::bad_alloc")
        print("               core dump. Reinstall torchvision from the SAME "
              "index as torch (see step 4), or")
        print("               `pip uninstall -y torchvision torchaudio` and "
              "accept that vLLM eval will not run.")
        ok = False
try:
    import flash_attn  # noqa: F401
    print("  flash_attn   present -- INLET_NO_FLASH_ATTN=1 still forces sdpa;")
    print("               never mix FA2 and sdpa numbers in one table")
except ImportError:
    print("  flash_attn   absent (expected; everything runs sdpa)")
# The whole training import chain in one shot -- this is what actually breaks.
sys.path.insert(0, ".")
import os
os.environ.setdefault("T2L_ROOT", os.environ.get("T2L_ROOT", ""))
try:
    import inlet.train_inlet  # noqa: F401
    print("  inlet.train_inlet imports cleanly")
except Exception as e:
    print(f"  inlet.train_inlet FAILED: {type(e).__name__}: {e}")
    ok = False
sys.exit(0 if ok else 1)
PY

say "7/8  disk after the install"
# The 14.5 GB model and ~5 GB of datasets still have to fit. If this number is
# under 25 GB, stop here and grow the volume -- the download will otherwise die
# partway and leave .incomplete blobs that make the next attempt look corrupted.
require_disk "${INLET_NEED_AFTER_GB:-25}" "$INLET_ROOT" || exit 1

say "8/8  next steps (you run these)"
cat <<MSG
  source $VENV/bin/activate
  export T2L_ROOT=$T2L_ROOT

  # 1. Hugging Face login -- NOT required. Both models below are ungated
  #    (verified: model_info(...).gated is False for each), so an anonymous
  #    download works. Only run this if you are behind a private mirror or
  #    hitting the anonymous rate limit, and type your own token if you do:
  #      huggingface-cli login

  # 2. weights, into the SAME cache the trainer will read. Source common.sh
  #    first -- it pins HF_HOME. Downloading with a different HF_HOME (or none)
  #    puts 14GB somewhere the trainer will not look, and the trainer then
  #    downloads its own second copy.
  #
  #    --exclude "*.bin" is NOT optional either: without it you get both the
  #    .bin and .safetensors copies, 24GB instead of 14GB, and a 50GB volume
  #    dies with "Disk quota exceeded" halfway through.
  source scripts/common.sh          # sets HF_HOME
  echo "downloading into \$HF_HOME"
  huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \\
      --exclude "*.bin" "*.pth" "*.gguf" "*.msgpack" "*.h5"
  huggingface-cli download Alibaba-NLP/gte-large-en-v1.5

  # 3. datasets: 20-40 min, no GPU, safe to run on a CPU box in a tmux
  ./scripts/warm_cache.sh

  # 4. pre-flight (~20 min). Do not skip before a multi-day run.
  ./scripts/smoke.sh 1

  # 5. the real thing. First argument is the GPU count; that is the only
  #    thing that changes between machines.
  ./scripts/train.sh 8 --run_name=full \\
      --checkpoint_steps=4000,20000,40000,80000,147500

  # 6. eval
  ./scripts/eval.sh train_outputs/hyper_lora/full/hypermod_inlet.pt
MSG
