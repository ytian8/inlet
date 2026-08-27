#!/usr/bin/env bash
# Disk checks. Sourced by common.sh AND, separately, by setup_env.sh.
#
# This is a separate file on purpose: setup_env.sh has to check disk before it
# clones text-to-lora, and common.sh hard-exits when that checkout is missing.
# Nothing here needs T2L_ROOT, a venv, or python.
#
#   source scripts/disk.sh
#   require_disk 60 /path        # returns 1 (and explains) when short

: "${INLET_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --------------------------------------------------------------------------- #
# Disk. This is not a nicety -- it is the failure mode that cost the most time.
#
# OBSERVED 2026-08-24 on a RunPod A100 box: `setup_env.sh` filled the volume to
# 97%. What that looks like is NOT "no space left on device". It looks like the
# machine dying: the web terminal stops echoing, gotty accepts the websocket and
# then never writes a byte, `nvidia-smi` shows 0% on everything, and the pod sits
# there billing by the hour looking merely idle. A full disk takes the shell down
# with it, so you cannot even log in to read the error. Check BEFORE, not after.
#
# Where the space goes, measured:
#     venv, torch 2.9+cu128 (the nvidia-* wheels dominate)   ~13 GB
#     Mistral-7B-Instruct-v0.2, safetensors only              14.5 GB
#     Alibaba-NLP/gte-large-en-v1.5                            1.7 GB
#     raw HF datasets + upstream's TRANSFORMED_DS_DIR          ~5 GB
#     pip's own wheel cache, if you let it keep one           ~8 GB
#     checkpoints (13.8M params x fp32 = 55 MB each)           <1 GB
#   ------------------------------------------------------------------
#     with the pip cache                                      ~43 GB
#     without it (PIP_NO_CACHE_DIR=1, which common.sh sets)    ~35 GB
#
# 60 GB free is the floor; a 100 GB volume is the comfortable answer.
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

inlet_free_gb() { df -PB1G "${1:-$INLET_ROOT}" 2>/dev/null | awk 'NR==2 {print $4+0}'; }
inlet_fstype()  { df -PT  "${1:-$INLET_ROOT}" 2>/dev/null | awk 'NR==2 {print $2}'; }

# `df` DOES NOT SEE A VOLUME QUOTA. Measured 2026-08-24 on a RunPod A100: the
# pod's 120 GB volume is a mount of a shared network filesystem
# (mfs#ca-mtl-3.runpod.net:9421), and `df` reports the whole cluster --
# 755 TB total, 292 TB free -- while the pod is limited to its 120 GB quota.
# So the check below reads "292000 GB free" and never fires, on exactly the
# machine where a previous run filled its volume to 97% and wedged the shell.
#
# There is no portable way to read the quota from inside the container, so this
# says so out loud rather than reporting a green tick. Anything above 5 TB is
# not a disk you were given; it is somebody's cluster.
inlet_quota_warning() {
  local free="$1" where="$2" fs
  (( free < 5000 )) && return 0
  fs="$(inlet_fstype "$where")"
  cat >&2 <<EOF

  WARNING: df reports ${free} GB free on $where (filesystem type: ${fs:-unknown}).
  That is a shared or network filesystem, and df is showing you the whole thing,
  not your share of it. If this is a cloud pod, your real limit is the VOLUME
  QUOTA, which df cannot see and this check therefore cannot enforce.

  Check it where it is actually reported -- the pod dashboard's volume gauge,
  not the container one -- and confirm you have 60+ GB before continuing.
  Filling it does not raise a clean error: it wedges the shell, and the machine
  then looks idle while it keeps billing.

EOF
}

require_disk() {
  local need="${1:-60}" where="${2:-$INLET_ROOT}" free
  free="$(inlet_free_gb "$where")"
  if [[ -z "$free" ]]; then
    echo "  disk: could not read df for $where -- skipping the check" >&2
    return 0
  fi
  echo "  disk: ${free} GB free on $(df -P "$where" | awk 'NR==2 {print $6}') (need ${need} GB)"
  inlet_quota_warning "$free" "$where"
  if (( free < need )); then
    cat >&2 <<EOF

  REFUSING TO CONTINUE: ${free} GB free, ${need} GB needed.

  Running out of disk here does not produce a clean error. It wedges the shell
  and the machine looks idle while it keeps billing. Free space or resize the
  volume first.

  Biggest things that are safe to delete:
    rm -rf "\$HF_HOME/hub/models--"*"/blobs/"*.incomplete   # aborted downloads
    pip cache purge
    rm -rf ~/.cache/uv ~/.cache/pip
  Then check what is actually large:
    du -xh --max-depth=2 "$where" 2>/dev/null | sort -h | tail -20

  Set INLET_SKIP_DISK_CHECK=1 to override (it will not end well).
EOF
    [[ "${INLET_SKIP_DISK_CHECK:-0}" == "1" ]] || return 1
    echo "  INLET_SKIP_DISK_CHECK=1 -- continuing anyway" >&2
  fi
  return 0
}
