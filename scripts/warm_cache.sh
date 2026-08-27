#!/usr/bin/env bash
# Download + tokenize + cache all 510 datasets (479 train + 31 eval).
#
#   ./scripts/warm_cache.sh                 # everything, 12 workers
#   ./scripts/warm_cache.sh --n-train 12    # smoke subset
#   ./scripts/warm_cache.sh --only-failed warm_failures.json
#
# I/O bound, no GPU. Run it on a CPU box or in a tmux BEFORE renting GPU time;
# ~500 small HF repos sequentially is 1-3h, with 12 workers 20-40 min. Every
# failure is recorded so a second pass only re-fetches what actually failed.
#
# The cache lands in $T2L_ROOT/data/transformed_datasets. On a cluster with a
# shared filesystem, warm it once and every node sees it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/common.sh
# Warming is the one thing that MUST reach the hub.
export HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0
inlet_banner
exec "$PYBIN" -m inlet.warm_datasets --workers "${WORKERS:-12}" "$@"
