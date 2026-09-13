#!/usr/bin/env bash
# Run after the documented setup, preparation and smoke checks.
# Forward the same --checkpoint, --manifest and --out arguments as the Python queue.
# Shared plan is created BEFORE workers start; each task has one exclusive worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYBIN="${PYBIN:-python}"
visible="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a devices <<< "$visible"
if [[ ${#devices[@]} -ne 2 || -z "${devices[0]}" || -z "${devices[1]}" || "${devices[0]}" == "${devices[1]}" ]]; then
  echo "Require exactly two distinct allocated GPUs in CUDA_VISIBLE_DEVICES; got '$visible'" >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    --execute|--only-task|--only-task=*)
      echo 'Do not pass --execute or --only-task to the two-GPU launcher' >&2
      exit 2 ;;
  esac
done
# Freeze all shared files serially. Existing matching files are not rewritten.
"$PYBIN" -m inlet.run1_diagnostic "$@"
worker() {
  local gpu="$1" first="$2" second="$3"
  shift 3
  for task in "$first" "$second"; do
    echo "GPU $gpu: $task"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m inlet.run1_diagnostic "$@" --execute --only-task "$task" || return $?
  done
}
worker "${devices[0]}" arc_challenge gsm8k "$@" &
worker0=$!
worker "${devices[1]}" winogrande mbpp "$@" &
worker1=$!
status0=0
status1=0
wait "$worker0" || status0=$?
wait "$worker1" || status1=$?
if [[ $status0 -ne 0 || $status1 -ne 0 ]]; then
  echo "Worker failures: GPU ${devices[0]}=$status0; GPU ${devices[1]}=$status1. Inspect job logs; completed jobs are resumable." >&2
  exit 1
fi
echo 'Both GPU workers completed.'
