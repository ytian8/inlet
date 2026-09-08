#!/usr/bin/env bash
# warm_cache.sh, paced against the Hugging Face API rate limit.
#
#   ./scripts/warm_cache_paced.sh
#
# The hub allows a fixed window of requests and says so in the headers:
#
#   RateLimit:        "api";r=0;t=7                       <- 0 left, resets in 7s
#   RateLimit-Policy: "fixed window";"api";q=500;w=300    <- 500 per 300s
#
# warm_datasets does not back off: on 429 it records the dataset as failed and
# moves on, so one pass with 12 workers burns the whole window in ~85 datasets
# and then "fails" the remaining 415 in a few seconds. Nothing is wrong with
# the network or the datasets -- rerunning the same command immediately just
# fails all of them again, which reads like a hard error and is not one.
#
# So: repeated --only-failed passes, sleeping out the window between them.
# Each pass gets through however many the budget allows and rewrites the
# failure list; the loop stops when the list is empty or stops shrinking.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILJSON="${FAILJSON:-$PWD/third_party/text-to-lora/warm_failures.json}"
WINDOW="${WINDOW:-310}"          # 300s window + a little slack
MAX_PASSES="${MAX_PASSES:-12}"
LOG="${LOG:-/root/warm_paced.log}"

# warm_datasets writes {"failed": [name, ...]} -- NOT a bare list. Taking
# len() of the dict gives 1 no matter how many datasets are in it, which makes
# the loop below think one pass fixed everything.
count_failed() {
  [[ -f "$FAILJSON" ]] || { echo 0; return; }
  python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0); raise SystemExit
if isinstance(d, dict):
    d = d.get("failed", d.get("failures", []))
print(len(d) if hasattr(d, "__len__") else 0)' "$FAILJSON" 2>/dev/null || echo 0
}

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Pass 1 is the full set unless a failure list already exists.
if [[ -f "$FAILJSON" ]] && [[ "$(count_failed)" != "0" ]]; then
  say "resuming from $FAILJSON ($(count_failed) datasets)"
else
  say "first pass: everything"
fi

prev=-1
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
  # warm_cache.sh only WRITES the failure file when something fails, so a pass
  # in which everything succeeds leaves the previous pass's file untouched.
  # Reading it afterwards then reports the old count, the loop sees "no
  # progress", and it stops with STALLED having actually just finished. Move
  # the file aside first: whatever is there after the pass was written by THIS
  # pass, and its absence means zero failures.
  IN=()
  if [[ -f "$FAILJSON" ]]; then
    mv "$FAILJSON" "$FAILJSON.in"
    IN=(--only-failed "$FAILJSON.in")
  fi

  say "pass $pass  (WORKERS=${WORKERS:-4})"
  # Keep the FAIL lines. They carry the reason (429, PermissionError, a broken
  # cache dir), and truncating them is what makes a stall undiagnosable.
  WORKERS="${WORKERS:-4}" ./scripts/warm_cache.sh "${IN[@]}" 2>&1 \
    | grep -av "examples/s" | grep -aE "FAIL|ok=|warmed|failures|Error" | tail -60 \
    | tee -a "$LOG"

  left="$(count_failed)"
  say "pass $pass done -- $left still failing"

  if [[ "$left" == "0" ]]; then
    say "ALL WARMED after $pass pass(es)"
    rm -f "$FAILJSON.in"
    echo "PACED_EXIT=0" | tee -a "$LOG"
    exit 0
  fi
  if [[ "$left" == "$prev" ]]; then
    say "STALLED: $left datasets failed twice in a row with no progress."
    say "That is no longer the rate limit. The FAIL lines above carry the reason;"
    say "debris from an interrupted run (a half-written cache dir, or a stale"
    say ".incomplete blob) is the usual cause -- delete it and rerun."
    echo "PACED_EXIT=2" | tee -a "$LOG"
    exit 2
  fi
  prev="$left"

  say "sleeping ${WINDOW}s for the rate-limit window"
  sleep "$WINDOW"
done

say "gave up after $MAX_PASSES passes, $(count_failed) still failing"
echo "PACED_EXIT=3" | tee -a "$LOG"
exit 3
