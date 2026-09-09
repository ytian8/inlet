# Run 2 — evaluate the Run 1 checkpoints properly

**This file is the whole assignment.** Read it start to finish before running
anything, then work through §1 to §7 in order and report §8. It is evaluation
only — one GPU, no training, roughly a day of wall clock. Nothing in it is
time-boxed or budget-boxed: run everything, and do not substitute a cheaper
measurement for one that is specified.

Repo: <https://github.com/ytian8/inlet>, commit `fe6e67b` or later.
Prerequisite: the checkpoints and environment from `docs/RUN1.md`.

Run 1 trained a 16,000-step model and swept prompt scale on four benchmarks with
a **debug** protocol: one task description, no junk-description control. That
produced a step/scale curve and one surprise. It did not produce a number that
can go in a paper.

This run does. It is evaluation only — **no training** — so it needs one GPU and
it can be run start to finish without supervision. Budget it at roughly a day of
wall clock and do not trim it to save time.

Follow this file top to bottom. `docs/RUN1.md` is still the reference for
environment setup, the shell preamble, and the traps; read its §2 and §3 before
starting and use the same five-line preamble in every shell here.

---

## 0. What Run 1 found, and why this run exists

Numbers below are Run 1's, all at the debug protocol, all selected on the test
set. Treat them as directions, not results.

**The optimal prompt scale is task-dependent, and the two task families want
opposite ends of the range.** At step 16,000:

| | s0.35 | s0.5 | s0.7 | s1.0 | frozen |
|---|---|---|---|---|---|
| arc_challenge | 67.15 | 70.39 | 71.67 | **72.95** | 65.70 |
| humaneval | 40.85 | **43.29** | 26.83 | 16.46 | 39.63 |
| gsm8k | 39.73 | **40.11** | 38.97 | 21.99 | 39.95 |
| mbpp | **44.71** | 39.95 | 36.51 | 2.38 | 42.86* |

\* mbpp's frozen baseline was never measured on that box; it is quoted from
RUN1.md. §2 below fixes that.

The mechanism is length collapse. A large-magnitude prompt truncates
generation — humaneval's completions go from 136 median words (frozen) to 16 at
s1.0 — and the accuracy drop tracks the truncation. **arc_challenge emits 8
words**, so truncation cannot hurt it and it keeps the full conditioning benefit
at s1.0. So the split is not "multiple choice vs generative"; it is **short
output vs long output**, and answer format is only a proxy for that.

Three things make those numbers unreportable, and this run fixes all three.

1. **No junk-description control anywhere.** `probe_prompt` shows the generated
   prompts do vary with the description — cos(real, junk) fell from 0.9999 at
   step 1,000 to 0.6705 at step 16,000 — but a difference between prompts is not
   a difference in accuracy. On the previous 147,500-step model the benchmark gap
   was +0.41 points on humaneval, i.e. nothing. **This is the single most
   important measurement in this run.**
2. **Six of ten benchmarks were never run**, and all six are short-output ones.
   The "two families" claim currently rests on `arc_challenge` alone.
3. **arc_challenge's optimum was not bracketed.** s1.0 was both the best scale
   and the largest one tried, and the column was still rising at step 16,000.

---

## 1. Setup

Same repo, same environment as Run 1. If the box still has Run 1's checkout and
caches, you need nothing but a `git pull`.

```bash
cd /root/inlet
git pull                                  # need fe6e67b or later
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # BEFORE common.sh -- RUN1.md §3

# Run 1 wrote to whatever INLET_OUTPUT_ROOT was set to at the time, and its
# RESULTS.md records `train_outputs/`, i.e. the default root -- NOT the
# /root/outputs that RUN1.md §5 suggests. Find where its checkpoints actually
# are and point at that, so new results land beside the old ones.
for R in "$PWD/train_outputs" /root/outputs "$INLET_OUTPUT_ROOT"; do
  if [[ -f "$R/hyper_lora/run1/hypermod_inlet_step16000.pt" ]]; then
    export INLET_OUTPUT_ROOT="$R"; break
  fi
done
source scripts/common.sh

echo "INLET_OUTPUT_ROOT=$INLET_OUTPUT_ROOT"
ls "$INLET_OUTPUT_ROOT/hyper_lora/run1/"
```

**If that loop found nothing, stop and locate Run 1's output directory by hand**
before going further — every path below hangs off `INLET_OUTPUT_ROOT`, and a
wrong one produces an empty results directory rather than an error.

`git pull` matters: `inlet.desc_control` (§3) did not exist before `fe6e67b`, and
before it `reported_protocol` in the result JSON counted *arms* rather than
descriptions, so every Run 1 file with 3+ scales is stamped
`reported_protocol: true` while being a one-description debug run. Do not trust
that field in any file written before this commit.

**Two checkpoints are in scope, and only these two:**

```bash
CKPT_SEL=$INLET_OUTPUT_ROOT/hyper_lora/run1/hypermod_inlet_best_val_unseen.pt
CKPT_END=$INLET_OUTPUT_ROOT/hyper_lora/run1/hypermod_inlet_step16000.pt
```

`best_val_unseen` is step 11,000 — the checkpoint the run selected on validation
loss, which is the honest reportable choice. `step16000` is the last checkpoint,
also defensible because it is fixed in advance. **`step4000` and `step1000` are
not in scope for any reported number**: Run 1 found them best on the test set,
which is exactly why they cannot be reported. Use them freely for diagnosis and
say that is what you did.

Confirm both exist and note their steps before starting:

```bash
ls -la $CKPT_SEL $CKPT_END
for C in $CKPT_SEL $CKPT_END; do
  python -c 'import torch, sys, os
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = d["config"]
print(os.path.basename(sys.argv[1]), "step=", c["curstep"],
      "desc_slots=", c["desc_slots"], "cond=", c["cond"])' "$C"
done
```

`desc_slots= 8 cond= cross` should print for both. If it does not, the checkpoint
came from a different configuration than Run 1's and nothing below is
comparable.

If `best_val_unseen` turns out not to be step 11,000, say so and report the step
it actually is — the rest of this run does not change, but the writeup does.

---

## 2. Baselines first (~40 min)

Run 1 measured the frozen model on three tasks and reproduced its references
exactly (65.70 / 39.95 / 39.63). The other seven were never measured on that box,
so no gain over frozen can be quoted for them.

```bash
TASKS="arc_easy boolq hellaswag openbookqa piqa winogrande mbpp" \
    ./scripts/eval.sh --zero-prompt
```

Published references, for comparison:

```
arc_challenge 65.70   arc_easy 77.48   boolq 71.56   hellaswag 49.67
openbookqa 55.00      piqa 73.01       winogrande 45.54
gsm8k 40.71           mbpp 44.44       humaneval 37.80
```

**If any measured baseline is more than ~1 point off its reference, stop and
report it.** RUN1.md's rule applies: if the eval path is wrong, nothing
downstream means anything. A small gap is normal — Run 1 measured gsm8k 39.95
against a published 40.71.

---

## 3. The measurement that matters: real minus junk (~4-6 h)

The reported protocol is **three real descriptions and three junk ones**, which
is `eval.sh` with **neither** `--max-eval-descs` **nor** `--skip-random-descs`.
Everything else in this section is secondary to it.

Run all ten benchmarks, both checkpoints, at the two scales that Run 1's curve
makes interesting. Each command is one scale so that a task's tokenisation is
done once per scale — see the trap in RUN1.md §6.

Six arms per task in one engine build, so `hellaswag` (10,042 examples) and
`boolq` (3,270 long passages) dominate the clock. Expect **12-20 hours** for all
four passes. Run it in this order so an interruption still leaves the cells that
matter — fast tasks first, and the last checkpoint at s0.5 first, since that is
the primary reportable cell:

```bash
FAST="arc_challenge arc_easy openbookqa piqa winogrande gsm8k mbpp humaneval"
SLOW="boolq hellaswag"

tmux new -s run2                 # or `setsid tmux new -d -s run2 "..."` -- RUN1.md §5

for CKPT in $CKPT_END $CKPT_SEL; do
  for S in 0.5 1.0; do
    TASKS="$FAST" EXTRA_EVAL_ARGS="--prompt-scales $S" ./scripts/eval.sh $CKPT
  done
done

for CKPT in $CKPT_END $CKPT_SEL; do
  for S in 0.5 1.0; do
    TASKS="$SLOW" EXTRA_EVAL_ARGS="--prompt-scales $S" ./scripts/eval.sh $CKPT
  done
done
```

`desc_control` and `sweep_report` both read whatever JSON exists, so you can run
them at any point to see how far it has got. **Do not stop early because the
numbers already look good or already look bad** — the eight-task partial and the
ten-task complete are different measurements and only the second is reportable.

Then:

```bash
python -m inlet.desc_control $INLET_OUTPUT_ROOT/eval_results_inlet
python -m inlet.desc_control $INLET_OUTPUT_ROOT/eval_results_inlet --scale 0.5 \
       --csv $INLET_OUTPUT_ROOT/desc_control_s0.5.csv
python -m inlet.desc_control $INLET_OUTPUT_ROOT/eval_results_inlet --scale 1.0 \
       --csv $INLET_OUTPUT_ROOT/desc_control_s1.0.csv
```

`desc_control` refuses to average across scales and refuses files that lack junk
arms or have fewer than three real descriptions, so if it prints a table at all,
the table is the reported protocol.

**Report the gap per task and its mean, at each (checkpoint, scale). Report it
whatever it is.** A gap near zero is a real and publishable finding about this
architecture; hiding it or reporting only the tasks where it is positive is not.
Do not select the scale or the checkpoint by which gives the larger gap — report
all four cells.

---

## 4. Is `arc_challenge` a special case, or is the whole short-output family? (~2 h)

§3 already gives every task at s0.5 and s1.0 under the reported protocol, which
answers this. Pull it out explicitly:

```bash
python -m inlet.sweep_report $INLET_OUTPUT_ROOT/eval_results_inlet --scale 1.0 --paper
python -m inlet.sweep_report $INLET_OUTPUT_ROOT/eval_results_inlet --scale 0.5 --paper
```

For each of the ten tasks, report **score at s1.0 minus score at s0.5**, next to
the task's **zero-prompt median completion length** from §2. The prediction is
that this difference is positive for every short-output task (arc_challenge,
arc_easy, boolq, hellaswag, openbookqa, piqa, winogrande) and negative for every
long-output one (gsm8k, mbpp, humaneval), and that its magnitude tracks
completion length.

Say plainly whether it holds. If some short-output task prefers s0.5, that is
the interesting result, not a nuisance — name it and give its completion length.

---

## 5. Bracket the scale for short-output tasks (~1.5 h)

s1.0 was the edge of Run 1's sweep and `arc_challenge` was still improving there.
Debug protocol is fine here — this is a shape question, not a reported number:

```bash
TASKS="arc_challenge openbookqa piqa winogrande" \
EXTRA_EVAL_ARGS="--prompt-scales 1.3,1.5,2.0 --max-eval-descs 1 --skip-random-descs" \
    ./scripts/eval.sh $CKPT_END
```

Report where each task turns over. If they are still rising at 2.0, say so and
do not extrapolate.

**These numbers use one description and no controls. They must not share a table
with anything from §3.**

---

## 6. Pin the length-collapse threshold (~30 min, mostly not GPU)

Run 1's data suggests generation survives while the prompt's per-token norm stays
below roughly the norm of a real token embedding (**0.1543**), and collapses
above it. That estimate used `probe_prompt`'s `base norm` column as a proxy,
which is about 13% below the true `prompt_norm`, so the threshold is only known
to about ±0.02.

`prompt_norm` is now logged every 100 steps. Get its true value at each
checkpoint step:

```bash
# The log is wherever the run tee'd it. RESULTS.md records logs/run1/.
LOG=$(ls -1 /root/inlet/logs/run1/*.log "$INLET_OUTPUT_ROOT"/run1*.log \
        /root/outputs/run1.train.log 2>/dev/null | head -1)
echo "reading $LOG"

for S in 500 1000 2000 4000 8000 11000 16000; do
  printf "%6s  " "$S"
  grep -a "\[step $S\] train:" "$LOG" \
    | tail -1 | grep -ao "prompt_norm=[0-9.]*" || echo "(no train line at $S)"
done
```

If nothing prints, the run predates the commit that put `prompt_norm` on a
`[step N] train:` line. In that case take it from the **validation** lines
instead — `grep -a "\[step $S\] val/" "$LOG"` — which have carried
`prompt_norm=` all along, and note in the writeup that the value is the one
measured on that validation split rather than on training batches.

Then build one table: **effective norm (`prompt_norm` × scale) against median
completion words**, using the `completions median N words` lines the eval prints,
for humaneval and gsm8k across every step and scale available. Report the
largest effective norm at which generation is intact and the smallest at which
it has collapsed. Those two numbers bracket the threshold; do not report a
single value as if it were measured exactly.

---

## 7. Explain mbpp's 2.38 (~20 min)

Run 1 scored mbpp 2.38 at step 16,000, s1.0 — far more extreme than humaneval's
16.46 at the same point. Before that appears anywhere as a data point, look at
what the model actually emitted:

```bash
python -c 'import json, glob, os
root = os.environ["INLET_OUTPUT_ROOT"]
for f in sorted(glob.glob(root + "/eval_results_inlet/mbpp__*.json")):
    d = json.load(open(f))
    print("==", os.path.basename(f))
    print("  scores :", d["results"]["mbpp"])
    print("  lengths:", d.get("completion_lengths"))'
```

The eval log also prints `Sanitized N out of M files` from `evalplus`. If N is
large for the collapsed arm, the 2.38 may be a code-extraction artefact rather
than a capability collapse. Say which it is, and quote a couple of raw
completions either way.

---

## 8. What to send back

- The `desc_control` tables and both CSVs from §3 — **this is the headline**.
- Both `sweep_report --paper` tables from §4, plus the per-task
  s1.0-minus-s0.5 column with completion lengths.
- The §2 baseline table with measured vs published.
- The §5 bracket, the §6 threshold pair, and the §7 verdict on mbpp.
- `prompt_norm` at each checkpoint step.
- Every JSON under `eval_results_inlet/` — they are small, send the directory.
- Anything that disagrees with §0. Run 1's numbers were debug protocol and
  test-set selected; if the reported protocol contradicts them, the reported
  protocol wins and that disagreement is itself worth reporting.

---

## 9. Rules

- **Do not train anything.** If a result makes a training change look attractive,
  write that down and send it; do not act on it.
- **Do not select the checkpoint or the scale by which scored best.** §3 fixes
  both in advance. Use anything you like for diagnosis and label it as such.
- **Never put a debug-protocol number (§5) in the same table as a reported one
  (§3, §4).** One description and no controls is a different measurement.
- **Use `./scripts/eval.sh` to cover several tasks. Never `eval_inlet --tasks`
  with a checkpoint** — that scores every task with the *first* task's
  description prompt. The code now refuses it; the refusal is not a bug.
- **A vLLM `ERROR ... EngineCore ... died unexpectedly` after `wrote <file>` is
  teardown noise.** The score is already written. Read the exit code.
- If a gate fails, report it. Do not loosen it to get a number out.
