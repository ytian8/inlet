# Overnight run, 2026-09-06

Live status. Updated as results land, so it can be read mid-run.

## The question

Inlet scores 54.63 on the 10-benchmark average against a frozen Mistral's
56.09. The whole deficit is three generative tasks:

```
7 multiple-choice   71.94   (zero-shot 62.57, T2L 76.99)   working
3 generative        14.24   (zero-shot 40.98, T2L 45.12)   26.7 points below doing nothing
```

Two failures are tangled here and they are NOT the same thing:

- **A — the generator ignores its description.** cos(real, junk) = 0.9999,
  C = +0.41, varying_fraction = 5.4%.
- **B — generation collapses.** gsm8k 23.20, mbpp 1.84, humaneval 17.68.

**T2L proves A does not cause B**: T2L has A (the description explains only a
few percent of its gain) and does not have B (gsm8k 44.45, humaneval 39.23).
So fixing A alone leaves the scores where they are. B is where the 26.7 points
are, and tonight is aimed at B.

## What rules out the easy explanations

**Not substrate capacity.** Per-task prompt tuning, on the same 32 input-layer
vectors, reaches gsm8k **49.61** — above T2L's 44.45 with per-layer LoRA. The
soft prompt can carry reasoning; our hypernetwork does not find what direct
optimisation finds. (Per-task prompt tuning is an oracle upper bound, not an
apples-to-apples baseline.)

**Not an eval bug.** A zero prompt through the identical eval path reproduces
humaneval 39.63 against a published zero-shot of 37.80.

**Not the text format prior.** Tested and retracted: adding the training
prompts' "without any explanation" sentence to a benchmark *gains* gsm8k 3.1
and costs humaneval 1.8. It cannot produce −20.7 and −33.

## The three candidate mechanisms for B

1. **Length-weighted loss.** `compute_loss(..., equally_weight_sample=True)`
   divides each sample's token losses by that sample's own target length
   (`sft_trainer.py:381`), so per-token gradient weight is proportional to
   1/target_length. Default is True (`configs.py:118`) and `train.sh` never
   sets it. The only config in upstream that overrides it is
   `configs/lora_gsm8k.yaml:23`, which sets it **false** — for the generative
   task, upstream themselves turned it off.

   Per-task prompt tuning is immune to this: within one task all targets are
   about the same length, so dividing by `seq_len` is a constant. The
   hypernetwork trains on 479 tasks with target lengths from 1 token to 200+,
   which is where the skew bites.

   *Known hole:* T2L trains on the same mixture with the same loss and does not
   collapse. So this is at best necessary, not sufficient — see 2.

2. **Prompt norm.** `prompt_norm` grew 0.1543 → 0.4056 over the 147.5k-step run
   (2.63x any real token embedding), and `bench_loss` worsened as it grew.
   humaneval was 21.54 at step 4,000 and 4.47 at step 130,000. `l2_reg_prompt`
   defaults to 0.0 and `train.sh` never sets it, so the norm was never
   constrained. Hypothesis for why T2L escapes: a LoRA delta on q_proj/v_proj
   degrades gracefully, while 32 off-distribution vectors prepended to the input
   compound over 200 autoregressive steps. *Not yet measured.*

3. **Channel redundancy** (this one is aimed at A, not B). Training uses
   `"{task_def}\n\n{problem}"`. On a frozen Mistral over 10 lol_ tasks, with
   `task_def` 28.37 rougeL and without it 4.64 — the text already delivers
   23.73 points of the task, so the prompt has no reason to learn to carry it.
   None of the ten benchmark templates carries a task spec.

## Arms

| # | Config | Targets |
|---|---|---|
| 1 | `--cond cross --desc_slots 8` | control |
| 2 | 1 + `--equally_weight_sample False` | B |
| 3 | 1 + `--l2_reg_prompt <x>` | B |
| 4 | 1 + `--strip_taskdef_in_training` | A |
| 5 | `--freeze_head` | decomposition: what the task-agnostic prior alone reaches |

Cut for tonight: `--prompt_diversity`, `--head_lr_mult`, `--contrastive`. All
three attack A, and A is not where the 26.7 points are. `varying_fraction` is
now logged on every arm regardless, so each arm still reports its effect on A
for free.

## Readouts, in priority order

1. **Generation length** vs zero-shot on gsm8k/humaneval (`completion_stats()`).
   If the prompt has learned "expect a short answer", this is where it shows.
2. **Benchmark scores** on gsm8k / mbpp / humaneval, with arc_challenge as the
   multiple-choice control.
3. **varying_fraction** — collapse shows up in the first few thousand steps.
4. **C = score(real descriptions) − score(junk descriptions).**

Benchmark averages at 4,000 steps with one seed are not the judge; a 1–2 point
difference between arms means nothing at this scale.

## Rules held to

- No checkpoint selected on benchmark scores. Debugging reference only, and
  labelled as such wherever it appears.
- A failing gate stops the run and gets reported, not loosened.
- Anything that has not actually executed on hardware is said to have not
  executed on hardware. As of the start of this run that includes all four new
  flags, the generation canary, and `--model_select_split`.

## Status

- [x] `varying_fraction` moved out of the `if prompt_diversity` guard — the
      baselines would otherwise have had no trajectory to be compared against
      (commit 44f478a)
- [x] Verified `--equally_weight_sample False` is CLI-reachable
- [x] Resolved 73.36 vs 61.46: different columns (prompt tuning vs TextGrad),
      a bookkeeping error of mine, not a discrepancy in the data
- [ ] Pod up
- [ ] `smoke.sh 2`
- [ ] Training-mix target-length distribution (no GPU needed)
- [ ] Arms
- [ ] Eval

## Results

_(filled in as they land)_
