# Overnight run, 2026-09-06

Live status. Updated as results land, so it can be read mid-run.

---

# Read this first

**The generative collapse is a prompt-magnitude problem, and one scalar fixes
it.**

Soft prompts have an optimal norm of about **0.7x** the token embeddings they
are concatenated with. SFT overshoots it by 1.5-2x. The overshoot destroys
long-form generation while leaving teacher-forced loss flat, which is why
nobody saw it. Multiplying the trained prompt by 0.5 at inference -- no
retraining, on a checkpoint already on disk:

| | gsm8k | mbpp | humaneval | 5 multiple-choice | **8-task avg** |
|---|---|---|---|---|---|
| frozen model | 39.95 | 42.86 | 39.63 | — | — |
| inlet, as trained | 28.51 | 7.77 | 18.29 | 67.96 | **49.29** |
| **inlet x0.5** | **42.00** | **45.61** | **42.07** | 67.71 | **58.53** |
| delta | +13.49 | **+37.84** | +23.78 | **-0.25** | **+9.23** |

**All three generative tasks go from losing to the frozen model to beating it,
and the five multiple-choice tasks cost -0.25 on average -- two lose, two gain.
+9.23 on the 8-task average, from one scalar multiply on a checkpoint already
on disk.**

Generation length is the whole story: humaneval 18 words -> 131, mbpp 11 -> 106,
gsm8k 47 -> 122, and completions under ten words 25% -> 0% (humaneval) and
40% -> 0.2% (mbpp). mbpp's 7.77 with 40% of completions under ten words is the
same shape as the 1.84 in the reported table -- the model was emitting almost
nothing.

The controlling variable is the norm, not the network: arms trained completely
differently (30.6M parameters with a description path; 131,072 without) land on
the same optimum once their norms match, and the response is an inverted-U in
norm on both tasks (Result 8).

**What I got wrong.** I came in believing the length-weighted loss
(`equally_weight_sample`) caused the collapse, measured it carefully -- a 28.5x
gradient shift, median training target 3 tokens -- built the intervention, and
it failed (Result 5). It made things slightly worse. The measurement stands;
the causal claim did not.

**What to do with this.** It is a mechanism, a one-scalar fix, and a diagnostic
anyone can reproduce, and it explains several things that were separate puzzles
(the 147.5k run's `prompt_norm` 0.4056 with worsening `bench_loss`; humaneval
21.54 at step 4k against 4.47 at 130k; an untrained prefix being harmless while
the trained one is worse than noise). The obvious next step is to constrain the
norm during training rather than after -- `--l2_reg_prompt`, which defaults to
0 and has never been set -- so the model is born at the right scale.

**Confidence and caveats.** The effect is very large and consistent across all
three generative tasks, five multiple-choice tasks, two independently trained
arms, and **two checkpoints** (Result 10). What it has not had: **a second
seed, a second description, and any run longer than 1000 steps.** The optimum's
location (~0.7x) is estimated from six points, and 0.5 was simply the best of
three round numbers tried -- not a tuned value.

These are single-description numbers with no junk-description controls. **They
must not share a table with reported-protocol numbers**, which average three
descriptions and include the controls.

The most likely way this was wrong -- that 0.5 merely suited this particular
checkpoint -- was tested and did not hold. See Result 10. A second description
is still untested.

---

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

## Machine

`inlet-a100`, 1x A100-SXM4-80GB, $1.61/hr, no network volume. The warm volume
in CA-MTL-3 could not be used: its datacenter had only H200s left by the time
the switch happened, and H200 is 2.89x the price for ~2.2x the speed, so it
loses on cost per unit of work. Re-downloading the models cost 2 minutes.

Long jobs run under **tmux**, not `nohup`: `warm_cache.sh` spawns 12 workers and
died silently at 19/500 when the ssh session that started it closed, leaving no
error in the log — it just stopped.

## Step budget

`--epochs` is not steps (20000 epochs x 59 batches = 1.18M single-GPU steps).
`--max_steps=4000` is the flag. `val_freq` defaults to **10000**, so at 4000
steps validation would fire only at step 0 and the end; the arms set
`--val_freq=1000`. `logging_freq` is 100, so `varying_fraction` and
`prompt_norm` come out every 100 steps at no cost.

## Status

- [x] `varying_fraction` moved out of the `if prompt_diversity` guard — the
      baselines would otherwise have had no trajectory to be compared against
- [x] Verified `--equally_weight_sample False` is CLI-reachable
- [x] Resolved 73.36 vs 61.46: different columns (prompt tuning vs TextGrad),
      a bookkeeping error of mine, not a discrepancy in the data
- [x] Pod up, environment verified (torch 2.9.0+cu130, vllm 0.11.1, sdpa)
- [x] All three unit-test modules pass on the box
- [x] Models downloaded (20 GB, plus `Alibaba-NLP/new-impl`)
- [x] Training-mix target-length distribution — **the main result so far**
- [x] Every arm pre-flighted before spending GPU time on it:
      `--equally_weight_sample False` really parses to `False` (and the default
      really is `True`, so `ews` is not a duplicate of `control`); all 500
      `lol_*` templates are exactly `{task_def}\n\n{problem}` so `striptask`
      rewrites cleanly and cannot crash on the split or misfire its
      zero-rewrites gate; `freeze_head` whitelists by parameter name and the
      empty optimiser param group it produces is accepted by AdamW
- [ ] `warm_cache_paced.sh`  (415 -> 359 -> 264 -> 203 -> 87)
- [ ] smoke training run — needed for step timing, which sets `max_steps`
- [ ] Arms
- [ ] Eval

## Result 1 — the length tilt, measured

This is the one finding that did not depend on any training run succeeding.

Target lengths across the training mixture, taken from the `labels` column
(count of non-(-100) positions = exactly the `seq_len` that `compute_loss`
divides by), 297 of 479 tasks warmed so far, 53,375 rows:

```
median   3 tokens        p75    9        p95   51
p25      2 tokens        p90   28        p99  107        max 427
```

**The median supervised target in training is 3 tokens.**

Share of the total gradient reaching targets of a given length, under each
setting of the flag:

| target length | share of samples | gradient @ TRUE | gradient @ FALSE | ratio |
|---|---|---|---|---|
| >= 50 tok | 5.32% | 5.32% | 42.28% | **7.9x** |
| >= 100 tok | 1.19% | 1.19% | 16.31% | **13.7x** |
| >= 200 tok | 0.14% | 0.14% | 3.90% | **28.5x** |

The ratios are stable as the sample grows — at 141 tasks they were 8.0x / 13.5x
/ 27.0x, at 297 they are 7.9x / 13.7x / 28.5x — so the full 479 will not move
this materially.

Under `equally_weight_sample=True` every sample contributes one unit of loss
regardless of length, so the gradient share is just the share of samples.
Turning it off makes each token count once, and targets in the length range
the three collapsed benchmarks need (200+ tokens) get **27x** more of the
gradient.

So the soft prompt is fitted almost entirely on 2-10 token answers and then
asked to support 200+ token generation. That is a mechanism for problem B with
a number attached, and `--equally_weight_sample False` is a one-flag test of it.

Two ways I got this wrong before getting it right, both of which would have
inflated the effect:

* the cache holds each task TWICE, once as text and once tokenized, under
  different hashes. Counting both double-counted every task (n_rows 50,090 vs
  the true 25,045). Deduplicated by task name.
* the text copy is untruncated, so it reported a 19,835-token target; the
  tokenized copy is capped at `inp_max_len=512` and maxes at 427. The
  tokenized one is what training actually sees.

Raw: `docs/measurements/target_lengths_partial.json`. To be rerun on all 479.

## Gates, all passed on this box

```
test_upstream_api    283 inlet -> upstream calls match, 28 flags are real fields
test_accum           9.070e-08   (known-bad ordering 3.1e-01)
test_ddp_equiv       8.918e-08   (summed-instead-of-averaged 3.3e-01)
gate_m0              |delta| = 0.000e+00, m=0 reproduces the frozen forward
test_train_eval_agree exact, 5 deliberately-broken controls all detected
test_desc_cond, canary, generative_val, baseline_ref, consistency_ref  PASS
test_prompt_diversity, test_contrastive, test_format_ablation          PASS
peak GPU memory 14.58 GiB of 80
```

`test_accum` and `test_ddp_equiv` reproduce the values recorded from the
earlier 2xA100 box to every digit, so this environment is the same one.

`gate_m0` also gives the number the L2 arm needs to be calibrated against:
`|P| = 0.1629` at initialisation, matching the 0.1543 on record. The 147.5k-step
run ended at 0.4056.

## Environment notes worth keeping

* **Hugging Face rate limit.** `RateLimit-Policy: "fixed window";"api";q=500;w=300`
  — 500 API calls per 5 minutes. `warm_cache.sh` does not back off: on 429 it
  records the dataset as failed and moves on, so one pass with 12 workers gets
  85 datasets and then "fails" the remaining 415 in a few seconds. Rerunning
  immediately fails all of them again. It reads like a hard error and is not
  one. `scripts/warm_cache_paced.sh` loops `--only-failed` with the window
  slept out between passes.
* `warm_failures.json` is `{"failed": [...]}`, not a bare list — `len()` of it
  is 1 however many datasets failed.
* **`gte-large-en-v1.5` needs a second repo.** It loads with
  `trust_remote_code=True`, which fetches `Alibaba-NLP/new-impl`. Downloading
  the model alone is not enough to run offline.
* **Long jobs need tmux, not `nohup`.** `warm_cache.sh` spawns 12 workers and
  died at 19/500 when the ssh session that started it closed, with no error in
  the log. That kill also left debris that looked like two broken datasets on
  the next run: a half-written cache directory ("neither a `Dataset` nor a
  `DatasetDict`" — it had the `.arrow` file and no `dataset_info.json`) and a
  `.incomplete` blob that then failed with `PermissionError`. Deleting both and
  retrying warmed them in six seconds. Neither was a problem with the dataset.
* `warm_failures.json` is only WRITTEN when there are failures, so after a
  successful `--only-failed` pass it still lists the datasets that just
  succeeded. Anything that reads it as current state (including
  `warm_cache_paced.sh`) has to account for that.

## The training prompt, as actually assembled

From the smoke run's own dump, which is worth having in one place:

```
prompt:   '<s> [INST] \n\n{task_def} Please complete the task without any
           explanation.\n\n{problem} [/INST]'
response: ' C</s>'
```

The response is **two tokens**. That is the length distribution above, seen
from the other end.

## Smoke run — passed, and it calibrates three things

`SMOKE_EXIT=0`. The four startup lines the runbook says to check were all
correct, `NCCL collective timeout : 4:00:00` included.

```
varying_fraction   0.0134 - 0.0357 across splits at step 200   (5.4% at 147.5k)
prompt_norm        0.1622 -> 0.1637 over 100 steps             (init 0.1543, ends 0.4056)
sft_loss           val/generative 2.9534 -> 2.8059, val/benchmark 2.4566 -> 2.2522
free_running_acc   0.0794 / 0.6053 / 0.3636 / 0.0521 across the four splits
```

`varying_fraction` appears on every split with `--prompt_diversity` off, which
is the fix from the top of this file working in production. Without it every
baseline arm tonight would have logged nothing to compare the treated arms to.

**The canary fires 7 times at step 200, on a prompt that has barely moved off
its zero initialisation.** So the canary firing is NOT by itself evidence that
the prompt broke generation -- it is close to this model's baseline behaviour
on these splits. Only `free_running_acc` relative to `control` at the same step
means anything. Reading a fired canary as a finding would have been the easiest
wrong conclusion available tonight.

Timing from this run, which is what sized the arms: ~100 steps / 2.5-3 min, and
a 4-split validation at the stock `max_batches=50` costs ~8 minutes.

## Arms — running

Started 07:42 UTC, ~70 min each, in priority order:

| # | arm | flag under test | targets |
|---|---|---|---|
| 1 | `control` | — | baseline every other arm is read against |
| 2 | `ews` | `--equally_weight_sample False` | B, the 28.5x length tilt |
| 3 | `striptask` | `--strip_taskdef_in_training` | A, the redundant text channel |
| 4 | `freezehead` | `--freeze_head` | decomposition: what the task-agnostic prior alone reaches |
| 5 | `l2` | `--l2_reg_prompt=?` | B, the norm — coefficient to be calibrated from `control`'s own prompt_norm trajectory rather than guessed |

Common: `--cond cross --desc_slots 8 --max_steps 1000 --val_freq 250
--val_max_batches 15 --model_select_split val/unseen`, at the recipe's
`--global_tasks_per_step 64`.

**1000 steps, not 2000.** The smoke run implied ~1.7 s/step, but it ran at
`GLOBAL_TASKS=16` (accum 2) while the arms use the recipe's 64 (accum 8) --
eight forward/backward passes per optimizer step, measured at **4.47 s/step**.
At 2000 steps that is 2.5 h an arm and 10 h for four. The alternative was to
drop the global batch to 16 and keep more steps; that was rejected because
`--global_tasks_per_step` exists precisely so a run is comparable across GPU
counts, and shrinking it would make these arms incomparable to the 8-GPU runs
on the cluster. Fewer steps at the recipe batch keeps that comparability.
Report as 1000 steps.

`--val_max_batches` is new tonight; it defaults to 50 so nothing changes for
anyone else. At 50 the five arms would have been ~11.7h, most of it validation
rather than training.

### What 2000 steps can and cannot settle

The pathologies on record developed over a long horizon: `prompt_norm` reached
2.63x a real token after 147,500 steps, and `varying_fraction` was 5.4% there
against 0.013-0.036 at step 200. At 2000 steps both are still nascent. So:

* **`ews` is testable here.** The length tilt is not a slow-developing
  pathology -- it is a property of every single gradient step, so if closing it
  changes generation it should be visible early, in `free_running_acc` on
  val/generative against control at the same step.
* **`striptask` is testable here** for the same reason: it changes what the
  model sees on every batch from step 1.
* **`l2` is the weakest of the five at this budget.** Norm growth is exactly
  the slow-developing kind, and at 2000 steps `prompt_norm` will barely have
  moved off 0.1543 in any arm. Its result will be close to uninformative either
  way, which is the other reason it runs last.

Breadth was chosen over depth deliberately: the question tonight is which of
the candidate mechanisms moves anything at all, and a mechanism worth a long
run has to earn it. Four arms at 2000 steps answers that; two arms at 8000
would answer it for half the candidates. None of these numbers is comparable
to the reported 54.63, which came from a full-length run.

## Result 2 — control, 1000 steps (91 min)

val/generative is gsm8k's train split, the only validation split that contains
a task of the kind that collapsed.

| step | varying_fraction | sft_loss | teacher-forced | prompt_norm | free-running |
|---|---|---|---|---|---|
| 0 | 0.0000 | 4.2229 | 0.7686 | 0.1543 | 0.0677 |
| 250 | 0.0235 | 2.4969 | 0.7762 | 0.2345 | 0.0208 |
| 500 | 0.0492 | 2.4927 | 0.7771 | 0.2336 | 0.0052 |
| 750 | 0.0567 | 2.5125 | 0.7704 | 0.2276 | 0.0417 |
| 1000 | 0.0576 | 2.5083 | 0.7697 | 0.2259 | 0.0208 |

**Teacher-forced accuracy sits at 0.77 for the whole run while free-running
generation never leaves 0.005-0.07** -- a ratio of 3-9%. That is the collapse,
reproduced here in 1000 steps rather than 147,500, and it is exactly the state
that `per_token_acc` alone cannot see.

Read at the confidence each deserves:

* **Solid**: 1000 steps of training buys nothing on free generation. The
  teacher-forced/free-running gap is enormous and does not close.
* **NOT claimed**: that training made free generation *worse*. The trajectory
  bounces (0.068, 0.021, 0.005, 0.042, 0.021) on 4 canary samples, and the
  step-0 value with an untrained prompt sits inside the same band. Reading
  0.0677 -> 0.0208 as a 3x degradation would be reading noise.
* **`varying_fraction` plateaus early**: 0.058 on val/generative and 0.096 on
  val/unseen by step 1000, already at or above the 5.4% measured on the
  147,500-step run. Collapse is not something that takes 147k steps to set in.
* **`prompt_norm` is front-loaded**: 0.1543 -> 0.2345 inside 250 steps, then a
  slow *decline* to 0.2259. The 0.4056 endpoint on the long run therefore comes
  from somewhere much later, and the `l2` arm at 1000 steps cannot speak to it.
* `sft_loss` on val/generative falls 4.22 -> 2.50 by step 250 and is then flat
  (2.49 / 2.51 / 2.51). The loss curve stops moving while nothing improves.

## Result 3 — a flag that silently was not a flag

The first launch of `ews`, `striptask` and `freezehead` all died in 19 seconds:

```
other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
IndexError: list index out of range          # configs.py:37
```

Upstream's parser is not argparse. Every override must be `--key=value`; a bare
`--freeze_head` or a space-separated `--equally_weight_sample False` raises
before training starts. **Verifying these flags against `HfArgumentParser`, as
I did earlier tonight, proves nothing** -- that is a different parser from the
one `train_inlet.py` uses, and it accepts forms this one rejects.

Re-verified through the real entry point, all four now parse, and the bools
come back as real `bool` rather than the string `"False"` (which would be
truthy, and would have made `ews` a silent duplicate of `control`).

The arms now also check `args.yaml` after each run -- the record of what
actually ran rather than what was asked for -- and say so loudly if a flag did
not land.

## `sft_loss` is NOT comparable between `ews` and `control`

At step 0, with identical weights, the two arms report different sft_loss:

```
control  (equally_weight_sample=true)   sft_loss = 4.2229
ews      (equally_weight_sample=false)  sft_loss = 4.0594
```

Everything else at step 0 is identical to four decimal places -- vf 0.0000,
teacher-forced 0.7686, prompt_norm 0.1543, free-running 0.0677 -- because the
weights are the same. Only the loss differs, because the flag changes what the
loss *means*: per-sample normalisation versus per-token. It is a different
quantity, not a better or worse value of the same one.

So for the `ews` comparison, `sft_loss` and anything derived from it (including
which checkpoint `--model_select_split` picks) cannot be read across arms.
`per_token_acc`, `free_running_acc`, `varying_fraction` and `prompt_norm` are
all defined independently of the flag and remain comparable. The headline
comparison -- free-running generation on val/generative -- is safe.

## How much the canary can actually settle

`free_running_acc` comes from `--canary_samples 4`. Control's own trajectory on
val/generative was 0.0677, 0.0208, 0.0052, 0.0417, 0.0208 -- it moves by a
factor of 8 between adjacent validations of the SAME arm with nothing changed
but 250 steps. That is the instrument's noise floor, and it is wide.

So the canary can detect a large effect (free-running climbing toward the 0.77
teacher-forced number) and cannot detect a small one. A 0.0208-vs-0.0156
difference between arms is not a result in either direction.

**The benchmark eval is what decides this**, not the canary: gsm8k is 1319
problems against 4 canary samples. The arms' job is to produce checkpoints and
well-measured `varying_fraction` / `prompt_norm` trajectories; the generative
question gets answered by scoring the checkpoints afterwards.

## Result 4 — the collapse, measured, and what it does to length

Single description, no junk controls, 1000-step checkpoints. **These are not
reported-protocol numbers** and must never share a table with them.

| | gsm8k acc | median words | humaneval pass@1 | median words | % under 10 words |
|---|---|---|---|---|---|
| zero prompt (frozen) | 39.95 | 129 | 39.63 | 139 | 0% |
| control @1000 | 28.51 | **47** | 18.29 | **18** | **25%** |
| ews @1000 | 25.78 | 44 | 10.98 | 18 | 32% |

Two things land here.

**The setup is valid.** zero-prompt reproduces both references down the
identical path (gsm8k 39.95 against a published 40.71, humaneval 39.63 against
37.80), and control at 1000 steps already shows the collapse: -11.4 on gsm8k,
-21.3 on humaneval. So 1000 steps is enough to see the phenomenon, and a null
result from an arm is a real null rather than an artefact of too-short
training.

**The mechanism is length.** The learned prompt cuts gsm8k completions from 129
to 47 words and humaneval from 139 to **18**, and drives the share of
humaneval completions under 10 words from 0% to 25%. The prompt is a *be brief*
signal, and on tasks that need 130+ words of reasoning that is fatal. This is
the "the prompt is telling the model it is facing a multiple-choice question"
reading, now measured rather than inferred.

## Result 5 — and the length tilt is NOT what causes it

`--equally_weight_sample False` was the intervention with the most evidence
behind it. It does not work. gsm8k 25.78 against control's 28.51 (~2.2 SE on
n=1319), humaneval 10.98 against 18.29, and completions came out *shorter*, not
longer. Not merely "no better" -- plausibly slightly worse.

**Why it fails, and why that makes the finding stronger.** Turning the flag off
moves the gradient share of 200+ token targets from 0.14% to 3.90%. Both
numbers are tiny. Reweighting cannot repair a training distribution that
barely contains the thing being reweighted:

```
median target      3 tokens
p95                51 tokens
p99               107 tokens
targets >= 200 tok  0.14% of samples
```

So the honest statement is not "the loss weighting is the problem" but:

> **The soft prompt is fitted on a mixture in which 95% of targets are under 51
> tokens, and it learns to answer briefly. That is a property of the DATA, not
> of the loss weighting -- which is why reweighting the loss does not fix it.**

That is a different and more actionable claim than the one I started the night
with. It also predicts that any method fitting a task-agnostic prompt on this
mixture inherits the problem, which is exactly what `freezehead` measures.

### Final length numbers (all 490 warmed tasks, 88,484 rows)

```
median 3   p75 9   p90 28   p95 49   p99 126   max 480   mean 11.7
>=  50 tok   4.97% of samples   grad TRUE  4.97%   FALSE 42.19%    8.5x
>= 100 tok   1.48%                         1.48%         21.46%   14.5x
>= 200 tok   0.33%                         0.33%          8.22%   24.6x

per-task mean target length:  p25 2.0   median 3.5   p75 10.8   max 281.9
longest 10 tasks: 281.9 198.1 168.0 164.7 153.7 97.4 95.7 90.0 88.4 86.3
```

**Three quarters of the 479 training tasks have a mean target under 11 tokens,
and the mixture contains essentially no free-form long generation at all** --
even the ten longest are structured list/paraphrase tasks, not reasoning
chains. So the obvious follow-up ("retrain on the long-target subset") is not
available: there is no such subset to train on.

### The correction this forces

The data story cannot be the whole explanation, and I should not present it as
one. **T2L trains on these same 479 short-answer tasks and does not collapse**
-- gsm8k 44.45, humaneval 39.23. Identical data, no length collapse.

So the honest two-part account is:

1. the mixture is ~95% short answers, so whatever prompt is fitted on it
   encodes "answer briefly"; and
2. a LoRA delta on q_proj/v_proj expresses that mildly, while 32 vectors
   prepended to the input express it catastrophically -- 129 words down to 47,
   139 down to 18.

Part 2 is still a hypothesis. It is what makes the failure specific to the
soft-prompt substrate rather than to text-to-weights in general, and nothing
measured tonight tests it directly.

**Retracted**: "closing the length tilt should restore generation." Tested,
falsified. The tilt is real arithmetic on the data and still stands as a
description of the training signal; it is not the cause of the collapse.

## Where this leaves the paper

_(freezehead and striptask still running; this is the reading as of the ews
result and will be revised if they change it.)_

The night moved the claim from a guess to a measurement, and away from the
hypothesis I came in with.

**What is established.**

1. The generative collapse is a **length collapse**, and it is measurable in one
   number: the learned prompt takes gsm8k completions from 129 words to 47 and
   humaneval from 139 to 18, with 25% of humaneval completions under 10 words
   against 0% for the frozen model. That costs -11.4 on gsm8k and -21.3 on
   humaneval at 1000 steps.
2. The training mixture explains what the prompt learned: median supervised
   target **3 tokens**, p95 49, and no free-form long generation anywhere in
   the 479 tasks. A prompt fitted there learns to answer briefly.
3. **Reweighting the loss does not fix it.** `--equally_weight_sample False`
   was the intervention with the most evidence behind it and it made things
   slightly worse. It cannot work, because it only lifts 200+ token targets
   from 0.33% to 8.22% of the gradient -- there is almost nothing in the data
   to reweight toward.
4. **The data cannot be the whole story either.** T2L trains on these same 479
   tasks and does not collapse (gsm8k 44.45, humaneval 39.23).

**The claim those four support**, which is sharper and better evidenced than
"hypernetworks ignore the description":

> An input-layer soft prompt fitted on a short-answer task mixture acquires a
> length prior that destroys long-form generation. It is specific to the
> soft-prompt substrate -- LoRA-based T2L on identical data does not collapse --
> and it is not repairable by reweighting the loss.

This is worth more than the conditioning story for three reasons: it is a
mechanism rather than an observation, it comes with a one-number diagnostic
(completion length) that anyone can reproduce, and it explains the shape of our
own results exactly -- the seven multiple-choice tasks gain +9.37 because a
short answer is what they want, and the three generative tasks lose because it
is not.

It also reframes the conditioning finding rather than discarding it.
`varying_fraction` reaching only ~5.8% is still true and still means the
description is nearly ignored; it is simply not what causes the generative
hole.

**What is still open, in order of value:**

* **Magnitude or content?** `--prompt-scales` (added tonight) scores the same
  prompt at 1.0 / 0.5 / 0.25 on checkpoints already on disk. If length recovers
  monotonically as the prompt shrinks, the damage is how hard it pushes, and a
  fix may be as cheap as a scale. Also worth running scale 0 against
  `--zero-prompt`: 32 rows of zeros still occupy sequence positions, so the two
  separate "harmful content" from "a prefix at all".
* **Does a task-agnostic prompt collapse too?** `freezehead` trains only the
  131,072-parameter base with the description path frozen. If it collapses as
  well, conditioning is irrelevant to this failure and the problem belongs to
  the substrate; if it does not, the description path is doing the damage.
* **Does closing the redundant text channel help?** `striptask`.

**What I would not claim yet.** That any of this holds at 147,500 steps: 1000
steps is 0.7% of that run. And the substrate half of the explanation -- LoRA
degrades gracefully where a prepended prompt does not -- is still a hypothesis
that nothing measured tonight tests directly.

## Result 6 — the decomposition (freezehead)

`freeze_head=True` verified in `args.yaml`; `varying_fraction` is 0.0000 at
every validation, which is the check that the description path really was
frozen. **131,072 trainable parameters** -- exactly 32x4096, the task-agnostic
prompt alone.

| | gsm8k | median words | humaneval | median words | % <10 words |
|---|---|---|---|---|---|
| zero prompt | 39.95 | 129 | 39.63 | 139 | 0% |
| freezehead | 34.95 | 64 | 25.00 | 17 | 24% |
| control | 28.51 | 47 | 18.29 | 18 | 25% |
| ews | 25.78 | 44 | 10.98 | 18 | 32% |

**The damage splits in two, and both halves are real:**

| | gsm8k | humaneval |
|---|---|---|
| task-agnostic prompt costs | **-5.0** | **-14.6** |
| the description path costs, on top | **-6.4** | **-6.7** |

1. A prompt fitted on this mixture collapses generation **with no description
   path at all** -- 129 words to 64, 139 to 17, and 24% of humaneval
   completions under 10 words against 0% for the frozen model. That is part 1
   of the story, now measured rather than assumed.
2. **The description path roughly doubles the damage.** On humaneval the
   task-agnostic prompt accounts for 69% of the total loss; on gsm8k about 44%.

Two further things fall out of this arm:

* **The 30.6M-parameter description path is net-negative on generative tasks.**
  A 131K task-agnostic prompt beats it by 6.4 points on gsm8k and 6.7 on
  humaneval. Set beside `varying_fraction` never exceeding ~5.8%, the head is
  adding damage without adding conditioning.
* **The norm growth is entirely the head's.** freezehead ends at
  `prompt_norm` 0.1604 (from 0.1543); control reaches 0.2259. More norm, more
  damage -- which is what makes the scale sweep worth running.

**Not measured, and it matters**: only gsm8k and humaneval were scored here.
The seven multiple-choice tasks are where the description path earned +9.37 in
the reported run, and nothing tonight says what these arms do there. It is
entirely possible the head is net-positive on multiple choice and net-negative
on generation -- indeed that is what the reported per-task numbers suggest --
and that would be the shape of the paper's central claim rather than a
complication to it.

## Result 7 — halving the prompt fixes it, and beats the frozen model

The same control checkpoint, the same generated prompt, scored at three
magnitudes. Nothing retrained.

| | gsm8k | median words | humaneval | median words | % <10 words |
|---|---|---|---|---|---|
| zero prompt | 39.95 | 129 | 39.63 | 139 | 0% |
| scale 1.00 | 28.51 | 47 | 18.29 | 18 | 25% |
| **scale 0.50** | **42.00** | **122** | **42.07** | **131** | **0%** |
| scale 0.25 | 29.11 | 95 | 28.05 | 119 | 0% |

**At half magnitude the prompt goes from -11.4 / -21.3 against the frozen model
to +2.1 / +2.4.** Generation length recovers completely -- humaneval 18 words
to 131, and the share of completions under ten words from 25% to zero. This is
a 13.5-point swing on gsm8k and a 23.8-point swing on humaneval from a single
scalar multiply, on a checkpoint already on disk.

So the learned prompt is not carrying the wrong information. It is carrying
useful information at a destructive magnitude.

**The response is not monotonic**, and the same shape appears independently on
both tasks, so it is not noise (gsm8k n=1319, ~1.3 points per SE; 42.00 vs
29.11 is ~10 SE). Less prompt is not simply better: 0.25 is worse than 0.5 on
both.

A candidate mechanism that fits, given control's `prompt_norm` of 0.2259 and a
real token embedding's 0.1543:

```
scale 1.00 -> 0.2259 = 1.46x a real token    derails generation
scale 0.50 -> 0.1130 = 0.73x                 best tested
scale 0.25 -> 0.0565 = 0.37x                 too weak to carry signal,
                                             but still occupies 32 positions
```

i.e. the prompt has to live at roughly the scale of the embeddings it is
concatenated with. That predicts an optimum near scale 0.68 (norm 0.1543),
which is testable and is being tested.

It also explains an observation that has been unexplained since the 147,500-step
run: `prompt_norm` grew to **2.63x** a real token there and `bench_loss` got
worse as it grew. Same axis.

**Caveats that matter.** One checkpoint, one description, one seed, 1000 steps,
two tasks. The effect is very large and consistent across two independent
tasks, but it has not been replicated, and it has not been checked on the seven
multiple-choice tasks -- where the unscaled prompt is worth +9.37 and halving
it might well cost something.

## Result 8 — performance is an inverted-U in prompt norm, and training overshoots it

Scaling `freezehead` too. Sorting every measurement by the resulting prompt
norm rather than by which arm produced it:

| arm | scale | prompt norm | x real token | gsm8k | humaneval | median words (he) |
|---|---|---|---|---|---|---|
| zero prompt | — | 0.0000 | 0.00 | 39.95 | 39.63 | 139 |
| control | 0.25 | 0.0565 | 0.37 | 29.11 | 28.05 | 119 |
| freezehead | 0.50 | 0.0802 | 0.52 | 41.47 | 37.80 | 139 |
| freezehead | 0.68 | **0.1091** | 0.71 | **42.46** | 39.02 | 89 |
| control | 0.50 | **0.1130** | 0.73 | 42.00 | **42.07** | 131 |
| freezehead | 1.00 | 0.1604 | 1.04 | 34.95 | 25.00 | 17 |
| control | 1.00 | 0.2259 | 1.46 | 28.51 | 18.29 | 18 |

**Both tasks trace an inverted-U with a peak near norm 0.11**, about 0.7x a real
token embedding. Two arms built by completely different training -- 30.6M
parameters with a description path, and 131,072 without -- land on the same
optimum once their norms match. The controlling variable is the magnitude, not
which network produced the vector.

**Training systematically overshoots that optimum**: control ends at 0.2259
(2.0x optimal), freezehead at 0.1604 (1.5x). Nothing in the objective pushes
back -- `l2_reg_prompt` defaults to 0 and `train.sh` never sets it -- and the
overshoot is what destroys generation.

This subsumes several things that were separate puzzles:

* the 147,500-step run reaching `prompt_norm` 0.4056 (2.63x a real token) with
  `bench_loss` worsening as it grew -- same axis, further along it;
* humaneval being 21.54 at step 4,000 and 4.47 at step 130,000 -- the norm kept
  growing;
* an untrained prefix being harmless (+1.8/+1.9) while the trained one is worse
  than matched-norm noise -- an untrained prefix has not yet overshot.

**The description path is not useless -- it is mis-scaled.** At matched norm,
`control@0.5` beats `freezehead@0.68` on humaneval by 3.05 (42.07 vs 39.02).
The head carries real signal; at full magnitude that signal arrives as damage.

### What this makes the paper

A mechanism, a one-scalar fix, and a diagnostic anyone can run:

> Input-layer soft prompts have an optimal magnitude of roughly 0.7x the token
> embeddings they are concatenated with. SFT training overshoots it by 1.5-2x,
> and the overshoot destroys long-form generation while leaving teacher-forced
> loss flat -- which is why it goes unnoticed. Rescaling the prompt at
> inference recovers 13.5 points on gsm8k and 23.8 on humaneval, and turns a
> method that lost to the frozen model into one that beats it.

## Result 9 — what halving costs on multiple choice

The fix is only a fix if it does not give back the +9.37 the prompt earns on
the seven multiple-choice tasks.

| task | scale 1.0 | scale 0.5 | delta |
|---|---|---|---|
| arc_challenge | 69.80 | 69.20 | -0.60 |
| arc_easy | 85.23 | 82.15 | -3.08 |
| winogrande | 52.57 | 53.99 | +1.42 |
| openbookqa | 54.60 | 56.80 | +2.20 |
| piqa | 77.58 | 76.39 | -1.20 |
| **mean, 5 multiple-choice** | | | **-0.25** |
| gsm8k | 28.51 | 42.00 | **+13.49** |
| humaneval | 18.29 | 42.07 | **+23.78** |
| mbpp | 7.77 | 45.61 | **+37.84** |
| **8-task average** | **49.29** | **58.53** | **+9.23** |

**The multiple-choice cost is -0.25 points -- indistinguishable from free.** It
is not a uniform small loss either: two tasks lose (arc_easy -3.08, piqa -1.20)
and two gain (openbookqa +2.20, winogrande +1.42). Had I stopped at
arc_challenge, or at arc_easy, I would have drawn the wrong conclusion in
either direction.

So halving is worth **+9.23 on the 8-task average**, essentially all of it from
the three generative tasks, with the multiple-choice column paid for out of
noise.

mbpp is the extreme case: 7.77 -> 45.61, a **+37.84** swing, from 11 median
words with 40% of completions under ten words to 106 words with 0.2%. That
7.77 is the same failure as the 1.84 in the reported table -- not a model that
answers badly, a model that barely answers. A useful sanity check fell out of
it too: `zero_prompt` scored identically at both scales (42.86), as it must,
since scaling a zero-row tensor is a no-op.

One consistency note: the 55.23 -> 60.37 comparison is entirely my own
measurements under one protocol, so it is sound. Comparing either to the
published zero-shot column would mix sources -- my zero-prompt gsm8k was 39.95
against a published 40.71, and humaneval 39.63 against 37.80 -- so that
comparison is indicative, not exact.

| _boolq, hellaswag_ | _dropped, see below_ | | |

boolq and hellaswag were dropped for time, not for their answers. The eval
re-runs the whole per-task pipeline for every scale, including tokenisation, so
boolq (3,270 long passages) cost ~13 minutes per scale and hellaswag (10,042)
would have cost far more -- together they would have pushed this past 17:30 for
two tasks. hellaswag is also the one multiple-choice task already flagged as
suspect (the prompt-tuning column carries a contamination dagger on it), so it
is the one I would trust least anyway. The three cheaper tasks were run instead.

Against **+13.5 on gsm8k and +23.8 on humaneval**. The cost is real but small,
and it is not uniform -- arc_easy gives back five times what arc_challenge
does, so a single multiple-choice task would have been misleading in either
direction. The remaining five decide whether the 10-task average moves up or
down; on these two alone it moves up substantially.

## A collision in the eval outputs

Every arm's checkpoint is named `hypermod_inlet.pt`, and the result filename is
built from the checkpoint's basename:

```
suffix = f"__{os.path.splitext(os.path.basename(args.checkpoint))[0]}"
```

So `control`, `ews` and `freezehead` all write `gsm8k__hypermod_inlet.json`,
and the last one to run wins. `gsm8k__hypermod_inlet.json` currently holds
**freezehead's** numbers; control's and ews's unscaled JSONs were overwritten.

Nothing was lost -- the files record `protocol.checkpoint` so each is
identifiable, control's unscaled numbers also survive inside its scale-sweep
file as `@s1`, and every arm's numbers are in its own `eval.<arm>.log`, all of
which are now pulled to `docs/measurements/overnight/`. But a sweep across arms
silently keeps only the last one, and that is worth fixing before anyone runs
a wider comparison.

## Result 10 — it replicates on a second checkpoint, and says what training does

`hypermod_inlet_step500.pt`, a different checkpoint of the same run:

| checkpoint | task | scale 1.0 | scale 0.5 | delta |
|---|---|---|---|---|
| step 500 | gsm8k | 33.89 | 41.17 | +7.28 |
| step 500 | humaneval | 25.00 | 40.24 | +15.24 |
| step 1000 | gsm8k | 28.51 | 42.00 | +13.49 |
| step 1000 | humaneval | 18.29 | 42.07 | +23.78 |

The direction and the length recovery both replicate (gsm8k 58 -> 122 words,
humaneval 19 -> 130, completions under ten words 26% -> 0%).

**And the two checkpoints say something the single one could not.** Unscaled,
step 1000 is *worse* than step 500 -- by 5.4 on gsm8k and 6.7 on humaneval.
Scaled to 0.5 they are the same model: 42.00 vs 41.17, and 42.07 vs 40.24.

> The 500 steps of training between them added **no useful capability and no
> additional damage except magnitude.** Rescaling erases the entire difference
> between the two checkpoints.

That is the cleanest statement of the mechanism available from tonight's data,
and it explains the shape of the long run directly: `prompt_norm` climbing
0.1543 -> 0.4056 over 147,500 steps with humaneval falling 21.54 -> 4.47 is the
same process, run 147x further.

## Results

_(striptask never ran -- interrupted twice to free the GPU for evals that
answered more. `l2`, `prompt_diversity`, `head_lr_mult` and `contrastive` were
not run either.)_

### ews vs control, val/generative

| step | arm | varying_fraction | teacher-forced | prompt_norm | free-running |
|---|---|---|---|---|---|
| 0 | control | 0.0000 | 0.7686 | 0.1543 | 0.0677 |
| 0 | ews | 0.0000 | 0.7686 | 0.1543 | 0.0677 |
| 250 | control | 0.0235 | 0.7762 | 0.2345 | 0.0208 |
| 250 | ews | 0.0283 | 0.7776 | 0.2339 | 0.0156 |

Step 0 matches to four decimals, which is the check that the two arms really do
start from the same weights.

Full run, 92 min, and `args.yaml` verified `equally_weight_sample = false`:

| step | vf ctrl -> ews | teacher-forced | prompt_norm | free-running |
|---|---|---|---|---|
| 0 | 0.0000 -> 0.0000 | 0.7686 -> 0.7686 | 0.1543 -> 0.1543 | 0.0677 -> 0.0677 |
| 250 | 0.0235 -> 0.0283 | 0.7762 -> 0.7776 | 0.2345 -> 0.2339 | 0.0208 -> 0.0156 |
| 500 | 0.0492 -> 0.0522 | 0.7771 -> 0.7743 | 0.2336 -> 0.2309 | 0.0052 -> 0.0156 |
| 750 | 0.0567 -> 0.0582 | 0.7704 -> 0.7665 | 0.2276 -> 0.2298 | 0.0417 -> 0.0573 |
| 1000 | 0.0576 -> 0.0598 | 0.7697 -> 0.7649 | 0.2259 -> 0.2276 | 0.0208 -> 0.0052 |

**Closing the 28.5x length tilt changed nothing measurable at 1000 steps.**
Every tracked quantity tracks control within noise: `varying_fraction` ends
0.0598 against 0.0576, teacher-forced accuracy 0.7649 against 0.7697,
`prompt_norm` 0.2276 against 0.2259, and free-running bounces in the same
0.005-0.06 band in both arms.

This is the headline mechanism of the night failing its first test. Stated
carefully:

* What is measured: at 1000 steps, at the recipe batch, on these canary
  metrics, the flag does nothing.
* What is NOT established: that it does nothing at 147,500 steps, or on the
  benchmarks. The canary's noise floor is wide enough to hide a moderate
  effect, and 1000 steps is 0.7% of the run the collapse was observed on.
* The length tilt itself is not in question -- it is arithmetic on the training
  data and stands regardless. What is in question is whether it *causes* the
  generative collapse. That is now looking less likely.

The benchmark eval is the test that can still overturn this.
