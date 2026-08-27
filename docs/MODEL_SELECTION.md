# Picking the checkpoint

## The problem, in three numbers

Same 147,500-step run, two checkpoints:

| | step 4,000 | step 130,000 |
|---|---|---|
| 7 multiple-choice tasks | 71.39 | 72.12 |
| 3 generative tasks (gsm8k, mbpp, humaneval) | **24.61** | **13.65** |
| 10-task average | **57.35** | 54.58 |

The reported number came from step 130,000. Nothing in training could have known
4,000 was better, because:

- `val/seen`, `val/unseen` and `val/benchmark` contain **only multiple-choice and
  short-answer tasks**. `gsm8k`, `mbpp` and `humaneval` are in none of them
  (upstream's `BENCHMARK_TASK_INFO` has seven keys, and those three are not among
  them).
- The rolling checkpoint followed whichever split a dict listed first.

So the generative tasks degraded for 126,000 steps with no instrument pointed at
them.

---

## 1. Launching a run

```bash
VENV=/root/venv-inlet INLET_OUTPUT_ROOT=/root/outputs HF_HOME=/workspace/hf \
./scripts/train.sh 8 --run_name=cross8 \
    --desc_slots=8 --cond=cross \
    --checkpoint_steps=500,1000,2000,4000,8000,16000,32000,64000,128000
```

| flag | why |
|---|---|
| `--desc_slots=8 --cond=cross` | cross-attention over 8 description vectors instead of one pooled vector. `--desc_slots=1` reproduces the old behaviour bit for bit, so the two are a controlled A/B. |
| `--checkpoint_steps=...` | **The only thing here that cannot be recovered afterwards.** Dense early: the known optimum is before step 4,000 and may be earlier still. |
| `INLET_OUTPUT_ROOT` on local disk | A checkpoint written to a wedged network volume comes out the right size and fails to load. See `ENVIRONMENT.md` NEW 30. |

Already on by default, nothing to pass:

- `--save_best_per_split` writes `hypermod_inlet_best_val_{seen,unseen,benchmark}.pt`
- `--canary_samples 4` reports free-running generation next to teacher-forced
  accuracy, and warns when they diverge

Check these lines appear, then leave it alone:

```
LR: 2.500e-05 x 1.0000 = 2.500e-05   -- independent of GPU count by construction
NCCL collective timeout : 4:00:00     -- 0:10:00 means the job will SIGABRT
permanent checkpoints will be kept at steps: [500, 1000, ...]
[neftune] alpha=5.0 ACTIVE on Embedding   -- arrives minutes in, after step-0 validation
```

---

## 2. Finding the peak, after the run

**On the 8-GPU node, sweep all ten and let it shard across the GPUs:**

```bash
ALL10="arc_challenge arc_easy boolq hellaswag openbookqa piqa winogrande gsm8k mbpp humaneval"
./scripts/sweep_checkpoints.sh train_outputs/hyper_lora/cross8 "$ALL10" 8
```

Eval uses one GPU, so the checkpoints run eight at a time. That is the whole
difference in what is affordable:

| | 1 GPU | 8 GPUs |
|---|---|---|
| 3 generative tasks, 9 checkpoints | ~4 h | **~30 min** |
| all 10 benchmarks, 9 checkpoints | ~15 h | **~2 h** |

Prints per-task scores against the frozen model, then the curve that matters:

```
=== 10-task average (this is the reported number) ===
     step       avg   vs zero-shot
zero-shot     56.09
    4,000     57.35          +1.26  <- peak
  130,000     54.58          -1.51
```

and writes a two-panel plot to `$INLET_OUTPUT_ROOT/sweep_curve.png`: every task
on the left, the ten-task average on the right with the frozen-model line.

**The ten-task average is the curve to read.** A per-task view hides it — between
step 4,000 and 130,000 the seven multiple-choice tasks moved 0.73 points while
the reported average moved 2.77.

The average is only computed at steps where all ten were evaluated. A partial
average is not comparable across steps: it would quietly reward whichever step
happened to be missing the hardest task.

Default (no task list) is the three generative tasks, which is the cheap way to
find the peak before spending on the full sweep.

---

## 3. Which checkpoint to report

**These are different jobs. Do not use one answer for both.**

| question | use | may it go in the paper? |
|---|---|---|
| When did the model peak? | the sweep (§2) | **Yes, as a curve.** "Performance peaks at step N and degrades after" is a finding. |
| Which number do we report? | `--model_select_split` | **Yes.** |
| Which checkpoint scored highest on the benchmarks? | the sweep | **No, not as the headline.** That is selection on the test set and the number carries an optimistic bias. |

For the reported number, pick the split **before** looking at any sweep output:

| split | what it is | note |
|---|---|---|
| `val/unseen` | 11 lol_* tasks held out of training | **Recommended.** Touches nothing the benchmarks use, so a number selected on it is genuinely zero-shot with respect to them. Bottomed out at step 10,000 on the 147,500-step run. |
| `val/benchmark` | 7 multiple-choice benchmark tasks | Uses a different split from the one scored, so not leakage — but it is still the benchmark tasks' own data, which sits awkwardly next to a zero-shot claim. Say so if you use it. Bottomed out at step 30,000. |
| `val/seen` | 10 lol_* tasks that are also in training | Measures description generalization only, and keeps improving while the model overfits. Bottomed out at step 130,000. Diagnostic only. |

```bash
./scripts/train.sh 8 --run_name=cross8 --model_select_split=val/unseen ...
```

The strongest result available is the sweep peak and the selected checkpoint
landing at the same step. Report both.

---

## 4. What is still missing

No validation split covers generative tasks, so **checkpoint selection cannot
see them** no matter which split you choose. §2 is a post-hoc workaround, not a
fix.

The fix is a `val/generative` split built from `gsm8k` train and a decontaminated
`mbpp` train. Not implemented. Two constraints when it is:

- `humaneval` cannot go in it — 164 problems, no train split. Using it would be
  selection on the scored data.
- `mbpp` must use the decontaminated list. 108 of the 120 problems in
  `mbpp/sanitized` train fall inside MBPP+, which is what the harness scores.
