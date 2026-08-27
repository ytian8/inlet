# Runbook — 8×H200

Two commands. Everything else here is why.

## 1. Train

```bash
git clone https://github.com/ytian8/inlet.git && cd inlet
bash scripts/setup_env.sh                    # ~15 min, no GPU

source .venv/bin/activate
source scripts/common.sh
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --exclude "*.bin" "*.pth" "*.gguf" "*.msgpack" "*.h5"
huggingface-cli download Alibaba-NLP/gte-large-en-v1.5
./scripts/warm_cache.sh                      # 20-40 min, CPU only

./scripts/smoke.sh 2                         # ~20 min. Do not skip.

VENV=$PWD/.venv INLET_OUTPUT_ROOT=/root/outputs \
./scripts/train.sh 8 --run_name=cross8 \
    --desc_slots=8 --cond=cross \
    --model_select_split=val/unseen \
    --checkpoint_steps=500,1000,2000,4000,8000,16000,32000,64000,128000
```

| flag | why |
|---|---|
| `--desc_slots=8 --cond=cross` | Cross-attention over 8 description vectors instead of one pooled vector. `--desc_slots=1` reproduces the old model bit for bit, so the two runs are a controlled A/B. |
| `--checkpoint_steps` | **The only thing here that cannot be recovered afterwards.** Dense early: the known optimum is at or before step 4,000. |
| `--model_select_split=val/unseen` | 11 tasks held out of training and untouched by the benchmarks, so a number selected on it is genuinely zero-shot. Default is `val/seen`, which on the last run selected step ~130,000 — 2.77 points of 10-task average worse than step 4,000. |
| `INLET_OUTPUT_ROOT` on local disk | A checkpoint written to a wedged network volume comes out the right size and fails to load. `ENVIRONMENT.md` NEW 30. |

On by default, nothing to pass:

- **`val/generative`** — gsm8k `train` added as a fourth validation split. Upstream's three cover only multiple-choice and short-answer tasks; gsm8k, mbpp and humaneval are in none of them, which is why their collapse was invisible for 147,500 steps. `--generative_val_tasks=""` disables.
- **`--save_best_per_split`** — writes `hypermod_inlet_best_val_{seen,unseen,benchmark,generative}.pt`, 55 MB each. No need to guess the right criterion in advance.
- **`--canary_samples 4`** — generates freely from the context at each validation and warns when free-running accuracy falls below half the teacher-forced number. Teacher forcing cannot see a prompt that has destroyed generation.

**Check these lines, then leave it alone:**

```
LR: 2.500e-05 x 1.0000 = 2.500e-05   -- independent of GPU count by construction
NCCL collective timeout : 4:00:00     -- 0:10:00 means SIGABRT ~10 min in
val/generative: gsm8k[train]
permanent checkpoints will be kept at steps: [500, 1000, ...]
[neftune] alpha=5.0 ACTIVE on Embedding    -- minutes in, after step-0 validation
```

If it hangs: `pgrep -f train_inlet`, then `kill -USR1 <pid>` for **every** rank.

## 2. Get the curve

**Start with the four best-per-split checkpoints.** If one of them is good
enough, the step ladder never has to be paid for.

```bash
ALL10="arc_challenge arc_easy boolq hellaswag openbookqa piqa winogrande gsm8k mbpp humaneval"
RUN=/root/outputs/hyper_lora/cross8

# 4 checkpoints x 10 tasks = 40 jobs, ~1 h on 8 GPUs
WHICH=best ./scripts/sweep_checkpoints.sh $RUN "$ALL10" 8
```

Read the table. If the best of the four beats the frozen model and the four do
not disagree much, stop — you have your answer and which split to select on.

**Only if they are all bad, or disagree, go back for the curve:**

```bash
# 9 checkpoints x 10 tasks = 90 jobs, ~2.5 h on 8 GPUs
WHICH=steps ./scripts/sweep_checkpoints.sh $RUN "$ALL10" 8
```

`WHICH=all` does both. `SWEEP_CKPTS="a.pt b.pt"` scores exactly those files.

Cheaper first pass either way — the three generative tasks carry the whole
signal, and this is ~15 min for the four:

```bash
WHICH=best ./scripts/sweep_checkpoints.sh $RUN "gsm8k mbpp humaneval" 8
```

One job per (checkpoint, task), spread across the GPUs. Each job builds its own
vLLM engine, and **the engine build is the cost, not the eval**: measured on one
A100, a 7B engine takes ~9 min to build and each description variant ~2.6 min
after that.

So a sweep runs **one** description per task — enough for the shape of the curve
at a sixth of the price. `SWEEP_FULL=1` runs the full reported protocol instead.

| | jobs | 8 GPUs |
|---|---|---|
| 3 generative tasks × 9 checkpoints | 27 | **~50 min** |
| all 10 benchmarks × 9 checkpoints | 90 | **~2.5 h** |
| **the reported number**: 10 tasks × 1 checkpoint, full protocol | 10 | **~50 min** |

**Time one task before trusting those.** They extrapolate from humaneval, which
is 164 problems of long generation; the multiple-choice tasks have far more
questions but emit one or two tokens each, and those pull in opposite directions:

```bash
time TASKS=arc_challenge ./scripts/eval.sh <ckpt>
```

A swept number and a reported number are **different measurements** — one
description versus an average over three — and must not go in the same table.
Each results file records which it was:

```json
"protocol": { "n_eval_descs": 1, "random_descs": false, "reported_protocol": false }
```

Prints per-task scores against the frozen model, then:

```
=== 10-task average (this is the reported number) ===
     step       avg   vs zero-shot
zero-shot     56.09
    4,000     57.35          +1.26  <- peak
  130,000     54.58          -1.51
```

and writes `sweep_curve.png` — every task on the left, the 10-task average on the
right with the frozen-model line.

Drop `"$ALL10" 8` to sweep only the three generative tasks; that is the cheap way
to find the peak before paying for the full sweep.

**The number for the paper** is one checkpoint under the full protocol:

```bash
TASKS="$ALL10" ./scripts/eval.sh /root/outputs/hyper_lora/cross8/hypermod_inlet.pt
```

That is 3 real descriptions averaged plus the 3 junk-description controls, which
is T2L's protocol and what makes the numbers comparable to theirs. The junk
controls are the C measurement: real minus random is how much the description is
worth, and on the last checkpoint measured it was +0.41 points on humaneval —
i.e. nothing.

## 3. Why the benchmarks are not evaluated during training

A full 10-benchmark eval is ~4 h on one GPU (ten engine builds plus six
description variants each). Every 4,000 steps over 147,500 steps is 36 of them,
and all eight GPUs are busy training, so each would have to pause the run and
rebuild a vLLM engine.

So: checkpoints during, sweep after. Same information, and the sweep parallelises
across the 8 GPUs when nothing else needs them.

During the run, the cheap in-loop signals are `val/generative` (teacher-forced
loss on long-generation targets) and the canary (free-running generation). They
are proxies, not benchmark scores. The benchmark numbers come from §2.

## 4. Reporting

**Finding the peak and choosing the reported number are different jobs.**

| question | tool | may it go in the paper? |
|---|---|---|
| When did the model peak? | the sweep | **Yes, as a curve.** "Peaks at step N, degrades after" is a finding. |
| Which number do we report? | `--model_select_split` | **Yes.** |
| Which checkpoint scored best on the benchmarks? | the sweep | **Not as the headline** — selecting on it biases the number. |

Decide the split before reading any sweep output. On the 147,500-step run they
bottomed out 120,000 steps apart:

| split | what it is | bottomed out at |
|---|---|---|
| `val/unseen` | 11 lol_* tasks held out of training | step 10,000 — **recommended**, touches nothing the benchmarks use |
| `val/generative` | gsm8k `train` | new; the only split that watches long generation |
| `val/benchmark` | 7 multiple-choice benchmark tasks | step 30,000 — different split from the scored one, so not leakage, but still in-task information |
| `val/seen` | 10 lol_* tasks also in training | step 130,000 — measures description generalization only; diagnostic |

The strongest available result is the sweep peak and the selected checkpoint
landing on the same step. Report both.

## 5. State of the code

Verified locally, every run:

```
inlet.test_upstream_api      21 upstream symbols, 27 upstream calls,
                             250 internal calls, 23 CLI flags
inlet.test_desc_cond         21 checks, 2 known-bad controls
inlet.canary                 10 checks, 1 known-bad control
inlet.generative_val          8 checks, 1 known-bad control
inlet.test_train_eval_agree   8 checks, 5 known-bad controls
```

Verified on 2× A100 previously: `gate_m0` |delta| exactly 0.000e+00,
`test_accum` 9.070e-08, `test_ddp_equiv` 8.918e-08, 200-step 2-GPU run.

**Not yet run on hardware**, so check these first if something breaks:
`val/generative` (needs datasets), `--save_best_per_split`,
`--model_select_split`, the canary's generation half, and the loop in
`sweep_checkpoints.sh`. Their logic and wiring are checked; the hardware paths
are not. `./scripts/smoke.sh 2` exercises all of them.
