# QUICKSTART

The whole thing, in order, with nothing to decide. `README.md` explains *why*
each step is what it is; this file is just the sequence. If a step's output
does not match what is written here, stop and read the troubleshooting table at
the bottom of `README.md` — every entry there is a failure that actually
happened on a clean box.

Total: ~1 h of setup (mostly downloads, no GPU), then ~27 h of training on
8× A100.

---

## 0. What you need

- 1–64 GPUs with ≥ 40 GB each. 8× A100-80GB is the reference.
- **60 GB of free disk minimum, 100 GB comfortable**, on a volume you control:
  ~13 GB venv (the `nvidia-*` wheels) + 14.5 GB weights + 1.7 GB gte-large +
  ~5 GB dataset cache + 55 MB per checkpoint. `setup_env.sh` refuses to start
  below 60 GB, because running out here does not raise an error — it wedges the
  shell and the machine then looks idle while it keeps billing. On a cloud box,
  check the **volume** gauge, not the container one.
- Python 3.10–3.12. **Not 3.13** — `vllm==0.11.1` has no wheels there.
- A Hugging Face account (the model is not gated; you still need to be logged
  in for the download to be reliable).

---

## 1. Install (~15 min, no GPU needed)

```bash
git clone git@github.com:ytian8/inlet.git
cd inlet
bash scripts/setup_env.sh
```

This checks free disk first, then creates `.venv`, picks the torch wheel index
from your driver, installs the pinned stack, clones
[text-to-lora](https://github.com/SakanaAI/text-to-lora) into `third_party/` if
you do not already have it, and verifies the install by importing
`inlet.train_inlet`.

**Then run the four no-GPU checks before anything else.** Seconds each, no
model weights, no dataset, no vLLM — and they catch the class of bug that
otherwise costs you a full training run:

```bash
python -m inlet.test_upstream_api         # this checkout still fits the
                                        # text-to-lora checkout next to it
python -m inlet.test_train_eval_agree     # the training and eval prompt
                                        # assemblies must be one operation
python -m inlet.test_desc_cond            # description slots, the padding mask,
                                        # and that cross-attention is not a no-op
python -m inlet.baseline_ref              # the reconstructed reference satisfies
python -m inlet.consistency_ref           # the contract the original was tested on
```

All five must print `PASS`. If `test_train_eval_agree` fails, do not train:
the prompt would be learned in one sequence layout and scored in another, and
nothing downstream would raise.

If you already have a text-to-lora checkout, point at it first so it is not
cloned twice:

```bash
export T2L_ROOT=/abs/path/to/text-to-lora
```

**Expected last line:** `setup complete`, followed by a "next steps" block that
repeats steps 2–5 below.

---

## 2. Weights (~10 min, ~14 GB)

```bash
source .venv/bin/activate
source scripts/common.sh        # <- sets HF_HOME. Do not skip this line.

# No `huggingface-cli login`. Both models are ungated, so the two downloads
# below work anonymously. Do not stop here waiting for a token.

huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --exclude "*.bin" "*.pth" "*.gguf" "*.msgpack" "*.h5"
huggingface-cli download Alibaba-NLP/gte-large-en-v1.5
```

Two things here are load-bearing:

- **`source scripts/common.sh` first.** It pins `HF_HOME` (default
  `<repo>/.hf`). Download with a different `HF_HOME`, or none, and training
  will not find these 14 GB — it will start a second copy and die with
  `Disk quota exceeded`. If your site has a shared cache, `export HF_HOME=…`
  *before* sourcing.
- **`--exclude "*.bin"`.** Without it you get the `.bin` *and* the
  `.safetensors` copies: 24 GB instead of 14 GB.

**Check:** `du -sh $HF_HOME` should be ~20 GB (14 GB Mistral + 5.6 GB gte).

---

## 3. Datasets (~20–40 min, CPU only — do this before you rent GPUs)

```bash
./scripts/warm_cache.sh
```

Downloads and tokenizes all 510 datasets (479 train + 31 eval) into
`$T2L_ROOT/data/transformed_datasets`. Failure-tolerant: a dead or rate-limited
repo is recorded in `warm_failures.json`, not fatal.

```bash
./scripts/warm_cache.sh --only-failed warm_failures.json   # retry just those
```

After this, everything runs with `HF_DATASETS_OFFLINE=1`, so a multi-day job
can never stall on the hub.

**Your first training run will still build some datasets, and that is normal.**
`warm_cache.sh` warms the *training* representation of each task. The trainer
also needs `val/seen`, `val/unseen` and `val/benchmark`, which upstream hashes
differently (it rewrites `split` to `train[:90%]` / `train[90%:]`, and merges
`BENCHMARK_TASK_INFO` into the benchmark tasks' kwargs). So expect a few extra
minutes before step 1 of the first run, once. It is not a warm failure, and it
is safe under DDP: the trainer builds them inside
`accelerator.main_process_first()`, so rank 0 writes and everyone else waits.

**Do not skip this before eval.** If a dataset is missing from the cache, offline
mode reports it as a network failure —
`ConnectionError: Couldn't reach 'allenai/ai2_arc' on the Hub
(OfflineModeIsEnabled)` — which sends you looking at the wrong thing. To force
one eval through without warming everything: `HF_DATASETS_OFFLINE=0 ./scripts/eval.sh …`.

---

## 4. Pre-flight (~20 min, 1 GPU). Do not skip before a multi-day run.

```bash
./scripts/smoke.sh 1
```

Four checks, in order. Each has caught a real break:

| # | Check | Must print |
|---|---|---|
| 1 | train-path == eval-path consistency | agreement, or a clean "baseline directory missing, skipping" |
| 1b | gradient accumulation == larger batch (CPU) | `PASS`, with `accum=2 vs batch=16 … 9.07e-08` |
| 1b2b | description slots / padding mask / cross-attention is live (CPU, no model) | `PASS`, 21 checks including 2 known-bad controls |
| 1b3 | train-side `[B, m+T, d]` == eval-side `[m+T, d]`, element for element (CPU, no model) | `PASS`, 8 checks + 4 deliberately-broken controls detected |
| 1c | 2 ranks == 1 process with 2× the batch (CPU, gloo, no GPU) | `PASS`, with `DDP vs 1 process … 8.92e-08` and the summed-gradient control at `3.33e-01` |
| 2 | m=0 gate: zero vectors reproduce the frozen model | `[gate] m=0 reduces to plain forward … injection point OK` |
| 3 | 200-step training run | loss falls |
| 4 | `--zero-prompt` eval | `zero-shot reproduction: got 65.xx, expected 65.70 -> PASS`. A one-question difference (65.61) is normal across hardware and passes; `WARN` or `FAIL` means the prompt is being injected in the wrong place. |

Then confirm multi-GPU actually works before committing to the long job:

```bash
./scripts/smoke.sh 2
```

**Check this line in the output:**

```
NCCL collective timeout : 4:00:00
```

If it says `0:10:00`, something re-introduced an early process-group creation
and the job will SIGABRT ten minutes in, during the first validation, with no
Python traceback. See README → troubleshooting.

Then check the checkpoint that 2-GPU run wrote — once, on the first multi-GPU
checkpoint you ever produce:

```bash
python -m inlet.test_ckpt_keys train_outputs/hyper_lora/smoke2/hypermod_inlet.pt
```

Must print `PASS`. It catches a DDP-wrapped `state_dict` whose keys all carry a
`module.` prefix — which loads fine nowhere and only shows up at eval, after
the training run is over.

---

### Check the LR line

The trainer prints, before the first step:

```
LR: 2.500e-05 x 1.0000 = 2.500e-05  [global batch 64 vs reference 64, 'sqrt']
    -- independent of GPU count by construction
NCCL collective timeout : 4:00:00
```

Both numbers must be the same whether you ran `train.sh 1`, `2` or `8`. They
used to not be: the LR was scaled by GPU count (2.50e-5 / 3.54e-5 / 7.07e-5) and
the cosine schedule was consumed `world` times too fast, so the three configs
the README calls reproductions of each other were three different experiments.
See `docs/ENVIRONMENT.md` → NEW 20.

## 5. Train

```bash
./scripts/train.sh 8 --run_name=full \
    --checkpoint_steps=4000,20000,40000,80000,147500
```

The first argument is the GPU count and it is the only thing that changes
between machines — 1, 2, 8, or (with `NNODES`/`NODE_RANK`/`MASTER_ADDR`) 64.
All of them run the *same* experiment: `GLOBAL_TASKS` (default 64) fixes the
global batch and gradient accumulation is derived from it, so the batch, the
step count and the LR schedule are identical and only wall time changes.

| GPUs | A100-80GB | H100 |
|---|---|---|
| 1 | ~8.7 days | ~5 days |
| 2 | ~4.4 days | ~2.5 days |
| 8 | ~27 hours | ~17–18 hours |

64 GPUs (8 nodes x 8) is the one case that needs a second variable, because
`GLOBAL_TASKS` must stay divisible by `TASKS_PER_RANK x world` = 8 x 64 = 512
and the default 64 is not. `train.sh` refuses to start rather than rounding:

```bash
GLOBAL_TASKS=512 NNODES=8 NODE_RANK=$i MASTER_ADDR=<node0-ip> \
    ./scripts/train.sh 8 --run_name=full        # on each node, with its NODE_RANK
```

`--checkpoint_steps` writes a **permanent** checkpoint at each listed optimizer
step, alongside the rolling best-val one. **Decide this list before launching.**
It exists for one measurement that cannot be reconstructed afterwards: how the
description-conditioning gain grows with training budget. A curve is worth far
more than the single endpoint, and it costs nothing at training time — but only
if the checkpoints were kept.

Out of memory? Lower `TASKS_PER_RANK` (default 8), never `GLOBAL_TASKS`.
Accumulation absorbs the difference and the optimization is untouched:

```bash
TASKS_PER_RANK=4 ./scripts/train.sh 8 --run_name=full
```

Resuming: the run writes to `train_outputs/hyper_lora/full/`. Relaunching with
the same `--run_name` overwrites it; use a new name for a new run.

**Check that these three lines appear.** Each one corresponds to
a bug that was silent until it was checked, so a missing line is a real finding,
not a cosmetic one:

```
[neftune] alpha=5 ACTIVE on Embedding (max |delta| over one probe: ...)
[ddp] per-rank RNG streams verified distinct across 8 ranks
NCCL collective timeout : 4:00:00
```

The first two arrive within seconds. **`[neftune]` does not**, and its absence
early is not a finding: it is activated deliberately last, after the m=0 gate and
after step-0 validation, because it draws fresh noise on every call and would
otherwise make those two comparisons fail against a differently-perturbed
forward (see `activate_neftune` in `inlet/train_inlet.py`). On a first run, step-0
validation also builds `val/seen`, `val/unseen` and `val/benchmark`, so the
neftune line can be **several minutes** in. Measured 2026-08-26 on 2x A100:
launch 05:29:35, neftune line 05:34:13.

Also read back the resolved step count and the LR line. `epochs` is epochs, not
steps — 479 tasks at 8 per batch is 59 batches per epoch, so `--epochs=20000` is
1,180,000 single-GPU steps, and reading it as a step count under-trains by 59×.
The LR is keyed off the global batch, not the GPU count.

---

## 6. Eval

```bash
./scripts/eval.sh train_outputs/hyper_lora/full/hypermod_inlet.pt
```

Scores the 10 benchmarks from T2L's Table 2. Each task is evaluated once per
description variant and averaged, matching T2L's protocol ("an average of three
generated LoRAs, each with a different instance of task descriptions") — so
this is 30 evaluations and 10 vLLM engine builds. One GPU is enough; pick it
with `CUDA_VISIBLE_DEVICES`.

```bash
TASKS="arc_challenge" ./scripts/eval.sh <ckpt>     # a single task, ~5 min
./scripts/eval.sh --zero-prompt                    # no checkpoint: the 65.70 gate
```

Results land in `train_outputs/eval_results_inlet/`.

### Reference numbers (arc_challenge, Mistral-7B-Instruct-v0.2)

| | arc_c |
|---|---|
| zero-shot | 65.70 |
| TextGrad | 63.42 |
| 3-shot ICL | 71.93 |
| per-task prompt tuning (32 vectors) | 74.89 |
| T2L | 77.28 |

An untrained Inlet checkpoint should land near zero-shot — the head is
zero-initialised. A large gap there means `base` init or `emb_rms` is wrong,
not that training failed.

---

## If it hangs

A multi-GPU job that hangs shows no error and looks busy — both GPUs at 100%,
because a rank spinning in an NCCL collective is indistinguishable from a rank
doing work. Get the stacks:

```bash
pgrep -f train_inlet                 # one pid per rank
kill -USR1 <pid>                   # do this for EVERY rank
tail -100 <your log>
```

The trainer installs a SIGUSR1 handler for exactly this. Compare the ranks: the
healthy one is parked in a collective (`backward`, `barrier`, `all_reduce`); the
stuck one is somewhere else, and that somewhere is the bug. (`py-spy` is the
usual tool and does not work in most containers — no `SYS_PTRACE`.)

The known instance of this, already fixed, is worth knowing as a rule: **any
`if is_main:` block that touches an `accelerator.prepare()`d module can
desynchronise the ranks**, because DDP issues collectives of its own (buffer
broadcast on the first forward) that `no_grad` does not suppress. Run
single-rank work against `accelerator.unwrap_model(...)`.

---

## When a check fails

Every gate in this repo guards a failure that is **silent** — the code runs, the
loss falls, and the numbers are wrong. So there is one rule:

> **Do not disable a check, loosen a tolerance, delete an assert, or pass a flag
> that skips a gate in order to make a run proceed.** If a gate fires, it has
> found something. Fix the cause, or stop and report it.

That applies to all of these, each of which will stop a run:

* any of the five no-GPU checks in step 1 printing `FAIL`
* `smoke.sh` failing at any step
* `m=0 path does not reduce to a plain forward pass` — the injection point is
  wrong and every number afterwards would be quietly wrong
* `RuntimeError: NEFTune was requested ... identical embeddings` — the hook is
  dead, so the recipe would differ from the T2L numbers it is compared against
* `RuntimeError: Per-rank sampler independence is broken` — the ranks would
  sample identical batches and the effective batch would be 1/N of what every
  log line claims
* the `--zero-prompt` eval missing the published zero-shot number by more than
  ~2 questions out of 1172
* `INLET_SKIP_DISK_CHECK=1` starting to look attractive

## When the run finishes, report

1. The **10-benchmark eval table**, with each description variant kept separate
   (`eval_descs` / `other_train_descs` / `random_descs` / `train_descs`). Do not
   collapse them: the gap between real and random descriptions is the number
   this project exists to measure.
2. `train_outputs/hyper_lora/full/train_summary.json`.
3. The three startup lines from step 5, verbatim.
4. `val/seen`, `val/unseen`, `val/benchmark` — loss and `per_token_acc` — at each
   checkpoint step.
5. **`prompt_std_across_batch` over training.** If it trends toward zero the
   generator has collapsed to a constant prompt and is ignoring its input. Say
   so loudly: no eval number reveals this on its own, and the trainer only warns.
6. Anything you had to change to make it run, and why.

---

## If something breaks

`README.md` ends with a symptom → cause → fix table covering every failure seen
on a clean box: the torchvision ABI abort with no traceback, the evalplus
`SUCCESS` import, the NCCL watchdog SIGABRT, the two ways to run out of disk,
the missing chat template, and the rest. Look up what you *see*, not what you
think it means — none of these errors names its own cause.

`docs/ENVIRONMENT.md` has the long-form version with the exact commands.
