# Inlet

**Adaptation through the only surface a black-box LLM exposes: vectors at the input.**

A hypernetwork maps a **natural-language task description** to **input-layer
soft-prompt vectors** for a **frozen** Mistral-7B-Instruct-v0.2.

The name is the argument. Every other way of adapting a frozen model — LoRA,
per-layer prefixes, adapters — needs to reach *inside* it, and therefore needs
white-box access at serving time. Vectors at the input are the one substrate an
embedding-level API can already accept, which makes adaptation a piece of *data*
rather than a piece of *surgery*: no per-layer changes, native batching across
users who each carry their own vectors.

Text-to-LoRA generates adapter weights; Inlet generates what fits through the
inlet. Same conditioning signal, same frozen base model, same 479/31
decontaminated task split, same eval harness — a different injection surface,
which is the whole point of the comparison.

```
description  ──▶  gte-large-en-v1.5  ──▶  TaskEncoder ──▶ 2× MLPResidual ──▶ head
   (text)          (frozen, pre-embedded)                                      │
                                                                               ▼
                                          P = base[32,4096] + head_out · emb_rms
                                                                               │
   input_ids ──▶ embed() ──────────────────────── concat(P, ·) ───────────────▶│
                                                                               ▼
                                                        frozen Mistral-7B  ──▶ loss
```

**13,814,784 trainable parameters** at the default `--desc_slots 1`
(30,618,624 with `--desc_slots 8 --cond cross`). Nothing else in the stack has a
gradient — the hypernetwork's size is not a claim of the paper, because it runs
on your own machine; only what reaches the frozen model does, and that is always
32×4096 vectors at the input.

---

## Start here

**The commands live in [`QUICKSTART.md`](QUICKSTART.md)** — the full sequence,
in order, with the expected output at each step and nothing to decide. Follow
that file. This one is reference material you come back to when something is
wrong or when you need to change a default.

The shape of it, so you know what you are in for:

| | | |
|---|---|---|
| 1 | `bash scripts/setup_env.sh` | ~15 min, no GPU |
| 2 | `python -m inlet.test_train_eval_agree` (+ 2 more) | seconds, no GPU, no weights — **all three must say PASS** |
| 3 | `huggingface-cli download …` | ~10 min, ~20 GB |
| 4 | `./scripts/warm_cache.sh` | 20–40 min, CPU only — do it before renting GPUs |
| 5 | `./scripts/smoke.sh 1` | ~20 min, 1 GPU — never skip before a multi-day job |
| 6 | `./scripts/train.sh 8 --run_name=full` | ~27 h on 8× A100 |
| 7 | `./scripts/eval.sh train_outputs/hyper_lora/full/hypermod_inlet.pt` | |

Two things that are load-bearing and easy to skip: `source scripts/common.sh`
**before** downloading weights (it pins `HF_HOME`; download to the wrong place
and the trainer silently fetches a second 14 GB copy), and `--exclude "*.bin"`
on the model download (24 GB instead of 14 GB otherwise). Both are in
QUICKSTART with the reasons.

## Where to look when

| You want to… | Go to |
|---|---|
| just run it | [`QUICKSTART.md`](QUICKSTART.md) |
| know what is actually verified vs. merely written | [Status](#status--what-is-verified-and-what-is-not), below |
| use a different number of GPUs | [Choosing the number of GPUs](#choosing-the-number-of-gpus) |
| understand a failure | [Troubleshooting](#troubleshooting-symptom--cause--fix) — symptom → cause → fix, every row a real failure |
| know why the environment is pinned the way it is | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — 23 numbered facts, each with how it was verified |
| **run this on a cluster, start to finish** | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — two commands and the reasons |
| launch a run, then pick which checkpoint to report | [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) — the commands, and why the obvious choice is the wrong one |
| see every number this project reports, with provenance | [`docs/RESULTS.md`](docs/RESULTS.md) — **the source of truth**; all baselines, the Inlet column, and what is superseded |
| check a result against a known-good run | [Known-good numbers](#known-good-numbers) |
| change the model or the architecture | [What the code does](#what-the-code-does), then [Repo layout](#repo-layout) |
| know which checks exist and what each one catches | [Gates](#gates--the-checks-that-must-pass) |
| ask "is the description path doing anything at all?" without running an eval | `python -m inlet.probe_prompt --checkpoint <ckpt>` — measures `P(desc) − base` directly. No LLM forward, no vLLM, no GPU. |

## Requirements

| | |
|---|---|
| GPU | 1× 80GB is enough (A100/H100). More GPUs = proportionally less wall time, same result. |
| Disk | **60 GB free, 100 GB comfortable.** venv with torch+cu128 ~13 GB · Mistral-7B safetensors 14.5 GB · gte-large 1.7 GB · dataset caches ~5 GB · checkpoints 55 MB each. `setup_env.sh` refuses to start below 60 GB and re-checks for 25 GB after the install. On a cloud box, read the **volume** gauge, not the container one — see the troubleshooting table. |
| Python | 3.10 – 3.12. **Not 3.13+**: `vllm==0.11.1` and `datasets==3.5.1` have no wheels there. |
| Accounts | A Hugging Face account that has accepted the `mistralai/Mistral-7B-Instruct-v0.2` licence. |
| Upstream | A checkout of [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora). |

Inlet is an **overlay**, not a fork. Task metadata, the hierarchical batch sampler,
the 479/31 decontaminated split, the collator, description pre-embedding, and the
vLLM eval harness are all *imported* from upstream, never copied — that is the
reason a Inlet number and a T2L number can go in the same table.

Eval also wants the reference `baseline_prompt_tuning/` directory, which lives
inside the text-to-lora checkout and is not distributed here. **Nothing requires
it** — a clean clone trains and evaluates without it — but it changes what a
comparison against the prompt-tuning baseline means. See
[Running without `baseline_prompt_tuning/`](#running-without-the-reference-baseline_prompt_tuning).

## Status — what is verified and what is not

**Read this before trusting any number in this file.** Last updated
**2026-08-24**. It says what was actually executed, not what is expected to
work; where a check has not been run since the code changed, it says so.

**Run these three first. No GPU, no model weights, no dataset, seconds each.**

```bash
python -m inlet.test_upstream_api         # does this fit the text-to-lora next to it
python -m inlet.test_train_eval_agree     # train-side vs eval-side assembly
python -m inlet.baseline_ref              # the reconstruction's own contract
python -m inlet.consistency_ref           # label masking, with controls
```

| check | status |
|---|---|
| `inlet.test_upstream_api` | **passes** against the real text-to-lora checkout — 19 imported symbols, 24 calls into upstream, 160 inlet→inlet calls, 22 CLI flags. All four checks mutation-tested (bogus import, bad keyword, wrong arity, invented flag): each is caught, and the file returns to PASS when reverted. |
| `inlet.test_train_eval_agree` | **passes under real PyTorch** — 2.8.0+cu128 on an A100-80GB PCIe, 2026-08-24. All 8 checks, all 5 controls, exit 0, including the bfloat16 dtype check that a stand-in could have got wrong.|
| `inlet.baseline_ref` | **passes under real PyTorch** (same box) — the five assertions lifted from the original `test_consistency.py`. |
| `inlet.consistency_ref` | **passes under real PyTorch** (same box) — plus four deliberately-malformed samples, each of which must raise, and the fully-supervised case, which must not. |
| `inlet.test_accum` | **passes under real PyTorch** (same box) — `accum=2 vs batch=16` at `9.070e-08`, misplaced-`zero_grad` control at `3.148e-01`, i.e. 3,470,892× worse. |
| `inlet.test_ddp_equiv` | **passes with two real ranks** (same box, gloo) — `DDP(2 ranks) vs 1 process` at `8.918e-08`, summed-gradient control at `3.328e-01` (3,731,188× worse), and the regression check for the 2026-08-23 hang: *rank-0-only forward on the UNWRAPPED module issued no collective*. |
| `inlet.gate_m0` | **passes on real hardware** — 2× A100-SXM4-80GB, torch 2.9.0+cu128, 2026-08-24. `|delta|` between the frozen forward and the m=0 Inlet forward is **exactly 0.000e+00**; the description path is dormant for one step and fully connected after. |
| `inlet.test_consistency_inlet` | **passes on the same box** (inside `smoke.sh` step 1/4, `--task arc_challenge`). |
| `train_inlet.py` end to end | **passes on the same box** — 2 GPUs, 200 steps, loss 14.08 → 2.22, `best_val_loss` 3.0393, `trainable_params` 13,814,784, checkpoint written. All three startup lines present: `[neftune] alpha=5.0 ACTIVE on Embedding (max |delta| 5.493e-02); off in eval mode`, `[ddp] per-rank RNG streams verified distinct across 2 ranks`, and `world=2 per-rank batch=8 grad_accum=1 -> GLOBAL BATCH 16`. |
| zero-prompt eval | **passes on the same box** — `TASKS=arc_challenge ./scripts/eval.sh --zero-prompt` gives **65.61 vs the published 65.70**, a difference of 1.0 question out of 1172. This is the eval half of the pipeline, vLLM 0.11.1 included. |
| `setup_env.sh` from a clean box | **passes, 8/8, from an empty `/workspace`** on 2026-08-24 — after fixing three things it exposed: a `trl` pin that made `pip` unsatisfiable, a torchvision check that failed a working install, and a missing `hf_transfer` that blocked every model download. See `docs/ENVIRONMENT.md` NEW 28. |

**The stand-in caveat is retired for those three.** They were first run against a small numpy-backed stand-in, because they were written on a machine with no PyTorch available. On 2026-08-24 all three were re-run on an A100-80GB PCIe under **real torch 2.8.0+cu128**, from a byte-verified copy of the source (sha256 checked after transfer), and all three print `PASS`. The stand-in is not in this repo and is on no code path.

## Running without the reference `baseline_prompt_tuning/`

That directory is not distributed here. Three modules cover what Inlet needs, so a
clean clone runs. **Whenever the real directory is importable it wins**, and every
consumer prints which one it got (`[eval] shared-code source: …`).

| file | covers | status |
|---|---|---|
| `inlet/baseline_ref.py` | `prepend_soft_prompt`, `init_soft_prompt`, `load_input_embeddings`, `build_train_val` | reconstructed, **not diffed against the original** |
| `inlet/consistency_ref.py` | `check_label_masking`, `check_prompt_matches_eval` | reconstructed, complete |
| `inlet/eval_common.py` | `BASE_MODEL`, `ZERO_SHOT` (9 tasks), `TASK_ORDER`, `SEEDS` | transcribed |

`get_tokenizer` is deliberately **not** reconstructed — the reference re-exports
`hyper_llm_modulator.utils.get_tokenizer`, so both paths import the same upstream
function and are identical by construction. `mbpp_clean_task_ids.json` is data,
not code, so the mbpp path raises with instructions rather than silently scoring
a different set of problems.

**What it costs you when the fallback is live.** The numbers are still real, but
Inlet and the prompt-tuning baseline are then scored by two implementations that
were never checked against each other, and a difference in tokenizer settings,
prompt position or label shift shows up as *"a slightly lower score"*, never as
an error. `python -m inlet.baseline_ref` and `python -m inlet.consistency_ref` run
the assertions the **original** was tested against — evidence, not proof. Full
provenance is in each file's module docstring.

## Choosing the number of GPUs

The first argument to `train.sh` is the GPU count. That is the whole interface.

```bash
./scripts/train.sh 1  --run_name=full
./scripts/train.sh 2  --run_name=full
./scripts/train.sh 8  --run_name=full

# 64 GPUs = 8 nodes × 8. Run this on each node, with its own NODE_RANK:
# GLOBAL_TASKS must stay divisible by TASKS_PER_RANK x world = 8 x 64 = 512,
# so the default 64 will not do here -- train.sh refuses rather than rounding.
GLOBAL_TASKS=512 NNODES=8 NODE_RANK=$i MASTER_ADDR=<node0-ip> \
    ./scripts/train.sh 8 --run_name=full
```

**All of these run the same experiment.** `GLOBAL_TASKS` (default 64) fixes the
number of task descriptions per *optimizer step* across all ranks, and gradient
accumulation is derived from it:

```
grad_accum = GLOBAL_TASKS / (TASKS_PER_RANK × world)
```

| GPUs | per-rank batch | grad accum | global batch | optimizer steps |
|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 64 | 147,500 |
| 2 | 8 | 4 | 64 | 147,500 |
| 8 | 8 | 1 | 64 | 147,500 |
| 64 | 8 | — | 512 → set `GLOBAL_TASKS=512` | 18,437 |

Same batch, same step count, same LR schedule — only wall time changes. That is
what makes a 2-GPU rerun of an 8-GPU result a *reproduction* and not a new
experiment. `GLOBAL_TASKS` must stay divisible by `TASKS_PER_RANK × world`; the
trainer refuses to start rather than silently round.

**Out of memory?** Lower `TASKS_PER_RANK`, never `GLOBAL_TASKS`. Accumulation
absorbs the difference and the optimization is untouched:

```bash
TASKS_PER_RANK=4 ./scripts/train.sh 8 --run_name=full
```

**Check this line on your first multi-GPU run.** The trainer prints, at startup:

```
NCCL collective timeout : 4:00:00
```

Validation runs on rank 0 only; every other rank waits inside a collective for
the whole pass, and torch's default NCCL timeout is **ten minutes** — shorter
than one real validation. When it expires, a non-main rank SIGABRTs with
`Watchdog caught collective operation timeout` and **no Python traceback**, and
it looks like a deadlock rather than a slow rank. `--ddp_timeout_hours`
(default 4) sets it. If that line ever reads `0:10:00`, the value did not take;
see the troubleshooting table.

**Why this scales well.** Only the 13.8M generator parameters (~55 MB in bf16)
are all-reduced — the 7B frozen base has no gradients at all. Estimated
communication overhead is 1.5–2.7% of step time, so NVLink is not required and
PCIe boxes are fine. *That estimate is arithmetic, not a measurement.*

Accumulation really is equivalent to the larger batch, and N ranks really do
equal one process with N times the batch — both measured, with controls, in
[Known-good numbers](#known-good-numbers). Two CPU tests, seconds, no GPU.

### Wall-time estimates

Measured: **1.57 optimizer steps/s** on one A100-80GB PCIe, sdpa, per-rank batch
8. Everything below is that number extrapolated at fixed global batch 64
(147,500 steps) — treat as ±20% and re-measure with `smoke.sh` on your hardware.

| | A100-80GB | H100 |
|---|---|---|
| 1 GPU | ~8.7 days | ~5 days¹ |
| 2 GPUs | ~4.4 days | ~2.5 days |
| 8 GPUs | ~27 hours | ~17–18 hours |

¹ matches the "~5 days on 1 H100" in T2L's own README, which is the cross-check
that this arithmetic is right.

## Gates — the checks that must pass

These are not unit tests for their own sake. Each one catches a failure that is
*silent* — the code runs, the loss falls, and every number is wrong.

| Gate | Command | What it catches |
|---|---|---|
| **upstream API** | `python -m inlet.test_upstream_api` | AST only — imports nothing, so it needs no torch, no GPU and no install; it runs before `setup_env.sh` has finished. Inlet is an overlay on an upstream it does not pin, so four things can rot underneath it: an imported symbol that no longer exists, a call whose signature changed, a inlet→inlet call broken by your own refactor, and a `--flag` that is no longer a dataclass field. All four otherwise surface as a `TypeError` *after* 14 GB of weights have loaded. Each of the four was mutation-tested. |
| **m=0** | `python -m inlet.gate_m0` | Prepending *zero* vectors must reproduce a plain forward pass to floating-point noise. If not, the sequence is being assembled wrong — wrong axis, wrong mask, wrong label shift. Verified: `abs delta = 0.000e+00`. |
| **train == eval** | `python -m inlet.test_consistency_inlet --task arc_challenge` | Training builds `[B, m+T, d]` on dim 1; vLLM eval builds `[m+T, d]` on dim 0 with no batch axis. Two implementations that must agree, and the m=0 gate cannot see the difference. |
| **train == eval** | `python -m inlet.test_train_eval_agree` | CPU, a second, no model and no vLLM. Training assembles `[B, m+T, d]` on dim 1; eval assembles `[m+T, d]` on dim 0 and hands it to vLLM. Two implementations of one operation — if they drift, the learned prompt is scored in a context it was never trained in and **nothing raises**. This gate did not exist until 2026-08-24: `test_consistency_inlet` compares two *training-side* implementations, `gate_m0` short-circuits at m=0, and `--zero-prompt` never executes the eval-side `cat` at all — so flipping `dim=0` to `dim=1` left every check green. |
| **reference contract** | `python -m inlet.baseline_ref`<br>`python -m inlet.consistency_ref` | CPU, no model, under a second each. On a clean clone the baseline this project is measured against is a *reconstruction* (see below), so these assert that the reconstruction satisfies the contract the **original** was tested against — the five assertions lifted from `baseline_prompt_tuning/test_consistency.py`, plus every deliberately-broken label-masking sample it exists to reject. Evidence, not proof; it is the strongest statement available without that directory. |
| **NEFTune is real** | *(automatic, at startup)* | Not a command — `activate_neftune()` runs two forward passes over the same ids and **raises** unless they differ, then two more in eval mode which must be identical. NEFTune was a no-op in every commit before 2026-08-24 while `--neftune_noise_alpha=5` sat in `train.sh` looking effective. |
| **rank independence** | *(automatic, at startup, 2+ GPUs)* | Each rank draws one integer off the global torch RNG after `set_seed(seed + rank)`; the trainer gathers them and refuses to start if any two match. If ranks shared a stream they would sample identical task batches and N GPUs would be N copies of one gradient at 1/N of the advertised effective batch — with a completely normal-looking loss curve. |
| **DDP equivalence** | `torchrun --nproc_per_node=2 -m inlet.test_ddp_equiv` | CPU, gloo, seconds, no GPU and no dataset. N ranks must compute the same update as one process with N times the batch. If DDP summed gradients instead of averaging them, the effective LR would scale with the GPU count and a 2-GPU rerun of an 8-GPU result would silently be a different experiment. Verified: `8.918e-08`, with the summed-gradient control at `3.328e-01`. |
| **DDP checkpoint** | `python -m inlet.test_ckpt_keys <ckpt>` | Run this once on the first checkpoint a multi-GPU job writes. `accelerator.prepare()` wraps the generator in `DistributedDataParallel`, and a naive `state_dict()` then prefixes every key with `module.`. Nothing complains at save time; the failure appears days later at eval as `Missing key(s): "base", … Unexpected key(s): "module.base", …`. `save_checkpoint()` calls `unwrap_model()` to prevent it — this proves it on a real DDP checkpoint instead of by reading the code. |
| **zero-shot** | `./scripts/eval.sh --zero-prompt` | The eval harness itself, with no prompt at all: it must reproduce the published zero-shot number. The tolerance is **one to two questions**, not an exact match — measured on a second machine (A100-80GB PCIe, vLLM 0.11.1, FLASH_ATTN), arc_challenge scored `65.61` against the recorded `65.70`: one question out of 1172, reproducible across runs and unchanged by switching the tokenizer, i.e. hardware, not this repo. An injection bug does not move one answer, it moves dozens — so the gate `WARN`s past ~2 questions and `FAIL`s past ~4. |
| **untrained ≈ zero-shot** | `./scripts/eval.sh <fresh-ckpt>` | The head is zero-initialised, so an untrained checkpoint should land near zero-shot. A large gap means `base` init or `emb_rms` is wrong. |

## Known-good numbers

Everything here was measured on 1× A100-80GB PCIe with **sdpa** (no flash-attn),
torch 2.9.0, transformers 4.57.6.

**12-dataset, 500-step run** (`done in 0.50h`, checkpoint 55,267,247 bytes):

| step | seen loss | seen acc | unseen loss | unseen acc | bench loss | bench acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.9849 | .3953 | 10.9437 | .6335 | 14.7562 | .4302 |
| 100 | 3.8270 | .4437 | 4.3778 | .6828 | 2.9019 | .7237 |
| 200 | 2.7328 | .5063 | 2.7328 | .7228 | 2.0370 | .8091 |
| 300 | 2.6285 | .5150 | 2.5673 | .7374 | 2.0266 | .8159 |
| 500 | 2.6061 | .5157 | 2.5931 | .7377 | 2.0859 | .8175 |

`prompt_std_across_batch` went 2.3e-05 → 1.26e-04 — growing, not collapsing.

**479-dataset scale check:** `warmed 500/500`, `train batches/epoch=59`,
200 steps in 127 s = **1.57 steps/s**, trainable params `13,814,784` — identical
at 12 and at 479 datasets, as it must be.

**arc_challenge, that 12-dataset checkpoint** (three real description variants
vs three junk ones):

| | run 1 | run 2 | run 3 | mean |
|---|---:|---:|---:|---:|
| real descriptions | 67.49 | 67.24 | 68.00 | **67.58** |
| random descriptions | 67.15 | 66.89 | 66.89 | 66.98 |

Real beats junk with no overlap, but with n=3 vs 3 the best achievable one-sided
p is 1/C(6,3) = 0.05. **Suggestive, not conclusive.** The decomposition it
suggests: 65.70 zero-shot → 66.98 (+1.28, task-agnostic prompt) → 67.58 (+0.60,
description conditioning). The +0.60 is the number the whole project is about,
and it is not yet established at this scale.

### Multi-GPU, verified 2026-08-24 on 2× A100-80GB PCIe

The DDP path is no longer "should work". What was actually run and what it
printed:

| what | result |
|---|---|
| 2 ranks, 60 steps, `--global_tasks_per_step` derived accum | `2.93 it/s`, `done in 0.09h`, exit 0 |
| loss falls | val/benchmark `sft_loss` 14.8238 → 2.8078, `per_token_acc` .4299 → .7742 |
| validation identical to the pre-fix single-rank numbers | val/seen 10.7737/.3916, val/unseen 10.9675/.6321 — unchanged, so unwrapping the model for validation changed nothing but the collectives |
| `--checkpoint_steps=30` | `permanent checkpoint -> …/hypermod_inlet_step30.pt`, 55,267,450 bytes |
| checkpoint is a plain checkpoint | 23 tensors, no `module.` prefix, `strict=True` load → 13,814,784 params, m=32, head=per_slot |
| effective NCCL timeout, read back from the live process group | `4:00:00` on both ranks |

**The `--global_tasks_per_step` equivalence claim, split in two and each half
tested on CPU in seconds** — no GPU, no dataset, run them on a login node:

```
python -m inlet.test_accum
  accum=2 vs batch=16                        9.070e-08     <- fp32 round-off
  the same, with zero_grad misplaced         3.148e-01     <- 31% wrong

torchrun --nproc_per_node=2 -m inlet.test_ddp_equiv
  DDP(2 ranks) vs 1 process, same batch      8.918e-08
  gradients SUMMED instead of averaged       3.328e-01     <- 3,731,188x worse
```

Together these are the whole claim: accumulation equals the larger batch, and
N ranks equal one process with N times the batch. Each test carries a
deliberately-broken control, because a test that cannot fail proves nothing.
The summed-gradient control is not academic — it is exactly what an effective
learning rate silently scaled by the GPU count looks like, and it neither
crashes nor warns.

## What the code does

**Generator** (`inlet/hyper_prompt.py`). Frozen `gte-large-en-v1.5` embeds every
description **once**, before training. Then: `Linear(1024→1024)` + LayerNorm →
2× residual MLP block (hidden 2048) → LayerNorm → head. Two heads:

- `per_slot` (default): `h.unsqueeze(1) + slot_queries[32,1024]` → `Linear(1024→4096)`
- `shared`: `Linear(1024 → 32·4096)`

**Description width** (`inlet/desc_pool.py`, `--desc_slots` / `--cond`). How much
of the description survives the trip into the generator:

| | |
|---|---|
| `--desc_slots 1` (default) | gte's CLS token only — **1024 numbers** to produce 32×4096 = 131,072. This is upstream's `cls_pool`, and every Inlet number before 2026-08-25 was produced this way. |
| `--desc_slots K` | slot 0 is still that CLS vector; slots 1..K-1 are mean-pooled contiguous segments of the description. Strictly more information, `K×` the cache (~125 MB → ~1 GB at K=8). |
| `--cond cross` | the 32 prompt slots **cross-attend** over those K description slots (2 layers, 8 heads, ~4M params), so slot *i* can read a different part of the description than slot *j*. Requires `--desc_slots > 1` and `per_slot`. |

Widened here rather than by forking upstream's dataloader: `pooling_fn` is a
parameter of both `data.get_task_embs` (training) and `utils.embed_texts`
(eval), and nothing upstream constrains the width it returns. So the collator,
the cache, the hierarchical sampler and the 479/31 split are all untouched —
Inlet stays an overlay. `test_upstream_api` check 5/5 pins that contract.

`--desc_slots 1 --cond pooled` reproduces the older computation graph tensor for
tensor, which is what makes a K=1 vs K=8 run a controlled A/B. This is checked
(`inlet.test_desc_cond`), not asserted. **Eval reads K from the checkpoint's own
config, never from a flag** — scoring a K=8 checkpoint under K=1 encoding would
just produce a worse number, with nothing raising.

> Widening K changes the **architecture**, not the training recipe — batch, LR
> schedule and step count are untouched — so K=1 and K=8 runs are comparable
> with each other and with the published T2L numbers. Always say which K.

Output is `P = base + head_out × emb_rms`, where `base[32,4096]` is initialised
from sampled vocabulary token embeddings (Lester et al. 2021, `top_k=5000`) and
the head's output layer is **zero-initialised**. So at step 0 the prompt is
already in-distribution, and `--freeze_head` gives a free base-only control: if
Inlet does not beat it, the generator is not reading descriptions at all.

`emb_rms` (measured 0.002705 for Mistral) rescales the head's output to the
input embedding's own scale, which is why a network with `LayerNorm` outputs can
write into an embedding space three orders of magnitude smaller without a
hand-tuned learning rate.

**Injection** (`inlet/loss.py`, `inlet/sequence.py`). `torch.cat([P, embed(input_ids)], dim=1)`,
with the attention mask extended by ones and the labels by `-100`. NEFTune noise
is applied by upstream's hook on the embedding module, so it reaches the token
embeddings; the soft prompt is deliberately *not* noised.

> NEFTune was a **silent no-op** before 2026-08-24 — the hook was never
> registered and would not have fired if it were. `activate_neftune()` now
> proves at startup that the noise is real and raises if it is not.
> `docs/ENVIRONMENT.md` NEW 22 has both halves of the bug.

**Training** (`inlet/train_inlet.py`). Mirrors upstream `train_custom_sft.py` with
exactly two substitutions: `HyperModulator → HyperPrompt` and
`get_loss_batch → get_loss_batch_inlet`. Under DDP each rank runs its **own**
hierarchical sampler seeded `seed + rank` and gradients are averaged — the
dataloader is deliberately *not* passed to `accelerator.prepare()`, because
upstream uses a custom hierarchical batch sampler that accelerate would silently
reshard.

**Two diagnostics worth watching.** `prompt_norm`, and `prompt_std_across_batch`
— the standard deviation of the generated prompt *across different descriptions
in one batch*. If that trends toward zero the generator has collapsed to a
constant prompt and is ignoring its input, and it shows up there long before any
eval would reveal it. The trainer warns below 1e-3.

**Evaluation** (`inlet/eval_inlet.py`). Swaps one function in-process
(`vllm_eval.eval_model`) and lets upstream own dataset loading, chat templates,
`assistant_prefill`, answer extraction and metrics. T2L Table 2 reports "an
average of three generated LoRAs, each with a different instance of task
descriptions", so each task is scored once per description variant and averaged:
10 benchmarks × 3 variants = 30 evaluations, but only 10 vLLM engine builds,
because the engine is built once per task and `model.soft_prompt` is swapped
between runs.

## Repo layout

```
inlet/
├── scripts/
│   ├── setup_env.sh        disk check + venv + pinned deps + verification
│   ├── common.sh           sourced by all of them: paths, env facts
│   ├── disk.sh             require_disk(); separate because setup_env.sh needs
│   │                       it BEFORE text-to-lora exists and common.sh exits
│   │                       when that checkout is missing
│   ├── warm_cache.sh       download + tokenize 510 datasets (CPU, 20-40 min)
│   ├── smoke.sh            the 20-minute "does this cluster work" check
│   ├── train.sh            <n_gpus> [overrides]   ← the main entry point
│   ├── eval.sh             the 10 Table-2 benchmarks
│   └── run_queue_inlet.sh    ablation sweep: a queue of whole runs
├── inlet/
│   ├── _env.py             finds the text-to-lora checkout; the only path logic
│   ├── hyper_prompt.py     the generator
│   ├── loss.py             injection + the collapse diagnostics
│   ├── train_inlet.py        training loop (DDP)
│   ├── eval_inlet.py         vLLM eval
│   ├── checkpoint.py       load a checkpoint back into a HyperPrompt
│   ├── warm_datasets.py    parallel, failure-tolerant dataset warmer
│   ├── probe_prompt.py     measure P(desc) - base directly, no LLM and no GPU
│   ├── sequence.py         BOTH prompt assemblies, side by side so they can be
│   │                       compared: train [B, m+T, d] dim 1, eval [m+T, d] dim 0
│   ├── baseline_ref.py     reconstruction of the reference soft_prompt_common.py
│   ├── consistency_ref.py  reconstruction of the reference test_consistency.py
│   ├── gate_m0.py          the m=0 gate
│   ├── test_upstream_api.py       does this fit upstream? AST only, no torch
│   ├── test_train_eval_agree.py   train assembly == eval assembly (CPU, 1 s)
│   ├── test_accum.py       accumulation == the larger batch (CPU)
│   ├── test_ddp_equiv.py   N ranks == one process with N x the batch (CPU, gloo)
│   ├── test_ckpt_keys.py   a DDP checkpoint has no `module.` prefix
│   └── test_consistency_inlet.py    needs the reference baseline_prompt_tuning/
├── docs/ENVIRONMENT.md     every environment fact, with how it was verified
├── requirements.txt
└── README.md
```

Outputs go to `train_outputs/` inside this repo (override with
`INLET_OUTPUT_ROOT`), never into the upstream checkout, so `git status` there
stays clean.

## Things that will bite you

**`epochs` is epochs, not steps.** Upstream's loop is
`for _ in range(args.epochs): for batch in train_dataloader:`. At 479 tasks and
8 per batch there are 59 batches per epoch, so the reference `--epochs=20000` is
**1,180,000** single-GPU optimizer steps. Reading it as a step count
under-trains by 59×. `train.sh` prints the resolved step count at startup —
read it.

**No flash-attn, on purpose.** No environment used here has a
wheel for torch 2.9/cu13, so everything runs sdpa and `INLET_NO_FLASH_ATTN=1` is
the default in `common.sh`. If you install flash-attn somewhere, *do not put its
numbers in the same table as sdpa numbers*: floating-point addition is not
associative, so a different attention kernel sums the same values in a different
order, and in bfloat16 (eps = 0.0078, ~65,000× coarser than float32) those
differences compound. Two runs under two kernels are effectively two random
seeds. **One comparison table = one environment.**

**piqa needs `HF_DATASETS_TRUST_REMOTE_CODE=1`.** It is a script-based dataset;
`datasets>=3` loads it but refuses to run its loader without consent. Exactly one
of ~510 repos fails without this and the error mentions neither piqa nor consent.
`common.sh` sets it.

**`src/fishfarm` is a nested checkout.** The importable package is at
`src/fishfarm/fishfarm`, so putting `src` alone on `PYTHONPATH` is not enough.
`common.sh` and `_env.py` both add the inner path.

**vLLM is required for TRAINING too, not just eval.** This surprised us:
`hyper_llm_modulator.utils` reaches `vllm_eval` through its import chain, so
`python -m inlet.train_inlet` dies with `ModuleNotFoundError: No module named
'vllm'` before the first step. Separately, vLLM 0.11.1 ships kernels built
against CUDA 12 while torch here may be cu13; the upstream `env.sh`
`LD_PRELOAD`s the cu12 cublas/cudart to bridge that, and `common.sh` sources it
if present.

**You do not need a Hugging Face token.** Mistral-7B-Instruct-v0.2 and
gte-large-en-v1.5 are both ungated, and every download in QUICKSTART works
anonymously. If you log in anyway, `huggingface-cli login` is the only place a
token should ever be typed -- never commit one; `.gitignore` covers the obvious
names but not a token pasted into a script.

---

## Troubleshooting: symptom → cause → fix

Every entry below is something that actually happened on a clean box, in the
order it happened. They are listed by **what you will see**, because none of
these errors names its own cause.

| You see | It is | Fix |
|---|---|---|
| **Nothing, again — but this time the whole box.** The web terminal connects and renders blank forever, screenshots time out, the console shows CPU/GPU/memory all at 0%, and the pod looks merely idle while it keeps billing | the **volume** is full. A full filesystem does not raise here, it takes the shell down with it, so you cannot log in to read the error. The console's `Disk` column has **two** gauges — container and volume — and it is almost always the volume. Observed 2026-08-24 at 97% | restarting resets the *container* layer and frees nothing on the volume. Free space instead: `rm -rf $HF_HOME/hub/models--*/blobs/*.incomplete`, `pip cache purge`, `rm -rf ~/.cache/uv ~/.cache/pip`, then `du -xh --max-depth=2 . \| sort -h \| tail -20`. Prevention: `require_disk` in `scripts/disk.sh` now gates every entry point, and `PIP_NO_CACHE_DIR=1` removes the ~8 GB row |
| `RuntimeError: NEFTune was requested (alpha=5) but two forward passes over the same ids gave identical embeddings` at startup | exactly what it says: the hook is registered but not firing, so the run would silently use a different recipe from the T2L numbers it is compared against | this is the check working. It means something re-entered `model.eval()` on the embedding module after `activate_neftune()`, or upstream's hook body changed. Fix the cause; `--neftune_noise_alpha=0` disables NEFTune deliberately and prints that it did |
| `RuntimeError: Per-rank sampler independence is broken` at startup on 2+ GPUs | after `set_seed(seed + rank)` two ranks still share an RNG stream, so they sample identical task batches and the effective batch is smaller than every log line claims | upstream's `HierachicalBatchSampler` must draw from the **global** torch RNG inside `__iter__`. If it was changed to capture a `Generator` at construction, re-seeding after `create_dataloaders` is a no-op — seed the sampler's own generator instead |
| `terminate called after throwing an instance of 'std::bad_alloc'` / `Aborted (core dumped)` on `import transformers` | the base image's `torchvision`/`torchaudio` are built against a different torch; the ABI mismatch aborts the process. Not an ImportError. | `pip uninstall -y torchvision torchaudio` |
| `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=1 …, Timeout(ms)=600000)` then `Signal 6 (SIGABRT)` on rank≠0, ~10 min in, **no Python traceback** | validation runs on rank 0 only; every other rank is parked inside a collective for the whole pass. Torch's default NCCL timeout is **10 minutes** and a real validation pass (three splits, one generative) beats that easily. | already fixed — `--ddp_timeout_hours` (default 4). Check the `NCCL collective timeout` line the trainer prints at startup: if it says `0:10:00`, the timeout did not take, and the reason is that something created the process group before the `Accelerator` (see below). |
| the timeout in the source says 1 hour but the crash says `Timeout(ms)=600000` | whichever call creates the process group sets the timeout. `PartialState()` creates it, so `InitProcessGroupKwargs` passed to `Accelerator` afterwards is silently ignored. | pass the timeout to `PartialState(timeout=…)` too. Both now take the same `--ddp_timeout_hours`. |
| `OSError: I/O error: IO Error: Disk quota exceeded (os error 122)` mid-run, on a volume that looks like it has room | a re-download of the same model leaves the **old blobs behind**. `snapshots/<rev>/` is re-symlinked to the new files; the previous ones stay in `blobs/` unreferenced, plus any `*.incomplete`. Measured here: the Mistral cache read 24 GB for a 14 GB model — two orphaned 4.7 GB shards and a 576 MB partial. | prune what the snapshot does not point at:<br>`M=$HF_HOME/hub/models--mistralai--Mistral-7B-Instruct-v0.2`<br>`for b in $M/blobs/*; do ls -l $M/snapshots/*/ \| grep -q $(basename $b) \|\| echo $b; done`<br>then `rm` the listed files and `$M/blobs/*.incomplete`. Downloading with `--exclude "*.bin"` in the first place avoids it. |
| **Nothing.** Validation prints correct numbers, the progress bar prints `0/60`, and then the job never moves again. No error, no traceback; both GPUs at 100% and the log file stops growing | `DistributedDataParallel.forward` broadcasts module buffers on its **first** call, and `no_grad` does not suppress it (it only suppresses the gradient all-reduce). Validation runs on rank 0 only, so that broadcast has no partner. NCCL matches collectives by order, not by kind, so it pairs with rank 1's next barrier and the ranks are one collective out of step forever. | already fixed: validation runs on `accelerator.unwrap_model(hypermod)` and DDP is constructed with `broadcast_buffers=False`. **The general rule: any `if is_main:` block that touches a `prepare()`d module is a collective hazard.** Run it against the unwrapped module. |
| you need to see where a hang actually is, and `py-spy dump` says `Permission Denied` | the container has no `SYS_PTRACE` | the trainer installs a SIGUSR1 handler: `kill -USR1 <pid>` dumps every thread's Python stack to the log. Send it to **both** ranks — the healthy one is parked in a collective, the stuck one is somewhere else. |
| `ConnectionError: Couldn't reach 'allenai/ai2_arc' on the Hub (OfflineModeIsEnabled)` during **eval** | the message blames the network; the cause is local. `common.sh` sets `HF_DATASETS_OFFLINE=1` on purpose (a multi-day job must not stall on the hub), and this eval dataset is simply not in the cache yet — `warm_cache.sh` was skipped, or did not finish. | run `./scripts/warm_cache.sh` — it fetches the 31 eval datasets too. To get one eval through immediately: `HF_DATASETS_OFFLINE=0 ./scripts/eval.sh …`. |
| `ModuleNotFoundError: No module named 'vllm'` from `train_inlet` | vLLM is not eval-only — `hyper_llm_modulator.utils` imports `vllm_eval` | install vLLM even on a training-only box |
| `ImportError: cannot import name 'SUCCESS' from 'evalplus.eval'. Did you mean: '_SUCCESS'?` | PyPI evalplus ≥0.3.1 renamed it; vendored fishfarm wants the old name | install the pinned commit, `--no-deps` (see `requirements.txt`) |
| `ModuleNotFoundError: tempdir` / `wget` / `termcolor` … | evalplus was installed `--no-deps`, so its own leaf imports are missing | `pip install --no-deps tempdir wget termcolor multipledispatch appdirs rich` |
| `ModuleNotFoundError: inflect` / `torchmetrics` / `fasttext` / `colorlog` / `matplotlib` | upstream's transitive deps, unpinned by its pyproject | all listed in `requirements.txt` |
| `AssertionError: Chat template not found for <model>` | upstream reads `chat_templates/{model_path}/chat_template.jinja`; only a few models have one | use `mistralai/Mistral-7B-Instruct-v0.2` (it has one), or add the directory |
| `OSError: I/O error: Disk quota exceeded` mid-download | `huggingface-cli download` fetched **both** `.bin` and `.safetensors`: 24GB instead of 14GB | re-download with `--exclude "*.bin" "*.pth" "*.gguf"` |
| `ValueError: The repository for ybisk/piqa contains custom code…` | piqa is a script-based dataset | `HF_DATASETS_TRUST_REMOTE_CODE=1` (`common.sh` sets it) |
| `FileNotFoundError: Directory data/transformed_datasets/<hash> is neither a Dataset directory nor a DatasetDict` | the warmer was killed mid-write and left a partial cache dir that never heals | delete that hash directory, re-run `warm_cache.sh --only-failed warm_failures.json` |
| `AttributeError: type object 'tqdm' has no attribute '_lock'` | tqdm's class lock is created lazily by the first thread; the warmer's pool races it | already fixed in `warm_datasets.py`; if you see it, you are on an old copy |
| `error: unrecognized arguments: configs/….yaml` | upstream's parser takes the YAML as `argv[1]`, not as a flag, and only ONE dataclass may take CLI overrides | use `scripts/train.sh`, which gets this right |
| `huggingface_hub.errors.LocalEntryNotFoundError: … outgoing traffic has been disabled` | `HF_HUB_OFFLINE=1`, but `gte-large-en-v1.5` resolves its `trust_remote_code` modelling files through the hub | `common.sh` now defaults `HF_HUB_OFFLINE=0` and keeps only `HF_DATASETS_OFFLINE=1` |
| `wandb.errors.errors.UsageError: No API key configured` | the trainer initialises a wandb tracker unconditionally | `common.sh` sets `WANDB_MODE=offline`; set `WANDB_MODE=online` after `wandb login` if you want the dashboard |
| a warm/train job dies when you press Ctrl-C on something unrelated | `nohup … &` alone still shares the terminal's process group | the scripts use `setsid nohup … < /dev/null & disown` |
| `torch.distributed…ChildFailedError` with no Python traceback | the real error is above it, per-rank | `grep -nE "Error\|Traceback" <logfile> \| head -20` |
