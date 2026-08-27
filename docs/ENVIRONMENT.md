# Environment notes — verified on RunPod A100-80GB PCIe, 2026-08-23

Everything below was found by actually building the environment from scratch and
running the code, not by reading a setup guide. Each item marked **NEW** is
something that bit a real install and is not obvious from the dependency files.

## Verified stack

| package | version | note |
|---|---|---|
| torch | 2.9.0+cu128 | `+cu130` also works — same torch, different CUDA build. Fine on sm_80. |
| vllm | 0.11.1 | |
| transformers | 4.57.6 | **NOT** the `4.46.2` that `pyproject.toml` pins — see below |
| peft | 0.20.0 | |
| datasets | 3.5.1 | |
| evalplus | 0.3.0.dev16 | **NEW** — git pin, see below |
| GPU | A100 80GB PCIe, driver 570.195.03 | |

The two vLLM facts this repo depends on, re-verified on this exact build:

    vllm.prompt_adapter gone:     True
    EmbedsPrompt importable:      True
    enable_prompt_embeds accepted: True

## NEW 1 — never `pip install -e .` on the repo

`pyproject.toml` pins `transformers==4.46.2`. The prompt-tuning baseline (and
therefore Inlet) runs on 4.57.6 because vllm 0.11.1 needs it. Installing the repo
as a package silently downgrades transformers and breaks the whole eval path.
Install the repo's *leaf* dependencies only, and re-check versions afterwards.

## NEW 2 — evalplus must come from a git commit, not PyPI

`uv.lock` pins:

    version = "0.3.0.dev16"
    source  = { git = "https://github.com/evalplus/evalplus?rev=1895d2f6aa8895044a7cf69defc24bd57695e885" }

PyPI evalplus 0.3.1 renamed `evalplus.eval.SUCCESS` -> `_SUCCESS`, and the
vendored fishfarm imports the old public name:

    ImportError: cannot import name 'SUCCESS' from 'evalplus.eval'.
    Did you mean: '_SUCCESS'?

    pip install --no-deps "git+https://github.com/evalplus/evalplus@1895d2f6aa8895044a7cf69defc24bd57695e885"

## NEW 3 — PYTHONPATH needs `src/fishfarm`, not just `src`

`src/fishfarm/` is a full nested repo checkout (its own pyproject.toml, tox.ini),
so the importable package lives at `src/fishfarm/fishfarm/`:

    PYTHONPATH=inlet:src:src/fishfarm

With only `src`, you get `ModuleNotFoundError: No module named 'fishfarm.models'`.

## NEW 4 — extra leaf deps not installed by the vllm/transformers set

    inflect torchmetrics rouge-score zstandard tensorboardX hf_transfer matplotlib
    # and, with --no-deps so they cannot move transformers:
    bitsandbytes fasttext-wheel
    # (trl used to be on this line; it is not imported by anything -- see NEW 17 row 1)
    # fishfarm's own: huggingface_hub pydantic colorlog

`inflect` is imported by `hyper_llm_modulator/utils/metric_fns.py`, which is on
the import path of *everything*, so nothing runs without it.

## Storage

`/workspace` on RunPod is a network filesystem (`mfs#...runpod.net`), not local
disk. It survives pod termination, which is what we want for the HF cache, but
model loading is slower than local disk. `HF_HOME=/workspace/.cache/huggingface`.

## NEW 5 — the argument parser is NOT argparse

`hyper_llm_modulator.configs.ArgumentParser` subclasses `HfArgumentParser` and is
used as:

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()          # NOT .parse_args()

`.parse()` takes `argv[1]` as a YAML config path and treats everything after it
as overrides. Consequences for anything built on top of it:

* extra options must be a **dataclass**, not `add_argument` calls — argparse-style
  additions never reach the parsed output;
* flag names use **underscores** (`--n_virtual_tokens`), not dashes;
* the YAML path is positional: `python -m inlet.train_inlet configs/....yaml --lr=...`.

## Upstream API shapes, verified rather than assumed

| call | actual |
|---|---|
| `get_emb_model_and_fns(name, device)` | returns **4**: `emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn` — do not build `pooling_fn` separately |
| `create_dataloaders(...)` | returns `{"train", "val/seen", "val/unseen", "val/benchmark"}` — the three val splits are the same three eval groups the config defines |

## Upstream API shapes, part 2

| call | actual |
|---|---|
| `get_datasets(dataset_names, metadata, ...)` | first arg is a **LIST**. It does `{k: metadata[k]["ds_kwargs"] for k in dataset_names}`, so a bare string iterates its characters and dies with `KeyError: 'l'`. Returns `{name: dataset}`. |

## NEW 6 — `datasets==3.5.1` is load-bearing, and it is why piqa needs a flag

`datasets==3.5.1` is pinned for a specific reason: whatever pip resolves by
default (5.x) **dropped support for script-based HF datasets**, and `ybisk/piqa`
ships a `piqa.py` loading script. On 5.x you get

    RuntimeError: Dataset scripts are no longer supported

3.5.1 is also what the repo's own `uv.lock` pins, so this is not a guessed
downgrade — do not let anything float it.

Corollary, found here: 3.5.1 *supports* script datasets but still refuses to
execute them silently. `warm_datasets.py` failed on exactly one of 33 repos:

    ValueError: The repository for ybisk/piqa contains custom code which must be
    executed to correctly load the dataset.

Fix, verified 2026-08-23 (`warmed 1/1`, n=16113):

    HF_DATASETS_TRUST_REMOTE_CODE=1

This env var must be set for **training, eval and warming alike** — anything
that touches piqa. It is exported by `scripts/run_queue_inlet.sh` and defaulted
inside `warm_datasets.py` so it cannot be forgotten.

## NEW 7 — evalplus: patch on disk, do not patch in process

I fixed the `cannot import name 'SUCCESS' from 'evalplus.eval'` break by
installing the `uv.lock` git commit (NEW 2). The same break can also be fixed by
appending to the *installed* file, which is worth knowing because it constrains
any future fix:

    # .venv-vllm/lib/python3.10/site-packages/evalplus/eval/__init__.py
    TIMEOUT = "timeout"
    SUCCESS = PASS          # appended after the TIMEOUT marker

The reason that alternative exists is the important part:

> vllm forces multiprocessing 'spawn' for its engine subprocess, which
> re-imports evalplus fresh from disk, so an in-process patch would not survive.

Both approaches are on-disk and therefore both survive the spawn. What would
**not** work is monkeypatching `evalplus.eval.SUCCESS` from Python at startup.

## NEW 8 — fishfarm may be pip-installed rather than on PYTHONPATH

    "$PIP" install ./src/fishfarm

Some environments install it as a package, which supersedes NEW 3 there:
`PYTHONPATH=src/fishfarm` becomes unnecessary. Keep the PYTHONPATH form on a
rented pod (nothing is installed there on purpose, so the checkout stays the
single source of truth), but `run_queue_inlet.sh` must tolerate both — having
`src/fishfarm` on PYTHONPATH when the package is also installed is harmless.

## NEW 9 — vllm 0.11.1 is a hard floor on a CUDA 13 driver

    vllm < 0.11.1  ->  undefined symbol: cublasHgemm

Install with `--torch-backend auto`. 0.11.1 works on both CUDA 12 and CUDA 13
drivers, so pinning it keeps every environment on the one version that matters
most.

## Restricted clusters (do not port these workarounds to the pod)

Some clusters this has been run on cannot reach the open internet, and need
local adjustments that would be wrong on a rented pod. They are listed as
*shapes* rather than as any site's actual settings:

* **Outbound fetches go through a proxy, and github.com egress may be blocked.**
  Anything that clones from GitHub — the evalplus git pin above is the one that
  bites — has to be vendored or replaced by the on-disk patch, which is why that
  section patches instead of pinning.
* **A site-wide pip config can force `--user`**, which breaks venv installs.
  Override it for the install.
* **The interpreter may not be the system python.** Pass `PYTHON=` to
  `setup_env.sh` rather than relying on what is first on PATH.
* **CUDA runtime libraries may need `LD_PRELOAD`.**

If you hit one of these, fix it in your own environment; none of it belongs in
this repo.

## NEW 10 — installing evalplus `--no-deps` leaves four holes that only bite at import

evalplus has to go in with `--no-deps` (otherwise it drags a dependency
resolution that can move transformers). The cost is that its own runtime imports
are missing, and they surface *far* from the install — the traceback that
actually appeared was on `python -m inlet.train_inlet`:

    inlet/train_inlet.py -> inlet/loss.py -> hyper_llm_modulator.sft_trainer
      -> utils.eval_hypermod -> vllm_eval -> fishfarm.tasks.evalplus
      -> evalplus.data.humaneval -> evalplus.data.utils
    ModuleNotFoundError: No module named 'tempdir'   (then 'wget', ...)

i.e. **every** import of `inlet.loss` pulls the whole eval stack, so a missing
evalplus leaf dep kills *training* too, not just eval. Install alongside it:

    pip install --no-deps tempdir wget termcolor multipledispatch appdirs rich

Verified afterwards that nothing moved: transformers 4.57.6, datasets 3.5.1,
vllm 0.11.1.

## NEW 11 — the upstream parser takes exactly ONE dataclass, and it is unforgiving

Three separate failures, all from the same place (`configs.py: ArgumentParser`):

1. **`parse()` dispatches on `sys.argv[1].endswith(".yaml")`.** With a YAML plus
   overrides it calls `parse_yaml_and_args(argv[1], argv[2:])`; with no YAML it
   falls through to plain argparse — which is what produces the misleading
   `error: unrecognized arguments: configs/....yaml` if anything upstream of it
   went wrong.
2. **Overrides must be `--key=value`**, because it does
   `arg.split("=")[0].strip("-")`. `--key value` silently becomes garbage.
3. **Two dataclasses cannot both take CLI overrides.** The override loop runs
   once *per dataclass* and the `else` branch is
   `raise ValueError(f"Argument provided not found in dataclass: {arg}")` — so
   passing `--run_name=...` with `(TrainingArguments, InletArguments)` dies while
   the loop is still on `TrainingArguments`. The only shape that works is a
   single dataclass, hence `class InletArguments(TrainingArguments)`.
4. With one dataclass, `parse()` returns **the object, not a 1-tuple**.

## NEW 12 — `save_dir` and `run_name` are namespace attributes, not fields

`train_custom_sft.py` sets them by hand after parsing:

    args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    args.save_dir = f"train_outputs/sft/{args.exp_setup}/{args.run_name}"
    logger = create_logger(args.save_dir, debug=args.debug)

So `create_logger()` takes a **log_dir** (it writes `debug.log` into it) and the
directory has to exist before training starts. Inlet makes `run_name` a real,
user-settable field — a timestamp+uuid name is unusable for `eval_inlet`, which
has to find the run again — and derives
`train_outputs/inlet/{exp_setup}/{run_name}`.

## The exact T2L Mistral recipe (scripts/train_t2l_mistral.sh), verbatim

    configs/hyper_lora_decontam_lol_tasks.yaml
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2
    --emb_model=Alibaba-NLP/gte-large-en-v1.5
    --warmup_frac=0.2 --lr=2.5e-5 --n_tasks_per_batch=8
    --n_points_per_task=1 --grad_accum_steps=1
    --epochs=20000 --n_descs_per_ds=128 --n_train_ds=479
    --exp_setup=hyper_lora --encoder_type=linear
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1
    --neftune_noise_alpha=5 --weight_decay=1e-2

Two things to note. **The YAML's own `model_dir` is a Llama-3.1-8B local path**
(`models/Llama-3.1-8B-Instruct/`), so `--model_dir` is not optional — without it
you get `HFValidationError: Repo id must be in the form 'repo_name' or
'namespace/repo_name'`. And `--l2_reg_generated_w` is the LoRA-weight
regulariser; Inlet's analogue is `--l2_reg_prompt`, so that flag is dropped rather
than translated.

## NEW 13 — three more things that only show up once the run actually starts

* **FlashAttention2 is the upstream default.** `get_model_and_tokenizer(...,
  use_flash_attn=True)`. The RunPod pod has no matching `flash_attn` wheel for
  torch 2.9.0+cu128 / py3.12, so it dies with
  `ImportError: FlashAttention2 has been toggled on, but ... flash_attn seems to
  be not installed`. `INLET_NO_FLASH_ATTN=1` falls back to sdpa **on the debug pod
  only** — never set it for a real run.
* **Only the train dataloader is `accelerator.prepare()`d.** The three val
  dataloaders are not, so their batches arrive on the CPU and the first
  `validate()` call (which runs at step 0, before any training) dies with
  `Expected all tensors to be on the same device`. Inlet moves the batch inside
  `loss_fn`, which covers both paths and is a no-op for train batches.
* **`nohup cmd > log 2>&1 &` after a `&&` does NOT detach from the caller's
  pipe.** `cd X && ENV cmd > log 2>&1 &` backgrounds the *whole list*, so bash
  forks a subshell whose own stdout is still the caller's pipe, and it sits there
  waiting for python. Anything reading that pipe (subprocess.run, a notebook
  cell) blocks for the entire training run. Wrap the list instead:
  `( cd X && ENV cmd ) > log 2>&1 &`.

## Measured on the pod, 2026-08-23 (A100-80GB PCIe, sdpa, 12 datasets)

Smoke run `smoke12`: 500 optimizer steps, `--n_train_ds=12`, otherwise the
verbatim T2L Mistral recipe. **0.50 h wall**, of which most is validation:
`val_freq=100` with `max_batches=50` over three splits costs ~7 min per call.

    [gate] m=0 reduces to plain forward (16.375000) -- injection point OK
    trainable params: 13,814,784  (22 tensors)
    checkpoint hypermod_inlet.pt = 55,267,247 bytes  (13.8M x fp32)

| step | seen loss | seen acc | unseen loss | unseen acc | bench loss | bench acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.9849 | .3953 | 10.9437 | .6335 | 14.7562 | .4302 |
| 100 | 3.8270 | .4437 | 4.3778 | .6828 | 2.9019 | .7237 |
| 200 | 2.7328 | .5063 | 2.7328 | .7228 | 2.0370 | .8091 |
| 300 | 2.6285 | .5150 | 2.5673 | .7374 | 2.0266 | .8159 |
| 400 | 2.6245 | .5158 | 2.5856 | .7347 | 2.0711 | .8182 |
| 500 | 2.6061 | .5157 | 2.5931 | .7377 | 2.0859 | .8175 |

Train-split loss falls monotonically; both held-out splits bottom out at step
300 and tick back up — mild overfitting on 12 datasets, which is the expected
shape and is itself evidence the held-out numbers measure generalization.

`prompt_std_across_batch` (the collapse alarm): 2.3e-05 at step 90 -> 1.26e-04
by step 400. Growing, not collapsing. Zero-init makes it start at 0 by
construction, so the *trend* is the signal, not the level.

Eval plumbing gate:

    [arc_challenge__zero_prompt] {'zero_prompt': {'acc': 0.6569965870307167}}
    zero-shot reproduction: got 65.70, expected 65.70 -> PASS

Note `val/seen` and `val/unseen` both read 2.7328 at step 200. The accuracies
differ (.5063 vs .7228) so the splits are genuinely different data; recorded
here as a coincidence to re-check on the next run rather than ignore.

## Eval on the trained smoke checkpoint — arc_challenge, all 6 description tags

    eval_descs__0      67.49        <- real descriptions
    eval_descs__1      67.24
    eval_descs__2      68.00
    random_descs__0    67.15        <- junk: "dogs;cats;bananas;", noise, "ggggg..."
    random_descs__1    66.89
    random_descs__2    66.89
    zero_prompt        65.70        <- m=0, the frozen-model baseline

Three things this establishes, in order of how load-bearing they are:

1. **Every prompted number differs from 65.70.** The generated prompt actually
   reaches the model through vLLM's `prompt_embeds` at eval time. A silently
   dropped prompt is the failure mode that would waste an entire training run,
   and it is now ruled out by measurement rather than by reading the code.
2. **Real > junk with no overlap** (min real 67.24 > max junk 67.15).
   mean(real)=67.58, mean(junk)=66.98, gap +0.60. Under a random relabeling of
   the six numbers, a perfect split happens with probability 1/C(6,3)=0.05, so
   this is suggestive at one-sided p=0.05 — *not* conclusive, and it should not
   be reported as more than that. It is a 500-step, 12-dataset run.
3. **Junk still beats zero-prompt by +1.28.** This is the `base` parameter doing
   its job: even a meaningless description yields `base` plus a near-zero
   residual, and `base` alone is a task-agnostic soft prompt. The design
   therefore decomposes the gain cleanly:

       65.70  ->  66.98   +1.28  task-agnostic base prompt
       66.98  ->  67.58   +0.60  description conditioning

   which is exactly the split `--freeze_head` is built to measure as an ablation.

## NEW 14 — two things that only appear at 500-dataset scale

Warming all 500 (479 train + 31 eval, deduped) surfaced two failures that the
33-dataset smoke subset never hit:

* **tqdm's class lock is created lazily, in whichever thread touches it first.**
  With `--workers 12`, several threads race on the very first `datasets`
  progress bar and a handful of datasets die with
  `AttributeError: type object 'tqdm' has no attribute '_lock'` — at random, so
  it looks like a flaky dataset rather than a threading bug. Fix is one line in
  the *main* thread before the pool exists:

      from tqdm import tqdm as _tqdm
      _tqdm.set_lock(threading.RLock())

* **Killing the warmer poisons its cache.** A dataset interrupted mid-write
  leaves a partial dir, and every later run fails on it with

      FileNotFoundError: Directory data/transformed_datasets/<hash> is neither a
      `Dataset` directory nor a `DatasetDict` directory.

  This never heals on its own — the failure is now the cache, not the download.
  Delete the named `<hash>` directory and re-run with `--only-failed`.

* And a matching footgun in the shell: `pkill -f warm_datasets.py` **kills the
  shell running it**, because that shell's own command line contains the
  pattern. Same class of bug as `pgrep -f inlet.train_inlet` looping forever against
  itself. Use the bracket trick: `pkill -f "warm[_]datasets"`.

## Full-scale check: 479 datasets, measured 2026-08-23

    train batches/epoch = 59
    HyperPrompt: emb_rms = 0.002705,  ||base|| = 0.154
    trainable params: 13,814,784  (22 tensors)     <- IDENTICAL to the 12-dataset run
    [gate] m=0 reduces to plain forward (18.375000) -- injection point OK
    {"run_name":"scale479","steps":200,"best_val_loss":2.9262,
     "wall_s":428.62,"trainable_params":13814784}

**The parameter count does not move between 12 and 479 tasks.** That is the whole
argument for Inlet over per-task methods, and it is now measured rather than
multiplied out on paper:

| method | params | to cover all 479 tasks |
|---|---|---|
| Oracle LoRA | 3.4M / task | ~1.63 B |
| Prompt tuning | 131k / task | ~63 M |
| **Inlet** | **13.8M, once** | **13.8 M** |

### Throughput, and what it implies for a full run

200 steps in 127 s of pure training = **1.57 steps/s (0.64 s/step)** on one
A100-80GB **with sdpa**; FlashAttention2, where available, should be faster
still, so treat this as a floor. Extrapolating the real recipe:

> **CORRECTED 2026-08-24.** The three bullets that used to sit here read
> `--epochs=20000` as a STEP count and concluded "~3.5 h/GPU" and "9-10 h for a
> complete config". Both were wrong by 59x. `epochs` is epochs: upstream's loop
> is `for _ in range(args.epochs): for batch in train_dataloader:`, and at 479
> tasks / 8 per batch there are 59 batches per epoch. README.md always had this
> right; this file did not. The corrected arithmetic:

```
--epochs=20000 x 59 batches/epoch      = 1,180,000  single-GPU optimizer steps
                                       = 9,440,000  task-samples  (x8 per step)

at --global_tasks_per_step=64:
  1,180,000 x 8 / 64                   =   147,500  optimizer steps
```

At the measured 1.57 steps/s (1x A100-80GB PCIe, sdpa, per-rank batch 8):

| | steps | wall clock |
|---|---:|---:|
| 1x A100 | 1,180,000 | **8.7 days** |
| 2x A100 | 147,500 | ~4.4 days |
| 8x A100 | 147,500 | **~27 hours** |
| 8x H100 | 147,500 | ~17-18 h (extrapolated) |

The 8x H100 row is the cross-check: T2L's own README says ~5 days on one H100,
and 1,180,000 steps / 5 days = 2.73 steps/s, which is 1.74x our A100+sdpa rate.
That ratio conflates hardware with FlashAttention2, so pure-hardware is probably
nearer 1.5x -- but it lands in the right place, which is the point.

* Validation is still a trap: `val_freq=500` with `max_batches=50` over three
  splits costs ~7 min a call. At 147,500 steps that is 295 calls = **34 hours**,
  more than the training. Use `val_freq=5000` or cut `max_batches`.
* Full eval is 10 benchmarks x 3 description variants = 30 passes, separate,
  ~4-8 h. (An earlier note here said 289 passes; that counted all 31 eval tasks
  x ~9 description splits, which is the in-training validation surface, not
  T2L's Table 2 protocol.)

---

## NEW 15 — packaging this as a standalone repo (2026-08-23)

Everything above was written against the **overlay layout**: `inlet/` living
inside a `text-to-lora` checkout, with `PYTHONPATH=inlet:src:src/fishfarm` and cwd
at the upstream root. That layout is still supported, but it cannot be pushed to
its own git repo without dragging upstream along.

`inlet/_env.py` now owns all of the path logic and supports both layouts:

| layout | how it resolves |
|---|---|
| `T2L_ROOT` exported | used as-is, validated |
| this repo IS the checkout | `./src/hyper_llm_modulator` exists |
| sibling / parent / `third_party/` | searched, three levels up |

`bootstrap()` runs at import time in every entry point. It inserts
`INLET_ROOT`, `$T2L_ROOT/src` and `$T2L_ROOT/src/fishfarm` onto `sys.path` and
**chdir's to `$T2L_ROOT`**, because the YAML config path, `models/` and
`data/transformed_datasets` are all relative to it. Inlet's own outputs go to
`$INLET_OUTPUT_ROOT` (default `<repo>/train_outputs`), so a `git status` in the
upstream checkout stays clean.

Evaluation additionally needs `baseline_prompt_tuning/` on the path — it imports
`get_tokenizer`, `load_input_embeddings`, `BASE_MODEL`, `ZERO_SHOT` from it, on
purpose, so the Inlet eval and the prompt-tuning eval cannot drift. Override with
`INLET_BASELINE_DIR`. **Training does not need it**, which matters: a training-only
box needs neither that directory nor vLLM.

## NEW 16 — DDP: the four things that were actually wrong

Multi-GPU support is `torchrun` + accelerate, with the dataloader deliberately
**not** passed to `accelerator.prepare()` (upstream's hierarchical batch sampler
would be silently resharded). Each rank runs its own sampler seeded
`args.seed + process_index`; gradients are averaged. Four bugs had to be fixed
to make that correct, and all four are the silent kind:

1. **`optimizer.zero_grad()` sat before `accelerator.backward()`.** At
   `grad_accum_steps=1` this is harmless, which is why it survived the smoke run
   — and at any higher accumulation it throws away every micro-batch but the
   last. It now goes after `optimizer.step()`.
2. **`curstep` incremented per micro-batch.** So `logging_freq`, `val_freq` and
   the total step count all meant "micro-batches" whenever accumulation was on.
   It now advances only when `accelerator.sync_gradients` is true.
3. **Checkpoints were saved from the wrapped module**, giving every key a
   `module.` prefix that would fail `load_inlet_checkpoint`'s `strict=True` load.
   `save_checkpoint` now takes the accelerator and calls `unwrap_model` — a
   no-op on one GPU, so single- and multi-GPU checkpoints are byte compatible.
4. **Every rank called `create_logger`**, which makes the directory and opens
   `debug.log` for writing — eight ranks interleaving one file and racing on
   mkdir. Rank 0 now owns the log, and every `logger.*` call inside `main()` is
   behind `if is_main`.

Validation and checkpointing run on **rank 0 only**, each followed by
`wait_for_everyone()`. That is safe because a `no_grad` forward triggers no
gradient all-reduce, so the other ranks are not waiting on a collective that
never comes. The NCCL collective timeout is raised to one hour
(`InitProcessGroupKwargs`) so a slow NFS checkpoint write cannot be mistaken for
a dead rank.

**`--global_tasks_per_step` is the reproducibility knob.** It fixes the number
of task descriptions per optimizer step globally and derives gradient
accumulation from the world size, so 2 GPUs and 8 GPUs execute the same
optimization — same batch, same step count, same LR schedule — and differ only
in wall time. It refuses to start on a non-divisible combination rather than
rounding.

**Communication cost.** Only 13.8M trainable parameters (~55 MB in bf16) are
all-reduced; the 7B frozen base has no gradients. At the measured 1.57 steps/s
that is an estimated 1.5–2.7% of step time, so PCIe is fine and NVLink is not
required. **Calculated, not measured** — no multi-GPU run has been executed yet.
The first 8-GPU run should be checked against it.

---

## NEW 17 — a clean-box install, in the order the failures actually arrive (2026-08-24)

Rebuilt the whole environment from scratch on a fresh 2× A100-80GB PCIe pod
(`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) to produce an install guide
that is *verified* rather than *believed*. Every item below cost one failed
launch. They are listed in the order they appeared, because that is the order
the next person will meet them.

| # | Symptom | Cause |
|---|---|---|
| 1 | `datasets==3.5.1` + `trl==1.9.2` → `ResolutionImpossible` | genuine pin conflict. **Fixed 2026-08-24: `trl` is gone from `requirements.txt`.** Nothing imports it -- `trl_activate_neftune` and `neftune_post_forward_hook` are defined in `hyper_llm_modulator/sft_trainer.py` itself, and there is no `import trl` anywhere in either repo. Do not re-add it |
| 2 | `std::bad_alloc` / `Aborted (core dumped)` on `import transformers` | a `torchvision` built for a different torch; the ABI mismatch aborts the process. **Not an ImportError**. **Corrected 2026-08-24:** the old fix (uninstall torchvision) does not hold -- vLLM *depends* on torchvision, so step 5 reinstalls it and the old presence check then failed the whole install. `setup_env.sh` now installs `torchvision==0.24.0` from the same index as torch, in the same command, and verifies by running `import torchvision, transformers` in a subprocess |
| 3 | `ModuleNotFoundError: inflect` | `hyper_llm_modulator.utils` imports `metric_fns` at package import time |
| 4 | `ModuleNotFoundError: torchmetrics` | same chain |
| 5 | `AssertionError: Chat template not found for <model>` | upstream reads `chat_templates/{model_path}/chat_template.jinja`; only google/gemma-2-2b-it, meta-llama/Llama-3.1-8B-Instruct and mistralai/Mistral-7B-Instruct-v0.2 ship one |
| 6 | `ModuleNotFoundError: matplotlib` | utils draws figures at import |
| 7 | `ModuleNotFoundError: vllm` **from the trainer** | the utils chain reaches `vllm_eval`. **vLLM is not eval-only** |
| 8 | `ModuleNotFoundError: fasttext` | `pip install fasttext-wheel` (source build otherwise) |
| 9 | `ImportError: cannot import name 'SUCCESS' from 'evalplus.eval'` | PyPI evalplus ≥0.3.1 renamed it; use the git commit upstream's `uv.lock` pins, `--no-deps` |
| 10 | `ModuleNotFoundError: colorlog` | vendored fishfarm's `logging.py` |
| 11 | `OSError: I/O error: Disk quota exceeded` at ~24GB | `huggingface-cli download` fetched **both** `.bin` and `.safetensors`. `--exclude "*.bin"` |
| 12 | `wandb.errors.UsageError: No API key configured` | `init_trackers(log_with="wandb")` is unconditional. `WANDB_MODE=offline` |

**Total: twelve launches to get one that starts.** All twelve are now encoded
in `requirements.txt`, `scripts/setup_env.sh` and `scripts/common.sh`, and
restated as a symptom→cause→fix table in the README.

Two of them are worth singling out because they mislead rather than merely
block:

* **#2 does not raise a Python exception.** The process aborts in C++. Nothing
  in the traceback mentions torchvision, because there is no traceback.
* **#7 contradicts what this repo's own README used to say.** "Training does
  not need vLLM" was written from the architecture (Inlet never generates text
  during training) and was wrong about the import graph.

### Also fixed while here

* `mistralai/Mistral-7B-Instruct-v0.2` is **no longer gated** — `model_info`
  succeeds anonymously. An earlier note assuming it needs an accepted licence
  is out of date.
* `eval_inlet.py` no longer *requires* the reference `baseline_prompt_tuning/`
  directory. It prefers it when present (so the two eval paths still cannot
  drift), and falls back to `inlet/eval_common.py` otherwise. Before this, the
  repo could train but not evaluate on any machine but one.
* The web terminal drops long-idle sessions. Background anything long with
  `setsid nohup … < /dev/null & disown`, not plain `nohup … &` — a Ctrl-C
  aimed at an unrelated foreground `sleep` killed a dataset warm that was
  nominally nohup'd.

---

## NEW 18 — the thirteenth failure, and it only appears on 2+ GPUs

The first real 2-GPU run reached step 0, validated `val/seen` and `val/unseen`
correctly, and then died at 01:40:35 with:

```
[rank1] Watchdog caught collective operation timeout:
        WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
                 Timeout(ms)=600000) ran for 600001 milliseconds before timing out.
terminate called after throwing an instance of 'c10::DistBackendError'
...
rank : 1 (local_rank: 1)  exitcode : -6   traceback : Signal 6 (SIGABRT)
```

Three things about this failure are worth writing down, because none of them
is visible from the message.

**1. It is not a deadlock.** `NumelIn=1` is a scalar all-reduce, and `SeqNum=4`
means it is the fourth collective of the job. Rank 1 was doing exactly what the
design says it should: waiting at a barrier while rank 0 validates alone.
Validation is three splits; two of them measured ~2.5 min each and the third
generates. The wait exceeded ten minutes, and NCCL's watchdog cannot tell
"slow" from "dead" — it takes the process down to avoid running later kernels
on corrupted state.

**2. The timeout in the source was one hour. The one that ran was ten minutes.**
`Timeout(ms)=600000` is torch's *default* NCCL timeout. The code said

```python
ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=1))
accelerator = Accelerator(..., kwargs_handlers=[ddp_kwargs])
```

which is correct in isolation and useless here, because a few lines earlier the
trainer calls `PartialState()` to read the world size before it can size
gradient accumulation. **`PartialState()` is what creates the process group.**
By the time the `Accelerator` is constructed the group already exists,
accelerate keeps it, and the kwargs are dropped without a warning. The timeout
has to be given to whichever call creates the group:

```python
ddp_timeout = timedelta(hours=args.ddp_timeout_hours)
world = PartialState(timeout=ddp_timeout).num_processes
...
kwargs_handlers=[InitProcessGroupKwargs(timeout=ddp_timeout)]
```

**3. A configured value that silently does not apply is worse than no value.**
The trainer now reads the timeout back off the live process group and prints it
at startup:

```
NCCL collective timeout : 4:00:00
```

If that line ever says `0:10:00` again, something re-introduced an early
process-group creation. Check it on the first cluster run; it costs nothing and
it is the only way to see this before it costs ten minutes and a job.

`--ddp_timeout_hours` defaults to **4**.

### The related design cost, not a bug

Validation runs on rank 0 only, so on 8 GPUs seven of them idle through every
validation pass. With `val_freq=10000` (upstream's default, unchanged) a
147,500-step run validates ~15 times; at ~10 min each that is ~2.5 h of 7-GPU
idle out of ~27 h, about 9%. That is a known, accepted cost — sharding
validation across ranks would remove it but is not worth the correctness risk
before the first full run. If you later want it, the change is in `validate()`
and its two call sites, both already guarded by `accelerator.wait_for_everyone()`.

---

## NEW 19 — the fourteenth failure: rank-0-only validation is not free under DDP

With the timeout raised, the 2-GPU run got further and then stopped dead:

```
[step 0] val/seen:      sft_loss=10.7737 per_token_acc=0.3916 prompt_norm=0.1543
[step 0] val/unseen:    sft_loss=10.9675 per_token_acc=0.6321 prompt_norm=0.1543
[step 0] val/benchmark: sft_loss=14.8238 per_token_acc=0.4299 prompt_norm=0.1543
  0%|          | 0/60 [00:00<?, ?it/s]
```

and then nothing, for as long as you leave it. No error. No traceback. `nvidia-smi`
shows both GPUs at 100% utilisation and both processes at ~100% CPU, because a
rank spinning in an NCCL collective is indistinguishable from a rank doing work.
The log file stops growing; GPU memory stops changing. Eight minutes in, the
numbers were byte-identical to the first minute.

### What is actually happening

`DistributedDataParallel.forward` broadcasts the module's buffers on its **first**
call. `require_forward_param_sync` is initialised to `True`, and the pre-forward
hook fires before anything clears it. `torch.no_grad()` does **not** prevent this
— `no_grad` suppresses the *gradient* all-reduce, which is a different collective.
`HyperPrompt` registers `emb_rms` as a buffer, so there is something real to
broadcast.

Validation runs on rank 0 only. So rank 0 issues a broadcast that no other rank
ever posts.

NCCL matches collectives by **order on a communicator**, not by kind or by size.
That stray broadcast pairs up with whatever rank 1 posts next — its
`wait_for_everyone()` barrier — and from that moment the two ranks are
permanently one collective out of step. The first gradient all-reduce of training
waits for a partner that will never arrive.

This is exactly the case torch's own timeout message warns about, in a sentence
that is easy to read past:

> This is most likely caused by incorrect usages of collectives, e.g., wrong
> sizes used across ranks, the order of collectives is not same for all ranks

It is invisible on one GPU. It is invisible in code review, because the code
that is wrong is the code that looks most obviously right:

```python
if is_main:
    validate(model, hypermod, ...)     # "only rank 0, so no collectives" -- false
accelerator.wait_for_everyone()
```

The docstring on `validate()` even said so: *"a forward pass under no_grad
triggers no gradient all-reduce, so the other ranks are not waiting on a
collective."* True about gradients, and wrong about buffers.

### The fix

Two locks on the same door:

1. **Validate the unwrapped module.** `accelerator.unwrap_model(hypermod)`
   returns the plain `HyperPrompt`: same parameters, same device, no hooks, no
   collectives, and a no-op on one GPU. `val_loss_fn` closes over that instead
   of over the DDP wrapper.
2. **`DistributedDataParallelKwargs(broadcast_buffers=False)`.** `emb_rms` is
   derived from the frozen embedding table, so it is bit-identical on every rank
   and never updated. There was never anything to broadcast.

Either alone is sufficient. Both, because this failure costs a day to find and
five characters to reintroduce.

### The general rule this is an instance of

**Any `if is_main:` block that touches a `prepare()`d module is a collective
hazard.** Not just backward — buffer sync, `all_gather` inside a metric,
`accelerator.gather`, anything. When only one rank runs code that can talk to
the others, the ranks stop agreeing on the order of collectives, and NCCL will
not tell you; it will hang, or worse, silently transfer the wrong bytes.

If you must run something on one rank, run it against the **unwrapped** module.

### Debugging a hang, since this will happen again

`py-spy` is the usual tool and it does not work here: the pod's container has no
`SYS_PTRACE`, so `py-spy dump` returns `Permission Denied`. The trainer therefore
installs a `SIGUSR1` handler:

```bash
kill -USR1 <rank0-pid>; kill -USR1 <rank1-pid>
```

Every thread's Python stack goes to the log. Send it to **both** ranks and
compare: the healthy rank is parked in a collective (`backward`, `barrier`,
`all_reduce`); the stuck rank is somewhere else, and that somewhere is the bug.

Do **not** use `faulthandler.dump_traceback_later(repeat=True)` for this. It
fires on whatever thread is running, and walking the frames of a thread inside a
CUDA kernel launch killed a run here with `exitcode: -11` (SIGSEGV) instead of
producing a usable trace. On-demand only.

One more environment note learned the hard way: the RunPod web terminal drops
the session during any long `sleep`, and the dropped shell takes the foreground
job with it unless it was started with `setsid nohup … < /dev/null & disown`.
Poll with short commands against the log file instead of sleeping in the shell.

---

## NEW 20 — two silent LR bugs that only exist above one GPU

Neither of these crashes, warns, or shows up on a single GPU. Both were found by
review, then confirmed against accelerate's source, after the DDP path was
already "working".

### 20a. `split_batches=True` was silently discarded

`train_inlet.py` constructed the Accelerator as

```python
Accelerator(mixed_precision="bf16", split_batches=True, ...)
```

which raises nothing on accelerate 1.14 and does nothing. In that version the
argument survives only as a signature placeholder with a sentinel default:

```python
_split_batches = object()
...
def __init__(self, ..., split_batches: bool = _split_batches, ...):
```

and **there is no code in `__init__` that copies it into the dataloader
config**. The live value comes from a property:

```python
@property
def split_batches(self):
    return self.dataloader_config.split_batches      # DataLoaderConfiguration() -> False
```

That matters here for exactly one reason. No dataloader is prepared in this
trainer, so `split_batches` has a single effect —
`AcceleratedScheduler.step()`:

```python
if self.split_batches:
    self.scheduler.step(...)                 # once per optimizer step
else:
    num_processes = AcceleratorState().num_processes
    for _ in range(num_processes):           # num_processes times!
        self.scheduler.step(...)
```

With it off, the cosine schedule is consumed `world` times too fast. On 8 GPUs,
a 147,500-step run finishes its 29,500-step warmup at optimizer step **3,687**
and bottoms out at step **18,437** — after which `transformers`' cosine lambda,
driven past `progress=1.0`, walks the LR back **up**. Roughly 87% of the run
would be on a schedule nobody designed. On 1 GPU `num_processes == 1` and none
of it happens, which is why it survived every single-GPU test.

Fixed by passing `dataloader_config=DataLoaderConfiguration(split_batches=True)`
and then asserting the value came back:

```python
assert accelerator.split_batches is True
```

A configured value that silently fails to apply is worse than no value. This is
the second instance of that exact shape today (the first was the NCCL timeout),
which is why both now read the setting back rather than trusting it.

### 20b. `lr_world_scale` scaled the LR by GPU count

```python
_scale = {"linear": num_processes, "sqrt": num_processes ** 0.5, "none": 1.0}[...]
lr = args.lr * _scale
```

Batch-size LR scaling is a real technique, and this was not it. Once
`--global_tasks_per_step` fixes the global batch, gradient accumulation absorbs
the GPU count: `train.sh 1`, `train.sh 2` and `train.sh 8` all present the same
64 task descriptions per optimizer step. Scaling by world size on top of that
is not batch scaling — it is just a different learning rate:

| command | global batch | steps | LR actually used |
|---|---:|---:|---:|
| `train.sh 1` | 64 | 147,500 | 2.50e-5 |
| `train.sh 2` | 64 | 147,500 | **3.54e-5** |
| `train.sh 8` | 64 | 147,500 | **7.07e-5** |

...while `scripts/train.sh` shipped `--lr_world_scale=sqrt` in the recipe and
the README called the three configurations reproductions of each other.

Now keyed off the **global batch** relative to `--lr_reference_batch` (64, the
batch the T2L LR was tuned at), so it is 1.0000 for every single-node config and
only engages when the batch genuinely changes (64 GPUs at `GLOBAL_TASKS=512` ->
sqrt(8) = 2.83). The trainer prints the arithmetic:

```
LR: 2.500e-05 x 1.0000 = 2.500e-05  [global batch 64 vs reference 64, 'sqrt']
    -- independent of GPU count by construction
```

### The pattern in all four of today's bugs

NCCL timeout, buffer broadcast, `split_batches`, `lr_world_scale`: every one is
a setting that *looks* applied, is not, and produces no error. Three of the four
are invisible on one GPU. The repo's answer is the same in each case — read the
value back at runtime and print it, so the first ten lines of a log say what is
actually in effect rather than what the source asked for.

---

## NEW 21 — running the volume out of space does not raise, it wedges the box (2026-08-24)

**What it looked like.** The RunPod A100 box stopped responding partway through
`setup_env.sh`. Not "slow" — dead. The gotty web terminal accepted the
websocket, rendered forty blank rows, and never wrote a byte; the browser tab
reported `Script injection timed out after 5000ms` on every screenshot;
`nvidia-smi` and every other diagnostic was out of reach because there was no
shell to run them in. The pod console showed CPU 0%, GPU 0%, memory 0%. It looks
exactly like an idle machine, and it bills exactly like a busy one.

**What it was.** The console's `Disk` column carries *two* gauges. Container disk
was at 0%. **Volume disk was at 97%.** Everything `setup_env.sh` writes —
`$INLET_ROOT/.venv`, `$INLET_ROOT/.hf`, the transformed-dataset cache — lands on the
volume, because `INLET_ROOT` is the checkout and the checkout is on `/workspace`.

**Read the right gauge.** A restart is the reflex, and here it is the wrong one:
restarting resets the *container* layer and leaves the volume exactly as full as
it was. `df -h` inside the pod distinguishes them; the console's two rings do
too, if you notice there are two.

**The budget, measured rather than guessed:**

| item | GB |
|---|---|
| venv with torch 2.9+cu128 (the `nvidia-*` wheels dominate) | ~13 |
| `mistralai/Mistral-7B-Instruct-v0.2`, safetensors only | 14.5 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1.7 |
| raw HF datasets + upstream `TRANSFORMED_DS_DIR` | ~5 |
| pip's wheel cache, if you let it keep one | ~8 |
| checkpoints (13.8M params × fp32 = 55 MB each) | <1 |
| **total, pip cache kept** | **~43** |
| **total, `PIP_NO_CACHE_DIR=1`** | **~35** |

A 50 GB volume is where this bites: it fits, until the pip cache is also there.

**What the repo does about it now.**

* `scripts/disk.sh` — `require_disk <GB> [path]`. Its own file, not `common.sh`,
  because `setup_env.sh` must check disk *before* it clones text-to-lora and
  `common.sh` hard-exits when that checkout is missing. Both directions of the
  helper are exercised in the pre-commit run.
* `setup_env.sh` step 1/8 requires **60 GB** free, step 7/8 requires **25 GB**
  still free after the install so the 14.5 GB download cannot die halfway and
  leave `.incomplete` blobs that read as corruption on the next attempt.
* `train.sh`, `eval.sh`, `smoke.sh` require 20/10/20 GB.
* `PIP_NO_CACHE_DIR=1` is exported by `disk.sh`, which removes the ~8 GB row.
* `INLET_SKIP_DISK_CHECK=1` overrides. It will not end well.

---

## NEW 22 — NEFTune was a silent no-op, twice over (2026-08-24)

`scripts/train.sh` has passed `--neftune_noise_alpha=5` since the first commit,
the README described the noise reaching the token embeddings but not the soft
prompt, and `inlet/loss.py` had a docstring paragraph about the asymmetry. None of
it was happening. **Two independent reasons, either one sufficient:**

1. `train_inlet.py` never called `trl_activate_neftune`, so the forward hook was
   never registered. `neftune_noise_alpha` is a field on upstream's
   `TrainingArguments`, so the flag parsed cleanly and went nowhere.
2. The hook body is `if module.training:` (`sft_trainer.py:86`). Inlet calls
   `model.eval()` on the frozen base model and never calls `model.train()`.
   Upstream calls `model.train()` at `sft_trainer.py:184`, *before* activating at
   `:211`. So even a registered hook would not have fired.

This is the shape of bug that survives review: no error, no warning, a
completely normal loss curve, and a recipe that quietly differs from the T2L run
the numbers are being compared against.

**The fix, and why it is narrower than upstream's.** `activate_neftune()`
registers the hook and puts **only the embedding module** in train mode, where
upstream puts the whole model there. For Mistral those are the same thing —
verified against transformers v4.57.1 `modeling_mistral.py`, which contains
exactly one dropout call, `nn.functional.dropout(attn_weights, p=dropout,
training=module.training)` in `eager_attention_forward`, with `dropout` coming
from `config.attention_dropout`, default `0.0`; the sdpa and flash paths this
repo runs never reach that function, and `MistralRMSNorm` keeps no running
statistics. Narrowing it to the embedding keeps the "base model is frozen" claim
literally true for a base model that *does* have dropout, where upstream's
`model.train()` would start perturbing it.

**Ordering, which is load-bearing:**

* activation happens **after** the m=0 gate and **after** step-0 validation.
  Both compare a Inlet forward against a plain frozen-model forward, and NEFTune
  draws fresh noise per call, so activating earlier makes the gate compare two
  different random perturbations and fail.
* `validate()` flips the embedding to `eval()` and restores it, mirroring
  upstream's `evaluating(model, hypermod)` context manager. Without this every
  validation loss is measured on noised embeddings and is not comparable to the
  checkpoint it selects.
* the handle is removed at teardown, as upstream does at `sft_trainer.py:313`.

**The check matters more than the fix.** `activate_neftune()` runs two forward
passes over the same ids and **raises** unless they differ — and a third and
fourth with the module in eval mode, which must be identical. A dead hook now
stops the run instead of silently changing the experiment.

---

## NEW 23 — per-rank sampler independence was asserted in a comment, not in code

`train_inlet.py` re-seeds with `set_seed(args.seed + accelerator.process_index)`
after `accelerator.prepare()` so each rank draws a different stream of tasks.
That is load-bearing: if it fails, N GPUs stop being an N-times bigger batch and
become N copies of the same gradient, at 1/N of the effective batch size that
every log line, the LR scale and the paper all claim. The loss curve looks
completely normal either way.

It works today only because upstream's `HierachicalBatchSampler` draws from the
**global** torch RNG lazily inside `__iter__` (`data.py:99` `torch.randperm`,
`:107` `torch.randint`) rather than from a `Generator` captured at construction.
Re-seeding after the sampler exists would be a no-op against the other design.

There is now a check: one draw off the global RNG per rank, `accelerator.gather`,
and a hard failure if any two ranks match. It costs one integer and it is the
difference between a comment and a fact.

---

## NEW 24 — an overlay needs a check that it still fits (2026-08-24)

Inlet imports upstream's metadata, sampler, dataloaders, loss and eval harness
rather than copying them. That is the whole design, and it is also a standing
liability: **upstream is a moving target this repo does not pin.** When it moves
the failure is not subtle, but it is *late* — argument parsing succeeds, the
model loads, 14 GB of weights come off disk, and then a `TypeError`.

`python -m inlet.test_upstream_api` is the cheap check that runs first. Pure AST on
both sides: it imports nothing, needs no torch, no vLLM, no GPU and no install,
so it runs on a laptop or before `setup_env.sh` has built the venv. Four checks:

| # | check | the failure it catches |
|---|---|---|
| 1 | every symbol Inlet imports from upstream still exists | a rename upstream |
| 2 | every call into upstream matches its signature | arity/keyword drift — e.g. `get_emb_model_and_fns` returning four values, not three |
| 3 | every inlet → inlet call matches | your own refactor |
| 4 | every `--flag` in `scripts/*.sh` is a real dataclass field | upstream's parser raises on the first unknown override, two seconds into a three-day job |

**Resolve by module, not by name.** The first version reported three failures
that did not exist. `get_tokenizer` is both `hyper_llm_modulator.utils`' function
and a zero-argument method on fishfarm's `VLLMModel`; matching by bare name found
the wrong one and declared every call site broken. The checker now resolves each
name against the module it was actually imported from.

**Mutation-tested, because a check that cannot fail proves nothing.** Against a
scratch copy, with the real text-to-lora as reference:

```
baseline                                              PASS
import a symbol upstream does not have             -> FAIL  upstream no longer provides ['no_such_function_xyz']
add a bogus keyword to compute_loss(...)           -> FAIL  unknown keyword(s) ['bogus_kwarg_xyz']
call activate_neftune with 5 positional args       -> FAIL  5 positional args, definition takes 3
add --totally_made_up_flag to train.sh             -> FAIL  not a field on InletArguments
restored                                              PASS
```

Verified against the real checkout: 19 imported symbols across 9 upstream
modules, 24 calls into upstream, 160 inlet → inlet calls, 22 CLI flags. Wired into
`setup_env.sh` step 2 (right after the checkout is located, before the venv) and
`smoke.sh` step 0.

**What it does not do.** Signatures, not semantics. A function can have the right
shape and the wrong behaviour — that is what `gate_m0`, `test_train_eval_agree`
and the smoke run are for.

---

## NEW 25 — the rank-RNG probe was testing the wrong generator (2026-08-24)

Caught by reading, not by running, and worth recording because it is the exact
shape of mistake the check was written to prevent.

NEW 23 added a probe: each rank draws one integer after
`set_seed(seed + rank)`, the trainer gathers them, and refuses to start if any
two ranks match. First version:

```python
rng_probe = torch.randint(0, 2 ** 31 - 1, (1,), device=device)   # CUDA
```

`device` is the accelerator, so that consumes the **CUDA** generator. The thing
the check is about — upstream's `HierachicalBatchSampler` — consumes the **CPU**
generator (`torch.randperm` at `data.py:99`, `torch.randint` at `:107`, both on
CPU). `set_seed` seeds both from the same value, so in practice the two agree and
the probe would have looked like it worked.

That is the failure mode, not a mitigation. A check whose result is *correlated*
with the property it asserts passes for the wrong reason, and it does so most
reliably in exactly the situation where nothing is wrong yet. The probe now draws
on CPU and moves to the device only for the gather, because NCCL gathers device
tensors.

Cost of getting it wrong: one extra draw off the CPU generator per rank, identical
on every rank, which shifts each rank's task stream by one and changes nothing
about their independence.

---

## NEW 26 — `df` does not see a volume quota, so the disk gate could not fire where it mattered (2026-08-24)

NEW 21 added `require_disk` after a pod filled its volume to 97% and wedged.
Measured on a fresh RunPod A100 the same day, that check would **not have
fired**:

```
$ df -PT /workspace
Filesystem                     Type  ...  Avail  Mounted on
mfs#ca-mtl-3.runpod.net:9421   fuse  ...   292T  /workspace

$ df -PB1G /workspace | awk 'NR==2{print $4}'
298426
```

The pod's "120 GB volume" is a mount of a shared network filesystem. `df`
reports the **whole cluster** — 755 TB total, 292 TB free — while the pod is
limited to its quota. `require_disk 60` reads 298426 and passes, on exactly the
machine where the failure happened.

A check whose result is unrelated to the property it asserts is worse than no
check: it produces a green line before the failure it was written to prevent.

There is no portable way to read the quota from inside the container, so
`disk.sh` now says so instead of pretending. Above 5 TB reported free it prints
`inlet_quota_warning`: df is showing a shared filesystem, the real limit is the
volume quota, df cannot see it, and the number to trust is the pod dashboard's
**volume** gauge — not the container one, and not this check. Both branches
(ordinary filesystem, silent; simulated 292 TB, warns) were exercised.

The threshold-and-warn behaviour is deliberate over a hard failure: plenty of
real clusters have a genuine multi-TB scratch filesystem, and refusing to start
on those would be wrong.

---

## NEW 27 — the base models are not gated

Checked directly rather than assumed:

```python
>>> from huggingface_hub import model_info
>>> model_info('mistralai/Mistral-7B-Instruct-v0.2').gated
False
>>> model_info('Alibaba-NLP/gte-large-en-v1.5').gated
False
```

So `huggingface-cli login` is **not required** to fetch either. It is still
worth doing — anonymous pulls are rate-limited and less reliable for a 14.5 GB
download — but nobody is blocked waiting for a licence acceptance, and an agent
running this does not need a token to get to a first training step.

---

## Verified on real hardware, 2026-08-24 — A100-80GB PCIe, torch 2.8.0+cu128

The three CPU gates, re-run under real PyTorch after first passing against a
numpy stand-in. Source transferred and **sha256-verified** before running
(`ef75a40f…ba045`), so this is the same code that is in the repo:

```
python -m inlet.test_train_eval_agree
  ok  train[b] == eval(prompt[b], tok[b]) for all 3 rows, shape (3, 11, 5)
  ok  prompt occupies [0, m) and tokens [m, m+T) on both paths
  ok  mask extended with ones, labels with -100, originals intact
  ok  supervised target set unchanged under the shift (18 tokens)
  ok  m=0 is the identity on both paths (None and 0-row)
  ok  both paths cast the fp32 prompt to the token dtype identically
  ok  [m, d] broadcasts to every row and matches the eval path
  ok  all 5 deliberately-broken controls are detected
  PASS  exit 0

python -m inlet.baseline_ref       PASS  exit 0
python -m inlet.consistency_ref    PASS  exit 0
```

The bfloat16 check is the one that mattered here: the stand-in emulated bf16 as
float32 truncated to 8 mantissa bits, and this run confirms the two assemblies
round identically under the real dtype.

Two more, on the same box, both with their deliberately-broken controls:

```
python -m inlet.test_accum
  accum=2 vs batch=16   max relative param diff : 9.070e-08
  same, with zero_grad misplaced                : 3.148e-01
  PASS  (tolerance 1e-04; the known-bad ordering is 3470892x worse)

python -m torch.distributed.run --nproc_per_node=2 -m inlet.test_ddp_equiv
  ok    rank-0-only forward on the UNWRAPPED module issued no collective
        (the barrier after it completed; against DDP(...) this would hang)
  world size                              : 2
  global batch                            : 8 (4 per rank)
  DDP(2 ranks) vs 1 process, same batch   : 8.918e-08
  gradients SUMMED instead of averaged    : 3.328e-01   <- must be caught
  PASS  (tolerance 1e-05; the known-bad variant is 3731188x worse)
```

Those reproduce the numbers recorded on 2026-08-23 to the digit, on different
hardware and a different torch build — and the first line of `test_ddp_equiv`
is the standing regression check for the rank-0-validation hang (NEW 19).

**Still not executed (as of 2026-08-24 morning; all three ran later the same day -- see NEW 28):** `gate_m0`, `test_consistency_inlet`, and the trainer end
to end (NEFTune activation, the per-rank RNG gather). Those need the full
upstream checkout, the 14.5 GB of weights and the warmed dataset cache.
`smoke.sh` is where they all run, and it is therefore the acceptance test.

---

## NEW 28 — the clean-box install, run again on 2026-08-24 (2× A100-SXM4-80GB)

The whole of `setup_env.sh` was run from an empty `/workspace` on a fresh pod
(`torch 2.9.0+cu128`, driver 570.172.08, 120 GB **real** block device, so
`require_disk` was live and `inlet_quota_warning` correctly stayed silent). It
failed twice before it finished. Both failures were in the repo, not the box,
and both are now fixed — but note what they have in common: **each was already
described in this document and still shipped broken.** Prose in `docs/` does
not help an agent that runs the script.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ERROR: Cannot install -r /tmp/inlet_req.txt (line 4) and datasets==3.5.1` → `ResolutionImpossible`, step 5/8, install dead | `requirements.txt` pinned `trl==1.9.2`. Nothing imports `trl`: `trl_activate_neftune` and `neftune_post_forward_hook` are defined in `hyper_llm_modulator/sft_trainer.py` itself, and `grep -rn '^\s*(from trl\|import trl)'` over both checkouts returns nothing. Upstream's own `uv.lock` pins trl 0.12.2, so my 1.9.2 was never upstream's version either | `trl` removed from `requirements.txt`, with a comment saying not to re-add it |
| 2 | `torchvision STILL INSTALLED -- uninstall it or transformers will core dump`, step 6/8 exits 1, steps 7 and 8 never run | The check tested **presence**. vLLM *depends* on torchvision, so step 5 always reinstalls it after step 4's `pip uninstall`. Following the message would have broken vLLM | `setup_env.sh` now installs `torchvision==0.24.0` from the same index as torch **in the same command**, so step 5 finds it satisfied; and the check is now behavioural — it runs `import torchvision, transformers` in a subprocess and fails only on a non-zero exit (the version is read from metadata, not by importing, so a genuinely broken ABI cannot abort the checker itself) |
| 3 | `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available` — every model download fails, 0 bytes moved | The RunPod image exports `HF_HUB_ENABLE_HF_TRANSFER=1` in the **container environment** (not in any file we control, so nothing in this repo could have overridden it), and `hf_transfer` was in NEW 4's prose list but never in `requirements.txt` | `hf_transfer` added to `requirements.txt` |

Also removed: the instruction to run `huggingface-cli login`. Both models are
ungated (NEW 27), so an anonymous download works — and an unattended agent told
to log in stops and waits for a token that never arrives. `setup_env.sh` step 8,
`QUICKSTART.md` §2 and `README.md` now all say a token is not needed.

### What then ran, on the same box

```
setup_env.sh                8/8, clean, from an empty /workspace
  test_upstream_api           19 symbols / 24 upstream calls / 160 inlet calls
                              / 22 flags -- against a REAL fresh clone of
                              SakanaAI/text-to-lora, not a stand-in
  verify                      torch 2.9.0+cu128 cuda=12.8 gpus=2,
                              transformers 4.57.6, accelerate 1.14.0,
                              datasets 3.5.1, peft 0.20.0, vllm 0.11.1,
                              torchvision 0.24.0 (imports with transformers),
                              flash_attn absent, inlet.train_inlet imports cleanly

weights                     Mistral-7B-Instruct-v0.2 + gte-large-en-v1.5,
                            20 GB, NO huggingface-cli login

warm_cache.sh --n-train 12  warmed 33/33 in 1.3 min, no failures

gate_m0
  reference loss (plain forward) : 12.9375000000
  Inlet loss, m=0                  : 12.9375000000
  |delta|                        : 0.000e+00
  GATE m=0 reduces to plain forward -> PASS
  GATE description path dormant 1 step, then fully connected -> PASS
  peak GPU mem: 14.58 GiB
```

### The 2-GPU run, and the two things that stopped it

```
GLOBAL_TASKS=16 ./scripts/train.sh 2 --run_name=smoke2 --n_train_ds=12 \
    --n_descs_per_ds=8 --max_steps=200 --val_freq=100 --logging_freq=20
```

**Stop 1 — DDP raced itself building the dataset cache.** Rank 1 died with

```
FileNotFoundError: Directory data/transformed_datasets/94ca448f... is neither
a `Dataset` directory nor a `DatasetDict` directory.
```

while rank 0's log showed `Saving the dataset (0/1 shards)` on that exact hash,
same second. Upstream's cache-hit test (`data.py:128`) is

```python
if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
    tokenized_dataset = datasets.load_from_disk(...)
```

— the directory **existing** counts as built, and `save_to_disk` creates it
before filling it. Two ranks, one writer, one reader of a half-written
directory. With 8 ranks it is 8 writers on one path.

Fixed in `train_inlet.py`: the `create_dataloaders` call is now inside
`accelerator.main_process_first()`. Rank 0 builds, everyone else waits, then
they all read a complete cache. One serialized build, first run only.

Do not assume a warmed cache makes this unreachable — see the next paragraph.

**What `warm_cache.sh` actually covers.** It warms the *training* representation
of each task. `create_dataloaders` builds four splits, and three of them hash
differently:

| split | why the hash differs from the warmed one |
|---|---|
| `train` | for any task that is also in `eval_ds_info`, upstream **rewrites** `ds_kwargs["split"]` to `train[:90%]` (`sft[:90%]` for longreward) before hashing |
| `val/seen` | same tasks again at `train[90%:]` |
| `val/unseen` | the held-out tasks, never in `train_ds_names` |
| `val/benchmark` | eval tasks with `BENCHMARK_TASK_INFO` merged into `ds_kwargs` |

So the first training run always builds *something* — on this box, warming 33
datasets left 93 directories after one run. That is expected, not a warm
failure. It costs a few minutes before step 1 and it happens once.

**Stop 2 — `train.sh` ran under the system python.** Launched from a shell
where `.venv` had not been activated, the run died with

```
ModuleNotFoundError: No module named 'wandb'
```

from `/usr/local/lib/python3.12/dist-packages/torch/...` — the system torch,
not the venv's. Two causes, both fixed:

* `common.sh` now activates `$INLET_ROOT/.venv` when `VIRTUAL_ENV` is unset, and
  then refuses to continue unless `import torch, wandb` succeeds, with a message
  that names `setup_env.sh`. Every script sources `common.sh`, so this covers
  all of them.
* `train.sh` launched bare `torchrun`, which resolves off `PATH` and can belong
  to a different interpreter than the one `common.sh` chose. It now runs
  `"$PYBIN" -m torch.distributed.run`. Every other script already used `$PYBIN`;
  this was the only hole.

**Then it ran, clean, with no venv activated by hand:**

```
optimizer: base lr=1.25e-05  head lr=1.25e-05 (head_lr_mult=1.0)
world=2  per-rank batch=8  grad_accum=1  -> GLOBAL BATCH 16 tasks/step
[ddp]     per-rank RNG streams verified distinct across 2 ranks
[gate]    m=0 reduces to plain forward (16.375000) -- injection point OK
[neftune] alpha=5.0 ACTIVE on Embedding (max |delta| over one probe: 5.493e-02);
          off in eval mode, so validation and the m=0 gate are clean
loss 14.0777 -> 10.5887 -> 3.0048 -> 2.2209   (steps 20 / 40 / 100 / 200)
[step 200] val/seen 3.0393  val/unseen 3.0222  val/benchmark 2.2203
           (per_token_acc 0.4830 / 0.7196 / 0.7988)
done in 0.18h -- train_outputs/hyper_lora/smoke2/hypermod_inlet.pt
{"steps": 200, "best_val_loss": 3.0393, "world_size": 2, "grad_accum_steps": 1,
 "global_tasks_per_step": 16, "trainable_params": 13814784}
```

That `[neftune]` line is the first time the 2026-08-24 NEFTune fix has been seen
firing on real hardware: the hook is registered, its probe measures real noise,
and it is off during validation and during the m=0 gate — which is why the gate
still reads exactly the frozen-model loss.

**The `prompt_std_across_batch` warning at steps 20 and 40** ("the generator is
emitting nearly the same prompt for every description", 1.1e-05 of `|P|`) is the
diagnostic working, not a failure: the head starts near-constant and separates
as it trains. It is gone by step 200.

### And the eval half

```
TASKS=arc_challenge ./scripts/eval.sh --zero-prompt

Evaluating soft prompt: zero_prompt  (m=0)
Processed prompts: 1172/1172 [00:16<00:00, 72.64it/s]
[arc_challenge__zero_prompt] {'zero_prompt': {'acc': 0.6561433447098977}}
zero-shot reproduction: got 65.61, expected 65.70 -> PASS (1.0 question of 1172)
```

vLLM 0.11.1 loads, the m=0 eval path assembles the same sequence the trainer
does, and the number lands one question away from the published zero-shot
baseline. Install → weights → warm → gates → 2-GPU training → checkpoint →
eval has now been executed end to end on one machine.

**Nothing in the "still not executed" list above remains.** `gate_m0`,
`test_consistency_inlet` and the trainer end to end have all run.

### `./scripts/smoke.sh 2` — the acceptance test, green

After the four fixes above, the whole script was run once more from a shell with
no venv activated, and finished:

```
### 0/4  does this checkout still fit the text-to-lora next to it? ###   PASS
### 1/4  no-GPU consistency checks (train path == eval path) ###
  ALL CHECKS PASSED
  1b/4  gradient-accumulation equivalence                               PASS
  1b2/4 train-side vs eval-side prompt assembly                         PASS
  1b3/4 the reconstructed reference checks its own contract             PASS
  1c/4  DDP equivalence (gloo, 2 ranks)                                 PASS
### 2/4  m=0 gate ###                                                   PASS
### 3/4  short training run on 2 GPU(s): loss must fall ###             ok
### 4/4  zero-prompt eval must reproduce the published zero-shot ###
  zero-shot reproduction: got 65.61, expected 65.70 -> PASS
smoke OK -- see train_outputs/hyper_lora/smoke2/train_summary.json
```

This is the state the repo is being handed over in.

## NEW 29 — a venv on the network volume does not fail, it hangs (2026-08-25)

**Symptom.** `bash scripts/setup_env.sh` reaches

```
=== 4/8  torch, matched to the driver ===
  using cu130 wheels
```

and then prints nothing for half an hour. Nothing errors. `pip` is alive and the
network is fine — `curl -sI https://download.pytorch.org/...` returns HTTP 200 in
0.08 s — so it reads as a slow download and the instinct is to keep waiting. On a
rented 2×A100 that instinct costs $3.21/hr.

The tell is that pip's own I/O counters are **frozen**:

```
$ P=$(pgrep -f "pip install"); grep -E '^rchar|^wchar' /proc/$P/io; sleep 5; grep -E '^rchar|^wchar' /proc/$P/io
rchar: 6868453326      rchar: 6868453326      <- not one byte in 5 s
wchar: 5216871291      wchar: 5216871291
$ ps -o wchan -p $P
request_wait_answer                            <- blocked in FUSE, not on a socket
```

**Root cause.** The pod's `/workspace` is a MooseFS volume mounted over FUSE
(`df -PT` reports fstype `fuse`). The default venv path is `$INLET_ROOT/.venv`,
which put it there. Installing torch writes tens of thousands of small files,
which is the worst case for FUSE, and the process parks in
`request_wait_answer` indefinitely.

Measured on the same pod, same command, only the venv path changed:

| venv on | bytes written in 6 s | outcome |
|---|---|---|
| `/workspace` (fuse) | **0** | wedged after ~25 min, killed |
| `/root` (overlay) | **741 MB** | 13 MB → 4.6 GB in ~2 min |

The weights are *not* affected — a few large files read sequentially is what a
network volume is good at. It is specifically the venv that must be local.

**Fix (in the code, not here).** `setup_env.sh` already accepted `VENV=`, but
nothing told you when you needed it and the failure was silent. It now reads the
venv parent's filesystem type and **refuses** on `fuse*`/`nfs*`/`cifs*`/`9p`/
`lustre`/`ceph`/`glusterfs`, printing the remedy:

```
VENV=/root/venv-inlet bash scripts/setup_env.sh
```

`INLET_ALLOW_NETWORK_VENV=1` overrides it. Verified on the pod that the check
fires on the path that actually hung and passes on the one that worked:

```
/workspace/inlet   fstype=fuse      -> REFUSE (network)
/root            fstype=overlay   -> allow  (local)
```

**Also worth knowing.** A stopped RunPod pod can lose its GPUs to another user.
Restarting then offers only "CPU Only"/fewer GPUs, and the reason appears only in
a dialog reached through the pod row's ⋮ → Start Pod. "Automatically migrate your
Pod data" starts an identical pod immediately, but the volume copy is not
necessarily quick — on 2026-08-25 it sat at "getting volume information" with
`/workspace` still empty after 15 minutes, and a from-scratch install was faster.

## NEW 30 — a wedged volume hangs DDP and writes a checkpoint that looks fine (2026-08-26)

Two failures, one cause, both silent. Same pod and same FUSE volume as NEW 29.

### Symptom 1: the job finishes training and then hangs forever

`smoke.sh 2` completed all 200 steps and all three step-200 validations, wrote
its last log line, and then sat for 23 minutes producing nothing while both GPUs
stayed allocated and the pod kept billing. The give-away was **asymmetric** GPU
use — 0% on one card, 100% on the other — which is *not* the pattern README
describes for a hang (it warns that both sit at 100%).

`kill -USR1` on every rank, as QUICKSTART instructs, produced the answer at once:

```
rank 1 (pid 7178)  accelerator.wait_for_everyone()   train_inlet.py:935   <- at the barrier
rank 0 (pid 7177)  wchan = request_wait_answer                          <- stuck in FUSE
```

Rank 0 wedged on a filesystem write after the checkpoint save; rank 1 waited at
the barrier that follows it. Neither will ever move.

### Symptom 2: the checkpoint is the right size and cannot be loaded

That run's `hypermod_inlet.pt` is **55,267,695 bytes** — exactly what a healthy one
weighs. It fails on load:

```
EOFError: Ran out of input
  torch/serialization.py:1802 in _legacy_load
```

Nothing at write time noticed. The file passes every check anyone would think to
run — it exists, it is not empty, it is the expected size — and the training that
produced it is gone by the time anyone tries to use it.

### Root cause

`INLET_OUTPUT_ROOT` defaults to `$INLET_ROOT/train_outputs`, which on this pod is the
MooseFS/FUSE volume. Checkpoint writes go there. When the mount wedges, the write
neither completes nor errors.

### Fixes (in the code)

1. **`save_checkpoint` now reads the file back** immediately after writing and
   checks that `state_dict` and `config` are present and that the tensor count
   matches what was written. A second on 55 MB, and it tests the failure that
   actually occurred rather than guessing at filesystem types. It raises with the
   remedy — point `INLET_OUTPUT_ROOT` at local disk — instead of leaving a file
   that will be trusted later.
2. `setup_env.sh` already refuses to build the venv on a network filesystem
   (NEW 29). The same reasoning applies to the output root, but the read-back
   check subsumes it: it catches truncation from any cause, including ones that
   have nothing to do with FUSE.

### Operationally

On a rented box, put **both** the venv and `INLET_OUTPUT_ROOT` on local disk and
leave only the model weights on the network volume:

```bash
VENV=/root/venv-inlet INLET_OUTPUT_ROOT=/root/outputs HF_HOME=/workspace/hf ./scripts/train.sh 2 ...
```

Weights are a few large files read sequentially, which is what a network volume
is good at. Checkpoints and venvs are not.
