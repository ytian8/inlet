# Run 1 — the step/scale curve

This is a complete, self-contained instruction for one training run and its
evaluation. Follow it top to bottom. Everything it asks you to check exists
because skipping it has produced a wrong number at least once.

---

## 1. What this run is for

Inlet generates an **input-layer soft prompt** (32 vectors prepended to the
embedded input) from a task description. T2L generates a **LoRA injected into
every layer's q_proj/v_proj**. The paper's claim is about the interface: the
input layer is the only substrate that can be delivered through a black-box
embedding API, and we want to show it carries task adaptation about as well as
per-layer weight injection.

The blocker was that the generated prompt's **magnitude** grows during training
until it destroys long-form generation. Measured on a 1000-step checkpoint:

| | gsm8k | mbpp | humaneval | 5 multiple-choice | 8-task avg |
|---|---|---|---|---|---|
| frozen model | 39.95 | 42.86 | 39.63 | — | — |
| inlet as trained | 28.51 | 7.77 | 18.29 | 67.96 | 49.29 |
| inlet, prompt x0.5 | 42.00 | 45.61 | 42.07 | 67.71 | **58.53** |

It is a length collapse: humaneval completions go from 18 words back to 131.
Performance traces an inverted-U in prompt norm peaking near **0.11**, about
0.7x a real token embedding (0.1543). Training overshoots it.

**This run produces:** a step curve, a `prompt_norm` curve, and six checkpoints,
so we can find where the scaled score peaks and how many steps we actually
need. Report as "N steps", never rounded up.

---

## 2. Machine

2x A100-80GB (or any 2 GPUs with 80GB). One GPU works too — pass `1` instead of
`2` to `train.sh`; the global batch is pinned by `--global_tasks_per_step` so
the result is the same, just slower (~4.5 s/step on 1 A100, ~2.2 s/step on 2).

**Disk — measured, not estimated:**

| | size | where it must live |
|---|---|---|
| `.venv` | 8.2 GB | **local disk** (pip onto a network mount wedges silently) |
| `HF_HOME` (models + processed datasets) | **32 GB** | anywhere with room; a network volume is fine, it is read-mostly |
| `third_party/.../data/transformed_datasets` | 6.7 GB | with the repo |
| checkpoints (6 x 122 MB) + eval scratch | ~2 GB | with `INLET_OUTPUT_ROOT` |

That is ~47 GB before any headroom, and `smoke.sh` additionally refuses to
start unless **20 GB** is free. **A 60 GB local disk is not enough for all of
it** — on a 60 GB box, `smoke.sh` stops with a disk-check failure after
everything is installed. Either give the repo's filesystem ~90 GB, or put
`HF_HOME` on a separate large volume, which is what §3 does.

---

## 3. Setup

**Clone onto local disk, not a network volume.** The venv goes in
`<repo>/.venv`, and pip installing onto a FUSE-mounted network volume wedges
silently — the I/O counters freeze and it looks like a slow download.
`setup_env.sh` refuses outright if it detects this. On a RunPod box `/root` is
local and `/workspace` is the network volume, so:

```bash
cd /root
git clone https://github.com/ytian8/inlet.git && cd inlet
bash scripts/setup_env.sh                      # ~15 min, no GPU needed

cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- see below
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh
```

Those last five lines are the **standard preamble**. Every shell in this
document starts with them, and they are repeated each time rather than assumed.

**Both exports must precede `source scripts/common.sh`.** `common.sh` fills each
one in only when it is unset, and everything downstream reads its *derived*
values, so an export that arrives afterwards is ignored:

- `HF_HOME` — the caches are 32 GB and will not fit beside the venv on a 60 GB
  local disk. A shell that forgets it re-downloads Mistral into a second cache.
- `INLET_OUTPUT_ROOT` — defaults to `<repo>/train_outputs`. Forget it in one
  shell and that shell's checkpoints or eval JSON land there instead of under
  `/root/outputs`, so §6 and §7 look in an empty directory. Use any local path
  you like, but use the **same** one in every shell.

**`setup_env.sh` can stop before installing anything**, on purpose. At the end
of step `2/8`, before the venv exists, it runs `inlet.test_upstream_api` — an
AST-only check that every symbol, call signature and flag name in this checkout
still matches the `text-to-lora` next to it. If it fails you will see

```
FAILED  1 problem(s). This checkout does not fit the text-to-lora next to it.
```

and **no venv is created**. That is the intended behaviour — the alternative is
the same crash 20 minutes and 13 GB later. Report the failure rather than
working around it.

**If you pipe the output, `$?` is the pipe's exit code, not the script's.**
`bash scripts/setup_env.sh | tee setup.log; echo $?` prints tee's status and
will say `0` on a failed setup. Use `${PIPESTATUS[0]}`, or just check that
`.venv/` exists and that the log ends with step `8/8`.

Then the model weights. **Three repos, not two:**

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --exclude "*.bin" "*.pth" "*.gguf" "*.msgpack" "*.h5"
huggingface-cli download Alibaba-NLP/gte-large-en-v1.5
huggingface-cli download Alibaba-NLP/new-impl        # <-- easy to miss
```

**The mechanism, since it is worth seeing once.** `common.sh` does

```bash
export HF_HOME="${HF_HOME:-$INLET_ROOT/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
```

so `HF_HUB_CACHE` is fixed at the moment `common.sh` runs, and nothing reads
`HF_HOME` again. Setting it afterwards changes nothing the downloader consults:
the 32 GB goes to `<repo>/.hf` while the directory you named stays at a few
megabytes, and `huggingface-cli` reports success the whole way. Nothing
announces it until `smoke.sh` refuses to start on a full disk, long after the
download. `INLET_OUTPUT_ROOT` is derived the same way and fails the same way,
just more quietly — an empty results directory rather than a full disk.

`gte-large-en-v1.5` loads with `trust_remote_code=True`, which fetches its code
from the separate repo `Alibaba-NLP/new-impl`. Downloading the model alone is
not enough to run offline; you get a `LocalEntryNotFoundError` from deep inside
`transformers` that does not name the missing repo.

Then the datasets:

```bash
tmux new -s warm
cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- see §3
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh
WORKERS=4 ./scripts/warm_cache_paced.sh
```

(`tmux new` does inherit the environment of the shell that starts it, so the
preamble is redundant *if* you never detach and never open a second shell.
They are here because that assumption is the one that breaks: a reattached or
second shell without them warms 12 GB into a different cache, and the failure
does not surface until a later step reports a dataset it just downloaded as
missing.)

**Use `warm_cache_paced.sh`, not `warm_cache.sh` directly.** The Hugging Face
API allows 500 calls per 300 seconds and says so in its response headers
(`RateLimit-Policy: "fixed window";"api";q=500;w=300`). `warm_datasets` does not
back off: on HTTP 429 it records the dataset as failed and moves on, so one
12-worker pass warms ~85 of 500 datasets and then "fails" the other 415 in a few
seconds. Rerunning immediately fails all of them again. **This looks like a hard
error and is not one.** The paced script loops `--only-failed` with the window
slept out between passes; expect ~40 minutes and 7-8 passes.

Expect `PACED_EXIT=0` and about 1,000 directories under
`third_party/text-to-lora/data/transformed_datasets` (each task is cached twice,
once as text and once tokenised, so ~1,000 for 500 tasks).

If it ends with `PACED_EXIT=2` (stalled), read the `FAIL` lines it printed
before assuming a network problem. Debris from an earlier interrupted run is
the usual cause: a half-written cache directory ("neither a `Dataset` nor a
`DatasetDict`" — it has the `.arrow` file and no `dataset_info.json`), or a
stale `.incomplete` blob that then fails with `PermissionError`. Deleting those
and rerunning warmed them in six seconds.

To retry a specific set by hand, **pass an absolute path** —
`--only-failed third_party/.../warm_failures.json` fails with
`FileNotFoundError` because the script changes directory first:

```bash
WORKERS=2 ./scripts/warm_cache.sh \
    --only-failed $PWD/third_party/text-to-lora/warm_failures.json
```

Note that `warm_cache.sh` only *writes* `warm_failures.json` when something
fails, so after a fully successful retry that file still lists the datasets
that just succeeded. Do not read it as current state.

### Pre-flight

```bash
./scripts/smoke.sh 1        # ~30 min on one A100. Do not skip.
```

It catches the class of bug this codebase actually has: runs that complete, show
a falling loss, and report wrong numbers. Most of the half-hour is the two
validations inside the 200-step training run — `smoke.sh` deliberately does not
lower `val_max_batches`, so this is the same validation the real run does.
Reference values, measured on 1x A100 on 2026-09-08:

```
test_accum        9.070e-08     (known-bad ordering 3.1e-01)
test_ddp_equiv    8.918e-08     (summed-instead-of-averaged 3.3e-01)
gate_m0           |delta| = 0.000e+00
peak GPU memory   14.58 GiB
step 4/4          got 65.61, expected 65.70 -> PASS  (arc_challenge zero-shot)
```

The last one is the whole eval path — engine, sequence assembly, scoring —
checked against a published number. If it passes, an eval result that later
looks wrong is not the harness.

**A vLLM `ERROR` after the result is written is teardown noise, not a failure:**

```
wrote .../arc_challenge__zero_prompt.json
ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
smoke OK -- ...
```

The score is already written by then. Read `SMOKE_EXIT` / `smoke OK`, not the
last line that says ERROR.

---

## 4. Datasets

**Training — 479 tasks.** From `configs/hyper_lora_decontam_lol_tasks.yaml`,
selected by `--n_train_ds=479` (already in the recipe, do not change it). These
are Super-Natural-Instructions tasks from the `Lots-of-LoRAs` collection. T2L
started from 500, held out 11 for validation, and **removed 10 for
contamination with the evaluation benchmarks**, leaving 479. Using the same 479
is what makes an Inlet number comparable to a T2L number.

**Validation during training — four splits, automatic, nothing to pass:**

| split | what it is |
|---|---|
| `val/seen` | 10 `lol_*` tasks that are in training |
| `val/unseen` | 11 held-out `lol_*` tasks (task035, 039, 202, 304, 362, 614, 701, 706, 710, 726, 1557) |
| `val/benchmark` | benchmark-derived |
| `val/generative` | gsm8k `train` split |

`val/generative` exists because upstream's three splits contain no long-form
generation at all, which is why the collapse was invisible for 147,500 steps.

**Final evaluation — the 10 benchmarks in T2L's Table 2**, in their order:

```
arc_challenge arc_easy boolq hellaswag openbookqa piqa winogrande gsm8k mbpp humaneval
```

---

## 5. Train

```bash
tmux new -s run1
cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- see §3
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh

./scripts/train.sh 2 --run_name=run1 \
    --desc_slots=8 --cond=cross \
    --model_select_split=val/unseen \
    --max_steps=16000 \
    --checkpoint_steps=500,1000,2000,4000,8000,16000 \
    --val_freq=1000 \
    --val_max_batches=15 \
  2>&1 | tee /root/outputs/run1.train.log
```

~10 hours on 2x A100.

**Keep the `tee`.** `train.sh` writes no log of its own, and every later step —
reading `prompt_norm` per checkpoint, deriving the eval scale, reporting the
startup lines — reads `run1.train.log`. Without it the only copy is tmux
scrollback, which a 16,000-step run overruns.

**Every override must be `--key=value`.** Upstream's parser is not argparse; it
builds its override dict as `arg.split("=")[1]`, so a bare `--freeze_head` or a
space-separated `--max_steps 16000` raises `IndexError: list index out of range`
in `configs.py` before training starts. Checking a flag against
`HfArgumentParser` proves nothing — that is a different parser from the one
`train_inlet.py` uses, and it accepts forms this one rejects.

**Booleans are stricter than they look.** The cast is `val in ["true", "True"]`
and everything else falls through to `False`, so `--freeze_head=1`,
`--freeze_head=yes` and `--freeze_head=TRUE` all mean **False** — the run trains
fine and is a duplicate of the control. Write `True` / `False` exactly. An
unknown flag name, by contrast, is loud: `ValueError: Argument provided not
found in dataclass`.

**Run it under tmux, not `nohup`.** `warm_cache.sh` spawns 12 workers and died
silently at 19/500 when the ssh session that started it closed, leaving no error
in the log.

### Check these five lines, then leave it alone

```
LR: 2.500e-05 x ... = ...          -- independent of GPU count by construction
NCCL collective timeout : 4:00:00  -- 0:10:00 means SIGABRT ~10 min in, no traceback
val/generative: gsm8k[train]
permanent checkpoints will be kept at steps: [500, 1000, 2000, 4000, 8000, 16000]
[neftune] alpha=5.0 ACTIVE on Embedding    -- minutes in, after step-0 validation
```

### Watch for

- **`CANARY:` warnings.** Free-running generation next to teacher forcing. Note
  that the canary fires even at step 200 on an essentially untrained prompt, so
  a fired canary is close to this model's baseline and is **not on its own**
  evidence that anything broke. Only the trend against step count means
  something.
- **`prompt_std_across_batch` warnings** — the generator is emitting nearly the
  same prompt for every description.
- **`prompt_norm`**, on the `[step N] train:` line every 100 steps, next to
  `varying_fraction` and `prompt_std_across_batch`. This is the number the whole
  run is about; expect it to start at 0.1543 and climb. (`|P|` in the tqdm bar is
  the same number, rounded to 2 dp.) Validation lines have the same shape, so
  `grep -a "\[step " run1.train.log` reads both.

If it hangs: `pgrep -f train_inlet`, then `kill -USR1 <pid>` for **every** rank.

---

## 6. Evaluate

Six checkpoints x 10 tasks x several scales is too much to run blindly. Two
stages. Open the eval shell the same way as the training one — the two lines
are not optional here either, and an eval shell that forgets them re-downloads
Mistral into `~/.cache` rather than failing:

```bash
cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- see §3
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh
```

### Stage A — find the scale, on three tasks (~2-3 h)

For each checkpoint, read its `prompt_norm` out of the training log and derive
the sweep range. **The optimum is near norm 0.11**, so:

```
scale ≈ 0.11 / prompt_norm
```

A checkpoint at `prompt_norm` 0.2259 wants ~0.49; one at 0.4056 wants ~0.27.
Sweep three values bracketing that estimate, e.g. `1.0,0.5,0.35`.

```bash
# read prompt_norm at, say, step 4000
grep -a "\[step 4000\] train:" /root/outputs/run1.train.log \
    | grep -ao "prompt_norm=[0-9.]*"

CKPT=/root/outputs/hyper_lora/run1/hypermod_inlet_step4000.pt
TASKS="gsm8k humaneval arc_challenge" \
EXTRA_EVAL_ARGS="--prompt-scales 1.0,0.5,0.35 --max-eval-descs 1 --skip-random-descs" \
    ./scripts/eval.sh $CKPT
```

**Use `eval.sh` to cover several tasks, never `eval_inlet --tasks`.** With a
checkpoint the prompts are generated from `--task` alone and the engine is built
once, so `--task gsm8k --tasks gsm8k humaneval` scores humaneval with *gsm8k's*
description and writes it to `humaneval__<ckpt>.json` — no error, no warning.
`eval.sh` re-invokes the module once per task, so each task gets its own
description. The raw form now exits with an explanation instead of running;
`--tasks` remains correct for `--zero-prompt` and `--synthetic-prompt`, whose
prompts are task-independent.

`--max-eval-descs 1 --skip-random-descs` makes this a **debug protocol**: one
description, no junk-description controls. Numbers from it must never share a
table with reported-protocol numbers, which average three descriptions and
include the controls.

Also get the zero-prompt reference once — it is the only number that says
whether a prompt helps or hurts at all, and it validates the eval path:

```bash
python -m inlet.eval_inlet --task gsm8k --tasks gsm8k humaneval \
    --zero-prompt --max-eval-descs 1 --skip-random-descs \
    --out-dir /root/outputs/eval_results_inlet
```

(`--tasks` is correct *here*: the zero prompt is the same empty tensor for every
task, so one engine build genuinely covers both.)

Expect gsm8k ~39.95 and humaneval ~39.63 (published zero-shot: 40.71 / 37.80).
**If these are far off, stop — the eval path is wrong and nothing downstream
means anything.**

### Stage B — full table at the chosen (checkpoint, scale) (~1.5 h)

```bash
EXTRA_EVAL_ARGS="--prompt-scales $BEST_SCALE --max-eval-descs 1 --skip-random-descs" \
    ./scripts/eval.sh $BEST_CKPT
```

`eval.sh` already defaults `TASKS` to the ten benchmarks in T2L's Table 2 order,
so there is nothing to list.

Then the **reported protocol** on the same checkpoint — three descriptions plus
the junk-description controls, which is what goes in the paper:

```bash
./scripts/eval.sh $BEST_CKPT
```

### Two eval traps

1. **Result filenames are built from the checkpoint's basename.** Two
   checkpoints both called `hypermod_inlet.pt` (e.g. from different runs) write
   the same `gsm8k__hypermod_inlet.json` and the last one wins. The step
   checkpoints are fine (`..._step4000.pt` differs), but do not evaluate two
   runs' final checkpoints into the same `--out-dir`. Each file records
   `protocol.checkpoint`, so you can tell after the fact.
2. **The eval re-runs per-task tokenisation for every scale.** boolq (3,270 long
   passages) costs ~13 minutes *per scale* and hellaswag (10,042) far more.
   Keep multi-scale sweeps to the three fast tasks; run the full 10 at a single
   scale.

---

## 7. Send back

- The five startup lines, verbatim.
- `train_summary.json` and the full training log.
- `prompt_norm` at each checkpoint step.
- Every JSON under `eval_results_inlet/` (they are small).
- Any `CANARY:` or `prompt_std_across_batch` warnings.
- The Stage A table: checkpoint x scale x {gsm8k, humaneval, arc_challenge},
  and the completion-length lines that `eval_inlet` prints next to each score —
  `completions median N words` is the diagnostic the whole run turns on.

## 8. The question this run answers

Plot **scaled score against step count**. `sweep_report` builds it from the
JSON `eval.sh` already wrote — it reads the step out of each checkpoint's own
config, so nothing has to be assembled by hand:

```bash
python -m inlet.sweep_report /root/outputs/eval_results_inlet --scale 0.5 --paper
python -m inlet.sweep_report /root/outputs/eval_results_inlet --scale 1   --paper
```

**`--scale` is required once any result carries `--prompt-scales`,** and each
scale is its own series. They are not averaged and must not be: the x1.0 and
x0.5 arms live in the same JSON under keys that both begin `eval_descs`, and
meaning them together produces a number that is nobody's measurement — on the
worked example it moved the peak from step 1,000 to step 500, which is the
answer this run exists to give. Without `--scale`, `sweep_report` lists the
scales it found and stops.

- Peaks at 2,000-4,000 and flattens → long training is unnecessary; report the
  short run and say so.
- Still climbing at 16,000 → the earlier "optimum before step 4,000" was an
  artefact of the norm growing, and the fix moves the optimum later.

Both outcomes are useful. Either way, `prompt_norm` against step count should
be a clean monotone curve, and the scaled score should be far flatter than the
unscaled one — that is the mechanism, stated as a prediction so it can fail.
