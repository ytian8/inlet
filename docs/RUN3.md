# Run 3 — full step budget, faster head

**Two training runs, one per 2xA100 machine.** Training only — do not evaluate
benchmarks during either. The scale sweep and the benchmark table come
afterwards, from the checkpoints.

Setup, shell preamble and traps: `docs/RUN1.md` §2 and §3. Nothing changes there.

| | `--head_lr_mult` | machine |
|---|---|---|
| Run 1 (already done) | 1 | — |
| **run3a** | **20** | machine 1 |
| **run3b** | **60** | machine 2 |

Everything else is identical across all three, so this is a clean learning-rate
sweep on the description path — which is the diagnosed bottleneck (§1) and the
one number this whole plan rests on. `RESULTS.md` recommends 10-30, but the
per-task prompt-tuning baseline needed **3e-3, a 120x multiple**, so 20 may be
too conservative and 60 is the hedge. Three points on that axis is also the
ablation figure the paper needs.

Start both now. §3's probe check kills both within three hours if the premise is
wrong.

---

## 1. What this changes, and why

Exactly **two** things differ from Run 1. Run 1 is the control.

### `--head_lr_mult` (20 and 60)

`RESULTS.md` diagnosed this before Run 1 was designed and it has never been
tested: the head that reads the description trains at T2L's `lr=2.5e-5`, while
per-task prompt tuning needed **3e-3** for vectors of the same shape. The note
there reads *"the head does not move … which is an optimization problem, not an
information one"*, and it ranks `--head_lr_mult 10–30` as the fix with
cross-attention as only a precondition. Run 1 shipped the precondition.

Run 1's own curves fit that diagnosis. `cos(real, junk)` separation, from
`probe_prompt`:

| step | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---|---|---|---|---|---|
| separation | 0.0000 | 0.0002 | 0.0061 | 0.0516 | 0.1584 | 0.2117 |

The head is zero-initialised, so the whole run is spent climbing out of that
initialisation — and it had not finished at 16,000.

The flag already exists (`train_inlet.py:177`, separate optimizer param group at
:880). `base` keeps `lr`; everything else — the encoder MLP, the cross-attention,
the output head, i.e. the only path the description travels — gets `lr × mult`.

### `--max_steps=147500`

The recipe's native budget, inherited from T2L: `--epochs=20000` × 59
batches/epoch ÷ `--global_tasks_per_step=64` = **147,500**. Run 1 ran 16,000,
which is **11%** of it.

This matters more than the raw step count, because **warmup and the cosine decay
are both scaled to `max_steps`**:

| | `max_steps=16000` | `max_steps=147500` |
|---|---|---|
| warmup | 3,200 | 29,500 |
| lr factor at step 4,000 | 0.990 (peak) | 0.136 |
| lr factor at step 8,000 | 0.691 | 0.271 |
| lr factor at step 16,000 | **0.000** | 0.542 |

So Run 1's separation slowing between 8,000 and 16,000 is **the schedule
annealing to zero, not the model saturating** — and the whole of Run 1 fits
inside this recipe's warmup period.

Nothing else changes. In particular: **no prompt normalisation and no
`--l2_reg_prompt`.** Magnitude stays free and is dealt with at inference by the
scale sweep, because short-output and long-output tasks want opposite magnitudes
and a single fixed norm loses points on the short-output side.

---

## 2. Train

Run this on machine 1 with `MULT=20 NAME=run3a`, and on machine 2 with
`MULT=60 NAME=run3b`.

```bash
MULT=20; NAME=run3a          # machine 2: MULT=60; NAME=run3b

tmux new -s $NAME
cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- RUN1.md §3
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh

./scripts/train.sh 2 --run_name=$NAME \
    --desc_slots=8 --cond=cross \
    --model_select_split=val/unseen \
    --head_lr_mult=$MULT \
    --max_steps=147500 \
    --checkpoint_steps=1000,2000,4000,8000,16000,24000,32000,48000,64000,96000,128000,147500 \
    --val_freq=4000 \
    --val_max_batches=15 \
  2>&1 | tee /root/outputs/$NAME.train.log
```

**~109 hours (4.6 days)** on 2×A100 at Run 1's measured 2.67 s/step.

Twelve step checkpoints plus four `best_val_*` plus the final — about 2 GB. They
are deliberately dense: the scale sweep needs the whole curve, and re-training to
recover a missing checkpoint costs days.

Startup: check the same five lines as RUN1.md §5, plus one new one:

```
optimizer: base lr=2.500e-05  head lr=5.000e-04 (head_lr_mult=20.0)
```

`5.000e-04` for run3a, `1.500e-03` for run3b. **If the line is missing or says
`head_lr_mult=1.0`, stop — the flag did not land and the run is a duplicate of
Run 1.**

run3b's head learning rate is high enough to diverge. If `sft_loss` goes to NaN
or climbs for more than a few hundred steps, report it and stop that run — it is
a result about the usable range, not a failure of the plan.

---

## 3. Kill it early if the premise is wrong

Do not wait 4.6 days to find out. Run `probe_prompt` on the first checkpoints as
they appear:

```bash
for S in 1000 2000 4000; do
  python -m inlet.probe_prompt \
      --checkpoint $INLET_OUTPUT_ROOT/hyper_lora/$NAME/hypermod_inlet_step$S.pt \
      --task arc_challenge --out /root/outputs/${NAME}_probe_step$S.json
done
```

Compare `cos(real, junk)` separation against Run 1 at the same step:

| step | Run 1 | run3 must beat |
|---|---|---|
| 2,000 | 0.0061 | **0.05** |
| 4,000 | 0.0516 | **0.15** |

This is a **harder** test than it looks: at `max_steps=147500` the learning rate
at step 2,000 is a factor 0.068, against Run 1's 0.625 at the same step. run3
has to separate faster on a tenth of the learning rate. If it cannot, the
optimization diagnosis is wrong, and four more days will not fix it.

**If the step-4,000 probe is at or below Run 1's 0.0516 on BOTH runs, stop both
and report that.** It is a real result: it says the amortization gap is not an
optimization speed problem, and no amount of further training fixes it.

If only one of the two clears the bar, keep that one and restart the other at a
multiple between the two.

---

## 4. Watch `prompt_norm`

It is on the `[step N] train:` line every 100 steps.

Run 1 reached 0.2426 and plateaued after step 11,000. The old 147,500-step run
(collapsed architecture) finished at **0.4056**. A 20× head learning rate over a
full budget can push it well past either.

That is expected and is not by itself a reason to stop — the scale sweep exists
to deal with it, and the dense checkpoints mean an over-grown late model does not
cost the run. Record the value at each checkpoint step:

```bash
for S in 1000 2000 4000 8000 16000 24000 32000 48000 64000 96000 128000 147500; do
  printf "%7s  " "$S"
  grep -a "\[step $S\] train:" /root/outputs/$NAME.train.log \
    | tail -1 | grep -ao "prompt_norm=[0-9.]*" || echo "(none)"
done
```

Report it. It sets the scale sweep range: the generative optimum is near an
effective norm of **0.12–0.14**, so `scale ≈ 0.13 / prompt_norm` for each
checkpoint.

---

## 5. Do not

- **Do not evaluate benchmarks during training.** `val_freq=4000` already costs
  enough; benchmark eval belongs in the sweep afterwards, on chosen checkpoints.
- **Do not add a third changed variable.** The only difference between run3a and
  run3b is `--head_lr_mult`. `--contrastive`, `--l2_reg_prompt`,
  `--prompt_diversity` and `--desc_slots=32` are candidates for the round after
  this one; putting any of them in here makes the result unattributable.
  `--desc_slots=32` in particular is downstream of this run — `RESULTS.md`:
  *"widening the input to a head that is barely learning does not help"* — and it
  also needs the description cache rebuilt at 32x1024, which the trainer refuses
  to do silently.
- **Do not change `--n_train_ds`, the LR, or the global batch.** They are what
  make an Inlet number comparable to a T2L number.
- **Do not delete intermediate checkpoints** to save disk. 2 GB is cheaper than
  a rerun.

---

## 6. Report back

- The startup lines, including the `optimizer: base lr=… head lr=…` line.
- The step-2,000 and step-4,000 probe separations **as soon as they exist** —
  before the run finishes. This is the go/no-go.
- `prompt_norm` at every checkpoint step (§4).
- `probe_prompt` separation at every checkpoint step, so the Run 1 table above
  can be extended.
- `train_summary.json` and the full log.
- The twelve checkpoints, or tell us where they are.

The scale sweep and the benchmark table are a separate job and will be specified
once the `prompt_norm` curve is known.
