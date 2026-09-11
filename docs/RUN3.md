# Run 3 — full step budget, faster head

**One machine, one command.** Start it and leave; it decides for itself whether
to commit the 4.6 days.

```bash
./scripts/head_lr_probe.sh
```

It probes `--head_lr_mult=20` and `60` to step 4,000 each (~3 h apiece), then
either launches the full 147,500-step run at whichever won, or stops and tells
you the premise was wrong. Training only — no benchmark eval during any of it.

Setup, shell preamble and traps: `docs/RUN1.md` §2 and §3. Nothing changes there.

| | `--head_lr_mult` | when |
|---|---|---|
| Run 1 (already done) | 1 | — |
| probe | 20, then 60 | hours 0-6 |
| the real run | whichever won | hours 6-116 |

Everything else is identical across all of them, so this is a clean
learning-rate sweep on the description path — the diagnosed bottleneck (§1) and
the one number this plan rests on. `RESULTS.md` recommends 10-30, but the
per-task prompt-tuning baseline needed **3e-3, a 120x multiple**, so 20 may be
too conservative and 60 is the hedge. Three points on that axis is also the
ablation figure the paper needs.

**Why two probes and not one.** "mult=20 did not separate" is either *the
diagnosis is wrong* or *20 is too small*, and those want opposite next steps. One
value cannot tell them apart; two can. The second probe costs three hours
against a 4.6-day commitment.

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

`head_lr_probe.sh` does all of this. The commands are here so you can read what
it will run, and so you can drive it by hand if you prefer.

Each probe starts a **real** `--max_steps=147500` run and is killed once
`hypermod_inlet_step4000.pt` is on disk. A short `--max_steps=4000` job would
measure something else entirely: warmup is 20% of `max_steps`, so step 4,000 is
at lr factor 0.99 in a short run and 0.136 in the real one. Nothing is wasted —
the first 4,000 steps are the same steps either way.

```bash
tmux new -s run3
cd /root/inlet
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache        # both BEFORE common.sh -- RUN1.md §3
export INLET_OUTPUT_ROOT=/root/outputs
source scripts/common.sh

./scripts/head_lr_probe.sh
```

The run it commits to is:

```bash
./scripts/train.sh 2 --run_name=run3_m<MULT> \
    --desc_slots=8 --cond=cross \
    --model_select_split=val/unseen \
    --head_lr_mult=<MULT> \
    --max_steps=147500 \
    --checkpoint_steps=1000,2000,4000,8000,16000,24000,32000,48000,64000,96000,128000,147500 \
    --val_freq=4000 \
    --val_max_batches=15
```

**~109 hours (4.6 days)** on 2×A100 at Run 1's measured 2.67 s/step.

Twelve step checkpoints plus four `best_val_*` plus the final — about 2 GB. They
are deliberately dense: the scale sweep needs the whole curve, and re-training to
recover a missing checkpoint costs days.

Startup: check the same five lines as RUN1.md §5, plus one new one:

```
optimizer: base lr=2.500e-05  head lr=5.000e-04 (head_lr_mult=20.0)
```

`5.000e-04` at mult=20, `1.500e-03` at mult=60. **If the line is missing or says
`head_lr_mult=1.0`, stop — the flag did not land and the run is a duplicate of
Run 1.**

The mult=60 head learning rate is high enough to diverge. If `sft_loss` goes to
NaN or climbs for more than a few hundred steps during that probe, that probe
simply loses and the script moves on — it is a result about the usable range,
not a failure of the plan. Report it.

---

## 3. Kill it early if the premise is wrong

Do not wait 4.6 days to find out. Run `probe_prompt` on the first checkpoints as
they appear:

```bash
for S in 1000 2000 4000; do
  python -m inlet.probe_prompt \
      --checkpoint $INLET_OUTPUT_ROOT/hyper_lora/run3_m<MULT>/hypermod_inlet_step$S.pt \
      --task arc_challenge --out /root/outputs/probe_step$S.json
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

`head_lr_probe.sh` applies this automatically and **refuses to start the long
run** if neither probe clears the bar (exit 3). That refusal is a result, not a
failure: it says the amortization gap is not an optimization speed problem, and
no amount of further training at these settings fixes it. Report it as such.

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
  grep -a "\[step $S\] train:" /root/outputs/run3_m*.train.log \
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
- **Do not add a third changed variable.** The only thing that varies between
  the two probes is `--head_lr_mult`. `--contrastive`, `--l2_reg_prompt`,
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
