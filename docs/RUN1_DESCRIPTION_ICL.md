# Run 1 follow-up: description usefulness and ICL complementarity

This is the complete assignment for the cluster agent. Evaluation only. Do not
train, start RUN3, run head_lr_probe.sh, change the backbone, search scales, or
select another checkpoint. Older RUN1/RUN2/RUN3 documents are background, not
additional execution assignments. Read this entire document before starting.

## Purpose and evidence already available

The user's RUN1_EVIDENCE.md (2026-09-10) establishes that scaling often recovers
long generation, but the Run 1 measurements used one real description and no
junk control. At step 16k, ARC-C scored 72.95 at scale 1 and 70.39 at .5;
GSM8K scored 21.99 and 40.11; MBPP 2.38 and 39.95. Those are historical exploratory
observations, not targets to force reproduction of. The 54.63 headline column
and BoolQ 85.51 are not a complete table measured on this 16k checkpoint.
Real/junk vector cosine separation reached .2117 at 16k; that does not establish
an accuracy benefit. The 1k scale experiment cannot be substituted for 16k.
Generation varied with engine batch composition (HumanEval by 4.27 points even
at scale .5). Therefore freeze arm order and use fresh-process repeats.

Two questions only:
1. At fixed checkpoint and scale, does the correct description improve accuracy
   relative to wrong real descriptions and junk? Is this an absolute benefit
   over the no-prompt model, or merely a worse control?
2. Does adding the SAME ICL examples improve Inlet beyond ICL alone, and close
   the gap to T2L evaluated with the SAME examples?

This is a four-task exploratory experiment, NOT Avg10, a new SOTA claim, an
unbiased test after model selection, or proof that descriptions carry semantics.
Do not combine the best scales per task or choose results based on score.

## Fixed design

- Frozen Mistral-7B-Instruct-v0.2, existing Run1 checkpoint with
  config curstep=16000, desc_slots=8, cond=cross; normally
  hypermod_inlet_step16000.pt. Do not use an old model with a similar name.
- Tasks: arc_challenge, winogrande, gsm8k, mbpp. Full evaluation split per task;
  do not subsample scored questions.
- Scales: 0.5 and 1.0, each in a SEPARATE process/engine build. Do not add .35,
  .7, 1.3 or scale search. The same scales apply to all tasks.
- No ICL: 3 correct descriptions + 3 wrong real descriptions + 3 junk strings,
  in that fixed order. Wrong donors fixed before any results:
  ARC-C <- OpenBookQA, WinoGrande <- BoolQ, GSM8K <- MBPP, MBPP <- GSM8K.
  ARC/OBQA tests a relatively similar task; math/code swaps also change output
  format. Report donor names; do not overinterpret this as a pure semantic test.
- ICL: the 3 correct descriptions only; both scales. Control descriptions are
  intentionally only measured without ICL in this bounded experiment.
- Base without/with ICL, using the Inlet m=0 embedding path (NOT 32 zero vectors).
- Released Mistral T2L SFT L without/with ICL, same 3 correct descriptions as
  Inlet. Never substitute supervised per-benchmark oracle LoRAs.
- ARC-C and WinoGrande: one pass. GSM8K and MBPP: TWO total fresh-process passes
  for EVERY configuration, with identical seed, request order and arm layout.
  These are inference reproducibility repeats, not training seeds or independent
  test datasets. Identical repeats are possible under greedy decoding.
- HumanEval is omitted from this ICL experiment because upstream examples are
  empty. Do not invent demonstrations or call empty-prefix evaluation 3-shot.

Full queue: 72 process jobs, 192 method/description/scale evaluation arms. This
is not a 72-training-run job. Runtime depends on engine startup and evaluator;
measure first completed tasks and give an ETA, not a promised one-day finish.

## Setup and preflight

1. Use the existing Run1 environment, model/data caches, and checkpoint. Inspect
   working directories and GPU allocation. Do not rebuild or upgrade the venv
   merely because this branch is newer. Read RUN1.md sections 2-3 only if you
   need its environment context; do not execute its training instructions.
2. Fetch branch codex/run1-description-icl into an isolated worktree/clone if the
   existing checkout is dirty. Do not reset or overwrite local experiment edits.
   Record Inlet commit, upstream T2L commit AND local diff, GPU model, Python,
   torch, transformers, vLLM, fishfarm/evalplus versions, CONFIG path and hash.
   If GitHub egress is blocked, report that exact blocker; do not silently run
   old code. A git bundle of this branch can be transferred separately.
3. Set existing HF_HOME/HF_HUB_CACHE, T2L_ROOT and INLET_OUTPUT_ROOT BEFORE
   sourcing scripts/common.sh. Activate the existing compatible venv. Use
   absolute paths below. Ensure this branch is on PYTHONPATH (common.sh does
   this); do not accidentally import an older installed inlet package.
4. Locate the 16k checkpoint; inspect torch.load(..., map_location='cpu',
   weights_only=False)['config']; record its SHA256, curstep, model_dir,
   desc_slots, cond, n_virtual_tokens if present. The queue checks 16k/8/cross.
   Keep all source checkpoints intact. No intermediate checkpoint is required.
5. Locate the official released Mistral T2L SFT L checkpoint and sidecars
   (args.yaml, adapter_config.json, hypermod.pt). Do not demand old Inlet models.
   If absent, locate/download the official SakanaAI/text-to-lora release using
   its documented repository instructions and record provenance. If access or
   storage blocks this, proceed with explicit partial Inlet/Base evaluation
   and mark T2L missing; do not replace it with historical scores.
6. Run CPU checks using the CLUSTER venv Python (3.10+):

```bash
python -m unittest inlet.test_eval_protocol
python -m compileall -q inlet
python -m inlet.test_upstream_api
```

Local development passed dependency-light tests and compilation, not GPU end to
end. Cluster preflight is mandatory. If a dependency API differs, diagnose it;
a narrowly scoped compatibility fix is allowed, but record the patch and rerun
checks. Do not change dataset splits, generation settings or scoring to make a
run succeed. Stop affected jobs if a safe compatibility fix is unavailable.

## Freeze descriptions and generate T2L adapters

Set these variables to the actual absolute paths; placeholders are not defaults:

```bash
RUN1_CKPT=/absolute/path/hypermod_inlet_step16000.pt
T2L_CKPT=/absolute/path/released_mistral_sft/hypermod.pt
DIAG_ROOT=/absolute/path/run1_description_icl_v1
mkdir -p "$DIAG_ROOT"
python -m inlet.prepare_run1_diagnostic \
  --out "$DIAG_ROOT/inputs" \
  --t2l-checkpoint "$T2L_CKPT"
```

Pass --config /actual/training_description_config.yaml if the default upstream
configs/hyper_lora_decontam_lol_tasks.yaml differs from the Run1 setup. The
preparation command freezes 3+3+3 descriptions per task and a manifest. It
performs only forward adapter generation for 12 real descriptions. It sets
hypermod.peft_config.use_rslora=False IN MEMORY before saving the newly generated
adapters (the known released Mistral metadata issue). It never edits the source
checkpoint. Verify saved adapters have correct r, alpha, target modules and
base model, and that generated file provenance matches each description.

If T2L is genuinely unavailable, omit --t2l-checkpoint and later explicitly pass
--allow-missing-t2l. Report the partial scope. A later full run needs a new
output root/plan; do not change inputs inside an existing run.

## ICL correctness gate

The new --use-icl flag is forwarded to upstream. Both Inlet and T2L paths call
inlet.eval_protocol.audit_icl at the actual evaluator sample boundary:
- All samples must contain the intended nonempty upstream ICL prefix.
- GSM8K's historical dropped-prefix bug is repaired there, for ALL methods.
  An already corrected checkout is detected and is not double-prefixed.
- A missing prefix on other tasks or empty HumanEval examples fails closed.
- The actual ICL text, SHA256, repair count and first sample are recorded.
- No-ICL remains the original path. Do not compare corrected GSM8K ICL to the
  old published/Run1 nominal ICL column as if the protocols were identical.

Before the full queue, run one no-prompt ARC-C smoke evaluation in a separate
scratch output directory, plus a no-prompt GSM8K ICL evaluation. Inspect their
full rendered raw inputs: correct chat template, actual demonstrations,
question, assistant prefill; no ground-truth query answer accidentally inserted.
Count actual demonstrations from saved ICL text, don't assume every task has 3.
ARC-C no-prompt should be close to 65.70. A >1 point discrepancy is a diagnosis
trigger, not a tolerance to silently accept. Generative reference differences
are not fixed by changing decoding settings. Reuse a completed smoke result
only if its full protocol matches and provenance is preserved.

```bash
python -m inlet.eval_inlet --task arc_challenge --zero-prompt --save-raw \
  --out-dir "$DIAG_ROOT/smoke_arc"
python -m inlet.eval_inlet --task gsm8k --zero-prompt --use-icl --save-raw \
  --out-dir "$DIAG_ROOT/smoke_gsm_icl"
```

The raw JSONL capture contains full rendered inputs and UNSANITIZED model
outputs before EvalPlus extraction. It does not currently capture vLLM finish
reasons. Do not invent EOS/truncation explanations from length alone.

## Execute the queue

First print/freeze the plan (no GPU evaluation), inspect it, then execute:

```bash
python -m inlet.run1_diagnostic --checkpoint "$RUN1_CKPT" \
  --manifest "$DIAG_ROOT/inputs/manifest.json" --out "$DIAG_ROOT/results"
python -m inlet.run1_diagnostic --checkpoint "$RUN1_CKPT" \
  --manifest "$DIAG_ROOT/inputs/manifest.json" --out "$DIAG_ROOT/results" --execute
```

Use the cluster's scheduler or a persistent session so the queue outlives the
agent turn. On the user's TWO A100s, use the launcher below: one independent evaluation
process per GPU, no tensor parallelism or DDP. Do not launch unrelated jobs on
these GPUs. The queue launches fresh
processes, writes per-job process.log, and resumes only from done.json markers.
--only-task TASK allows running one task from the same frozen plan. Preserve
identical hardware/environment for inference repeats.

### Two A100s (preferred on this cluster)

After preparing adapters and passing the smoke checks, use this INSTEAD OF the
single-GPU --execute command above:

```bash
bash scripts/run1_diagnostic_2gpu.sh --checkpoint "$RUN1_CKPT" \
  --manifest "$DIAG_ROOT/inputs/manifest.json" --out "$DIAG_ROOT/results"
```

Activate the same venv as before; PYBIN must resolve to that venv Python. The
launcher respects the two device IDs/UUIDs in CUDA_VISIBLE_DEVICES set by your
scheduler; when it is unset it uses 0,1. Verify two GPUs are actually allocated.
It first freezes the shared plan serially, then assigns:
- first GPU: ARC-Challenge, then GSM8K, including both GSM8K repeats;
- second GPU: WinoGrande, then MBPP, including both MBPP repeats.

Both GPUs work concurrently, but each GPU runs one process at a time. All
methods, scales and repeats for a task remain on the same GPU. A failure stops
that worker; the other worker may finish its independent tasks. The launcher
returns failure if either worker fails. Rerun the same launcher to resume after
inspecting partial artifacts. Use the scheduler to cancel the entire allocation
if needed, including child evaluator processes. Do not run the single-GPU queue
against the same output root while the two-GPU launcher is active.

The total work is unchanged (72 processes, 192 arms). Wall time may decrease,
but startup, CPU-based code evaluation and uneven task durations prevent a
promise of exactly 2x speedup. If the older commit already created identity.json,
use a new results output root for this code revision; do not edit its commit hash.

A failed process stops the queue. Inspect logs and partial JSON/raw files before
retry. Move the WHOLE failed job directory into a sibling failed_attempts area
outside the results root, then rerun the queue; never append to partial raw files.
A vLLM teardown error after writing metrics is not automatically treated as
success. Verify raw completeness and process state; either document a carefully
validated salvage separately or rerun the failed job. Do not manufacture a
successful done.json solely because the metric file exists.

After the first ARC-C results, compare raw rendered-input hashes across
Base/Inlet/T2L for the SAME ICL condition. Vector vs LoRA interventions differ,
but the underlying formatted text must match, including assistant prefill.
Repeat this check for every task before interpreting differences. Within an
Inlet task/scale/no-ICL job, ensure the same questions are scored under all 9
arms (raw input hashes must match; prompts themselves intentionally differ).

Do not expand to the other six benchmarks, other scales or training based on
these exploratory numbers. Finish this fixed queue even if initial scores are
encouraging or disappointing, unless a validity/infrastructure failure blocks it.

## Report and hand back

```bash
python -m inlet.report_run1_diagnostic "$DIAG_ROOT/results"
```

Do not use the old sweep_report/desc_control across mixed ICL directories;
they were designed for single-protocol outputs. The new reporter groups by
task, method, ICL, scale, description group and repetition. It averages the
three descriptions WITHIN each repeat first, then reports repeat mean/range.
Two repeats do not establish statistical significance. Do not count their
examples as extra independent test questions.

Write "$DIAG_ROOT/REPORT.md" with:
1. Provenance/environment and scope, completed and missing jobs; all local fixes.
2. For each task and each fixed scale WITHOUT ICL: matched, mismatched, junk,
   matched-minus-mismatched, matched-minus-junk, matched-minus-base. Include both
   absolute scores and differences. A gap caused only by worsening controls is
   not evidence that useful adaptation improved.
3. Per task: Base, Base+ICL, T2L, T2L+ICL, Inlet(.5), Inlet(1),
   Inlet(.5)+ICL, Inlet(1)+ICL. Three-description means and separate repeat
   values/ranges; do not select a winning scale for a combined table.
4. ICL marginal gain for each method and Inlet-minus-T2L at MATCHED ICL status.
   State whether adding Inlet improves on ICL alone.
5. GSM8K and MBPP median completion length, empty/short output rate, full raw
   examples. For MBPP include representative malformed output, extraction
   failure and valid-but-test-failing code if present; distinguish categories
   using observed evidence. Never assume MBPP 2.38 was only an extraction bug.
6. Input-hash/ICL-prefix checks, actual demonstration counts, correction counts,
   repeat variability and any sample coverage mismatches. If stable per-question
   correctness is available in sample_details, a paired bootstrap may be added,
   grouping repeated observations of the same question rather than multiplying n.
7. A bounded recommendation: ICL-compatible training, information-channel
   alignment, or generation-preserving training, based on the measured results.
   No claim of full Avg10 victory or a definitive semantic mechanism.

Return REPORT.md, summary.json, scores.csv, frozen plan+input manifest, raw
JSONL files, metrics, and logs (or a portable archive with relative paths). Do
not send model weights unless explicitly requested. This agent's job ends with
this report; do not automatically start a training run.
