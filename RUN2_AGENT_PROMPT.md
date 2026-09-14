# Cluster agent task — Run 2 aligned training, two A100 80GB GPUs

Execute this task in the existing Inlet cluster checkout. The user has authorized preparing and running this experiment. Complete the audit and smoke checks yourself; do not ask the user to approve routine reversible work. Do not launch the 32k run with unresolved data/protocol defects.

## Objective and frozen recipe

Train **one fresh generator for 32,000 optimizer updates**, using the actual Run 1 `args.yaml` as the source of all unchanged settings. Do not resume the 16k weights. Same frozen `mistralai/Mistral-7B-Instruct-v0.2`, 8 description slots, cross attention, 32 virtual tokens, 479 training tasks, global batch 64. Exactly two GPUs, DDP, no tensor parallelism. At Run 1's microbatch 8/rank this means accumulation 4.

The main training change is: generator still sees task descriptions; LLM text sees the problem without the dataset's `task_def` paragraph. Apply the same removal to seen/unseen validation. Preserve the target answer, masking, descriptions, data mixture, seed, architecture, optimizer settings and LR recipe. The 32k schedule inherits Run 1's warmup fraction; it is a new full schedule, not the old 16k schedule resumed. Training prompt scale remains 1.0. No added ICL, no diversity/contrastive penalty, no higher head LR, no length reward/KL, no new generative training data, no per-task scaling.

This is a candidate improved recipe, not a controlled proof against the 16k run: training budget and schedule also differ. A same-budget original-recipe control is a later experiment.

## 1. Preserve the existing checkout and results

Work in `/data/users/yuxin1211/inlet` if it is the active checkout; verify rather than assume. Inspect branch, remote and local changes. Pull GitHub `main` while preserving the existing **MBPP serialization patch**, cached data, outputs and failed-attempt records. Do not use `reset --hard`, `clean`, delete caches, or replace the working upstream checkout. Save local diffs before any merge. If a conflict occurs, resolve it preserving both changes and record the diff.

Read `docs/RUN2_ALIGNED_TRAINING.md`. Record Inlet commit, local diff, upstream source hashes, environment versions, model/tokenizer revision and Run 1 args/checkpoint paths. The upstream directory need not be a Git checkout: hash its source. Do not claim an unknown upstream commit.

Reuse the working Run 1 Python/CUDA environment and large-volume HF cache. Do not upgrade torch/transformers/vLLM. Keep `INLET_NO_FLASH_ATTN=1`. Reuse the known working cache/environment values; `HF_DATASETS_OFFLINE=0` may be necessary for this checkout. Preserve the Run 1 eval workaround `INLET_BASELINE_DIR=/nonexistent/baseline_prompt_tuning`; never inherit the sibling Gemma configuration. Training does not require vLLM.

## 2. Freeze configs from the real Run 1 args

Locate the `args.yaml` beside `hypermod_inlet_step16000.pt`; verify it is Run 1. Activate the existing environment and export `T2L_ROOT`, `HF_HOME`, `INLET_OUTPUT_ROOT` and `CUDA_VISIBLE_DEVICES=0,1` with the actual paths. Then run, substituting real paths:

```bash
python -m inlet.run2_setup \
  --run1-args /data/users/yuxin1211/inlet/train_outputs/hyper_lora/run1/args.yaml \
  --t2l-root /data/users/yuxin1211/text-to-lora \
  --output /data/users/yuxin1211/inlet/train_outputs/run2_aligned_plan_v1 \
  --run-name run2_aligned_32k_v1
```

Inspect `provenance.json` changes. Do not substitute a generic config if the source is missing or rejected. Resolve the specific source mismatch first and document it. Source/config changes require a new plan, not editing hashes to bypass checks.

## 3. Input audit, then an eight-update smoke run

```bash
bash scripts/run2_aligned_2gpu.sh /data/users/yuxin1211/inlet/train_outputs/run2_aligned_plan_v1 audit
```

Audit outputs are under `<INLET_OUTPUT_ROOT>/<exp_setup>/run2_aligned_32k_v1_audit/`. Review `metadata_audit.json`, `input_audit.jsonl`, `label_coverage.json`, `input_protocol.json` and `audit_hashes.json`.

- Check all training/validation task description sets actually used, alongside removed definitions and two example rows/task. Produce `description_sufficiency_review.md` and a per-task review table. Look specifically for label mappings (yes/no vs entailment/contradiction, etc.), required output format, task-specific rules, and multi-answer formatting. Classify sufficient / ambiguous / missing-rule with concrete evidence. Merely nonempty strings do not pass semantic review.
- The upstream preprocessing sometimes adds a multi-answer instruction based on the output list. Do not move answer-derived information into the generator at training or inference. If removing that instruction leaves an ambiguous supervision convention, report the actual affected tasks and propose a task-level convention supported by the task definition; do not silently rewrite the recipe or drop tasks.
- Confirm no task definition remains in the LLM template, question and targets are intact, descriptions are still passed to the encoder, target tokens remain after truncation, and no target benchmark is used for validation/model selection. Zero supervised-token rows are a blocker, not grounds for silent filtering.
- This audit samples rows; do not call it an exhaustive example-level semantic proof. The token-label coverage check does cover all loaded rows.

If there are unresolved missing-rule/truncation defects, stop before 32k, preserve outputs and give the user concrete findings and a recommended resolution. Otherwise:

```bash
bash scripts/run2_aligned_2gpu.sh /data/users/yuxin1211/inlet/train_outputs/run2_aligned_plan_v1 smoke
```

Verify two ranks, global batch 64, finite loss/gradients, frozen backbone, model zero-prompt gate, 8 optimizer updates, checkpoint saving and validation completion. Inspect step 0/8 `generation_monitor.jsonl` and `generation_monitor_manifest.json`: raw output, target length, actual generated token count and EOS/length-limit cause. No forced minimum output length. Base and Inlet must see identical question text. The untrained zero-initialized head need not distinguish descriptions at step 0; do not mistake that for a bug.

The monitor deterministically chooses short/long examples from independent `val/unseen`, at most 8 different tasks; selection only sees the first 64 rows/task. Record whether long targets/code are actually represented. If code or genuinely long targets are absent, explicitly record the coverage limitation; do not call the monitor evidence that MBPP or long-form generation is fixed, and do not add target test questions to compensate. This limitation alone does not invalidate the alignment experiment.

Write `<plan>/readiness.json` only after checking the evidence:

```json
{
  "plan_sha256": "SHA256 of provenance.json",
  "descriptions_sufficient": true,
  "rendered_inputs_checked": true,
  "label_coverage_checked": true,
  "smoke_passed": true,
  "monitor_coverage_reviewed": true,
  "blocking_findings": [],
  "evidence": {
    "audit_directory": "actual path",
    "audit_hashes": "actual path and hash",
    "semantic_review": "actual path",
    "smoke_log": "actual path",
    "monitor_coverage_limitations": "actual findings"
  }
}
```

These fields record your completed checks, not permission requests. Do not fill them blindly.

## 4. Run 32k and preserve progress

Launch using the site's allocation/scheduler or a durable session that retains both allocated GPUs; capture stdout/stderr and exit code:

```bash
bash scripts/run2_aligned_2gpu.sh /data/users/yuxin1211/inlet/train_outputs/run2_aligned_plan_v1 train
```

The launcher calls torchrun directly from the frozen YAML; **do not route through `scripts/train.sh`**, which applies recipe defaults. The eight-update smoke is separate and is never resumed into training. Keep 8k/16k/24k/32k checkpoints and best validation checkpoints. Monitor every 4k, plus step 0 and final. Do not stop solely because early descriptions produce similar vectors. Investigate nonfinite values, OOM, rank hangs, missing targets, or malformed input. Preserve failed attempts. There is no exact optimizer-state resume implemented by this change: do not claim a restart from generator weights is an exact continuation. A restart needs a distinct output name/plan and explicit provenance.

Estimate runtime from measured update throughput after warmup; do not promise a fixed number of hours. Log allocated GPU model/memory, actual effective batch, observed training time, validation overhead, learning rates and prompt norms. Do not perform an unbounded scale/seed/architecture sweep.

## 5. Return a reviewable result bundle

Include `REPORT.md`, frozen plan/configs/provenance/readiness, semantic audit, training and smoke logs, checkpoint SHA256s, checkpoint selection records, validation-loss curves, prompt norm/variation curves, the monitor manifest/raw JSONL, environment versions and local diffs. State clearly whether 32k completed and which checks/limitations remain.

**Do not automatically select checkpoint/scale on the ten test benchmarks.** The main candidate is the terminal 32k checkpoint; the best `val/unseen` checkpoint is a separately labelled secondary candidate. The next evaluation plan will freeze a common deployment rule, tune scale only on independent held-out tasks, and compare no-ICL Base / text description / Inlet / T2L under the same harness. Existing target-benchmark exploration must be disclosed. This assignment ends with the trained artifacts and diagnostics; do not report unmeasured benchmark improvements.
