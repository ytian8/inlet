# Run 2: align the information channel, 32k fresh training

Status: code prepared; no Run 2 GPU result is claimed. See [the complete cluster-agent task](../RUN2_AGENT_PROMPT.md).

| Component | Run 2 |
|---|---|
| Initialization | Fresh, same Run 1 seed and initialization recipe |
| Budget | 32,000 optimizer updates; 2 A100 80GB; global batch 64 |
| Backbone | Frozen Mistral-7B-Instruct-v0.2 |
| Generator | Run 1 cross attention, 8 description slots, 32 virtual tokens |
| Generator input | Original task descriptions, unchanged |
| LLM text input | Original problem, without `task_def`, in both train and validation |
| Training loss/scale | Original loss and scaling, prompt scale 1.0; no ICL |
| Validation | Original seen/unseen task splits, target benchmarks excluded |
| Generation diagnostic | Fixed independent val/unseen examples, free EOS, step 0 and every 4k |
| Checkpoints | 8k,16k,24k,32k plus named best validation checkpoints |

The objective is to test a plausible recipe, not guarantee a win over T2L. Compared with Run 1 16k, duration and the LR schedule also change; a same-budget original-recipe control is needed for causal attribution. A 16k snapshot inside a 32k schedule is not a matched 16k control.

## New components

- `inlet.run2_setup`: derives three frozen YAMLs from real Run 1 args; validates key assumptions, records changes and file hashes.
- `inlet.aligned_protocol`: removes only the known upstream definition template, checks descriptions structurally, exports before/after metadata, input examples and full token-label coverage.
- `inlet.generation_monitor`: records raw free generation and termination cause on independent validation tasks. It is not benchmark accuracy and does not change the loss or length constraints.
- `scripts/run2_aligned_2gpu.sh`: launches audit, smoke and train from the frozen plan, preserving the two-GPU effective batch.

All flags are opt-in; old Run 1 commands keep their defaults. Audit unknown templates and missing rules instead of silently slicing strings. An invalid semantic description is an experiment blocker even if the tensor shapes work.

## Local validation

Run `python -m inlet.test_run2`, `python -m inlet.test_train_eval_agree`, `python -m inlet.test_eval_protocol`, `python -m compileall -q inlet`, and `bash -n scripts/run2_aligned_2gpu.sh`.

CPU checks do not validate the actual cluster dependency stack, CUDA memory, HF caches or dataset semantics. The separate audit and two-GPU eight-update smoke must pass before 32k.

## Interpretation and evaluation boundaries

Monitor lengths alone cannot prove reasoning or code correctness, and the available held-out mixture may have little long-form coverage. Length collapse may persist after alignment; this run neither assumes nor claims it is fixed. Scale is not fixed to 0.5 at deployment: a later bounded validation sweep should choose one common rule, without selecting the best value per test task. No ICL is required in this training assignment; any future ICL evaluation must use matched demonstrations for Base/Inlet/T2L and be reported separately.
