# Prompt to hand a cluster agent

Copy everything below the line.

---

Train and evaluate one Inlet model on our cluster.

Repo: https://github.com/ytian8/inlet (public)
Follow `docs/RUN1.md`. It has every command. **Make sure you are on a commit at
or after `a5f8d88`** — earlier ones fail on the very first command.

## What this run is for

Inlet generates an input-layer soft prompt (32 vectors prepended to the
embedded input) from a task description. T2L generates a LoRA injected into
every layer. We are asking whether the input layer — the only substrate you can
deliver through a black-box embedding API — carries task adaptation as well as
per-layer weight injection does.

The blocker was that the generated prompt's **magnitude** grows during training
until it destroys long-form generation: humaneval completions drop from 139
words to 18. Multiplying the trained prompt by 0.5 at inference restores them
and moves the 8-task average by +9.23, with no retraining.

**This run produces a step curve and a `prompt_norm` curve** with six
checkpoints, so we can find where the *scaled* score peaks and how many steps
we actually need. That last question is the point: a previous 147,500-step run
scored no better than a 1,000-step one.

## Steps

1. Setup, then **run `./scripts/smoke.sh 1` and do not skip it** (~20 min). It
   catches the class of bug this codebase has: runs that finish, show a falling
   loss, and report wrong numbers.
2. Train with the command in RUN1.md §5 (~10 h on 2 GPUs). Check the four
   startup lines it lists, especially `NCCL collective timeout : 4:00:00` —
   `0:10:00` means the job aborts ~10 minutes in with no Python traceback.
3. Evaluate in two stages, RUN1.md §6. Stage A finds the scale on three tasks;
   Stage B runs the full ten at the chosen scale. Do not run all ten at several
   scales — the eval re-tokenises per scale and boolq alone is ~13 min each.

## Three things that will bite you

- **`setup_env.sh` can stop at step 0 with no venv created.** That is intended:
  it is an AST check that this checkout still fits the `text-to-lora` beside
  it. Report the failure, do not work around it.
- **Do not set `HF_HOME` after sourcing `common.sh`.** `HF_HUB_CACHE` is fixed
  when `common.sh` runs, so a later `HF_HOME` is ignored and 20 GB lands
  somewhere you did not ask for — while the download reports success.
- **Long jobs need tmux, not `nohup`.** `warm_cache.sh` spawns 12 workers and
  died silently at 19/500 when the ssh session that started it closed, leaving
  no error in the log.

## Please report back

- The four startup lines from step 2, verbatim.
- `train_summary.json` and the training log.
- `prompt_norm` at each of the six checkpoint steps — this run is about that
  number.
- Every JSON under `eval_results_inlet/` (they are small), plus the
  `completions median N words` lines the eval prints next to each score.
- Any `CANARY:` or `prompt_std_across_batch` warnings. Note that the canary
  fires even on an untrained prompt, so a fired canary is not by itself
  evidence anything broke — the trend against step count is what matters.

## Do not

- Skip the smoke test, or loosen a failing gate to get a run started. Every gate
  exists because a specific wrong number got past its absence.
- Change the recipe. `--n_train_ds=479`, the LR, and the global batch are what
  make an Inlet number comparable to a T2L number.
- Select the checkpoint or the scale you report by which scored best on the
  benchmarks. Use it freely for debugging and say that is what you did.
- Put a single-description number in the same table as a reported-protocol
  number. Stage A uses one description and no junk-description controls;
  the reported protocol averages three and adds the controls.
