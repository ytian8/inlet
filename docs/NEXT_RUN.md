# The next training run

Read this before booking GPU time. It is ordered by what to do, not by what we
learned; the evidence is in `docs/DIAGNOSIS_FORMAT_PRIOR.md`.

## What is wrong

The generator is not conditioning on its input. cos(prompt from a real
description, prompt from junk) = **0.9999**; swapping every description for
gibberish costs **0.41 points**. Whatever else is true, a text-to-parameter
method whose output does not depend on the text is not doing its job, and this
is the first thing a reviewer will check.

This has a name. Hyper-DFS, Appendix B.2: *"A known failure mode of
hypernetworks is representation collapse: the hypernetwork may learn to produce
nearly identical parameters regardless of the conditioning input, rendering the
conditioning redundant."* Our setup makes the collapsed solution optimal for two
independent reasons, and both have to be removed.

**1. The conditioning really is redundant.** Upstream trains on
`LOL_TEMPLATE = "{task_def}\n\n{problem}"`, and `task_def` is the entire
"Definition: ..." paragraph — a complete natural-language spec of the task, in
the LM's own context, on all 479 training tasks. lol_751's says *"only use
subtraction"*, which no model infers from the word problem.

Measured on a frozen Mistral-7B, 10 `lol_*` tasks, nothing else changed:

| | rougeL | completion length |
|---|---|---|
| with `task_def` (what training shows the model) | **28.37** | 1–65 words |
| without it (what the benchmarks show it) | **4.64** | 79–237 words |
| | **−23.73** | |

So the text channel is already delivering ~24 points of the task for free. That
is how little is left for the prompt to be worth carrying. And **none** of the
ten benchmark templates carries a task spec — `"Please answer the following
question: {question}"` and the like. The generator is trained where it is
redundant and deployed where it is the only channel.

**2. Nothing in the loss forbids a constant.** Plain SFT never rewards telling
two descriptions apart. There is no diversity, contrastive, or variance term.

## What to change

Both changes are in the repo, both are cheap, and they are complementary: the
first makes conditioning *useful*, the second makes it *obligatory*.

```
--strip_taskdef_in_training      # removes {task_def} from the training prompt
--prompt_diversity 0.01          # hinge that forbids a constant prompt
--prompt_diversity_target 0.3    # sweep this; see below
```

Two more, added 2026-09-07:

```
--contrastive 0.1                # the prompt must beat a neighbour's ON ITS OWN
--contrastive_margin 0.5         #   task. ~2x step time (second LM forward).
--freeze_base                    # `base` stops absorbing the task-agnostic part
```

`prompt_diversity` only asks the prompts to differ; `contrastive` asks them to
differ *usefully*, which is what the random-description control measures. It is
the more direct objective and the more expensive one — read the 2x before
budgeting. `contrastive_shift` must be a multiple of `n_points_per_task`; every
row is checked and a row paired with its own task raises.

`--prompt_diversity` costs **no extra forward pass**: the spread is taken across
the batch the SFT term already ran. It is a hinge on `varying_fraction` — the
exact number `probe_prompt.py` reports, which measured **5.4%** on the 147.5k
run — so it is bounded in `[0, target]`, stops pushing once the target is met,
and the quantity it optimises is the quantity the diagnostic prints.

(Hyper-DFS uses a raw `-Var`. Don't: it is unbounded below, and its cheapest
minimiser is to blow the prompt's scale up.)

## Verify on a short run first — do NOT go straight to a long one

Collapse is visible in the first few thousand steps, so this costs hours, not
days. Run the 2×2 and read **`varying_fraction` and the random-description
control — not benchmark scores**:

```bash
for taskdef in "" "--strip_taskdef_in_training"; do
  for div in "" "--prompt_diversity 0.01 --prompt_diversity_target 0.3"; do
    ./scripts/train.sh 8 --run_name="probe$(echo $taskdef$div | tr -d ' -')" \
        --max_steps 4000 $taskdef $div
  done
done
```

Then, on each resulting checkpoint:

```bash
python -m inlet.probe_prompt --checkpoint <ckpt> --out <name>.json
```

**Read it like this:**

| `varying_fraction` after 4k steps | what it means |
|---|---|
| still ≈ 5% | that arm collapsed; the change did not take |
| rises toward the target | conditioning is being learned — take this arm to the long run |

If the `--strip_taskdef_in_training` arms show a much higher training loss,
that is expected and is the point: the task is genuinely harder once the answer
is not written in the prompt. What matters is whether the prompt starts
carrying it.

## Gate the long run

Add to the run and check at the first validation, then every validation:

* `varying_fraction` — abort if it is still under ~0.1 at 5k steps
* `prompt_std_across_batch` — already logged every step
* the generative canary (`inlet/canary.py`) and `val/generative`
* `--model_select_split val/unseen` and `--save_best_per_split`

The 147.5k-step run had no gate on conditioning. It ran to completion and the
loss curve looked fine the whole way.

## What we are NOT claiming

The mechanism behind the second failure — the trained prompt being *actively*
harmful on generative tasks, worse than same-magnitude noise — is not settled.
Removing the redundancy and forbidding the constant may or may not fix it. The
short runs above will say. See `docs/DIAGNOSIS_FORMAT_PRIOR.md` §5c–5d for the
hypothesis we tested and retracted.
