"""The "no explanation" instruction, isolated so it can be switched off.

Upstream's `preprocessing.py:29` appends, unconditionally, to every `lol_*`
task:

    task_def += " Please complete the task without any explanation."

All 479 training tasks are `lol_*`, so 479/479 training prompts carry that
sentence.  Of the ten reported benchmarks, the seven multiple-choice ones
carry the same clause in their own templates and the three generative ones
(gsm8k, mbpp, humaneval) do not.  That is a perfect split along exactly the
line where Inlet's sign flips, which makes it a hypothesis rather than a
coincidence -- and hypotheses get measured.

Two directions, both applied to the evaluator's samples so that all ten
tasks go through one code path:

    strip   remove the clause from a benchmark that has it   (7 MC tasks)
    add     prepend the training sentence to one that lacks it (3 generative)

Going through `evaluator.samples` rather than the per-task templates also
side-steps upstream's dead `in_context_message` in `eval_gsm8k` (the local
`problem` it builds is discarded by the second `template.format` call), which
would otherwise make the gsm8k arm a silent no-op.
"""

import dataclasses

TRAIN_SENTENCE = "Please complete the task without any explanation."

# The substring shared with the multiple-choice templates.  Stripping only the
# clause, not the whole sentence, keeps every other format constraint intact
# ("You must respond with the letter corresponding to the correct choice."),
# so the contrast isolates the phrase and not "format guidance" in general.
CLAUSE = " without any explanation"

MODES = ("keep", "strip", "add", "strip_taskdef", "desc_instead_of_taskdef")

# QASample.question, MathSample.problem, evalplus TextToCodeProblem.instruction,
# RougeSample.prompt. No class carries two of these, so the match is unambiguous
# -- `text_field` raises rather than guessing if that ever stops being true.
_TEXT_FIELDS = ("question", "problem", "instruction", "prompt")


def text_field(sample) -> str:
    """Name of the attribute holding a sample's prompt text."""
    found = [f for f in _TEXT_FIELDS if hasattr(sample, f)]
    if len(found) != 1:
        raise RuntimeError(
            f"cannot locate the prompt text on {type(sample).__name__}: "
            f"matched {found!r} of {list(_TEXT_FIELDS)}"
        )
    return found[0]


def strip_clause(text: str) -> str:
    return text.replace(CLAUSE, "")


def add_sentence(text: str) -> str:
    """Prepend the training sentence the way training composes it.

    `LOL_TEMPLATE` is "{task_def}\\n\\n{problem}" and the sentence sits at the
    end of `task_def`, i.e. immediately before a blank line and the problem.
    """
    return f"{TRAIN_SENTENCE}\n\n{text}"


def strip_task_def(text: str) -> str:
    """Drop the task definition from a `lol_*` prompt, keeping the problem.

    `LOL_TEMPLATE` is "{task_def}\n\n{problem}", and `task_def` is the whole
    "Definition: ..." paragraph -- a complete natural-language spec of the task.
    At training time it sits in the LM's own context, so the soft prompt has
    nothing left to carry; at benchmark time the templates are generic
    boilerplate and carry no task spec at all. Removing it here measures how
    much of the task the text channel was supplying.

    `task_def` can contain a single "\n" (the comma-separated-list line
    upstream appends) but never a blank line, so the first "\n\n" is the
    template's own separator.
    """
    head, sep, rest = text.partition("\n\n")
    if not sep or not head.strip() or not rest.strip():
        raise RuntimeError(
            "cannot split a task definition off this prompt: expected "
            "'{task_def}\\n\\n{problem}' with both halves non-empty, got "
            f"{text[:120]!r}"
        )
    return rest


def replace_task_def(text: str, description: str) -> str:
    """Swap the task definition for the task's own description.

    This is the experiment that decides whether closing the text channel can
    work at all. `strip_taskdef` costs the frozen model 20+ rougeL, which is the
    headroom a conditioned prompt would have to fill. But the generator is fed a
    DESCRIPTION, not the definition, and the two are not the same text: lol_751's
    definition says "only use subtraction" while its LLM-written descriptions
    talk about "a math word problem". If the description is the vaguer of the
    two, removing the definition does not make the task harder -- it makes it
    unanswerable, and no generator can recover it.

    Scoring the frozen model with the description in the definition's place puts
    a number on that:

        keep                    28.37   the definition
        desc_instead_of_taskdef     ?   the description, same slot
        strip_taskdef            4.64   nothing

    Near the top, the description carries the task and the fix has something to
    learn. Near the bottom, it does not and the fix is doomed for a reason that
    has nothing to do with the generator.
    """
    head, sep, rest = text.partition("\n\n")
    if not sep or not head.strip() or not rest.strip():
        raise RuntimeError(
            "cannot swap a task definition into this prompt: expected "
            "'{task_def}\\n\\n{problem}' with both halves non-empty, got "
            f"{text[:120]!r}"
        )
    if not description or not description.strip():
        raise ValueError("desc_instead_of_taskdef needs a non-empty description")
    return f"{description.strip()}\n\n{rest}"


def _replace_text(sample, field: str, new: str):
    try:
        object.__setattr__(sample, field, new)   # works for frozen dataclasses too
        return sample
    except (AttributeError, TypeError):
        if dataclasses.is_dataclass(sample):
            return dataclasses.replace(sample, **{field: new})
        raise


def apply_to_evaluator(evaluator, mode: str, description: str | None = None) -> dict:
    """Rewrite the evaluator's prompts in place.  Returns a provenance record.

    Raises when the rewrite changes nothing.  A silent no-op would reproduce
    the baseline number exactly and read as "the ablation had no effect",
    which is the one wrong conclusion this experiment can produce.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "keep":
        return {"mode": "keep", "n_samples": len(evaluator.samples), "n_changed": 0}

    samples = evaluator.samples
    if mode == "desc_instead_of_taskdef":
        if not description:
            raise ValueError(
                "mode 'desc_instead_of_taskdef' needs `description=`; without one "
                "there is nothing to put in the definition's place."
            )

        def fn(t):
            return replace_task_def(t, description)
    else:
        fn = {"strip": strip_clause, "add": add_sentence,
              "strip_taskdef": strip_task_def}[mode]
    changed = 0
    first_i = None
    before = after = None
    for i, s in enumerate(samples):
        field = text_field(s)
        old = getattr(s, field)
        new = fn(old)
        if new != old:
            if before is None:
                before, after, first_i = old, new, i
            samples[i] = _replace_text(s, field, new)
            # read back: `_replace_text` writes through `object.__setattr__`,
            # which is silent on a class that ignores the write.
            if getattr(samples[i], field) != new:
                raise RuntimeError(
                    f"rewrite did not stick on {type(s).__name__}.{field}"
                )
            changed += 1

    if changed == 0:
        raise RuntimeError(
            f"--format-instruction {mode} changed 0 of {len(samples)} prompts.\n"
            + (
                f"This task's template does not contain {CLAUSE!r}, so there is "
                "nothing to strip -- `strip` only applies to the seven "
                "multiple-choice benchmarks."
                if mode == "strip"
                else "this mode should always change the text; this is a bug."
            )
        )

    # `samples` was bound once.  If `evaluator.samples` is a property that
    # hands out a fresh list, every rewrite above went into a throwaway and the
    # eval would quietly report the baseline number.
    live = evaluator.samples
    if getattr(live[first_i], text_field(live[first_i])) != after:
        raise RuntimeError(
            "rewrites are not visible through evaluator.samples -- it is "
            "handing out a copy, so this ablation would silently be a no-op"
        )

    return {
        "mode": mode,
        "description": description,
        "n_samples": len(samples),
        "n_changed": changed,
        "example_before": before[-200:],
        "example_after": after[-200:] if mode == "strip" else after[:200],
    }
