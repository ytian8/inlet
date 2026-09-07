"""Checks for `format_ablation`, including a real known-bad control.

Run: python -m inlet.test_format_ablation
"""

import dataclasses
import sys

from inlet.format_ablation import (
    CLAUSE,
    TRAIN_SENTENCE,
    add_sentence,
    apply_to_evaluator,
    strip_clause,
    text_field,
)

# The real templates, copied verbatim from upstream vllm_eval.py so that a
# change upstream shows up here as a failing check rather than as a silent
# no-op at eval time.
ARC = (
    "Answer the question below by choosing the correct choice.\n\n{question}\n\n"
    "A: a\nB: b\nC: c\nD: d\n\n"
    "You must respond with the letter corresponding to the correct choice without any explanation."
)
HUMANEVAL = "Write a solution to the following problem:\n```python\ndef f():\n```"

_fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(name)


@dataclasses.dataclass
class QASample:
    question: str
    answer: str = ""


@dataclasses.dataclass(frozen=True)
class MathSample:
    problem: str
    answer: int = 0


class EvalplusSample:            # not a dataclass, like upstream's
    def __init__(self, instruction):
        self.instruction = instruction


class Evaluator:
    def __init__(self, samples):
        self.samples = list(samples)


def main():
    print("format_ablation")

    # --- pure text transforms ---
    stripped = strip_clause(ARC)
    check("strip removes the clause", CLAUSE not in stripped)
    check(
        "strip keeps the rest of the format constraint",
        stripped.endswith("You must respond with the letter corresponding to the correct choice."),
        stripped[-90:],
    )
    check(
        "strip changes exactly len(CLAUSE) characters",
        len(ARC) - len(stripped) == len(CLAUSE),
        f"{len(ARC) - len(stripped)} vs {len(CLAUSE)}",
    )
    check("strip is a no-op where the clause is absent", strip_clause(HUMANEVAL) == HUMANEVAL)

    added = add_sentence(HUMANEVAL)
    check(
        "add composes the way LOL_TEMPLATE does",
        added == TRAIN_SENTENCE + "\n\n" + HUMANEVAL,
    )

    # --- field discovery on all three sample shapes ---
    check("field: QASample", text_field(QASample("q")) == "question")
    check("field: MathSample", text_field(MathSample("p")) == "problem")
    check("field: evalplus", text_field(EvalplusSample("i")) == "instruction")
    try:
        text_field(object())
        check("field: unknown shape raises", False)
    except RuntimeError:
        check("field: unknown shape raises", True)

    # --- apply, including the frozen dataclass path ---
    ev = Evaluator([QASample(ARC), QASample(ARC)])
    rec = apply_to_evaluator(ev, "strip")
    check("apply/strip counts every sample", rec["n_changed"] == 2, str(rec["n_changed"]))
    check("apply/strip actually rewrote", CLAUSE not in ev.samples[0].question)

    frozen = Evaluator([MathSample("solve this " + CLAUSE)])
    apply_to_evaluator(frozen, "strip")
    check("apply works on a frozen dataclass", CLAUSE not in frozen.samples[0].problem)

    gen = Evaluator([EvalplusSample(HUMANEVAL)])
    rec = apply_to_evaluator(gen, "add")
    check("apply/add prepends", gen.samples[0].instruction.startswith(TRAIN_SENTENCE))
    check("apply/add reports one change", rec["n_changed"] == 1)

    keep = Evaluator([QASample(ARC)])
    rec = apply_to_evaluator(keep, "keep")
    check("keep is a no-op", keep.samples[0].question == ARC and rec["n_changed"] == 0)

    # --- KNOWN-BAD CONTROL ---
    # The failure this experiment can actually produce: `strip` applied to a
    # task whose template never had the clause silently reproduces the baseline
    # number, and the run reads as "the ablation had no effect".  The guard has
    # to turn that into a crash.  If this check ever passes-by-not-raising, the
    # ablation is untrustworthy on every task.
    try:
        apply_to_evaluator(Evaluator([EvalplusSample(HUMANEVAL)]), "strip")
        check("KNOWN-BAD: silent no-op strip is rejected", False, "no exception raised")
    except RuntimeError as e:
        check("KNOWN-BAD: silent no-op strip is rejected", "changed 0 of 1" in str(e), str(e)[:80])

    # A second control: a sample shape we cannot rewrite must raise, not be
    # skipped.  Skipping is the same silent-baseline failure as above.
    class ReadOnly:
        def __init__(self, t):
            self._t = t

        @property
        def instruction(self):
            return self._t

    try:
        apply_to_evaluator(Evaluator([ReadOnly(HUMANEVAL)]), "add")
        check("KNOWN-BAD: unwritable sample raises", False, "no exception")
    except (RuntimeError, AttributeError, TypeError) as e:
        check("KNOWN-BAD: unwritable sample raises", True, str(e)[:60])

    # A third control: an evaluator whose `.samples` hands out copies would
    # make every rewrite invisible to the eval.
    class Copying:
        def __init__(self, s):
            self._s = list(s)

        @property
        def samples(self):
            return [dataclasses.replace(x) for x in self._s]

    try:
        apply_to_evaluator(Copying([MathSample(ARC)]), "strip")
        check("KNOWN-BAD: copying .samples is rejected", False, "no exception")
    except RuntimeError as e:
        check("KNOWN-BAD: copying .samples is rejected", "handing out a copy" in str(e), str(e)[:80])


    # --- strip_taskdef, the channel the task actually travelled down ---
    from inlet.format_ablation import strip_task_def

    LOL = ("In this task you are given commands and must decide if the "
           "interpretation is correct. Please complete the task without any "
           "explanation.\n\nCommand: eq { count { all_rows } ; 17 }")
    got = strip_task_def(LOL)
    check("strip_taskdef keeps only the problem",
          got == "Command: eq { count { all_rows } ; 17 }", repr(got[:60]))
    check("strip_taskdef survives a single newline inside task_def",
          strip_task_def("Def line one.\nAnswer as a list.\n\nthe problem")
          == "the problem")

    @dataclasses.dataclass
    class RougeSample:
        prompt: str
        response: str = ""

    check("field: RougeSample", text_field(RougeSample("x")) == "prompt")

    ev = Evaluator([RougeSample(LOL), RougeSample(LOL)])
    rec = apply_to_evaluator(ev, "strip_taskdef")
    check("apply/strip_taskdef rewrote both", rec["n_changed"] == 2)
    check("apply/strip_taskdef dropped the definition",
          "Please complete" not in ev.samples[0].prompt)

    # KNOWN-BAD: a prompt with no "\n\n" has no definition to split off, and
    # silently returning it unchanged would report the with-definition number
    # as if it were the without-definition one.
    for bad, why in [("no blank line here", "no separator"),
                     ("\n\nproblem only", "empty task_def"),
                     ("task def only\n\n   ", "empty problem")]:
        try:
            strip_task_def(bad)
            check(f"KNOWN-BAD: {why} raises", False, "no exception")
        except RuntimeError:
            check(f"KNOWN-BAD: {why} raises", True)


    # --- desc_instead_of_taskdef: the experiment that decides the fix ---
    from inlet.format_ablation import replace_task_def

    DESC = "Analyze the given command and decide whether the interpretation fits."
    got = replace_task_def(LOL, DESC)
    check("desc replaces the definition, keeps the problem",
          got == DESC + "\n\nCommand: eq { count { all_rows } ; 17 }", repr(got[:70]))
    check("the definition is gone", "without any explanation" not in got)

    ev = Evaluator([RougeSample(LOL), RougeSample(LOL)])
    rec = apply_to_evaluator(ev, "desc_instead_of_taskdef", description=DESC)
    check("apply/desc rewrote every row", rec["n_changed"] == 2)
    check("the description is recorded in the provenance", rec["description"] == DESC)
    check("the rewritten prompt starts with the description",
          ev.samples[0].prompt.startswith(DESC))

    # KNOWN-BAD: the mode needs a description. Falling back to "leave it alone"
    # would score the WITH-definition prompt and report it as the description's
    # number -- the one confusion this experiment exists to avoid.
    try:
        apply_to_evaluator(Evaluator([RougeSample(LOL)]), "desc_instead_of_taskdef")
        check("KNOWN-BAD: a missing description raises", False, "no exception")
    except ValueError as e:
        check("KNOWN-BAD: a missing description raises", "needs" in str(e), str(e)[:60])

    for bad in ("", "   "):
        try:
            replace_task_def(LOL, bad)
            check(f"KNOWN-BAD: blank description ({bad!r}) raises", False, "no exception")
        except ValueError:
            check(f"KNOWN-BAD: blank description ({bad!r}) raises", True)

    print(f"\n{'FAILED: ' + ', '.join(_fails) if _fails else 'all checks passed'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
