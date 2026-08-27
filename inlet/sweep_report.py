"""Collect a checkpoint sweep into one table: score vs training step.

    python -m inlet.sweep_report <eval_results_dir> [--zero-shot]

Reads the JSON files `eval_inlet.py` writes, groups them by the training step
baked into the checkpoint filename, and prints score-vs-step per task with the
peak marked.

Written for one question -- **when did the model peak?** -- which the reported
numbers could not answer. On the 147,500-step run the seven multiple-choice
tasks moved by -0.73 between step 4,000 and step 130,000 while the three
generative tasks moved by -10.96, so the whole curve lives in the generative
tasks and a sweep that skips them sees nothing.

This scores checkpoints on benchmark test data. That is the right tool for
debugging and the wrong tool for choosing what to report: pick the reported
checkpoint with `--model_select_split`, which never sees these numbers.
"""

import argparse
import json
import pathlib
import re
import sys

# Frozen-model reference, so a row can be read as "better or worse than not
# using Inlet at all" without looking anything up.
ZERO_SHOT = {
    "arc_challenge": 65.70, "arc_easy": 77.48, "boolq": 71.56, "hellaswag": 49.67,
    "openbookqa": 55.00, "piqa": 73.01, "winogrande": 45.54,
    "gsm8k": 40.71, "mbpp": 44.44, "humaneval": 37.80,
}


def _score(metrics: dict):
    """One number per task, preferring the metric the results table reports."""
    for k in ("humaneval_base_pass@1", "mbpp_base_pass@1", "acc", "accuracy", "pass@1"):
        if k in metrics:
            return float(metrics[k]) * (100.0 if float(metrics[k]) <= 1.0 else 1.0)
    # single-metric files: take the only value
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    if len(vals) == 1:
        return float(vals[0]) * (100.0 if float(vals[0]) <= 1.0 else 1.0)
    return None


def collect(results_dir: pathlib.Path):
    """-> {task: {step: mean score over eval_descs}}, and any files skipped."""
    out, skipped = {}, []
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as exc:
            skipped.append((f.name, str(exc)))
            continue
        ck = d.get("checkpoint")
        task = d.get("task")
        if not task or not ck:
            skipped.append((f.name, "no task/checkpoint field"))
            continue
        m = re.search(r"step(\d+)", str(ck))
        if not m:
            skipped.append((f.name, f"no step in checkpoint name: {ck}"))
            continue
        step = int(m.group(1))
        # average the real-description variants only; random_descs is a control
        per_tag = d.get("results", {}).get(task, {})
        reals = [_score(v) for k, v in per_tag.items() if k.startswith("eval_descs")]
        reals = [r for r in reals if r is not None]
        if not reals:
            skipped.append((f.name, "no eval_descs entries"))
            continue
        out.setdefault(task, {})[step] = sum(reals) / len(reals)
    return out, skipped


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    a = p.parse_args(argv)
    d = pathlib.Path(a.results_dir)
    if not d.is_dir():
        print(f"not a directory: {d}", file=sys.stderr)
        return 1

    curves, skipped = collect(d)
    if not curves:
        print(f"no usable results in {d}")
        for n, why in skipped[:10]:
            print(f"  skipped {n}: {why}")
        return 1

    steps = sorted({s for c in curves.values() for s in c})
    tasks = sorted(curves)
    w = max(12, max(len(t) for t in tasks) + 1)

    print(f"{'step':>9}" + "".join(f"{t:>{w}}" for t in tasks))
    print(f"{'zero-shot':>9}" + "".join(
        f"{ZERO_SHOT[t]:>{w}.2f}" if t in ZERO_SHOT else f"{'—':>{w}}" for t in tasks))
    print("-" * (9 + w * len(tasks)))
    peaks = {t: max(curves[t], key=curves[t].get) for t in tasks}
    for s in steps:
        row = f"{s:>9,}"
        for t in tasks:
            v = curves[t].get(s)
            if v is None:
                row += f"{'':>{w}}"
            else:
                row += f"{v:>{w-2}.2f}" + (" *" if peaks[t] == s else "  ")
        print(row)
    print("\n* = this task's best step in the sweep")

    print("\npeak per task:")
    for t in tasks:
        s = peaks[t]
        v = curves[t][s]
        zs = ZERO_SHOT.get(t)
        tail = f"   vs zero-shot {zs:.2f} -> {v - zs:+.2f}" if zs else ""
        print(f"  {t:<14} {v:6.2f} at step {s:,}{tail}")

    if len(set(peaks.values())) > 1:
        print("\nThe tasks do not peak at the same step. Whichever checkpoint you keep is")
        print("a trade between them -- say which task drove the choice.")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for n, why in skipped[:5]:
            print(f"  {n}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
