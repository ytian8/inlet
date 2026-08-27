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
    out, skipped, labels = {}, [], {}
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
        # The step comes from the checkpoint's own config, which save_checkpoint
        # writes as `curstep`. Reading it from the filename works only for the
        # --checkpoint_steps ladder; the best-per-split files carry no step in
        # their name, and those are the ones worth evaluating first.
        step = None
        tc = d.get("train_config") or {}
        if isinstance(tc, dict) and isinstance(tc.get("curstep"), int):
            step = tc["curstep"]
        if step is None:
            m = re.search(r"step(\d+)", str(ck))
            if m:
                step = int(m.group(1))
        if step is None:
            skipped.append((f.name, f"no step in config or filename: {ck}"))
            continue
        label = pathlib.Path(str(ck)).stem
        labels.setdefault(step, set()).add(label)
        # average the real-description variants only; random_descs is a control
        per_tag = d.get("results", {}).get(task, {})
        reals = [_score(v) for k, v in per_tag.items() if k.startswith("eval_descs")]
        reals = [r for r in reals if r is not None]
        if not reals:
            skipped.append((f.name, "no eval_descs entries"))
            continue
        out.setdefault(task, {})[step] = sum(reals) / len(reals)
    return out, skipped, labels


def _plot(curves, full_steps, path):
    """Two panels: every task, and the ten-task average. Zero-shot as a dashed line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping --plot)")
        return

    has_avg = bool(full_steps)
    fig, axes = plt.subplots(1, 2 if has_avg else 1, figsize=(13 if has_avg else 7, 4.5))
    axes = axes if has_avg else [axes]

    _x = lambda v: max(v, 1)          # log axis cannot place 0
    for t in sorted(curves):
        xs = sorted(curves[t])
        axes[0].plot([_x(v) for v in xs], [curves[t][x] for x in xs],
                     marker="o", ms=3, label=t)
        if t in ZERO_SHOT:
            axes[0].axhline(ZERO_SHOT[t], ls=":", lw=0.6, alpha=0.35)
    # log, not symlog: steps are positive, and symlog spends half the axis on
    # negative values that cannot occur. Step 0 (if a run checkpoints there) is
    # nudged onto the axis rather than dropped.
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("score")
    axes[0].set_title("per task (dotted = frozen model)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25)

    if has_avg:
        zs_avg = sum(ZERO_SHOT.values()) / len(ZERO_SHOT)
        ys = [sum(curves[t][s] for t in ZERO_SHOT) / len(ZERO_SHOT) for s in full_steps]
        axes[1].plot([_x(v) for v in full_steps], ys, marker="o", color="tab:blue")
        axes[1].axhline(zs_avg, ls="--", color="k", lw=1,
                        label=f"frozen model {zs_avg:.2f}")
        pk = full_steps[ys.index(max(ys))]
        axes[1].annotate(f"peak {max(ys):.2f}\nstep {pk:,}", (_x(pk), max(ys)),
                         textcoords="offset points", xytext=(8, -14), fontsize=8)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("training step"); axes[1].set_ylabel("10-task average")
        axes[1].set_title("the reported number")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


# The ten benchmarks in the order the results table lists them.
TASK_ORDER = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "openbookqa",
              "piqa", "winogrande", "gsm8k", "mbpp", "humaneval"]


def _paper_table(curves, labels):
    """Markdown, laid out the way the results table is: a row per task.

    The per-step view above is for reading a curve. This is for pasting next to
    the baseline columns, which live outside this repo -- only the frozen model's
    own scores are here, because those are published numbers for Mistral and not
    ours to withhold or to leak.
    """
    steps = sorted({st for c in curves.values() for st in c})
    if not steps:
        return
    tasks = [t for t in TASK_ORDER if t in curves] + \
            [t for t in sorted(curves) if t not in TASK_ORDER]

    def col(st):
        names = labels.get(st, set())
        best = next((n for n in names if "best_" in n), None)
        return best.replace("hypermod_inlet_best_", "") if best else f"step {st:,}"

    hdr = ["Task", "zero-shot"] + [col(st) for st in steps]
    print("\n=== paste-ready ===\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))

    def fmt(v):
        return f"{v:.2f}" if v is not None else ""

    for t in tasks:
        row = [t, fmt(ZERO_SHOT.get(t))] + [fmt(curves[t].get(st)) for st in steps]
        print("| " + " | ".join(row) + " |")

    # Averages only where every task in the set was scored. A partial average is
    # a different measurement and would flatter whichever column has a gap.
    for label, subset in (("**Avg (10 tasks)**", TASK_ORDER),
                          ("**Avg (9, excl. humaneval)**", [t for t in TASK_ORDER if t != "humaneval"])):
        zs = sum(ZERO_SHOT[t] for t in subset) / len(subset)
        cells = []
        for st in steps:
            if all(t in curves and st in curves[t] for t in subset):
                cells.append(f"**{sum(curves[t][st] for t in subset) / len(subset):.2f}**")
            else:
                cells.append("")
        print(f"| {label} | **{zs:.2f}** | " + " | ".join(cells) + " |")
    print("\nBlank average = not every task in that set was scored at that step.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--plot", metavar="PNG", help="also write a two-panel curve")
    p.add_argument("--paper", action="store_true",
                   help="also print a markdown table laid out like the results "
                        "table: one row per task, one column per checkpoint")
    a = p.parse_args(argv)
    d = pathlib.Path(a.results_dir)
    if not d.is_dir():
        print(f"not a directory: {d}", file=sys.stderr)
        return 1

    curves, skipped, labels = collect(d)
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
    if labels and any(len(v) > 1 or not next(iter(v)).endswith(f"step{k}")
                      for k, v in labels.items()):
        print("\ncheckpoints:")
        for k in sorted(labels):
            print(f"  step {k:>9,}  {', '.join(sorted(labels[k]))}")

    print("\npeak per task:")
    for t in tasks:
        s = peaks[t]
        v = curves[t][s]
        zs = ZERO_SHOT.get(t)
        tail = f"   vs zero-shot {zs:.2f} -> {v - zs:+.2f}" if zs else ""
        print(f"  {t:<14} {v:6.2f} at step {s:,}{tail}")

    # THE HEADLINE CURVE. The paper reports an average over the ten benchmarks,
    # so that average as a function of step is the curve that answers "when was
    # the model best". A per-task view can hide it: between step 4,000 and
    # 130,000 the seven multiple-choice tasks moved 0.73 points while the
    # ten-task average moved 2.77.
    #
    # Only computed where every one of the ten was evaluated. A partial average
    # is not comparable across steps and would silently reward whichever step
    # happened to be missing the hardest task.
    full = [s for s in steps if all(t in curves and s in curves[t] for t in ZERO_SHOT)]
    if full:
        print(f"\n=== {len(ZERO_SHOT)}-task average (this is the reported number) ===")
        zs_avg = sum(ZERO_SHOT.values()) / len(ZERO_SHOT)
        print(f"{'step':>9}{'avg':>10}{'vs zero-shot':>15}")
        print(f"{'zero-shot':>9}{zs_avg:>10.2f}")
        best = max(full, key=lambda s: sum(curves[t][s] for t in ZERO_SHOT))
        for s in full:
            v = sum(curves[t][s] for t in ZERO_SHOT) / len(ZERO_SHOT)
            print(f"{s:>9,}{v:>10.2f}{v - zs_avg:>+15.2f}" + ("  <- peak" if s == best else ""))
    else:
        missing = sorted(set(ZERO_SHOT) - set(curves))
        print(f"\n(no {len(ZERO_SHOT)}-task average: never evaluated {missing})")
        print("  Run the sweep over all ten to get the curve the paper reports:")
        print('    ./scripts/sweep_checkpoints.sh <run_dir> "' + " ".join(ZERO_SHOT) + '"')

    if a.paper:
        _paper_table(curves, labels)

    if a.plot:
        _plot(curves, full, a.plot)

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
