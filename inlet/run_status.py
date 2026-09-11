"""One table of where a training run currently is. Safe to call at any time.

    python -m inlet.run_status --run run3_m20

Reads only files that are already on disk -- the training log, `probe_prompt`
JSON, and whatever `eval_inlet` has written -- so it never touches the GPU, never
loads a checkpoint, and cannot disturb a run in progress. Missing cells are
blank rather than an error: a checkpoint that has not been evaluated yet is the
normal state, not a problem.

Columns, and why each is here:

  prompt_norm      the magnitude the whole scale story turns on. Run 1 ended at
                   0.2426; the old collapsed run reached 0.4056.
  separation       mean cos(real,real) - mean cos(real,junk) from probe_prompt.
                   This is the number that says whether the generator reads the
                   description at all. Run 1: 0.0002 at step 1,000, 0.2117 at
                   16,000.
  per-task scores  at x1.0 and x0.5, because short-output and long-output tasks
                   want opposite ends of that range and a single column would
                   hide it.
"""

import argparse
import glob
import json
import os
import re
import sys

from inlet._env import bootstrap, output_root, user_path  # noqa: E402

STEP_RE = re.compile(r"hypermod_inlet_step(\d+)")
# Same metric order as sweep_report._score and desc_control._score. Three tools
# that disagree about which metric a task reports produce three tables that
# cannot be read against each other.
METRIC_ORDER = ("humaneval_base_pass@1", "mbpp_base_pass@1", "acc", "accuracy", "pass@1")


def _score(metrics):
    for k in METRIC_ORDER:
        if k in metrics:
            v = float(metrics[k])
            return v * (100.0 if v <= 1.0 else 1.0)
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    if len(vals) == 1:
        v = float(vals[0])
        return v * (100.0 if v <= 1.0 else 1.0)
    return None


def prompt_norms(log_path):
    """-> {step: prompt_norm} from the `[step N] train:` lines."""
    out = {}
    if not log_path or not os.path.isfile(log_path):
        return out
    pat = re.compile(r"\[step (\d+)\] train:.*?prompt_norm=([0-9.]+)")
    with open(log_path, "rb") as f:
        for raw in f:
            m = pat.search(raw.decode("utf-8", "replace"))
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def separations(out_dir, run):
    """-> {step: separation} from probe_prompt JSON."""
    out = {}
    for f in glob.glob(os.path.join(out_dir, f"{run}_probe_step*.json")):
        try:
            d = json.load(open(f))
            rr = list(d["cos_real_real"].values())
            rj = list(d["cos_real_junk"].values())
            step = d.get("curstep")
            if step is None:
                m = re.search(r"probe_step(\d+)", f)
                step = int(m.group(1)) if m else None
            if step is not None and rr and rj:
                out[int(step)] = sum(rr) / len(rr) - sum(rj) / len(rj)
        except Exception:
            continue
    return out


def scores(eval_dir):
    """-> {(task, step, scale): score} for every checkpoint result on disk."""
    out = {}
    for f in glob.glob(os.path.join(eval_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        task, ck = d.get("task"), d.get("checkpoint") or ""
        if not task or task not in d.get("results", {}):
            continue
        step = (d.get("train_config") or {}).get("curstep")
        if step is None:
            m = STEP_RE.search(str(ck))
            if not m:
                continue                       # best_val_* / final: no step to place it at
            step = int(m.group(1))
        for arm, metrics in d["results"][task].items():
            if not arm.startswith("eval_descs"):
                continue                       # junk arms belong to desc_control
            sc = float(arm.rsplit("@s", 1)[1]) if "@s" in arm else 1.0
            v = _score(metrics)
            if v is not None:
                out.setdefault((task, int(step), sc), []).append(v)
    return {k: sum(v) / len(v) for k, v in out.items()}


def main(argv=None):
    bootstrap()
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_name, e.g. run3_m20")
    p.add_argument("--out-dir", default=None, help="INLET_OUTPUT_ROOT (default: env)")
    p.add_argument("--tasks", nargs="+", default=["arc_challenge", "humaneval"])
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.5])
    a = p.parse_args(argv)

    out_dir = user_path(a.out_dir) if a.out_dir else output_root()
    log = os.path.join(out_dir, f"{a.run}.train.log")
    norms = prompt_norms(log)
    seps = separations(out_dir, a.run)
    scs = scores(os.path.join(out_dir, "eval_results_inlet"))

    steps = sorted(set(norms) | set(seps) | {k[1] for k in scs})
    if not steps:
        print(f"nothing yet for run '{a.run}' under {out_dir}")
        print(f"  looked for: {log}")
        print(f"              {out_dir}/{a.run}_probe_step*.json")
        print(f"              {out_dir}/eval_results_inlet/*.json")
        return 1

    cols = [(t, s) for t in a.tasks for s in a.scales]
    # 13, not 12: "humanev x0.5" is exactly 12 characters, so at w=12 adjacent
    # headers run together with no gap.
    w = 13
    hdr = f"{'step':>8}{'prompt_norm':>13}{'separation':>12}"
    hdr += "".join(f"{t.split('_')[0][:7] + ' x' + f'{s:g}':>{w}}" for t, s in cols)
    print(hdr)
    print("-" * len(hdr))
    for st in steps:
        row = f"{st:>8,}"
        row += f"{norms[st]:>13.4f}" if st in norms else f"{'':>13}"
        row += f"{seps[st]:>12.4f}" if st in seps else f"{'':>12}"
        for t, s in cols:
            v = scs.get((t, st, s))
            row += f"{v:>{w}.2f}" if v is not None else f"{'':>{w}}"
        print(row)

    print("\nreference -- Run 1 (head_lr_mult=1):")
    print(f"{'1,000':>8}{0.1547:>13.4f}{0.0002:>12.4f}")
    print(f"{'4,000':>8}{0.1755:>13.4f}{0.0516:>12.4f}")
    print(f"{'16,000':>8}{0.2426:>13.4f}{0.2117:>12.4f}")
    print("\nBlank = not measured yet. Scores are one description, no junk")
    print("control -- progress only, not reportable. See docs/RUN2.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
