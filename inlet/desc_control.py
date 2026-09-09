"""Real-minus-junk: does the generated prompt actually use the description?

    python -m inlet.desc_control <eval_results_dir> [--scale S] [--csv out.csv]

Reads the JSON `eval_inlet.py` writes and, for every (task, checkpoint, scale),
prints the mean over the real `eval_descs__*` arms, the mean over the junk
`random_descs__*` arms, and the gap between them.

That gap is the number the paper's claim rests on. `probe_prompt` can show the
generated prompts DIFFER between a real and a junk description -- cosine 0.67
rather than 0.9999 -- without that difference being worth any accuracy. Only
this measurement settles it. On the 147,500-step model the gap was +0.41 points
on humaneval, i.e. nothing.

It requires the REPORTED protocol: three real descriptions and three junk ones,
which is `./scripts/eval.sh <ckpt>` with no --max-eval-descs / --skip-random-descs.
A run that trimmed either has nothing to compare and is listed as skipped
rather than silently averaged over one description.

Scales are kept apart. The x1.0 and x0.5 arms of one description are two
measurements of two different prompts, and a gap computed across them is not a
gap in anything.
"""

import argparse
import collections
import json
import pathlib
import re
import sys

SCALE_RE = re.compile(r"@s([0-9.eE+-]+)$")


def _score(metrics: dict):
    """One number per task, preferring the metric the results table reports.

    Same order as inlet.sweep_report._score, deliberately: two tools that
    disagree about which metric a task reports produce two tables that cannot
    be put next to each other.
    """
    for k in ("humaneval_base_pass@1", "mbpp_base_pass@1", "acc", "accuracy", "pass@1"):
        if k in metrics:
            v = float(metrics[k])
            return v * (100.0 if v <= 1.0 else 1.0)
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    if len(vals) == 1:
        v = float(vals[0])
        return v * (100.0 if v <= 1.0 else 1.0)
    return None


def _arm_scale(tag: str):
    m = SCALE_RE.search(tag)
    return float(m.group(1)) if m else None


def collect(results_dir: pathlib.Path, want_scale=None):
    """-> rows [(task, ckpt_label, scale, real, junk, gap, n_real, n_junk)], skipped."""
    rows, skipped = [], []
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as exc:
            skipped.append((f.name, str(exc)))
            continue
        task = d.get("task")
        if not task or task not in d.get("results", {}):
            skipped.append((f.name, "no task/results"))
            continue
        per_arm = d["results"][task]

        # Group arms by scale, then split real vs junk inside each scale.
        by_scale = collections.defaultdict(lambda: {"real": [], "junk": []})
        for arm, metrics in per_arm.items():
            sc = _arm_scale(arm)
            base = arm.rsplit("@s", 1)[0]
            if base.startswith("eval_descs"):
                side = "real"
            elif base.startswith("random_descs"):
                side = "junk"
            else:
                continue                      # zero_prompt, vocab, gauss: not this test
            v = _score(metrics)
            if v is not None:
                by_scale[sc][side].append(v)

        ck = d.get("checkpoint") or "<none>"
        label = pathlib.Path(str(ck)).stem
        for sc, sides in sorted(by_scale.items(), key=lambda kv: (kv[0] is None, kv[0])):
            if want_scale is not None and sc != want_scale:
                continue
            real, junk = sides["real"], sides["junk"]
            if not real:
                continue
            if not junk:
                skipped.append((
                    f.name,
                    f"no junk arms at scale {sc if sc is not None else 'unscaled'} "
                    "-- run without --skip-random-descs"))
                continue
            if len(real) < 3:
                skipped.append((
                    f.name,
                    f"{len(real)} real description(s), not 3 -- this is the debug "
                    "protocol, drop --max-eval-descs"))
                continue
            r, j = sum(real) / len(real), sum(junk) / len(junk)
            rows.append((task, label, sc, r, j, r - j, len(real), len(junk)))
    return rows, skipped


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--scale", type=float, default=None, metavar="S",
                   help="only the '@sS' arms (e.g. --scale 0.5)")
    p.add_argument("--csv", metavar="PATH", help="also write the rows as CSV")
    a = p.parse_args(argv)

    d = pathlib.Path(a.results_dir)
    if not d.is_dir():
        print(f"not a directory: {d}", file=sys.stderr)
        return 1

    rows, skipped = collect(d, a.scale)
    if not rows:
        print(f"no real-vs-junk pairs in {d}")
        for n, why in skipped[:20]:
            print(f"  skipped {n}: {why}")
        if skipped:
            print("\nThis measurement needs the reported protocol: three real "
                  "descriptions\nand three junk ones. That is `./scripts/eval.sh "
                  "<ckpt>` with NEITHER\n--max-eval-descs NOR --skip-random-descs.")
        return 1

    w = max(14, max(len(r[0]) for r in rows) + 1)
    print(f"{'task':<{w}}{'scale':>7}{'real':>9}{'junk':>9}{'gap':>9}  checkpoint")
    print("-" * (w + 34 + 12))
    for task, label, sc, r, j, gap, nr, nj in sorted(rows):
        scs = "—" if sc is None else f"{sc:g}"
        print(f"{task:<{w}}{scs:>7}{r:>9.2f}{j:>9.2f}{gap:>+9.2f}  {label}")

    # The average gap is the headline, but only over one (checkpoint, scale).
    groups = collections.defaultdict(list)
    for task, label, sc, r, j, gap, nr, nj in rows:
        groups[(label, sc)].append(gap)
    print()
    for (label, sc), gaps in sorted(groups.items()):
        scs = "unscaled" if sc is None else f"x{sc:g}"
        print(f"mean gap over {len(gaps)} task(s), {label} {scs}: "
              f"{sum(gaps) / len(gaps):+.2f}")

    print("\ngap = mean(3 real descriptions) - mean(3 junk descriptions).")
    print("A gap near zero means the generator is not using the description,")
    print("however different the prompts look to probe_prompt.")

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for n, why in skipped[:20]:
            print(f"  {n}: {why}")

    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("task,checkpoint,scale,real,junk,gap,n_real,n_junk\n")
            for task, label, sc, r, j, gap, nr, nj in sorted(rows):
                fh.write(f"{task},{label},{'' if sc is None else sc},"
                         f"{r:.4f},{j:.4f},{gap:.4f},{nr},{nj}\n")
        print(f"\ncsv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
