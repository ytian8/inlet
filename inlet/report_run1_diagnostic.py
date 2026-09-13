"""Report the fixed queue without mixing tasks, scales, ICL, or repetitions."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def score(metrics):
    for key in ('mbpp_base_pass@1', 'humaneval_base_pass@1', 'acc', 'accuracy'):
        if key in metrics:
            value = float(metrics[key])
            if not 0 <= value <= 1:
                raise ValueError(f'{key} must be a fraction, got {value}')
            return 100 * value
    raise ValueError(f'No declared metric: {list(metrics)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    a = p.parse_args()
    root = Path(a.out).resolve()
    jobs = json.loads((root/'plan.json').read_text())
    values = collections.defaultdict(list)
    missing = []
    raw_rows = []
    for job in jobs:
        done = root/job['id']/'done.json'
        if not done.exists():
            missing.append(job['id'])
            continue
        data = json.loads(Path(json.loads(done.read_text())['result']).read_text())
        for tag, metrics in data['results'][job['task']].items():
            group = ('matched' if tag.startswith('eval_descs__') or job['method'] == 't2l'
                     else 'mismatched' if tag.startswith('mismatch_descs__')
                     else 'junk' if tag.startswith('random_descs__') else 'base')
            key = (job['task'], job['method'], job['icl'], job['scale'], group, job['repeat'])
            val = score(metrics)
            values[key].append(val)
            raw_rows.append(dict(zip(('task', 'method', 'icl', 'scale', 'group', 'repeat'), key),
                                 arm=tag, score=val))
    collapsed = collections.defaultdict(list)
    for key, vals in values.items():
        expected = 1 if key[1] == 'base' else 3
        if len(vals) != expected:
            # Do not present a two-description partial as a three-description mean.
            missing.append(f'incomplete group {key}: {len(vals)}/{expected}')
            continue
        collapsed[key[:-1]].append((key[-1], statistics.mean(vals)))
    summary = []
    for key, reps in sorted(collapsed.items(), key=lambda kv: str(kv[0])):
        vals = [v for _, v in reps]
        row = dict(zip(('task', 'method', 'icl', 'scale', 'group'), key),
                   mean=statistics.mean(vals), repeat_min=min(vals), repeat_max=max(vals),
                   n_repeats=len(vals), repeats=dict(reps))
        summary.append(row)
        print(f'{key}: {row["mean"]:.2f} (repeat range {min(vals):.2f}–{max(vals):.2f}, n={len(vals)})')
    (root/'summary.json').write_text(json.dumps({'cells': summary, 'missing': missing}, indent=2)+'\n')
    if raw_rows:
        with (root/'scores.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
            w.writeheader()
            w.writerows(raw_rows)
    print(f'{len(missing)} missing/incomplete entries. This is a FOUR-task exploratory experiment, not Avg10.')


if __name__ == '__main__':
    main()
