"""Print or run a bounded, resumable evaluation queue. Never starts training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASKS = ('arc_challenge', 'winogrande', 'gsm8k', 'mbpp')


def build_jobs(checkpoint, manifest, out, repeats=2):
    jobs = []
    for task in TASKS:
        entry = manifest['tasks'][task]
        desc = Path(entry['descriptions'])
        real_path = out / 'descriptions' / f'{task}_real.json'
        all_descs = json.loads(desc.read_text())
        real = {k: v for k, v in all_descs.items() if k.startswith('eval_descs__')}
        if len(real) != 3:
            raise ValueError('exactly three real descriptions required')
        real_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(real, indent=2) + '\n'
        if real_path.exists() and real_path.read_text() != encoded:
            raise ValueError('description inputs changed in an existing run')
        if not real_path.exists():
            real_path.write_text(encoded)
        for rep in range(repeats if task in ('gsm8k', 'mbpp') else 1):
            for icl in (False, True):
                base_id = f'{task}/rep{rep}/base_icl{int(icl)}'
                jobs.append({'id': base_id, 'task': task, 'method': 'base', 'repeat': rep,
                             'icl': icl, 'scale': None,
                             'argv': [sys.executable, '-m', 'inlet.eval_inlet', '--task', task,
                                      '--zero-prompt', '--save-raw', '--out-dir', str(out/base_id)]
                                     + (['--use-icl'] if icl else [])})
                for scale in (0.5, 1.0):
                    job_id = f'{task}/rep{rep}/inlet_s{scale:g}_icl{int(icl)}'
                    jobs.append({'id': job_id, 'task': task, 'method': 'inlet', 'repeat': rep,
                                 'icl': icl, 'scale': scale,
                                 'argv': [sys.executable, '-m', 'inlet.eval_inlet', '--task', task,
                                          '--checkpoint', checkpoint, '--descriptions', str(real_path if icl else desc),
                                          '--prompt-scales', str(scale), '--save-raw', '--out-dir', str(out/job_id)]
                                         + (['--use-icl'] if icl else [])})
                for tag, adapter in entry.get('adapters', {}).items():
                    job_id = f'{task}/rep{rep}/t2l_{tag}_icl{int(icl)}'
                    jobs.append({'id': job_id, 'task': task, 'method': 't2l', 'repeat': rep,
                                 'icl': icl, 'scale': None, 'description_tag': tag,
                                 'argv': [sys.executable, '-m', 'inlet.eval_t2l_control', '--task', task,
                                          '--adapter', adapter, '--descriptions', str(desc),
                                          '--description-tag', tag, '--out', str(out/job_id/'result.json')]
                                         + (['--use-icl'] if icl else [])})
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--execute', action='store_true', help='default only prints/writes the plan')
    p.add_argument('--allow-missing-t2l', action='store_true', help='explicit partial experiment, no beat-T2L claim')
    p.add_argument('--only-task', choices=TASKS)
    a = p.parse_args()
    checkpoint = str(Path(a.checkpoint).resolve())
    if not Path(checkpoint).is_file():
        raise SystemExit('checkpoint not found')
    manifest = json.loads(Path(a.manifest).read_text())
    if not a.allow_missing_t2l and any(len(manifest['tasks'][t].get('adapters', {})) != 3 for t in TASKS):
        raise SystemExit('Need 3 released T2L adapters per task; or explicitly --allow-missing-t2l for partial progress')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    identity = {'checkpoint_sha256': digest.hexdigest(),
                'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                'python': sys.executable,
                'inlet_commit': subprocess.check_output(
                    ['git', '-C', str(Path(__file__).resolve().parent.parent), 'rev-parse', 'HEAD'],
                    text=True).strip()}
    identity_path = out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise SystemExit('Checkpoint/manifest/code/runtime identity changed; use a new output root')
    if not identity_path.exists():
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    jobs = build_jobs(checkpoint, manifest, out)
    plan = out/'plan.json'
    encoded = json.dumps(jobs, indent=2) + '\n'
    if plan.exists() and plan.read_text() != encoded:
        raise SystemExit('Existing plan differs. Use a new output root.')
    if not plan.exists():
        plan.write_text(encoded)
    if a.execute:
        import torch
        config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        if (config.get('curstep') != 16000 or config.get('desc_slots') != 8
                or config.get('cond') != 'cross'):
            raise SystemExit(f'Expected Run1 16000/8/cross; found {config}')
    for job in jobs:
        if a.only_task and a.only_task != job['task']:
            continue
        print(job['id'], flush=True)
        if not a.execute:
            continue
        folder = out/job['id']
        folder.mkdir(parents=True, exist_ok=True)
        done = folder/'done.json'
        if done.exists():
            continue
        if any(folder.glob('*.json')) or any(folder.rglob('*.jsonl')):
            raise SystemExit(f'Partial artifacts in {folder}; inspect and move this whole directory before retry')
        with (folder/'process.log').open('w') as log:
            proc = subprocess.run(job['argv'], stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise SystemExit(f"{job['id']} exited {proc.returncode}; inspect process.log. No automatic success on teardown errors.")
        outputs = list(folder.glob('*.json'))
        if len(outputs) != 1:
            raise SystemExit(f'Expected one metric JSON in {folder}, found {outputs}')
        data = json.loads(outputs[0].read_text())
        if job['task'] not in data.get('results', {}):
            raise SystemExit('missing task metrics')
        done.write_text(json.dumps({'result': str(outputs[0]), 'job': job}, indent=2) + '\n')


if __name__ == '__main__':
    main()
