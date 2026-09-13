"""CPU tests: lost/doubled ICL, mutation visibility, raw capture, fixed queue."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from inlet.eval_protocol import audit_icl, evaluate_recorded
from inlet.run1_diagnostic import build_jobs, TASKS


class ProtocolTests(unittest.TestCase):
    def test_gsm8k_repair_and_idempotence(self):
        e = NS(samples=[NS(problem='question')])
        first = audit_icl(e, 'gsm8k', True, {'gsm8k': 'demo'})
        self.assertEqual(e.samples[0].problem, 'demo\n\nquestion')
        self.assertEqual(first['gsm8k_samples_repaired'], 1)
        self.assertEqual(audit_icl(e, 'gsm8k', True, {'gsm8k': 'demo'})['gsm8k_samples_repaired'], 0)

    def test_non_gsm_missing_icl_fails(self):
        with self.assertRaises(ValueError):
            audit_icl(NS(samples=[NS(question='question')]), 'arc_challenge', True, {'arc_challenge': 'demo'})

    def test_empty_humaneval_fails(self):
        with self.assertRaises(ValueError):
            audit_icl(NS(samples=[NS(instruction='q')]), 'humaneval', True, {'humaneval': ''})

    def test_zero_shot_is_unchanged(self):
        e = NS(samples=[NS(problem='question')])
        audit_icl(e, 'gsm8k', False, {'gsm8k': 'demo'})
        self.assertEqual(e.samples[0].problem, 'question')

    def test_actual_output_saved_and_no_overwrite(self):
        class Model:
            def _into_prompt(self, messages):
                return messages
            def generate(self, requests):
                for request in requests:
                    yield NS(request=request, generation='raw code\nreturn 1')
        class RawCaptureEvaluator:
            def evaluate(self, model):
                return list(model.generate([NS(messages='formatted input')]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'raw.jsonl'
            model = Model()
            original = model.generate
            evaluate_recorded(RawCaptureEvaluator(), model, path)
            data = json.loads(path.read_text())
            self.assertEqual(data['output'], 'raw code\nreturn 1')
            self.assertEqual(data['rendered_prompt'], 'formatted input')
            self.assertEqual(model.generate, original)
            with self.assertRaises(FileExistsError):
                evaluate_recorded(RawCaptureEvaluator(), model, path)

    def test_queue_keeps_scale_and_icl_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            desc = root/'desc.json'
            desc.write_text(json.dumps({f'eval_descs__{i}': str(i) for i in range(3)}))
            manifest = {'tasks': {t: {'descriptions': str(desc), 'adapters':
                        {f'eval_descs__{i}': f'/adapter/{t}/{i}' for i in range(3)}} for t in TASKS}}
            jobs = build_jobs('/checkpoint.pt', manifest, root)
            self.assertEqual(len(jobs), 72)
            self.assertEqual(len({j['id'] for j in jobs}), 72)
            for job in jobs:
                self.assertEqual('--use-icl' in job['argv'], job['icl'])
                if job['method'] == 'inlet':
                    i = job['argv'].index('--prompt-scales')
                    self.assertNotIn(',', job['argv'][i+1])
            self.assertEqual(sum(j['task']=='gsm8k' for j in jobs), 24)
            before = {p: p.stat().st_mtime_ns for p in (root/'descriptions').glob('*.json')}
            self.assertEqual(build_jobs('/checkpoint.pt', manifest, root), jobs)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def _launch_fake_gpus(self, fail_task=''):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stub = root/'python-stub'
            stub.write_text('#!' + sys.executable + '\n' +
                'import os, sys, json\n'
                'a = sys.argv\n'
                'task = a[a.index("--only-task")+1] if "--only-task" in a else "plan"\n'
                'with open(os.environ["TEST_LOG"], "a") as f: f.write(json.dumps([task, os.environ.get("CUDA_VISIBLE_DEVICES")])+"\\n")\n'
                'sys.exit(3 if task == os.environ.get("FAIL_TASK") else 0)\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYBIN=str(stub), CUDA_VISIBLE_DEVICES='GPU-A,GPU-B',
                       TEST_LOG=str(root/'log'), FAIL_TASK=fail_task)
            launcher = Path(__file__).resolve().parent.parent/'scripts/run1_diagnostic_2gpu.sh'
            result = subprocess.run(['bash', str(launcher), '--checkpoint', '/unused',
                                     '--manifest', '/unused', '--out', '/unused'],
                                    env=env, capture_output=True, text=True)
            rows = [json.loads(line) for line in (root/'log').read_text().splitlines()]
            return result, rows

    def test_two_gpu_assignment(self):
        result, rows = self._launch_fake_gpus()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0], ['plan', 'GPU-A,GPU-B'])
        self.assertEqual(dict(rows[1:]), {'arc_challenge':'GPU-A', 'gsm8k':'GPU-A',
                                         'winogrande':'GPU-B', 'mbpp':'GPU-B'})

    def test_two_gpu_worker_failure_propagates(self):
        result, rows = self._launch_fake_gpus('winogrande')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('mbpp', dict(rows))
        self.assertIn('gsm8k', dict(rows))



if __name__ == '__main__':
    unittest.main()
