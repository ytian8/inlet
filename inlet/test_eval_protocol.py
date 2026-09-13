"""CPU tests: lost/doubled ICL, mutation visibility, raw capture, fixed queue."""
import json
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


if __name__ == '__main__':
    unittest.main()
