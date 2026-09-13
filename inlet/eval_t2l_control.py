"""One released T2L adapter per process, with the same ICL audit as Inlet.

Pass a manifest produced by inlet.prepare_run1_diagnostic. The adapter must be
from the released Mistral SFT generator, never a benchmark-trained oracle.
"""
import argparse
import json
import os
from pathlib import Path

from inlet._env import bootstrap, user_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--descriptions', required=True)
    p.add_argument('--description-tag', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-dir', default='mistralai/Mistral-7B-Instruct-v0.2')
    p.add_argument('--use-icl', action='store_true')
    a = p.parse_args()
    a.adapter, a.out, a.descriptions = map(user_path, (a.adapter, a.out, a.descriptions))
    bootstrap(need_baseline=False)
    from inlet.eval_common import get_tokenizer
    from inlet.eval_protocol import audit_icl, evaluate_recorded
    import hyper_llm_modulator.vllm_eval as ve

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    cfg = json.loads((Path(a.adapter) / 'adapter_config.json').read_text())
    if cfg.get('use_rslora', False):
        raise SystemExit('Released Mistral SFT adapter requires use_rslora=false; verify provenance')
    desc = json.loads(Path(a.descriptions).read_text())[a.description_tag]
    native = ve.eval_model
    audit = {}

    def wrapped(model_dir, lora_dirs, chat_template, gpu_memory_utilization,
                evaluator, prefill_text='', per_sample_lora=False):
        audit.update(audit_icl(evaluator, a.task, a.use_icl, ve.IN_CONTEXT_EXAMPLES))
        audit['prefill_text'] = prefill_text
        # Proxy keeps the real evaluator unmodified and records before extraction.
        class Proxy:
            def __getattr__(self, key):
                return getattr(evaluator, key)
            def evaluate(self, model):
                return evaluate_recorded(evaluator, model, str(out) + '.raw.jsonl')
        return native(model_dir, lora_dirs, chat_template, gpu_memory_utilization,
                      Proxy(), prefill_text=prefill_text, per_sample_lora=False)
    ve.eval_model = wrapped
    result = ve.eval(a.model_dir, [a.adapter], a.task,
                     chat_template=get_tokenizer(a.model_dir).chat_template,
                     use_icl=a.use_icl)
    payload = {'task': a.task, 'method': 't2l', 'adapter': a.adapter,
               'model_dir': a.model_dir, 'use_icl': a.use_icl,
               'description_tag': a.description_tag, 'description': desc,
               'input_audit': audit,
               'sample_details': {k: getattr(v, 'sample_details', None) for k, v in result.items()},
               'results': {a.task: {k: v.aggregate_metrics for k, v in result.items()}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
