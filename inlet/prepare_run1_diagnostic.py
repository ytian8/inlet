"""Freeze description controls; optionally generate only 12 released T2L adapters.

No training. Generation uses the SAME real descriptions as Inlet. Corrected
LoRA scaling is applied in memory, leaving the source checkpoint unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

DONORS = {'arc_challenge': 'openbookqa', 'winogrande': 'boolq',
          'gsm8k': 'mbpp', 'mbpp': 'gsm8k'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--t2l-checkpoint', default=None)
    a = p.parse_args()
    root = Path(a.out).resolve()
    config = Path(a.config).resolve() if a.config else None
    checkpoint = str(Path(a.t2l_checkpoint).resolve()) if a.t2l_checkpoint else None
    from inlet._env import bootstrap
    upstream = bootstrap(need_baseline=False)
    import yaml
    config = config or Path(upstream) / 'configs/hyper_lora_decontam_lol_tasks.yaml'
    cfg = yaml.safe_load(config.read_text())
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for task, donor in DONORS.items():
        real = cfg['eval_ds_info'][task]['descriptions'][:3]
        wrong = cfg['eval_ds_info'][donor]['descriptions'][:3]
        junk = cfg['additional_eval_descs'][:3]
        if any(len(x) != 3 for x in (real, wrong, junk)):
            raise ValueError('three descriptions per group required')
        groups = {'eval_descs': real, 'mismatch_descs': wrong, 'random_descs': junk}
        descs = {f'{group}__{i}': d for group, ds in groups.items() for i, d in enumerate(ds)}
        if any(not isinstance(d, str) or not d.strip() for d in descs.values()):
            raise ValueError('empty description')
        if set(real) & (set(wrong) | set(junk)):
            raise ValueError('matched and control descriptions overlap')
        path = root / f'{task}.json'
        encoded = json.dumps(descs, indent=2, ensure_ascii=False) + '\n'
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f'{path} differs; use a fresh directory')
        path.write_text(encoded)
        manifests[task] = {'donor': donor, 'descriptions': str(path),
                           'sha256': hashlib.sha256(encoded.encode()).hexdigest()}
    manifest = {'config': str(config), 'tasks': manifests, 't2l_checkpoint': checkpoint}
    if checkpoint:
        import torch
        from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
        from hyper_llm_modulator.utils import get_layers
        from hyper_llm_modulator.utils.eval_hypermod import gen_and_save_lora
        args, hypermod, model, tokenizer, enc, enc_tok, fmt, pool = load_hypermod_checkpoint(checkpoint, 'cuda')
        if args.model_dir != 'mistralai/Mistral-7B-Instruct-v0.2':
            raise ValueError(f'Wrong T2L backbone: {args.model_dir}')
        if hypermod.training_task != 'sft':
            raise ValueError(f'Expected SFT T2L, got {hypermod.training_task}')
        hypermod.peft_config.use_rslora = False
        indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device='cuda')
        for task, entry in manifests.items():
            descs = json.loads(Path(entry['descriptions']).read_text())
            entry['adapters'] = {}
            for i in range(3):
                tag = f'eval_descs__{i}'
                dest = root / 'adapters' / task / tag
                if dest.exists():
                    raise ValueError(f'{dest} exists; use a fresh directory to avoid stale adapters')
                gen_and_save_lora(args.model_dir, 'cuda', indices, enc, enc_tok,
                                  fmt, pool, hypermod, str(dest), descs[tag])
                entry['adapters'][tag] = str(dest)
        manifest['scaling'] = 'released Mistral SFT: use_rslora=false, set in memory before saving'
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
