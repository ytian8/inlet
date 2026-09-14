"""Freeze Run 2 configs from actual Run 1 args, without importing GPU dependencies."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import yaml

from inlet.aligned_protocol import write_json


def dataclass_fields(path, name):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def make_config(source, run_name):
    cfg = copy.deepcopy(source)
    for key, value in {'desc_slots':8, 'cond':'cross', 'n_virtual_tokens':32,
                       'model_dir':'mistralai/Mistral-7B-Instruct-v0.2',
                       'n_train_ds':479, 'n_points_per_task':1}.items():
        if cfg.get(key) != value: raise ValueError(f'Run 1 {key} must be {value!r}, got {cfg.get(key)!r}')
    for key in ('freeze_head','freeze_base','use_inp_as_desc','use_default_desc','use_per_sample_desc','use_one_hot_task_emb'):
        if cfg.get(key, False): raise ValueError(f'Unexpected Run 1 option {key}')
    for key in ('prompt_diversity','contrastive','l2_reg_prompt'):
        if cfg.get(key, 0) != 0: raise ValueError(f'Unexpected Run 1 regularizer {key}')
    if not cfg.get('use_per_task_emb') or cfg.get('sft_mode') != 'completion':
        raise ValueError('Run 2 requires description-conditioned completion SFT')
    if cfg.get('head_lr_mult',1.0) != 1.0: raise ValueError('Run 1 head_lr_mult must be 1')
    if cfg.get('global_tasks_per_step') != 64:
        raise ValueError('Expected Run 1 global_tasks_per_step=64; inspect actual batch budget')
    if len(cfg.get('train_ds_names', [])) < 479 or not cfg.get('eval_ds_info'):
        raise ValueError('Actual Run 1 task lists and validation metadata are required')
    if cfg.get('strip_taskdef_in_training', False):
        raise ValueError('Source already removes task definitions; not the expected Run 1 control')
    cfg.update(run_name=run_name, max_steps=32000, strip_taskdef_in_training=True,
               strip_taskdef_in_validation=True, exclude_benchmark_validation=True,
               generative_val_tasks='', model_select_split='val/unseen',
               global_tasks_per_step=64, checkpoint_steps='8000,16000,24000,32000',
               save_best_per_split=True, generation_monitor_samples=8,
               generation_monitor_freq=4000, generation_monitor_max_new_tokens=256,
               input_audit_only=False)
    if 64 % (2*cfg['n_tasks_per_batch']):
        raise ValueError('Run 1 microbatch does not divide global batch 64 on two GPUs')
    cfg['grad_accum_steps'] = 64//(2*cfg['n_tasks_per_batch'])
    return cfg


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    upstream = Path(args.t2l_root).resolve()
    src = Path(args.run1_args).resolve()
    source = yaml.safe_load(src.read_text())
    fields = dataclass_fields(root/'inlet/train_inlet.py','InletArguments')
    fields |= dataclass_fields(upstream/'src/hyper_llm_modulator/configs.py','TrainingArguments')
    # args.yaml includes derived runtime fields that upstream's parser rejects.
    excluded = {k:v for k,v in source.items() if k not in fields}
    allowed_derived = {'save_dir','device','world_size','num_processes','local_rank','rank'}
    unexpected = set(excluded)-allowed_derived
    if unexpected: raise ValueError(f'Unrecognized Run 1 fields: {sorted(unexpected)}; review, do not silently drop')
    clean = {k:v for k,v in source.items() if k in fields}
    cfg = make_config(clean,args.run_name)
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'run1_args_source.yaml').write_bytes(src.read_bytes())
    stages = {'train':cfg, 'audit':dict(cfg,run_name=args.run_name+'_audit',input_audit_only=True),
              'smoke':dict(cfg,run_name=args.run_name+'_smoke',max_steps=8,
                          checkpoint_steps='8',val_freq=8,generation_monitor_freq=8)}
    hashes = {}
    for stage, data in stages.items():
        path = destination/f'{stage}.yaml'
        path.write_text(yaml.safe_dump(data,sort_keys=False))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    code_files = sorted((root/'inlet').glob('*.py'))
    code_files += sorted((upstream/'src/hyper_llm_modulator').rglob('*.py'))
    code_files += [root/'scripts/common.sh',root/'scripts/run2_aligned_2gpu.sh']
    task_names = set(cfg['train_ds_names'][:cfg['n_train_ds']]) | set(cfg['eval_ds_info'])
    code_files += [upstream/'tasks'/name/'metadata.yaml' for name in sorted(task_names)]
    write_json(destination/'provenance.json', {
        'run1_args_path':str(src),'run1_args_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'inlet_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'inlet_root':str(root),'upstream_root':str(upstream),'omitted_runtime_fields':excluded,
        'changes':{k:{'before':clean.get(k),'after':v} for k,v in cfg.items() if clean.get(k)!=v},
        'config_sha256':hashes,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
        'environment':{k:os.environ.get(k) for k in ['INLET_NO_FLASH_ATTN','HF_HOME','HF_DATASETS_OFFLINE','CUDA_VISIBLE_DEVICES']},
        'note':'Fresh initialization; no checkpoint loading. Review source/config changes before launch.'})
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run1-args',required=True)
    parser.add_argument('--t2l-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--run-name',default='run2_aligned_32k')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name): parser.error('Use a simple run name')
    prepare(args)

if __name__=='__main__': main()
