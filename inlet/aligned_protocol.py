"""Explicit definition removal for Run 2; no upstream edits or description rewrites."""
import copy
import hashlib
import json
from pathlib import Path
from string import Formatter

BENCHMARKS = frozenset('arc_challenge arc_easy boolq hellaswag openbookqa piqa winogrande gsm8k mbpp humaneval'.split())

def strip_definitions(metadata, require_change=False):
    result = copy.deepcopy(metadata)
    changed = []
    for name, md in result.items():
        tpl = md.get('user_prompt_template', '')
        fields = [f for _, f, _, _ in Formatter().parse(tpl) if f]
        if 'task_def' in fields:
            # Known upstream shape only. Unknown templates must be reviewed,
            # not silently sliced at the first blank line.
            if tpl.strip() != '{task_def}\n\n{problem}':
                raise ValueError(f'{name}: unsupported task definition template {tpl!r}')
            md['user_prompt_template'] = '{problem}'
            changed.append(name)
        for key in ('user_prompt_template', 'system_message', 'assistant_prefill', 'assistant_postfill'):
            if '{task_def' in md.get(key, ''):
                raise ValueError(f'{name}: task_def remains in {key}')
        if name.startswith('lol_') and name not in changed:
            raise ValueError(f'{name}: expected a removable definition; inspect upstream metadata')
        descs = md.get('descriptions', [])
        if not descs or any(not isinstance(d, str) or not d.strip() for d in descs):
            raise ValueError(f'{name}: empty or invalid generator descriptions')
    if require_change and not changed:
        raise ValueError('Definition removal changed zero templates')
    return result, changed


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n')


def export_input_audit(save_dir, args, original, metadata, val_metadata, tokenizer, loaders):
    """Read cached formatted data, showing rules beside unchanged descriptions.

    This is structural evidence, not an automated proof of semantic sufficiency.
    Samples are deterministic; no train sampler or RNG is advanced.
    """
    from hyper_llm_modulator.data import load_and_format_dataset
    root = Path(save_dir)
    write_json(root/'metadata_audit.json', {'train_original': original, 'train_effective': metadata,
               'validation_effective': val_metadata, 'semantic_sufficiency': 'requires agent review'})
    with (root/'input_audit.jsonl').open('w') as out:
        for split, mdset in [('train', metadata), ('validation', val_metadata)]:
            for task, md in mdset.items():
                ds = load_and_format_dataset(mdset, tokenizer, args.sft_mode, True, task, md['ds_kwargs'])
                for index in range(min(2, len(ds))):
                    row = ds[index]
                    record = {'split': split, 'task': task, 'row_index': index,
                              'task_definition_removed': row.get('task_def'),
                              'problem': row.get('problem'), 'rendered_llm_input': row.get('prompt'),
                              'supervised_response': row.get('response'),
                              'generator_descriptions': md['descriptions'][:args.n_descs_per_ds],
                              'description_count': min(len(md['descriptions']), args.n_descs_per_ds)}
                    if record['rendered_llm_input'] is None or not record['supervised_response']:
                        raise ValueError(f'{task}: expected nonempty completion-format audit fields')
                    out.write(json.dumps(record, ensure_ascii=False, default=str)+'\n')
    # Check label availability in the actual tokenized data, not only source text.
    coverage = {}
    for split, loader in loaders.items():
        missing, total = 0, 0
        for dataset in loader.dataset.datasets:
            for labels in dataset.tokenized_dataset['labels']:
                total += 1
                if not bool((labels != -100).any()):
                    missing += 1
        coverage[split] = {'rows': total, 'rows_without_target_tokens': missing}
    write_json(root/'label_coverage.json', coverage)
    if any(x['rows_without_target_tokens'] for x in coverage.values()):
        raise ValueError('Some tokenized rows have no supervised target; see label_coverage.json')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [root/'metadata_audit.json', root/'input_audit.jsonl', root/'label_coverage.json']}
    write_json(root/'audit_hashes.json', hashes)
