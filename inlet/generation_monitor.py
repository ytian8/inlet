"""Small fixed held-out generation monitor. Diagnostic only, not benchmark accuracy."""
import ast
import json
from pathlib import Path
import torch
from inlet.canary import split_context_and_target
from inlet.sequence import build_eval_sequence
from inlet.aligned_protocol import write_json


def build_monitor(loader, tokenizer, save_dir, max_samples=8):
    """Fixed short and long examples across val/unseen tasks, selected by target length.

    No benchmark tasks; no random sampling. Scan first 64 rows per task. The
    manifest reveals if this split cannot provide genuinely long/code targets.
    """
    candidates = []
    for task_index, ds in enumerate(loader.dataset.datasets):
        for index in range(min(64, len(ds))):
            row = ds[index]
            batch = loader.collate_fn([row])
            got = split_context_and_target(batch['input_ids'][0], batch['attention_mask'][0], batch['labels'][0])
            if got:
                context, target = got
                candidates.append((len(target), task_index, index, batch, context, target))
    long = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))
    short = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))
    chosen, used_tasks = [], set()
    # Half longest and half shortest, spread across tasks.
    for pool, count in [(long, (max_samples+1)//2), (short, max_samples//2)]:
        if count == 0: continue
        n = 0
        for item in pool:
            if item[1] in used_tasks: continue
            chosen.append(item); used_tasks.add(item[1]); n += 1
            if n == count: break
    if not chosen:
        raise ValueError('No usable held-out generation monitor examples')
    manifest = []
    for length, task, idx, batch, ctx, target in chosen:
        manifest.append({'task_index': task, 'row_index': idx, 'target_tokens': length,
                         'context': tokenizer.decode(ctx), 'target': tokenizer.decode(target)})
    write_json(Path(save_dir)/'generation_monitor_manifest.json', {
        'split': 'val/unseen', 'selection': 'first 64 rows/task; longest+shortest; distinct tasks',
        'examples': manifest, 'long_targets_ge64': sum(x['target_tokens']>=64 for x in manifest),
        'caveat': 'Not benchmark accuracy or a code test suite. Inspect coverage before training.'})
    return chosen


def generation_end(tokens, eos_token_id, budget):
    eos = set(eos_token_id if isinstance(eos_token_id, (list,tuple)) else [eos_token_id])
    if tokens and tokens[-1] in eos: return 'eos'
    if len(tokens) >= budget: return 'length_limit'
    return 'other'


@torch.no_grad()
def run_monitor(model, hypermod, tokenizer, examples, save_dir, step, max_new_tokens=256):
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    htrain, etrain = hypermod.training, emb.training
    hypermod.eval(); emb.eval()
    path = Path(save_dir)/'generation_monitor.jsonl'
    rows = []
    try:
        # Fork RNG so diagnostics cannot change the subsequent training stream.
        with torch.random.fork_rng(devices=[dev] if dev.type=='cuda' else []):
            for length, task, idx, batch, ctx, target in examples:
                sp = hypermod(batch['task_embs'].to(dev))[0]
                for arm in ('base', 'inlet'):
                    x = build_eval_sequence(sp if arm=='inlet' else None, emb(ctx.to(dev))).unsqueeze(0)
                    result = model.generate(inputs_embeds=x,
                        attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev),
                        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                        min_new_tokens=0, min_length=0, use_cache=True)
                    ids = result[0].tolist()
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    reference = tokenizer.decode(target, skip_special_tokens=True)
                    code_like = 'def ' in text or 'class ' in text
                    parse = None
                    if code_like:
                        try: ast.parse(text); parse = True
                        except SyntaxError: parse = False
                    rows.append({'step':step, 'arm':arm, 'task_index':task, 'row_index':idx,
                        'target_tokens':length, 'generated_tokens':len(ids), 'words':len(text.split()),
                        'end':generation_end(ids, model.generation_config.eos_token_id, max_new_tokens),
                        'output':text, 'target':reference, 'exact_match':text.strip()==reference.strip(),
                        'code_like':code_like, 'raw_python_parses':parse,
                        'note':'Raw parsing only; markdown not stripped; no code executed.'})
        with path.open('a') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    finally:
        hypermod.train(htrain); emb.train(etrain)
    return rows
