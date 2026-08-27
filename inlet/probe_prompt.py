"""Measure the generated prompt vectors directly. No LLM forward, no vLLM, no GPU needed.

Answers, without any benchmark eval:

    Is the description path doing anything at all, and how big is it next to
    the task-agnostic `base` prompt?

Because HyperPrompt.forward is

    P(desc) = base + head(enc(desc)) * emb_rms

the head's whole contribution is exactly `P(desc) - base`, which this file
isolates and measures. Three numbers matter:

  * |head| / |base|   -- if this is 1e-2 or smaller the head is decoration.
  * cos between head outputs for DIFFERENT real descriptions -- if ~1.0 the
    head emits one constant vector and ignores its input (collapse), which is
    invisible to any single-task score.
  * |head(real)| vs |head(junk)| -- if equal, the head reacts to "some text"
    rather than to what the text says.

    python -m inlet.probe_prompt --checkpoint train_outputs/.../hypermod_inlet.pt
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap, user_path  # noqa: E402

bootstrap()

import torch  # noqa: E402
import yaml  # noqa: E402

from inlet.checkpoint import load_inlet_checkpoint, load_description_encoder  # noqa: E402


def per_token_norm(x):
    """x: [m, d] -> mean over the m tokens of the per-token L2 norm."""
    return x.float().norm(dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--config", default="configs/hyper_lora_decontam_lol_tasks.yaml")
    ap.add_argument("--out", default=None, help="write the numbers as json here too")
    args = ap.parse_args()
    # bootstrap() has already chdir'd into the upstream checkout, so a relative
    # path the user typed no longer points where they meant it to.
    args.checkpoint = user_path(args.checkpoint)
    args.out = user_path(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyper, cfg = load_inlet_checkpoint(args.checkpoint, device)
    # whole cfg, not cfg["emb_model"]: the description must be pooled at the
    # checkpoint's own desc_slots, or "P(desc) - base" is measured against a
    # description encoded in a layout this checkpoint never saw.
    emb_model, emb_tok, fmt_fn, pool_fn = load_description_encoder(cfg, device)
    from hyper_llm_modulator.utils import embed_texts

    y = yaml.safe_load(open(args.config))
    real = list(y["eval_ds_info"][args.task]["descriptions"])[:3]
    junk = list(y["additional_eval_descs"])[:3]

    base = hyper.base.detach().float()                      # [m, d]
    emb_rms = float(hyper.emb_rms)

    def head_of(desc):
        te = embed_texts([desc], emb_model, emb_tok, fmt_fn, pool_fn, device)
        p = hyper(te)[0].detach().float()                   # [m, d] = base + head*rms
        return p - base.to(p.device)                        # the head's whole contribution

    heads = {f"real_{i}": head_of(d) for i, d in enumerate(real)}
    heads.update({f"junk_{i}": head_of(d) for i, d in enumerate(junk)})

    nb = per_token_norm(base)
    res = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "curstep": cfg.get("curstep"),
        "m": hyper.n_virtual_tokens,
        "d": hyper.model_dim,
        "emb_rms": emb_rms,
        "base_per_token_norm": nb,
        "base_total_norm": base.norm().item(),
        "head": {},
    }
    for k, h in heads.items():
        nh = per_token_norm(h)
        res["head"][k] = {"per_token_norm": nh, "ratio_to_base": nh / max(nb, 1e-12)}

    # Collapse check: cosine between the head outputs of different descriptions,
    # flattened over all m*d coordinates. ~1.0 means one constant vector.
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

    res["cos_real_real"] = {f"{a}|{b}": cos(heads[a], heads[b])
                            for a, b in itertools.combinations([f"real_{i}" for i in range(len(real))], 2)}
    res["cos_junk_junk"] = {f"{a}|{b}": cos(heads[a], heads[b])
                            for a, b in itertools.combinations([f"junk_{i}" for i in range(len(junk))], 2)}
    res["cos_real_junk"] = {f"real_{i}|junk_{j}": cos(heads[f"real_{i}"], heads[f"junk_{j}"])
                            for i in range(len(real)) for j in range(len(junk))}
    # How much of the head output is the SAME for every description (the
    # constant bias B) vs how much varies with the description (C)?
    stack = torch.stack(list(heads.values()))
    mean = stack.mean(0)
    res["head_mean_per_token_norm"] = per_token_norm(mean)
    res["head_resid_per_token_norm"] = per_token_norm((stack - mean).reshape(-1, stack.shape[-1]))
    res["varying_fraction"] = res["head_resid_per_token_norm"] / max(res["head_mean_per_token_norm"], 1e-12)

    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
