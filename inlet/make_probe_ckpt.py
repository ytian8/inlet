"""Build checkpoints whose prompt is chosen, not learned, so the prompt itself
can be varied while everything else is held fixed.

The question these answer: **how much of the damage is caused by prepending 32
vectors at all, and how much by what training put in them?**

Nothing else separates those. Every measurement so far compares a trained prompt
against no prompt, which confounds "there is a prompt" with "the prompt was
trained on 479 short-answer tasks". These make prompts with no training in them:

  vocab      32 real token embeddings sampled from the frequent vocab -- exactly
             `base` at initialisation, before a single gradient step. In
             distribution by construction.
  random     Gaussian noise scaled to the same norm. Off distribution in content
             but not in magnitude.
  scaled     the vocab prompt at 2.63x that norm -- the magnitude the 147,500-step
             run drifted to.

Read the three against zero-prompt (humaneval 39.63) and against a trained prompt
(20.33 measured, 21.54 at step 4,000):

  vocab ~= 39   prompts are harmless until trained; the damage is learned, and
                the fix is in the objective or the data mix.
  vocab ~= 20   prepending 32 vectors before a frozen model costs ~16 points on
                its own, and no amount of better conditioning recovers it. That
                is a ceiling on the whole approach and belongs in the paper.
  random << vocab   content matters, not just magnitude.
  scaled << vocab   magnitude matters; --l2_reg_prompt is the lever.

The head is zeroed, so P(desc) == base for every description and the checkpoint
loads through the normal eval path with no special casing.

    python -m inlet.make_probe_ckpt --like <trained.pt> --kind vocab --out vocab.pt
"""

import argparse
import sys

import torch


def build(like: str, kind: str, out: str, scale: float, seed: int) -> str:
    ck = torch.load(like, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["config"]

    base = sd["base"]                                  # [m, d]
    m, d = base.shape
    g = torch.Generator().manual_seed(seed)

    if kind == "vocab":
        # `base` at init is already vocab-sampled; reuse the recipe rather than
        # the trained tensor, so this carries no training at all.
        emb = ck.get("embedding_weight")
        if emb is None:
            raise SystemExit(
                "kind=vocab needs the input embedding matrix. Pass --embeddings "
                "<path to a .pt holding the [vocab, d] tensor>, or use "
                "--kind random, which needs only the norm."
            )
        hi = min(5000, emb.shape[0])
        ids = torch.randint(0, hi, (m,), generator=g)
        new = emb[ids].clone().float()
    elif kind == "random":
        new = torch.randn(m, d, generator=g)
        # match the PER-TOKEN norm of the reference, so magnitude is controlled
        # and only content differs.
        new = new / new.norm(dim=-1, keepdim=True) * base.norm(dim=-1, keepdim=True)
    elif kind == "keep":
        new = base.clone()
    else:
        raise SystemExit(f"unknown --kind {kind}")

    new = new * scale

    sd["base"] = new
    # Zero the head so P(desc) == base for every description. Without this the
    # trained head would still add its constant and the prompt would not be the
    # one named on the tin.
    for k in list(sd):
        if k.startswith("out.") and ("weight" in k or "bias" in k):
            sd[k] = torch.zeros_like(sd[k])

    ck["state_dict"] = sd
    ck["extra"] = dict(ck.get("extra") or {},
                       probe_kind=kind, probe_scale=scale, probe_seed=seed,
                       probe_note="synthetic prompt; head zeroed; NOT a trained model")
    cfg["curstep"] = -1        # so a sweep never mistakes it for a training step
    torch.save(ck, out)

    pn = new.norm(dim=-1).mean().item()
    bn = base.norm(dim=-1).mean().item()
    print(f"{out}\n  kind={kind} scale={scale} seed={seed}")
    print(f"  per-token norm {pn:.4f}   (source base {bn:.4f}, ratio {pn / bn:.2f}x)")
    print(f"  head zeroed: P(desc) == base for every description")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--like", required=True, help="a trained checkpoint to copy the shape/config from")
    p.add_argument("--kind", required=True, choices=["vocab", "random", "keep"])
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embeddings", default=None,
                   help="path to a .pt with the [vocab, d] input embedding matrix (kind=vocab)")
    a = p.parse_args(argv)

    if a.kind == "vocab" and a.embeddings:
        ck = torch.load(a.like, map_location="cpu", weights_only=False)
        ck["embedding_weight"] = torch.load(a.embeddings, map_location="cpu")
        tmp = a.out + ".withemb"
        torch.save(ck, tmp)
        a.like = tmp
    build(a.like, a.kind, a.out, a.scale, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
