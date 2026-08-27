"""The m=0 gate: Inlet's injection point must reduce to the frozen model.

If prepending ZERO virtual tokens does not reproduce a plain forward pass to
floating-point noise, then build_prompted_inputs is assembling the sequence
wrong -- wrong axis, wrong mask, wrong label shift -- and every number Inlet ever
produces will be quietly wrong while the code runs without error.

This is the training-side twin of `eval_soft_prompt.py --no-soft-prompt`.
Synthetic batch on purpose: no dataset, no tokenizer templates, nothing that
could mask an indexing bug.

    PYTHONPATH=inlet:src:src/fishfarm python inlet/gate_m0.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap  # noqa: E402

bootstrap()

from hyper_llm_modulator.sft_trainer import compute_loss  # noqa: E402
from inlet.hyper_prompt import HyperPrompt  # noqa: E402
from inlet.loss import build_prompted_inputs, get_loss_batch_inlet  # noqa: E402

MODEL = os.environ.get("INLET_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


def main():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    print(f"loading {MODEL} (frozen, bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.get_input_embeddings().weight.shape[1]
    V = model.get_input_embeddings().weight.shape[0]
    print(f"model loaded: d={d} vocab={V}")

    bs, L, n_sup = 2, 24, 6
    ids = torch.randint(3, V - 1, (bs, L), device="cuda")
    labels = ids.clone()
    labels[:, :-n_sup] = -100  # supervision is a suffix, as upstream requires
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones(bs, L, dtype=torch.long, device="cuda"),
        "labels": labels,
    }
    kw = dict(equally_weight_sample=False, label_smoothing=0.0)

    # ---- reference: the frozen model, untouched -------------------------- #
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=batch["attention_mask"])
        l_ref = compute_loss(labels, ref.logits, **kw).item()

    # ---- Inlet path with m = 0 --------------------------------------------- #
    hp = HyperPrompt(task_emb_size=1024, model_dim=d, n_virtual_tokens=0).cuda()
    with torch.no_grad():
        zero = torch.zeros(bs, 0, d, device="cuda", dtype=torch.bfloat16)
        l_m0 = get_loss_batch_inlet(batch, model=model, hypermod=hp, override_prompt=zero, **kw)["sft_loss"].item()

    delta = abs(l_m0 - l_ref)
    print(f"\nreference loss (plain forward) : {l_ref:.10f}")
    print(f"Inlet loss, m=0                  : {l_m0:.10f}")
    print(f"|delta|                        : {delta:.3e}")
    gate = "PASS" if delta < 1e-4 else "FAIL"
    print(f"GATE m=0 reduces to plain forward -> {gate}")

    # ---- shapes and gradient flow at m = 8 -------------------------------- #
    hp8 = HyperPrompt(task_emb_size=1024, model_dim=d, n_virtual_tokens=8).cuda()
    hp8.fit_output_scale(model.get_input_embeddings().weight)
    hp8.init_base_from_vocab(model.get_input_embeddings().weight)
    batch["task_embs"] = torch.randn(bs, 1024, device="cuda")

    e, m, lab = build_prompted_inputs(model, batch, hp8(batch["task_embs"]))
    assert e.shape == (bs, 8 + L, d), e.shape
    assert m.shape == (bs, 8 + L) and int(m[:, :8].sum()) == bs * 8
    assert bool((lab[:, :8] == -100).all())
    print(f"\nshapes at m=8: embeds {tuple(e.shape)} mask {tuple(m.shape)} labels {tuple(lab.shape)}  OK")

    # Zero-init means the description path is DORMANT FOR EXACTLY ONE STEP:
    # for y = Wh + b with W = 0, dL/dW = dL/dy (x) h^T is nonzero, but
    # dL/dh = W^T dL/dy is exactly zero, so nothing upstream of `out` moves.
    # Once `out.weight` leaves zero, gradient reaches the whole trunk. Same
    # mechanism as LoRA's B=0 init. Asserting it rather than assuming it.
    opt = torch.optim.AdamW([p for p in hp8.parameters() if p.requires_grad], lr=1e-3)

    def step():
        opt.zero_grad()
        o = get_loss_batch_inlet(batch, model=model, hypermod=hp8, **kw)
        o["sft_loss"].backward()
        g = {n: (p.grad is not None and p.grad.abs().sum().item() > 0)
             for n, p in hp8.named_parameters() if p.requires_grad}
        opt.step()
        return o, g

    o1, g1 = step()
    got1 = sorted(k for k, v in g1.items() if v)
    print(f"\nstep 1  loss {o1['sft_loss'].item():.6f}  |P| {o1['prompt_norm'].item():.4f}  "
          f"std_across_batch {o1['prompt_std_across_batch'].item():.6f}")
    print(f"        gradient reached: {got1}")
    assert got1 == ["base", "out.bias", "out.weight"], got1
    assert o1["prompt_std_across_batch"].item() < 1e-6, "zero-init must ignore the description at step 1"

    o2, g2 = step()
    missing = sorted(k for k, v in g2.items() if not v)
    print(f"step 2  loss {o2['sft_loss'].item():.6f}  |P| {o2['prompt_norm'].item():.4f}  "
          f"std_across_batch {o2['prompt_std_across_batch'].item():.6f}")
    print(f"        params still without gradient: {missing}")
    assert not missing, f"description path never woke up: {missing}"
    assert o2["prompt_std_across_batch"].item() > 0, "prompts still identical across descriptions at step 2"
    print("GATE description path dormant 1 step, then fully connected -> PASS")

    print(f"\npeak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
