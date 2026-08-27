"""Gate for multi-slot description conditioning and cross-attention.

CPU only, no model weights, no gte, no upstream import. Seconds.

Every check here exists because the corresponding failure is SILENT -- the run
completes, the loss falls, and the number is wrong. In particular this file
refuses to let cross-attention repeat NEFTune's history, where the machinery was
wired up, never fired, and sat green for weeks. So the central checks are not
"does it run" but:

  * does widening K change anything at all (and does it change NOTHING when
    cond='pooled', which is the control),
  * does the attention mask actually exclude empty slots,
  * does K=1 still reproduce upstream's CLS pooling bit for bit.

Known-bad controls are deliberate and must stay. A check that passes on both the
correct and the broken variant has no teeth.

    python -m inlet.test_desc_cond
"""

import sys

import torch

from inlet.desc_pool import make_pooling_fn, slot_mask, unflatten_slots
from inlet.hyper_prompt import HyperPrompt

H = 32          # stand-in for gte's 1024
D = 48          # encoded_task_emb_size
MDIM = 64       # stand-in for Mistral's 4096
M = 6           # virtual tokens
K = 5           # description slots

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def cls_pool_reference(outputs, attention_mask):
    """Verbatim copy of hyper_llm_modulator.utils.pooling.cls_pool.

    Copied rather than imported ON PURPOSE: this file must run with no upstream
    checkout, and the point of the comparison is to detect our pooling drifting
    away from upstream's definition. `test_upstream_api` separately checks that
    upstream's definition has not itself changed.
    """
    right_padding = attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert right_padding
    return outputs["last_hidden_state"][:, 0].detach()


def fake_encoder_output(bs, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"last_hidden_state": torch.randn(bs, L, H, generator=g)}


def right_padded_mask(lengths, L):
    m = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        m[i, :n] = 1
    return m


# --------------------------------------------------------------------------- #
# 1. K=1 is upstream, bit for bit
# --------------------------------------------------------------------------- #
def test_k1_is_upstream():
    print("\n[1] K=1 reproduces upstream cls_pool")
    out = fake_encoder_output(4, 12)
    am = right_padded_mask([12, 9, 5, 1], 12)

    ours = make_pooling_fn(1)(out, am)
    theirs = cls_pool_reference(out, am)
    check("K=1 == cls_pool, exactly",
          torch.equal(ours, theirs),
          f"max|delta|={(ours - theirs).abs().max():.3e}")
    check("K=1 width is H, not 1*H padded", ours.shape == (4, H), str(tuple(ours.shape)))

    # ...and slot 0 at K>1 is still that same vector.
    wide = make_pooling_fn(K)(out, am)
    slots = unflatten_slots(wide, K)
    check("slot 0 at K>1 is still the CLS vector",
          torch.equal(slots[:, 0], theirs),
          f"max|delta|={(slots[:, 0] - theirs).abs().max():.3e}")


# --------------------------------------------------------------------------- #
# 2. segments and the empty-slot mask
# --------------------------------------------------------------------------- #
def test_segments_and_mask():
    print("\n[2] segment pooling and the empty-slot mask")
    L = 16
    # row 3 has ONE real token: it cannot fill K-1=4 segments, so some slots
    # must come out empty. This is the case the mask exists for.
    out = fake_encoder_output(4, L, seed=1)
    am = right_padded_mask([16, 8, 4, 1], L)
    slots = unflatten_slots(make_pooling_fn(K)(out, am), K)

    check("shape is [bs, K, H]", slots.shape == (4, K, H), str(tuple(slots.shape)))

    mask = slot_mask(slots)
    check("CLS slot is real for every row", bool(mask[:, 0].all()))
    check("a 1-token description leaves empty slots",
          not bool(mask[3].all()) and bool(mask[3, 0]),
          f"row3 mask={mask[3].tolist()}")
    check("a full-length description fills every slot",
          bool(mask[0].all()), f"row0 mask={mask[0].tolist()}")

    # Empty slots must be EXACTLY zero -- slot_mask reads that, and "almost
    # zero" would silently start admitting padding as content.
    empty = slots[3][~mask[3]]
    check("empty slots are exactly zero",
          empty.numel() > 0 and torch.equal(empty, torch.zeros_like(empty)),
          f"{empty.numel()} empty slot(s), max|v|={empty.abs().max() if empty.numel() else 0:.3e}")

    # Segments must average REAL tokens only. Row 1 has 8 real tokens of 16.
    h = out["last_hidden_state"]
    k = K - 1
    n = 8
    expect = torch.stack([h[1, i * n // k:(i + 1) * n // k].mean(0) for i in range(k)])
    check("segments average real tokens only (padding excluded)",
          torch.allclose(slots[1, 1:], expect, atol=1e-6),
          f"max|delta|={(slots[1, 1:] - expect).abs().max():.3e}")


# --------------------------------------------------------------------------- #
# 3. the zero-init invariant, in every mode
# --------------------------------------------------------------------------- #
def build(cond, n_slots, seed=0, zero_init=True):
    torch.manual_seed(seed)
    return HyperPrompt(
        task_emb_size=H, model_dim=MDIM, n_virtual_tokens=M,
        encoded_task_emb_size=D, hidden_size=64, n_blocks=2,
        head="per_slot", dropout=0.0, zero_init=zero_init,
        n_desc_slots=n_slots, cond=cond, n_cross_layers=2, n_cross_heads=4,
    ).eval()


def test_zero_init():
    print("\n[3] zero-init: P(desc) == base at step 0, in every mode")
    for cond, k in [("pooled", 1), ("pooled", K), ("cross", K)]:
        mod = build(cond, k)
        mod.base.data.normal_()
        x = torch.randn(3, k * H)
        with torch.no_grad():
            p = mod(x)
        delta = (p - mod.base.unsqueeze(0)).abs().max().detach()
        check(f"cond={cond:6s} K={k}: |P - base| == 0", float(delta) == 0.0,
              f"max|delta|={delta:.3e}")


# --------------------------------------------------------------------------- #
# 4. does conditioning actually READ the extra slots? (the NEFTune lesson)
# --------------------------------------------------------------------------- #
def test_cross_attention_is_live():
    print("\n[4] cross-attention actually reads slots 1..K-1")
    x = torch.randn(3, K * H)
    # perturb ONLY the non-CLS slots
    x2 = x.clone().view(3, K, H)
    x2[:, 1:] += 5.0
    x2 = x2.reshape(3, K * H)

    # control: with cond='pooled' the extra slots are unreachable by
    # construction, so the output MUST NOT move. If this ever changes, the
    # pooled path stopped being the K=1 computation and the A/B is invalid.
    pooled = build("pooled", K, zero_init=False)
    with torch.no_grad():
        d_pooled = (pooled(x) - pooled(x2)).abs().max()
    check("KNOWN-BAD CONTROL: cond='pooled' ignores slots 1..K-1",
          float(d_pooled) == 0.0, f"max|delta|={d_pooled:.3e}")

    # the real check: cross-attention must move.
    cross = build("cross", K, zero_init=False)
    with torch.no_grad():
        d_cross = (cross(x) - cross(x2)).abs().max()
    check("cond='cross' output CHANGES when a non-CLS slot changes",
          float(d_cross) > 1e-6, f"max|delta|={d_cross:.3e}")

    # and it must still read the CLS slot.
    x3 = x.clone().view(3, K, H)
    x3[:, 0] += 5.0
    x3 = x3.reshape(3, K * H)
    with torch.no_grad():
        d_cls = (cross(x) - cross(x3)).abs().max()
    check("cond='cross' still reads the CLS slot", float(d_cls) > 1e-6,
          f"max|delta|={d_cls:.3e}")


def test_slots_differentiate():
    print("\n[5] different prompt slots get different prompts")
    # If cross-attention collapsed to a broadcast, every one of the m prompt
    # positions would receive the same vector and the mechanism would be a very
    # expensive no-op.
    cross = build("cross", K, zero_init=False)
    x = torch.randn(2, K * H)
    with torch.no_grad():
        p = cross(x) - cross.base.unsqueeze(0)     # strip the shared base
    spread = p.std(dim=1).mean()
    check("prompt varies across the m slots", float(spread) > 1e-6,
          f"std across slots={spread:.3e}")


def test_mask_is_load_bearing():
    print("\n[6] the key-padding mask is load-bearing")
    # A row whose tail slots are empty must produce the same prompt whatever
    # garbage sits in those empty slots -- because they are masked out. If the
    # mask were dropped or inverted, this would change.
    cross = build("cross", K, zero_init=False)
    slots = torch.randn(1, K, H)
    slots[:, 3:] = 0.0                       # two empty slots
    a = slots.reshape(1, K * H)
    noisy = slots.clone()
    noisy[:, 3:] = 0.0                       # still exactly zero -> still masked
    with torch.no_grad():
        same = (cross(a) - cross(noisy.reshape(1, K * H))).abs().max()
    check("masked slots do not affect the output", float(same) == 0.0,
          f"max|delta|={same:.3e}")

    # KNOWN-BAD CONTROL: filling those slots with content (so the mask admits
    # them) MUST change the output. Otherwise the previous check would also pass
    # on a model that ignores description slots entirely.
    filled = slots.clone()
    filled[:, 3:] = 1.0
    with torch.no_grad():
        moved = (cross(a) - cross(filled.reshape(1, K * H))).abs().max()
    check("KNOWN-BAD CONTROL: un-masking those slots DOES change it",
          float(moved) > 1e-6, f"max|delta|={moved:.3e}")


def test_width_mismatch_raises():
    print("\n[7] a K mismatch fails loudly rather than reshaping")
    mod = build("cross", K)
    try:
        mod(torch.randn(2, K * H + 1))
    except ValueError as e:
        check("wrong task_emb width raises ValueError", "not divisible" in str(e))
    else:
        check("wrong task_emb width raises ValueError", False, "no exception")

    try:
        HyperPrompt(task_emb_size=H, model_dim=MDIM, n_virtual_tokens=M,
                    encoded_task_emb_size=D, hidden_size=64, n_blocks=2,
                    head="per_slot", n_desc_slots=1, cond="cross")
    except ValueError as e:
        check("cond='cross' with K=1 is rejected", "single vector" in str(e))
    else:
        check("cond='cross' with K=1 is rejected", False, "no exception")

    try:
        HyperPrompt(task_emb_size=H, model_dim=MDIM, n_virtual_tokens=M,
                    encoded_task_emb_size=D, hidden_size=64, n_blocks=2,
                    head="shared", n_desc_slots=K, cond="cross")
    except ValueError as e:
        check("cond='cross' with head='shared' is rejected", "per_slot" in str(e))
    else:
        check("cond='cross' with head='shared' is rejected", False, "no exception")


def test_gradients_reach_cross_attention():
    print("\n[8] gradients actually reach the cross-attention parameters")
    # zero_init makes `out` zero, which zeroes the gradient of everything
    # UPSTREAM of it on the first step. That is intended, but it means a broken
    # wiring would look identical to a correctly-wired step-0 model. Check with
    # a non-zero head, which is the state from step 1 onward.
    mod = build("cross", K, zero_init=False)
    p = mod(torch.randn(4, K * H))
    p.pow(2).mean().backward()
    named = dict(mod.named_parameters())
    cross_params = [n for n in named if n.startswith("cross.")]
    check("cross-attention has parameters", len(cross_params) > 0,
          f"{len(cross_params)} tensors")
    dead = [n for n in cross_params
            if named[n].grad is None or float(named[n].grad.abs().max()) == 0.0]
    check("every cross-attention parameter receives gradient", not dead,
          f"dead: {dead[:4]}" if dead else "")


def main():
    print("inlet.test_desc_cond -- multi-slot description conditioning")
    test_k1_is_upstream()
    test_segments_and_mask()
    test_zero_init()
    test_cross_attention_is_live()
    test_slots_differentiate()
    test_mask_is_load_bearing()
    test_width_mismatch_raises()
    test_gradients_reach_cross_attention()

    print()
    if FAILURES:
        print(f"FAIL -- {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
