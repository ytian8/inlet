"""The training-side and eval-side prompt assemblies must be the same operation.

    python -m inlet.test_train_eval_agree

CPU, under a second, no GPU, no model weights, no vLLM, no dataset, and no
`hyper_llm_modulator`. Run it on a laptop.

WHY THIS EXISTS
---------------
Inlet prepends m soft-prompt vectors at the input-embedding layer, and it does so
twice, in two independently written implementations:

    training   torch.cat([P, E], dim=1)   on [B, m+T, d], plus mask and labels
    eval       torch.cat([P, e], dim=0)   on [m+T, d],    one request at a time

If they disagree -- wrong axis, reversed order, a dtype cast on one side only,
the mask or the labels extended by the wrong amount -- then the prompt that was
learned in one context is scored in another. **Nothing raises.** The run
completes, the loss falls, the checkpoint loads, and every number in the paper
is quietly wrong.

The repo already advertised a gate for this. It did not do it:

  * `test_consistency_inlet.py` compares `build_prompted_inputs` against the
    baseline's `prepend_soft_prompt` -- but BOTH are training-side, dim=1,
    batched. It never touches the eval path.
  * `gate_m0` is structurally blind to it: at m=0 both paths short-circuit and
    agree trivially.
  * `--zero-prompt` passes a 0-row tensor, so the eval-side `torch.cat` never
    executes at all.

So before this file, changing `dim=0` to `dim=1` in the eval path left every
gate in the repo green.

Every check below carries a deliberately-broken control, and the file fails if a
control is NOT detected. A test that cannot fail proves nothing.
"""

import sys

import torch

from inlet.sequence import build_eval_sequence, build_train_sequence

B, T, D, M = 3, 7, 5, 4
TOL = 0.0  # these are copies of the same numbers; anything but exact is a bug


def _fixtures(dtype=torch.float32, prompt_dtype=None, m=M):
    g = torch.Generator().manual_seed(0)
    tok = torch.randn(B, T, D, generator=g).to(dtype)
    prompt = torch.randn(B, m, D, generator=g).to(prompt_dtype or dtype)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, -2:] = 0  # a padded row, so mask handling is actually exercised
    labels = torch.arange(B * T, dtype=torch.long).reshape(B, T)
    labels[:, 0] = -100  # the collator masks the prompt region of the real data
    return tok, prompt, mask, labels


def _fail(msg):
    print(f"FAIL  {msg}")
    return 1


def check_sequences_match():
    """The core claim: row b of the training batch == the eval sequence for row b."""
    tok, prompt, mask, labels = _fixtures()
    embeds, _, _ = build_train_sequence(tok, mask, labels, prompt)
    bad = 0
    for b in range(B):
        want = build_eval_sequence(prompt[b], tok[b])
        got = embeds[b]
        if got.shape != want.shape:
            bad += _fail(f"row {b}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        elif not torch.equal(got, want):
            bad += _fail(f"row {b}: max abs diff {(got - want).abs().max().item():.3e}")
    if not bad:
        print(f"ok    train[b] == eval(prompt[b], tok[b]) for all {B} rows, "
              f"shape {tuple(embeds.shape)}")
    return bad


def check_prompt_comes_first():
    """Order, not just contents. A reversed cat has the same shape and norm."""
    tok, prompt, mask, labels = _fixtures()
    embeds, _, _ = build_train_sequence(tok, mask, labels, prompt)
    if not torch.equal(embeds[:, :M], prompt):
        return _fail("training: the first m rows are not the prompt")
    if not torch.equal(embeds[:, M:], tok):
        return _fail("training: the rows after m are not the tokens")
    ev = build_eval_sequence(prompt[0], tok[0])
    if not torch.equal(ev[:M], prompt[0]) or not torch.equal(ev[M:], tok[0]):
        return _fail("eval: prompt is not first")
    print("ok    prompt occupies [0, m) and tokens [m, m+T) on both paths")
    return 0


def check_mask_and_labels():
    tok, prompt, mask, labels = _fixtures()
    _, am, lb = build_train_sequence(tok, mask, labels, prompt)
    bad = 0
    if am.shape != (B, M + T):
        bad += _fail(f"attention_mask shape {tuple(am.shape)} != {(B, M + T)}")
    elif not torch.equal(am[:, :M], torch.ones(B, M, dtype=mask.dtype)):
        bad += _fail("virtual tokens are not attended to")
    elif not torch.equal(am[:, M:], mask):
        bad += _fail("the original attention mask was altered")
    if lb.shape != (B, M + T):
        bad += _fail(f"labels shape {tuple(lb.shape)} != {(B, M + T)}")
    elif not torch.equal(lb[:, :M], torch.full((B, M), -100, dtype=labels.dtype)):
        bad += _fail("virtual tokens are NOT masked out of the loss (-100)")
    elif not torch.equal(lb[:, M:], labels):
        bad += _fail("the original labels were altered")
    if not bad:
        print("ok    mask extended with ones, labels with -100, originals intact")
    return bad


def check_supervised_targets_unchanged():
    """The set of predicted tokens must not move when the prompt is prepended.

    This is the property that makes the m=0 gate meaningful and keeps
    `equally_weight_sample` normalisation comparable across m.
    """
    tok, prompt, mask, labels = _fixtures()
    _, _, lb = build_train_sequence(tok, mask, labels, prompt)
    # standard causal shift: logits[..., :-1] predict labels[..., 1:]
    before = labels[:, 1:][labels[:, 1:] != -100]
    after = lb[:, 1:][lb[:, 1:] != -100]
    if not torch.equal(before, after):
        return _fail(f"supervised targets changed: {before.numel()} -> {after.numel()}")
    print(f"ok    supervised target set unchanged under the shift "
          f"({before.numel()} tokens)")
    return 0


def check_m_zero_is_identity():
    tok, _, mask, labels = _fixtures()
    empty = torch.zeros(B, 0, D)
    e, am, lb = build_train_sequence(tok, mask, labels, empty)
    bad = 0
    if not (torch.equal(e, tok) and torch.equal(am, mask) and torch.equal(lb, labels)):
        bad += _fail("training: m=0 did not reduce to the plain inputs")
    for sp in (None, torch.zeros(0, D)):
        if not torch.equal(build_eval_sequence(sp, tok[0]), tok[0]):
            bad += _fail(f"eval: m=0 ({type(sp).__name__}) did not reduce to the tokens")
    if not bad:
        print("ok    m=0 is the identity on both paths (None and 0-row)")
    return bad


def check_dtype_follows_tokens():
    """The prompt is fp32; the model runs bf16. Both paths must cast the same way."""
    tok, prompt, mask, labels = _fixtures(dtype=torch.bfloat16, prompt_dtype=torch.float32)
    embeds, _, _ = build_train_sequence(tok, mask, labels, prompt)
    ev = build_eval_sequence(prompt[0], tok[0])
    bad = 0
    if embeds.dtype is not torch.bfloat16:
        bad += _fail(f"training result dtype {embeds.dtype}, expected bfloat16")
    if ev.dtype is not torch.bfloat16:
        bad += _fail(f"eval result dtype {ev.dtype}, expected bfloat16")
    if not bad and not torch.equal(embeds[0], ev):
        bad += _fail("the two paths round the fp32 prompt to bf16 differently")
    if not bad:
        print("ok    both paths cast the fp32 prompt to the token dtype identically")
    return bad


def check_shared_prompt_broadcasts():
    """[m, d] must behave exactly like [B, m, d] with the row repeated."""
    tok, _, mask, labels = _fixtures()
    g = torch.Generator().manual_seed(7)
    shared = torch.randn(M, D, generator=g)
    e_shared, _, _ = build_train_sequence(tok, mask, labels, shared)
    e_expand, _, _ = build_train_sequence(
        tok, mask, labels, shared.unsqueeze(0).expand(B, -1, -1)
    )
    if not torch.equal(e_shared, e_expand):
        return _fail("[m, d] and [B, m, d] disagree")
    for b in range(B):
        if not torch.equal(e_shared[b], build_eval_sequence(shared, tok[b])):
            return _fail(f"[m, d] row {b} disagrees with the eval path")
    print("ok    [m, d] broadcasts to every row and matches the eval path")
    return 0


# --------------------------------------------------------------------------- #
# Controls. Each reproduces a plausible bug and MUST be caught, or the checks
# above are decoration.

def _controls():
    tok, prompt, mask, labels = _fixtures()
    embeds, am, lb = build_train_sequence(tok, mask, labels, prompt)
    good = build_eval_sequence(prompt[0], tok[0])

    # A wrong axis usually raises, but not when m == T or d happens to line up.
    # Build the m == T case explicitly so the axis check is real rather than
    # accidental.
    sq_p, sq_t = torch.randn(T, D), torch.randn(T, D)
    axis0 = build_eval_sequence(sq_p, sq_t)                 # [2T, d]  -- correct
    axis1 = torch.cat([sq_p, sq_t], dim=1)                  # [T, 2d]  -- wrong

    cases = {
        "cat on dim=1 does not produce the same tensor as dim=0 (m == T)":
            axis0.shape != axis1.shape or not torch.equal(axis0, axis1),
        "reversed order (tokens first) is detected":
            not torch.equal(good, torch.cat([tok[0], prompt[0]], dim=0)),
        "labels NOT masked on the prompt region is detected":
            not torch.equal(lb[:, :M], torch.zeros(B, M, dtype=labels.dtype)),
        "mask of zeros on the prompt region is detected":
            not torch.equal(am[:, :M], torch.zeros(B, M, dtype=mask.dtype)),
        "a one-off shift of the prompt block is detected":
            not torch.equal(embeds[:, :M], torch.roll(prompt, 1, dims=1)),
    }
    bad = 0
    for name, detected in cases.items():
        if not detected:
            bad += _fail(f"control not detected -- {name}")
    if not bad:
        print(f"ok    all {len(cases)} deliberately-broken controls are detected")
    return bad


def main() -> int:
    print("train-side [B, m+T, d] (dim 1)  vs  eval-side [m+T, d] (dim 0)")
    print(f"B={B} T={T} d={D} m={M}, tolerance {TOL:g} (exact)\n")
    bad = 0
    for fn in (
        check_sequences_match,
        check_prompt_comes_first,
        check_mask_and_labels,
        check_supervised_targets_unchanged,
        check_m_zero_is_identity,
        check_dtype_follows_tokens,
        check_shared_prompt_broadcasts,
        _controls,
    ):
        # A broken assembly usually raises before it can return a verdict -- a
        # wrong `dim` is a shape error, not a wrong number. Catch it here so the
        # run reports which check died instead of dumping a traceback and
        # skipping every check after it.
        try:
            bad += fn()
        except Exception as exc:  # noqa: BLE001 -- reporting, not handling
            bad += _fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
    print()
    if bad:
        print(f"FAILED  {bad} check(s). The two assemblies are NOT the same operation. "
              "Do not trust any score produced by this checkout.")
        return 1
    print("PASS  the training and eval assemblies are the same operation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
