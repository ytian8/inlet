"""The generative canary: free-running generation next to teacher forcing.

Why this file exists. The 147,500-step run reported `val/benchmark`
`per_token_acc` between 0.80 and 0.82 for its entire length -- a completely
healthy-looking curve -- while its humaneval score was 4.47 against a frozen
model's 37.80. Nothing in validation could see that, because `per_token_acc` is
**teacher forced**: it asks "given the correct prefix, is the next token right?"
and the correct prefix is handed back after every single step. A prompt can
destroy a model's ability to generate 200 coherent tokens on its own and barely
move that number.

So the run trained for over 100,000 steps past the point where its benchmark
loss had started degrading, and the only instrument that would have said so was
an eval nobody runs mid-training because it needs vLLM.

This measures the same batch **both** ways:

  per_token_acc        teacher forced, the existing number
  free_running_acc     generate from the context alone, no peeking

On a healthy model the two track each other. When the prompt starts breaking
generation, teacher forcing holds up and free running collapses, and **the gap
is the signal**. It costs a handful of short greedy generations per validation,
needs no vLLM, no new dataset, and no task-specific answer parsing -- it reuses
the validation batches that are already loaded.

It is a canary, not a benchmark. Do not report `free_running_acc` as a score;
report the eval harness. Use it to find out, at step 20,000 rather than step
147,500, that something has gone wrong.
"""

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["split_context_and_target", "free_running_accuracy"]


def split_context_and_target(input_ids, attention_mask, labels):
    """One padded row -> (context_ids, target_ids), or None if unusable.

    Pure tensor algebra, no model, so `inlet.test_desc_cond` can check it.

    The collator right-pads, and marks everything that is not a supervised
    target with -100. The context is what the model is allowed to see; the
    target is what it has to produce on its own.

      input_ids       [t0 t1 t2 t3 t4 PAD PAD]
      attention_mask  [ 1  1  1  1  1   0   0]
      labels          [-100 -100 -100 t3 t4 -100 -100]
                                      ^^^^^ supervised
      -> context = [t0 t1 t2],  target = [t3 t4]

    Returns None when there is no supervised token, or nothing to condition on:
    generating from an empty context measures the prompt against a void, which
    is not what this is for.
    """
    real = int(attention_mask.sum())
    if real <= 1:
        return None
    ids, labs = input_ids[:real], labels[:real]

    sup = (labs != -100).nonzero(as_tuple=True)[0]
    if sup.numel() == 0:
        return None
    start = int(sup[0])
    if start == 0:
        # Supervised from the first position: there is no context to generate
        # from. Rare, but it would otherwise produce a meaningless comparison.
        return None

    target = labs[start:]
    target = target[target != -100]
    if target.numel() == 0:
        return None
    return ids[:start], target


@torch.no_grad()
def free_running_accuracy(model, batch, soft_prompts, max_samples=4,
                          max_new_tokens=64):
    """-> (accuracy, n_tokens_compared, n_samples_used).

    Greedy generation from the context alone, scored against the supervised
    target.

    `soft_prompts` is [B, m, d] -- ONE PROMPT PER SAMPLE, because that is what
    Inlet actually does at inference: sample b carries task b's description and
    must be scored under task b's prompt. Passing a single [m, d] and sharing it
    across the batch would measure a model nobody runs. [m, d] and None are
    still accepted, for the zero-prompt and shared-prompt controls.

    Prepended exactly as `inlet.sequence.build_eval_sequence` does -- the same
    assembly the real eval uses, so this cannot drift into measuring a different
    model than the one being scored.

    Accuracy is over min(len(target), max_new_tokens) positions. Truncation is
    deliberate: the point is whether generation holds together, and the first
    tokens carry that. Samples whose context is empty or which have no
    supervised target are skipped, not counted.
    """
    from inlet.sequence import build_eval_sequence

    emb = model.get_input_embeddings()
    device = next(emb.parameters()).device
    hits = total = used = 0

    for b in range(min(max_samples, batch["input_ids"].shape[0])):
        got = split_context_and_target(
            batch["input_ids"][b], batch["attention_mask"][b], batch["labels"][b]
        )
        if got is None:
            continue
        ctx_ids, target = got
        n = min(int(target.numel()), max_new_tokens)
        if n == 0:
            continue

        ctx_ids = ctx_ids.to(device)
        if soft_prompts is None:
            sp = None
        elif soft_prompts.dim() == 3:
            sp = soft_prompts[b].to(device)      # this sample's own prompt
        else:
            sp = soft_prompts.to(device)
        inputs_embeds = build_eval_sequence(sp, emb(ctx_ids)).unsqueeze(0)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=n,
            min_new_tokens=n,          # do not stop early: we compare n positions
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        # With inputs_embeds the returned sequence is the NEW tokens only.
        gen = out[0][-n:].to(target.device)
        hits += int((gen == target[:n]).sum())
        total += n
        used += 1

    return (hits / total if total else float("nan")), total, used


# --------------------------------------------------------------------------- #
# Self-test for the tensor algebra. No model, no GPU, under a second:
#     python -m inlet.canary
# The generation half needs a model and is exercised by smoke.sh.
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    L = 7
    def row(ids, mask, labs):
        return (torch.tensor(ids), torch.tensor(mask), torch.tensor(labs))

    print("inlet.canary -- context/target split")

    # the ordinary case: 5 real tokens, last 2 supervised, 2 pads
    ids, mask, labs = row([10, 11, 12, 13, 14, 0, 0],
                          [1, 1, 1, 1, 1, 0, 0],
                          [-100, -100, -100, 13, 14, -100, -100])
    got = split_context_and_target(ids, mask, labs)
    check("context stops where supervision starts",
          got is not None and got[0].tolist() == [10, 11, 12], str(got and got[0].tolist()))
    check("target is the supervised span only",
          got is not None and got[1].tolist() == [13, 14], str(got and got[1].tolist()))

    # padding must never reach either side -- this is the one that would silently
    # score the model on predicting PAD, which it does very well.
    check("padding is excluded from the target",
          got is not None and 0 not in got[1].tolist())
    check("padding is excluded from the context",
          got is not None and len(got[0]) == 3)

    # KNOWN-BAD CONTROL, and it has to be a case where the mask ACTUALLY changes
    # the answer -- a control that agrees with the correct result by construction
    # is not a control. Here a pad position is wrongly labelled as supervised, so
    # a split that trusts `labels` and skips `attention_mask` scores the model on
    # predicting PAD, which it does essentially perfectly. That would read as a
    # healthy canary on a model that had stopped generating entirely.
    bad_labs = torch.tensor([-100, -100, -100, 13, 14, 0, 0])   # pads marked supervised
    masked = split_context_and_target(ids, mask, bad_labs)
    unmasked = bad_labs[bad_labs != -100]                       # what skipping the mask gives
    check("KNOWN-BAD CONTROL: skipping attention_mask admits PAD as a target",
          masked is not None and masked[1].tolist() == [13, 14]
          and unmasked.tolist() == [13, 14, 0, 0],
          f"masked={masked[1].tolist()} vs mask-free={unmasked.tolist()}")

    # rows that cannot be scored are skipped, not silently counted as 0 or 1
    check("no supervised token -> None",
          split_context_and_target(ids, mask, torch.full((L,), -100)) is None)
    check("supervised from position 0 (no context) -> None",
          split_context_and_target(ids, mask, torch.tensor(
              [10, 11, 12, 13, 14, -100, -100])) is None)
    check("empty row -> None",
          split_context_and_target(ids, torch.zeros(L, dtype=torch.long), labs) is None)

    # a supervised span with a -100 hole in it must not re-admit the hole
    ids2, mask2, labs2 = row([20, 21, 22, 23, 24, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0],
                             [-100, -100, 22, -100, 24, -100, -100])
    got2 = split_context_and_target(ids2, mask2, labs2)
    check("interior -100 is dropped from the target",
          got2 is not None and got2[1].tolist() == [22, 24], str(got2 and got2[1].tolist()))
    check("context still ends at the FIRST supervised position",
          got2 is not None and got2[0].tolist() == [20, 21], str(got2 and got2[0].tolist()))

    print()
    if fails:
        print(f"FAIL -- {len(fails)}: {fails}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
