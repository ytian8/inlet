"""The two sequence assemblies, in one file so they can be held against each other.

Inlet prepends m soft-prompt vectors at the input-embedding layer. That happens
twice, in two independent implementations:

  * TRAINING builds ``[B, m+T, d]`` on dim 1, and must extend the attention mask
    and the labels in lockstep.
  * EVALUATION builds ``[m+T, d]`` on dim 0, one request at a time, with no
    batch axis, and hands it to vLLM as ``prompt_embeds``.

If those two ever disagree -- wrong axis, reversed order, a dtype cast on one
side only -- the learned prompt is evaluated in a context it was never trained
in. Nothing raises. Every number the project reports is quietly wrong.

Both used to live somewhere untestable: the training one behind
``model.get_input_embeddings()``, the eval one as three inline statements inside
a method that needs a live vLLM engine to reach. So the repo's own
"train == eval" gate (``test_consistency_inlet.py``) ended up comparing two
*training-side* implementations with each other, and flipping ``dim=0`` to
``dim=1`` in the eval path left every gate green.

This module holds the pure tensor algebra of both, with no model, no vLLM, no
``hyper_llm_modulator`` and no GPU, so ``inlet/test_train_eval_agree.py`` can
compare them element for element in about a second.
"""

import torch

__all__ = ["build_eval_sequence", "build_train_sequence"]


def build_eval_sequence(soft_prompt, tok_embeds):
    """EVAL side. -> ``[m+T, d]``, prompt first, in ``tok_embeds``' dtype.

    Args:
        soft_prompt: ``[m, d]``, or None, or a 0-row tensor (the zero-prompt gate).
        tok_embeds:  ``[T, d]`` -- embedding rows for one request's token ids.
    """
    if soft_prompt is None or soft_prompt.numel() == 0:
        return tok_embeds
    return torch.cat([soft_prompt.to(tok_embeds.dtype), tok_embeds], dim=0)


def build_train_sequence(tok_embeds, attention_mask, labels, soft_prompt):
    """TRAINING side. -> ``(inputs_embeds, attention_mask, labels)``.

    Args:
        tok_embeds:     ``[B, T, d]`` -- already embedded, so this stays model-free.
        attention_mask: ``[B, T]``
        labels:         ``[B, T]``
        soft_prompt:    ``[m, d]`` (shared across the batch) or ``[B, m, d]``
                        (per-sample, which is what Inlet actually produces -- one
                        prompt per description).

    m == 0 returns the three inputs untouched, so the m=0 gate is exact rather
    than approximately exact.
    """
    bs = tok_embeds.shape[0]
    if soft_prompt.dim() == 2:
        soft_prompt = soft_prompt.unsqueeze(0).expand(bs, -1, -1)
    m = soft_prompt.shape[1]
    if m == 0:
        return tok_embeds, attention_mask, labels

    inputs_embeds = torch.cat([soft_prompt.to(tok_embeds.dtype), tok_embeds], dim=1)
    # The virtual tokens are always attended to...
    attention_mask = torch.cat(
        [attention_mask.new_ones((bs, m)), attention_mask], dim=1
    )
    # ...and are never a target. With the standard shift-by-one, position m-1
    # predicts the first real token: the prompt conditions, and is not predicted.
    labels = torch.cat([labels.new_full((bs, m), -100), labels], dim=1)
    return inputs_embeds, attention_mask, labels
