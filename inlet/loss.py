"""
Inlet loss / forward. Drop-in replacement for
hyper_llm_modulator.sft_trainer.get_loss_batch, minus the LoRA hook machinery.

T2L installs forward hooks on q_proj/v_proj to add a generated low-rank delta.
Inlet needs none of that: the soft prompt is prepended to the input embedding
sequence, which is the whole point of the paper. The only fiddly parts are the
three tensors that must be extended in lockstep (embeds / attention_mask /
labels) and the label shift, both handled here.
"""

import logging

import torch

from hyper_llm_modulator.sft_trainer import compute_loss

from inlet.sequence import build_train_sequence

logger = logging.getLogger(__name__)


def build_prompted_inputs(model, batch, soft_prompt):
    """Prepend `soft_prompt` [bs, m, d] to the embedded batch.

    Returns (inputs_embeds, attention_mask, labels).

    Note on NEFTune: the hook lives on the input embedding module, so the
    `get_input_embeddings()(input_ids)` call below picks the noise up and the
    soft prompt, concatenated afterwards, does not. That asymmetry is deliberate:
    the prompt is the thing being learned, not data to be perturbed.

    That was aspirational until 2026-08-24. `train_inlet.py` never registered the
    hook, and would not have fired it if it had -- see `activate_neftune` there
    for both halves of the bug and the runtime check that now refuses to train
    if the noise is not actually present.
    """
    # The only model-dependent step. Everything after it is pure tensor algebra
    # and lives in inlet/sequence.py, next to the eval-side assembly it has to
    # agree with -- see inlet/test_train_eval_agree.py.
    tok_embeds = model.get_input_embeddings()(batch["input_ids"])   # [bs, L, d]
    return build_train_sequence(
        tok_embeds, batch["attention_mask"], batch["labels"], soft_prompt
    )


def get_loss_batch_inlet(
    batch,
    model,
    hypermod,
    equally_weight_sample,
    l2_reg_prompt=0.0,
    label_smoothing=0.0,
    return_per_token_acc=False,
    return_entropy=False,
    override_prompt=None,
):
    """`override_prompt`: [bs, m, d] or [1, m, d] to bypass the hypernet.

    Used by the smoke test to inject a zero prompt (must reproduce zero-shot)
    and by ablations that need a fixed prompt.
    """
    out = {"prompt_l2_loss": torch.zeros((), device=model.device)}

    if override_prompt is not None:
        soft_prompt = override_prompt.expand(batch["input_ids"].shape[0], -1, -1)
    else:
        soft_prompt = hypermod(batch["task_embs"])                 # [bs, m, d]

    if l2_reg_prompt:
        out["prompt_l2_loss"] = (soft_prompt.float() ** 2).mean() * l2_reg_prompt

    inputs_embeds, attention_mask, labels = build_prompted_inputs(model, batch, soft_prompt)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    out["sft_loss"] = compute_loss(
        labels,
        outputs.logits,
        equally_weight_sample=equally_weight_sample,
        label_smoothing=label_smoothing,
    )

    if return_per_token_acc or return_entropy:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        idx = torch.where(shift_labels != -100)
    if return_per_token_acc:
        out["per_token_acc"] = (shift_logits.argmax(-1) == shift_labels)[idx].float().mean()
    if return_entropy:
        prob = torch.nn.functional.softmax(shift_logits[idx], dim=-1)
        out["entropy"] = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1).mean()

    # diagnostics: the prompt-tuning runs showed a real failure mode where the
    # learned prompt collapses (||P|| 1.44 vs 21.5 for healthy runs). Log it
    # every step so the collapse is visible in wandb before the run finishes.
    with torch.no_grad():
        out["prompt_norm"] = soft_prompt.float().norm(dim=-1).mean()
        # std over the batch axis is only defined with >=2 samples; with one
        # sample (or m=0) torch warns about dof<=0 and returns nan.
        sp = soft_prompt.float()
        out["prompt_std_across_batch"] = (
            sp.std(dim=0).mean() if sp.shape[0] > 1 and sp.shape[1] > 0
            else torch.zeros((), device=sp.device)
        )

    return out
