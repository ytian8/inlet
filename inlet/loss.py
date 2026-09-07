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


def prompt_diversity_loss(soft_prompt, base, target: float):
    """Hinge that refuses to let the generated prompt become a constant.

    `representation collapse` is the named failure mode of hypernetworks: the
    network learns to emit nearly identical parameters whatever the conditioning
    input, which makes the conditioning redundant (Hyper-DFS, App. B.2). Inlet
    has the textbook signature -- cos(prompt from a real description, prompt
    from junk) = 0.9999 and a random-description control worth +0.41 points --
    and plain SFT contains nothing that rewards telling two descriptions apart.

    Hyper-DFS penalises this with `-Var`, which is unbounded below: the cheapest
    way to minimise it is to blow the variance up. This is a hinge on the SAME
    quantity `probe_prompt.py` reports as `varying_fraction` -- the size of the
    description-dependent part of the head output relative to the part that is
    constant across the batch, measured at 5.4% on the 147.5k-step run. It is
    bounded in [0, target], it stops pushing once the target is met, and the
    number it optimises is the number the diagnostic prints, so a run can be
    read against its own gate.

    Costs no extra forward pass: the spread is taken across the batch the SFT
    term already ran.

    Returns (hinge_loss, varying_fraction); both zero with fewer than two rows.
    """
    if soft_prompt.shape[0] < 2:
        z = soft_prompt.new_zeros(())
        return z, z
    head = soft_prompt.float() - base.float().unsqueeze(0)      # [bs, m, d]
    mean = head.mean(0)                                          # [m, d]
    resid = (head - mean).reshape(-1, head.shape[-1]).norm(dim=-1).mean()
    const = mean.norm(dim=-1).mean().clamp_min(1e-12)
    varying_fraction = resid / const
    hinge = torch.relu(torch.as_tensor(target, device=head.device) - varying_fraction)
    return hinge, varying_fraction


def contrastive_task_loss(
    batch, model, soft_prompt, matched_loss, *, margin, shift,
    equally_weight_sample, label_smoothing,
):
    """Make the description's prompt work better on ITS task than on another's.

    `prompt_diversity_loss` only asks the prompts to differ. It does not ask
    them to differ *usefully*: a generator could satisfy it with task-shaped
    noise. This asks for the thing the random-description control actually
    measures -- score(real description) - score(junk) -- by scoring each
    example twice, once under its own prompt and once under a neighbour's:

        L = relu(margin - (L_mismatched - L_matched))

    The hinge stops once the mismatched pairing is `margin` nats worse, so the
    term cannot keep distorting the SFT objective after the point is made.

    COSTS A SECOND FORWARD AND BACKWARD through the frozen LM -- roughly 2x the
    step time. That is the price of a signal defined on pairs.

    `shift` rolls the prompts along the batch axis. The sampler lays a batch out
    as n_tasks_per_batch groups of n_points_per_task rows, so a shift smaller
    than n_points_per_task would pair some rows with their OWN task's prompt and
    silently weaken the signal; every row is checked and the mismatch raises.
    """
    bs = soft_prompt.shape[0]
    if bs < 2:
        return soft_prompt.new_zeros(()), soft_prompt.new_zeros(())

    rolled = soft_prompt.roll(shift, dims=0)
    embs = batch["task_embs"]
    same = (embs == embs.roll(shift, dims=0)).flatten(1).all(dim=1)
    if bool(same.any()):
        raise RuntimeError(
            f"contrastive_task_loss: {int(same.sum())} of {bs} rows kept their own "
            f"task after roll({shift}). The batch groups n_points_per_task rows per "
            f"task, so shift must be a multiple of it and n_tasks_per_batch must be "
            f">= 2. Those rows would contribute a mismatched loss that is not "
            f"mismatched, which reads as 'the term is not working'."
        )

    inputs_embeds, attention_mask, labels = build_prompted_inputs(model, batch, rolled)
    logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    mismatched = compute_loss(
        labels, logits,
        equally_weight_sample=equally_weight_sample,
        label_smoothing=label_smoothing,
    )
    gap = mismatched - matched_loss
    return torch.relu(torch.as_tensor(margin, device=gap.device) - gap), gap.detach()


def get_loss_batch_inlet(
    batch,
    model,
    hypermod,
    equally_weight_sample,
    l2_reg_prompt=0.0,
    label_smoothing=0.0,
    prompt_diversity=0.0,
    prompt_diversity_target=0.5,
    contrastive=0.0,
    contrastive_margin=0.5,
    contrastive_shift=1,
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

    zero = torch.zeros((), device=model.device)
    out["prompt_diversity_loss"] = zero
    out["contrastive_loss"] = zero
    if override_prompt is None:
        # `hypermod` may be a DDP wrapper; `base` lives on the module.
        inner = getattr(hypermod, "module", hypermod)
        hinge, vf = prompt_diversity_loss(soft_prompt, inner.base, prompt_diversity_target)
        # Logged on EVERY run, not only the ones that penalise it. The whole
        # point of the treated arms is to be read against an untreated one, and
        # a trajectory that exists for a single arm compares to nothing. Costs
        # two norms over a tensor the step already materialised.
        out["varying_fraction"] = vf.detach()
        if prompt_diversity:
            out["prompt_diversity_loss"] = hinge * prompt_diversity

    inputs_embeds, attention_mask, labels = build_prompted_inputs(model, batch, soft_prompt)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    out["sft_loss"] = compute_loss(
        labels,
        outputs.logits,
        equally_weight_sample=equally_weight_sample,
        label_smoothing=label_smoothing,
    )

    if contrastive and override_prompt is None:
        hinge, gap = contrastive_task_loss(
            batch, model, soft_prompt, out["sft_loss"],
            margin=contrastive_margin, shift=contrastive_shift,
            equally_weight_sample=equally_weight_sample,
            label_smoothing=label_smoothing,
        )
        out["contrastive_loss"] = hinge * contrastive
        out["contrastive_gap"] = gap

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
