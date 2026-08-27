"""Rebuild a trained HyperPrompt from a checkpoint written by train_inlet.py.

`eval_inlet.py` imports the two functions here. They are deliberately the *only*
place that knows the checkpoint layout, so training and eval cannot drift: the
architecture is reconstructed from the `config` dict that `save_checkpoint`
wrote, never from CLI flags supplied again at eval time.
"""

import logging

import torch

from inlet.hyper_prompt import HyperPrompt

logger = logging.getLogger(__name__)


def load_inlet_checkpoint(path: str, device: str = "cuda"):
    """-> (hypermod in eval mode on `device`, config dict).

    `zero_init=False` because the saved weights are about to overwrite the head
    anyway; zero-initializing first would only waste time and, worse, would hide
    a shape mismatch behind a tensor of zeros.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    hypermod = HyperPrompt(
        task_emb_size=cfg["task_emb_size"],
        model_dim=cfg["model_dim"],
        n_virtual_tokens=cfg["n_virtual_tokens"],
        encoded_task_emb_size=cfg["encoded_task_emb_size"],
        hidden_size=cfg["hypernet_hidden_size"],
        n_blocks=cfg["hypernet_n_blocks"],
        head=cfg["head"],
        zero_init=False,
        # .get with the pre-2026-08-25 defaults: a checkpoint written before
        # description slots existed is a K=1 pooled model, which is exactly what
        # these defaults build.
        n_desc_slots=int(cfg.get("desc_slots", 1)),
        cond=cfg.get("cond", "pooled"),
        n_cross_layers=int(cfg.get("n_cross_layers", 2)),
        n_cross_heads=int(cfg.get("n_cross_heads", 8)),
    )
    # strict: a silently-missing tensor here would evaluate a partly random
    # generator and look like a bad result rather than a bad load.
    missing, unexpected = hypermod.load_state_dict(ck["state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    hypermod.to(device).eval()
    for p in hypermod.parameters():
        p.requires_grad_(False)

    logger.info(
        "loaded Inlet checkpoint %s: m=%d d=%d head=%s desc_slots=%d cond=%s "
        "step=%s emb_rms=%.4f",
        path, hypermod.n_virtual_tokens, hypermod.model_dim, hypermod.head,
        hypermod.n_desc_slots, hypermod.cond,
        cfg.get("curstep"), float(hypermod.emb_rms),
    )
    return hypermod, cfg


def load_description_encoder(config: dict, device: str = "cuda"):
    """-> (emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn).

    Same call training uses, so a description is embedded identically in both
    places. Returned as a 4-tuple because eval_inlet splats it straight into
    `generate_prompts_for_task(hypermod, descs, *enc, device=...)`.

    Takes the whole CHECKPOINT CONFIG, not an `emb_model` string, and that is
    the point: the description pooling width has to be the one the checkpoint
    was trained with. Scoring a K=8 checkpoint against a K=1 encoding feeds the
    generator a description in a layout it never saw, and the only symptom is a
    worse number. Passing the config makes the mismatch unrepresentable instead
    of merely discouraged.

    A checkpoint written before description slots existed has no "desc_slots"
    key and gets K=1 -- upstream's CLS pooling, unchanged.
    """
    from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns

    from inlet.desc_pool import make_pooling_fn

    # This used to take the emb_model STRING, and both call sites passed one.
    # Changing it to the config dict kept the arity identical, so
    # test_upstream_api's signature check could not see the difference -- it
    # compares argument counts and names, not types. Fail here with the reason
    # rather than deeper in, on `config["emb_model"]` indexing a str.
    if not isinstance(config, dict):
        raise TypeError(
            "load_description_encoder takes the whole checkpoint config dict, not "
            f"an emb_model name (got {type(config).__name__}). It needs 'desc_slots' "
            "as well, so that eval pools the description the way training did."
        )

    emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
        config["emb_model"], device
    )
    n_slots = int(config.get("desc_slots", 1))
    if n_slots != 1:
        pooling_fn = make_pooling_fn(n_slots)
        logger.info("description encoder: %d slots (from checkpoint config)", n_slots)
    return emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn
