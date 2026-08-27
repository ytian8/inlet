"""Reconstruction of the parts of `baseline_prompt_tuning/soft_prompt_common.py`
that Inlet depends on, so this repo runs standalone.

PROVENANCE — read this before trusting a comparison that rests on this file.
--------------------------------------------------------------------------
The per-task prompt-tuning baseline (74.89 arc_challenge / 76.33 8-task average)
lives in a `baseline_prompt_tuning/` directory *inside* a text-to-lora
checkout. A fresh upstream clone does not contain it, and it is not being
distributed with this repo.

The functions below were **reconstructed by reading the reference file**, at the
author's instruction, so that a clean clone can run. They are not a copy taken
from disk, and they have not been diffed against the original. Two consequences,
both of which matter:

  1. If the real module IS importable (see `inlet/_env.py:baseline_dir`), it wins.
     Every consumer in this repo prefers it. This file is the fallback.
  2. When this file is what runs, any check that compares Inlet against "the
     baseline" is comparing Inlet against a *reconstruction* of the baseline. That
     is weaker than it sounds and the code says so out loud rather than
     reporting a green tick — see `inlet/test_consistency_inlet.py`.

`test_consistency.py` IS reconstructed, in `inlet/consistency_ref.py`, and it is
complete -- both of its checks were visible end to end.

Deliberately NOT reconstructed:

  * `tasks_config.TRAIN_DS` — not visible, and nothing in Inlet reads it.
  * `train_prompt_tuning.py` / `eval_soft_prompt.py` — those produce the baseline
    numbers themselves. Inlet consumes the numbers, not the scripts.
  * `save_soft_prompt` / `load_soft_prompt` / `make_collate_fn` — reconstructed
    once, then removed on 2026-08-24 because nothing in Inlet calls them. An
    unused reconstruction of someone else's code is a liability: it looks
    authoritative, no test touches it, and the first person to use it would be
    trusting a transcription nobody ever checked. Inlet writes its own checkpoints
    (`inlet/checkpoint.py`) and gets its collator from upstream directly.

`get_tokenizer` is not reconstructed either, and must not be: the reference
re-exports `hyper_llm_modulator.utils.get_tokenizer`, so both paths import the
same public upstream function. That is the one piece that is identical by
construction rather than by transcription.
"""

import json
import os

import torch

__all__ = [
    "N_VIRTUAL_TOKENS",
    "INP_MAX_LEN",
    "SOURCE",
    "build_train_val",
    "init_soft_prompt",
    "load_input_embeddings",
    "prepend_soft_prompt",
]

SOURCE = "inlet.baseline_ref (reconstructed, not the reference file)"

N_VIRTUAL_TOKENS = 32
INP_MAX_LEN = 1024


# --------------------------------------------------------------------------- #
# the soft prompt itself
# --------------------------------------------------------------------------- #

def prepend_soft_prompt(soft_prompt, inputs_embeds, attention_mask, labels=None):
    """Concatenate the soft prompt on the SEQUENCE axis, in front of everything.

        soft_prompt    [L, d]        inputs_embeds [B, T, d]
        returns        [B, L+T, d],  mask [B, L+T], labels [B, L+T] (-100 on the L)

    The virtual tokens never contribute to the loss and are always attended to.
    L == 0 is legal and must reduce to a plain forward pass -- that is what the
    zero-shot reproduction check exercises.
    """
    bsz = inputs_embeds.shape[0]
    n_virtual = soft_prompt.shape[0]
    if n_virtual == 0:
        return inputs_embeds, attention_mask, labels

    prefix = soft_prompt.to(inputs_embeds.dtype).unsqueeze(0).expand(bsz, -1, -1)
    inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
    attention_mask = torch.cat(
        [attention_mask.new_ones((bsz, n_virtual)), attention_mask], dim=1
    )
    if labels is not None:
        labels = torch.cat([labels.new_full((bsz, n_virtual), -100), labels], dim=1)
    return inputs_embeds, attention_mask, labels


def init_soft_prompt(embedding_weight, n_virtual=N_VIRTUAL_TOKENS, seed=0, top_k=5000):
    """Initialize from real token embeddings sampled out of the frequent vocab.

    Lester et al. 2021: sampled-vocab init beats random init, and the gap widens
    as the model gets smaller. Token ids are roughly frequency-ordered in a BPE
    vocab, so the first `top_k` are the common ones.
    """
    g = torch.Generator().manual_seed(seed)
    hi = min(top_k, embedding_weight.shape[0])
    ids = torch.randint(0, hi, (n_virtual,), generator=g)
    return embedding_weight[ids].clone().float()


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #

def load_input_embeddings(model_dir, dtype=torch.bfloat16):
    """Read just `model.embed_tokens.weight` (262MB) instead of the whole 7B.

    Used by the eval path, which needs to embed prompts itself before handing
    them to vLLM as `prompt_embeds`.
    """
    from huggingface_hub import snapshot_download

    local = model_dir
    if not os.path.isdir(local):
        local = snapshot_download(model_dir, allow_patterns=["*.json", "*.safetensors"])

    key = "model.embed_tokens.weight"
    index_path = os.path.join(local, "model.safetensors.index.json")
    if os.path.exists(index_path):
        shard = json.load(open(index_path))["weight_map"][key]
        shards = [os.path.join(local, shard)]
    else:
        from glob import glob

        shards = sorted(glob(os.path.join(local, "*.safetensors")))

    from safetensors import safe_open

    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key).to(dtype)
    raise KeyError(f"{key} not found in {local}")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def build_train_val(task, tokenizer, inp_max_len=INP_MAX_LEN, val_frac=0.1):
    """Tokenized (train, val) for one task, mirroring upstream's per-task recipe.

    Upstream trains a per-task adapter on `train[:90%]` and selects on
    `train[90%:]`; this reproduces that with an index split rather than a
    shuffle, so the two are reproducible and disjoint in the same way.

    Everything task-specific -- preprocessing, prompt templates, tokenization,
    label masking -- comes from upstream `hyper_llm_modulator`, never copied.
    """
    import datasets
    from hyper_llm_modulator.utils import (
        get_inp_tokenize_fn,
        get_preprocessing_fn,
        get_prompt_formatting_fn,
    )
    from hyper_llm_modulator.utils.task_metadata import get_metadata_for_task

    metadata = get_metadata_for_task(task)
    ds_kwargs = dict(metadata["ds_kwargs"])
    path = ds_kwargs.pop("path")

    if task == "mbpp":
        # Upstream's `sanitized` train split is 90% inside MBPP+, so the
        # reference uses `full` minus the overlapping ids, listed in
        # baseline_prompt_tuning/mbpp_clean_task_ids.json. That JSON is a data
        # file, not code: it was not available to reconstruct, and inventing an
        # id list would silently change which problems the model is scored on.
        clean_ids = os.environ.get("MBPP_CLEAN_IDS")
        if not clean_ids or not os.path.isfile(clean_ids):
            raise SystemExit(
                "mbpp needs baseline_prompt_tuning/mbpp_clean_task_ids.json, which is "
                "not reconstructible from code.\n"
                "  * point INLET_BASELINE_DIR at the real directory, or\n"
                "  * set MBPP_CLEAN_IDS=/abs/path/to/mbpp_clean_task_ids.json, or\n"
                "  * skip mbpp (TASKS=... on scripts/eval.sh)."
            )
        # `full` names the problem statement `text`; `sanitized` (and therefore
        # the prompt template) calls it `prompt`.
        ds_kwargs["name"] = "full"
        ds = datasets.load_dataset(path, **ds_kwargs)
        clean = set(json.load(open(clean_ids))["clean_task_ids"])
        ds = ds.filter(lambda x: x["task_id"] in clean)
        ds = ds.rename_column("text", "prompt")
    else:
        ds = datasets.load_dataset(path, **ds_kwargs)

    ds = ds.map(get_preprocessing_fn(task), batched=False)
    fmt_fn = get_prompt_formatting_fn(
        metadata, "completion", tokenizer.apply_chat_template, is_intx_model=True
    )
    ds = ds.map(fmt_fn, batched=True)
    tok_fn = get_inp_tokenize_fn(tokenizer, "completion", True, inp_max_len)
    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    n_train = int(len(ds) * (1 - val_frac))
    return ds.select(range(n_train)), ds.select(range(n_train, len(ds)))


# --------------------------------------------------------------------------- #
# The reference's own contract for prepend_soft_prompt.
#
# These five assertions are lifted verbatim from `baseline_prompt_tuning/
# test_consistency.py::main` -- the test the ORIGINAL was checked against. They
# are the strongest statement available about this reconstruction: not "the
# assertions I thought of pass", but "the assertions the original was tested
# with pass".
#
#     python -m inlet.baseline_ref
#
# `int(m1[:, :32].sum()) == 64` is 2 x 32: every virtual token on every row must
# be attended to, not merely most of them.
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    hidden = 4096
    embeds = torch.zeros(2, 7, hidden)
    mask = torch.ones(2, 7, dtype=torch.long)
    labels = torch.full((2, 7), -100)

    h0, m0, l0 = prepend_soft_prompt(torch.zeros(0, hidden), embeds, mask, labels)
    assert h0.shape == (2, 7, hidden), h0.shape

    h1, m1, l1 = prepend_soft_prompt(torch.randn(32, hidden), embeds, mask, labels)
    assert h1.shape == (2, 39, hidden), h1.shape
    assert m1.shape == (2, 39) and int(m1[:, :32].sum()) == 64
    assert bool((l1[:, :32] == -100).all())

    print("ok    L=0 is a no-op, L=32 gives [B, 32+T, d]")
    print("ok    mask covers every virtual token on every row (2 x 32 = 64)")
    print("ok    labels are -100 across the whole prompt region")
    print(f"\nPASS  {SOURCE}")
    print("      satisfies the contract baseline_prompt_tuning/test_consistency.py")
    print("      asserts on the original. That is evidence the reconstruction is")
    print("      faithful; it is not proof. The other half of that file --")
    print("      check_label_masking and check_prompt_matches_eval -- is")
    print("      reconstructed in inlet/consistency_ref.py; run it with")
    print("        python -m inlet.consistency_ref")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
