"""How much of the task description survives the trip into the hypernetwork.

This is the *only* place that decides it, and it is deliberately one function
with one integer knob.

Upstream hands `pooling_fn` to both `data.get_task_embs` (training, via
`create_dataloaders`) and `utils.embed_texts` (eval, via `eval_inlet`). Its
contract is

    pooling_fn(outputs, attention_mask) -> [B, D]

and nothing upstream constrains `D`. `get_task_embs` stores whatever comes back,
`data.py:199` stacks one row per sample with `torch.stack`, and the result
arrives as `batch["task_embs"]`. So a pooling function that returns K vectors
flattened to `[B, K*H]` needs **no upstream change at all** -- not to the
collator, not to the dataset, not to the cache. That is why the description
bottleneck is widened here rather than by forking `data.py`.

Why it needed widening: gte's `cls_pool` returns `last_hidden_state[:, 0]`, so
the entire task description reached the generator as **1024 numbers**, from
which it had to produce 32x4096 = 131,072. Nothing in that path let different
soft-prompt slots look at different parts of the description -- `slot_queries`
made positions distinguishable but carried no description content.

## The one knob

`n_slots` (K):

* **K = 1 reproduces upstream exactly, bit for bit.** Slot 0 is always the CLS
  vector, which *is* `cls_pool`'s output. This is checked in
  `inlet.test_desc_cond`, not asserted in prose.
* **K > 1** appends K-1 mean-pooled contiguous segments of the real (non-pad)
  tokens. The representation is therefore a strict superset of the K=1 one: any
  result at K>1 that is worse than K=1 is a training problem, never a loss of
  information.

Storage is exactly K times the pooled cache (~125 MB at K=1, ~1 GB at K=8) and
is the reason to prefer segments over raw token states: the full-token variant
is ~8 GB and needs memmapping, which buys little that K=8..32 does not.

## Padding

Slots with no content are exactly zero, and the model derives its key-padding
mask from that (`slot_mask` below) rather than carrying a second tensor through
upstream's collator. A real hidden state is never exactly all-zero, so this is
safe; `test_desc_cond` checks a short-description case where it matters.
"""

import torch

__all__ = ["make_pooling_fn", "unflatten_slots", "slot_mask", "DEFAULT_SLOTS"]

# The doc's four-week compromise: most of the benefit of token-level
# conditioning at 1/8 of the storage of raw token states.
DEFAULT_SLOTS = 8


def make_pooling_fn(n_slots: int = 1):
    """-> a `pooling_fn(outputs, attention_mask) -> [B, n_slots * H]`.

    Drop-in for `hyper_llm_modulator.utils.pooling.get_pooling_fn("cls")`, which
    is what `get_emb_model_and_fns` returns for gte. At `n_slots=1` it returns
    the identical tensor.
    """
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}")

    def pooling_fn(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        # Same assertion upstream's cls_pool makes. gte is a BERT-style encoder
        # tokenized right-padded; CLS at position 0 is only the CLS token if the
        # padding really is on the right.
        right_padding = attention_mask[:, 0].sum() == attention_mask.shape[0]
        assert right_padding, 'tokenizer.padding_side should be "right"'

        h = outputs["last_hidden_state"].detach()          # [B, L, H]
        cls = h[:, 0]                                      # [B, H]  == cls_pool
        if n_slots == 1:
            return cls

        B, L, H = h.shape
        k = n_slots - 1                                    # segment count
        mask = attention_mask.to(h.dtype)                  # [B, L]
        lengths = mask.sum(dim=1).clamp(min=1)             # [B]

        # Assign every real token to one of k contiguous segments, by its
        # position among the real tokens. Vectorized because this runs over all
        # 479 x 128 descriptions at cache-build time.
        pos = torch.arange(L, device=h.device).unsqueeze(0).expand(B, L)
        seg = (pos.to(h.dtype) * k / lengths.unsqueeze(1)).floor().long()
        seg = seg.clamp(0, k - 1)                          # [B, L]

        sums = h.new_zeros((B, k, H))
        sums.scatter_add_(1, seg.unsqueeze(-1).expand(B, L, H), h * mask.unsqueeze(-1))
        counts = h.new_zeros((B, k))
        counts.scatter_add_(1, seg, mask)                  # [B, k]

        # A segment with no real tokens stays exactly zero -- that is the signal
        # `slot_mask` reads. Do NOT switch this to clamp-then-divide without
        # keeping the zero, or the mask silently starts admitting empty slots.
        segs = torch.where(counts.unsqueeze(-1) > 0, sums / counts.clamp(min=1).unsqueeze(-1),
                           torch.zeros_like(sums))
        out = torch.cat([cls.unsqueeze(1), segs], dim=1)   # [B, n_slots, H]
        return out.flatten(1)                              # [B, n_slots * H]

    pooling_fn.n_slots = n_slots
    return pooling_fn


def unflatten_slots(task_emb: torch.Tensor, n_slots: int) -> torch.Tensor:
    """[bs, K*H] (or [bs, 1, K*H]) -> [bs, K, H]."""
    if task_emb.dim() == 3 and task_emb.shape[1] == 1:
        task_emb = task_emb.squeeze(1)
    bs, flat = task_emb.shape
    if flat % n_slots:
        raise ValueError(
            f"task_emb width {flat} is not divisible by n_slots={n_slots}. "
            f"The checkpoint's desc_slots and the pooling_fn that built the "
            f"cache disagree -- rebuild the cache or load the matching config."
        )
    return task_emb.view(bs, n_slots, flat // n_slots)


def slot_mask(slots: torch.Tensor) -> torch.Tensor:
    """[bs, K, H] -> [bs, K] bool, True where the slot carries content.

    Empty segments are exactly zero by construction in `make_pooling_fn`.
    """
    return slots.abs().sum(dim=-1) > 0
