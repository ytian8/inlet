"""
Inlet: Text-to-Prompt hypernetwork.

Maps a task-description embedding to a sequence of input-layer soft-prompt
vectors for a frozen causal LM.

Design constraints (do not relax without a reason):
  * The intervention point is the INPUT EMBEDDING SEQUENCE ONLY. No per-layer
    prefixes, no weight deltas. This is what makes the substrate injectable
    through a black-box embedding-level API, and it is what makes the
    per-task prompt-tuning upper bound a like-for-like reference.
  * The description encoder (gte-large-en-v1.5) is FROZEN and its outputs are
    pre-computed once, exactly as T2L does in data.get_task_embs. This module
    therefore consumes a pre-embedded tensor, never raw text.

Description conditioning comes in two widths, set by `n_desc_slots` (K) and
produced by `inlet.desc_pool.make_pooling_fn`:

  K = 1   the description arrives as ONE 1024-d vector (gte's CLS token). This
          is upstream's `cls_pool` and was Inlet's only mode until 2026-08-25.
  K > 1   slot 0 is still that CLS vector; slots 1..K-1 are mean-pooled
          contiguous segments of the description. With `cond="cross"` the m
          soft-prompt slots then CROSS-ATTEND over those K vectors, so different
          prompt positions can read different parts of the description.

The K=1 path is preserved exactly, and not approximately: the CLS vector still
flows through `task_encoder -> trunk -> trunk_norm` as it always did, and
cross-attention is added on top of that residual stream. Setting K=1 and
cond="pooled" reproduces the pre-2026-08-25 computation graph tensor for tensor,
which is what makes an A/B between them a controlled comparison rather than two
different models. `inlet.test_desc_cond` checks this rather than trusting it.

NOTE FOR COMPARISONS: widening K changes the ARCHITECTURE, not the training
recipe (data, batch, LR schedule and step count are untouched). Inlet numbers at
different K are comparable with each other and with the published T2L numbers;
say which K produced a number whenever one is reported.
"""

import logging

import torch
import torch.nn as nn

from inlet.desc_pool import slot_mask, unflatten_slots

logger = logging.getLogger(__name__)


class TaskEncoder(nn.Module):
    """Mirrors hyper_llm_modulator.hyper_modulator.TaskEncoder (encoder_type='linear').

    Kept byte-compatible in spirit so that a Inlet run and a T2L run differ only
    in the output head, not in how the description is consumed.
    """

    def __init__(self, task_emb_size: int, encoded_task_emb_size: int):
        super().__init__()
        self.encoded_task_emb_size = encoded_task_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"encoded_task_emb": self.mlp(x)}


class MLPResidualBlock(nn.Module):
    """Mirrors T2L's MLPResidualBlock."""

    def __init__(self, size: int, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)


class CrossAttnBlock(nn.Module):
    """One pre-norm cross-attention + feed-forward block.

    Queries are the m soft-prompt slots; keys/values are the K description
    slots. `key_padding_mask` is True where a slot must be IGNORED, matching
    `nn.MultiheadAttention`'s convention -- note that this is the negation of
    `desc_pool.slot_mask`, which is True where a slot is REAL. Getting that
    backwards attends to padding only and is silent, so the conversion happens
    in exactly one place (`HyperPrompt.forward`) and is covered by
    `inlet.test_desc_cond`.
    """

    def __init__(self, dim: int, n_heads: int, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, key_padding_mask=None):
        kv = self.kv_norm(kv)
        a, _ = self.attn(self.q_norm(q), kv, kv,
                         key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        return q + self.ff(self.ff_norm(q))


class HyperPrompt(nn.Module):
    """description embedding -> [bs, n_virtual_tokens, model_dim] soft prompt.

    Two heads are supported, selected by `head`:

      'shared'   one trunk, then a single Linear to n_virtual_tokens * model_dim.
                 Cheapest; all positions produced by one matrix.

      'per_slot' one trunk, then a learned per-slot query added before a shared
                 output Linear. Lets slots specialize without an n-fold blowup
                 in parameters. This is the default.

    Initialization is base + residual, and this is load-bearing:

        P(desc) = base + head(trunk(encode(desc)))

    `base` is a task-agnostic [m, d] parameter initialized from real token
    embeddings sampled out of the frequent vocab (Lester et al. 2021, and the
    same recipe as baseline_prompt_tuning.soft_prompt_common.init_soft_prompt).
    `head` is ZERO-initialized, so at step 0 the generator contributes nothing
    and the model sees an in-distribution prompt rather than a block of zeros --
    which is the most off-distribution thing you could hand a frozen LM.

    This also buys a free ablation. `base` alone is exactly a task-agnostic soft
    prompt: freeze the head and you have a control that uses no description at
    all. If Inlet does not clearly beat base-only, the generator is not reading
    descriptions, and no amount of eval will disguise that.

    Description width is set by `n_desc_slots` (K) and `cond`:

      cond='pooled'  the trunk output is broadcast to every prompt slot, which
                     differ only by `slot_queries`. Only slot 0 (CLS) of the
                     description is read, whatever K is.
      cond='cross'   the prompt slots additionally cross-attend over all K
                     description slots, so slot i can read a different part of
                     the description than slot j. Requires head='per_slot',
                     because 'shared' has no per-slot queries to attend WITH.

    `zero_init` still zeroes `out`, so P(desc) == base at step 0 in every
    configuration -- cross-attention does not disturb that, because it acts on
    `out`'s input rather than bypassing it.
    """

    def __init__(
        self,
        task_emb_size: int,
        model_dim: int,
        n_virtual_tokens: int = 32,
        encoded_task_emb_size: int = 1024,
        hidden_size: int = 2048,
        n_blocks: int = 2,
        head: str = "per_slot",
        dropout: float = 0.05,
        zero_init: bool = True,
        emb_rms: float | None = None,
        learnable_base: bool = True,
        n_desc_slots: int = 1,
        cond: str = "pooled",
        n_cross_layers: int = 2,
        n_cross_heads: int = 8,
    ):
        super().__init__()
        assert head in ("shared", "per_slot"), head
        assert cond in ("pooled", "cross"), cond
        if cond == "cross" and head != "per_slot":
            raise ValueError(
                "cond='cross' requires head='per_slot': cross-attention needs one "
                "query per prompt slot, and head='shared' has none."
            )
        if cond == "cross" and n_desc_slots < 2:
            raise ValueError(
                f"cond='cross' with n_desc_slots={n_desc_slots} would attend over a "
                f"single vector, which is pooled conditioning with extra parameters. "
                f"Use --desc_slots 8 (or more), or cond='pooled'."
            )
        self.n_virtual_tokens = n_virtual_tokens
        self.model_dim = model_dim
        self.head = head
        self.zero_init = zero_init
        self.n_desc_slots = n_desc_slots
        self.cond = cond
        # a buffer, not a constant: it travels with the checkpoint so eval cannot
        # silently use a different scale than training did.
        self.register_buffer("emb_rms", torch.tensor(float(emb_rms if emb_rms else 1.0)))
        # task-agnostic base prompt; filled by init_base_from_vocab().
        self.base = nn.Parameter(torch.zeros(n_virtual_tokens, model_dim), requires_grad=learnable_base)

        self.task_encoder = TaskEncoder(task_emb_size, encoded_task_emb_size)
        self.trunk = nn.Sequential(
            *[MLPResidualBlock(encoded_task_emb_size, hidden_size, dropout) for _ in range(n_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(encoded_task_emb_size)

        if head == "shared":
            self.out = nn.Linear(encoded_task_emb_size, n_virtual_tokens * model_dim)
        else:
            self.slot_queries = nn.Parameter(torch.randn(n_virtual_tokens, encoded_task_emb_size) * 0.02)
            self.out = nn.Linear(encoded_task_emb_size, model_dim)

        # Cross-attention operates in the ENCODED space, so description slots go
        # through the same `task_encoder` the CLS vector does -- Linear+LayerNorm
        # apply over the last dim, so [bs, K, H] -> [bs, K, D] needs no reshape.
        if cond == "cross":
            self.cross = nn.ModuleList(
                [CrossAttnBlock(encoded_task_emb_size, n_cross_heads, hidden_size, dropout)
                 for _ in range(n_cross_layers)]
            )
        else:
            self.cross = None

        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def init_base_from_vocab(self, input_embedding_weight: torch.Tensor, seed: int = 0, top_k: int = 5000):
        """Sampled-vocab init, byte-identical to
        baseline_prompt_tuning.soft_prompt_common.init_soft_prompt, so the Inlet
        base prompt and the prompt-tuning upper bound start from the same place.

        Token ids are roughly frequency-ordered in a BPE vocab, so the first
        `top_k` are the common ones.
        """
        g = torch.Generator().manual_seed(seed)
        hi = min(top_k, input_embedding_weight.shape[0])
        ids = torch.randint(0, hi, (self.n_virtual_tokens,), generator=g)
        self.base.copy_(input_embedding_weight[ids].clone().float())
        logger.info(
            f"HyperPrompt: base initialized from {self.n_virtual_tokens} sampled vocab embeddings "
            f"(top_k={top_k}, seed={seed}), ||base||={self.base.norm(dim=-1).mean():.3f}"
        )

    @torch.no_grad()
    def fit_output_scale(self, input_embedding_weight: torch.Tensor) -> float:
        """Record the RMS of the frozen model's token embeddings."""
        rms = input_embedding_weight.float().pow(2).mean().sqrt().item()
        self.emb_rms.fill_(rms)
        logger.info(f"HyperPrompt: emb_rms set to {rms:.6f} from input embedding matrix")
        return rms

    def encode(self, task_emb: torch.Tensor) -> torch.Tensor:
        """-> [bs, D], the trunk output for the CLS slot.

        Unchanged from the pooled-only version: whatever K is, the trunk sees
        slot 0, which is gte's CLS vector. Everything K>1 adds is applied later,
        as a residual on top of this.
        """
        slots = unflatten_slots(task_emb, self.n_desc_slots)
        kv = self.task_encoder(slots)["encoded_task_emb"]      # [bs, K, D]
        return self.trunk_norm(self.trunk(kv[:, 0]))

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """task_emb: [bs, n_desc_slots * task_emb_size]
        -> [bs, n_virtual_tokens, model_dim]
        """
        slots = unflatten_slots(task_emb, self.n_desc_slots)   # [bs, K, H]
        bs = slots.shape[0]
        # Linear+LayerNorm act on the last dim, so the K slots go through the
        # same task_encoder the CLS vector does with no reshape. At K=1 this is
        # elementwise-identical to the pooled path it replaces.
        kv = self.task_encoder(slots)["encoded_task_emb"]      # [bs, K, D]
        h = self.trunk_norm(self.trunk(kv[:, 0]))              # [bs, D]

        if self.head == "shared":
            p = self.out(h).view(bs, self.n_virtual_tokens, self.model_dim)
        else:
            # [bs, 1, D] + [1, m, D] -> [bs, m, D]
            q = h.unsqueeze(1) + self.slot_queries.unsqueeze(0)
            if self.cross is not None:
                # MultiheadAttention wants True == IGNORE; slot_mask is
                # True == REAL. Inverted here, once, deliberately.
                # Slot 0 is the CLS vector and is never empty, so no row of this
                # mask is all-True and attention cannot produce NaN.
                key_padding_mask = ~slot_mask(slots)           # [bs, K]
                for blk in self.cross:
                    q = blk(q, kv, key_padding_mask=key_padding_mask)
            p = self.out(q)  # [bs, m, model_dim]

        # residual on the task-agnostic base. head is zero-initialized, so at
        # step 0 this is exactly `base` -- in every conditioning mode, because
        # `out` is downstream of the cross-attention rather than parallel to it.
        return self.base.unsqueeze(0) + p * self.emb_rms

    # ------------------------------------------------------------------ #

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
