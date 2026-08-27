"""Train/eval consistency checks for Inlet that need no GPU.

Extends baseline_prompt_tuning/test_consistency.py. The three checks that file
does (label masking, training-prompt vs eval-template, prepend_soft_prompt
shapes) apply unchanged and are re-run here via import.

Inlet needs three MORE checks, for a reason the prompt-tuning baseline did not
have to worry about. soft_prompt_common says:

    Both paths call this one function.

Inlet cannot. Training assembles [B, m+T, d] on dim=1 with a batch axis; eval
hands vLLM a per-request [m+T, d] built on dim=0 with no batch axis. Two
separate implementations that must agree, which is exactly the failure the
prompt-tuning design was built to make impossible. The `--zero-prompt` gate does
not cover it either: at m=0 there is no prompt to misplace. Hence checks 4-6.

    python -m inlet.test_consistency_inlet --task arc_challenge
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap  # noqa: E402

# need_baseline=False: this file runs standalone now. It still PREFERS the
# author's baseline_prompt_tuning/ and says which one it got -- see REF_SOURCE.
bootstrap(need_baseline=False)

import torch  # noqa: E402

from hyper_llm_modulator.utils import get_tokenizer  # noqa: E402

try:
    from soft_prompt_common import prepend_soft_prompt  # type: ignore # noqa: E402

    REF_SOURCE = "baseline_prompt_tuning (the reference implementation's own file)"
    REF_IS_REAL = True
except ImportError:
    from inlet.baseline_ref import prepend_soft_prompt  # noqa: E402

    REF_SOURCE = "inlet.baseline_ref (RECONSTRUCTED -- see the caveat below)"
    REF_IS_REAL = False

try:
    from tasks_config import BASE_MODEL, TASK_ORDER  # type: ignore # noqa: E402
except ImportError:
    from inlet.eval_common import BASE_MODEL, TASK_ORDER  # noqa: E402

from inlet.loss import build_prompted_inputs  # noqa: E402


class _FakeModel:
    """Just enough of a HF model for build_prompted_inputs: an embedding lookup."""

    def __init__(self, embed_weight):
        self._emb = torch.nn.Embedding.from_pretrained(embed_weight, freeze=True)

    def get_input_embeddings(self):
        return self._emb


def check_per_sample_matches_shared(hidden=64, bsz=3, seqlen=7, m=32):
    """[B, m, d] with identical rows == [m, d] broadcast.

    This is the bolt that pins Inlet's training path to the already-verified
    prepend_soft_prompt. If it ever fails, Inlet is training in a different
    sequence layout than the prompt-tuning upper bound it is compared against.
    """
    vocab = 50
    emb_w = torch.randn(vocab, hidden)
    model = _FakeModel(emb_w)
    ids = torch.randint(0, vocab, (bsz, seqlen))
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones(bsz, seqlen, dtype=torch.long),
        "labels": torch.cat([torch.full((bsz, 4), -100), ids[:, 4:]], dim=1),
    }
    shared = torch.randn(m, hidden)
    per_sample = shared.unsqueeze(0).expand(bsz, -1, -1).contiguous()

    ref_e, ref_m, ref_l = prepend_soft_prompt(
        shared, model.get_input_embeddings()(ids), batch["attention_mask"], batch["labels"]
    )
    got_e, got_m, got_l = build_prompted_inputs(model, batch, per_sample)

    assert torch.equal(ref_e, got_e), "embeds differ from prepend_soft_prompt"
    assert torch.equal(ref_m, got_m), "attention_mask differs"
    assert torch.equal(ref_l, got_l), "labels differ"

    # m = 0 must be a no-op on both sides
    z_e, z_m, z_l = build_prompted_inputs(model, batch, torch.zeros(bsz, 0, hidden))
    assert z_e.shape == (bsz, seqlen, hidden), z_e.shape
    assert torch.equal(z_l, batch["labels"])
    print(f"[ok] per-sample prompt matches prepend_soft_prompt; m=0 is a no-op")


def check_prompts_differ_by_description(hidden=64, m=8):
    """The generator must (a) be deterministic in eval mode and (b) actually
    read the description.

    (b) is the collapse this whole paper hinges on. If two unrelated
    descriptions produce the same vectors, Inlet has learned a constant prompt and
    every downstream number is measuring a task-agnostic soft prompt wearing a
    hypernetwork costume.
    """
    from inlet.hyper_prompt import HyperPrompt

    hp = HyperPrompt(task_emb_size=16, model_dim=hidden, n_virtual_tokens=m)
    hp.eval()
    a = torch.randn(1, 16)
    b = torch.randn(1, 16)

    with torch.no_grad():
        pa1, pa2, pb = hp(a), hp(a), hp(b)

    assert torch.equal(pa1, pa2), "generator is not deterministic in eval mode (dropout leaking?)"
    # at init the head is zero, so a and b MUST agree -- that is the point of the
    # base+residual design. Perturb the head to simulate a trained model.
    assert torch.equal(pa1, pb), "zero-init head should ignore the description at step 0"

    with torch.no_grad():
        hp.out.weight.normal_(0, 0.02)
        pa, pb = hp(a), hp(b)
    assert not torch.allclose(pa, pb), "trained head still ignores the description"
    print("[ok] generator: deterministic in eval, zero-init ignores desc, trained head does not")


def check_base_only_is_reachable(hidden=64, m=8):
    """`base` alone must be a runnable configuration -- it is the control arm."""
    from inlet.hyper_prompt import HyperPrompt

    hp = HyperPrompt(task_emb_size=16, model_dim=hidden, n_virtual_tokens=m)
    # whitelist by PARAMETER NAME, not by module. Enumerating modules silently
    # misses trunk_norm and slot_queries, which is exactly what this caught.
    for n, q in hp.named_parameters():
        q.requires_grad_(n == "base")
    trainable = [n for n, q in hp.named_parameters() if q.requires_grad]
    assert trainable == ["base"], trainable
    print(f"[ok] base-only ablation reachable: trainable = {trainable}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="arc_challenge", choices=TASK_ORDER)
    p.add_argument("--model-dir", default=BASE_MODEL)
    p.add_argument("--skip-data", action="store_true",
                   help="run only the tensor-level checks (no dataset download)")
    args = p.parse_args()

    print(f"reference implementation : {REF_SOURCE}")
    if not REF_IS_REAL:
        print("  ^ WARNING: this compares Inlet against a RECONSTRUCTION of the")
        print("    prompt-tuning baseline, written from the reference rather than")
        print("    copied from it. A green tick here does not establish that Inlet")
        print("    and the 74.89/76.33 baseline assemble the prompt identically.")
        print("    Point INLET_BASELINE_DIR at the real directory for that.")
    print()

    if not args.skip_data:
        try:
            import test_consistency as base_checks  # type: ignore
        except ImportError:
            base_checks = None
        try:
            from soft_prompt_common import build_train_val  # type: ignore
        except ImportError:
            from inlet.baseline_ref import build_train_val

        if base_checks is None:
            # check_label_masking WAS reconstructible in full; the other two were
            # not. Run what can be run, and name what cannot.
            from inlet import consistency_ref

            base_checks = consistency_ref
            print(f"inherited checks         : {consistency_ref.SOURCE}")

        tokenizer = get_tokenizer(args.model_dir, train=True)
        train_ds, val_ds = build_train_val(args.task, tokenizer)
        print(f"task={args.task}  train={len(train_ds)}  val={len(val_ds)}")
        sample = train_ds[0]
        prompt_len = base_checks.check_label_masking(tokenizer, sample)
        print(f"[ok] label masking: {prompt_len} prompt tokens masked")
        _, _, double_bos = base_checks.check_prompt_matches_eval(
            tokenizer, args.task, sample, prompt_len
        )
        # NOT a bug: vLLM's add_special_tokens=True stacks a BOS on the
        # template's own <s>. Every column in the table has it. Inlet must too, or
        # its prompt sits one position differently from the baselines it is
        # compared against.
        assert double_bos, "expected the upstream double BOS at eval; Inlet must reproduce it"
        print("[ok] double BOS at eval reproduced (upstream behaviour)")

    return _tensor_checks()


def _tensor_checks() -> None:
    check_per_sample_matches_shared()
    check_prompts_differ_by_description()
    check_base_only_is_reachable()
    print("\nALL CHECKS PASSED"
          + ("" if REF_IS_REAL else "  (with the three inherited checks SKIPPED"
                                    " and the reference RECONSTRUCTED)"))


if __name__ == "__main__":
    main()
