"""Checks for `contrastive_task_loss`'s pairing guard.

The hinge arithmetic is one line; the part that can be silently wrong is WHICH
prompt each row gets rolled onto. A batch is laid out as n_tasks_per_batch
groups of n_points_per_task rows, so rolling by less than n_points_per_task
pairs some rows with their OWN task's prompt. Those rows then contribute a
"mismatched" loss that is not mismatched, the gap collapses toward zero, and
the run reads as "the contrastive term does nothing" -- the one wrong
conclusion this experiment can produce.

The guard fires before the model is ever touched, so these run on CPU with no
model at all.

Run: python -m inlet.test_contrastive
"""

import sys

import torch

from inlet.loss import contrastive_task_loss

_fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


def batch_with(task_ids, emb_dim=16):
    """One row per entry; rows sharing a task id share a task embedding."""
    torch.manual_seed(0)
    table = {t: torch.randn(emb_dim) for t in set(task_ids)}
    return {"task_embs": torch.stack([table[t] for t in task_ids])}


def guard_fired(task_ids, shift, m=4, d=8):
    """Call with a model that would explode if reached.

    Returns True if the pairing guard raised, False if it let the call through
    (which then dies in build_prompted_inputs on the stub -- proof it passed).
    """
    bs = len(task_ids)
    prompt = torch.randn(bs, m, d, requires_grad=True)
    try:
        contrastive_task_loss(
            batch_with(task_ids), object(), prompt, torch.zeros(()),
            margin=0.5, shift=shift, equally_weight_sample=True, label_smoothing=0.0,
        )
    except RuntimeError as e:
        if "kept their own task" in str(e):
            return True
        raise
    except Exception:
        return False          # got past the guard, died on the stub model
    return False


def main():
    print("contrastive_task_loss")

    # --- degenerate batch: no pair to form, and the model is never touched ---
    z_hinge, z_gap = contrastive_task_loss(
        batch_with(["a"]), object(), torch.randn(1, 4, 8), torch.zeros(()),
        margin=0.5, shift=1, equally_weight_sample=True, label_smoothing=0.0,
    )
    check("bs=1 is a no-op", z_hinge.item() == 0.0 and z_gap.item() == 0.0)

    # --- the normal case: one point per task, shift 1 ---
    check("n_points_per_task=1, shift=1 passes the guard",
          not guard_fired(["a", "b", "c", "d"], shift=1))

    # --- KNOWN-BAD CONTROL -------------------------------------------------
    # Rows are grouped by task: [A A B B]. Rolling by 1 gives row 1 task A's
    # prompt (its own) and row 3 task B's (its own). Half the batch would be
    # scored against itself and the gap would look small for a reason that has
    # nothing to do with the generator.
    check("KNOWN-BAD: n_points_per_task=2 with shift=1 raises",
          guard_fired(["a", "a", "b", "b"], shift=1))
    check("KNOWN-BAD: n_points_per_task=3 with shift=2 raises",
          guard_fired(["a", "a", "a", "b", "b", "b"], shift=2))

    # --- and the matching shift is accepted ---
    check("n_points_per_task=2, shift=2 passes",
          not guard_fired(["a", "a", "b", "b"], shift=2))
    check("n_points_per_task=3, shift=3 passes",
          not guard_fired(["a", "a", "a", "b", "b", "b"], shift=3))

    # --- a whole batch of one task can never be contrasted ---
    check("KNOWN-BAD: every row the same task raises",
          guard_fired(["a", "a", "a", "a"], shift=1))

    # --- the message says how many rows were wrong, not just that it failed ---
    try:
        contrastive_task_loss(
            batch_with(["a", "a", "b", "b"]), object(),
            torch.randn(4, 4, 8), torch.zeros(()),
            margin=0.5, shift=1, equally_weight_sample=True, label_smoothing=0.0,
        )
        check("the error counts the offending rows", False, "no exception")
    except RuntimeError as e:
        check("the error counts the offending rows", "2 of 4 rows" in str(e), str(e)[:70])

    # --- the hinge itself: relu(margin - (mismatched - matched)) ---
    for matched, mismatched, margin, want in [
        (1.0, 3.0, 0.5, 0.0),      # mismatched much worse -> satisfied
        (1.0, 1.0, 0.5, 0.5),      # no separation -> full penalty
        (1.0, 1.3, 0.5, 0.2),      # partway
        (1.0, 0.2, 0.5, 1.3),      # mismatched BETTER -> penalty above margin
    ]:
        got = torch.relu(torch.tensor(margin) - (torch.tensor(mismatched) - matched)).item()
        check(f"hinge({matched}, {mismatched}, m={margin}) = {want}",
              abs(got - want) < 1e-6, f"{got:.4f}")

    print(f"\n{'FAILED: ' + ', '.join(_fails) if _fails else 'all checks passed'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
