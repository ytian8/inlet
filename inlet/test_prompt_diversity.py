"""Checks for `prompt_diversity_loss`, the anti-collapse hinge.

Run: python -m inlet.test_prompt_diversity
"""

import ast
import inspect
import sys

import torch

import inlet.loss as loss_mod
from inlet.loss import prompt_diversity_loss

_fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


def probe_prompt_varying_fraction(head):
    """The definition probe_prompt.py uses, copied so a drift shows up here."""
    mean = head.mean(0)
    resid = (head - mean).reshape(-1, head.shape[-1]).float().norm(dim=-1).mean().item()
    const = mean.float().norm(dim=-1).mean().item()
    return resid / max(const, 1e-12)


def main():
    print("prompt_diversity_loss")
    torch.manual_seed(0)
    bs, m, d = 8, 32, 4096
    base = torch.randn(m, d) * 0.5
    target = 0.5

    # --- collapsed: every row identical ---
    head = torch.randn(1, m, d).expand(bs, m, d).contiguous()
    hinge, vf = prompt_diversity_loss(base.unsqueeze(0) + head, base, target)
    check("collapse gives vf ~ 0", vf.item() < 1e-5, f"vf={vf.item():.3e}")
    check("collapse pays the full hinge", abs(hinge.item() - target) < 1e-5, f"{hinge.item():.4f}")

    # --- diverse: rows independent ---
    head = torch.randn(bs, m, d)
    hinge, vf = prompt_diversity_loss(base.unsqueeze(0) + head, base, target)
    check("independent rows give vf >> target", vf.item() > 2.0, f"vf={vf.item():.3f}")
    check("hinge is zero once the target is met", hinge.item() == 0.0, f"{hinge.item():.4f}")

    # --- the hinge is bounded, unlike -Var ---
    huge = torch.randn(bs, m, d) * 1e3
    hinge_huge, vf_huge = prompt_diversity_loss(base.unsqueeze(0) + huge, base, target)
    check("blowing the scale up cannot drive the loss below 0",
          hinge_huge.item() == 0.0 and vf_huge.item() > target,
          f"hinge={hinge_huge.item()}, vf={vf_huge.item():.1f}")

    # --- matches the diagnostic it is named after ---
    head = torch.randn(bs, m, d) * 0.05
    _, vf = prompt_diversity_loss(base.unsqueeze(0) + head, base, target)
    ref = probe_prompt_varying_fraction(head)
    check("vf equals probe_prompt's varying_fraction",
          abs(vf.item() - ref) < 1e-4, f"{vf.item():.6f} vs {ref:.6f}")

    # --- gradient points toward more spread ---
    head = (torch.randn(1, m, d).expand(bs, m, d).contiguous()
            + torch.randn(bs, m, d) * 1e-3).requires_grad_(True)
    hinge, vf0 = prompt_diversity_loss(base.unsqueeze(0) + head, base, target)
    hinge.backward()
    with torch.no_grad():
        stepped = head - 0.5 * head.grad          # descend the hinge
    _, vf1 = prompt_diversity_loss(base.unsqueeze(0) + stepped, base, target)
    check("a gradient step increases varying_fraction",
          vf1.item() > vf0.item(), f"{vf0.item():.5f} -> {vf1.item():.5f}")

    # --- degenerate batch ---
    hinge, vf = prompt_diversity_loss(base.unsqueeze(0) + torch.randn(1, m, d), base, target)
    check("a single row is a no-op", hinge.item() == 0.0 and vf.item() == 0.0)

    # --- KNOWN-BAD CONTROL ---
    # Measuring the spread of P instead of P - base looks almost identical in
    # code and is wrong: `base` is 32 real token embeddings and dominates the
    # norm, so a head that varies plenty still reports a tiny fraction and the
    # hinge stays saturated no matter what the head does. If these two ever
    # agree, the term is measuring the wrong thing.
    head = torch.randn(bs, m, d) * 0.05
    P = base.unsqueeze(0) + head
    _, vf_right = prompt_diversity_loss(P, base, target)
    vf_wrong = probe_prompt_varying_fraction(P)
    check("KNOWN-BAD: measuring P instead of P-base understates the spread",
          vf_wrong < vf_right / 5,
          f"P-base {vf_right.item():.4f} vs P {vf_wrong:.4f}")

    # --- the diagnostic must not be gated on the treatment ------------------
    # `varying_fraction` is the readout the whole experiment is decided on, and
    # the treated arms are only interpretable against the untreated ones. When
    # this was nested inside `if prompt_diversity:` every arm still trained,
    # still logged a falling loss, and still produced checkpoints -- only the
    # baselines silently had no vf trajectory to be compared with. Nothing at
    # runtime says so, hence a static check.
    print("\nget_loss_batch_inlet wiring")
    fn = next(n for n in ast.parse(inspect.getsource(loss_mod)).body
              if isinstance(n, ast.FunctionDef) and n.name == "get_loss_batch_inlet")

    def guarded_by_prompt_diversity(node):
        """Is `node` inside an `if prompt_diversity...` within this function?"""
        for parent in ast.walk(fn):
            if not isinstance(parent, ast.If):
                continue
            names = {n.id for n in ast.walk(parent.test) if isinstance(n, ast.Name)}
            if "prompt_diversity" in names and any(node is d for d in ast.walk(parent)):
                return True
        return False

    writes = [n for n in ast.walk(fn)
              if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
              and n.value.id == "out" and isinstance(n.slice, ast.Constant)
              and n.slice.value == "varying_fraction"]
    check("varying_fraction is assigned in get_loss_batch_inlet", len(writes) >= 1,
          f"{len(writes)} assignment(s)")
    ungated = [w for w in writes if not guarded_by_prompt_diversity(w)]
    check("varying_fraction is logged regardless of --prompt_diversity",
          bool(ungated),
          f"{len(ungated)}/{len(writes)} assignment(s) outside the guard")

    # and the control: the penalty itself SHOULD be gated, or every run is treated
    pen = [n for n in ast.walk(fn)
           if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
           and n.value.id == "out" and isinstance(n.slice, ast.Constant)
           and n.slice.value == "prompt_diversity_loss"]
    check("KNOWN-BAD: the penalty is still gated on --prompt_diversity",
          bool(pen) and any(guarded_by_prompt_diversity(p) for p in pen),
          f"{sum(guarded_by_prompt_diversity(p) for p in pen)}/{len(pen)} gated")

    print(f"\n{'FAILED: ' + ', '.join(_fails) if _fails else 'all checks passed'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
