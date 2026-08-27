"""Unit test for the gradient-accumulation contract in train_inlet.py's loop.

The bug this exists to catch is silent: `optimizer.zero_grad()` placed before
`accelerator.backward()` throws away every micro-batch but the last, and at
grad_accum_steps=1 -- which is what a single-GPU smoke run uses -- it changes
nothing at all. It only appears once accumulation is on, i.e. exactly when you
switch GPU counts.

The test asserts the property the whole `--global_tasks_per_step` design rests
on:

    one step over a batch of 2N   ==   two accumulated micro-steps of N

It runs the SAME loop body as train_inlet.py (accelerator.accumulate, backward,
clip, step, scheduler, zero_grad -- in that order) against a tiny model and
fixed synthetic data, so there is no sampler and no dataloader to introduce a
difference. CPU only, runs in seconds.

    python -m inlet.test_accum
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import GradientAccumulationPlugin, set_seed  # noqa: E402


def _model(seed=0):
    set_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))


def run(batches, accum, lr=1e-2, order="correct"):
    """`batches` is a list of (x, y). One optimizer step per `accum` batches."""
    plugin = GradientAccumulationPlugin(num_steps=accum, sync_with_dataloader=False)
    acc = Accelerator(gradient_accumulation_plugin=plugin, cpu=True)
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    model, opt, sched = acc.prepare(model, opt, sched)

    for x, y in batches:
        with acc.accumulate(model):
            loss = torch.nn.functional.mse_loss(model(x), y)
            if order == "buggy":
                # what the loop did before: zero_grad before backward
                opt.zero_grad()
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            else:
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
    return [p.detach().clone() for p in acc.unwrap_model(model).parameters()]


def maxrel(a, b):
    num = max((x - y).abs().max().item() for x, y in zip(a, b))
    den = max(x.abs().max().item() for x in a)
    return num / max(den, 1e-12)


def main():
    torch.manual_seed(1234)
    N, STEPS = 8, 10
    halves = [(torch.randn(N, 16), torch.randn(N, 4)) for _ in range(2 * STEPS)]
    wholes = [(torch.cat([halves[2 * i][0], halves[2 * i + 1][0]]),
               torch.cat([halves[2 * i][1], halves[2 * i + 1][1]])) for i in range(STEPS)]

    ref = run(wholes, accum=1)               # 10 steps over batches of 16
    got = run(halves, accum=2)               # 10 steps, each 2 micro-batches of 8
    bug = run(halves, accum=2, order="buggy")

    r_ok, r_bug = maxrel(ref, got), maxrel(ref, bug)
    print(f"accum=2 vs batch=16   max relative param diff : {r_ok:.3e}")
    print(f"same, with zero_grad misplaced                : {r_bug:.3e}")

    # fp32 on CPU: the only difference is summation order, so this is tight.
    TOL = 1e-4
    assert r_ok < TOL, (
        f"gradient accumulation is NOT equivalent to the larger batch "
        f"({r_ok:.3e} >= {TOL}). --global_tasks_per_step cannot be trusted: "
        f"runs at different GPU counts are different experiments."
    )
    # and the test must actually be able to see the bug it is guarding against
    assert r_bug > 100 * TOL, (
        f"the misplaced-zero_grad control only moved params by {r_bug:.3e}; "
        "this test is not sensitive enough to catch the bug it exists for."
    )
    print(f"PASS  (tolerance {TOL:.0e}; the known-bad ordering is "
          f"{r_bug / max(r_ok, 1e-12):.0f}x worse, so the test has teeth)")


if __name__ == "__main__":
    main()
