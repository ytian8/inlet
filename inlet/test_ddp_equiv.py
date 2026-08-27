"""Prove that N ranks compute the same update as one process with the N-times batch.

    torchrun --nproc_per_node=2 -m inlet.test_ddp_equiv

CPU only, gloo backend, a few seconds, no GPU and no dataset. Run it on a login
node before you book multi-GPU time.

WHY THIS EXISTS
---------------
`--global_tasks_per_step` rests on a claim that is asserted everywhere in this
repo and, until this file, tested nowhere:

    1 GPU x 8 tasks x 8 accum  ==  2 GPUs x 8 tasks x 4 accum  ==  8 GPUs x 8 x 1

`test_accum.py` proves half of it -- that accumulation equals the larger batch
on one process. The other half is that DDP *averages* gradients across ranks
rather than summing them, because if it summed them the effective learning rate
would scale with the GPU count and a 2-GPU rerun of an 8-GPU result would be a
different experiment wearing the same name. That failure does not crash, does
not warn, and looks like "multi-GPU training is a bit unstable".

You cannot test this with the real trainer: each rank runs its own hierarchical
sampler, so 1 GPU and 2 GPUs never see the same data and the curves are only
comparable through a noise band. Here the data is synthetic and deterministic,
so the comparison is exact.

WHAT IT COMPARES
----------------
  A: `world` ranks under DDP, rank r gets shard r of a fixed batch, one step.
  B: one process, no DDP, the whole batch, one step.

Same init, same optimizer, same LR. A and B must agree to floating-point noise.

It also runs a deliberately-wrong control -- gradients summed instead of
averaged, which is what `world`-times the learning rate looks like -- and
asserts the check is sharp enough to catch it. A test that cannot fail is not a
test.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

TOL = 1e-5
BATCH_PER_RANK = 4
DIM = 8
LR = 0.1


class Tiny(nn.Module):
    """Two Linears and -- crucially -- a BUFFER.

    The buffer is not decoration. `DistributedDataParallel.forward` broadcasts
    module buffers on its FIRST call, and the pre-forward hook is gated on
    `len(self.modules_buffers) > 0`. A buffer-free model therefore CANNOT
    reproduce the bug that hung this repo's first real 2-GPU run, no matter what
    else the test does. `HyperPrompt` has `emb_rms`; this has `scale`.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, 2)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x))) * self.scale


def make_model(seed=0):
    torch.manual_seed(seed)
    return Tiny()


def fixed_batch(world):
    """The same tensor on every rank, so shard r is well defined everywhere."""
    g = torch.Generator().manual_seed(1234)
    n = BATCH_PER_RANK * world
    return torch.randn(n, DIM, generator=g), torch.randn(n, 2, generator=g)


def one_step(model, x, y, scale=1.0):
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    if scale != 1.0:
        for p in model.parameters():
            p.grad.mul_(scale)
    opt.step()
    return [p.detach().clone() for p in model.parameters()]


def max_rel_diff(a, b):
    return max(
        ((p - q).abs().max() / q.abs().max().clamp_min(1e-12)).item()
        for p, q in zip(a, b)
    )


def main() -> int:
    if "RANK" not in os.environ:
        print(__doc__)
        print("Not running under torchrun. Launch it as:")
        print("    torchrun --nproc_per_node=2 -m inlet.test_ddp_equiv")
        return 2

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        if rank == 0:
            print("This test is about what happens BETWEEN ranks; it needs at "
                  "least 2.\n    torchrun --nproc_per_node=2 -m inlet.test_ddp_equiv")
        dist.destroy_process_group()
        return 2

    x, y = fixed_batch(world)
    lo, hi = rank * BATCH_PER_RANK, (rank + 1) * BATCH_PER_RANK

    # ------------------------------------------------------------------ #
    # REGRESSION TEST for the hang, before anything else.
    #
    # Run a forward on rank 0 ONLY, exactly as rank-0-only validation does.
    # Against the DDP WRAPPER this broadcasts `scale` with no partner, the
    # ranks go one collective out of step, and the barrier below never
    # completes. Against the UNWRAPPED module it is inert -- which is the fix
    # the trainer now uses.
    #
    # gloo has no watchdog, so a regression surfaces as this test hanging
    # rather than failing. That is still better than a 27-hour job hanging.
    # ------------------------------------------------------------------ #
    wrapped = DDP(make_model())
    if rank == 0:
        with torch.no_grad():
            wrapped.module(x[lo:hi])          # unwrapped: no collective
    dist.barrier()
    if rank == 0:
        print("ok    rank-0-only forward on the UNWRAPPED module issued no "
              "collective")
        print("      (the barrier after it completed; against DDP(...) this "
              "would hang)")

    # A: this rank's shard, through DDP.
    ddp_params = one_step(wrapped, x[lo:hi], y[lo:hi])

    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # B: the whole batch, one process, no DDP.
    single_params = one_step(make_model(), x, y)
    d_ok = max_rel_diff(ddp_params, single_params)

    # Control: what a SUM instead of a MEAN would produce. DDP averages, so
    # reproducing the sum means scaling the single-process gradient by `world`.
    summed = one_step(make_model(), x, y, scale=float(world))
    d_bad = max_rel_diff(summed, single_params)

    print(f"world size                                  : {world}")
    print(f"global batch                                : {BATCH_PER_RANK * world} "
          f"({BATCH_PER_RANK} per rank)")
    print(f"DDP({world} ranks) vs 1 process, same batch     : {d_ok:.3e}")
    print(f"gradients SUMMED instead of averaged        : {d_bad:.3e}   <- must be caught")

    ok = d_ok < TOL
    sharp = d_bad > TOL * 100
    if ok and sharp:
        print(f"PASS  (tolerance {TOL:g}; the known-bad variant is "
              f"{d_bad / max(d_ok, 1e-30):.0f}x worse, so the test has teeth)")
        rc = 0
    else:
        if not ok:
            print(f"FAIL  DDP does not reproduce the single-process update "
                  f"({d_ok:.3e} > {TOL:g}).")
        if not sharp:
            print("FAIL  the control was not detected -- this test proves nothing here. "
                  "Raise LR or DIM until the summed variant separates.")
        rc = 1

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
