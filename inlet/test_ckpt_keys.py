"""Assert a checkpoint written under DDP is loadable outside DDP.

    python -m inlet.test_ckpt_keys train_outputs/hyper_lora/full/hypermod_inlet.pt

Why this needs its own check. Under `torchrun`, `accelerator.prepare()` wraps
the generator in `DistributedDataParallel`, and a naive `model.state_dict()`
then prefixes every key with `module.`. Nothing complains at save time. The
failure surfaces days later, at eval, as

    RuntimeError: Error(s) in loading state_dict for HyperPrompt:
        Missing key(s): "base", "task_encoder.0.weight", ...
        Unexpected key(s): "module.base", "module.task_encoder.0.weight", ...

by which point the training run is over. `save_checkpoint()` calls
`accelerator.unwrap_model()` to prevent this; this file is what proves it, on
an actual multi-GPU checkpoint rather than by reading the code.

Exit 0 = the checkpoint is a plain single-process checkpoint.
"""
import sys

import torch

from inlet._env import bootstrap  # noqa: F401  (sets up sys.path)

bootstrap(chdir=False)

from inlet.checkpoint import load_inlet_checkpoint  # noqa: E402


def main(path: str) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    bad = [k for k in sd if k.startswith("module.")]
    print(f"checkpoint     : {path}")
    print(f"tensors        : {len(sd)}")
    print(f"trainable      : {sum(v.numel() for v in sd.values()):,d} elements")
    print(f"saved at step  : {ck['config'].get('curstep')}")
    if bad:
        print(f"FAIL  {len(bad)} keys carry the DDP 'module.' prefix, e.g. {bad[:3]}")
        print("      save_checkpoint() is not unwrapping the model.")
        return 1
    print("ok             : no 'module.' prefix")

    # The real test: strict load into a freshly constructed generator. This also
    # catches a config/shape drift that a key-name check alone would miss.
    hypermod, cfg = load_inlet_checkpoint(path, device="cpu")
    n = sum(p.numel() for p in hypermod.parameters())
    print(f"ok             : strict load into HyperPrompt, {n:,d} params, "
          f"m={hypermod.n_virtual_tokens} d={hypermod.model_dim} head={hypermod.head}")
    print("PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
