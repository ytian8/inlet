"""Train the Inlet generator.

Mirrors scripts/train_custom_sft.py. Everything upstream owns -- task metadata,
the hierarchical sampler, the 479/31 split, decontamination, the collator,
description pre-embedding -- is imported, not reimplemented. Two things are
replaced:

    HyperModulator  ->  HyperPrompt        (input-layer vectors, not LoRA)
    get_loss_batch  ->  get_loss_batch_inlet (concat, not forward hooks)

    python -m inlet.train_inlet --run-name smoke --n-train-ds 20 --epochs 200

NOTE (2026-08-22): written without execution -- this box has no torch and no
GPU. Lines marked  # VERIFY  are the ones where I inferred an upstream API
shape from reading rather than from running. Check those first if it explodes.
"""

import faulthandler
import gc
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap, output_root  # noqa: E402

# --------------------------------------------------------------------------- #
# Hang forensics. A multi-GPU job that hangs is the worst failure mode there is:
# no traceback, no exit code, and it burns the whole wall-clock allocation while
# looking busy (a rank spinning in an NCCL collective reports 100% GPU and 100%
# CPU). Both hooks below cost nothing when nothing is wrong.
#
#   kill -USR1 <pid>        dump every thread's Python stack for that rank
#
# Send it to BOTH ranks and compare: the healthy rank is the one parked in a
# collective (`backward`, `barrier`, `all_reduce`); the stuck one is somewhere
# else, and that somewhere is the bug.
#
# Deliberately NOT using faulthandler.dump_traceback_later(repeat=True): a
# periodic dump fires on whatever thread happens to be running, and walking the
# frames of a thread that is inside a CUDA kernel launch killed a run here with
# SIGSEGV (exitcode -11) instead of telling us anything. On-demand only.
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):  # platforms without SIGUSR1
    pass


# chdir into the upstream checkout: the YAML config path, models/ and
# data/transformed_datasets are all relative to it. Anything Inlet writes goes to
# output_root() instead, which lives in THIS repo.
T2L_ROOT = bootstrap()

import torch
import wandb
from accelerate import Accelerator, PartialState
from accelerate.utils import (DataLoaderConfiguration, DistributedDataParallelKwargs,
                             GradientAccumulationPlugin, InitProcessGroupKwargs)
from transformers import get_scheduler, set_seed
from tqdm import tqdm

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import create_dataloaders
from hyper_llm_modulator.utils import (
    create_logger,
    get_metadata,
    get_model_and_tokenizer,
    save_yaml,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns

from inlet.canary import free_running_accuracy
from inlet.generative_val import build_generative_val_dataloader
from inlet.desc_pool import make_pooling_fn

from inlet.hyper_prompt import HyperPrompt
from inlet.loss import get_loss_batch_inlet

logger = logging.getLogger()


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #

# VERIFIED on the pod: upstream's ArgumentParser subclasses HfArgumentParser and
# is used as `ArgumentParser((TrainingArguments,))` then `.parse()` -- NOT
# argparse's `.parse_args()`. It takes a tuple of DATACLASSES, so Inlet's own
# options have to be a dataclass too; argparse-style add_argument would never
# reach the parsed output. Flag names use UNDERSCORES (--n_virtual_tokens).
# `.parse()` accepts `argv[1]` as a YAML config plus CLI overrides, exactly like
# scripts/train_t2l_mistral.sh does.
@dataclass
class InletArguments(TrainingArguments):
    # m. 32 matches the per-task prompt-tuning upper bound exactly.
    n_virtual_tokens: int = 32
    hypernet_head: str = "per_slot"  # "per_slot" | "shared"
    encoded_task_emb_size: int = 1024
    hypernet_hidden_size: int = 2048
    hypernet_n_blocks: int = 2
    hypernet_zero_init: int = 1
    base_init_seed: int = 0
    # ---- description conditioning width (see inlet/desc_pool.py) ----
    # How many vectors the frozen gte encoder hands to the generator. 1 is
    # upstream's CLS pooling and reproduces every Inlet number before 2026-08-25.
    # >1 adds mean-pooled segments of the description; slot 0 stays the CLS
    # vector, so the representation is a strict superset.
    #
    # Changing this needs no cache rebuild. Inlet runs with use_per_task_emb, and
    # that path (data.get_task_embs) re-embeds every description on each run --
    # only data.get_inp_prompt_emb, used by use_inp_as_desc / use_default_desc /
    # use_per_sample_desc, writes to EMBS_DIR. That cache's key is
    # ds_name + descriptions + tokenizer and does NOT include the pooling
    # function, so if Inlet ever moves onto that path a K=1 cache would be loaded
    # for a K>1 run. See the emb width assertion below, which catches exactly
    # that; do not replace it with a comment here.
    desc_slots: int = 1
    # "pooled": every prompt slot sees the same trunk output, as before.
    # "cross":  prompt slots cross-attend over the K description slots, so slot
    #           i can read a different part of the description than slot j.
    #           Requires desc_slots > 1 and hypernet_head='per_slot'.
    cond: str = "pooled"
    n_cross_layers: int = 2
    n_cross_heads: int = 8
    # base-only control: train the task-agnostic prompt and nothing else. If Inlet
    # does not beat this, the generator is not reading descriptions.
    freeze_head: bool = False
    l2_reg_prompt: float = 0.0
    # Stop after this many optimizer steps regardless of `epochs`. 0 = disabled.
    # Use it for smoke runs and for "give me a signal in 6 hours" runs; leave it
    # at 0 to reproduce the T2L recipe faithfully.
    max_steps: int = 0
    # Multiplies the LR of everything EXCEPT `base`. The description path is
    # zero-initialized, so at T2L's lr=2.5e-5 it moves very slowly -- the
    # 12-dataset run left prompt_std_across_batch at ~1e-4 after 500 steps.
    # Meanwhile the per-task prompt-tuning baseline needed lr=3e-3 to fit
    # vectors of this shape at all. 1.0 keeps the recipe verbatim; >1 is the
    # hedge against "the head never learns to read the description".
    head_lr_mult: float = 1.0
    # Which validation split decides the rolling best-val checkpoint.
    #
    # This used to be `next(iter(vi))` -- whichever split upstream happened to
    # put first in its dict, which is "val/seen". Nobody chose that; it fell out
    # of insertion order, and it would have changed silently if upstream ever
    # reordered its return value.
    #
    # It also selects the WRONG checkpoint for what this project reports. On the
    # 147.5k-step run, val/seen kept improving to ~step 130k while val/benchmark
    # bottomed out at ~step 30k, so the saved checkpoint was 9.9% worse in
    # benchmark loss (1.9266 -> 2.1169) and 1.11 pt worse in benchmark accuracy
    # than the one the run had passed through 100k steps earlier.
    #
    # The default stays "val/seen" so that this change does not silently move
    # anyone's numbers. Pick deliberately:
    #   val/seen       training tasks, unseen DESCRIPTIONS. Measures description
    #                  generalization only; keeps improving while the model
    #                  overfits the 479 training tasks.
    #   val/unseen     held-out TASKS. No benchmark data involved, and the
    #                  closest honest proxy for what Inlet claims to do.
    #   val/benchmark  the benchmark tasks' own train split. Closest to the
    #                  reported metric, but it is selection on the eval tasks --
    #                  say so if you use it.
    model_select_split: str = "val/seen"
    # Generative canary (inlet/canary.py). Free-running generation next to the
    # teacher-forced per_token_acc, on validation batches already loaded.
    #
    # This exists because the 147,500-step run showed val/benchmark
    # per_token_acc 0.80-0.82 for its whole length while humaneval was 4.47.
    # Teacher forcing hands the correct prefix back every step, so it cannot see
    # a prompt that has destroyed free generation. Nothing else in validation
    # could either, which is why that run went 100k+ steps past its own optimum.
    #
    # Costs a few short greedy generations per split per validation -- seconds,
    # no vLLM, no new dataset. 0 disables it.
    # Save the best checkpoint for EVERY validation split, not only the one
    # --model_select_split names. 55 MB each, three splits. The alternative is
    # having to guess the right criterion before the run, and guessing wrong
    # cost 2.77 points of 10-task average on the 147,500-step run.
    # Long-generation validation. None of upstream's three splits contains
    # gsm8k, mbpp or humaneval -- the three tasks that collapsed -- so nothing in
    # the loop was measuring them. "" disables. See inlet/generative_val.py for
    # why humaneval can never be here and mbpp needs decontamination.
    generative_val_tasks: str = "gsm8k"
    save_best_per_split: bool = True
    canary_samples: int = 4
    canary_max_new_tokens: int = 48
    # DDP. Each rank runs its OWN hierarchical sampler with its own seed and
    # gradients are averaged, so the effective number of distinct task
    # descriptions per optimizer step is n_tasks_per_batch * world_size. We do
    # NOT shard the dataloader: upstream uses a custom hierarchical batch
    # sampler, and letting accelerate wrap it would silently change which tasks
    # get sampled. Independent samplers + grad averaging is the same thing a
    # bigger batch would be, with no sampler surgery.
    #   "linear" | "sqrt" | "none" -- how to scale LR for the larger batch.
    #
    # SCALED BY GLOBAL BATCH, NOT BY GPU COUNT. This used to key off world size,
    # which quietly destroyed the property the rest of this file is built on:
    # once --global_tasks_per_step fixes the batch, gradient accumulation
    # absorbs the GPU count, so `train.sh 1`, `train.sh 2` and `train.sh 8`
    # present the SAME 64 tasks per optimizer step. Scaling the LR by world size
    # on top of that is not batch-size scaling, it is just a different learning
    # rate: 2.50e-5 on 1 GPU, 3.54e-5 on 2, 7.07e-5 on 8 -- three different
    # experiments wearing the same run_name, and the README calling them
    # reproductions of each other.
    #
    # Keyed off the global batch it does what it says: 1.0 for every
    # single-node config at the default GLOBAL_TASKS=64, and it only engages
    # when you actually enlarge the batch (e.g. 64 GPUs at GLOBAL_TASKS=512 ->
    # sqrt(8) = 2.83).
    lr_world_scale: str = "sqrt"
    # The global batch `lr_world_scale` is measured against. The T2L recipe this
    # LR was copied from ran at 64 tasks/step, so 64 means "no change unless you
    # deviate from the reference batch".
    lr_reference_batch: int = 64
    # The number of task descriptions per OPTIMIZER step, counted globally.
    # This is the knob that makes runs comparable across machines: gradient
    # accumulation is derived from it, so 8 GPUs and 2 GPUs execute the SAME
    # optimization (same batch, same step count, same LR) and differ only in
    # wall time. 0 = off (fall back to per_rank * world * grad_accum_steps).
    #   global = n_tasks_per_batch * world_size * grad_accum_steps
    global_tasks_per_step: int = 0
    run_name: str = "inlet"
    # Comma-separated optimizer steps at which to write a PERMANENT checkpoint,
    # e.g. "4000,20000,40000,80000,147500". These are kept alongside (and never
    # overwritten by) the rolling best-val checkpoint.
    #
    # This exists for one measurement that cannot be reconstructed afterwards:
    # how the description-conditioning gain C = score(real descs) - score(junk
    # descs) grows with training budget. A single number at the end says much
    # less than a curve, and the curve costs nothing at training time -- but
    # only if the checkpoints were kept. Decide the step list BEFORE launching.
    checkpoint_steps: str = ""
    # NCCL collective timeout, in hours.
    #
    # This is NOT cosmetic. Validation runs on rank 0 only; every other rank is
    # parked inside a collective for the whole of it. If validation (three
    # splits, one of which generates) takes longer than the timeout, the
    # watchdog on a non-main rank decides the job is deadlocked and SIGABRTs the
    # process -- the failure looks like "Watchdog caught collective operation
    # timeout: WorkNCCL(... ALLREDUCE, NumelIn=1 ...)" with no Python traceback.
    # torch's default for NCCL is 10 minutes, which a real validation pass beats
    # easily. Measured on 2x A100: ~2.5 min per non-generative split.
    ddp_timeout_hours: float = 4.0


# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, hypermod, val_dataloaders, loss_fn, curstep, max_batches=50,
             is_main=True, canary_samples=0, canary_max_new_tokens=48):
    """Validation runs on the MAIN RANK ONLY.

    Safe under DDP: the val dataloaders are not sharded, and a forward pass
    under no_grad triggers no gradient all-reduce, so the other ranks are not
    waiting on a collective. The caller must follow this with
    accelerator.wait_for_everyone().
    """
    hypermod.eval()
    # NEFTune is gated on `module.training` (upstream sft_trainer.py:86). Upstream
    # keeps it off during validation by wrapping the whole pass in `evaluating(
    # model, hypermod)`. Inlet holds only the embedding module in train mode, so
    # that is the one thing to flip here -- otherwise every val loss is measured
    # on noised embeddings and is not comparable to the checkpoint it selects.
    emb_mod = model.get_input_embeddings()
    emb_was_training = emb_mod.training
    emb_mod.eval()
    out = {}
    for name, dl in val_dataloaders.items():
        acc = defaultdict(list)
        batch = None
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            info = loss_fn(batch, return_per_token_acc=True)
            for k, v in info.items():
                acc[k].append(float(v))
        out[name] = {k: sum(v) / max(len(v), 1) for k, v in acc.items()}

        # The canary. Same batch, generated instead of teacher-forced.
        if canary_samples and batch is not None:
            try:
                dev = next(model.get_input_embeddings().parameters()).device
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                sp = hypermod(b["task_embs"])                    # [B, m, d]
                fr, ntok, nsamp = free_running_accuracy(
                    model, b, sp, max_samples=canary_samples,
                    max_new_tokens=canary_max_new_tokens)
                if ntok:
                    out[name]["free_running_acc"] = fr
                    tf = out[name].get("per_token_acc")
                    # The GAP is the signal, not either number on its own.
                    # Teacher forcing holding up while free running collapses is
                    # exactly the state that was invisible for 147,500 steps.
                    if is_main and tf and tf > 0.3 and fr < 0.5 * tf:
                        logger.warning(
                            "[step %d] %s CANARY: free-running acc %.4f is %.0f%% of "
                            "teacher-forced %.4f over %d tokens (%d samples). Teacher "
                            "forcing cannot see generative collapse -- if this gap keeps "
                            "widening, the prompt is breaking generation and the loss "
                            "curve will not tell you.",
                            curstep, name, fr, 100 * fr / tf, tf, ntok, nsamp)
            except Exception as exc:            # never let a probe kill a run
                if is_main:
                    logger.warning("[step %d] %s canary skipped: %s", curstep, name, exc)

        if is_main:
            for k, v in out[name].items():
                wandb.log({f"{name}/{k}": v}, step=curstep)
            logger.info(f"[step {curstep}] {name}: " +
                        " ".join(f"{k}={v:.4f}" for k, v in out[name].items()))
    hypermod.train()
    emb_mod.train(emb_was_training)
    return out


def activate_neftune(model, alpha, is_main=True):
    """Turn NEFTune on, or say out loud that it is off. Returns the hook handle.

    THIS WAS A SILENT NO-OP UNTIL 2026-08-24 AND EVERY DOC IN THE REPO CLAIMED
    OTHERWISE. Two independent reasons, either one sufficient:

      1. `train_inlet.py` never called `trl_activate_neftune`, so the forward hook
         was never registered at all. `--neftune_noise_alpha=5` in train.sh is a
         field on upstream's TrainingArguments, so it parsed cleanly and did
         nothing.
      2. The hook body is `if module.training:` (upstream sft_trainer.py:86).
         Inlet calls `model.eval()` to freeze the frozen base model and never calls
         `model.train()`, where upstream calls `model.train()` before activating
         (sft_trainer.py:184). So even with the hook registered it would not fire.

    Upstream puts the WHOLE model in train mode. Inlet puts only the embedding
    module there, which for Mistral is the same thing. VERIFIED against
    transformers v4.57.1 modeling_mistral.py: the file contains exactly one
    dropout call, `nn.functional.dropout(attn_weights, p=dropout,
    training=module.training)` in `eager_attention_forward`, and `dropout` comes
    from `config.attention_dropout`, which MistralConfig defaults to 0.0 (and
    the sdpa/flash paths this repo runs do not go through that function at all).
    MistralRMSNorm keeps no running statistics. So for Mistral, train vs eval
    changes nothing except whether this hook fires.

    Narrowing it to the embedding keeps that guarantee true even for a base model
    that *does* have dropout, where upstream's `model.train()` would quietly
    start perturbing a model the paper describes as frozen.

    The verification below matters more than the activation. A no-op NEFTune is
    invisible: the loss curve looks fine and the recipe silently differs from the
    T2L run it is being compared against. So we prove the noise is real -- same
    input ids, twice, must give different embeddings -- and refuse to train if it
    is not.
    """
    if not alpha or alpha <= 0:
        if is_main:
            logger.info("[neftune] disabled (alpha=%s)", alpha)
        return None

    from hyper_llm_modulator.sft_trainer import trl_activate_neftune

    handle = trl_activate_neftune(model, alpha)
    emb = model.get_input_embeddings()
    emb.train()

    with torch.no_grad():
        probe_ids = torch.zeros(1, 8, dtype=torch.long, device=next(emb.parameters()).device)
        a, b = emb(probe_ids), emb(probe_ids)
        noisy = not torch.equal(a, b)
        emb.eval()
        c, d = emb(probe_ids), emb(probe_ids)
        clean = torch.equal(c, d)
        emb.train()

    if not noisy:
        handle.remove()
        raise RuntimeError(
            f"NEFTune was requested (alpha={alpha}) but two forward passes over the "
            "same ids gave identical embeddings, so the hook is not firing. Training "
            "now would silently use a different recipe from the T2L numbers this is "
            "compared against. Refusing to start."
        )
    if not clean:
        handle.remove()
        raise RuntimeError(
            "NEFTune noise is applied even with the embedding module in eval mode. "
            "Validation and the m=0 gate would both be measured on noised inputs."
        )
    if is_main:
        delta = (a - b).abs().max().item()
        logger.info(
            "[neftune] alpha=%s ACTIVE on %s (max |delta| over one probe: %.3e); "
            "off in eval mode, so validation and the m=0 gate are clean",
            alpha, type(emb).__name__, delta,
        )
    return handle


def save_checkpoint(save_dir, hypermod, args, curstep, extra=None, accelerator=None,
                    filename="hypermod_inlet.pt"):
    """Write hypermod_inlet.pt. MAIN RANK ONLY -- the caller guards this.

    Under DDP the module is wrapped and every key gains a `module.` prefix,
    which would make load_inlet_checkpoint's strict=True load fail. unwrap_model
    is a no-op on one GPU, so the single- and multi-GPU checkpoints are byte
    compatible.
    """
    if accelerator is not None:
        hypermod = accelerator.unwrap_model(hypermod)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    sd = {k: v.detach().cpu() for k, v in hypermod.state_dict().items()}
    torch.save(
        {
            "state_dict": sd,
            "config": {
                "n_virtual_tokens": hypermod.n_virtual_tokens,
                "model_dim": hypermod.model_dim,
                "head": hypermod.head,
                "task_emb_size": hypermod.task_encoder.mlp[0].in_features,
                "encoded_task_emb_size": args.encoded_task_emb_size,
                "hypernet_hidden_size": args.hypernet_hidden_size,
                "hypernet_n_blocks": args.hypernet_n_blocks,
                # Eval rebuilds BOTH the model and the description pooling_fn
                # from these four. They are not cosmetic: a checkpoint scored
                # under a different desc_slots is being fed a description it was
                # never trained on, and nothing raises.
                "desc_slots": hypermod.n_desc_slots,
                "cond": hypermod.cond,
                "n_cross_layers": args.n_cross_layers,
                "n_cross_heads": args.n_cross_heads,
                "emb_model": args.emb_model,
                "model_dir": args.model_dir,
                "n_train_ds": args.n_train_ds,
                "n_descs_per_ds": args.n_descs_per_ds,
                "inp_max_len": args.inp_max_len,
                "lr": args.lr,
                "seed": args.seed,
                "curstep": curstep,
            },
            "extra": extra or {},
        },
        path,
    )
    # READ IT BACK. A checkpoint written to a wedged network volume can come out
    # the right SIZE and still be truncated: the smoke2 run on 2026-08-26 left a
    # 55,267,695-byte hypermod_inlet.pt -- exactly the size a healthy one is --
    # that died on load with `EOFError: Ran out of input`. Nothing at write time
    # noticed, and the run that produced it had already been killed, so the
    # failure surfaced days-equivalent later, at eval, with the training gone.
    #
    # Loading 55 MB back costs about a second and tests the thing that actually
    # broke, rather than guessing at filesystem types.
    try:
        _back = torch.load(path, map_location="cpu", weights_only=False)
        _missing = [k for k in ("state_dict", "config") if k not in _back]
        if _missing:
            raise RuntimeError(f"reloaded checkpoint is missing {_missing}")
        if len(_back["state_dict"]) != len(sd):
            raise RuntimeError(
                f"reloaded checkpoint has {len(_back['state_dict'])} tensors, wrote {len(sd)}"
            )
        del _back
    except Exception as exc:
        raise RuntimeError(
            f"checkpoint at {path} did not survive a read-back: {exc}. The file may "
            f"look the right size and still be truncated -- this happens when the "
            f"output directory is on a network/FUSE volume that wedged mid-write. "
            f"Point INLET_OUTPUT_ROOT at local disk and rerun; do not trust this file."
        ) from exc
    return path


# --------------------------------------------------------------------------- #

def effective_timeout():
    """The timeout NCCL is ACTUALLY using, read back from the live process group.

    Printed at startup on purpose. The value in the source is only a request;
    if anything created the process group earlier it wins silently, and the
    first symptom is a SIGABRT with no traceback ten minutes into the run.
    """
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            return "n/a (single process)"
        pg = dist.distributed_c10d._get_default_group()
        # The timeout lives on the BACKEND's options, not on the ProcessGroup
        # wrapper -- `pg.options` raises AttributeError on torch 2.9. Never fall
        # back to reporting torch's default here: a diagnostic that prints the
        # default when it cannot read the real value is worse than one that
        # admits it does not know, because the whole point is to catch the case
        # where the real value IS the default by accident.
        for dev in (torch.device("cuda"), torch.device("cpu")):
            try:
                opts = pg._get_backend(dev).options
            except Exception:
                continue
            t = getattr(opts, "_timeout", None)
            if t is not None:
                return str(t)
        return "unreadable"
    except Exception as e:  # never let a diagnostic kill the run
        return f"unknown ({type(e).__name__})"


def main(args):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    save_dir = args.save_dir  # __main__ already appended run_name -- create_logger needs it early
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)

    # PartialState brings up the distributed backend without building the full
    # Accelerator, so we can size gradient accumulation from the world size --
    # which the Accelerator constructor already needs to know.
    # The timeout must be handed to whichever call creates the process group.
    # PartialState() creates it, so passing InitProcessGroupKwargs to the
    # Accelerator below is too late: the group already exists and accelerate
    # silently keeps the existing one. That is how a "1 hour" timeout in the
    # source ends up as torch's 10-minute NCCL default at runtime.
    ddp_timeout = timedelta(hours=args.ddp_timeout_hours)
    world = PartialState(timeout=ddp_timeout).num_processes
    per_rank = args.n_tasks_per_batch
    if args.global_tasks_per_step > 0:
        denom = per_rank * world
        if args.global_tasks_per_step % denom != 0:
            raise SystemExit(
                f"--global_tasks_per_step={args.global_tasks_per_step} is not divisible by "
                f"n_tasks_per_batch({per_rank}) * world({world}) = {denom}. "
                f"Pick a global batch that divides evenly, or change n_tasks_per_batch."
            )
        accum = args.global_tasks_per_step // denom
    else:
        accum = args.grad_accum_steps
    args.grad_accum_steps = accum
    global_tasks = per_rank * world * accum
    plugin = GradientAccumulationPlugin(num_steps=accum, sync_with_dataloader=False)
    # NCCL's default collective timeout is 10-30 min. Validation, checkpoint
    # writes to a shared filesystem and the very first cudagraph capture all
    # happen on rank 0 while the other ranks sit in wait_for_everyone(); one
    # slow NFS write should not look like a dead rank. An hour is generous and
    # costs nothing when nothing is wrong.
    ddp_kwargs = InitProcessGroupKwargs(timeout=ddp_timeout)
    # broadcast_buffers=False: the only buffer is `emb_rms`, computed from the
    # frozen embedding table, so it is bit-identical on every rank and never
    # updated. Broadcasting it every forward is pure cost -- and, because the
    # broadcast fires on the first forward regardless of no_grad, it is the
    # thing that turns a rank-0-only validation pass into a permanent hang.
    # Validation also runs on the unwrapped module (see below); this is the
    # second lock on the same door.
    ddp_perf = DistributedDataParallelKwargs(broadcast_buffers=False)
    # split_batches MUST be set through DataLoaderConfiguration.
    #
    # `Accelerator(split_batches=True)` still *accepts* the argument on
    # accelerate 1.14 -- its default is a sentinel (`_split_batches = object()`)
    # -- but nothing in __init__ copies it into the dataloader config, and
    # `Accelerator.split_batches` is a property reading
    # `self.dataloader_config.split_batches`. So the keyword is silently
    # discarded and the value is False.
    #
    # That is not cosmetic here. We prepare no dataloader, so split_batches has
    # exactly ONE effect in this trainer -- AcceleratedScheduler.step():
    #
    #     if self.split_batches:
    #         self.scheduler.step()                   # once per optimizer step
    #     else:
    #         for _ in range(num_processes):          # num_processes times!
    #             self.scheduler.step()
    #
    # With it off, the cosine schedule is consumed `world` times too fast. On 8
    # GPUs a 147,500-step run finishes its 29,500-step warmup at optimizer step
    # 3,687 and bottoms out at step 18,437 -- after which transformers' cosine
    # lambda, driven past progress=1.0, walks the LR back UP. 87% of the run
    # would be on a schedule nobody designed. On 1 GPU (num_processes=1) it is
    # invisible, which is why it survived every local test.
    dl_cfg = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=plugin,
                              dataloader_config=dl_cfg, log_with="wandb",
                              kwargs_handlers=[ddp_kwargs, ddp_perf])
    # Read it back. A configured value that silently fails to apply is the whole
    # reason the line above is three lines of comment.
    assert accelerator.split_batches is True, (
        "accelerator.split_batches is False -- the LR schedule would advance "
        f"{accelerator.num_processes}x per optimizer step. accelerate has moved "
        "this setting again; find where DataLoaderConfiguration went."
    )
    is_main = accelerator.is_main_process
    accelerator.init_trackers(
        os.getenv("WANDB_PROJECT", "inlet"),
        config=vars(args),
        init_kwargs=dict(wandb={"group": args.run_name, "name": args.run_name}),
    )
    device = accelerator.device

    # ---------------- frozen base model ----------------
    # requires_grad=False everywhere: Inlet trains only the generator. Gradients
    # still flow THROUGH the model to reach the input embeddings, which is why
    # this is ~2x forward cost and not 3x -- no weight grads are computed.
    model, tokenizer = get_model_and_tokenizer(
        args.model_dir, train=True, requires_grad=False, peft_config=None,
        # FlashAttention2 is the upstream default.
        # The RunPod debug pod has no flash_attn wheel for this torch/CUDA build,
        # so INLET_NO_FLASH_ATTN=1 falls back to sdpa there. Numerically slightly
        # different, structurally identical -- never set it for a real run.
        use_flash_attn=os.getenv("INLET_NO_FLASH_ATTN", "0") != "1",
        model_kwargs={"output_hidden_states": False, "output_attentions": False},
        device=device,
    )
    model.eval()  # freeze dropout/norm stats in the base model
    for p in model.parameters():
        p.requires_grad_(False)
    assert tokenizer.chat_template is not None, "Only chat models are supported"

    # ---------------- description encoder (frozen, pre-embedded) ----------------
    # get_emb_model_and_fns returns the gte encoder plus the formatting/pooling
    # helpers; create_dataloaders embeds every description ONCE under no_grad.
    # VERIFIED 2026-08-23 on the pod: this returns FOUR values, and pooling_fn is
    # one of them -- do NOT build it separately with get_pooling_fn.
    emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
        args.emb_model, device
    )
    # The description bottleneck. Upstream's pooling_fn keeps only gte's CLS
    # token; ours keeps K vectors. Replacing the FUNCTION rather than the
    # dataloader is what keeps Inlet an overlay: create_dataloaders, the collator,
    # the cache and the hierarchical sampler are all untouched, because nothing
    # upstream constrains the width pooling_fn returns.
    #
    # The same factory is called at eval time from the CHECKPOINT's config, not
    # from a flag -- see inlet.checkpoint.load_description_encoder. That is the
    # only reason a K mismatch between training and eval cannot happen silently.
    # Captured now because emb_model is freed before the generator is built, and
    # this is the number the per-slot width has to match.
    emb_hidden_size = int(emb_model.config.hidden_size)
    if args.desc_slots != 1:
        pooling_fn = make_pooling_fn(args.desc_slots)
        if is_main:
            logger.info(
                f"[desc] pooling widened to {args.desc_slots} slots "
                f"(slot 0 = CLS, {args.desc_slots - 1} mean-pooled segments); "
                f"cond={args.cond}"
            )

    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)

    # ---------------------------------------------------------------------
    # main_process_first is NOT a nicety here -- without it multi-GPU training
    # dies on any dataset that is not already in the cache.
    #
    # create_dataloaders reaches get_datasets (upstream data.py:128), whose
    # cache hit test is
    #     if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
    #         datasets.load_from_disk(...)
    # i.e. it treats the mere EXISTENCE of the directory as "already built".
    # `save_to_disk` creates that directory and then fills it, so a second rank
    # arriving mid-write sees a truthy glob and calls load_from_disk on a
    # half-written directory:
    #     FileNotFoundError: Directory data/transformed_datasets/<hash> is
    #     neither a `Dataset` directory nor a `DatasetDict` directory.
    # Observed exactly this way on 2 GPUs, 2026-08-24. With 8 ranks it is 8
    # writers on the same path.
    #
    # Do not assume warm_cache.sh makes this unreachable. It warms the TRAIN
    # representation of each task; create_dataloaders additionally builds
    # "val/seen" (the same tasks re-hashed after it rewrites their split to
    # train[:90%] / train[90%:]), "val/unseen", and "val/benchmark" (eval tasks
    # with BENCHMARK_TASK_INFO merged into ds_kwargs) -- all different hashes.
    # Some dataset is essentially always built on the first run.
    #
    # main_process_first gives rank 0 the write, then releases the others onto
    # a complete cache. Cost is one serialized build on the first run only.
    # ---------------------------------------------------------------------
    with accelerator.main_process_first():
        dataloaders = create_dataloaders(
            args, train_metadata, val_metadata, use_hypernet=True, device=device,
            tokenizer=tokenizer, is_intx_model=True, emb_model=emb_model,
            emb_tokenizer=emb_tokenizer, task_desc_format_fn=task_desc_format_fn,
            pooling_fn=pooling_fn,
        )
    train_dataloader = dataloaders.pop("train")

    # val/generative. Built after create_dataloaders and inside the same
    # main_process_first region's shadow: get_datasets writes to the same
    # transformed_datasets cache, and a second rank arriving mid-write dies with
    # the FileNotFoundError described above. Rank 0 builds, everyone else waits.
    if args.generative_val_tasks.strip():
        _gen_names = tuple(t.strip() for t in args.generative_val_tasks.split(",") if t.strip())
        with accelerator.main_process_first():
            _gen_dl = build_generative_val_dataloader(
                args, val_metadata, tokenizer, True,
                emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device,
                task_names=_gen_names,
            )
        if _gen_dl is not None:
            dataloaders["val/generative"] = _gen_dl
        elif is_main:
            logger.warning(
                "val/generative requested (%s) but not built -- checkpoint selection "
                "and the canary stay blind to long-generation tasks",
                args.generative_val_tasks)
    # VERIFIED: upstream returns exactly
    #   {"train", "val/seen", "val/unseen", "val/benchmark"}
    # -- the three val splits are the same three eval groups the config defines
    # (training tasks w/ unseen descriptions, unseen tasks, benchmark tasks).
    val_dataloaders = dataloaders
    if is_main:
        logger.info(f"train batches/epoch={len(train_dataloader)}  "
                    f"val splits={list(val_dataloaders)}")

    # the encoder has done its job; 1.7GB back to vLLM-free headroom
    del emb_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- generator ----------------
    probe = next(iter(train_dataloader))
    # The cache is K*H wide; HyperPrompt wants the PER-SLOT width H. If these
    # disagree the cache was built at a different desc_slots -- fail here, with
    # the numbers, rather than reshaping into a silently wrong model.
    flat_w = probe["task_embs"].shape[-1]
    if flat_w % args.desc_slots:
        raise RuntimeError(
            f"cached task_embs are {flat_w} wide, which is not divisible by "
            f"--desc_slots={args.desc_slots}. The description cache was built "
            f"with a different number of slots. Rebuild it, or match the flag."
        )
    task_emb_size = flat_w // args.desc_slots
    # Divisibility is not enough. A pooled 1024-wide vector divides evenly by
    # desc_slots=8 into 128-wide "slots", and everything downstream would then be
    # self-consistent: task_encoder would be built as Linear(128, ...), training
    # would run, the loss would fall, and the generator would be reading eight
    # slices of one vector as if they were eight parts of the description.
    # Nothing else in this file can catch that, so check the actual width.
    if task_emb_size != emb_hidden_size:
        raise RuntimeError(
            f"description slots are {task_emb_size} wide but {args.emb_model} emits "
            f"{emb_hidden_size}. task_embs came back {flat_w} wide for "
            f"--desc_slots={args.desc_slots}, which means the embeddings were NOT "
            f"produced by inlet.desc_pool.make_pooling_fn({args.desc_slots}) -- most "
            f"likely a stale EMBS_DIR cache from a different desc_slots, reachable "
            f"if this run is on the use_inp_as_desc / use_default_desc / "
            f"use_per_sample_desc path. Clear EMBS_DIR or fix the flag."
        )
    emb_w = model.get_input_embeddings().weight
    hypermod = HyperPrompt(
        task_emb_size=task_emb_size,
        model_dim=emb_w.shape[1],
        n_virtual_tokens=args.n_virtual_tokens,
        encoded_task_emb_size=args.encoded_task_emb_size,
        hidden_size=args.hypernet_hidden_size,
        n_blocks=args.hypernet_n_blocks,
        head=args.hypernet_head,
        zero_init=bool(args.hypernet_zero_init),
        n_desc_slots=args.desc_slots,
        cond=args.cond,
        n_cross_layers=args.n_cross_layers,
        n_cross_heads=args.n_cross_heads,
    ).to(device)
    hypermod.fit_output_scale(emb_w)
    hypermod.init_base_from_vocab(emb_w, seed=args.base_init_seed)

    if args.freeze_head:
        # whitelist by PARAMETER NAME. Freezing module-by-module missed
        # trunk_norm and slot_queries, so the "base-only" control was not.
        for nm, p in hypermod.named_parameters():
            p.requires_grad_(nm == "base")
        if is_main:
            logger.warning("BASE-ONLY CONTROL: description path frozen, training `base` alone")

    trainable = [n for n, p in hypermod.named_parameters() if p.requires_grad]
    if is_main:
        logger.info(f"trainable params: {hypermod.num_trainable():,d}  "
                    f"({len(trainable)} tensors)")

    # two param groups so the description path can be given a faster LR than the
    # task-agnostic `base` prompt. head_lr_mult=1.0 makes this exactly one group.
    # Ratio of the batch we are actually running to the batch the LR was tuned
    # for -- NOT the GPU count. See lr_world_scale's docstring.
    _ratio = global_tasks / float(max(args.lr_reference_batch, 1))
    _scale = {"linear": _ratio,
              "sqrt": _ratio ** 0.5,
              "none": 1.0}[args.lr_world_scale]
    lr = args.lr * _scale
    if is_main:
        logger.info(
            f"LR: {args.lr:.3e} x {_scale:.4f} = {lr:.3e}  "
            f"[global batch {global_tasks} vs reference {args.lr_reference_batch}, "
            f"'{args.lr_world_scale}']  -- independent of GPU count by construction"
        )
    base_p = [p for n, p in hypermod.named_parameters() if n == "base" and p.requires_grad]
    head_p = [p for n, p in hypermod.named_parameters() if n != "base" and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": base_p, "lr": lr},
         {"params": head_p, "lr": lr * args.head_lr_mult}],
        lr=lr, weight_decay=args.weight_decay,
    )
    if is_main:
        logger.info(f"optimizer: base lr={lr:.2e}  head lr={lr * args.head_lr_mult:.2e} "
                    f"(head_lr_mult={args.head_lr_mult})")
    # `epochs` REALLY IS EPOCHS. Upstream sft_trainer.py is
    #     for _ in tqdm(range(args.epochs), total=num_training_steps):
    #         for batch in train_dataloader:
    # -- an outer epoch loop with an inner dataloader loop. An earlier version
    # of this file read `epochs` as a step count and so trained 59x too little.
    # At n_train_ds=479 / n_tasks_per_batch=8 there are 59 batches per epoch, so
    # the T2L recipe's --epochs=20000 is 1,180,000 optimizer steps. That is why
    # their README says ~5 days on one H100.
    keep_steps = sorted({int(x) for x in str(args.checkpoint_steps).split(",") if x.strip()})
    if keep_steps and is_main:
        logger.info(f"permanent checkpoints will be kept at steps: {keep_steps}")

    steps_per_epoch = len(train_dataloader)
    # What T2L does on one GPU: epochs * batches_per_epoch optimizer steps, each
    # over n_tasks_per_batch tasks. With `world` ranks each drawing their own
    # batch, one of our steps covers `world` times as many tasks, so we need
    # `world` times fewer of them to see the same amount of data.
    single_gpu_steps = args.epochs * steps_per_epoch
    num_training_steps = (args.max_steps if args.max_steps > 0
                          else max(1, single_gpu_steps // (world * accum)))
    if is_main:
        logger.info(
            f"steps/epoch={steps_per_epoch}  epochs={args.epochs}\n"
            f"  world={world}  per-rank batch={per_rank}  grad_accum={accum}  "
            f"-> GLOBAL BATCH {global_tasks} tasks/step\n"
            f"  T2L-equivalent on 1 GPU : {single_gpu_steps:,d} steps x {per_rank} tasks\n"
            f"  this run                : {num_training_steps:,d} steps x {global_tasks} tasks\n"
            f"  total task-samples      : {num_training_steps * global_tasks:,d} "
            f"(T2L: {single_gpu_steps * per_rank:,d})"
            + ("  [OVERRIDDEN by --max_steps]" if args.max_steps > 0 else "")
        )
        logger.info(f"  NCCL collective timeout : {effective_timeout()}")
    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=int(args.warmup_frac * num_training_steps),
        num_training_steps=num_training_steps,
    )
    # The dataloader is deliberately NOT prepared -- see lr_world_scale above.
    hypermod, optimizer, scheduler = accelerator.prepare(hypermod, optimizer, scheduler)
    # Re-seed AFTER construction so every rank draws a DIFFERENT stream of tasks.
    # Weights are already identical across ranks (DDP broadcasts from rank 0 on
    # wrap), so this only affects sampling, which is exactly what we want.
    #
    # This works only because upstream's HierachicalBatchSampler draws from the
    # GLOBAL torch RNG lazily, inside `__iter__` (data.py:99 `torch.randperm`,
    # :107 `torch.randint`) rather than from a Generator captured at
    # construction. If that ever changes, re-seeding here becomes a no-op, every
    # rank samples the SAME tasks, and N GPUs stop being an N-times bigger batch
    # -- they become N copies of the same gradient. The loss curve looks
    # completely normal while the effective batch is 1/N of what every log line,
    # the LR scale and the paper all claim.
    #
    # So do not trust the comment: check it. One draw off the global RNG,
    # gathered across ranks; they must all differ.
    if accelerator.num_processes > 1:
        set_seed(args.seed + accelerator.process_index)
        # Draw on the CPU generator, NOT the CUDA one. `torch.randint(...,
        # device="cuda")` consumes the CUDA RNG, and the sampler consumes the
        # CPU RNG (`torch.randperm` / `torch.randint` on CPU, data.py:99/:107).
        # Probing the wrong generator would only be correlated evidence -- both
        # are seeded from the same value, so they usually agree -- and this
        # check exists precisely for the case where the usual thing is false.
        # `.to(device)` afterwards because NCCL gathers device tensors.
        # NOT named `probe`: that is the m=0 gate's batch, live at this point.
        rng_probe = torch.randint(0, 2 ** 31 - 1, (1,))          # CPU
        gathered = accelerator.gather(rng_probe.to(device)).tolist()
        if len(set(gathered)) != len(gathered):
            raise RuntimeError(
                "Per-rank sampler independence is broken: after set_seed(seed + rank) "
                f"the global torch RNG gave {gathered} -- at least two ranks share a "
                "stream, so they will sample identical task batches and the effective "
                "batch size is smaller than every log line claims. Refusing to train."
            )
        if is_main:
            logger.info("[ddp] per-rank RNG streams verified distinct across "
                        f"{accelerator.num_processes} ranks")

    def _loss(batch, module, **kw):
        # No dataloader goes through accelerator.prepare() (see above), so no
        # batch arrives on the GPU by itself. Moving here covers train and val
        # alike and is a no-op for anything already on the right device.
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        return get_loss_batch_inlet(
            batch, model=model, hypermod=module,
            equally_weight_sample=args.equally_weight_sample,
            l2_reg_prompt=args.l2_reg_prompt,
            label_smoothing=args.label_smoothing,
            **kw,
        )

    def loss_fn(batch, **kw):
        """Training loss: goes through the DDP wrapper, so gradients all-reduce."""
        return _loss(batch, hypermod, **kw)

    # THE VALIDATION LOSS MUST NOT GO THROUGH THE DDP WRAPPER. This is not a
    # micro-optimization; it is the difference between a job that trains and a
    # job that hangs forever with both GPUs pinned at 100%.
    #
    # `DistributedDataParallel.forward` broadcasts the module's buffers on its
    # FIRST call -- `require_forward_param_sync` is initialised to True, and the
    # pre-forward hook fires before anything has cleared it. `no_grad` does not
    # prevent that; it only prevents the *gradient* all-reduce. HyperPrompt has a
    # buffer (`emb_rms`), so the broadcast is real.
    #
    # Validation runs on rank 0 only. So rank 0 issues a broadcast that no other
    # rank ever posts. NCCL matches collectives by ORDER on a communicator, not
    # by kind or size, so that stray broadcast pairs up with whatever rank 1
    # posts next -- its `wait_for_everyone()` barrier -- and from that moment the
    # two ranks are permanently one collective out of step. The first gradient
    # all-reduce of training then waits for a partner that will never come. What
    # you see is: validation prints correct numbers, the progress bar prints
    # `0/60`, and nothing ever happens again. No error, no traceback, both GPUs
    # at 100% (a rank spinning in an NCCL collective looks exactly like a rank
    # doing work).
    #
    # `unwrap_model` returns the plain HyperPrompt: same parameters, same device,
    # no hooks, no collectives. It is a no-op on one GPU.
    hypermod_eval = accelerator.unwrap_model(hypermod)

    def val_loss_fn(batch, **kw):
        return _loss(batch, hypermod_eval, **kw)

    # ---------------- gate: an m=0 prompt must reproduce the frozen model ----------------
    # Cheap, runs before any training, and catches a wrong injection point
    # immediately instead of 7 hours later. See baseline_prompt_tuning's
    # --no-soft-prompt check, of which this is the training-side twin.
    with torch.no_grad():
        b = {k: v.to(device) for k, v in probe.items()}
        zero_m = torch.zeros(b["input_ids"].shape[0], 0, emb_w.shape[1], device=device)
        l_none = loss_fn(b, override_prompt=zero_m)["sft_loss"].item()
        ref = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        from hyper_llm_modulator.sft_trainer import compute_loss
        l_ref = compute_loss(b["labels"], ref.logits,
                             equally_weight_sample=args.equally_weight_sample,
                             label_smoothing=args.label_smoothing).item()
        assert abs(l_none - l_ref) < 1e-4, (
            f"m=0 path does not reduce to a plain forward pass: {l_none} vs {l_ref}. "
            "The injection point is wrong; do not train."
        )
        if is_main:
            logger.info(f"[gate] m=0 reduces to plain forward ({l_none:.6f}) "
                        "-- injection point OK")

    # ---------------- train ----------------
    # DDP contract for this loop, stated once because every bug below came from
    # breaking it:
    #   * EVERY rank runs the same number of optimizer steps. curstep advances
    #     only when accelerator.sync_gradients is True, i.e. once per optimizer
    #     step, never once per micro-batch. All the `curstep % freq` triggers
    #     therefore fire on every rank at the same moment.
    #   * Only rank 0 validates, logs and writes checkpoints; every such block
    #     is followed by wait_for_everyone() so no rank runs ahead.
    #   * `done` is decided from curstep alone, which is identical on all ranks,
    #     so the loop cannot exit on one rank while another blocks in all-reduce.
    hypermod.train()
    if is_main:
        validate(model, hypermod_eval, val_dataloaders, val_loss_fn, curstep=0, is_main=True,
                 canary_samples=args.canary_samples,
                 canary_max_new_tokens=args.canary_max_new_tokens)
        save_checkpoint(save_dir, hypermod, args, 0, accelerator=accelerator)
    accelerator.wait_for_everyone()

    # NEFTune goes on LAST -- after the m=0 gate and after step-0 validation.
    # Both of those compare a Inlet forward against a plain frozen-model forward,
    # and NEFTune draws fresh noise on every call, so activating it any earlier
    # makes the gate compare two different random perturbations and fail.
    # Every rank activates: the hook is local and adds no collective.
    neftune_handle = activate_neftune(model, args.neftune_noise_alpha, is_main=is_main)

    best_val = float("inf")
    split_best = {}          # split -> (best sft_loss, step) for the final divergence report
    curstep, grad_norm = 0, 0.0
    avg = defaultdict(list)
    t0 = time.time()
    pbar = tqdm(total=num_training_steps, disable=not is_main)
    done = False
    while not done:
        for batch in train_dataloader:
            with accelerator.accumulate(hypermod), accelerator.autocast():
                info = loss_fn(batch)
                loss = info["sft_loss"] + info["prompt_l2_loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        hypermod.parameters(), args.max_grad_norm)
                # AcceleratedOptimizer.step()/zero_grad() and AcceleratedScheduler
                # .step() are no-ops while gradients are still accumulating, so
                # these four lines are correct at any grad_accum_steps. zero_grad
                # goes AFTER step -- putting it before backward() (as an earlier
                # revision did) throws away every micro-batch but the last.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k, v in info.items():
                avg[f"train/{k}"].append(float(v))

            # ---- everything below is per OPTIMIZER STEP, not per micro-batch --
            if not accelerator.sync_gradients:
                continue
            curstep += 1
            if is_main:
                pbar.update(1)
                pbar.set_description(f"loss {float(info['sft_loss']):.4f} "
                                     f"|P| {float(info['prompt_norm']):.2f} "
                                     f"sd {float(info['prompt_std_across_batch']):.3f}")

            if curstep % args.logging_freq == 0 or curstep == num_training_steps:
                logged = {k: sum(v) / len(v) for k, v in avg.items() if v}
                logged["train/grad_norm"] = float(grad_norm)
                logged["train/lr"] = scheduler.get_last_lr()[0]
                logged["train/steps_per_sec"] = curstep / max(time.time() - t0, 1e-9)
                # These are RANK-LOCAL averages (each rank sees its own tasks).
                # Cheap and good enough for a curve; the numbers that have to be
                # exact -- validation loss/accuracy -- come from rank 0 only.
                if is_main:
                    wandb.log(logged, step=curstep)
                    # prompt_std_across_batch is the collapse alarm: if the
                    # generator stops distinguishing descriptions it trends to 0
                    # long before any eval would show it.
                    sd = logged.get("train/prompt_std_across_batch", float("nan"))
                    # Relative to the prompt's own scale, not an absolute
                    # number. The absolute 1e-3 threshold this used to carry
                    # fired on every line of the known-good 12-dataset run
                    # (README records 2.3e-05 -> 1.26e-04), i.e. it was a
                    # permanently-on alarm, which is the same as no alarm.
                    # `prompt_std_across_batch` is bounded by the prompt norm,
                    # so the scale-free quantity is the ratio.
                    pn = float(avg.get("train/prompt_norm", [0.0])[-1]) if avg.get(
                        "train/prompt_norm") else 0.0
                    rel = sd / pn if pn > 0 else float("nan")
                    if rel == rel and rel < 1e-4:
                        logger.warning(
                            f"[step {curstep}] prompt_std_across_batch={sd:.2e} "
                            f"is {rel:.1e} of |P|={pn:.3f} -- the generator is "
                            "emitting nearly the same prompt for every description")
                avg = defaultdict(list)

            if curstep in keep_steps and is_main:
                kp = save_checkpoint(save_dir, hypermod, args, curstep,
                                     extra={"scale_curve_point": True},
                                     accelerator=accelerator,
                                     filename=f"hypermod_inlet_step{curstep}.pt")
                logger.info(f"[step {curstep}] permanent checkpoint -> {kp}")

            if curstep % args.val_freq == 0 or curstep == num_training_steps:
                if is_main:
                    vi = validate(model, hypermod_eval, val_dataloaders, val_loss_fn,
                                  curstep, is_main=True,
                                  canary_samples=args.canary_samples,
                                  canary_max_new_tokens=args.canary_max_new_tokens)
                    # NAMED, never positional. This was `next(iter(vi))`, i.e.
                    # whichever split upstream's dict happened to list first.
                    if args.model_select_split not in vi:
                        raise RuntimeError(
                            f"--model_select_split={args.model_select_split!r} is not one "
                            f"of the validation splits {list(vi)}. Refusing to fall back "
                            f"to an arbitrary one -- that is the bug this replaced."
                        )
                    cur = vi[args.model_select_split].get("sft_loss", float("inf"))
                    # Keep the best checkpoint for EVERY split, not just the
                    # selected one. Three files at 55 MB each against a run that
                    # costs days: there is no reason to make anyone choose in
                    # advance, and the 4,000 vs 130,000 step comparison showed the
                    # choice is worth 2.77 points of 10-task average.
                    #
                    # This does NOT make the choice for you. Decide which split
                    # the paper reports BEFORE looking at benchmark scores --
                    # keeping all three and then reporting whichever scored best
                    # downstream is selection on the test set. val/unseen is the
                    # only one that touches no benchmark data.
                    for _sp, _m in vi.items():
                        _l = _m.get("sft_loss", float("inf"))
                        if _l < split_best.get(_sp, (float("inf"), -1))[0]:
                            split_best[_sp] = (_l, curstep)
                            if args.save_best_per_split:
                                _fn = "hypermod_inlet_best_" + _sp.replace("/", "_") + ".pt"
                                save_checkpoint(save_dir, hypermod, args, curstep,
                                                extra={"best_split": _sp, "best_loss": _l,
                                                       "val": vi},
                                                accelerator=accelerator, filename=_fn)
                                logger.info(f"[step {curstep}] new best {_sp} {_l:.4f} -> {_fn}")
                    if cur < best_val:
                        best_val = cur
                        save_checkpoint(save_dir, hypermod, args, curstep,
                                        extra={"best_val_loss": best_val, "val": vi,
                                               "model_select_split": args.model_select_split},
                                        accelerator=accelerator)
                        logger.info(f"[step {curstep}] new best {args.model_select_split} "
                                    f"{best_val:.4f} -- checkpoint saved")
                accelerator.wait_for_everyone()
                # Decided on rank 0 and BROADCAST. Reading the file on every
                # rank does not make the ranks agree -- it makes each rank form
                # its own opinion from its own filesystem view at its own
                # moment. On a non-shared filesystem `save_dir` does not even
                # exist on nodes >= 1 (only global rank 0 creates it), so node 0
                # would break out to wait_for_everyone() while node 1 posted the
                # next gradient all-reduce: barrier vs all-reduce on the same
                # communicator, i.e. the silent-hang failure again. NFS
                # attribute caching reintroduces it even on a shared FS.
                stop = torch.tensor(
                    [1 if (is_main and os.path.isfile(f"{save_dir}/earlystop_info.yaml"))
                     else 0],
                    device=accelerator.device,
                )
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(
                        stop, op=torch.distributed.ReduceOp.MAX
                    )
                if stop.item():
                    if is_main:
                        logger.info("early stop signal")
                    done = True
                    break

            if curstep >= num_training_steps:
                done = True
                break
    pbar.close()

    accelerator.wait_for_everyone()
    if is_main:
        path = save_checkpoint(save_dir, hypermod, args, curstep,
                               extra={"final": True}, accelerator=accelerator)
        # WHERE EACH SPLIT BOTTOMED OUT.
        #
        # The rolling checkpoint follows ONE split. On the 147.5k-step run those
        # optima were 100k steps apart -- val/seen at ~130k, val/benchmark at
        # ~30k -- so the saved checkpoint was 9.9% worse in benchmark loss than a
        # checkpoint the run had already passed through. Nothing said so at the
        # time: the loss curve looked healthy, because the split being watched
        # really was still improving.
        #
        # Printing all three, with the step each was best at, is what makes that
        # visible without reading a wandb chart afterwards.
        sel = args.model_select_split
        if split_best:
            logger.info("validation optima by split (the checkpoint follows %s):", sel)
            for _sp, (_l, _st) in split_best.items():
                mark = "  <- selected on this" if _sp == sel else ""
                logger.info("    %-14s best sft_loss %.4f at step %d%s", _sp, _l, _st, mark)
            sel_step = split_best.get(sel, (None, None))[1]
            elsewhere = [(sp, st) for sp, (_l, st) in split_best.items()
                         if sp != sel and st != sel_step]
            if elsewhere:
                logger.warning(
                    "SAVED CHECKPOINT IS NOT THE BEST ON EVERY SPLIT. Selected on "
                    "%s (step %s); %s. If the number you report comes from a split "
                    "you did not select on, re-evaluate the --checkpoint_steps "
                    "checkpoint nearest ITS optimum, or rerun with "
                    "--model_select_split set to it.",
                    sel, sel_step,
                    "; ".join(f"{sp} was best at step {st}" for sp, st in elsewhere),
                )
        with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
            json.dump({"run_name": args.run_name, "steps": curstep,
                       "best_val_loss": best_val, "wall_s": time.time() - t0,
                       "model_select_split": sel,
                       "val_optima_by_split": {k: {"sft_loss": v[0], "step": v[1]}
                                               for k, v in split_best.items()},
                       "world_size": accelerator.num_processes,
                       "grad_accum_steps": accum,
                       "global_tasks_per_step": global_tasks,
                       "trainable_params": accelerator.unwrap_model(hypermod).num_trainable()},
                      f, indent=2)
        logger.info(f"done in {(time.time() - t0) / 3600:.2f}h -- {path}")
    accelerator.wait_for_everyone()
    if neftune_handle is not None:
        neftune_handle.remove()          # upstream sft_trainer.py:313
        model.get_input_embeddings().eval()
    accelerator.end_training()


if __name__ == "__main__":
    # ONE dataclass only. Upstream parse_yaml_and_args() raises on the first
    # CLI override it cannot find in the dataclass it is currently looping over,
    # so two dataclasses can never both take CLI overrides. Subclassing is the
    # only shape that works.
    parser = ArgumentParser((InletArguments,))
    args = parser.parse()   # one dataclass in -> the object itself out, not a tuple
    # save_dir and run_name are NOT dataclass fields upstream -- train_custom_sft.py
    # sets them on the namespace after parsing:
    #     args.run_name = time.strftime(...) + f"_{uuid}"
    #     args.save_dir = f"train_outputs/sft/{args.exp_setup}/{args.run_name}"
    # Inlet keeps run_name a real, user-settable field instead of a timestamp+uuid, so
    # that eval_inlet can find a run again by name. save_dir is still derived here, and
    # has to exist before main() because create_logger writes debug.log into it.
    args.save_dir = os.path.join(output_root(), str(args.exp_setup), args.run_name)
    # Under torchrun every rank reaches this line. create_logger() makes the
    # directory and opens debug.log for writing; letting 8 ranks do that
    # interleaves the file and races on mkdir. Rank 0 owns the log; the others
    # keep the module-level no-op logger, and every logger.* call inside main()
    # is already behind `if is_main`.
    if int(os.environ.get("RANK", "0")) == 0:
        logger = create_logger(args.save_dir, debug=args.debug)
    main(args)
