"""Self-contained versions of the four things eval_inlet.py needs.

Why this file exists: `eval_inlet.py` originally imported `get_tokenizer`,
`load_input_embeddings`, `BASE_MODEL` and `ZERO_SHOT` from the reference implementation's own
`baseline_prompt_tuning/` directory. That was deliberate -- it guaranteed the
Inlet eval and the per-task prompt-tuning eval could not drift apart, which
matters because the whole comparison rests on them injecting at the same point.

But it also made the repo unrunnable for anyone who does not have that
directory. So:

    * if `baseline_prompt_tuning/` is importable, we import from it and the
      no-drift guarantee holds exactly as before;
    * otherwise we fall back to the definitions here, which are written to be
      behaviourally identical.

Set INLET_BASELINE_DIR to point at the directory if it lives somewhere unusual.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = os.environ.get("INLET_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Published zero-shot accuracy of the frozen base model, used by the
# `--zero-prompt` gate: prepending zero vectors must reproduce these exactly.
# Only tasks listed here are checked, so an incomplete dict is safe -- it just
# checks less. arc_challenge is the one value reproduced end to end in this
# repo (got 65.70, expected 65.70). Add your own as you measure them.
# Transcribed from the reference baseline_prompt_tuning/tasks_config.py so that
# `./scripts/eval.sh --zero-prompt` is a real gate on every task, not just on
# arc_challenge. When that directory IS importable its own values win (see the
# import at the bottom of this file) -- these are the fallback for a machine
# that does not have it.
#
# humaneval is absent on purpose: upstream reports it as N/A for this model.
ZERO_SHOT = {
    "arc_challenge": 65.70,
    "arc_easy": 77.48,
    "boolq": 71.56,
    "hellaswag": 49.67,
    "openbookqa": 55.00,
    "piqa": 73.01,
    "winogrande": 45.54,
    "gsm8k": 40.71,
    "mbpp": 44.44,
}

# Smallest training set first, so a broken pipeline surfaces in the first job.
TASK_ORDER = [
    "arc_challenge", "openbookqa", "arc_easy", "winogrande",
    "gsm8k", "boolq", "piqa", "hellaswag", "mbpp",
]
SEEDS = [0, 1, 2]
_zs = os.environ.get("INLET_ZERO_SHOT_JSON")
if _zs and os.path.isfile(_zs):
    ZERO_SHOT.update(json.load(open(_zs)))


# Upstream's own tokenizer factory, not a re-implementation.
#
# This matters more than it looks. The first version of this file built an
# AutoTokenizer by hand -- reasonable-looking code, and WRONG in a way that only
# a number shows: `./scripts/eval.sh --zero-prompt` scored arc_challenge 65.61
# against a recorded 65.70. One question out of 1172. No error, no warning; just
# a slightly lower score, which is exactly how a broken comparison hides.
#
# The reference baseline_prompt_tuning/soft_prompt_common.py re-exports
# `hyper_llm_modulator.utils.get_tokenizer`, so importing the same function is
# the only way the two eval paths can be identical rather than merely similar --
# padding side, pad token, chat template and any future upstream change come
# along for free. `hyper_llm_modulator` is a hard dependency of this repo
# already, so there is no cost to depending on it here.
try:
    from hyper_llm_modulator.utils import get_tokenizer  # type: ignore # noqa: F401
except ImportError:  # last resort only -- see the warning above
    def get_tokenizer(model_dir: str = BASE_MODEL, train: bool = False, **_kw):
        import warnings

        from transformers import AutoTokenizer

        warnings.warn(
            "hyper_llm_modulator.utils.get_tokenizer is not importable, so eval "
            "is using a hand-rolled AutoTokenizer. This is known to shift scores "
            "by ~0.1 point and invalidates any comparison against the "
            "prompt-tuning baseline. Fix T2L_ROOT instead of trusting the number.",
            RuntimeWarning,
            stacklevel=2,
        )
        tok = AutoTokenizer.from_pretrained(model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok


def load_input_embeddings(model_dir: str = BASE_MODEL):
    """-> the input embedding matrix [vocab, d] on CPU, without loading the LLM.

    Reads only the shard that actually holds `model.embed_tokens.weight`, via
    the safetensors index. Loading the whole 7B model just to read one matrix
    costs ~15GB of RAM and a minute; this costs ~0.5GB and a second.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    key = "model.embed_tokens.weight"

    def _local(path):
        return path if os.path.isdir(path) else None

    local = _local(model_dir)

    def fetch(fn):
        if local:
            p = os.path.join(local, fn)
            return p if os.path.isfile(p) else None
        try:
            return hf_hub_download(model_dir, fn)
        except Exception:
            return None

    idx = fetch("model.safetensors.index.json")
    if idx:
        shard = json.load(open(idx))["weight_map"][key]
    else:
        shard = "model.safetensors"
    path = fetch(shard)
    if path is None:
        raise FileNotFoundError(
            f"could not find {shard} for {model_dir}. If this is a local path, it must "
            f"contain safetensors weights; if a hub id, you may need to log in."
        )
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# ---------------------------------------------------------------------------
# Resolution order, best first. Whichever wins, `source()` says so out loud --
# the point of this block is that a weaker configuration is visible, not silent.
#
#   1. baseline_prompt_tuning/   the reference implementation's own files. Inlet eval and the
#                                prompt-tuning eval then cannot drift, because
#                                they are literally the same functions.
#   2. inlet.baseline_ref          a reconstruction of (1) written from the
#                                reference. Runs standalone; NOT diffed against
#                                the original.
#   3. this file                 last resort. Only reached if even the
#                                reconstruction fails to import.
_SOURCE = "inlet.eval_common (last resort)"
try:
    from soft_prompt_common import (  # type: ignore  # noqa: F401
        HERE,
        get_tokenizer,
        load_input_embeddings,
    )
    from tasks_config import (  # type: ignore  # noqa: F401
        BASE_MODEL,
        SEEDS,
        TASK_ORDER,
        ZERO_SHOT,
    )

    _SOURCE = "baseline_prompt_tuning (the reference implementation's own files)"
except ImportError:
    try:
        from inlet.baseline_ref import load_input_embeddings  # noqa: F401

        _SOURCE = "inlet.baseline_ref (reconstructed from the reference, not diffed)"
    except ImportError:
        pass


def source() -> str:
    """Which implementation is live. Print it before trusting a comparison.

    "baseline_prompt_tuning" means eval is running the same tokenizer, embedding
    loader and reference numbers that produced the per-task prompt-tuning column
    (74.89 arc_c / 76.33 8-task avg) -- the comparison is like-for-like.

    Anything else means Inlet and that baseline are being scored by two
    implementations that were never checked against each other. The numbers are
    still real; the *comparison* carries an asterisk, and a divergence in
    tokenizer settings, prompt position or label shift would show up as "a
    slightly lower score", never as an error.
    """
    return _SOURCE
