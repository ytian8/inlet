"""Locate the upstream text-to-lora checkout and put it on sys.path.

Inlet is an OVERLAY on SakanaAI/text-to-lora, not a fork. Everything upstream
owns -- task metadata, the hierarchical sampler, the 479/31 decontaminated
split, the collator, description pre-embedding, the vLLM eval harness -- is
imported, never copied. That is a deliberate choice: it is what lets us say our
numbers are comparable to theirs.

So every entry point needs to answer one question: where is text-to-lora?
Three answers, in priority order:

  1. $T2L_ROOT is set                       -- explicit, use it
  2. this repo IS the checkout              -- ./src/hyper_llm_modulator exists
  3. a sibling or parent directory is       -- walk up and across looking for it

Upstream also expects to be run from its own root: config paths in the YAML,
the `data/transformed_datasets` cache and `models/` are all relative. So
`bootstrap()` chdir's there. Anything Inlet writes goes to $INLET_OUTPUT_ROOT
instead, which defaults to <this repo>/train_outputs so a `git status` in the
upstream checkout stays clean.
"""

import os
import sys

# The directory the user ran the command from. Captured at import time, BEFORE
# bootstrap() chdir's into the upstream checkout, because after that chdir every
# relative path the user typed on the command line points somewhere else.
#
# This is not hypothetical: `python -m inlet.probe_prompt --checkpoint
# train_outputs/.../hypermod_inlet.pt` -- the exact form the README documents --
# failed with FileNotFoundError for precisely this reason. Any CLI argument that
# names a file the USER chose must go through user_path().
INVOCATION_CWD = os.getcwd()


def user_path(p):
    """Resolve a user-supplied path against the directory they ran the command in.

    Absolute paths and None pass through unchanged.
    """
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(INVOCATION_CWD, p))

INLET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MARKER = os.path.join("src", "hyper_llm_modulator")


def _is_t2l(path: str) -> bool:
    return os.path.isdir(os.path.join(path, _MARKER))


def find_t2l_root() -> str:
    env = os.environ.get("T2L_ROOT")
    if env:
        env = os.path.abspath(os.path.expanduser(env))
        if not _is_t2l(env):
            raise SystemExit(
                f"T2L_ROOT={env} does not look like a text-to-lora checkout "
                f"(no {_MARKER}/ inside it)."
            )
        return env

    candidates = [INLET_ROOT]
    cur = INLET_ROOT
    for _ in range(3):  # up to three levels up, plus every sibling on the way
        cur = os.path.dirname(cur)
        candidates.append(cur)
        candidates.append(os.path.join(cur, "text-to-lora"))
        candidates.append(os.path.join(cur, "text-to-lora-main"))
    candidates.append(os.path.join(INLET_ROOT, "third_party", "text-to-lora"))

    for c in candidates:
        if _is_t2l(c):
            return os.path.abspath(c)

    raise SystemExit(
        "Cannot find the text-to-lora checkout.\n"
        "Inlet is an overlay on https://github.com/SakanaAI/text-to-lora and needs\n"
        "hyper_llm_modulator importable. Either clone it and export the path:\n"
        "    export T2L_ROOT=/path/to/text-to-lora\n"
        "or place this repo next to / inside that checkout.\n"
        f"Looked in: {', '.join(candidates)}"
    )


def baseline_dir(t2l_root: str) -> str:
    """Your own per-task prompt-tuning baseline directory.

    eval_inlet.py imports get_tokenizer / load_input_embeddings / BASE_MODEL /
    ZERO_SHOT from it on purpose: Inlet's eval and the prompt-tuning baseline's
    eval must not be allowed to drift, since the whole comparison rests on them
    injecting at the same point. TRAINING does not need this directory.
    """
    d = os.environ.get("INLET_BASELINE_DIR") or os.path.join(t2l_root, "baseline_prompt_tuning")
    return os.path.abspath(os.path.expanduser(d))


def output_root() -> str:
    return os.path.abspath(os.path.expanduser(
        os.environ.get("INLET_OUTPUT_ROOT") or os.path.join(INLET_ROOT, "train_outputs")
    ))


def bootstrap(chdir: bool = True, need_baseline: bool = False) -> str:
    """Put upstream on sys.path, optionally chdir into it. Returns T2L_ROOT."""
    root = find_t2l_root()
    paths = [INLET_ROOT, os.path.join(root, "src"), os.path.join(root, "src", "fishfarm")]
    # The baseline directory is added whenever it exists, so that anything able
    # to prefer the reference originals does so; it is only REQUIRED when the
    # caller says it is (the train/eval consistency test, which has nothing to
    # compare against without it).
    b = baseline_dir(root)
    if os.path.isdir(b):
        paths.append(b)
    elif need_baseline:
        raise SystemExit(
            f"This entry point compares against the per-task prompt-tuning baseline,\n"
            f"and that directory is not here:\n"
            f"    {b}\n"
            f"Set INLET_BASELINE_DIR to point at it, or skip this check -- everything\n"
            f"else in the repo runs without it."
        )
    for p in paths:
        # src/fishfarm is a NESTED checkout: the package is at
        # src/fishfarm/fishfarm, so `src` alone does not make it importable.
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("T2L_ROOT", root)
    if chdir:
        os.chdir(root)
    return root
