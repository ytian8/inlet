"""A validation split that contains long-generation tasks.

Why this file exists. `val/seen`, `val/unseen` and `val/benchmark` between them
cover ten held-out lol_* tasks, eleven more, and seven multiple-choice
benchmarks. **None of them contains gsm8k, mbpp or humaneval** -- upstream's
`BENCHMARK_TASK_INFO` has seven keys and those three are not among them.

Those are exactly the three tasks that collapsed. On the 147,500-step run the
seven multiple-choice tasks moved 0.73 points between step 4,000 and step
130,000 while the three generative tasks moved 10.96, and no instrument in the
training loop was pointed at them for any of it. `val/benchmark` read 0.80-0.82
the whole way because it was measuring the tasks that were fine.

This adds `val/generative`, built with upstream's own public functions -- no fork,
no copied dataloader.

## Which tasks, and which splits

**gsm8k**: `train`, which the harness never scores (it scores `test`). Long
chain-of-thought targets and a non-empty `assistant_prefill`, so it exercises the
shape the collapsing tasks have.

**mbpp**: opt in, and only with decontamination. `mbpp/sanitized` train is 120
problems and **108 of them are inside MBPP+**, which is the set the harness
scores. Putting that in validation is selection on scored data. `--generative_val_tasks`
accepts it, and the build refuses unless the MBPP+ ids can be loaded and removed.

**humaneval**: cannot be here at all. 164 problems, no train split. Any use of it
for selection is selection on the scored set.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ["SAFE_SPLITS", "vet_tasks", "build_generative_val_dataloader"]

# Splits the eval harness does NOT score, so validating on them is not selection
# on scored data. Checked against vllm_eval.DS_KWARGS, which scores gsm8k `test`.
SAFE_SPLITS = {"gsm8k": "train"}

# Never, under any flag: no train split exists.
FORBIDDEN = {"humaneval"}


def _decontaminated_mbpp_ids():
    """task_ids in mbpp train that are NOT in MBPP+, or None if unknowable.

    Returning None is a refusal, not a fallback: without the MBPP+ ids there is
    no way to tell a clean problem from a scored one, and guessing would put
    scored data into model selection silently.
    """
    try:
        from evalplus.data import get_mbpp_plus
    except Exception as exc:
        logger.warning("mbpp excluded from val/generative: cannot import evalplus (%s)", exc)
        return None
    try:
        plus = get_mbpp_plus()
    except Exception as exc:
        logger.warning("mbpp excluded from val/generative: MBPP+ unavailable (%s)", exc)
        return None
    ids = set()
    for k in plus:
        tail = str(k).rsplit("/", 1)[-1]
        if tail.isdigit():
            ids.add(int(tail))
    if not ids:
        logger.warning("mbpp excluded from val/generative: MBPP+ yielded no task ids")
        return None
    logger.info("MBPP+ contains %d task ids; those are excluded from val/generative", len(ids))
    return ids


def vet_tasks(val_metadata, task_names):
    """-> (names, metadata) for the tasks that may safely be validated on.

    Pure: no datasets, no model, no upstream import, so `python -m
    inlet.generative_val` can check it. This is where the leakage rules live, and
    they are rules rather than warnings: a task with no vetted split is dropped,
    not guessed at, because the failure mode is putting scored data into model
    selection where nothing downstream would notice.
    """
    from copy import deepcopy

    names, meta = [], {}
    for t in task_names:
        if t in FORBIDDEN:
            raise ValueError(
                f"{t!r} cannot be a validation task: it has no train split, so validating "
                f"on it is selection on the data the harness scores."
            )
        if t not in val_metadata:
            logger.warning("val/generative: %r not in eval_ds_info, skipping", t)
            continue
        m = deepcopy(val_metadata[t])
        if t in SAFE_SPLITS:
            m["ds_kwargs"]["split"] = SAFE_SPLITS[t]
        elif t == "mbpp":
            if _decontaminated_mbpp_ids() is None:
                continue          # already logged why
            m["ds_kwargs"]["split"] = "train"
        else:
            logger.warning(
                "val/generative: no vetted split for %r -- skipping rather than "
                "guessing one the harness might also score", t)
            continue
        names.append(t)
        meta[t] = m
    return names, meta


def build_generative_val_dataloader(
    args, val_metadata, tokenizer, is_intx_model,
    emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device,
    task_names=("gsm8k",),
):
    """-> a DataLoader over long-generation tasks, or None if none are usable.

    Uses upstream's `get_datasets`, `get_embs_dict` and `get_dataloader` with the
    same arguments `create_dataloaders` binds, so this split is built the way the
    other three are and cannot drift from them.
    """
    from hyper_llm_modulator.data import get_dataloader, get_datasets, get_embs_dict

    names, meta = vet_tasks(val_metadata, task_names)

    if not names:
        logger.warning("val/generative: no usable tasks, split not built")
        return None

    logger.info("val/generative: %s", ", ".join(f"{n}[{meta[n]['ds_kwargs']['split']}]" for n in names))

    ds_dict = get_datasets(
        names, meta, tokenizer=tokenizer, sft_mode=args.sft_mode,
        is_intx_model=is_intx_model, inp_max_len=args.inp_max_len,
    )
    embs = get_embs_dict(
        args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
        names, meta, device,
    )
    return get_dataloader(
        ds_dict, embs, tokenizer=tokenizer,
        use_per_task_emb=args.use_per_task_emb,
        use_inp_as_desc=args.use_inp_as_desc,
        use_per_sample_desc=args.use_per_sample_desc,
        n_tasks_per_batch=args.n_tasks_per_batch,
        n_points_per_task=args.n_points_per_task,
        use_hierarchical_sampler=False,
        batch_size=args.val_batch_size,
        validation=True,
    )


# --------------------------------------------------------------------------- #
#     python -m inlet.generative_val
# Checks the leakage rules. No datasets, no model, under a second.
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    md = {
        "gsm8k":     {"ds_kwargs": {"name": "main", "path": "gsm8k", "split": "train"}},
        "mbpp":      {"ds_kwargs": {"name": "sanitized", "path": "g/mbpp", "split": "train"}},
        "humaneval": {"ds_kwargs": {"path": "openai/openai_humaneval", "split": "test"}},
        "boolq":     {"ds_kwargs": {"split": "train"}},
    }

    print("inlet.generative_val -- leakage rules")

    names, meta = vet_tasks(md, ("gsm8k",))
    check("gsm8k is accepted", names == ["gsm8k"], str(names))
    check("gsm8k validates on `train`, which the harness never scores",
          meta["gsm8k"]["ds_kwargs"]["split"] == "train",
          meta["gsm8k"]["ds_kwargs"]["split"])
    check("the caller's metadata is not mutated",
          md["gsm8k"]["ds_kwargs"]["split"] == "train")

    try:
        vet_tasks(md, ("humaneval",))
        check("humaneval is refused outright", False, "no exception")
    except ValueError as e:
        check("humaneval is refused outright", "no train split" in str(e))

    try:
        vet_tasks(md, ("gsm8k", "humaneval"))
        check("humaneval is refused even alongside a legal task", False, "no exception")
    except ValueError:
        check("humaneval is refused even alongside a legal task", True)

    n, _ = vet_tasks(md, ("not_a_task",))
    check("an unknown task is skipped, not invented", n == [], str(n))

    # KNOWN-BAD CONTROL: a task present in metadata but with no vetted split must
    # be DROPPED. Accepting it would validate on whatever split the metadata
    # happened to carry -- which for a benchmark task is often the scored one,
    # and nothing downstream would notice.
    n, _ = vet_tasks(md, ("boolq",))
    check("KNOWN-BAD CONTROL: a task with no vetted split is dropped, not guessed",
          n == [], f"got {n}")

    n, _ = vet_tasks(md, ("gsm8k", "not_a_task", "boolq"))
    check("mixed input keeps only the vetted task", n == ["gsm8k"], str(n))

    print()
    if fails:
        print(f"FAIL -- {len(fails)}: {fails}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
