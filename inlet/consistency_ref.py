"""Reconstruction of `baseline_prompt_tuning/test_consistency.py`.

Same provenance and caveats as `inlet/baseline_ref.py`: written from the reference
at the reference implementation's stated contract, not copied from disk, not diffed against the
original. The real module wins whenever it is importable.

Unlike `baseline_ref`, this one is COMPLETE -- both checks were visible end to
end. `inlet/test_consistency_inlet.py` extends them rather than replacing them.

From the reference's own header:

    Train/eval consistency checks that need no GPU.

    The expensive gate is the zero-shot reproduction run (eval_soft_prompt.py
    --no-soft-prompt). This file catches the cheap failure modes first: label
    masking, and whether the prompt the model is trained on is token-for-token
    the prompt it is evaluated on.
"""

__all__ = ["SOURCE", "check_label_masking", "check_prompt_matches_eval"]

SOURCE = "inlet.consistency_ref (reconstructed from the reference)"


def check_label_masking(tokenizer, sample) -> int:
    """The prompt must be fully masked and the response fully supervised.

    Returns the index of the first supervised token, i.e. the prompt length.

    Three distinct failures, three distinct assertions:
      * nothing supervised at all -> completion masking is inverted or empty;
      * supervision is not a contiguous suffix -> the mask is on the wrong span,
        which no loss curve would reveal;
      * a supervised label differing from its input id -> the labels were shifted
        somewhere they should not have been.
    """
    ids, labels = sample["input_ids"], sample["labels"]
    assert len(ids) == len(labels), (len(ids), len(labels))
    supervised = [i for i, lab in enumerate(labels) if lab != -100]
    assert supervised, "no supervised tokens -- completion masking is broken"
    first = supervised[0]
    assert supervised == list(range(first, len(ids))), "supervision is not a suffix"
    for i in supervised:
        assert labels[i] == ids[i], f"label != input at {i}"
    return first


def check_prompt_matches_eval(tokenizer, task, sample, prompt_len):
    """Training prompt tokens vs what fishfarm's `_into_prompt` would produce.

    Eval builds `apply_chat_template([system, user], add_generation_prompt=True)`
    and vLLM then tokenizes that string with add_special_tokens=True (verified in
    vllm/inputs/preprocess.py: only whisper overrides it). Training builds the
    same conversation with add_generation_prompt=False and tokenizes with
    add_special_tokens=False, because the repo's chat template emits `bos_token`
    itself. Those two must land on the same tokens modulo that extra BOS.
    """
    from hyper_llm_modulator.utils.task_metadata import get_metadata_for_task

    metadata = get_metadata_for_task(task)
    train_prompt_ids = sample["input_ids"][:prompt_len].tolist()
    train_text = tokenizer.decode(train_prompt_ids)

    # rebuild the eval-side string from the same conversation
    eval_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": metadata["system_message"]},
            {"role": "user", "content": "<<<USER>>>"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    eval_ids = tokenizer(eval_text, add_special_tokens=True)["input_ids"]

    # vLLM's add_special_tokens=True prepends a BOS on top of the template's own
    # literal <s>. That double BOS is upstream behaviour, shared with every other
    # column in the table, so we reproduce it rather than "fix" it.
    double_bos = eval_ids[0] == tokenizer.bos_token_id == eval_ids[1]
    return train_text, eval_text, double_bos


# --------------------------------------------------------------------------- #
# `check_label_masking` is pure Python -- no torch, no tokenizer, no model -- so
# it can be exercised directly, with controls:
#
#     python -m inlet.consistency_ref
#
# `check_prompt_matches_eval` cannot be: it needs a real tokenizer and upstream's
# task metadata. It is covered by `test_consistency_inlet.py`, which runs on a box
# that has both.
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    class _Tok:  # check_label_masking never touches the tokenizer
        pass

    good = {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}
    first = check_label_masking(_Tok(), good)
    assert first == 2, first
    print(f"ok    a well-formed sample reports prompt_len={first}")

    # every failure this function exists to catch, each of which must raise
    controls = {
        "nothing supervised (inverted or empty completion mask)":
            {"input_ids": [1, 2, 3], "labels": [-100, -100, -100]},
        "supervision is not a suffix (mask on the wrong span)":
            {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, -100, 4]},
        "a supervised label does not match its input id (labels shifted)":
            {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 9, 4]},
        "labels and input_ids are different lengths":
            {"input_ids": [1, 2, 3], "labels": [-100, 2]},
    }

    bad = 0
    for name, sample in controls.items():
        try:
            check_label_masking(_Tok(), sample)
        except AssertionError:
            continue
        print(f"FAIL  control not caught -- {name}")
        bad += 1
    if not bad:
        print(f"ok    all {len(controls)} deliberately-broken samples raise")

    # A fully-supervised sample is legal (prompt_len 0) and must NOT raise --
    # otherwise the check would reject the no-prompt zero-shot configuration.
    n = check_label_masking(_Tok(), {"input_ids": [7, 8], "labels": [7, 8]})
    if n != 0:
        print(f"FAIL  fully-supervised sample reported prompt_len={n}, expected 0")
        bad += 1
    else:
        print("ok    a fully-supervised sample is accepted with prompt_len=0")

    print()
    if bad:
        print(f"FAILED  {bad} check(s).")
        return 1
    print(f"PASS  {SOURCE} -- check_label_masking behaves as the reference describes")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
