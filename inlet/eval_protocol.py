"""Shared, dependency-light input audit and raw-output capture for Inlet/T2L."""
import hashlib
import json
from pathlib import Path

from inlet.format_ablation import text_field, _replace_text


def audit_icl(evaluator, task, use_icl, examples):
    """Verify actual sample inputs; repair ONLY upstream GSM8K's dropped prefix.

    Already corrected checkouts are supported without double-prepending. Missing
    examples fail closed instead of calling an empty HumanEval prefix '3-shot'.
    """
    prefix = examples.get(task, "") if use_icl else ""
    if use_icl and not prefix.strip():
        raise ValueError(f"{task}: requested ICL but upstream examples are empty")
    if not evaluator.samples:
        raise ValueError("cannot audit an empty evaluator")
    repairs = 0
    texts = []
    for i, sample in enumerate(evaluator.samples):
        field = text_field(sample)
        text = getattr(sample, field)
        if prefix and not text.startswith(prefix):
            if task != "gsm8k":
                raise ValueError(f"{task}: sample {i} does not start with the ICL examples")
            text = prefix + "\n\n" + text
            evaluator.samples[i] = _replace_text(sample, field, text)
            repairs += 1
        live = getattr(evaluator.samples[i], field)
        if prefix and not live.startswith(prefix):
            raise RuntimeError("ICL repair did not persist")
        texts.append(live)
    return {
        "use_icl": use_icl, "icl_nonempty": bool(prefix),
        "icl_text": prefix, "icl_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "gsm8k_samples_repaired": repairs, "n_samples": len(texts),
        "sample_texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        "first_sample_text": texts[0],
        "protocol": "effective_icl_v1" if use_icl else "no_icl_v1",
    }


def evaluate_recorded(evaluator, model, raw_path=None):
    """Record full rendered inputs and unsanitized generations before scoring.

    Files use exclusive creation: a retry cannot silently append to an old run.
    A completed metric JSON (written by the caller) is the completion marker.
    """
    if raw_path is None:
        return evaluator.evaluate(model)
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = model.generate
    count = 0
    with path.open("x") as stream:
        def recorded(requests, *args, **kwargs):
            nonlocal count
            for result in original(requests, *args, **kwargs):
                rendered = model._into_prompt(result.request.messages)
                stream.write(json.dumps({
                    "index": count,
                    "rendered_prompt": rendered,
                    "input_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "output": result.generation,
                }, ensure_ascii=False) + "\n")
                stream.flush()
                count += 1
                yield result
        model.generate = recorded
        try:
            result = evaluator.evaluate(model)
        finally:
            model.generate = original
    if count == 0:
        raise RuntimeError("raw capture recorded no generations; do not report this run")
    return result
