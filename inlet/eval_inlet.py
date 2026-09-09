"""Score a Inlet checkpoint with the official T2L evaluation harness.

Structure is lifted from baseline_prompt_tuning/eval_soft_prompt.py: swap exactly
one function in-process (`vllm_eval.eval_model`) and let upstream own dataset
loading, chat templates, assistant_prefill, answer extraction and metrics.
Upstream files are not modified.

Two deliberate differences from the prompt-tuning version:

  1. The soft prompt is not a constant. Inlet produces one prompt per task
     DESCRIPTION, and every eval task carries several description splits
     (eval_descs / other_train_descs / random_descs / train_descs -- see
     utils/eval_hypermod.py). So this file restores the loop that upstream's
     eval_model has over `lora_dirs` and that eval_soft_prompt.py dropped:
     build the vLLM engine ONCE per task, then swap model.soft_prompt for each
     description. With ~9 descriptions per task and 31 eval tasks that is 289
     evaluations but only 31 engine constructions.

  2. `--zero-prompt` is the Inlet analogue of `--no-soft-prompt`. It must
     reproduce the zero-shot column exactly, and because the Inlet generator is
     zero-initialized at the output layer, an UNTRAINED checkpoint must also
     reproduce it. Both are checked.

Sanity check (no checkpoint needed):

    python -m inlet.eval_inlet --task arc_challenge --zero-prompt
"""

import argparse
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap, output_root, user_path  # noqa: E402
from inlet.format_ablation import MODES as FORMAT_MODES  # noqa: E402
from inlet.format_ablation import apply_to_evaluator  # noqa: E402
from inlet.sequence import build_eval_sequence  # noqa: E402

# need_baseline=False: eval prefers the prompt-tuning baseline's own tokenizer
# and embedding loaders (so the two injection paths cannot drift), but does not
# REQUIRE them -- inlet.eval_common falls back to equivalent definitions when the
# directory is absent, which is what makes this repo runnable on its own.
T2L_ROOT = bootstrap(need_baseline=False)

import torch  # noqa: E402
import vllm  # noqa: E402
from fishfarm.models.base import GenerationResult  # noqa: E402
from fishfarm.models.vllm_model import VLLMModel  # noqa: E402

import hyper_llm_modulator.vllm_eval as vllm_eval  # noqa: E402

from inlet.eval_common import (  # noqa: E402
    BASE_MODEL,
    ZERO_SHOT,
    get_tokenizer,
    load_input_embeddings,
    source as eval_impl_source,
)


class SoftPromptVLLMModel(VLLMModel):
    """fishfarm model that prepends vectors instead of loading an adapter.

    Copied from baseline_prompt_tuning/eval_soft_prompt.py with one change:
    `soft_prompt` is a mutable attribute so the caller can swap it between
    evaluator.evaluate() calls without rebuilding the engine.
    """

    def __init__(self, soft_prompt, embed_weight, prefill_text=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if prefill_text:
            base_into_prompt = self._into_prompt
            self._into_prompt = lambda messages: base_into_prompt(messages) + prefill_text
        self.soft_prompt = soft_prompt
        self.embed_weight = embed_weight
        # vLLM tokenizes text prompts with add_special_tokens=True; we must match
        # it exactly or the BOS handling silently diverges from the other columns.
        self.add_special_tokens = True

    def set_soft_prompt(self, soft_prompt):
        """[m, d] or a 0-row tensor. Engine is untouched."""
        self.soft_prompt = soft_prompt

    def generate(self, requests):
        tokenizer = self.get_tokenizer()
        inputs = []
        for request in requests:
            prompt = self._into_prompt(request.messages)
            ids = tokenizer(prompt, add_special_tokens=self.add_special_tokens)["input_ids"]
            embeds = build_eval_sequence(self.soft_prompt, self.embed_weight[ids])
            inputs.append({"prompt_embeds": embeds})

        completions = self.llm.generate(inputs, sampling_params=self.sampling_params)
        for request, completion in zip(requests, completions):
            yield GenerationResult(request=request, generation=completion.outputs[0].text)


# One vLLM engine per (model, memory fraction), reused across tasks in the same
# process. Building it costs minutes; scoring a small task costs seconds, so a
# sweep over tasks otherwise spends nearly all its wall-clock on rebuilds.
_ENGINES: dict = {}


def _engine(model_dir, gpu_memory_utilization):
    key = (model_dir, round(float(gpu_memory_utilization), 3))
    if key not in _ENGINES:
        _ENGINES[key] = vllm.LLM(
            model_dir,
            seed=42,
            max_model_len=2**12,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
        )
    return _ENGINES[key]


def synthetic_prompts(kinds, embed_weight, n_tokens: int, seed: int) -> dict:
    """Untrained prompts of the right shape, to separate two explanations.

    "The 32 prepended vectors hurt generation" and "training taught the prompt
    to be terse" both predict a low humaneval score for Inlet. They differ on an
    untrained prompt:

      vocab  32 rows sampled from the frequent vocab -- byte-identical to
             `HyperPrompt.init_base_from_vocab`, i.e. the model at step 0.
      gauss  Gaussian noise rescaled to the same per-token norm, controlling
             for magnitude without any real token direction.

    If these land near the zero-prompt score, prefix presence is harmless and
    the damage is in what training put there.
    """
    out = {}
    for kind in kinds:
        if kind == "vocab":
            g = torch.Generator().manual_seed(seed)
            hi = min(5000, embed_weight.shape[0])
            ids = torch.randint(0, hi, (n_tokens,), generator=g)
            p = embed_weight[ids].clone()
        elif kind == "gauss":
            g = torch.Generator().manual_seed(seed)
            hi = min(5000, embed_weight.shape[0])
            ids = torch.randint(0, hi, (n_tokens,), generator=g)
            ref = embed_weight[ids].float()
            noise = torch.randn(n_tokens, embed_weight.shape[1], generator=g)
            noise = noise / noise.norm(dim=-1, keepdim=True) * ref.norm(dim=-1, keepdim=True)
            p = noise.to(embed_weight.dtype)
        else:
            raise ValueError(f"unknown synthetic prompt kind {kind!r}")
        out[f"synthetic_{kind}"] = p.to(embed_weight.dtype)
    return out


def completion_stats(task_result) -> dict:
    """Length of what the model actually wrote.

    Accuracy alone cannot distinguish "the prompt made the model wrong" from
    "the prompt made the model stop early". All three evaluator classes record
    the raw text under sample_details[i]["output"], so the distinction is free.
    """
    details = getattr(task_result, "sample_details", None) or []
    outs = [d.get("output", "") or "" for d in details if isinstance(d, dict)]
    if not outs:
        return {}
    chars = sorted(len(o) for o in outs)
    words = sorted(len(o.split()) for o in outs)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    return {
        "n": len(outs),
        "chars_mean": round(sum(chars) / len(chars), 1),
        "chars_median": pct(chars, 0.5),
        "chars_p90": pct(chars, 0.9),
        "words_mean": round(sum(words) / len(words), 1),
        "words_median": pct(words, 0.5),
        "frac_empty": round(sum(c == 0 for c in chars) / len(chars), 4),
        "frac_under_10_words": round(sum(w < 10 for w in words) / len(words), 4),
        "examples": [o[:300] for o in outs[:2]],
    }


def make_eval_model(prompts_by_tag: dict, embed_weight,
                    format_instruction="keep", format_description=None,
                    provenance=None):
    """Drop-in replacement for `vllm_eval.eval_model` (same signature).

    `prompts_by_tag` maps a result key -> soft prompt tensor [m, d] (or a 0-row
    tensor). Every entry is scored against the same engine.
    """

    def eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text="",
        per_sample_lora=False,
    ):
        assert lora_dirs is None, "Inlet passes prompts through the closure, not lora_dirs"
        llm = _engine(model_dir, gpu_memory_utilization)
        model = SoftPromptVLLMModel(
            None,
            embed_weight,
            prefill_text=prefill_text,
            llm=llm,
            # identical to upstream eval_model
            sampling_params=vllm.SamplingParams(
                temperature=0, top_p=1, max_tokens=2**9, repetition_penalty=1.0
            ),
            chat_template=chat_template,
        )
        rec = apply_to_evaluator(evaluator, format_instruction,
                                 description=format_description)
        if provenance is not None:
            provenance.update(rec)
        if rec["n_changed"]:
            print(f"[format-ablation] {rec['mode']}: rewrote "
                  f"{rec['n_changed']}/{rec['n_samples']} prompts", flush=True)

        results = {}
        for tag, soft_prompt in prompts_by_tag.items():
            print(f"Evaluating soft prompt: {tag}  (m={0 if soft_prompt is None else soft_prompt.shape[0]})")
            model.set_soft_prompt(soft_prompt)
            results[tag] = evaluator.evaluate(model)
        return results

    return eval_model


# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_prompts_for_task(hypermod, descriptions: dict, emb_model, emb_tokenizer,
                              task_desc_format_fn, pooling_fn, device):
    """descriptions: {tag: description_string} -> {tag: [m, d] cpu tensor}.

    The description encoder is frozen and used exactly as T2L's
    data.get_task_embs does, so a Inlet prompt and a T2L LoRA are conditioned on
    byte-identical embeddings.
    """
    from hyper_llm_modulator.utils import embed_texts

    out = {}
    for tag, desc in descriptions.items():
        task_emb = embed_texts([desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device)
        p = hypermod(task_emb)              # [1, m, d]
        out[tag] = p[0].detach().to("cpu")  # [m, d]
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   help="the task to score; --tasks scores several in one engine")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="score these tasks in one process (one engine build). "
                        "Only valid with --zero-prompt / --synthetic-prompt, where "
                        "the prompts do not depend on the task.")
    p.add_argument("--checkpoint", default=None, help="path to hypermod_inlet.pt")
    p.add_argument(
        "--zero-prompt",
        action="store_true",
        help="evaluate with zero virtual tokens; must reproduce the zero-shot column",
    )
    p.add_argument("--descriptions", default=None,
                   help="optional json {tag: description}; defaults to the task's eval_descs")
    # A full scoring run is 3 eval_descs + 3 random_descs per task, which is the
    # reported protocol and is what you want for the number that goes in a table.
    # A checkpoint sweep wants the SHAPE of the curve over a ladder of
    # checkpoints, and paying 6x for it is the difference between a sweep that
    # fits in an afternoon and one that does not. These two trim it.
    #
    # A number produced with --max-eval-descs 1 is NOT comparable to a reported
    # one: it is a single description rather than an average over three. The
    # results JSON records how many were used so the two cannot be confused.
    p.add_argument("--max-eval-descs", type=int, default=0, metavar="N",
                   help="use only the first N real descriptions (0 = all). "
                        "For sweeps; not comparable to a reported number.")
    p.add_argument("--skip-random-descs", action="store_true",
                   help="skip the junk-description control. For sweeps.")
    p.add_argument("--model-dir", default=BASE_MODEL)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    # output_root(), not HERE: when baseline_prompt_tuning is importable HERE is
    # rebound to it, and Inlet results would be written into the upstream
    # checkout -- the one thing INLET_OUTPUT_ROOT exists to prevent.
    p.add_argument("--prompt-scales", default="",
                   help="comma-separated multipliers, e.g. '1.0,0.5,0.25'. Scores the "
                        "SAME generated prompt at several magnitudes. NOTE scale 0 is "
                        "NOT --zero-prompt: it is 32 rows of zeros, which still occupy "
                        "sequence positions and are attended to, whereas --zero-prompt "
                        "prepends nothing at all. Empty = unscaled, the default.")
    p.add_argument("--synthetic-prompt", default="",
                   help="comma-separated untrained control prompts scored in the "
                        "same engine: 'vocab' (the model at step 0) and/or 'gauss'.")
    p.add_argument("--include-zero-prompt", action="store_true",
                   help="also score a 0-row prompt in the same engine, so the "
                        "frozen base and a checkpoint share one vLLM build.")
    # choices come from the module that implements them, so a new mode can never
    # be added there and silently rejected here.
    p.add_argument("--format-instruction", choices=FORMAT_MODES, default="keep",
                   help="rewrite the evaluator's prompts before scoring. 'strip' "
                        "removes ' without any explanation'; 'add' prepends the "
                        "sentence every training task carries; 'strip_taskdef' "
                        "drops a lol_* task definition; 'desc_instead_of_taskdef' "
                        "swaps the definition for the task's own description. "
                        "See inlet/format_ablation.py.")
    p.add_argument("--format-description", default=None,
                   help="description to use with --format-instruction "
                        "desc_instead_of_taskdef (default: the task's first eval desc)")
    p.add_argument("--out-dir", default=os.path.join(output_root(), "eval_results_inlet"))
    a = p.parse_args()
    # bootstrap() chdir'd into the upstream checkout at import time, so relative
    # paths from the command line -- including the one the README documents,
    # `./scripts/eval.sh train_outputs/hyper_lora/full/hypermod_inlet.pt` -- would
    # otherwise resolve against the wrong directory.
    a.checkpoint = user_path(a.checkpoint)
    a.descriptions = user_path(a.descriptions)
    a.out_dir = user_path(a.out_dir)
    return a


def main() -> None:
    args = parse_args()
    assert args.checkpoint or args.zero_prompt, "pass --checkpoint or --zero-prompt"
    # --tasks scores ONE set of prompts on several tasks in one engine build.
    # That is only sound when the prompts do not depend on the task, i.e. the
    # zero and synthetic arms. With a checkpoint the prompts come from
    # _default_descs(args.task), so `--task gsm8k --tasks gsm8k humaneval`
    # scored humaneval with gsm8k's description and wrote it to
    # humaneval__<ckpt>.json -- no error, no warning, and a number that
    # contradicts the one claim this project is testing. argparse's help has
    # said "only valid with --zero-prompt / --synthetic-prompt" all along;
    # nothing enforced it. Use scripts/eval.sh to loop tasks: it re-invokes this
    # module once per task, so each gets its own description.
    if args.checkpoint and args.tasks and set(args.tasks) != {args.task}:
        raise SystemExit(
            "--tasks with --checkpoint would score "
            f"{sorted(set(args.tasks) - {args.task})} with the prompt generated "
            f"from {args.task}'s description.\n"
            "  Prompts are generated from --task alone; the engine is built once "
            "and reused.\n"
            f"  Loop instead:  TASKS=\"{' '.join(args.tasks)}\" ./scripts/eval.sh "
            f"{args.checkpoint}\n"
            "  --tasks is for --zero-prompt / --synthetic-prompt, whose prompts "
            "are task-independent.")
    # Which tokenizer / embedding loader / reference numbers are live. If this
    # says "(fallback)", the Inlet-vs-prompt-tuning comparison is between two
    # implementations that were never checked against each other -- the numbers
    # are still real, the comparison is weaker than it looks.
    print(f"[eval] shared-code source: {eval_impl_source()}", flush=True)

    embed_weight = load_input_embeddings(args.model_dir)
    tokenizer = get_tokenizer(args.model_dir)

    if args.zero_prompt and args.include_zero_prompt:
        raise SystemExit("--zero-prompt already is the zero arm; drop --include-zero-prompt")

    if args.zero_prompt:
        prompts = {"zero_prompt": torch.zeros(0, embed_weight.shape[1], dtype=embed_weight.dtype)}
        suffix = "__zero_prompt"
        train_config = None
    else:
        from inlet.checkpoint import load_inlet_checkpoint, load_description_encoder
        hypermod, train_config = load_inlet_checkpoint(args.checkpoint, device="cuda")
        # The checkpoint's `base` is 32 rows lifted out of a specific embedding
        # table, and `emb_rms` was measured on that same table. Scoring it
        # against a different base model loads cleanly (model_dim matches for any
        # 4096-d 7B), reports no warning, and is meaningless.
        trained_on = train_config.get("model_dir")
        if trained_on and trained_on != args.model_dir:
            raise SystemExit(
                f"checkpoint was trained against {trained_on!r} but eval is using "
                f"{args.model_dir!r}.\n"
                "`base` and `emb_rms` come from the training model's embedding table; "
                "this comparison would be silently wrong.\n"
                f"  pass --model-dir {trained_on}  (or re-train)"
            )
        # whole config, not just emb_model: the pooling width must be the
        # checkpoint's own. See load_description_encoder.
        enc = load_description_encoder(train_config, device="cuda")
        descs = (json.load(open(args.descriptions)) if args.descriptions
                 else _default_descs(args.task, args.max_eval_descs, args.skip_random_descs))
        prompts = generate_prompts_for_task(hypermod, descs, *enc, device="cuda")
        # the generator and the description encoder are done; give the GPU back
        # to vLLM before the engine is built.
        del hypermod, enc
        torch.cuda.empty_cache()
        suffix = f"__{os.path.splitext(os.path.basename(args.checkpoint))[0]}"
        if args.include_zero_prompt:
            # 0 rows: `build_eval_sequence` concatenates nothing, so this arm is
            # the frozen base under byte-identical sampling and prompts.
            prompts["zero_prompt"] = torch.zeros(
                0, embed_weight.shape[1], dtype=embed_weight.dtype)
            suffix += "__plus_zero"

    # Scale sweep. The learned prompt cuts gsm8k completions from 129 words to
    # 47 and humaneval from 139 to 18; the question this answers is whether that
    # is the prompt's CONTENT or simply its MAGNITUDE. Scaling the whole prompt
    # (base included) walks it toward zero, so a monotonic recovery of length as
    # the scale falls localises the damage to how hard the prompt pushes.
    #
    # scale 0 is NOT the same arm as --zero-prompt: it is 32 rows of zeros,
    # which still take up 32 sequence positions and are attended to, while
    # --zero-prompt concatenates nothing. Running both separates "the prefix has
    # harmful content" from "a 32-token prefix is harmful at all".
    scales = [float(x) for x in args.prompt_scales.split(",") if x.strip()]
    if scales:
        scaled = {}
        for nm, pt in prompts.items():
            for sc in scales:
                scaled[f"{nm}@s{sc:g}"] = (pt.float() * sc).to(pt.dtype)
        prompts = scaled
        suffix += "__scales_" + "-".join(f"{sc:g}" for sc in scales)

    kinds = [k for k in args.synthetic_prompt.split(",") if k]
    if kinds:
        prompts.update(synthetic_prompts(kinds, embed_weight, 32, seed=0))
        suffix += "__syn_" + "-".join(kinds)
    if args.format_instruction != "keep":
        suffix += f"__fmt_{args.format_instruction}"

    for task in (args.tasks or [args.task]):
        stem = f"{task}{suffix}"
        fmt_rec = {}
        fmt_desc = args.format_description
        if args.format_instruction == "desc_instead_of_taskdef" and fmt_desc is None:
            # the task's own description -- the exact string the generator is fed
            fmt_desc = next(iter(_default_descs(task, 1, True).values()))
        vllm_eval.eval_model = make_eval_model(
            prompts, embed_weight, args.format_instruction, fmt_desc, fmt_rec)
        results = vllm_eval.eval(
            args.model_dir,
            None,
            task,
            chat_template=tokenizer.chat_template,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        metrics = {k: v.aggregate_metrics for k, v in results.items()}
        lengths = {k: completion_stats(v) for k, v in results.items()}

        os.makedirs(args.out_dir, exist_ok=True)
        # Count DESCRIPTIONS, not arms. With --prompt-scales every description
        # becomes one arm per scale, so counting arms made a single-description
        # run with 3+ scales report reported_protocol=true -- which is the exact
        # confusion this block exists to prevent, and it mislabelled every file
        # from the first 16k-step sweep (1 description x 4 scales -> "4").
        _desc_tags = {k.rsplit("@s", 1)[0] for k in prompts}
        _n_eval = sum(1 for k in _desc_tags if k.startswith("eval_descs"))
        _has_random = any(k.startswith("random_descs") for k in _desc_tags)
        out_path = os.path.join(args.out_dir, f"{stem}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "task": task,
                    "model_dir": args.model_dir,
                    "checkpoint": args.checkpoint,
                    "n_virtual_tokens": {k: int(v.shape[0]) for k, v in prompts.items()},
                    "results": {task: metrics},
                    # Which protocol produced these numbers. A sweep run with
                    # --max-eval-descs 1 is a single description, not an average over
                    # three, and putting the two in one table would be comparing
                    # different measurements. Recorded here so they cannot be.
                    "protocol": {
                        "n_eval_descs": _n_eval,
                        "random_descs": _has_random,
                        "reported_protocol": _n_eval >= 3,
                    },
                    "train_config": train_config,
                    "format_instruction": fmt_rec,
                    "completion_lengths": lengths,
                },
                f,
                indent=2,
            )
            f.write("\n")

        print(f"[{stem}] {metrics}")
        for _k, _st in lengths.items():
            if _st:
                print(f"[{stem}] {_k}: completions median {_st['words_median']} words "
                      f"(mean {_st['words_mean']}), "
                      f"{100 * _st['frac_under_10_words']:.0f}% under 10 words")
        if (args.zero_prompt and task in ZERO_SHOT
                and args.format_instruction == "keep"):
            got = 100 * next(iter(next(iter(metrics.values())).values()))
            want = ZERO_SHOT[task]
            # Tolerance is expressed in QUESTIONS, not in points, because that is
            # the unit the difference actually comes in.
            #
            # Measured 2026-08-24 on an A100-80GB PCIe with vLLM 0.11.1 (FLASH_ATTN
            # backend): arc_challenge scored 65.61 against a recorded 65.70 -- one
            # question out of 1172, reproduced exactly across two runs, and
            # unchanged when the tokenizer was switched from a hand-rolled
            # AutoTokenizer to upstream's own `get_tokenizer`. So it is not this
            # repo: it is a different GPU / attention backend / vLLM build flipping
            # one borderline answer.
            #
            # A strict equality gate therefore greets every new cluster with a red
            # FAIL that means nothing, which is how gates get ignored. A gate that
            # tolerates a couple of hundred questions would be useless. One or two
            # questions is the honest line: an injection bug does not move one
            # answer, it moves dozens.
            n = _N_EVAL_QUESTIONS.get(task)
            if n:
                q = abs(got - want) * n / 100.0
                if q <= 1.5:
                    verdict, why = "PASS", f"({q:.1f} question of {n})"
                elif q <= 3.5:
                    verdict, why = ("WARN",
                                    f"({q:.1f} questions of {n} -- larger than the "
                                    "1-question hardware jitter seen so far; check the "
                                    "vLLM version and attention backend before trusting "
                                    "any comparison)")
                else:
                    verdict, why = ("FAIL",
                                    f"({q:.0f} questions of {n} -- too many to be "
                                    "hardware. The prompt is being injected in the "
                                    "wrong place, or the tokenizer/chat template "
                                    "differs. Do not train through this.)")
            else:
                verdict = "PASS" if abs(got - want) < 0.05 else "FAIL"
                why = "(exact match required: question count for this task unknown)"
            print(f"zero-shot reproduction: got {got:.2f}, expected {want:.2f} "
                  f"-> {verdict} {why}")
        print(f"wrote {out_path}")


# Size of the eval split, used to express the zero-shot tolerance in questions
# rather than in points. Only tasks whose size is known are listed; anything
# missing falls back to requiring an exact match.
_N_EVAL_QUESTIONS = {
    "arc_challenge": 1172,
}


def _default_descs(task: str, max_eval_descs: int = 0, skip_random: bool = False) -> dict:
    """The task's eval_descs from the T2L config, tagged by split+index.

    `max_eval_descs` / `skip_random` trim the set for a checkpoint sweep. A
    number produced with fewer than all three real descriptions is NOT the
    reported protocol and must not be put in the same table as one that is.
    """
    import yaml

    # Resolve against T2L_ROOT, never against HERE.
    #
    # HERE has two different values depending on whether the reference
    # baseline_prompt_tuning/ is importable: eval_common defines it as the inlet/
    # package directory, and then REBINDS it to the baseline directory when the
    # real module is present. So `HERE/../configs` meant the T2L checkout on one
    # machine and this repo on another -- and this repo has no configs/ dir, so
    # a fresh clone died with FileNotFoundError on the documented eval command.
    # The config belongs to T2L; ask T2L where it is.
    cfg_path = os.environ.get("CONFIG") or "configs/hyper_lora_decontam_lol_tasks.yaml"
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(T2L_ROOT, cfg_path)
    if not os.path.isfile(cfg_path):
        raise SystemExit(
            f"description config not found: {cfg_path}\n"
            "This is the T2L config that lists each task's eval descriptions.\n"
            "  * check T2L_ROOT (currently: " + str(T2L_ROOT) + ")\n"
            "  * or set CONFIG=/abs/path/to/hyper_lora_decontam_lol_tasks.yaml\n"
            "  * or pass --descriptions with your own json {tag: description}"
        )
    cfg = yaml.safe_load(open(cfg_path))
    reals = cfg["eval_ds_info"][task]["descriptions"]
    if max_eval_descs and max_eval_descs > 0:
        reals = reals[:max_eval_descs]
    out = {f"eval_descs__{i}": d for i, d in enumerate(reals)}
    if not skip_random:
        out.update({f"random_descs__{i}": d for i, d in enumerate(cfg["additional_eval_descs"])})
    return out


if __name__ == "__main__":
    main()
