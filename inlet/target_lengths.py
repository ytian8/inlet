"""How hard does `equally_weight_sample` tilt training toward short answers?

`compute_loss(..., equally_weight_sample=True)` divides each sample's token
losses by that sample's own target length before averaging over the batch
(upstream `sft_trainer.py`):

    loss = (loss.view(bs, max_seq_len) / seq_len).sum(-1).mean()

Every sample contributes 1 unit of loss whatever its length, so PER TOKEN a
target of length T carries weight 1/T. That is a reasonable default for mixing
datasets with different prompt lengths, and upstream turns it off exactly once
-- for the generative task (`configs/lora_gsm8k.yaml` sets it false). Inlet
trains on 479 tasks whose targets range from one token to hundreds, leaves the
flag at its default of True, and is then evaluated on gsm8k / mbpp / humaneval,
which need 200+ tokens of coherent generation.

This measures the tilt rather than asserting it.

Three facts about the cache this reads, all learned the hard way:

* the directories under `data/transformed_datasets` are HASH-named; the task
  name lives in `dataset_info.json` under `dataset_name`;
* the cache holds TWO shapes. Some entries are tokenized
  (`input_ids`/`attention_mask`/`labels`), some are still text
  (`prompt`/`response`/`task_def`/...);
* where `labels` exists it is the exact answer: the count of non-(-100)
  positions is the `seq_len` that `compute_loss` divides by. Where it does not,
  the config's `sft_mode: completion` means the supervised span is `response`,
  which then has to be tokenized here. `length_source_counts` in the output
  says how many tasks came from each, because the two are not interchangeable
  -- `response` alone omits any template and EOS tokens the collator kept.

No GPU, no model weights -- only the tokenizer and the cache warm_cache builds.

    python -m inlet.target_lengths --out docs/measurements/target_lengths.json
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--max-rows-per-task", type=int, default=200,
                    help="subsample within a task; the mixture is what matters")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--field", default="response",
                    help="supervised span; `completion` mode supervises `response`")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    root = os.environ.get("T2L_ROOT", "third_party/text-to-lora")
    cache = os.path.join(root, "data", "transformed_datasets")
    if not os.path.isdir(cache):
        raise SystemExit(f"no cache at {cache} -- run scripts/warm_cache.sh first")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dirs = sorted(os.listdir(cache))
    if not dirs:
        raise SystemExit(f"{cache} is empty")

    candidates = {}          # task name -> (source, lens)
    per_task = {}
    all_lens = []
    skipped = []
    sources = {}
    for i, d in enumerate(dirs):
        path = os.path.join(cache, d)
        info = os.path.join(path, "dataset_info.json")
        if not os.path.isfile(info):
            skipped.append((d, "no dataset_info.json"))
            continue
        try:
            name = json.load(open(info)).get("dataset_name") or d
        except Exception as e:
            skipped.append((d, f"unreadable info: {e}"))
            continue
        # The Lots-of-LoRAs training mixture is the taskNNN_* namespace; the
        # benchmarks cached alongside it are not part of the training mix and
        # would distort the distribution being measured.
        if not name.startswith("task"):
            continue
        try:
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):
                ds = ds[list(ds.keys())[0]]
            if len(ds) > args.max_rows_per_task:
                ds = ds.select(range(args.max_rows_per_task))

            # The cache holds BOTH shapes. Prefer `labels`: the count of
            # non-(-100) positions is exactly what compute_loss divides by, so
            # it needs no assumption about which field is supervised and it
            # includes whatever template and EOS tokens the collator kept.
            if "labels" in ds.column_names:
                source = "labels"
                lens = [int((np.asarray(r) != -100).sum()) for r in ds["labels"]]
            elif args.field in ds.column_names:
                source = args.field
                texts = [t for t in ds[args.field] if isinstance(t, str)]
                if not texts:
                    skipped.append((name, "no text rows"))
                    continue
                lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
            else:
                skipped.append((name, f"neither `labels` nor `{args.field}` in {ds.column_names}"))
                continue
            lens = [x for x in lens if x > 0]
            if not lens:
                skipped.append((name, f"all rows empty via {source}"))
                continue
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}: {e}"))
            continue

        # The SAME task appears twice under different hashes -- once as text and
        # once tokenized. Counting both double-counts every task, so keep one
        # per task name and prefer `labels`, which is what the loss actually
        # divides by.
        prev = candidates.get(name)
        if prev is None or (prev[0] != "labels" and source == "labels"):
            candidates[name] = (source, lens)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dirs)} dirs, {len(candidates)} distinct tasks")

    for name, (source, lens) in candidates.items():
        sources[source] = sources.get(source, 0) + 1
        per_task[name] = float(np.mean(lens))
        all_lens.extend(lens)

    if not all_lens:
        raise SystemExit("read no rows -- the cache is unusable; that is not a finding about tasks")
    if len(per_task) != len(candidates):
        raise RuntimeError(f"dedup lost tasks: {len(candidates)} -> {len(per_task)}")

    a = np.asarray(all_lens, dtype=np.float64)
    tm = np.asarray(sorted(per_task.values()), dtype=np.float64)

    median_len = float(np.median(a))

    def share_of_weight(threshold):
        """Share of the total gradient that lands on targets >= threshold.

        TRUE:  every sample contributes exactly 1 unit of loss (T tokens at
               1/T each), so the share is just the fraction of SAMPLES.
        FALSE: every token contributes 1, so the share is the fraction of
               TOKENS.
        The gap between the two numbers is the tilt.
        """
        long_mask = a >= threshold
        if not long_mask.any():
            return {"n_samples_at_or_above": 0, "frac_samples": 0.0,
                    "share_of_gradient_TRUE": 0.0, "share_of_gradient_FALSE": 0.0}
        return {
            "n_samples_at_or_above": int(long_mask.sum()),
            "frac_samples": float(long_mask.mean()),
            "share_of_gradient_TRUE": float(long_mask.sum() / a.size),
            "share_of_gradient_FALSE": float(a[long_mask].sum() / a.sum()),
            # Under TRUE a token's weight is 1/T, so a token in a target this
            # long is worth this many times LESS than a token in a
            # median-length target.
            "per_token_weight_penalty_vs_median": float(a[long_mask].mean() / median_len),
        }

    pcts = {f"p{p}": float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 90, 95, 99)}
    out = {
        "n_tasks": len(per_task),
        "n_skipped": len(skipped),
        "length_source_counts": sources,
        "n_rows": int(a.size),
        "supervised_field": "labels where present, else " + args.field,
        "target_len_tokens": {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "min": int(a.min()), "max": int(a.max()), **pcts,
        },
        "per_task_mean_target_len": {
            "min": float(tm.min()), "p25": float(np.percentile(tm, 25)),
            "median": float(np.median(tm)), "p75": float(np.percentile(tm, 75)),
            "max": float(tm.max()),
        },
        "long_targets": {str(t): share_of_weight(t) for t in (50, 100, 200)},
        "shortest_10_tasks": sorted(per_task.items(), key=lambda kv: kv[1])[:10],
        "longest_10_tasks": sorted(per_task.items(), key=lambda kv: kv[1])[-10:],
        "skipped": skipped[:20],
    }

    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("shortest_10_tasks", "longest_10_tasks", "skipped")},
                     indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
