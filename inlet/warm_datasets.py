"""Pre-tokenize and cache the T2L datasets. I/O bound -- no GPU needed.

Run it on any CPU box, not on rented GPU time. Datasets are
fetched and tokenized in parallel because the cost is per-request latency, not
bandwidth: ~500 small HF repos sequentially is 1-3h, with 12 workers it is
20-40min.

Failure-tolerant on purpose: one dead or rate-limited repo must not abort the
run. Every failure is recorded, and the summary lists exactly which ids to
retry, so a second pass only re-fetches what actually failed.

    # everything (479 train + 31 eval)
    python inlet/warm_datasets.py --workers 12

    # smoke subset
    python inlet/warm_datasets.py --n-train 12 --workers 8

    # retry only what failed last time
    python inlet/warm_datasets.py --only-failed warm_failures.json
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# piqa (ybisk/piqa) is a script-based HF dataset. datasets 3.5.1 still loads
# those, but refuses to execute the script without explicit consent. Without
# this, exactly one of the 510 repos fails and the failure looks unrelated.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

# Resolve the upstream checkout and chdir into it -- the config path and the
# data/transformed_datasets cache below are relative to it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import bootstrap  # noqa: E402

REPO = bootstrap()

import yaml  # noqa: E402

from hyper_llm_modulator.data import get_datasets  # noqa: E402
from hyper_llm_modulator.utils import get_metadata, get_tokenizer  # noqa: E402

CFG = "configs/hyper_lora_decontam_lol_tasks.yaml"
_print_lock = threading.Lock()


def warm_one(name, metadata, tokenizer, inp_max_len):
    t0 = time.time()
    try:
        # get_datasets takes a LIST of names (it does {k: metadata[k] for k in names}
        # internally) -- passing a bare string iterates its characters and dies with
        # KeyError: 'l'.
        ds = get_datasets([name], metadata, tokenizer, sft_mode="completion",
                          is_intx_model=True, inp_max_len=inp_max_len)
        n = sum(len(v) for v in ds.values()) if isinstance(ds, dict) else len(ds)
        return name, True, n, time.time() - t0, ""
    except Exception as e:
        return name, False, 0, time.time() - t0, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=None, help="limit training tasks (smoke runs)")
    ap.add_argument("--inp-max-len", type=int, default=512, help="match T2L's default, not the 1024 the prompt-tuning baseline used")
    ap.add_argument("--only-failed", default=None, help="path to a previous warm_failures.json")
    ap.add_argument("--out", default="warm_failures.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG))
    train_names = list(cfg["train_ds_names"])
    if args.n_train:
        train_names = train_names[: args.n_train]
    eval_names = list(cfg["eval_ds_info"].keys())
    names = list(dict.fromkeys(train_names + eval_names))  # dedupe, keep order

    if args.only_failed:
        names = json.load(open(args.only_failed))["failed"]
        print(f"retry mode: {len(names)} datasets")

    print(f"{len(names)} datasets ({len(train_names)} train + {len(eval_names)} eval, deduped)")
    print(f"workers={args.workers} inp_max_len={args.inp_max_len}")

    tokenizer = get_tokenizer(args.model_dir)
    metadata = get_metadata(names, False)

    # `datasets` builds its progress bars inside whatever thread calls it, and
    # tqdm creates its class-level lock lazily in the first thread to touch it.
    # With 12 workers racing on the very first bar you get
    #     AttributeError: type object 'tqdm' has no attribute '_lock'
    # on a handful of datasets, at random. Claim the lock from the main thread
    # before the pool exists and the race cannot happen.
    from tqdm import tqdm as _tqdm
    _tqdm.set_lock(threading.RLock())

    ok, failed, t0 = [], [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(warm_one, n, metadata, tokenizer, args.inp_max_len): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            name, good, n, dt, err = f.result()
            (ok if good else failed).append(name)
            with _print_lock:
                mark = "ok  " if good else "FAIL"
                print(f"[{i:4d}/{len(names)}] {mark} {name:14s} n={n:<6d} {dt:5.1f}s "
                      f"| ok={len(ok)} fail={len(failed)} | {time.time()-t0:6.0f}s elapsed"
                      + (f"  {err}" if err else ""), flush=True)

    print(f"\n{'='*70}")
    print(f"warmed {len(ok)}/{len(names)} in {(time.time()-t0)/60:.1f} min")
    if failed:
        print(f"\n{len(failed)} FAILED -- retry just these with --only-failed {args.out}:")
        for n in failed:
            print(f"  {n}")
        json.dump({"failed": failed}, open(args.out, "w"), indent=2)
    else:
        print("no failures")


if __name__ == "__main__":
    main()
