"""Does this checkout still fit the text-to-lora checkout next to it?

    python -m inlet.test_upstream_api

Pure AST on both sides. Imports nothing from either project, so it needs **no
torch, no vLLM, no GPU and no install** -- run it on a login node, or before
`setup_env.sh` has finished, or on a laptop.

WHY THIS EXISTS
---------------
Inlet is an overlay: it imports upstream's task metadata, sampler, dataloaders,
loss and eval harness rather than copying them. That is what makes a Inlet number
comparable to a T2L number, and it is also a standing liability -- upstream is a
moving target this repo does not pin. When it moves, the failure is not subtle,
but it is *late*: `python -m inlet.train_inlet` gets through argument parsing, model
loading and 14 GB of weights before dying on a TypeError.

Four things are checked, all of them things that have actually broken here:

  1. every symbol Inlet imports from upstream still exists there;
  2. every call into upstream still matches upstream's signature -- arity,
     keyword names, required arguments;
  3. every inlet -> inlet call matches too, which catches your own refactors;
  4. every `--flag` the shell scripts pass is a real dataclass field. Upstream's
     parser raises on the first override it does not recognise, so a typo is a
     hard failure two seconds into a job you queued for three days.

What it does NOT check: whether any of it *behaves* correctly. A function can
have the right signature and the wrong semantics -- that is what `gate_m0`,
`test_train_eval_agree` and the smoke run are for. This is the cheap check that
runs first.
"""

import ast
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inlet._env import find_t2l_root  # noqa: E402

INLET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names that belong to some other program on the command line, not the trainer.
NOT_TRAINER_FLAGS = {
    # torchrun / torch.distributed.run
    "nproc_per_node", "nproc", "master_port", "master_addr", "nnodes", "node_rank",
    "rdzv_backend", "rdzv_endpoint", "standalone",
    # pip / git / huggingface-cli
    "exclude", "no_deps", "index_url", "upgrade", "recurse_submodules", "depth", "init",
    # nvidia-smi
    "format", "query_gpu", "noheader",
    # inlet's own non-trainer entry points (eval_inlet, warm_datasets, probe_prompt, ...)
    "zero_prompt", "no_soft_prompt", "task", "tasks", "checkpoint", "out_dir",
    "model_dir", "only_failed", "workers", "max_depth", "quiet", "help", "version",
    # literal placeholder in train.sh's usage string
    "flag",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _parse(path):
    try:
        return ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return None


def _upstream_files(t2l):
    for p in glob.glob(f"{t2l}/src/**/*.py", recursive=True):
        # src/fishfarm is a nested checkout; the importable package is one level in
        if "/fishfarm/" in p and "/src/fishfarm/fishfarm/" not in p:
            continue
        yield p


def _index_defs(files):
    """name -> [(path, FunctionDef)]. Classes are indexed by their __init__."""
    defs = {}
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(n.name, []).append((path, n))
            elif isinstance(n, ast.ClassDef):
                for b in n.body:
                    if isinstance(b, ast.FunctionDef) and b.name == "__init__":
                        defs.setdefault(n.name, []).append((path, b))
    return defs


def _module_file(t2l, mod):
    for base in (f"{t2l}/src", f"{t2l}/src/fishfarm"):
        p = os.path.join(base, *mod.split("."))
        for c in (p + ".py", os.path.join(p, "__init__.py")):
            if os.path.isfile(c):
                return c
    return None


def _exported(path, t2l, seen=None):
    """Top-level names a module provides, following `from X import *`."""
    seen = seen or set()
    if path in seen:
        return set()
    seen.add(path)
    tree = _parse(path)
    if tree is None:
        return set()
    names, pkgdir = set(), os.path.dirname(path)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Import):
            names.update((a.asname or a.name.split(".")[0]) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if not any(a.name == "*" for a in n.names):
                names.update((a.asname or a.name) for a in n.names)
                continue
            sub, cand = n.module or "", None
            if n.level:
                p = os.path.join(pkgdir, *sub.split(".")) if sub else pkgdir
                for c in (p + ".py", os.path.join(p, "__init__.py")):
                    if os.path.isfile(c):
                        cand = c
                        break
            else:
                cand = _module_file(t2l, sub)
            if cand:
                names |= _exported(cand, t2l, seen)
    return names


def _signature(fn):
    a = fn.args
    pos = [x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
    if pos and pos[0] in ("self", "cls"):
        pos = pos[1:]
    nd = len(a.defaults)
    required = pos[: len(pos) - nd] if nd else list(pos)
    kwonly = [x.arg for x in a.kwonlyargs]
    kwreq = [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    return pos, required, kwonly, kwreq, a.vararg is not None, a.kwarg is not None


def _call_errors(call, fn):
    pos, required, kwonly, kwreq, star, dstar = _signature(fn)
    n_pos = sum(1 for a in call.args if not isinstance(a, ast.Starred))
    kw = [k.arg for k in call.keywords if k.arg]
    has_star = any(isinstance(a, ast.Starred) for a in call.args)
    has_dstar = any(k.arg is None for k in call.keywords)
    errs = []
    if not star and not has_star and n_pos > len(pos):
        errs.append(f"{n_pos} positional args, definition takes {len(pos)}")
    if not dstar:
        unknown = [k for k in kw if k not in pos and k not in kwonly]
        if unknown:
            errs.append(f"unknown keyword(s) {unknown}")
    if not has_star and not has_dstar:
        supplied = set(pos[:n_pos]) | set(kw)
        missing = [r for r in required if r not in supplied]
        missing += [k for k in kwreq if k not in kw]
        if missing:
            errs.append(f"missing required {missing}")
    return errs, (n_pos, len(kw))


def _inlet_sources():
    return sorted(glob.glob(os.path.join(INLET_ROOT, "inlet", "*.py")))


def _upstream_imports():
    """local name -> module it came from, for every upstream import in inlet/."""
    out = {}
    for p in _inlet_sources():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and (
                    n.module.startswith("hyper_llm_modulator")
                    or n.module.startswith("fishfarm")):
                for a in n.names:
                    out[a.asname or a.name] = n.module
    return out


# --------------------------------------------------------------------------- #
# check 1 -- the import surface
# --------------------------------------------------------------------------- #

def check_import_surface(t2l, verbose):
    wanted = {}
    for p in _inlet_sources():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and (
                    n.module.startswith("hyper_llm_modulator")
                    or n.module.startswith("fishfarm")):
                wanted.setdefault(n.module, set()).update(a.name for a in n.names)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith(("hyper_llm_modulator", "fishfarm")):
                        wanted.setdefault(a.name, set())

    bad = 0
    for mod, syms in sorted(wanted.items()):
        f = _module_file(t2l, mod)
        if f is None:
            print(f"  FAIL  {mod} -- module not found under {t2l}/src")
            bad += 1
            continue
        missing = sorted(s for s in syms if s not in _exported(f, t2l))
        if missing:
            print(f"  FAIL  {mod} -- upstream no longer provides {missing}")
            bad += 1
        elif verbose:
            print(f"  ok    {mod} ({len(syms)} symbol(s))")
    if not bad:
        print(f"  ok    all {sum(len(v) for v in wanted.values())} imported symbol(s) "
              f"exist across {len(wanted)} upstream module(s)")
    return bad


# --------------------------------------------------------------------------- #
# check 2 -- calls into upstream
# --------------------------------------------------------------------------- #

def check_upstream_calls(t2l, verbose):
    targets = _upstream_imports()
    defs = _index_defs(_upstream_files(t2l))

    def pick(name):
        # Resolve by the module the name was imported FROM. Resolving by name
        # alone is wrong: `get_tokenizer` is both hyper_llm_modulator.utils'
        # function and a 0-arg method on fishfarm's VLLMModel, and matching the
        # wrong one invents failures that are not there.
        want = targets[name].replace(".", "/")
        cands = defs.get(name, [])
        exact = [c for c in cands
                 if f"/{want}.py" in c[0] or f"/{want}/__init__.py" in c[0]]
        if exact:
            return exact[0]
        pkg = want.split("/")[0]
        under = [c for c in cands if f"/{pkg}/" in c[0]]
        return under[0] if under else None

    bad = checked = 0
    for p in _inlet_sources():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            if n.func.id not in targets or n.func.id not in defs:
                continue
            picked = pick(n.func.id)
            if picked is None:
                continue
            path, fn = picked
            errs, shape = _call_errors(n, fn)
            checked += 1
            rel = os.path.relpath(p, INLET_ROOT)
            if errs:
                pos = _signature(fn)[0]
                print(f"  FAIL  {rel}:{n.lineno} {n.func.id}(...) -- " + "; ".join(errs))
                print(f"        upstream: {n.func.id}({', '.join(pos)})"
                      f"  in {os.path.relpath(path, t2l)}")
                bad += 1
            elif verbose:
                print(f"  ok    {rel}:{n.lineno} {n.func.id}"
                      f"({shape[0]} pos, {shape[1]} kw)")
    if not bad:
        print(f"  ok    all {checked} call(s) into upstream match its signatures")
    return bad


# --------------------------------------------------------------------------- #
# check 3 -- inlet calling inlet
# --------------------------------------------------------------------------- #

def check_internal_calls(verbose):
    defs = {}
    for p in _inlet_sources():
        tree = _parse(p)
        if tree is None:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(n.name, []).append((p, n))
            elif isinstance(n, ast.ClassDef):
                for b in n.body:
                    if isinstance(b, ast.FunctionDef) and b.name == "__init__":
                        defs.setdefault(n.name, []).append((p, b))

    bad = checked = 0
    for p in _inlet_sources():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            cands = defs.get(n.func.id)
            if not cands or len(cands) != 1:   # ambiguous name -> cannot resolve statically
                continue
            dp, fn = cands[0]
            errs, _ = _call_errors(n, fn)
            checked += 1
            if errs:
                print(f"  FAIL  {os.path.relpath(p, INLET_ROOT)}:{n.lineno} "
                      f"{n.func.id}(...) -- " + "; ".join(errs))
                print(f"        defined at {os.path.relpath(dp, INLET_ROOT)}:{fn.lineno}"
                      f" as {n.func.id}({', '.join(_signature(fn)[0])})")
                bad += 1
    if not bad:
        print(f"  ok    all {checked} inlet -> inlet call(s) match")
    return bad


# --------------------------------------------------------------------------- #
# check 4 -- CLI flags
# --------------------------------------------------------------------------- #

def check_cli_flags(t2l, verbose):
    def class_fields(path, cls):
        tree = _parse(path)
        if tree is None:
            return set(), []
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == cls:
                own = {b.target.id for b in n.body
                       if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)}
                bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                         for b in n.bases]
                return own, bases
        return set(), []

    own, bases = class_fields(os.path.join(INLET_ROOT, "inlet", "train_inlet.py"), "InletArguments")
    if not own:
        print("  FAIL  could not find the InletArguments dataclass")
        return 1
    fields, seen, queue = set(own), set(), list(bases)
    while queue:
        b = queue.pop()
        if b in seen:
            continue
        seen.add(b)
        for cand in glob.glob(f"{t2l}/src/hyper_llm_modulator/*.py"):
            f, bb = class_fields(cand, b)
            if f:
                fields |= f
                queue += bb
                break

    used = {}
    for sh in sorted(glob.glob(os.path.join(INLET_ROOT, "scripts", "*.sh"))):
        for m in re.finditer(r"--([a-zA-Z_][a-zA-Z0-9_]*)[= ]",
                             open(sh, encoding="utf-8").read()):
            used.setdefault(m.group(1), set()).add(os.path.basename(sh))

    bad = 0
    checked = sorted(f for f in used if f not in NOT_TRAINER_FLAGS)
    for f in checked:
        if f not in fields:
            print(f"  FAIL  --{f} (in {', '.join(sorted(used[f]))}) is not a field on "
                  "InletArguments; upstream's parser will refuse to start")
            bad += 1
        elif verbose:
            print(f"  ok    --{f}")
    if not bad:
        print(f"  ok    all {len(checked)} trainer flag(s) exist "
              f"({len(own)} own + {len(fields) - len(own)} inherited fields)")
    return bad


# --------------------------------------------------------------------------- #

def check_pooling_contract(t2l, verbose):
    """Inlet REPLACES upstream's pooling_fn instead of forking the dataloader.

    That trick rests on four facts about upstream, none of which Inlet controls.
    If any of them changes, `inlet.desc_pool` silently starts feeding the
    generator something other than what it thinks -- so they are checked here
    rather than assumed in a comment.

      1. `pooling.cls_pool(outputs, attention_mask)` exists with that signature.
         `desc_pool.make_pooling_fn(1)` must reproduce it, and
         `test_desc_cond` compares against a copy of its body.
      2. `get_emb_model_and_fns` still hands gte a CLS pooling function, i.e.
         the thing being replaced is the thing we think it is.
      3. `data.get_task_embs` still takes `pooling_fn` and passes it to
         `embed_texts` -- that is the training-side injection point.
      4. `utils.embed_texts` still takes `pooling_fn` -- the eval-side one.

    What this canNOT check by AST is the load-bearing claim that nothing
    upstream constrains the WIDTH pooling_fn returns. `test_desc_cond` covers
    the shape algebra, and the smoke run covers it end to end.
    """
    bad = 0
    defs = _index_defs(list(_upstream_files(t2l)))

    def _params(name):
        if name not in defs:
            return None
        return [a.arg for a in defs[name][0][1].args.args]

    for fn, needed in (
        ("cls_pool", ["outputs", "attention_mask"]),
        ("get_task_embs", ["pooling_fn"]),
        ("embed_texts", ["pooling_fn"]),
        ("get_emb_model_and_fns", ["emb_model_name", "device"]),
    ):
        got = _params(fn)
        if got is None:
            print(f"  FAIL  upstream no longer defines {fn}()")
            bad += 1
            continue
        missing = [a for a in needed if a not in got]
        if missing:
            print(f"  FAIL  {fn}() lost parameter(s) {missing}; has {got}")
            bad += 1
        elif verbose:
            print(f"  ok    {fn}({', '.join(got)})")

    # get_emb_model_and_fns must still choose CLS pooling for gte -- if upstream
    # switched gte to mean pooling, desc_pool's slot 0 would stop being the
    # vector every pre-2026-08-25 Inlet number was conditioned on.
    path = _module_file(t2l, "hyper_llm_modulator.utils.model_loading")
    src = open(path, encoding="utf-8").read() if path else ""
    if 'get_pooling_fn("cls")' not in src and "get_pooling_fn('cls')" not in src:
        print("  FAIL  get_emb_model_and_fns no longer selects CLS pooling for gte; "
              "inlet.desc_pool slot 0 would no longer be upstream's pooled vector")
        bad += 1
    elif verbose:
        print('  ok    gte still gets get_pooling_fn("cls")')

    # data.py must still stack one row per sample. desc_pool returns K*H-wide
    # rows and relies on torch.stack accepting any fixed width.
    path = _module_file(t2l, "hyper_llm_modulator.data")
    src = open(path, encoding="utf-8").read() if path else ""
    if "torch.stack" not in src or "task_emb" not in src:
        print("  FAIL  data.py no longer stacks per-sample task_emb rows; "
              "the K*H-wide description cache depends on that")
        bad += 1
    elif verbose:
        print("  ok    data.py still stacks per-sample task_emb rows")

    return bad


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verbose = "-v" in argv or "--verbose" in argv

    try:
        t2l = find_t2l_root()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL  cannot locate the text-to-lora checkout: {exc}")
        return 1

    print(f"inlet     {INLET_ROOT}")
    print(f"t2l     {t2l}")
    print("AST only -- nothing is imported, so this needs no torch and no GPU.\n")

    bad = 0
    for title, fn in (
        ("1/5  every imported upstream symbol still exists", lambda: check_import_surface(t2l, verbose)),
        ("2/5  every call into upstream matches its signature", lambda: check_upstream_calls(t2l, verbose)),
        ("3/5  every inlet -> inlet call matches", lambda: check_internal_calls(verbose)),
        ("4/5  every --flag is a real dataclass field", lambda: check_cli_flags(t2l, verbose)),
        ("5/5  the pooling_fn contract desc_pool overrides", lambda: check_pooling_contract(t2l, verbose)),
    ):
        print(title)
        bad += fn()
        print()

    if bad:
        print(f"FAILED  {bad} problem(s). This checkout does not fit the text-to-lora")
        print("        next to it. Fix these before installing anything -- every one of")
        print("        them is a crash that would otherwise arrive after the 14 GB of")
        print("        weights have loaded.")
        return 1
    print("PASS  this checkout fits the text-to-lora checkout next to it.")
    print("      Signatures only -- behaviour is what gate_m0, test_train_eval_agree")
    print("      and the smoke run are for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
