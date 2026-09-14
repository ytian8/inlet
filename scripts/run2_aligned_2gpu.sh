#!/usr/bin/env bash
# Usage: bash scripts/run2_aligned_2gpu.sh /absolute/run2-plan audit|smoke|train
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PLAN="${1:?pass the directory produced by inlet.run2_setup}"
STAGE="${2:?audit, smoke or train}"
[[ "$STAGE" == audit || "$STAGE" == smoke || "$STAGE" == train ]] || exit 2
PLAN="$(cd "$PLAN" && pwd)"
source scripts/common.sh
# Configs inherit Run 1 rather than the defaults in train.sh.
"$PYBIN" - "$PLAN" "$STAGE" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
import torch,yaml
p,stage=Path(sys.argv[1]),sys.argv[2]
manifest=json.loads((p/'provenance.json').read_text())
assert Path(os.environ['T2L_ROOT']).resolve()==Path(manifest['upstream_root']), 'T2L_ROOT differs from frozen plan'
assert Path(os.environ['INLET_ROOT']).resolve()==Path(manifest['inlet_root']), 'Inlet checkout differs from frozen plan'
for name,digest in manifest['config_sha256'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest, f'{name} changed; create a new plan'
for name,digest in manifest['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, f'Source changed: {name}; regenerate plan'
cfg=yaml.safe_load((p/f'{stage}.yaml').read_text())
assert torch.cuda.device_count()==2, 'Expose exactly two GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1'
save=Path(os.environ['INLET_OUTPUT_ROOT'])/str(cfg['exp_setup'])/cfg['run_name']
assert not save.exists(), f'{save} already exists. Preserve it; use a fresh run name/plan.'
if stage=='train':
    review=json.loads((p/'readiness.json').read_text())
    assert review['plan_sha256']==hashlib.sha256((p/'provenance.json').read_bytes()).hexdigest()
    for key in ['descriptions_sufficient','rendered_inputs_checked','label_coverage_checked','smoke_passed','monitor_coverage_reviewed']:
        assert review.get(key) is True, f'Cluster agent must complete {key} before training'
    assert not review.get('blocking_findings'), 'Resolve and document blockers before launch'
print(f'Run 2 {stage}: output={save}; global batch=64; fresh initialization')
PY
require_disk 30 "$INLET_OUTPUT_ROOT" || exit 1
PORT="${MASTER_PORT:-$((29500 + $$ % 1000))}"
exec "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  --master_port="$PORT" -m inlet.train_inlet "$PLAN/$STAGE.yaml"
