#!/usr/bin/env bash
# TEMPLATE ONLY: operator copies to the NEXT available gpuq_job_NNN_rollout_smoke.sh.
# No enqueueing, GPU access, or conductor changes happen just by adding this file.
set -euo pipefail
ROOT=${ROOT:-/workspace/slava/exp/wavefield_video}
LOG_DIR=${LOG_DIR:-/workspace/slava/logs}
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/$(basename "$0" .sh).log" 2>&1
printf 'ROLLOUT BOX PREFLIGHT %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv
cd "$ROOT"
export PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
export PYTHONHASHSEED=0
export REQUIRE_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# New timestamped OUT, no stale sentinel or automatic overwrite.
export OUT=${OUT:-runs_rollout_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
bash run_rollout_smoke.sh
printf 'ROLLOUT BOX JOB DONE %s\n' "$(date -u +%FT%TZ)"
