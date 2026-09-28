#!/bin/bash
# LONG_HORIZON box smoke: validates the pushed file set on the 4090 before any
# suite slice -- unit tests, then a tiny GPU pass of run_long_horizon.sh (all 5
# arms, 4 steps, 60-frame streams) into a throwaway OUT. Must print
# 'smoke long-horizon PASS'.
LOG=/workspace/slava/logs/gpuq_job_600_long_horizon_smoke.log
mkdir -p /workspace/slava/logs
exec > "$LOG" 2>&1
set -o pipefail                 # a failing test must not hide behind `| tail`
echo "=== job600 long-horizon boxsmoke start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
cd /workspace/slava/exp/wavefield_video || exit 1
export PY=/workspace/slava/comfy-house/venv/bin/python
$PY -m pytest -q tests/test_long_horizon.py || exit 1
$PY test_fusion_r14.py | tail -1 || exit 1
rm -rf runs_long_horizon_smoke
OUT=runs_long_horizon_smoke GRID=8 DIM=16 LAYERS=1 HEADS=2 BATCH=2 TARGET_PARAMS=0 \
STEPS=4 T_LONG=24 T_ATTN=16 GAP=8 EVAL_ROLLOUT=48 EVAL_SEEDS=2 SEEDS=0 \
STREAM_FRAMES=60 SLICE_S=900 bash run_long_horizon.sh || exit 1
grep -q DONE runs_long_horizon_smoke/status.txt || exit 1
[ "$(ls runs_long_horizon_smoke/result_*.json | wc -l)" = "5" ] || exit 1
echo "smoke long-horizon PASS $(date -u +%FT%TZ)"
exit 0
