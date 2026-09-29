#!/bin/bash
set -euo pipefail
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01p_fa2.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT=runs_longlive_cfg16c LENGTHS=10 PRECISION=fp8 WINDOW=16 LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
