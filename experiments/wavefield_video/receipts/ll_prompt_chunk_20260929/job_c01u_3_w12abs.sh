#!/bin/bash
# c01u_3: rope isolation at our window — window 12, absolute RoPE
# [mission-pump pre-fire fix 2026-09-29, before first fire: see c01u_1 (PY gate,
#  log redirect, real timeout wrapper; LL + LL_PROMPT_CHUNK pinned). Staged original
#  kept in gpu_queue/superseded/.]
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01u_3_w12abs.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01u_3_w12abs
export OUT=runs_probe_w12abs
export LENGTHS=10
export PRECISION=fp8
export WINDOW=12
export SINK=8
export RELATIVE_ROPE=off
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
