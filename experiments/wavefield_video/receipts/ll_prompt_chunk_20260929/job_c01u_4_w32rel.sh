#!/bin/bash
# c01u_4: window isolation with current rope — window 32, relative RoPE on
# [mission-pump pre-fire fix 2026-09-29, before first fire: see c01u_1 (PY gate,
#  log redirect, real timeout wrapper; LL + LL_PROMPT_CHUNK pinned). Staged original
#  kept in gpu_queue/superseded/.]
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01u_4_w32rel.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01u_4_w32rel
export OUT=runs_probe_w32rel
export LENGTHS=10
export PRECISION=fp8
export WINDOW=32
export SINK=8
export RELATIVE_ROPE=on
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
