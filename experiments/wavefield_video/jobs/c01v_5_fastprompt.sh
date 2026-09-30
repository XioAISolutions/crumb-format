#!/bin/bash
# c01v_5: fast-travel prompt variant (ladder prompt says "slow steadicam walk").
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01v_5_fastprompt.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01v_5_fastprompt
export OUT=runs_probe_fastprompt
export LENGTHS=10
export PRECISION=fp8
export WINDOW=12
export SINK=8
export RELATIVE_ROPE=on
export LL_OFFLOAD_IDLE=1
export PROMPTS=/workspace/slava/exp/pr63/prompts_street_fast.txt
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
