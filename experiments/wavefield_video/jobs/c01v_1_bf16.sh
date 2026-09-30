#!/bin/bash
# c01v_1: bf16 control probe — is fp8 quant damping dynamics? street prompt, W24, 10s
# (PY gate + log redirect + timeout wrapper per the pump's pre-fire template.)
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01v_1_bf16.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01v_1_bf16
export OUT=runs_probe_bf16
export LENGTHS=10
export PRECISION=bf16
export WINDOW=24
export SINK=8
export RELATIVE_ROPE=on
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 3000 bash run_longlive.sh
