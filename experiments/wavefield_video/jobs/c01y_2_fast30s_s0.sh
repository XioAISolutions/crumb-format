#!/bin/bash
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01y_2_fast30s_s11.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_PROMPT_CHUNK=16 LL_OFFLOAD_IDLE=1 LL=/root/LongLive
export JOB_ID=c01y_2_fast30s_s0 OUT=runs_probe_fast30s_s0 LENGTHS=30 PRECISION=fp8 WINDOW=12 SINK=8 SEED=0 RELATIVE_ROPE=on
export PROMPTS=/workspace/slava/exp/pr63/prompts_street_fast.txt
exec /usr/bin/timeout 3600 bash run_longlive.sh
