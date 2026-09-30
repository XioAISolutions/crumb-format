#!/bin/bash
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01ae_3_nofollow10s_s0.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_PROMPT_CHUNK=16 LL_OFFLOAD_IDLE=1 LL=/root/LongLive
export JOB_ID=c01ae_3_nofollow10s_s0 OUT=runs_probe_nofollow10s_s0 LENGTHS=10 PRECISION=fp8 WINDOW=12 SINK=8 SEED=0 RELATIVE_ROPE=on
export PROMPTS=/workspace/slava/exp/pr63/prompts_nofollow.txt
exec /usr/bin/timeout 2400 bash run_longlive.sh
