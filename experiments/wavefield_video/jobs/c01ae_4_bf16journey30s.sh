#!/bin/bash
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01ae_4_bf16journey30s.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_PROMPT_CHUNK=16 LL_OFFLOAD_IDLE=1 LL=/root/LongLive
export JOB_ID=c01ae_4_bf16journey30s OUT=runs_probe_journey30s_bf16 LENGTHS=30 PRECISION=bf16 WINDOW=12 SINK=8 SEED=0 RELATIVE_ROPE=on
export PROMPTS=/workspace/slava/exp/pr63/prompts_journey.txt
exec /usr/bin/timeout 3600 bash run_longlive.sh
