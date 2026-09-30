#!/bin/bash
# c01v_3: seed sweep 2 — street prompt, ladder config, seed 2.
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01v_3_seed2.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01v_3_seed2
export OUT=runs_probe_seed2
export LENGTHS=10
export PRECISION=fp8
export WINDOW=12
export SINK=8
export SEED=2
export RELATIVE_ROPE=on
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
