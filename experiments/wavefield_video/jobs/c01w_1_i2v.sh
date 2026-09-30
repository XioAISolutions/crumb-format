#!/bin/bash
# c01w_1: I2V probe — continue forward from the c01q street frame (clamped anchor).
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01w_1_i2v.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
mkdir -p /workspace/slava/exp/pr63/i2v_data
cp -f /workspace/slava/street_frame0.png /workspace/slava/exp/pr63/i2v_data/street.png
printf '%s\n' "The camera continues walking forward down the sunlit cobblestone street from this exact viewpoint, steadily advancing past the buildings at a brisk pace, continuous forward dolly motion, the street opening up ahead, same warm light and weather." > /workspace/slava/exp/pr63/i2v_data/street.txt
export I2V=1
export PROMPTS=/workspace/slava/exp/pr63/i2v_data
export JOB_ID=c01w_1_i2v
export OUT=runs_probe_i2v
export LENGTHS=10
export PRECISION=fp8
export WINDOW=12
export SINK=8
export RELATIVE_ROPE=on
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
