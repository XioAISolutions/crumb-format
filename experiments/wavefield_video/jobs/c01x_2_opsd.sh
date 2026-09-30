#!/bin/bash
set -e
cd /root/opsd-v
exec > /workspace/slava/logs/gpuq_c01x_2_opsd.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /usr/bin/timeout 5400 /workspace/slava/venvs/opsd/bin/python inference.py $(tr '\n' ' ' < /workspace/slava/opsd_args_opsd.txt)
