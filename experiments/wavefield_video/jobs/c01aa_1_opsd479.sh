#!/bin/bash
set -e
cd /root/opsd-v
exec > /workspace/slava/logs/gpuq_c01aa_1_opsd479.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /usr/bin/timeout 7200 /workspace/slava/venvs/opsd/bin/python inference.py $(tr '\n' ' ' < /workspace/slava/opsd_args_479.txt)
