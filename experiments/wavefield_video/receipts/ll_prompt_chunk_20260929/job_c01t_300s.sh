#!/bin/bash
# c01t: 300s rung — same config (fp8, W12, sink 8, idle-offload).
# v3 (2026-09-29, post first fire): v2 died at TE prompt-encode before denoise —
# one batched call for all 226 block-prompts allocates attn_bias O(batch)
# (t5.py:102; 7.06 GiB needed vs 1.45 free). Fixed by LL_PROMPT_CHUNK_V1
# (chunked prompt encode, /root/LongLive/utils/prompt_conditioning.py md5
# 8a8fe8a544f5a646e38eed78ba44314f); LL_PROMPT_CHUNK=16 pinned below. Fresh OUT
# (cfg16j): cfg16g holds the failed attempt's logs.
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01t_300s.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01t_300s
export OUT=runs_longlive_cfg16j
export LENGTHS=300
export PRECISION=fp8
export WINDOW=12
export SINK=8
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 5280 bash run_longlive.sh
