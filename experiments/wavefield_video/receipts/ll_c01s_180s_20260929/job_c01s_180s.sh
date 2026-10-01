#!/bin/bash
# c01s: 180s rung — same config as c01r (fp8, W12, sink 8, idle-offload).
# v3 (2026-09-29, post first fire): v2 died at TE prompt-encode before denoise —
# one batched call for all 136 block-prompts allocates attn_bias O(batch)
# (t5.py:102; 4.25 GiB needed vs 3.43 free). Box now carries LL_PROMPT_CHUNK_V1
# (chunked prompt encode, /root/LongLive/utils/prompt_conditioning.py md5
# 8a8fe8a544f5a646e38eed78ba44314f); LL_PROMPT_CHUNK=16 pinned below. Fresh OUT
# (cfg16i): cfg16f holds the failed attempt's logs.
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01s_180s.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01s_180s
export OUT=runs_longlive_cfg16i
export LENGTHS=180
export PRECISION=fp8
export WINDOW=12
export SINK=8
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 4800 bash run_longlive.sh
