#!/bin/bash
# c01u_1: prompt-dynamics probe — dynamic upstream prompt at the current config (W12, rel-rope on)
# [mission-pump pre-fire fix 2026-09-29, before first fire: the staged version set no
#  PY (under the conductor's clean env run_longlive.sh exits rc=2 at its interpreter
#  gate), had no own log redirect, and TIMEOUT=2400 was a no-op. Added PY +
#  PYTORCH_CUDA_ALLOC_CONF, own log, real timeout wrapper; LL pinned;
#  LL_PROMPT_CHUNK=16 pinned (prompt-chunk patch carries on box). Staged original
#  kept in gpu_queue/superseded/. prompts_dynamic.txt copied into pr63/.]
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01u_1_prompt.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LL_PROMPT_CHUNK=16
export JOB_ID=c01u_1_prompt
export OUT=runs_probe_prompt
export LENGTHS=10
export PRECISION=fp8
export WINDOW=12
export SINK=8
export RELATIVE_ROPE=on
export PROMPTS=/workspace/slava/exp/pr63/prompts_dynamic.txt
export LL_OFFLOAD_IDLE=1
export LL=/root/LongLive
exec /usr/bin/timeout 2400 bash run_longlive.sh
