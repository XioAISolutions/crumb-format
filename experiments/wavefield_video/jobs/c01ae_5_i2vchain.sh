#!/bin/bash
# c01ae_5: i2v chain probe — continue from the LAST FRAME of the fast 30s travel clip,
# seed 0, fast dialect. If flow >= ~1.0 this unlocks stitched multi-scene long videos.
set -e
cd /workspace/slava/exp/pr63
exec > /workspace/slava/logs/gpuq_c01ae_5_i2vchain.log 2>&1
export PY=/workspace/slava/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_PROMPT_CHUNK=16 LL_OFFLOAD_IDLE=1 LL=/root/LongLive
mkdir -p /workspace/slava/exp/pr63/i2v_chain
FF=/workspace/slava/exp/pr63/runs_probe_fast30s_s0/len_30s/rank0-0-0_regular.mp4
/usr/bin/ffmpeg -y -sseof -0.2 -i "$FF" -frames:v 1 /workspace/slava/exp/pr63/i2v_chain/street.png -loglevel error
printf '%s\n' "Brisk forward walk continuing down the sunlit cobblestone alley, camera advancing steadily and quickly at eye level, continuous fast forward dolly motion, buildings sliding past on both sides, the alley opening onto a wide sunny square ahead, consistent daylight, realistic colours." > /workspace/slava/exp/pr63/i2v_chain/street.txt
export I2V=1 PROMPTS=/workspace/slava/exp/pr63/i2v_chain
export JOB_ID=c01ae_5_i2vchain OUT=runs_probe_i2vchain LENGTHS=10 PRECISION=fp8 WINDOW=12 SINK=8 SEED=0 RELATIVE_ROPE=on
exec /usr/bin/timeout 2400 bash run_longlive.sh
