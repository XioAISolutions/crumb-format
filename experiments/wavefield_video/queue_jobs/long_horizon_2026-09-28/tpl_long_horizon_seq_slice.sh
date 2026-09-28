#!/bin/bash
# LONG_HORIZON carried-state suite slice (SEQ=only; LONG_HORIZON.md 8): copy to gpuq_job_6NN_long_horizon_slice.sh
# as many times as needed (5 arms x 3 seeds at 4000 steps ~ 10-20 slices; copies
# no-op once runs_long_horizon_seq/status.txt reads DONE). Each copy runs one slice
# (<= SLICE_S per arm, under the conductor's 5400s cap) and resumes the next.
# Keep STEPS fixed across slices of one OUT.
LOG=/workspace/slava/logs/$(basename "$0" .sh).log
mkdir -p /workspace/slava/logs
exec > "$LOG" 2>&1
cd /workspace/slava/exp/wavefield_video || exit 1
grep -qs DONE runs_long_horizon_seq/status.txt && { echo "suite DONE -- no-op"; exit 0; }
export PY=/workspace/slava/comfy-house/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rc=0
SEQ=only OUT=runs_long_horizon_seq STEPS=4000 SEEDS="0 1 2" SLICE_S=4800 bash run_long_horizon.sh || rc=$?
tail -5 runs_long_horizon_seq/progress.txt
exit "$rc"                      # FAILED suites must fail the queue job
