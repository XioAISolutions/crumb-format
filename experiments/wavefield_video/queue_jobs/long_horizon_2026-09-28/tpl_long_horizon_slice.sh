#!/bin/bash
# LONG_HORIZON suite slice (template): copy to gpuq_job_*_long_horizon_slice.sh
# as many times as needed (5 arms x 3 seeds at 4000 steps ~ 10-20 slices; copies
# no-op once runs_long_horizon/status.txt reads DONE). Each copy runs one slice
# (<= SLICE_S per arm, under the conductor's 5400s cap) and resumes the next.
# Keep STEPS fixed across slices of one OUT.
# v2 (Zeph 2026-09-28): VRAM pre-gate -- releases ComfyUI's cached models and
# defers if <21 GiB free (co-tenancy OOMs the arm); OOM-rescue -- an arm killed
# by CUDA OOM under co-tenancy is reset to SLICED (same class as a box time-cap
# kill, which the resume design already tolerates) and retried by a later copy;
# real failures still stop and fail the queue job.
LOG=/workspace/slava/logs/$(basename "$0" .sh).log
mkdir -p /workspace/slava/logs
exec > "$LOG" 2>&1
cd /workspace/slava/exp/wavefield_video || exit 1
grep -qs DONE runs_long_horizon/status.txt && { echo "suite DONE -- no-op"; exit 0; }
export PY=/workspace/slava/comfy-house/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- v2 VRAM pre-gate -------------------------------------------------------
curl -s -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' \
     -d '{"unload_models":true,"free_memory":true}' >/dev/null 2>&1 || true
sleep 5
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${FREE:-0}" -lt 21000 ]; then
  echo "defer: only ${FREE:-?} MiB free (<21000) -- another tenant holds VRAM; next copy retries"
  exit 0
fi

rc=0
OUT=runs_long_horizon STEPS=4000 SEEDS="0 1 2" SLICE_S=4800 bash run_long_horizon.sh || rc=$?
tail -5 runs_long_horizon/progress.txt

# --- v2 OOM-rescue ----------------------------------------------------------
if [ "$rc" -ne 0 ] && grep -qs "^FAILED" runs_long_horizon/status.txt; then
  tag=$(awk 'NR==1{print $2}' runs_long_horizon/status.txt)
  if [ -n "$tag" ] && grep -qs "OutOfMemoryError" "runs_long_horizon/log_${tag}.txt"; then
    echo "OOM-rescue: ${tag} killed by CUDA OOM under co-tenancy -> SLICED for retry" \
         | tee -a runs_long_horizon/progress.txt
    echo SLICED > runs_long_horizon/status.txt
    exit 0
  fi
fi
exit "$rc"                      # FAILED suites must fail the queue job
