#!/bin/bash
# LONG_HORIZON phase 2 slice (template): the wave model on LTX latents of real
# video. EDIT VIDEOS below to a folder of mp4s on the box, then copy to
# gpuq_job_7NN_latent_slice.sh as many times as needed; copies no-op once
# runs_latent/status.txt reads DONE. First run needs Hugging Face access for
# Lightricks/LTX-Video (or set VAE_PATH to a local copy of its vae/ folder).
VIDEOS=${VIDEOS:-/workspace/slava/videos}
LOG=/workspace/slava/logs/$(basename "$0" .sh).log
mkdir -p /workspace/slava/logs
exec > "$LOG" 2>&1
set -o pipefail
cd /workspace/slava/exp/wavefield_video || exit 1
grep -qs DONE runs_latent/status.txt && { echo "latent run DONE -- no-op"; exit 0; }
export PY=/workspace/slava/comfy-house/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c "import diffusers, imageio_ffmpeg" || { echo "missing deps: pip install -r requirements-video.txt"; exit 1; }
rc=0
VIDEOS="$VIDEOS" OUT=runs_latent SLICE_S=4800 bash run_latent.sh || rc=$?
tail -5 runs_latent/progress.txt
exit "$rc"
