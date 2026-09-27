#!/usr/bin/env bash
# One sliced queue job of the R15 occlusion suite (HYPERCOMPLEX_STUDY.md sec 5.4:
# real jobs via the conductor, one at a time). run_occlusion.sh in RESUME mode
# skips finished arms, resumes live ones from their ckpt and stops at the slice
# boundary; queue this repeatedly until occlusion_status.txt reads DONE.
#
# Recipe knobs -- set after the --const-lr pre-flight probe and KEEP FIXED across
# slices of one OUT:
#   CONST_LR=1 STEPS=8000   escalated: probe froze at 2k (the campaign's unfreeze
#                           recipe is budget x schedule together, see sec 5.2)
#   CONST_LR=1 STEPS=2000   probe showed training at the short budget
set -euo pipefail
cd /workspace/slava/exp/wavefield_video
export RESUME=1
export CONST_LR=${CONST_LR:-1}
export STEPS=${STEPS:-8000}
export SLICE_S=${SLICE_S:-4200}
export PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
exec bash run_occlusion.sh
