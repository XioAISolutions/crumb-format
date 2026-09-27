#!/bin/bash
# Attention-ladder suite slice (card t_d77fd4cd): one self-organizing queue unit
# -- skips arms with a result JSON, resumes live ones from ckpt, stops at the
# SLICE_S wall. All copies are identical; trailing ones no-op once DONE.
set -euo pipefail
cd /workspace/slava/exp/wavefield_video
export RESUME=1 CONST_LR=1 STEPS=8000 SLICE_S=4200
export PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
export OUT=runs_attn_ladder
exec bash run_attn_ladder.sh
