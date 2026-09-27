#!/usr/bin/env bash
# R15 pre-flight (HYPERCOMPLEX_STUDY.md sec 5.2): one cheap 2k-step wave probe
# with --const-lr at the SUITE config. Its copy_ratio gates the suite budget
# escalation: frozen (~0.02-0.1) -> CONST_LR=1 STEPS=8000 for ALL arms.
# Safe to re-run: skips when the result exists, resumes from ckpt otherwise
# (queue a couple of copies; each one advances or verifies).
set -euo pipefail
cd /workspace/slava/exp/wavefield_video
PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_occlusion}
STEPS=${STEPS:-2000}
mkdir -p "$OUT"
if [ -f "$OUT/result_wave_r15_preflight.json" ]; then
    echo "preflight already done: $OUT/result_wave_r15_preflight.json"
    exit 0
fi
resume_args=()
if [ -f "$OUT/ckpt_wave_r15_preflight.pt" ]; then
    resume_args=(--resume "$OUT/ckpt_wave_r15_preflight.pt")
fi
"$PY" train_compare.py --data-source occlusion --grid 64 --n-balls 8 --frames 17 \
    --batch 8 --target-params 4000000 --causal --residual --linear-pad \
    --dim 192 --layers 6 --heads 8 --occ-start 64 --occ-end 320 --motion-loss \
    --rollout-loss 1 --eval-rollout 1024 --eval-seeds 8 --eval-chunk 1 \
    --auto-batch --save-every 500 --const-lr --kind wave --kernel-version dispersion \
    --steps "$STEPS" --out "$OUT" --tag _r15_preflight \
    ${resume_args[@]+"${resume_args[@]}"} > "$OUT/log_r15_preflight.txt" 2>&1
echo "preflight done $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$OUT/r15_preflight.txt"
