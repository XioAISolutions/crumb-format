#!/bin/bash
# Long-budget g32 run: 4x steps vs the matrix (per-token budget control)
cd /workspace/slava/exp/wavefield_video || exit 1
P=/workspace/slava/comfy-house/venv/bin/python
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --causal --residual --linear-pad --collisions --motion-loss \
  --rollout-ramp --steps 8000 --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 \
  --auto-batch --save-every 500 --out runs_deep --tag _g32_long8k > log_longbudget_g32.txt 2>&1
echo "longbudget exit=$? $(date)" >> runs_deep/longbudget_progress.txt
echo DONE > runs_deep/longbudget_status.txt
