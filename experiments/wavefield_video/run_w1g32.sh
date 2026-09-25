#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse --causal --residual --linear-pad --collisions --steps 2000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 --target-params 4000000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _w1g32 > log_w1g32.txt 2>&1
echo DONE > runs_v2/w1g32_status.txt
