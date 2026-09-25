#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
C="--causal --residual --linear-pad --collisions"
B32="--steps 2000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 --target-params 4000000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2"
$P train_compare.py --kind ssm $C $B32 --tag _s1g32 > log_s1g32.txt 2>&1
$P train_compare.py --kind attn $C $B32 --tag _a1g32 > log_a1g32.txt 2>&1
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C $B32 --tag _w1g32 > log_w1g32.txt 2>&1
echo G32-DONE > runs_v2/status_g32.txt
