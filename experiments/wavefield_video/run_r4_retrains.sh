#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
C="--steps 3000 --dim 192 --layers 6 --heads 8 --grid 16 --frames 16 --batch 64 --target-params 1600000 --causal --residual --linear-pad --collisions --motion-loss --rollout-ramp --eval-rollout 256 --eval-seeds 16 --auto-batch --out runs_r4"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C --tag _w1r4 > log_r4_w1.txt 2>&1
$P train_compare.py --kind ssm $C --tag _s1r4 > log_r4_ssm.txt 2>&1
$P train_compare.py --kind attn $C --tag _a1r4 > log_r4_attn.txt 2>&1
echo DONE > runs_r4/r4_status.txt
