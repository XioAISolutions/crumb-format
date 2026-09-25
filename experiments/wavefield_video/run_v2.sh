#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs_v2
C="--causal --residual --linear-pad --collisions"
$P train_compare.py --kind ssm $C --steps 3000 --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --batch 128 --target-params 1600000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _s1 > log_s1.txt 2>&1
$P train_compare.py --kind attn $C --steps 3000 --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --batch 128 --target-params 1600000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _a1 > log_a1.txt 2>&1
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C --steps 3000 --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --batch 64 --target-params 1600000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _w1 > log_w1.txt 2>&1
$P train_compare.py --kind wave --kernel-version separable --gate --local-fuse $C --steps 3000 --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --batch 64 --target-params 1600000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _w0 > log_w0.txt 2>&1
$P train_compare.py --kind ssm $C --steps 2000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 64 --target-params 4000000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _s1g32 > log_s1g32.txt 2>&1
$P train_compare.py --kind attn $C --steps 2000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 64 --target-params 4000000 --ckpt --rollout-loss 3 --eval-rollout 256 --eval-seeds 16 --out runs_v2 --tag _a1g32 > log_a1g32.txt 2>&1
echo DONE-ALL > runs_v2/status.txt
