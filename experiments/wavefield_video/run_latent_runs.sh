#!/bin/bash
cd /workspace/slava/exp/wavefield_video
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p ckpts runs_latent
$P latent_ae.py --train --data-source balls --grid 32 --steps 4000 --batch 64 --out ckpts/ae_g32.pt > log_latent_ae.txt 2>&1
L="--latent --ae-ckpt ckpts/ae_g32.pt --steps 3000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17"
L="$L --batch 64 --target-params 1600000 --causal --residual --linear-pad --collisions"
L="$L --rollout-ramp --eval-rollout 128 --eval-seeds 8 --auto-batch --save-every 500 --out runs_latent"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $L --tag _w1lat > log_latent_w1.txt 2>&1
$P train_compare.py --kind attn $L --tag _a1lat > log_latent_attn.txt 2>&1
echo DONE > runs_latent/latent_status.txt
[ -f run_pde_redo.sh ] && bash run_pde_redo.sh
