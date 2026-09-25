#!/bin/bash
cd /workspace/slava/exp/wavefield_video
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 900); do [ -f runs_pde/pde_status.txt ] && break; sleep 60; done
sleep 20
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs_deep
C="--steps 2000 --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16"
C="$C --target-params 4000000 --causal --residual --linear-pad --collisions"
C="$C --motion-loss --rollout-ramp --eval-rollout 256 --eval-seeds 16 --auto-batch --save-every 500 --out runs_deep"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C --tag _w1g32b > log_deep_w1g32b.txt 2>&1
$P train_compare.py --kind attn $C --steps 800 --tag _a1g32b > log_deep_a1g32b.txt 2>&1
G="--steps 1500 --dim 192 --layers 6 --heads 8 --grid 64 --frames 9 --batch 6 --target-params 4000000"
G="$G --causal --residual --linear-pad --collisions --motion-loss --rollout-ramp --eval-rollout 64 --eval-seeds 8 --auto-batch --save-every 500 --out runs_deep"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $G --tag _w1g64 > log_deep_w1g64.txt 2>&1
echo DONE > runs_deep/deep1_status.txt
[ -f run_latent_runs.sh ] && bash run_latent_runs.sh
