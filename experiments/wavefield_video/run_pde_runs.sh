#!/bin/bash
cd /workspace/slava/exp/wavefield_video
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs_pde
for i in $(seq 1 900); do [ -f runs_r4/r4_clean_done.txt ] && break; sleep 60; done
sleep 20
P=/workspace/slava/comfy-house/venv/bin/python
C="--data-source waves --field wave --steps 3000 --dim 192 --layers 6 --heads 8 --grid 16 --frames 16 --batch 64 --target-params 1600000 --causal --residual --linear-pad --motion-loss --rollout-ramp --eval-rollout 256 --eval-seeds 16 --auto-batch --out runs_pde"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C --tag _w1wave > log_pde_w1.txt 2>&1
$P train_compare.py --kind attn $C --tag _a1wave > log_pde_attn.txt 2>&1
echo DONE > runs_pde/pde_status.txt
