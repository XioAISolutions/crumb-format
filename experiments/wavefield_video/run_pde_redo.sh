#!/bin/bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs_pde
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--data-source waves --field wave --steps 3000 --dim 192 --layers 6 --heads 8 --grid 16 --frames 16"
C="$C --batch 64 --target-params 1600000 --causal --residual --linear-pad --motion-loss --rollout-ramp"
C="$C --eval-rollout 256 --eval-seeds 16 --auto-batch --save-every 500 --out runs_pde"
$P train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse $C --tag _w1wave > log_pde_w1.txt 2>&1
$P train_compare.py --kind attn $C --tag _a1wave > log_pde_attn.txt 2>&1
echo DONE > runs_pde/pde_status.txt
[ -f run_retune.sh ] && bash run_retune.sh
