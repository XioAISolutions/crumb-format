#!/bin/bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
BASE="--kind wave --kernel-version dispersion --gate --local-fuse --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 --target-params 4000000 --causal --residual --linear-pad --collisions --motion-loss --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --save-every 500 --out runs_deep"
NC="--kind wave --kernel-version dispersion --gate --local-fuse --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 --target-params 4000000 --causal --residual --linear-pad --motion-loss --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --save-every 500 --out runs_deep"
echo "retune start $(date)" > runs_deep/retune_progress.txt
run(){ tag=$1; shift; echo "== $tag $(date)"; $P train_compare.py "$@" --tag $tag > log_retune${tag}.txt 2>&1; echo "$tag exit=$? $(date)" >> runs_deep/retune_progress.txt; }
run _g32_ctrl_K1    $BASE --rollout-loss 1
run _g32_balresid   $BASE --rollout-loss 1 --resid-balanced
run _g32_motionw    $BASE --rollout-loss 1 --motion-weighted
run _g32_bal_mw     $BASE --rollout-loss 1 --resid-balanced --motion-weighted
run _g32_K1_nocoll  $NC --rollout-loss 1
run _g32_ramp_nocoll $NC --rollout-ramp
run _g32geo_big     $BASE --rollout-loss 1 --radius 3.2 --speed 2.30
echo DONE > runs_deep/retune_status.txt
