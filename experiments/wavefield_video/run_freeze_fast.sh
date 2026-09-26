#!/bin/bash
# Freeze fast-suite (DEEP_DIVE_3_opus 1.4): B3/B4/B5/B6 + S1. One knob per arm.
cd /workspace/slava/exp/wavefield_video || exit 1
PY=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs_dd3
BASE="--kernel-version dispersion --gate --local-fuse --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 --target-params 4000000 --causal --residual --linear-pad --collisions --motion-loss --rollout-loss 1 --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --telemetry-every 25 --out runs_dd3"
run(){ tag=$1; shift; echo "== $tag start $(date -u +%H:%M)" >> runs_dd3/progress.txt; $PY train_compare.py $BASE "$@" --tag "_$tag" > log_dd3_$tag.txt 2>&1; echo "$tag exit=$? $(date -u +%H:%M)" >> runs_dd3/progress.txt; }
run B3 --const-lr --steps 2000 --kind wave --data-source balls --n-balls 3 --speed 2.3 --radius 1.6
run B4 --steps 2000 --batch 64 --kind wave --data-source balls --n-balls 3 --speed 2.3 --radius 1.6
run B5 --fp32 --steps 2000 --kind wave --data-source balls --n-balls 3 --speed 2.3 --radius 1.6
run B6 --no-decay-norm-head --steps 2000 --kind wave --data-source balls --n-balls 3 --speed 2.3 --radius 1.6
run S1 --steps 500 --grid 16 --kind wave --data-source balls --n-balls 3 --speed 2.3 --radius 1.6
echo SUITE-FAST-DONE > runs_dd3/fast_status.txt
