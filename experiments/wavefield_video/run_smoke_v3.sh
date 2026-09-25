#!/usr/bin/env bash
# CPU smoke for the v3 streaming + hardening round. Tiny + fast: exercises the
# new code paths (dispersion step(), --stream-test, --auto-batch, --save-every /
# --resume) for shape/NaN/regression -- NOT an optimization result.
#
# Operator runs this (python is session-gated). One approval runs all of it:
#   cd experiments/wavefield_video && bash run_smoke_v3.sh
#   # or: PY=/path/to/venv/bin/python bash run_smoke_v3.sh
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_v3}"
mkdir -p "$OUT"

echo "### (1) sanity_check.py  (now incl. dispersion forward==step + wave stream==forward)"
"$PY" sanity_check.py

# Small model shared by every arm below.
M="--dim 48 --layers 2 --heads 4 --grid 8 --frames 8 --target-params 120000"

echo
echo "### (2) --stream-test : wave (dispersion, O(1) step) vs attn/ssm (windowed)"
# 384 frames -> checkpoints at 128/256/384; batch small so it runs on CPU quickly.
STREAM="--stream-test --stream-frames 384 --stream-batch 4 --out $OUT --collisions"
"$PY" train_compare.py --kind wave --kernel-version dispersion --linear-pad $M $STREAM --tag _wave
"$PY" train_compare.py --kind attn $M $STREAM --tag _attn
"$PY" train_compare.py --kind ssm  $M $STREAM --tag _ssm
echo "    -> $OUT/stream_{wave,attn,ssm}_*.json  (peak_gb + fps per 128 frames)"

echo
echo "### (3)+(4) checkpoint roundtrip: save-every, then --resume continues"
CKPT="--kind wave --kernel-version dispersion --linear-pad $M --batch 8 \
      --eval-batches 2 --eval-rollout 16 --eval-seeds 4 --collisions --auto-batch --out $OUT"
# Run A: 10 steps, checkpoint every 5 -> $OUT/ckpt_wave_ck.pt at step 10.
"$PY" train_compare.py $CKPT --steps 10 --save-every 5 --tag _ck | tail -4
# Run B: resume and train to step 20 (total_steps must match the resumed run's
# --steps for the OneCycleLR fast-forward; here we extend on purpose to show the
# loop restarts at step 11, not 1).
"$PY" train_compare.py $CKPT --steps 20 --save-every 5 --tag _ck \
      --resume "$OUT/ckpt_wave_ck.pt" | tail -6

echo
echo "### smoke v3 done -> $OUT/"
