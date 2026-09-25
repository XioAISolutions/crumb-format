#!/usr/bin/env bash
# CPU smoke for the v2 suite: sanity gate, then each arm at grid 8 and 16.
# Tiny (few steps) -- catches shape/NaN/regressions, NOT an optimization result.
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_v2}"
STEPS="${STEPS:-30}"

echo "### sanity_check.py"
"$PY" sanity_check.py

# Small but honest eval so the divergence-horizon plumbing exercises for real.
COMMON="--steps $STEPS --dim 48 --layers 2 --heads 4 --batch 8 \
        --eval-batches 3 --eval-rollout 24 --eval-seeds 8 --out $OUT \
        --target-params 120000 --collisions"

for G in 8 16; do
  echo "### grid $G : wave (dispersion + gate + local-fuse + rollout-loss)"
  "$PY" train_compare.py --kind wave --grid $G --frames 8 --tag "_g${G}" $COMMON \
      --kernel-version dispersion --gate --local-fuse --linear-pad \
      --rollout-loss 3 --causal | tail -8

  echo "### grid $G : attn (param-matched)"
  "$PY" train_compare.py --kind attn --grid $G --frames 8 --tag "_g${G}" $COMMON \
      --causal | tail -8

  echo "### grid $G : ssm (param-matched)"
  "$PY" train_compare.py --kind ssm --grid $G --frames 8 --tag "_g${G}" $COMMON \
      --causal | tail -8

  echo "### grid $G : wave separable (v1 kernel, v2 harness) reference"
  "$PY" train_compare.py --kind wave --grid $G --frames 8 --tag "_g${G}_sep" $COMMON \
      --kernel-version separable | tail -8
done

echo "### smoke done -> $OUT/"
