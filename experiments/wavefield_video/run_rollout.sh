#!/usr/bin/env bash
# Fresh paired training probe. No changes to run_long_horizon.sh or its defaults.
# Explicit opt-in recipe, not a quality gate. Use a NEW OUT for each invocation.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
: "${OUT:?Set OUT to a fresh directory (never reuse a completed probe)}"
if [ -e "$OUT" ]; then
    printf 'OUT already exists; use a fresh directory: %s\n' "$OUT" >&2; exit 2
fi
# Conservative preflight, not a claimed 24 GB measurement. FP32 spectral math,
# single-sample micro-batches, one chunk of backprop, activation recomputation.
GRID=${GRID:-16}; DIM=${DIM:-64}; LAYERS=${LAYERS:-2}; HEADS=${HEADS:-4}
CHUNK=${CHUNK:-8}; SEQ_FRAMES=${SEQ_FRAMES:-32}; ROLLOUT_K=${ROLLOUT_K:-2}
STEPS=${STEPS:-20}; SEED=${SEED:-0}; BATCH=${BATCH:-2}; MICRO_BATCH=${MICRO_BATCH:-1}
EVAL_ROLLOUT=${EVAL_ROLLOUT:-64}; EVAL_SEEDS=${EVAL_SEEDS:-4}
if [ "$ROLLOUT_K" -lt 1 ]; then printf 'ROLLOUT_K must be positive\n' >&2; exit 2; fi
mkdir -p "$OUT"
BASE=(--kind wave --data-source balls --pole-param halflife --dense --grad-ckpt
      --seq-frames "$SEQ_FRAMES" --chunk "$CHUNK" --tbptt-chunks 1
      --grid "$GRID" --dim "$DIM" --layers "$LAYERS" --heads "$HEADS"
      --steps "$STEPS" --seed "$SEED" --batch "$BATCH" --micro-batch "$MICRO_BATCH"
      --lr 0.002 --eval-rollout "$EVAL_ROLLOUT" --eval-seeds "$EVAL_SEEDS"
      --eval-chunk 1 --eval-batch 1 --save-every 1 --out "$OUT")
# Identical initialization/data/steps/targets. Difference: only final K chunks'
# inputs are generated. Equal steps are NOT equal wall time: report train_sec.
printf '%q ' "$PY" train_long.py "${BASE[@]}" > "$OUT/command.txt"
printf '\nROLLOUT_K=%s\n' "$ROLLOUT_K" >> "$OUT/command.txt"
for k in 0 "$ROLLOUT_K"; do
    "$PY" train_long.py "${BASE[@]}" --rollout-k "$k" --tag "_k${k}" \
        2>&1 | tee "$OUT/train_k${k}.log"
done
"$PY" verify_rollout_run.py "$OUT" --rollout-k "$ROLLOUT_K" --steps "$STEPS"
printf 'PAIRED ROLLOUT PROBE COMPLETE: %s (not a quality verdict)\n' "$OUT"
