#!/usr/bin/env bash
# CPU correctness and a tiny end-to-end pair; on the box training uses CUDA.
# The gpuq wrapper serializes GPU ownership. Never enqueue from this script.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PY=${PY:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# Keep test scratch contained even if host TMPDIR points at a system temp dir.
mkdir -p runs_rollout_test_tmp
export TMPDIR="$PWD/runs_rollout_test_tmp"
: "${OUT:?Set OUT to a fresh smoke directory}"
"$PY" -m pytest -q tests/test_rollout_training.py tests/test_stateful.py tests/test_long_horizon.py
"$PY" mutation_check_rollout.py
GRID=8 DIM=16 LAYERS=2 HEADS=2 CHUNK=4 SEQ_FRAMES=16 ROLLOUT_K=2 \
STEPS=4 BATCH=2 MICRO_BATCH=1 EVAL_ROLLOUT=16 EVAL_SEEDS=2 bash run_rollout.sh
# Exercise the EXISTING checkpoint loader and no-grad carried-state streamer.
"$PY" long_horizon.py stream --pole-param halflife --ckpt "$OUT/model_wave_k2.pt" \
    --grid 8 --frames 4 --dim 16 --layers 2 --heads 2 --time-pos none \
    --batch 1 --stream-frames 48 --chunk 12 --out "$OUT/stream_smoke.json"
verify_args=(--stream-smoke)
if [ "${REQUIRE_CUDA:-0}" = 1 ]; then verify_args+=(--require-cuda); fi
"$PY" verify_rollout_run.py "$OUT" --rollout-k 2 --steps 4 "${verify_args[@]}"
printf 'ROLLOUT SMOKE PASS (execution only; collapse flags are NOT quality passes)\n'
