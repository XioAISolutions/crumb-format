#!/usr/bin/env bash
# crumb_coherence M0 smoke — CPU-only. Runs unit tests then the synthetic-drift
# M0 harness for BOTH scenarios. No commits, no GPU. From wavefield_video/:
#     bash run_m0_smoke.sh
# Use the project torch env (system python usually lacks torch):
#     PYTHON=~/vibevoice-env/bin/python bash run_m0_smoke.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""          # force CPU
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
echo "using python: $PY"

echo "### 1/3  unit tests (invariants + state_bytes) ###########################"
"$PY" crumb_coherence/tests/test_core.py

echo
echo "### 2/3  M0 gain_field (exposure/color = magnitude drift) ################"
"$PY" crumb_coherence/scripts/run_m0.py --frames 48 --grid 64 --scenario gain_field --no-video

echo
echo "### 3/3  M0 hotspot (wandering bump = positional/phase drift) ############"
"$PY" crumb_coherence/scripts/run_m0.py --frames 48 --grid 64 --scenario hotspot --no-video

echo
echo "M0 smoke complete."
