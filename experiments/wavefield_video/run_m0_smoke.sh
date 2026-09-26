#!/usr/bin/env bash
# crumb_coherence M0 smoke — CPU-only. Runs unit tests then the synthetic-drift
# M0 harness. No commits, no GPU. From the wavefield_video/ directory:
#     bash run_m0_smoke.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""          # force CPU
cd "$(dirname "$0")"

echo "### 1/2  unit tests (invariants + state_bytes) ###########################"
python crumb_coherence/tests/test_core.py

echo
echo "### 2/2  M0 synthetic-drift harness ######################################"
python crumb_coherence/scripts/run_m0.py --frames 48 --grid 64

echo
echo "M0 smoke complete."
