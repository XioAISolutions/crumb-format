#!/bin/bash
# Anti-collapse suite (RESEARCH_SWEEP_20260926 sec 2-4). Four arms, ONE anti-
# collapse knob each, layered on the g32 --kind wave arm that FREEZES
# (copy_ratio ~ 0) even after the const-lr / fp32 / no-decay freeze fixes from
# the DEEP_DIVE_3 Part-1 suite all fail to unfreeze it. seed 0, 2000 steps,
# telemetry ON (--telemetry-every 25) so gate_g / head_gn / copy_r are logged.
#
#   varreg    --var-reg 0.5      VICReg-style variance-match on frame deltas   (sec 2)
#   drift     --drift-pert 0.05  drift-perturbed context augmentation          (sec 3)
#   histdisc  --hist-disc bits   16-level discrete history representation      (sec 2/4)
#   combo     best-two           default = var-reg + drift-pert (see BLOCKING)
#
# Read copy_ratio first (NOT the 256-mean MSE): ~0 = still frozen, ~1 = correct
# motion magnitude, >>1 = unstable. An arm "unfreezes" if copy_ratio climbs off
# ~0.005 toward ~1 without the rollout blowing up.
#
# ============================ BLOCKING NOTE ================================ #
# "combo best-two" is PRE-WIRED to var-reg + drift-pert -- the loss-level and
# data-level knobs, which are the most mechanistically complementary (one
# punishes the collapsed output directly, the other invalidates the copy-last
# shortcut in the data). This is a defensible default, NOT a measured result.
# AFTER the three single-knob arms finish, look at their copy_ratio in
# runs_anticollapse/result_wave_*.json; if hist-disc lands in the top two,
# re-run the combo with the actual best pair, e.g.:
#     COMBO="--drift-pert 0.05 --hist-disc bits" bash run_anticollapse.sh
# (the single-knob arms are idempotent by tag; only _combo is rewritten).
# =========================================================================== #
cd /workspace/slava/exp/wavefield_video || exit 1
PY=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs_anticollapse
SEED="${SEED:-0}"
STEPS="${STEPS:-2000}"
# Same base as run_freeze_fast.sh (the arm under study), plus telemetry.
BASE="--kernel-version dispersion --gate --local-fuse --dim 192 --layers 6 --heads 8 \
--grid 32 --frames 17 --batch 16 --target-params 4000000 --causal --residual --linear-pad \
--collisions --motion-loss --rollout-loss 1 --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 \
--auto-batch --telemetry-every 25 --kind wave --data-source balls --n-balls 3 --speed 2.3 \
--radius 1.6 --seed $SEED --steps $STEPS --out runs_anticollapse"
run(){ tag=$1; shift; echo "== $tag start $(date -u +%H:%M)" >> runs_anticollapse/progress.txt; \
  $PY train_compare.py $BASE "$@" --tag "_$tag" > log_ac_$tag.txt 2>&1; \
  echo "$tag exit=$? $(date -u +%H:%M)" >> runs_anticollapse/progress.txt; }

run varreg   --var-reg 0.5
run drift    --drift-pert 0.05
run histdisc --hist-disc bits
# combo best-two -- override with COMBO="..." after reading the single-knob copy_ratios.
COMBO="${COMBO:---var-reg 0.5 --drift-pert 0.05}"
run combo $COMBO
echo SUITE-ANTICOLLAPSE-DONE > runs_anticollapse/ac_status.txt
echo "anti-collapse suite done -> runs_anticollapse/ (read copy_ratio in result_wave_*.json)"
