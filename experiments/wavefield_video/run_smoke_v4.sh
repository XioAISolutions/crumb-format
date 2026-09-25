#!/usr/bin/env bash
# CPU smoke for the v4 objective-fix round (Astra #2 + Opus #4). Tiny + fast:
# exercises the new flags for shape/NaN/regression -- NOT an optimization result.
# Each block below PROVES one flag is wired end-to-end.
#
# New this round:
#   * data.moving_mask + make_clip_batch(return_moving=True)  -- motion mask batch
#   * --motion-loss   0.5*E[moving] + 0.5*E[static] + 0.25*MSE(pred-last,tgt-last)
#   * copy_ratio      in eval, result JSON, and the printed table
#   * --rollout-ramp  K 3->8 linearly over the second half of training
#
# Operator runs this (python is session-gated). One approval runs all of it:
#   cd experiments/wavefield_video && bash run_smoke_v4.sh
#   # or: PY=/path/to/venv/bin/python bash run_smoke_v4.sh
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_v4}"
mkdir -p "$OUT"

# Small model shared by every training arm below.
M="--dim 48 --layers 2 --heads 4 --grid 8 --frames 8 --target-params 120000"
EV="--eval-batches 2 --eval-rollout 16 --eval-seeds 4"
COMMON="--kind wave --kernel-version dispersion --linear-pad $M $EV --batch 8 --out $OUT"

echo "### (1) data.py: moving-mask batch (make_clip_batch return_moving / moving_mask)"
"$PY" - <<'PY'
import torch
from data import make_clip_batch, moving_mask, MOVE_THRESH
B, T, G = 4, 8, 8
# return_moving alone -> (out, moving); moving is [B,T,H,W] bool (T frame-pairs).
out, mov = make_clip_batch(B, T, G, G, seed=1, collisions=True, return_moving=True)
assert out.shape == (B, T + 1, 3, G, G), out.shape
assert mov.shape == (B, T, G, G) and mov.dtype == torch.bool, (mov.shape, mov.dtype)
# meta + moving together -> (out, meta, moving), moving always last.
out2, meta, mov2 = make_clip_batch(B, T, G, G, seed=1, collisions=True, return_meta=True, return_moving=True)
assert torch.equal(mov, mov2) and set(meta) == {"pos", "col"}
# standalone helper matches, and threshold monotonicity: higher thresh -> fewer movers.
assert torch.equal(moving_mask(out), mov)
frac_lo = moving_mask(out, MOVE_THRESH).float().mean()
frac_hi = moving_mask(out, MOVE_THRESH * 4).float().mean()
assert 0.0 < frac_lo < 1.0 and frac_hi <= frac_lo, (frac_lo.item(), frac_hi.item())
print(f"OK moving-mask: moving_frac@{MOVE_THRESH}={frac_lo:.3f} @{MOVE_THRESH*4}={frac_hi:.3f}")
PY

echo
echo "### (2) --motion-loss : motion-balanced objective + copy_ratio in table/JSON"
"$PY" train_compare.py $COMMON --steps 15 --rollout-loss 2 --motion-loss --collisions --tag _motion \
    | grep -E "copy_ratio|eval_mse|RESULT " | tail -3
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_v4/result_wave_motion.json").read_text())
assert r["motion_loss"] is True and "copy_ratio" in r, r.get("motion_loss")
assert isinstance(r["copy_ratio"], (int, float))
print(f"OK motion-loss JSON: motion_loss={r['motion_loss']} copy_ratio={r['copy_ratio']} "
      f"move_thresh={r['move_thresh']}")
PY

echo
echo "### (3) --rollout-ramp : K holds 3 through first half, ramps 3->8 over the second"
# 50 steps -> log rows at step 1 (K=3), 25 (K=3, still first half), 50 (K=8, end of ramp).
"$PY" train_compare.py $COMMON --steps 50 --rollout-ramp --collisions --tag _ramp \
    | grep -E '"step":' | tail -3
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_v4/result_wave_ramp.json").read_text())
assert r["rollout_ramp"] is True and r["rollout_loss_K"] == 8, (r["rollout_ramp"], r["rollout_loss_K"])
Ks = [row["K"] for row in r["log_tail"]]
assert min(Ks) >= 3 and max(Ks) == 8, Ks   # ramp reached the top by the final rows
print(f"OK rollout-ramp: rollout_loss_K(max)={r['rollout_loss_K']}  logged K tail={Ks}")
PY

echo
echo "### (4) both flags together (motion-loss + rollout-ramp)"
"$PY" train_compare.py $COMMON --steps 20 --rollout-ramp --motion-loss --collisions --tag _both \
    | grep -E "copy_ratio|RESULT " | tail -2

echo
echo "### (5) regression: no new flags -> plain MSE path unchanged (copy_ratio still reported)"
"$PY" train_compare.py $COMMON --steps 10 --rollout-loss 2 --collisions --tag _base \
    | grep -E "copy_ratio|eval_mse" | tail -2
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_v4/result_wave_base.json").read_text())
assert r["motion_loss"] is False and r["rollout_ramp"] is False, r
assert "copy_ratio" in r
print(f"OK baseline: motion_loss={r['motion_loss']} rollout_ramp={r['rollout_ramp']} "
      f"copy_ratio={r['copy_ratio']}")
PY

echo
echo "### smoke v4 done -> $OUT/"
