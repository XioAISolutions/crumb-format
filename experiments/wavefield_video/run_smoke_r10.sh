#!/usr/bin/env bash
# CPU smoke for R10 -- SCALE-AWARE OBJECTIVE + EVAL HARDENING. Tiny + fast:
# it PROVES the four new knobs are wired end-to-end (shapes / NaN / JSON fields /
# metric-equivalence / default-path regression), NOT that they improve accuracy.
#
# What this proves, in order:
#   (0) motion_balanced_loss unit tests: default path byte-identical; --resid-balanced
#       restricts the residual term to moving pixels; --motion-weighted gives a soft
#       per-pixel weight in [1, w_max] with static pixels == 1.
#   (1) --resid-balanced : tiny train runs, JSON records resid_balanced=True, no NaN.
#   (2) --motion-weighted : tiny train runs, JSON records motion_weighted + w_max, no NaN.
#   (3) --radius / --speed : data.py geometry passthrough (bigger radius => bigger
#       moving footprint; CONTACT scales), and a tiny train run records radius/speed.
#   (4) --eval-chunk : the chunked rollout is METRIC-EQUIVALENT to the all-at-once
#       rollout (chunk=1 vs 2 vs S give the same curve / div_horizon), so the OOM fix
#       does not change the numbers.
#   (5) default-path regression : no new flags -> deterministic eval_mse across two
#       fixed-seed runs, and the new JSON fields carry their documented defaults.
#
# Operator runs this (python is session-gated). One approval runs all of it:
#   cd experiments/wavefield_video && bash run_smoke_r10.sh
#   # or: PY=/path/to/venv/bin/python bash run_smoke_r10.sh
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_r10}"
mkdir -p "$OUT"

# Tiny model / tiny eval so every step is seconds on CPU.
M="--kind wave --kernel-version dispersion --linear-pad --dim 32 --layers 2 --heads 4 \
   --grid 8 --frames 6 --batch 8 --steps 6 --rollout-loss 2 --collisions"
EV="--eval-batches 2 --eval-rollout 6 --eval-seeds 4"

echo "### (0) motion_balanced_loss unit tests (default equivalence + both knobs)"
"$PY" - <<'PY'
import torch
from train_compare import motion_balanced_loss
torch.manual_seed(0)
B, C, H, W = 2, 3, 8, 8
pred = torch.rand(B, C, H, W)
last = torch.rand(B, C, H, W)
target = last.clone()
# make a few pixels "move" by a large amount so the mask is non-trivial
target[:, :, 2, 3] += 0.9
target[:, :, 5, 6] += 0.5
moving = (target - last).abs().amax(1) > 0.05          # [B,H,W]

# --- default path: byte-identical to the hand-written formula --------------
E = (pred - target).pow(2).mean(1)
ref = 0.5 * E[moving].mean() + 0.5 * E[~moving].mean() \
      + 0.25 * torch.nn.functional.mse_loss(pred - last, target - last)
got = motion_balanced_loss(pred, target, last, moving)
assert torch.allclose(got, ref, atol=1e-7), (got.item(), ref.item())

# --- --resid-balanced: residual term restricted to moving pixels -----------
ref_rb = 0.5 * E[moving].mean() + 0.5 * E[~moving].mean() + 0.25 * E[moving].mean()
got_rb = motion_balanced_loss(pred, target, last, moving, resid_balanced=True)
assert torch.allclose(got_rb, ref_rb, atol=1e-7), (got_rb.item(), ref_rb.item())
assert not torch.allclose(got_rb, got), "resid-balanced must change the loss"

# --- --motion-weighted: soft weights in [1, w_max], static pixels == 1 ------
delta = (target - last).abs().mean(1)
w = (delta / (delta.mean() + 1e-6)).clamp(1.0, 5.0)
assert float(w.min()) >= 1.0 - 1e-6 and float(w.max()) <= 5.0 + 1e-6
assert torch.allclose(w[~moving], torch.ones_like(w[~moving])), "static pixels must weigh 1"
ref_mw = 0.5 * (w * E).mean() + 0.5 * E[~moving].mean() \
         + 0.25 * torch.nn.functional.mse_loss(pred - last, target - last)
got_mw = motion_balanced_loss(pred, target, last, moving, motion_weighted=True, w_max=5.0)
assert torch.allclose(got_mw, ref_mw, atol=1e-7), (got_mw.item(), ref_mw.item())
print("OK loss: default byte-identical; resid_balanced + motion_weighted match spec")
PY

echo
echo "### (1) --resid-balanced : tiny train run, JSON field set, no NaN"
"$PY" train_compare.py $M --motion-loss --resid-balanced $EV --out "$OUT" --tag _rb \
    | grep -E "eval_mse|copy_ratio|RESULT " | tail -2
"$PY" - <<'PY'
import json, pathlib, math
r = json.loads(pathlib.Path("smoke_r10/result_wave_rb.json").read_text())
assert r["motion_loss"] is True and r["resid_balanced"] is True, r
assert r["motion_weighted"] is False
assert not math.isnan(r["eval_mse"]), r["eval_mse"]
print(f"OK resid-balanced: resid_balanced={r['resid_balanced']} eval_mse={r['eval_mse']}")
PY

echo
echo "### (2) --motion-weighted : tiny train run, JSON field + w_max recorded, no NaN"
"$PY" train_compare.py $M --motion-loss --motion-weighted --motion-w-max 8 $EV \
    --out "$OUT" --tag _mw | grep -E "eval_mse|copy_ratio|RESULT " | tail -2
"$PY" - <<'PY'
import json, pathlib, math
r = json.loads(pathlib.Path("smoke_r10/result_wave_mw.json").read_text())
assert r["motion_loss"] is True and r["motion_weighted"] is True, r
assert r["motion_w_max"] == 8.0, r["motion_w_max"]
assert r["resid_balanced"] is False
assert not math.isnan(r["eval_mse"]), r["eval_mse"]
print(f"OK motion-weighted: motion_weighted={r['motion_weighted']} w_max={r['motion_w_max']} "
      f"eval_mse={r['eval_mse']}")
PY

echo
echo "### (3a) data.py geometry passthrough : bigger radius => bigger moving footprint"
"$PY" - <<'PY'
from data import make_clip_batch, RADIUS, SPEED, moving_mask
# same seed, same everything except radius: the moving-pixel fraction must grow.
small = make_clip_batch(16, 8, 16, 16, seed=7, radius=RADIUS, speed=SPEED, collisions=True)
big   = make_clip_batch(16, 8, 16, 16, seed=7, radius=2.0 * RADIUS, speed=SPEED, collisions=True)
fs = moving_mask(small).float().mean().item()
fb = moving_mask(big).float().mean().item()
assert fb > fs, (fs, fb)
# default radius/speed must reproduce the module constants exactly (regression).
import torch
a = make_clip_batch(4, 5, 16, 16, seed=11, collisions=True)
b = make_clip_batch(4, 5, 16, 16, seed=11, radius=RADIUS, speed=SPEED, collisions=True)
assert torch.equal(a, b), "explicit default radius/speed must equal the constants"
print(f"OK geometry: moving_frac radius={RADIUS}->{fs:.3f}  radius={2*RADIUS}->{fb:.3f} (grows)")
PY

echo "### (3b) --radius/--speed train passthrough : JSON records the geometry"
"$PY" train_compare.py $M --radius 2.4 --speed 1.7 $EV --out "$OUT" --tag _geo \
    | grep -E "eval_mse|RESULT " | tail -1
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_r10/result_wave_geo.json").read_text())
assert r["radius"] == 2.4 and r["speed"] == 1.7, (r["radius"], r["speed"])
assert r["div_thresh_px"] == round(2.4, 3), r["div_thresh_px"]   # div threshold scaled
print(f"OK geometry train: radius={r['radius']} speed={r['speed']} div_thresh_px={r['div_thresh_px']}")
PY

echo
echo "### (4) --eval-chunk : chunked rollout is metric-equivalent to all-at-once"
"$PY" - <<'PY'
import argparse, torch
import train_compare as tc
from wfvideo import VideoPredictor
torch.manual_seed(0)
dev = "cpu"
# one fixed untrained model; chunking must not change any rollout number
m = VideoPredictor(32, 2, 4, 6, 8, 8, "wave", causal=False, residual=True,
                   ffn_mult=2.0, kernel_version="dispersion", linear_pad=True).to(dev).eval()

def run(chunk):
    a = argparse.Namespace(eval_seeds=6, eval_rollout=8, eval_chunk=chunk,
                           frames=6, grid=8, kicks=False, collisions=True,
                           radius=tc.RADIUS, speed=tc.SPEED)
    return tc.rollout_eval(m, a, dev)

full = run(6)      # all seeds at once (old behavior)
c1   = run(1)      # one seed at a time (worst-case OOM avoidance)
c2   = run(2)
for other, name in ((c1, "chunk=1"), (c2, "chunk=2")):
    a = torch.tensor(full["rollout_mse_curve"]); b = torch.tensor(other["rollout_mse_curve"])
    assert torch.allclose(a, b, atol=1e-4), (name, a, b)
    assert full["divergence_horizon"] == other["divergence_horizon"], name
    assert abs(full["identity_survival"] - other["identity_survival"]) < 1e-4, name
    assert abs(full["mean_centroid_err"] - other["mean_centroid_err"]) < 1e-3, name
assert full["eval_chunk"] == 6 and c1["eval_chunk"] == 1
print(f"OK eval-chunk: chunk=1/2/6 agree  div_h={full['divergence_horizon']} "
      f"id_surv={full['identity_survival']} curve[0]={full['rollout_mse_curve'][0]}")
PY

echo
echo "### (5) default-path regression : deterministic + documented defaults"
for TAG in _def _def2; do
  "$PY" train_compare.py $M $EV --out "$OUT" --tag "$TAG" \
      | grep -E "eval_mse|RESULT " | tail -1
done
"$PY" - <<'PY'
import json, pathlib
from data import RADIUS, SPEED
r  = json.loads(pathlib.Path("smoke_r10/result_wave_def.json").read_text())
r2 = json.loads(pathlib.Path("smoke_r10/result_wave_def2.json").read_text())
# new knobs default to no-op
assert r["resid_balanced"] is False and r["motion_weighted"] is False, r
assert r["motion_w_max"] is None
assert r["radius"] == RADIUS and r["speed"] == SPEED, (r["radius"], r["speed"])
assert r["eval_chunk"] == 4, r["eval_chunk"]                 # new default (OOM fix)
assert r["motion_loss"] is False
# fixed seeds => identical eval_mse AND identical rollout curve across two runs
assert r["eval_mse"] == r2["eval_mse"], (r["eval_mse"], r2["eval_mse"])
assert r["rollout_mse_curve"] == r2["rollout_mse_curve"]
print(f"OK default regression: resid_balanced=False motion_weighted=False "
      f"radius={r['radius']} eval_chunk={r['eval_chunk']} eval_mse={r['eval_mse']} (deterministic)")
PY

echo
echo "### smoke r10 done -> $OUT/"
