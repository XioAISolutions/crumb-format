#!/usr/bin/env bash
# CPU smoke for R7 -- LATENT SPACE (the gate to real pixels). Tiny + fast:
# it PROVES the pipeline is wired end-to-end (shapes/NaN/fields/regression),
# NOT that latent prediction is accurate.
#
# Three things this proves:
#   (1) latent_ae.py --train  : the 3->32->64 conv AE trains and its held-out
#       recon round-trip beats a gray-constant baseline; encode/decode shapes
#       are [B,64,H/4,W/4] <-> [B,3,H,W] in [0,1].
#   (2) train_compare.py --latent : freeze the AE, train the predictor on encoded
#       tokens, predict next latent, decode for pixel-space eval -- the result
#       JSON still carries EVERY pixel-space field plus the new latent_* fields.
#   (3) balls regression : the non-latent pixel path is byte-for-byte unchanged
#       (latent=False, original fields present, eval_mse deterministic across
#       two identical fixed-seed runs).
#
# Operator runs this (python is session-gated). One approval runs all of it:
#   cd experiments/wavefield_video && bash run_smoke_r7.sh
#   # or: PY=/path/to/venv/bin/python bash run_smoke_r7.sh
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_r7}"
AE="${AE:-$OUT/ae.pt}"
mkdir -p "$OUT"

EV="--eval-batches 2 --eval-rollout 8 --eval-seeds 4"

echo "### (1) latent_ae.py --train : 3->32->64 AE trains, reports held-out recon MSE"
"$PY" latent_ae.py --train --data-source balls --grid 32 --base 32 --steps 150 --batch 16 \
    --out "$AE" | grep -E "AE base|recon_mse|AE-DONE|AE-RESULT" | tail -6
"$PY" - <<'PY'
import torch, torch.nn.functional as F
from data import make_clip_batch
from latent_ae import load_ae
ae = load_ae("smoke_r7/ae.pt")
x = make_clip_batch(8, 8, 32, 32, seed=123).reshape(-1, 3, 32, 32)
z = ae.encode(x)
assert tuple(z.shape[1:]) == (64, 8, 8), z.shape          # 3->...->64 @ /4 grid
xr = ae.decode(z)
assert xr.shape == x.shape, (xr.shape, x.shape)
assert 0.0 <= float(xr.min()) and float(xr.max()) <= 1.0   # sigmoid range
recon = F.mse_loss(xr, x).item()
gray = F.mse_loss(torch.full_like(x, 0.5), x).item()       # trivial baseline
assert recon < gray, (recon, gray)                          # AE actually learned
print(f"OK AE roundtrip: z={tuple(z.shape)} recon_mse={recon:.5f} < gray={gray:.5f}")
PY

echo
echo "### (2) train_compare.py --latent : predict next latent, decode for pixel eval"
"$PY" train_compare.py --kind wave --kernel-version dispersion --linear-pad \
    --dim 48 --layers 2 --heads 4 --grid 32 --frames 6 --batch 8 \
    --latent --ae-ckpt "$AE" --steps 10 --rollout-loss 2 $EV --out "$OUT" --tag _latent \
    | grep -E "LATENT|eval_mse|RESULT " | tail -3
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_r7/result_wave_latent.json").read_text())
assert r["latent"] is True, r["latent"]
assert r["latent_ch"] == 64 and r["latent_grid"] == 8, (r["latent_ch"], r["latent_grid"])
assert r["params"] > 0 and r["ae_params"] > 0
# SAME pixel-space JSON fields the non-latent path produces:
for k in ("eval_mse", "baselines", "copy_ratio", "eval_mse_over_copylast",
          "r_persist_over_model", "divergence_horizon", "identity_survival",
          "rollout_mse_curve"):
    assert k in r, f"missing pixel-space field {k!r}"
assert isinstance(r["eval_mse"], (int, float)) and r["eval_mse"] == r["eval_mse"]  # not NaN
print(f"OK latent JSON: latent_ch={r['latent_ch']} latent_grid={r['latent_grid']} "
      f"params={r['params']} eval_mse={r['eval_mse']}")
PY

echo
echo "### (3) balls regression : non-latent pixel path unchanged + deterministic"
for TAG in _balls _balls2; do
  "$PY" train_compare.py --kind wave --kernel-version dispersion --linear-pad \
      --dim 48 --layers 2 --heads 4 --grid 8 --frames 8 --batch 8 \
      --steps 10 --rollout-loss 2 --collisions $EV --out "$OUT" --tag "$TAG" \
      | grep -E "eval_mse|RESULT " | tail -1
done
"$PY" - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path("smoke_r7/result_wave_balls.json").read_text())
r2 = json.loads(pathlib.Path("smoke_r7/result_wave_balls2.json").read_text())
assert r["latent"] is False and r["latent_ch"] is None, r["latent"]
assert r["motion_loss"] is False and r["rollout_ramp"] is False
for k in ("eval_mse", "baselines", "copy_ratio", "divergence_horizon"):
    assert k in r, k
# fixed seeds => identical eval_mse across two runs => pixel RNG path untouched
assert r["eval_mse"] == r2["eval_mse"], (r["eval_mse"], r2["eval_mse"])
print(f"OK balls regression: latent=False deterministic eval_mse={r['eval_mse']} "
      f"(matched across 2 runs)")
PY

echo
echo "### smoke r7 done -> $OUT/"
