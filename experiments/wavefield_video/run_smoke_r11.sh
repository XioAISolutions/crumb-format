#!/usr/bin/env bash
# CPU smoke for R11 -- LATENT DECODE-RENDER + PIXEL-SPACE EVAL.
#
# This PROVES the render_rollout.py --latent plumbing is wired end-to-end
# (load LatentCore + frozen AE, roll autoregressively at the latent grid, decode
# EVERY predicted latent, score in pixel space, write PNGs/MP4 + metrics.json).
# It does NOT prove latent prediction is accurate -- weights are RANDOM here, so
# no trained checkpoint is required.
#
# Three things this proves:
#   (1) --latent decode path : synth a tiny frozen ConvAE + a tiny LatentCore
#       predictor (random weights), render 6 frames; metrics.json carries PIXEL-
#       space mse, copy_last_mse, centroid err + divergence; PNGs/MP4 exist.
#   (2) resilience : rendering that same latent ckpt WITHOUT --latent is refused
#       (it has no pixel VideoPredictor state), and a pixel ckpt WITH --latent is
#       refused (no vp.* keys). The two paths never silently cross.
#   (3) default-path regression : a tiny PIXEL VideoPredictor renders unchanged
#       (no --latent), producing the original metrics.json fields.
#
# Operator runs this (python is session-gated). One approval runs all of it:
#   cd experiments/wavefield_video && bash run_smoke_r11.sh
#   # or: PY=/path/to/venv/bin/python bash run_smoke_r11.sh
set -e
PY="${PY:-python3}"
cd "$(dirname "$0")"
OUT="${OUT:-smoke_r11}"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "### (0) synthesize tiny RANDOM checkpoints (no training needed)"
"$PY" - "$OUT" <<'PY'
import json, sys, pathlib, torch
sys.path.insert(0, ".")
from wfvideo import VideoPredictor
from latent_ae import ConvAE, DOWNSAMPLE
from train_compare import LatentCore

out = pathlib.Path(sys.argv[1])
torch.manual_seed(0)
GRID, BASE, DIM, LAYERS, HEADS, FRAMES = 16, 4, 16, 1, 2, 4
LAT = GRID // DOWNSAMPLE                      # latent grid 4x4
CZ = 2 * BASE                                 # ConvAE latent_ch = 2*base = 8

# --- frozen conv AE (random) -> {state, config}, mirrors latent_ae.py --train ---
ae = ConvAE(base=BASE)
ae_cfg = {"base": BASE, "latent_ch": CZ, "downsample": DOWNSAMPLE, "grid": GRID,
          "latent_grid": LAT, "data_source": "balls", "field": "wave"}
torch.save({"state": ae.state_dict(), "config": ae_cfg}, out / "ae_smoke.pt")
(out / "ae_smoke.json").write_text(json.dumps({"eval_recon_mse": None, **ae_cfg}))

# --- latent predictor (LatentCore over a VideoPredictor at the latent grid) ---
vp = VideoPredictor(DIM, LAYERS, HEADS, FRAMES, LAT, LAT, "wave",
                    kernel_version="dispersion", linear_pad=True, residual=True)
core = LatentCore(vp, DIM, CZ)
torch.save({"state": core.state_dict()}, out / "model_wave_latsmoke.pt")
# result JSON in the exact schema train_compare.py writes for a --latent run:
res = {"kind": "wave", "params": sum(p.numel() for p in core.parameters()),
       "ffn_mult": 4.0, "dim": DIM, "layers": LAYERS, "heads": HEADS,
       "grid": GRID, "frames": FRAMES, "kernel_version": "dispersion",
       "causal": False, "residual": True, "linear_pad": True,
       "gate": False, "local_fuse": False, "kicks": False, "collisions": True,
       "latent": True, "latent_ch": CZ, "latent_grid": LAT,
       "ae_ckpt": str(out / "ae_smoke.pt"), "ae_params": sum(p.numel() for p in ae.parameters())}
(out / "result_wave_latsmoke.json").write_text(json.dumps(res, indent=1))

# --- a tiny PIXEL VideoPredictor for the default-path regression ---
pvp = VideoPredictor(DIM, LAYERS, HEADS, FRAMES, GRID, GRID, "wave",
                     kernel_version="dispersion", linear_pad=True, residual=True)
torch.save({"state": pvp.state_dict()}, out / "model_wave_pixsmoke.pt")
pres = {"kind": "wave", "dim": DIM, "layers": LAYERS, "heads": HEADS,
        "grid": GRID, "frames": FRAMES, "ffn_mult": 4.0, "kernel_version": "dispersion",
        "causal": False, "residual": True, "linear_pad": True, "gate": False,
        "local_fuse": False, "data_source": "balls", "field": "wave",
        "kicks": False, "collisions": True}
(out / "result_wave_pixsmoke.json").write_text(json.dumps(pres, indent=1))
print("OK synth: ae_smoke.pt, model_wave_latsmoke.pt (+result), model_wave_pixsmoke.pt (+result)")
PY

echo
echo "### (1) --latent decode path : roll @latent grid, decode every latent, score in pixels"
"$PY" render_rollout.py --latent --ckpt "$OUT/model_wave_latsmoke.pt" \
    --ae-ckpt "$OUT/ae_smoke.pt" --frames 6 --side-by-side --grid 16 \
    --out "$OUT/latent_render" | tail -3
"$PY" - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
m = json.loads((out / "latent_render" / "metrics.json").read_text())
assert m["status"] == "complete", m["status"]
assert m["config"]["latent"] is True and m["config"]["latent_grid"] == 4
mt = m["metrics"]
for k in ("mse", "copy_last_mse", "mse_curve", "copy_last_mse_curve",
          "mean_centroid_err", "final_centroid_err", "divergence_horizon", "div_thresh_px"):
    assert k in mt, f"missing pixel-space metric {k!r}"
assert isinstance(mt["mse"], (int, float)) and mt["mse"] == mt["mse"]        # not NaN
assert len(mt["mse_curve"]) == 6 and len(mt["copy_last_mse_curve"]) == 6
assert mt["mean_centroid_err"] is not None                                    # balls -> centroids computed
rd = out / "latent_render"
for name in ("prediction", "ground_truth", "comparison"):
    pngs = list((rd / name).glob("*.png"))
    assert len(pngs) == 6, (name, len(pngs))
    assert (rd / f"{name}.mp4").is_file(), f"{name}.mp4 missing"
prov = m["provenance"]
assert "latent_ae.py" in prov["source_sha256"] and "train_compare.py" in prov["source_sha256"]
assert prov["ae_sha256"]
print(f"OK latent render: mse={mt['mse']:.5f} copy_last={mt['copy_last_mse']:.5f} "
      f"centroid={mt['mean_centroid_err']} div_horizon={mt['divergence_horizon']}")
PY

echo
echo "### (2) resilience : the two paths refuse each other's checkpoints"
if "$PY" render_rollout.py --ckpt "$OUT/model_wave_latsmoke.pt" \
     --frames 4 --grid 16 --out "$OUT/x_lat_as_pixel" 2>"$OUT/err_lat_as_pixel.txt"; then
  echo "FAIL: pixel path accepted a latent checkpoint"; exit 1
fi
if grep -q "fc1.weight\|unsupported checkpoint\|posemb" "$OUT/err_lat_as_pixel.txt"; then
  echo "OK pixel path refused latent ckpt"
else
  echo "FAIL: pixel path gave an unexpected error:"; cat "$OUT/err_lat_as_pixel.txt"; exit 1
fi
if "$PY" render_rollout.py --latent --ckpt "$OUT/model_wave_pixsmoke.pt" \
     --ae-ckpt "$OUT/ae_smoke.pt" --frames 4 --grid 16 --out "$OUT/x_pixel_as_lat" \
     2>"$OUT/err_pixel_as_lat.txt"; then
  echo "FAIL: latent path accepted a pixel checkpoint"; exit 1
fi
if grep -q "not a --latent checkpoint\|vp\." "$OUT/err_pixel_as_lat.txt"; then
  echo "OK latent path refused pixel ckpt"
else
  echo "FAIL: latent path gave an unexpected error:"; cat "$OUT/err_pixel_as_lat.txt"; exit 1
fi

echo
echo "### (3) default-path regression : pixel VideoPredictor renders unchanged"
"$PY" render_rollout.py --ckpt "$OUT/model_wave_pixsmoke.pt" \
    --frames 6 --side-by-side --grid 16 --out "$OUT/pixel_render" | tail -2
"$PY" - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
m = json.loads((out / "pixel_render" / "metrics.json").read_text())
assert m["status"] == "complete", m["status"]
assert "latent" not in m["config"], "pixel report must not carry a latent block"
mt = m["metrics"]
for k in ("mse", "copy_last_mse", "mse_curve", "copy_last_mse_curve", "model_fps"):
    assert k in mt, f"pixel path lost field {k!r}"
assert len(mt["mse_curve"]) == 6
rd = out / "pixel_render"
assert (rd / "prediction.mp4").is_file() and len(list((rd / "prediction").glob("*.png"))) == 6
print(f"OK pixel regression: mse={mt['mse']:.5f} copy_last={mt['copy_last_mse']:.5f} "
      f"model_fps={mt['model_fps']:.1f}")
PY

echo
echo "### smoke r11 done -> $OUT/"
