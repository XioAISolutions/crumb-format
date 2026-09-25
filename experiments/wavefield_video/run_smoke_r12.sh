#!/usr/bin/env bash
# R12 CPU checks: frozen R11 bytes, ball counts/collisions, semantic correctness,
# tiny real training + pixel rendering, and the R11 latent decode regression.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUT="${OUT:-smoke_r12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$OUT" ]] && [[ ! -d "$OUT" || -n "$(ls -A "$OUT")" ]]; then
  echo "Use a new or empty OUT directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo '### (1) frozen R11 data bytes + arbitrary ball counts/collisions'
"$PY" test_data_r12.py
echo '### (2) known-position semantic frames + eval/geometry integration'
"$PY" test_semantic_r12.py
"$PY" test_eval_r12.py
echo '### (2b) eight-job ladder commands and status handling (no training)'
"$PY" test_difficulty_r12.py

echo '### (3) omitted --n-balls and explicit default: identical trained tensors'
BASE=(--kind wave --kernel-version dispersion --dim 16 --layers 1 --heads 2
      --grid 16 --frames 4 --batch 2 --steps 6 --collisions --motion-loss
      --eval-batches 1 --eval-rollout 4 --eval-seeds 3 --eval-chunk 2 --out "$OUT")
"$PY" train_compare.py "${BASE[@]}" --tag _default > "$OUT/train_default.log"
"$PY" train_compare.py "${BASE[@]}" --n-balls 3 --tag _explicit > "$OUT/train_explicit.log"
"$PY" train_compare.py "${BASE[@]}" --n-balls 12 --radius 0.8 --speed 2.30 \
  --tag _n12 > "$OUT/train_n12.log"
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys, torch
out = pathlib.Path(sys.argv[1])
a, b = [torch.load(out / f"model_wave_{tag}.pt", weights_only=True)["state"]
        for tag in ("default", "explicit")]
assert a.keys() == b.keys()
assert all(torch.equal(a[key], b[key]) for key in a), "default weights changed"
ra, rb, rn = [json.loads((out / f"result_wave_{tag}.json").read_text())
              for tag in ("default", "explicit", "n12")]
for key in ("eval_mse", "baselines", "copy_ratio", "rollout_mse_curve", "semantic"):
    assert ra[key] == rb[key], key
assert ra["n_balls"] == 3
assert rn["n_balls"] == 12 and rn["semantic"]["expected_n"] == 12
assert rn["radius"] == 0.8 and rn["speed"] == 2.3
assert len(rn["semantic"]["frames"]) == 4
print("OK default training tensors/metrics identical; n=12 reached training + eval")
PY

echo '### (4) n=12 pixel render restores checkpoint geometry + semantic JSON'
"$PY" render_rollout.py --ckpt "$OUT/model_wave_n12.pt" --frames 4 \
  --side-by-side --scale 1 --device cpu --out "$OUT/n12_render" > "$OUT/n12_render.log"
tail -1 "$OUT/n12_render.log"

echo '### (5) R11 latent decode + checkpoint refusal + pixel path regression'
PY="$PY" OUT="$OUT/r11" bash run_smoke_r11.sh > "$OUT/r11.log" 2>&1
tail -2 "$OUT/r11.log"
"$PY" eval_only.py "$OUT/model_wave_n12.pt" "$OUT/result_wave_n12.json" \
  "$OUT/eval_pixel.json" --eval-seeds 2 --eval-rollout 3 > "$OUT/eval_pixel.log"
"$PY" eval_only.py "$OUT/r11/model_wave_latsmoke.pt" "$OUT/r11/result_wave_latsmoke.json" \
  "$OUT/eval_latent.json" --eval-seeds 2 --eval-rollout 3 > "$OUT/eval_latent.log"
"$PY" - "$OUT" <<'PY'
import json, math, pathlib, sys
out = pathlib.Path(sys.argv[1])
for folder, n, horizon in (("n12_render", 12, 4), ("r11/latent_render", 3, 6),
                           ("r11/pixel_render", 3, 6)):
    m = json.loads((out / folder / "metrics.json").read_text())
    assert m["status"] == "complete" and m["config"]["n_balls"] == n
    s = m["metrics"]["semantic"]
    assert s["expected_n"] == n and len(s["frames"]) == horizon
    for frame in s["frames"]:
        assert frame["expected_n"] == n
        assert 0 <= frame["matched_count"] <= min(n, frame["detected_n"])
        error = frame["mean_matched_position_error"]
        assert (error is None) == (frame["matched_count"] == 0)
        assert error is None or (math.isfinite(error) and error >= 0)
        assert len(frame["samples"]) == 1
    assert "semantic_metrics.py" in m["provenance"]["source_sha256"]
    if n == 12:
        assert m["config"]["radius"] == 0.8 and m["config"]["speed"] == 2.3
for filename, n in (("eval_pixel.json", 12), ("eval_latent.json", 3)):
    s = json.loads((out / filename).read_text())["semantic"]
    assert s["expected_n"] == n and len(s["frames"]) == 3
    assert all(len(frame["samples"]) == 2 for frame in s["frames"])
print("OK pixel/latent render + saved-checkpoint eval semantic metrics")
PY
echo "### smoke r12 PASS -> $OUT/"
