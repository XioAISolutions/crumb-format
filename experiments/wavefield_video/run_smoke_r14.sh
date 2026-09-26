#!/usr/bin/env bash
# R14 CPU smoke: occlusion-scenario correctness, gated-fusion wrapper parity +
# state-byte accounting, the mixer regression sanity, the runner dry-run, and a
# micro end-to-end train+occlusion-eval for all five arms. Proves the plumbing of
# the decisive experiment, not the accuracy of a trained predictor.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUT="${OUT:-smoke_r14}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$OUT" ]] && [[ ! -d "$OUT" || -n "$(ls -A "$OUT")" ]]; then
  echo "Use a new or empty OUT directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo '### (1) occlusion scenario: target hidden exactly on the window, deterministic'
"$PY" test_occlusion_r14.py

echo '### (2) gated fusion wrapper parity + persistent state-byte accounting'
"$PY" test_fusion_r14.py

echo '### (2b) mixer regression (existing wave/attn/ssm paths unchanged)'
"$PY" sanity_check.py

echo '### (3) run_occlusion.sh dry-run: five arms x seeds, equal budget (no training)'
"$PY" test_occlusion_runner_r14.py

echo '### (4) micro end-to-end: train + 1024-style occlusion eval + state bytes, all arms'
BASE=(--data-source occlusion --grid 16 --frames 4 --batch 2 --steps 4
      --dim 16 --layers 1 --heads 2 --occ-start 6 --occ-end 12 --motion-loss
      --causal --residual --linear-pad --eval-batches 1
      --eval-rollout 16 --eval-seeds 2 --eval-chunk 1 --out "$OUT")
"$PY" train_compare.py "${BASE[@]}" --kind attn --tag _A > "$OUT/train_A.log"
"$PY" train_compare.py "${BASE[@]}" --kind ssm  --tag _B > "$OUT/train_B.log"
"$PY" train_compare.py "${BASE[@]}" --kind wave --kernel-version dispersion --tag _C > "$OUT/train_C.log"
"$PY" train_compare.py "${BASE[@]}" --kind ssm  --fuse local_ssm --tag _D > "$OUT/train_D.log"
"$PY" train_compare.py "${BASE[@]}" --kind wave --kernel-version dispersion \
      --fuse local_wave --tag _E > "$OUT/train_E.log"

"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
arms = {"attn_A": ("none", False, True, False),   # (fuse, streaming, keeps_window, has_persistent)
        "ssm_B": ("none", True, False, True),
        "wave_C": ("none", True, False, True),
        "ssm_D": ("local_ssm", True, True, True),
        "wave_E": ("local_wave", True, True, True)}
required = ("mode", "streaming", "divergence_horizon", "exit_direction_accuracy",
           "position_error_at_emergence", "velocity_error_at_emergence",
           "target_identity_survival", "motion_preservation", "ms_per_frame",
           "persistent_state_bytes", "window_bytes", "state_bytes_total",
           "emergence_frame", "occluder", "target_centroid_curve")
for name, (fuse, streaming, keeps_window, has_persistent) in arms.items():
    r = json.loads((out / f"result_{name}.json").read_text())
    for key in required:
        assert key in r, (name, "missing", key)
    assert r["mode"] == "occlusion_rollout"
    assert r["fuse"] == fuse
    assert r["streaming"] is streaming, (name, "streaming")
    assert r["emergence_frame"] == 12 - 4, name           # occ_end - frames
    assert len(r["target_centroid_curve"]) == 16
    assert (r["persistent_state_bytes"] > 0) == has_persistent, name
    assert (r["window_bytes"] > 0) == keeps_window, name
    assert r["state_bytes_total"] == r["persistent_state_bytes"] + r["window_bytes"]
    acc = r["exit_direction_accuracy"]
    assert acc is None or 0.0 <= acc <= 1.0, name
    assert r["divergence_horizon"] is not None and 0 <= r["divergence_horizon"] <= 16
# The whole point: attention keeps a big finite window and NO persistent state;
# the recurrent arms carry a fixed persistent state independent of horizon.
attn = json.loads((out / "result_attn_A.json").read_text())
ssm = json.loads((out / "result_ssm_B.json").read_text())
assert attn["persistent_state_bytes"] == 0 and attn["window_bytes"] > 0
assert ssm["persistent_state_bytes"] > 0 and ssm["window_bytes"] == 0
# Hybrid persistent state equals its global-only arm's (local buffer is not counted).
d = json.loads((out / "result_ssm_D.json").read_text())
assert d["persistent_state_bytes"] == ssm["persistent_state_bytes"], "hybrid==global persistent bytes"
print("OK all five arms trained + occlusion-evaluated; state-byte asymmetry verified")
PY
echo "### smoke r14 PASS -> $OUT/"
