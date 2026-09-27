#!/usr/bin/env bash
# R15 CPU smoke: hypercomplex modules + wiring invariants (E1/E2/E3), the mixer
# regression and the runner contract, and a micro end-to-end train+occlusion-eval
# for the three new arms plus the baselines whose bytes they are compared against.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUT="${OUT:-smoke_r15}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$OUT" ]] && [[ ! -d "$OUT" || -n "$(ls -A "$OUT")" ]]; then
  echo "Use a new or empty OUT directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo '### (1) hypercomplex modules + wiring invariants (r15 test file)'
"$PY" test_hypercomplex_r15.py

echo '### (2) mixer regression (existing wave/attn/ssm paths unchanged)'
"$PY" sanity_check.py

echo '### (3) occlusion runner contract (arms x seeds, const-lr, slice mode)'
"$PY" test_occlusion_runner_r14.py

echo '### (4) micro end-to-end: baselines + the three new arms'
BASE=(--data-source occlusion --grid 16 --frames 4 --batch 2 --steps 4
      --dim 16 --layers 1 --heads 2 --occ-start 6 --occ-end 12 --motion-loss
      --causal --residual --linear-pad --eval-batches 1
      --eval-rollout 16 --eval-seeds 2 --eval-chunk 1 --out "$OUT")
"$PY" train_compare.py "${BASE[@]}" --kind ssm  --tag _B > "$OUT/train_B.log"
"$PY" train_compare.py "${BASE[@]}" --kind wave --kernel-version dispersion --tag _C > "$OUT/train_C.log"
"$PY" train_compare.py "${BASE[@]}" --kind qssm --tag _F > "$OUT/train_F.log"
"$PY" train_compare.py "${BASE[@]}" --kind wave --kernel-version dispersion --q-mix --tag _G > "$OUT/train_G.log"
"$PY" train_compare.py "${BASE[@]}" --kind wave --kernel-version dispersion --quat-color --tag _H > "$OUT/train_H.log"

"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
required = ("mode", "streaming", "divergence_horizon", "eval_mse", "copy_ratio",
            "persistent_state_bytes", "window_bytes", "state_bytes_total", "q_mix",
            "quat_color", "params", "log_tail")
r = {name: json.loads((out / f"result_{name}.json").read_text())
     for name in ("ssm_B", "wave_C", "qssm_F", "wave_G", "wave_H")}
for name, res in r.items():
    for key in required:
        assert key in res, (name, "missing", key)
    assert res["mode"] == "occlusion_rollout", name
    assert res["streaming"] is True, name
# E1: equal persistent bytes vs the generic SSM baseline; both carry no window.
assert r["qssm_F"]["persistent_state_bytes"] == r["ssm_B"]["persistent_state_bytes"], \
    "qssm state bytes must equal ssm (the per-byte comparison premise)"
assert r["qssm_F"]["window_bytes"] == 0 and r["ssm_B"]["window_bytes"] == 0
# E2/E3: wave variants, flags recorded in the result JSON, state bytes unchanged.
assert r["wave_G"]["q_mix"] is True and r["wave_G"]["quat_color"] is False
assert r["wave_H"]["quat_color"] is True and r["wave_H"]["q_mix"] is False
assert r["wave_G"]["persistent_state_bytes"] == r["wave_C"]["persistent_state_bytes"]
assert r["wave_H"]["persistent_state_bytes"] == r["wave_C"]["persistent_state_bytes"]
print("OK new arms trained + occlusion-evaluated; qssm state bytes == ssm; flags recorded")
PY
echo "### smoke r15 PASS -> $OUT/"
