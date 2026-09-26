#!/usr/bin/env bash
# DEEP_DIVE_3 Rank-1 HYBRID gated-fusion smoke (CPU-only plumbing check).
#
# Proves the four things the hybrid build task requires, and nothing about a
# trained predictor's accuracy:
#
#   (1) gated-fusion wrapper + streaming parity + persistent-state-byte accounting
#       -- reuses test_fusion_r14.py, which checks the exact spec math
#          h = g*h_local + (1-g)*h_global,  g = sigmoid(Linear([h_local; h_global]))
#          and that FusedMix.step() == FusedMix.forward() at the last frame, for
#          BOTH globals (local_wave, local_ssm).
#   (2) the DEFAULT (fuse=none) path is BYTE-IDENTICAL whether the arm is selected
#       by the canonical --fuse or the DEEP_DIVE_3 --fusion alias: two same-seed
#       runs must agree on every deterministic metric (the additive flag is a pure
#       pass-through and cannot perturb the default arm's numerics).
#   (2b) the existing wave/attn/ssm mixer numerics are unchanged (sanity_check.py).
#   (3) BOTH hybrid arms (local_wave, local_ssm) train end-to-end on CPU at BOTH
#       g16 AND g32, with equal-params accounting (--target-params matches each arm
#       to the same budget within match_ffn_mult's 10% tolerance).
#
# Usage:  bash run_smoke_hybrid.sh          # writes ./smoke_hybrid/
#         OUT=/tmp/x bash run_smoke_hybrid.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUT="${OUT:-smoke_hybrid}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$OUT" ]] && [[ ! -d "$OUT" || -n "$(ls -A "$OUT")" ]]; then
  echo "Use a new or empty OUT directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo '### (1) gated-fusion wrapper parity + streaming + state-byte accounting'
"$PY" test_fusion_r14.py

echo '### (2b) mixer regression (default wave/attn/ssm numerics unchanged)'
"$PY" sanity_check.py

# Shared tiny end-to-end config: balls + collisions, K=1 (isolate the one-step
# map -- the freeze, not the ramp/drift), mirroring DEEP_DIVE_3 BASE knobs shrunk
# for CPU. target-params 20000 is reachable inside match_ffn_mult's [0.05,64] range
# at this dim, so both arms match the SAME budget.
COMMON=(--frames 5 --batch 2 --steps 3 --dim 16 --layers 1 --heads 2
        --causal --residual --linear-pad --motion-loss --rollout-loss 1
        --collisions --eval-batches 1 --eval-rollout 12 --eval-seeds 2
        --eval-chunk 1 --out "$OUT")

echo '### (2) default-path byte-identical: --fuse none vs --fusion none (same seed)'
"$PY" train_compare.py "${COMMON[@]}" --grid 16 --kind wave --kernel-version dispersion \
      --fuse   none --seed 0 --tag _def_fuse
"$PY" train_compare.py "${COMMON[@]}" --grid 16 --kind wave --kernel-version dispersion \
      --fusion none --seed 0 --tag _def_fusion

echo '### (3) hybrid arms train end-to-end at g16 AND g32, matched budget'
for G in 16 32; do
  "$PY" train_compare.py "${COMMON[@]}" --grid "$G" --kind wave \
        --kernel-version dispersion --fusion local_wave \
        --target-params 20000 --tag "_lw_g${G}"
  "$PY" train_compare.py "${COMMON[@]}" --grid "$G" --kind ssm \
        --fusion local_ssm \
        --target-params 20000 --tag "_ls_g${G}"
done

echo '### (4) assertions'
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])

def load(kind, tag):
    return json.loads((out / f"result_{kind}{tag}.json").read_text())

# (2) --fusion is a byte-identical alias of --fuse on the default path: same seed,
#     same config -> every deterministic field must match exactly. (Timing fields
#     train_sec/steps_s/ms_per_frame are excluded because they are wall-clock.)
a = load("wave", "_def_fuse")
b = load("wave", "_def_fusion")
det = ["fuse", "params", "ffn_mult", "eval_mse", "copy_ratio",
       "eval_mse_over_copylast", "baselines", "divergence_horizon",
       "rollout_mse_curve", "identity_survival"]
for k in det:
    assert a.get(k) == b.get(k), f"alias diverged on {k}: {a.get(k)!r} != {b.get(k)!r}"
assert a["fuse"] == "none", "default arm must record fuse=none"
print(f"OK  default path byte-identical (--fuse none == --fusion none): "
      f"eval_mse={a['eval_mse']} copy_ratio={a['copy_ratio']}")

# (3) both hybrid arms train + eval at both grids, and equal-params accounting
#     holds: each arm within 10% of the shared 20000 budget, and the two arms are
#     within 10% of each other at each grid (matched-budget comparison is valid).
TARGET = 20000
for G in (16, 32):
    lw = load("wave", f"_lw_g{G}")
    ls = load("ssm",  f"_ls_g{G}")
    assert lw["fuse"] == "local_wave" and ls["fuse"] == "local_ssm", G
    for name, r in (("local_wave", lw), ("local_ssm", ls)):
        off = abs(r["params"] - TARGET) / TARGET
        assert off <= 0.10, f"g{G} {name} params {r['params']} off {off:.1%} from {TARGET}"
        # end-to-end sanity: finite metrics were actually produced.
        assert isinstance(r["eval_mse"], (int, float)) and r["eval_mse"] == r["eval_mse"], (G, name)
        assert r["divergence_horizon"] is not None, (G, name)
    pair_off = abs(lw["params"] - ls["params"]) / TARGET
    assert pair_off <= 0.10, f"g{G} arms not matched: {lw['params']} vs {ls['params']}"
    # the hybrid killer metric: persistent state > 0 (global recurrence) AND a
    # finite local window is kept -- both true for local+global fusion.
    assert lw["persistent_state_bytes"] > 0 and lw["window_bytes"] > 0, f"g{G} local_wave bytes"
    assert ls["persistent_state_bytes"] > 0 and ls["window_bytes"] > 0, f"g{G} local_ssm bytes"
    print(f"OK  g{G}: local_wave params={lw['params']} state={lw['persistent_state_bytes']}B "
          f"| local_ssm params={ls['params']} state={ls['persistent_state_bytes']}B "
          f"(matched to {TARGET} within {pair_off:.1%})")

print("SMOKE-HYBRID ALL-OK")
PY
echo "### smoke hybrid PASS -> $OUT/"
