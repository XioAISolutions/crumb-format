#!/usr/bin/env bash
# Anti-collapse CPU smoke (plumbing only -- proves the three knobs are wired and,
# crucially, that the DEFAULT path is a byte-identical no-op). Mirrors the shape
# of run_smoke_hybrid.sh. It asserts, in order:
#
#   (1) helper unit checks (no training):
#         - quantize_frames -> <=16 unique levels, values in [0,1]
#         - apply_drift_pert -> same shape, clamped [0,1], and does NOT touch the
#           GLOBAL RNG stream (uses a private torch.Generator) -- the property the
#           byte-identical guarantee rests on
#         - var-reg hinge relu(std_gt - std_pred) >= 0 and == 0 when deltas match
#   (2) DEFAULT-PATH NO-OP: a plain run vs one with --var-reg 0 --drift-pert 0
#       --hist-disc off (explicit defaults), same seed, must agree on every
#       deterministic field (CPU is fp32 => bit-deterministic).
#   (3) every knob trains end-to-end on CPU (var-reg, drift-pert, hist-disc, and
#       all three combined), reports finite metrics, records the flags in the
#       result JSON, and the var-reg arm (and ONLY it) emits a `var_reg` field in
#       its step rows -- the required per-step logging.
#
# Usage:  bash run_anticollapse_smoke.sh          # writes ./smoke_anticollapse/
#         OUT=/tmp/x bash run_anticollapse_smoke.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
OUT="${OUT:-smoke_anticollapse}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
if [[ -e "$OUT" ]] && [[ ! -d "$OUT" || -n "$(ls -A "$OUT")" ]]; then
  echo "Use a new or empty OUT directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo '### (1) helper unit checks (quantize / drift-pert RNG isolation / var-reg hinge)'
"$PY" - <<'PY'
import torch
from train_compare import apply_drift_pert, quantize_frames, HIST_DISC_LEVELS

torch.manual_seed(0)
frames = torch.rand(2, 5, 3, 8, 8)

# quantize_frames: <= HIST_DISC_LEVELS unique bins, stays in [0,1]
q = quantize_frames(frames, HIST_DISC_LEVELS)
assert q.shape == frames.shape
assert q.min() >= 0.0 and q.max() <= 1.0, (q.min().item(), q.max().item())
assert q.unique().numel() <= HIST_DISC_LEVELS, q.unique().numel()

# apply_drift_pert: same shape, clamped, and GLOBAL RNG untouched (private gen).
torch.manual_seed(123)
ctrl = torch.rand(4)              # advance the global stream
torch.manual_seed(123)
d = apply_drift_pert(frames, 0.05, seed=7)
after = torch.rand(4)            # global stream must be exactly where ctrl left it
assert d.shape == frames.shape
assert d.min() >= 0.0 and d.max() <= 1.0, (d.min().item(), d.max().item())
assert torch.equal(ctrl, after), "drift-pert perturbed the GLOBAL RNG stream!"
assert not torch.equal(d, frames), "drift-pert did nothing"
# determinism: same seed -> same field
assert torch.equal(d, apply_drift_pert(frames, 0.05, seed=7)), "drift-pert non-deterministic"

# var-reg hinge: relu(std_gt - std_pred) >= 0; == 0 exactly when deltas match.
gt = torch.randn(2, 3, 8, 8); pred = gt.clone()
assert torch.relu(gt.std() - pred.std()).item() == 0.0
frozen = torch.zeros_like(gt)   # collapsed prediction -> hinge must be > 0
assert torch.relu(gt.std() - frozen.std()).item() > 0.0
print("OK  helpers: quantize<=16 levels, drift-pert RNG-isolated+deterministic, hinge sign correct")
PY

# Tiny CPU end-to-end config (the freezing arm family: wave + dispersion).
COMMON=(--kind wave --kernel-version dispersion --grid 16 --frames 5 --batch 2 --steps 3
        --dim 16 --layers 1 --heads 2 --causal --residual --linear-pad --motion-loss
        --rollout-loss 1 --collisions --eval-batches 1 --eval-rollout 12 --eval-seeds 2
        --eval-chunk 1 --telemetry-every 1 --out "$OUT")

echo '### (2) default-path no-op: plain vs explicit --var-reg 0 --drift-pert 0 --hist-disc off'
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --tag _base
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --var-reg 0.0 --drift-pert 0.0 --hist-disc off --tag _off

echo '### (3) each knob trains end-to-end (var-reg / drift-pert / hist-disc / all-on)'
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --var-reg 0.5                              --tag _vr
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --drift-pert 0.05                          --tag _dp
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --hist-disc bits                           --tag _hd
"$PY" train_compare.py "${COMMON[@]}" --seed 0 --var-reg 0.5 --drift-pert 0.05 --hist-disc bits --tag _combo

echo '### assertions'
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
def load(tag): return json.loads((out / f"result_wave{tag}.json").read_text())

# (2) OFF == BASE on every deterministic field: the flags at their defaults are a
#     pure no-op (they allocate nothing and draw no RNG). Timing fields excluded.
base, off = load("_base"), load("_off")
det = ["params", "ffn_mult", "eval_mse", "copy_ratio", "eval_mse_over_copylast",
       "baselines", "divergence_horizon", "rollout_mse_curve", "identity_survival",
       "var_reg", "drift_pert", "hist_disc"]
for k in det:
    assert base.get(k) == off.get(k), f"default no-op broken on {k}: {base.get(k)!r} != {off.get(k)!r}"
assert base["var_reg"] == 0.0 and base["drift_pert"] == 0.0 and base["hist_disc"] == "off"
# base/off step rows must NOT carry a var_reg field (var-reg is off).
assert all("var_reg" not in r for r in base["log_tail"]), "base leaked a var_reg step field"
print(f"OK  default no-op: base==off  eval_mse={base['eval_mse']} copy_ratio={base['copy_ratio']}")

# (3) knobs recorded, metrics finite, var-reg logged per step ONLY when on.
vr, dp, hd, combo = load("_vr"), load("_dp"), load("_hd"), load("_combo")
assert vr["var_reg"] == 0.5 and vr["drift_pert"] == 0.0 and vr["hist_disc"] == "off"
assert dp["drift_pert"] == 0.05 and dp["var_reg"] == 0.0 and dp["hist_disc"] == "off"
assert hd["hist_disc"] == "bits" and hd["var_reg"] == 0.0 and hd["drift_pert"] == 0.0
assert combo["var_reg"] == 0.5 and combo["drift_pert"] == 0.05 and combo["hist_disc"] == "bits"
for name, r in (("vr", vr), ("dp", dp), ("hd", hd), ("combo", combo)):
    em = r["eval_mse"]; assert isinstance(em, (int, float)) and em == em, (name, em)
    assert r["divergence_horizon"] is not None, name
# var-reg step-row logging: present + finite for the var-reg arms, absent otherwise.
assert any("var_reg" in row for row in vr["log_tail"]), "var-reg arm did not log var_reg per step"
assert any("var_reg" in row for row in combo["log_tail"]), "combo arm did not log var_reg per step"
assert all("var_reg" not in row for row in dp["log_tail"]), "drift arm leaked a var_reg field"
assert all("var_reg" not in row for row in hd["log_tail"]), "hist-disc arm leaked a var_reg field"
vr_vals = [row["var_reg"] for row in vr["log_tail"] if "var_reg" in row]
assert all(v >= 0.0 for v in vr_vals), f"var_reg contribution went negative: {vr_vals}"
print(f"OK  var-reg logged per step (contrib>=0): {vr_vals[:3]}...")
print(f"OK  knobs recorded: vr={vr['var_reg']} dp={dp['drift_pert']} hd={hd['hist_disc']} "
      f"combo=({combo['var_reg']},{combo['drift_pert']},{combo['hist_disc']})")
print("SMOKE-ANTICOLLAPSE ALL-OK")
PY
echo "### smoke anticollapse PASS -> $OUT/"
