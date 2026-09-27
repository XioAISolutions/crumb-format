#!/usr/bin/env bash
# M3 — velocity-coherence GATE: full acceptance suite + gate diagnostics.
# One command, CPU-only, a few minutes. Numbers land in crumb_coherence/out/.
#
#   PY=~/vibevoice-env/bin/python bash run_m3.sh 2>&1 | tee reviews/claude_m3_log.txt
#
# Acceptance (all must hold):
#   1. TRAP gated PASS both horizons (retention>=95, distortion<5, hf>=0.98, var in [.9,1.1])
#   2. 2nd legit scene (accelerate) PASS the same, both horizons  (no trap overfit)
#   3. M0 hotspot removal >=60% WITH the gate on; gain_field unaffected; tests green;
#      M1 low-band drop>=50% & detail>=0.98
#   4. legacy (ungated) still reachable and still FAILS the trap (for the record)
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/vibevoice-env/bin/python}"
RT=crumb_coherence/scripts/run_trap.py
M0=crumb_coherence/scripts/run_m0.py
OUT=crumb_coherence/out
mkdir -p "$OUT"
echo "PY=$PY"; "$PY" --version

echo; echo "############ 0. GATE SEPARATION — coherence C: legit(want HIGH) vs wander(want LOW)"
echo "   pick hi BELOW the translate C and ABOVE the settled wander C; lo a bit under that."
for d in 0.80 0.90 0.95 0.97; do
  echo "==== velocity-EMA decay=$d  (pos-smoother fixed at 0.5) ===="
  echo "-- translate (coherent: settled C should be HIGH, g->0) --"
  "$PY" $RT --scene translate --frames 96 --diag --gate-decay $d | sed -n '/gate diagnostic/,/^=====/p'
  echo "-- wander (incoherent: settled C should be LOW, g->1) --"
  "$PY" $RT --scene wander    --frames 96 --diag --gate-decay $d | sed -n '/gate diagnostic/,/^=====/p'
done

echo; echo "############ 1. TRAP — LEGACY (pre-M3, no gate): should still FAIL (record)"
"$PY" $RT --scene translate --frames 96  --legacy
"$PY" $RT --scene translate --frames 192 --legacy

echo; echo "############ 2. TRAP — M3 GATED, translate, BOTH horizons (acceptance 1)"
"$PY" $RT --scene translate --frames 96  --diag --json $OUT/trap_translate_96.json
"$PY" $RT --scene translate --frames 192 --diag --json $OUT/trap_translate_192.json

echo; echo "############ 3. TRAP — M3 GATED, accelerate, BOTH horizons (acceptance 2)"
"$PY" $RT --scene accelerate --frames 96  --diag --json $OUT/trap_accel_96.json
"$PY" $RT --scene accelerate --frames 192 --diag --json $OUT/trap_accel_192.json

echo; echo "############ 4. TRAP — M3 GATED, curve (STRESS, reported — turning-arc limit)"
"$PY" $RT --scene curve --frames 96 --diag --json $OUT/trap_curve_96.json

echo; echo "############ 5. REGRESSION — M0 hotspot: gated (SHIP) then ungated (removal>=60%)"
"$PY" $M0 --scenario hotspot --frames 48 --no-video --hotspot-mc-gate
"$PY" $M0 --scenario hotspot --frames 48 --no-video --no-hotspot-mc-gate

echo; echo "############ 6. REGRESSION — M0 gain_field (magnitude default; gate not involved)"
"$PY" $M0 --scenario gain_field --frames 48 --no-video

echo; echo "############ 7. UNIT TESTS — core invariants incl. the gate"
"$PY" crumb_coherence/tests/test_core.py

echo; echo "############ 8. M1 integration (self-referential; low-band drop + detail)"
"$PY" crumb_coherence/scripts/m1_integration.py --frames 64

echo; echo "############ DONE. JSONs in $OUT/ ; interpret against the acceptance bars above."
