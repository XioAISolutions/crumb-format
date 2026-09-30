# Fast-prompt travel — LongLive-2.0-5B + OPSD-V, 2026-09-30

## Claim
On our 4090 (24 GB), the **fast-travel prompt** flips LongLive-2.0-5B from
creep-class motion (~0.4 px/f, no travel) to a **follow-the-walker travel
regime** (~1.0–1.1 px/f, net ≈ 0, contact-verified landmark progression), and
travel **sustains at 30 s** (flow 1.132). Separately, the released **OPSD-V
post-trained 1.3B** stack produces 40 s of travel-class footage on the same card
after three surgical memory patches.

## Numbers (median px/frame @640w)
| run | length | flow | net px | rmse | verdict (contact) |
|---|---|---|---|---|---|
| ladder s0 (W12 fp8) | 10 s | 0.404 | −24.8 | — | creep |
| ladder s1/s2/s3 | 10 s | .594/.614/.203 | ≈0 | — | no travel |
| bf16 s0 W24 | 10 s | 0.578 | −20.1 | 11.35 | creep (quant ruled out) |
| **fast s0** | 10 s | **1.118** | −0.6 | — | **TRAVEL ✓** |
| fast s11 | 10 s | 0.630 | −12.9 | 13.57 | no travel |
| **fast s2** | 10 s | **1.006** | +0.6 | 20.78 | **TRAVEL ✓** |
| **fast s0** | 30 s | **1.132** | −1.6 | 19.47 | **TRAVEL ✓ (sustained)** |
| OPSD orig | 39.5 s | 1.740 | +28.8 | 26.2 | TRAVEL ✓ (1.3B, rough) |
| OPSD lora | 39.5 s | 1.960 | +13.7 | 28.2 | TRAVEL ✓ |
| **fast s0** | **180 s** | **1.176** | −11.9 | 19.42 | **TRAVEL ✓ (3 min, halves 1.156/1.196)** |
| OPSD 477 | 119 s | 1.835 | −225.4 | 27.72 | TRAVEL ✓ (2 min, 1.3B) |
| **fast s0** | **300 s** | **1.165** | −18.6 | 19.30 | **TRAVEL ✓ (5 min, halves 1.160/1.173) — MISSION RUNG** |

Travel signature: flow ≥ ~1.0 with net ≈ 0 (first↔last phase-correlate);
non-travel = flow ≤ 0.63 and/or large net. Contacts are the arbiter.

## Evidence highlights
- `c01z_1` (180 s): contact shows a new street segment in every panel — doors,
  window grilles, flower boxes, bins passing; walker held at constant distance
  (the model's follow-the-walker behavior) for the full 3 minutes.
- `c01v_5` / `c01y_2` / `c01y_3`: landmark-passing verified by frame comparison;
  walker follows with new façades throughout.
- OPSD pair: threading alleys, landmark progression; rougher texture (1.3B).

## Caveats
- Single seed family (0) at length; s11 = no-travel control. s2 travels at 10 s.
- net ≈ 0 is the travel SIGNATURE here (phase-correlate first↔last), not a
  contradiction: the walked path returns near its start or is non-rigid;
  contacts are the arbiter of progress.
- Flow is a proxy: walker motion inflates it (s11 = 0.63 no-travel). Treat
  flow ≥ ~1.0 + net small + contact progression as the joint criterion.
- 5B quality: some softness/blur spots; rmse 19-21 typical for travel runs.
- 1.3B OPSD: 832×480 @16 fps, rmse 26-28 — acceptable as a motion proof, below
  our 5B quality bar.

## Artifacts (local harvest)
- `clip_harvest/c01ab_1_fast300s_s0/rank0-0-0_regular.mp4` (1280×704, 24 fps,
  **7229 fr = 301.2 s**, sha256 aa4bd761a574abcb…)
- `.../send.mp4` (960w crf31, 30.0 MB)
- `clip_harvest/c01z_1_fast180s_s0/rank0-0-0_regular.mp4` (1280×704, 24 fps,
  4349 fr, sha256 04c86d9d84100e2e5ff8…)
- `.../send.mp4` (960w crf28, sha256 16dad4582de9c9dff3c8…)

