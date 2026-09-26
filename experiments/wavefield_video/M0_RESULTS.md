# M0_RESULTS - owner-run final (2026-09-26 05:10Z, all CPU)

## Scenario A: gain_field (exposure ramp + color cast + wandering field)

| engine | drift%(var) | drift%(dE) | hf_ssim | motion% | state | verdict |
|---|---|---|---|---|---|---|
| stats-EMA | 89.5 | 87.4 | 0.9995 | 101.0 | 24B | PASS [DSM] |
| spectral dc_only | 88.3 | 87.4 | 1.0000 | 101.0 | 24B | PASS [DSM] |
| spectral magnitude | 91.7 | 87.0 | 0.9998 | 100.6 | 864B | PASS [DSM] |
| spectral complex | 89.8 | 86.8 | 0.9987 | 100.0 | 864B | PASS [DSM] |

Verdict: M0 PASSES. All engines remove 87-92% of drift, detail untouched (ssim>=0.9987), tracker-based motion 100-101%, no false cut triggers (141 segs). Spectral magnitude leads on drift removal (91.7%) at 864B state; dc_only matches the 24B stats baseline. NOTE: at this scenario difficulty the 24B baseline is competitive - the spectral edge is small here.

## Scenario B: hotspot (mean-preserving wandering low-freq bump = phase drift)

| engine | drift%(pos) | |dE|corr | hf_ssim | motion% | state | verdict |
|---|---|---|---|---|---|---|
| stats-EMA | -0.5 | 0.008 | 0.9999 | 99.8 | 24B | fail (blind) |
| spectral dc_only | 0.3 | 0.012 | 1.0000 | 99.8 | 24B | fail (blind) |
| spectral magnitude | -18.0 | 0.124 | 0.9994 | 99.7 | 1944B | fail (worse) |
| spectral complex | 50.2 | 0.121 | 0.9957 | 99.7 | 1944B | fail (below 60 gate) |

Verdict: the expected separator. stats/dc_only are structurally blind to mean-preserving phase drift; magnitude (phase-free by design) cannot fix it; complex - the only phase-anchoring mode - removes 50.2% of the wander while keeping motion at 99.7% (no freeze cost!). Below the 60% gate; next: stronger phase blending (per-cell weights, slower phase rho) without motion damage.

## Status + next

- rho fix (0.95 to 0.995): drift removal 46% -> 92% on scenario A (the anchor no longer chases the drift).
- Metrics fixed this round: tracker-based motion% (was centroid-wobble contaminated); bounded |dE|; segment/cut accounting.
- Next M0.2: tune phase-anchor strength on hotspot (target >=60% without motion loss); then M1 synthetic AR vs vanilla generator.
