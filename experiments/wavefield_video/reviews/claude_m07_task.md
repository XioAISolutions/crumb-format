# M0.7 TASK: the oracle four-arm test (+ X−B̂+B̂_stable variant)

FIRE NOTE: do not launch before 2:30pm EDT (Claude session limit resets then).

Context: read IMPL_NOTES_M0_6.md (both plugin axes pass offline; complex_mc band=0.03, mc=0.5, est=corr, edge=hard = 64.0% [DSME]) and reviews/DEEP_DIVE_M05_astra.md (the oracle four-arm test design + the "even better variant").

GOAL: locate the remaining headroom precisely — estimator error vs metric error vs decomposition error — before touching the estimator further.

Build crumb_coherence/scripts/run_oracle.py implementing the four-arm oracle test on the hotspot scenario:

- Arm 1 (ceiling): ORACLE component — replace the bump's low-band with the CLEAN control's low-band (perfect nuisance removal; upper bound of any component-space fix). Report: drift%pos, hf_ref raw, hf_ref after oracle registration (align corrected to clean via the existing phase-correlation helper before scoring), ball motion%. (CW-SSIM: skip and note — needs a new dep.)
- Arm 2: ORACLE band-phase — apply the TRUE per-frame bump displacement (known from the injector) as the band-limited phase correction (perfect estimator, same band/edge as the engine). Separates estimator error from decomposition error.
- Arm 3: ENGINE band-phase — the M0.6 winner config, same metrics (context row).
- Arm 4: X − B̂ + B̂_stable — estimate B̂ as the coherent low-band component that moves (from the engine's own displacement estimate); output = X − B̂_drifting + B̂_at_stable_position. Simplest faithful version; document the exact definition used.

Deliverables: scripts/run_oracle.py + outputs pasted into IMPL_NOTES_M0_7.md + a decision paragraph mapping results to Astra's Outcome 1/2/3/4 (state which one we are in and the consequences for the next round).

Constraints: do NOT modify crumb_coherence/ source in this round (script reuses helpers read-only); no new deps; runtime < 5 min per arm; honest notes (no invented numbers).
