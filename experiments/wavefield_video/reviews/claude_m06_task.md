# M0.6 TASK: land the band-limited motion-compensated engine (the E3 win) + Astra refinements

Context: read IMPL_NOTES_M0_5.md and reviews/DEEP_DIVE_M05_astra.md (sections A/B/C, the "even better variant", and the oracle four-arm test design).

OWNER-RUN RESULT (run_m05.py already executed): restricting the mc correction to the inner radial band r<0.03 with mc=0.5 gives 64.0% drift removal + hf_ref 0.9946 + hf_chg 0.9955 + motion 99.7% = FIRST [DSM] PASS on the hotspot scenario. E1 proved the hf gate floor: a perfect subpixel translation of 1.5px scores ~0.94 by construction, 3px ~0.87 — so large translations cannot pass 0.98 on hf; the inner-band trick sidesteps this because the bump sits below the HF measure's scale. E2 broadband-consistent = falsified (moves the balls; motion collapsed).

Deliverables:
1. Promote the E3 mechanism into crumb_coherence/core.py as an additive kwarg on the complex_mc engine: mc_band: float = 0.0 (0.0 = current behavior byte-identical; >0 = restrict the mc correction to radial band r < mc_band). Reuse run_m05.py's band-mask definition EXACTLY — factor it into core so run_m05.py and the engine share one definition.
2. Implement Astra's refinements as separate additive kwargs, defaults = current behavior:
   a. mc_est="phase_plane" option: weighted phase-plane displacement estimate (trimmed/Huber over the lowest non-DC cells, magnitude weights) — keep the current estimate as default.
   b. mc_smooth=True option: causal alpha-beta trajectory filter on the estimate (x, y, vx, vy + confidence) -> correction delta_t = p_star - p_t. Default off.
   c. mc_edge="flat_top" option: cosine-transition edge centered on mc_band (transitions inward, width ~0.3x the band) rather than a hard cut; apply F' = [(1-W) + W * exp(-i k . delta)] F. Default = hard cut.
3. Tests: extend test_core.py (19 -> 21+): band mask invariants (unity inside, zero outside, radial symmetry), default-off byte-identity for each new kwarg.
4. Run these and paste outputs into IMPL_NOTES_M0_6.md:
   - python crumb_coherence/tests/test_core.py
   - python crumb_coherence/scripts/run_m0.py --scenario gain_field --no-video   (must stay all PASS, rows unchanged)
   - python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video      (complex_mc row expected [DSM])
   - python crumb_coherence/scripts/run_m05.py                                   (E3 table through the engine path)
   - a small sweep over (mc_band x mc x estimator x smooth x edge) combos, reported as a table.
5. Verdict: state the recommended default for complex_mc (the combo that passes [DSM] with the best drift) + honest caveats, in the notes.

Constraints: no new deps; additive public API only; every existing engine's behavior must be byte-identical when new kwargs sit at defaults; keep runtime sane (< 4 min per full run); no emoji; write notes to IMPL_NOTES_M0_6.md.
