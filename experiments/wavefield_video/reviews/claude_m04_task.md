# M0.4 TASK: lossless k-space phase-ramp re-centering

Context: read IMPL_NOTES_M0_3.md (incl. the OWNER-RUN VERDICT at the end). M0.3's motion-compensated re-centering VALIDATED the direction: at mc=0.5 the hotspot pos-drift removal hits 70.2% (target >=60). BUT hf_ssim collapses to ~0.93 at every mc>0 (gate: >=0.98), and gain_field's complex_mc shows 0.9678 despite ~zero wander - the damage happens even at an identity/near-zero shift, which points at the spatial-warp INTERPOLATION as the cause, not the correction itself.

THEORY TO TEST: the re-centering is currently applied as a spatial resample (interpolation). Replace it with an EXACT, UNITARY k-space phase-ramp shift: for a shift (dy,dx), multiply the spectrum by exp(-2j*pi*(kh*dy/Hp + kw*dx/Wp)) - no interpolation, no HF loss by construction. At ~zero estimated shift this becomes a bit-exact identity (make that an explicit invariant: when the estimator returns zero shift, output is byte-identical to the non-mc path).

Also: only run complex_mc's correction in the hotspot scenario (in gain_field it must be a no-op OR not be listed as a graded engine - the S-gate failure there is an artifact of running a hotspot-specific correction on a scenario without wander; preserve existing engine rows' numbers byte-identical).

Requirements:
- All changes in crumb_coherence/ only; keep public API stable; extend tests (>=17 invariants: zero-shift identity for complex_mc, shift round-trip accuracy).
- Run: (1) tests, (2) gain_field (all pre-existing engines byte-identical; complex_mc either absent or PASS), (3) hotspot sweep mc=0.0/0.5/0.7.
- Target: hotspot mc=0.5 with D>=60 AND S>=0.98 AND M AND E all pass. If k-space shift still shows hf_ssim < 0.98, report it honestly with the residual cause analysis (e.g. phase estimator noise feeding back).
- Append an OWNER-VERIFIABLE table into IMPL_NOTES_M0_4.md; do not touch IMPL_NOTES_M0_3.md.

No emoji. Runtime < 3 min CPU per scenario.