# M0.5 TASK: ghosting test + scale-selective phase correction

Context: read the OWNER-VERIFIED ADDENDUM in IMPL_NOTES_M0_3.md (frontier tables) first.

Theory to test (owner hypothesis): complex_mc's HF damage (hf_ref 0.93) =
BAND INCONSISTENCY - it shifts the LOW band (pin the bump) while the HIGH band
stays put, so every edge gets a double-image/ghost. A consistent correction
should not do this.

Predictions:
P1: applying the same correction as a CONSISTENT broadband full-frame shift
    removes ghosting: hf_ref recovers ~0.99 while pos-drift removal stays >=60
    IF the wander is translation-like on the corrected content.
P2: metric floor check - hf_ssim(img, exact_subpixel_shift(img, d)) is ~0.9x
    even for a perfectly consistent shift (HF decorrelates under subpixel
    movement). Measure d = 0.5, 1.5, 3.0 px on the hotspot INPUT.

Experiments (own scripts allowed; keep in repo):
E1: metric-floor calibration per P2. Table: d -> hf_ssim. This calibrates what
    the S gate can even mean for translation-type corrections.
E2: mc variants on hotspot: (a) broadband consistent shift (apply estimated
    correction to ALL spectral cells = equivalent to shifting the frame);
    (b) current lowband-only. Report drift%pos / hf_ref / hf_chg / motion% for
    both at mc=0.5 and 0.85.
E3: scale-selective: split into 2-3 radial bands (FFT radius). Run a small
    analysis: where does the injected bump energy live vs the balls (per-band
    energy attribution). Then apply the wander correction ONLY on the bump's
    dominant band; sweep strength. Report the frontier table.

Deliver IMPL_NOTES_M0_5.md with all tables + an honest verdict: which variant
is best on (drift>=60, hf_ref>=0.98, motion within 5), or none.

Constraints: 19 invariants must stay green; unchanged flags = byte-identical
behavior; no new dependencies (numpy/torch only); CPU-runnable (<3 min per
scenario run).
