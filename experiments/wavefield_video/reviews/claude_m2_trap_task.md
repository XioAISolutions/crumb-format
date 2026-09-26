# M2 TASK: legit-motion TRAP test (Astra's top-priority falsification)

Context: read reviews/DEEP_DIVE_M1_assess.md section 2. Rename note: internal names are now "geometric-displacement control" / "known-component replacement" (not "oracle").

Goal: write scripts/run_trap.py at 64×64: a large soft Gaussian object translating at 0.15 px/frame + slow sinusoidal illumination + 3 small balls on independent trajectories; NO injected wander. Freeze the shipping config (complex_mc band=0.03 strength=0.5 corr hard, alpha=0.95 rho=0.995 cutoff=0.14).

Measure: (1) intentional centroid-motion retention %; (2) low-frequency trajectory distortion (|plugin−clean| range vs clean range); (3) HF-SSIM vs clean; (4) clean low-band variance ratio plugin/raw (should be ~1). Kill criteria: retention >95%, distortion <5%, HF>0.98, low-band var ratio in [0.9, 1.1].

Deliverables: scripts/run_trap.py + IMPL_NOTES_M2.md with commands, pre-registered expectations, and a verdict paragraph naming which hypothesis dies if it fails. Runs are owner-run as usual. No engine changes; runtime < 5 min.
