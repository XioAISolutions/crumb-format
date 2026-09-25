# BRIEF 2 (2026-09-25 eve): post-objective results + a g32 collapse to diagnose

## Numbers
- w1r4 (g16, motion-loss + K-ramp 3->8): 1-step/copy 0.434 | copy_ratio 0.719 | div_h 1 | curve tail ~0.11
- w1g32b (g32, SAME objective): 1-step/copy 1.012 | copy_ratio 0.115 (under-motion collapse) | div_h 0 | curve tail 0.94
- Render protocol (fair): model MSE 256 frames, 5 seeds: wave 0.074-0.082 vs attn 0.146-0.183 (wave wins 5/5). Frozen-frame baseline 0.043.
- Render and eval agree on model MSE (0.0777 curve mean vs 0.0819 render, same seed family).
- MOVE_THRESH = 0.05; g16 moving-frac 0.746. RADIUS 1.6px, SPEED 1.15 px/frame (constants, grid-independent).

## Questions
1) Why would the wave arm collapse toward freezing at g32 under this objective (0.719 -> 0.115 copy_ratio)? Threshold scale-dependence? Kernel capacity? K-ramp interaction?
2) Concrete objective/schedule fix to run next (rank 2-3 experiments, most informative first).
3) Sanity-check the protocol reconciliation above; any flaw?
