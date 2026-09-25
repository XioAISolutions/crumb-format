# SPECTRAL PRIOR ART + STRATEGIC MAPPING (2026-09-25)

## FreqForcing (arXiv 2607.27110, SJTU/Tencent HY, Jul 2026)
- Autoregressive video DIFFUSION: error accumulation over long horizons = color drift, MOTION STAGNATION, visual collapse.
- Frequency-domain diagnosis: accumulation shows as pronounced energy DRIFT in DC + low-frequency bands (color, layout, identity); high-freq (motion, detail) becomes unstable.
- Attention sink delays but cannot stop spectral drift; Deep Forcing's KV compression causes spectral JITTER (flicker).
- FIX (training-free): Spectral Self-Anchoring. Dual-branch: local attn (windowed, high-freq) + anchor attn (early high-quality frames, low-freq). Fuse with Gaussian low-pass: A_fused = A_loc + lam*H_lp*(A_anc - A_loc); lam=0.6, sigma=0.125. Anchor cache=6 frames, applied only at first 2 denoising steps. +16.5% latency.
- VBench-Long: best Dynamic Degree at 60s/120s (59.58/58.97) while competitive elsewhere; 24x extrapolation (5s -> 2min).

## Strategic mapping to OUR stack (wave-field AR, native k-space state)
1) Our streaming state z_t(k) IS frequency-domain. Spectral anchoring = trivial at kernel level: blend low-k state components toward a frozen anchor state: z_tilde(k) = (1-g(k)) z_t(k) + g(k) z_anchor(k), g(k) big for low k. No post-hoc FFT fusion needed (they bolt FFT onto attention outputs; we LIVE there).
2) Motion stagnation (our g32 copy_ratio collapse) = their named failure mode; spectral diagnostics = the missing metric. Add per-band energy tracking (low/mid/high) to eval + render: catches stagnation/collapse numerically.
3) Anchor cache of early high-quality frames (their design; also 'geometrically-spaced keyframe memory' in CSSC bounded-drift theory, CVPR26 workshop) = our streaming should keep sparse ANCHOR STATES, not just roll one state forward. Log-drift guarantee per their theory.
4) Curriculum: Diagonal-Distillation-style (ICLR26: more steps early, fewer late) parallels our K-ramp; note for the schedule redesign.
5) External benchmark standard: VBench-Long (60s/120s) = the reporting standard IF we ever wrap into T2V; all competitors are transformer+diffusion — NOBODY does spectral-recurrent kernels. Our whitespace intact; FreqForcing = closest philosophy, bolted on.

## Next experiments to queue (post-retune)
- E1: spectral diagnostic in eval (per-band energy of pred vs GT over rollout; stagnation + collapse signatures).
- E2: kernel-level spectral anchoring (g(k) = exp(-k2/2s^2) blend toward anchor state; anchor = state from context window start; tune g).
- E3: anchor-state streaming (K anchor states, geometric spacing) + long-run drift measurement.
- E4: re-run difficulty D6 (1024 frames) WITH E1 metrics for the collapse map.
