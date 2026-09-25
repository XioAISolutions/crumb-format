# HARDENING DIGEST (Astra deep-research, 2026-09-25) — actionable core

## The reframe (accepted)
Stop selling wave as 'attention replacement'. Position: WaveMemory = ultra-compact recurrent dynamical MEMORY inside a hybrid (local neural perception + structured recurrent dynamics + bounded adaptive control = long-horizon world state). Aligns with SSM-video direction (global state + local attention); our edge = structured, interpretable, ultra-cheap state. BrainSNN tie-in: perception -> persistent state -> prediction -> action.

## Priority table (expected payoff / cost / kill signal)
1. Finish g32 objective matrix (RUNNING retune) — kill: motion still collapses.
2. Generic SSM baseline arms — CRITICAL — kill: SSM dominates wave.
3. Local+Wave hybrid (gated fusion) — kill: wave adds little.
4. Self-rollout training schedule (GT -> generated history 0/25/50-75%) — kill: no horizon gain.
5. AE qualification (is 16x latent actually safe?) — kill: latent destroys state.
6. Divergence-horizon eval (have div_h; keep as core).
7. Memory probe = occlusion benchmark (THE experiment; r14 build fires tonight) — kill: no long-memory advantage.
8. Adaptive WaveField (alpha_t = alpha0 + deltaalpha_t etc, content-conditional) — kill: complexity w/o horizon gain.
9. Triton recurrence — PARKED until model survives (r16 spec parked).

## Stop / Continue
STOP: chasing pixel MSE; loss coefficients without mechanism; blind resolution scaling; Triton now; attention-only baselines; assuming 16x latent; victory claims from balls; indiscriminate params.
CONTINUE: multi-seed; semantic+anti-cheat metrics; difficulty ladder; 4090 constraint; causal recurrence; fair baseline protocol.

## The decisive experiment (occlusion memory-probe)
64x64, 8 balls + occluder; target visible 0-64, hidden 64-320, emerges 320+; exit determined by pre-occlusion state; 1024-frame rollout; arms: local-attn / ssm / wave / local+ssm / local+wave (equal compute, 5 seeds); metrics incl. STATE BYTES. Milestone: WaveMemory must beat generic SSM on divergence-horizon-PER-BYTE. If yes: double down. If no: don't polish the wave idea because it's elegant.
