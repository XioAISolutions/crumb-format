# Crumb Video — Master Plan ("Boil the Ocean")
_2026-09-24/25 · wave-field video research · single RTX 4090 · owner: Slava_

## Thesis
Replace O(N^2) attention with Fourier-domain spectral mixing (separable damped-
cosine kernels). Wedge = the frontier's weakest point: shot-length caps (Seedance
2.5 / Kling 3.0 / Veo 3.1 = 8-15s) and quadratic long-context cost. Target =
minutes-long coherent video at constant memory on commodity GPUs.

## Receipts so far
- 2D/3D kernel FFT vs brute-force conv: max err ~5e-6 (correct). Causality mask OK.
- Scaling (64ch field, CPU): N=1k: 0.7ms vs attn 6ms · N=4k: 0.8 vs 105ms ·
  N=16k: 3.7 vs 1546ms (417x) · N=64k: 13ms (attention dies).
- v1 wave on 4090 (grid32, T=17, dim256/6L, 1200 steps): eval MSE 0.0082,
  drift 0.0083->0.016 over 16 steps, 3.95M params, 15GB peak, 30 min.
- v1 attn baseline: running (~0.24 st/s; ~83 min) — historical only, superseded.

## Frontier consults (2026-09-24/25)
- Claude Opus 5.5: reviews/claude_opus_review.md (14.8KB)
- GPT-6 Astra: reviews/chatgpt_astra_review.md (14.2KB, RAN the code + built baselines)
Both agree: harness must be fixed before architecture science.

## Findings that resculpt the plan (must-fix)
1. TRIVIAL BASELINES WIN: copy-last ≈ 0.0155 MSE at 8x8; both v1 models lost to it
   (attn 0.092, wave 0.153). => every table needs P0 baselines; residual prediction
   (pred = last_frame + zero-init delta) is MANDATORY and instantly starts at P0.
2. PARAM MISMATCH: v1 "same budget" was false (attn up to 5.7x larger; giant [N,dim]
   pos tables). => match within ~10%, ONE compact shared factorized t+y+x encoding.
3. 1-STEP TRAINING vs AUTOREGRESSIVE GOAL: add short rollout loss (4 steps).
4. CIRCULAR BOUNDARY vs WALLS: 3D FFT is toroidal; data bounces off walls.
   => zero-pad (linear conv) or reflection padding, time AND space.
5. SEPARABILITY = THE DEEP LIMIT: K(t)K(y)K(x) can't express propagating fields;
   moving objects need omega_t ~ v.k coupling. Standing field vs PROPAGATING field.
   => W1: temporal pole depends on spatial frequency: lambda(kx,ky) =
   exp(-alpha(kx,ky)) exp(i(Omega(kx,ky))), Omega = vx*kx + vy*ky + beta*sqrt(kx^2+ky^2),
   2-4 modes/head. Recurrence-compatible (z_t(k) = lambda(k) z_{t-1}(k) + B x_t(k)).
6. CONTENT INDEPENDENCE ceiling => cheap Hyena-style gate: y = out(g * wave(v)).
7. LOCAL PATH: parallel 3x3 depthwise conv fused with the global field.
8. MUST become causal+linear when the recurrent state (not window) era begins.
9. LATENT SPACE mandatory for minutes (dense pixels cap at ~2-4s at 32^2 on 4090).

## Target architecture (converged)
latent frame -> local 3x3 -> 2D FFT -> lambda(kx,ky) oscillator bank (2-4 modes,
gated) -> causal recurrent temporal state -> iFFT -> fuse local -> head (residual).

## Tonight's falsifiable protocol (Astra design)
Config: grid16, T=16, D=128, L=4, H=8, 3 balls, elastic collisions, deterministic
(no stochastic kicks); 1-step + 0.25*L2 + 0.25*L3 + 0.25*L4 rollout loss.
Arms: P0 copy-last | W0 separable wave | W1 propagation wave (+gate) | S1
diagonal/complex SSM baseline | A1 attention (matched, compact pos).
Eval: 256-step rollouts, 32 unseen seeds. Metrics: r = MSE(copy-last)/MSE(model)
(must be >1 first); ball centroid error (mean/final), identity survival;
divergence horizon = first frame where median centroid error > 1 ball radius for
8 consecutive frames (radius 1.6px — fixed BEFORE running); efficiency table
(peak VRAM, steps/s, rollout frames/s, mem at 32 and 256).
KILL: W1 <= SSM on all metrics + no efficiency edge => drop wave frame, use SSM.
PROVE: W1 ~ attention quality, > SSM horizon, << attention memory => recurrence
form + 512/1024-frame streaming test next.

## Queue (GPU never idle)
1. [running] v1 attn p1 (baseline record).
2. v2 impl (Claude round 1: causal/linear/posemb/ssm_lite/centroid; round 2:
   residual/baselines/W1 dispersion/gate/fuse/modes/param-match/4-step-loss/
   horizon-eval) -> smoke on Mac -> box suite P0/W0/W1/S1/A1 (~2h).
3. Recurrence streaming (512/1024 frames, constant memory) on winner.
4. Latent AE (4x downsample) + latent-space rerun of the suite.
5. Scale grids {16,32,64}; efficiency curves vs attention wall.
6. Report + (eventually) crumb-video open release candidate.

## MILESTONE 2026-09-25: streaming recurrence VERIFIED
- dispersion step() == forward() (1e-3), causality + stability gates green.
- 384-frame stream flat 2600 fps CPU (attn windowed 1000).
- GPU 1024-frame stream test staged (fires after w1g32).

## 2026-09-25 eager-mix finding: torch.compile CANNOT fuse the complex path
- inductor: 'does not support code generation for complex operators' -> falls back to eager.
- CPU bench: fwd 62.3 vs 61.0 ms; stream step 2.924 vs 2.927 ms/step = no gain.
- => hand-Triton (real/imag packed) is the ONLY route to fusion for fft>mult>gate>ifft.

## MILESTONE 2 (2026-09-25 PM): build sprint
- r4 objective (motion-loss + copy_ratio
 + rollout-ramp) built + smoke-verified; copy_ratio metric live.
- r5: data_waves.py PDE fields (analytic err 1e-12) + audit + proofpack tools; wired, 29 tests pass.
- r6: Triton fused kernels correct (2e-7); cplxmul 2.2x at grid32; step +13%; chain 0.90x (cuFFT-bound).
- Queue: w1g32 -> clean stream -> bench -> motion-balanced retrains -> PDE runs.
