Implement the v2 fairness + architecture suite in this directory. FIRST read
reviews/claude_opus_review.md and reviews/chatgpt_astra_review.md — they define
requirements and rationale. Keep v1 behavior as defaults; gate new behavior
behind flags; run CPU smokes for each arm at grid 8/16 and record results.

CHANGES
1. train_compare.py: residual prediction: pred = frames[:, -1] + delta, where
   delta comes from the prediction head applied to the LAST context frame's
   tokens; zero-initialize the head's final linear weights and bias. Flag
   --residual (default ON for v2 runs; off reproduces v1).
2. Baselines reported in EVERY result JSON + printed table: zero predictor,
   copy-last (MSE of last context frame vs target), constant-velocity (predict
   frame T as frame T-1 + (T-1 frame delta)).
3. Parameter matching: --target-params N; auto-adjust ffn_mult (and n_heads if
   needed) per arm to land within ~10% of target; print achieved counts.
4. ONE compact shared factorized positional encoding (t,y,x) added at the input
   for ALL arms; delete the giant [1,N,dim] attention position table entirely.
5. wfvideo.py WaveMix3D: --kernel-version {separable, dispersion}. Dispersion:
   temporal poles lambda(kx,ky) = exp(-alpha(kx,ky)) * exp(i*Omega(kx,ky)),
   Omega = vx*kx + vy*ky + beta*sqrt(kx^2+ky^2); alpha,beta,vx,vy learned
   (small MLP or low-rank factors over (kx,ky) per head); 2-4 modes per head
   (sum of damped oscillators, per-mode complex state). Implement as a
   frequency-domain multiplier over the time axis per spatial-frequency cell
   (rfft over time). Document the recurrence form z_t(k) = lambda(k) z_{t-1}(k)
   + B x_t(k) in comments for the later streaming test.
6. Content gate (Hyena-style): y = po( g * wave(pi(x)) ), g = sigmoid of a small
   MLP on pooled per-head features, broadcast back. Flag --gate.
7. Local path: parallel 3x3 depthwise conv per head fused (add) with the wave
   output before po. Flag --local-fuse.
8. Linear (zero-padded) convolution over time AND space (fft with s > input,
   crop after inverse) so the FFT does not see a torus. Flag --linear-pad
   (default ON for v2).
9. Multi-step rollout loss: --rollout-loss K with total L = L1 + 0.25*L2 + ...
   + 0.25*LK; each step's forward uses the model's own previous predictions
   (detach between steps so each contributes gradients only through its own
   forward pass).
10. Divergence-horizon eval (--eval-rollout 256, 32 unseen seeds): metrics in
    result JSON: r = MSE(copy-last)/MSE(model), mean + final ball-centroid
    error (color-matched), identity survival fraction, divergence horizon =
    first frame where median centroid error > 1.6 px for 8 consecutive frames,
    efficiency: steps/s, rollout frames/s, peak VRAM, memory at rollout 32 vs
    256. Fixed before running; no threshold movement afterward.
11. data.py: ball-ball elastic collisions (swap velocities on overlap) exposed
    behind --collisions; expose ball radius as a config constant.
12. Keep it minimal and correct. IMPL_NOTES_V2.md: what changed, smoke numbers,
    open risks. Do not commit to git.
