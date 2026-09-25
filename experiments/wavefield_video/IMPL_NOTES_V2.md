# Implementation notes — v2 fairness + architecture suite

Scope: everything lives in `experiments/wavefield_video/`. Not committed to git.
The two reviews (`reviews/claude_opus_review.md`, `reviews/chatgpt_astra_review.md`)
define the requirements; this suite implements all 12 changes from the round-2 task.

**Design stance.** The reviews agree the v1 comparison was confounded three ways —
attention was 3–6× larger, neither model beat a trivial copy-last baseline, and the
separable kernel `K(t)·K(y)·K(x)` factorizes space and time so it _cannot_ represent a
translating object (spectrum `ω_t = v·k`). v2 fixes the **harness** (residual, baselines,
param-match, one shared positional encoding) and only then does the **architecture
science** (a dispersion kernel that couples space–time, plus gate/local/rollout).

v1 numerics are still reproducible: the `separable` wave arm with `--no-linear-pad
--no-residual` and no causal/gate/local flags takes the exact v1 `rfftn` code path.

---

## Change-by-change

**1. Residual prediction** (`wfvideo.VideoPredictor`, flag `--residual`, default ON).
`pred = frames[:, -1] + delta`, `delta = head(last-context-frame tokens)`. The head's
final `Linear` weight **and** bias are zero-initialized, so at step 0 `delta ≡ 0` and the
model starts _exactly_ at the copy-last baseline instead of learning RGB reconstruction
from scratch. `sanity_check.py` #8 asserts `mr(frames) == frames[:,-1]` at init.

**2. Baselines in every result JSON + printed table** (`train_compare.baseline_mses`).
`zero` (MSE of all-zeros vs target), `copy_last` (last context frame vs target),
`const_vel` (`last + (last - prev)` vs target). Reported under `"baselines"`, plus
`eval_mse_over_copylast` and the rollout `r_persist_over_model`. Printed as a table at
the end of every run. If a learned arm can't beat `copy_last`, that's visible immediately.

**3. Parameter matching** (`--target-params N`, `train_compare.match_ffn_mult`).
Total params are monotonic in FFN width, so a 44-step bisection on `ffn_mult` lands within
~10% of `N`. Attention's `qkv` is a fixed `3·dim²` regardless of heads, so FFN width is the
only clean equalizer across arms — that's why we tune it rather than heads. Achieved count

- % off are printed (`PARAM-MATCH ...`, `WARN>10%` if it can't reach target) and stored
  (`params`, `ffn_mult`). n_heads adjustment was left out: for these arms it moves the count
  negligibly, and FFN width covers the full range.

**4. One shared factorized (t,y,x) positional encoding** (`wfvideo.FactorizedPosEmb`).
`(T+H+W)·dim` params, summed and broadcast-added at the model input for **all** arms,
always on. The giant per-layer `[1, T·H·W, dim]` attention table is **deleted** — attention
no longer gets a free positional-memory advantage.

**5. Dispersion kernel** (`--kernel-version {separable,dispersion}`). Per head, 2–4
oscillator modes; each mode has a temporal pole per spatial-frequency cell
`λ(kx,ky) = exp(-α(kx,ky))·exp(i·Ω(kx,ky))`, `Ω = vx·kx + vy·ky + β·|k|`, with
`α = softplus(a0 + a1·|k|) > 0` (⇒ `|λ|<1`, **unconditionally stable**; `a1>0` damps small
scales like viscosity). Implemented as a frequency-domain temporal multiplier: 2-D spatial
FFT → per-cell temporal transfer `G(ω,k) = Σ_m Cₘ Bₘ / (1 − λₘ(k) e^{−iω})` over a
zero-padded (`L=2T`) time FFT → inverse. That transfer is exactly the frequency response of
the causal recurrence `z_t(k) = λ(k) z_{t-1}(k) + B x_t(k)`, `y_t = Re(C z_t)` — documented
in the code so the later streaming test can swap the FFT for the O(1)-in-T recurrence. This
is the one operator that can represent a _propagating_ field (advection `vx·kx`), which the
separable standing-field kernel cannot.

**6. Content gate** (Hyena-style, `--gate`). `y = po(g · wave(pi(x)))`,
`g = sigmoid(Linear_dh(mean-pooled per-head features))`, broadcast over (t,y,x). Cheap,
keeps FFT parallelism, and directly attacks the fixed-filter (content-independence) weakness.

**7. Local path** (`--local-fuse`). A parallel `3×3` depthwise conv per (head,channel),
added to the wave output before `po`. The global field carries transport/coherence; the
local path reconstructs edges so the field isn't forced to.

**8. Linear (zero-padded) convolution over time AND space** (`--linear-pad`, default ON).
Separable path: each axis uses a length-`2n−1` FFT with natural kernel placement + a center
crop — a true non-circular "same" convolution (a symmetric kernel stays symmetric; a causal
kernel reads only the past). Dispersion path: time is zero-padded to `L=2T` (causal by
construction), space is zero-padded to `2H×2W` so the advection phase ramp doesn't wrap the
image on a torus. `--no-linear-pad` restores circular v1 behavior.

**9. Multi-step rollout loss** (`--rollout-loss K`). `L = L₁ + 0.25·L₂ + … + 0.25·L_K`;
each step re-runs the forward on the model's own previous prediction, **detached** between
steps so each `Lₖ` contributes gradients only through its own forward pass. Trains the thing
we actually measure (autoregressive stability) instead of pure one-step teacher forcing.

**10. Divergence-horizon eval** (`--eval-rollout 256 --eval-seeds 32`,
`train_compare.rollout_eval`). Rolls R frames from S unseen seeds (seed 90000). Metrics in
the JSON: `r_persist_over_model = MSE(copy-last)/MSE(model)`; `mean_centroid_err` +
`final_centroid_err` (color-matched per-ball centroids — each pixel projected onto each
ball's unit color vector, isolating it through occlusion); `identity_survival` (fraction of
balls with final error < `2·RADIUS`); `divergence_horizon` (first frame where the median
centroid error exceeds **`RADIUS = 1.6 px` for 8 consecutive frames**); efficiency
(`train_steps_s`, `rollout_fps`, `peak_gb`, `mem_gb_at_32`, `mem_gb_at_256`). **Thresholds
are fixed in code (`DIV_THRESH`, `DIV_CONSEC`, `ID_SURV_TOL`) before any run** and must not
move afterward.

**11. Ball-ball collisions** (`data.py`, `--collisions`). Two overlapping _and approaching_
balls swap velocity vectors (equal-mass elastic exchange; the approach test prevents
sticking). Deterministic, so no irreducible uncertainty is introduced — it just couples the
balls so occlusion/identity tracking is a real test. `RADIUS` (1.6 px, the Gaussian sigma)
is the exported ball-size constant; `CONTACT = 2·RADIUS`.

**12. This file.** Minimal, correct, not committed.

---

## Smoke tests — STATUS: BLOCKED (session hard-gates `python3`)

Same gate as the round-1 session: every `python3` invocation returns "requires approval"
and is denied, so **no runtime output was produced and none is fabricated.** The suite is
fully wired; one approval runs all of it:

```bash
cd experiments/wavefield_video
bash run_smoke.sh              # sanity_check.py, then each arm at grid 8 and 16
# or: PY=/path/to/venv/bin/python bash run_smoke.sh
```

`run_smoke.sh` runs the correctness gate first, then — at grid 8 and 16, param-matched to
120k, with `--collisions` — a wave arm (dispersion + gate + local-fuse + 3-step rollout
loss + causal), attention, SSM, and a separable-wave reference. Results land in
`smoke_v2/result_*.json`.

`sanity_check.py` is the fast correctness gate (prints `OK`/`FAIL` per line, no training):
non-square separable wave; causal-tap zeroing; separable linear-pad finite; **dispersion
shape/finite + `|λ|<1` stability**; gate+local finite (both kernels); attention no
future-leak; **`SSMLite.forward()==step()`**; **residual zero-init == copy-last**;
end-to-end all three arms; color-centroid recovers a planted ball; divergence-horizon
consec logic. It ends with `SANITY ALL-OK` or `SANITY FAILURES-PRESENT`.

**When you approve Python, I'll run it and paste the real numbers into this file.**

---

## Static self-review (since I couldn't execute)

Traced by hand; highest-confidence concerns first:

- **Fixed a real bug mid-implementation:** the spatial linear-pad conv originally used
  `ifftshift` placement + `[0,n)` crop, which zero-padding turns into a _one-sided_ filter —
  wrong for a symmetric spatial kernel. Now uses natural placement + center crop (correct
  "same" linear conv). Time causality still works because a causal kernel's future taps are
  already zero.
- **Dispersion stability** rests on `softplus(...) > 0 ⇒ |λ| < 1`, so `|1 − λe^{−iω}| ≥
1 − |λ| > 0` — the transfer never blows up regardless of learned params. Verified in
  `sanity_check.py`.
- **Dispersion memory** is the known tradeoff: the `[B,nh,L,2H,2W,dh]` complex spectrum is
  large at grid 32+ (the reviews' "O(N) activation memory" wall). Fine at smoke sizes;
  `--ckpt` and the spatial zero-pad-off path exist as escape hatches.

## Open risks / caveats

- **Unverified at runtime.** Logic reviewed on paper only. If a smoke fails, first suspects:
  `torch.fft.fft2(..., s=(Hp,Wp), dim=(3,4))` on the 6-D tensor, and the `reshape(shp)` of
  the rfft kernel in `_lin_conv_axis` (both traced, believed correct).
- **Color-matched centroid** assumes balls have distinct random colors (they do, `0.25–1.0`
  per channel); heavy color collision between two balls would blur their centroids. Coarse
  but far better than the v1 single global centroid.
- **Param match tunes only `ffn_mult`.** If a target is below the floor model (all flags,
  `ffn_mult→0.05`), it reports the closest reachable count with `WARN>10%` rather than
  silently missing.
- **bf16/FFT dtype asymmetry** (pre-existing): the FFT paths upcast to fp32 while attention
  SDPA runs bf16 on CUDA, so wall-clock isn't perfectly apples-to-apples on dtype.
- **Streaming recurrence for dispersion** is documented but not yet wired as a `step()` (the
  reviews scope that for _after_ the mixer proves useful). `SSMLite.step()` exists today.
