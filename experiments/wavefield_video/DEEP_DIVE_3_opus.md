# DEEP DIVE 3 (opus) — the g32 freeze root-cause, an airtight budget suite, and the next architecture move

Answers `CONSULT_BRIEF_3.md` Q1 + Q2. Scope: `experiments/wavefield_video/`. No commits.
Read against `train_compare.py` (OneCycleLR L724, `k_at` L733, train loop L756-817,
single-step eval + `copy_ratio` L831-864), `wfvideo.py` (`VideoPredictor` L440, zero-init
head L488-490, `WaveMix3D`/`_transfer` L200-219, `FusedMix` L363-424), `ssm_lite.py`,
`data.py` (`make_clip_batch` L47), and the two prior dives (`DEEP_DIVE_2_opus.md`,
`HARDENING_astra.md`).

---

## 0. TL;DR

1. **The new data invalidates the DEEP_DIVE_2 story.** The geometry control (radius 3.2,
   speed 2.3, `moving_frac` 0.09→0.55) still froze (copy_ratio 0.006). A majority-moving
   canvas that freezes anyway **rules out canvas-sparsity / objective dilution as the
   driver**, which is exactly why the `--resid-balanced` / `--motion-weighted` patches had
   _zero_ effect — they repair a leak that is not the cause.

2. **The freeze splits into two distinct failures that must not be averaged together:**
   - **K=1 arms → FREEZE** (copy_ratio ≈ 0.005, tail-MSE 0.016; "frozen is
     metric-optimal"). This is a _one-step-map_ failure.
   - **Ramp arms → DRIFT** (copy_ratio 0.116, tail 0.94). This is a _long-horizon /
     exposure-bias_ failure.
     These have **different root causes and different fixes.** Treating "the freeze" as one
     phenomenon is the trap.

3. **Is budget-per-token the driver? Partly — but the stated mechanism ("¼ the updates
   per token") is wrong, and the real mechanism is sharper.** The wave/SSM operators are
   spatially _shared_ (per-head kernels), so g32 feeds them **4× more** gradient samples
   per step, not fewer. What actually scales adversely with grid is (a) the number of
   **sequential optimizer steps** needed to climb out of the zero-init copy-last basin,
   coupled to (b) **`OneCycleLR(total_steps=a.steps)` annealing the LR to ≈0** before the
   climb finishes, plus two grid-fragile numerics: (c) **bf16 spectral precision** and
   (d) **AdamW weight decay dragging the zero-init head back toward copy-last**.

4. **Before the budget suite, run one cheaper control that may already be answered:** a
   **deterministic-future** run (no collisions, no kicks). If the future is a deterministic
   function of the 17-frame context, a _moving_ predictor achieves ≈0 loss — strictly below
   freeze — so **if it still freezes, freeze is provably NOT the loss optimum → it is a
   trainability failure** and the budget/schedule/precision suite applies. If your R10
   C-suite `_g32_K1_nocoll` arm was in the "ALL variants freeze" matrix, this is _already
   answered in favor of trainability_ — confirm and skip.

5. **Architecture pick:** build **hybrid gated fusion (`local_wave`) first.** It is already
   scaffolded (`FusedMix`, `wfvideo.py:363`), it gives the model an _easy_ motion path that
   does not depend on the hard-to-train global FFT operator (so it directly attacks the
   freeze), and it doubles as the decisive `local+wave` vs `local+ssm` experiment that
   HARDENING_astra names as the single most informative run. Conditional-λ and
   self-conditioned rollouts are ranked 2nd and 3rd and are **gated** on the freeze being
   resolved first.

---

## Part 1 — Freeze root-cause

### 1.1 What the new data proves and disproves

| Evidence                                                                  | What it kills / establishes                                                                                          |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Geometry control (moving_frac 0.09→0.55) **still froze**                  | Kills canvas-sparsity / `moving_frac` dilution as the driver. Kills the DEEP_DIVE_2 §1.1–1.2 mechanism as _primary_. |
| Objective patches (`resid-balanced`, `motion-weighted`) = **zero effect** | Confirms the above: they fix the diluting `0.25·resid` term, which is not what's binding.                            |
| K=1 arms freeze; ramp arms drift                                          | Two failures, two causes (§1.2).                                                                                     |
| Freeze survives at 4× radius/speed                                        | The cause is **grid/optimization-structural**, not scene-statistical.                                                |

So the suspect list is now short and mostly about _optimization and numerics at 4× sequence
length_, not about the loss's shape over pixels.

### 1.2 Two failures, separated

**K=1 FREEZE ("metric-optimal").** With `--rollout-loss 1` the loss is a single
`F.mse_loss(pred, tgt)` (L797). Two sub-hypotheses produce the same symptom:

- _(H-opt) Freeze is a genuine loss optimum._ With `--collisions` on and only 17 context
  frames, some one-step futures are aliased (a bounce the model can't infer). Under MSE the
  optimal predictor is the **conditional mean**, which for an ambiguous ball position is a
  blurred/attenuated frame ≈ copy-last. The brief's own "frozen is metric-optimal, tail
  0.016" is exactly this signature.
- _(H-train) Freeze is a trainability floor._ The zero-init residual head (L488-490) starts
  the model _exactly_ at copy-last; the escape gradient is weak/fragile and the model never
  leaves the basin within the high-LR window.

**These are disambiguated by one control** (§1.4, Arm 0-det): make the future deterministic.
Under H-opt, deterministic data has a ≈0-loss moving solution and the model will move. Under
H-train, it will still freeze. _This is the single most valuable experiment in this document
and it is nearly free._

**Ramp DRIFT.** With `--rollout-ramp`, K climbs 3→8 over the second half (`k_at`, L733-740);
multi-step terms feed the model's own **detached** predictions (L800). Far-horizon collision
positions are unpredictable, so those `e_move` gradients are near-zero-mean noise, and the
model settles on a low-motion compromise that _drifts_ (copy_ratio 0.116, tail 0.94). This is
**exposure bias**, and its fix is self-conditioned rollout training (Part 2, rank 3) — _not_
the same fix as the K=1 freeze.

### 1.3 Ranked suspects (with code-aware verdicts)

**Suspect A — MSE-mean-seeking under an unpredictable future (H-opt).** _Plausibility: high
if `nocoll` was never run; likely already ruled out if it was._ Mechanism above.
**Test:** Arm 0-det. **Fix if confirmed:** train/eval on the deterministic benchmark, or move
off pixel-MSE (perceptual/adversarial/mode-covering loss) — but first confirm, because the
geometry-big freeze hints the optimum isn't the whole story.

**Suspect B — sequential-step budget × OneCycleLR anneal (H-train, leading actionable).**
_Plausibility: high._ The framing "g32 gets ¼ the updates per token" is **mechanically
wrong**: the operator weights (`WaveMix3D` kernel params, `pi`/`po`, FFN) are shared across
all `H·W` cells and receive gradients from **all 1024** cells each step at g32 vs 256 at g16
— that's **4× more** samples per step, lower-variance gradients, _not_ undertraining of those
weights. What _does_ bind:

- Escaping the flat copy-last basin needs a number of **sequential** high-LR updates. More
  samples per step lower gradient _variance_ but do not add sequential steps.
- `OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)` (**L724**) welds the whole schedule to
  `--steps=2000`: `initial_lr = max_lr/25 = 1.2e-5`, peak at step 600 (`pct_start=0.3`),
  annealed to `max_lr/1e4 ≈ 3e-8` by step 2000. If the g32 climb is even slightly slower, the
  LR is already ≈0 before it completes → the model is **frozen into the basin by the
  schedule.**
  **Test:** Arms B1–B4 (§1.4) decouple _step count_ from _schedule shape_ from _samples/step_.
  **Fix:** floor `min_lr` / stretch `pct_start` / cosine-to-nonzero, and/or scale `--steps`
  with token count. Note `--resume` re-aligns OneCycleLR by replaying `sched.step()` and
  "requires same `--steps`" (L748) — any schedule change must be consistent across resume.

**Suspect C — bf16 spectral precision at 4× FFT size (H-train, novel).** _Plausibility:
medium, cheap to test, un-flagged by prior dives._ Training runs under
`autocast(dtype=bfloat16)` (L766). In `WaveMix3D.forward`, `pi(x)` executes in bf16, then
`h = h.float()` (**wfvideo.py:313**) upcasts — but upcasting _after_ a bf16 matmul does not
recover lost mantissa. The delta the model must learn is a **small, localized** perturbation
that lives in high-`k` spectral cells, computed as a small difference on top of a large DC/low
band. bf16 has ~8 mantissa bits; at g32 the FFT sums over 4× more elements and the
small-ball spectral content can be swamped, making the escape gradient underflow. The DC and
low bands (background) are represented fine — so the model preferentially learns the static
majority → copy-last. This mechanism is **grid-dependent by construction.**
**Test:** Arm B5 (fp32, autocast disabled). **Fix:** force the operator's FFT path to fp32
regardless of autocast (compute `pi` in fp32 for the mixer, not just `.float()` after).

**Suspect D — AdamW weight decay pulling the zero-init head back to copy-last (H-train).**
_Plausibility: medium._ `AdamW(..., weight_decay=0.01)` (L722-723) applies decoupled decay
`w ← w − lr·wd·w` to _all_ params including the residual head. The head starts at 0; as it
grows away from copy-last, WD resists. If the escape is slow, and once the LR is small but
nonzero, WD can dominate the tiny gradient and **drag the head back toward zero (copy-last)**.
Grid-coupled only via "g32 escape is slower." **Test:** Arm B6 (`weight_decay=0`, or exclude
the head/biases from WD — the standard no-decay-on-1D-params rule). **Fix:** exclude head +
norm + biases from weight decay.

**Ruled out (do not spend GPU-hours here):**

- **Normalization across 4× tokens.** `RMSNorm` reduces over the feature dim (`x.pow(2)
.mean(-1)`, **wfvideo.py:35**) — per-token, **grid-invariant**. Torch FFT default norm is
  `backward` (forward unnormalized, inverse ÷N), so the operator's per-cell gain is
  grid-invariant; the gate pools with `.mean(dim=(2,3,4))` (L306) — a mean, grid-invariant.
  No normalization term scales with token count.
- **Kernel init scaling at larger N.** Dispersion params are continuous in `k = 2π·fftfreq`
  (`_transfer`, **wfvideo.py:200-219**); a 1.15-cell advection is the _same_ `vx,vy` at both
  grids; `Cg = randn·n_modes^-0.5` is grid-independent. Poles are stable at DC (`alpha =
softplus(0.5) ≈ 0.97 → |λ| ≈ 0.38`). Confirmed already in DEEP_DIVE_2 §1.5. (The only
  per-position params that grow with grid are `FactorizedPosEmb.py/px` (**wfvideo.py:61-62**),
  32×dim vs 16×dim — negligible, and each row sees many samples/step.)
- **Objective dilution / `moving_frac`.** Killed by the geometry control.

### 1.4 The airtight budget-experiment suite

Design goal: **separate the four confounded axes** — (i) sequential step count, (ii)
OneCycleLR shape, (iii) samples per step, (iv) precision/WD — that the brief's single
"8000-step" run collapses into one knob. Everything below is **g32, wave/dispersion, K=1**
(to isolate the _freeze_, not the drift), collisions on unless noted, **one shared init seed**
first, then replicate survivors over 3 seeds. Shared base (from R10):

```
BASE="--kind wave --kernel-version dispersion --gate --local-fuse \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --causal --residual --linear-pad --collisions \
  --motion-loss --rollout-loss 1 \
  --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --out runs_dd3"
```

| Arm        | Change vs BASE                                                                        | Isolates                                              | Expected if suspect true                                                      |
| ---------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| **0-ctrl** | none, 2000 steps                                                                      | the freeze (control)                                  | copy_ratio ≈ 0.005                                                            |
| **0-det**  | `--no-collisions` (no kicks), 2000 steps                                              | H-opt vs H-train                                      | **moves** ⇒ H-opt (loss optimum); **freezes** ⇒ H-train (trainability)        |
| **B1**     | `--steps 8000` (OneCycle stretched to 8000)                                           | steps **+** schedule (the brief's run — _confounded_) | copy_ratio recovers                                                           |
| **B2**     | `--steps 8000` **+ constant LR** (patch: build schedule as `ConstantLR`/`LambdaLR≡1`) | steps _without_ schedule stretch                      | isolates pure step count                                                      |
| **B3**     | 2000 steps **+ constant LR = 3e-4** (no anneal)                                       | schedule anneal _without_ extra steps                 | **moves** ⇒ the anneal, not step count, is the killer (cheapest fix)          |
| **B4**     | 2000 steps, `--batch 64` (4× samples/step)                                            | samples/step _without_ extra steps                    | **moves** ⇒ gradient SNR/variance, fix = batch / grad-accum                   |
| **B5**     | 2000 steps, **fp32** (disable `autocast`)                                             | bf16 spectral precision                               | **moves** ⇒ Suspect C                                                         |
| **B6**     | 2000 steps, `weight_decay=0` (or head/norm/bias no-decay)                             | AdamW WD                                              | **moves** ⇒ Suspect D                                                         |
| **S1**     | **g16**, `--steps 500`                                                                | symmetric budget test                                 | **freezes** ⇒ starving g16 to ¼ steps reproduces the freeze (supports budget) |
| **S2**     | **g16**, `--steps 8000`                                                               | mirror sanity                                         | best copy_ratio (should not regress)                                          |

**Instrumentation (2-line diff, do this before running):** in the logging block (L820-824),
also record the **head gradient norm** and **copy_ratio** every 25 steps, e.g. after
`clip_grad_norm_` capture `head_gn = m.head.weight.grad.norm().item()` and a cheap
single-batch `copy_ratio`. The **trajectory of copy_ratio against the LR curve** is the
smoking gun:

- copy_ratio rises then collapses as LR decays → **Suspect B (schedule)**.
- copy_ratio never rises even at peak LR, head grad-norm ≈ 0 → **Suspect A/C** (no signal:
  optimum or underflow).
- copy_ratio never rises but head grad-norm is healthy → **basin / WD (D)**.

**Controls that make it airtight:**

- **Same init seed** across all arms; replicate only the winner(s) over 3 seeds
  (`--seed 0/1/2`) — `--seed` seeds both weights and clip offsets (L664), so this is a clean
  independent-run sweep with the eval seeds fixed at 90000.
- **Fixed eval seeds** (already 90000) and fixed `--eval-chunk` so the metric is invariant.
- **Normalize tail-MSE by the frozen-frame baseline at the same grid/geometry** before any
  cross-arm MSE comparison (DEEP_DIVE_2 §3.5); lead with `copy_ratio` (a ratio, grid-robust),
  `divergence_horizon`, and short-horizon (frames 1–8) MSE, **not** the saturated 256-mean.
- **One knob per arm.** The brief's "8000 steps" is B1 = B2 ⊕ B3 (steps ⊕ schedule); never
  read B1 alone.

**Decision table (what each outcome buys you):**

| Outcome              | Root cause                                                                     | Fix                                                                        |
| -------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| 0-det moves          | H-opt: MSE-mean under ambiguity                                                | deterministic benchmark, or non-mean-seeking loss                          |
| B3 moves             | LR anneal kills the escape                                                     | floor `min_lr`, longer `pct_start`, cosine-to-nonzero — _no extra compute_ |
| B2 moves, B3 doesn't | genuine sequential-step budget                                                 | scale `--steps` with tokens/grid                                           |
| B4 moves             | gradient variance/SNR                                                          | larger batch or gradient accumulation                                      |
| B5 moves             | bf16 spectral underflow                                                        | fp32 operator FFT path                                                     |
| B6 moves             | WD drags head to zero                                                          | exclude head/norm/bias from WD                                             |
| **nothing moves**    | **structural**: global FFT operator cannot emit localized motion deltas at 32² | → **architecture change required (Part 2, hybrid)**                        |

This suite costs ~7 short g32 runs + one 8000-step run + two g16 runs, and every arm changes
exactly one axis, so the confound the brief flagged is fully broken.

---

## Part 2 — Next architecture change: rank the three, pick one

The freeze data reshapes HARDENING_astra's ranking. HARDENING ranked self-conditioned
rollouts #1 (general exposure-bias argument), hybrid #2, conditional-λ #3. **Given a one-step
map that will not move, the priority inverts:** you cannot fix long-horizon rollout of a model
that produces copy-last, and you should not add conditional dynamics to an operator you can't
yet train. Fix the one-step _motion_ first.

### Rank 1 — Hybrid gated fusion (`local_wave`) — **BUILD THIS FIRST**

**Why first (three reasons):**

1. **It directly attacks the freeze.** The local path (`AttnMix`, causal, window = T) can
   compute constant-velocity extrapolation from the last two frames _trivially_ — an easy,
   well-conditioned function with a strong gradient to the head. So the zero-init head escapes
   copy-last through the local path even if the global FFT operator's escape gradient is weak
   or bf16-underflowed. The gate then learns to route. This is the one architecture change
   that plausibly unfreezes without first winning the budget fight.
2. **It's already scaffolded** (`FusedMix`, `wfvideo.py:363-424`; wired through `--fuse
local_wave`, streaming `step`, and `state_bytes`). Build cost ≈ verification + a training
   run, not new architecture.
3. **It doubles as the decisive experiment.** `local+wave` vs `local+ssm` vs `local-only` vs
   `wave-only` is _the_ experiment HARDENING_astra says matters most. If the gate collapses to
   all-local (`g→1`), that cheaply falsifies the wave thesis; if `local+wave` beats
   `local+ssm` on divergence-horizon-per-byte, you have the headline result.

**Exact spec (matches the existing code):**

- `h = g·h_local + (1−g)·h_global`, `g = sigmoid(Linear([h_local; h_global]))` (`_fuse`,
  wfvideo.py:400-402). Keep as-is.
- `local = AttnMix(causal=True)` (window = T=17); `glob = WaveMix3D(dispersion, causal_time,
linear_pad)`. Keep as-is.
- Run: `--kind wave --fuse local_wave --kernel-version dispersion --causal --residual
--linear-pad`, plus the **freeze fix that Part 1 identifies** (e.g. `--no-collisions` or
  the schedule/precision fix). K=1 first.
- **Add one diagnostic:** log the mean gate value `g` per layer (mean of `sigmoid(...)`). This
  is the "is wave pulling its weight" readout. If mean `g > 0.9`, wave is dead weight.
- **Matched-params comparison is mandatory:** run `local+wave`, `local+ssm`, `local-only`
  (attn causal), `wave-only`, `ssm-only`, all at `--target-params 4000000`. **Kill criterion:**
  if `local+wave` does not beat `local+ssm` on `divergence_horizon` (and on
  `div_horizon/KB_state`, already computed at L934) in the occlusion probe, the _structured_
  wave formulation is not earning its keep.

### Rank 2 — Conditional λ_t (content-adaptive, bounded)

**Why second, not first:** it adds a controller and optimization difficulty to an operator you
cannot yet train past copy-last — likely to _worsen_ the freeze. It is the right move _after_
the hybrid proves the wave state is worth conditioning. HARDENING_astra explicitly gates it
behind #2–#7.

**Exact spec (bounded residuals around the stable prior — do NOT go to unrestricted Mamba):**
Keep the per-(head, mode) base params `a0, a1, vx, vy, beta, Bg, Cg` (`WaveMix3D.__init__`,
wfvideo.py:126-133). Add a tiny controller producing **bounded deltas** from a pooled content
summary (reuse the gate's pooled feature, `h.mean(dim=(2,3,4))`, one MLP per param group):

```
Δα_t = τ_α · tanh(Wα · pooled)     # τ_α ≈ 0.5  (keeps softplus argument near its prior)
ΔΩ_t = τ_Ω · tanh(WΩ · pooled)     # τ_Ω ≈ 0.25·π  (bounded phase nudge)
ΔB_t = τ_B · tanh(WB · pooled)
alpha = softplus(a0 + a1·|k| + Δα_t);  Omega = (vx+ΔB-shaped)... ; lam = exp(-alpha)·exp(iΩ)
```

Constraints that keep it defensible and stable: (i) `tanh`-bounded so `|λ|<1` is preserved by
construction (softplus argument stays positive); (ii) controller is **≤1% of params**; (iii)
zero-init the controller's final linear so training **starts exactly at the fixed-kernel
prior** (same discipline as the residual head). **Streaming caveat:** the current `step()`
(wfvideo.py:270) builds `lam` from static params; a content-dependent `lam_t` must be threaded
through `step()` from the current frame's pooled feature — a real (but contained) change.
**Kill criterion (HARDENING):** if adaptive params improve short-horizon reconstruction but do
**not** increase divergence horizon on ≥3 hard tasks, delete the controller.

### Rank 3 — Self-conditioned rollouts (scheduled sampling)

**Why third _now_ (it is #1 in general):** it fixes the **ramp/drift** failure, not the K=1
**freeze**. Rolling out a copy-last model just yields copy-last — no new signal — so it cannot
unfreeze a static one-step map. It becomes rank 1 the moment the one-step map moves (after the
hybrid), because _then_ compounding error is the binding constraint.

**Exact spec (cheap Self-Forcing, extends the existing K-step loss):** the loop already feeds
**detached** predictions for `k≥1` (L800). Add two things:

1. **Corrupt the initial context window**, not just the horizon: with probability `p(step)`,
   replace the last `j` context frames with the model's own generated frames (a short warmup
   rollout) before computing the K-step loss — this exposes training to the _generated_
   distribution the eval sees.
2. **Schedule `p`:** `0→30%` steps `p=0`; `30–60%` `p=0.25`; `60–100%` `p∈[0.5,0.75]`
   (HARDENING's ladder). Keep `K` modest (do not stack with the aggressive 3→8 ramp that
   _caused_ the drift — cap `K≤4` while `p>0`).
   **Kill criterion:** if longer self-conditioning yields no divergence-horizon gain at 256/1024
   frames, the apparent teacher-forced advantage was an artifact — a legitimately important
   negative result (HARDENING Rank 1 kill).

### The pick, stated plainly

**Build hybrid gated fusion (`local_wave`) first**, because it is (a) the only change that
plausibly _unfreezes_ the one-step map without first winning the budget fight, (b) already
implemented, and (c) the setup for the `local+wave` vs `local+ssm` occlusion experiment that
is the project's highest-information run. Sequence: **freeze-diagnosis suite (Part 1) →
hybrid → self-conditioned rollouts → conditional-λ**, each gated by the prior's kill criteria.

---

## Appendix — one-command reproducibility hook (HARDENING §B, worth doing alongside)

Every arm above should emit an immutable manifest (git_sha, seed, params, schedule, precision,
wd, grid, geometry, eval seeds). `train_compare.py` already writes most of this to
`result_*.json` (L894-919); add `git rev-parse HEAD`, `torch.__version__`, the LR-schedule
descriptor, and `weight_decay` so a run is self-describing. This is what lets the decision
table in §1.4 be read months later without re-deriving which knob each tag changed.
