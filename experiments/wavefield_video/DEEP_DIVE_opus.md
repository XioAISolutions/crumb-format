# DEEP DIVE — wave-field video, g16/g32 read + streaming plan

_Opus 4.8 · 2026-09-25 · one RTX 4090 · numbers from `runs_v2_box/result_*.json` +
`eeval_*.json` (clean re-run fps) + `RESULTS_BRIEF.md` for g32-in-progress._

Ground rules for reading below: **one-step** = `eval_mse` on held-out T=16 context →
frame 17. **rollout** = 256-frame autoregressive free-run from 16 unseen seeds.
`r = MSE(copy_last)/MSE(model)` (want > 1). Baselines fixed for all arms:
`zero 0.0426 · copy_last 0.00572 · const_vel 0.00439`.

---

## (a) What g16/g32 say about the architecture race

### g16 table (grid 16, T=16, D=128, 4L, 8H, 1.60M params ±0.02%, 3000 steps)

| arm         | kernel                | one-step mse | /copy_last | rollout r | rollout plateau |     fps |  peak GB | cent-err px | id-surv | div-horizon |
| ----------- | --------------------- | -----------: | ---------: | --------: | --------------: | ------: | -------: | ----------: | ------: | ----------: |
| **attn_a1** | —                     |  **0.00125** |  **0.218** |     0.041 |       **~0.18** |      64 | **0.59** |        6.06 |   0.125 |           0 |
| **wave_w1** | dispersion+gate+local |      0.00137 |      0.239 |     0.027 |           ~0.32 |     123 |     2.06 |    **5.90** |   0.104 |         0–1 |
| **wave_w0** | separable+gate+local  |      0.00158 |      0.276 |     0.032 |           ~0.42 |     238 |     0.65 |        6.33 |   0.125 |           0 |
| **ssm_s1**  | S4D-lite              |      0.00263 |      0.459 |     0.018 |           ~0.48 | **354** |     0.55 |        5.85 |   0.104 |           0 |

Five things the numbers actually decide, mapped to the reviews' kill/prove gates:

1. **Harness is fixed (Astra's invalidity gate CLEARED).** All four arms beat
   copy-last _and_ const_vel one-step (best learned 0.00125 vs const_vel 0.00439).
   The residual + zero-init head did its job; the v1 "loses to copy-last" failure is
   gone. Every arm is now a legitimate learner, so architecture comparison is valid.

2. **Dispersion > separable — CONFIRMED, w0 is dominated.** w1 beats w0 by 13%
   one-step (0.239 vs 0.276 /copy) and by a wide margin on the rollout plateau
   (0.32 vs 0.42). Separability was a real ceiling, exactly as both reviews argued
   (`ω_t = v·k` coupling matters). **w0 can be retired from future suites.**

3. **Wave bias beats generic SSM — hypothesis NOT killed.** w1 nearly **halves**
   ssm's one-step error (0.239 vs 0.459 /copy) and wins the rollout plateau
   (0.32 vs 0.48). The oscillatory/dispersion prior earns its keep over a vanilla
   diagonal SSM on this wave-like motion — this is the one genuinely pro-thesis result.

4. **…but the "<< attention memory" prove-criterion FAILS at g16.** w1 costs
   **2.06 GB vs attention's 0.59 GB — 3.5× _more_, the opposite of the thesis.** The
   L=2T temporal zero-pad × 2H×2W spatial zero-pad × complex64 spectrum is the
   activation-memory tax both reviews predicted. At N=16·16·16 attention's O(N²) is
   still cheap, so dense-forward dispersion is currently the _heavier_ operator. The
   thesis only survives if the **streaming recurrence** (§c) drops w1 below attention —
   which it should, because streaming memory is flat in T while attention's grows.

5. **Nobody sustains rollout — the metric that matters is unwon by all.**
   `divergence_horizon ≈ 0`, `r_persist < 0.05` (models 24–55× worse than freezing
   frame 0 over 256 frames), centroid error ~6px on a 16px grid, identity survival
   ~10% everywhere. Attention "wins" the rollout plateau (0.18) only by regressing to
   blur — it hedges lowest MSE while also losing the balls. **One-step skill does not
   transfer to 256-frame coherence for any architecture.** Trained rollout horizon was
   K=3; evaluated at 256. That 85× mismatch, not the mixer, dominates the divergence.

### g32 (grid 32, T=17, B16, 4M params) — partial

- **ssm_s1 COLLAPSES:** 0.99× copy-last (barely beats the trivial baseline), rollout
  0.0015→0.07 with no plateau. The cheap/fast SSM that looked fine at g16 **cannot fit
  the harder task at 4× the tokens.** This is the "degrades with N" signal Opus/Astra
  wanted — and it lands against the SSM.
- **attn + w1 g32 still running.** The decisive comparison is unresolved: _if w1 holds
  quality at g32 while ssm has already collapsed, wave wins the graceful-degradation
  criterion._ This is the single most important number still outstanding. Watch its
  peak GB — at g32 the dispersion spectrum is ~4× the g16 2.06 GB (~8 GB/forward
  region), which is why `--ckpt` is on and why OOM hardening (§b) gates the run.

### Race verdict (one paragraph)

Dispersion wave is **the best sub-quadratic arm** — beats SSM ~2× and separable ~13%,
sits within 10% of attention on one-step. But it currently **loses on memory** (3.5×
attention at g16) and, like everything else, **fails long-rollout coherence**
(horizon ≈ 0). The thesis is neither proved nor killed: it hinges entirely on two
things the current dense-forward g16 suite can't test — (1) does the streaming
recurrence flip w1's memory from 3.5× _worse_ to constant-in-T _better_, and (2) can a
proper rollout-loss curriculum lift divergence horizon off zero. Both are the 48h plan.

---

## (b) Ranked hardening + quality upgrades (payoff-first)

Payoff estimated against the specific g16 numbers above. H = hardening, Q = quality.

| #   | Change                                                                                 | Type          | Cost    | Expected payoff                                                                                                                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------------------- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **`--auto-batch`**: catch CUDA OOM → halve batch, `zero_grad`, `empty_cache`, retry ×3 | H             | ~30 LOC | **Unblocks g32/g64 entirely.** Today an OOM = dead run + lost 2–3 h. This is the prerequisite for every scale number, so ROI is effectively infinite until it exists.                                                                                                                |
| 2   | **`--save-every N` + `--resume`** ({state, opt, step})                                 | H             | ~40 LOC | w1 g16 = 7626 s (2.1 h); attn = 9931 s (2.8 h). A crash mid-suite currently loses hours. Makes the 48h plan survivable and lets long g32/g64 runs checkpoint.                                                                                                                        |
| 3   | **Streaming `step()` for dispersion** (§c)                                             | Q             | ~80 LOC | **Thesis-deciding.** Turns w1's 2.06 GB dense-forward into flat-in-T state (~few hundred MB), enabling 1024-frame runs where dense-forward OOMs (est. ~34 GB/layer at L=2048). This is what could flip criterion #4 above from FAIL to PASS.                                         |
| 4   | **Rollout-loss curriculum K=3→8, ramp weight, +steps**                                 | Q             | config  | Directly attacks the 85× train/eval horizon gap that is _the_ cause of `div_horizon=0`. Highest-leverage quality knob: expected to lift horizon off zero and lower the rollout plateau below the current 0.18–0.48 band. Try `L = L1 + Σ_{k=2..8} 0.25·Lk`, warm K up over training. |
| 5   | **Cache `_transfer()` / kernel FFT** (recomputed every forward, `wfvideo.py:227`)      | H/Q           | ~15 LOC | Depends only on 6·nh·n_modes params. Recomputing the [nh,L,Hp,Wp] complex transfer every step is negligible at g16, ~real at g32+. Free speed on w1's 123 fps and on train step time. Cache during eval; compute-once-per-forward during train.                                      |
| 6   | **Latent tokens** (frozen 4× patch/AE → 8×8 latent)                                    | Q             | ~1 day  | Both reviews' "big one." 480 frames × 64 latent tokens = 30k vs 480×256=123k pixel tokens. Cuts w1's memory ~4× _and_ extends usable T for the same budget — the only route past the 2–4 s dense-pixel wall to "minutes". Do **after** #3 proves the mixer streams.                  |
| 7   | **Causal gate** (running mean, not global `h.mean(2,3,4)`)                             | Q/correctness | ~10 LOC | Prerequisite for streaming equivalence: the current gate pools over _all_ T frames (`wfvideo.py:241`), which is non-causal and breaks `step()==forward()`. Must be a running/EMA pool over past frames only.                                                                         |
| 8   | **Retire w0; add const-vel-residual arm**                                              | Q/hygiene     | config  | w0 is dominated (§a.2); its slot is better spent on a stronger baseline (predict const-velocity delta then learn the correction).                                                                                                                                                    |
| 9   | Document/equalize **bf16 vs fp32-FFT** dtype asymmetry                                 | H             | note    | FFT path upcasts to fp32; attention SDPA runs bf16. w1's fps is measured fp32, attention's bf16 — not apples-to-apples. Flag in results, don't silently compare wall-clock.                                                                                                          |

**Order to execute:** 1 → 2 → 5 (cheap hardening, unblocks everything) → 7 → 3 (the
pivotal capability) → 4 (the coherence fix) → 6 → 8/9.

---

## (c) Streaming experiment design for the dispersion kernel

### State math (exact, matches `_wave_dispersion` / `_transfer`)

The dense-forward computes, per (head h, mode m), per spatial-frequency cell `k=(kx,ky)`
on the **zero-padded** grid `Hp=2H, Wp=2W`, the temporal transfer
`G(ω,k) = Σ_m C_m B_m / (1 − λ_m(k) e^{−iω})`. That transfer is _exactly_ the frequency
response of this causal recurrence — so streaming is a drop-in, not an approximation:

```
per frame t, per head, per mode m, on the padded spatial-frequency grid:
  X_t(k)         = fft2(h_t, s=(Hp,Wp))          # spatial FFT of frame t, per channel dh
  z_t^{(m)}(k)   = λ_m(k) · z_{t-1}^{(m)}(k) + B_m · X_t(k)      # complex recurrence
  Y_t(k)         = Σ_m C_m · z_t^{(m)}(k)
  y_t            = ifft2(Y_t).real[:H, :W]        # crop the zero-pad
with  λ_m(k) = exp(−softplus(a0+a1·|k|)) · exp(i·(vx·kx + vy·ky + β·|k|))   # |λ|<1
      z_{-1} = 0
```

- **State tensor:** `z ∈ ℂ[B, nh, n_modes, Hp, Wp, dh]`. Size = `B·nh·M·(2H)(2W)·dh`
  complex64, **independent of t and of total sequence length**. At g16
  (B=16, nh=8, M=4, Hp=Wp=32, dh=16): ≈ 16·8·4·32·32·16 ·8 B ≈ **1.1 GB** for all
  layers — vs the dense-forward's 2.06 GB that _also grows with L=2T_. At 1024 frames
  the dense L=2048 spectrum is ~34 GB/layer (OOM); streaming stays flat at ~1.1 GB.
- `B_m, C_m` are the existing per-head scalars `Bg[:,m], Cg[:,m]`; `λ_m(k)` is the
  existing `lam` from `_transfer`. **No new parameters** — `init_state()` + `step()`
  reuse the trained weights verbatim.

### Verification (the correctness gate — add to `sanity_check.py`)

**Test 1 — recurrence ≡ forward (the load-bearing one).**
With `z_{-1}=0`, unroll `step()` over the T training frames, collect `[y_0..y_{T-1}]`,
compare to `forward(h)`. The dense path zero-pads time to `L=2T` and takes `[:T]`, which
_is_ the finite causal linear convolution `y_t = Σ_{s≤t} g_{t−s} x_s` — so they must
match to FFT round-off. **Assert max-abs-diff < 1e-3** (fp32 FFT tolerance). Run per
kernel, per (H≠W) shape, at n_modes ∈ {2,4}.

**Test 2 — causality of `step()`.** Feed an impulse at frame t0; assert `y_t = 0` for
all `t < t0`. Confirms no future leak, independent of the FFT equivalence.

**Test 3 — stability / no blow-up.** Feed 4096 random frames; assert `‖z_t‖` and `‖y_t‖`
stay bounded (guaranteed by `|λ|<1` from softplus>0, but verify numerically — this is
the "unbounded rollout doesn't explode" claim).

**Gate caveat — the gate must be causal first (#b.7).** `forward`'s gate pools over all
T frames; a streaming `step()` can only see the past. Either (a) run Test 1 with
`--no-gate` to validate the _pure_ recurrence, then (b) validate the causal running-mean
gate against a _causal-forward_ reference, not the global-pool forward. Don't claim
`step()==forward()` while the global gate is live — they legitimately differ.

### Metrics (`--stream-test`, 1024 frames, log every 128)

Per 128-frame checkpoint, to JSON:

1. **peak GB** — must be **flat** across 128…1024 for w1-stream (the headline); attn/ssm
   run windowed rollout for reference (their memory is flat too but their _quality_
   window is fixed at T).
2. **fps** (streaming vs windowed) — expect w1-stream fps roughly constant; dense-forward
   fps would degrade with the growing L.
3. **rollout MSE + centroid-err + identity-survival curves** to 1024 — does coherence
   hold or diverge? (This is where #b.4's rollout curriculum should show up.)
4. **equivalence residual** on the first T frames (Test 1 value) logged alongside, so the
   run self-certifies that the streamed trajectory started from the validated operator.

**Prove:** w1-stream memory flat to 1024 frames **and** its divergence horizon ≥ attention's
windowed horizon → the frontier gap (constant-memory long coherence) is real.
**Kill:** if streaming memory is flat but horizon is still ~0 after the rollout
curriculum, the mixer coheres no better long than short — fold to latent-space + SSM.

---

## (d) Concrete 48h plan on one 4090

Wall-clock anchors from the g16 suite: w1 = 2.1 h, attn = 2.8 h, ssm/w0 ≈ 1 h each (B and
kernel differences). g32 ~2–3× heavier. Budget assumes serial GPU, coding overlaps
GPU-idle windows.

**H0–4 · Hardening + streaming code (GPU mostly idle).**
Land #b.1 `--auto-batch`, #b.2 `--save-every/--resume`, #b.5 transfer cache, #b.7 causal
gate, and the `init_state()/step()` recurrence. Wire Tests 1–3 into `sanity_check.py`.
Gate the whole plan on **Test 1 < 1e-3 with `--no-gate`** — nothing downstream is
trustworthy until the recurrence provably equals forward.

**H4–7 · Finish g32 race (GPU).** Let `run_g32.sh` complete attn + w1 (ssm already
collapsed to 0.99× copy-last). Record w1 peak GB and one-step /copy vs attn. **Decision:**
w1 ≤ attn quality _and_ ssm-collapsed ⇒ wave wins graceful-degradation; log it.

**H7–12 · Streaming test (GPU).** `--stream-test` 1024 frames: w1-stream vs attn/ssm
windowed, log memory/fps/coherence every 128. **The headline plot:** w1-stream flat
memory to 1024 vs dense-forward OOM curve. This is criterion #a.4 — prove or kill the
memory thesis here.

**H12–28 · Fix rollout coherence (GPU, the big retrain).** Retrain w1 g16 with #b.4:
rollout-loss K ramped 3→8, weight ramp, 6000 steps (~2× → ~4.5 h), `--save-every 500`.
Rerun attn g16 identically for a fair comparison. **Target:** `divergence_horizon > 0`
and rollout plateau below the current 0.18/0.32 band. If horizon stays 0 for _all_ arms,
the task is chaotic-unpredictable at 256 frames (collisions amplify) — reduce to
deterministic no-collision or shorten the coherence target; report honestly.

**H28–42 · Latent space (GPU + code).** Frozen 4× patch/AE → 8×8 latent grid (64
tokens/frame). Rerun w1/attn/ssm in latent at T=32–48. **Target:** w1 memory ~4× lower
_and_ usable rollout length ~4× longer for the same budget — the first real step toward
"minutes."

**H42–48 · Attention-wall demo + report.** g64 (N≈70k): show attention OOM/blowup on
the 4090 training batch while w1-stream survives at flat memory. Write results into
`RESULTS_BRIEF.md` + a plot: (i) g16/g32 quality table, (ii) streaming flat-memory curve,
(iii) divergence-horizon before/after rollout curriculum, (iv) g64 attention wall.

### Kill / prove checkpoints baked into the plan

- **After H4** — Test 1 ≥ 1e-3 → streaming is wrong; stop and debug before any GPU spend.
- **After H12** — w1-stream memory NOT flat → the recurrence isn't buying constant memory;
  the whole "minutes at constant memory" thesis is dead, pivot to latent+SSM.
- **After H28** — divergence horizon still 0 for w1 _and_ attn → coherence is task-limited,
  not architecture-limited; fix the task/objective before claiming any arm.
- **After H42** — w1 ≤ SSM on latent-space horizon with no memory edge → drop the wave
  framing, ship the SSM.

**One-line status going in:** dispersion wave is the best sub-quadratic arm (beats SSM
~2×, separable ~13%, ~within 10% of attention one-step) but today loses on memory (3.5×
attention) and, like everything, fails 256-frame coherence (horizon ≈ 0). The next 48h
exist to answer exactly two questions the g16 suite couldn't: does streaming flip the
memory sign, and does a rollout curriculum lift the horizon off zero.
