# Implementation notes — v3 streaming + hardening

Scope: everything lives in `experiments/wavefield_video/`. Not committed to git.
Builds on the v2 suite (see `IMPL_NOTES_V2.md`). **All v2 defaults are unchanged** —
every v3 flag defaults to a no-op, so `run_smoke.sh` / `run_v2.sh` behave exactly as before.

**Design stance.** v2 built the dispersion kernel as a frequency-domain temporal
transfer and _documented_ that it is the transfer of a causal recurrence, deferring the
streaming form. v3 makes that recurrence real (`step()`), then proves the streaming
rollout scales and hardens the training loop for large runs.

---

## Change-by-change

### (1) Recurrent `step()` for the dispersion wave kernel — `wfvideo.WaveMix3D`

New `init_state(B, device)` + `step(x_t, state)` implement, per head, per oscillator
mode `m`, and per spatial-frequency cell `k=(kx,ky)`:

```
z_t^m(k) = lam_m(k) · z_{t-1}^m(k) + Bg_m · x̂_t(k)      # x̂ = fft2(pi(x_t))
out_hat(k) = Σ_m Cg_m · z_t^m(k)
out = to_out( Re( iFFT2(out_hat) )[:H,:W] )              # to_out = self.po
```

`lam_m(k) = exp(-α_m(k))·exp(i·Ω_m(k))`, with `α = softplus(a0 + a1·|k|)` and
`Ω = vx·kx + vy·ky + β·|k|` — **the identical formula `_transfer()` uses in `forward()`**
(factored into `_dispersion_lam()`; `_transfer` itself is left byte-for-byte untouched so
the trained forward numerics can't drift). `Bg`, `Cg` are the same real per-mode input/
output gains. So `lam/B/C` reproduce the forward math exactly.

**Why they match to a tolerance, not to the bit.** The transfer
`1/(1 − λe^{−iω})` is the DTFT of the _infinite_ causal impulse response `λ^n` (n≥0).
`forward()` samples it at `L = 2T` DFT bins, which time-aliases that response to
`λ^n/(1 − λ^L)` plus an anti-causal wrap term. Both correction terms are `O(|λ|^{T+1})`,
and since the poles are stable (`|λ|<1`) with `L = 2T`, they vanish fast. `step()` is the
clean, un-aliased recurrence; `forward()` is its FFT approximation — exactly the
`SSMLite.forward()`↔`SSMLite.step()` relationship, now for the wave arm.

**Verification (in `sanity_check.py`):**

- **7b** `dispersion forward == step()` — unroll `step()` over `T` frames, compare to
  `forward()`, `max|Δ| < 1e-3`. Uses its own `T=16` so `|λ|^{T+1}` (≈`0.38^17 ≈ 1e-7` at
  default poles) sits far under tol; at the harness's global `T=5` the aliasing is ~3e-3
  and would be a misleading "fail", so the check owns its size.
- **8b** `wave stream_step == forward (T frames)` — the whole-model claim: ingest `T`
  context frames through `VideoPredictor.stream_step` and match `forward(context)` to
  `1e-3`. Per-frame `step==forward` composes through every block because each block's
  streamed input sequence equals its windowed input sequence, and `pt[t]` is added for
  `t<T` in both paths.

`VideoPredictor.stream_init/stream_step` (wave-only) thread the per-layer `WaveMix3D`
state through `embed → posemb → blocks → norm → head → residual`. For `t ≥ T` the
temporal position embedding is clamped to `pt[T-1]` (documented steady-state; the
recurrence itself is time-invariant, so this only shifts the additive input phase).
`step()` does **not** apply `gate` (a non-causal global `(t,y,x)` pool) or `local_fuse`;
those are v2 forward-only add-ons and out of scope for the streaming path.

### (2) `--stream-test` — `train_compare.stream_test`

A long streaming rollout (`--stream-frames`, default **1024**; `--stream-batch`, default 8)
that **builds the model, optionally loads `--resume` weights, runs the benchmark, writes
JSON, and exits without training**.

- **wave** → the O(1)-in-T `step()` recurrence: warm the state on the `T` context frames,
  then autoregress, feeding each prediction back. Constant state, no window, no recompute.
- **attn / ssm** → windowed rollout: keep the last `T` frames and re-run the **full**
  `forward()` every frame (the only way an attention window can "stream").

Every 128 frames it logs `{frame, peak_gb, seg_fps, cum_fps}` — `peak_gb` is the
**per-segment** peak (`reset_peak_memory_stats()` after each checkpoint) so a flat wave
curve vs the window's re-compute cost is directly visible. Output:
`stream_{kind}{tag}.json`. Requires `--kernel-version dispersion` for wave; warns (does not
fail) if `--gate/--local-fuse` are set, since streaming ignores them.

### (3) `--auto-batch` — OOM recovery in `train_compare.main`

Both the train step and the eval passes are wrapped in a `1 try + 3 retry` loop. On a CUDA
`RuntimeError` containing "out of memory" it **zeros grads, drops references,
`torch.cuda.empty_cache()`, halves the batch, and retries** (train: `cur_batch`; single-step
eval: `eb`; rollout eval: `a.eval_seeds`). The shrink persists for later steps (no thrash
back up). `empty_cache()` is also called **between eval passes** and before the rollout eval.
Non-OOM errors and a 4th failure re-raise unchanged. On CPU or without the flag it is inert.
The scheduler steps exactly once per _successful_ optimizer step, so a shrink never desyncs
OneCycleLR.

### (4) `--save-every N` + `--resume PATH` — `train_compare.main`

`--save-every N` writes `ckpt_{kind}{tag}.pt = {state, opt, step}` every `N` steps (and once
more at the end). `--resume PATH` restores `state` + `opt`, sets `start_step = ckpt["step"]`,
fast-forwards OneCycleLR by stepping it `start_step` times, and continues the loop from
`start_step+1`. **Caveat:** OneCycleLR is parameterized by `total_steps`, so a resumed run
must keep the same `--steps` as the original for the LR schedule to line up (the smoke
deliberately extends `--steps` to show the loop restarts mid-way, accepting the schedule
offset). `--stream-test --resume` loads only `state`.

### (5) This file + `run_smoke_v3.sh`.

---

## Smoke tests — STATUS: BLOCKED (session hard-gates `python3`)

Same gate as prior rounds: every `python3` invocation is denied, so **no runtime output was
produced and none is fabricated.** The suite is fully wired; one approval runs all of it:

```bash
cd experiments/wavefield_video
bash run_smoke_v3.sh          # sanity (incl. new step checks), stream-test x3, ckpt roundtrip
# or: PY=/path/to/venv/bin/python bash run_smoke_v3.sh
```

`run_smoke_v3.sh` runs, in order: (1) `sanity_check.py` — now including
`dispersion forward == step()` and `wave stream_step == forward`; (2) `--stream-test` for
wave/attn/ssm at 384 frames → `smoke_v3/stream_*.json`; (3)+(4) a 10-step
`--save-every 5` run then a `--resume` continuation to step 20, with `--auto-batch` on.

**When you approve Python, I'll run it and paste the real numbers here.**

---

## Static self-review (couldn't execute)

Traced by hand; highest-confidence concerns first.

- **`step()` broadcast shapes.** State `[B, n_modes, nh, Hp, Wp, dh]`; `lam` broadcast as
  `[1, n_modes, nh, Hp, Wp, 1]`; gains `Bg/Cg` are `[nh, n_modes] → .t()[None,:,:,None,None,None]`
  = `[1, n_modes, nh, 1, 1, 1]`. `x̂_t[:,None]` is `[B,1,nh,Hp,Wp,dh]`. All align on the mode
  axis (dim 1). The token (y,x)-order in/out `permute/reshape` mirrors `forward()` exactly.
- **`.real` placement.** Taken after `iFFT2`, matching `_wave_dispersion` (`Ys.real`), not
  before — a common off-by-one that would drop the imaginary leakage differently.
- **Aliasing tolerance is real, not cosmetic.** Documented above; the sanity check picks a
  `T` where it is `~1e-7`. If someone reuses `step()` with near-unit poles (`α→0`) at small
  `T`, the `1e-3` agreement can loosen — this is inherent to the L=2T truncation, not a bug.
- **Streaming posemb past T** is an approximation (`pt[T-1]`); the sanity check only asserts
  equivalence for `t<T`, which is the honest, provable claim. The >T rollout is a cost
  benchmark, not a numerical-equivalence claim.
- **`--auto-batch` alters gradient statistics** on a shrink (smaller effective batch for
  subsequent steps). Intended: finishing at a smaller batch beats crashing. It never grows
  back, so no oscillation.
- **`torch.load` uses the default `weights_only=False`** for `--resume`. These are
  self-produced local checkpoints (trusted). If loading untrusted checkpoints, pass
  `weights_only=True` (works for the tensor-only `state`; optimizer state may need
  allowlisting on some torch versions).

## Open risks / caveats

- **Unverified at runtime.** Logic reviewed on paper only. First suspects if a smoke fails:
  the 6-D broadcast in `step()` and the `fft2(..., s=(Hp,Wp), dim=(2,3))` on the per-frame
  5-D tensor (both traced, believed correct).
- **`--stream-test` memory on GPU** grows with `Hp·Wp·dh·n_modes` state for wave; still O(1)
  in T (the whole point). At grid 32+ the state is a few hundred MB — fine, and it does not
  grow over the 1024 frames.
- **Windowed attn fps** in the stream test recomputes O(T²·N) attention every frame; that is
  the intended contrast, not an unfair penalty — it is the actual cost of streaming with a
  window.
