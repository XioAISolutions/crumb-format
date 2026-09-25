# Implementation notes — wave/attn/ssm blocker fixes + upgrades

Scope: everything below lives in `experiments/wavefield_video/`. **All v1 behavior is
preserved as the default** — with no new flags, `wave` and `attn` reproduce the original
runs (bit-for-bit on square grids; see "v1 equivalence" below). Not committed to git.

## What changed, file by file

### `wfvideo.py`

- **Per-axis kernel bug fixed** (`WaveMix3D`). v1 built `ds = arange(W) - W//2` and reused
  it for _both_ spatial axes, so any non-square grid (`H != W`) silently broadcast a
  W-length kernel against an H-length FFT axis. Now three separate buffers: `dt` from `T`,
  `dy` from `H`, `dx` from `W`. The spatial damping/frequency params (`a_s,w_s,p_s`) are
  still shared between y and x (isotropic, as in v1), so on square grids `dy == dx == v1's
ds` and the result is numerically identical.
- **`causal_time` flag.** Zeroes the future-reaching half of the temporal kernel. With the
  FFT convolution semantics `out[t] = Σ_u kc[u]·h[t-u]` (`kc = ifftshift(k)`), a tap at
  `dt>0` reads a **past** frame `h[t-dt]` and a tap at `dt<0` reads a **future** frame
  `h[t+|dt|]`. So causal masking keeps `dt >= 0` and zeroes `dt < 0` — `kernels_1d()` does
  `kt *= (dt >= 0)`. (Note: `kernels2d_smoke.py`'s demo comment labels `dt>0` as "future",
  which is the opposite of the convolution sign; the version here is the one that actually
  makes the last-frame read-out autoregressive. It's a one-character flip — `>= 0` ↔
  `<= 0` — if you want the other convention.)
- **`linear_time` flag.** Zero-padded (linear) convolution over the time axis instead of
  the circular FFT conv, so there is no wraparound between frame 0 and frame T-1. Space
  stays circular. Implemented in `_time_conv`: place the kernel at `ifftshift` indices in a
  length-`L` buffer (`L = 2T-1` for linear, `L = T` for circular), FFT along time (which
  zero-pads), multiply, crop to `T`. The `dt.long() % L` index map reproduces `ifftshift`
  exactly at `L = T` and spreads negative-lag taps to the buffer tail (with zero-pad
  between) at `L = 2T-1`.
  - **Important coupling:** `causal_time` alone is _not_ fully causal — the circular conv
    still wraps _past_ taps to future frames (`out[0]` with a `dt=+1` tap reads `h[T-1]`).
    True autoregressive causality needs `causal_time` **and** `linear_time`. The two flags
    are independent (as requested); `train_compare.py --causal` sets **both** for the wave
    arm, which is the correct combination.
  - When both flags are off, `forward` takes the **exact v1 code path** (joint `rfftn` over
    `(T,H,W)`), so the default is unchanged. The separable space-then-time path only runs
    when a flag is set.
- **`FactorizedPosEmb`** (new module). Learned separable `(t,y,x)` tables summed and
  broadcast-added: `(T + H + W)·dim` params instead of attention's flat `(T·H·W)·dim`
  table. Shared by **all** arms for a fair comparison.
- **`AttnMix` reworked.** Signature now `(dim, n_heads, T, H, W, causal=False, flat_pos=True)`.
  - `causal`: temporal causal mask — token in frame `t` attends only to frames `<= t`
    (spatial attention within a frame stays full). Mask = `frame[j] <= frame[i]` over the
    `(t,y,x)`-flattened token grid, passed to `scaled_dot_product_attention`.
  - `flat_pos`: keeps v1's flat learned position table by default. When the model uses the
    shared `FactorizedPosEmb`, this is turned off so position isn't double-counted (the
    review's fairness fix — v1 gave attention a huge dedicated pos table and wave none).
- **`Block` / `mix_dim` cleanup.** The fragile `mix_dim()` helper (which read `pi.out_features`
  or fell back to `po.in_features` "by luck", flagged in the review) is gone; `Block` now
  takes `dim` explicitly.
- **`VideoPredictor`** gains `kind="ssm"`, `causal=False`, `posemb=False`. Wiring:
  `wave` → `WaveMix3D(causal_time=causal, linear_time=causal)`; `attn` →
  `AttnMix(causal=causal, flat_pos=not posemb)`; `ssm` → `SSMLite(causal=causal)`.

### `ssm_lite.py` (new)

S4D-style diagonal state-space mixing block, **same interface** as the others (`pi`/`po`
Linear in/out, `forward: [B,N,dim]->[B,N,dim]`, reshapes internally to `[B,nh,T,H,W,dh]`,
causal over time, spatial cells treated as independent channels).

- Diagonal poles `A = exp(-softplus(a_log) + i·a_im)` — strictly inside the unit circle,
  shared per head over `d_state=16` modes. Input map `B` per head/state; output map `C`
  per head/channel/state; skip `D`.
- Closed-form kernel `k[c,n] = Re(Σ_s C·B·A^n)` — a sum of damped complex exponentials, the
  generic-learned cousin of the wave arm's hand-built damped cosines. This is the _fair_
  long-context competitor the review asks for.
- **Training** uses a zero-padded FFT convolution (`L=2T`), which is causal by construction
  (kernel defined only for `n>=0`).
- **`step()`** implements the equivalent O(1)-state recurrence `x_s = A·x_s + B·u`,
  `y = Re(C·x_s) + D·u`, for constant-memory streaming rollout of arbitrary length. Verified
  algebraically to equal the conv (`sanity_check.py` asserts `forward == step()`).
- Param budget is dominated by `pi`/`po` (`2·dim²`), same order as wave; the SSM tensors add
  `~ nh·(dh+2)·d_state`, much smaller.

### `data.py`

- **`kicks` flag** (default `False` → v1 exact, no extra RNG draws). When on, every
  `kick_every` (default 20) frames each ball gets a Gaussian velocity kick (`kick_scale`,
  default = `speed`), breaking determinism so a fixed kernel can't memorize the whole
  trajectory — the harder regime for measuring rollout coherence.

### `train_compare.py`

- `--kind` now accepts `ssm`.
- New flags: `--causal`, `--posemb`, `--kicks`, `--centroid-tol` (default 2.0 px). All
  default off → v1 behavior. Flags recorded in the result JSON.
- **Centroid-localization accuracy** added to the drift/rollout eval: `centroids()` computes
  the brightness-weighted `(x,y)` centroid of predicted vs ground-truth frames;
  `centroid_acc()` counts the fraction of (valid) frames whose predicted centroid lands
  within `tol` px of GT. Reported as `centroid_acc` (overall) and `centroid_acc_step`
  (per rollout step) alongside the existing `drift` MSE curve. `kicks` is threaded through
  every `make_clip_batch` call (train, eval, drift).

## v1 equivalence (why defaults are safe)

- Wave default path is the original `rfftn((T,H,W))` code, untouched.
- Per-axis kernel change is transparent on square grids (`dy == dx == old ds`).
- Attention default keeps its flat pos table and no mask.
- `data.py` draws identical RNG unless `kicks=True`.
- New result-JSON keys are additive.

## Smoke tests — STATUS: NOT RUN (blocked)

The intended CPU smoke (grid 8, frames 5, 20 steps) for all three kinds — plus a
`--causal --posemb --kicks` wave run — is wired in `run_smoke.sh`, and a correctness
harness in `sanity_check.py`. **I could not execute them: this session hard-gates
`python3` execution and the approval was declined, so there is no runtime output to
report and I will not fabricate any.** To run:

```bash
cd experiments/wavefield_video
bash run_smoke.sh            # sanity_check.py + train_compare.py smoke for wave/attn/ssm
# or, if you use a specific interpreter:
PY=/path/to/venv/bin/python bash run_smoke.sh
```

`sanity_check.py` checks (all designed to print `OK`/`FAIL`, no training):

1. WaveMix3D default runs on a **non-square** grid (H≠W) — would have crashed pre-fix.
2. `causal_time` zeroes the `dt<0` taps.
3. `linear_time` and `causal+linear` produce finite output.
4. AttnMix `causal`: blasting the last frame's input leaves earlier-frame outputs unchanged
   (no future leak).
5. `SSMLite.forward()` == unrolled `step()` recurrence (tol 1e-3).
6. End-to-end `VideoPredictor` for wave/attn/ssm with `causal+posemb` returns `[B,3,H,W]`
   finite output.

`run_smoke.sh` then writes `smoke2/result_<kind>.json` (+ strip PNGs) with `eval_mse`,
`drift`, `centroid_acc`, and the new flag record.

## Remaining risks / caveats

- **Unverified at runtime.** The logic above is reviewed by hand and the einsum/index maps
  checked on paper, but nothing was executed here. Run `sanity_check.py` first — it is the
  fast gate. Highest-risk spots if something fails: `torch.view_as_complex` contiguity in
  `SSMLite`, and complex-base `**` (guarded by a float exponent).
- **`causal_time` without `linear_time` is only partially causal** (circular wrap of past
  taps). Documented above; use `--causal` (sets both) for real autoregressive runs.
- **bf16/FFT dtype asymmetry unchanged.** The FFT path still upcasts to fp32 (`h.float()`)
  while attention SDPA runs bf16 on CUDA — so wall-clock comparisons remain not-quite
  apples-to-apples on dtype (a pre-existing caveat the review raised; not in scope here).
- **SSM `step()` cost.** Streaming state is `[B, H*W, nh, dh, d_state]` complex — O(1) in T
  (the point) but O(spatial·channels·state) per step; fine for these grids, watch it at
  large H·W.
- **Centroid metric is single-blob.** It uses one global brightness-weighted centroid, so
  with 3 balls it measures the centroid of the _ensemble_, not per-ball matching — a coarse
  localization proxy, as the review framed it. Tighten later with per-ball assignment if
  needed.
- **Kernel recomputed every forward** (v1 behavior kept). Cheap at these sizes; cache for
  large grids.

```

```
