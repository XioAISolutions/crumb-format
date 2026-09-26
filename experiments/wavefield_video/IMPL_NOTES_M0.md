# IMPL_NOTES_M0 — `crumb_coherence` plugin (M0 build)

Implements `PLUGIN_SPEC_crumb_coherence.md` §1–§7 M0. Training-free, pipeline-
agnostic low-band spectral coherence for long video. **No commits.**

## What was built

```
crumb_coherence/
  __init__.py            # public API re-exports + honest scope docstring (§0)
  core.py                # §6 functions + SpectralCoherenceEngine, StatsEMAEngine
  adapters.py            # wrap_generator (§6)
  metrics.py             # lowband_trajectory_variance, delta_e_vs_ref,
                         #   highfreq_ssim, temporal_flicker (§6)
  scripts/run_m0.py      # M0 synthetic-drift harness (§7 M0)
  tests/test_core.py     # §6 invariants + state-byte claims
  out/                   # side-by-side mp4 lands here
run_m0_smoke.sh          # CPU-only: unit tests + M0 harness
```

## Exact commands (CPU-only, no commits)

From `experiments/wavefield_video/`:

```bash
# everything (unit tests then M0 harness), forced CPU:
bash run_m0_smoke.sh

# just the invariant/state-byte unit tests:
python crumb_coherence/tests/test_core.py
#   or: python -m pytest crumb_coherence/tests -q

# just the M0 synthetic-drift harness (defaults: 48 frames, 64x64):
python crumb_coherence/scripts/run_m0.py
python crumb_coherence/scripts/run_m0.py --frames 60 --grid 96 --alpha 0.9
python crumb_coherence/scripts/run_m0.py --no-video        # skip mp4
```

The mp4 (`crumb_coherence/out/m0_sidebyside.mp4`, columns _control | drifted |
corrected-magnitude_) is written only if `imageio` is importable; otherwise it
falls back to a `.gif`, and if neither works the run still completes and prints
all metrics. `CUDA_VISIBLE_DEVICES=""` in the smoke script pins CPU.

## Design decisions worth knowing

### The load-bearing §3.2 choice — `target_lowband` (core.py)

This is _the_ function the spec flags as "where your judgment should shape the
plugin". It decides which low-band content is **drift** (anchor it) vs **motion**
(leave it free):

- `complex` — anchor the full complex low band. Strongest drift removal, but
  freezes low-band **phase** = large-scale motion/pans. Ships as an option only.
- `magnitude` _(default)_ — anchor the low-band **magnitude** (exposure / color-
  energy = the drift), keep each frame's own **phase** (where things are = the
  motion). Implemented as `|anchor| · (Xlo/|Xlo|)`: re-inject the anchored
  magnitude onto the live phase. This is what lets drift die while the scene
  still moves.
- `dc_only` — anchor only the DC cell (per-channel global mean). A rolling
  exposure/white-balance lock; the 24-byte MVP floor. In this mode the engine
  keeps a **1×1** anchor box (hence `state_bytes(512,512) == 24`, exactly the
  spec value).

### FFT frequency layout (the subtle bit)

`rfft2` gives `[C,H,W//2+1]`: width holds only non-negative freqs (low band is a
plain `[:kw]` slice) but height holds the full spectrum with negative freqs
wrapped to the end. So `low_band_box` / `put_low_band` `fftshift` the **height**
axis, take/write a **centered** block, and `ifftshift` back. After the shift, DC
lives at box row `kh//2`, col `0` — that is where the Gaussian weight peaks and
where `dc_only` reads.

### Gaussian band weight (§3.3)

`gaussian_band_weight(kh, kw, sigma_k_frac, device, cutoff_frac=0.10)` — 1 at DC,
tapering smoothly to ~0 by the band edge (no Gibbs ringing). The spec's four
positional params are unchanged; `cutoff_frac` is threaded in as a trailing
keyword so `sigma_k_frac` keeps its physical meaning ("fraction of Nyquist"):
with `kh = cutoff_frac·H`, Nyquist is `kh/(2·cutoff_frac)` cells, so
`sigma_cells = sigma_k_frac·kh/(2·cutoff_frac)`.

### EMA anchor (§2 step 3)

First-value initialization (`anchor := Xlo` on frame 1) then a plain leaky
integrator `anchor ← ρ·anchor + (1−ρ)·Xlo`. Seeding with the first real value
removes the from-zero warmup bias that would otherwise need `1/(1−ρⁿ)`
correction, so no explicit debiasing is needed. This is exactly the
`z_t = λ·z_{t-1} + B·x_t` recurrence of `wfvideo.py` with `λ=ρ`, one mode per
low-k cell — the honest link to the research model.

### Cut guard (§2 step 6, §3.4)

`detect_cut` = relative low-band L2 change > `cut_thresh`. On a cut with
`reset_on_cut=True`: skip the blend (frame passes through) and reseed the anchor
to the new scene. With `reset_on_cut=False`: drop `ρ→0.5` for
`CUT_COOLDOWN_N=5` frames so the anchor adapts fast instead of dragging scene A
into scene B.

### Baseline (§5)

`StatsEMAEngine` — rolling EMA of per-channel mean/std matched into each frame
(AdaIN-in-time), no FFT, 24 bytes, **same interface**. Per the spec this is the
floor: the FFT engine only earns its complexity if it beats this on real
segments. The M0 harness runs both so you can see the gap (the FFT engine should
pull ahead specifically on the _spatial_ low-frequency field, which global
mean/std matching cannot touch).

## Invariants under test (`tests/test_core.py`)

1. **`alpha=0` ⇒ byte-identical no-op** — short-circuited before the FFT round-
   trip, so an A/B toggle is exact (`torch.equal`), spectral and stats.
2. **`process_segment` == `process_frame` looped** — the segment path _is_ the
   frame loop, so batched and streaming agree exactly.
3. **Cut guard triggers** — a hard scene change passes through and reseeds the
   anchor; a smooth exposure nudge does **not** false-trigger.
4. **`state_bytes` matches spec** — 512² magnitude = **62 424 B (~62 KB)**,
   256² = **16 224 B (~15 KB)**, `dc_only` = **24 B**, stats = 24 B; and the
   claimed count equals the real anchor tensor size.

## M0 harness (`scripts/run_m0.py`) — what it does

1. `data.make_clip_batch(1, T, G, G, collisions=True)` → a coherent moving-ball
   **control** clip (CPU, small grid).
2. `inject_drift`: a known **exposure ramp** (gain 1.0→1.25, bias 0→0.08) plus a
   **wandering low-frequency color field** (a 1-cycle spatial pattern, per-
   channel amplitude = color cast, phase drifting over time). Both are low-band
   and leave ball _positions_ untouched.
3. Run **stats-EMA**, **spectral dc_only / magnitude / complex**.
4. Report per engine: drift-reduction % (`lowband_trajectory_variance` and
   `delta_e_vs_ref`), `highfreq_ssim(drifted, corrected)`, **motion %**
   (corrected centroid path length ÷ clean-control path length), state bytes,
   and a PASS/fail verdict against the M0 gate (**≥60 % drift removed,
   ssim ≥ 0.98, motion within 5 %**).

**Reading the table.** Expected shape of the result (confirm on the real run):
`magnitude` should remove both the exposure ramp _and_ the spatial field and
clear the 60 % bar while keeping ssim ≥ 0.98 and motion ≈ 100 %. `dc_only` and
`stats-EMA` remove the exposure/DC drift but leave the spatial field, so they
should reduce drift **less** — this is the evidence the FFT engine earns its
complexity (§5). At 64×64 the balls (σ≈1.6 px) are largely _high_-frequency, so
even `complex` may not visibly freeze them; if `complex` shows `motion% << 100`
that is the §3.2 freeze failure the `magnitude` default avoids. Report whatever
the run actually prints — do not assume the numbers.

## Verification status

⚠️ **Not yet executed here** — this sandbox gates `python`; every run returned
"requires approval". The code is written and reviewed but the M0 numbers and the
unit-test pass are **pending an approved run of `bash run_m0_smoke.sh`**. Run it
(or approve the call) to get the actual verdict, then paste the table into
`RESULTS` if you want it recorded.

## Honest scope (README/§0)

Fixes **global / low-band drift** (exposure, color balance, contrast, large-
scale layout). Does **not** fix object-identity drift, texture crawl, or semantic
incoherence — those are high-frequency and/or need the generator's own memory.
