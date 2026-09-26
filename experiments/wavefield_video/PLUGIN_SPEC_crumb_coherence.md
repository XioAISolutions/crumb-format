# PLUGIN SPEC — `crumb_coherence`: a spectral coherence engine for 2–5 minute video

Answers `CONSULT_BRIEF_3.md` Q3 (the product ask). Scope: a **pipeline-agnostic, training-free
plugin** that gives _any_ segment/AR video generator low-frequency coherence over 2–5 minutes.
No commits. This spec is buildable stand-alone; it borrows the `z_t = λ·z_{t-1} + B·x_t`
recurrence idea from `wfvideo.py` but does **not** depend on the research model.

---

## 0. The thesis, and where it can break

**Claim.** Long AR/segment video drifts in its **low-frequency, slowly-varying content** —
global exposure, color balance, contrast, and large-scale layout — long before it loses
high-frequency detail. FreqForcing's result (low-band spectral anchoring fixes AR long-video
drift) says: hold the low band steady across segments and the video _reads_ as coherent even
when fine detail evolves. A tiny rolling state over low-band FFT coefficients + a soft blend
into each new segment can do this **outside the generator, in plain pixel/FFT space.**

**The single thing that must be true, or the whole plugin fails.** _The low band you anchor
must carry drift, not motion._ If you anchor the **full complex** low band you will freeze
camera pans, global motion, and legitimate scene evolution — you will have reinvented the g32
freeze in a plugin. **You must anchor the part of the low band that is drift (exposure/color
statistics; DC and near-DC magnitude) while leaving the part that is motion (low-band _phase_,
which encodes large-scale translation) free.** This is the load-bearing design decision (§3.2).

**Honest scope.** This plugin fixes **global/low-band drift**. It does **not** fix object
identity drift, texture crawl, or semantic incoherence — those live in high frequencies and/or
require the generator's own memory. Say so in the README; a plugin that over-claims will be
judged on failures it was never meant to fix.

---

## 1. Interface

```python
class SpectralCoherenceEngine:
    def __init__(self,
                 cutoff_frac: float = 0.10,   # low-band half-extent, as a fraction of Nyquist
                 sigma_k_frac: float = 0.06,  # Gaussian falloff scale (fraction of Nyquist)
                 alpha: float = 0.5,          # blend strength toward the anchor at DC
                 rho: float = 0.95,           # anchor EMA memory (λ of the k-space recurrence)
                 color_space: str = "ycbcr",  # decorrelate exposure (Y) from color (CbCr)
                 anchor_mode: str = "magnitude",  # {"magnitude","dc_only","complex"} — see §3.2
                 reset_on_cut: bool = True,
                 cut_thresh: float = 0.35):
        ...

    def init_state(self, H: int, W: int, device="cpu") -> "CoherenceState": ...
    def process_frame(self, frame: "Tensor[3,H,W]", state) -> tuple["Tensor[3,H,W]", state]: ...
    def process_segment(self, frames: "Tensor[T,3,H,W]", state) -> tuple["Tensor[T,3,H,W]", state]: ...
    def reset(self, state) -> state: ...     # hard reset the anchor (e.g. on a known cut)
    def state_bytes(self, H, W) -> int: ...  # for the "state = tiny" claim
```

- **Input:** frames or segments in `[0,1]`, shape `[3,H,W]` / `[T,3,H,W]`. Streaming
  (`process_frame`) and batched-segment (`process_segment`) both supported; a generator that
  emits 2–5 s clips calls `process_segment` per clip and threads `state`.
- **Output:** corrected frames, same shape/range. Identity when `alpha=0` (so it is a no-op
  you can A/B toggle).
- **State (`CoherenceState`):** a dataclass holding the rolling low-band anchor and counters
  — **the only persistent memory, and it is tiny:**

```python
@dataclass
class CoherenceState:
    anchor: Tensor        # complex, [C, kh, kw]  — the EMA'd low-band box (kh,kw ≈ cutoff·H,W)
    prev_lowband: Tensor  # complex, [C, kh, kw]  — for cut detection
    n_seen: int           # frames folded into the anchor (for EMA warmup / bias-correction)
    warm: bool
```

`state_bytes` = `C · kh · kw · 8` (complex64). For `cutoff_frac=0.1` at 512×512, `kh=kw≈51`,
`C=3` → **~62 KB**; at 256×256, ~15 KB; DC-only mode, ~24 bytes. **Horizon-independent** —
this is the same O(1)-in-T story as the research recurrence, and it is the plugin's headline.

---

## 2. Algorithm

Per frame `x` (after RGB→color_space):

1. **Transform.** `X = rfft2(x)` per channel → complex `[C, H, W//2+1]`.
2. **Extract low band.** `Xlo = low_band_box(X, cutoff_frac)` → `[C, kh, kw]` (the block of
   lowest spatial frequencies; a _box_, then softened by the Gaussian weight so there is no
   hard cut).
3. **Anchor (rolling k-space recurrence).** On the first frame, `anchor = Xlo`. Thereafter, EMA
   — which is literally `z_t = ρ·z_{t-1} + (1−ρ)·B·x_t` with scalar `λ=ρ` per low-`k` cell (a
   degenerate 1-mode wave/SSM state; this is the honest link to `wfvideo.py`):
   `anchor ← ρ·anchor + (1−ρ)·A(Xlo)` where `A(·)` selects the _driftful_ part per
   `anchor_mode` (§3.2).
4. **Blend the new frame toward the anchor**, Gaussian-weighted so the pull is strong at DC and
   fades to zero by the cutoff (no Gibbs ringing):
   `w = gaussian_band_weight(sigma_k_frac)` (`[kh,kw]`, `w(0)=1`, →0 at band edge)
   `Xlo_new = (1 − alpha·w) ⊙ Xlo + alpha·w ⊙ target(Xlo, anchor)`
   where `target(·)` re-injects the driftful anchor while **preserving the frame's own
   low-band phase** when `anchor_mode="magnitude"` (§3.2).
5. **Write back + invert.** Put `Xlo_new` into `X`, `irfft2`, color_space→RGB, `clamp(0,1)`.
6. **Cut guard.** If `‖Xlo − prev_lowband‖ / ‖prev_lowband‖ > cut_thresh` → a scene cut: skip
   the blend for this frame and either hard-`reset` the anchor or set `rho→0.5` for a few
   frames so the anchor adapts fast instead of fighting the new scene.

Cost: two rFFTs per frame over the full image but the recurrence/blend touch only the low-band
box → dominated by the FFT, ~ms/frame on GPU, negligible state.

---

## 3. Design decisions that make or break it

### 3.1 Operate in a decorrelated color space

Fix **exposure** on luminance (Y) and **color cast** on chroma (Cb,Cr) independently. Doing the
blend in RGB couples channels and shifts hue as you correct brightness. `ycbcr` is the cheap
default; `oklab` is better if available.

### 3.2 What to anchor — the load-bearing choice (`anchor_mode`)

This is the decision that separates "fixes drift" from "freezes the video":

- `"complex"` — anchor the full complex low band. **Strongest drift removal, but freezes
  large-scale motion / pans.** Only safe for near-static scenes. Ships as an option, not the
  default.
- `"magnitude"` _(default)_ — anchor the low-band **magnitude spectrum**, keep the frame's own
  **phase**. Magnitude carries exposure/contrast/color-energy (the drift); phase carries
  _where things are_ (the motion). This fixes drift while letting the scene move. This is the
  right default for real generated video.
- `"dc_only"` — anchor just the DC term (per-channel global mean) + first ring. Cheapest,
  most conservative; equivalent to a rolling exposure/white-balance lock. Great MVP floor.

> **This is where your judgment should shape the plugin.** The `target(Xlo, anchor)` function
> in step 4 encodes _exactly which spectral content is drift vs motion_ for your content. The
> spec ships `magnitude` as the default, but the crossover (how much phase to touch, whether to
> also stabilize the lowest phase ring for a locked-camera product vs a moving-camera product)
> is a product decision. Implement `target()` deliberately and A/B it against `dc_only` — do
> not accept the default blindly.

### 3.3 Gaussian, not a box

A hard band cut produces ringing (Gibbs) at edges. `gaussian_band_weight` makes the correction
taper smoothly to zero by the cutoff — the "Gaussian blend" the brief asked for, applied in
`k`-space where it belongs.

### 3.4 Adaptive `rho` / `alpha`

Fixed `rho` fights legitimate slow scene evolution and can accumulate a cast. Make the anchor
adapt when the low-band residual is large (fast pan/cut) and hold when it is small (drift).
Simple rule: `rho_eff = rho` normally, `→0.5` for N frames after a cut trigger.

---

## 4. Failure modes (and the mitigation each forces into the design)

| Failure                                  | Cause                                                     | Mitigation (already in the interface)                                   |
| ---------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Video freezes / pans locked**          | `anchor_mode="complex"` freezes low-band phase (= motion) | default `"magnitude"`; never ship `complex` as default                  |
| **Ghosting / color bleed at scene cuts** | anchor from scene A dragged into scene B                  | `reset_on_cut`, `cut_thresh`, adaptive `rho`                            |
| **Washed-out / hazy, low contrast**      | `alpha` too high or band too wide                         | narrow `cutoff_frac`, modest `alpha`, Gaussian taper                    |
| **Ringing at strong edges**              | hard band cut                                             | `gaussian_band_weight` (not a box mask)                                 |
| **Slow color cast accumulation**         | biased EMA over many frames                               | periodic re-anchor to a reference frame; bias-correct EMA during warmup |
| **Hue shift while fixing exposure**      | RGB channel coupling                                      | `color_space="ycbcr"`/`oklab`                                           |
| **"It didn't fix anything"**             | the drift is high-freq (identity/texture), out of band    | scope honestly; this plugin is low-band only                            |

---

## 5. Is there a better plugin shape? (tear-apart)

**Baseline you must beat before shipping the FFT engine:** a **rolling global-statistics
matcher** — EMA of per-channel mean/std (or a rolling histogram) matched into each new frame
(AdaIN-in-time). It is `dc_only` plus variance, needs _no FFT_, ~24 bytes of state, and removes
most exposure/white-balance drift. **If the FFT low-band engine does not measurably beat this
stats-EMA baseline on real segments, ship the baseline.** The FFT version earns its complexity
only by anchoring **large-scale spatial structure** (layout, vignette, gradient) beyond global
stats — verify that it does.

Two shapes to reject for this product: (a) a **learned adapter** — violates training-free /
pipeline-agnostic; (b) **optical-flow warping between segments** — expensive, brittle at cuts,
and solves motion continuity (a different problem) not low-band drift.

So the recommended shape is: **stats-EMA baseline as the floor, spectral engine as the
upgrade, both behind the same `process_segment` interface** so you can A/B them and the
generator never knows which is running.

---

## 6. Concrete build list (classes / functions)

```python
# crumb_coherence/core.py
def rgb_to_ycbcr(x): ...          # [C,H,W] in [0,1]  (and inverse)
def ycbcr_to_rgb(x): ...
def low_band_box(X, cutoff_frac): ...          # crop the rfft2 tensor to [C,kh,kw]
def put_low_band(X, Xlo): ...                  # write the box back (out-of-place)
def gaussian_band_weight(kh, kw, sigma_k_frac, device): ...   # [kh,kw], 1 at DC
def ema_update(anchor, x, rho, n_seen): ...    # bias-corrected during warmup
def target_lowband(Xlo, anchor, anchor_mode):  # <-- the load-bearing §3.2 function
    # magnitude: keep angle(Xlo), swap |·| toward |anchor|; dc_only: only [...,0,0]; complex: anchor
    ...
def detect_cut(Xlo, prev_lowband, thresh): ... # -> bool

class CoherenceState: ...                       # dataclass in §1
class SpectralCoherenceEngine: ...              # interface in §1; process_frame orchestrates §2
class StatsEMAEngine: ...                       # the §5 baseline, SAME interface (for A/B)

# crumb_coherence/adapters.py
def wrap_generator(gen_segment_fn, engine, state):
    """Decorate any `gen_segment_fn() -> [T,3,H,W]` so each emitted segment is passed through
    `engine.process_segment` with threaded state. Pipeline-agnostic entry point."""

# crumb_coherence/metrics.py
def lowband_trajectory_variance(frames, cutoff_frac): ...   # drift magnitude over time
def delta_e_vs_ref(frames, ref_idx=0): ...                  # color drift (CIEDE2000 on means)
def highfreq_ssim(frames_in, frames_out): ...               # did we harm detail? (must stay high)
def temporal_flicker(frames): ...                           # short-time frame-diff energy
```

**Two invariants worth a unit test:** (i) `alpha=0` ⇒ output byte-identical to input (true
no-op toggle); (ii) `process_segment` threaded frame-by-frame == `process_frame` looped (so the
batched and streaming paths agree — the same discipline `sanity_check.py` applies to the
research recurrence).

---

## 7. MVP milestones + kill criteria

**M0 — Synthetic drift (days).** Take a _real, coherent_ clip and inject known low-band drift:
a slow exposure ramp + a wandering low-frequency brightness/color field. Run the engine.

- **Success:** `lowband_trajectory_variance` and `delta_e_vs_ref` drop by ≥60% while
  `highfreq_ssim(in,out) ≥ 0.98` (detail preserved) and no motion is frozen (a moving synthetic
  object still moves the same distance).
- **Kill:** cannot remove _injected, known_ drift without harming high-freq SSIM, or freezes
  the moving object → the low-band-anchoring premise is wrong; stop.

**M1 — Real generated segments (1–2 weeks).** Chain an off-the-shelf 2–5 s segment generator to
~30–60 s and run with/without the engine (and vs the stats-EMA baseline).

- **Success:** drift metrics improve materially over _both_ off and stats-EMA, with no new
  visible artifacts (ghosting/wash-out); on a small blind A/B, viewers prefer engine-on.
- **Kill:** engine ≤ stats-EMA on drift → **ship stats-EMA, drop the FFT engine.** Or: any
  visible artifact that makes engine-off preferred → back to §3 knobs; if unfixable, kill.

**M2 — 5-minute demo (2–3 weeks).** One 2–5 min stitched video, side-by-side, plus VBench-Long
consistency dims (background/subject consistency, temporal flicker, motion smoothness).

- **Success:** clear human A/B preference for engine-on **and** VBench-Long
  background/subject-consistency up with dynamic-degree not down (i.e. coherence gained without
  motion lost). This is the shippable artifact for brainsnn.com.
- **Kill:** no human preference and no VBench improvement → the drift this fixes isn't what
  makes long video feel incoherent; pivot to identity/high-freq memory (a different plugin).

**Gate order:** M0 → M1 → M2, stop at the first kill. Do not build M2's demo harness before M1
beats the stats-EMA floor — that floor is the honest bar, and clearing it cheaply is the whole
point.
