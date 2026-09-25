# DEEP DIVE 2 (opus) — g32 copy_ratio collapse, next experiments, protocol audit

Answers to `RESULTS_BRIEF_2.md`. Scope: `experiments/wavefield_video/`. No commits.
Read against `train_compare.py` (`motion_balanced_loss` L111, `k_at` L464, train loop
L487-544, single-step eval + `copy_ratio` L557-590), `data.py` (`moving_mask` L26,
`make_clip_batch` L38), `wfvideo.py` (`VideoPredictor` L360, `WaveMix3D` L95),
`render_rollout.py` (L297-353).

Both arms compared were run at **identical hyperparameters** — only `--grid` differs
(`run_deep1.sh` for w1g32b, its g16 sibling for w1r4): `dim 192 layers 6 heads 8
--target-params 4000000 --batch 16 --steps 2000 --frames 17 --collisions --causal
--residual --linear-pad --kernel-version dispersion --gate --local-fuse
--motion-loss --rollout-ramp`.

---

## TL;DR verdict

The collapse is **real** (not a metric artifact) and it is **scale-dependent, but
the scale that matters is the _benchmark's_, not the model's.** `RADIUS` and `SPEED`
are constants in grid-_cell_ units, so g32 is not "g16 at higher resolution" — it is
the _same 1.6px balls moving 1.15px/frame_ dropped into a 4× larger, ~4× emptier
canvas. `moving_frac` falls from 0.746 (g16) to an estimated ~0.19 (g32).

The motion-balanced objective was engineered to be invariant to `moving_frac`, and
its **first two terms genuinely are.** But three scale leaks survive, all pushing
toward freeze as the canvas empties, and none is compensated by a grid-aware term or
by more compute:

1. **The `0.25·resid` term is full-frame and un-balanced** — its motion content
   dilutes ∝ `moving_frac`, so ~25% of the objective becomes ~4× more
   copy-last-rewarding at g32 (the one term that re-imports the shortcut the split
   removed).
2. **Rollout-K ramp × emptier canvas rewards freezing across all 8 horizons** —
   with `--collisions` on, far-horizon ball positions are genuinely unpredictable,
   so those `e_move` gradients are near-zero-mean noise; freezing protects the
   (now-larger) static + residual mass across every horizon. The ramp _helps_ at
   g16 (freezing is catastrophic at 75% moving) and _hurts_ at g32.
3. **Fixed optimization budget for a 4× longer token sequence** — the model is
   per-_pixel_-token (`embed = Conv2d(3,dim,3)`, one token/pixel, L404), params
   matched to 4M at _both_ grids, so g32 runs 1024·T tokens through the same
   capacity in the same 2000 steps, starting exactly at copy-last (zero-init
   residual head, L396-398). It lands in the nearest basin: copy-last.

Two hypotheses from the brief are **ruled out** as primary drivers: threshold
scale-dependence (§1.4) and kernel capacity (§1.5). Keep them off the suspect list.

---

## Part 1 — Where the objective is scale-dependent

### 1.1 The physical setup is FOV-scaled, not resolution-scaled

`data.py`: `SPEED=1.15`, `RADIUS=1.6`, positions `rand()*maxs` with `maxs=grid-1`.
Everything the ball _does_ is expressed in grid cells. So going g16→g32:

- ball Gaussian σ stays 1.6 **cells**; per-frame displacement stays 1.15 **cells**;
- the ball's self-overlap frame-to-frame is identical → _per-ball motion difficulty
  is unchanged_;
- but the canvas area 4×'s, so 3 fixed-size balls occupy ~¼ the fraction, and the
  static background dominates.

`moving_frac` (fraction of pixels with |Δ|>0.05 between consecutive frames) is the
scalar that captures this. Measured g16 = 0.746 (brief). Estimate for g32: three
balls' >0.05 footprint ≈ 3·π(3σ)² ≈ 3·72 ≈ 216 px of 1024, plus motion rings, minus
overlap → **~0.19** (about ÷3.9, matching the area ratio). **Measure this first**
(Experiment A) — the whole diagnosis rests on it.

> This is the single most important reframing: at g16 the model that freezes is
> wrong on 3 of every 4 pixels; at g32 it is _right_ on ~4 of every 5. The scene
> stopped rewarding motion, and the objective didn't fully make up the difference.

### 1.2 Loss-at-copy-last decomposition (why the split isn't enough)

The model starts at δ=0 (copy-last). Write `M` = mean squared copy-last error over
_moving_ pixels (grid-invariant, same balls). At the copy-last init, with
`f = moving_frac`:

```
e_move = M                     (mean over moving pixels — f-independent)
e_stat ≈ 0                     (δ=0 copies static pixels perfectly)
resid  = mean_all (tgt-last)²  = f·M          (full-frame; nonzero only on moving px)

L_init = 0.5·M + 0.5·0 + 0.25·f·M = M·(0.5 + 0.25f)
   g16 (f=0.75): 0.69·M      resid share = 0.27
   g32 (f≈0.19): 0.55·M      resid share = 0.086
```

Two consequences:

- **The escape gradient is dominated by `e_move` at both grids, and `e_move` is
  f-independent** — so on paper the split _works_ and the collapse is _not_ a loss
  optimum (perfect prediction has L=0 at every grid). The collapse is therefore an
  **optimization/trainability** failure, not a landscape failure. That is why the
  same objective can be fine at g16 and fail at g32 without any term flipping sign.
- **The `0.25·resid` term is the only piece that carries `f` into the loss**, and it
  carries it the _wrong_ way: as `f→0`, `resid` becomes almost entirely "keep the
  static majority static" i.e. δ=0 i.e. copy-last. It is a full-frame MSE with no
  moving/static balancing — literally the shortcut the 0.5/0.5 split was built to
  kill, re-entering at 25% weight and strengthening (relatively) as the canvas
  empties. Confirm with Experiment B (drop/reweight `resid`).

### 1.3 The K-ramp reinforces freeze at g32 (and only at g32)

Training loss = Σₖ wₖ·motion_balanced(predₖ, tgtₖ, lastₖ, movingₖ), wₖ = 1.0 (k=0)
else 0.25, K ramped 3→8 over the second half (`k_at`, L464-471). At K=8 the multi-step
terms hold 1.75 of 2.75 total weight (64%), each fed the model's own **detached**
prediction (L527), each masked by the **GT** future mask `moving[:, frames+k-1]`
(good — the mask is honest).

With `--collisions` on, a ball's position 4-8 frames out depends on bounces the model
cannot infer from the context window → those far-horizon `e_move` targets are
effectively unpredictable, so their gradient is **near-zero-mean noise**. The
optimizer's only reliable way to reduce the _sum_ of 8 terms is to keep the rollout
close to a stable, mostly-static scene — i.e. **freeze**, which protects `e_stat` and
`resid` across all 8 horizons at once. Crucially this bites in the **second half**,
exactly when OneCycleLR (L455) is decaying, so there's little step size left to climb
back out.

At g16 this same mechanism is _benign_: freezing is catastrophic (75% of pixels
moving, every horizon), so the ramp does what it was meant to. **The ramp is not
wrong; it's grid-conditional.** Confirm with Experiment C (ramp off / K=1 at g32) and
by turning `--collisions` off (removes the unpredictable far-horizon noise).

### 1.4 Ruled out: threshold scale-dependence

`MOVE_THRESH=0.05` is an absolute intensity threshold, but a pixel's |Δ| as a ball
edge sweeps past is set by the Gaussian's intensity gradient, which is governed by σ
in _cells_ (1.6) and displacement in _cells_ (1.15) — **both grid-invariant.** So the
per-pixel classification "moving vs static" is identical in shape at g16 and g32; only
the _count_ of far-from-ball static pixels changes (and those are safely <0.05 at
both). The threshold changes `moving_frac` only through canvas area, which §1.1
already covers. Sweeping the threshold will not fix the collapse (though it's a cheap
confirmation — Experiment A can log frac at 2-3 thresholds).

### 1.5 Ruled out: kernel capacity

The dispersion kernel parameterizes advection as `Omega = vx·kx + vy·ky + β|k|`
(`_transfer`, L200-219) where `kx,ky = 2π·fftfreq(grid)` are radians/**cell**. A pure
translation by d cells is phase `−d·k`, so the _learned_ `vx,vy` that represent a
1.15-cell shift are the **same numbers at g16 and g32** — the advection
parameterization is grid-invariant in cell units. Likewise the separable damped-cosine
kernel's effective support ≈ 1/a_s **cells** is grid-invariant. g32 adds more spectral
bins, but the response is parameterized continuously in `k`, so there's no capacity
cliff. The model _can_ represent the g32 motion with the same weights; it just doesn't
get _trained_ into that region under fixed compute (§1.2, §1.6). Adding capacity is
therefore a weak lever vs. fixing the schedule/objective — deprioritize it.

### 1.6 The compute confound (state it honestly)

Per-pixel tokens quadruple (256→1024 per frame) at fixed params/steps/LR. Even with a
perfect objective, 2000 steps at batch 16 is ~4× less gradient-signal-per-token at
g32, and the residual init means every "unlearned" token defaults to copy-last. Some
of the 0.719→0.115 gap is plain undertraining, independent of the objective. Any
objective fix must be evaluated **at matched signal**, not matched steps — see
Experiment A/D controls.

---

## Part 2 — Next experiments, ranked by information value

Ranked by _bits about the mechanism per GPU-hour_. All are single-arm wave/dispersion
runs unless noted; reuse the `run_deep1.sh` config and change only the listed flags.

### A. (cheapest, do first) Instrument the scene, don't train — confirm the premise

Zero training. Prove `moving_frac`, the `e_move/e_stat/resid` split, and `L_init`
across grids. This validates the entire §1 chain before spending a GPU-hour, and its
numbers calibrate every fix below.

```bash
# CPU, seconds. Writes moving_frac + copy-last loss decomposition for g16/g32/g64.
python3 - <<'PY'
import torch; from data import make_clip_batch
for g in (16,32,64):
    out,mv = make_clip_batch(64, 8, g, g, seed=1234, return_moving=True, collisions=True)
    last,tgt = out[:,:-1], out[:,1:]
    E = (tgt-last).pow(2).mean(2); st=~mv
    em=E[mv].mean().item(); es=E[st].mean().item(); rf=E.mean().item()
    L=0.5*em+0.5*es+0.25*rf
    print(g, dict(frac=round(mv.float().mean().item(),4),
                  e_move=round(em,5), e_stat=round(es,7), resid=round(rf,5),
                  L_init=round(L,5), resid_share=round(0.25*rf/L,3)))
PY
```

**Expected if §1 is right:** `frac` ≈ 0.75/0.19/0.05; `e_move` roughly flat across
grids; `resid_share` falls 0.27→0.09→~0.02. **Kills or confirms the diagnosis for
free.** (Note: I could not run this here — the Python session is operator-gated.)

### B. (highest-value training run) Balance or drop the residual term

Directly tests leak #1 (§1.2). The `0.25·resid` full-frame term is the only un-balanced
piece; make it moving-only or remove it. This is a ~2-line change to
`motion_balanced_loss` (replace `F.mse_loss(pred-last, target-last)` with the same
restricted to `moving`, or gate it behind a flag). Suggested new flag `--resid-balanced`.

```bash
# g32, ramp OFF to isolate the resid effect from the K-ramp effect (C tests that)
train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse \
  --causal --residual --linear-pad --collisions --motion-loss \
  --rollout-loss 1 --resid-balanced \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --steps 2000 --eval-rollout 256 --eval-seeds 16 \
  --auto-batch --out runs_deep --tag _w1g32_balresid
```

**Reads:** if `copy_ratio` climbs materially (say >0.4) with resid balanced/off, leak
#1 is a real contributor. Cheap, decisive, one run.

### C. (isolates the schedule) K-ramp off + collisions off, 2×2

Tests leak #2 (§1.3). Four short runs at g32: {ramp on, K=1} × {collisions on, off}.

```bash
BASE="--kind wave --kernel-version dispersion --gate --local-fuse --causal \
  --residual --linear-pad --motion-loss --dim 192 --layers 6 --heads 8 \
  --grid 32 --frames 17 --batch 16 --target-params 4000000 --steps 2000 \
  --eval-rollout 256 --eval-seeds 16 --auto-batch --out runs_deep"
train_compare.py $BASE --collisions --rollout-loss 1            --tag _g32_K1_coll
train_compare.py $BASE --collisions --rollout-ramp             --tag _g32_ramp_coll
train_compare.py $BASE             --rollout-loss 1            --tag _g32_K1_nocoll
train_compare.py $BASE             --rollout-ramp             --tag _g32_ramp_nocoll
```

**Reads:** if `copy_ratio(K1) ≫ copy_ratio(ramp)` _only with collisions_, the
unpredictable-far-horizon → freeze mechanism is confirmed. If K1 helps regardless of
collisions, the ramp is simply too aggressive at fixed compute (still actionable:
ramp later / cap K by grid).

### D. (control for the confound) Matched-signal g32

Before crediting any objective fix, rule out §1.6. Give g32 the gradient budget g16
had — same tokens·steps means ~4× steps or ~4× batch at g32. One run:

```bash
train_compare.py --kind wave --kernel-version dispersion --gate --local-fuse \
  --causal --residual --linear-pad --collisions --motion-loss --rollout-ramp \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --steps 8000 --eval-rollout 256 --eval-seeds 16 \
  --auto-batch --save-every 1000 --out runs_deep --tag _w1g32_4xsteps
```

**Reads:** if `copy_ratio` recovers toward 0.7 with 4× steps alone, much of the
"collapse" is undertraining and the objective is _fine_ — which would reprioritize B/C
as polish rather than fixes. This is the honest null hypothesis; run it alongside B.

**Ranking rationale:** A is ~free and gates everything (do first). B is the single most
likely objective fix and is one run. D is the null control that tells you whether B/C
are even necessary — run B and D together. C is the richest but is 4 runs; run it only
if A shows the ramp-half is where copy_ratio dies (check the per-25-step `K`/`loss`
log tail: collapse coinciding with the K>3 half is the tell).

---

## Part 3 — Sanity-check of the protocol reconciliation

The brief's §Numbers makes two protocol claims. One is **sound**; the framing around
it **buries the headline** and **conflates three different baselines.**

### 3.1 SOUND: model-mean-MSE agreement across the two harnesses

`rollout_eval` (train, L129-171): 256-frame open-loop autoregressive rollout, 32 seeds,
seed family 90000, `mean_model = mean(step_mse)`. `render_rollout` (L314-352): 256-frame
open-loop autoregressive rollout, `mean_mse`. **Same protocol** (both feed own
predictions, both 256 horizons, both native-grid float32 after clamp). 0.0777 vs 0.0819
(~5%) across the two implementations is a legitimate agreement.

⚠️ One correction: the brief says "same seed family," but train uses seed **90000** and
render's default is **170001**. If they were genuinely different seeds, the agreement is
_better_ than claimed (cross-seed generalization, not a re-run of the same clip). Please
confirm which — as written the label is inconsistent with the code defaults.

### 3.2 FLAW: the frozen-frame baseline beats **both** arms, and that's the buried lede

`render_rollout` baseline `copy_last_mse` = **frozen seed frame `fixed_last` at every
one of the 256 horizons** (L333, `metric_notes` L382). The brief reports it as 0.043 —
**below** wave (0.074–0.082) and attn (0.146–0.183). So over 256 frames a _frozen
frame_ has lower mean pixel MSE than either model. "wave wins 5/5" is true **only
wave-vs-attn**; against the trivial frozen baseline **both arms lose.** The
reconciliation should state this outright rather than list "Frozen-frame baseline
0.043" as a footnote after "wave wins."

Why it happens (and why it's expected, not a bug): on a chaotic collision scene the
autoregressive rollout decorrelates from truth within a few frames, and the
residual/blur-prone model _fades_ balls, so its tail MSE exceeds that of a frozen frame
that at least preserves the correct ball count, colors, and contrast statistics. Which
leads to:

### 3.3 FLAW: 256-frame mean MSE is a saturated, low-discrimination metric

Once the rollout decorrelates, per-frame MSE saturates near the scene's pixel variance
for _every_ model, so `mean(step_mse)` is dominated by a tail where all arms look
equally wrong — and where a frozen frame can win (3.2). Two saturated numbers agreeing
(3.1) is therefore **weak evidence of correctness**: it mostly confirms both harnesses
compute the same saturated quantity. The discriminating signals live at **short
horizons** and in **identity**, not in the 256-mean:

- early-horizon MSE (say frames 1–8),
- `divergence_horizon` (centroid, `DIV_THRESH=RADIUS`),
- `identity_survival`.

The brief's own `div_h` = 1 (g16) and 0 (g32) already says the balls are effectively
lost almost immediately at both grids — consistent with "beaten by a frozen frame."
Lead the reconciliation with div_horizon / early-MSE; use 256-mean only as a coarse
cross-harness checksum.

### 3.4 FLAW: three different "persistence" baselines are listed as if comparable

The brief places these adjacently without distinguishing them:

- **render** `copy_last` = frozen-at-t0, 256-horizon (0.043) — a _hard, open-loop_
  baseline;
- **train** `rollout_eval.step_persist` = `MSE(prev_gt, gt)`, **one-step** GT
  persistence updated every frame (feeds `r_persist_over_model`) — a _much easier_
  baseline (only 1 frame apart);
- **train** single-step `copy_last` / `copy_ratio` denominator `‖tgt−last‖`, **one-step**
  at eval.

These are not the same number and shouldn't be read against each other. In particular
`r_persist_over_model` (built from one-step GT persistence) can look favorable while the
frozen-t0 256-horizon baseline (0.043) says the opposite. Report each with its horizon
explicitly.

### 3.5 FLAW: cross-grid absolute-MSE comparison ("tail 0.11 vs 0.94") is confounded

Absolute pixel MSE is not comparable across grids because scene pixel variance scales
with `moving_frac` and canvas emptiness (§1.1). g16 tail 0.11 vs g32 tail 0.94 mixes a
real quality difference with a scale-of-the-metric difference. Normalize before
comparing — e.g. tail MSE ÷ frozen-frame MSE at the same grid, or ÷ scene variance — so
the ratio is dimensionless. (This also makes the `copy_ratio` metric your most
grid-robust single scalar, since it's already a ratio; lean on it over raw MSE for
cross-grid claims.)

### 3.6 Net

- Keep: the two-harness model-MSE agreement is a valid implementation checksum.
- Fix the framing: **both arms are beaten by a frozen frame over 256 frames**; wave's
  win is only over attn. Say so.
- Re-anchor the reconciliation on short-horizon MSE + divergence_horizon +
  identity_survival, report each baseline with its horizon, and normalize any
  cross-grid MSE comparison. Confirm the seed label in 3.1.

```

```
