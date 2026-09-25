# R10 — scale-aware objective + eval hardening

Answers the DEEP_DIVE_2 diagnoses of the **g32 copy_ratio collapse** (0.719 → 0.115
under the _same_ hyperparameters) with three optional objective/geometry knobs plus
an eval-memory fix. Every knob is **default-OFF / no-op**; the default pixel path is
byte-identical except that rollout-eval now sub-batches its seeds (metric-preserving).

Edits: `train_compare.py`, `data.py`. No commits.

## What shipped

### (1) `--resid-balanced` — Opus leak #1

`motion_balanced_loss`'s third term is `MSE(pred-last, target-last)`, which is
algebraically `MSE(pred, target)` over the **full frame** (the `last` cancels). So it
re-imports the copy-last shortcut the 0.5/0.5 moving/static split was built to kill,
and its motion content dilutes ∝ `moving_frac` — ~4× weaker at g32. `--resid-balanced`
restricts that term to the moving pixels (`E[moving]`), making it grid-invariant.
Requires `--motion-loss`. Default off ⇒ full-frame residual (unchanged).

### (2) `--motion-weighted` — Astra #1 (soft variant)

Replaces the **hard** moving-pixel mean with a soft per-pixel weighting by motion
magnitude: `w = clamp(|target-last| / (mean|target-last| + 1e-6), 1, w_max)`
(channel-mean Δ), then `e_move = mean(w · E)`. Static pixels clamp to `w=1`
(unchanged); the fastest pixels weigh up to `w_max` (`--motion-w-max`, default 10).
Smoother than a binary mask when the mask is unstable at low `moving_frac`. `e_stat`
and the residual term are untouched. Requires `--motion-loss`. Default off.

> `--resid-balanced` and `--motion-weighted` compose (you can set both); they modify
> orthogonal terms of the objective.

### (3) `--radius` / `--speed` — geometry control (Astra exp 2)

`RADIUS`/`SPEED` are **grid-cell** constants, so g16→g32 shrinks the ball's _relative_
size (÷2) and speed (÷2) — g32 is a different motion regime, not "g16 at higher res".
These flags pass ball geometry straight to the clip generator; defaults equal
`data.RADIUS` (1.6) / `data.SPEED` (1.15) so behavior is unchanged. The collision
`CONTACT` distance scales with `radius` (still `2r`), and the rollout-eval divergence
thresholds scale too (`div_thresh_px = radius`, `id_survival_tol = 2·radius`) so the
geometry control isn't confounded by a fixed 1.6px threshold. To reproduce g16
geometry at g32: `--radius 3.2 --speed 2.30`.

### (4) `--eval-chunk` (default 4) — the OOM fix that killed 3 runs

`rollout_eval` now forwards at most `--eval-chunk` seeds through the model at once,
running each chunk as an independent R-frame rollout and combining: per-frame MSE is a
seed-count-weighted mean (exact), centroid errors are concatenated along the seed axis.
Result is **metric-equivalent** to the old all-at-once path (proven in the smoke,
chunk=1/2/S agree) with a bounded activation footprint. The auto-batch OOM handler now
shrinks `eval_chunk` first (keeps all seeds ⇒ metric unchanged) and only drops
`eval_seeds` once chunk hits 1.

New JSON fields: `resid_balanced`, `motion_weighted`, `motion_w_max`, `radius`,
`speed`, `eval_chunk` (also inside the rollout block).

## Smoke (CPU, session-gated python; one approval runs all of it)

```bash
cd experiments/wavefield_video
bash run_smoke_r10.sh
#   or: PY=/path/to/venv/bin/python bash run_smoke_r10.sh
```

Proves: loss unit tests (default byte-identical + both knobs), each flag wired with no
NaN, `--radius` grows the moving footprint, chunked rollout == all-at-once, and the
default path is deterministic with documented defaults.

## Retune suite (GPU) — exact commands

Shared base = the collapsing g32 config from `run_deep1.sh`. `--eval-chunk 4` is on by
default now; it's written explicitly below so the OOM fix is obvious in the logs.

```bash
P=python   # or your venv python
BASE="--kind wave --kernel-version dispersion --gate --local-fuse \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --causal --residual --linear-pad --collisions --motion-loss \
  --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --save-every 500 --out runs_deep"
```

### B — balance / drop the residual leak (Opus exp B; ramp OFF to isolate it)

```bash
# B1: residual restricted to moving pixels, K fixed at 1
$P train_compare.py $BASE --rollout-loss 1 --resid-balanced        --tag _g32_balresid
# B2: soft motion weighting instead of the hard mask, K fixed at 1
$P train_compare.py $BASE --rollout-loss 1 --motion-weighted       --tag _g32_motionw
# B3: both leaks addressed together
$P train_compare.py $BASE --rollout-loss 1 --resid-balanced --motion-weighted --tag _g32_bal_mw
# B0: control — same config, both knobs OFF, K=1 (isolates the objective change)
$P train_compare.py $BASE --rollout-loss 1                         --tag _g32_ctrl_K1
```

Read `copy_ratio`: if B1/B2/B3 climb materially above B0 (say >0.4), the residual /
mask-dilution leak is a real contributor.

### C — K-ramp × collisions, 2×2 (Opus exp C; schedule isolation)

```bash
CBASE="--kind wave --kernel-version dispersion --gate --local-fuse \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --causal --residual --linear-pad --motion-loss \
  --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --save-every 500 --out runs_deep"
$P train_compare.py $CBASE --collisions --rollout-loss 1   --tag _g32_K1_coll
$P train_compare.py $CBASE --collisions --rollout-ramp     --tag _g32_ramp_coll
$P train_compare.py $CBASE              --rollout-loss 1   --tag _g32_K1_nocoll
$P train_compare.py $CBASE              --rollout-ramp     --tag _g32_ramp_nocoll
```

If `copy_ratio(K1) ≫ copy_ratio(ramp)` **only with collisions**, the
unpredictable-far-horizon → freeze mechanism is confirmed; if K1 helps regardless, the
ramp is simply too aggressive at fixed compute.

### Geometry — resolution-equivalent scene at g32 (Astra exp 2)

```bash
GBASE="--kind wave --kernel-version dispersion --gate --local-fuse \
  --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16 \
  --target-params 4000000 --causal --residual --linear-pad --collisions --motion-loss \
  --rollout-loss 1 --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch \
  --save-every 500 --out runs_deep"
# g32-A: current geometry (relatively small/slow ball) — the collapsing regime
$P train_compare.py $GBASE --radius 1.6 --speed 1.15 --tag _g32geo_small
# g32-B: g16-equivalent geometry (2× radius, 2× speed) — same normalized scene as g16
$P train_compare.py $GBASE --radius 3.2 --speed 2.30 --tag _g32geo_big
```

If g32-B recovers toward g16 behavior while g32-A stays collapsed, the collapse is
task/loss scaling, **not** wave-model inability at 32×32.

### Optional null control (Opus exp D) — undertraining vs objective

```bash
$P train_compare.py $BASE --rollout-ramp --steps 8000 --save-every 1000 --tag _g32_4xsteps
```

If `copy_ratio` recovers with 4× steps alone, much of the "collapse" is undertraining
(fixed budget on 4× the tokens) and B/C are polish rather than fixes — run this
alongside B.

## Reading the results (DEEP_DIVE_2 §3 caveats)

- Prefer `copy_ratio` (a ratio ⇒ grid-robust) and short-horizon / divergence-horizon
  reads over the 256-frame mean MSE, which saturates and can be beaten by a frozen
  frame at both grids.
- When comparing across grids or across `--radius`, normalize MSE (÷ frozen-frame or ÷
  scene variance) — absolute pixel MSE is not comparable when `moving_frac` changes.
