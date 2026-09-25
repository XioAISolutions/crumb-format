# Implementation notes — v4 objective fix (Astra #2 + Opus #4)

Scope: everything lives in `experiments/wavefield_video/`. Not committed to git.
Builds on v3 (see `IMPL_NOTES_V3.md`). **All prior defaults are unchanged** — every
v4 flag defaults to a no-op, so `run_smoke.sh` / `run_v2.sh` / `run_smoke_v3.sh`
behave exactly as before. `copy_ratio` is the one always-on addition (a reported
metric, not a behavior change).

**Why this round.** `DEEP_DIVE_astra.md` §1: the g32 SSM "collapsing to copy-last"
is at least as much a statement about the _objective_ as the model. On
mostly-static frames, `predict next ≈ copy current` gets a deceptively good pixel
MSE because most pixels didn't change. Until the loss stops rewarding that and we
report an explicit collapse detector, a larger sweep mostly measures which arm
best exploits the shortcut. v4 fixes the objective and instruments the collapse.

---

## Change-by-change

### (1) Moving-mask batch — `data.py`

`moving_mask(frames, thresh=MOVE_THRESH)` takes `[B, N, 3, H, W]` and returns a
bool `[B, N-1, H, W]`: entry `t` is True where `frames[:, t+1]` differs from
`frames[:, t]` by more than `thresh` in _any_ channel (per-channel abs diff,
`amax` over channel). These are exactly the pixels for which copy-last is wrong —
the ground-truth motion mask.

`make_clip_batch(..., return_moving=True, move_thresh=...)` appends the mask to
its return. Return order is fixed and unambiguous: `out`, then `meta` (if
`return_meta`), then `moving` (if `return_moving`) — **moving is always last**.
So callers get `out` / `(out, meta)` / `(out, moving)` / `(out, meta, moving)`.

`MOVE_THRESH = 0.05` (frames are in `[0,1]`). Design choice: the mask is built
from **ground-truth consecutive-frame diffs**, not from the model's own rollout
predictions — so it's deterministic and decoupled from whichever arm is training.

### (2) Motion-balanced loss — `train_compare.py --motion-loss`

`motion_balanced_loss(pred, target, last, moving)` implements the §1 objective:

```
E = mean_channels((pred - target)^2)                     # [B,H,W]
L = 0.5*mean(E[moving]) + 0.5*mean(E[static])
      + 0.25*MSE(pred - last, target - last)
```

The 0.5/0.5 split makes the handful of moving pixels count as much as the entire
static background, so copy-last is no longer a good optimum. The third term
directly supervises the _motion residual_ (what changed since `last`). Empty-mask
guard: if a batch has no moving (or no static) pixels, that sub-term is 0 rather
than a NaN mean over an empty slice.

`last` is the **ground-truth previous frame** `clips[:, frames+k-1]` and the mask
index is the matching `frames+k-1` — the same persistence baseline in both the
mask and the residual term, consistent across every rollout step `k` (even when
the window is being fed the model's own detached predictions). Default OFF; when
off the loop uses the original `F.mse_loss` path byte-for-byte.

### (3) `copy_ratio` — always reported

```
copy_ratio = ||pred - last|| / (||target - last|| + eps)
```

computed in the single-step eval (`last = ctx[:, -1]`), averaged over eval
batches. Interpretation: `≈0` collapsed to persistence · `≈1` correct motion
magnitude · `≫1` unstable/excessive motion. Added to the result JSON
(`"copy_ratio"`) and the printed RESULT TABLE. This is the scalar that would have
made the g32 collapse obvious at a glance.

### (4) Rollout-K ramp — `train_compare.py --rollout-ramp`

Curriculum for the rollout-loss horizon: hold `K=3` through the first half of
training, then ramp `K` linearly `3 → 8` over the second half
(`K = round(3 + 5·frac)`, `frac = (step-half)/(steps-half)`). Rationale: learn a
decent one-step map first, then push long-horizon stability. `gen_frames` always
provisions for `Kmax=8`, so the extra frames exist from step 1; only the number
of rollout terms used grows. The per-25-step log row now includes `"K"` so the
ramp is visible, and `rollout_loss_K` in the JSON reports the max (8) with a new
`"rollout_ramp"` flag. Default OFF → fixed `K = --rollout-loss` as before.

---

## Smoke — `run_smoke_v4.sh` (CPU, python session-gated; operator runs it)

```
cd experiments/wavefield_video && bash run_smoke_v4.sh
# or: PY=/path/to/venv/bin/python bash run_smoke_v4.sh
```

Each block proves one flag end-to-end (tiny model, not an optimization result):

1. **moving-mask** — shapes `[B,T,H,W]` bool; `(out,meta,moving)` ordering;
   standalone `moving_mask` matches `return_moving`; threshold monotonicity
   (higher thresh ⇒ fewer moving pixels; `0 < frac < 1`).
2. **`--motion-loss`** — trains, then asserts `motion_loss=True` and a numeric
   `copy_ratio` in the result JSON; greps `copy_ratio` out of the table.
3. **`--rollout-ramp`** — 50 steps so log rows land at step 1/25/50; asserts the
   logged `K` tail reaches 8 and `rollout_loss_K == 8`.
4. **both flags together** — motion-loss + ramp in one run.
5. **regression** — no new flags ⇒ `motion_loss=False`, `rollout_ramp=False`,
   plain-MSE path, `copy_ratio` still reported.

Outputs land in `smoke_v4/`. **No commit** — experiment scratch, per the standing
rule for this directory.
