# R7 — LATENT SPACE: the gate to real pixels

Predicting raw 32×32 RGB is a toy. Real video is bigger, and the mixer cost
scales with the token count `T·H·W`. R7 adds a small **frozen conv autoencoder**
so the wave/attn/ssm predictors can operate on a compact **latent** — 16× fewer
tokens per frame — and still be scored in pixel space.

Nothing here changes the default (pixel) path: every latent behavior is gated
behind `--latent`, and the balls regression is proven byte-for-byte in the smoke.

## What shipped

- **`latent_ae.py`** (new) — `ConvAE`, a per-frame autoencoder.
  - Encoder `3 → 32 → 64`, two stride-2 convs ⇒ spatial `/4` (32×32 → 8×8×64).
  - Decoder mirrors it with `ConvTranspose2d 64 → 32 → 3`, ending in `sigmoid`
    so reconstructions live in `[0,1]`. The latent itself is left **unbounded**
    (linear) so a residual predictor can add/subtract deltas without saturating.
  - APIs: `ae.encode(x[B,3,H,W]) → z[B,64,H/4,W/4]`, `ae.decode(z) → x̂[B,3,H,W]`.
  - `--train` mode reports **held-out recon MSE** and saves `{state, config}`.
  - `load_ae(path)` rebuilds it frozen (`eval()`, `requires_grad=False`,
    `weights_only=True`).

- **`train_compare.py --latent`** (edited — the only allowed edit) —
  - `--latent` + `--ae-ckpt PATH`. Loads the frozen AE, reads `latent_ch`
    (=64) and sets the internal predictor grid to `grid/4`.
  - The wave/attn/ssm **mixing stack is reused unchanged**. Two thin adapters
    (defined in `train_compare.py`, so `wfvideo.py` is untouched):
    - `LatentCore` — builds a `VideoPredictor` at the latent grid, swaps only its
      I/O (`embed: Conv2d(64,dim)`, `head: Linear(dim,64)`), and re-runs the
      identical block loop. The zero-init residual head is preserved, so training
      still starts at "copy last latent" == copy last frame.
    - `LatentWrapper` — pixel-in/pixel-out (`decode(core(encode(frames)))`) so
      **every** pixel-space eval path (`rollout_eval`, `baseline_mses`,
      `centroids_by_color`, single-step eval) runs **unchanged** and the result
      JSON carries all the same fields.
  - Training loss is computed **in latent space** (predict next latent, K-step
    rollout MSE, feed predicted latent back detached). Eval **decodes** to pixels.
  - New JSON fields: `latent`, `latent_ch`, `latent_grid`, `ae_ckpt`,
    `ae_params`. All prior pixel-space fields are still present and populated.

- **`run_smoke_r7.sh`** (new) — CPU-quick, one approval runs all three proofs.
- **`IMPL_NOTES_R7.md`** — this file.

## Operator commands (python is session-gated)

```bash
cd experiments/wavefield_video

# One-shot smoke: AE roundtrip + tiny latent train + balls regression unchanged.
bash run_smoke_r7.sh
#   or with a specific interpreter:
PY=/path/to/venv/bin/python bash run_smoke_r7.sh
```

### Individual steps

```bash
# (1) Train + freeze an AE (3->32->64, grid 32 -> 8x8x64 latent). Reports recon MSE.
python latent_ae.py --train --data-source balls --grid 32 --steps 300 --out ckpts/ae.pt
python latent_ae.py --train --data-source waves --field wave --grid 32 --steps 300 --out ckpts/ae.pt

# (2) Train a predictor IN LATENT SPACE; eval is decoded to pixels (same JSON fields).
python train_compare.py --kind wave --kernel-version dispersion --linear-pad \
    --grid 32 --frames 12 --dim 384 --layers 8 \
    --latent --ae-ckpt ckpts/ae.pt --steps 2000 --rollout-loss 4 \
    --out runs_latent --tag _wave

# Head-to-head on the SAME frozen AE (fair: identical tokens for every arm):
python train_compare.py --kind attn --grid 32 --frames 12 --latent --ae-ckpt ckpts/ae.pt \
    --steps 2000 --out runs_latent --tag _attn

# (3) Pixel path (regression) — no --latent, unchanged behavior:
python train_compare.py --kind wave --kernel-version dispersion --linear-pad \
    --grid 32 --frames 12 --steps 2000 --collisions --out runs_pixel --tag _wave
```

## Design notes / boundaries

- **Fair comparison.** Freeze **one** AE and reuse it for every arm, so the
  wave/attn/ssm arms see identical latent tokens — the AE is not part of the
  contest, only the mixer is. `params` in the JSON counts the **predictor** only;
  `ae_params` is reported separately.
- **Honest pixel metrics.** Each rollout step does a full AE round-trip on the
  fed-back frame, so the reported pixel MSE / centroid / divergence numbers
  _include_ the AE's reconstruction error — the real cost of "the gate to pixels".
  Read `eval_recon_mse` from the AE's own `.json` as the noise floor.
- **Grid.** `--latent` requires `--grid` divisible by 4 (the two stride-2 layers).
  The AE is fully convolutional, so it runs at any /4 grid, but recon quality is
  best at the grid it was trained on.
- **Not wired for latent** (guarded with a clear error, by design):
  - `--motion-loss` — its moving-pixel mask is pixel-space; latent training uses
    plain latent MSE.
  - `--stream-test` — uses the wave-only O(1) `step()` recurrence in `wfvideo.py`,
    which is not threaded through the AE. Streaming stays a pixel-space feature.
- **Checkpoints.** In latent mode only the predictor (`LatentCore`) weights are
  saved/loaded/optimized; the AE ships separately as `--ae-ckpt`.

## Possible next knob (not yet built)

Training minimizes **latent** MSE. If decoded-pixel fidelity lags, add a small
`λ · pixel_MSE(decode(zpred), target_frame)` term to the latent loss — it costs a
decode per step but directly supervises what we actually score. Left out for now
to keep step cost low; it's a one-liner in the `--latent` branch if wanted.
