# R11 — LATENT DECODE-RENDER + PIXEL-SPACE EVAL

R7 gave us a frozen conv AE and let `train_compare.py --latent` train a predictor
in latent space (scored in pixels _at train time_). R11 closes the loop on the
**inference/demo** side: `render_rollout.py --latent` loads a trained latent
predictor + the same frozen AE, rolls the predictor **autoregressively at the
latent grid**, decodes **every** predicted latent through the AE to pixels, and
scores in **pixel space** against the pixel ground-truth clip.

Nothing about the default (pixel) render path changes: everything latent is gated
behind `--latent`, and `render()` returns into `render_latent()` on the very first
line — the entire pixel rollout below it is byte-for-byte unchanged.

## What shipped

- **`render_rollout.py --latent --ae-ckpt PATH`** (edited) —
  - `render(args)` early-returns `render_latent(args)` when `--latent` is set; the
    pixel path (`load_model`, recurrent/windowed loop, report) is untouched.
  - `load_latent_model()` — rebuilds the `LatentCore(VideoPredictor@latent_grid)`
    from the checkpoint's `vp.*` state and its `result_<tag>.json`. The exact FFN
    width is recovered from `vp.blocks.0.ffn.fc1.weight` (the result JSON rounds
    `ffn_mult` to 3 dp), so `load_state_dict(strict=True)` matches the trained
    1.6M-param arms. Loads the frozen AE via `latent_ae.load_ae` and asserts
    `ae.latent_ch == predictor latent_ch`.
  - Rollout: encode the seed context **once** → latent window `[1,T,cz,h,w]`; each
    step `z_next = core(z_win)`, `x_pred = ae.decode(z_next)`, then feed **the
    predicted latent** back (`z_win = cat(z_win[:,1:], z_next)`). Decode is used
    only for scoring/PNGs — the autoregression itself stays in latent space, so
    the error reflects predictor drift, not repeated AE re-quantization.
  - **Pixel-space metrics** vs the same-seed `make_clip_batch` pixel clip:
    `mse` (decoded pred vs pixel GT), `copy_last_mse` (frozen last seed frame at
    every horizon), per-ball **centroid error** + **divergence_horizon**
    (`centroids_by_color` / `divergence_horizon` imported from `train_compare`,
    `div_thresh = data.RADIUS`, balls only — waves carry no centroids, exactly as
    `rollout_eval`), plus `mse_curve`, `copy_last_mse_curve`, `centroid_err_curve`.
  - `metrics.json` also records `ae_eval_recon_mse` (the AE's own held-out floor),
    a `latent` config block (`latent_ch`, `latent_grid`, `ae_ckpt`, `ae_params`,
    `predictor_params`), and provenance sha256s incl. `latent_ae.py` + `train_compare.py`.
  - Refusals: a latent ckpt without `vp.*` state, a config with `latent!=true`,
    an AE channel mismatch, or `--mode recurrent` (not threaded through the AE)
    all error clearly; the pixel path likewise refuses the latent ckpt.

- **`run_smoke_r11.sh`** (new) — CPU-quick, **no trained ckpt needed**: synthesizes
  a tiny **random** ConvAE + LatentCore + result JSON, renders the `--latent` decode
  path, checks the two paths refuse each other's checkpoints, and re-renders a tiny
  **pixel** VideoPredictor (default-path regression). Proves plumbing, not accuracy.

- **`IMPL_NOTES_R11.md`** — this file.

## Operator commands (python is session-gated)

```bash
cd experiments/wavefield_video

# One-shot smoke: latent decode path + regression, all synthetic (no ckpt/train).
bash run_smoke_r11.sh
#   or with a specific interpreter:
PY=/path/to/venv/bin/python bash run_smoke_r11.sh
```

## Box render commands (the real deliverable)

Run on the box, where the trained latent arms and the frozen AE live. The AE is
`ckpts/ae_g32.pt` (grid-32 balls AE, latent_ch 64, latent grid 8); both latent
arms were trained on **balls + collisions**, so centroid/divergence metrics apply.

```bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p demo_out

# (a) WAVE latent arm  -> pixel metrics + mp4 into demo_out/
$P render_rollout.py --latent \
    --ckpt   runs_latent/model_wave_w1lat.pt \
    --ae-ckpt /workspace/slava/exp/wavefield_video/ckpts/ae_g32.pt \
    --kind wave --grid 32 --frames 256 --seed 170001 \
    --side-by-side --fps 24 --scale 8 \
    --out demo_out/latent_wave_w1lat

# (b) ATTN latent arm (run once runs_latent/model_attn_a1lat.pt exists)
$P render_rollout.py --latent \
    --ckpt   runs_latent/model_attn_a1lat.pt \
    --ae-ckpt /workspace/slava/exp/wavefield_video/ckpts/ae_g32.pt \
    --kind attn --grid 32 --frames 256 --seed 170001 \
    --side-by-side --fps 24 --scale 8 \
    --out demo_out/latent_attn_a1lat
```

Each produces, under its `--out` dir:

- `metrics.json` — pixel-space `mse`, `copy_last_mse`, `copy_last_over_model`,
  `mean/final_centroid_err`, `divergence_horizon`, `identity_survival`,
  `ae_eval_recon_mse`, full curves, config + provenance.
- `prediction.mp4`, `ground_truth.mp4`, `comparison.mp4` (+ PNG dirs), at
  `scale`×-upscaled `grid` for viewing (metrics stay at native grid).

Notes:

- `--kind`/`--grid` are **assertions** (they must match the checkpoint's config;
  drop them and the config still drives the build) — kept here as a guardrail.
- `--ae-ckpt` may be omitted if the predictor's `result_*.json` carries a valid
  `ae_ckpt` path reachable from the run dir; passing it explicitly is safer.
- `--seed 170001` matches the pixel demos so the latent and pixel renders share
  the same ground-truth clip for a like-for-like comparison.

## Design notes / boundaries

- **Latent feedback, not pixel feedback.** The demo rolls the predicted **latent**
  back (decode is score-only). This differs from `train_compare`'s `LatentWrapper`
  (`decode∘core∘encode` each step); the R11 loop is truer to "run the latent model
  autoregressively at latent_grid" and avoids an extra encode/decode per step.
- **Honest pixel numbers.** Because every scored frame is a real AE decode, the
  reported MSE/centroid/divergence **include** the AE reconstruction floor. Read
  `ae_eval_recon_mse` in `metrics.json` (mirrored from the AE's `.json`) as the
  noise floor the predictor can't beat.
- **Windowed only.** Latent rollout keeps a `T`-frame latent window and re-runs the
  full forward each step. The wave-only O(1) `step()` recurrence is **not** threaded
  through the AE (same boundary as R7's `--stream-test`), so `--mode recurrent`
  errors under `--latent`.
- **Untouched pixel path.** `render_latent` is a sibling of `render`; no pixel-path
  code was refactored. `run_smoke_r11.sh` step (3) re-renders a pixel checkpoint to
  prove the default path is unchanged.
