# I2V harness support (2026-09-30)

LongLive's repo ships `configs/inference_i2v.yaml` (first latent clamped to a conditioning image; the AR rollout continues from it) but our harness had no path to it. Added via `scripts/longlive_i2v_patch.py` (fail-closed anchors; backups `.before_i2v`):

- `longlive_long.py`: `generate --i2v` sets overlay `i2v: true` + `inference.independent_first_frame: true` and drops the LoRA adapter section; `n_prompts_of` counts images for i2v data dirs (flat `name.png` + `name.txt`, or `images/` + `prompts/`).
- `run_longlive.sh`: `I2V=1` accepts a data dir as `PROMPTS`, auto-selects `configs/inference_i2v.yaml` when `BASE_CONFIG` is unset, and passes `--i2v`.

Evidence: `bash -n` clean; box venv pytest **19/19** (17 prior + 2 new: image-layout counting, overlay i2v flags + adapter removal). Applied on the box ~00:24Z (pre-patch md5 `366eedb8…` = repo copy).

First use: `c01w_1_i2v` — continue forward from `street_frame0.png` (frame 0 of c01q), 10 s, W12 rel-rope. Motivation: the model does not unlock camera travel under any config tested so far (matrix in `OPERATIONS.md`); i2v is also the product path (photo → living shot).
