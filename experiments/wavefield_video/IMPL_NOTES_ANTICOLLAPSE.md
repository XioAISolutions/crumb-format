# IMPL_NOTES_ANTICOLLAPSE — VICReg var-reg / drift-pert / discrete history

Scope: `experiments/wavefield_video/`. No commits. Three anti-collapse knobs
motivated by `RESEARCH_SWEEP_20260926.md` (§2 FramePack history discretization,
§3 collapse taxonomy + SurgVista drift-perturbed training, §4 VICReg variance
regularization), targeting the **freeze** the DEEP_DIVE_3 Part-1 suite could not
break: at g32, `--const-lr`, `--fp32`, and `--no-decay-norm-head` all leave the
`--kind wave` arm frozen (`copy_ratio ≈ 0.005`) after 2000 steps.

All three act at the **training** level only — the model (`wfvideo.py`) is
unchanged (see §5) — and every one is gated so the default path is a provable
byte-identical no-op.

## 0. What changed (files)

- `train_compare.py`
  - imports: `+ math`.
  - `apply_drift_pert(frames, magnitude, seed)` and `quantize_frames(frames,
levels)` helpers + `HIST_DISC_LEVELS = 16` (just after `make_clip_batch`).
  - three flags: `--var-reg FLOAT` (0.0=off), `--drift-pert FLOAT` (0.0=off),
    `--hist-disc {off,bits}` (off default).
  - validation: `--drift-pert`/`--hist-disc` rejected under `--latent`
    (pixel-space augmentations); negative magnitudes rejected. `--var-reg` is
    allowed under `--latent` (it matches variance in latent space).
  - context augmentation applied right after `ctx = clips[:, :a.frames].clone()`
    (drift first, then discretize).
  - var-reg hinge accumulated inside both the latent and pixel rollout loss
    loops and added to `loss` (weighted per rollout step like the MSE terms).
  - `var_reg` contribution logged in each step row when on; `var_reg`,
    `drift_pert`, `hist_disc` recorded in `result_*.json`.
- `run_anticollapse.sh` — the 4-arm g32/2000-step/seed-0 suite (box-ready).
- `run_anticollapse_smoke.sh` — CPU tiny-grid smoke, all knobs on/off + a hard
  default-path no-op assertion.
- `wfvideo.py` — **intentionally untouched** (see §5).

## 1. `--var-reg` — VICReg-style variance-matching on frame deltas

Per rollout step `k` we form two one-step deltas:

- `pred_delta = pred − win[:, −1]` — the motion the model actually produced,
  relative to the last frame it was fed (its own previous prediction for `k>0`).
- `gt_delta   = tgt_k − clips[:, frames+k−1]` — the true motion (clean).

The added loss term is

```
var_reg * Σ_k w_k · relu( std(gt_delta) − std(pred_delta) )      w_0=1, w_{k>0}=0.25
```

std is a **batch-aggregated scalar** over all elements of the delta tensor (the
simplest VICReg-flavored aggregate), matching the per-step MSE weighting so the
regularizer and the objective ramp together.

### ⚠️ SIGN DECISION — read this (deliberate deviation from the task shorthand)

The task wrote the hinge as `(std(pred_delta) − std(gt_delta))_+`. **I
implemented the opposite subtraction order:** `relu(std(gt_delta) −
std(pred_delta))`. Reason: the failure we are fighting is the **freeze**, where
the prediction is nearly constant ⇒ `std(pred_delta) → 0`. Only
`relu(std(gt) − std(pred))` is **positive exactly when the model under-moves**
(collapses) and pushes variance **up**. The literal `(std(pred) − std(gt))_+`
fires when the model moves _more_ than ground truth and would penalize motion —
i.e. it would _reward_ the freeze. That is the VICReg variance term's whole
point (`max(0, γ − std)` penalizes low variance), so I read the task shorthand as
having the difference written the wrong way round for an anti-collapse tool.

If you truly want the literal form (penalize over-motion, e.g. to damp the
`copy_ratio ≫ 1` explosion instead), flip the two `torch.relu(gd.std() −
pd.std())` lines to `torch.relu(pd.std() − gd.std())` in `train_compare.py`
(latent + pixel branches) — it is a one-token change in each. **Confirm which
direction you meant before reading the runs.**

`--var-reg 0.5` is the suite value (start moderate; the hinge is unit-consistent
with the MSE since both are in pixel-value units).

## 2. `--drift-pert` — drift-perturbed training augmentation

`apply_drift_pert` overlays, on the **context frames only**, a per-sample smooth
low-frequency field of the same family as `crumb_coherence/scripts/run_m0.py`
`inject_drift` (slow exposure ramp + brightness lift + low-freq spatial color
cast, wandering across the T frames), amplitude-scaled by the flag. Targets stay
clean. Anti-collapse rationale (SurgVista, RESEARCH §3): a **drifted** history no
longer equals the clean future, so copy-last stops being a valid shortcut and the
model is forced to infer real dynamics.

Two properties the byte-identical guarantee rests on, both asserted by the smoke:

1. **Private RNG.** The field is drawn from a dedicated `torch.Generator`
   (seeded `2_000_000 + step + seed_off`), so it never advances the global RNG
   stream that seeds clip generation — with the flag off, zero draws happen.
2. **Determinism.** Same `(frames, magnitude, seed)` ⇒ same field.

`--drift-pert 0.05` is the suite value (a mild cast; `inject_drift` uses ~0.08–0.25
for a _detectable_ drift, so 0.05 is deliberately below the "obvious" band).

## 3. `--hist-disc {off,bits}` — discrete history representation

`bits` rounds the context frames to `HIST_DISC_LEVELS = 16` uniform bins in
`[0,1]` before feeding the model; targets stay continuous. FramePack v1's
"History Discretization" (RESEARCH §2/§4). Anti-collapse rationale: coarsening
the history strips the sub-bin precision the model would otherwise reuse to copy
the last frame pixel-exactly, so verbatim echo is no longer representable.

Applied **after** drift (the model sees the drifted-then-coarsened history a real
streaming decoder would). Only the _real_ context is discretized — rollout
feed-back predictions stay continuous.

## 4. Default-path no-op (why "byte-identical" is provable, not hopeful)

With `--var-reg 0 --drift-pert 0 --hist-disc off`:

- the drift/quantize calls are skipped (no tensors, no RNG draws);
- the var-reg `if a.var_reg > 0:` blocks are skipped — `var_term = 0.0` is a bare
  float assignment that is never added to `loss`;
- the step row gets no `var_reg` key.

So the numeric training path is identical to pre-change. `run_anticollapse_smoke.sh`
step (2) proves it: a plain run and an explicit-defaults run at the same seed must
agree on every deterministic field (CPU is fp32 ⇒ bit-deterministic), and the base
run must carry no `var_reg` step field. The three `result_*.json` keys
(`var_reg/drift_pert/hist_disc`) are metadata only and do not enter training.

## 5. Why `wfvideo.py` was not touched

The task named `train_compare.py + wfvideo.py` as the edit surface. All three
knobs are cleanly training-side: var-reg is a loss term, drift-pert and hist-disc
are context-input transforms. Implementing them at the training level (a) keeps
the model forward — and thus the "byte-identical default" and streaming-parity
guarantees from IMPL_NOTES_HYBRID — intact, and (b) keeps the **eval** path clean
(augmentations are train-only) so `copy_ratio` / `divergence_horizon` stay
comparable across arms. No `wfvideo.py` change was needed; forcing one would only
add risk. Flagging this explicitly since it deviates from the literal file list.

## 6. Box commands (fire when trained arms finish)

Verification note: the CPU smoke could not be run in the authoring session
(`python3` was blocked by the sandbox permission mode); it is written box-ready
and reviewed by inspection. Run it first on any machine with the deps:

```
bash run_anticollapse_smoke.sh          # CPU plumbing + default no-op; ~1 min
```

Then the decisive suite on the GPU box (mirrors `run_freeze_fast.sh` paths/BASE):

```
bash run_anticollapse.sh                # 4 arms, g32, 2000 steps, seed 0, telemetry on
# -> runs_anticollapse/result_wave_{varreg,drift,histdisc,combo}.json
```

Read **`copy_ratio` first** (not the 256-mean MSE): ~0 = still frozen, ~1 =
correct motion magnitude, ≫1 = unstable. An arm unfroze if `copy_ratio` climbs
off ~0.005 toward ~1 without the rollout blowing up. Cross-check `head_gn`
(head grad-norm should be non-vanishing) and, on the fused variants, `gate_g`
in the `TELE` lines.

### BLOCKING: `combo best-two`

`run_anticollapse.sh` pre-wires the combo to **var-reg + drift-pert** (the
loss-level and data-level knobs — the most mechanistically complementary pair).
This is a defensible default, **not** a measured result. After the three
single-knob arms finish, if `--hist-disc` lands in the top two on `copy_ratio`,
re-run the combo with the true best pair:

```
COMBO="--drift-pert 0.05 --hist-disc bits" bash run_anticollapse.sh
```

(only the `_combo` tag is rewritten; the single-knob arms are idempotent).

## 7. Cross-refs

- `RESEARCH_SWEEP_20260926.md` — motivation (§2 discretization, §3 drift-pert &
  collapse taxonomy, §4 VICReg).
- `IMPL_NOTES_HYBRID.md` — the Rank-1 hybrid; the freeze fix (whichever Part-1
  knob wins) is meant to be _appended to all arms identically_, and the same
  applies here — an anti-collapse knob can be stacked on the winning freeze fix.
- `run_freeze_fast.sh` — the Part-1 suite whose g32 wave arm is the frozen
  baseline these knobs attack (same BASE reused in `run_anticollapse.sh`).
- `crumb_coherence/scripts/run_m0.py` `inject_drift` — the drift field family.
