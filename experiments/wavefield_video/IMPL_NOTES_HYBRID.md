# IMPL_NOTES_HYBRID — DEEP_DIVE_3 Rank 1: hybrid gated fusion (`local_wave` / `local_ssm`)

Scope: `experiments/wavefield_video/`. No commits. This note pins the exact box
commands for the decisive matched-budget run set at g32 and records what the
hybrid build task actually changed.

## 0. What was already there vs. what this task added

The gated-fusion architecture DEEP_DIVE_3 ranks #1 was **already implemented in
R14** and matches the spec verbatim — this task did **not** re-derive it:

- `FusedMix` (`wfvideo.py:363`) computes exactly
  `h = g·h_local + (1−g)·h_global`, `g = sigmoid(Linear([h_local; h_global]))`
  (`_fuse`, `wfvideo.py:401-408`). The gate mean is cached **detached**
  (`_gate_mean`) so the "is the global state dead weight?" diagnostic never enters
  autograd.
- Local path = `AttnMix(causal=True)` (finite window = T). Global path =
  `WaveMix3D(dispersion)` for `local_wave` or `SSMLite` for `local_ssm`
  (`wfvideo.py:388-397`).
- `VideoPredictor` swaps in `FusedMix` **only when `fuse != "none"`**
  (`wfvideo.py:476-478`), so the default (non-fused) numeric path is untouched by
  construction — this is why "byte-identical default" is provable, not hopeful.
- Wired end-to-end in `train_compare.py`: `build()` passes `fuse=a.fuse`
  (`:688`), `--target-params` bisects `ffn_mult` through the same `build()`
  (`:692-698`) so hybrids are held to the single-mixer budget, the eval path is
  shared, and the result JSON records `fuse`, `persistent_state_bytes`,
  `window_bytes` (`:922-929`).
- Streaming parity + state-byte accounting tested in `test_fusion_r14.py`
  (`step()==forward()` at the last frame for both globals; hybrid persistent bytes
  == its global path's).

**This task added two things only:**

1. **`--fusion` alias** for `--fuse` (DEEP_DIVE_3 names the flag `--fusion`).
   `train_compare.py:608` — `ap.add_argument("--fuse", "--fusion", dest="fuse", …)`,
   choices reordered to `{none, local_wave, local_ssm}` to match the spec spelling.
   Both flags write the same `a.fuse`, so every existing script keeps working.
2. **`run_smoke_hybrid.sh`** — a focused CPU smoke (see §3).

## 1. The decisive matched-budget run set at g32 (balls)

DEEP_DIVE_3 §Part-2 Rank 1: the hybrid **doubles as** the highest-information
experiment — `local+wave` vs `local+ssm` vs `wave-only` vs `local-only`, all at one
parameter budget, so the only thing that differs is the temporal mechanism. K=1 to
isolate the **one-step map (the freeze)**, not the ramp/drift. Eval seeds are fixed
at 90000 inside `train_compare.py`, so the metric is invariant across arms.

Shared base (DEEP_DIVE_3 §1.4 BASE, g32, ~4M params):

```
BASE="--grid 32 --frames 17 --batch 16 --dim 192 --layers 6 --heads 8 \
  --target-params 4000000 --causal --residual --linear-pad --collisions \
  --motion-loss --rollout-loss 1 --steps 2000 \
  --eval-rollout 256 --eval-seeds 16 --eval-chunk 4 --auto-batch --out runs_dd3"
```

The four arms (run each; add `--seed 0/1/2` to replicate survivors):

```
# local-only  (finite causal attention window; NO persistent state)
python3 train_compare.py $BASE --kind attn --tag _local_only

# wave-only   (structured dispersion state; NO finite window)
python3 train_compare.py $BASE --kind wave --kernel-version dispersion --tag _wave_only

# local+wave  (HYBRID: causal window fused with the structured wave state)
python3 train_compare.py $BASE --kind wave --kernel-version dispersion \
    --fusion local_wave --tag _local_wave

# local+ssm   (HYBRID control: causal window fused with a generic diagonal SSM)
python3 train_compare.py $BASE --kind ssm --fusion local_ssm --tag _local_ssm
```

Optional 5th arm for the full HARDENING five-way (generic recurrence, no window):

```
python3 train_compare.py $BASE --kind ssm --tag _ssm_only
```

### FREEZE-FIX (apply the winner from the DEEP_DIVE_3 Part-1 suite)

DEEP_DIVE_3 Rank 1 says to run the hybrid **with the Part-1 freeze fix appended**.
Until Part-1 names the winner, the candidate one-knob fixes (all already wired) are:

- `--const-lr` (Suspect B: kill the OneCycle anneal that welds LR→0)
- `--fp32` (Suspect C: bf16 spectral underflow at 4× FFT size)
- `--no-decay-norm-head` (Suspect D: AdamW WD dragging the zero-init head to copy-last)
- `--no-collisions` (Suspect A / 0-det: make the future deterministic)

Append the winning flag to **all four arms identically** so the budget/mechanism
comparison stays clean, e.g. `… --fusion local_wave --const-lr --tag _local_wave`.

### What to read (do NOT lead with the 256-mean MSE)

Per-arm, from `result_<kind><tag>.json`:

- `copy_ratio` — grid-robust ratio (~0 frozen | ~1 correct motion | ≫1 unstable).
  **The freeze readout.** If `local+wave` moves (`copy_ratio` climbs off ~0.005)
  while `wave-only` freezes, the local path unfroze the one-step map — the whole
  point of Rank 1.
- `gate_g=[…]` in the `TELE` log lines (enable with `--telemetry-every 25`) — mean
  gate per layer. **Kill signal:** mean `g → 0.9+` ⇒ the wave/ssm persistent state
  is dead weight (the model routed everything through local attention).
- `divergence_horizon` and, on the occlusion probe, **`div_horizon/KB_state`**
  (`train_compare.py:966`) — the per-byte-of-persistent-state killer metric.
  **Kill criterion (DEEP_DIVE_3 Rank 1):** if `local+wave` does not beat
  `local+ssm` on divergence-horizon-per-byte, the structured wave formulation is
  not earning its keep.

## 2. Occlusion-harness arm wiring note

The long-memory decision is made on the **occlusion** data, not balls. That harness
is already wired — `run_occlusion.sh` runs the identical five arms at the fixed
`--target-params 4000000` budget over seeds 0–4 with a 1024-frame rollout:

```
A_localattn   --kind attn                                             = local-only
B_ssm         --kind ssm                                             = ssm-only
C_wave        --kind wave --kernel-version dispersion                 = wave-only
D_local_ssm   --kind ssm  --fuse local_ssm                           = local+ssm
E_local_wave  --kind wave --kernel-version dispersion --fuse local_wave = local+wave
```

(`run_occlusion.sh:56-70`; the task's four arms are A, C, D, E — B is the optional
generic-recurrence control.) Run it with:

```
STEPS=2000 SEEDS="0 1 2" OUT=runs_occlusion bash run_occlusion.sh
```

`--data-source occlusion` switches the eval to `occlusion_rollout_eval`, which
reports `divergence_horizon`, `exit_direction_accuracy`,
`position/velocity_error_at_emergence`, `target_identity_survival`, and the
persistent-vs-window state bytes — the metrics the kill criterion needs. To append
a Part-1 freeze fix here, add it to the shared `BASE` array
(`run_occlusion.sh:56-61`) so all arms inherit it.

**Note on `--fusion` in the harness:** `run_occlusion.sh` uses the canonical
`--fuse` spelling; the new `--fusion` alias is interchangeable but the harness was
left on `--fuse` to avoid churn in a checked-in script. Both resolve to `a.fuse`.

## 3. Smoke — `run_smoke_hybrid.sh`

CPU-only plumbing check (not an accuracy run). Run:

```
bash run_smoke_hybrid.sh        # writes ./smoke_hybrid/
```

It asserts, in order:

1. **Gated-fusion wrapper + streaming parity + state bytes** — reuses
   `test_fusion_r14.py`: `FusedMix.step() == FusedMix.forward()` at the last frame
   for both globals, and the exact `h = g·h_local + (1−g)·h_global` shape.
2. **Default path byte-identical** — two same-seed `--kind wave` runs, one via
   `--fuse none` and one via `--fusion none`, must agree on every deterministic
   field (`eval_mse`, `copy_ratio`, `divergence_horizon`, `rollout_mse_curve`, …).
   Proves the alias is a pure pass-through (CPU runs are fp32, so bit-deterministic).
   2b. **Mixer regression** — `sanity_check.py` (existing wave/attn/ssm numerics).
3. **Both hybrid arms train end-to-end at g16 AND g32**, matched to a shared
   `--target-params 20000` (reachable inside `match_ffn_mult`'s `[0.05, 64]`
   ffn-mult range at the tiny smoke dim) within 10%, with the arms within 10% of
   each other and both reporting `persistent_state_bytes > 0` and `window_bytes > 0`.

> Note: the smoke targets 20000 params (not the 4M of the real runs) because
> `match_ffn_mult` bisects `ffn_mult ∈ [0.05, 64]`; at the tiny smoke dim=16 a 4M
> target is unreachable and would report a spurious `WARN>10%`. The real decisive
> runs use dim=192, where 4M is comfortably inside the range.

## 4. Equal-params accounting

`--target-params` bisects `ffn_mult` via `match_ffn_mult` (`train_compare.py:94`,
`:692-698`) over the **full** parameter count returned by `build()`, which includes
the `FusedMix` two-mixer stack and the gate `Linear(2·dim, dim)`. So a hybrid arm
is held to the **same** budget as a single-mixer arm — the two-mixer cost is paid by
a smaller FFN, not by extra parameters. The `PARAM-MATCH … off=X%` line prints the
achieved match; `off > 10%` prints `WARN>10%`. At dim=192 all five arms match 4M to
well under 1%.
