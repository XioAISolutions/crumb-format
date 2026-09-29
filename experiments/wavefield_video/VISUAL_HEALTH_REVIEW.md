# Visual health hardening — review for Zeph

Predeclared screen (before implementation or failure-media measurement): flag
`texture_collapse` after `patience` consecutive frames with coarse spatial power
below **25%** of that sample's clean context and high-band power at least **50%**
of current spatial power. Per-channel spatial DC is removed; coarse is radial
frequency <=1/8 cycles/pixel, high is >=1/4. Context coarse power must exceed
1e-8. These are conservative engineering ratios, not fitted thresholds or
semantic certification. Motion is not required: jitter can hide texture collapse
from a freeze check. Thresholds have not been measured against the failure media.

Per-sample anchors, streak resets, onset indices, and save/load parity are tested.
Nonfinite generated frames still flag immediately; nonfinite context is rejected.
New checkpoints include the thresholds/version and coarse anchor. Legacy states
without that anchor require a fresh run, rather than learning it from generation.

`long_eval` now divides similarity by window 0's self-similarity, not generated
window 1's similarity. A synthetic early drop to positive similarity 0.2 now flags
instead of normalizing to 1. `NO_FLAGS` replaces `PASS`; `diagnostic_horizon_s`
replaces the coherence claim (the old key remains a deprecated numerical alias).
Reports and CLI, including reused receipts, say semantic quality is unverified.
Scope identifies the actual encoder, including pixel fallback. Window 0 is still
unverified: this does not provide an independently clean evaluator reference.

No flags means only that these heuristics did not fire. Wrong objects/scenes,
blur, texture retaining coarse power, already-degraded context, and defects below
the chosen spatial/temporal scales can escape. Resize/zoom and legitimate scene
changes can flag. A test demonstrates static fine texture present from the start
returns `NO_FLAGS` in `long_eval` while explicitly remaining unverified. Moving,
static natural-looking, and high-detail controls are synthetic scenes, not a
representative natural-video false-positive study. Full semantic quality remains
unverified until independent pixels review. No DINO inference or failure-media
evaluation was performed; no DINO evidence is claimed.

## Actual local validation (2026-09-29)

Used the requested interpreter, with existing Python 3.14 dependencies read-only
from the local cache; no installs. Without that path, imports failed (not TDD RED).
All commands below use this prefix from the worktree root:

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/Users/slavaz/.cache/uv/archive-v0/pop5eciiSH0JOZP1b9fzk/lib/python3.14/site-packages
/Users/slavaz/crumb-format/.venv/bin/python -m pytest -p no:cacheprovider ...
```

Tests preceded production changes; actual RED -> GREEN summaries:

| Cycle / test selection | RED output | GREEN output |
| --- | --- | --- |
| `test_long_horizon.py -k structure_to_static_or_jittering_texture` | `2 failed, 1 passed, 17 deselected` (subtests: `None != 4`) | module: `18 passed, 4 subtests passed` |
| Monitor invalid context, tiny frame, legacy state | `5 failed, 4 passed, 18 deselected, 1 warning, 3 subtests passed` | module: `24 passed, 10 subtests passed` |
| `test_long_eval.py -k early_positive` | `1 failed, 9 deselected` | module: `10 passed` |
| Evaluator reporting/scope | `4 failed, 7 passed` | module: `11 passed` |
| `test_long_eval.py -k nonfinite` | `6 failed, 11 deselected, 5 warnings` | module: `17 passed` |

The monitor edge-case selection was `-k 'context_is_rejected or tiny_flat or
legacy_state or texture_streaks or structured_moving or nonfinite_frame'`.
Later coverage-only additions check heterogeneous anchors and the explicit
initial-texture miss; they required no production changes.

Final focused command: `experiments/wavefield_video/tests/test_long_horizon.py
experiments/wavefield_video/tests/test_long_eval.py -q`:
**`43 passed, 10 subtests passed in 1.91s`**.

Then full command: `experiments/wavefield_video/tests -q --tb=short -rs`:
**`9 failed, 124 passed, 19 skipped, 66 subtests passed in 11.24s`**.
Eight failures initially came from a missing optional dep (`einops`); 19 optional VAE
tests skip when `diffusers` is absent. One further failure was
`test_waves.py::WaveTests::test_api_signature_and_optional_returns`, `KeyError:
'radius'` — a real API-parity gap in `data_waves.make_clip_batch` (its legacy
ball-motion parameters lacked `radius`). Fixed pre-merge by adding `radius=RADIUS`
to the signature (docstring updated). With the optional deps installed and that fix
in place the full suite is GREEN: `152 passed, 68 subtests passed` (pytest, Python
3.12 venv). `git diff --check` is clean.

Changed files: the two modules, their tests, this review, plus the pre-merge
`data_waves.py` signature fix and the `run_longlive.sh` wording update. Diff left
uncommitted for Zeph; no push, PR, release, GPU/SSH work, or failure-media
modifications.

## Zeph addendum (pre-merge, 2026-09-29)

- Independent fresh-context re-review (read-only): PASS with limitations — it flagged
  only two wording items, both fixed (this file and `run_longlive.sh`).
- `run_longlive.sh` pre-registered read now says NO_FLAGS / diagnostic horizon,
  mirroring `long_eval`'s new contract.
- Box evidence with this evaluator: c55 (DINOv2) on the 40-min latent clip reports
  `FAIL at 300s (diagnostic flags: colour); diagnostic horizon 300s of 2400s` —
  diagnostic only; semantic quality unverified.
- Known limits, accepted for merge: thresholds are conservative engineering ratios
  measured only on synthetic scenes; no natural-video false-positive study; no DINO
  validation of thresholds; window 0 remains a non-independent reference. A decoded
  pixels eye-check remains the gate for any quality claim.
