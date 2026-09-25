# R5 implementation and validation

Delivered as new files only. No existing Python sources were edited and no git
commits were made. SHA-256 comparison against the session-start snapshot confirmed
all 13 pre-existing Python sources were unchanged.

## Files

- `data_waves.py`: deterministic spectral PDE clips, raw evolution API, motion mask.
- `audit_wavefield.sh`: isolated compilation and v3/v4 smoke audit with logs and manifest.
- `proofpack.py`: completed-run bundler with configuration and exact reproduction commands.
- `verify_proofpack.py`: standalone SHA-256 verifier, standard library only.
- `tests/test_waves.py`: analytic, numerical, determinism, and API checks.
- `tests/test_audit.py`: audit failure propagation and output isolation checks.
- `tests/test_proofpack.py`: integrity, malformed input, CLI, and portability checks.

## PDE clips

```python
from data_waves import make_clip_batch

frames, meta, moving = make_clip_batch(
    4, 64, 32, 32, seed=123, field="vortex", device="cpu",
    return_meta=True, return_moving=True,
)
# frames: float32 [4, 65, 3, 32, 32], values in [0, 1]
# moving: bool [4, 64, 32, 32]
physical = (2 * frames.double() - 1) * meta["scale"][:, None, :, None, None]
```

The original `data.make_clip_batch` positional arguments, defaults, and optional
return order are preserved. New arguments are keyword-only: `field`, `time_step`,
`velocity`, and `viscosity`. Supported spatial dimensions are independently
16/32/64; T is 0 through 64. Channels encode field, continuous time derivative
(`dt`), and spatial Laplacian, respectively. The time derivative is not a finite
frame difference or the scalar time step.

Coordinates use the periodic domain `[0, 2*pi)` in both axes. The default time step
is 0.15; frame t represents time `t * time_step`. Spatial modes are limited to
absolute frequency 3 per axis, below Nyquist on every supported grid.

| Field | Equation and evolution |
| --- | --- |
| `wave` | `u_tt = c^2 Δu`; exact cosine/sine Fourier oscillator with `c=speed`. Initial velocities select traveling branches. |
| `advection` | `u_t + v·∇u = 0`; multiply each Fourier coefficient by `exp(-i k·v t)`. |
| `vortex` | Translated Taylor–Green vorticity; multiply by `exp((-i k·v - ν|k|²)t)`, default `ν=0.02`. |

Vortex initial conditions are products of two cosines on a single Laplacian
eigenspace. Their induced incompressible velocity has zero vorticity self-advection,
so translation plus diffusion is also an exact solution of the full 2-D
Navier–Stokes vorticity equation with a uniform background drift. These are smooth
cellular vortices, not general interacting or turbulent vortex simulations.

Randomness comes from a local CPU generator; explicit seeds reproduce exactly
within the same runtime without changing the global torch RNG. An omitted seed
draws local entropy; `meta["seed"]` allows replay. FFT calculations use float64 on
CPU, then encoded clips transfer to the requested device as float32. Channel scales
are initial Fourier L1 bounds per sample, fixed for the entire trajectory. They do
not inspect future frames; changing T preserves the existing frame prefix. Physical
zero maps to 0.5, including identically zero derivative channels. Clamping handles
floating-point roundoff at the encoded bounds.

`moving_mask` has the same any-channel absolute-difference semantics as `data.py`.
Legacy ball-specific kicks/collisions arguments are accepted as no-ops. `nb`
controls Fourier mode count for wave/advection; vortex uses one cell lattice.
PDE metadata contains scales, drift, frequencies, and times instead of ball
positions/colors. The existing ball centroid evaluator is therefore not a PDE
evaluator. No training script imports were changed.

`spectral_evolve(initial, T, ..., initial_dt=...)` exposes raw physical channels
for analytic verification. It accepts CPU float32/float64 `[B,H,W]` tensors. Wave
DC modes evolve correctly as `u0 + initial_dt*t`. For arbitrary initial data the
vortex branch solves transported diffusion; its full vorticity interpretation
requires the eigenspace initial conditions described above. Inputs to this raw
API should avoid even-grid Nyquist modes for unambiguous fractional translations.

## Audit

```sh
PY=/Users/slavaz/crumb-format/.venv-llm/bin/python bash audit_wavefield.sh
# Or choose a new retained output directory:
PY=/path/to/python bash audit_wavefield.sh /tmp/my-new-wavefield-audit
```

The default output is a fresh retained temporary directory; `AUDIT_OUT` is also
supported. Existing destinations are refused. The audit snapshots project Python
sources recursively plus both smoke scripts, records source SHA-256 values, and
compiles every snapshotted `.py` with `py_compile`. Git/cache/virtualenv directories
and its current output directory are excluded.

Both smokes run from the snapshot with CUDA hidden and CPU thread defaults of 2.
This isolates existing artifacts and accommodates v4's hardcoded `smoke_v4` JSON
paths. `bash -o pipefail` exposes failures hidden by existing pipelines. The audit
also catches `SANITY FAILURES-PRESENT`, since the existing sanity script can report
failures without setting a failing exit code. Every check is attempted, with logs
and aggregate PASS/FAIL in `manifest.json`; any failed check yields exit 1.

## Proofpack

```sh
python3 proofpack.py path/to/completed_run path/to/new_bundle \
  --config path/to/config.json \
  --repro-cwd /absolute/path/to/training/source \
  --repro-command 'python3 train_compare.py --kind wave --steps 10 --out path/to/completed_run'
python3 verify_proofpack.py path/to/new_bundle
# The verifier travels with the bundle:
cd path/to/new_bundle
python3 verify_proofpack.py .
```

Supply the actual complete command used for your run; the command above illustrates
CLI syntax. Repeat `--repro-command` for a multi-stage run. Commands can alternatively
come from JSON config keys `repro_commands` (list) or `repro_command` (string).
Default reproduction cwd is the bundler's invocation directory. Existing result
JSONs omit some training options, so commands are never guessed from metrics.

The input must contain `result.json`, `results.json`, or `result_*.json` and at least
one `.pt`, `.pth`, `.ckpt`, or `.safetensors` checkpoint. Without `--config`, the tool
discovers `config.json/yaml/yml/toml/ini` inside the run. Result and JSON config files
must contain JSON objects. All run files are copied under `run/`; an external config
is copied under `config/`. A new output directory is required outside the input run.

`manifest.json` records relative paths, SHA-256, byte sizes, artifact roles,
reproduction commands/cwd, and producer environment. `VERIFY.md` includes the
standalone verification command and exact supplied reproduction commands. Both it
and the copied verifier are hashed. The CLI prints the manifest's own hash as an
external integrity anchor; the manifest cannot hash itself.

Verification rehashes every file and rejects changed, missing, extra, unsafe-path,
duplicate, symlink, or malformed entries. It does not deserialize checkpoints or
execute recorded commands. This proves file integrity against the manifest; it does
not prove authorship or numerical reproducibility. Source/dependency environments
are not implicitly bundled, and reproduction requires the original source checkout.

## Executed validation — 2026-09-25

Environment: macOS arm64, Python 3.12.12, torch 2.12.0, CPU. System `python3` lacks
torch on this machine; the existing `.venv-llm` interpreter was used for PDE/smoke
tests. Proofpack itself requires only Python 3.10+ standard library.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /Users/slavaz/crumb-format/.venv-llm/bin/python -m unittest discover -s tests -v
PY=/Users/slavaz/crumb-format/.venv-llm/bin/python \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash audit_wavefield.sh
```

- **29 tests passed**: 9 PDE/API, 5 audit, 15 proofpack.
- **21 Python files compiled**, zero failures in the final source snapshot.
- **Smoke v3 PASS** (11.136 s), **smoke v4 PASS** (29.031 s), aggregate exit 0.
- Analytic 2-D traveling wave used `kx=2`, `ky=3`, `c=1.15`, phase 0.37, all
  65 timestamps. Worst absolute errors across field/time derivative/Laplacian:

| Grid | Maximum absolute error |
| --- | ---: |
| 16 | 2.20e-13 |
| 32 | 4.82e-13 |
| 64 | 2.48e-12 |

These are raw physical-channel comparisons against closed form, all below 1e-3.
Additional checks cover advection conservation, exact viscous vortex evolution,
stationary/DC limits, supported dimensions and T=64, range, derivative/Laplacian
consistency after decoding, seeded replay, global RNG isolation, prefix equality,
all optional return forms, and equivalence to the original moving mask.

Final audit artifacts and logs are retained locally at:

```text
/private/var/folders/ls/41r_37r57rlcx2jfdfx95l000000gn/T/wavefield-audit-r5-mirgudtj/
```

Its completed `source/smoke_v3` run was also bundled with an explicit harness config
and reproduction command into `proofpack_v3/`. Running the copied standalone verifier
there returned **PASS: 9 files rehashed**. This integration example contains the
existing moving-ball smoke run, including its real saved/resumed checkpoint.
The example pack's manifest SHA-256 is:

```text
fc987b074bed3eebb31ffba54ab87fb7bfae331956a66e67fa8f196d9cc2af9f
```
