# HYPERCOMPLEX STUDY — which hypercomplex math earns its place in our model

**Card:** `t_02a3dd20` (study + design). **Child:** `t_7877bc01` (implement + run the top 1–3 arms).
**Date:** 2026-09-27 (EDT). **Repo:** `crumb-format`, branch `crumb-llm-standalone`, path `experiments/wavefield_video/reports/HYPERCOMPLEX_STUDY.md`.
**Owner directive, this card:** "Use hyper complex number where needed and all other LLM building skills you have."
This document is the *where needed* mapping: what to build, what to refuse, and the exact runnable
experiments with kill criteria. Top 1–3 (E1–E3) are scoped for the child card and designed to be
implemented as **additive arms** in `train_compare.py` without touching the existing wave/ssm paths.

**Integrity statement.** Every external claim below is tied to a paper whose arXiv ID, title and
authors were verified against the arXiv API in this study's session (see §6 and Appendix A);
claims *made by a paper* are marked as such; numbers about our own stack are cited to committed
campaign docs. No fabricated results. Where the honest answer is "no", it says no.

---

## 0 · TL;DR — the verdict table

| Family | What the literature actually shows | Our stack slot | Verdict |
|---|---|---|---|
| **Quaternion layers (Hamilton-structured weights)** | Same quality with ~3.3×–4× fewer free params in ASR/NLP (QRNN, Quaternion-Transformer); QCNN beats matched real CNNs on color tasks; QUAN (2025) fewer params + rotation handling | mixer input/output maps (wave arm), stem (color) | **BUILD (E2, E3)** — literature-backed, cheap, params-matched protocol handles the claim honestly |
| **Quaternion-valued *state* in a recurrent/SSM memory** | **No ML prior art found.** 0 hits for quaternion Mamba/selective scan; the only "quaternion state-space" work is classical time-series statistics (H-IAR, 2026) | the SSM arm (`ssm_lite.py`) and the wave streaming state | **BUILD (E1)** as a pre-registered *novel* test (equal bytes) — strict kill; do not lean on it until measured |
| **Quaternion color algebra (RGB as pure quaternion)** | Quaternion color-video processing works: Q-DMD beats exact DMD for background modeling, uQRPCA+ is SOTA on moving-target detection/background recovery | engine low-band state; model stem | **BUILD-part (E3) / TRACK (E4)** — real for video matrix methods; our arms test the neural version |
| **Clifford / geometric-algebra networks** | Clifford layers consistently improve neural PDE surrogates at similar params (Navier-Stokes/weather/Maxwell); CGENN/GATr give O(n)/E(3) equivariance and win on geometric data | 3D lane (authored, not learned), future pose/geometry models | **WATCH — do not build for the current 2D pixel model.** No integration point today; triggers in Appendix B |
| **Hypercomplex FFTs** | They exist and are usable (quaternion FFT = 2 complex FFTs via block diagonalization), but there is **no proper pointwise convolution theorem** — "a main stumbling block for filter design" (Bujack et al. 2013) | engine spectral core | **CONDITIONAL — not a speed feature.** Usable for compare/blend ops (no filtering); one testable use = E4 |
| **Dual numbers** | Forward-mode AD's implementation vehicle; niche = few inputs → many outputs | anywhere | **NO.** Reverse-mode autograd already covers the stack; no seam hits the forward-mode niche |
| **Dual quaternions** | Big literature in robotics/SLAM/skinning (dual-quaternion networks, hand-eye calibration) | 3D lane | **NO.** We author poses; we don't skin characters or estimate pose from footage |
| **Split-complex numbers** | No ML literature found (81 arXiv hits, all pure math/physics) | — | **NO** |
| **Octonions** | Exists: "Deep Octonion Networks" (2019) claims better convergence/accuracy than real/complex/quaternion nets on CIFAR; non-associativity makes algebra and composition awkward; one line of work | — | **NO (novelty check: open but unsupported)** — do not build |
| **Quaternion / geometric conditioning of diffusion models** | Exists: ResQu quaternion-wavelet conditioning for SR; Clifford diffusion for molecules | our Wan conditioning lane | **WATCH** — template exists; no seam in our control-video conditioning today |

One-line summary: **build the quaternion *parameterization* tests (E2/E3) — they have the
literature behind them; build the quaternion *memory* test (E1) because the state shelf is
genuinely empty and the per-byte story is our wedge — but gate it with a kill; refuse octonions,
split-complex, dual numbers, dual quaternions, and any GA architecture for the current model.**

---

## 1 · The families — what is actually measured (with sources)

### 1.1 Quaternion neural networks (parameter efficiency, rotational equivariance)

The measured core, from the primary sources:

- **QRNN/QLSTM** (Parcollet et al., 2018, arXiv:1806.04418): "QRNN and QLSTM reduce by a maximum
  factor of **3.3×** the number of free parameters needed, compared to real-valued RNNs and LSTMs
  to reach better results" — on automatic speech recognition. This is the canonical measured
  *parameter-efficiency* result, and the paper frames quaternions as coding "internal dependencies
  by composing and processing multidimensional features as single entities" (capsule-like).
- **Quaternion CNN for ASR** (Parcollet et al., 2018, arXiv:1806.07789): same family, CNN version,
  mel-filter-bank + derivatives as quaternion input entities.
- **QCNN** (Zhu et al., 2019, arXiv:1903.00658): color images as quaternion matrices; redesigned
  convolution/FC layers; "outperform the real-valued CNNs with same structures" on color
  classification and denoising.
- **Deep Quaternion Networks** (Gaudet & Maida, 2017, arXiv:1712.04604): quaternion weight
  initialization + quaternion batch normalization; "improved convergence compared to real-valued
  and complex-valued networks, especially on the segmentation task, while having fewer parameters"
  (CIFAR-10/100, KITTI road).
- **Lightweight NLP with quaternion networks** (Tay et al., 2019, arXiv:1906.04393): "up to 75%
  reduction in parameter size without significant loss in performance" (verbatim — "lesser
  degrees of freedom in the Hamilton product" is their stated reason); introduces Quaternion
  attention/Transformer — the 4× number usually quoted as "quaternion = 4× cheaper".
- **QUAN** (Grant & Wang, 2025, arXiv:2509.05512): quaternion approximation via Hamilton-product
  decomposition with real-valued ops + CUDA kernels; "higher accuracy with fewer parameters and
  faster convergence" than existing convolution and quaternion-based models (their wording), on
  image-classification and oriented-object-detection benchmarks.
- **Rotational equivariance**: the sharpest statement is REQNN (Shen et al., 2019,
  arXiv:1911.09040): "when a neural network uses quaternion features under certain conditions, the
  network feature naturally has the rotation-equivariance property" — for **3D point-cloud**
  processing. That is a real, measured property in its domain; it is **not** a property of 2D
  image-plane models and **not** 3D camera motion (see §3's "no free camera motion").
- Engineering reality check: quaternion nets are not free — they need their own init, batch-norm,
  activations and backprop treatment (arXiv:2406.16481 activations; arXiv:2212.13082 quaternion
  backprop; arXiv:2311.16771 HR-calculus provides the derivative machinery). Training via
  autograd on real-component ops avoids custom calculus (see E1/E2 designs).
- The 2026 **shared-score quaternion self-attention** (arXiv:2605.24920) proves that component-wise
  quaternion attention largely "re-parameterizes the same interactions" — a useful caution: not
  every quaternionized block adds capacity; the gains that are real are at the *linear/convolution*
  parameterization level, not by multiplying exotic attention heads.

**Verdict-relevant reading:** quaternion *layers* earn their reputation in parameter efficiency on
signal-like multivariate data with natural 4-fold structure (speech features, color). Our arms
should test exactly that: does Hamilton-structured mixing buy anything, per parameter, on our
long-memory task? (E2/E3.)

### 1.2 Clifford / geometric-algebra networks (rotors)

- **Clifford Neural Layers for PDE Modeling** (Brandstetter et al., 2022, arXiv:2209.04934): first
  use of multivector fields + Clifford convolutions and **Clifford Fourier transforms** in deep
  learning; "For similar parameter count, Clifford neural layers **consistently improve
  generalization**" on 2D Navier-Stokes, weather modeling, 3D Maxwell. This is the strongest
  measured GA result — and note what the tasks are: PDE fields, i.e., data with genuine
  vector/scalar field structure.
- **CGENN** (Ruhe, Brandstetter, Forré, 2023, arXiv:2305.11141): O(n)/E(n)-equivariant models built
  from the Clifford group; multivectors carry higher-grade components (bivectors etc.) as features.
- **GATr: Geometric Algebra Transformer** (Brehmer et al., 2023, arXiv:2305.18415): inputs,
  outputs and hidden states in projective geometric algebra (16-dim representation of common
  geometric objects and **rotors as operators**); E(3)-equivariant; "consistently outperforms both
  non-geometric and equivariant baselines in terms of error, data efficiency, and scalability" on
  n-body modeling, arterial meshes, robotic motion planning. GATr is the reference architecture if
  we ever make a model whose tokens are geometric (poses, rays, points).
- Ecosystem: Clifford-steerable CNNs (arXiv:2402.14730), Clifford diffusion for molecules
  (arXiv:2504.15773), fast Clifford layers (arXiv:2507.01040), GLGENN (arXiv:2506.09625).
- **What it is not:** none of this demonstrates gains on 2D pixel-grid video models, and the rotor
  representation buys equivariance only when the data and the task carry the symmetry. Our current
  model has no 3D-rotation-covariant structure in its tokens.

### 1.3 Hypercomplex FFTs (the mathematically heavy, practically conditional one)

- The transforms exist in many flavors: two-sided quaternion FT (introduced for 2D linear
  time-invariant systems — Ell 1993, as referenced in arXiv:1306.2157), the unified discrete
  framework via matrix-exponential Euler formula (Sangwine & Ell, 2010, arXiv:1001.4379), fast
  complexified QFT via four complex FFTs (Said, Le Bihan, Sangwine, 2006, math/0603578), steerable
  QFT (Hitzer & Sangwine, 2013, arXiv:1306.2157), quaternion convolution in matrix form (Sfikas &
  Retsinas, 2023, arXiv:2307.01836), dual-quaternion FT (Kenwright, 2023, arXiv:2305.02802),
  octonion FT (Grigoryan & Agaian, 2019, arXiv:1905.12631).
- **The load-bearing caveat**: "A main stumbling block for further applications, in particular
  concerning **filter design in the Fourier domain, is the lack of a proper convolution theorem**"
  for hypercomplex Fourier transforms (Bujack et al., 2013, arXiv:1303.1752). Also: a quaternion
  circulant matrix **cannot** be diagonalized by a quaternion DFT — it is only
  **block-diagonalizable** into complex blocks (Pan & Ng, 2023, arXiv:2302.04086). Practical
  consequence: a quaternion FFT is *computable* (≈2 complex FFTs of the same size), but you cannot
  port "multiply spectra pointwise" filtering naively. For our engine, the operations we do in the
  spectral domain are **compare / measure / blend** (anchor updates, phase correlation, band
  re-weighting) — not convolution — so the caveat bites less there; for the *model*, it bites
  hard.

### 1.4 Hypercomplex state-space models — the empty shelf (novelty check)

Search receipts (Appendix A): `abs:"quaternion" AND (abs:"mamba" OR abs:"selective scan")` → **0
results**; `abs:"hypercomplex" AND abs:"state space"` → **2 results**, both classical time-series
statistics (quaternion irregular autoregressive, arXiv:2609.06866; octonion AR, arXiv:2609.18794);
no ML sequence-model work found. Meanwhile complex-diagonal states are standard in the SSM field
(S4/S4D and Mamba appear in our ENGINE_RESEARCH_SWEEP_2026 §1.3; Liquid-S4, arXiv:2209.12951,
is the canonical complex-diagonal-dynamics reference; our own
`ssm_lite.py` already uses complex poles A = exp(−softplus(a) + i·a_im)). So a quaternion-state
SSM is a *genuine open question*, not a known win. Treat it as research with a kill criterion —
and note the prior: complex eigenvalues already give rotation in one plane; the quaternion upgrade
is rotation in three planes **per state slot**, which is only useful if the data's memory is
rotational. Nothing in our ball/occlusion data says it is; that is exactly what E1 measures.

### 1.5 Dual numbers, dual quaternions, split-complex

- **Dual numbers** are the algebraic vehicle of forward-mode AD (Baydin et al., 2015 survey,
  arXiv:1502.05767; Neuenhofen's hyper-dual review, arXiv:1801.03614; dual-number reverse AD
  arXiv:2205.11368 / arXiv:2507.12640; arbitrary-order AD arXiv:2501.04159). Forward mode is efficient when inputs ≪ outputs:
  Jacobian-vector products, small parameter blocks — e.g., pose/geometry Jacobians.
- **Dual quaternions** own the rigid-transform composition space in robotics: dual-quaternion
  network layers predict rigid-body dynamics (arXiv:2011.08734), hand-eye calibration/SLAM
  (arXiv:2206.14406), SE(3) synchronization (arXiv:2602.00324), survey in arXiv:2303.14765, plus a
  dual-quaternion Fourier transform (arXiv:2305.02802).
- **Split-complex** (hyperbolic numbers): no ML literature found; the arXiv hits are mathematics
  and physics (mostly Minkowski-adjacent). Nothing to map.

### 1.6 Octonions — the honest novelty check

"Deep Octonion Networks" (Wu et al., 2019, arXiv:1903.08478) is essentially the one NN attempt: it
claims "better convergence and higher classification accuracy" than real/complex/quaternion nets
on CIFAR-10/100. Beyond that: octonion phase retrieval (arXiv:2308.15784) and algebra papers.
Weigh against: non-associativity (composition/substitution rules stop being free), 8-real
components, no follow-up literature, no video evidence, no measured advantage over quaternions
anywhere except that one paper's CIFAR table. **Not worth an arm.** (The related trend — 8-dim
hypercomplex "axial" networks, arXiv:2301.04626 — likewise has no mechanism that maps to our
task.)

---

## 2 · Mapping each family to our actual stack (concrete)

### (a) The wave-field model — `wfvideo.py` (`WaveMix3D`, dispersion path)

**Precise current state.** Learnable parameters are **real**; the *field arithmetic* is complex64:
the per-(head, mode) damping/phase/gain parameters (a0, a1, vx, vy, beta, Bg, Cg) generate (i) a
complex transfer `G[nh, L, Hp, Wp]` (`_transfer`, dtype `torch.cfloat`), (ii) complex temporal
poles `lam` (`_dispersion_lam`), and (iii) a complex streaming state
`[B, n_modes, nh, Hp, Wp, dh]` (`init_state`), updated `state = lam_b * state + Bg * x_hat` and
read out `out_hat = (Cg * state).sum(1)` (`step`). Per head/mode the kernel is a damped complex
exponential e^(−αt)·e^(iΩt), i.e. a scalar damped *phase* per mode (one rotation plane, one angle
field Ω(k)), and the modes are mixed by learned gains. So when the card says "complex64 spectral
params", the exact facts are: the *coordinates* are real, the *operators and state* are
complex64, and the kernel's temporal operator per mode is a single complex exponential.

**What a quaternion upgrade would change, concretely.**

1. *Quaternion-structured mixing maps* (cheap, this is E2): replace the real Linear
   `pi`/`po` (dim×dim) with Hamilton-structured block maps over quaternion slots: group each
   head's dh channels into dh/4 quaternion slots; a quaternion linear stores one quaternion (4
   reals) per slot pair instead of a scalar per channel pair — **exactly ¼ the parameters** for
   the same nominal width, with cross-component coupling built in (a·b − c·d etc.). At matched
   total parameters, the saving gets reinvested (FFN grows via the existing ffn_mult bisection) —
   so the experiment asks the real question: *is this parameterization worth its structure on our
   task?*
2. *Quaternion field + quaternion transfer* (expensive, deferred v2): make field values
   quaternion-valued and let the per-mode propagator be a unit-quaternion rotation
   q_m(k)^t·e^(−αt) applied by left-multiplication — 3-plane precession instead of 1-plane phase.
   Implementation-wise this is a 4×4 real (or 2×2 complex) matrix per (k, temporal-freq) cell; the
   math is standard (block diagonalization, arXiv:2302.04086), but it is a new kernel, not a knob,
   and the convolution-theorem caveat (arXiv:1303.1752) means the transfer must be constructed
   carefully (we build G analytically per mode, so this is feasible, but it is real work). **Do not
   build v2 unless E2 shows signal or the owner extends budget** (§4).

3. *What it would NOT buy:* **"rotational equivariance = camera motion for free" is false as
   stated** for this model. (i) Image-plane rotation (roll) equivariance is a 2D-symmetry property
   one can constrain into kernels (steerable/harmonic filters — arXiv:2104.12229 class of ideas;
   for a spectral field, restrict kernels per |k|-shell), but our generators never test roll and
   the product lane authors camera motion explicitly. (ii) 3D camera rotation projects to an image
   homography — perspective, not rotation of the image plane — no equivariant pixel-grid
   architecture hands that to you. (iii) REQNN-style quaternion equivariance is measured on 3D
   point sets, not 2D grids. Write the claim down as: *quaternion structure can couple color/motion
   channels and enrich temporal operators; it does not absorb the 3D-control lane's job.*

### (b) The 3D-control lane (Blender → depth/edge control → Wan 2.2 Fun-Control 4-step → stitch → crumb)

**Current state.** Cameras are Blender quaternions + locations, authored per chunk; control is
depth/edge **video** (fixed clip bounds), not raw poses; a 10-chunk 60 s chain shows zero
cumulative drift slope and no seam pops (`~/.hermes/workspaces/brainsnn/3d_first_pilot/NOTES.md`);
Fun-Camera was eliminated as a full motion path (B-roll only) and FLF2V/HQ/canvas arms were
eliminated by the method ladder (`reports/METHODS_SPACE_2026.md`). Pose-conditioning research
lives elsewhere (CameraCtrl: camera-pose control for video diffusion, arXiv:2404.02101; MotionCtrl,
arXiv:2312.03641) and in our stack pose is **authored data**, not a network input.

**Where quaternion/rotor math genuinely earns its place here (and where it doesn't).**

- *Earns, today, as tooling:* camera-path **continuity invariants**. A camera pose sequence is
  quaternion-valued; the right distance metric between consecutive poses is the geodesic angle
  (2·acos|⟨q1,q2⟩|), and naive linear interpolation across the q ↔ −q double cover produces a
  360° spin artifact; slerp is the correct interpolation. Our chains are long and stitched, so a
  cheap per-chunk **pose-continuity check** (geodesic jumps, twist rate, sign flips at seams) is a
  real guardrail — E5.
- *Does not earn (yet):* rotor-conditioned model inputs. "Pose conditioning via rotors" is a
  principled idea (GATr-style tokenization, rotors as operators, arXiv:2305.18415) but (i) our
  working interface is control video, (ii) Fun-Camera-as-full-path already lost, (iii) quaternion
  I/O has the double-cover discontinuity — "continuous representation" caveats apply (Zhou et al.,
  2018, arXiv:1812.07035 recommends 6D rotation representations precisely for network I/O;
  CameraCtrl's Plücker-style trajectory parameterization is the practical alternative). If we ever
  train a pose-conditioned model, revisit with GATr as reference — Appendix B trigger.

### (c) The engine spectral core (`crumb_coherence/core.py`)

**Current state.** `CoherenceState` holds a complex64 anchor `[C, kh, kw]` (C=3 channels in
YCbCr), a `prev_lowband` box for cut detection, and scalar trajectory state; the estimator
(`estimate_lowband_shift`, plus a `phase_plane` variant) computes a sub-pixel displacement from
the low-band spectrum; corrections are band-selective (radial mask, r≈0.03 hotspot band, default),
and the mode envelope is: `magnitude` safe default; `complex_mc` reachable behind a flag (M3: no
content-blind gate — stop rule stands; `IMPL_NOTES_M3.md` RESULTS). Persistent state: ~1.7 KB @64²,
~32.5 KB @256², ~1.86 MiB @1080p (`reports/ENGINE_RESEARCH_SWEEP_2026.md` §2.3).

**What "hypercomplex FFT = multi-channel coherence in one transform" really offers.**

- The three low-band boxes (Y, Cb, Cr — or R, G, B) are naturally a quaternion field; one QFT
  transforms all four components jointly (computable as ~2 complex FFTs via block diagonalization,
  arXiv:2302.04086; or the complexified route, math/0603578). Per-band statistics like the
  quaternion magnitude are rotation-covariant summaries a per-channel stack cannot express.
- The catch: no proper convolution theorem for hypercomplex FFTs (arXiv:1303.1752) — so this is
  **not** "do multi-channel filtering in one transform". Our engine mostly *measures* (phase
  correlation, variance) and *blends* (band re-weighting), not filters — which is why the one
  testable use is the **estimator**: a color-coupled (quaternion) phase correlation for the
  displacement/trajectory estimate, E4. Its natural consumer is the M3 Plan-C **residual
  anchoring** build (synthesis decision D1), which needs the best possible trajectory estimate to
  separate constant-velocity motion from drift.
- Band-limited caveat continues to apply (grid ≥ 16 sampling fundamental — M0.7 note), and the
  M3 lesson stands: estimator refinements previously *regressed* vs plain phase correlation on
  disjoint-band content — any new estimator must win on the actual removal metric, not just
  estimator error (pre-registered in E4).

### (d) The SSM arm (per-byte comparison to the ENGINE_RESEARCH_SWEEP spec)

**Current state.** `ssm_lite.py`: S4D-style diagonal SSM with **complex poles**
(A = exp(−softplus(a_log) + i·a_im), `state_bytes()` = horizon-independent complex64 state);
it is one of the five R14 arms (A–E) that the 1024-frame occlusion probe evaluates
(`IMPL_NOTES_R14.md`; metrics: `target_identity_survival`, `divergence_horizon`,
`persistent_state_bytes`, console `div_horizon/KB_state`). Measured per-byte-adjacent values from
`runs_hybrid8` (2026-09-26 suite, g32, 8k steps, const-lr recipe): hybrid wave
`persistent_state_bytes` 113,246,208 (108 MiB) with copy_ratio 0.428 / eval_mse_over_copylast
0.947; hybrid ssm 150,994,944 (144 MiB) with copy_ratio 0.718 / 0.566. The occlusion suite itself
has **not yet run on the box** (no `runs_occlusion/` as of this study) — the child card starts
from zero and must run baselines and new arms in the *same* suite for fairness.

**The quaternion-state question, stated honestly.** Our SSM state is already complex (rotation in
one plane per state slot). A quaternion state replaces each complex pole with a unit quaternion
(magnitude for decay × unit quaternion for rotation — three rotational planes per slot) and each
input/output map with quaternion-valued maps. Per *byte*, the state carries 4 real numbers instead
of 2; per *slot pair* the algebra couples components non-commutatively. There is no ML prior art
(§1.4), so the expected value is unknown; the per-byte protocol (`div_horizon/KB_state`) is the
pre-registered currency, and the kill condition from the child card is final: *if no hypercomplex
arm beats both baselines (plain wave and plain ssm) on div_horizon per KB state, hypercomplex
expansion ends.* Design: E1.

### (e) Dual numbers for differentiable depth/geometry passes

**Current state.** The 3D lane renders control passes with Blender (exact, non-differentiable by
design), feeds a frozen Wan generator, and post-processes with crumb (analytic). No gradient flows
to geometry today. Where our stack *does* differentiate: training the wave/ssm arms (reverse-mode
autograd), and nothing else.

**Verdict: no build.** Dual numbers earn where forward-mode JVPs of few-input/many-output maps are
needed (bundle adjustment-style pose refinement, analytic Jacobians of rotor algebra). We do not
refit poses from footage — poses are authored; we do not run differentiable rendering loops; our
autodiff needs are already satisfied. One legitimate *test-time* use if ever needed: verifying
hand-derived quaternion backprop against dual-number/finite-difference checks (a unit test, not a
feature). Dual quaternions similarly have no seam (no skinning, no SE(3) estimation). Say no, keep
the door labeled for a future pose-fitting need (Appendix B).

---

## 3 · The honest filter — earned vs cargo-cult

| Claim | Status | Basis |
|---|---|---|
| Quaternion layers reduce parameter count for matched quality | **EARNED in-domain** (~3.3×–4× on ASR/NLP; QCNN/QUAN on color vision) | §1.1 sources; tested in our setting by E2/E3 |
| Quaternion weights couple components "for free" | **EARNED but bounded** — the coupling is the structure that costs you expressivity elsewhere; shared-score attention result shows not all quaternionizations add capacity | arXiv:2605.24920 |
| Rotational equivariance comes naturally from quaternion features | **EARNED for 3D point-cloud nets under stated conditions** (REQNN) | arXiv:1911.09040 |
| Quaternion/GA models help generic 2D image/video generation | **NOT SHOWN anywhere we found** | Appendix A receipts |
| Hypercomplex FFT = one transform for multi-channel coherence | **PARTLY TRUE** — transforms yes; but no proper convolution theorem (no pointwise filtering), diagonalization only block-wise | arXiv:1303.1752, arXiv:2302.04086 |
| Quaternion state = better long memory per byte | **UNPROVEN, no prior art** — this study's E1 is the experiment | §1.4 receipts |
| Quaternion color (RGB as pure quaternion) helps video processing | **EARNED for matrix methods** (Q-DMD beats DMD; uQRPCA+ SOTA) — neural version untested for us | arXiv:2112.13982, arXiv:2507.19730; E3 |
| Clifford layers improve field-modeling nets at similar params | **EARNED on PDE/weather tasks** | arXiv:2209.04934 |
| GA/rotors help pose/skinning/robotics | **EARNED in robotics** — seamless for us | §1.5, §2(b) |
| Dual numbers unlock differentiable geometry | **ALREADY HAVE** (autodiff); forward-mode niche not hit | §2(e) |
| Octonions are the "next step" for capacity | **CARGO-CULT for us** — one unreplicated line, non-associativity costs, no video evidence | §1.6 |
| Split-complex for ML | **NOTHING FOUND** | Appendix A |
| "Rotation equivariance gives camera motion for free" | **FALSE as stated** (2D plane rotation ≠ 3D camera homography; generators don't test roll) | §2(a) item 3 |

**The NO list for the child card (do not build):** octonions; split-complex; dual numbers / dual
quaternions; GA/rotor architecture for the pixel model; quaternion diffusion conditioning; any
hypercomplex FFT *speed* work (no convolution theorem; and per the optimization gate, kernels are
not the lever on unproven mechanisms).

---

## 4 · Ranked runnable experiments (exact integration points, designs, kill criteria)

Ordering rule: (i) literature-backing × (ii) directness to the per-byte/long-memory wedge ×
(iii) implementation cost. E1–E3 are scoped for the child card (`train_compare.py` arms, real jobs
via `gpu_queue/pending/`); E4–E5 are follow-ups in their own subsystems (see notes).

### E1 (must-run) — `qssm`: quaternion-state SSM at equal persistent bytes

**Question.** Does a quaternion-valued recurrent state (3-plane rotation per slot, non-commutative
maps) buy long-memory recall **per state byte** over the complex-diagonal SSM and the wave arm?

**Integration points.**
- New file `ssm_quat.py` (do **not** edit `ssm_lite.py` — baseline integrity). Class `QuatSSM`
  with the exact `SSMLite` interface: `pi`/`po` Linears; `forward([B,N,D]) -> [B,N,D]`;
  `init_state(B, device)`; `step(x_t, state)`; `state_bytes(B=1, device)`.
- `train_compare.py`: extend `--kind` choices with `"qssm"`; dispatch in
  `VideoPredictor.__init__`; add `"qssm"` to `_streamable()`; make sure `stream_init`,
  `persistent_state_bytes`, `stream_step` route through the same mix interface (they already are
  kind-agnostic).
- `run_occlusion.sh`: append one arm entry to `ARMS` (keep A–E untouched): `"F_qssm|qssm|"` (plus
  optionally `--const-lr` in BASE per the unfreeze rule, §5).

**Design (reference).**
- Layout: state `S` in H^(B × HW × nh × dhq × ds) stored float32 with a trailing component axis
  `[..., 4]`; **dhq = dh // 2** (dh = dim/heads ⇒ 24 → 12), ds = 16, nh = 8 — chosen so
  `state_bytes()` equals `SSMLite`'s by construction: per (HW, nh, ds) cell,
  `16·dhq = 8·dh` bytes (16 B per quaternion float32 vs 8 B per complex64). Assert this in a test.
- Dynamics: poles `A = m ⊙ U` with magnitude `m = exp(−softplus(a_log))` (init ≈ 0.05 like
  SSMLite) and unit quaternion `U` per (nh, ds) (init: fixed angle spread linspace(0, π, ds) about
  per-head fixed random axes — mirrors SSMLite's `a_im` linspace). Update
  `S ← A ⊙ S + B ⊗ u` (⊙ slot-wise Hamilton product, ⊗ Hamilton product), readout
  `y = Σ_ds C ⊗ S` taking all four quaternion components; `pi: dim→2·dim`, `po: 2·dim→dim`
  (reals packed/unpacked to quaternion groups), feedthrough D per component as in SSMLite.
- Training: direct unrolled recurrence over T=17 (no FFT needed at this T); Hamilton products
  implemented as pure torch real ops on the component axis (autograd handles gradients — **no
  custom HR-calculus/backprop machinery**; that literature is for bespoke learning rules we don't
  need).
- Invariants: (1) `forward` last-frame output == `step` chain output within 1e-5; (2) byte
  equality formula asserted; (3) existing `sanity_check.py` unchanged; (4) default paths of all
  existing arms byte-identical (additive change only).

**Protocol.** Equal-bytes comparison via the suite's own accounting: report
`persistent_state_bytes`, `divergence_horizon`, `div_horizon/KB_state`, `ms/frame`
(`rollout_fps`), `mem_gb_peak`; ≥3 seeds; the fixed eval seed; baselines (A–E) from the same
suite run, not from history.

**Kill.** If `qssm` does not beat **both** `wave` and `ssm` (C and B) on `div_horizon/KB_state`
(median across seeds) — document, stop, no further hypercomplex model expansion (child-card rule).

**Cost.** One mixer class + flags + one arm entry; training cost ≈ SSMLite order (T=17 unrolled;
Hamilton products are ~4 real multiplies per slot pair).

### E2 (must-run) — `qwave`: quaternion-structured mixing in the wave arm

**Question.** Do Hamilton-structured input/output maps (¼ the free parameters at equal nominal
width, cross-component coupling for free) improve the wave arm at matched total parameters — i.e.,
does the quaternion *parameterization* earn its place in our spectral core?

**Integration points.**
- New file `wfvideo_quat.py` (keep `wfvideo.py`'s numeric paths untouched): class `WaveQuatMix`
  mirroring `WaveMix3D`'s dispersion path exactly (kernel/transfer/state identical), but with
  `pi`/`po` as `QuaternionLinear`: split dim into nh heads × dh channels → dh/4 quaternion slots
  per head; each slot pair (in,out) stores a quaternion (4 reals) instead of a scalar — ¼ the
  params of a real `Linear(dim, dim)`. Optional `--q-ffn` extends the same to the block FFN
  (default off).
- `train_compare.py`: add a `--q-mix` flag (BooleanOptionalAction, default off) accepted by
  `--kind wave`; dispatch to `WaveQuatMix` when set; **assert the default path stays
  byte-identical** (flag off ⇒ existing class, no new tensors).
- `run_occlusion.sh`: append `"G_qwave|wave|--kernel-version dispersion --q-mix"` to `ARMS`.

**Protocol.** Same suite/suite-run requirement as E1. Because the parameter saving is reinvested
by the existing `--target-params` bisection, the honest summary is "same total params, different
allocation + structure". Secondary diagnostic (optional, cheap): one extra arm at
`--target-params 3000000` to probe "matched quality at fewer params" directly.

**Kill.** No beat on `div_horizon/KB_state` vs both baselines → log as a negative; keep the flag
behind the no-op default (same discipline as prior falsified mechanisms: revert default, keep code,
record).

### E3 (third arm) — `qcolor`: quaternion color coupling at the stem

**Question.** Does treating RGB as a pure quaternion at the model's input boundary (hue-structure
prior; the Zhu-style color representation) help the occlusion probe — whose identity metric is
*chroma*-weighted (`relu(dominant channel − luminance)`, `IMPL_NOTES_R14.md`)?

**Integration points.**
- In `wfvideo_quat.py`: `QuatEmbed` — replace `self.embed = nn.Conv2d(3, dim, 3, padding=1)` with
  a quaternion conv: frame pixels as pure quaternion (0, r, g, b) per pixel → `dim/4` quaternion
  channels, 3×3 Hamilton-structured kernels (≈ 9·dim params vs 27·dim real — cheaper); head stays
  `Linear(dim, 3)`; residual add unchanged.
- `train_compare.py`: `--quat-color` flag (default off; byte-identical default); applies to
  `--kind wave`.
- `run_occlusion.sh`: `"H_qcolor|wave|--kernel-version dispersion --quat-color"`.

**Honest note.** Multiplication by a unit quaternion rotates the RGB vector about an axis through
the gray diagonal — the standard "color rotation" of quaternion color processing; it is a smooth,
structure-preserving transform of color space (not identical to HSL hue for all colors; do not
claim "exact hue equivariance" in any write-up). The motivation is the measured color-video
literature (§1.1, §1.5) plus our chroma-based identity metric.

**Kill.** Same protocol and kill as E1/E2. If E3 and E2 both show nothing, the quaternion
*parameterization* family is closed for this model with a documented negative.

### E4 (follow-up, engine track) — quaternion phase-correlation estimator `mc_est="quat_corr"`

**Not in the child card's `train_compare.py` scope; recommended as its own small card, gated on
the model verdict per the stop rule.** Consumer: the M3 Plan-C residual-anchoring build
(synthesis D1) which needs the best trajectory estimator.

**Integration points.**
- `crumb_coherence/core.py`: add `estimate_lowband_shift_quatcorr(Xlo, anchor, ...)` (the three
  complex low-band boxes treated as a complexified quaternion field; conjugate-left multiply,
  inverse transform, quaternion-magnitude correlation surface, 2D parabolic sub-pixel peak) and
  the `mc_est="quat_corr"` choice; default stays `"corr"` (byte-identical no-op).
- Validation: `crumb_coherence/scripts/run_m0.py`, `run_m05.py`, `run_trap.py`, `run_eng1_1080.py`;
  same frozen windows/thresholds.

**Kill (pre-registered).** Displacement error reduction ≥20% vs `corr` on ≥2 of {dc, hotspot,
ENG-1} **and** no regression on the M0/ENG-1 removal-fidelity bars **and** trap bars unchanged
**and** ≤1.2× wall per frame → adopt into the D1 build; else revert default to `corr`, keep code
behind the flag, log the falsification. (The M3 lesson: flat phase-pull and every estimator
"refinement" regressed on disjoint-band content — this must win on the actual removal metric, not
estimator error alone.)

### E5 (follow-up, 3D lane tooling) — quaternion camera-path continuity QA

**Not a model arm; a guardrail for the authored-motion lane.** Fold into the next 3D-chain task
rather than a standalone card.

**What.** `check_pose_continuity.py` over the per-chunk camera exports (quaternion + location):
per-step geodesic angle `2·acos|⟨q_t, q_{t+1}⟩|`; twist rate relative to path tangent; seam
metrics at chunk boundaries (geodesic jump, angular-velocity discontinuity); **sign-flip
detection** (q vs −q are the same pose; lerp across a flip = 360° artifact — the classic
double-cover bug). Run on the existing 10-chunk 60 s chain.

**Kill / success.** If the chain passes with no flags: keep as a regression invariant (near-zero
cost, prevents future flips) and stop. If a real discontinuity is flagged: fix by slerp
reparameterization of the affected segment, re-render that chunk, document the before/after — that
result is a concrete deliverable of the study's (b) mapping.

### Deferred-v2 list (conditions, not now)

- **Quaternion field + quaternion transfer in the wave kernel** (3-plane precession): build only
  if E2 shows signal or owner extends budget; spec sketch in §2(a).2.
- **GATr-style rotor tokenization / pose-conditioned model**: trigger = we start training a
  model that consumes poses/rays/points (not pixels). Reference architectures: arXiv:2305.18415,
  arXiv:2305.11141.
- **Quaternion/geometric conditioning of a diffusion/refiner stage** (ResQu template,
  arXiv:2505.00334): trigger = we add a learned refiner where conditioning is a feature vector,
  not a control video.
- **Quaternion attention blocks**: only if the campaign revives the attention arm; the shared-score
  result (arXiv:2605.24920) says expect re-parameterization, not new capacity.

---

## 5 · Protocol & ops for the child run (so the comparison is honest)

1. **The suite is the instrument; run it once, whole.** Baselines (A–E) and new arms (F/G/H) in
   the SAME `run_occlusion.sh` invocation (same session, same thermal state, same seeds), not
   against historical numbers. Seed-outer loop already guarantees arm/seed coverage on partial runs.
2. **Budget and the unfreeze rule.** The R14 suite defaults to `STEPS=2000` and predates the
   freeze verdict; the campaign rule is *budget × schedule are a pair*: scaling steps while
   leaving LR annealing welded to `total_steps` starves the model and "freezes everything
   identically" (`HARDENING_DIGEST.md`, `DEEP_DIVE_3_opus.md`). Pre-flight before the full suite:
   one cheap 2k-step wave probe **with `--const-lr`**; if its `copy_ratio` indicates a frozen
   (copy-last) regime (≈0.02–0.1; reference: unfrozen wave reached ≈0.73 at 8k const-lr on g32),
   escalate the suite to `--const-lr` + `STEPS=8000` for ALL arms. Record the choice and the probe
   numbers in the run notes. Never compare arms at budgets where every arm is frozen.
3. **Read verdicts from result JSONs, never from log tails.** Per arm: `copy_ratio` (freeze
   diagnostic), `divergence_horizon`, `persistent_state_bytes` and the printed
   `div_horizon/KB_state`, `rollout_fps`/`ms_per_frame`, `mem_gb_peak`; median across seeds; MSE
   columns are diagnostics only (pixel MSE rewards staleness — standing campaign rule).
4. **Queue mechanics.** Real jobs via `/workspace/slava/gpu_queue/pending/` (plain bash scripts;
   the conductor runs one at a time; never two at once; never as fillers). Check for orphaned
   CUDA contexts before a big arm (`nvidia-smi --query-compute-apps`); watch for `AUTO-BATCH`
   lines before assuming a stall.
5. **Code hygiene.** Additive-only changes to existing files (new flags default off; default
   paths byte-identical — assert in smoke); new modules for new classes; run the repo smokes
   (`run_smoke_r14.sh` pattern) and `sanity_check.py` before any suite; commit code + results
   table to `crumb-format`; scp the complete file set to the box and verify md5 both ends.
6. **Honest labels.** These arms are *research preview* items; no external claim, number, or demo
   built on them until they pass the pre-registered bars. The campaign claim boundary in
   `THE_POSITION.md` is unchanged by anything in this study.

---

## 6 · Sources (verified in this session via the arXiv API; citation counts approximate,
OpenAlex scrape 2026-09-27, shown only where they match cleanly)

Quaternions — networks / layers:
- Parcollet et al., *Quaternion Recurrent Neural Networks*, arXiv:1806.04418 (2018) — 3.3× fewer params, better ASR. [~59 cites]
- Parcollet et al., *Quaternion Convolutional Neural Networks for End-to-End ASR*, arXiv:1806.07789 (2018).
- Zhu et al., *Quaternion Convolutional Neural Networks*, arXiv:1903.00658 (2019) — color-image QCNN. [~196 cites]
- Gaudet & Maida, *Deep Quaternion Networks*, arXiv:1712.04604 (2017) — init, batch-norm; CIFAR/KITTI. [~200 cites]
- Trabelsi et al., *Deep Complex Networks*, arXiv:1705.09792 (2017) — complex building blocks (the immediate predecessor). [~167 cites]
- Tay et al., *Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks*, arXiv:1906.04393 (2019) — "up to 75% reduction in parameter size"; Quaternion Transformer/attention.
- Grant & Wang, *Quaternion Approximation Networks for Enhanced Image Classification and Oriented Object Detection*, arXiv:2509.05512 (2025) — higher accuracy with fewer params (their claim).
- Shen et al., *3D-Rotation-Equivariant Quaternion Neural Networks*, arXiv:1911.09040 (2019).
- Mandic et al., *The HR-Calculus: Enabling Information Processing with Quaternion Algebra*, arXiv:2311.16771 (2023).
- Pöppelbaum & Schwung, *Quaternion Backpropagation*, arXiv:2212.13082 (2022); *Quaternionic Activation Functions*, arXiv:2406.16481 (2024).
- *Quaternion Self-Attention with Shared Scores*, arXiv:2605.24920 (2026) — component-wise attention mostly re-parameterizes.
- Valle & Lobo, *Quaternion-Valued Recurrent Projection Neural Networks*, arXiv:1909.09227 / 2001.11846 (2019-2020); Granero et al., *QCNN for ALL diagnosis*, arXiv:2112.06685 (2021); Miao, Kou et al., *Quaternion Matrix Completion for Color Inpainting*, arXiv:2305.00416 (2023); Nguyen et al., *Quaternion Graph Neural Networks*, arXiv:2008.05089 (2020); Qiu et al., *QNN for multi-channel distant speech*, arXiv:2005.08566 (2020); survey: Altamirano-Gómez & Gershenson, *Quaternion Convolutional Neural Networks: Current Advances and Future Directions*, arXiv:2307.08663 (2023).

Quaternion color-video processing:
- Han, Kou, Miao et al., *Quaternion-based dynamic mode decomposition for background modeling in color videos*, arXiv:2112.13982 — "Q-DMD outperforms the exact DMD method" (their wording), comparable to SOTA.
- Wang, Wu, Fang, *Quaternion-Based Robust PCA for Efficient Moving Target Detection and Background Recovery in Color Videos*, arXiv:2507.19730 — "uQRPCA+ achieves State Of The Art (SOTA) performance on moving target detection and background recovery tasks" (verbatim).

Clifford / geometric algebra:
- Brandstetter et al., *Clifford Neural Layers for PDE Modeling*, arXiv:2209.04934 (2022) — consistently improves at similar params. [~22 cites]
- Ruhe et al., *Clifford Group Equivariant Neural Networks*, arXiv:2305.11141 (2023).
- Brehmer et al., *Geometric Algebra Transformer*, arXiv:2305.18415 (2023) — rotors, E(3) equivariance.
- Zhdanov et al., *Implicit Convolutional Kernels for Steerable CNNs*, arXiv:2212.06096 (2022); *Clifford-Steerable Convolutional Neural Networks*, arXiv:2402.14730 (2024); *Fast Clifford Neural Layers*, arXiv:2507.01040 (2025); Liu et al., *Clifford Group Equivariant Diffusion Models for 3D Molecular Generation*, arXiv:2504.15773 (2025).

Hypercomplex FFTs:
- Sangwine & Ell, *Complex and Hypercomplex DFTs Based on Matrix Exponential Form of Euler's Formula*, arXiv:1001.4379 (2010).
- Bujack et al., *Convolution products for hypercomplex Fourier transforms*, arXiv:1303.1752 (2013) — the missing convolution theorem.
- Said, Le Bihan, Sangwine, *Fast complexified quaternion Fourier transform*, math/0603578 (2006).
- Sfikas & Retsinas, *On the Matrix Form of the QFT and Quaternion Convolution*, arXiv:2307.01836 (2023).
- Pan & Ng, *Block Diagonalization of Quaternion Circulant Matrices*, arXiv:2302.04086 (2023).
- Hitzer & Sangwine, *The Orthogonal 2D Planes Split of Quaternions and Steerable Quaternion Fourier Transformations*, arXiv:1306.2157 (2013); Kenwright, *Dual-Quaternion Fourier Transform*, arXiv:2305.02802 (2023); Błaszczyk, *A Generalization of the Octonion Fourier Transform to 3-D Octonion-Valued Signals*, arXiv:1905.12631 (2019).

SSMs (the per-byte bar, already in our sweep doc — anchor citations):
- Gu et al.: HiPPO arXiv:2008.07669; S4 arXiv:2111.00396; S4D arXiv:2206.11893; Mamba (Gu & Dao) arXiv:2312.00752; Hasani et al., Liquid-S4 arXiv:2209.12951.

Dual numbers / dual quaternions:
- Baydin et al., *Automatic differentiation in machine learning: a survey*, arXiv:1502.05767 (2015).
- Neuenhofen, *Review of theory and implementation of hyper-dual numbers for first and second order automatic differentiation*, arXiv:1801.03614 (2018); Peñuñuri et al., *Dual Numbers for Arbitrary Order AD*, arXiv:2501.04159 (2025); dual-numbers reverse AD arXiv:2205.11368 / arXiv:2507.12640.
- Pöppelbaum & Schwung, *Predicting Rigid Body Dynamics using Dual Quaternion Recurrent Neural Networks with Quaternion Attention*, arXiv:2011.08734 (2020); *Standard Dual Quaternion Optimization* (hand-eye calibration & SLAM), arXiv:2206.14406 (2022); *Dual Quaternion SE(3) Synchronization with Recovery Guarantees*, arXiv:2602.00324 (2026); Kenwright, *A Survey on Dual-Quaternions*, arXiv:2303.14765 (2023).

Octonions:
- Wu et al., *Deep Octonion Networks*, arXiv:1903.08478 (2019).
- *Octonion phase retrieval*, arXiv:2308.15784 (2023).

Pose / camera conditioning (context for the 3D lane):
- He et al., *CameraCtrl*, arXiv:2404.02101 (2024); Wang et al., *MotionCtrl*, arXiv:2312.03641 (2023).
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, arXiv:1812.07035 (2018).
- Wan et al., *Wan: Open and Advanced Large-Scale Video Generative Models*, arXiv:2503.20314 (2025).

Internal (committed campaign docs this study builds on):
- `IMPL_NOTES_M3.md` (gate RED, margin 0.002; trap 54.7/70.8% retention; band-cut hotspot 64.0%/0.9946; falsified refinements), `reports/ENGINE_RESEARCH_SWEEP_2026.md` (§2 SSM per-byte spec + engine byte table; §1.3-1.4 state-size literature), `reviews/ENGINE_SYNTHESIS_v2.md` (D1 residual anchoring BUILD; product boundaries), `IMPL_NOTES_R14.md` (occlusion probe; `div_horizon/KB_state`; state-byte accounting), `reports/METHODS_SPACE_2026.md` (eliminated arms incl. Fun-Camera/FLF2V/HQ), `~/.hermes/workspaces/brainsnn/3d_first_pilot/NOTES.md` (60 s zero-drift chain), `HARDENING_DIGEST.md` / `DEEP_DIVE_3_opus.md` (freeze diagnosis; budget × schedule rule).

---

## Appendix A · Search receipts (negative results, same session)

- `abs:"quaternion" AND (abs:"mamba" OR abs:"selective scan")` → **0**. No quaternion SSM/Mamba line in ML.
- `abs:"hypercomplex" AND abs:"state space"` → **2**, both statistics time-series (arXiv:2609.06866 H-IAR quaternion irregular AR; arXiv:2609.18794 octonion AR). No sequence-modeling prior art.
- `abs:"quaternion" AND abs:"state space"` → 26 hits, none a quaternion neural SSM (math/control/stats).
- `abs:"split-complex" OR abs:"split complex numbers"` → 81 hits, all pure math/physics; **no ML**.
- `ti:"Clifford" AND ti:"equivariant"` → 9 hits: CGENN lineage only (all geometric-graph/point data; no pixel-video).
- `abs:"rotor" AND (abs:"neural network" OR abs:"transformer")` → dominated by aircraft-rotor acoustics; **no pose-rotor conditioning line** (the rotor-as-operator literature is GATr and the Clifford group papers).
- `abs:"quaternion" AND abs:"video"` → color-video matrix analysis (DMD/RPCA/completion/watermarking); **no quaternion video *generation* model**.
- `abs:"quaternion" AND abs:"attention"` → ~2 usable lines (QEAN dance-generation attention; shared-score attention 2026).
- Octonion NN line: essentially one paper (1903.08478) + phase retrieval.

## Appendix B · Falsifiers / triggers (what would change the verdicts)

- E1–E3 negative across the board → close the hypercomplex model line per the stop rule; keep
  `--q-mix`/`--quat-color`/`qssm` code behind default-off flags; cite this study in the negative
  record.
- E2 or E3 positive → unlock the deferred v2 (quaternion field + quaternion transfer, §2(a).2) and
  consider a wider sweep of the parameter-efficiency question (params-reduced arms).
- Engine E4 positive → wire into the D1 residual-anchoring build; negative → keep `corr` default.
- Any future model that consumes poses/rays/points, or trains inside a differentiable geometry
  loop → GATr-style rotor tokenization and dual-number/forward-mode tooling become live again
  (§1.2, §2(e)).
- Any future learned refiner with feature-vector conditioning → revisit quaternion/geometric
  conditioning (ResQu template).
