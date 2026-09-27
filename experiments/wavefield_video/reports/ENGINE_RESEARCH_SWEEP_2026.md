# ENGINE RESEARCH SWEEP 2026 — prior art, the SSM per-byte bar, and an honest benchmark protocol

Internal research note · 2026-09-27 · kanban t_fc77f3bd · companion to the engine deep-dive v2 consults
(`reviews/engine_deepdive_v2_task.md` sent to Astra + Opus 2026-09-27 03:22 UTC).
Watch-gate: nothing in this document is public. Sources are linked; quoted numbers are as-reported by
the cited work unless marked "campaign-verified" (our own runs).

Read with: `IMPL_NOTES_M0_ADDENDUM.md`, `IMPL_NOTES_M2.md`, `reviews/claude_m3_gate_task.md`,
`reports/ENG1_1080.*`, `THE_POSITION.md`, `~/.hermes/workspaces/brainsnn/3d_first_pilot/NOTES.md`.

---

## 0 · TL;DR for the consult synthesis

1. The forcing family has converged on **memory policies over the KV cache**, and three of its ideas
   land directly on our roadmap: **anchor caches of early frames applied selectively** (FreqForcing),
   **trusted-alignment editing of recalled memory** (= our anchor, independently invented; TetherCache
   TAME), and **measure the context-response profile before redesigning memory** (Recency Forcing).
2. **SNF-Bench (Aug 2026) mechanically validates our measurement philosophy**: whole-frame metrics
   *reward* injected drift (Dynamic Degree 1.07× at max corruption while drift-specific factors rise
   1.32–1.86×). Its injection protocol is the template for validating our own instruments — adopt it
   before any public number.
3. The **coherence half-life** idea already has public precedent: VBench issue #206 proposes a
   long-horizon DINO Δt-curve (frame pairs at Δt = 2–50 frames) with a small human-alignment study —
   and shows Veo3 *highest at Δt=2, lowest overall*. Our headline metric should be the half-life
   **of that curve** (defined in §3.1), reported with motion metrics beside it.
4. **SSM per-byte bar**: the literature backs "state bytes bind recall" (TTT-video: Mamba states
   "small and less expressive"; ACL'25: linear-attention state 16× smaller than attention at d=128/L=2048;
   StateX: recall improves with state expansion). The middleware baseline is specified in §2: same task,
   same correction blend, **equal complex64 state bytes**, one diagonal S4D-style tracker. Kill criteria
   pre-registered (§2.4). Model-track bar = R14's occlusion probe, restated there.
5. **Publish first** = a neutral half-life *audit* of frontier generators (methodology + curves, no
   product claims), then one frozen-settings third-party before/after. Both under watch-gate; owner
   approves every outbound artifact (THE_POSITION rule).
6. Roadmap-changers from the last 90 days: **Recency Forcing** (response-profile methodology),
   **SNF-Bench** (metric validation by injection), **MARCH** (content-routed state anchors — the
   architecture our anchor-cache plan resembles, now with a published precedent), **FreqForcing**
   (spectral anchoring in mainstream diffusion), **TetherCache** (TAME).

---

## 1 · Prior-art sweep (last ~6 months; 5 older anchors marked)

Scope note: web-level sweep (abstracts + project pages), Mar–Sep 2026 with older items included only
where the campaign already builds on them. No claim below has been replicated by us unless marked.

### 1.1 The forcing lineage — long-horizon AR video diffusion

- **Self Forcing** (Jun 2025 → NeurIPS'25) — https://arxiv.org/abs/2506.08009 · https://self-forcing.github.io/
  *Mechanism:* train-time autoregressive rollout with KV caching closes the train/test gap; near-real-time
  streaming on one 4090. *Steal:* the canonical train-test-gap framing for our self-rollout decision tree
  (already adopted); their KV-rollout discipline.
- **Self-Forcing++** (Oct 2025 → ICLR'26) — https://arxiv.org/abs/2510.02283
  *Mechanism:* supervision from segments **sampled out of the student's own long self-generated videos**;
  scales 20× beyond the teacher; up to 4m15s; avoids over-exposure/error accumulation *without recomputing
  overlapping frames*. *Steal:* (a) "sample segments from your own long rollouts" is free long-horizon
  supervision — the cleanest published recipe for the exposure-bias arm; (b) the no-recompute overlap
  note is a stitching discipline for our P3 60s chain; (c) their improved long-horizon benchmark.
- **Deep Forcing** (Dec 2025 → ICML'26) — https://arxiv.org/abs/2512.05081 · https://cvlab-kaist.github.io/DeepForcing/
  *Mechanism:* Deep Sink + Participative Compression, training-free; 12× extrapolation (5s → 60s+). Reports
  naive StreamingLLM-style sinks cause fidelity degradation + motion stagnation. *Steal:* the
  sink-length/compression tradeoff; **caution**: FreqForcing reports this style of KV compression causes
  *spectral jitter* (flicker) — our own M0/M1 work agrees that compression changes spectral behavior and
  must be measured for band side-effects.
- **Rolling Forcing** (ICLR'26) — https://arxiv.org/abs/2509.25161 · https://github.com/TencentARC/RollingForcing
  *Mechanism:* rolling-window joint denoising (current + future-noise frames denoised together); real-time
  multi-minute streams, minimal error accumulation. *Steal:* rolling-denoise mechanics if we ever go
  in-loop; their drift instrumentation.
- **Sparse Forcing** (Apr 2026) — https://arxiv.org/abs/2604.21221
  *Mechanism:* attention concentrates on a *persistent subset of salient blocks* (implicit spatiotemporal
  memory); trainable sparse attention reduces decoding latency (as-reported). *Steal:* block-salience memory — we already
  compute low-band energy per cell; a salience channel could drive cut/revisit logic cheaply.
- **Reward Forcing** (CVPR'26 Highlight) — https://github.com/JaydenLyh/Reward-Forcing
  *Mechanism:* reward-weighted distillation (vs vanilla DMD) into a 4-step AR student, 23.1 FPS. *Steal:*
  distillation-with-reward recipe for the model track (later; the freeze first).
- **Relax Forcing** (BMVC'26) — https://arxiv.org/abs/2603.21366
  *Mechanism:* replace dense history with a *functional* decomposition — **Sink** (global stability),
  **Tail** (short-term continuity), **selected History** (motion guidance); attention length 21→7 latent
  frames; VBench-Long. *Steal:* the functional-role vocabulary maps onto our state regions (anchor =
  sink/stability; recent context = tail; motion-bearing past = selected history). "Selected history for
  motion" is exactly our band-split logic — low-band held, motion free.
- **Recency Forcing** (Sep 17 2026 — newest) — https://arxiv.org/abs/2609.19729
  *Mechanism:* names the **KV eviction mismatch** (train on full context, infer with evictions);
  introduces a perturbation-based **positional response** R(Δt, denoise-step) showing context influence
  decays steeply with temporal distance and varies across denoise steps; applies **Temporal Response Bias**
  on pre-softmax logits (exact zero-overhead reformulation via BAR); training-free + training modes; SOTA
  long-horizon on VBench(-Long). *Steal:* (a) **measure the response profile before changing context
  mechanics** — this is what our M0 rho sweep started; make it systematic (correction strength vs Δt);
  (b) fade-don't-cut: distant context kept but de-weighted — the principle behind our slow anchor;
  (c) their perturbation methodology as an instrument recipe.
- **TetherCache** (Jun 2026) — https://arxiv.org/abs/2606.13035
  *Mechanism:* training-free cache policy; regions sink/memory/recent; **GRAB** (gated recall: attention
  relevance + temporal diversity) picks long-range frames; **TAME** aligns recalled memory-token
  statistics to a *trusted distribution*, de-polluting drifted features; quality drift 7.84 → 1.33 at
  240s (VBench-Long 30/60/240). *Steal:* **TAME is our anchor mechanism at token level** — independent
  confirmation that statistics-alignment of a trusted reference is the right fix and survives 240s;
  GRAB's **diversity term**: our anchor cache should keep *diverse* references (geometric spacing,
  cf. CSSC), not just recent ones.
- **Memento** (Jun 2026) — https://arxiv.org/abs/2606.14667
  *Mechanism:* memory quality enforced by a **subject-reconstruction** objective (a memory bank must
  support reconstructing the subject from memory alone); dual-query memory (identity vs short-context).
  *Steal:* "memory must reconstruct" as an anchor-health test: can our anchor reconstruct the start-window
  low-band? A cheap canary for anchor staleness/corruption.
- **Memorize When Needed** (Apr 2026) — https://arxiv.org/abs/2604.18215
  *Mechanism:* decoupled memory branch; per-frame cross-attention to the most spatially relevant history;
  **camera-aware gating** — "memory conditioning only when meaningful historical references exist."
  *Steal:* direct prior art for M3-style gating: condition the correction on the *informativeness of
  history*; also revisit handling ("only when meaningful references exist" = our cut/reset logic).
- **A2RD + Google's long-form framework** (May 2026 + Google Research blog, Sep 24 2026) —
  https://arxiv.org/abs/2605.06924 · https://research.google/blog/coherent-long-form-video-generation/
  *Mechanism:* agentic retrieve-synthesize-refine memory loop; interpolation↔extrapolation mode switch at
  boundaries; 10-minute multimodal movie; benchmark LVBench-C. *Steal:* the mode switch generalizes our
  cut detector (revisits = interpolation against stored state; wander = extrapolation); LVBench-C's
  gap-based evaluation rules.
- **MilliVid** (Jun 2026) — https://arxiv.org/abs/2606.09056 · https://davidcharatan.com/millivid/
  *Mechanism:* hierarchical multi-scale tokens; coarse-to-fine rollout; the **coarsest levels carry
  layout/semantics** across long range (consistency in geometry/object permanence) while detail levels
  cost less; Minecraft long-video dataset. *Steal:* explicit hierarchy for the model track; independent
  support for "spend the long-range budget on the coarse/low-band content" — our operating principle.

### 1.2 Spectral / frequency-domain work

- **FreqForcing** (Jul 2026) — https://arxiv.org/abs/2607.27110 · https://jiatongli2024.github.io/freqforcing.github.io/
  *Mechanism (campaign-verified summary in `SPECTRAL_PRIOR_ART.md`):* training-free **Spectral
  Self-Anchoring** — dual attention branches fused by a Gaussian low-pass (λ=0.6, σ=0.125; anchor cache =
  6 early frames; applied only at the first 2 denoising steps; +16.5% latency); 24× extrapolation (5s →
  2min); best Dynamic Degree at 60s/120s on VBench-Long. *Steal:* headline validation of the thesis;
  actionable extras: (a) apply low-band anchoring at *early denoise steps only* (compute saving if we go
  in-loop); (b) anchor-cache of early frames; (c) their band-energy drift diagnostics — wire per-band
  energy telemetry into our evals (already planned as E1).
- **Spectral State Space Models** (2023, foundational) — https://arxiv.org/abs/2312.06837
  *Mechanism:* SSM formulation via spectral filtering of long-range dependencies; learnable linear
  dynamical systems. *Steal:* theoretical home for our k-space recurrence — our dispersion kernel is a
  hand-built spectral filter bank; `ssm_lite.py`'s learned poles are its generic cousin. Cite in the
  model write-up; keeps our claims in established language.
- **HFMamba** (Jul 2026) — https://link.springer.com/chapter/10.1007/978-981-92-3420-2_1
  *Mechanism:* frequency-enhanced SSM (image restoration). *Steal:* SSM+frequency hybrids are now
  mainstream-adjacent; supports the (model-track) hybrid direction; domain caveat (restoration ≠ video).
- **FoSS** (OpenReview) — https://openreview.net/forum?id=icsrG0ijaA
  *Mechanism:* Fourier decomposition where **amplitude = global intent, phase = local variation**, with
  SSM submodules over each. *Steal:* external language for our "anchor magnitude, preserve phase" rule —
  amplitude holds the regime, phase carries the motion. Useful in the next write-up.

### 1.3 SSM / state-size literature (why the per-byte bar is the right bar)

- **Long-Context State-Space Video World Models** (ICCV'25) — https://arxiv.org/abs/2505.20171
  *Mechanism:* SSM memory (write/erase) for long-context video world models. *Steal:* baseline
  architecture reference; memory-write mechanisms as a comparison point for our anchor EMA.
- **One-Minute Video Generation with Test-Time Training** (ICML'25; IEEE 11095233) —
  https://arxiv.org/abs/2504.05298 · https://test-time-training.github.io/video-dit/
  *Mechanism:* TTT layers as "hidden states that are neural networks"; beats Mamba-2, Gated DeltaNet and
  sliding-window on minute-scale coherence; explicit statement: **"Mamba layers struggle to produce
  coherent scenes because their hidden states are small and less expressive."** *Steal:* (a) the quote is
  the field's own admission that fixed small states have coherence limits — use it to justify the SSM bar
  in §2; (b) TTT cells as a third "bigger-state" comparator for the R14 model track; (c) their storyboard
  benchmark protocol.
- **Scaling up the State Size of RNN LLMs** (ACL'25) — https://aclanthology.org/2025.acl-long.564.pdf
  *Mechanism:* linear-attention state is **16× smaller** than self-attention for d=128, L=2048; recall
  tasks scale with state size. *Steal:* quantitative grounding for the per-byte protocol rationale
  (state bytes ↔ recall capacity).
- **StateX** (Findings ACL'26) — https://aclanthology.org/2026.findings-acl.1073.pdf
  *Mechanism:* post-training state expansion improves recall. *Steal:* fairness lever — give the SSM
  baseline its best shot (expanded states) before claiming anything.
- **MARCH** (Aug 2026) — https://arxiv.org/abs/2608.12435
  *Mechanism:* **content-routed state anchors** — per-token anchor queries attending over all causally
  available anchors; the memory bank grows with context at controllable cost, without leaving the
  recurrent path; beats linear-attention variants on LongBench/ICL retrieval. *Steal:* published
  precedent for our anchor-state streaming plan (E3: K anchor states, geometric spacing) — and a naming
  anchor ("state anchors") that reviewers recognize.

### 1.4 Camera / 3D-control long-clip work (feeds the 3D-first stack)

- **CamDirector** (CVPR'26) — https://arxiv.org/abs/2603.02256
  *Mechanism:* world cache (static-content fusion) + history-guided AR diffusion for long-term coherent
  trajectory edits; dedicated benchmark. *Steal:* world-cache idea = geometry-side memory that our
  Blender/depth-control pipeline can compute exactly, not estimate.
- **Closing the Loop: Training-Free Revisit Consistency** (Jul 2026) — https://arxiv.org/abs/2607.21848
  *Mechanism:* loop-closure via pose-matched historical latents + spatial correspondence bias; training-
  free revisit consistency. *Steal:* **revisit-consistency as a metric**, and loop-closure retrieval; for
  our 3D-first chain we own exact poses + depth from Blender — strictly stronger correspondences than
  their estimation, near-free to implement.
- **OmniRoam** (SIGGRAPH'26) — https://github.com/yuhengliu02/OmniRoam
  *Mechanism:* preview (trajectory-controlled) → refine (temporal extend + spatial upsample) world
  wandering. *Steal:* the preview-then-refine pattern for long chains.
- **3D Scene Prompting** (KAIST) — https://cvlab-kaist.github.io/3DScenePrompt/ — dual spatio-temporal
  conditioning: recent frames + rendered point cloud. *Steal:* reinforces our depth/edge control design;
  "render geometry, generate texture."
- **ReCamDriving** — https://recamdriving.github.io/ — 3DGS novel-trajectory renderings as dense structure
  for precise camera control. **CameraSquad** — https://rabberk.github.io/CameraSquad/ — parallel
  multi-trajectory content consistency via cross-view attention. *Steal:* both are "geometry as control
  signal" instances; CameraSquad only if we ever do multi-view chains.
- **Practical stitching ecosystem** (community: ComfyUI VideoChunkTools-style rolling-reference recipes;
  the widely-cited ~135-frame identity-reversion failure; Memorize-When-Needed's revisit note). *Steal:*
  our P3 chain (10×121-frame chunks, 25-frame overlap, stitch ramp, style pass, crumb over the full clip)
  already implements the consensus practice; add revisit-checks at the two doorway crossings as the first
  loop-closure probe.

### 1.5 Commercial / adjacent tooling (context only — we have not audited these tools)

- Topaz Video AI (Starlight/Apollo/Aion) ecosystem guides — e.g.
  https://vegavid.com/blog/troubleshoot-ai-video-artifacts-warping-flickering · deflicker tool posts
  (https://astraml.com/blog/how-to-smooth-video-flicker, https://higsfield.com/articles/best-fix-ai-vidoe-flicker).
- The 2026 "flicker vs identity drift vs melting" customer vocabulary —
  https://uncutly.ai/ai/how-to-fix-ai-video-flicker-and-identity-drift.
  *Steal:* their taxonomy matches ours (flicker = per-frame texture/lighting instability; identity drift =
  accumulating subject change; melting = warp) and is good copy voice. **None of the public material we
  found markets spectral low-band coherence or a half-life metric** — the white space holds *at the level
  of public messaging* (not proof of absence internally).

---

## 2 · SSM baseline design — the per-byte claim

Two scopes. (A) **Middleware bar (new, specified here)**: what crumb_coherence must beat at equal
persistent-state bytes. (B) **Model bar (R14, restated briefly)** — already designed in
`IMPL_NOTES_R14.md`; the per-byte machinery (`ssm_lite.py`, `state_bytes()`, arms A–E) exists.

### 2.1 Why an SSM baseline

Constant-state recurrent models are the field's standard alternative for long memory (S4 → Mamba → MARCH).
Both buckets of evidence say small states lose long-range recall (TTT-video's Mamba critique; ACL'25
state-size scaling; StateX). Therefore: if our spectral anchor cannot beat an **equal-byte learned SSM
tracker** on a task that *requires* memory beyond the local context, the spectral framing is not
load-bearing (our own R14 kill-criterion logic, applied to the middleware). If it can, the per-byte claim
is the correct form of the claim (already the campaign rule: per-byte advantage, not raw quality).

### 2.2 The task that requires memory beyond local context

**Middleware task "T-ref" — reference-regime tapes.** Long tapes (240–480 s) assembled from our synthetic
generators (`run_m0.py` scenarios: gain_field, hotspot, multi-motion) plus real footage (ENG-1 clips),
with drift injected at known severity in regimes:
 (i) static drift — DC/low-band ramp (exposure/color/layout);
 (ii) low-band wander — mean-preserving spatial hotspot;
 (iii) **legit slow motion (the trap class)** — must be retained;
 (iv) revisit — return to an earlier appearance regime (tests cut/reset logic).

The memory requirement is structural: (i)/(ii) must be corrected **against a reference that is minutes
old**, while (iii) must be left alone; a frame-local estimator provably conflates (ii) and (iii) — that is
exactly our M2 trap failure, campaign-verified (`IMPL_NOTES_M2.md`: retention 54.7%/70.8% on frozen
defaults). So the task cannot be won from local information, and it is *the same instrument* we already
gate on. Companion model-task: R14 occlusion probe (target hidden 256 frames; only persistent state
bridges; 1024-frame rollout).

### 2.3 The baseline — minimal and honest

- **Architecture.** One diagonal SSM cell (S4D-style; reuse the structure of `ssm_lite.py`) processing the
  same per-frame low-band boxes the engine sees: input u_t = flattened complex [C, kh, kw] → m = 3·kh·kw
  channels; state S ∈ C^{m×d_s} (complex64); update S ← A⊙S + B·u_t; readout r̂_t = Re(C·S) → predicted
  *clean* low-band box. The correction then uses the engine's own blend machinery with r̂ in place of the
  anchor (same alpha, band mask, phase rules) so the **only** difference between arms is the reference
  source. New file (when built): `crumb_coherence/ssm_tracker.py` + `scripts/run_perbyte.py`.
- **State-size matching.** B_engine = anchor + prev_lowband + scalars = 2·(3·kh·kw·8) + ~48 B. Choose d_s
  so B_ssm = m·d_s·8 ≤ 1.25·B_engine; run at **1× and 2× engine bytes**; the claim is strongest if the
  engine wins even at 2×. Concrete budgets (cutoff defaults; complex64 both sides):

  | res | kh×kw (cutoff) | engine anchor | engine total | SSM m | d_s=1 | d_s=2 |
  |---|---|---|---|---|---|---|
  | 64² | 6×6 (0.10) | 864 B | ~1.7 KB | 108 | 864 B | 1.7 KB |
  | 256² | 26×26 (0.10) | 16.2 KB | ~32.5 KB | 2,028 | 16.2 KB | 32.4 KB |
  | 1080p | 151×269 (0.14, ENG-1 frozen) | 0.93 MiB | ~1.86 MiB | 121,857 | 0.93 MiB | 1.86 MiB |

- **Training / fairness.** SSM arm trained on the synthetic tapes only (ground-truth clean low-band
  available by construction; we generate the drift ourselves) with an equal compute budget per variant,
  ≥3 seeds, fixed held-out eval scenes shared by all arms. **State the asymmetry explicitly**: the engine
  is training-free; the claim is "equal persistent bytes, engine needs no training data". Variants for
  fairness: (a) float32 state (half bytes); (b) slower-pole init; (c) d_s=1 (the degenerate EMA control);
  (d) engine rows in both magnitude and complex_mc modes. No hidden tuning per arm.
- **Kill criteria (pre-registered, before running).**
  A. If SSM (≤1.25× bytes, any fair variant) ≥ engine on CHL_corr **and** passes the trap bar → the
     spectral anchor is not load-bearing for the middleware; pivot per M3 plan-C (learned tracker or drop
     "spectral advantage" language; keep the training-free engine with an honest envelope).
  B. If SSM wins removal but fails the trap → expected; drift/motion confusion is the discriminating
     signature — record it and continue (this is corroborating, not falsifying).
  C. If engine wins at 1× but not 2× → claim narrows to "at equal-or-smaller bytes"; say it exactly that way.

### 2.4 Exact metric list

Middleware arms (frozen windows/thresholds; all multi-seed, median + IQR; no single-clip claims):
- **drift removal** per class: low-band temporal-variance ratio (corrected/raw), centroid-displacement
  reduction, DC-path variance ratio — existing `metrics.py` instruments;
- **non-interference / trap**: tracker-based legit-motion retention ≥95%, trajectory distortion <5%,
  HF-SSIM ≥0.98, low-band var ratio ∈ [0.90, 1.10] (the M3 acceptance set);
- **coherence half-life** CHL_corr (§3.1) — headline aggregate;
- **jitter**: corrected/raw frame-diff ratio (max/p95);
- **detail retention**: HF Pearson correlation (median);
- **resources**: persistent bytes, ms/frame, peak RAM — report per-byte columns (removal @1 KB, CHL per KB).

Model track (R14, restated): `target_identity_survival`, `exit_direction_accuracy`,
`position_error_at_emergence`, `velocity_error_at_emergence`, `divergence_horizon`,
`motion_preservation`, `rollout_fps`/`ms_per_frame`/`mem_gb_peak`; headline = **divergence-horizon per KB
of persistent state**. R14's kill criterion stands verbatim: if local+wave does not beat local+generic-SSM
on div-horizon-per-byte, the wave formulation stops pulling weight.

---

## 3 · Honest benchmark protocol draft

### 3.1 "Coherence half-life" (CHL) — definition and measurement

Use three instruments, one shape (a decay curve → its half-point; report the **curve + fit + number**):

1. **Generator-side (pair-similarity curve).** Precedent: VBench issue #206 `long_horizon_coherence`
   (https://github.com/Vchitect/VBench/issues/206) — frame pairs at Δt ∈ {2,5,10,20,50}, mean DINO
   cosine. Normalize: c(Δt) = (S(Δt) − S_∞)/(S(0+) − S_∞); fit exponential c = exp(−Δt/τ); report
   **CHL = τ·ln 2** in seconds. Their study shows the pattern we care about (Veo3: highest at Δt=2,
   lowest overall; community issue + n=30 Prolific study, Spearman r = 0.854 — directional, not gospel).
2. **Middleware-side (correction effectiveness).** E(t) = 1 − D_corr(t)/D_raw(t) on the same drift
   instrument D (low-band trajectory variance primary; centroid displacement secondary), computed in
   sliding windows W (default 96 frames). **CHL_corr = first t with E < 0.5** (linear interpolation).
   This measures how long the correction keeps working — the honest version of "generates N minutes".
3. **Model-side.** Divergence horizon (R14) is already the primary; CHL optional on rollouts.

Guards (learned the hard way, campaign + SNF-Bench):
- Never report CHL alone — ship motion + texture + jitter beside it. A freeze raises consistency; SNF-Bench
  shows whole-frame metrics literally *reward* corruption.
- D must be validated by **injection before use** (§3.2), not by intuition.
- Reference discipline: drift measured against a *fixed* early-window reference (or ground truth for
  synthetic), never rolling self-similarity (which decays ~0 by construction).
- Minimum clip length ≥ 5× W; per-clip curves published or archived; aggregate only across completed clips.

### 3.2 Metric validation by injection (adopt SNF-Bench's protocol)

SNF-Bench (https://arxiv.org/abs/2608.28694 · https://minar09.github.io/snfbench/, Aug 2026) separates
**static support** from **dynamic flow**, injects global translation/rotation/scale drift and progressive
late freezing **at known severity**, and requires each factor to (a) respond in its stated direction and
(b) stay selective against corruptions it does not target — validated *mechanically*, not by preference
correlation. Their headline warning matches our own: at max injected drift, whole-frame Dynamic Degree
reaches only 1.07× (rewarding it) while drift-specific factors rise 1.32×/1.86×.

Our version: `bench/inject.py` — inject (i)–(iv) drift classes at graded severities into clean synthetic
tapes and real footage; require the instrument table: each metric fires on its class, stays quiet on the
others; publish the response matrix. **A suite passing while the mechanism fails is not a result** (the
honest-benchmark rule from `honest-benchmark-gating`: harness checks are labeled and never promoted).

### 3.3 Blind third-party clip protocol

- **Clips.** ≥2 generators (≥3 for release-grade) × ≥6 clips each, 10–60 s raw (plus a long chain when
  testing chain paths). Selection rule **pre-registered** (e.g., first N public clips matching an
  independent difficulty label — no "clips we know we fix"). Public-domain real footage as a non-AI
  control class (motion preservation must hold there; nothing to "fix").
- **Freeze.** One config for all clips; record + publish the config (hash it). No per-clip tuning, no
  prompt changes, no regeneration framing, no cherry-picking: publish **all** outcomes; every excluded
  clip gets a count + reason.
- **Blindness.** Processing sees uninformative filenames (generator identity/provenance hidden). Raters
  (human and any VLM judge) see randomized A/B with no labels; include catch trials (known-degraded
  fakes) to detect rater drift/laziness. Deterministic scripts are the gate; VLM judge is a third signal,
  never the gate.
- **Evidence pack per clip.** original · processed · config · script versions · metric table · failure
  classification when it fails. Third-party footage only for public claims. Watch-gate: owner approves
  every outbound artifact; this document stays internal.

### 3.4 What to publish first (ranked, each gated)

1. **"Coherence audit" of frontier generators** — half-life curves on public prompts + methodology +
   scripts; **no product claims**. Gate: 2 generators × ≥6 clips, reproducible, instrument
   injection-validated (§3.2). Why first: owns the *measurement* brand, survives even if the rescue offer
   slips, and is honest by construction.
2. **One frozen-settings third-party before/after** (the skill's public-proof recipe): one hard clip from
   a generator we did not build; original + processed + method/version; metrics beside it (including trap
   metrics, so "smoother" isn't the only claim). Gate: trap-safe defaults (post-M3, or magnitude mode for
   legit-motion content) + numbers hold + owner review.
3. **Per-byte technical note** (engine vs stats-EMA vs SSM; model-track div-horizon/KB). Gate: §2 results
   in; audience = technical/design partners.
4. **Product / rescue claims** — only after 10+ real customer clips (THE_POSITION ladder); never before.

---

## 4 · Ranked next moves (with kill criteria)

1. **Finish M3, then run the trap-class matrix** — extend the (already-required) second-scene test to a
   4-case grid: fast pan, occlusion, parallax, multi-object, each with the same retention/distortion bars.
   *Kill:* gate fails >1 legit class → ship magnitude default, publish the envelope; plan-C region/velocity
   variants stay behind flags. *Expected gain:* the one thing between us and a customer-safe default.
2. **Build the injection-validated instrument + half-life harness** (§3.1–3.2) as a small deterministic
   CPU suite (`bench/`), reusing `run_m0`/`run_trap`/`run_eng1_1080` machinery. *Kill:* if CHL fails the
   selectivity test on injected drift, fix the instrument before any number leaves the lab.
3. **Run the SSM per-byte middleware baseline** (§2). *Kill criteria inside.* This is the decisive new
   number for the deep-dive-v2 synthesis and the "per-byte" claim the position already leans on.
4. **Wire the top steals** (each additive, flag-gated, measured on existing suites; no default changes
   without a pre-registered bar):
   (a) anchor-cache of N early snapshots, geometric spacing, for cut/revisit recovery — prior art:
   FreqForcing anchor cache + MARCH content-routed anchors + A2RD's interpolation switch;
   (b) response-profile instrumentation of the anchor (Recency-style perturbation: correction strength vs
   Δt / rho / scenario) — formalizes what the M0 rho sweep found;
   (c) revisit handling in the 3D chain via loop-closure from known poses (Closing the Loop, adapted; we
   own the poses);
   (d) model-track: sample-segments-from-own-rollouts supervision (Self-Forcing++) when the unfreeze
   recipe resumes.
5. **The publish ladder** (§3.4) — strictly behind the owner gate; nothing public from this document.

### Not to steal / cautions

- Don't port KV-eviction mechanics into the plugin — we have no KV to evict; the transferable content is
  *influence shaping* (fade-don't-cut), which the slow anchor already does structurally.
- "Attention sink" is a transformer artifact; our anchor is a *state* anchor. The naming overlap has
  confused before; use "state anchor / anchor memory" in write-ups (MARCH gives us the vocabulary).
- Eviction-style compression can *cause* jitter (FreqForcing on Deep Forcing) — any compression of our
  anchor state must be measured for spectral side effects before it becomes a default.
- TTT-style bigger states are a model-track comparator, not a plugin direction; the plugin's edge is
  being training-free and tiny-state.

---

## Appendix · Source index (all links)

Forcing lineage: 2506.08009 · 2510.02283 · 2512.05081 · 2509.25161 · 2604.21221 ·
github.com/JaydenLyh/Reward-Forcing · 2603.21366 · 2609.19729 · 2606.13035 · 2606.14667 · 2604.18215 ·
2605.06924 + research.google/blog/coherent-long-form-video-generation · 2606.09056 ·
Spectral: 2607.27110 · 2312.06837 · HFMamba (springer 978-981-92-3420-2_1) · FoSS (openreview icsrG0ijaA) ·
SSM/state: 2505.20171 · 2504.05298 · aclanthology 2025.acl-long.564 · 2026.findings-acl.1073 ·
2608.12435 · Camera/3D: 2603.02256 · 2607.21848 · github.com/yuhengliu02/OmniRoam ·
cvlab-kaist.github.io/3DScenePrompt · recamdriving.github.io · rabberk.github.io/CameraSquad ·
Benchmarks: 2608.28694 · github.com/Vchitect/VBench/issues/206 · deepwiki VBench-Long ·
imerit.ai/resources/blog/solving-temporal-drift-in-ai-generated-video ·
Tooling context: vegavid.com Topaz guide · astraml.com · higsfield.com · uncutly.ai.

Campaign-internal cross-refs: `SPECTRAL_PRIOR_ART.md` (FreqForcing numbers, E1–E4 queue) ·
`IMPL_NOTES_M0_ADDENDUM.md` (rho, hotspot) · `IMPL_NOTES_M2.md` + `reviews/claude_m3_gate_task.md`
(trap/gate) · `reports/ENG1_1080.md` (1080p frozen config) · `IMPL_NOTES_R14.md` (occlusion probe,
state-bytes protocol) · `ssm_lite.py` (`state_bytes()`) · `THE_POSITION.md` (claim boundary, ladder).
