Everything checks out against the code: the `rho` EMA recurrence (`anchor ← rho·anchor + (1-rho)·A(Xlo)`, line 8), the **causal alpha-beta trajectory filter** (line 32, `beta = a²/(2-a)` — a constant-velocity predictor), the M3 velocity-coherence gate (lines 40-55), band-selective `mc_band` masking, and a soft-cut cooldown (`rho→0.5`, line 30) are all present. That grounding matters for one of my main divergences below.

`★ Insight ─────────────────────────────────────`
The crux the whole campaign turns on: in the 2D FFT, **magnitude carries texture/energy, phase carries position** — so anchoring magnitude and preserving phase *should* be motion-safe. M2/ENG-1 prove it isn't, because `complex_mc` re-centers the low band, which *is* phase. And there is a hard wall underneath: **statistics alone cannot separate legitimate unstructured low-band motion from drift** — both are random-walk-shaped. In-loop methods (FreqForcing) escape the wall because the model's own next-frame prediction supplies the missing "intended motion" prior; 3D-control escapes it because motion is *authored*. Middleware on foreign footage has neither — which caps it, and dictates the product story.
`─────────────────────────────────────────────────`

I read the default-models memo (`engine_deepdive_v2_default_models.md`) and the Astra Blender consult. Both are strong; I won't re-emit them. Below is the **ranked deliverable**, then the **five specific divergences** where the third opinion actually differs — that's the useful part for synthesis.

---

## RANKED ACTIONS (with kill criteria) — save this

**1. Read before you build: run the frozen gate across *full* ENG-1 + re-run the trap ladder. [Q1]** This is the gate to revenue and it's nearly free.
- Gain: tells you whether *any* analytic gate can clear real-footage retention (ENG-1 motion 0.54–0.70 is unaddressed by every current acceptance test). Cost: hours, no new code. Kill: n/a — it's a reading, not a build. Measure: tracker retention/distortion + jitter max/p95-vs-raw per clip, worst-case reported.

**2. Build *residual anchoring* now — in parallel with M3, not as "plan C". [Q1/Q2]** *My biggest divergence — see D1.* Anchor to the alpha-beta filter's *prediction*; correct only the unpredicted residual, strength ∝ residual significance.
- Gain: constant-velocity translation (the trap) has ~zero residual → passes untouched; wander is all residual → corrected. It is the offline twin of FreqForcing's published dual-branch, buildable on machinery already in `core.py`. Cost: moderate (one `core.py` path + battery re-run). Kill: trap + second-scene retention <95% at removal ≥30% → fall back to the honest tier split. Measure: retention/distortion on trap **and** on an *unpredictable-legit-motion* scene (acceleration/handheld) — which it will fail by construction; publish that as the envelope, don't hide it.

**3. Legit-motion battery v2 (6 classes) + pre-register the wall. [Q1]** Handheld ramble, parallax/two-layer, occlusion, fast pan (aliases the phase peak ≈>17 px/frame), multi-object, zoom/dolly — each with analytic ground truth.
- Gain: turns "what breaks it" from speculation into an acceptance harness. Cost: ~1 day CPU harness. Kill: gate passes battery but ENG-1 still fails → synthetic-to-real gap is the blocker; stop making scenes, tune on real clips only. Measure: per-class retention ≥95% / distortion <5% **and** no false-correction on the drift controls.

**4. Ship the lane that sidesteps the wall: "3D-control + crumb finishing." [Q3]** Sell this now (concierge) for authored content; call foreign-footage work an **appearance stabilizer**, not "coherence," until 2–3 pass.
- Gain: authored geometry supplies the intended-motion prior the middleware lacks → the M2/ENG-1 ambiguity disappears. Cost: none beyond the pilot in flight. Kill: doorway-occlusion (probe 1) or probe 2 (D5) shows cross-chunk identity collapse the crumb pass can't touch → the sellable unit is per-shot, not per-minute; say so. Measure: edge-recall vs control on revisit; DINO/LPIPS identity distance across chunks.

**5. Model track: run the wired occlusion arms, decide fusion from the per-byte column — no architecture budget on faith. [Q2]** local+wave vs local+ssm, matched params, 1024-frame rollout.
- Gain: settles hybrid gated fusion empirically (H8wav already *lost* to plain wave on balls: copy_ratio 0.428 vs 0.728). Cost: run pending arms only. Kill (on record): local+wave doesn't beat local+ssm on `div_horizon/KB_state` → the structured wave formulation doesn't earn its keep; no kernel work, no per-byte claim. Measure: `divergence_horizon`, `div_horizon/KB_state`, ms/frame.

**6. Proof: coherence half-life *and* the field's own drift metric, on third-party footage. [Q4]** *See D4.* Publish method note + one real before/after first.
- Gain: a skeptic can't dismiss the metric as self-serving if you report the published **VBench-Long Drift Score** (slope of DINOv2 consec-sim + LPIPS + HSV-sat + Laplacian-var over 5 segments) alongside your intuitive H50. Cost: metric wiring + blind rater panel (n≥20, not-us). Kill: blind 2AFC win-rate CI includes 50% → no perceptible coherence gain; do not publish a win. Measure: H50(crumb)/H50(raw) worst-case+median, ≥3 seeds, equal compute, publish the losses.

**Straight ranking of the brief's four mechanisms [Q2]:** (1) analytic gate hardening + residual/per-region — has the receipts (every estimator "refinement" regressed; rho alone 46%→92%); (2) self-conditioned rollouts — cheap p-ladder behind a `div_horizon` readout, externally validated (Self-Forcing++); (3) learned gating — *fashion until the battery proves an unseparable class* (and you'd risk training on your own generator); (4) conditional kernel — last, gated behind 1–2.

---

## Five divergences from the default-models memo (the third-opinion delta)

- **D1 [Q1/Q2] — Promote residual anchoring to "build now," not "plan C."** The memo files it as a fallback if M3 fails. I disagree: it's cheap, the alpha-beta predictor it needs *already exists* (`core.py:32`), and FreqForcing independently validates exactly this shape (local branch keeps recent change, anchor branch supplies stable low-freq). M3's binary gate and C1's continuous residual are complementary; C1 almost certainly generalizes to real footage better than a hand-tuned wander classifier. Run both against the same battery and keep the winner.

- **D2 [Q1] — The wall is the organizing principle, and it's strategic, not just technical.** No statistics-based gate (M3), residual scheme, or FreqForcing local-branch escapes the fact that *unstructured legitimate motion ≡ drift* to a statistical observer. Consequence the memo under-weights: **middleware on foreign footage is fundamentally capped**; the 3D-control and in-loop lanes lift the cap because they inject the missing intended-motion signal. This is *why* action 4 outranks the foreign-footage rescue as the product.

- **D3 [Q3] — Whitespace is narrower than "intact."** FreqForcing (2607.27110) *and* FreeLong++ (2507.00162) now own **training-free spectral low-band anchoring inside open diffusion models** — the exact "in-loop for other generators" pitch. That lane is contested, not white. Defensible whitespace = (a) authored-geometry bundle, (b) closed/foreign-footage stabilizer where no attention access exists, (c) model-agnostic, no-training. And because generator-side fixes are landing monthly, the foreign-footage proof must target **current commercial closed models** and happen *soon* — the "first blind third-party before/after" is a decaying asset.

- **D4 [Q4] — The "beat SSM per-byte" framing is a category trap; quarantine it.** Per-byte belongs to the *model* track (wave-field vs SSM, occlusion probe) — the constraint's own wording ("beats SSM on a memory-requiring task") is about the model. Do **not** benchmark the middleware crumb pass on per-byte; a stats-EMA baseline is blind to spatial wander *by construction* (hotspot: 6% vs 45%), so the honest middleware claim is **capability-class + Pareto (removal vs bytes, removal vs ms)**, never per-byte dominance. Mixing the two tracks in one chart hands the skeptic a free rebuttal.

- **D5 [Q3] — Probe 2 should stress the crumb/motion conflict, not geometry again.** The memo's "long-gap revisit" re-tests geometry memory (same as probe 1). The untested, higher-risk class is **an independently-moving foreground subject that isn't in the control geometry** crossing the scene: does its identity survive across chunks (failure-class 3, which geometry can't fix and crumb doesn't touch), *and* does the gate leave its legitimate low-band motion intact? That single probe exercises the two things most likely to break the bundle at 5–10 min.

**One method both memos missed:** **Recency Forcing** (2609.19729, Sept 2026) — directly on "the long-horizon gap in autoregressive video." Within the 90-day window; worth a read before finalizing the self-conditioned-rollout ladder. Otherwise I concur with the memo's prior-art table (FreqForcing → anchor-cache design; Rolling Sink → bounded-budget discipline = your rho finding stated rigorously; Pathwise TTC → "correct *after* structure stabilizes, not during," the closest published analog of the middleware thesis).

**Bottom line:** the field is converging *onto* the band-selective-anchoring mechanism from inside the models — that validates the science and compresses the commercial window. Bet on the two lanes the wall can't cap (authored 3D-control + closed-model stabilizer), build residual anchoring now, and get one honest blind before/after out fast.

---
**Sources** (verified 2026-09-27; several post-date my Jan-2026 cutoff — treat method claims as their abstracts state, not as endorsements):
- [FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring (2607.27110)](https://arxiv.org/abs/2607.27110)
- [Self-Forcing++: Towards Minute-Scale High-Quality Video Generation (2510.02283)](https://arxiv.org/pdf/2510.02283)
- [Recency Forcing: Bridging the Long-Horizon Gap in Autoregressive Video Generation (2609.19729)](https://arxiv.org/pdf/2609.19729)
- [FreeLong++: Training-Free Long Video Generation via Multi-band Spectral Fusion (2507.00162)](https://arxiv.org/pdf/2507.00162)
- [Attend Locally, Remember Linearly: Linear Attention as Cross-Frame Memory (2605.16579)](https://arxiv.org/pdf/2605.16579)
- [VBench-Long / Drift Score (long-horizon drift metric)](https://www.emergentmind.com/topics/vbench-long)
- [SSM Meets Video Diffusion Models (structured state spaces for long video, 2403.07711)](https://arxiv.org/pdf/2403.07711); Mamba-3 (ICLR 2026 oral) per [SSM survey](https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801)

I modified no files, per the brief. Want me to pressure-test any single recommendation (most likely **D1 residual anchoring** — I can sketch the exact `core.py` change and the discriminating experiment without touching the file), or verify the Recency Forcing / FreeLong++ abstracts more closely before you fold this into the synthesis?
