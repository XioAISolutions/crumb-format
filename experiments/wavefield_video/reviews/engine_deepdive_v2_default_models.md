# Engine deep dive v2 — default-models consult (ranked improvements + kill criteria)

Consultant: Hermes default profile (deepseek-flash), headless kanban worker `t_c5fd567c`, 2026-09-27.
Brief: `reviews/engine_deepdive_v2_task.md`. Grounded in the campaign memory (IMPL_NOTES_M0_ADDENDUM,
reviews/claude_m3_gate_task, reports/ENG1_1080.*, DEEP_DIVE_3_opus, THE_POSITION, IMPL_NOTES_HYBRID,
plus the 3D pilot NOTES) and a fresh prior-art sweep (Jul–Sep 2026; sources verified live).
**Status: M3 result pending — this memo designs for both outcomes. Watch-gate: nothing here is public.**

## 0. The program in six lines
1. The M3 gate is the right shape; the gate-to-offer work is **real-footage robustness**, not more
   synthetic scenes. ENG-1 already fails retention on real 1080p (motion 0.54–0.70) and the jitter
   regression (2.91 vs raw 1.69) is a design smell, not a scene property. [Q1]
2. Middleware: keep the analytic gate; add **multi-anchor state only if** removal is actually measured
   to decay along long rollouts. Learned gating is unjustified until the battery proves a class the
   analytic gate cannot separate. [Q2]
3. Model: **don't spend architecture budget on faith**. At 8k const-LR the fused hybrid underperformed
   plain wave on balls (copy_ratio 0.428 vs 0.728; mse/copy 0.947 vs 0.571) — the decisive readout is
   the pending **occlusion** arms (div-horizon-per-byte, local_wave vs local_ssm). [Q2]
4. Product: "3D-control generation + crumb finishing pass" sells **now** for controlled content; for
   other generators the honest lane is "correct the conditioning window" (open weights = true in-loop;
   extend-style APIs = at extension points; one-shot closed = stabilizer envelope only). [Q3]
5. Proof: pre-registered, frozen, multi-seed, worst-case-first. Headline = **coherence half-life**
   measured on third-party footage; first publication item = one real before/after + method (post-gate,
   owner-approved). [Q4]
6. The 2026 field is converging on *our* thesis from inside the models (persistent bounded state +
   inference-time reference anchoring). Whitespace (model-agnostic, spectral, outside) is intact — but
   the window to be first with a blind third-party before/after is a "now" asset. [Q5]

## Q1 — Beyond the M3 gate

**If M3 passes as specified**, the classes that still break it, ranked by risk (engine terms: everything
below attacks the single-global-displacement + single-EMA-anchor design):

1. **Handheld / freehand camera ramble — the in-principle blind spot.** Legitimate low-band motion whose
   statistics *are* a random walk is indistinguishable from drift by any trajectory-statistics gate
   (the separator would have to be semantic intent — beyond this machinery). Consequence: the honest
   envelope is "safe where legit low-band motion is
   structured (constant velocity / smooth curves)". Measure: random-walk camera over a static synthetic
   scene; report retention + removal; publish the number as the envelope statement.
2. **Parallax / two-layer motion** (moving foreground + camera): the cross-power surface splits; the
   estimator locks the dominant layer and drags the other. ENG-1 `e_curve` (motion 0.699) is plausibly
   this class. Measure: two-blob scene at different velocities; per-region retention.
3. **Occlusion / disappearance**: trajectory discontinuity at hide/reveal; the gate must classify "no
   signal" as *hold, don't drag*. Real probe already in flight (3D doorway crossing). Measure: synthetic
   occluder + the 3D clip; drag-at-event.
4. **Fast pans**: per-frame low-band shift approaching half the band wavelength (≈1/(2·0.03) ≈ 17 px in
   working-frame pixels — a brisk pan) aliases the phase-correlation peak; a wrong measurement cannot be
   gated away. Measure: pan ladder 1→20 px/frame; estimator error vs known shift.
5. **Multi-object independent motion**: global estimate ≈ blend; confidence drops → correction fades
   (safe, ineffective) or locks one blob. Measure: opposing-blobs; removal should vanish, retention hold.
6. **Zoom / dolly**: radial field vs translation-only estimator; gate likely suppresses (safe). Measure:
   zoom ladder; retention + removal.

**Next robustness step, best data support (do in this order):**
- **(a) Real-data reading (hours, no new code):** run the frozen gate across the full ENG-1 clips and
  re-run the trap ladder; ENG-1 is the only real-footage evidence and its failures are currently
  unaddressed by any acceptance test. No kill — it's a reading.
- **(b) Legit-motion battery v2 (~1 day harness, CPU):** extend `run_trap.py`'s scene generator to the
  six classes above, each with analytic ground truth; acceptance per class = retention ≥95%, distortion
  <5%, plus no false-correction on the drift scenarios. Kill: gate passes battery but ENG-1 still fails →
  synthetic-to-real gap is the blocker; stop building scenes, tune on real clips only.
- **(c) If ≥2 classes break: per-region displacement fields** (blockwise phase correlation → per-region
  coherence + per-region taper). The radial band-selectivity (hotspot, 64.0%) already proves
  per-region targeting works in this engine. Kill: no lift on ≥2 breaking classes without hotspot
  regression (<60%) → accept the envelope instead.

**If M3 fails — plan C beyond "magnitude default + flag":**
- **C1 (build first): residual anchoring.** Split the estimated low-band shift into a coherent
  (predictable) component and a residual; anchor to the *prediction* (the alpha-beta state machinery
  already exists in `core.py`), correct only the residual, scale strength by residual significance.
  The trap's constant-velocity translation has ~zero residual → untouched; wander is all residual →
  corrected. Gain: converts hard classify-then-gate into continuous "remove only the unpredicted".
  Cost: moderate (core.py edit + battery re-run). Kill: trap + second-scene retention <95% at any
  removal ≥30% → accept envelope, don't force. Note the bound: legit motion that is itself
  unpredictable (acceleration, handheld) is *by construction* outside the residual definition; say so.
- **C2: per-region fields** (same as (c); also the parallax/multi-object fix).
- **C3 (do regardless if C1/C2 miss): honest tier split.** Magnitude mode = "appearance stabilizer"
  default for moving content; complex_mc = gated engine for content whose low-band motion is
  structured; document the per-class envelope from battery + ENG-1 numbers. This is a claim-boundary
  decision that feeds THE_POSITION's small print — it is not a failure.
- Strategic note: the 3D-control stack **sidesteps this ambiguity** (motion is authored; geometry comes
  from the control pass), so the middleware-on-foreign-footage lane is the fragile one, and the
  in-loop lane is where this failure mode loses its teeth.

## Q2 — The engine's deeper evolution (ranked; evidence vs fashion)

**Middleware layer (crumb_coherence):**
1. **Analytic gate hardening (Q1 battery + per-region fields).** Evidence: the strongest in-program
   signal we have — every estimator "refinement" regressed (M0.7 controls: injected-truth displacement
   39.9% vs engine 64.0%; perfect component swap −43%); phase correlation + hard band cut + rho=0.995
   are the load-bearing parts (46%→92% via rho alone). Fashion: none. This is unglamorous and it has
   the receipts.
2. **Multi-anchor state (temporal memory for the plugin).** External evidence: FreqForcing anchor
   cache (6 frames, λ=0.6), MemRoPE dual-EMA memory tokens, Rolling Sink's bounded-cache discipline —
   all 2026, all converging on "keep sparse stable references, don't roll one state forever".
   Internal evidence: none yet — removal-vs-time has not been measured past the current rollouts.
   Minimal discriminating experiment: log removal as a function of t on the 256/1024-frame rollouts
   with the single anchor; if removal decays >10% beyond ~frame 500, add K anchors (geometric spacing)
   and re-measure. Kill: no decay → non-problem, don't build. Gain if decay: the 5–10-min single-take
   product claim; cost: state grows by K× band box (still KBs).
3. **Adaptive rho/alpha per content.** Evidence: rho is the single most load-bearing knob we have; a
   content-adaptive horizon is cheap. Expected gain: single-digit % (removal already 85–94% on ramps
   across rho 0.99–0.999) but it removes the stationary-vs-drifting tension. Kill: no gain over frozen
   rho across the battery → skip. Fashion risk: low.
4. **Learned gating (small CNN/MLP on low-band stats).** Evidence: none; and the M0.7 lesson plus the
   trap result both say analytic-beats-refinement here. Fashion risk: high (and training pairs would be
   synthesized by us — we'd risk learning our own generator). Verdict: do NOT build until the battery
   shows ≥2 classes the analytic gate provably cannot separate. Then: labels are free (we inject the
   drift), train on the battery, test ZERO-SHOT on held-out classes. Kill: doesn't beat the analytic
   gate on those classes at equal false-correction, or its rule can't be stated in one sentence.

**Model layer (wave-field / crumb-llm):**
5. **Hybrid gated fusion (`--fusion local_wave` / `local_ssm`).** Evidence so far is *mixed-negative*:
   H8wav (8k const-LR, g32) escaped freeze (copy_ratio 0.428) but lost to plain wave (B2 0.728) on the
   same recipe — the fusion gates cost accuracy on balls. The decisive experiment is already wired:
   the occlusion arms (`run_occlusion.sh`, 4M matched params, 1024-frame rollout); read
   `divergence_horizon` and `div_horizon/KB_state`; kill criterion on record — if `local+wave` doesn't
   beat `local+ssm` per byte, the structured wave formulation doesn't earn its keep on that task.
   Minimal experiment = run the pending arms and read the per-byte column; do not add budget until then.
6. **Self-conditioned rollouts.** Evidence: ramp-arm drift is the exposure-bias signature; externally
   Self-Forcing / Self-Forcing++ / Causal Forcing++ / Rolling Forcing all validate the recipe family;
   our detached-forward machinery exists. Fashion risk: moderate (field-wide adoption, our evidence
   indirect). Minimal experiment: self-cond p-ladder (0/0.25/0.5) at K≤4, same budget; readout
   `divergence_horizon` @256/1024 + copy_ratio trajectory. Kill: no div-horizon gain → keep teacher
   forcing, close the item.
7. **Conditional kernel (λ_t).** Evidence: none yet; gate behind 5–6 by construction. Minimal
   experiment: bounded-Δλ controller vs fixed kernel at matched params on ≥3 hard tasks incl. occlusion.
   Kill: short-horizon gain without div-horizon gain → delete (this one is the most fashion-shaped of
   the four; the burden is on it, not on us).

**The brief's four, straight-ranked:** (1) hybrid gated fusion — decide now from the wired occlusion
arms (cheap, decisive; don't add budget before the per-byte column reads); (2) self-conditioned
rollouts — cheap ladder behind a div-horizon readout; (3) learned gating — middleware; fashion until
the battery proves an unseparable class; (4) conditional kernel — last, gated behind (1)–(2).

Cross-layer note: "temporal memory in diffusion" (Rolling Sink, Deep Forcing) and our anchor-state
plan are the same principle at two addresses — *keep the recurrent state within its stable, clean
distribution; don't let it roll into the drifted regime*. Our rho finding IS a one-knob version of
this. Cite that alignment, don't reinvent it.

## Q3 — The stack play

**Judgment: yes — "3D-control generation + crumb finishing pass" is the sellable bundle, with a sharp
boundary on the in-loop-for-others claim.**
- Sell now (concierge): controlled content, where geometry is authored (Blender path) and crumb
  finishes; and the honest stabilizer envelope on customer footage until the gate passes.
- Other-generator frontier, in honesty tiers: (i) open-weights pipelines (Wan/LTX/…) → true in-loop
  conditioning correction; (ii) closed APIs with extension/keyframe conditioning → correct the
  conditioning window at each extension point (still context, not export; verify each API's actual
  surface before claiming); (iii) one-shot closed generation → post-hoc only, and then we are a
  *stabilizer* — say it, don't dress it as middleware. The middleware label survives only where we sit
  inside a generation loop.

**What breaks first at 60s → 5–10min (rank):**
1. **Cross-chunk appearance/grade drift** — ten chunks tuned identical at t=0 diverge in
   white balance / material response; stitch ramps hide boundaries, not slow divergence. Measure:
   chroma drift + DINO/LPIPS identity distance on tracked objects across chunks (ASTRA 3D review list).
2. **Occlusion/re-entry content** (probe 1) — occluded surfaces are re-invented per chunk; the
   "orbit away → orbit back → same face/material?" test is the nasty form.
3. **Dynamic-object identity not present in the 3D scene** — moving extras / sway / curtains are
   free-styled per chunk and can morph at 5min. Geometry can't help; crumb doesn't touch mid/high band.
   Name it as outside the current envelope.
4. **Seam/stitch integrity at scale**: verify the 60s chain's stitch does NOT trip the cut detector
   (if it does, effective coherence is per-chunk only — a silent claim downgrade). Cheap audit.
5. **Cost/ops**: ~10 chunks/min → 50–100 chunks per 5–10min at 832×480 (~2min/chunk incl. load/swap
   amortized → 2–4h wall); 25/121 ≈ 21% overlap overhead. Fine for concierge, not for self-serve.

**How overlap-stitch + scene resets + crumb scale:** keep 25-frame overlap; validate no reset trip;
keep crumb AFTER the style pass (chain order is right). The 5–10min product is mostly ONE long take
(that's the wedge, no cuts to hide behind) — so scene resets matter less and **anchor horizon becomes
the product requirement** (ties Q2-2 directly into the roadmap: long single takes are the forcing
function for multi-anchor state). Crumb itself scales linearly (streamed; 134.8 ms/frame @1080p CPU,
less at 832×480 windows) — no kernel work (constraint honored).

**Probe 2 (recommend): the long-gap revisit.** Leave the room, spend ≥20–30s elsewhere, return.
Check: (a) same furniture positions on return (edge recall vs control ≥ first-visit), (b) no re-entry
pop (frame-diff continuity at crossing), (c) crumb retention on the return approach ≥95% (a re-approach
is exactly the legit low-band motion the gate must not fight). It tests the unique claim — geometry
remembers WHERE — at the timescale nobody else holds. It also maps 1:1 onto LVBench-C's
appear/disappear/reappear design (A²RD, 2605.06924), so results are comparable to published work.
**Probe 3 (cheap): cross-chunk identity ledger** — freeze matched poses of ≥5 scene objects across
chunks, score DINO/LPIPS identity distance; first quantitative chart of failure class 1.

## Q4 — Proof: the honest benchmark

**Definitions first (pre-register before any run):**
- Per scenario, pick the drift statistic d(t) (e.g., low-band centroid displacement from reference, or
  trajectory variance) and the failure level d_fail (the raw clip's own worst-horizon value).
- **Coherence half-life H50 = first t where d(t) − d(0) ≥ 0.5·(d_fail − d(0)).** Customer-legible
  headline: H50(with crumb)/H50(raw) on the same generator, same protocol. Engine-side diagnostic:
  "removal half-life" (t where retained removal <50% of initial) — internal only.
- Multi-seed ≥3, equal compute, frozen settings, report **worst-case and median**, publish the losses.

**Protocol A — engine vs baselines (per clip-class, both synthetic + real):** engines = gated
complex_mc, magnitude, stats-EMA (24 B — already in `core.py`), no-op. Metrics: per-class removal,
fidelity **vs the clean control** (the wiring lesson: vs-input hides differences), tracker
retention/distortion (never centroid/argmax proxies), HF-SSIM, jitter max/p95-vs-raw, ms/frame,
state_bytes. Success bar: on the classes it exists for (spatially-wandering drift) the margin over
stats-EMA is large and structural (stats is blind by construction — hotspot 0%-class); on scalar
classes it must be *no worse* than stats within noise. Report the Pareto (removal vs bytes; removal vs
ms) — the honest framing is capability-class difference, not per-byte superiority on every class.

**Protocol B — the SSM per-byte claim (model):** the memory-requiring task is the **occlusion probe**
(object exits, must re-emerge with correct position/identity after 30+ frames) + 1024-frame
divergence. Arms: local+wave vs local+ssm at matched params (wired); optional external reference:
VideoSSM-class hybrid (arXiv 2512.04519) as the literature anchor. Report `divergence_horizon`,
`div_horizon/KB_state`, ms/frame. Kill (on record): no per-byte win → no kernel work, no per-byte
claim — the constraint holds.

**Protocol C — the third-party blind test (what actually convinces a skeptic):**
1. One hard clip from a real generator **we didn't make**, fixed prompt decided in advance, frozen
   settings + version, publish original + processed + method + the full metric table including
   failures ("no prompt changes, no regeneration, no cherry-picked frames").
2. Blind 2AFC: randomized A/B pairs, pre-registered question ("which drifts less?"), n≥20 raters who
   are not us, report win rate + CI, include clips where we lose.
3. Reproducibility: config JSON + exact package version + raw metric files.
Optionally the strongest move: invite skeptics to send their own broken clip, process blind, publish.

**Publish order (all post-gate, owner-approved, nothing public before):**
(1) method note + ONE real before/after ("research preview", honest labels);
(2) 3-generator before/afters;
(3) benchmark table vs stats-EMA on real footage;
(4) the falsified list (flat phase-pull, phase-ramp, etc.) — this is what makes the rest credible.
Synthetic numbers are mechanism evidence; only real footage is product evidence. Never mix the two
in one chart.

## Q5 — Prior art to steal from (2026; Jul–Sep window as requested)

| method | what it is | take |
| --- | --- | --- |
| **FreqForcing** (2607.27110, Jul 2026) | training-free spectral self-anchoring; anchor cache 6 frames; DC/low-freq drift diagnosis; +16.5% latency | anchor-cache design → Q2-2; external validation of band-selective correction; closest published philosophy to ours |
| **Self-Forcing / ++ (ICLR'26), Rolling Forcing, Causal Forcing++** (2605.15141) | minute-scale AR video; train-test gap; few-step distillation | recipe validation for self-cond rollouts (Q2-6); benchmark discipline |
| **Rolling Sink** (2602.07775) | cache-behavior mismatch = drift carrier; training-free bounded cache (K=6, 5/6 retained), 5–30min, 0.56% throughput | the "keep state within its stable distribution" principle = our rho/anchor horizon, stated rigorously; bounded-budget discipline for our multi-anchor spec |
| **MemRoPE** (2603.12513) | dual EMA memory tokens (long+short), online RoPE indexing, hour-scale | candidate dual-timescale anchor for the plugin (slow+fast rho) if single-rho decays |
| **Deep Forcing** (2512.05081) | deep sink + participative KV compression; shows naive StreamingLLM-style sinks degrade fidelity / stall motion; training-free 12x extrapolation (5s→60s+) | its "a sink alone is not enough" result parallels our gate finding (blocking one failure mode exposes another); add its temporal-repetition / motion-deceleration metrics to our occlusion eval |
| **Pathwise TTC** (2602.05871) | initial-frame reference anchor; corrects *stochastic sampling states* after structure stabilizes; 30s+ stable | closest published analog of our middleware thesis inside a model; adopt "correct after stabilization, not during" for our in-loop integration; cross-cite in the method note |
| **A²RD + LVBench-C** (2605.06924, May) | agentic AR diffusion; multimodal memory; benchmark with appear/disappear/reappear at 3/5/10min | use LVBench-C's design AS the probe-2/3 eval shape; their 10-min scale = our target horizon |
| **VideoSSM** (2512.04519) | AR diffusion + hybrid state-space memory (short/long context) | strongest external signal that hybrid SSM+video is live; reference arm in Protocol B |
| **ReHyAt** (CVPR 2026) | recurrent hybrid attention (softmax+linear, constant memory) | architecture reference for fusion design; nothing to adopt now |
| **TTT for long video**: TTT Done Right (ICLR'26), Forget-Anticipate-Adapt (2606.26515), Spatial-TTT (ECCV 2026) | fast weights as temporary memory; long-sequence TTT is the field's own weak spot; Spatial-TTT accumulates 3D evidence in fast weights | do not chase until the occlusion probe shows recurrent state underperforming; keep as the conditional-kernel alternative for the model track; Spatial-TTT is a future option for the 3D-cache lane |
| **GEN3C / GaussFusion** (3D track) | 3D cache → camera-precise generation; buffer-driven coherent refinement | cache-the-world validates the Blender-control variant; nothing to change now (our authored scene is cheaper than building a cache from video) |

**What changes the roadmap: nothing supersedes the mechanism; the field is converging on it.** In the
Jul–Sep 2026 window itself the only major new item in our lane was FreqForcing; the load-bearing wave
is Feb–Jul 2026 (Rolling Sink, MemRoPE, TTC, Deep Forcing, A²RD). Nothing in 2026 refutes
band-selective anchoring or ships a comparable *outside-the-model* system. The live risk is the
opposite: generator-side fixes (Rolling Sink-class, TTC-class) make
post-hoc drift correction less necessary for *newest* models — so the middleware proof (blind
third-party before/after) should target **current-generation commercial output** and should happen
sooner, not later. Keep the in-loop lane warm precisely because that's where those fixes live.

---
Constraints honored: middleware-not-filter (all Q3 tiers respect it, with explicit honesty where we
cannot), magnitude/phase policy unchanged, no kernel work proposed, every claim carries an envelope.
Evidence notes: all numeric values above are from repo docs (M0/M1/M2/M0.7/ENG-1/IMPL_NOTES_HYBRID/H8wav
and the 3D pilot NOTES); external methods verified live on 2026-09-27; no new measurements were run for
this consult.
