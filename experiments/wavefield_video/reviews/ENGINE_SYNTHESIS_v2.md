# ENGINE SYNTHESIS v2 — four consults + one hard verdict (2026-09-27 ~01:30 EDT)

## TL;DR — the decision

The content-blind in-loop correction (`complex_mc` as a default) is **NOT shippable
and NOT sellable** — M3 proved the velocity gate cannot separate legitimate motion
from drift (measured separation margin 0.002 on [0,1]). The honest product is
**triage-form rescue ($99, human-in-the-loop)** plus the **authored-motion
pipeline** (3D-control + crumb: proven through 60 s, zero cumulative drift). The
engine's research arc continues in parallel on a pre-registered kill path
(residual anchoring), and the cash lane (audit / bids / leads) is independent of
all of this and fully staged.

## Inputs (all committed, all timestamped)

| # | Artifact | Task | Status |
| - | -------- | ---- | ------ |
| 1 | reports/ENGINE_RESEARCH_SWEEP_2026.md | t_fc77f3bd | ✓ 26+ verified sources, SSM per-byte baseline spec, SNF-Bench injection protocol |
| 2 | reviews/engine_deepdive_v2_default_models.md | t_c5fd567c | ✓ ranked improvements + kill criteria + handheld blind spot |
| 3 | reviews/engine_deepdive_v2_opus.md | 01:00 CLI run | ✓ code-grounded (verified gate/filter lines), 5 divergences |
| 4 | reports/METHODS_SPACE_2026.md | t_11e322ff | ✓ 9 method families mapped vs the box; runnable flags |
| 5 | method-ladder run (5 alternates vs baseline) | t_70c59e88 | ✓ none beat dense-depth; HQ-mode loses (−0.131); 4-step LoRA load-bearing |
| 6 | IMPL_NOTES_M3.md incl. RESULTS | M3 gate | ✗ **RED** — separation margin 0.002; late misfire (307 % retention at 192 f); hotspot removal dead gated (−0.2 %) |
| 7 | 3D pilot NOTES (P0–P3) + 60 s demo | t_80b6e826 / t_a021b496 | ✓ zero cumulative drift over 960 frames; crumb −43.6 % drift / −27.1 % flicker; occlusion (doorway ×2) held |

## The convergent finding

Four independent readers of the situation plus one hard experimental result agree:

1. **The wall:** statistics alone cannot separate legitimate unstructured low-band
   motion from drift — both are random-walk-shaped. M3 measured the margin: 0.002.
   Escape requires a **prior**:
   - the generator's own next-frame prediction (**in-loop** — FreqForcing et al.),
   - or **authored motion** (our 3D-control pipeline — proven).
2. **The field moved to generator-side implementations** (FreqForcing dual-attention
   early-denoise anchoring; TetherCache TAME; CANVAS plan+persist). Middleware on
   foreign footage has neither prior — that caps it, and dictates the product story.
3. **Our spine is validated**: dense-depth 4-step control + crumb; nothing in the
   ladder beat it; the 20-step "HQ" arm scored worse (0.517 vs 0.648); the 4-step
   LoRA is load-bearing for control fidelity, not a shortcut.
4. **Safe remnants**: `magnitude` mode is unaffected and safe (M1 regression: −85.9 %
   low-band var, detail 0.988); legacy `complex_mc` stays reachable behind a flag;
   the gate code stays in-tree, default OFF.

## Decisions (each with a kill criterion)

**D1 — Residual anchoring [BUILD NOW, parallel].** Anchor to the alpha–beta
prediction (constant-velocity prior); correct only the unpredicted residual,
strength ∝ significance. *Build:* a mode in `core.py` + acceptance via the same
pre-registered tables. *KILL:* no config achieves BOTH trap distortion <5 % AND
M0 hotspot ≥60 % within 3 tuning rounds → middleware correction defaults OFF;
product = triage-only.

**D2 — Product formalization [OWNER REVIEW].** Offer v1 = triage rescue
("$99 — send clip; improved version + comparison; no charge if not materially
improved"), scoped honestly: human triage, magnitude-mode + restore stack.
Flagship demo = the 60 s authored-motion chain (cut; landing hero after
watch-gate). Claim-boundary edit staged for THE_POSITION.md: no implication of
content-blind automated correction; "automated correction prototypes remain in
research preview."

**D3 — Restoration stack (F6) [BUILD].** Wire deflicker node + SeedVR2 (installed)
+ frame interpolation + magnitude-mode; test on kk77785 (ghost mirror) & kk77783
(max flicker) & the 60 s. *KILL:* no measurable flicker/drift reduction on the
worst picks vs source.

**D4 — Benchmark / proof [BUILD, slow].** SNF-Bench injection protocol = mandatory
instrument validation before any external number. Coherence half-life audit of
frontier generators (frozen settings, neutral) = the first publishable piece.
SSM per-byte baseline = spec'd in the sweep §2.

**D5 — Model layer [DEFERRED, gated].** Hybrid gated fusion / self-conditioned
rollouts / conditional kernel: no architecture budget until the occlusion arms
produce the div-horizon/KB-state column (per-byte). Opus's items carry kill
criteria on record (H8wav already lost to plain wave: copy_ratio 0.428 vs 0.728).

**D6 — Cash lane [OWNER DECISIONS].** Independent of the engine: audit wedge
(4 decisions), top-10 bid dossier (go), 20-lead outreach list (approval).

## The honest one-liner

We know exactly what the wall is (statistics can't separate drift from motion
without a prior), exactly where our proven ground is (authored motion: geometry
remembers where, crumb remembers how), and exactly what we can sell today (triage
rescue + restoration) — while the research path (residual anchoring → in-loop
proof) runs on pre-registered kill criteria.

## Owner decision queue

1. **Audit wedge**: founding rate ($500 × first 3) or list $750? Freelancer+Contra,
   add Fiverr? "XIO AI Solutions" name OK? Sample-report sharing OK?
2. **Bids**: run the next bid round on the dossier's top picks?
3. **Leads**: outreach approval for v1 (channel + tone), or hold?
4. **Mac reboot** once for hygiene (swap clear) — pick a moment.
5. **Landing hero swap** to the authored-motion demo (after watch-gate) — yes/no.

## D1 OUTCOME (t_3e1d4c9f, IMPL_NOTES_M4.md)

**KILL — pre-registered outcome triggered, round 1 = decisive wipeout.** 120-config sweep
(window×strength×cutoff on trap96/192 + hotspot48): **0/120 co-pass.** Trap distortion floor
= 61.2% (bar 5%); hotspot removal max = 51.2% (bar 60). The no-op floor sits at 61.7% ⇒
corrections at every usable strength only ADD damage; no grid refinement crosses a structural floor.

**Mechanism finding (the valuable part): the premise inverts the useful correction.** On these
traces the drift IS the predictable slow component (illumination sinusoid / anchor-lag ramp), and
the innovation is estimator noise. The legacy anchor-pull's 64% removal lived precisely in the
predictable term that residual anchoring refuses to touch. Additionally: the fixed blend's
*magnitude* alone contributes ≈50% removal — i.e. the **safe magnitude mode carries half the
benefit with none of the risk.**

**Consequence:** the middleware correction research = **CLOSED** (M2 fail → M3 RED → M4 kill:
three mechanisms, three honest negatives, one structural explanation). No rounds 2–3
(structural floor, per pre-registration). Code stays in-tree, default OFF.
Coherence must come from **authorship** (3D-control lane: proven) or the **model itself**
(wave-field / qssm lane); product = triage + authored-motion + magnitude envelope.
The 4090 keeps cooking the model lane.
