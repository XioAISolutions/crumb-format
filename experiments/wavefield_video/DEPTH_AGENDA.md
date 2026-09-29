# DEPTH AGENDA — building beyond the industry's approach

*Zeph, 2026-09-29. Owner directive: "go deeper than all other companies and build this." Companion to RESEARCH_VIDEO_RECIPES.md.*

## Thesis

Every leading system (Seedance, Veo, Kling, Sora) is a **bounded-context model + chained extensions + bolted-on memory**. Their "long" is stitching; their memory is retrieval; their continuity is a patch. Our substrate is structurally different: a **constant-size carried state** (wave-field recurrence) where the memory IS the model state, O(1) at any horizon. The places we can go deeper are exactly the places their architecture cannot follow cheaply:

### 1. Closed-loop minute-scale training (the big one)
Train through the actual objective: 5-minute self-rollouts with gradients through minutes of carried state. Their attention/KV grows superlinearly with horizon — minute-scale closed-loop training is unaffordable for them. Our constant state makes it O(T) memory. Nobody in the field trains the full horizon in one closed loop (Self-Forcing++ cheats with segments; we don't have to). First scaffold = card t_ed03a741 (Astra, gpt-6-astra).

### 2. Multi-timescale state = engineered permanence
Give the state a LADDER of update rates — fast dynamics / motion / scene identity — each level with its own write policy (§8.5 generalized per level). Object permanence becomes a designed property, not an emergent hope from scale. Sora's permanence is a scale prayer; ours is a circuit.

### 3. State-native product features (nobody ships these)
- **Save / resume**: continue a video tomorrow, identically — the state is the artifact.
- **Fork**: branch one story into two from any moment.
- **State-edit**: change the scene IN the state (not the prompt) and the video follows.
- **Share**: send the state, not a render.
The industry re-encodes every request (Veo's Extend is v2v re-encode: drift per chain, no true resume).

### 4. Structured-dynamics depth (the wave substrate)
Exploit the math: stability guarantees by construction (no blow-ups), native continuity dials (pole halflives = explicit % memory per scale), hypercomplex/field kernels, and formal statements of what a wave-field recurrence represents that attention cannot cheaply.

### 5. The scoreboard (measured claims)
long_eval (drift / luma / contrast / motion / saturation per 30 s window) as the common gate. Public-behavior comparisons: our 5-min stream vs LongLive vs Veo-style chaining, drift curves side by side. "Deeper" must be a number, not a vibe.

## Sequencing

- **A. Now** — 5-min proof artifacts: LongLive ladder (running tonight) + latent rounds (queued) + Astra's rollout prototype (t_ed03a741).
- **B. Next** — closed-loop minute-scale training experiment (needs A's prototype + the box free): the single statement the field can't match.
- **C. Then** — multi-timescale state design (level ladder + per-level write policies).
- **D. Product** — state-native features once B/C show the mechanism.
- **E. Continuous** — theory notes + the eval scoreboard published per experiment.

## Guardrails

- Every claim measured by long_eval, pre-registered thresholds; honest horizons (report the number, whatever it is).
- One box, one queue (gpuq); co-resident ComfyUI respected; no manual GPU launches.
- Repo-only deploys; owner watches anything public before it ships.
