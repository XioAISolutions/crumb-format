# VIDEO LLM — DEEPER RESEARCH (2026-09-30)

*Owner directive: "do deeper research to find out how we can do the Video LLM better." Companion to RESEARCH_VIDEO_RECIPES.md + DEPTH_AGENDA.md — sweeps what changed since those were written and ranks the moves for our two lanes.*

## TL;DR — top moves
1. **SeedVR2 delivery validation** (installed, unvalidated) — biggest visual-quality-per-hour currently sitting idle on the box.
2. **Motion LoRA on our Wan-family bases trained on our own travel battery** — bakes motion in, kills the seed lottery + world-attractor.
3. **Finetuned-base bake-off**: HunyuanVideo 1.5 (8.3B, step-distilled, ~75 s / 4090 at 480p i2v) and Wan 2.7 TI2V-5B as segment engines under the orchestrator.
4. **Prompt-expansion layer** (local Qwen3-4B, Seedance-style dense dynamic+static captions) — removes the studio's prompt-skill floor.
5. **Wave lane**: verify the running long-horizon training includes self-rollout sampling — that remains the single highest-value coherence experiment (field-converged).

## Lane map (as of tonight)
- **A. Sellable generator lane**: LongLive-2.0-5B streaming (5-min works; travel = prompt-gated, seed-sensitive; variety solved today via scene-noun swap — probe flows 1.15–2.01 across 4 distinct worlds) + Wan2.2-FunControl control-spine (ghosting/melt ladder open) + journey orchestrator v1.6 (plan/takes/stitch, built tonight).
- **B. Wave-field research lane**: long_horizon slices training (job 994 running); rollout/self-forcing fix = the stated next big one; long_eval = the gate.

## What the external sweep adds (Sept 2026)

### 1. Open-weight landscape moved
- **Wan 2.2/2.7** (Apache-2.0) = quality leader of the open family; TI2V-5B fuses T2V+I2V; "motion holds up on demanding prompts". Wan 3.0 (60B, 30-s single pass) on the mid-2026 roadmap.
- **HunyuanVideo 1.5** (8.3B): step-distilled — ~75 s renders on a single RTX 4090 (480p i2v); cinematic-motion reputation. Commercial use <100M MAU.
- **LTX-2.3** (22B): 4K@50fps + synchronized audio, up to 20 s/clip — length+audio leader.
- *(Roundup-sourced; verify exact weights/licensing at download time before committing.)*
- **Reading**: our 5B LongLive base is now ~two tiers behind the open frontier. The multi-minute streaming capability is still ours (nobody open ships multi-minute single-pass) — keep it as the wedge, but segment quality should come from newer bases.

### 2. Single-GPU LoRA for video = proven mature
- Wan2.1-I2V-**14B** cinematic LoRA on ONE GPU "within hours": rank 8, lr 3e-5, <50 clips, CFG 3.8–4.2, 28–32 steps; two-stage (style, then motion) decoupling.
- **LiON-LoRA**: motion-amplitude token → linear motion scaling with minimal data (camera + object control).
- **Follow-Your-Motion**: spatial-LoRA → temporal-LoRA two-stage motion transfer; naive joint LoRA fails (3000 steps ≈ no motion reproduced).
- **Reading**: a small motion LoRA trained on OUR travel clips is a weekend job on the 4090, not a research project. It converts today's prompt+seed lottery into a baked-in property.

### 3. Drift/coherence family (newest additions on top of the recipes doc)
- Self-Forcing++ (minute-scale without long-video teachers), **Rolling Forcing** (real-time multi-minute streaming, ICLR 2026), Head-Heterogeneity (error accumulation, arXiv 2605.14487), Light-Forcing (sparse-attention speed), training-free horizon extension (2602.14027), on-manifold steering (ICML 2026 F2S).
- **Reading**: the whole field converged on *train on your own rollouts*. Our constant-size state makes that cheaper for us than for anyone — proceed (this is DEPTH_AGENDA item #1).

### 4. Upscaling
- **SeedVR2** = one-step DiT restoration, 1080p in a single forward pass (ICLR 2026); FlashVSR = rival. Ours is **installed, unvalidated for delivery** (PRODUCT_PATH r.1). Likely the fastest "looks premium" upgrade available on the box today.

## Ranked experiment cards (proposed for the queue)

- **Card 1 — SeedVR2 texture battery (fast, ~1 day).** Upscale the 5-min + day→night journey 2x; run the texture battery + owner eyes. Kill: texture-battery fail or temporal shimmer.
- **Card 2 — Travel LoRA (medium, weekend).** 40–60 clips from our archives (fast battery + world probes + OPSD) → rank-16 LoRA on the Wan2.2-TI2V-5B family → eval: travel at seeds {0,2,11} WITHOUT the prompt lottery. Kill: texture regression exceeding motion gain.
- **Card 3 — Segment-engine bake-off (medium).** HunyuanVideo 1.5 vs Wan 2.7 TI2V-5B vs current Fun-Control spine on the same 4 scenes; pick per-scene engines for the orchestrator. Kill: no visible win per scene (disk is fine: 711 GB free).
- **Card 4 — Prompt expansion (small).** Local Qwen3-4B rewrite layer: "walk through a market at night" → dense dynamic+static caption in our dialect; A/B on 8 prompts. Kill: no motion/adherence win.
- **Card 5 — Rollout-loss verification (train lane).** Confirm long_horizon training includes self-rollout sampling; if not, that becomes the next training experiment (DEPTH_AGENDA #1).

## Sequencing
1. Card 1 this week (installed; 1 day).
2. Card 4 (small, unlocks studio UX).
3. Card 2 (the weekend job — biggest leverage on lane A).
4. Card 3 gated on model downloads + a spare window.
5. Lane B continues independently (994 → rollout).

## The one-line thesis
Quality moves come from **data-side training** (LoRA/rollouts) and **model-side upgrades** (newer bases, one-step restoration) — NOT from more inference-knob twisting. We proved the knobs' ceiling this week; these are the levers that remain.
