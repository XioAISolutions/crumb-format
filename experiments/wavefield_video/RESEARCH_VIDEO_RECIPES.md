# How Seedance and the long-video field build video models — research notes

*Zeph, 2026-09-29. Sources: Seedance 1.0 tech report (arXiv 2506.09113), Seedance 1.5 pro paper page, the AR-video memory survey ("The Past Frames the Future", arXiv 2609.28466 + its repo), Self-Forcing / Self-Forcing++ / CausVid / Rolling Forcing line, VideoSSM (arXiv 2512.04519), Sana-WM, WorldMem, LongLive. Written against OUR problem: 5 minutes of the SAME continuous content on ONE 4090.*

---

## 1. Seedance 1.0 — the industrial recipe (arXiv 2506.09113)

**Model stack**
- **Temporally-causal VAE**: (r_t, r_h, r_w) = (4, 16, 16), 48 channels; L1 + KL + LPIPS + adversarial (hybrid PatchGAN-like discriminator); causal design handles images as T=0. No patchification on the DiT side.
- **DiT with decoupled spatial/temporal layers**: spatial layers = within-frame attention + all text cross-modality; temporal layers = across-frame attention with window partition (global temporal receptive field, efficient). MMDiT-style dual weight sets; Q/K normalized for stability.
- **MM-RoPE**: 3D visual RoPE + extra 1D text RoPE on concatenated sequences → interleaved text/visual, and **natively multi-shot** (shots as temporal blocks, each with its own dense caption).
- **Diffusion refiner** cascade (480p base → 720p/1080p, initialized from base, conditioned on LR video) — their answer to "smooth final pixels".
- **Prompt engineering LLM** (Qwen2.5-14B, SFT then DPO with LoRA): rewrites user prompts into the exact dense-caption format the DiT was trained on.

**Data pipeline** (their words: performance "inextricably linked" to it)
1. Diversity-oriented sourcing → 2. **shot-aware temporal segmentation, max ~12 s clips** → 3. overlay rectification (logo/watermark detection + adaptive crop) → 4. quality & safety filters (blur, jitter, poor composition, **"predominantly static content" removed**) → 5. **semantic dedup** (embed → cluster → keep best) → 6. distribution rebalancing (downsample head, upweight tail).
- **Precision captions**: dense, separated into *dynamic* (actions, camera movement) and *static* (appearance, scene, style) features; caption model trained on manual annotations (frozen vision encoder + fine-tuned LM).

**Training ladder** (each stage visibly improves output — their Figure 6)
- **PT**: flow matching, logit-normal timesteps + resolution-aware shift; *progressive*: 256px images → 256px video 3–12 s @12fps → 640px → finally 24fps ("to improve smoothness"); i2v ratio 20%.
- **CT (continue training)**: i2v 20→40%; **select data by aesthetic scorer + optical-flow motion evaluators**; motion-only short captions for i2v. 
- **SFT**: human-curated category-balanced set; train several specialists → **merge models**; early stop.
- **RLHF**: three reward models (foundational / **motion** / aesthetic); **directly maximize composite reward predicting x0** — they report this beats DPO/PPO/GRPO; multi-round model↔RM iteration; separate RLHF pass on the refiner and even on the *distilled* model.

**Acceleration** (targets for us): TSCD trajectory-segmented consistency distillation (4×), RayFlow score distillation, adversarial multi-step tuning with human-preference supervision; **thin VAE decoder** (narrow channels near pixel space → 2× decode, no quality loss); kernel fusion (RoPE/norm fused, −90% memory traffic on those); FP8 communication; async offloading; hybrid-parallel VAE decode (spatial+temporal partition).

**1.5 pro**: dual-branch DiT (video branch + audio branch + cross-modal joint module), multi-stage A/V data pipeline — the A/V direction, not our fight right now.

---

## 2. The long-video problem — the memory lineage

**The survey framing (2609.28466)**: AR models have bounded context but unbounded history → *memory mechanisms* preserve history beyond the context window. Carriers: **Visual / Implicit-State (KV caches, recurrent states, SSMs) / Explicit-State (entities, geometry) / Adaptive-Parametric**. Operations: **write / read / update / manage / integrate**. This is the exact vocabulary our §8.4/§8.5 write-path work lives in.

**Closest relatives to our system**
- **VideoSSM (2512.04519)**: hybrid state-space memory — an evolving global SSM memory + a context window for local motion; linear-time; minute-scale consistency. *Our carried state + write-policy work is this idea in a wave recurrence.*
- **Sana-WM (2605.15178)**: hybrid linear DiT, minute-scale on one GPU — efficiency reference.
- **Recurrent/SSM family**: StateSpaceDiffuser, EDELINE, Recurrent AR Diffusion (global memory + local attention), Long-context SSM world models.

**The drift fix that everyone converged on (train on your own rollouts)**
- **CausVid**: block-causal + KV cache; shows the saturation/error-accumulation problem (we run its descendant: LongLive).
- **Self-Forcing**: close the train/test gap — train the student on *its own* generated rollouts with a distribution-matching (DMD) signal from a bidirectional teacher. 
- **Self-Forcing++ (2510.02283)**: minute-scale without long-video teachers: sampled segments from the student's OWN long videos guide the update; per-chunk noise increasing over time; up to 4m15s (20× the teacher's horizon), fixes error-accumulation and over-exposure; **no retraining on long-video datasets required**.
- **Rolling Forcing / Causal Forcing / Context Forcing / Rolling Sink / BaggEr**: the 2025-26 wave of drift/over-exposure fixes and real-time extensions; **MemRoPE** (training-free infinite via evolving memory tokens); **PackForcing** ("short video training suffices for long video sampling"); **Lol**: hour-scale scaling. Also QA-style: Reward Forcing, Sparse/Light Forcing, Anchor Forcing.
- **Explicit-state alternatives**: WorldMem, DecMem/MemLearner (learned decoupled memory), StoryMem, Memento, LongLive-RAG (retrieval-augmented long video).

**Benchmarks/eval**: the survey's own "Memory-oriented Benchmarks" list (M-Bench, EntityBench, WorldRoamBench, Long-CODE isolating pure long-context ...) — plus our own long_eval style of per-window drift read.

---

## 3. What WE take — mapped to our stack

Our position: single 4090; wave/crumb model with an O(1) carried state; wheel already built (streams 5 min at constant memory); the open problem = *the same content staying the same for 5 minutes* (measured: coherent horizon ~30 s on the 120 s pilot).

1. **We are an implicit-state model and the field agrees that this is the right family for unbounded video.** Priorities are the *operations*: what to write, when to gate, what to skip. Our §8.4 (clean write) and §8.5 (change-gated write) verdicts are first steps; next: learned write/read gating, and memory-aware sampling. VideoSSM's hybrid (global state + local window) = the closest published blueprint to test against ours.
2. **The missing half of our training: rollout/self-forcing-style objectives.** Teacher-forced training is why the state drifts at inference. The documented fix: train through the model's own carried-state rollouts (long horizons, self-generated), with a strong teacher/preference signal — no long-video dataset required (Self-Forcing++). Concretely for us: add a long-rollout loss to train_long (the machinery — stream + carried state — already exists and is proven), and later a DMD-style distribution-matching term. **This is the single highest-value experiment on the board for the 5-minute goal.**
3. **Stage ladder = our roadmap**: PT (ball-scene arms) → **CT**: longer, motion-rich, higher-quality data (our concat reel is the first CT data; add an optical-flow motion scorer to select/filter clips, mirroring Seedance CT) → SFT/merge: curate best takes; merge specialist checkpoints (our multi-arm suite is literally a collection of specialists) → preference/RL: direct reward maximization over our long_eval dims (drift/motion/colour) — the evaluator doubles as the reward model.
4. **Data discipline to adopt now**: ≤12 s shot segmentation, drop near-static clips, semantic dedup, rebalance; dense dynamic+static captions when we go prompt-conditioned (Prompt-Eng LLM = Qwen-class SFT→DPO; not yet on the critical path for video-only latents).
5. **Decode/pipeline tricks for "hyper smooth"**: thin decoder (2× decode), precision experiments at decode time (the flicker question), decode-side temporal consistency checks; compile/CUDA-graph the sequential chunk loop (our measured launch-bound cost); distillation later for consumer speed (TSCD/RayFlow family) — our version would be a few-step student of the wave model.
6. **Keep the evaluator as THE gate**: every run judged by long_eval's per-window read (drift ≥0.9, luma/contrast/sat ±25%, motion ≥25%); report the coherent horizon whatever it is. Same gate for LongLive (baseline) and our model (should beat it on continuity of *the same content* — that is our wedge).

**Immediate queue consequences** (already moving): LongLive 5-min ladder running as the baseline (evaluated on the same gate); latent-v2 (SEQ=12) then latent-v3 (SEQ=64 on the 226 s concat reel); write-mode audit for the latent runs (gate vs clean per §8.5); then the rollout-loss experiment.

---

## Key links
- Seedance 1.0 tech report — https://arxiv.org/abs/2506.09113
- Seedance 1.5 pro — https://seed.bytedance.com/public_papers/seedance-1-5-pro-a-native-audio-visual-joint-generation-foundation-model
- Survey: The Past Frames the Future — https://arxiv.org/abs/2609.28466 (list: https://github.com/HaroldChen19/Awesome-AR-Video-Memory)
- Self-Forcing++ — https://arxiv.org/abs/2510.02283 · Self-Forcing — https://arxiv.org/abs/2506.08009 · CausVid — https://causvid.github.io/
- VideoSSM — https://arxiv.org/abs/2512.04519 · Sana-WM — https://arxiv.org/abs/2605.15178
- LongLive — https://arxiv.org/abs/2509.22622
- Rolling Forcing — https://arxiv.org/abs/2509.25161
