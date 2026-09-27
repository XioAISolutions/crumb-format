# METHODS SPACE 2026 — every viable family for long-form coherent AI video (including what we've never touched)

Filed 2026-09-26 EDT / 09-27 box-UTC · Zeph · kanban `t_11e322ff`
Owner directive: "Still can't sell yet — research OTHER METHODS, use all skills + new skills we haven't considered."
Context read first: `reviews/engine_deepdive_v2_task.md`, `reports/ENGINE_RESEARCH_SWEEP_2026.md`,
`~/.hermes/workspaces/brainsnn/3d_first_pilot/NOTES.md`, `reports/3D_FIRST_RESEARCH.md`, `reports/ASTRA_3D_REVIEW.md`.
Watch-gate: nothing public. This file queues no GPU jobs — the ladder card `t_70c59e88` was running on the box during composition (GPU 100 %, 19.1 GB resident at read time).

---

## 0 · Method + honesty rules

- **"Runnable on our 4090 today"** was checked LIVE against the box inventory (`/workspace/slava/comfy-house/ComfyUI/models/*`, `runners/`, `custom_nodes/`, RAM/disk, logs). Raw dump: `~/.hermes/kanban/workspaces/t_11e322ff/box_models_inventory.txt`. Box facts used throughout: ComfyUI **v0.3.65** core-only, custom nodes = {Frame-Interpolation, seedvr2_videoupscaler, GielisGeometry, websocket_image_save}; RAM 62 GB (43 used / 18 avail during a job); disk 826 GB free; Wan 2.2 family + FLUX + LTXV as the model set (§7.1).
- **No fabricated results.** Every measured number is cited from an existing artifact by path; every forward-looking statement is marked *(projection)*. Web facts are dated and linked (§7.2); "last 60 days" = 2026-07-27 → 2026-09-27.
- **"vs our stack"** = vs the current spine: 3D-control generation (Blender → depth/edge passes → Wan 2.2 Fun-Control 4-step → overlap-stitched chunks) + crumb_coherence (magnitude default; complex_mc flagged) + the $99 rescue lane. Engine internals are NOT this card's scope (see deep dive v2).

## 1 · The map at a glance

| # | Family | Runnable on our 4090 today? | Verdict vs our stack | Kill criterion (one line) |
|---|--------|------------------------------|----------------------|---------------------------|
| F1 | Keyframe / interpolation control (FLF2V-class) | **YES** (Wan FLF2V template + RIFE on box; already exercised — `kk` series). LTX-2.5 needs quant work; H3 NO (RAM) | Confirmed multi-shot bridge; complements crumb (pins endpoints per shot) | Bridges don't beat plain overlap seams at matched seeds ⇒ rescue-use only |
| F2 | Per-video adaptation (LoRA / test-time training) | **PARTIAL** (LoRA: proven for images on box; video-LoRA needs musubi install. TTT: NO) | Cheap scene-lock for chains; TTT = watch item | 1-scene video-LoRA ≤2 h shows no seam/identity gain vs base+crumb ⇒ park |
| F3 | Latent planning (plan-then-execute) | **YES** (process + JSON shotplan + existing QA scripts) | Multiplies our control; Google's own long-form answer is planning | No ≥30 % retry reduction or seam gain at equal GPU-time ⇒ human-side only |
| F4 | Temporal memory architectures (streaming/FIFO/anchor caches) | **PARTIAL/NO** (nothing drop-in on box; that's our middleware's territory) | crumb's home turf; external recipes stay citations | Can't run ≤24 GB AND can't beat chain+crumb at equal GPU-min ⇒ citation only |
| F5 | Video state-space models | **NO** (research code; no weights found) | Closest external sibling of our thesis; audit + steal T-RFlow's band metric | When ≤24 GB weights drop, run honest protocol; else audit only |
| F6 | V2V stabilization / deflicker (existing-footage lane → $99 rescue) | **YES — highest readiness** (1 git clone + already-installed SeedVR2/RIFE) | Second engine for the rescue offer that does NOT wait on the M3 gate | Combined pass < best single stage +20 % flicker win (no motion loss) ⇒ ship single stage |
| F7 | Scene-reset / multi-shot editing pipelines | **YES** (composition discipline on existing stack) | Required for the 5–10 min product; needs crumb reset semantics | Cross-cut bleed persists after one fix cycle ⇒ claims stay single-take |
| F8 | Render / 3D-based (longer + bigger-res path) | **YES** (spine; 720 p test + Fun-Camera + chunk math pending) | This IS the product spine; scale hinges on chunk math, not new models | 720 p no perceptual win vs 480 p+SeedVR2 (or >2× cost) ⇒ stay 480 p |
| F9 | Hybrid stack play | **YES** (assembly of the above) | The assembled recommendation (§2·F9) | Per-layer kills; stop if seams don't hold past ~3 min |

**Three headline conclusions**

1. The single most runnable NEW lane is **F6 — the existing-footage rescue stack** (deflicker node + SeedVR2 + RIFE + crumb magnitude, labeled post-hoc). It uses the box as-is and feeds the $99 offer without touching the M3-gate dependency.
2. **F1 is already partly in our hands** — the Wan FLF2V-class template has been executed on our box and drove the rescue-demo `kk` material. The un-built piece is using it as a *multi-shot bridge* between 3D-control chunks, where it looks structurally right.
3. Our whole generative box runs **draft mode by construction** (lightx2v 4-step, CFG 1.0 — the cheapest mode of a 14B model; `open-weight-video-generation` skill). The biggest quality lever before any new-model spend is an **HQ profile arm** (no distillation LoRA, ~20 steps, CFG ≈3.5) on one hero chunk.

---

## 2 · The families in detail

### F1 · Keyframe / interpolation control (first-last-frame class)

**Mechanism.** Pin endpoints (first/last images), let the model fill the interval; interpolate (RIFE/FILM) to raise fps; loops via first==last. Turns "generate a shot" into "generate an interval between two known states" — the natural unit for multi-shot composition.

**Best evidence.**
- Wan FLF2V is open (Apache 2.0, 720 p): official ComfyUI guide <https://docs.comfy.org/tutorials/video/wan/wan-flf>; model docs <https://deepwiki.com/Wan-Video/Wan2.1/6.3-first-last-frame-to-video-(wanflf2v)>; wiki note that FLF2V is a Wan2.1 open model while Wan 2.2 I2V sees the first frame <https://wan2.video/wiki/first-last-frame-video>.
- Our own tested recipe: the `architectural-fpv-concept-video` skill cites the official template `video_wan2_2_14B_flf2v.json` and records it as **executed on an RTX 4090** on our stack versions (core-only ComfyUI v0.3.65, Torch 2.6.0+cu124 — the box still runs 0.3.65). We already produced with it: the rescue-demo `kk` series = Wan2.2 first/last-frame renders (bedroom → dressing-room glide, 81 f / 5.06 s), `~/.hermes/workspaces/brainsnn/rescue-demo/PICKS.md`; runners on box `comfy-house/runners/wan_gen.json`.
- LTX-2.5 ships an official **FLF2V workflow** (Aug 13, 2026, <https://www.earngenix.com/workflows/ltx-2-5-comfyui>; model page <https://comfy.org/ltx-2.5>).
- MiniMax **H3-Base-FL2VA**: 0/1/2 images → video+audio, up to 15 s, 24 fps (<https://huggingface.co/MiniMaxAI/MiniMax-H3>; tracker: open weights Aug 3, 2026 <https://magichour.ai/blog/ai-video-model-release-tracker-2026>). On our box: downloaded (465 GB) but the load smoke was **killed at 53 GB RAM used / 8 GB avail — RC=137 at weight 401/1058** (`/workspace/slava/h3_smoke_out.log`, `h3_smoke_ram.log`; box RAM 62 GB). Not runnable today.
- Long-span quality caveat: semantic degradation between distant keyframes is a named failure mode — "Anchor Frame Bridging" <https://github.com/roboticwink1/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB>; STAGE/STEP2 builds whole storyboards from start-end frame pairs (arXiv 2512.12372).

**Runnable today on our 4090: YES** (Wan template + Frame-Interpolation node on box). LTX-2.5: PARTIAL — 70–80 GB of files, officially 32 GB+ VRAM (<https://book.st-hakky.com/en/data-science/ltx-25-comfyui-workflow-guide>, Sep 18, 2026) ⇒ quant experiments only. H3: **NO** (RAM wall, above).

**What it means vs our stack.** FLF2V pins *endpoints per shot*; crumb holds *trajectories within runtime*. Together: pinned shots + live coherence. It is also the cleanest bridge between 3D-control chunks — each chunk already knows its exact control frames, so adjacent chunks can be FLF-bridged instead of only overlap-stitched.

**Kill criterion.** ≥3 seeds, matched budget: bridged joins vs plain overlap joins on the 60 s path. If seam frame-diff and drift slope don't beat plain overlap beyond a pre-registered bound, deprioritize to rescue-use only.

### F2 · Per-video adaptation (LoRA / test-time training)

**Mechanism.** Adapt model weights to *this* clip: (a) train a small LoRA on one scene/clip (minutes–hours, 24 GB class); (b) test-time training — adapt during inference (research frontier).

**Best evidence.**
- The box has a proven LoRA pipeline for images: kohya/sd-scripts venv + cached datasets (`kip-dataset/cap_*_flux_te.npz`) and two in-house Flux LoRAs live in the ComfyUI loras dir (`xiokip_flux`, `xiokip_v2`, 306 MB each).
- Wan 2.2 LoRA training is the community standard: kohya's **musubi-tuner** supports Wan 2.1/2.2 T2V/I2V/control with the dual high/low-noise scheme (GUI fork: <https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI>; ComfyUI node: <https://comfy.icu/node/MusubiWanLoraTrainer>). Landscape note from that ecosystem: *"since Alibaba stopped shipping open weights after 2.2, still the base most of the local video ecosystem is built on"*.
- TTT research (recent): "One-Minute Video Generation with Test-Time Training" (arXiv 2504.05298); "Forget, Anticipate and Adapt: TTT for Long Videos" (arXiv 2606.26515); TANGO test-time noise-guided adaptation (arXiv 2607.15849, Jul 2026); TTA dual distillation (arXiv 2607.24611, ACM MM'26).

**Runnable today: PARTIAL.** LoRA: yes for image models (proven); video-LoRA = install musubi-tuner (not yet exercised on box). TTT: NO — research stacks, no drop-in for 24 GB.

**What it means vs our stack.** A per-scene Wan LoRA is a cheap "scene lock" that attacks identity/style drift at the source and could shrink crumb's workload; TTT is the frontier rival story for temporal memory (watch, don't build).

**Kill criterion.** Train one Wan LoRA on the pilot scene (single 121 f chunk, rank 16–32, ≤2 h). If seam/identity metrics at equal budget don't improve vs base + crumb, park — novelty tax.

### F3 · Latent planning (plan-then-execute)

**Mechanism.** An LLM/agent plans shotlist, keystates, and constraints *before* generation; execution follows the plan; a verifier scores outputs and triggers replans.

**Best evidence.** Google's **CANVAS** "AI video co-director" — multi-agent framework planning visual continuity across multi-shot narratives, minutes-long outputs, HardContinuityBench, Gemini-3.1-Pro backbone (Sep 24, 2026, <https://research.google/blog/coherent-long-form-video-generation/>). STAGE (storyboards as start-end pairs, arXiv 2512.12372). Our own P3 chain already ran a *hand-baked plan* (10×121 f, step 96, 25-frame overlap) — planning works; it just isn't automated or machine-readable.

**Runnable today: YES** — no new model. Concretely: a JSON shotplan → per-chunk prompt/control/FLF-endpoint generation → auto-QA (`qa_drift.py` pattern + `video_analyze`) → retry only failed chunks.

**What it means vs our stack.** Planning doesn't fix a broken generator; it multiplies our control and cuts dead GPU-minutes. That even Google's long-form answer is planning + persistence (not a bigger model) is the strategic point.

**Kill criterion.** Plan-driven rerun of a 60 s chain: if retry count doesn't drop ≥30 % vs manual or seam QA doesn't improve at equal GPU-time, keep planning human-side.

### F4 · Temporal-memory architectures (streaming / FIFO / anchor caches)

**Mechanism.** Give the generator persistent state across windows: rolling KV caches, FIFO of denoising windows, anchor/sink frame caches, forcing-family training recipes, error-recirculation controls.

**Best evidence.** Our sweep already maps this family in depth — `reports/ENGINE_RESEARCH_SWEEP_2026.md` (Self-Forcing / Self-Forcing++ and the forcing variants incl. Recency Forcing Sep 17 2026; TetherCache/Memento; FramePack; streaming V2V lineage). Nothing new to add from this round except the landscape frame: these are *training-side* recipes for generators we don't control; the drop-in inference variants are the ones to watch.

**Runnable today: PARTIAL/NO.** No streaming-memory model or context-window node in the box inventory (custom_nodes = 4, listed in §7.1). Our working approximation IS overlap-stitch + crumb.

**What it means vs our stack.** This is crumb's home turf — the competitive frame for the per-byte claim. No new lane to run today beyond what the ladder card already tests.

**Kill criterion.** Any recipe that can't run ≤24 GB AND can't beat chain+crumb on drift slope at equal GPU-minutes stays a citation.

### F5 · Video state-space models

**Mechanism.** Linear-time recurrent state (Mamba-family) as video memory — constant state per frame, no quadratic attention.

**Best evidence.** VideoSSM (arXiv 2512.04519 — autoregressive video diffusion + hybrid SSM memory; reports minute-scale consistency gains vs LongLive/Self-Forcing while preserving dynamic degree); Long-Context State-Space Video World Models (arXiv 2505.20171, ICCV'25 — block-wise scan, constant per-frame cost); SSM Meets Video Diffusion (arXiv 2403.07711); and the most stealable piece: **"Towards Error-Free Long Video Generation" (T-RFlow, arXiv 2606.22370)** — a frequency-band decomposition that separately suppresses high-frequency over-sharpening and low-frequency color-bias accumulation. That is *exactly* the failure class our spectral docs name, published in the same language.

**Runnable today: NO** — research code, no released weights found; training pipeline out of 4090 class.

**What it means vs our stack.** The closest external sibling of our thesis; use as (i) the SSM per-byte baseline audit we already owe, (ii) a metric steal — band-resolved error-accumulation curves belong in our honest benchmark, (iii) demand validation for non-attention memory.

**Kill criterion.** When ≤24 GB weights appear, run the honest protocol head-to-head; if per-byte coherence half-life loses, re-scope the claim. Until then: audit only, no build.

### F6 · Video-to-video stabilization / deflicker — the existing-footage lane ($99 rescue)

**Mechanism.** Fix footage that already exists (customer or ours): temporal deflicker (brightness wander + per-chunk step artifacts), temporally-aware restore/upscale, interpolation. This is a **post-hoc restoration class — label it as such**; it must never be presented as the in-loop middleware claim (which stays for generation pipelines).

**Best evidence.**
- `comfyUi-deflicker` node — targets exactly our artifact classes: `step_removal` (chunk-boundary steps) + `temporal_smoothing` (brightness flicker), LAB colorspace, debug heatmaps; built for AI-video sources (WAN, VACE, FramePack). One `git clone` into custom_nodes. <https://github.com/karcsiha/comfyUi-deflicker>
- **SeedVR2** (video restoration/upscaler) is **already installed on our box** (`seedvr2_videoupscaler` custom node; outputs under `/workspace/slava/seedvr/` including 1080 p results).
- **FlashVSR** — CVPR'26 one-step streaming VSR, ~17 fps @ 768×1408 on one A100, code + weights open (<https://github.com/OpenImagingLab/FlashVSR>, project <https://zhuang2002.github.io/FlashVSR/>, arXiv 2510.12747). 4090-feasible class.
- RIFE/FILM interpolation: **on box** (`ComfyUI-Frame-Interpolation`).
- Classical deflicker lineage: NFFA "Blind Video Deflickering" (<https://github.com/chenyanglei/all-in-one-deflicker>); BurstDeflicker multi-frame flicker benchmark (arXiv 2510.09996, NeurIPS'25 dataset).
- In-house numbers already on record: crumb magnitude-mode pass = −68 % low-band trajectory variance / −20 % flicker (single clip, `3d_first_pilot` G5); −43.6 % / −27.1 % on the 60 s chain with tracking unchanged (NOTES P3).

**Runnable today: YES — highest readiness of any new lane.** No big downloads; one git clone + components already installed.

**What it means vs our stack.** Gives the rescue offer a **second engine that does not wait on the M3 gate**: deflicker → SeedVR2 → crumb (magnitude) on the `kk` "before" clips = a shippable "rescue pass v0.5". The ladder card can QA it this week without touching generation.

**Kill criterion.** Blind A/B on `kk` clips (flicker/drift via `rescue_qa.py` + vision): if the combined pass doesn't beat the best single stage by ≥20 % on flicker without motion loss, ship the single stage.

### F7 · Scene-reset / multi-shot editing pipelines

**Mechanism.** Compose many shots into a longer piece; reset state at cuts; maintain continuity via planning + per-shot anchoring (start/end frames, references).

**Best evidence.** Google CANVAS (Sep 24, 2026, above); STAGE (start-end pairs as shot units, arXiv 2512.12372); Kling 3.0 multi-shot storyboarding (≤6 cuts; closed — <https://www.digitalapplied.com>, Kling 3.0 4K/60 fps guide); Runway's multi-shot "combined last-frame technique" guide (Sep 11, 2026, <https://runway.com>); NVIDIA neural storyboard / character-consistency line (arXiv 2412.07750). Our assets: the 60 s single-path chain (P3), the style0 variant, byte-verified stitch receipts.

**Runnable today: YES** — it's a composition discipline on the existing stack. Missing piece: **crumb reset semantics across cuts** (the M0 cut detector exists; it now needs an explicit per-scene anchor reset + a scene manifest) and QA that proves no cross-cut correction bleed.

**What it means vs our stack.** A 5–10 min product is multi-scene by definition. This is the methods gap between our 60 s proof and the sellable long-form claim — smaller than it looks, but it must be measured at cut boundaries, not assumed.

**Kill criterion.** Build one 3-scene, ~3 min piece. If cross-cut color/history bleed or missed resets persist after one fix cycle, restrict claims to single-scene long takes.

### F8 · Render / 3D-based — the longer + bigger-resolution path

**Mechanism.** 3D scene → control passes → conditioned generation (our Pipeline B); pure render (A); scaling = resolution, chunk length, asset quality, camera control.

**Best evidence (ours, measured).** P0–P3 pilot: controlled runs track geometry ~3× above the uncontrolled floor (depth recall 0.648 / edges 0.725 vs gray 0.189); placement edits honored (crossed comparisons collapse, 0.487–0.497); **60 s chain = recall 0.684, slope +0.00006/frame — no cumulative drift**; uncontrolled 60 s repeats one ~6 s dream (recall 0.139); crumb on top: −43.6 % / −27.1 % with tracking unchanged (`3d_first_pilot/NOTES.md`, receipts in `p3/`). Box support: image→3D smokes produced GLBs (TRELLIS + Pixal3D, `trellis-smoke.log`, `pixal3d-smoke2.log`); `blender_mcp_addon.py` staged locally. External context: `3D_FIRST_RESEARCH.md` (GEN3C / SEVA / ReCamMaster / WonderWorld / GaussFusion family).

**Runnable today: YES — the spine.** Concrete upgrade paths:
- **(a) 720 p chunks** (1280×720) on the pilot scene — VRAM test at 81 f / 4-step (projection: likely fits fp8 + tiled decode; verify before promising).
- **(b) Fun-Camera as motion source** — models on disk (2×15.3 GB) + camera nodes present in 0.3.65 (`nodes_camera_trajectory.py`, `WanCameraImageToVideo`); trajectory control instead of depth for free-camera moves. (Also on the ladder card's list.)
- **(c) Chunk math for 5 min:** 4800 f = 40 chunks (121 f, 25-overlap) ≈ 4× the P3 chain; wall time ≈ 1.5–2 h generation at P3 rates *(projection from ~2 min/chunk)*; disk fine (826 GB free).
- **(d) SeedVR2 → 1080 p** already in use; **(e) TRELLIS.2 assets** for richer scenes.

**What it means vs our stack.** It *is* the product spine — "Geometry remembers WHERE the world is; crumb remembers HOW the video behaves" survived the pilot. The methods-space finding: scaling to minutes hinges on chunk math + look, not on new models.

**Kill criterion.** 720 p test (same seed/prompt; blind A/B vs 480 p+SeedVR2): no perceptual win or >2× cost ⇒ stay 480 p. 5-min build: crumb time scales superlinearly or storage blows up ⇒ re-plan before promising.

### F9 · Hybrid stack play (the assembly)

**The recommendation on the table:**
1. **Spine:** 3D-control generation (unchanged) — extend to 720 p test + Fun-Camera + scene manifest.
2. **Multi-shot:** FLF2V bridges between chunks + JSON shotplan + auto-QA retry loop (F1+F3+F7).
3. **Rescue (new lane, ships now):** deflicker → SeedVR2 → RIFE → crumb magnitude on existing footage, labeled post-hoc (F6) — the $99-offer engine that ignores the M3 gate.
4. **Quality:** HQ-profile arms for hero assets before any new-model spend (F2/LoRA + open-weight skill's settings lever).
5. **Sound:** our clips are silent; LTX-2.5 (native audio) and H3 (audio) are hardware-blocked — interim sound layer via existing audio skills; revisit if a quantized LTX-2.5 fits 24 GB.

**Where it breaks at 5–10 min:** 40+ interdependent chunks (planning needed, F3); scene resets (F7); audio absence (interim above); crumb pass cost growing over length (measure, don't assume). None of these kill the stack; each has a cheap probe (ladder).

---

## 3 · Local skills catalog scan — video / AI-film / generative media (task item a)

Method: scan of `~/.hermes/skills/` (30 video-adjacent SKILL.md files located), usage evidence = `rg` over `~/.hermes/workspaces` + `kanban/logs` + the capability map `t_5e5b981c` (`~/.hermes/workspaces/capability_map/CAPABILITY_MAP.md`). "Unused" = **no use found in records as of 2026-09-27**, not a claim it can't work.

**Top finds (unused or first-use-now) with concrete uses:**

| # | Skill | Status | Concrete use for this campaign |
|---|-------|--------|-------------------------------|
| 1 | `creative/comfyui` (v5.1.0) | **First use now** (ladder card loads it) | Script the box headless: REST/WebSocket runs, editor→API JSON conversion for official templates, install+run the deflicker node, batch sweeps |
| 2 | `creative/open-weight-video-generation` | **First use now** (ladder card) | The quality lever: HQ vs draft profiles (remove lightx2v, ~20 steps, CFG 3.5); audit settings BEFORE evaluating any new model |
| 3 | `creative/architectural-fpv-concept-video` | **Unused (0 records)** — purpose-built (Slava+Zeph) | Its tested Wan FLF2V doorway-transition recipe = invented FPV motion from house stills (exactly what rescue demos + house content need); its "exact geometry → 3D route" clause matches our spine |
| 4 | `creative/dream-loop` | **Unused (0 records)** | Concept-art fidelity loop for 3D scenes → for brainsnn web demo scenes (three.js), NOT the video pipeline |
| 5 | `creative/unreal-mcp` | **Unused (0 records)** | Alternative renderer for control passes if Blender's look caps out (needs UE install — prerequisite) |
| 6 | `video-use` (staged repo, `~/.hermes/workspaces/video-use/`, NOT installed) | **Unused** | Edit-by-conversation (ffmpeg/PIL helpers, hard production rules) — demo cuts + rescue deliverable finishing |
| 7 | `creative/ai-presenter-video` | **Unused (1 mention = map)** | Offer explainer / presenter-style demo video for the coherence + rescue services |
| 8 | `media-use` + `media/songsee` + `audiocraft` + `songwriting-and-ai-music` + `heartmula` | Planned/unused | Sound layer: our clips are silent; free-path BGM/SFX/voice; spectrograms for audio QA |
| 9 | `manim-video` / `ascii-video` / `remotion-best-practices` / `hyperframes` set | Mixed (hyperframes first-use planned) | Metric visualizations (coherence half-life charts), title cards, site/demo motion graphics |
| 10 | `video-review-harness`, `ffmpeg-video-assembly`, `short-form-video-assembly`, `shortform-ai-video-production`, `news-to-shortform-reels`, `kanban-video-orchestrator` | In use / conditional | Keep: review gate for ladder clips; assembly + compliance for content; orchestrator for the 5–10 min demo build |

Also available as tools (not skills): the **40-tool Blender MCP** deferred toolset (drive scene edits programmatically), `video_analyze` (QA reviewer for clips), `video_generate` (cloud T2V), and the staged `blender_mcp_addon.py` (`~/.hermes/cache/scratch/`).

## 4 · Last-60-days web scan (task item b) — what's not in our sweep

| Date (2026) | Item | Why it matters | Link |
|---|---|---|---|
| Jul 17 | **TANGO** — test-time noise-guided adaptation for AR video | TTT lineage for long video; watch-don't-build | arXiv 2607.15849 |
| Jul 31 → Aug 3 | **MiniMax H3** announce → **open weights**; Seedance 2.5 (closed) | H3-FL2VA = video+audio from first/last frames; on our box but RAM-blocked | <https://magichour.ai/blog/ai-video-model-release-tracker-2026>, <https://huggingface.co/MiniMaxAI/MiniMax-H3> |
| Aug 11 | **LTX-2.5** released, day-0 ComfyUI; native 4K + synchronized audio ≤50 fps; FLF2V workflow | The current open frontier; 70–80 GB files, 32 GB+ VRAM rec. → 4090 gets quant experiments only | <https://docs.comfy.org/tutorials/video/ltx/ltx-2-5>, <https://comfy.org/ltx-2.5>, <https://www.ithome.com/0/988/766.htm> |
| Aug 18–21 | LTX-2.5 guides + Prodia "pareto-optimal" deployment | Ecosystem maturing fast; check back monthly | <https://book.st-hakky.com/en/data-science/ltx-25-comfyui-workflow-guide>, <https://prodia.com/> |
| Sep 10–16 | Open-source landscape refreshes: LTX-2.3 "only OSS model with native audio, 4K/50fps"; Wan 2.2 / LTX-2 / HunyuanVideo 1.5 as the stack | Confirms Wan 2.2 = our base and LTX = upgrade path | <https://www.thundercompute.com/blog/best-open-source-ai-video-generation-models>, <https://www.hyperstack.cloud/blog/case-study/best-open-source-ai-video-generation-models>, <https://builderai.tools/blog/best-open-source-video-generation-stack-2026> |
| Sep 11 | **Runway multi-shot guide** ("combined last-frame technique") | Industry validates FLF-based multi-shot practice | <https://runway.com> |
| Sep 17 | Recency Forcing (already in our sweep — cross-note) | Forcing family still moving; check for inference code | (sweep) |
| Sep 24 | **Google CANVAS** AI video co-director + HardContinuityBench; multi-agent planning for minutes-long video | The strongest signal that long-form = planning + persistence | <https://research.google/blog/coherent-long-form-video-generation/> |
| ~ongoing | **comfyUi-deflicker** node; musubi-tuner GUI/ComfyUI node | The two most directly actionable new tools for us (F6, F2) | <https://github.com/karcsiha/comfyUi-deflicker>, <https://comfy.icu/node/MusubiWanLoraTrainer> |
| Watch | Wan: no open weights after 2.2; **Wan 3.0** (60B, 4K) "expected mid-2026" — still not shipped as of writing | If it opens under Apache 2.0 it's the base-model refresh event | <https://aiwiki.ai/wiki/wan_2_5>, <https://www.spheron.network/blog/deploy-wan-2-5-gpu-cloud/> |

Also access-shift context: OpenAI's Sora app closed Apr 26 and the Sora 2 API sunsets ~Sep 24, 2026 (tracker above) — the open-weight lane is where our stack strategy already is.

## 5 · Runnable-now shortlist — flagged for the experiment-ladder card (`t_70c59e88`)

Ordered by value/spend. Items 1, 3 already overlap the ladder's own plan; 2, 4, 5 are NEW from this round. Flagged via kanban comment on `t_70c59e88`.

1. **FLF2V keyframe control — CONFIRMED runnable.** Wan FLF2V-class template + on-box models; run endpoint-bridging across two adjacent P3 chunk boundaries vs plain overlap joins (same seeds). QA: seam frame-diff + drift slope. *Kill: bridges don't beat plain overlap beyond the pre-registered seam bound.*
2. **HQ-profile arm (NEW; zero new downloads).** Same hero chunk, `lightx2v` LoRAs **removed**, ~20 steps, CFG ≈3.5 (draft vs HQ). Compare vision + flicker/detail vs the 4-step draft. *Kill: no visible/felt gain within ≤40 min wall.*
3. **Fun-Camera trajectory arm** — models (2×15.3 GB) + camera nodes already on box; motion source swap test vs depth.
4. **Deflicker/restore rescue pass on `kk` clips (NEW; independent of generation).** `git clone karcsiha/comfyUi-deflicker` → `step_removal` + `temporal_smoothing`; SeedVR2 already installed; blind A/B vs single stages; feeds the $99 offer's honest language ("post-hoc restoration" vs in-loop coherence). *Kill: combined < best single +20 % flicker win, or motion loss.*
5. **Scene-reset probe (NEW; tiny).** 2-chunk chain with a hard cut; measure crumb cross-cut bleed and anchor reset behavior. *Kill: bleed persists after one fix cycle ⇒ single-take claims only.*

**Do NOT (this week):** H3 runs (RAM wall — needs a free-RAM retry or int8 loader work first); LTX-2.5 full download (files + VRAM; revisit as a quant experiment after the ladder); training runs mid-ladder (GPU budget conflict); any public posting (watch-gate).

## 6 · Kill-criteria index (quick reference)

- **F1:** bridge seams must beat plain overlap at matched seeds, else rescue-use only.
- **F2:** one scene-LoRA ≤2 h must beat base+crumb on seams/identity, else park; TTT: only when ≤24 GB runnable repo exists.
- **F3:** plan-driven rerun must cut retries ≥30 % or improve seams at equal GPU-time, else human-side.
- **F4:** ≤24 GB AND beats chain+crumb at equal GPU-minutes, else citation.
- **F5:** same bar when weights land; audit now.
- **F6:** combined rescue < best single +20 % flicker (no motion loss) ⇒ simplify.
- **F7:** cross-cut bleed after one fix cycle ⇒ single-take claims.
- **F8:** 720 p must show a blind perceptual win (else stay 480 p+SeedVR2); 5-min scaling must hold crumb time + storage.
- **F9:** per-layer above; stop if seams don't hold past ~3 min without new engineering.

## 7 · Sources

**7.1 Local (measured / read today)**
- Box inventory dump: `~/.hermes/kanban/workspaces/t_11e322ff/box_models_inventory.txt`
- Model set (comfy-house): diffusion_models = wan2.2 fun_camera hi/lo (2×15.30 GB), wan2.2 fun_control hi/lo (2×14.30 GB), wan2.2 i2v hi/lo (2×14.29 GB); checkpoints = flux1-dev-fp8 (17.25 GB), ltxv-13b-0.9.8-dev-fp8 (15.69 GB); loras = lightx2v 4-step hi/lo (2×1.23 GB), xiokip_flux / xiokip_v2 (0.31 GB each); controlnet = flux-depth-controlnet-v3 (1.49 GB); VAE = wan_2.1 + flux ae; text encoders = umt5_xxl fp8, t5xxl fp16, clip_l; clip_vision_h. custom_nodes = ComfyUI-Frame-Interpolation, seedvr2_videoupscaler, ComfyUI_GielisGeometry, websocket_image_save. ComfyUI `__version__ = "0.3.65"`.
- Box extras: `models-h3` (465 GB; FL2VA + Ref2VA diffusers trees) + `h3_smoke_out.log` (`H3-SMOKE-RC=137` at 401/1058 weights) + `h3_smoke_ram.log` (53 G used / 8 G avail); `seedvr/` (upscaled outputs); `trellis-smoke.log`, `pixal3d-smoke2.log` (GLB smokes exit 0); `FUNCAM_PLAN.md`; `gpu_queue/`; tmux `comfy8188b` + `gpuq`.
- Pilot + rescue artifacts: `~/.hermes/workspaces/brainsnn/3d_first_pilot/NOTES.md` (§P0–P3 numbers quoted); `~/.hermes/workspaces/brainsnn/rescue-demo/PICKS.md` (kk series); `~/.hermes/workspaces/brainsnn/3d_first_pilot/p3/` receipts.
- Repo docs: `reports/ENGINE_RESEARCH_SWEEP_2026.md`, `reports/3D_FIRST_RESEARCH.md`, `reports/ASTRA_3D_REVIEW.md`, `reviews/engine_deepdive_v2_task.md`, `reviews/engine_deepdive_v2_default_models.md`.
- Skills: `~/.hermes/skills/creative/{comfyui,open-weight-video-generation,architectural-fpv-concept-video,dream-loop,unreal-mcp,media-use,ai-presenter-video}/SKILL.md`; `~/.hermes/workspaces/video-use/SKILL.md` (staged); `~/.hermes/workspaces/capability_map/CAPABILITY_MAP.md` (t_5e5b981c).
- MiniMax H3 local docs: `~/.hermes/workspaces/h3-docs/{README.md,license_qa.md,prompt_base.md,prompt_ref.md}` (community license; territory scope EU/UK/SK/US).

**7.2 Web (all accessed 2026-09-27; last-60-days emphasis)**
1. <https://magichour.ai/blog/ai-video-model-release-tracker-2026> — release tracker (H3 open weights Aug 3; LTX-2.5 Aug 11; Sora sunset).
2. <https://huggingface.co/MiniMaxAI/MiniMax-H3> + <https://github.com/MiniMax-AI/MiniMax-H3> — H3 model card (FL2VA / Ref2VA specs).
3. <https://docs.comfy.org/tutorials/video/ltx/ltx-2-5> + <https://comfy.org/ltx-2.5> — LTX-2.5 ComfyUI (day-0; 4K + audio; FLF2V workflows).
4. <https://www.earngenix.com/workflows/ltx-2-5-comfyui> — LTX-2.5 official workflows incl. FLF2V (Aug 13).
5. <https://book.st-hakky.com/en/data-science/ltx-25-comfyui-workflow-guide> — LTX-2.5 VRAM/file-size reality (Sep 18).
6. <https://www.ithome.com/0/988/766.htm> — LTX-2.5 launch detail (Aug 12).
7. <https://www.thundercompute.com/blog/best-open-source-ai-video-generation-models> — LTX-2.3 native-audio OSS claim (Sep 10).
8. <https://www.hyperstack.cloud/blog/case-study/best-open-source-ai-video-generation-models> (Sep 16); <https://builderai.tools/blog/best-open-source-video-generation-stack-2026> (Aug 22); <https://www.pixazo.ai/blog/best-open-source-ai-video-generation-models> (Sep 8).
9. <https://research.google/blog/coherent-long-form-video-generation/> — CANVAS AI video co-director (Sep 24).
10. <https://docs.comfy.org/tutorials/video/wan/wan-flf> + <https://deepwiki.com/Wan-Video/Wan2.1/6.3-first-last-frame-to-video-(wanflf2v)> + <https://wan2.video/wiki/first-last-frame-video> + <https://comfyui-wiki.com/en/models/wan/flf2v> — Wan FLF2V.
11. <https://github.com/roboticwink1/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB> — FLF2V degradation fix.
12. <https://github.com/karcsiha/comfyUi-deflicker> — deflicker node (step removal + temporal smoothing).
13. <https://github.com/OpenImagingLab/FlashVSR> + <https://zhuang2002.github.io/FlashVSR/> + arXiv 2510.12747 — FlashVSR (CVPR'26).
14. <https://github.com/chenyanglei/all-in-one-deflicker> — NFFA blind video deflickering.
15. arXiv 2510.09996 (+ NeurIPS 2025 track PDF) — BurstDeflicker multi-frame flicker benchmark.
16. arXiv 2512.12372 — STAGE/STEP2 storyboarded start-end pair generation.
17. arXiv 2412.07750 — neural storyboard / multi-shot character consistency.
18. arXiv 2512.04519 — VideoSSM; arXiv 2505.20171 — Long-Context SSM Video World Models (ICCV'25); arXiv 2403.07711 — SSM + video diffusion.
19. arXiv 2606.22370 — "Towards Error-Free Long Video Generation" (T-RFlow, frequency-band error control).
20. arXiv 2504.05298 — TTT one-minute video generation; arXiv 2606.26515 — TTT for long videos; arXiv 2607.15849 — TANGO; arXiv 2607.24611 — TTA dual distillation (ACM MM'26).
21. <https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI> + <https://comfy.icu/node/MusubiWanLoraTrainer> — Wan 2.2 LoRA training (musubi).
22. <https://aiwiki.ai/wiki/wan_2_5> + <https://www.spheron.network/blog/deploy-wan-2-5-gpu-cloud/> — Wan 2.5/2.6/2.7 and Wan 3.0 status.
23. <https://ltx.io/blog/open-source-video-generation-models-guide>, <https://www.sevenlabs.site/blogs/best-open-source-video-generation-models-2026> — landscape guides.

*End of report. Nothing here ran on the GPU; every runnable-now item is flagged for `t_70c59e88` and ownership of running stays with the ladder card.*
