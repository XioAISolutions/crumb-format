# RESEARCH ASSEMBLY — everything we've researched, adopted, and built
Filed 2026-09-26/27 · Zeph · companion to this session's work. Canonical copies: this workspace + crumb repo `experiments/wavefield_video/reports/`.

## 0 · TL;DR map
Two products, one engine, one loop:
- **BrainSNN — The Coherence Layer for AI Video.** Today: $99 "AI Video Rescue" (fix drift/flicker on existing footage). Vision: long-form coherence past the 2–5 minute wall.
- **crumb_coherence engine** — the working core (spectral temporal-coherence pass; M0/M1 passed, M2 trap failed → M3 fix pending).
- **The loop** — kanban dispatcher + 3 specialist profiles + cron + webhooks + 4090 pipeline; it works while nobody watches.

## 1 · Mission & positioning (researched)
- **THE_POSITION.md** — "Generation is getting longer. Coherence isn't keeping up." Frontier models die at 8–25s (Veo ~8, Kling ~15, Sora 2 Pro ~25). Long-form is the open flank.
- Ladder: rescue ≤15s $49–79 · 15–60s $99–199 · 1–3min $249–499 · agency $500–1500+ · engine access $500–2,000/mo (design partners, research preview).
- Concierge-first (manual processing, 20–50 clips, then automate). No SaaS claims. No photorealism promises.
- Competitor benchmarks: Topaz $39–74/mo, UniFab $320–500; market truth + white space filed in THE_POSITION.

## 2 · 3D-first technology track (new, deep)
**3D_FIRST_RESEARCH.md** — the "Blender method" answer: control-first generation. 4 pipelines:
- A pure Blender render (TRELLIS.2 assets; 24GB VRAM = our 4090 exactly; MIT)
- **B Blender → depth/edge passes → Wan 2.2 Fun-Control [PILOT]**
- C GEN3C-style 3D-cache diffusion (watch)
- D video→3DGS→re-render loops (later; GaussFusion CVPR'26)
Plus: Stable Virtual Camera (non-commercial!), ReCamMaster (Kling, open), WonderWorld, GaussVideoDreamer. crumb = the finishing pass over ALL pipelines.
**In motion:** blender-mcp installed (Blender 5.1 addon + 36-tool Hermes MCP); Fun-Control + Fun-Camera fp8 models downloading to the box (55GB, tmux `wandl`); pilot card t_80b6e826 queued (P0 headless scene → P1 controlled render → P2 placement test → P3 60s+ chain).

## 3 · Engine science status
- **M0 arc COMPLETE** (luminance 87–92%, hotspot 64.0% / 0.9946 / 99.7%). **M1 DONE** (−86% variance, −56% centroid drift, 0.988 detail).
- **M2 trap FAILED** (96f kept 54.7% travel / 64% distortion; 192f 70.8% / 58.9%) — engine fights legitimate slow motion. Product-killer caught in lab, pre-sale.
- **ENG-1 1080p FAIL** confirmed on real renders (motion 70%/54%; 2.9× jitter; detail 0.96–0.98; 64% drift removal).
- **Fix = velocity-aware gate (M3)** — task armed in repo (`reviews/claude_m3_gate_task.md`); fires when Claude window resets. **Needs owner "re-arm"** (timer died in the gateway restart).
- Config: `--hotspot-mc-band 0.03 --hotspot-mc-strength 0.5 --hotspot-phase-anchor 0.0`.

## 4 · Hermes system mastery (the 10-hacks article → our stack)
Researched from the viral X playbook + Hermes docs. **Now active:**
- **Kanban dispatcher** — 8 cards through it today; workers run autonomously (design, QA, skills, renders), self-block correctly.
- **Cron** — gpu-temp-sampler, scheduled fires; **Dashboard** — localhost:9119 (skills/models/cron/profiles/kanban).
- **Desktop app** — installed + running on the Mac (native, same data dir).
- **Webhooks** — enabled; `signal` route live + end-to-end tested (accepted→agent→answered).
- **Profiles** — designer / video / ops (cloned config, kanban-routable). `gateway migrate --multiplex` available when needed.
- **Structured goals** — skill written (OUTCOME/SOURCES/CONSTRAINTS/DELIVERABLE).
- **MCP** — chatgpt (image gen) + **blender (36 tools)** registered.
- **dynamic-workflow** — official skill installed (plan-in-code fan-outs, adversarial convergence).

## 5 · Skill arsenal (installed today)
**Design:** frontend-design (Anthropic) · web-design-guidelines (Vercel) · anti-ui-slop + ui-design (uizze v1.3.0) · auteur (cinematic pages) · impeccable · apple-grade-web (ours).
**Video:** hyperframes set · remotion-best-practices · video-edit · media-use · kanban-video-orchestrator.
**Quality gates:** adversarial-ux-test · grill-me.
**Swarm/orchestration:** darwinian-evolver · agent-merge-conflict-arbiter.
**Ops/research:** watchers · searxng-search · duckduckgo-search.
Sources: skills.sh hub sweep (worker card, 14 verified) + official catalog (inventory scanned end-to-end).

## 6 · External tools evaluated (owner queue)
- **Composio** (composio.dev/hermes) — ~1,500 app tools via one MCP. REC: yes, pick toolkits. Needs owner account/OAuth. PENDING.
- **TypeUI** — Hermes plugin staged (hosted MCP, account). PENDING.
- **Firecrawl** — web-data standard; CLI+skills; self-host free; AGPL (no shipping embedded). Eval queued. NO ACTION NEEDED YET.
- **SkillClaw** — local no-paid-API skill-evolution loop proven feasible; spike card t_0a70b122 blocked on owner go/no-go. PENDING.
- **youtube-skills repo** — skipped (paid transcript API; we have free paths).

## 7 · GPU pipeline (4090, 24/7)
- ComfyUI :8188, Wan 2.2 I2V 14B fp8 (high/low), make_video.py two-stage runner.
- **kk corpus: 23 takes** (house FPV bedroom→dressing room; QA'd: motion/flicker/drift metrics; PICKS.md + contact sheets; picks for rescue demos incl. kk77780 ghost-veil, kk01 furniture ghosts, kk77785 mirror smears).
- Comfy3D + TRELLIS (pixal3d) + Depth-ControlNet assets on box; **Fun-Control/Fun-Camera downloading**.
- Runs: batch chain kk77772→77786+ (5-batch cadence, ~17 min/batch); GPU verified 100%/250W/62°C during, 0-lag between batches.

## 8 · Content & money lanes (state)
- **Funcyclopedia** (YouTube) — content lane staged; reels pipeline + watch-gate (Slava approves every cut personally).
- **Money:** Fiverr/Freelancer accounts live; Freelancer job 40731392 awaits owner e-sign; plus auto-renew $13.95 CAD Oct 23; quotas noted. 72h experiment doc filed (EXPERIMENT_72H.md).

## 9 · Owner decision queue
1. **"Re-arm"** — restart the engine-fix (M3) timer. (Blocked by safety gate awaiting confirmation.)
2. **SkillClaw spike** — go/no-go (local evolution loop).
3. **Composio** — wire it? (recommended: start with 1–2 toolkits.)
4. **TypeUI** — enable? (free tier.)
5. v4 landing — preview at localhost:4179/coherence (watch-gate: no merge/publish until your review).
6. DNS: naked brainsnn.com 404 (GoDaddy apex limitation; options filed `brainsnn-dns-fix.txt`).

## 10 · File index (where things live)
- Workspace: `~/.hermes/workspaces/brainsnn/` — THE_POSITION.md · DESIGN_APPLE_LEVEL.md · DESIGN_CRITIQUE_gpt.md · 3D_FIRST_RESEARCH.md · EXPERIMENT_72H.md · ASTRA_3D_REVIEW.md (in flight)
- Rescue material: `~/.hermes/workspaces/brainsnn/rescue-demo/` (PICKS.md, picks/, sheets/)
- Engine repo: `~/crumb-format/experiments/wavefield_video/` (plugin, reports/, reviews/, renders/)
- Site: `~/the-brain/brainsnn-r3f-app` (branch feat/coherence-landing; PR #168)
- Box: `/workspace/slava/comfy-house/` (runners/, output/kk/) · scripts `/workspace/slava/*.sh`
- Kanban: board `default` (~/.hermes/kanban/)

## 11 · How the loop runs itself
Cron + kanban dispatcher + specialist profiles do the recurring work (renders, QA, sweeps, monitors); webhooks take external triggers; I orchestrate, synthesize into reports like this one, and escalate only owner-level decisions. Every artifact gets committed to its repo; every claim carries evidence paths; nothing ships publicly without the watch-gate.
