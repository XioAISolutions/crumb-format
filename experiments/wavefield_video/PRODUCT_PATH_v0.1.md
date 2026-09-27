# PRODUCT PATH v0.1 — a video generator sellable in Higgsfield's mold

Task: kanban `t_278e5ba1` · Date: 2026-09-27 (EDT) · Author: Zeph (kanban worker)
Owner directive (verbatim, 2026-09-27): *"focus all energy on these newer pathways to get our video LLM running and sellable like higgsfield."*

Two halves of the directive: **RUNNING** (in motion on the box — REF chunk, ladder rungs, r15 suite, attn band; see §b.3) and **SELLABLE** (this document).
Constraints honored throughout: nothing public without an owner watch-gate on the exact asset; no accounts/payments without owner; never claim photorealism; coherence is the axis (not camera tricks); deploys are repo-only (the-brain). Every external fact below carries URL + fetch date; every local fact carries a receipt path.

**Status of this doc: v0.1 for owner review. It authorizes nothing by itself.**

---

## 0. TL;DR

1. **Higgsfield is a funnel company, not a model company.** Verified: $5.4B valuation / $400M Series B (Aug 2026), ~15M creators, 4.5M clips/day, a dozen+ rented models under one roof, and a growth engine that is part trend-jacking, part credit math, part community flywheel — plus a documented dark side (fabricated demos, unpaid creators, deepfake freeloading, "unlimited" plans that weren't). What's copyable is the *mold*. What's not is their scale — and some of their tactics must never be copied (§a).
2. **The field's ceiling is verified and it is exactly our wedge.** Native single generations: 8–30 s (Seedance 2.5 holds the 30 s record). Extension chains: 120 s (Sora API — sunsetting), 148 s (Veo — 720p-only), ~3 min (Kling — reviewers report drift past ~2 min). Verdict line from the market's own press: *"extension is a splice, not a memory."* Nobody sells multi-minute coherence because their architectures + economics degrade exactly there (§b.1).
3. **We have 2 minutes with a flat drift slope — mechanics proven, quality not yet.** The P4 chain: 1920 frames, 120.0 s, drift slope 3.1e-5/frame, no seam pops, uncontrolled baseline is a pixel-identical fixed point. But the ship verdict on the P4 output is **NOT ship-acceptable as convincing camera footage** (ghosted architecture at ~30/65–72/80/118 s). The quality ladder to fix that is *in flight right now* (§b.2, §c).
4. **Pick for v0: the authored-camera creator** ("Long Take Studio" working title) — scene library + directed camera + duration, rendered as one continuous take; concierge-operated behind the existing gpuq conductor; per-render pricing, no credits, no subscription. The extender form is a *feature* of it; both API forms are out this cycle, with different kill reasons (§d).
5. **The honest gap list is dominated by non-model items** — consumer UX, payments re-enable, queue UX, 1080p delivery, legal license check. The model side is the *least* of the missing pieces for v0; the generator quality ladder and the operator-time economics are what decide go/no-go (§c, §d.4).
6. **First sellable moment = first collected payment from a stranger for a delivered take they accept.** Not a demo, not a like, not a waitlist signup. Everything before it is pipeline (§f).

---

## (a) Higgsfield teardown — copyable vs not-copyable

### a.1 Verified facts (fetched in-run 2026-09-27)

| Fact | Value | Source (URL) | Date |
|---|---|---|---|
| Pricing | Starter $19/mo = 270 cr · Plus $47/mo (annual; $59 monthly) = 1,200 cr · Ultra $99/mo (annual; $129 monthly) = 3,000 cr | techsifted.com/roundups/higgsfield-ai-pricing-2026/ | Sep 5, 2026 |
| Credit mechanics | Credits don't roll over; auto-refill $1 = 18 cr; premium video ≈ 6.5 cr/s (Seedance 2.5 @720p), ≈ 9 cr/s (Seedance 2.0 @1080p), Kling 3.0 8 s ≈ 14 cr; "unlimited" models work web-app-only — MCP/CLI always burn credits | techsifted (same); krea.ai/blog/higgsfield-pricing-explained-2026 | Sep 5, 2026 |
| Price variations reported | $15/$39/$99 annual-billing variants; Team $65/seat | layer3labs.io/guides/higgsfield-ai-pricing; creatify.ai/blog/higgsfield-pricing-(2026) | Aug 2026 |
| Scale | Jan 2026: $80M Series A ext. at $1.3B valuation (Reuters-confirmed) · Aug 2026: $400M Series B led by DST Global at $5.4B; ~$700M annualized revenue claim; 390 Fortune-500 clients | techcrunch.com (Jan 15, 2026); pomegra.io (Aug 26, 2026); startupwired.com (Aug 17, 2026); thecodew.com (Aug 2026) | verified 2026-09-27 |
| Scale (earlier data point) | ~15M creators, ~4.5M clips/day, ~$300M ARR (Feb 2026) | forbes (Feb 2026, via nationalcybersecurity.com reprint) | Feb 2026 |
| Note — inconsistent ARR snapshots across outlets ($200M/$300M/$500M/$700M) | treat revenue claims as reported claims, not audited | getlatka.com/companies/higgsfield.ai (Sep 15, 2026) vs masternodeai.com | 2026 |
| Product surface (live) | Seedance 2.5 ("top model"), Nano Banana Pro, Genjutsu ("one upload in — endless new visions out"), Cinema Studio 4.0, "Supercomputer" agent (GPT-6 Astra), MCP for Claude + plugin for ChatGPT, API Cashback (100% back up to $100K), Global Film Festival (submissions live), SFX gallery with **Recreate** buttons, signup discount hook ("Get unlimited Nano Banana Pro + extra discount") | higgsfield.ai (live fetch) | 2026-09-27 |
| Dark side (documented) | Fabricated demos (stock templates passed off as AI-made); non-consensual/racist deepfake examples; refund-wall + mass bans of paying users after "unlimited" promos (Dec 2025); unpaid "Higgsfield Earn" creators; fake Trustpilot reviews; X account suspended for inauthentic behavior (~Feb 10, 2026). Cofounder on record: *"We fully admit that we push the envelope... it's more controversial content that gets attention."* | forbes.com investigation, Rashi Shrivastava (Feb 11, 2026), via nationalcybersecurity.com reprint (Feb 18, 2026) | Feb 2026 |

### a.2 Copyable mechanics (and what each becomes in our v0)

1. **Consumer UX: pick-a-shot, not prompt-an-API.** Their surface is browsable presets/effects/recreates — the user never faces a blank text box. -> Our studio ships as a *shot picker* over authored scenes (scene x camera move x duration), which is also the only UI our pipeline can honestly power today (§d.2).
2. **Credit-burn economics as a *pricing anchor*.** Their $19/$47/$99 tiers set what consumers already pay for AI video. -> We anchor v0 per-render prices to these numbers, but *without* the credit-expiry mechanic (see a.3).
3. **Viral content engine: clip-first publishing with a "Recreate" action.** Every gallery item is a shareable proof + a funnel entry. -> Our equivalent is the side-by-side format we already own (uncontrolled vs controlled; §e) — the single most shareable proof asset in the campaign.
4. **Community flywheel (film festival, creator program).** -> Later; a curated "first 10 takes" showcase is the v0-sized version. Payouts must be reliable and public — the direct contrast to their Earn program complaints.
5. **Distribution arbitrage (MCP/plugin into ChatGPT/Claude).** -> Not available to us yet (no hosted generator); park as v1 item.
6. **Signup incentive / founder-led scarcity.** -> "First 10 founding takes" is our honest version.

### a.3 Never copy (documented trust failures — these are the anti-brand)

- Fabricated demos, stock footage passed off as generated output. We publish **receipts** (seed, settings, drift numbers) instead — it's a differentiator, not overhead.
- "Unlimited" claims that aren't; credits that silently vanish; refund walls.
- Non-consensual likeness content and ragebait marketing. Our trust posture ("verifiable, no photorealism claims, labels everywhere") is the anti-Higgsfield posture and should be said out loud on the site eventually — **owner decision**, and only after our own receipts are public.
- Unpaid creator programs.

### a.4 Not copyable (their stack/scale)

- A rented-model marketplace (a dozen+ model contracts) — no leverage for us; we have a pipeline, not a fleet of partners.
- $5.4B / ~15M creators / 4.5M clips-per-day distribution and brand. Any plan that implicitly assumes their reach is fiction. Our v0 is concierge-scale by construction (§d.4).

---

## (b) Our wedge — multi-minute coherence

### b.1 The field's verified ceiling (the hole we sell into)

From the market's own length survey (invideo.io/blog/ai-video-length-limits/, Aug 7, 2026):

- **Native single generations: 8–30 s.** Seedance 2.5 = 30 s record; FLUX 3 = 20 s; Veo 3.1 = 8 s.
- **Extension chains:** Sora API 120 s (API sunsets Sep 24, 2026); Veo 148 s (720p-only, Veo-source-only, 2-day storage timer); Kling ~3 min (paid; "reviewers report drift past ~2 minutes"); PixVerse practical consensus ~60 s before coherence decays.
- The tell: *"Extension is a splice, not a memory; characters and lighting drift a little at every seam, which is why no model advertises unlimited chaining even when its docs set no cap."*

So: at 2 minutes, the market's options are (a) spliced extensions with documented drift, (b) resolution penalties, or (c) don't offer it. **Nobody's promise survives 2 minutes.** That is the regime our wedge lives in.

### b.2 What we've already proven (and the claim boundary)

Proven (receipts in §Receipts index):
- **2-minute chain (P4):** 20 chunks x 121 frames, 832x480 @16 fps, stitched to exactly 120.0 s (1920 f). Drift slope **3.1e-5/frame** (raw and crumb), geometry-tracking recall mean 0.679, seam ratios with no boundary pops; controlled vs uncontrolled ~3–5x tracking, and the uncontrolled baseline is a *pixel-identical fixed point* (no state = repeats the same ~6 s dream — this is itself a demo).
- **60-second chain (P3):** zero cumulative drift over 960 frames, byte-identical re-stitch from the same chunk files (md5), occlusion held (doorway crossed twice).
- **Crumb (our spectral pass) on top:** −51.3% low-band trajectory variance, −49.8% luma flicker on the 2-min, texture/detail preserved (retention 0.996–1.005) — with a disclosed tone trade (broad contrast 0.829 at alpha 0.5; alpha 0.25 halves both sides of the trade). Raw stays the default presentation until the owner rules on the veil trade-off.
- **Mechanism sentence (the important one):** every chunk is generated *anchored to an authored scene* (dense-depth control from a Blender render with fixed clip bounds) — the scene remembers geometry, so drift cannot accumulate across the chain. Their splices condition on the last clip; ours condition on the world.

**Claim boundary — say it in every caption:** single-take authored scenes; 832x480 validated; quality below ship bar today (ghosting defects — the ladder in flight targets exactly this); no photorealism claims; "research preview" labels on any external artifact.

### b.3 Quality ladder in flight (state as of 2026-09-27 11:00 EDT)

| Item | What it decides | State |
|---|---|---|
| L0–L5 arms (texture: steps/LoRA strength/canvas/split) | Is the softness the 4-step distill or the bare scene? | queued on box (gpuq jobs 1a), behind p4u jobs 18–19 |
| K1–K2 arms (kk/FLF2V family, steps) | Does sampling fix the FLF2V look? | queued (job 1b) |
| REF chunk (20-step, no LoRA, cfg 3.5) | TEXTURE CEILING — resolves the question above | queued (job 1c); ETA output ~12:05 EDT today |
| R2-A scene v2 (props/materials/light/edges) | The scene-side floor (“the scene is the content”) | built + independently verified; tone gap ~1 stop under photo target, documented; swaps into the winner A/B |
| r15 suite (264 jobs: wave/attn/hypercomplex per-byte arms) | Which model architecture carries the long-horizon claim | queued behind the audit jobs (conductor FIFO) |
| attn ladder (D6/1024-frame focus) | Attention vs wave at every rung, equal-bytes | deployed; job 574 queued |
| P4 2-min quality verdict | Ship gate | CLOSED (ghosting at ~30/65–72/80/118 s; raw default) |

The sellable lane consumes the ladder's winners; it does not gate them.

---

## (c) HAVE vs MISSING — the sellable-v0 gap table (brutally honest)

| # | Capability | Have (receipt) | Missing to sell v0 | Effort/time-box |
|---|---|---|---|---|
| 1 | **Generator** | Authored-motion pipeline: Blender depth control -> Wan 2.2 Fun-Control 4-step -> overlap stitch -> crumb (P3 60 s, P4 120 s receipts). Wavefield/attn model arms = research track (r15 suite), not shippable as the generator yet. | Ship-quality look (ghosting/melt) = the open ladder; 1080p delivery path (native 480p; 1024x576 parity at +40% wall; SeedVR2 upscale installed, unvalidated for delivery) | rung-1/2 ladder (in flight); upscale validation gated by texture battery |
| 2 | **Coherence story** | 2-min flat drift slope, no seam pops, uncontrolled-baseline proof, md5 re-stitch | Blind third-party comparison to survive skepticism (needs owner-decided account/budget); a frozen demo protocol doc | 1 day once quality gate passes |
| 3 | **Consumer UX** | Coherence landing v4 built on branch `feat/coherence-landing` (local; 707 tests green; placeholder 21 s demo assets) | Landing targets the OLD rescue offer — needs the generator offer block; a studio/config screen, order flow, delivery page (spec in §d.2) | Landing edit: 1–2 days (owner eyeball first). Studio: ~1–2 weeks after decisions |
| 4 | **Job queue (consumer-facing)** | gpuq conductor on the box (operator-driven, FIFO, per-arm logs, receipts discipline) | Request intake, status/progress surfacing, email notifications, per-customer isolation; box capacity is ONE 4090 | Concierge phase: operator IS the queue (no build). Self-serve: later, after N>=20 orders |
| 5 | **Payments** | the-brain repo has Stripe plumbing from the sponsorship era (checkout currently paused) | Re-enable decision (owner), new products/prices, refund policy text, tax/receipt sanity | 1 day after owner decides price |
| 6 | **Credits / pricing model** | Nothing built (and that's fine) | A price list. Recommendation: per-render, no credits, no expiry (§d.4) | Owner decision this week |
| 7 | **GPU $ per render** | Measured: ~170 s/chunk (121 f, 832x480, power-capped ~250 W); 20 chunks = ~57 min gen per 2-min; recoverable +25–30% uncapped | Upscale pass cost (unmeasured); multi-box capacity path (not needed for v0) | Measured; upscale cost from first validation run |
| 8 | **Legal** | — | Wan 2.2 + lightx2v LoRA + Fun-Control component licenses checked for commercial output sale; likeness policy; refund T&Cs | 1 day, before first paid order (non-negotiable gate) |
| 9 | **Support/ops** | Operator (owner+Zeph) can run the concierge flow | A written runbook (draft in §d.3); delivery email template; QA checklist is already battle-tested (metrics_v2 + eyes) | Half a day |
| 10 | **Analytics/funnel events** | Site event plumbing exists (landing v4 registers coherence funnel events; server allowlist fixed) | Re-point events to the new offer; define the funnel (visit -> demo view -> order -> paid -> delivered) | Part of landing edit |
| 11 | **Demo assets (owner-gated)** | `sbs_120s_uncontrolled_vs_crumb.mp4`, `pilot_120s*.mp4`, 60 s SBS, crumb-battery panels, sheets | Owner watch-gate on the exact cuts; web encodes (the 2-min SBS is 36 MB — needs web mp4/webm + poster); caption approval | Owner + 1 hour encode time AFTER approval |
| 12 | **Brand/name** | "BrainSNN" umbrella; landing in Space Grotesk / dark-violet palette (v4) | Product name for the studio (working: "Long Take Studio"); voice line; the anti-claim policy text | Owner decision |

**Reading:** items 3, 5, 6, 7, 9 are engineering- or decision-light and mostly exist; item 1 (quality) is the true gate; item 4's consumer build is deliberately deferred (concierge first); items 2/8 are the trust essentials we can't skip.

---

## (d) Candidate v0 forms + PICK

### d.1 The three candidates, scored straight

**C1 — "Make it long" extender.** User gives a clip (or scene); we extend it to 2–5 min with coherence.
- Foreign-footage extension (arbitrary user clip -> longer): **already-falsified class.** The content-blind middleware branch is closed (M3 gate RED, M4 killed 0/120; correction cannot separate drift from legitimate motion — receipts: IMPL_NOTES_M3.md / M4.md). Do not resell it.
- Our-scene extension ("keep this take going"): viable — it's really C2 with a different entry point. -> Fold into C2 as a feature, not a form.

**C2 — Authored-camera creator (PICK).** Pick a scene from a library, choose a camera move, choose duration; receive one continuous multi-minute take. This is Higgsfield's "cinematic camera controls" mold applied where our proof actually lives.
- Every subsystem is proven at 2 min today (P4) or in the ladder (scene v2, texture arms).
- The consumer interaction ("pick a shot") matches both their UX grammar and our real affordances; no fake text-to-video promises.
- The wedge feature is the duration slider that goes to 2:00 — a number their platforms structurally can't match in one coherent take.

**C3 — Coherence API.** Two flavors, both out this cycle:
- Post-process API ("video in, corrected video out"): dead — killed with the middleware branch; a mode that can't address a drift class can even make it worse (magnitude-mode hotspot: −18%).
- Generation API (hosted render endpoint for our pipeline): deferred, not rejected. Needs accounts, isolation, a hosted queue, and — per our own compute-economics discipline — "the GPU is the warehouse, not the product": raw API-hours lose money; the deliverable carries the price. Revisit after C2 has paying orders.

**PICK: C2, concierge-operated, per-render pricing.** C1-as-feature (extend a completed take) and C3-generation deferred.

### d.2 Wireframe-level spec (v0)

Customer flow (all screens re-use the landing v4 language: near-black, cyan/violet accents, Space Grotesk, receipt-forward boxes):

```
[1] /coherence  (edit of v4 landing — shell/player/method sections stay; offer block swaps)
+--------------------------------------------------------------+
| Generation is getting longer. Coherence isn't keeping up.    |
|                                                              |
|  [ 2:00 side-by-side player: uncontrolled | controlled+crumb ]|  <- owner-gated asset
|                                                              |
|  "A 2-minute single take. Drift slope 0.00003/frame.        |
|   No seam pops. Research preview - authored scenes, 480p."   |
|                                                              |
|  [ Configure a take ]   [ See the receipts ]                 |
+--------------------------------------------------------------+

[2] Studio config (one screen, three controls + price)
+--------------------------------------------------------------+
| Scene   ( ) House - dining/living       [preview still]       |
|         ( ) House - doorway reverse     [preview still]      |
| Camera  ( ) Glide-through  ( ) Close pass  ( ) Doorway swing |
| Length  ( ) 0:30  ( ) 1:00  ( ) 2:00                          |
| Stability  [x] tone-stabilize (crumb a=0.5, disclosure shown) |
|                                        Price: $__  [Order]    |
+--------------------------------------------------------------+
| Notes: scenes are authored (not text-to-video). Renders take  |
| ~15-60 min. No credits, no expiry. Refund if it melts.        |
+--------------------------------------------------------------+

[3] Order / queue status (simple authed page)
+--------------------------------------------------------------+
| Order #12 - rendering  chunk 7/20  [progress bar]  ETA ~35m  |
| You'll get an email + link when it's stitched and QA'd.      |
+--------------------------------------------------------------+

[4] Delivery page (email link)
+--------------------------------------------------------------+
| [player: your take]   [download mp4]                          |
| Receipts: scene v2.1 | path shot2 | seed 20260926 |          |
|   crumb a=0.5 | drift slope measured | vs uncontrolled clip  |
| [ report an artifact -> free re-render ]                      |
+--------------------------------------------------------------+
```

Ops runbook (concierge v0 — the operator is the queue):
1. Order lands (Stripe + form) -> owner/Zeph read it within the day.
2. Build control pass (Blender; seconds) -> queue chunks on gpuq behind anything already running (FIFO discipline as today).
3. Stitch + crumb on the Mac (streamed; never buffer 1080p in RAM) -> QA: `metrics_v2` gates + eye panel + the owner watch-gate for the first N.
4. Deliver link + receipts; log to `rescue_orders.md`-style tracker (new: `product_orders.md`).
5. First 10 orders: measure operator minutes per order and artifact/refund rate (feeds §f kill gates).

Reuse inventory (so this is not a from-scratch build): landing v4 branch (copy/player/events) · the-brain Stripe plumbing (paused; re-enable decision) · gpuq conductor + watcher discipline · metrics_v2 QA + panels · all demo assets. The genuinely NEW build is screen [2] + [3] + [4] (~1–2 weeks if/when green-lit) — concierge v0 can start with just [1]-edit + a form.

### d.3 GPU unit economics (measured) vs their price points (verified)

Measured on our 4090 (power-capped ~250 W; +25–30% slower than an uncapped card):

| Quantity | Measured | Notes |
|---|---|---|
| Chunk cost | 121 frames @832x480 ~= 170 s | P4 receipts, 20/20 chunks exit=0 |
| 2-min take, generation | ~3,400 s ≈ 57 min GPU (~0.94 GPU-h) | + stitch/crumb ~minutes local (Mac) |
| 576p penalty | +40% wall (1024x576 parity run) | ~1.3 GPU-h per 2-min |
| Upscale (SeedVR2 -> 1080p) | UNMEASURED | add from first validation run; do not promise 1080p until measured |
| GPU cost per 2-min take | **$0.38–$0.70** ($0.40/hr internal rate — $0.74/hr market hosting anchor) | compute-economics reference: raw GPU-hour resale LOSES money; price the deliverable |
| Capacity (single 4090, 24/7) | ~25 x 2-min takes/day @480p; ~18 @576p | v0 is capacity-bound, not compute-cost-bound |
| Per output-second COGS | ~$0.003–0.006 @480p | vs **$0.36–0.50 per premium second** on Higgsfield consumer credits (6.5–9 cr/s @ $1=18 cr) |

Their anchor numbers for pricing context: Plus $47/mo ≈ 133 s of 1080p premium video per month; Ultra $99/mo ≈ 5.5 min/mo, credits expire monthly.

**Proposed v0 pricing (owner decision):** per-render, no credits, no expiry — 0:30 $19 / 1:00 $39 / 2:00 $79, founding batch "first 10 takes" optionally discounted, refund if the take shows a new artifact class (re-render first). Rationale: (i) $79 buys 2 *continuous* minutes where $99 at Higgsfield buys ~5.5 spliced ones — the comparison is ours to make honestly; (ii) per-render keeps every promise checkable and avoids the credit-expiry trap; (iii) compute COGS is <2% of price at every tier — the binding v0 cost is operator QA time, which caps throughput at a handful of orders/day (measure it on the first 5).
**No subscription and no "unlimited" at v0** — capacity is one box; we will not repeat their most-documented sin.

---

## (e) Demo funnel — assets, captions, places, cadence (ALL owner-gated)

**Rule zero: nothing moves until the owner watches the exact file.** The 2-min is mechanism-proof, not yet ship-quality; the owner watch-gate (`sbs_120s_raw_left_crumb_right.mp4` + `sbs_120s_uncontrolled_vs_crumb.mp4`) decides what is even eligible.

Asset inventory (existing, local):

| Asset | Path | Use | State |
|---|---|---|---|
| 2-min uncontrolled vs controlled+crumb SBS | `3d_first_pilot/p4/sbs_120s_uncontrolled_vs_crumb.mp4` | Hero proof (mechanics: path vs fixed-point loop) | NEEDS OWNER WATCH; contains ghosting moments on the controlled side; web encode pending |
| 2-min raw vs crumb SBS | `generator_audit/panels/p4_ship/sbs_120s_raw_left_crumb_right.mp4` | The crumb trade-off decision | Owner watch (this week) |
| 2-min raw / crumb / uncontrolled singles | `3d_first_pilot/p4/pilot_120s{,_crumb,_u}.mp4` | Source cuts / excerpting | same |
| 60 s SBS + 60 s sbs | `3d_first_pilot/p3/sbs_uncontrolled_vs_crumb.mp4`, `pilot_60s_sbs.mp4` | Shorter fallback proof (excerpting likely lands better on social) | owner gate |
| Sheets / stills | `arm_results/sheet_p4_120s.png`, `qa120*/`, crumb-battery `eye/sbs_t*.png` | Static posts / receipts | owner gate |

Caption drafts (honest-label versions; owner edits at will):
1. X/social: *"AI video hits a wall around 30 s. Extensions are splices — every seam drifts. This is 2 continuous minutes with a flat drift slope: every 7-second chunk anchored to the same authored scene. Research preview. Receipts in reply."*
2. Hero sub-line (site): keep *"Generation is getting longer. Coherence isn't keeping up."* + mechanical caption: *"2:00 single take · drift slope 0.00003/frame · no seam pops · raw, unlisted numbers."*
3. Receipt-thread style: *"No photorealism claims. 480p. Authored scenes. What we measured: [drift slope, seam check, uncontrolled baseline is a fixed point]. What we didn't: [resolution, free-prompt flexibility]."*

Places (owner decisions flagged): the-brain `/coherence` (post-merge; owned surface, first home) · X (needs an account decision — the existing brand is Funcyclopedia, a different voice) · Reddit/communities LAST, no spam, only after proof assets are up and with owner-approved text.

Weekly cadence proposal (post-gate only): Tue = one clip (30–60 s excerpt of best windows, full video linked); Fri = one receipt card (numbers, honest band); backlog 4+ approved assets BEFORE the first post so cadence never forces a weak post. During ladder/quality weeks: internal comms only (board + owner), zero public posts.

---

## (f) Kill criteria + the first "sellable moment"

**Definitions (staged):**
- Sellable-0: an asset the owner approves as publishable, with honest labels.
- Sellable-1: a live order path (form -> payment -> delivery) that runs end-to-end without hand-holding beyond operator QA.
- **First sellable moment (Sellable-2): the first collected payment from someone who is not a friend, for a delivered take they accept (no refund).** That is the moment the pathway stops being a plan. Everything else is instrumentation.

**Kill criteria (pre-registered; each has a measurement):**
1. **Quality (the gate):** if the current ladder cycle (L/K arms + R2-A scene + REF texture ceiling) cannot produce a take that passes the owner watch-gate as convincing footage — i.e., ghost/melt artifacts survive every advancing arm — then the consumer lane is killed this cycle; remaining value = research + investor/design-partner demos. Measurement: owner watch-gate verdict on the winner chain, one pass.
2. **Wedge survival (blind):** a blind side-by-side of our 2-min take vs the best publicly available ~3-min chain (e.g., Kling extension; requires owner-decided account + budget) must show decisively better stability (drift slope / seam pops / white-clip counts). If blind viewers can't see the difference, the coherence claim dies even if we can — reposition or park. Escalation: if the owner declines the external account, this gate defers and the wedge remains "structural limits (cited) + our own uncontrolled-baseline comparison."
3. **Economics:** operator time + GPU per delivered take must fit under ~20% of price (measured over the first 5 orders). If operator QA is eating the product, v0 shrinks (30 s takes only) or prices change — one rework, then decide.
4. **Delivery time:** 2:00 takes must deliver in <= ~90 min wall end-to-end (today: ~60–70 min gen+post, compatible); if a future quality fix blows past 2 h, cap v0 at 0:30–1:00.
5. **Demand:** after one owner-approved outreach round (10 real prospects, e.g., AI-film communities), >= 1 paid order within two weeks -> continue; 0 paid after two rounds -> offer/messaging wrong — one messaging rework, then park the consumer form.
6. **Legal:** Wan 2.2 / lightx2v LoRA / Fun-Control license check for sold outputs must pass before the first paid order. Failing = no paid orders until resolved (hard gate).

**Explicitly NOT kill criteria:** slow social growth, zero likes, "nobody watched the video" — at concierge scale these are noise until Sellable-1 exists.

---

## Owner decision queue (things this doc intentionally does NOT decide)

1. Watch-gate the two 2-min SBS files this week (crumb trade-off + publishability)?
2. Green-light the `/coherence` landing edit (generator offer block) after deciding (1)?
3. v0 price + refund text (proposed in §d.3) and Stripe re-enable?
4. Product name ("Long Take Studio" is a working title) + whether to open a dedicated social account.
5. Blind-comparison budget/account (Kling or similar) for wedge gate #2 — yes/no/defer.
6. Crumb default on deliveries: alpha 0.5 with disclosure vs 0.25 "more snap" (both measured; raw stays default until decided).

## Receipts index

Local (all paths relative to `~/.hermes/workspaces/brainsnn/` unless noted):
- Generator audit: `generator_audit/GENERATOR_AUDIT_v0.1.md` + addenda; `generator_audit/arm_results/crumb_battery/CRUMB_KILL_BATTERY.md`; `generator_audit/judge_moves/REF_CHUNK.md`, `R2A_VERIFY.md`.
- P4 2-min: `3d_first_pilot/p4/qa120/p4_summary.json` (drift slope 3.1e-5, seams), `p4/` video assets, `p4/wait_pull.log`.
- P3 60 s: `3d_first_pilot/NOTES.md` (P0–P3), `3d_first_pilot/p3/`.
- Method ladder: `method_ladder/METHOD_LADDER_RUN1.md`; engine synthesis: `ENGINE_SYNTHESIS_v2.md`; repo mirrors in `~/crumb-format/experiments/wavefield_video/` (PRODUCT_SPEC_brainsnn.md, THE_POSITION.md, METHODS_SPACE_2026.md, IMPL_NOTES_M3/M4.md).
- Box/compute: `remote-gpu-runtime-operations` skill `references/compute-economics.md`; box probe 2026-09-27 ~11:00 EDT (gpuq pending/done lists; P4-RESULT walls 170 s; GPU idle between jobs, 21 GB resident).
- Site: `~/the-brain` branch `feat/coherence-landing` (v4 landing; 707 tests green; assets `brainsnn-r3f-app/public/videos/coherence-demo*.mp4`, 516x256x21.3 s placeholders). Live `brainsnn.com` fetched 2026-09-27: "Sapient Playground" surface; `/coherence` -> 404 (landing not deployed).

External (fetched 2026-09-27 unless dated):
- Higgsfield pricing: techsifted.com/roundups/higgsfield-ai-pricing-2026/ (Sep 5, 2026); krea.ai/blog/higgsfield-pricing-explained-2026; layer3labs.io/guides/higgsfield-ai-pricing; creatify.ai/blog/higgsfield-pricing-(2026).
- Higgsfield scale: techcrunch.com/2026/01/15/ai-video-startup-higgsfield-founded-by-ex-snap-exec-lands-1-3b-valuation/; techstartups.com (Jan 15, 2026); pomegra.io (Aug 26, 2026); startupwired.com (Aug 17, 2026); thecodew.com (Aug 2026); getlatka.com/companies/higgsfield.ai (Sep 15, 2026).
- Higgsfield controversies: forbes.com investigation (Feb 11, 2026; reprint nationalcybersecurity.com, Feb 18, 2026).
- Higgsfield live surface: higgsfield.ai (homepage + pricing fetch).
- Length limits: invideo.io/blog/ai-video-length-limits/ (updated Aug 7, 2026).

*End of v0.1. Committed with the campaign docs; iterations tracked as v0.2+. This document itself: nothing public without the owner watch-gate.*
