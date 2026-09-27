# SITE-ASSET render round — preparation note (pre-M3)

By: kanban t_90265ba9 ("Post-gate: render the synchronized comparison asset for the hero player")
Written: 2026-09-26 ~20:30 EDT. Status: **NOT FIRED — gate unmet** (see §6).

Execution map so the round can fire the moment M3 lands green. Do not render the hero asset
before §1 passes — with the current engine the "processed" side is visibly worse than the
original (renders/eng1_e_curve_sidebyside.mp4), which is exactly what this gate exists to prevent.

## 1. Gate check — run first; do NOT render if any line fails

- [ ] `reviews/IMPL_NOTES_M3.md` exists, has gate numbers + the honest envelope of what is safe to claim
- [ ] `reports/ENG1_1080.md` (updated post-gate): `e_curve` AND `fib_spiral` — detail >= 0.97,
      motion ratio in [0.90, 1.10], jitter max <= 1.5x the clip's own p95; sanity drift removal >= 60%
- [ ] `run_trap.py` both horizons: retention >= 95%, trajectory distortion < 5%, HF-SSIM >= 0.98,
      low-band var ratio in [0.90, 1.10]; second legit-motion scene also retained >= 95%
- [ ] plugin version bumped (crumb-format commit) — this becomes the caption's "engine vX.Y" field
- [ ] `THE_POSITION.md` claim-boundary section updated (what we are allowed to say now)

Check commands:

    ls crumb-format/experiments/wavefield_video/IMPL_NOTES_M3.md
    grep -n "PASS\|FAIL\|motion\|jitter" crumb-format/experiments/wavefield_video/reports/ENG1_1080.md
    cd crumb-format && git log --oneline -5

## 2. Render spec

The card points at `reviews/claude_render_task.md` as the round's task file. **It does not exist**
(checked 20:23: nothing at that path, and no such file by name under the wavefield tree or
~/.hermes/context|handoffs|workspaces). Author is intended to
be the M3/orchestrator session. If it is still missing when M3 is green: STOP and ask — do not
guess the source clip or panel geometry. (Alternatively the owner may authorize authoring it from
the card text + §3/§4 here.)

## 3. What the round must produce (card text + live page)

- one synchronized Original|Processed comparison composite, 8-12s
- poster jpg
- mp4 + webm pair (current pair: h264 + vp9 — keep formats)
- burnt-in pane labels in the panels' top strip (current asset: "RAW rollout" / "PLUGIN magnitude";
  the page ALSO overlays Original/Processed chips at the bottom — v4 keeps them clear of the
  burnt-in captions). Labels must be consistent with the mode actually used in the render.
- honest caption line updated in `CoherenceLanding.jsx` (~line 144): source type, native res,
  duration, engine version — the four fields the critique requires

Geometry — decide explicitly in the spec:
- current site asset: 516x256, ~21.3s, two ~square halves (2.016:1 total); the page layout and the
  mobile half-switch assume ~square halves. A portrait pair (ENG-1 style, 1086x960 with 540x960
  panels) will NOT fit the current layout unchanged — either render square-ish halves for the
  site or re-layout the player (design lane).
- ENG-1 renderer for reference: `crumb_coherence/scripts/run_eng1_1080.py` `_compose_sxs()`
  (view_w=540, view_h=H*540/W, sep=6) burns "RAW f0123"/"CORRECTED f0123" engineering labels —
  the site asset should use customer-facing labels per the page promise ("same timestamps, same
  framing, nothing regenerated").

## 4. Source clip candidates (spec decides — do not guess)

1. **kk rescue pick** (from t_aa99ef3b: clips with natural slow motion + visible drift) — most
   demo-worthy if the gated pass visibly improves it; that card files its picks in the brainsnn
   workspace.
2. **ENG-1 clips** `data_real/e_curve.mp4`, `data_real/fib_spiral.mp4` (1080x1920, 30fps rendered
   math reels) — already plumbed through the pipeline, but thin high-contrast content; as of
   ENG-1 they FAIL with the frozen engine (motion 0.699/0.536; e_curve jitter 2.91 vs raw 1.69).
3. current synthetic fixture (256px) — last resort only; moving OFF it is the point of the critique.

Caption template (keep the current page format):

    Source: <source-type>, <WxH native>, <T>s, engine v<version> (<mode>). "Original" is the raw
    clip; "Processed" is after the correction pass. Targets unwanted brightness and color drift.
    Not a general repair for changing identities, geometry, or missing detail.

and update the "What this example proves" details with measured numbers **for the chosen clip**
(drift/flicker removed %, detail, motion retained) — only numbers the engine actually reported.

## 5. Swap map (the-brain, branch feat/coherence-landing)

Replace (keep filenames → no JSX source change needed):

    brainsnn-r3f-app/public/videos/coherence-demo-scrub.mp4
    brainsnn-r3f-app/public/videos/coherence-demo-scrub.webm
    brainsnn-r3f-app/public/videos/coherence-poster.jpg

(`coherence-demo.mp4` is the legacy non-scrub copy, referenced nowhere in src — replace/remove only
if the spec says so.)

JSX caption text: `brainsnn-r3f-app/src/app/CoherenceLanding.jsx` ~lines 144-156.

⚠ **Collision**: design round t_c326104e is editing this same file right now (v4 pass uncommitted
as of 20:25 on the same branch). Land the asset swap AFTER v4 commits, rebased onto it.

Rebuild + preview:

    cd the-brain/brainsnn-r3f-app && npm run build     # vite build + esbuild server; dist/ is gitignored
    npm run dev                                        # tsx server.ts; vite middleware on PORT 3000
    # preview: http://localhost:3000/coherence

Owner gate: **no merge** (watch-gate) — Slava reviews the actual video + page preview first;
deploys are repo-only (merge → Railway) and nothing goes public without his explicit approval.

## 6. Evidence of block (2026-09-26 ~20:25)

- Gate spec written 19:58 (`reviews/claude_m3_gate_task.md`); never run — no IMPL_NOTES_M3.md, no
  m3 log, latest crumb-format commit is 16:02 ("auto-sync artifacts").
- Engine state: M2 trap FAILED (legit-motion retention 54.7%/70.8%, distortion 64.1%/58.9%);
  ENG-1 FAILED both real clips (e_curve motion 0.699, jitter 2.91 vs raw 1.69, lowband var 0.275;
  fib_spiral motion 0.536, detail 0.957). Corrected side visibly dimmer on e_curve.
- Current site asset: "RAW rollout" | "PLUGIN magnitude", abstract 256px fixture; caption already
  discloses "Source: synthetic drift fixture, 256px, 21s, engine v0.4 (complex_mc)".
- kk rescue-pick card (t_aa99ef3b) also still gated on its 4090 batch; it may feed the source clip.
