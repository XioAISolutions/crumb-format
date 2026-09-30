# Receipt — journey probes (c01ae) + Studio v0.1 — 2026-09-30

## Journey / no-follow probes

LongLive-2.0-5B, fp8, W12 sink8, seed 0, 832×576@24fps-class 10–30 s runs, scored as median px/frame flow + net displacement + rmse (see tooling in `jobs/`).

| run | flow px/f | net | rmse | verdict |
|---|---|---|---|---|
| c01ae_1 journey30s | 0.853 | −2.8 | 17.05 | one-street treadmill persists; NO zone variety; no walker visible |
| c01ae_2 journey10s | 0.869 | −0.5 | 16.08 | same street class as 30s; clean look (yellow courtyard alley, walker in blue) |
| c01ae_3 nofollow10s | 1.142 | −2.1 | 15.93 | travels clean; a walker appears despite the prompt not asking for one |
| c01ae_4 bf16journey30s | — | — | — | OOM (GPU race with a concurrent slice job); requeued |

**Conclusions:**
1. "Journey" prompt wording (alley → plaza → café) does **not** produce scene transitions on this base — the model renders one consistent street regardless. Prompt is the lever for *travel yes/no* (fast dialect + seed 0/2), not for *environment variety*.
2. Remaining path to scene evolution = **segmentation / i2v chaining** — probe `c01ae_5_i2vchain` (continue from the LAST frame of the fast 30 s clip, new-scene prompt, seed 0). Flow ≥ ~1.0 would unlock stitched multi-scene long videos.

## Studio v0.1

End-to-end loop proven by `studio_260930101921_ac2` (5-min retry at the owner's request):

- submit (web) → box render rc=0 (2135 s) → auto-fetch (787 MB) → preview build (28.7 MB) → phone-readable URL via tunnel.
- Auth passcode gate + cookie session + `jobs.json` persistence added the same day.
- Code: `experiments/wavefield_video/studio/`.
