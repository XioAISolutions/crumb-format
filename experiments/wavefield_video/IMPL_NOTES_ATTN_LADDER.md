# IMPL_NOTES_ATTN_LADDER.md — card t_d77fd4cd ("Use attention actively")

Status: **QUEUED** (2026-09-27) — 75 jobs deployed to the box queue behind the
r15 band (FIFO position verified). No results yet; this doc is the
pre-registered design + rubric, written BEFORE any data lands.

## 1. What this is

The attention arm (`--kind attn`) has been a stub: one 2k-step run at rung D3
(copy_ratio 0.002, i.e. frozen) and nothing at the current (unfreeze) recipe.
This card makes it compete honestly with the wave spine:

- **`run_attn_ladder.sh`** — attention vs wave (`--kind wave`, dispersion
  kernel) at every rung of the difficulty ladder from `run_difficulty.sh`, at
  the unfreeze recipe (`--const-lr` + 8000 steps; budget and schedule scale
  together — see `references/freeze-diagnosis-suite.md`). Focus rung D6
  (1024-frame eval horizon).
- **`run_attn_hybrid_retest.sh`** — matched-budget retest of the H8 hybrid
  family (attention+wave fusion, generic-SSM version) at the same recipe.
- **`attn_ladder_report.py`** — byte-column table + the pre-registered D5
  guard (exit 0 TRIGGER / 1 SKIP / 2 WAIT).
- **`test_attn_ladder_runner.py`** — 10 contract tests on fake runners (CPU).
- **`run_smoke_attn_ladder.sh`** — portable smoke (bash -n + contract tests +
  report behavior); the box job 500 adds a micro GPU train key-check.

## 2. Base condition decision (recorded before results)

`run_difficulty.sh`'s base used the 1.15 speed default with D1/D3 overriding to
2.30. Every 8k-era run this suite is compared against ran at **speed 2.30**:
runs_deep B2 (0.728), freeze-suite S2 (g16), hybrid8 H8wav/H8ssm. The suite
base therefore adopts `--speed 2.30` (rung distinctions unchanged; D1/D3 speed
overrides kept as no-op lineage). Consequence: the D3 rung is era-matched to
the old attn D3 pin (both speed 2.30) — the one old-vs-new apple-to-apple.

## 3. The ladder (rungs x arms{attn, wave}, seed 0)

Rung distinctions identical to `run_difficulty.sh`; all at `--grid 32
--dim 192 --layers 6 --heads 8 --frames 17 --const-lr --steps 8000`, speed
2.30, collisions on, gated local fusion on. Order: D6 first (attn then wave),
then D1..D4; slice-resumable queue units.

| rung | distinction | attn window B | wave persistent B | status |
|------|-------------|---------------|-------------------|--------|
| D6   | `--eval-rollout 1024` (long horizon) | 13,369,344 | 113,246,208 | queued |
| D1   | `--speed 2.30` (base speed) | 13,369,344 | 113,246,208 | queued |
| D2   | `--n-balls 8` | 13,369,344 | 113,246,208 | queued |
| D3   | `--n-balls 12 --speed 2.30` | 13,369,344 | 113,246,208 | queued |
| D4   | `--radius 0.8` | 13,369,344 | 113,246,208 | queued |
| D5   | `--grid 64 --n-balls 8` | 53,477,376 | 452,984,832 | GUARDED (see §6) |

Attention totals = window only (attention keeps no persistent state); wave total =
persistent only (no window). Bytes: attn g32 = 17*1024*192*4 = 13,369,344 B
(the same convention as H8's `window_bytes`); wave g32 measured
113,246,208 B (B2/H8 JSONs); g64 both scale x4 (grid²).

## 4. Hybrid retest (matched budget, same protocol as H8)

5 arms at g32 / 8k / const-lr / speed 2.30, run_major order:

| arm | kind | fuse | persistent B | window B | total B | H8 prior |
|-----|------|------|--------------|----------|---------|----------|
| local_wave | wave | local_wave | 113,246,208 | 13,369,344 | 126,615,552 | cr 0.428 / 0.947 |
| local_ssm  | ssm  | local_ssm  | 150,994,944 | 13,369,344 | 164,364,288 | cr 0.718 / 0.566 |
| wave_only  | wave | none       | 113,246,208 | 0 | 113,246,208 | — |
| attn_only  | attn | none       | 0 | 13,369,344 | 13,369,344 | — |
| ssm_only   | ssm  | none       | 150,994,944 | 0 | 150,994,944 | — |

H8 prior column = the equivalent H8 run (same recipe/family, H8-era scripts);
H8ssm was the best hybrid (cr 0.718) while H8wav underperformed plain wave
(0.428 vs 0.728 for B2) — the retest exists to check both under the current
runner.

## 5. Existing data (context; measured, not new)

| run | recipe | cr | mse/ctl | note |
|-----|--------|----|---------|------|
| B2 wave g32 | 8k const-lr, speed 2.3 | 0.728 | 0.571 | the unfreeze anchor |
| S2 wave g16 | 8k const-lr | 0.915 | 0.157 | small-grid control |
| H8wav | 8k const-lr 2.3 | 0.428 | 0.947 | hybrid worse than spine |
| H8ssm | 8k const-lr 2.3 | 0.718 | 0.566 | best hybrid |

Superseded (starved budget, 2k OneCycle — everything frozen; kept for
lineage only): wave D1 0.021 / D2 0.003 / D3 0.08 / D4 0.012, attn D3 0.002,
all mse/ctl ~1.0; D5 never ran (suite was interrupted).

## 6. Pre-registered rubric + gates (no results seen yet)

**Competitive(rung)** := both arms have result JSONs AND
`cr_attn >= 0.5` AND `cr_attn >= 0.9 x cr_wave` AND
`mse/ctl_attn <= 1.15 x mse/ctl_wave`.

**D5 guard** (as coded in `attn_ladder_report.py --d5-guard`): TRIGGER if any
of the 5 g32 pairs (D6, D1..D4) is Competitive; SKIP (recorded once to
`runs_attn_ladder_d5/d5_status.txt`) if all 5 pairs complete and none is;
WAIT (exit 2) while pairs are incomplete. The D5 band (24 slice copies,
`gpuq_job_551..574`) runs the g64 pair ONLY on TRIGGER — attention at g64
costs ~16x its g32 compute (~8-26 h/arm), so it never runs on a hunch.

**Kill criterion**: if attn cr < 0.25 x wave cr at every completed g32 rung,
archive the attention lane (report as elim). No D5, no rescue rounds.

**Per-resource claim discipline**: report bytes (above) and ms/frame
(from logs) per arm; the attention claim is "competitive per byte", not raw
quality; never compare against a non-frozen baseline.

**D6 focus**: the 1024-frame rollout is the long-horizon statement — report
its tracker/semantic metrics like any long-horizon run (copy_ratio first,
then mse/ctl; motion verdicts need the tracker metric, per the campaign's
measurement rules).

## 7. Queue + deployment receipts (2026-09-27)

- Box: `/workspace/slava/exp/wavefield_video` (code) and
  `/workspace/slava/gpu_queue/pending/` (jobs). Pending went 270 -> 345 with
  these 75; FIFO position: strictly after `gpuq_job_461_r15_slice.sh` (last
  r15 job), verified by sorted listing.
- Bands: `500` smoke (portable + micro GPU key-check) | `501-530` ladder
  slices | `531-550` hybrid slices | `551-574` D5-conditional slices.
- Slice contract: `RESUME=1 SLICE_S=4200 STEPS=8000 CONST_LR=1`; each copy
  skips finished arms, resumes from `ckpt_*.pt`, stops at the slice boundary
  ("SLICED d/t" status); repeat copies converge the suite. Conductor kills
  whole jobs at 5400 s — the per-arm 4200 s cap lands before it.
- md5 both ends (Mac == box), 2026-09-27:
  - run_attn_ladder.sh `6b07b5530449d76d8b3c296b0a02a660`
  - run_attn_hybrid_retest.sh `fb0dbc7dc0dfe276d06d5a2284e372a6`
  - attn_ladder_report.py `f358e5eb9b278813160b9b6fd4a506bc`
  - test_attn_ladder_runner.py `94d6a049ddd3935815aaaeecad98849f`
  - run_smoke_attn_ladder.sh `2a292812ee64ad9291c9f21b2318e57a`
  - train_compare.py (edited) `8057f6ffc62d2fc13ae9e6488c9419ca`
- Box-side portable verification PASSED (bash -n, 10/10 contract tests,
  CPU-forced micro train with recipe keys, guard WAIT rc=2); `train_compare.py`
  backup at `train_compare.py.bak_t_d77fd4cd` on the box.
- `train_compare.py` edit: result JSON now records `const_lr`, `seed`, `fp32`,
  `no_decay_norm_head`, `telemetry_every` (previously unrecorded — needed to
  distinguish recipes in the table).

## 8. Results: where they land / how to read

- Files: `runs_attn_ladder/result_{attn,wave}_{rung}.json` (+ `ckpt_*`,
  `log_attn_ladder_*.txt`, `attn_ladder_{progress,status}.txt`);
  D5 pair in `runs_attn_ladder_d5/`; hybrid in `runs_hybrid_retest/`.
- Table: `PY attn_ladder_report.py` (any host with the repo; `--root` to point
  at the box tree). Guard: `--d5-guard` (0/1/2 = TRIGGER/SKIP/WAIT).
- Watcher: Hermes cron `attn-ladder-watch` reports only when new result JSONs
  appear (silent otherwise).
- To queue more copies (e.g. after a partial band): copy any
  `gpuq_job_5xx_*.sh` into `gpu_queue/pending/` under a NEW name that still
  sorts after 574 (rename to `gpuq_job_6xx_...`).

## 9. Limits / honest caveats

- Attention's g32 compute is untested at this recipe — wall estimates (8-26 h
  for g64; ~1.5 h for g32) are scaling estimates, not measurements; watch the
  first slice's log.
- D2/D4 rungs now run at speed 2.30 (harder than the old 1.15 versions);
  rung distinctions preserved, absolute numbers intentionally NOT comparable
  to the superseded 2k ladder rows.
- The hybrid retest compares against H8's *prior* numbers (different scripts,
  same recipe); its own wave_only/ssm_only controls make the within-suite
  comparison primary.
