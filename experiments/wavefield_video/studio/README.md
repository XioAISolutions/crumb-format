# BrainsNN Studio (v0.1)

Local web console that turns a typed prompt + length into a rendered long video on the GPU box and streams the result back. Stdlib-only Python — no dependencies; orchestrates entirely via ssh/scp.

## What it does

- `POST /api/generate {prompt, minutes, seed}` → places a `gpuq` job on the box → polls status → scp's the mp4 back → builds an in-browser preview (960w h264 crf ~30, audio-free).
- Job states: queued → running → fetching → done/failed. Persisted in `jobs.json`; survives restarts.
- Passcode gate (`passcode.txt`) + cookie session for tunneled / remote access. Local 127.0.0.1 use needs no code only when no passcode file exists.

## Run

    python3 app.py     # binds 127.0.0.1:9321

`config.json` holds the box ssh host/port/user (intentionally kept out of the repo).

## Live proof (2026-09-30)

studio job `studio_260930101921_ac2`: 300 s / seed 0 / street-fast prompt → box rc=0 in 2135 s; app fetched 787 MB raw + built a 28.7 MB preview; served to a phone over a Pinggy tunnel. Full loop verified end-to-end.

## Files

- `app.py` — single-file server + job worker
- `index.html` — UI
- `passcode.txt` / `jobs.json` — runtime state only (not secret to the owner; do not commit live copies)
