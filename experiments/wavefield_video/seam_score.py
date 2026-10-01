#!/usr/bin/env python3
"""seam_score.py — detect periodic streaming-block-boundary discontinuities in generated video.

Usage:
  python3 seam_score.py VIDEO [VIDEO ...] [--fps 12] [--period 16] [--offset 15] [--topk 7] [--json OUT.json]

Method:
  - Samples frame-to-frame luma change (mean abs diff, YAVG) at `fps` samples/sec via ffmpeg
    (tblend=all_mode=difference + signalstats).
  - LongLive streaming joins sit every 32 pixel frames @24fps. At 12 samples/s that is every
    16 samples (0-based: the join shows up on sample numbers 15, 31, 47, ... 1-based).
  - boundary_score = median(change at boundary samples) / median(change elsewhere).
  - topk_hits = how many of the clip's top-k strongest changes sit on boundary samples.
  - verdict: SEAMS if score >= 1.35 and topk_hits >= 5; WEAK if score >= 1.15; else CLEAN.

Output: one JSON object per video + a compact table. Exit 0.
"""
import subprocess, sys, json, statistics, argparse

def yavg_series(path, fps):
    cmd = ["ffmpeg", "-v", "info", "-i", path, "-vf",
           "fps=%d,scale=256:144,tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG" % fps,
           "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    vals = []
    for line in p.stderr.splitlines():
        i = line.find("YAVG=")
        if i >= 0:
            tok = line[i + 5:].split()
            if tok:
                try:
                    vals.append(float(tok[0]))
                except ValueError:
                    pass
    # first sample has no predecessor frame in the diff stream; drop if ~0
    if vals and vals[0] == 0.0:
        vals = vals[1:]
    return vals

def score(path, fps=12, period=16, offset=15, topk=7):
    vals = yavg_series(path, fps)
    n = len(vals)
    if n < period + 3:
        return {"video": path, "samples": n, "error": "too short"}
    boundary_idx = [i for i in range(n) if (i + 1) % period == offset % period]
    # offset semantics: 1-based sample numbers where joins land == offset (mod period), i.e. 15,31,...
    rest_idx = [i for i in range(n) if i not in set(boundary_idx)]
    b = [vals[i] for i in boundary_idx]
    r = [vals[i] for i in rest_idx]
    med_b = statistics.median(b)
    med_r = statistics.median(r) if r else 0.0
    score_v = (med_b / med_r) if med_r > 0 else 0.0
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)[:topk]
    topk_hits = sum(1 for i in order if i in set(boundary_idx))
    if score_v >= 1.35 and topk_hits >= 5:
        verdict = "SEAMS"
    elif score_v >= 1.15:
        verdict = "WEAK"
    else:
        verdict = "CLEAN"
    return {
        "video": path, "samples": n, "expected_joins": len(boundary_idx),
        "boundary_median": round(med_b, 2), "rest_median": round(med_r, 2),
        "boundary_score": round(score_v, 3), "top%dk_on_boundary" % topk: "%d/%d" % (topk_hits, topk),
        "topk_positions_1based": [i + 1 for i in order], "verdict": verdict,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--period", type=int, default=16)
    ap.add_argument("--offset", type=int, default=15)
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = []
    for v in a.videos:
        try:
            res = score(v, a.fps, a.period, a.offset, a.topk)
        except Exception as e:
            res = {"video": v, "error": str(e)}
        out.append(res)
        print(json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
