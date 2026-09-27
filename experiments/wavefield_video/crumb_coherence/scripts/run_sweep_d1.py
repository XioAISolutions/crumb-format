"""D1 residual-anchoring sweep driver (M4).

Runs a config grid (window x strength x cutoff) x scenes through the existing
acceptance scripts (run_trap.py / run_m0.py with the D1 flags), appends every
run to a CSV, and prints a per-round summary with the pre-registered co-pass
check: translate distortion <5% AND M0 hotspot removal >=60%.

Usage (from the wavefield_video dir):
  python crumb_coherence/scripts/run_sweep_d1.py --round 1 \
      --windows 2,4,8,16,32 --strengths 0.1,0.25,0.5,1.0 \
      --cutoffs 0.0,0.25,1.0,4.0 --tag r1

Scenes: "translate:96,translate:192,hotspot:48" (hotspot = M0 regressor).
Artifacts: crumb_coherence/out/sweep_d1/sweep_d1_rounds.csv (appended),
           crumb_coherence/out/sweep_d1/manifest_round{R}.json,
           crumb_coherence/out/sweep_d1/summary_round{R}.md
Per-run JSONs go to --devout (default: out/sweep_d1/runs/).
Resume-safe: a run whose JSON exists and parses is skipped (its row is read
from the JSON again, so a re-run does not duplicate evidence).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # wavefield_video/
PY = sys.executable

CSV_FIELDS = [
    "round", "tag", "ts", "git_rev", "scene", "frames", "window", "strength",
    "cutoff",
    # trap fields
    "retention", "distortion", "distortion_mean", "hf_ssim", "var_ratio",
    "all_pass",
    # hotspot fields
    "drift_pct_pos", "de_corr", "hf_ssim_clean", "hf_ssim_chg", "motion_pct",
    "verdict", "flags",
    # judgments
    "trap_pass", "hotspot_pass", "copass",
]


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _git_rev(root):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=root, capture_output=True, text=True)
        return out.stdout.strip() or "?"
    except Exception:                                        # noqa: BLE001
        return "?"


def run_key(round_, tag, scene, frames, w, s, c):
    return f"r{round_}_{tag}_{scene}{frames}_w{w}_s{s}_c{c}"


def run_one(devout, round_, tag, scene, frames, w, s, c):
    key = run_key(round_, tag, scene, frames, w, s, c)
    jpath = os.path.join(devout, key + ".json")
    if os.path.exists(jpath):
        try:
            with open(jpath) as fh:
                return json.load(fh), True
        except Exception:                                    # noqa: BLE001
            pass
    if scene == "hotspot":
        cmd = [PY, "crumb_coherence/scripts/run_m0.py",
               "--scenario", "hotspot", "--frames", str(frames),
               "--no-video", "--hotspot-mc-residual",
               "--hotspot-only-residual",
               "--hotspot-res-window", str(w),
               "--hotspot-res-strength", str(s),
               "--hotspot-res-cutoff", str(c),
               "--json", jpath]
    else:
        cmd = [PY, "crumb_coherence/scripts/run_trap.py",
               "--scene", scene, "--frames", str(frames),
               "--residual", "--res-window", str(w),
               "--res-strength", str(s), "--res-cutoff", str(c),
               "--json", jpath]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  !! RUN FAILED ({key}): rc={res.returncode}\n{res.stdout[-500:]}\n"
              f"{res.stderr[-800:]}")
        return None, False
    with open(jpath) as fh:
        rec = json.load(fh)
    rec["_wall_s"] = round(time.time() - t0, 1)
    return rec, False


def row_from(rec, round_, tag, scene, frames, w, s, c, git_rev):
    row = {k: "" for k in CSV_FIELDS}
    row.update(round=round_, tag=tag, ts=time.strftime("%Y-%m-%d %H:%M:%S"),
               git_rev=git_rev, scene=scene, frames=frames,
               window=w, strength=s, cutoff=c)
    if scene == "hotspot":
        row["drift_pct_pos"] = round(rec["rows"][0]["drift_pct_pos"], 3)
        row["de_corr"] = round(rec["rows"][0]["de_corr"], 4)
        row["hf_ssim_clean"] = round(rec["rows"][0]["hf_ssim_clean"], 5)
        row["hf_ssim_chg"] = round(rec["rows"][0]["hf_ssim_chg"], 5)
        row["motion_pct"] = (round(rec["rows"][0]["motion_pct"], 2)
                             if rec["rows"][0]["motion_pct"] is not None else "")
        row["verdict"] = rec["rows"][0]["verdict"]
        row["flags"] = rec["rows"][0]["flags"]
        hp = (rec["rows"][0]["ok_drift"] and rec["rows"][0]["ok_ssim"]
              and rec["rows"][0]["ok_motion"])
        row["hotspot_pass"] = bool(hp)
    else:
        row["retention"] = round(rec["retention"], 3)
        row["distortion"] = round(rec["distortion"], 4)
        row["distortion_mean"] = round(rec["distortion_mean"], 4)
        row["hf_ssim"] = round(rec["hf_ssim"], 5)
        row["var_ratio"] = round(rec["var_ratio"], 4)
        row["all_pass"] = bool(rec["all_pass"])
        row["trap_pass"] = bool(rec["all_pass"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--windows", type=str, default="2,4,8,16,32")
    ap.add_argument("--strengths", type=str, default="0.1,0.25,0.5,1.0")
    ap.add_argument("--cutoffs", type=str, default="0.0,0.25,1.0,4.0")
    ap.add_argument("--scenes", type=str,
                    default="translate:96,translate:192,hotspot:48")
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(_ROOT, "crumb_coherence", "out",
                                         "sweep_d1"))
    args = ap.parse_args()
    tag = args.tag or f"r{args.round}"

    windows = [float(x) for x in args.windows.split(",") if x.strip()]
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    cutoffs = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    scenes = []
    for spec in args.scenes.split(","):
        scene, fr = spec.split(":")
        scenes.append((scene.strip(), int(fr)))

    devout = os.path.join(args.outdir, "runs")
    os.makedirs(devout, exist_ok=True)
    csv_path = os.path.join(args.outdir, "sweep_d1_rounds.csv")
    git_rev = _git_rev(_ROOT)

    todo = [(sc, fr, w, s, c) for w in windows for s in strengths
            for c in cutoffs for (sc, fr) in scenes]
    print(f"D1 sweep round {args.round} ({tag}): {len(todo)} runs "
          f"[{len(windows)}x{len(strengths)}x{len(cutoffs)} configs x "
          f"{len(scenes)} scenes]  git={git_rev}  py={PY}")
    rows = []
    n_new = 0
    t0 = time.time()
    for i, (sc, fr, w, s, c) in enumerate(todo):
        rec, cached = run_one(devout, args.round, tag, sc, fr, w, s, c)
        if rec is None:
            continue
        n_new += 0 if cached else 1
        rows.append(row_from(rec, args.round, tag, sc, fr, w, s, c, git_rev))
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s")

    # merge this round's rows into the CSV (keep prior rounds; fill copass)
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            existing = list(csv.DictReader(fh))
    def _key(r):
        return (str(r["round"]), r["tag"], r["scene"], str(r["frames"]),
                str(r["window"]), str(r["strength"]), str(r["cutoff"]))
    merged = {_key(r): r for r in existing}
    for r in rows:
        merged[_key(r)] = r
    merged_rows = [merged[k] for k in merged]
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wtr.writeheader()
        for r in merged_rows:
            r2 = {k: r.get(k, "") for k in CSV_FIELDS}
            wtr.writerow(r2)

    # per-config co-pass: translate96 + translate192 trap_pass, hotspot_pass
    by_cfg = {}
    for r in rows:
        key = (r["window"], r["strength"], r["cutoff"])
        by_cfg.setdefault(key, {})[r["scene"]] = r
    winners = []
    for key, cfg in sorted(by_cfg.items()):
        t96 = cfg.get("translate")
        t192 = cfg.get("translate")
        # scenes stored by name: translate appears twice (96/192) -> handle by frames
        t96 = next((r for r in rows if (r["window"], r["strength"], r["cutoff"]) == key
                    and r["scene"] == "translate" and r["frames"] == 96), None)
        t192 = next((r for r in rows if (r["window"], r["strength"], r["cutoff"]) == key
                     and r["scene"] == "translate" and r["frames"] == 192), None)
        hp = cfg.get("hotspot")
        trap_ok = bool(t96 and t192 and t96.get("trap_pass") and t192.get("trap_pass"))
        hp_ok = bool(hp and hp.get("hotspot_pass"))
        copass = trap_ok and hp_ok
        for r in (t96, t192, hp):
            if r is not None:
                r["copass"] = copass
        if copass:
            winners.append(key)

    # rewrite csv with copass filled (small file: rewrite atomically).
    # NOTE: must write the MERGED row set — writing only this round's rows
    # silently dropped every prior round on multi-round appends (fixed
    # 2026-09-27, t_3e1d4c9f).
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wtr.writeheader()
        for r in merged_rows:
            wtr.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    # markdown summary
    lines = [f"# D1 sweep round {args.round} ({tag})", "",
             f"- runs this round: {len(rows)} (new: {n_new}), wall {time.time()-t0:.0f}s",
             f"- git={git_rev} py={PY}",
             "- co-pass = translate 96 AND 192 trap-PASS and hotspot d_pos >= 60", "",
             "| window | strength | cutoff | tr96 ret% | tr96 dist% | tr96 pass | "
             "tr192 dist% | tr192 pass | hotspot drift% | hotspot pass | CO-PASS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(by_cfg):
        w, s, c = key
        t96 = next((r for r in rows if (r["window"], r["strength"], r["cutoff"]) == key
                    and r["scene"] == "translate" and r["frames"] == 96), None)
        t192 = next((r for r in rows if (r["window"], r["strength"], r["cutoff"]) == key
                     and r["scene"] == "translate" and r["frames"] == 192), None)
        hp = next((r for r in rows if (r["window"], r["strength"], r["cutoff"]) == key
                   and r["scene"] == "hotspot"), None)
        copass = any((r is not None and r.get("copass")) for r in (t96, t192, hp))
        lines.append(
            f"| {w} | {s} | {c} | "
            f"{t96['retention'] if t96 else '-'} | {t96['distortion'] if t96 else '-'} | "
            f"{'PASS' if t96 and t96.get('trap_pass') else 'fail'} | "
            f"{t192['distortion'] if t192 else '-'} | "
            f"{'PASS' if t192 and t192.get('trap_pass') else 'fail'} | "
            f"{hp['drift_pct_pos'] if hp else '-'} | "
            f"{'PASS' if hp and hp.get('hotspot_pass') else 'fail'} | "
            f"{'YES' if copass else 'no'} |")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(args.outdir, f"summary_round{args.round}.md"), "w") as fh:
        fh.write(summary)

    manifest = dict(round=args.round, tag=tag, git_rev=git_rev, tsp=time.time(),
                    n_runs=len(rows), n_new=n_new,
                    md5=dict(core=_md5(os.path.join(_ROOT, "crumb_coherence", "core.py")),
                             run_trap=_md5(os.path.join(_ROOT, "crumb_coherence", "scripts",
                                                        "run_trap.py")),
                             run_m0=_md5(os.path.join(_ROOT, "crumb_coherence", "scripts",
                                                      "run_m0.py"))),
                    copass_configs=[list(k) for k in winners])
    with open(os.path.join(args.outdir, f"manifest_round{args.round}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(summary)
    print(f"CO-PASS configs: {winners if winners else 'NONE'}")
    print(f"csv: {csv_path}")
    print(f"manifest: {os.path.join(args.outdir, f'manifest_round{args.round}.json')}")


if __name__ == "__main__":
    main()
