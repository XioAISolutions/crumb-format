#!/usr/bin/env python3
"""Compact results table + pre-registered D5 guard for the attention-lane suites
(card t_d77fd4cd). Stdlib only; safe to run on the box with plain python3 and
from the Mac watcher over ssh.

  python3 attn_ladder_report.py                 # tables for both suites
  python3 attn_ladder_report.py --phase d6     # the D6 centerpiece comparison
  python3 attn_ladder_report.py --phase ladder
  python3 attn_ladder_report.py --phase hybrid
  python3 attn_ladder_report.py --d5-guard     # exit 0 TRIGGER / 1 SKIP / 2 WAIT

D5 guard (pre-registered in IMPL_NOTES_ATTN_LADDER.md): the g64 rung D5 pair
runs only if attention is demonstrably competitive at >=1 g32 rung, i.e. for a
rung with BOTH result files present:
    attn.copy_ratio >= 0.5
    attn.copy_ratio >= 0.9 * wave.copy_ratio
    attn.eval_mse_over_copylast <= 1.15 * wave.eval_mse_over_copylast
Any passing rung -> TRIGGER. All five g32 rungs present and none passing ->
SKIP (record the decision; do not run D5). Otherwise -> WAIT.
"""

import argparse
import json
import sys
from pathlib import Path

LADDER_RUNGS = ["D6", "D1", "D2", "D3", "D4", "D5"]
G32_RUNGS = ["D1", "D2", "D3", "D4", "D6"]
HYBRID_LABELS = [
    ("wave", "local_wave", "E hybrid local+wave"),
    ("ssm", "local_ssm", "D hybrid local+ssm"),
    ("wave", "wave_only", "C wave-only (spine)"),
    ("attn", "attn_only", "A attn-only (local)"),
    ("ssm", "ssm_only", "B ssm-only"),
]
FIELDS = ["copy_ratio", "eval_mse_over_copylast", "eval_mse"]


def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mb(v):
    if v is None:
        return "n/a"
    return f"{v / 1e6:.1f}MB"


def row(name, res):
    if res is None:
        return f"  {name:<26} --"
    psb = res.get("persistent_state_bytes")
    win = res.get("window_bytes")
    tot = res.get("state_bytes_total")
    return (f"  {name:<26} steps={res.get('steps')!s:<5} "
            f"cr={fmt(res.get('copy_ratio'), 3):<6} "
            f"mse/ctl={fmt(res.get('eval_mse_over_copylast'), 3):<6} "
            f"div_h={res.get('divergence_horizon')!s:<5} "
            f"id_surv={fmt(res.get('identity_survival'), 3):<6} "
            f"state p/w/t={mb(psb)}/{mb(win)}/{mb(tot)}")


def status_line(root, name):
    try:
        s = (root / name).read_text().strip().splitlines()
        return s[-1] if s else "(empty)"
    except Exception:
        return "(missing)"


def load_rung(root, kind, rung):
    """D5 (the conditional g64 pair) may live in its own OUT dir; look there too."""
    for sub in ("runs_attn_ladder", "runs_attn_ladder_d5"):
        res = load(root / sub / f"result_{kind}_{rung}.json")
        if res is not None:
            return res
    return None


def ladder_report(root, rungs):
    d = root / "runs_attn_ladder"
    print("== attention ladder suite (runs_attn_ladder) ==")
    print(f"   status: {status_line(d, 'attn_ladder_status.txt')}")
    print(f"   d5 dir: {status_line(root / 'runs_attn_ladder_d5', 'attn_ladder_status.txt')}")
    for rung in rungs:
        a = load_rung(root, "attn", rung)
        w = load_rung(root, "wave", rung)
        if a is None and w is None:
            print(f"  rung {rung}: (pending)")
            continue
        print(f"  rung {rung}:")
        print(row(f"attn {rung}", a))
        print(row(f"wave {rung}", w))
        if a and w and a.get("copy_ratio") is not None and w.get("copy_ratio"):
            ratio = a["copy_ratio"] / w["copy_ratio"]
            print(f"    -> copy_ratio attn/wave = {ratio:.3f}")
    print(f"   d5 guard: {guard_verdict(d)}")


def hybrid_report(root):
    d = root / "runs_hybrid_retest"
    print("== hybrid retest suite (runs_hybrid_retest) ==")
    print(f"   status: {status_line(d, 'hybrid_retest_status.txt')}")
    for kind, tag, label in HYBRID_LABELS:
        res = load(d / f"result_{kind}_{tag}.json")
        print(row(label, res))


def d6_focus(root):
    d = root / "runs_attn_ladder"
    a = load(d / "result_attn_D6.json")
    w = load(d / "result_wave_D6.json")
    print("== D6 centerpiece (1024-frame horizon, g32) ==")
    print(row("attn D6", a))
    print(row("wave D6", w))
    if a is None or w is None:
        print("   (pending: needs result_attn_D6.json + result_wave_D6.json)")
        return
    ca, cw = a.get("copy_ratio"), w.get("copy_ratio")
    if ca is not None and cw:
        print(f"   copy_ratio attn/wave = {ca / cw:.3f}  "
              f"(>1 = attention beats the spine on motion)")
    print(f"   attn state: persistent={mb(a.get('persistent_state_bytes'))} "
          f"window={mb(a.get('window_bytes'))} | "
          f"wave state: persistent={mb(w.get('persistent_state_bytes'))}")


def guard_verdict(d):
    passes = []
    present = 0
    for rung in G32_RUNGS:
        a = load(d / f"result_attn_{rung}.json")
        w = load(d / f"result_wave_{rung}.json")
        if a is None or w is None:
            continue
        present += 1
        ca, cw = a.get("copy_ratio"), w.get("copy_ratio")
        ma, mw = a.get("eval_mse_over_copylast"), w.get("eval_mse_over_copylast")
        if None in (ca, cw, ma, mw) or not cw:
            continue
        ok = (ca >= 0.5 and ca >= 0.9 * cw and ma <= 1.15 * mw)
        if ok:
            passes.append(rung)
    if passes:
        return f"TRIGGER (competitive at {', '.join(passes)})"
    if present >= len(G32_RUNGS):
        return "SKIP (no competitive g32 rung; D5 stays deferred)"
    return f"WAIT ({present}/{len(G32_RUNGS)} g32 pairs complete)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent),
                    help="experiment dir containing runs_attn_ladder/ and runs_hybrid_retest/")
    ap.add_argument("--phase", choices=["all", "d6", "ladder", "hybrid"], default="all")
    ap.add_argument("--d5-guard", action="store_true",
                    help="print the guard verdict and exit 0 TRIGGER / 1 SKIP / 2 WAIT")
    a = ap.parse_args()
    root = Path(a.root)
    if a.d5_guard:
        v = guard_verdict(root / "runs_attn_ladder")
        print(v)
        if v.startswith("TRIGGER"):
            return 0
        if v.startswith("SKIP"):
            return 1
        return 2
    if not (root / "runs_attn_ladder").is_dir() and not (root / "runs_hybrid_retest").is_dir():
        print(f"no suites found under {root}")
        return 0
    if a.phase in ("all", "d6"):
        if a.phase == "all":
            d6_focus(root)
            print()
        else:
            d6_focus(root)
    if a.phase in ("all", "ladder"):
        ladder_report(root, LADDER_RUNGS)
        print()
    if a.phase in ("all", "hybrid"):
        hybrid_report(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
