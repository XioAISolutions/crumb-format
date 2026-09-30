#!/usr/bin/env python3
"""Add --relative-rope on/off to longlive_long.py + RELATIVE_ROPE passthrough to run_longlive.sh.

Fail-closed: exact anchors, count==1 each, backups .before_ropeflag, py_compile + bash -n,
diff printed. Idempotent: refuses if half-applied.
"""
import subprocess, sys
from pathlib import Path

LL = Path("/workspace/slava/exp/pr63/longlive_long.py")
SH = Path("/workspace/slava/exp/pr63/run_longlive.sh")

def fail(m):
    print("FAIL:", m); sys.exit(1)

src = LL.read_text()
if "--relative-rope" in src:
    if 'relative_rope=(a.relative_rope == "on")' in src and "bool(relative_rope)" in src \
       and 'def build_overlay(base, *, latent_frames, prompts, ckpt, out_dir, window=32, sink=8,' in src:
        print("ALREADY APPLIED (idempotent OK)"); sys.exit(0)
    fail("half-applied state detected in longlive_long.py — inspect manually")

rep_ll = [
    ('seed=0, fp8=True, compile=False):',
     'seed=0, fp8=True, compile=False, relative_rope=True):'),
    ('    cfg["use_relative_rope"] = True',
     '    cfg["use_relative_rope"] = bool(relative_rope)'),
    ('fp8=a.precision == "fp8", compile=a.compile)',
     'fp8=a.precision == "fp8", compile=a.compile,\n                        relative_rope=(a.relative_rope == "on"))'),
    ('            s.add_argument("--force", action="store_true")',
     '            s.add_argument("--force", action="store_true")\n'
     '            s.add_argument("--relative-rope", choices=["on", "off"], default="on",\n'
     '                           help="store raw K in cache / window-relative RoPE (default on)")'),
]
for old, new in rep_ll:
    n = src.count(old)
    if n != 1:
        fail(f"anchor count {n} != 1 in longlive_long.py for: {old[:70]!r}")
    src = src.replace(old, new)
LL.with_name(LL.name + ".before_ropeflag").write_text(LL.read_text())
LL.write_text(src)
subprocess.run(["python3", "-m", "py_compile", str(LL)], check=True)

sh = SH.read_text()
if '--relative-rope "$RELATIVE_ROPE"' in sh:
    print("SH ALREADY APPLIED"); sys.exit(0)
rep_sh = [
    ("PRECISION=${PRECISION:-fp8}; WINDOW=${WINDOW:-32}; SINK=${SINK:-8}; SEED=${SEED:-0}",
     "PRECISION=${PRECISION:-fp8}; WINDOW=${WINDOW:-32}; SINK=${SINK:-8}; SEED=${SEED:-0}\nRELATIVE_ROPE=${RELATIVE_ROPE:-on}"),
    ('--decode-device "$DECODE_DEVICE" ${base_args[@]+"${base_args[@]}"}',
     '--decode-device "$DECODE_DEVICE" --relative-rope "$RELATIVE_ROPE" ${base_args[@]+"${base_args[@]}"}'),
]
for old, new in rep_sh:
    n = sh.count(old)
    if n != 1:
        fail(f"anchor count {n} != 1 in run_longlive.sh for: {old[:70]!r}")
    sh = sh.replace(old, new)
SH.with_name(SH.name + ".before_ropeflag").write_text(SH.read_text())
SH.write_text(sh)
subprocess.run(["bash", "-n", str(SH)], check=True)

for f in (LL, SH):
    d = subprocess.run(["diff", "-u", str(f) + ".before_ropeflag", str(f)], capture_output=True, text=True)
    print(f"=== diff {f.name} ===")
    print(d.stdout)
print("APPLIED OK")
