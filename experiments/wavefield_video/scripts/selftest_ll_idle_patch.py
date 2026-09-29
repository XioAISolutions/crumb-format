"""Self-test for the idle-offload patcher/verifier (v3).

Place beside the patcher + verifier, with the box's pre-idle source as
base.py and its live patched source as v2_box.py. Checks:
  1. clean apply -> APPLIED, verify PASS, re-apply -> ALREADY PATCHED (verified)
  2. a v1-style broken file (insert before the else) must FAIL verify AND apply
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
try:
    from longlive_idle_offload_patch import ANCHOR, INSERT
except ImportError:
    from apply_ll_idle import ANCHOR, INSERT


def find(names):
    for d in (HERE.parent, HERE):
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
    raise SystemExit("missing: %s" % (names,))


APPLY = find(["apply_ll_idle.py", "longlive_idle_offload_patch.py"])
VERIFY = find(["verify_ll_idle.py"])


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    ok = True

    def check(name, cond, out):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + " | " + out.replace("\n", " / ")[:300])
        ok = ok and bool(cond)

    base_p = HERE / "base.py"
    if not base_p.exists():
        raise SystemExit("missing fixture base.py (box pre-idle source)")
    base = base_p.read_text()
    clean = HERE / "t_clean.py"
    clean.write_text(base)

    rc, out = run([sys.executable, APPLY, str(clean)])
    check("clean apply -> APPLIED", rc == 0 and out.startswith("APPLIED"), out)

    rc, out = run([sys.executable, VERIFY, str(clean)])
    check("verify clean", rc == 0 and out.startswith("PASS"), out)

    rc, out = run([sys.executable, APPLY, str(clean)])
    check("re-apply idempotent + validated",
          rc == 0 and "ALREADY PATCHED (structure verified)" in out, out)

    broken = HERE / "t_broken.py"
    broken.write_text(base.replace(ANCHOR, INSERT + ANCHOR))  # the v1 mistake

    rc, out = run([sys.executable, VERIFY, str(broken)])
    check("broken file rejected by verify", rc != 0 and "else" in out.lower(), out)

    rc, out = run([sys.executable, APPLY, str(broken)])
    check("broken file rejected by apply (fail closed)", rc != 0, out)

    v2 = HERE / "v2_box.py"
    if v2.exists():
        rc, out = run([sys.executable, VERIFY, str(v2)])
        check("live v2 file passes verify", rc == 0, out)

    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
