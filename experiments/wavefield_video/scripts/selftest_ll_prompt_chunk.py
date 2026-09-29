"""Self-test for the prompt-chunk patcher + verifier (v1).

Place beside the patcher + verifier with the box's pre-patch source as base.py.
Checks:
  1. clean apply -> APPLIED, verify PASS, re-apply -> ALREADY PATCHED (verified)
  2. a subtly-broken variant (slice restarts at 0) is rejected by verify, and
     demonstrably mis-encodes under the functional mock (detector sensitivity)
  3. functional: chunked runs (default 16, chunk 7) equal the single-call
     encoding (order preserved); call-size sequences [16,16,8] / [7,7,7,7,7,5]
     / [40]; every prompt arrives at the encoder in original order.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
try:
    from longlive_prompt_chunk_patch import MARK, ANCHOR, _ANCHOR_HEAD  # noqa
except ImportError:
    from apply_ll_prompt_chunk import MARK, ANCHOR, _ANCHOR_HEAD  # noqa: F401


def find(names):
    for d in (HERE, HERE.parent):
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
    raise SystemExit("missing: %s" % (names,))


APPLY = find(["apply_ll_prompt_chunk.py", "longlive_prompt_chunk_patch.py"])
VERIFY = find(["verify_ll_prompt_chunk.py"])


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


class FakeT:
    """(n,1,1) prompt-embed stand-in: reshape + per-block batch slicing."""

    def __init__(self, vals, shape):
        self.vals = list(vals)
        self.shape = tuple(shape)

    def reshape(self, *dims):
        total = 1
        for d in dims:
            total *= d
        if total != len(self.vals):
            raise ValueError("reshape %s vs %d vals" % (dims, len(self.vals)))
        return FakeT(self.vals, dims)

    def __getitem__(self, key):
        sl, i = key
        b, s = self.shape[0], self.shape[1]
        per = 1
        for d in self.shape[2:]:
            per *= d
        vals = []
        for bi in range(b):
            base = (bi * s + i) * per
            vals.extend(self.vals[base:base + per])
        return FakeT(vals, (b,) + self.shape[2:])


def _fake_torch():
    m = types.ModuleType("torch")

    def cat(tensors, dim=0):
        if dim != 0:
            raise ValueError("prompt-chunk patch must cat at dim=0")
        tails = {tuple(t.shape[1:]) for t in tensors}
        if len(tails) != 1:
            raise ValueError("cat shape mismatch: %s" % (tails,))
        vals = []
        for t in tensors:
            vals.extend(t.vals)
        return FakeT(vals, (len(vals),) + next(iter(tails)))

    m.cat = cat
    return m


def _load(path):
    sys.modules["torch"] = _fake_torch()  # deterministic; box python may lack torch
    g = {}
    exec(compile(Path(path).read_text(), path, "exec"), g)
    return g["encode_prompt_blocks"]


PROMPTS = ["p%02d" % i for i in range(40)]
IDS = {p: i for i, p in enumerate(PROMPTS)}


def _run(fn, chunk):
    if chunk is None:
        os.environ.pop("LL_PROMPT_CHUNK", None)
    else:
        os.environ["LL_PROMPT_CHUNK"] = str(chunk)
    calls = []

    def enc(*, text_prompts):
        calls.append(list(text_prompts))
        vals = [float(IDS[p]) for p in text_prompts]
        return {"prompt_embeds": FakeT(vals, (len(text_prompts), 1, 1))}

    cd, cdl = fn(enc, [list(PROMPTS)], 1)
    return (
        [len(c) for c in calls],
        [c for c in calls],
        [e["prompt_embeds"].vals for e in cdl],
        cd["prompt_embeds"].vals,
    )


def main():
    ok = True

    def check(name, cond, out=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name
              + ((" | " + str(out)[:220]) if out != "" else ""))
        ok = ok and bool(cond)

    base_p = HERE / "base.py"
    if not base_p.exists():
        raise SystemExit("missing fixture base.py (box pre-patch source)")
    base = base_p.read_text()
    clean = HERE / "t_clean.py"
    clean.write_text(base)

    rc, out = run([sys.executable, VERIFY, str(clean)])
    check("unpatched rejected by verify", rc != 0 and "marker missing" in out, out)
    rc, out = run([sys.executable, APPLY, str(clean)])
    check("clean apply -> APPLIED", rc == 0 and out.startswith("APPLIED"), out)
    rc, out = run([sys.executable, VERIFY, str(clean)])
    check("verify clean", rc == 0 and out.startswith("PASS"), out)
    rc, out = run([sys.executable, APPLY, str(clean)])
    check("re-apply idempotent + validated",
          rc == 0 and "ALREADY PATCHED (structure verified)" in out, out)

    clean_src = clean.read_text()
    if MARK not in clean_src or "flat_prompts[_ll_i:_ll_i + _ll_chunk]" not in clean_src:
        check("mangle target present", False, "source drift")
        print("SELFTEST FAILED")
        return 1
    broken = HERE / "t_broken.py"
    broken.write_text(
        clean_src.replace("flat_prompts[_ll_i:_ll_i + _ll_chunk]",
                          "flat_prompts[0:_ll_chunk]"))
    rc, out = run([sys.executable, VERIFY, str(broken)])
    check("broken rejected by verify", rc != 0 and "slice" in out.lower(), out)
    rc, out = run([sys.executable, APPLY, str(broken)])
    check("broken rejected by apply (fail closed)", rc != 0, out)

    fn = _load(str(clean))
    sizes_def, calls_def, out_def, cd_def = _run(fn, None)
    sizes_7, calls_7, out_7, _ = _run(fn, 7)
    sizes_single, calls_single, out_single, _ = _run(fn, 0)
    os.environ.pop("LL_PROMPT_CHUNK", None)

    check("default call sizes [16,16,8]", sizes_def == [16, 16, 8], sizes_def)
    check("chunk-7 call sizes [7x5,5]",
          sizes_7 == [7, 7, 7, 7, 7, 5], sizes_7)
    check("explicit single-call sizes [40]", sizes_single == [40], sizes_single)
    check("prompts arrive in order",
          calls_def == [PROMPTS[0:16], PROMPTS[16:32], PROMPTS[32:40]],
          sizes_def)
    check("per-block embeddings match single-call",
          out_def == out_single, "equal" if out_def == out_single else out_def[:3])
    check("chunk-7 embeddings match single-call",
          out_7 == out_single, "equal" if out_7 == out_single else out_7[:3])
    check("block i -> prompt i end-to-end",
          out_single == [[float(i)] for i in range(40)], out_single[:3])
    check("conditional_dict prompt_embeds in order",
          cd_def == [float(i) for i in range(40)], cd_def[:3])

    # sensitivity 1: slice bug (verify rejects it; if it ever slipped through,
    # the module's own count guard catches it at runtime)
    try:
        fbn = _load(str(broken))
        _, _, out_broken, _ = _run(fbn, None)
        det = out_broken != out_single
        detail = out_broken[:3]
    except Exception as e:
        det = True
        detail = "runtime guard: %s" % (e,)
    check("slice-bug variant caught (runtime/equality)", det, detail)

    # sensitivity 2: reversed cat order (bypasses verify here on purpose) --
    # proves the equality test alone catches ordering defects
    if "cat(_ll_parts, dim=0)" not in clean_src:
        check("reversal mangle target present", False, "source drift")
    else:
        rev = HERE / "t_reversed.py"
        rev.write_text(clean_src.replace("cat(_ll_parts, dim=0)",
                                         "cat(_ll_parts[::-1], dim=0)"))
        frv = _load(str(rev))
        _, _, out_rev, _ = _run(frv, None)
        os.environ.pop("LL_PROMPT_CHUNK", None)
        check("reversed-order variant caught by equality test",
              out_rev != out_single, out_rev[:3])

    v2 = HERE / "v2_box.py"
    if v2.exists():
        rc, out = run([sys.executable, VERIFY, str(v2)])
        check("live box file passes verify", rc == 0, out)

    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
