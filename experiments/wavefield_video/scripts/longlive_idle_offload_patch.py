"""TE+VAE idle-offload for pinned LongLive; fail closed.

v3: validate-don't-trust marker path, explicit raises (no bare asserts),
enforced guard + TE/VAE transfer checks, ordering check, truthful reporting.
Insert goes AFTER the complete `use_cfg` if/else — never between `if` and its
`else:` (that reattaches the else and silently unsets `unconditional_dict`).
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARK = "# LONGLIVE_IDLE_OFFLOAD_V1"
ANCHOR = "        else:\n            unconditional_dict = None\n\n        output = torch.zeros(\n"

INSERT = '''        import os as _ll_os  # noqa: F401 (local handle; exec-safe)

        if _ll_os.environ.get("LL_OFFLOAD_IDLE") == "1" and self.text_encoder is not None:
            # LONGLIVE_IDLE_OFFLOAD_V1 (sm89 24GB edge: TE + VAE idle during denoise)
            _ll_te = "skipped"
            _ll_vae = "skipped"
            try:
                self.text_encoder.to("cpu")
                _ll_te = "cpu"
            except Exception as _ll_exc:
                _ll_te = "FAILED: %s" % (_ll_exc,)
            try:
                self.vae.to("cpu")
                _ll_vae = "cpu"
            except Exception as _ll_exc:
                _ll_vae = "FAILED: %s" % (_ll_exc,)
            torch.cuda.empty_cache()
            print(
                "[mem] idle-offload: TE=%s VAE=%s; alloc=%.2f GiB"
                % (_ll_te, _ll_vae, torch.cuda.memory_allocated() / 2**30),
                flush=True,
            )

'''

REPLACEMENT = (
    "        else:\n            unconditional_dict = None\n\n" + INSERT
    + "        output = torch.zeros(\n"
)


def fail(msg):
    raise SystemExit("STRUCTURE FAIL: " + msg)


def fn_inference(src):
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name == "CausalDiffusionInferencePipeline"), None)
    if cls is None:
        fail("pipeline class not found")
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == "inference"), None)
    if fn is None:
        fail("inference() not found")
    return fn


def is_cpu_call(node, owner):
    f = getattr(node, "func", None)
    v = getattr(f, "value", None)
    return (isinstance(node, ast.Call) and isinstance(f, ast.Attribute)
            and f.attr == "to" and isinstance(v, ast.Attribute) and v.attr == owner
            and isinstance(v.value, ast.Name) and v.value.id == "self"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "cpu")


def assert_structure(src):
    fn = fn_inference(src)
    stmts = fn.body
    uif = oif = None
    imp_i = oi = None
    for i, st in enumerate(stmts):
        if isinstance(st, ast.If):
            t = ast.unparse(st.test)
            if t == "use_cfg":
                if uif is not None:
                    fail("duplicate use_cfg if")
                uif = st
            if "LL_OFFLOAD_IDLE" in t:
                if oif is not None:
                    fail("duplicate LL_OFFLOAD_IDLE if")
                oif, oi = st, i
        if isinstance(st, ast.Import) and any(a.asname == "_ll_os" for a in st.names):
            imp_i = i
    if uif is None:
        fail("use_cfg if not found")
    ok = False
    for s in uif.orelse:
        if (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id == "unconditional_dict"
                and isinstance(s.value, ast.Constant) and s.value.value is None):
            ok = True
    if not ok:
        fail("use_cfg else must assign unconditional_dict = None "
             "(else reattachment / else lost)")
    if oif is None:
        fail("offload if not found")
    if oif.orelse:
        fail("offload if must not have an else (else stealing)")
    t = ast.unparse(oif.test)
    if '"1"' not in t and "'1'" not in t:
        fail('offload guard must compare to "1"')
    if "text_encoder is not None" not in t:
        fail("offload guard must include text_encoder is not None")
    calls = list(ast.walk(oif))
    if not any(is_cpu_call(n, "text_encoder") for n in calls):
        fail("body must move self.text_encoder to cpu")
    if not any(is_cpu_call(n, "vae") for n in calls):
        fail("body must move self.vae to cpu")
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "empty_cache" for n in calls):
        fail("body must call empty_cache")
    if imp_i is None:
        fail("_ll_os import not found")
    if not (stmts.index(uif) < imp_i < oi):
        fail("ordering: use_cfg if/else < import < offload if")


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/LongLive/pipeline/causal_diffusion_inference.py")
    src = src_path.read_text()
    if MARK in src:
        assert_structure(src)  # v3: validate, never trust the marker alone
        print("ALREADY PATCHED (structure verified):", src_path)
        return 0
    if src.count(ANCHOR) != 1:
        print("ANCHOR FAIL: found %d in %s" % (src.count(ANCHOR), src_path))
        return 2
    new = src.replace(ANCHOR, REPLACEMENT)
    assert_structure(new)
    ast.parse(new)
    backup = Path(str(src_path) + ".before_idle")
    if not backup.exists():
        backup.write_text(src)
    tmp = Path(str(src_path) + ".tmp")
    tmp.write_text(new)
    os.replace(tmp, src_path)
    print("APPLIED", hashlib.sha256(new.encode()).hexdigest(), src_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
