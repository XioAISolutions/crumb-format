"""Verify the LL_OFFLOAD_IDLE insert in the pinned LongLive pipeline (AST-level).

Checks the failure mode that compiled and passed naive regressions: the insert
must not sit between `if use_cfg:` and its `else:` (else-stealing), must keep
that else intact, and must itself carry no else clause.
"""
import ast
import sys
from pathlib import Path

MARK = "# LONGLIVE_IDLE_OFFLOAD_V1"


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/LongLive/pipeline/causal_diffusion_inference.py")
    src = p.read_text()
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + msg)
        ok = ok and bool(cond)

    chk(MARK in src, "marker present")
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "CausalDiffusionInferencePipeline"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
    use_cfg_if = offload_if = None
    for st in fn.body:
        if isinstance(st, ast.If):
            t = ast.unparse(st.test)
            if t == "use_cfg":
                use_cfg_if = st
            if "LL_OFFLOAD_IDLE" in t:
                offload_if = st
    chk(use_cfg_if is not None, "use_cfg if found")
    if use_cfg_if is not None:
        chk(
            any(
                isinstance(s, ast.Assign)
                and any(isinstance(tx, ast.Name) and tx.id == "unconditional_dict" for tx in s.targets)
                for s in use_cfg_if.orelse
            ),
            "use_cfg else assigns unconditional_dict = None (intact)",
        )
    chk(offload_if is not None, "offload if found")
    if offload_if is not None:
        chk(not offload_if.orelse, "offload if has NO else (else not stolen)")
        t = ast.unparse(offload_if.test)
        chk("LL_OFFLOAD_IDLE" in t and "text_encoder is not None" in t, "offload guard: env + TE not None")
        body = ast.unparse(offload_if.body)
        chk("to('cpu')" in body, "offload to cpu present")
        chk("self.vae.to('cpu')" in body, "VAE offload present")
        chk("empty_cache" in body, "empty_cache present")
    order_ok = False
    if use_cfg_if is not None and offload_if is not None:
        idx = {id(s): i for i, s in enumerate(fn.body)}
        order_ok = idx[id(use_cfg_if)] < idx[id(offload_if)]
    chk(order_ok, "offload ordered after use_cfg if/else")

    print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
