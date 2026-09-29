"""Offload idle TE + VAE before denoise in pinned LongLive; fail closed on drift.

sm89 24GB edge: `inference.py` leaves the text encoder (~11 GB bf16) and the VAE
on the card for the whole run, but both are idle once prompts are encoded
(decoded output is produced separately for latents-only streaming). The insert
goes AFTER the complete `use_cfg` if/else — never between `if use_cfg:` and its
`else:` (v1 did that: the else reattached to the inserted `if` and silently
unset `unconditional_dict` on the env=1 path; it compiled and passed naive
regressions, caught by region readback before the first run). Guarded by
LL_OFFLOAD_IDLE=1; _assert_structure() fails closed on else reattachment.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARK = "# LONGLIVE_IDLE_OFFLOAD_V1"

ANCHOR = (
    "        else:\n"
    "            unconditional_dict = None\n"
    "\n"
    "        output = torch.zeros(\n"
)

INSERT = '''        import os as _ll_os  # noqa: F401  (local handle keeps the block exec-safe)

        if _ll_os.environ.get("LL_OFFLOAD_IDLE") == "1" and self.text_encoder is not None:
            # LONGLIVE_IDLE_OFFLOAD_V1 (sm89 24GB edge: TE + VAE idle during denoise)
            try:
                self.text_encoder.to("cpu")
            except Exception:
                pass
            try:
                self.vae.to("cpu")
            except Exception:
                pass
            torch.cuda.empty_cache()
            print(
                "[mem] idle-offload: TE+VAE -> cpu; alloc=%.2f GiB"
                % (torch.cuda.memory_allocated() / 2**30),
                flush=True,
            )

'''

# Keep the whole if/else intact, then the offload block, then the next statement.
REPLACEMENT = (
    "        else:\n"
    "            unconditional_dict = None\n"
    "\n"
    + INSERT
    + "        output = torch.zeros(\n"
)


def _assert_structure(new_src: str) -> None:
    """Fail closed if the use_cfg if/else was split or the insert stole an else."""
    tree = ast.parse(new_src)
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
    assert use_cfg_if is not None, "use_cfg if/else not found"
    assigns = [
        st for st in use_cfg_if.orelse
        if isinstance(st, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "unconditional_dict" for t in st.targets)
    ]
    assert assigns, "use_cfg if/else lost its else (unconditional_dict = None)"
    assert offload_if is not None, "offload if not found"
    assert not offload_if.orelse, "offload if must NOT have an else clause"
    test = ast.unparse(offload_if.test)
    assert "LL_OFFLOAD_IDLE" in test and "text_encoder is not None" in test
    body_txt = ast.unparse(offload_if.body)
    assert "to('cpu')" in body_txt, "offload moves missing from body"


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/LongLive/pipeline/causal_diffusion_inference.py"
    )
    src = src_path.read_text()
    if MARK in src:
        print("ALREADY PATCHED:", src_path)
        return 0
    count = src.count(ANCHOR)
    if count != 1:
        print(f"ANCHOR FAIL: expected 1, found {count} in {src_path}")
        return 2
    new = src.replace(ANCHOR, REPLACEMENT)
    _assert_structure(new)
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
