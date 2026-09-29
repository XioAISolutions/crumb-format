"""Offload idle TE + VAE before denoise in pinned LongLive; fail closed on drift.

sm89 24GB edge: `inference.py` leaves the text encoder (~11 GB bf16) and the VAE
on the card for the whole run, but both are idle once prompts are encoded
(decoded output is produced separately for latents-only streaming). This insert
moves them to CPU right after the encode block, only when LL_OFFLOAD_IDLE=1.
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
    new = src.replace(ANCHOR, INSERT + ANCHOR)
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
