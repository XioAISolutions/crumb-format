"""Standalone verifier for the LL_PROMPT_CHUNK_V1 insert (AST checks via the patcher)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from longlive_prompt_chunk_patch import MARK, assert_structure
except ImportError:
    from apply_ll_prompt_chunk import MARK, assert_structure


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/root/LongLive/utils/prompt_conditioning.py")
    src = p.read_text()
    if MARK not in src:
        print("FAIL  marker missing")
        return 1
    try:
        assert_structure(src)
    except SystemExit as e:
        print("FAIL ", e)
        return 1
    print("PASS  structure verified: chunk guard, single-call path, ordered "
          "sliced chunk loop, cat dim=0, prompt_embeds read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
