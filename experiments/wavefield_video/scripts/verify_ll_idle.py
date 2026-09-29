"""Standalone verifier for the LL_OFFLOAD_IDLE insert (delegates AST checks to the patcher)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from longlive_idle_offload_patch import MARK, assert_structure
except ImportError:
    from apply_ll_idle import MARK, assert_structure


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/root/LongLive/pipeline/causal_diffusion_inference.py")
    src = p.read_text()
    if MARK not in src:
        print("FAIL  marker missing")
        return 1
    try:
        assert_structure(src)
    except SystemExit as e:
        print("FAIL ", e)
        return 1
    print("PASS  structure verified: else intact, no else stealing, guard, "
          "TE+VAE->cpu, empty_cache, ordering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
