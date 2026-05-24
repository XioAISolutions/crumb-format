"""``python -m wave_field_llm`` → quick health check.

Prints a one-line summary of available subcommands and a smoke-test
result (random weights forward pass on a tiny model). Useful for
verifying the package imports cleanly in a fresh environment.
"""

from __future__ import annotations

import sys

import torch

from . import __version__
from .model import WaveFieldLM, WaveFieldConfig


def main(argv: list[str] | None = None) -> int:
    print(f"wave_field_llm {__version__}")
    print("subcommands:")
    print("  python -m wave_field_llm.train  --help")
    print("  python -m wave_field_llm.sample --help")
    print("  python -m wave_field_llm.bench  --help")
    print()
    cfg = WaveFieldConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, field_size=32)
    model = WaveFieldLM(cfg)
    x = torch.randint(0, 64, (1, 16))
    out = model(x)
    print(f"smoke test: forward OK — logits shape={tuple(out['logits'].shape)}, "
          f"params={model.num_parameters():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
