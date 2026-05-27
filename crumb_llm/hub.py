"""HuggingFace Hub-compatible model saving and loading.

Saves Crumb LLM checkpoints in a format that can be uploaded to
HuggingFace Hub and loaded back. Includes:
  - Model weights (safetensors-compatible state dict)
  - Config (JSON, maps to WaveFieldConfig)
  - Tokenizer metadata
  - Model card (auto-generated README.md)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model import WaveFieldLM, WaveFieldConfig


MODEL_CARD_TEMPLATE = """\
---
library_name: crumb-llm
tags:
- crumb-llm
- wave-field
- physics-based
- O(N-log-N)
- experimental
license: mit
---

# {model_name}

A **Crumb LLM** checkpoint — an experimental O(N log N) language model
that replaces transformer self-attention with physics-based wave-equation
dynamics.

## Architecture

- **Type:** Wave-Field Language Model (decoder-only)
- **Parameters:** {params:,}
- **Layers:** {n_layers}
- **Heads:** {n_heads} (each with 3 learnable physics params: α, ω, φ)
- **Dim:** {dim}
- **Field size:** {field_size}
- **Vocab:** {vocab_size}
- **Boundary:** {boundary}
- **Dispersion:** {dispersion}
- **Interference mixer:** {interference_mixer}

## How it works

Each layer performs **scatter → FFT-convolve → gather** instead of QKᵀV:

1. Tokens deposit state onto a continuous 1-D field (scatter)
2. A learnable damped-cosine kernel propagates info via FFT (O(F log F))
3. Tokens read back from the field at their positions (gather)

The field size F is independent of sequence length N — the wave-mix
cost stays constant while attention's O(N²) cost grows.

## Usage

```python
from crumb_llm.hub import load_hub_model

model, tok = load_hub_model("{model_name}")
```

Or via CLI:
```bash
pip install 'crumb-format[llm]'
crumb llm generate --ckpt {model_name} --prompt "Hello"
```

## Training

{training_info}

## Citation

```bibtex
@misc{{crumb_llm,
  title={{Crumb LLM: O(N log N) Language Modeling via Wave-Equation Dynamics}},
  year={{2026}},
  url={{https://github.com/XioAISolutions/crumb-format}}
}}
```
"""


def save_for_hub(
    model: WaveFieldLM,
    tokenizer,
    out_dir: str | Path,
    model_name: str = "crumb-llm-tiny",
    training_info: str = "Trained on the crumb-format examples corpus.",
) -> Path:
    """Save a checkpoint in HuggingFace Hub format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Config.
    cfg_dict = asdict(model.cfg)
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    # Weights.
    torch.save(model.state_dict(), out / "model.pt")

    # Tokenizer.
    tokenizer.save(out / "tokenizer.json")

    # Model card.
    n_params = model.num_parameters()
    card = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        params=n_params,
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        dim=model.cfg.dim,
        field_size=model.cfg.field_size,
        vocab_size=model.cfg.vocab_size,
        boundary=model.cfg.boundary,
        dispersion=model.cfg.dispersion,
        interference_mixer=model.cfg.interference_mixer,
        training_info=training_info,
    )
    (out / "README.md").write_text(card)

    print(f"[hub] saved to {out}/")
    print(f"  config.json, model.pt, tokenizer.json, README.md")
    print(f"  Upload: huggingface-cli upload {model_name} {out}")
    return out


def load_hub_model(
    path: str | Path,
    device: str = "cpu",
) -> tuple[WaveFieldLM, object]:
    """Load a Crumb LLM from a Hub-format directory."""
    from .tokenizer import ByteTokenizer, CharTokenizer

    p = Path(path)
    cfg_dict = json.loads((p / "config.json").read_text())
    # Filter out keys that aren't WaveFieldConfig fields.
    valid_keys = set(WaveFieldConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = WaveFieldConfig(**filtered)
    model = WaveFieldLM(cfg)
    state = torch.load(p / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tok_path = p / "tokenizer.json"
    tok_meta = json.loads(tok_path.read_text())
    if tok_meta.get("type") == "byte":
        tok = ByteTokenizer()
    elif tok_meta.get("type") == "char":
        tok = CharTokenizer.load(tok_path)
    else:
        tok = ByteTokenizer()

    return model, tok
