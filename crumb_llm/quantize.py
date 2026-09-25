"""int8 dynamic quantization for Wave-Field LLMs.

PyTorch's ``quantize_dynamic`` quantizes weights of selected modules
(``nn.Linear`` by default) to int8 and dequantizes on the fly at matmul
time. For CPU inference this is the cheapest size/speed win: roughly 4x
smaller weights and 2-3x faster matmuls, with minimal accuracy loss for
language modeling at small-to-mid scales.

Why only Linear:
    The wave-field block's FFT path uses complex tensors (rfft/irfft)
    which PyTorch's dynamic quantization does not support. The Linear
    layers — ``proj_in``, ``proj_out``, ``ffn`` MLPs, optional ``lm_head``
    — are the matmul-heavy parts and the right quantization targets.

Embeddings are intentionally left in float: int8 embedding quantization
needs static observers + calibration data, which is out of scope here.
For tiny models the embedding table is often the single biggest weight,
so the practical size win is bounded by ``(model_size - embed_size) / 4``.

Usage:
    from crumb_llm.quantize import quantize_dynamic_linear, model_size_mb
    qmodel = quantize_dynamic_linear(model)
    print(f"{model_size_mb(model):.1f} MB -> {model_size_mb(qmodel):.1f} MB")
"""

from __future__ import annotations

import io
import torch
import torch.nn as nn


def _ensure_qengine() -> None:
    """Pick a quantization engine that matches the host CPU.

    PyTorch ships engines named ``fbgemm`` (x86) and ``qnnpack`` (ARM).
    On Apple Silicon and other ARM hosts, ``fbgemm`` is unavailable and
    leaving the engine unset raises ``Didn't find engine for operation
    quantized::linear_prepack NoQEngine``.
    """
    supported = torch.backends.quantized.supported_engines
    current = torch.backends.quantized.engine
    if current in supported and current != "none":
        return
    for candidate in ("qnnpack", "fbgemm"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return
    raise RuntimeError(
        f"No usable PyTorch quantization engine. supported={supported}"
    )


def quantize_dynamic_linear(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Return a copy of ``model`` with ``nn.Linear`` layers int8-quantized.

    Note: PyTorch dynamic quantization returns a new model; the original
    is left intact. The returned model is CPU-only — dynamic quant has no
    CUDA path. Run ``model.cpu()`` first if needed.
    """
    _ensure_qengine()
    return torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(), {nn.Linear}, dtype=dtype,
    )


def model_size_mb(model: nn.Module) -> float:
    """Return the serialized model size in MB.

    Uses an in-memory buffer to avoid touching the filesystem. Accounts
    for both float and quantized tensors (quantized tensors serialize
    with their scale/zero-point metadata).
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def linear_param_breakdown(model: nn.Module) -> dict:
    """Diagnostic — how many params are in Linear vs other layers.

    Useful for sizing the practical win from int8 quantization: only the
    ``linear`` chunk is affected; the rest stays float.
    """
    linear = 0
    embedding = 0
    other = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear += sum(p.numel() for p in m.parameters())
        elif isinstance(m, nn.Embedding):
            embedding += sum(p.numel() for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    other = total - linear - embedding
    return {
        "total_params": total,
        "linear_params": linear,
        "embedding_params": embedding,
        "other_params": other,
        "linear_pct": (linear / total * 100) if total else 0.0,
    }
