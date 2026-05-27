"""Linear-interpolation scatter and gather between tokens and the field.

Tokens live at integer positions 0..N-1; the field has F cells. We map
each token's position to a continuous field coordinate

    x_n = (n + 0.5) · F / N

then scatter its state onto the two nearest field cells with linear
weights, and gather by reading those same two cells back. Scatter is
the adjoint of gather, which is what makes the layer's gradient flow
symmetric and numerically clean.

We expose two functions:

* ``scatter_linear(values, positions, field_size)`` → field tensor
* ``gather_linear(field, positions)``                → token tensor

Both are written with native torch ops (``index_add_``, ``gather``) so
they run on GPU without Python loops and are autodiff-friendly.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _continuous_positions(N: int, F: int, device, dtype) -> Tensor:
    """Token-index → field-coordinate. Returns ``[N]`` floats in [0, F)."""
    idx = torch.arange(N, device=device, dtype=dtype)
    # Centre tokens inside their cells; clamp keeps the last token's right
    # neighbour in-bounds at index F-1.
    return torch.clamp((idx + 0.5) * (F / N), max=float(F - 1) - 1e-6)


def scatter_linear(
    values: Tensor,
    field_size: int,
    weights: Tensor | None = None,
) -> Tensor:
    """Deposit token values onto a continuous field with linear weights.

    Args:
        values:     ``[B, H, N, D]`` per-token state.
        field_size: F.
        weights:    optional ``[B, N]`` per-token amplitude (e.g. crumb
                    @priority). Multiplies the deposit. Defaults to ones.

    Returns:
        ``[B, H, F, D]`` field tensor.
    """
    B, H, N, D = values.shape
    F = field_size
    device, dtype = values.device, values.dtype

    pos = _continuous_positions(N, F, device, dtype)        # [N]
    left = pos.floor().long()                                # [N]
    frac = (pos - left.to(dtype)).view(1, 1, N, 1)           # [1,1,N,1]
    right = (left + 1).clamp(max=F - 1)                       # [N]

    if weights is not None:
        amp = weights.view(B, 1, N, 1)
        values = values * amp

    field = values.new_zeros((B, H, F, D))
    # index_add_ wants 1-D indices for the indexed axis; we expand the
    # value tensors so the (1 - frac) and frac weights ride along.
    # Use float scatter via index_add_ on a flattened (B*H, F, D) view.
    # ``values`` may not be contiguous (came from transpose); reshape it.
    flat_field = field.reshape(B * H, F, D)
    flat_values = values.reshape(B * H, N, D)
    flat_field.index_add_(1, left, flat_values * (1.0 - frac).view(1, N, 1))
    flat_field.index_add_(1, right, flat_values * frac.view(1, N, 1))
    return flat_field.reshape(B, H, F, D)


def gather_linear(field: Tensor, num_tokens: int) -> Tensor:
    """Read token states from the field at their continuous positions.

    Adjoint of ``scatter_linear`` (with weights=None): same left/right
    cells, same linear weights.

    Args:
        field:      ``[B, H, F, D]``.
        num_tokens: N.

    Returns:
        ``[B, H, N, D]`` per-token state.
    """
    B, H, F, D = field.shape
    N = num_tokens
    device, dtype = field.device, field.dtype

    pos = _continuous_positions(N, F, device, dtype)
    left = pos.floor().long()
    frac = (pos - left.to(dtype)).view(1, 1, N, 1)
    right = (left + 1).clamp(max=F - 1)

    # Advanced indexing: field[:, :, idx, :] → [B, H, N, D].
    left_vals = field[:, :, left, :]
    right_vals = field[:, :, right, :]
    return left_vals * (1.0 - frac) + right_vals * frac
