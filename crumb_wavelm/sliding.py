"""Sliding-window field for sequences longer than the field size.

When N > F, the basic scatter/gather approach breaks: tokens beyond the
field's right edge have nowhere to deposit. The sliding-window variant
solves this by moving the field's "viewport" as the sequence progresses,
re-centring the field around the recent context.

Two strategies:

1. **Truncate-right** (current default): only the last F tokens are
   mapped to the field. Simple, fast, but throws away long-range context.

2. **Sliding field**: the field maintains a "position offset" that shifts
   as new tokens arrive. Old tokens that fall off the left edge of the
   field are forgotten (their energy has already propagated via the wave
   kernel). New tokens are scattered relative to the shifted origin.

   This is analogous to a sliding-window attention (like Mistral's), but
   the "window" is the physics field, and information from old tokens
   persists as residual wave energy even after their source position
   leaves the viewport.

For the field-state cache (incremental generation), sliding works
naturally: when seq_pos > F, we shift the field by rolling it left and
zeroing the vacated right edge. The accumulated wave energy from past
tokens is preserved in the remaining field cells.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sliding_scatter(
    values: Tensor,         # [B, H, N, D]
    field_size: int,
    window_start: int = 0,  # first token index visible in the field
    weights: Tensor | None = None,
) -> Tensor:
    """Scatter tokens [window_start, window_start+N) onto a field of size F.

    Tokens outside the field viewport are silently dropped. This is the
    sliding-window variant of ``scatter_linear``.
    """
    from .scatter_gather import _continuous_positions

    B, H, N, D = values.shape
    F = field_size
    device, dtype = values.device, values.dtype

    # Map absolute token positions to field coordinates.
    # Token at absolute position (window_start + n) maps to field coord:
    #   x = (n + 0.5) * F / N
    # This is the same as the regular scatter, but offset.
    pos = _continuous_positions(N, F, device, dtype)
    left = pos.floor().long()
    frac = (pos - left.to(dtype)).view(1, 1, N, 1)
    right = (left + 1).clamp(max=F - 1)

    if weights is not None:
        amp = weights.view(B, 1, N, 1)
        values = values * amp

    field = values.new_zeros((B, H, F, D))
    flat_field = field.reshape(B * H, F, D)
    flat_values = values.reshape(B * H, N, D)
    flat_field.index_add_(1, left, flat_values * (1.0 - frac).view(1, N, 1))
    flat_field.index_add_(1, right, flat_values * frac.view(1, N, 1))
    return flat_field.reshape(B, H, F, D)


def shift_field(
    field: Tensor,          # [B, H, F, D]
    shift_cells: int,       # how many cells to shift left
) -> Tensor:
    """Roll the field left by ``shift_cells`` and zero the vacated edge.

    Used during sliding-window generation: old tokens' positions move
    off the left edge, but their wave energy is preserved in the remaining
    cells. The vacated cells on the right are zeroed to accept new tokens.
    """
    if shift_cells <= 0:
        return field
    F = field.shape[-2]
    if shift_cells >= F:
        return torch.zeros_like(field)
    # Roll left.
    shifted = torch.roll(field, shifts=-shift_cells, dims=-2)
    # Zero the right edge that wrapped around.
    shifted[:, :, -shift_cells:, :] = 0.0
    return shifted


class SlidingFieldManager:
    """Track the sliding window's position and manage field shifts.

    During generation, once ``seq_pos`` exceeds the field size, the
    manager shifts the field left by ``stride`` cells, making room for
    new tokens while preserving accumulated wave energy.
    """

    def __init__(self, field_size: int, stride: int | None = None):
        self.field_size = field_size
        self.stride = stride or (field_size // 4)
        self.window_start = 0  # absolute position of the leftmost token in the field

    def should_shift(self, seq_pos: int) -> bool:
        """True if the next token would fall outside the current field."""
        return (seq_pos - self.window_start) >= self.field_size

    def shift(self, field: Tensor) -> Tensor:
        """Shift the field left by ``stride`` cells."""
        field = shift_field(field, self.stride)
        self.window_start += self.stride
        return field

    def token_field_pos(self, absolute_pos: int) -> int:
        """Map absolute token position to field-relative position."""
        return absolute_pos - self.window_start
