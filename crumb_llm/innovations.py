"""Novel architectural innovations unique to Crumb LLM.

These go beyond the base Wave-Field LLM architecture (Badaramoni 2026)
and constitute Crumb LLM's original contributions:

1. **Adaptive Kernels** — input-conditioned physics parameters.
   Instead of fixed α/ω/φ per head, a small network predicts them from
   the input. The kernel shape adapts to content, like how attention is
   content-dependent.

2. **Field-Attention Hybrid** — a learned gate between wave-field
   mixing and standard attention within the same block. Captures both
   global patterns (wave field, O(N log N)) and local patterns
   (attention, O(N²) but only on a small window).

3. **Resonance Memory** — standing waves that persist across layers
   form natural memory patterns. A cross-layer residual field
   accumulates and reinforces dominant frequencies.

4. **Spectral Gating** — instead of the entire spectrum propagating
   equally, a learned gate in frequency space selectively amplifies
   or suppresses spectral bands. The model learns which frequencies
   carry useful information for the current token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch import Tensor


# ── 1. Adaptive Kernels ──────────────────────────────────────────────


class AdaptiveKernelHead(nn.Module):
    """Input-conditioned wave kernel: α, ω, φ are predicted from context.

    Standard wave-field heads have static per-head parameters. This head
    runs a small MLP on the mean-pooled input to produce per-sample,
    per-head physics parameters. The kernel adapts to what's being
    processed — analogous to how Q/K/V in attention are input-dependent.

    This is Crumb LLM's answer to "why should the kernel be static when
    the input is dynamic?"
    """

    def __init__(self, dim: int, field_size: int, causal: bool = True):
        super().__init__()
        self.field_size = field_size
        self.causal = causal
        # Small predictor: pool(x) → 3 physics params.
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 3),  # α, ω, φ
        )
        # Learnable base values (the predictor outputs residuals).
        self.alpha_base = nn.Parameter(torch.tensor(0.05))
        self.omega_base = nn.Parameter(torch.tensor(1.0))
        self.phi_base = nn.Parameter(torch.tensor(0.0))
        # Pre-compute time grid.
        t = torch.arange(field_size, dtype=torch.float32) - field_size // 2
        self.register_buffer("t", t)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D] → kernel spectrum [B, F//2+1] complex.

        Returns per-sample kernels (batch dimension preserved).
        """
        # Pool input to get a summary vector.
        pooled = x.mean(dim=1)  # [B, D]
        params = self.predictor(pooled)  # [B, 3]
        alpha = (self.alpha_base + params[:, 0]).abs()  # [B]
        omega = self.omega_base + params[:, 1]           # [B]
        phi = self.phi_base + params[:, 2]               # [B]

        # Build per-sample kernels.
        # t: [F], alpha/omega/phi: [B] → k: [B, F]
        a = alpha.unsqueeze(-1)  # [B, 1]
        w = omega.unsqueeze(-1)
        p = phi.unsqueeze(-1)
        k = torch.exp(-a * self.t.abs()) * torch.cos(w * self.t + p)

        if self.causal:
            mask = torch.zeros_like(k)
            mask[:, self.field_size // 2:] = 1.0
            k = k * mask

        k = torch.fft.ifftshift(k, dim=-1)
        return torch.fft.rfft(k, n=self.field_size, dim=-1)  # [B, F//2+1]


# ── 2. Field-Attention Hybrid ────────────────────────────────────────


class FieldAttentionGate(nn.Module):
    """Learned gate between wave-field mixing and windowed attention.

    For each token, a scalar gate decides how much to rely on the
    wave-field (global, O(N log N)) vs local attention (windowed,
    O(N·W)). The model learns when global context matters vs when
    local detail is more important.

    gate=1.0 → pure wave-field (long-range patterns)
    gate=0.0 → pure local attention (short-range detail)
    """

    def __init__(self, dim: int, window_size: int = 64):
        super().__init__()
        self.window_size = window_size
        self.gate_proj = nn.Linear(dim, 1)
        # Local attention components.
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def local_attention(self, x: Tensor) -> Tensor:
        """Windowed causal attention, O(N·W)."""
        B, N, D = x.shape
        W = min(self.window_size, N)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simple causal attention (no flash, just reference impl).
        # Mask: each token attends to at most W previous tokens.
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(D)
        # Causal + window mask.
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool).triu(1)
        # Window: mask out tokens more than W positions back.
        window_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril(-W)
        mask = mask | window_mask
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn, v)

    def forward(self, x: Tensor, wave_output: Tensor) -> Tensor:
        """Blend wave-field output with local attention output.

        Args:
            x: [B, N, D] — input to the block.
            wave_output: [B, N, D] — output from the wave-field mixer.

        Returns: [B, N, D] — gated blend.
        """
        gate = torch.sigmoid(self.gate_proj(x))  # [B, N, 1]
        local_out = self.local_attention(x)
        return gate * wave_output + (1.0 - gate) * local_out


# ── 3. Resonance Memory ──────────────────────────────────────────────


class ResonanceMemory(nn.Module):
    """Cross-layer field accumulator that reinforces standing waves.

    Maintains a "memory field" that accumulates spectral energy across
    layers. Strong frequencies (standing waves) that persist across
    multiple layers get amplified — they represent stable structural
    patterns in the input (like section boundaries in a crumb).

    The memory field is a learnable exponential moving average of each
    layer's field spectrum, with a per-frequency decay rate.
    """

    def __init__(self, n_heads: int, field_size: int):
        super().__init__()
        spec_size = field_size // 2 + 1
        # Per-frequency, per-head decay rate (how fast old layers fade).
        self.decay = nn.Parameter(torch.full((n_heads, spec_size), 0.9))
        # Injection strength (how much new layer data enters memory).
        self.inject = nn.Parameter(torch.full((n_heads, spec_size), 0.1))

    def forward(
        self,
        memory_spec: Tensor,   # [B, H, F//2+1] — accumulated memory
        new_field: Tensor,     # [B, H, F, D] — this layer's output field
    ) -> tuple[Tensor, Tensor]:
        """Update memory with this layer's field, return amplified field.

        Returns:
            (updated_memory [B, H, F//2+1], amplified_field [B, H, F, D])
        """
        # Compute power spectrum of new field (average over D).
        new_spec = torch.fft.rfft(new_field, dim=-2).abs().mean(dim=-1)  # [B, H, F//2+1]
        # EMA update.
        decay = self.decay.abs().clamp(0, 0.99).unsqueeze(0)
        inject = self.inject.abs().clamp(0, 0.5).unsqueeze(0)
        updated = decay * memory_spec + inject * new_spec

        # Amplify the field at frequencies where memory is strong.
        # Convert memory strength to a per-frequency gain.
        gain = 1.0 + torch.tanh(updated)  # [B, H, F//2+1], range [0, 2]
        spec = torch.fft.rfft(new_field, dim=-2)
        amplified_spec = spec * gain.unsqueeze(-1)
        amplified_field = torch.fft.irfft(amplified_spec, n=new_field.shape[-2], dim=-2)

        return updated, amplified_field


# ── 4. Spectral Gating ───────────────────────────────────────────────


class SpectralGate(nn.Module):
    """Learned frequency-domain gate: selectively pass/block spectral bands.

    Instead of all frequencies propagating equally after convolution,
    this gate learns which frequency bands carry useful signal for the
    current input. Low-info bands are suppressed, high-info bands
    amplified.

    This is the frequency-domain analog of a sigmoid gate in a GRU/LSTM,
    but operating on the spatial spectrum of the wave field.
    """

    def __init__(self, n_heads: int, field_size: int, dim: int):
        super().__init__()
        spec_size = field_size // 2 + 1
        # Input-conditioned gate: pool → project → per-frequency gate.
        self.gate_net = nn.Sequential(
            nn.Linear(dim, n_heads * spec_size),
        )
        self.n_heads = n_heads
        self.spec_size = spec_size

    def forward(self, field: Tensor, x_pooled: Tensor) -> Tensor:
        """Apply learned spectral gate to the convolved field.

        Args:
            field: [B, H, F, D] — field after convolution.
            x_pooled: [B, D] — pooled input (for conditioning the gate).

        Returns: [B, H, F, D] — spectrally gated field.
        """
        B = field.shape[0]
        F_ = field.shape[-2]
        # Predict gate values.
        gate_flat = torch.sigmoid(self.gate_net(x_pooled))  # [B, H * spec]
        gate = gate_flat.view(B, self.n_heads, self.spec_size, 1)  # [B, H, spec, 1]
        # Apply in frequency domain.
        spec = torch.fft.rfft(field, dim=-2)  # [B, H, F//2+1, D]
        gated = spec * gate
        return torch.fft.irfft(gated, n=F_, dim=-2)
