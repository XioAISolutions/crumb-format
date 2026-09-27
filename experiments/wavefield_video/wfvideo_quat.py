"""Quaternion-structured modules for the wave arm -- the E2/E3 hypercomplex arms.

E2 ``WaveQuatMix`` mirrors ``wfvideo.WaveMix3D``'s dispersion path exactly by
INHERITANCE (kernels, transfer function, ``init_state`` / ``step`` / ``state_bytes``
are literally the same code -- only the construction call differs), and replaces
the dense real ``pi`` / ``po`` Linears with Hamilton-structured maps:
``dim = nh * dh`` is read as ``nh`` heads x ``dh/4`` quaternion slots, so each
in/out slot pair stores a quaternion (4 reals) instead of a scalar -- fewer free
parameters at equal nominal width, with cross-component coupling built in. The
``--target-params`` bisection in ``train_compare.py`` reinvests the saving in the
FFN, so the honest summary is "same total params, different allocation + structure".

E3 ``QuatEmbed`` replaces the stem conv: frame RGB -> pure quaternion (0, r, g, b)
per pixel -> Hamilton-structured 3x3 conv, ``dim/4`` quaternion channels
(~9*dim params vs 27*dim for ``Conv2d(3, dim, 3)``). Honest note (study §4-E3):
unit-quaternion multiplication rotates the RGB vector about an axis through the
gray diagonal -- a smooth structure-preserving color transform, NOT exactly HSL
hue; no "hue equivariance" claim is made anywhere.

Both are wired in ``train_compare.py`` behind default-off flags (``--q-mix``,
``--quat-color``); the flag-off paths construct the existing classes and are
asserted byte-identical in ``test_hypercomplex_r15.py``. ``QuatFFN`` below is the
study's optional ``--q-ffn`` sketch -- kept in-tree unwired (default-off scope).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wfvideo import WaveMix3D


def qmul(a, b):
    """Hamilton product on the trailing size-4 axis, broadcasting."""
    a0, a1, a2, a3 = a.unbind(-1)
    b0, b1, b2, b3 = b.unbind(-1)
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], dim=-1)


def _qbasis(x):
    """[e_p ⊗ x] for p = 0..3 as 4 signed permutations of the trailing 4-axis."""
    x0, x1, x2, x3 = x.unbind(-1)
    return [torch.stack([x0, x1, x2, x3], -1),
            torch.stack([-x1, x0, -x3, x2], -1),
            torch.stack([-x2, x3, x0, -x1], -1),
            torch.stack([-x3, -x2, x1, x0], -1)]


class QuaternionLinear(nn.Module):
    """Hamilton-structured per-head linear map on quaternion slot groups.

    Input/output ``[..., dim_in] -> [..., dim_out]`` is read as quaternion slot
    groups ``[..., nh, slots, 4]`` (``slots = dim / (4 * nh)``); the weight is one
    quaternion per (out-slot, in-slot) pair. Computes ``sum_p W_p * (e_p ⊗ x)``
    over the 4 basis quaternions -- the bilinear expansion of ``W ⊗ x`` into real
    linear maps (no quaternion algebra in the autograd graph beyond elementwise
    ops and einsums).
    """

    def __init__(self, dim_in, dim_out, n_heads, bias=True):
        super().__init__()
        if dim_in % (4 * n_heads) or dim_out % (4 * n_heads):
            raise ValueError("QuaternionLinear needs both dims divisible by 4*n_heads")
        self.nh = n_heads
        self.in_slots = dim_in // (4 * n_heads)
        self.out_slots = dim_out // (4 * n_heads)
        self.weight = nn.Parameter(torch.empty(n_heads, self.out_slots, self.in_slots, 4))
        k = 1.0 / math.sqrt(4 * self.in_slots)     # real fan-in = in-slots * 4 comps
        nn.init.uniform_(self.weight, -k, k)
        self.bias = nn.Parameter(torch.empty(dim_out)) if bias else None
        if bias:
            nn.init.uniform_(self.bias, -k, k)

    def forward(self, x):  # [..., dim_in]
        shp = x.shape
        xq = x.view(*shp[:-1], self.nh, self.in_slots, 4)
        v = _qbasis(xq)                          # 4 x [..., nh, in_slots, 4]
        y = None
        for p in range(4):
            term = torch.einsum("hoi,...hij->...hoj", self.weight[..., p], v[p])
            y = term if y is None else y + term
        y = y.reshape(*shp[:-1], self.nh * self.out_slots * 4)
        if self.bias is not None:
            y = y + self.bias
        return y


class WaveQuatMix(WaveMix3D):
    """WaveMix3D dispersion path with quaternion-structured pi/po (E2).

    Everything except the two dense Linears is inherited unchanged, so the
    kernel, transfer function, streaming state and ``step()`` are numerically
    identical code to the real arm -- the comparison isolates the
    parameterization of the in/out maps.
    """

    def __init__(self, dim, n_heads, T, H, W, **kw):
        super().__init__(dim, n_heads, T, H, W, **kw)
        self.pi = QuaternionLinear(dim, dim, n_heads)
        self.po = QuaternionLinear(dim, dim, n_heads)


class QuatEmbed(nn.Module):
    """Pure-quaternion 3x3 conv stem (E3): [N,3,H,W] -> [N,dim,H,W].

    Pixels become pure quaternions (0, r, g, b); the kernel is one quaternion
    per 3x3 tap; output = Hamilton product summed over taps, expanded into real
    convs over the 4 basis products (``sum_p W_p ⊗ (e_p x)``), giving
    ``slots * 4 * 9 = 9*dim`` params (vs ``27*dim`` for the real stem).
    """

    def __init__(self, dim, padding=1):
        super().__init__()
        if dim % 4:
            raise ValueError("QuatEmbed needs dim divisible by 4")
        self.dim = dim
        self.slots = dim // 4
        self.weight = nn.Parameter(torch.empty(self.slots, 4, 3, 3))   # [out, p, kh, kw]
        k = 1.0 / math.sqrt(4 * 9)              # real fan-in = 4 components * 9 taps
        nn.init.uniform_(self.weight, -k, k)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.padding = padding

    def forward(self, x):  # [N, 3, H, W]
        N = x.shape[0]
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        o = torch.zeros_like(r)
        # e_p ⊗ (0, r, g, b) components, stacked as channels indexed (p, j):
        # p0: (0, r, g, b)   p1: (-r, 0, -b, g)   p2: (-g, b, 0, -r)   p3: (-b, -g, r, 0)
        V = torch.stack([o, r, g, b,
                         -r, o, -b, g,
                         -g, b, o, -r,
                         -b, -g, r, o], dim=1)                      # [N, 16, H, W]
        # For output component j: y_j = sum_p conv(V[:, p*4+j], weight[:, p]);
        # channels j::4 are exactly (p = 0..3) for fixed j.
        ys = [F.conv2d(V[:, j::4], self.weight, padding=self.padding) for j in range(4)]
        y = torch.stack(ys, dim=2).reshape(N, self.dim, *x.shape[2:])   # (slot, j)
        return y + self.bias.view(1, -1, 1, 1)


class QuatFFN(nn.Module):
    """FFN variant with quaternion-structured projections (optional ``--q-ffn``).

    Hidden width is rounded up to a multiple of ``4 * n_heads`` so both
    projections carry the quaternion slot structure.
    """

    def __init__(self, dim, mult=4.0, n_heads=1):
        super().__init__()
        if dim % (4 * n_heads):
            raise ValueError("QuatFFN needs dim divisible by 4*n_heads")
        unit = 4 * n_heads
        self.hidden = max(unit, (int(dim * mult) // unit) * unit)
        self.fc1 = QuaternionLinear(dim, self.hidden, n_heads)
        self.fc2 = QuaternionLinear(self.hidden, dim, n_heads)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
