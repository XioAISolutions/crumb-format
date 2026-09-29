"""Per-frame rectified-flow head: sample the next frame instead of averaging it (LONG_HORIZON.md phase 3).

An MSE head predicts the conditional MEAN of the next frame. When the future is
uncertain (a kick, a cut, an occluded object's exit side) the mean is a blend of
futures: blobs smear, and feeding the smear back compounds it into the fade/
flatten collapse HealthMonitor flags. A generative head predicts a velocity
field v(x_tau, tau | c) and integrates it from noise, so each rollout commits
to one plausible future.

  target  x1 = next - last        (the per-frame change; zero-mean-ish, like the residual head)
  noise   x0 ~ N(0, sigma^2)
  train   x_tau = (1 - tau) x0 + tau x1,  loss = || v(x_tau, tau, c) - (x1 - x0) ||^2
  sample  x <- x0; repeat n_steps: x <- x + v(x, tau, c) / n_steps;  next = last + x

c is the backbone's causal per-cell feature for that frame -- the same token the
residual head reads -- so the wave/SSM backbone, carried state and streaming are
unchanged; only what is read out of it changes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tau_embed(tau, dim):
    """Sinusoidal embedding of tau in [0,1]: [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / max(half - 1, 1))
    ang = tau[:, None] * freqs[None, :] * 2 * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)


class FlowHead(nn.Module):
    """Small conv velocity net over one frame: in = [x_tau (C) | proj(c) (hid)], tau
    enters by FiLM. Spatial 3x3 convs let the sample stay spatially coherent."""

    def __init__(self, dim, in_ch, hidden=64, tau_dim=32, sigma=1.0):
        super().__init__()
        self.in_ch, self.sigma, self.tau_dim = in_ch, sigma, tau_dim
        self.cproj = nn.Linear(dim, hidden)
        self.c1 = nn.Conv2d(in_ch + hidden, hidden, 3, padding=1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.c3 = nn.Conv2d(hidden, in_ch, 3, padding=1)
        self.film = nn.Linear(tau_dim, 2 * hidden)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def velocity(self, x, tau, c):
        """x [N,C,H,W], tau [N], c [N,H,W,dim] -> v [N,C,H,W]."""
        h = self.cproj(c).permute(0, 3, 1, 2)                        # [N,hid,H,W]
        h = F.gelu(self.c1(torch.cat([x, h], 1)))
        scale, shift = self.film(_tau_embed(tau, self.tau_dim)).chunk(2, -1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.gelu(self.c2(h))
        return self.c3(h)

    def loss(self, x1, c, generator=None, noise=None):
        """Rectified-flow loss for targets x1 [N,C,H,W] given features c [N,H,W,dim].
        ``noise`` = (unit Gaussian [N,C,H,W], tau [N]) replaces the draws."""
        N = x1.shape[0]
        if noise is None:
            noise = (torch.randn(x1.shape, generator=generator, device=x1.device),
                     torch.rand(N, generator=generator, device=x1.device))
        x0 = noise[0].to(device=x1.device, dtype=x1.dtype) * self.sigma
        tau = noise[1].to(device=x1.device, dtype=x1.dtype)
        xt = (1 - tau)[:, None, None, None] * x0 + tau[:, None, None, None] * x1
        return F.mse_loss(self.velocity(xt, tau, c), x1 - x0)

    @torch.no_grad()
    def sample(self, c, shape, n_steps=16, generator=None, noise=None):
        """Euler-integrate from noise: -> x1 sample [N,C,H,W]. ``noise`` (unit
        Gaussian, [N,C,H,W]) replaces the draw from ``generator``."""
        if noise is None:
            noise = torch.randn(shape, generator=generator, device=c.device, dtype=c.dtype)
        x = noise.to(device=c.device, dtype=c.dtype) * self.sigma
        N = shape[0]
        for k in range(n_steps):
            tau = torch.full((N,), k / n_steps, device=c.device, dtype=c.dtype)
            x = x + self.velocity(x, tau, c) / n_steps
        return x
