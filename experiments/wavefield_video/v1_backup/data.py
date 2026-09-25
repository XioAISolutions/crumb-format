"""Procedural moving-ball videos, vectorized (CPU gen -> device).

Returns [B, T+1, 3, H, W] float in [0,1]: frames 0..T-1 = context,
frame T = prediction target."""
import torch
import math

SPEED = 1.15
RADIUS = 1.6


def make_clip_batch(bs, T, H, W, nb=3, device="cpu", seed=None, speed=SPEED):
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    maxs = torch.tensor([W - 1.0, H - 1.0])
    pos = torch.rand(bs, nb, 2, generator=g) * maxs
    vel = (torch.rand(bs, nb, 2, generator=g) * 2 - 1) * speed
    col = torch.rand(bs, nb, 3, generator=g) * 0.75 + 0.25
    ys = torch.arange(H).view(1, 1, H, 1).float()
    xs = torch.arange(W).view(1, 1, 1, W).float()
    frames = []
    for t in range(T + 1):
        dy = ys - pos[:, :, 1][:, :, None, None]
        dx = xs - pos[:, :, 0][:, :, None, None]
        blobs = torch.exp(-(dy * dy + dx * dx) / (2 * RADIUS * RADIUS))
        img = (blobs[:, :, None] * col[:, :, :, None, None]).sum(1)
        frames.append(img.clamp(0, 1))
        pos = pos + vel
        low = pos < 0
        high = pos > maxs
        pos = torch.where(low, -pos, pos)
        pos = torch.where(high, 2 * maxs - pos, pos)
        vel = torch.where(low | high, -vel, vel)
    out = torch.stack(frames, 1)
    return out.to(device)
