"""Long-memory occlusion scenario -- the decisive WaveMemory probe (R14).

See HARDENING_astra.md "The experiment that matters most". A target ball is
visible, then hidden behind a static occluder for a long stretch, then emerges.
Its trajectory *continues deterministically* behind the occluder (wall bounces
included), so the exit position/direction at emergence is fully determined by the
pre-occlusion state. A model can only predict the emergence if it *remembered*
the hidden object's dynamical state across a gap far longer than any local
attention window -- which is exactly what a persistent recurrent state buys you.

Scene (defaults, 64x64):
  * ``nb`` balls; ball ``TARGET_IDX`` (0) is the *target* and carries a unique
    identity color (pure red anchor). The other balls roam the whole canvas and
    provide the always-on motion signal (so freezing is never a good solution).
  * The target is confined to a *chamber* rectangle: it bounces off the chamber
    walls, so it is ALWAYS inside the chamber region.
  * The occluder is a static rectangle covering the chamber (plus a radius pad).
    It is composited OPAQUELY over the frame only during ``[occ_start, occ_end)``.
    Because the target never leaves the chamber and the occluder covers it, the
    target is hidden EXACTLY during that window and visible otherwise -- a crisp,
    checkable invariant (see test_occlusion_r14.py).

Determinism: same RNG discipline as data.py (a local CPU ``Generator`` seeded by
``seed``; never touches global RNG). Draw order is pos, vel, color-jitter.

API mirrors data.py ``make_clip_batch`` -> [B, T+1, 3, H, W] in [0,1], with the
same optional ``return_meta`` / ``return_moving`` extras (meta adds the occlusion
bookkeeping the eval needs: target index, occluder rect, window, per-frame
velocity, and a target-visibility mask).
"""
import math
import operator

import torch

from data import PALETTE_ANCHORS, MOVE_THRESH, RADIUS, SPEED, moving_mask, _resolve_collisions

N_BALLS = 8
TARGET_IDX = 0
OCC_START = 64               # target hidden on frame indices [OCC_START, OCC_END)
OCC_END = 320
# Chamber the target is confined to, as (fx0, fy0, fx1, fy1) fractions of (W-1,H-1).
# Central band, wide in x (so horizontal bounces count) and shorter in y.
CHAMBER_FRAC = (0.28, 0.36, 0.72, 0.64)
OCCLUDER_RGB = torch.tensor([0.15, 0.15, 0.15])   # opaque neutral gray, not a palette hue


def occluder_rect(H, W, radius=RADIUS, chamber=CHAMBER_FRAC):
    """Integer inclusive pixel rect (x0, y0, x1, y1) covering the chamber + a
    ``3*radius`` pad, clamped to the canvas. The pad guarantees the target's whole
    Gaussian (not just its center) is overwritten, so occlusion is total."""
    fx0, fy0, fx1, fy1 = chamber
    pad = 3.0 * radius + 1.0
    x0 = max(0, int(math.floor(fx0 * (W - 1) - pad)))
    y0 = max(0, int(math.floor(fy0 * (H - 1) - pad)))
    x1 = min(W - 1, int(math.ceil(fx1 * (W - 1) + pad)))
    y1 = min(H - 1, int(math.ceil(fy1 * (H - 1) + pad)))
    return x0, y0, x1, y1


def draw_occluder(img, rect, rgb=OCCLUDER_RGB):
    """Opaquely overwrite ``rect`` (x0,y0,x1,y1 inclusive) of ``img`` [B,3,H,W]
    with ``rgb``. Returns a new tensor (does not mutate ``img``)."""
    x0, y0, x1, y1 = rect
    out = img.clone()
    out[:, :, y0:y1 + 1, x0:x1 + 1] = rgb.to(out.device, out.dtype)[None, :, None, None]
    return out


def occlude_observation(frame, t, rect, occ_start, occ_end, rgb=OCCLUDER_RGB):
    """Environment occlusion of an *observation*: if absolute frame index ``t`` is
    inside the window, paint the occluder over ``frame`` [B,3,H,W]; else pass it
    through unchanged. Used by the rollout eval to hide the target in the frames
    fed back to the model, so only persistent state (not the pixel input) can
    carry the hidden object across the gap."""
    if occ_start <= t < occ_end:
        return draw_occluder(frame, rect, rgb)
    return frame


def visibility_mask(n_frames, occ_start, occ_end):
    """Bool [n_frames]: True where the target is visible (outside the window)."""
    t = torch.arange(n_frames)
    return ~((t >= occ_start) & (t < occ_end))


def make_clip_batch(bs, T, H, W, nb=N_BALLS, device="cpu", seed=None, speed=SPEED,
                    radius=RADIUS, occ_start=None, occ_end=None, chamber=CHAMBER_FRAC,
                    target_idx=TARGET_IDX, collisions=False, return_meta=False,
                    return_moving=False, move_thresh=MOVE_THRESH):
    """Generate ``bs`` occlusion clips of ``T+1`` frames.

    ``occ_start``/``occ_end`` default to the module constants; pass smaller values
    for CPU smoke tests. ``collisions`` (optional) resolves elastic ball-ball
    collisions among the *non-target* balls only -- the target stays confined to
    its chamber so its determinism (and thus the exit-direction ground truth) is
    never perturbed by a collision.
    """
    try:
        count = operator.index(nb)
    except TypeError as exc:
        raise ValueError("nb must be a positive integer") from exc
    if isinstance(nb, bool) or count < 1:
        raise ValueError("nb must be a positive integer")
    nb = count
    if not (0 <= target_idx < nb):
        raise ValueError("target_idx must index a ball")
    occ_start = OCC_START if occ_start is None else int(occ_start)
    occ_end = OCC_END if occ_end is None else int(occ_end)
    if not (0 <= occ_start <= occ_end):
        raise ValueError("require 0 <= occ_start <= occ_end")

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    maxs = torch.tensor([W - 1.0, H - 1.0])
    fx0, fy0, fx1, fy1 = chamber
    clo = torch.tensor([fx0 * (W - 1), fy0 * (H - 1)])
    chi = torch.tensor([fx1 * (W - 1), fy1 * (H - 1)])
    # Per-ball reflection bounds: canvas for everyone, the chamber for the target.
    lo = torch.zeros(nb, 2)
    hi = maxs.expand(nb, 2).clone()
    lo[target_idx] = clo
    hi[target_idx] = chi
    rect = occluder_rect(H, W, radius=radius, chamber=chamber)
    contact = 2.0 * radius

    # Positions initialized inside each ball's own bounds; the target starts in
    # the chamber and can never leave it.
    pos = torch.rand(bs, nb, 2, generator=g) * (hi - lo)[None] + lo[None]
    vel = (torch.rand(bs, nb, 2, generator=g) * 2 - 1) * speed

    # Colors: target = pure red identity anchor (no jitter, so its unique hue is
    # unambiguous to the color-matched centroid tracker). Others cycle the green/
    # blue anchors with brightness jitter for visual variety.
    jitter = 0.85 + 0.15 * torch.rand(bs, nb, 1, generator=g)
    col = PALETTE_ANCHORS[0].expand(bs, nb, 3).clone()
    others = [i for i in range(nb) if i != target_idx]
    other_anchors = PALETTE_ANCHORS[1:]
    for k, i in enumerate(others):
        col[:, i] = other_anchors[k % len(other_anchors)]
    col = col * jitter
    col[:, target_idx] = PALETTE_ANCHORS[0]                      # keep target hue crisp

    ys = torch.arange(H).view(1, 1, H, 1).float()
    xs = torch.arange(W).view(1, 1, 1, W).float()
    frames, positions, vels = [], [], []
    for t in range(T + 1):
        dy = ys - pos[:, :, 1][:, :, None, None]
        dx = xs - pos[:, :, 0][:, :, None, None]
        blobs = torch.exp(-(dy * dy + dx * dx) / (2 * radius * radius))
        img = (blobs[:, :, None] * col[:, :, :, None, None]).sum(1).clamp(0, 1)
        if occ_start <= t < occ_end:
            img = draw_occluder(img, rect)                        # opaque, drawn last
        frames.append(img)
        positions.append(pos.clone())
        vels.append(vel.clone())
        pos = pos + vel
        if collisions and len(others) > 1:
            # Resolve collisions among non-target balls only. _resolve_collisions
            # is pairwise over all balls, so isolate the non-target sub-system.
            sub = _resolve_collisions(pos[:, others], vel[:, others], contact)
            vel = vel.clone()
            vel[:, others] = sub
        low = pos < lo[None]
        high = pos > hi[None]
        pos = torch.where(low, 2 * lo[None] - pos, pos)
        pos = torch.where(high, 2 * hi[None] - pos, pos)
        vel = torch.where(low | high, -vel, vel)

    out = torch.stack(frames, 1).to(device)
    extra = []
    if return_meta:
        extra.append({
            "pos": torch.stack(positions, 1).to(device),         # [B, T+1, nb, 2] (x,y)
            "vel": torch.stack(vels, 1).to(device),              # [B, T+1, nb, 2]
            "col": col.to(device),                               # [B, nb, 3]
            "target_idx": target_idx,
            "occluder": rect,                                    # (x0,y0,x1,y1) inclusive
            "occ_start": occ_start,
            "occ_end": occ_end,
            "visible": visibility_mask(T + 1, occ_start, occ_end).to(device),  # [T+1] bool
        })
    if return_moving:
        extra.append(moving_mask(out, move_thresh))              # [B, T, H, W] bool
    if extra:
        return (out, *extra)
    return out
