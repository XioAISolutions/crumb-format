"""Procedural moving-ball videos, vectorized (CPU gen -> device).

Returns [B, T+1, 3, H, W] float in [0,1]: frames 0..T-1 = context,
frame T = prediction target. With ``return_meta=True`` also returns per-frame
ball positions and colors (used by the color-matched centroid metrics).

Motion modes (all off = v1 exact behavior, identical RNG draws):
  * ``kicks``      -- stochastic velocity kick every ~kick_every frames, breaking
                      determinism so a fixed kernel cannot memorize the trajectory.
  * ``collisions`` -- ball-ball elastic collisions: two overlapping, approaching
                      balls swap velocity vectors (equal-mass head-on exchange).
                      Deterministic, but couples the balls so occlusion/identity
                      tracking becomes a real test.

``RADIUS`` (the Gaussian blob sigma, in pixels) is the single ball-size config
constant, exported for the divergence-horizon threshold in the eval.
"""
import operator

import torch

N_BALLS = 3
SPEED = 1.15
RADIUS = 1.6                 # ball size (Gaussian sigma, px); also the divergence threshold
CONTACT = 2.0 * RADIUS       # center distance at which two balls are "touching"
MOVE_THRESH = 0.05           # per-channel |Δ| above which a pixel counts as "moving"
# Three independent RGB anchors can be linearly unmixed for semantic blob
# detection. Larger scenes repeat these hues; spatial peaks distinguish balls
# of the same hue. Keep these exact values for default-path reproducibility.
PALETTE_ANCHORS = torch.tensor([[1.0, 0.22, 0.22],
                                [0.25, 1.0, 0.30],
                                [0.30, 0.45, 1.0]])


def moving_mask(frames, thresh=MOVE_THRESH):
    """Per-pixel motion mask between consecutive frames.

    ``frames`` is [B, N, 3, H, W]; returns a bool [B, N-1, H, W] where entry
    ``t`` is True wherever ``frames[:, t+1]`` differs from ``frames[:, t]`` by
    more than ``thresh`` in any channel -- i.e. the pixels for which copy-last
    (persistence) would be *wrong*. This is the ground-truth motion mask used to
    stop the loss from rewarding a copy-last collapse (DEEP_DIVE_astra sec 1)."""
    diff = (frames[:, 1:] - frames[:, :-1]).abs().amax(dim=2)      # [B,N-1,H,W]
    return diff > thresh


def make_clip_batch(bs, T, H, W, nb=N_BALLS, device="cpu", seed=None, speed=SPEED,
                    radius=RADIUS, kicks=False, kick_every=20, kick_scale=None,
                    collisions=False, return_meta=False,
                    return_moving=False, move_thresh=MOVE_THRESH):
    """Generate ``nb`` balls (``train_compare.py --n-balls``), defaulting to 3.

    Changing the count leaves the collision rule unchanged: every unordered
    pair is checked, including pairs sharing a palette anchor. The default
    three-ball path retains the original RNG draws and floating-point work.
    """
    try:
        count = operator.index(nb)
    except TypeError as exc:
        raise ValueError("nb must be a positive integer") from exc
    if isinstance(nb, bool) or count < 1:
        raise ValueError("nb must be a positive integer")
    nb = count
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    if kick_scale is None:
        kick_scale = speed
    # ``radius``/``speed`` are grid-cell geometry (DEEP_DIVE_2 Astra exp 2): pass
    # radius=2*grid-ratio to keep the ball's *relative* size/speed fixed as the
    # canvas grows. CONTACT (collision trigger) scales with radius so pairs still
    # touch at 2r regardless of the override.
    contact = 2.0 * radius
    maxs = torch.tensor([W - 1.0, H - 1.0])
    pos = torch.rand(bs, nb, 2, generator=g) * maxs
    vel = (torch.rand(bs, nb, 2, generator=g) * 2 - 1) * speed
    # Distinct anchor colors (well-separated hues) so the color-matched centroid
    # metric can actually separate balls; random jitter for variety. Random colors
    # caused ~5px centroid bleed between near-parallel ball colors (2026-09-25).
    anchors = PALETTE_ANCHORS
    if nb <= 3:
        idx = torch.rand(bs, nb, generator=g).argsort(1)
        col = anchors[idx] * (0.85 + 0.15 * torch.rand(bs, nb, 1, generator=g))
    else:
        # Balanced repeated anchors avoid nearly parallel random RGB vectors
        # while allowing arbitrary counts. Identity remains the ball index;
        # equal colors do not affect collision physics.
        idx = torch.rand(bs, nb, generator=g).argsort(1) % len(anchors)
        col = anchors[idx] * (0.85 + 0.15 * torch.rand(bs, nb, 1, generator=g))
    ys = torch.arange(H).view(1, 1, H, 1).float()
    xs = torch.arange(W).view(1, 1, 1, W).float()
    frames, positions = [], []
    for t in range(T + 1):
        dy = ys - pos[:, :, 1][:, :, None, None]
        dx = xs - pos[:, :, 0][:, :, None, None]
        blobs = torch.exp(-(dy * dy + dx * dx) / (2 * radius * radius))
        img = (blobs[:, :, None] * col[:, :, :, None, None]).sum(1)
        frames.append(img.clamp(0, 1))
        positions.append(pos.clone())
        pos = pos + vel
        if kicks and t > 0 and t % kick_every == 0:
            vel = vel + torch.randn(bs, nb, 2, generator=g) * kick_scale
        if collisions and nb > 1:
            vel = _resolve_collisions(pos, vel, contact)
        low = pos < 0
        high = pos > maxs
        pos = torch.where(low, -pos, pos)
        pos = torch.where(high, 2 * maxs - pos, pos)
        vel = torch.where(low | high, -vel, vel)
    out = torch.stack(frames, 1).to(device)
    # Optional returns. Order is fixed: (out[, meta][, moving]) so callers can
    # request either or both without ambiguity (moving is always last).
    extra = []
    if return_meta:
        extra.append({"pos": torch.stack(positions, 1).to(device),  # [B, T+1, nb, 2] (x,y)
                      "col": col.to(device)})                        # [B, nb, 3]
    if return_moving:
        extra.append(moving_mask(out, move_thresh))                  # [B, T, H, W] bool
    if extra:
        return (out, *extra)
    return out


def _resolve_collisions(pos, vel, contact=CONTACT):
    """Equal-mass elastic ball-ball collisions (swap velocities on overlap).

    For each unordered pair (i,j): if the centers are within ``contact`` *and* the
    balls are approaching (relative velocity has a negative component along the
    center-to-center line), swap their velocity vectors. The approach test
    prevents overlapping balls from swapping every frame (sticking). ``contact``
    defaults to the module CONTACT but scales with a ``radius`` override so the
    trigger distance stays 2r under the geometry control."""
    nb = pos.shape[1]
    vel = vel.clone()
    for i in range(nb):
        for j in range(i + 1, nb):
            rp = pos[:, j] - pos[:, i]                      # [B,2]
            rv = vel[:, j] - vel[:, i]
            dist = rp.norm(dim=-1)                          # [B]
            approaching = (rv * rp).sum(-1) < 0
            hit = (dist < contact) & approaching           # [B]
            if hit.any():
                vi, vj = vel[:, i].clone(), vel[:, j].clone()
                vel[:, i] = torch.where(hit[:, None], vj, vi)
                vel[:, j] = torch.where(hit[:, None], vi, vj)
    return vel
