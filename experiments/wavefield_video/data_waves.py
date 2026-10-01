"""Periodic 2-D PDE clips with the calling convention of :mod:`data`.

``make_clip_batch`` returns float32 [B, T+1, 3, H, W] in [0, 1]. Channels
encode [u, partial_t u, Laplacian u] as ``0.5 + raw / (2 * scale)``.
Scales are per-sample, per-channel Fourier amplitude bounds, fixed from the
initial conditions (never from future frames). Zero derivatives encode as 0.5.
Coordinates span [0, 2*pi) in both directions; time is t * time_step.

Fields:
* wave: u_tt = speed**2 * Laplacian(u), exact spectral oscillator evolution.
* advection: u_t + velocity . grad(u) = 0, exact spectral translation.
* vortex: drifting Taylor--Green vorticity with viscosity. Initial vorticity
  lies on one Laplacian eigenspace, so its induced incompressible velocity
  has zero self-advection. Translation and viscous decay therefore solve the
  full 2-D vorticity equation exactly. This is not general turbulent flow.

Generation and FFTs run on CPU in float64, followed by one transfer to device.
Explicit seeds use a local RNG and leave the global torch RNG untouched.
The legacy ball-motion options (kicks, kick_every, kick_scale, collisions)
are accepted but have no effect on these deterministic PDEs. ``nb`` controls
the number of random Fourier modes for wave/advection; vortex uses one cell
lattice. Metadata contains PDE parameters/scales, not ball positions/colors;
the ball centroid metrics in train_compare.py do not apply to these channels.
"""

import math
import operator

import torch

SPEED = 1.15
RADIUS = 1.6  # Legacy import compatibility only; not a PDE length scale.
CONTACT = 2.0 * RADIUS
MOVE_THRESH = 0.05
FIELDS = ("wave", "advection", "vortex")


def moving_mask(frames, thresh=MOVE_THRESH):
    """Match data.moving_mask: any-channel change > thresh, [B, N-1, H, W]."""
    return (frames[:, 1:] - frames[:, :-1]).abs().amax(dim=2) > thresh


def _integer(name, value, minimum):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _nonnegative(name, value, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _frequencies(H, W, dtype):
    ky = torch.fft.fftfreq(H, d=1.0 / H, dtype=dtype).view(H, 1)
    kx = torch.fft.fftfreq(W, d=1.0 / W, dtype=dtype).view(1, W)
    return kx, ky, kx.square() + ky.square()


def _velocity(value, bs, dtype, speed):
    if value is None:
        value = (speed, 0.0)
    vel = torch.as_tensor(value, dtype=dtype, device="cpu")
    if vel.shape == (2,):
        vel = vel.expand(bs, 2)
    if vel.shape != (bs, 2) or not torch.isfinite(vel).all():
        raise ValueError("velocity must be finite [2] or [B, 2], ordered (vx, vy)")
    return vel


def spectral_evolve(initial, T, *, field="wave", time_step=0.15, speed=SPEED,
                    velocity=None, initial_dt=None, viscosity=0.02):
    """Return unscaled CPU [B, T+1, 3, H, W] (field, time derivative, Laplacian).

    ``initial`` is a real floating CPU [B,H,W] tensor; float32/64 are retained.
    Wave initial_dt defaults to zero; its zero Fourier mode evolves linearly.
    Advection/vortex derive the time derivative from the PDE. Vortex evolves
    transported diffusion, which is also Navier--Stokes vorticity evolution
    for the single-eigenvalue initial conditions used by make_clip_batch.
    General inputs to that branch solve transported diffusion only.
    Use resolved modes below the Nyquist frequency: fractional translations
    of a sampled even-grid Nyquist mode have an ambiguous real interpolation.
    The clip generator only uses modes of magnitude <= 3 in each direction.
    """
    T = _integer("T", T, 0)
    time_step = _nonnegative("time_step", time_step, positive=True)
    speed = _nonnegative("speed", speed)
    viscosity = _nonnegative("viscosity", viscosity)
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}")
    if (not isinstance(initial, torch.Tensor) or initial.ndim != 3
            or initial.device.type != "cpu"
            or initial.dtype not in (torch.float32, torch.float64)
            or min(initial.shape) < 1 or not torch.isfinite(initial).all()):
        raise ValueError("initial must be a finite float32/64 CPU [B,H,W] tensor")
    bs, H, W = initial.shape
    kx, ky, k2 = _frequencies(H, W, initial.dtype)
    u0 = torch.fft.fft2(initial)
    times = torch.arange(T + 1, dtype=initial.dtype).view(1, -1, 1, 1) * time_step
    if field == "wave":
        if initial_dt is None:
            initial_dt = torch.zeros_like(initial)
        if (not isinstance(initial_dt, torch.Tensor)
                or initial_dt.shape != initial.shape or initial_dt.dtype != initial.dtype
                or initial_dt.device != initial.device or not torch.isfinite(initial_dt).all()):
            raise ValueError("initial_dt must match initial shape, dtype and device and be finite")
        v0 = torch.fft.fft2(initial_dt)[:, None]
        omega = speed * k2.sqrt()
        angle = omega * times
        # sinc gives sin(omega*t)/omega including the omega=0 limit t.
        sin_over_omega = times * torch.sinc(angle / math.pi)
        uhat = u0[:, None] * angle.cos() + v0 * sin_over_omega
        dhat = -u0[:, None] * omega * angle.sin() + v0 * angle.cos()
    else:
        if initial_dt is not None:
            raise ValueError("initial_dt is only used by the wave equation")
        vel = _velocity(velocity, bs, initial.dtype, speed)
        rate = -1j * (vel[:, 0, None, None] * kx + vel[:, 1, None, None] * ky)
        if field == "vortex":
            rate = rate - viscosity * k2
        uhat = u0[:, None] * torch.exp(rate[:, None] * times)
        dhat = rate[:, None] * uhat
    raw = torch.stack((torch.fft.ifft2(uhat).real,
                       torch.fft.ifft2(dhat).real,
                       torch.fft.ifft2(-k2 * uhat).real), dim=2)
    return raw


def make_clip_batch(bs, T, H, W, nb=3, device="cpu", seed=None, speed=SPEED,
                    kicks=False, kick_every=20, kick_scale=None,
                    collisions=False, return_meta=False,
                    return_moving=False, move_thresh=MOVE_THRESH, *,
                    field="wave", time_step=0.15, velocity=None, viscosity=0.02):
    """Generate a seeded PDE clip, preserving data.py's argument/return order.

    H and W may independently be 16, 32, or 64; T may be 0 through 512.
    Optional returns are ``(frames[, meta][, moving])``. Decode physical
    channels with ``(2*frames-1) * meta['scale'][:,None,:,None,None]``.
    Meta velocity is (vx,vy); it sets drift for advection/vortex and selects
    a propagation branch for wave modes (wave speed is still ``speed``).
    """
    bs, T = _integer("bs", bs, 1), _integer("T", T, 0)
    H, W, nb = _integer("H", H, 1), _integer("W", W, 1), _integer("nb", nb, 1)
    if H not in (16, 32, 64) or W not in (16, 32, 64):
        raise ValueError("H and W must each be 16, 32, or 64")
    if T > 64:
        raise ValueError("T must be <= 512")
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}")
    speed = _nonnegative("speed", speed)
    viscosity = _nonnegative("viscosity", viscosity)
    time_step = _nonnegative("time_step", time_step, positive=True)
    move_thresh = _nonnegative("move_thresh", move_thresh)
    g = torch.Generator(device="cpu")
    if seed is None:
        g.seed()
    else:
        g.manual_seed(operator.index(seed))
    actual_seed = g.initial_seed()
    dtype = torch.float64
    y = torch.arange(H, dtype=dtype).view(1, H, 1) * (2 * math.pi / H)
    x = torch.arange(W, dtype=dtype).view(1, 1, W) * (2 * math.pi / W)
    direction = 2 * math.pi * torch.rand(bs, generator=g, dtype=dtype)
    if velocity is None:
        vel = speed * torch.stack((direction.cos(), direction.sin()), dim=1)
    else:
        vel = _velocity(velocity, bs, dtype, speed)
    if field == "vortex":
        modes = torch.randint(1, 4, (bs, 2), generator=g)
        phase = 2 * math.pi * torch.rand(bs, 2, generator=g, dtype=dtype)
        initial = ((modes[:, 0, None, None] * x + phase[:, 0, None, None]).cos()
                   * (modes[:, 1, None, None] * y + phase[:, 1, None, None]).cos())
    else:
        initial = torch.zeros(bs, H, W, dtype=dtype)
        modes = torch.randint(-3, 4, (bs, nb, 2), generator=g)
        # Avoid a DC-only sampled mode.
        modes[:, :, 0] = torch.where((modes == 0).all(dim=-1), 1, modes[:, :, 0])
        phase = 2 * math.pi * torch.rand(bs, nb, generator=g, dtype=dtype)
        amplitudes = 0.5 + torch.rand(bs, nb, generator=g, dtype=dtype)
        amplitudes = amplitudes / amplitudes.sum(dim=1, keepdim=True)
        for j in range(nb):
            angle = (modes[:, j, 0, None, None] * x
                     + modes[:, j, 1, None, None] * y + phase[:, j, None, None])
            initial += amplitudes[:, j, None, None] * angle.cos()
    kx, ky, k2 = _frequencies(H, W, dtype)
    # Remove roundoff-only high frequencies and DC, preserving a real field.
    support = (kx.abs() <= 3) & (ky.abs() <= 3) & (k2 > 0)
    uhat = torch.fft.fft2(initial) * support
    initial = torch.fft.ifft2(uhat).real
    initial_dt = None
    if field == "wave":
        omega = speed * k2.sqrt()
        projection = vel[:, 0, None, None] * kx + vel[:, 1, None, None] * ky
        # A deterministic odd branch even for zero/orthogonal drift vectors.
        fallback = torch.where(kx != 0, kx.sign(), ky.sign())
        branch = torch.where(projection != 0, projection.sign(), fallback)
        dhat = -1j * omega * branch * uhat
        initial_dt = torch.fft.ifft2(dhat).real
        # The sampled wave is a superposition of traveling modes, each with
        # constant amplitude; these bounds are exact Fourier L1 bounds.
        bounds = (uhat.abs(), omega * uhat.abs(), k2 * uhat.abs())
    else:
        rate = -1j * (vel[:, 0, None, None] * kx + vel[:, 1, None, None] * ky)
        if field == "vortex":
            rate = rate - viscosity * k2
        bounds = (uhat.abs(), rate.abs() * uhat.abs(), k2 * uhat.abs())
    scale = torch.stack([b.sum(dim=(-2, -1)) / (H * W) for b in bounds], dim=1)
    scale = scale.clamp_min(torch.finfo(dtype).eps)
    raw = spectral_evolve(initial, T, field=field, time_step=time_step,
                          speed=speed, velocity=vel, initial_dt=initial_dt,
                          viscosity=viscosity)
    out = (0.5 + raw / (2 * scale[:, None, :, None, None])).clamp(0, 1).float().to(device)
    extras = []
    if return_meta:
        extras.append({"field": field, "seed": actual_seed, "time_step": time_step,
                       "times": (torch.arange(T + 1, dtype=dtype) * time_step).to(device),
                       "domain_length": 2 * math.pi, "speed": speed,
                       "velocity": vel.to(device), "viscosity": viscosity,
                       "scale": scale.to(device), "modes": modes.to(device)})
    if return_moving:
        extras.append(moving_mask(out, move_thresh))
    return (out, *extras) if extras else out
