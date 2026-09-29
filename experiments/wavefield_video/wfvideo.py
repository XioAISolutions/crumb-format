"""Wave-field video predictor + transformer / SSM baselines (identical scaffolds).

Mixing is 3D over (time, y, x), applied via FFT. Two wave kernels:

  * ``separable``  (v1): damped-cosine kernels K_t(t)*K_y(y)*K_x(x). A *standing*
                   field -- time and space factorize, so it cannot represent a
                   translating pattern (whose spectrum obeys omega_t = v . k).
  * ``dispersion`` (v2): each spatial-frequency cell (kx,ky) gets its own temporal
                   pole  lambda(kx,ky) = exp(-alpha(kx,ky)) * exp(i*Omega(kx,ky)),
                   Omega = vx*kx + vy*ky + beta*|k|, with 2-4 damped-oscillator
                   modes per head. A *propagating* field: heads learn direction/
                   speed. Implemented as a per-cell temporal transfer function
                   (frequency-domain multiplier over the rfft/fft time axis) --
                   the transfer of the causal recurrence
                       z_t(k) = lambda(k) z_{t-1}(k) + B x_t(k),  y_t = Re(C z_t),
                   so the SAME operator streams O(1)-in-T later (see ssm_lite.py).

Fairness (v2): ALL arms share ONE compact factorized (t,y,x) positional encoding
at the input; attention's giant per-layer [1,N,dim] table is gone. v1 numerics
are still reproduced by the ``separable`` wave arm with every new flag off.
"""
import math
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class FFN(nn.Module):
    def __init__(self, dim, mult=4.0, bias=True):
        super().__init__()
        h = max(1, int(round(dim * mult)))
        self.fc1 = nn.Linear(dim, h, bias=bias)
        self.fc2 = nn.Linear(h, dim, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class FactorizedPosEmb(nn.Module):
    """Shared learned separable (t, y, x) positional embeddings, summed.

    ONE compact table per axis, broadcast-added at the model input for every
    arm. Parameter count is (T + H + W)*dim instead of the flat (T*H*W)*dim
    table the v1 attention arm carried per layer -- so no arm can win just by
    memorizing absolute position in a huge dedicated table."""

    def __init__(self, dim, T, H, W, time=True, space=True):
        super().__init__()
        self.space = space
        self.T, self.H, self.W = T, H, W
        # time=False drops the absolute temporal table (LONG_HORIZON.md phase 1):
        # a length-T table cannot describe frame 5,000, and stream_step had to clamp
        # it to pt[T-1]; without it training, chunked training and streaming see
        # the same inputs and the causal recurrence alone carries order.
        self.pt = nn.Parameter(torch.zeros(T, dim)) if time else None
        # space=False (clean_write): no per-cell constant added to every frame --
        # a shared input projection cannot cancel it, so long poles integrate it.
        self.py = nn.Parameter(torch.zeros(H, dim)) if space else None
        self.px = nn.Parameter(torch.zeros(W, dim)) if space else None
        for p in (self.pt, self.py, self.px):
            if p is not None:
                nn.init.normal_(p, std=0.02)

    def spatial(self):
        """[H*W, dim] per-cell embedding (zeros without a spatial table)."""
        if self.py is None:
            return None
        return (self.py[:, None, :] + self.px[None, :, :]).reshape(self.H * self.W, -1)

    def forward(self, x):  # x: [B, N, dim], N = T*H*W in (t, y, x) order
        B, N, D = x.shape
        if self.py is None:
            if self.pt is None:
                return x
            return x + self.pt[:, None, :].expand(-1, self.H * self.W, -1).reshape(1, N, D)
        if self.pt is not None:                  # original summation order (byte-identical)
            pos = (self.pt[:, None, None, :]
                   + self.py[None, :, None, :]
                   + self.px[None, None, :, :])      # [T, H, W, dim]
        else:
            pos = (self.py[None, :, None, :] + self.px[None, None, :, :]).expand(
                N // (self.H * self.W), -1, -1, -1)
        return x + pos.reshape(1, N, D)


def _damped_cos(a, w, p, d):
    """Per-head damped cosine e^{-|a||d|} cos(w d + p). a,w,p: [nh]; d: [L]."""
    env = torch.exp(-a.abs()[:, None] * d.abs()[None, :])
    return env * torch.cos(w[:, None] * d[None, :] + p[:, None])


class WaveMix3D(nn.Module):
    """Wave-field spatiotemporal mixer applied per head via FFT.

    kernel_version: "separable" (v1 standing field) | "dispersion" (v2 propagating).
    causal_time:    zero the future-reaching temporal taps (separable path).
    linear_pad:     zero-padded (linear) convolution over time AND space so the
                    FFT does not treat the clip as a torus (default ON for v2).
    gate:           Hyena-style content gate  y = po(g * wave(pi(x))),
                    g = sigmoid(MLP(pooled per-head features)), broadcast back.
    local_fuse:     parallel 3x3 depthwise conv per head, added to the wave output
                    before po (global field carries transport, local path edges).

    All flags off with kernel_version="separable" => exact v1 rfftn code path.
    """

    def __init__(self, dim, n_heads, T, H, W, kernel_version="separable",
                 causal_time=False, linear_pad=False, gate=False, local_fuse=False,
                 n_modes=3, pole_param="softplus", hl_min=2.0, hl_max=4096.0,
                 write_gate=False, pi_bias=True):
        super().__init__()
        self.nh = n_heads
        self.dh = dim // n_heads
        self.T, self.H, self.W = T, H, W
        self.kernel_version = kernel_version
        self.causal_time = causal_time
        self.linear_pad = linear_pad
        self.gate = gate
        self.local_fuse = local_fuse
        self.n_modes = n_modes
        if pole_param not in ("softplus", "halflife"):
            raise ValueError(f"unknown pole_param {pole_param!r} (want softplus | halflife)")
        if pole_param != "softplus" and kernel_version != "dispersion":
            raise ValueError("pole_param='halflife' needs kernel_version='dispersion'")
        if not (0.0 < hl_min < hl_max):
            raise ValueError(f"need 0 < hl_min < hl_max (got {hl_min}, {hl_max})")
        self.pole_param = pole_param
        self.hl_min, self.hl_max = float(hl_min), float(hl_max)
        # pi_bias=False (pi and po): a blank input writes exactly nothing into the
        # state and adds nothing back to the residual stream. With a
        # bias, every frame writes the same constant and long poles integrate it
        # into a background that drowned a frame-0 memory ~1000:1 by frame 128
        # (LONG_HORIZON.md 8.4).
        self.pi = nn.Linear(dim, dim, bias=pi_bias)
        self.po = nn.Linear(dim, dim, bias=pi_bias)
        # write_gate: per-token, per-head sigmoid on what enters the wave, so the
        # model can learn to write events and skip static content. It scales the
        # input before the (linear, time-invariant) recurrence, so FFT training,
        # carried-state chunks and step() stay exactly equivalent.
        self.write_gate = write_gate
        if write_gate:
            self.wg = nn.Linear(dim, n_heads)
            nn.init.zeros_(self.wg.weight)
            nn.init.constant_(self.wg.bias, 2.0)           # starts open (~0.88)

        if kernel_version == "separable":
            self.a_t = nn.Parameter(torch.rand(n_heads) * 0.05 + 0.01)
            self.a_s = nn.Parameter(torch.rand(n_heads) * 0.05 + 0.01)
            self.w_t = nn.Parameter(torch.rand(n_heads) * 2.0)
            self.w_s = nn.Parameter(torch.rand(n_heads) * 2.0)
            self.p_t = nn.Parameter(torch.zeros(n_heads))
            self.p_s = nn.Parameter(torch.zeros(n_heads))
            self.register_buffer("dt", (torch.arange(T) - T // 2).float(), persistent=False)
            self.register_buffer("dy", (torch.arange(H) - H // 2).float(), persistent=False)
            self.register_buffer("dx", (torch.arange(W) - W // 2).float(), persistent=False)
        elif kernel_version == "dispersion":
            # Per (head, mode): damping alpha = softplus(a0 + a1*|k|) (>0 => |lambda|<1,
            # unconditionally stable); phase Omega = vx*kx + vy*ky + beta*|k|; input/
            # output gains B, C. |k| grows with spatial frequency so a1>0 damps fast
            # (small-scale) structure more, like physical viscosity.
            sh = (n_heads, n_modes)
            if pole_param == "softplus":
                self.a0 = nn.Parameter(torch.full(sh, 0.5))
                self.a1 = nn.Parameter(torch.full(sh, 0.1))
            else:
                # LONG_HORIZON.md: parameterize memory as a half-life in FRAMES,
                #   hl = hl_min * (hl_max/hl_min)^sigmoid(hl_raw),  alpha_DC = ln2/hl,
                # so |lam| < 1 always (stable) yet modes can hold ~hl_max frames. The
                # softplus default starts at |lam| = e^-0.97 = 0.38 (half-life ~0.7
                # frame) -- a ripple that is gone before the next frame. Init spreads
                # the (head, mode) half-lives log-uniformly over [hl_min, hl_max] so
                # every timescale from "blink" to "minutes" has a mode from step 0.
                # Viscosity visc*|k| still damps fine detail faster than layout.
                n = n_heads * n_modes
                u = (torch.arange(n, dtype=torch.float32) + 0.5) / n
                self.hl_raw = nn.Parameter(torch.logit(u).view(n_modes, n_heads).t().contiguous())
                self.visc = nn.Parameter(torch.full(sh, -4.0))   # softplus(-4) ~ 0.018/|k|
            self.vx = nn.Parameter(torch.zeros(sh))
            self.vy = nn.Parameter(torch.zeros(sh))
            self.beta = nn.Parameter(torch.zeros(sh))
            self.Bg = nn.Parameter(torch.ones(sh))
            self.Cg = nn.Parameter(torch.randn(sh) * (n_modes ** -0.5))
        else:
            raise ValueError(f"unknown kernel_version {kernel_version!r}")

        if local_fuse:
            # Depthwise 3x3 per (head, channel); grouped conv over the spatial plane.
            self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        if gate:
            self.gate_fc = nn.Linear(self.dh, self.dh)

    # ---- separable (v1) kernels ---------------------------------------------
    def _kernels_1d(self):
        kt = _damped_cos(self.a_t, self.w_t, self.p_t, self.dt)   # [nh, T]
        ky = _damped_cos(self.a_s, self.w_s, self.p_s, self.dy)   # [nh, H]
        kx = _damped_cos(self.a_s, self.w_s, self.p_s, self.dx)   # [nh, W]
        if self.causal_time:
            kt = kt * (self.dt >= 0).to(kt.dtype)[None, :]        # drop future (dt<0) taps
        return kt, ky, kx

    def kernels(self):
        """Full centered 3D kernel (v1 circular-FFT path)."""
        kt, ky, kx = self._kernels_1d()
        k = kt[:, :, None, None] * ky[:, None, :, None] * kx[:, None, None, :]
        return torch.fft.ifftshift(k, dim=(1, 2, 3))

    def _lin_conv_axis(self, h, k1d, axis, n):
        """1D "same" conv of h along ``axis`` with the centered per-head kernel
        k1d [nh, n] (sampled at d = arange(n)-n//2).

          linear_pad on  -> L = 2n-1, natural kernel placement + center crop:
                            true zero-padded (non-circular) convolution, so a
                            symmetric kernel stays symmetric and a causal kernel
                            (dt<0 taps zeroed) reads only the past -- no torus wrap.
          linear_pad off -> L = n, ifftshift placement + [0,n) crop: the v1
                            circular convolution (wraps frame 0 <-> T-1)."""
        if self.linear_pad:
            L = 2 * n - 1
            kb = h.new_zeros(self.nh, L)
            kb[:, :n] = k1d.to(kb.dtype)                  # k1d[j] at lag (j - n//2)
            start = n // 2                                # center crop of the full conv
        else:
            L = n
            d = torch.arange(n, device=k1d.device) - n // 2
            kb = h.new_zeros(self.nh, L)
            kb[:, d.long() % L] = k1d.to(kb.dtype)        # ifftshift: zero-lag -> index 0
            start = 0
        shp = [1, self.nh] + [1] * (h.dim() - 2)
        shp[axis] = L // 2 + 1
        Kf = torch.fft.rfft(kb, n=L, dim=1).reshape(shp)
        Hf = torch.fft.rfft(h, n=L, dim=axis)
        y = torch.fft.irfft(Hf * Kf, n=L, dim=axis)
        return y.narrow(axis, start, n)

    def _wave_separable(self, h):  # h: [B,nh,T,H,W,dh]
        if not (self.causal_time or self.linear_pad):
            # exact v1 joint circular rfftn over (T,H,W)
            K = torch.fft.rfftn(self.kernels(), s=(self.T, self.H, self.W),
                                dim=(1, 2, 3)).unsqueeze(0).unsqueeze(-1)
            X = torch.fft.rfftn(h, dim=(2, 3, 4))
            return torch.fft.irfftn(X * K, s=(self.T, self.H, self.W), dim=(2, 3, 4))
        kt, ky, kx = self._kernels_1d()
        h = self._lin_conv_axis(h, ky, 3, self.H)     # y
        h = self._lin_conv_axis(h, kx, 4, self.W)     # x
        h = self._lin_conv_axis(h, kt, 2, self.T)     # time
        return h

    # ---- dispersion poles (shared by forward's transfer and step()) ----------
    def half_lives(self):
        """DC half-life in frames per (head, mode): ln2 / alpha(k=0). Diagnostic for
        'how long does a ripple carry information' (long_horizon.py)."""
        if self.kernel_version != "dispersion":
            return None
        if self.pole_param == "halflife":
            return self.hl_min * (self.hl_max / self.hl_min) ** torch.sigmoid(self.hl_raw)
        return math.log(2.0) / F.softplus(self.a0)

    def _mode_pole(self, m, kx, ky, knorm):
        """Mode m's complex pole lam[nh,Hp,Wp] and its input normalizer (None for
        the softplus default, which keeps the v2 math op-for-op). For halflife
        poles the normalizer is sqrt(1-|lam|^2) (LRU-style): as |lam| -> 1 the
        state would otherwise grow ~1/(1-|lam|) and swamp the residual stream."""
        if self.pole_param == "halflife":
            hl = self.hl_min * (self.hl_max / self.hl_min) ** torch.sigmoid(self.hl_raw[:, m])
            alpha = (math.log(2.0) / hl)[:, None, None] \
                + F.softplus(self.visc[:, m])[:, None, None] * knorm
        else:
            alpha = F.softplus(self.a0[:, m][:, None, None] + self.a1[:, m][:, None, None] * knorm)
        Omega = (self.vx[:, m][:, None, None] * kx[None, None, :]
                 + self.vy[:, m][:, None, None] * ky[None, :, None]
                 + self.beta[:, m][:, None, None] * knorm)             # [nh,Hp,Wp]
        lam = torch.exp(-alpha) * torch.exp(1j * Omega)                # [nh,Hp,Wp], |lam|<1
        if self.pole_param == "halflife":
            return lam, torch.sqrt(-torch.expm1(-2.0 * alpha))          # sqrt(1-|lam|^2)
        return lam, None

    # ---- dispersion (v2) transfer function ----------------------------------
    def _transfer(self, L, Hp, Wp, device, exact=None):
        """G[nh, L, Hp, Wp] complex: temporal transfer per spatial-freq cell,
        G(w,k) = sum_m C_m B_m / (1 - lambda_m(k) e^{-i w}) -- the frequency
        response of z_t = lambda z_{t-1} + B x_t summed over modes."""
        ky = (2 * math.pi) * torch.fft.fftfreq(Hp, device=device)     # [Hp]
        kx = (2 * math.pi) * torch.fft.fftfreq(Wp, device=device)     # [Wp]
        knorm = torch.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)       # [Hp,Wp]
        w = (2 * math.pi) * torch.arange(L, device=device) / L        # [L]
        eiw = torch.exp(-1j * w)                                      # [L]
        G = torch.zeros(self.nh, L, Hp, Wp, dtype=torch.cfloat, device=device)
        # halflife: truncate the impulse response lam^n to n < T. The untruncated
        # transfer is the DTFT of an infinite response sampled at L=2T points, i.e. a
        # CIRCULAR kernel: frame s > t leaks into output t with weight lam^(2T-(s-t)).
        # For the softplus init (|lam| = 0.38) that is ~1e-7 and the v2 math is kept
        # as-is; for minutes-scale poles it is O(1) future leakage (the model could
        # read the frame it is asked to predict). sum_{n<T} lam^n e^{-iwn} =
        # (1 - lam^T e^{-iwT}) / (1 - lam e^{-iw}) -> exact causal linear conv.
        # ``exact`` forces the truncation for any pole family (the stateful path
        # needs chunk == full == stream exactly); None keeps the per-family default.
        if exact is None:
            exact = self.pole_param == "halflife"
        eiwT = torch.exp(-1j * w * self.T) if exact else None
        for m in range(self.n_modes):
            lam, norm = self._mode_pole(m, kx, ky, knorm)             # [nh,Hp,Wp]
            denom = 1 - lam[:, None, :, :] * eiw[None, :, None, None]  # [nh,L,Hp,Wp]
            gain = (self.Cg[:, m] * self.Bg[:, m])[:, None, None, None]
            if norm is not None:
                gain = gain * norm[:, None, :, :]
            if eiwT is not None:
                gain = gain * (1 - (lam ** self.T)[:, None, :, :] * eiwT[None, :, None, None])
            G = G + gain.to(torch.cfloat) / denom
        return G

    def _wave_dispersion(self, h):  # h: [B,nh,T,H,W,dh]
        Hp = 2 * self.H if self.linear_pad else self.H   # spatial zero-pad kills advection wrap
        Wp = 2 * self.W if self.linear_pad else self.W
        L = 2 * self.T                                   # temporal zero-pad => causal linear conv
        Xs = torch.fft.fft2(h, s=(Hp, Wp), dim=(3, 4))   # [B,nh,T,Hp,Wp,dh] complex
        Xt = torch.fft.fft(Xs, n=L, dim=2)               # over time -> [B,nh,L,Hp,Wp,dh]
        G = self._transfer(L, Hp, Wp, h.device)[None, :, :, :, :, None]
        Yt = torch.fft.ifft(Xt * G, n=L, dim=2)[:, :, :self.T]
        Ys = torch.fft.ifft2(Yt, dim=(3, 4))[..., :self.H, :self.W, :]
        return Ys.real

    def _wave_dispersion_stateful(self, h, z0):
        """Chunk of the dispersion recurrence with a carried state (LONG_HORIZON.md
        phase 1). h: [B,nh,T,H,W,dh]; z0: the step()/init_state() layout
        [B,n_modes,nh,Hp,Wp,dh] or None (zeros). Returns (y [B,nh,T,H,W,dh], zT).

        With z_{-1} = z0 the recurrence z_t = lam z_{t-1} + Bin x_t gives
            out_t = sum_m Cg_m lam_m^(t+1) z0_m  +  (exact causal conv of x),
            zT    = lam^T z0 + sum_s lam^(T-1-s) Bin x_s,
        so walking a long clip in T-frame chunks equals one long forward and
        equals step() frame by frame -- training can learn dependencies longer
        than the chunk it fits in memory."""
        B = h.shape[0]
        T = self.T
        Hp = 2 * self.H if self.linear_pad else self.H
        Wp = 2 * self.W if self.linear_pad else self.W
        L = 2 * T
        Xs = torch.fft.fft2(h, s=(Hp, Wp), dim=(3, 4))                 # [B,nh,T,Hp,Wp,dh]
        Xt = torch.fft.fft(Xs, n=L, dim=2)
        G = self._transfer(L, Hp, Wp, h.device, exact=True)[None, :, :, :, :, None]
        Yt = torch.fft.ifft(Xt * G, n=L, dim=2)[:, :, :T]              # zero-state part
        lam, norm, _, _ = self._dispersion_poles(h.device)             # [M,nh,Hp,Wp]
        n = torch.arange(T + 1, device=h.device, dtype=torch.float32).to(torch.cfloat)
        pw = lam[..., None] ** n                                       # [M,nh,Hp,Wp,T+1]
        Cg = self.Cg.t().to(torch.cfloat)                              # [M,nh]
        Bin = self.Bg.t().to(torch.cfloat)[:, :, None, None]           # [M,nh,1,1]
        if norm is not None:
            Bin = Bin * norm
        if z0 is not None:
            # out_t += sum_m Cg_m lam_m^(t+1) z0_m
            Yt = Yt + torch.einsum("mn,mnyxt,bmnyxd->bntyxd", Cg, pw[..., 1:], z0)
        # zT = lam^T z0 + sum_s lam^(T-1-s) Bin x_s
        rev = pw[..., :T].flip(-1)                                     # lam^(T-1-s), s=0..T-1
        zT = torch.einsum("mnyx,mnyxs,bnsyxd->bmnyxd", Bin, rev, Xs)
        if z0 is not None:
            zT = zT + pw[..., T][None, ..., None] * z0
        Ys = torch.fft.ifft2(Yt, dim=(3, 4))[..., :self.H, :self.W, :]
        return Ys.real, zT

    # ---- O(1)-in-T streaming recurrence (dispersion path only) --------------
    def _dispersion_poles(self, device):
        """Per-mode complex temporal pole lam[n_modes, nh, Hp, Wp] -- the SAME
        lambda(k) = exp(-alpha(k)) * exp(i*Omega(k)) that ``_transfer`` builds,
        so the recurrence below reproduces ``forward``'s dispersion math -- plus
        the matching input normalizer [n_modes, nh, Hp, Wp] (None for softplus)."""
        Hp = 2 * self.H if self.linear_pad else self.H
        Wp = 2 * self.W if self.linear_pad else self.W
        ky = (2 * math.pi) * torch.fft.fftfreq(Hp, device=device)      # [Hp]
        kx = (2 * math.pi) * torch.fft.fftfreq(Wp, device=device)      # [Wp]
        knorm = torch.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)        # [Hp,Wp]
        poles = [self._mode_pole(m, kx, ky, knorm) for m in range(self.n_modes)]
        lam = torch.stack([p[0] for p in poles], 0)                    # [n_modes,nh,Hp,Wp]
        norm = None if poles[0][1] is None else torch.stack([p[1] for p in poles], 0)
        return lam, norm, Hp, Wp

    def _dispersion_lam(self, device):
        lam, _, Hp, Wp = self._dispersion_poles(device)
        return lam, Hp, Wp

    def init_state(self, B, device):
        """Streaming state for ``step()`` (dispersion path only): complex zeros of
        shape [B, n_modes, nh, Hp, Wp, dh]. Mirrors ``SSMLite.init_state(B, device)``
        so the sanity harness can drive both mixers identically. O(1) in T."""
        assert self.kernel_version == "dispersion", "step() is dispersion-only"
        Hp = 2 * self.H if self.linear_pad else self.H
        Wp = 2 * self.W if self.linear_pad else self.W
        return torch.zeros(B, self.n_modes, self.nh, Hp, Wp, self.dh,
                           dtype=torch.cfloat, device=device)

    def state_bytes(self, B=1, device="cpu"):
        """Persistent recurrent-state size in bytes (the R14 killer metric). Only
        the dispersion path has an O(1)-in-T recurrence; the separable path is not
        streamable, so it reports 0 (it is never used as a streaming arm)."""
        if self.kernel_version != "dispersion":
            return 0
        s = self.init_state(B, device)
        return s.element_size() * s.nelement()

    def step(self, x_t, state):
        """Advance one frame of the dispersion wave recurrence. ``x_t``: [B, H*W, D]
        (raw input for frame t); returns ([B, H*W, D], new_state).

        Per mode m and spatial-frequency cell k=(kx,ky):
            z_t^m(k) = lam_m(k) * z_{t-1}^m(k) + Bg_m * x_hat_t(k)   # Bg = input gain
            out_hat(k) = sum_m Cg_m * z_t^m(k)                       # Cg = output gain
        then iFFT2 over the (Hp,Wp) spatial spectrum, crop to (H,W), take Re, and
        apply ``po``. This is the O(d_state)-memory-in-T equivalent of the
        frequency-domain transfer in ``_wave_dispersion`` -- they agree up to the
        |lam|^{T+1} time-aliasing that forward's L=2T zero-pad leaves behind
        (negligible for stable poles; verified in sanity_check.py to < 1e-3)."""
        assert self.kernel_version == "dispersion", "step() is dispersion-only"
        B, S, D = x_t.shape
        h = self.pi(x_t).view(B, self.H, self.W, self.nh, self.dh).permute(0, 3, 1, 2, 4)
        h = self._write(x_t, h.float())                                # [B,nh,H,W,dh]
        lam, norm, Hp, Wp = self._dispersion_poles(x_t.device)         # [n_modes,nh,Hp,Wp]
        x_hat = torch.fft.fft2(h, s=(Hp, Wp), dim=(2, 3))              # [B,nh,Hp,Wp,dh] cfloat
        Bg = self.Bg.t()[None, :, :, None, None, None]                 # [1,n_modes,nh,1,1,1]
        Cg = self.Cg.t()[None, :, :, None, None, None]                 # [1,n_modes,nh,1,1,1]
        lam_b = lam[None, :, :, :, :, None]                            # [1,n_modes,nh,Hp,Wp,1]
        Bin = Bg.to(torch.cfloat)
        if norm is not None:                                           # halflife: sqrt(1-|lam|^2)
            Bin = Bin * norm[None, :, :, :, :, None]
        state = lam_b * state + Bin * x_hat[:, None]                   # [B,n_modes,nh,Hp,Wp,dh]
        out_hat = (Cg.to(torch.cfloat) * state).sum(1)                 # [B,nh,Hp,Wp,dh]
        out = torch.fft.ifft2(out_hat, dim=(2, 3))[..., :self.H, :self.W, :].real
        out = out.permute(0, 2, 3, 1, 4).reshape(B, S, D)              # (y,x,nh,dh) -> [B,H*W,D]
        return self.po(out.to(x_t.dtype)), state

    # ---- shared post-ops (gate + local fuse) --------------------------------
    def _apply_local(self, out, h):
        B = h.shape[0]
        hp = h.permute(0, 2, 1, 5, 3, 4).reshape(B * self.T, self.nh * self.dh, self.H, self.W)
        loc = self.local(hp).reshape(B, self.T, self.nh, self.dh, self.H, self.W)
        loc = loc.permute(0, 2, 1, 4, 5, 3)              # [B,nh,T,H,W,dh]
        return out + loc

    def _apply_gate(self, out, h):
        if self.causal_time:
            # Prefix mean over frames <= t. The all-T mean the gate used before
            # let a "causal" model read future frames through g (LONG_HORIZON.md).
            per_t = h.mean(dim=(3, 4))                                 # [B,nh,T,dh]
            cnt = torch.arange(1, h.shape[2] + 1, device=h.device, dtype=h.dtype)
            pooled = per_t.cumsum(2) / cnt[None, None, :, None]
            g = torch.sigmoid(self.gate_fc(pooled))                    # [B,nh,T,dh]
            return out * g[:, :, :, None, None, :]
        pooled = h.mean(dim=(2, 3, 4))                    # [B,nh,dh] per-head content summary
        g = torch.sigmoid(self.gate_fc(pooled))           # [B,nh,dh]
        return out * g[:, :, None, None, None, :]

    def _write(self, x, h):
        """Wave input: h, scaled by the write gate when on. x: [B, N, D] tokens,
        h: [B, nh, T, H, W, dh] (or [B, nh, H, W, dh] for one frame)."""
        if not self.write_gate:
            return h
        g = torch.sigmoid(self.wg(x)).float()                       # [B, N, nh]
        if h.dim() == 5:                                            # one frame (step)
            g = g.view(x.shape[0], self.H, self.W, self.nh).permute(0, 3, 1, 2)
            return h * g[..., None]
        g = g.view(x.shape[0], self.T, self.H, self.W, self.nh).permute(0, 4, 1, 2, 3)
        return h * g[..., None]

    def forward(self, x):  # x: [B, N, D], N = T*H*W
        B, N, D = x.shape
        h = self.pi(x).view(B, self.T, self.H, self.W, self.nh, self.dh).permute(0, 4, 1, 2, 3, 5)
        h = h.float()
        if self.write_gate:
            if self.kernel_version != "dispersion":
                raise ValueError("write_gate applies to the dispersion operator")
            out = self._wave_dispersion(self._write(x, h))
        elif self.kernel_version == "dispersion":
            out = self._wave_dispersion(h)
        else:
            out = self._wave_separable(h)
        if self.local_fuse:
            out = self._apply_local(out, h)
        if self.gate:
            out = self._apply_gate(out, h)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(B, N, D)
        return self.po(out.to(x.dtype))

    def forward_stateful(self, x, state=None):
        """forward() over one T-frame chunk starting from ``state`` (None = zeros);
        returns (out, state_after_chunk) in the step() layout."""
        if self.kernel_version != "dispersion" or self.gate:
            raise ValueError("stateful chunks need the dispersion operator without --gate "
                             "(the gate's running mean is not carried)")
        B, N, D = x.shape
        h = self.pi(x).view(B, self.T, self.H, self.W, self.nh, self.dh).permute(0, 4, 1, 2, 3, 5)
        h = h.float()
        out, zT = self._wave_dispersion_stateful(self._write(x, h), state)
        if self.local_fuse:
            out = self._apply_local(out, h)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(B, N, D)
        return self.po(out.to(x.dtype)), zT


class AttnMix(nn.Module):
    """Full attention baseline over flattened tokens. Position comes only from the
    shared FactorizedPosEmb at the model input -- the v1 per-layer [1,N,dim] table
    is deleted so attention no longer gets a free positional-memory advantage.

    causal: temporal causal mask -- a token in frame t attends only to tokens in
    frames <= t (spatial attention within/behind the current frame stays full)."""

    def __init__(self, dim, n_heads, T, H, W, causal=False):
        super().__init__()
        self.nh = n_heads
        self.T, self.H, self.W = T, H, W
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.po = nn.Linear(dim, dim)
        if causal:
            frame = torch.arange(T * H * W) // (H * W)             # frame id per token
            mask = frame[None, :] <= frame[:, None]                # [N,N] bool, keep <= t
            self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.nh, D // self.nh).transpose(1, 2) for t in (q, k, v))
        mask = self.attn_mask if self.causal else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.po(y)

    def state_bytes(self, B=1, device="cpu"):
        """Attention has NO persistent recurrent state -- its 'memory' is the whole
        T-frame window it re-attends every step, which the model must keep O(T·H·W·
        dim). That bounded window is reported separately as ``window_bytes``; the
        persistent (horizon-independent) state is 0. This asymmetry is the point."""
        return 0


class FusedMix(nn.Module):
    """Gated fusion of a finite-context LOCAL path and a persistent GLOBAL memory
    path (HARDENING_astra.md Rank 2):  h = g*h_local + (1-g)*h_global, with
    g = sigmoid(Linear([h_local; h_global])).

      fuse="local_ssm"  -> global = SSMLite            (generic diagonal SSM)
      fuse="local_wave" -> global = WaveMix3D dispersion (the structured wave state)

    The local path is *causal* windowed attention (window = T): finite context, so
    it cannot see past the T most-recent frames. The global path is the O(1)-in-T
    recurrence that CAN carry information across an arbitrarily long occlusion.

    Streaming (``step``): the global path advances its O(1) recurrence; the local
    path keeps a rolling buffer of the last T frames of (normed) tokens and
    recomputes causal attention over them. Only the global recurrence is the
    *persistent* state (``state_bytes``); the local buffer is bounded O(T) and is
    reported as ``window_bytes``. Because the local attention is causal, ingesting a
    T-frame clip one frame at a time reproduces ``forward`` on that clip at the last
    frame (checked by test_fusion_r14.py, mirroring sanity_check.py test 8b)."""

    def __init__(self, dim, n_heads, T, H, W, fuse, causal=True, linear_pad=True,
                 pole_param="softplus", hl_min=2.0, hl_max=4096.0):
        super().__init__()
        self.dim, self.nh = dim, n_heads
        self.T, self.H, self.W = T, H, W
        self.fuse = fuse
        self.local = AttnMix(dim, n_heads, T, H, W, causal=True)   # always causal for streaming parity
        if fuse == "local_ssm":
            from ssm_lite import SSMLite
            self.glob = SSMLite(dim, n_heads, T, H, W, causal=causal)
        elif fuse == "local_wave":
            self.glob = WaveMix3D(dim, n_heads, T, H, W, kernel_version="dispersion",
                                  causal_time=causal, linear_pad=linear_pad,
                                  gate=False, local_fuse=False, pole_param=pole_param,
                                  hl_min=hl_min, hl_max=hl_max)
        else:
            raise ValueError(f"unknown fuse {fuse!r} (want local_ssm | local_wave)")
        self.gate = nn.Linear(2 * dim, dim)
        self._gate_mean = None            # diagnostic: mean g of the last forward/step

    def _fuse(self, hl, hg):
        g = torch.sigmoid(self.gate(torch.cat([hl, hg], dim=-1)))
        # Diagnostic only (DEEP_DIVE_3 Rank 1): mean gate value -- "is the global
        # state pulling its weight?". g routes toward the LOCAL path, so mean g -> 1
        # means the wave/ssm persistent state is dead weight. Detached so it never
        # enters autograd and cannot perturb training or the default (fuse=none) path.
        self._gate_mean = g.detach().mean()
        return g * hl + (1.0 - g) * hg

    def forward(self, x):  # x: [B, N, D], N = T*H*W
        return self._fuse(self.local(x), self.glob(x))

    # ---- O(1)-in-T streaming (global) + O(T) rolling window (local) ----------
    def init_state(self, B, device):
        return {"glob": self.glob.init_state(B, device),
                "buf": torch.zeros(B, self.T, self.H * self.W, self.dim, device=device),
                "n": torch.zeros((), dtype=torch.long, device=device)}   # filled slots

    def step(self, x_t, state):
        """x_t: [B, H*W, D] (current frame's normed tokens) -> ([B,H*W,D], state).

        Only the current frame's queries are computed, against the FILLED window
        slots. (Attending the zero-initialized slots during the first T-1 frames
        made warm-up diverge from ``forward``, where frame t sees frames <= t
        only; the zero-init residual head hid it from the parity test.)"""
        B, S, D = x_t.shape
        hg, gstate = self.glob.step(x_t, state["glob"])
        buf = state["buf"].to(x_t.dtype)
        buf = torch.cat([buf[:, 1:], x_t.unsqueeze(1)], dim=1)     # slide: current at last slot
        n = torch.clamp(state.get("n", torch.tensor(self.T - 1)) + 1, max=self.T)
        k_ = int(n)
        att = self.local
        dh = D // att.nh
        q = att.qkv(x_t).chunk(3, dim=-1)[0].view(B, S, att.nh, dh).transpose(1, 2)
        _, k, v = att.qkv(buf[:, self.T - k_:].reshape(B, k_ * S, D)).chunk(3, dim=-1)
        k, v = (t.view(B, k_ * S, att.nh, dh).transpose(1, 2) for t in (k, v))
        y = F.scaled_dot_product_attention(q, k, v)                # current frame sees <= t
        hl = att.po(y.transpose(1, 2).reshape(B, S, D))
        return self._fuse(hl, hg), {"glob": gstate, "buf": buf, "n": n}

    def state_bytes(self, B=1, device="cpu"):
        # Persistent (horizon-independent) memory is the global recurrence ONLY.
        return self.glob.state_bytes(B, device)


class Block(nn.Module):
    def __init__(self, mix, dim, ffn_mult=4.0, bias=True):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.mix = mix
        self.n2 = RMSNorm(dim)
        self.ffn = FFN(dim, mult=ffn_mult, bias=bias)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))

    def forward_stateful(self, x, state):
        if not hasattr(self.mix, "forward_stateful"):
            raise ValueError(f"{type(self.mix).__name__} has no carried-state forward "
                             "(stateful chunks support the wave dispersion and ssm arms)")
        m, state = self.mix.forward_stateful(self.n1(x), state)
        x = x + m
        return x + self.ffn(self.n2(x)), state


class VideoPredictor(nn.Module):
    """Frames [B,T,3,H,W] -> per-cell tokens -> mix -> predict next frame.

    residual: predict the next frame as last_frame + delta, delta = head(last
              tokens) with the head's final linear zero-initialized, so training
              starts exactly at the copy-last baseline instead of learning RGB
              reconstruction from scratch. (default ON for v2; off reproduces v1.)
    kind: "wave" | "attn" | "ssm" | "qssm" (quaternion-state SSM, R15).
    q_mix / quat_color: R15 hypercomplex arms on --kind wave (wfvideo_quat.py);
              both default off and the flag-off paths are byte-identical.
    pole_param: "softplus" (v2 default) | "halflife" (LONG_HORIZON.md): the wave
              poles are parameterized by a half-life in frames in [hl_min, hl_max]
              so the recurrent state can remember minutes, not ~1 frame."""

    def __init__(self, dim, n_layers, n_heads, T, H, W, kind,
                 causal=False, residual=True, ffn_mult=4.0,
                 kernel_version="separable", linear_pad=False,
                 gate=False, local_fuse=False, fuse="none",
                 q_mix=False, quat_color=False,
                 pole_param="softplus", hl_min=2.0, hl_max=4096.0, time_pos="table",
                 write_gate=False, clean_write=False, in_ch=3):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.grad_ckpt = False       # train_long.py --grad-ckpt (stateful path only)
        # in_ch: channels per frame -- 3 for pixels, or a video VAE's latent
        # channels (LONG_HORIZON.md phase 2); the head predicts the same channels.
        if quat_color and in_ch != 3:
            raise ValueError("--quat-color embeds RGB; it needs in_ch=3")
        self.in_ch = in_ch
        self.kind = kind
        if time_pos not in ("table", "none"):
            raise ValueError(f"unknown time_pos {time_pos!r} (want table | none)")
        self.time_pos = time_pos
        self.fuse = fuse
        self.residual = residual
        # R15 hypercomplex arms (HYPERCOMPLEX_STUDY.md sec 4-E2/E3): wave-scope,
        # default-off. Flag-off paths below construct the existing classes in the
        # same order, so default init is byte-identical (asserted r15 test).
        if q_mix and kind != "wave":
            raise ValueError(f"--q-mix applies to kind 'wave' (got {kind!r})")
        if q_mix and fuse != "none":
            raise ValueError("--q-mix is not supported together with --fuse hybrids")
        if pole_param != "softplus" and not (kind == "wave" and kernel_version == "dispersion"
                                             and not q_mix) and fuse != "local_wave":
            raise ValueError("--pole-param halflife needs the dispersion wave operator "
                             "(--kind wave --kernel-version dispersion, or --fuse local_wave)")
        if quat_color and kind != "wave":
            raise ValueError(f"--quat-color applies to kind 'wave' (got {kind!r})")
        if quat_color and dim % 4:
            raise ValueError(f"--quat-color needs dim divisible by 4 (got {dim})")
        if quat_color:
            from wfvideo_quat import QuatEmbed
            self.embed = QuatEmbed(dim)
        else:
            self.embed = nn.Conv2d(in_ch, dim, 3, padding=1, bias=not clean_write)
        # clean_write (LONG_HORIZON.md 8.4): a blank frame must write exactly zero --
        # no spatial table, no embed bias, no wave input bias.
        if (write_gate or clean_write) and not (kind == "wave" and kernel_version == "dispersion"
                                                 and fuse == "none" and not q_mix):
            raise ValueError("write_gate/clean_write need the plain dispersion wave arm")
        if clean_write and time_pos != "none":
            # a temporal table adds a nonzero vector to every (blank) frame -> the
            # constant background clean_write exists to remove (LONG_HORIZON.md 8.4)
            raise ValueError("clean_write needs time_pos='none'")
        self.posemb = FactorizedPosEmb(dim, T, H, W, time=(time_pos == "table"),
                                       space=not clean_write)          # shared, all arms
        if fuse != "none":
            # Hybrid arm: the global memory kind must agree with --kind so config,
            # streaming, and state-byte accounting all name the same operator.
            want = {"local_ssm": "ssm", "local_wave": "wave"}.get(fuse)
            if want is None:
                raise ValueError(f"unknown fuse {fuse!r}")
            if kind != want:
                raise ValueError(f"--fuse {fuse} requires --kind {want} (got {kind})")
        blocks = []
        for _ in range(n_layers):
            if fuse != "none":
                mix = FusedMix(dim, n_heads, T, H, W, fuse,
                               causal=causal, linear_pad=linear_pad, pole_param=pole_param,
                               hl_min=hl_min, hl_max=hl_max)
            elif kind == "wave":
                if q_mix:
                    from wfvideo_quat import WaveQuatMix
                    mix = WaveQuatMix(dim, n_heads, T, H, W, kernel_version=kernel_version,
                                      causal_time=causal, linear_pad=linear_pad,
                                      gate=gate, local_fuse=local_fuse)
                else:
                    mix = WaveMix3D(dim, n_heads, T, H, W, kernel_version=kernel_version,
                                    causal_time=causal, linear_pad=linear_pad,
                                    gate=gate, local_fuse=local_fuse, pole_param=pole_param,
                                    hl_min=hl_min, hl_max=hl_max, write_gate=write_gate,
                                    pi_bias=not clean_write)
            elif kind == "attn":
                mix = AttnMix(dim, n_heads, T, H, W, causal=causal)
            elif kind == "ssm":
                from ssm_lite import SSMLite
                mix = SSMLite(dim, n_heads, T, H, W, causal=causal)
            elif kind == "qssm":
                from ssm_quat import QuatSSM
                mix = QuatSSM(dim, n_heads, T, H, W, causal=causal)
            else:
                raise ValueError(f"unknown kind {kind!r}")
            # clean_write: every layer bias-free (wave pi/po, FFN) so a blank frame
            # stays exactly zero through the whole stack -- RMSNorm maps 0 to 0.
            blocks.append(Block(mix, dim, ffn_mult=ffn_mult, bias=not clean_write))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, in_ch)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.use_ckpt = False

    def forward(self, frames, states=None, dense=False):
        """frames: [B, T, 3, H, W] (context frames).

        dense=False (default): predict the frame after the last one, [B,3,H,W].
        dense=True: predict the next frame at EVERY position, [B,T,3,H,W]
            (pred[:, t] targets frames[t+1]) -- T targets per causal forward
            instead of one (LONG_HORIZON.md phase 1).
        states: list with one entry per block (entries may be None = zeros) to
            run this T-frame chunk from a carried recurrent state; the call then
            returns (pred, new_states) so long clips can be trained chunk by chunk."""
        B, T, C, H, W = frames.shape
        f = frames.reshape(B * T, C, H, W)
        e = self.embed(f).reshape(B, T, self.H * self.W, -1).reshape(B, T * self.H * self.W, -1)
        x = self.posemb(e)
        new_states = None
        if states is not None:
            if len(states) != len(self.blocks):
                raise ValueError(f"need {len(self.blocks)} states, got {len(states)}")
            new_states = []
            for blk, st in zip(self.blocks, states):
                if self.grad_ckpt and torch.is_grad_enabled():
                    # Keep only each block's input; recompute its FFT activations
                    # in backward (one layer's spectra at a time, not all layers').
                    x, st = torch.utils.checkpoint.checkpoint(blk.forward_stateful, x, st,
                                                              use_reentrant=False)
                else:
                    x, st = blk.forward_stateful(x, st)
                new_states.append(st)
        else:
            for blk in self.blocks:
                if self.use_ckpt and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        x = self.norm(x)
        if dense:
            delta = self.head(x).reshape(B, T, self.H, self.W, self.in_ch).permute(0, 1, 4, 2, 3)
            pred = frames + delta if self.residual else delta         # [B,T,3,H,W]
        else:
            last = x[:, (self.T - 1) * self.H * self.W: self.T * self.H * self.W]  # [B, HW, D]
            delta = self.head(last).reshape(B, self.H, self.W, self.in_ch).permute(0, 3, 1, 2)  # [B,C,H,W]
            pred = frames[:, -1] + delta if self.residual else delta
        return pred if states is None else (pred, new_states)

    # ---- O(1)-in-T streaming rollout (wave / ssm / local+global hybrids) -----
    def _streamable(self):
        return self.fuse != "none" or self.kind in ("wave", "ssm", "qssm")

    def stream_init(self, B, device):
        """Per-layer streaming state (list, one entry per block). Supported for the
        recurrent arms: wave (dispersion), ssm, qssm, and the local+ssm /
        local+wave hybrids. Attention has no recurrence and must roll out
        windowed instead."""
        assert self._streamable(), "stream_step needs a recurrent arm (wave/ssm/fused)"
        return [blk.mix.init_state(B, device) for blk in self.blocks]

    def persistent_state_bytes(self, B=1, device="cpu"):
        """Total persistent recurrent-state bytes across blocks (0 for attention)."""
        return sum(blk.mix.state_bytes(B, device) for blk in self.blocks)

    def gate_means(self):
        """Per-layer mean gate value g from the most recent forward pass (empty for
        non-fused arms). g -> 1 => local causal path dominates, the persistent global
        (wave/ssm) state is dead weight (DEEP_DIVE_3 Rank 1 kill signal); g -> 0 =>
        the model routes through the persistent global memory. Reads the detached
        scalar cached in FusedMix._fuse, so it must be called after a forward()."""
        out = []
        for blk in self.blocks:
            gm = getattr(blk.mix, "_gate_mean", None)
            if gm is not None:
                out.append(float(gm))
        return out

    def stream_step(self, frame, states, t_index):
        """Ingest ONE frame [B,3,H,W] and predict the next, threading the O(1)
        recurrence state through every block. ``t_index`` is the absolute frame
        index; the temporal positional embedding is clamped to pt[min(t,T-1)] --
        a documented steady-state choice so rollout can run past T frames (the
        recurrence itself is time-invariant, so this only affects the additive
        input phase, not the dynamics). Returns (next_frame [B,3,H,W], states)."""
        B = frame.shape[0]
        e = self.embed(frame).reshape(B, self.H * self.W, -1)            # [B,HW,D]
        pe = self.posemb
        spatial = pe.spatial()
        x = e
        if pe.pt is not None:                    # time_pos="table": clamp past T
            x = x + pe.pt[min(t_index, self.T - 1)][None, None, :]
        if spatial is not None:                  # (main's summation order: e + pt + spatial)
            x = x + spatial[None]
        for i, blk in enumerate(self.blocks):
            m, states[i] = blk.mix.step(blk.n1(x), states[i])
            x = x + m
            x = x + blk.ffn(blk.n2(x))
        x = self.norm(x)
        delta = self.head(x).reshape(B, self.H, self.W, self.in_ch).permute(0, 3, 1, 2)
        nxt = frame + delta if self.residual else delta
        return nxt, states
