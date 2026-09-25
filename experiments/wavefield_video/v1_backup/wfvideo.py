"""Wave-field video predictor + transformer baseline (identical scaffolds).

Mixing is 3D: separable damped-cosine kernels over (time, y, x), applied
via n-dimensional real FFT. The transformer baseline uses full attention
over the same flattened tokens. A diagonal-SSM baseline lives in ssm_lite.py.

v1 behavior is the default: with ``causal_time=False, linear_time=False,
posemb=False`` the wave arm and attention arm reproduce the original runs
bit-for-bit on square grids. The new flags (causal/linear time, factorized
positional embeddings) are opt-in so head-to-head comparisons stay valid.
"""
import torch
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
    def __init__(self, dim, mult=4.0):
        super().__init__()
        h = int(dim * mult)
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(h, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class FactorizedPosEmb(nn.Module):
    """Learned separable (t, y, x) positional embeddings, summed.

    Shared scheme for every mixer arm so the comparison is fair: instead of
    a flat [N, dim] table (which scales with T*H*W and lets attention simply
    memorize absolute position), we learn one table per axis and broadcast-add.
    Parameter count is (T + H + W) * dim instead of (T*H*W) * dim.
    """

    def __init__(self, dim, T, H, W):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.pt = nn.Parameter(torch.zeros(T, dim))
        self.py = nn.Parameter(torch.zeros(H, dim))
        self.px = nn.Parameter(torch.zeros(W, dim))
        for p in (self.pt, self.py, self.px):
            nn.init.normal_(p, std=0.02)

    def forward(self, x):  # x: [B, N, dim], N = T*H*W in (t, y, x) order
        B, N, D = x.shape
        pos = (self.pt[:, None, None, :]
               + self.py[None, :, None, :]
               + self.px[None, None, :, :])          # [T, H, W, dim]
        return x + pos.reshape(1, N, D)


def _damped_cos(a, w, p, d):
    """Per-head damped cosine e^{-|a||d|} cos(w d + p). a,w,p: [nh]; d: [L]."""
    env = torch.exp(-a.abs()[:, None] * d.abs()[None, :])
    return env * torch.cos(w[:, None] * d[None, :] + p[:, None])


class WaveMix3D(nn.Module):
    """Separable 3D wave kernel (time, y, x) applied via rfftn per head.

    Flags (all default False -> v1 behavior):
      causal_time: zero the future-reaching half of the temporal kernel so a
                   frame's output depends only on frames <= t (see note below).
      linear_time: zero-padded (linear) convolution over time instead of the
                   circular FFT conv, removing wraparound between frame 0 and T-1.

    Causality note: with ``out[t] = sum_u kc[u] h[t-u]`` and ``kc=ifftshift(k)``,
    a tap at dt>0 reads a PAST frame h[t-dt] and a tap at dt<0 reads a FUTURE
    frame h[t+|dt|]. So causal masking zeroes the dt<0 half. (This is the sign
    that makes the last-frame read-out truly autoregressive; flip the ``< 0`` to
    ``> 0`` if the opposite convention is desired.)
    """

    def __init__(self, dim, n_heads, T, H, W, causal_time=False, linear_time=False):
        super().__init__()
        self.nh = n_heads
        self.T, self.H, self.W = T, H, W
        self.causal_time = causal_time
        self.linear_time = linear_time
        self.pi = nn.Linear(dim, dim)
        self.po = nn.Linear(dim, dim)
        self.a_t = nn.Parameter(torch.rand(n_heads) * 0.05 + 0.01)
        self.a_s = nn.Parameter(torch.rand(n_heads) * 0.05 + 0.01)
        self.w_t = nn.Parameter(torch.rand(n_heads) * 2.0)
        self.w_s = nn.Parameter(torch.rand(n_heads) * 2.0)
        self.p_t = nn.Parameter(torch.zeros(n_heads))
        self.p_s = nn.Parameter(torch.zeros(n_heads))
        # Per-axis offsets: dt from T, dy from H, dx from W (was W for all -> bug).
        self.register_buffer("dt", (torch.arange(T) - T // 2).float(), persistent=False)
        self.register_buffer("dy", (torch.arange(H) - H // 2).float(), persistent=False)
        self.register_buffer("dx", (torch.arange(W) - W // 2).float(), persistent=False)

    def _kernels_1d(self):
        kt = _damped_cos(self.a_t, self.w_t, self.p_t, self.dt)   # [nh, T]
        ky = _damped_cos(self.a_s, self.w_s, self.p_s, self.dy)   # [nh, H]
        kx = _damped_cos(self.a_s, self.w_s, self.p_s, self.dx)   # [nh, W]
        if self.causal_time:
            kt = kt * (self.dt >= 0).to(kt.dtype)[None, :]        # drop future (dt<0) taps
        return kt, ky, kx

    def kernels(self):
        """Full centered 3D kernel (used by the v1 circular-FFT path)."""
        kt, ky, kx = self._kernels_1d()
        k = kt[:, :, None, None] * ky[:, None, :, None] * kx[:, None, None, :]
        return torch.fft.ifftshift(k, dim=(1, 2, 3))

    def _time_conv(self, h, kt):
        """Convolve h [B,nh,T,H,W,dh] along time (dim=2) with kt [nh,T].

        Uses a length-L FFT: L=T for circular, L=2T-1 for linear (zero-padded,
        no wraparound). The kernel is placed at ifftshift indices so zero-lag
        sits at index 0 and negative lags wrap to the buffer tail; with L>=2T-1
        the zero-pad region absorbs what would otherwise wrap."""
        T = self.T
        L = (2 * T - 1) if self.linear_time else T
        kb = h.new_zeros(self.nh, L)
        idx = (self.dt.long() % L)                                # dt=0->0, +j->j, -j->L-j
        kb[:, idx] = kt.to(kb.dtype)
        Kf = torch.fft.rfft(kb, n=L, dim=1)[None, :, :, None, None, None]
        Hf = torch.fft.rfft(h, n=L, dim=2)
        y = torch.fft.irfft(Hf * Kf, n=L, dim=2)
        return y[:, :, :T]

    def forward(self, x):  # x: [B, N, D], N = T*H*W
        B, N, D = x.shape
        dh = D // self.nh
        h = self.pi(x).view(B, self.T, self.H, self.W, self.nh, dh).permute(0, 4, 1, 2, 3, 5)
        h = h.float()
        if not self.causal_time and not self.linear_time:
            # v1 exact path: joint circular rfftn over (T, H, W).
            K = torch.fft.rfftn(self.kernels(), s=(self.T, self.H, self.W),
                                dim=(1, 2, 3)).unsqueeze(0).unsqueeze(-1)
            X = torch.fft.rfftn(h, dim=(2, 3, 4))
            y = torch.fft.irfftn(X * K, s=(self.T, self.H, self.W), dim=(2, 3, 4))
        else:
            # Separable: spatial circular conv (2D FFT) then temporal conv.
            kt, ky, kx = self._kernels_1d()
            ks = torch.fft.ifftshift(ky[:, :, None] * kx[:, None, :], dim=(1, 2))
            Ks = torch.fft.rfft2(ks, s=(self.H, self.W), dim=(1, 2))
            Ks = Ks[None, :, None, :, :, None]
            Xs = torch.fft.rfft2(h, dim=(3, 4))
            hs = torch.fft.irfft2(Xs * Ks, s=(self.H, self.W), dim=(3, 4))
            y = self._time_conv(hs, kt)
        y = y.permute(0, 2, 3, 4, 1, 5).reshape(B, N, D)
        return self.po(y.to(x.dtype))


class AttnMix(nn.Module):
    """Full attention baseline over flattened tokens.

    flat_pos: keep the original flat [1,N,dim] learned position table (v1). When
              the model uses the shared FactorizedPosEmb instead, this is turned
              off so position is not double-counted.
    causal:   temporal causal mask -- a token in frame t attends only to tokens
              in frames <= t (spatial attention within/behind the current frame
              stays full)."""

    def __init__(self, dim, n_heads, T, H, W, causal=False, flat_pos=True):
        super().__init__()
        self.nh = n_heads
        self.T, self.H, self.W = T, H, W
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.po = nn.Linear(dim, dim)
        self.flat_pos = flat_pos
        if flat_pos:
            self.pos = nn.Parameter(torch.zeros(1, T * H * W, dim))
            nn.init.normal_(self.pos, std=0.02)
        if causal:
            frame = torch.arange(T * H * W) // (H * W)             # frame id per token
            mask = frame[None, :] <= frame[:, None]                # [N,N] bool, keep <= t
            self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        B, N, D = x.shape
        if self.flat_pos:
            x = x + self.pos
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.nh, D // self.nh).transpose(1, 2) for t in (q, k, v))
        mask = self.attn_mask if self.causal else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.po(y)


class Block(nn.Module):
    def __init__(self, mix, dim):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.mix = mix
        self.n2 = RMSNorm(dim)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


class VideoPredictor(nn.Module):
    """Frames [B,T,3,H,W] -> per-cell tokens -> mix -> predict next frame.

    Token at (t,y,x) is the embedded pixel; the head reads the LAST frame's
    tokens and outputs the next frame's pixels at the same cell.

    kind: "wave" | "attn" | "ssm".
    causal: enable temporal causality in the chosen mixer.
    posemb: use the shared factorized (t,y,x) positional embedding for ALL arms
            (and drop attention's flat table) for a fair comparison. Default off
            reproduces v1 (attention keeps its flat table; wave/ssm get none)."""

    def __init__(self, dim, n_layers, n_heads, T, H, W, kind,
                 causal=False, posemb=False):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.kind = kind
        self.embed = nn.Conv2d(3, dim, 3, padding=1)
        self.posemb = FactorizedPosEmb(dim, T, H, W) if posemb else None
        flat_pos = not posemb
        blocks = []
        for _ in range(n_layers):
            if kind == "wave":
                mix = WaveMix3D(dim, n_heads, T, H, W,
                                causal_time=causal, linear_time=causal)
            elif kind == "attn":
                mix = AttnMix(dim, n_heads, T, H, W, causal=causal, flat_pos=flat_pos)
            elif kind == "ssm":
                from ssm_lite import SSMLite
                mix = SSMLite(dim, n_heads, T, H, W, causal=causal)
            else:
                raise ValueError(f"unknown kind {kind!r}")
            blocks.append(Block(mix, dim))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 3)
        self.use_ckpt = False

    def forward(self, frames):  # frames: [B, T, 3, H, W] (context frames)
        B, T, C, H, W = frames.shape
        f = frames.reshape(B * T, C, H, W)
        e = self.embed(f).reshape(B, T, self.H * self.W, -1).reshape(B, T * self.H * self.W, -1)
        x = e
        if self.posemb is not None:
            x = self.posemb(x)
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        last = x[:, (self.T - 1) * self.H * self.W : self.T * self.H * self.W]  # [B, HW, D]
        px = self.head(last)                      # [B, HW, 3]
        out = px.reshape(B, self.H, self.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
        return out
