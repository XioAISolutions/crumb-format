"""S4D-style diagonal state-space mixing block (the fair long-context baseline).

Same interface as ``WaveMix3D`` / ``AttnMix``:
  * ``pi`` (Linear in) / ``po`` (Linear out) submodules,
  * ``forward([B, N, dim]) -> [B, N, dim]`` with N = T*H*W in (t, y, x) order,
  * mixes causally over time only (spatial cells are independent channels),
  * reshapes internally to ``[B, nh, T, H, W, dh]`` like the others.

A diagonal SSM has the closed-form convolution kernel

    k[c, n] = Re( sum_s C[c, s] * B[c, s] * A[c, s]^n ),   n = 0 .. T-1

which is a sum of damped complex exponentials -- the generic-learned cousin of
the wave arm's hand-built damped cosines. Training uses an FFT convolution
(linear, zero-padded, so it is causal with no wraparound); ``step()`` runs the
equivalent O(1)-state recurrence for streaming rollout.

Parameter budget is dominated by ``pi``/``po`` (2*dim^2), same order as wave;
the SSM tensors add ~n_heads*(dh+2)*d_state, far smaller.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMLite(nn.Module):
    def __init__(self, dim, n_heads, T, H, W, d_state=16, causal=True):
        super().__init__()
        self.nh = n_heads
        self.dh = dim // n_heads
        self.T, self.H, self.W = T, H, W
        self.d_state = d_state
        self.causal = causal  # SSM is causal by construction; flag kept for parity
        self.pi = nn.Linear(dim, dim)
        self.po = nn.Linear(dim, dim)

        # Diagonal state matrix A = exp(-softplus(a_log) + i*a_im): poles strictly
        # inside the unit circle. Shared per head across its dh channels.
        a_log = torch.log(torch.expm1(torch.full((n_heads, d_state), 0.05)))  # softplus^-1(0.05)
        self.a_log = nn.Parameter(a_log)
        a_im = torch.linspace(0.0, 3.1416, d_state)[None, :].repeat(n_heads, 1)
        self.a_im = nn.Parameter(a_im.clone())
        # Input map B (per head/state) and output map C (per head/channel/state).
        self.B = nn.Parameter(torch.ones(n_heads, d_state))
        self.C = nn.Parameter(torch.randn(n_heads, self.dh, d_state, 2) * (d_state ** -0.5))
        self.D = nn.Parameter(torch.zeros(n_heads, self.dh))  # skip / feedthrough

    def _poles(self, device, dtype):
        mag = torch.exp(-F.softplus(self.a_log))              # |A| in (0,1)
        ang = self.a_im
        return torch.complex(mag * torch.cos(ang), mag * torch.sin(ang))  # [nh, ds]

    def _kernel(self, device, dtype):
        """Real conv kernel k[nh, dh, T]."""
        A = self._poles(device, dtype)                       # [nh, ds] complex
        n = torch.arange(self.T, device=A.device).float()    # float exponent for complex pow
        An = A[:, :, None] ** n[None, None, :]               # [nh, ds, T]
        C = torch.view_as_complex(self.C.to(torch.float32))  # [nh, dh, ds]
        CB = C * self.B[:, None, :]                          # [nh, dh, ds]
        k = torch.einsum("hds,hst->hdt", CB, An).real        # [nh, dh, T]
        return k.to(dtype)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        u = self.pi(x).view(B, self.T, self.H, self.W, self.nh, self.dh)
        u = u.permute(0, 4, 5, 2, 3, 1).contiguous()         # [B, nh, dh, H, W, T]
        u = u.float()
        k = self._kernel(u.device, u.dtype)                  # [nh, dh, T]

        L = 2 * self.T                                        # zero-pad -> causal linear conv
        Uf = torch.fft.rfft(u, n=L, dim=-1)
        Kf = torch.fft.rfft(k, n=L, dim=-1)[None, :, :, None, None, :]
        y = torch.fft.irfft(Uf * Kf, n=L, dim=-1)[..., :self.T]
        y = y + self.D[None, :, :, None, None, None] * u     # feedthrough
        y = y.permute(0, 5, 3, 4, 1, 2).reshape(B, N, D)     # back to [B, T, H, W, nh, dh]
        return self.po(y.to(x.dtype))

    # ---- O(1)-state streaming ------------------------------------------------
    def init_state(self, B, device):
        """Recurrent state for step(): complex [B, H*W, nh, dh, d_state]."""
        return torch.zeros(B, self.H * self.W, self.nh, self.dh, self.d_state,
                           dtype=torch.cfloat, device=device)

    def state_bytes(self, B=1, device="cpu"):
        """Persistent recurrent-state size in bytes (the R14 killer metric):
        B * H*W * n_heads * dh * d_state complex64 -- independent of horizon T."""
        s = self.init_state(B, device)
        return s.element_size() * s.nelement()

    def step(self, x_t, state):
        """Advance one frame. x_t: [B, H*W, D]; returns ([B, H*W, D], new_state).

        Equivalent to the frame-t slice of forward() but O(d_state) memory in T,
        so it supports unbounded autoregressive rollout at constant cost."""
        B, S, D = x_t.shape
        u = self.pi(x_t).view(B, S, self.nh, self.dh).float()   # [B,S,nh,dh]
        A = self._poles(x_t.device, torch.float32)               # [nh, ds]
        Bmat = self.B                                            # [nh, ds]
        # x_s = A * x_{s-1} + B * u_t   (broadcast B over channels)
        state = A[None, None, :, None, :] * state \
            + (Bmat[None, None, :, None, :] * u[..., None]).to(torch.cfloat)
        C = torch.view_as_complex(self.C.to(torch.float32))      # [nh, dh, ds]
        y = torch.einsum("hcz,bnhcz->bnhc", C, state).real       # reduce over state z
        y = y + self.D[None, None, :, :] * u
        y = y.reshape(B, S, D)
        return self.po(y.to(x_t.dtype)), state
