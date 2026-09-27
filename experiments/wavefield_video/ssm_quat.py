"""Quaternion-state diagonal SSM -- the E1 hypercomplex arm (study: HYPERCOMPLEX_STUDY.md §4-E1).

Same interface as ``SSMLite`` / ``WaveMix3D`` / ``AttnMix``:
  * ``pi`` (Linear in) / ``po`` (Linear out) submodules,
  * ``forward([B, N, dim]) -> [B, N, dim]`` with N = T*H*W in (t, y, x) order,
  * mixes causally over time only (spatial cells are independent channels),
  * ``init_state(B, device)`` / ``step(x_t, state)`` / ``state_bytes(B, device)``.

At equal persistent bytes: the state is ``[B, HW, nh, dhq, ds, 4]`` float32 with a
trailing quaternion-component axis, dhq = dh // 2 (dh = dim / n_heads). Per
(HW, nh, ds) cell that is dhq quaternions = dhq * 4 float32 = dhq * 16 B =
8*dh bytes -- exactly SSMLite's dh complex64 = dh * 8 B = 8*dh bytes;
``state_bytes()`` equals ``SSMLite``'s by construction (asserted in
``test_hypercomplex_r15.py``).

Dynamics (per (nh, dhq, ds) slot, Hamilton products ``qmul``):
    S <- A ⊗ S + B ⊗ u        A = m ⊙ U  (m = exp(-softplus(a_log)) ~ 0.95,
                              U = unit quaternion per (nh, ds))
    y  = sum_ds C ⊗ S  (all four components read out) + D ⊙ u

Training path: ``forward()`` evaluates the exact closed-form impulse response
    K[n] = sum_s C_s ⊗ A_s^n ⊗ B_s,  A^n = m^n * (cos n*ang, sin n*ang * axis)
    y_t  = sum_{k<=t} K[t-k] ⊗ u_k
as a causal linear convolution via real FFTs over time (linear, zero-padded to
2T -- no wraparound). This is algebraically the unrolled recurrence (the same
FFT-conv trick SSMLite uses over its complex kernel); the direct unrolled
recurrence would keep T=17 full states per layer alive for backward
(~13.7 GB/layer at the occlusion suite's batch 8 / grid 64 -- not viable).
``step()`` runs the O(1) recurrence directly; forward/step parity is asserted
in the tests at the standard 1e-3 fp32-FFT tolerance used for ssm/wave.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def qmul_basis(x):
    """[e_p ⊗ x] for p = 0..3 as 4 signed permutations of the trailing 4-axis.

    Used twice: (a) in QuaternionLinear-style maps to turn a quaternion product
    into 4 real linear maps; (b) in the freq-domain conv below."""
    x0, x1, x2, x3 = x.unbind(-1)
    return [torch.stack([x0, x1, x2, x3], -1),
            torch.stack([-x1, x0, -x3, x2], -1),
            torch.stack([-x2, x3, x0, -x1], -1),
            torch.stack([-x3, -x2, x1, x0], -1)]


class QuatSSM(nn.Module):
    def __init__(self, dim, n_heads, T, H, W, d_state=16, causal=True):
        super().__init__()
        if dim % (2 * n_heads):
            raise ValueError("QuatSSM needs dim divisible by 2*n_heads (dh even)")
        self.nh = n_heads
        self.dh = dim // n_heads
        self.dhq = self.dh // 2
        self.T, self.H, self.W = T, H, W
        self.d_state = d_state
        self.causal = causal  # causal by construction; flag kept for interface parity
        self.pi = nn.Linear(dim, 2 * dim)      # reals -> quaternion groups [nh, dhq, 4]
        self.po = nn.Linear(2 * dim, dim)

        # Poles A = m ⊙ U: same magnitude parameterization as SSMLite, with the
        # angle spread about per-head fixed random axes instead of the complex
        # unit circle. Angle is learnable (mirrors SSMLite's a_im); axis is a buffer.
        a_log = torch.log(torch.expm1(torch.full((n_heads, d_state), 0.05)))
        self.a_log = nn.Parameter(a_log)
        self.a_ang = nn.Parameter(torch.linspace(0.0, 3.1416, d_state)[None, :]
                                  .repeat(n_heads, 1).clone())
        axis = torch.randn(n_heads, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("axis", axis)     # per-head fixed rotation axis

        Bq = torch.zeros(n_heads, d_state, 4)
        Bq[..., 0] = 1.0                       # identity input map, as SSMLite's ones
        self.B = nn.Parameter(Bq)
        self.C = nn.Parameter(torch.randn(n_heads, self.dhq, d_state, 4) * (d_state ** -0.5))
        self.D = nn.Parameter(torch.zeros(n_heads, self.dhq, 4))   # per-component skip

    # ---- shared math ---------------------------------------------------------
    def _pole(self, device):
        """A [nh, ds, 4] = m * (cos ang, sin ang * axis)."""
        mag = torch.exp(-F.softplus(self.a_log))                    # [nh, ds]
        ca, sa = torch.cos(self.a_ang), torch.sin(self.a_ang)
        return torch.stack([mag * ca,
                            mag * sa * self.axis[:, None, 0],
                            mag * sa * self.axis[:, None, 1],
                            mag * sa * self.axis[:, None, 2]], dim=-1)

    def _kernel(self, device, dtype):
        """K [nh, dhq, T, 4]: exact impulse response sum_s C ⊗ A^n ⊗ B."""
        m = torch.exp(-F.softplus(self.a_log))                      # [nh, ds]
        n = torch.arange(self.T, device=device, dtype=torch.float32)
        mag = m[..., None] ** n                                     # [nh, ds, T]
        ang = self.a_ang[..., None] * n                             # [nh, ds, T]
        ca, sa = torch.cos(ang), torch.sin(ang)
        An = torch.stack([mag * ca,
                          mag * sa * self.axis[:, None, None, 0],
                          mag * sa * self.axis[:, None, None, 1],
                          mag * sa * self.axis[:, None, None, 2]], dim=-1)   # [nh,ds,T,4]
        M1 = qmul(An, self.B[:, :, None, :])                        # [nh,ds,T,4]
        K = qmul(self.C[:, :, :, None, :], M1[:, None])             # [nh,dhq,ds,T,4]
        return K.sum(dim=2).to(dtype)                               # [nh,dhq,T,4]

    # ---- training path: causal FFT conv of the impulse response ---------------
    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        u = self.pi(x).view(B, self.T, self.H, self.W, self.nh, self.dhq, 4)
        u = u.float().permute(0, 4, 5, 2, 3, 1, 6)                  # [B,nh,dhq,H,W,T,4]
        u = u.reshape(B, self.nh, self.dhq, self.H * self.W, self.T, 4)

        K = self._kernel(u.device, torch.float32)                   # [nh,dhq,T,4]
        L = 2 * self.T                                              # zero-pad: causal conv
        Kp = [torch.fft.rfft(K[..., p], n=L, dim=-1)[None, :, :, None, :] for p in range(4)]
        uh = [torch.fft.rfft(u[..., p], n=L, dim=-1) for p in range(4)]   # [B,nh,dhq,HW,L/2+1]
        ys = [torch.fft.irfft(Kp[0] * uh[0] - Kp[1] * uh[1] - Kp[2] * uh[2] - Kp[3] * uh[3],
                              n=L, dim=-1)[..., :self.T],
              torch.fft.irfft(Kp[0] * uh[1] + Kp[1] * uh[0] + Kp[2] * uh[3] - Kp[3] * uh[2],
                              n=L, dim=-1)[..., :self.T],
              torch.fft.irfft(Kp[0] * uh[2] - Kp[1] * uh[3] + Kp[2] * uh[0] + Kp[3] * uh[1],
                              n=L, dim=-1)[..., :self.T],
              torch.fft.irfft(Kp[0] * uh[3] + Kp[1] * uh[2] - Kp[2] * uh[1] + Kp[3] * uh[0],
                              n=L, dim=-1)[..., :self.T]]
        y = torch.stack(ys, dim=-1)                                 # [B,nh,dhq,HW,T,4]
        y = y + self.D[None, :, :, None, None, :] * u               # feedthrough
        y = y.permute(0, 4, 3, 1, 2, 5).reshape(B, N, 2 * D)        # (t, y, x, nh, dhq, comp)
        return self.po(y.to(x.dtype))

    # ---- O(1)-state streaming -------------------------------------------------
    def init_state(self, B, device):
        """Recurrent state [B, HW, nh, dhq, ds, 4] float32 (trailing component axis)."""
        return torch.zeros(B, self.H * self.W, self.nh, self.dhq, self.d_state, 4,
                           dtype=torch.float32, device=device)

    def state_bytes(self, B=1, device="cpu"):
        """Persistent recurrent-state size in bytes -- equals SSMLite's at the same
        (dim, heads, H, W): 16 B per quaternion float32 vs 8 B per complex64, with
        half the per-head slots (dhq = dh/2)."""
        s = self.init_state(B, device)
        return s.element_size() * s.nelement()

    def step(self, x_t, state):
        """Advance one frame. x_t: [B, H*W, D]; returns ([B, H*W, D], new_state)."""
        B, S, D = x_t.shape
        u = self.pi(x_t).view(B, S, self.nh, self.dhq, 4).float()   # [B,S,nh,dhq,4]
        A = self._pole(x_t.device)                                  # [nh, ds, 4]
        state = (qmul(A[None, None, :, None], state)
                 + qmul(self.B[None, None, :, None], u[..., None, :]))
        y = qmul(self.C[None, None], state).sum(dim=-2)             # [B,S,nh,dhq,4]
        y = y + self.D[None, None] * u
        return self.po(y.reshape(B, S, 2 * D).to(x_t.dtype)), state
