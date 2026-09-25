"""Fast CPU sanity checks for the v2 mixer paths (not a training run).

Each line prints OK/FAIL. This is the real correctness gate before smokes."""
import torch
from wfvideo import WaveMix3D, AttnMix, VideoPredictor
from ssm_lite import SSMLite
from train_compare import centroids_by_color, divergence_horizon

torch.manual_seed(0)
B, dim, nh, T, H, W = 2, 32, 4, 5, 6, 7   # non-square H!=W to catch the axis bug
N = T * H * W
x = torch.randn(B, N, dim)
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"{name:42s} {'OK' if cond else 'FAIL'}")


# 1) separable wave on a non-square grid (v1 axis bug would broadcast-fail).
check("separable wave non-square shape", WaveMix3D(dim, nh, T, H, W)(x).shape == x.shape)

# 2) causal_time zeroes the future-reaching (dt<0) taps.
kt, _, _ = WaveMix3D(dim, nh, T, H, W, causal_time=True)._kernels_1d()
dt = (torch.arange(T) - T // 2).float()
check("causal_time zeroes dt<0 taps", kt[:, dt < 0].abs().max().item() == 0.0)

# 3) linear_pad separable finite.
check("separable linear_pad finite",
      torch.isfinite(WaveMix3D(dim, nh, T, H, W, linear_pad=True)(x)).all())

# 4) dispersion kernel: correct shape + finite + stable across many modes.
wd = WaveMix3D(dim, nh, T, H, W, kernel_version="dispersion", linear_pad=True, n_modes=4)
yd = wd(x)
check("dispersion shape + finite", yd.shape == x.shape and torch.isfinite(yd).all())
# stability: all poles strictly inside the unit circle -> |lambda|<1.
import torch.nn.functional as F
with torch.no_grad():
    knorm = torch.linspace(0, 6, 20)
    lam_mag = torch.exp(-F.softplus(wd.a0[:, :, None] + wd.a1[:, :, None] * knorm))
check("dispersion poles |lambda|<1", lam_mag.max().item() < 1.0)

# 5) gate + local-fuse paths finite (both kernels).
for kv in ["separable", "dispersion"]:
    w = WaveMix3D(dim, nh, T, H, W, kernel_version=kv, linear_pad=True,
                  gate=True, local_fuse=True)
    check(f"{kv} +gate +local finite", torch.isfinite(w(x)).all())

# 6) AttnMix causal mask: perturbing the last frame must not change earlier outputs.
am = AttnMix(dim, nh, T, H, W, causal=True).eval()
with torch.no_grad():
    x2 = x.clone(); x2[:, (T - 1) * H * W:] += 5.0
    early = slice(0, (T - 1) * H * W)
    leak = (am(x)[:, early] - am(x2)[:, early]).abs().max().item()
check("attn causal no future-leak", leak < 1e-5)

# 7) SSMLite forward == unrolled step() recurrence.
s = SSMLite(dim, nh, T, H, W, d_state=8).eval()
with torch.no_grad():
    yf = s(x).view(B, T, H * W, dim)
    xt = x.view(B, T, H * W, dim)
    outs, st = [], s.init_state(B, x.device)
    for t in range(T):
        o, st = s.step(xt[:, t], st)
        outs.append(o)
    ys = torch.stack(outs, 1)
check("ssm forward == step()", (yf - ys).abs().max().item() < 1e-3)

# 7b) dispersion wave forward() == unrolled step() recurrence (causal, tol 1e-3).
# Own (larger) T so the L=2T time-aliasing |lam|^{T+1} sits far below tol.
Td = 16
xd = torch.randn(B, Td * H * W, dim)
wd2 = WaveMix3D(dim, nh, Td, H, W, kernel_version="dispersion", linear_pad=True,
                gate=False, local_fuse=False, n_modes=3).eval()
with torch.no_grad():
    yf = wd2(xd).view(B, Td, H * W, dim)
    xt = xd.view(B, Td, H * W, dim)
    outs, st = [], wd2.init_state(B, xd.device)
    for t in range(Td):
        o, st = wd2.step(xt[:, t], st)
        outs.append(o)
    ys = torch.stack(outs, 1)
check("dispersion forward == step()", (yf - ys).abs().max().item() < 1e-3)

# 8) residual zero-init: at init delta==0, so prediction == last input frame exactly.
frames = torch.rand(B, T, 3, H, W)
mr = VideoPredictor(dim, 2, nh, T, H, W, "wave", residual=True,
                    kernel_version="dispersion", linear_pad=True).eval()
with torch.no_grad():
    check("residual zero-init == copy-last",
          (mr(frames) - frames[:, -1]).abs().max().item() < 1e-6)

# 8b) whole-model streaming: ingesting T context frames via stream_step must
# reproduce the windowed forward() prediction (per-frame step==forward composes
# through every block; posemb pt[t] matches for t<T). Dispersion, no gate/local.
ms = VideoPredictor(dim, 2, nh, Td, H, W, "wave", residual=True,
                    kernel_version="dispersion", linear_pad=True).eval()
fd = torch.rand(B, Td, 3, H, W)
with torch.no_grad():
    pred_fwd = ms(fd)
    sts = ms.stream_init(B, fd.device)
    pred_stream = None
    for t in range(Td):
        pred_stream, sts = ms.stream_step(fd[:, t], sts, t)
    check("wave stream_step == forward (T frames)",
          (pred_fwd - pred_stream).abs().max().item() < 1e-3)

# 9) end-to-end predictor for all arms (dispersion wave + gate + local + causal).
for kind, kw in [("wave", dict(kernel_version="dispersion", linear_pad=True,
                               gate=True, local_fuse=True)),
                 ("attn", {}), ("ssm", {})]:
    m = VideoPredictor(dim, 2, nh, T, H, W, kind, causal=True, residual=True, **kw).eval()
    out = m(frames)
    check(f"predictor {kind} end-to-end", out.shape == (B, 3, H, W) and torch.isfinite(out).all())

# 10) metric helpers: color-matched centroid recovers a planted ball; horizon logic.
img = torch.zeros(1, 3, 10, 10); img[0, 0, 7, 3] = 1.0    # red pixel at (x=3,y=7)
cxy = centroids_by_color(img, torch.tensor([[[1.0, 0, 0]]]))
check("color centroid finds planted ball", abs(cxy[0, 0, 0] - 3) < 0.5 and abs(cxy[0, 0, 1] - 7) < 0.5)
med = torch.tensor([0.0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0])   # 8-long run of >1.6 starts at idx 2
check("divergence_horizon consec logic", divergence_horizon(med, 1.6, 8) == 2)

print("SANITY", "ALL-OK" if ok_all else "FAILURES-PRESENT")
