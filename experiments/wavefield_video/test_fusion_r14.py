"""R14 gated-fusion wrapper checks; run directly, no pytest dependency.

Mirrors sanity_check.py's streaming-parity tests for the new local+global hybrid
mixer and the persistent-state-byte accounting that the occlusion probe reports."""
import torch

from wfvideo import FusedMix, VideoPredictor, AttnMix, WaveMix3D
from ssm_lite import SSMLite

torch.manual_seed(0)
B, dim, nh, H, W = 2, 32, 4, 6, 7
Td = 16
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"{name:48s} {'OK' if cond else 'FAIL'}")


# 1) FusedMix forward is finite and gate-shaped for both global kinds.
x = torch.randn(B, Td * H * W, dim)
for fuse in ("local_ssm", "local_wave"):
    fm = FusedMix(dim, nh, Td, H, W, fuse, causal=True, linear_pad=True).eval()
    y = fm(x)
    check(f"{fuse} forward shape + finite", y.shape == x.shape and torch.isfinite(y).all())

# 2) FusedMix streaming reproduces forward at the last frame of a T-frame clip
#    (causal local window + O(1) global recurrence compose to the joint forward).
for fuse in ("local_ssm", "local_wave"):
    fm = FusedMix(dim, nh, Td, H, W, fuse, causal=True, linear_pad=True).eval()
    with torch.no_grad():
        yf = fm(x).view(B, Td, H * W, dim)[:, -1]
        xt = x.view(B, Td, H * W, dim)
        st = fm.init_state(B, x.device)
        out = None
        for t in range(Td):
            out, st = fm.step(xt[:, t], st)
        check(f"{fuse} step()==forward (last frame)", (yf - out).abs().max().item() < 1e-3)

# 3) whole predictor: fused stream_step == windowed forward on a T-frame clip.
frames = torch.rand(B, Td, 3, H, W)
for kind, fuse in (("wave", "local_wave"), ("ssm", "local_ssm")):
    m = VideoPredictor(dim, 2, nh, Td, H, W, kind, causal=True, residual=True,
                       kernel_version="dispersion", linear_pad=True, fuse=fuse).eval()
    with torch.no_grad():
        pred_fwd = m(frames)
        sts = m.stream_init(B, frames.device)
        pred_stream = None
        for t in range(Td):
            pred_stream, sts = m.stream_step(frames[:, t], sts, t)
        check(f"{fuse} predictor stream==forward",
              (pred_fwd - pred_stream).abs().max().item() < 1e-3)

# 4) predictor still builds for every occlusion arm and is finite end-to-end.
for kind, kw in [("attn", {}), ("ssm", {}),
                 ("wave", dict(kernel_version="dispersion", linear_pad=True)),
                 ("ssm", dict(fuse="local_ssm")),
                 ("wave", dict(kernel_version="dispersion", linear_pad=True, fuse="local_wave"))]:
    m = VideoPredictor(dim, 2, nh, Td, H, W, kind, causal=True, residual=True, **kw).eval()
    tag = kw.get("fuse", kind)
    check(f"predictor {tag} end-to-end finite",
          m(frames).shape == (B, 3, H, W) and torch.isfinite(m(frames)).all())

# 5) fuse/kind mismatch is rejected (config, streaming and state bytes must agree).
try:
    VideoPredictor(dim, 1, nh, Td, H, W, "wave", fuse="local_ssm")
    check("fuse/kind mismatch rejected", False)
except ValueError:
    check("fuse/kind mismatch rejected", True)

# 6) STATE BYTES: attention has no persistent recurrence; recurrent arms do; a
#    hybrid's persistent state equals its global path's (the local window is O(T),
#    not persistent, so it is NOT counted here).
attn = AttnMix(dim, nh, Td, H, W, causal=True)
ssm = SSMLite(dim, nh, Td, H, W)
wave = WaveMix3D(dim, nh, Td, H, W, kernel_version="dispersion", linear_pad=True)
hyb_s = FusedMix(dim, nh, Td, H, W, "local_ssm")
hyb_w = FusedMix(dim, nh, Td, H, W, "local_wave")
check("attn persistent state bytes == 0", attn.state_bytes(1) == 0)
check("ssm persistent state bytes > 0", ssm.state_bytes(1) > 0)
check("wave persistent state bytes > 0", wave.state_bytes(1) > 0)
check("hybrid state bytes == global path", hyb_s.state_bytes(1) == ssm.state_bytes(1)
      and hyb_w.state_bytes(1) == wave.state_bytes(1))

# 7) predictor-level persistent bytes scale with layers and stay 0 for attention.
m_attn = VideoPredictor(dim, 3, nh, Td, H, W, "attn", causal=True)
m_ssm = VideoPredictor(dim, 3, nh, Td, H, W, "ssm", causal=True)
check("predictor attn persistent bytes == 0", m_attn.persistent_state_bytes(1) == 0)
check("predictor ssm persistent bytes == 3 layers",
      m_ssm.persistent_state_bytes(1) == 3 * ssm.state_bytes(1))

print("SANITY-R14", "ALL-OK" if ok_all else "FAILURES-PRESENT")
if not ok_all:
    raise SystemExit(1)
