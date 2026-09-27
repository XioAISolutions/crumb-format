"""R15 hypercomplex-arm tests: module algebra, invariants, wiring, byte parity.

Run directly (``python test_hypercomplex_r15.py``); no pytest dependency. Covers
the three arms of reports/HYPERCOMPLEX_STUDY.md: E1 qssm (ssm_quat.QuatSSM,
--kind qssm), E2 qwave (wfvideo_quat.WaveQuatMix, --q-mix) and E3 qcolor
(wfvideo_quat.QuatEmbed, --quat-color).

The load-bearing invariants:
  * qmul / qmul_basis agree with the explicit 4x4 left-multiplication matrices;
  * QuatSSM.state_bytes == SSMLite.state_bytes (the per-byte comparison premise);
  * QuatSSM FFT-conv training path == O(1) step() recurrence (fp32 and fp64);
  * FFT conv == explicit sum_k K[t-k] (x) u_k from the module's own kernel;
  * default (flag-off) VideoPredictor init is byte-identical (same-seed hash);
  * flag scope errors fail fast at the CLI.
"""
import hashlib
import os
import subprocess
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import wfvideo                                            # noqa: E402
from ssm_lite import SSMLite                              # noqa: E402
from ssm_quat import QuatSSM, qmul, qmul_basis            # noqa: E402
from wfvideo_quat import QuaternionLinear, QuatEmbed, WaveQuatMix  # noqa: E402


def lmat(a):
    """Left-multiplication matrix: ``lmat(a) @ b == qmul(a, b)``."""
    a0, a1, a2, a3 = a.unbind(-1)
    return torch.stack([
        torch.stack([a0, -a1, -a2, -a3], -1),
        torch.stack([a1, a0, -a3, a2], -1),
        torch.stack([a2, a3, a0, -a1], -1),
        torch.stack([a3, -a2, a1, a0], -1)], dim=-2)


def sd_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_qmul_algebra():
    torch.manual_seed(0)
    a, b = torch.randn(5, 4), torch.randn(5, 4)
    assert (qmul(a, b) - (lmat(a) @ b[..., None])[..., 0]).abs().max() < 1e-6
    x = torch.randn(3, 4)
    for p in range(4):
        ep = torch.zeros(4)
        ep[p] = 1.0
        assert (qmul_basis(x)[p] - qmul(ep, x)).abs().max() < 1e-6, p
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    assert ((qmul(a, b).norm(dim=-1) - na * nb).abs().max() < 1e-5)
    print("OK quaternion algebra: qmul == Lmat, basis e_p, norm-multiplicative")


def test_qssm_interface_and_state_bytes():
    for dim, nh, T, H, W in ((32, 4, 6, 6, 7), (192, 8, 5, 8, 8)):
        q = QuatSSM(dim, nh, T, H, W)
        s = SSMLite(dim, nh, T, H, W)
        for attr in ("pi", "po", "forward", "init_state", "step", "state_bytes"):
            assert hasattr(q, attr), attr
        assert q.state_bytes(1, "cpu") == s.state_bytes(1, "cpu"), (dim, nh)
        st = q.init_state(2, "cpu")
        assert st.shape == (2, H * W, nh, dim // nh // 2, q.d_state, 4), st.shape
        assert st.dtype == torch.float32
    print("OK qssm interface + equal persistent bytes (dim 32 and 192)")


def test_qssm_forward_step_parity():
    B, dim, nh, T, H, W = 2, 32, 4, 8, 6, 7
    torch.manual_seed(1)
    q = QuatSSM(dim, nh, T, H, W).eval()
    x = torch.randn(B, T * H * W, dim)
    with torch.no_grad():
        yf = q(x).view(B, T, H * W, dim)
        xt = x.view(B, T, H * W, dim)
        outs, st = [], q.init_state(B, "cpu")
        for t in range(T):
            o, st = q.step(xt[:, t], st)
            outs.append(o)
        ys = torch.stack(outs, 1)
    d = (yf - ys).abs().max().item()
    assert d < 1e-3, d
    print(f"OK qssm forward == step chain (maxdiff {d:.2e})")


def test_qssm_fp64_kernel_vs_recurrence():
    B, dim, nh, T, H, W, ds = 2, 32, 4, 8, 6, 7, 8
    torch.manual_seed(2)
    qd = QuatSSM(dim, nh, T, H, W, d_state=ds).double().eval()
    x = torch.randn(B, T * H * W, dim)
    xt = x.view(B, T, H * W, dim)
    with torch.no_grad():
        yf64 = qd(x.double()).view(B, T, H * W, dim).double()
        st = torch.zeros(B, H * W, nh, qd.dhq, ds, 4, dtype=torch.float64)
        A = qd._pole("cpu")[None, None, :, None]            # [1,1,nh,1,ds,4]
        u = qd.pi(xt.double()).view(B, T, H * W, nh, qd.dhq, 4)
        outs = []
        for t in range(T):
            st = ((lmat(A) @ st[..., None])[..., 0]
                  + (lmat(qd.B[None, None, :, None]) @ u[:, t][..., None, :, None])[..., 0])
            y = (lmat(qd.C[None, None]) @ st[..., None])[..., 0].sum(dim=-2)  # sum over ds
            y = y + qd.D[None, None] * u[:, t]
            outs.append(qd.po(y.reshape(B, H * W, 2 * dim)))
        ys64 = torch.stack(outs, 1)
    # The module conv runs fp32 internally by design (u.float() + fp32 kernel,
    # mirroring SSMLite), so fp64 parity is bounded by fp32 rounding (~1e-6).
    d = (yf64 - ys64).abs().max().item()
    assert d < 1e-4, d
    print(f"OK qssm fp64 FFT-conv == fp64 Lmat recurrence (maxdiff {d:.2e})")


def test_qssm_conv_matches_kernel_sum():
    """FFT-conv output == explicit sum_k K[t-k] (x) u_k from the module kernel."""
    B, dim, nh, T, H, W = 1, 16, 2, 4, 2, 3
    torch.manual_seed(3)
    q = QuatSSM(dim, nh, T, H, W).eval()
    x = torch.randn(B, T * H * W, dim)
    with torch.no_grad():
        yf = q(x).view(B, T, H * W, dim)
        N = T * H * W
        u = q.pi(x).view(B, T, H, W, nh, q.dhq, 4)
        u = u.permute(0, 4, 5, 2, 3, 1, 6).reshape(B, nh, q.dhq, H * W, T, 4).float()[0]
        K = q._kernel(u.device, torch.float32)                       # [nh,dhq,T,4]
        outs = []
        for t in range(T):
            acc = torch.zeros(nh, q.dhq, H * W, 4)
            for k in range(t + 1):
                kn = lmat(K[:, :, t - k, :])                         # [nh,dhq,4(o),4(c)]
                acc = acc + torch.einsum("aboi,abei->abeo", kn, u[..., k, :])
            acc = acc + q.D[..., None, :] * u[..., t, :]
            outs.append(q.po(acc.permute(2, 0, 1, 3).reshape(H * W, nh * q.dhq * 4)))
        ys = torch.stack(outs, 1)                                    # [HW,T,dim]
    d = (ys.permute(1, 0, 2) - yf[0]).abs().max().item()
    assert d < 1e-4, d
    print(f"OK qssm FFT conv == K-sum conv (maxdiff {d:.2e})")


def test_quaternion_linear():
    dim, nh = 32, 4
    slots = dim // (4 * nh)
    ql = QuaternionLinear(dim, dim, nh)
    n = sum(p.numel() for p in ql.parameters())
    assert n == nh * slots * slots * 4 + dim, n
    assert n < dim * dim
    # scalar-rotation identity: W = alpha * e_0 component-only -> y = alpha * x
    torch.manual_seed(4)
    alpha = torch.rand(nh, slots, slots) + 0.5
    ql0 = QuaternionLinear(dim, dim, nh, bias=False)
    with torch.no_grad():
        ql0.weight.zero_()
        ql0.weight[..., 0] = alpha
        x = torch.randn(3, 11, dim)
        y = ql0(x)
        ref = torch.einsum("hoi,bshij->bshoj", alpha,
                           x.view(3, 11, nh, slots, 4)).reshape(3, 11, dim)
        assert (y - ref).abs().max() < 1e-6
    # dense per-slot reference: full Hamilton product, slot by slot
    ql2 = QuaternionLinear(dim, dim, nh, bias=False)
    x = torch.randn(5, dim)
    y2 = ql2(x).view(5, nh, slots, 4)
    xq = x.view(5, nh, slots, 4)
    for h in range(nh):
        for o in range(slots):
            acc = torch.zeros(5, 4)
            for i in range(slots):
                acc = acc + qmul(ql2.weight[h, o, i][None], xq[:, h, i])
            assert (acc - y2[:, h, o]).abs().max() < 1e-6, (h, o)
    print("OK QuaternionLinear: param formula, scalar-rotation identity, per-slot Hamilton")


def test_wave_quat_mix():
    dim, nh, T, H, W = 32, 4, 4, 2, 3
    torch.manual_seed(5)
    wq = WaveQuatMix(dim, nh, T, H, W, kernel_version="dispersion", causal_time=True)
    wr = wfvideo.WaveMix3D(dim, nh, T, H, W, kernel_version="dispersion", causal_time=True)
    assert isinstance(wq.pi, QuaternionLinear) and isinstance(wq.po, QuaternionLinear)
    assert wq.state_bytes(1, "cpu") == wr.state_bytes(1, "cpu")
    slots = dim // (4 * nh)
    q_mixer = sum(p.numel() for p in wq.parameters())
    real_mixer = sum(p.numel() for p in wr.parameters())
    expect = real_mixer - 2 * (dim * dim + dim) + 2 * (nh * slots * slots * 4 + dim)
    assert q_mixer == expect, (q_mixer, expect)
    x = torch.randn(2, T * H * W, dim)
    xt = x.view(2, T, H * W, dim)
    with torch.no_grad():
        yf = wq(x)
        assert yf.shape == x.shape and torch.isfinite(yf).all()
        st = wq.init_state(2, "cpu")
        outs = []
        for t in range(T):
            o, st = wq.step(xt[:, t], st)
            outs.append(o)
        d = (yf.view(2, T, H * W, dim) - torch.stack(outs, 1)).abs().max().item()
        st2 = wr.init_state(2, "cpu")
        outs2 = []
        for t in range(T):
            o, st2 = wr.step(xt[:, t], st2)
            outs2.append(o)
        dref = (wr(x).view(2, T, H * W, dim) - torch.stack(outs2, 1)).abs().max().item()
    assert d <= max(10 * dref, 1e-3), (d, dref)
    print(f"OK WaveQuatMix: pi/po 1/(4*{nh}) of Linear params, equal state bytes, "
          f"forward==step {d:.2e} (wave ref {dref:.2e})")


def test_quat_embed():
    torch.manual_seed(6)
    dim = 8
    emb = QuatEmbed(dim)
    slots = dim // 4
    nb = sum(p.numel() for p in emb.parameters())
    assert nb == dim * 9 + dim, nb                                  # 9 taps x dim + bias
    x = torch.rand(1, 3, 3, 3)
    with torch.no_grad():
        y = emb(x)
        assert y.shape == (1, dim, 3, 3) and torch.isfinite(y).all()
        W = emb.weight
        # manual Hamilton conv, tap by tap (independent of the channel trick)
        for oh in range(3):
            for ow in range(3):
                for slot in range(slots):
                    acc = torch.zeros(4)
                    for kh in range(3):
                        for kw in range(3):
                            ih, iw = oh + kh - 1, ow + kw - 1
                            if not (0 <= ih < 3 and 0 <= iw < 3):
                                continue
                            pix = torch.zeros(4)
                            pix[1:] = x[0, :, ih, iw]
                            acc = acc + qmul(W[slot, :, kh, kw][None], pix[None])[0]
                    ref = acc + emb.bias[slot * 4:(slot + 1) * 4]
                    got = y[0, slot * 4:(slot + 1) * 4, oh, ow]
                    assert (got - ref).abs().max() < 1e-5, (slot, oh, ow)
    emb.zero_grad()
    emb(x).sum().backward()
    assert emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0
    print("OK QuatEmbed: 9*dim params, manual Hamilton conv match, grads flow")


def test_videopredictor_wiring():
    from wfvideo import VideoPredictor
    dim, nh, T, H, W = 16, 2, 4, 4, 4
    frames = torch.rand(2, T, 3, H, W)
    # E1: qssm arm builds, forwards, streams
    m = VideoPredictor(dim, 1, nh, T, H, W, "qssm", causal=True, residual=True).eval()
    assert type(m.blocks[0].mix) is QuatSSM
    with torch.no_grad():
        assert m(frames).shape == (2, 3, H, W)
        st = m.stream_init(2, "cpu")
        nxt, st = m.stream_step(frames[:, 0], st, 0)
        assert nxt.shape == (2, 3, H, W) and torch.isfinite(nxt).all()
    assert m.persistent_state_bytes(1, "cpu") > 0
    # E2: q-mix coexists with the plain wave arm and keeps its state bytes
    wq = VideoPredictor(dim, 1, nh, T, H, W, "wave", causal=True,
                        kernel_version="dispersion", q_mix=True)
    wr = VideoPredictor(dim, 1, nh, T, H, W, "wave", causal=True,
                        kernel_version="dispersion")
    assert type(wq.blocks[0].mix).__name__ == "WaveQuatMix"
    assert isinstance(wq.embed, nn.Conv2d)
    assert wq.persistent_state_bytes(1, "cpu") == wr.persistent_state_bytes(1, "cpu")
    with torch.no_grad():
        assert wq(frames).shape == (2, 3, H, W)
    # E3: quaternion stem
    wc = VideoPredictor(dim, 1, nh, T, H, W, "wave", causal=True,
                        kernel_version="dispersion", quat_color=True)
    assert isinstance(wc.embed, QuatEmbed)
    with torch.no_grad():
        assert wc(frames).shape == (2, 3, H, W)
    # flag scope errors
    for kwargs in (dict(q_mix=True), dict(quat_color=True)):
        try:
            VideoPredictor(dim, 1, nh, T, H, W, "ssm", **kwargs)
            raise AssertionError(f"expected ValueError for kind ssm {kwargs}")
        except ValueError:
            pass
    try:
        VideoPredictor(dim, 1, nh, T, H, W, "wave", q_mix=True,
                       fuse="local_wave", kernel_version="dispersion")
        raise AssertionError("expected ValueError for --q-mix + --fuse")
    except ValueError:
        pass
    print("OK VideoPredictor wiring: qssm / wave+q-mix / wave+quat-color; scope enforced")


def test_default_paths_byte_identical():
    from wfvideo import VideoPredictor
    dim, nh, T, H, W = 16, 2, 4, 4, 4
    torch.manual_seed(7)
    a = VideoPredictor(dim, 2, nh, T, H, W, "wave", causal=True,
                       kernel_version="dispersion")
    torch.manual_seed(7)
    b = VideoPredictor(dim, 2, nh, T, H, W, "wave", causal=True,
                       kernel_version="dispersion", q_mix=False, quat_color=False)
    assert type(a.blocks[0].mix) is type(b.blocks[0].mix) is wfvideo.WaveMix3D
    assert isinstance(a.embed, nn.Conv2d) and isinstance(b.embed, nn.Conv2d)
    assert sd_hash(a) == sd_hash(b)
    print("OK default paths byte-identical (same-seed state_dict hash equality)")


def test_train_compare_cli_validation():
    base = ["train_compare.py", "--data-source", "occlusion",
            "--occ-start", "64", "--occ-end", "320"]
    cases = [
        (["--kind", "attn", "--q-mix"], "--q-mix / --quat-color apply to --kind wave"),
        (["--kind", "wave", "--kernel-version", "dispersion",
          "--fuse", "local_wave", "--q-mix"], "not supported together with --fuse"),
        (["--kind", "qssm", "--dim", "30", "--heads", "8"], "divisible by 2*--heads"),
    ]
    for extra, needle in cases:
        p = subprocess.run([sys.executable] + base + extra, cwd=HERE,
                           capture_output=True, text=True, timeout=300)
        assert p.returncode != 0, (extra, p.returncode)
        assert needle in p.stderr, (extra, p.stderr[-500:])
    print("OK train_compare CLI validation: flag scope + divisibility fail fast")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    test_qmul_algebra()
    test_qssm_interface_and_state_bytes()
    test_qssm_forward_step_parity()
    test_qssm_fp64_kernel_vs_recurrence()
    test_qssm_conv_matches_kernel_sum()
    test_quaternion_linear()
    test_wave_quat_mix()
    test_quat_embed()
    test_videopredictor_wiring()
    test_default_paths_byte_identical()
    test_train_compare_cli_validation()
    print("ALL r15 hypercomplex tests passed")
