"""Run: python -m unittest discover -s tests -p 'test_stateful.py' -v.

LONG_HORIZON.md phase 1: carried-state chunks, dense next-frame loss,
time-invariant positions, causal gate."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wfvideo import VideoPredictor, WaveMix3D

N, TC, H, W, DIM, NH = 12, 4, 5, 4, 32, 4
ARMS = [
    ("wave", dict(kernel_version="dispersion", linear_pad=True)),
    ("wave", dict(kernel_version="dispersion", linear_pad=True, pole_param="halflife",
                  hl_max=64.0)),
    ("ssm", {}),
]


def _pair(kind, kw):
    """Same weights at two window lengths (possible because time_pos='none'
    makes every parameter independent of T)."""
    torch.manual_seed(0)
    long = VideoPredictor(DIM, 2, NH, N, H, W, kind, causal=True, time_pos="none", **kw)
    torch.manual_seed(1)
    torch.nn.init.normal_(long.head.weight, std=0.1)   # zero-init head would hide mismatches
    chunk = VideoPredictor(DIM, 2, NH, TC, H, W, kind, causal=True, time_pos="none", **kw)
    chunk.load_state_dict(long.state_dict())
    return long.eval(), chunk.eval()


class StatefulTests(unittest.TestCase):
    def test_chunked_equals_full_equals_stream(self):
        fr = torch.rand(2, N, 3, H, W)
        for kind, kw in ARMS:
            with self.subTest(kind=kind, pole=kw.get("pole_param", "softplus")):
                long, ch = _pair(kind, kw)
                with torch.no_grad():
                    full, _ = long(fr, states=[None, None], dense=True)
                    sts, outs = [None, None], []
                    for c in range(N // TC):
                        p, sts = ch(fr[:, c * TC:(c + 1) * TC], states=sts, dense=True)
                        outs.append(p)
                    chunked = torch.cat(outs, 1)
                    st, so = ch.stream_init(2, "cpu"), []
                    for t in range(N):
                        o, st = ch.stream_step(fr[:, t], st, t)
                        so.append(o)
                    stream = torch.stack(so, 1)
                self.assertGreater((full - fr).abs().max().item(), 1e-2)   # non-trivial
                self.assertLess((chunked - full).abs().max().item(), 1e-4)
                self.assertLess((chunked - stream).abs().max().item(), 1e-4)
                for a, b in zip(sts, st):                                  # same carried state
                    self.assertLess((a - b).abs().max().item(), 1e-4)

    def test_dense_last_position_equals_default_forward(self):
        torch.manual_seed(0)
        m = VideoPredictor(DIM, 2, NH, TC, H, W, "wave", causal=True,
                           kernel_version="dispersion", linear_pad=True)
        torch.nn.init.normal_(m.head.weight, std=0.1)
        fr = torch.rand(2, TC, 3, H, W)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(fr, dense=True)[:, -1], m(fr), atol=1e-6))

    def test_carried_state_actually_carries(self):
        """Chunk 2 must depend on chunk 1 through the state (else 'carry' is a no-op)."""
        _, ch = _pair(*ARMS[1])
        a, b = torch.rand(1, TC, 3, H, W), torch.rand(1, TC, 3, H, W)
        with torch.no_grad():
            _, s1 = ch(a, states=[None, None], dense=True)
            with_state, _ = ch(b, states=s1, dense=True)
            fresh, _ = ch(b, states=[None, None], dense=True)
        self.assertGreater((with_state - fresh).abs().max().item(), 1e-4)

    def test_stateful_rejects_unsupported_arms(self):
        for kw in (dict(kind="attn"), dict(kind="wave", kernel_version="dispersion", gate=True),
                   dict(kind="wave", kernel_version="separable")):
            with self.subTest(**kw):
                m = VideoPredictor(DIM, 1, NH, TC, H, W, causal=True, **kw)
                with self.assertRaises(ValueError):
                    m(torch.rand(1, TC, 3, H, W), states=[None])

    def test_time_pos_none_has_no_temporal_table(self):
        m = VideoPredictor(DIM, 1, NH, TC, H, W, "wave", kernel_version="dispersion",
                           time_pos="none")
        self.assertIsNone(m.posemb.pt)
        with self.assertRaises(ValueError):
            VideoPredictor(DIM, 1, NH, TC, H, W, "wave", time_pos="sinusoid")

    def test_causal_gate_does_not_read_the_future(self):
        torch.manual_seed(0)
        # halflife -> exact truncated kernel, so any leak left is the gate's own.
        # (The softplus default's circular kernel leaks ~0.7% per future frame at
        # T=4 -- measured in LONG_HORIZON.md -- which would mask the gate here.)
        w = WaveMix3D(DIM, NH, TC, H, W, kernel_version="dispersion", causal_time=True,
                      linear_pad=True, gate=True, pole_param="halflife")
        x = torch.randn(1, TC * H * W, DIM)
        x2 = x.clone()
        x2[:, (TC - 1) * H * W:] += 5.0                  # change only the last frame
        with torch.no_grad():
            y, y2 = w(x), w(x2)
        early = slice(0, (TC - 1) * H * W)
        self.assertLess((y - y2)[:, early].abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
