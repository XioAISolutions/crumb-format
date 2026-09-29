"""Run: python -m unittest discover -s tests -p 'test_flow_head.py' -v.

LONG_HORIZON.md phase 3: rectified-flow head on the causal backbone."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flow_head import FlowHead
from wfvideo import VideoPredictor

T, H, W, DIM = 4, 5, 4, 32


def _flow(**kw):
    torch.manual_seed(0)
    return VideoPredictor(DIM, 2, 4, T, H, W, "wave", causal=True, kernel_version="dispersion",
                          linear_pad=True, time_pos="none", head="flow", flow_steps=4, **kw)


class FlowHeadTests(unittest.TestCase):
    def test_loss_decreases_on_a_fixed_batch(self):
        torch.manual_seed(0)
        head = FlowHead(DIM, 3)
        c = torch.randn(8, H, W, DIM)
        x1 = torch.randn(8, 3, H, W) * 0.3
        opt = torch.optim.Adam(head.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(0)
        first = None
        for _ in range(200):
            loss = head.loss(x1, c, generator=g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), 0.6 * first)

    def test_seeded_sampling_is_reproducible_and_seed_dependent(self):
        m = _flow().eval()
        fr = torch.rand(2, T, 3, H, W)
        with torch.no_grad():
            m.set_flow_sampler(seed=7)
            a = m(fr)
            m.set_flow_sampler(seed=7)
            b = m(fr)
            m.set_flow_sampler(seed=8)
            c = m(fr)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_flow_works_with_carried_state_and_streaming(self):
        m = _flow().eval()
        fr = torch.rand(1, 2 * T, 3, H, W)
        loss, st = m.flow_loss(fr[:, :T], fr[:, 1:T + 1], states=[None, None])
        self.assertTrue(torch.isfinite(loss))
        loss2, _ = m.flow_loss(fr[:, T:2 * T], torch.roll(fr, -1, 1)[:, T:2 * T], states=st)
        self.assertTrue(torch.isfinite(loss2))
        with torch.no_grad():
            s = m.stream_init(1, "cpu")
            o, s = m.stream_step(fr[:, 0], s, 0)
        self.assertEqual(o.shape, (1, 3, H, W))

    def test_config_errors_and_residual_default(self):
        with self.assertRaises(ValueError):
            VideoPredictor(DIM, 1, 4, T, H, W, "wave", head="diffusion")
        with self.assertRaises(ValueError):
            VideoPredictor(DIM, 1, 4, T, H, W, "wave", head="flow", residual=False)
        m = VideoPredictor(DIM, 1, 4, T, H, W, "wave", kernel_version="dispersion")
        self.assertFalse(hasattr(m, "flow"))
        with self.assertRaises(ValueError):
            m.flow_loss(torch.rand(1, T, 3, H, W), torch.rand(1, T, 3, H, W))

    def test_trainer_refuses_flow_with_self_rollout(self):
        # rollout_sequence_loss supervises an MSE on the model's own feedback;
        # it has no flow objective, so the combination must fail closed.
        import train_long
        with self.assertRaises(SystemExit):
            train_long.main(["--head", "flow", "--rollout-k", "1", "--seq-frames", "8",
                             "--chunk", "4", "--steps", "1", "--out", "unused.json"])


if __name__ == "__main__":
    unittest.main()
