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

    def test_flow_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            _flow(**{}).set_flow_sampler(n_steps=0)
        with self.assertRaises(ValueError):
            VideoPredictor(DIM, 1, 4, T, H, W, "wave", head="flow", flow_steps=0)
        import train_long
        with self.assertRaises(SystemExit):
            train_long.main(["--head", "flow", "--flow-steps", "0", "--seq-frames", "8",
                             "--chunk", "4", "--steps", "1", "--out", "unused"])

    def test_stream_sampling_is_keyed_to_the_frame_index(self):
        # a stream resumed from saved state must sample what the uninterrupted one
        # would: the draw depends on (seed, t_index), not on how many came before
        import copy
        m = _flow().eval()
        m.set_flow_sampler(seed=7)
        frame = torch.rand(1, 3, H, W)
        with torch.no_grad():
            _, st = m.stream_step(frame, m.stream_init(1, "cpu"), 0)
            a, _ = m.stream_step(frame, copy.deepcopy(st), 5)
            m.stream_step(frame, copy.deepcopy(st), 3)          # an unrelated draw
            b, _ = m.stream_step(frame, copy.deepcopy(st), 5)
            c, _ = m.stream_step(frame, copy.deepcopy(st), 6)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_trainer_flow_resume_equals_uninterrupted_and_stream_loads_it(self):
        import tempfile
        import long_horizon as lh
        import train_long
        base = ["--head", "flow", "--flow-steps", "2", "--seq-frames", "8", "--chunk", "4",
                "--grid", "4", "--dim", "8", "--layers", "1", "--heads", "2", "--batch", "2",
                "--eval-rollout", "2", "--eval-seeds", "1", "--eval-chunk", "1",
                "--eval-batch", "1", "--save-every", "1"]
        with tempfile.TemporaryDirectory() as d:
            full, split = Path(d) / "full", Path(d) / "split"
            train_long.main(base + ["--steps", "3", "--out", str(full)])
            train_long.main(base + ["--steps", "1", "--out", str(split)])
            torch.manual_seed(999)                  # a resumed process has its own global RNG
            torch.rand(17)
            train_long.main(base + ["--steps", "3", "--out", str(split),
                                    "--resume", str(split / "ckpt_wave.pt")])
            f = torch.load(full / "model_wave.pt", weights_only=True)
            s = torch.load(split / "model_wave.pt", weights_only=True)
            self.assertTrue(any(k.startswith("flow.") for k in f["state"]))
            for k in f["state"]:
                self.assertTrue(torch.equal(f["state"][k], s["state"][k]), k)
            # the long-horizon stream rebuilds the flow head from the checkpoint
            # (a residual rebuild fails the strict load) and samples reproducibly
            cfg = f["config"]
            args = ["stream", "--pole-param", cfg["pole_param"], "--time-pos", "none",
                    "--ckpt", str(full / "model_wave.pt"), "--grid", "4", "--frames", "4",
                    "--dim", "8", "--layers", "1", "--heads", "2", "--stream-frames", "8",
                    "--chunk", "4", "--batch", "1", "--device", "cpu"]
            r1, r2 = lh.main(args), lh.main(args)
            self.assertTrue(r1["trained"])
            self.assertEqual([row["last"] for row in r1["log"]], [row["last"] for row in r2["log"]])
            # ... and a sliced stream (--checkpoint, resumed) samples what the
            # uninterrupted one did: the draw is keyed to the frame, not the RNG
            cp = str(Path(d) / "stream.pt")
            sliced = [a if a != "8" or args[i - 1] != "--stream-frames" else "4"
                      for i, a in enumerate(args)] + ["--checkpoint", cp]
            lh.main(sliced)
            torch.manual_seed(12345)
            r3 = lh.main(args + ["--checkpoint", cp])
            self.assertEqual(r1["log"][-1]["last"], r3["log"][-1]["last"])
            # render_rollout: the same command renders the same sampled video
            import render_rollout
            pngs = []
            for i in range(2):
                torch.manual_seed(100 + i)              # different ambient RNG each time
                out = Path(d) / f"render{i}"
                render_rollout.render(render_rollout.parser().parse_args(
                    ["--ckpt", str(full / "model_wave.pt"), "--out", str(out), "--frames", "4",
                     "--device", "cpu"]))
                pngs.append([p.read_bytes() for p in sorted((out / "prediction").glob("*.png"))])
            self.assertTrue(pngs[0])
            self.assertEqual(pngs[0], pngs[1])


if __name__ == "__main__":
    unittest.main()
