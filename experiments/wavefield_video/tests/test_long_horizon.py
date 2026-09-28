"""Run: python -m unittest discover -s tests -p 'test_long_horizon.py' -v."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import long_horizon as lh
from wfvideo import VideoPredictor, WaveMix3D

DIM, NH, T, H, W = 32, 4, 8, 6, 5


def _randomize_head(m):
    # The residual head is zero-initialized, which makes every predictor output
    # "copy last frame" -- stream/forward parity would then hold trivially.
    torch.nn.init.normal_(m.head.weight, std=0.1)
    torch.nn.init.normal_(m.head.bias, std=0.1)
    return m


class PoleParamTests(unittest.TestCase):
    def test_softplus_default_is_the_v2_operator(self):
        """Default path keeps v2's parameters and pole math op-for-op."""
        torch.manual_seed(0)
        w = WaveMix3D(DIM, NH, T, H, W, kernel_version="dispersion", linear_pad=True)
        names = {n for n, _ in w.named_parameters()}
        self.assertTrue({"a0", "a1"} <= names)
        self.assertFalse({"hl_raw", "visc"} & names)
        lam, Hp, Wp = w._dispersion_lam("cpu")
        ky = (2 * math.pi) * torch.fft.fftfreq(Hp)
        kx = (2 * math.pi) * torch.fft.fftfreq(Wp)
        knorm = torch.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
        for m in range(w.n_modes):   # the pre-LONG_HORIZON formula, verbatim
            alpha = F.softplus(w.a0[:, m][:, None, None] + w.a1[:, m][:, None, None] * knorm)
            Omega = (w.vx[:, m][:, None, None] * kx[None, None, :]
                     + w.vy[:, m][:, None, None] * ky[None, :, None]
                     + w.beta[:, m][:, None, None] * knorm)
            self.assertTrue(torch.equal(lam[m], torch.exp(-alpha) * torch.exp(1j * Omega)))

    def test_halflife_init_spans_the_range_and_is_stable(self):
        w = WaveMix3D(DIM, NH, T, H, W, kernel_version="dispersion", linear_pad=True,
                      pole_param="halflife", hl_min=2.0, hl_max=4096.0)
        hl = w.half_lives().detach()
        self.assertGreaterEqual(hl.min().item(), 2.0)
        self.assertLessEqual(hl.max().item(), 4096.0)
        self.assertLess(hl.min().item(), 4.0)        # a fast mode exists ...
        self.assertGreater(hl.max().item(), 2048.0)  # ... and a minutes-scale one
        lam, _, _ = w._dispersion_lam("cpu")
        self.assertLess(lam.abs().max().item(), 1.0)

    def test_halflife_rejected_where_it_has_no_meaning(self):
        with self.assertRaises(ValueError):
            WaveMix3D(DIM, NH, T, H, W, kernel_version="separable", pole_param="halflife")
        with self.assertRaises(ValueError):
            VideoPredictor(DIM, 1, NH, T, H, W, "attn", pole_param="halflife")
        with self.assertRaises(ValueError):
            WaveMix3D(DIM, NH, T, H, W, kernel_version="dispersion", pole_param="halflife",
                      hl_min=10.0, hl_max=5.0)

    def test_halflife_stream_matches_forward(self):
        """step() recurrence == FFT forward for halflife poles, incl. the input
        normalizer, for the plain wave arm and the local+wave hybrid."""
        torch.manual_seed(1)
        frames = torch.rand(2, T, 3, H, W)
        for kw in ({}, {"fuse": "local_wave"}):
            with self.subTest(**kw):
                m = VideoPredictor(DIM, 2, NH, T, H, W, "wave", causal=True,
                                   kernel_version="dispersion", linear_pad=True,
                                   pole_param="halflife", hl_min=2.0, hl_max=64.0, **kw)
                m = _randomize_head(m).eval()
                with torch.no_grad():
                    fwd = m(frames)
                    st = m.stream_init(2, "cpu")
                    for t in range(T):
                        out, st = m.stream_step(frames[:, t], st, t)
                self.assertGreater((fwd - frames[:, -1]).abs().max().item(), 1e-3)
                self.assertLess((fwd - out).abs().max().item(), 1e-3)


    def test_halflife_forward_is_causal(self):
        """Changing frame s must not change the forward output at frames < s.
        The circular (untruncated) transfer leaks lam^(2T-(s-t)) -- O(1) for
        long poles -- so this guards the truncation in _transfer."""
        torch.manual_seed(3)
        w = WaveMix3D(DIM, NH, T, H, W, kernel_version="dispersion", causal_time=True,
                      linear_pad=True, pole_param="halflife", hl_min=64.0, hl_max=4096.0)
        h = torch.randn(1, NH, T, H, W, DIM // NH)
        h2 = h.clone()
        h2[:, :, T - 1] += 10.0                         # perturb only the last frame
        with torch.no_grad():
            y, y2 = w._wave_dispersion(h), w._wave_dispersion(h2)
        self.assertLess((y - y2)[:, :, :T - 1].abs().max().item(), 1e-4)
        self.assertGreater((y - y2)[:, :, T - 1].abs().max().item(), 1e-2)


class BudgetMemoryTests(unittest.TestCase):
    def test_five_minutes_through_ltx_is_900_latent_steps(self):
        r = lh.frame_budget(300, fps=24, height=480, width=848, vae="ltx")
        self.assertEqual(r["frames"], 7200)
        self.assertEqual(r["latent_steps"], 900)
        self.assertEqual(r["latent_hw"], [15, 27])

    def test_wave_state_does_not_grow_with_length_but_kv_does(self):
        a = lh.frame_budget(120, vae="wan21")
        b = lh.frame_budget(300, vae="wan21")
        self.assertEqual(a["wave_state_bytes"], b["wave_state_bytes"])
        self.assertGreater(b["kv_cache_bytes"], 2 * a["kv_cache_bytes"])

    def test_memory_report_softplus_forgets_halflife_reaches_minutes(self):
        torch.manual_seed(0)
        mk = lambda pp: VideoPredictor(DIM, 2, NH, T, H, W, "wave", causal=True,
                                       kernel_version="dispersion", linear_pad=True,
                                       pole_param=pp)
        soft = lh.memory_report(mk("softplus"), fps=24)["layers"][0]
        half = lh.memory_report(mk("halflife"), fps=24)["layers"][0]
        self.assertTrue(soft["modes_alive"]["10s"].startswith("0/"))
        self.assertFalse(half["modes_alive"]["120s"].startswith("0/"))
        self.assertGreater(half["reach_s"], 300)


class StreamSessionTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(2)
        m = VideoPredictor(DIM, 2, NH, T, H, W, "wave", causal=True,
                           kernel_version="dispersion", linear_pad=True, pole_param="halflife")
        return _randomize_head(m)

    def test_resume_is_bit_exact_and_state_is_constant(self):
        m = self._model()
        ctx = torch.rand(2, T, 3, H, W)
        full = lh.StreamSession(m).warm(ctx)
        sizes = set()
        chunks = []
        for c in full.generate(60, chunk=20):
            chunks.append(c)
            sizes.add(full.state_bytes())
        ref = torch.cat(chunks, 1)
        self.assertEqual(len(sizes), 1)

        s = lh.StreamSession(m).warm(ctx)
        first = next(s.generate(30, chunk=30))
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "state.pt")
            s.save(p)
            r = lh.StreamSession(m).load(p)
        second = next(r.generate(30, chunk=30))
        self.assertTrue(torch.equal(torch.cat([first, second], 1), ref))

    def test_resume_refuses_other_weights(self):
        m = self._model()
        s = lh.StreamSession(m).warm(torch.rand(1, T, 3, H, W))
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "state.pt")
            s.save(p)
            other = self._model()
            with torch.no_grad():
                other.head.bias.add_(1.0)
            with self.assertRaises(ValueError):
                lh.StreamSession(other).load(p)


class HealthMonitorTests(unittest.TestCase):
    def test_state_round_trip_keeps_streaks_across_a_resume(self):
        """A freeze streak that straddles a save/load must fire at the same frame."""
        ctx = self._ctx()
        frames = [torch.rand(1, 3, 8, 8) for _ in range(6)] + [torch.full((1, 3, 8, 8), 0.5)] * 10
        full = lh.HealthMonitor(patience=5).calibrate(ctx)
        for f in frames:
            full.update(f)
        a = lh.HealthMonitor(patience=5).calibrate(ctx)
        for f in frames[:9]:
            a.update(f)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "mon.pt")
            torch.save(a.state_dict(), p)
            b = lh.HealthMonitor(patience=5).load_state_dict(torch.load(p, weights_only=True))
            self.assertEqual(b._prev.device.type, "cpu")
        for f in frames[9:]:
            b.update(f)
        self.assertIn("freeze", full.first)
        self.assertEqual(b.first, full.first)

    def _ctx(self):
        g = torch.Generator().manual_seed(0)
        return torch.rand(1, 8, 3, 8, 8, generator=g)

    def test_healthy_motion_raises_nothing(self):
        mon = lh.HealthMonitor(patience=5).calibrate(self._ctx())
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            mon.update(torch.rand(1, 3, 8, 8, generator=g))
        self.assertTrue(mon.healthy, mon.first)

    def test_fade_to_black_is_flagged_where_it_starts(self):
        mon = lh.HealthMonitor(patience=5).calibrate(self._ctx())
        g = torch.Generator().manual_seed(1)
        for t in range(40):
            scale = 1.0 if t < 10 else 0.05
            mon.update(torch.rand(1, 3, 8, 8, generator=g) * scale)
        self.assertEqual(mon.first.get("fade"), 10)   # motion shrinks too: freeze may co-fire

    def test_one_collapsed_sample_is_not_hidden_by_a_healthy_one(self):
        g = torch.Generator().manual_seed(2)
        ctx = torch.rand(2, 8, 3, 8, 8, generator=g)
        mon = lh.HealthMonitor(patience=5).calibrate(ctx)
        for _ in range(20):
            f = torch.rand(2, 3, 8, 8, generator=g)
            f[1] *= 0.05                                  # sample 1 fades, sample 0 is fine
            mon.update(f)
        self.assertEqual(mon.first.get("fade"), 0)
        self.assertEqual(mon.first_sample.get("fade"), 1)

    def test_frozen_rollout_is_flagged(self):
        mon = lh.HealthMonitor(patience=5).calibrate(self._ctx())
        f = torch.rand(1, 3, 8, 8)
        for _ in range(20):
            mon.update(f)
        self.assertIn("freeze", mon.first)
        self.assertNotIn("fade", mon.first)


class RunnerTests(unittest.TestCase):
    def test_failed_arm_never_reports_done(self):
        """Slices no-op on DONE, so a crashed arm must leave status FAILED."""
        import os
        import subprocess
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PY="false", OUT=str(Path(d) / "out"), SEEDS="0",
                       SLICE_S="600")
            r = subprocess.run(["bash", str(root / "run_long_horizon.sh")], env=env,
                               capture_output=True, text=True)
            status = (Path(d) / "out" / "status.txt").read_text()
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue(status.startswith("FAILED"), status)


if __name__ == "__main__":
    unittest.main()
