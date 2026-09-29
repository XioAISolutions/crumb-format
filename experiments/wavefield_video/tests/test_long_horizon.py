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
    @staticmethod
    def _structured(t, detail=0.02):
        """Synthetic smooth landscape and a moving broad object, with fine detail."""
        y, x = torch.meshgrid(torch.arange(32) / 32, torch.arange(48) / 48,
                              indexing="ij")
        scene = (0.50 + 0.15 * torch.cos(2 * math.pi * y)
                 + 0.12 * torch.sin(2 * math.pi * x)
                 + 0.2 * torch.exp(-((x - (0.5 + 0.15 * math.sin(t))) ** 2
                                     + (y - 0.5) ** 2) / 0.025))
        scene += detail * torch.cos(2 * math.pi * (17 * x + 11 * y) + t)
        return scene[None, None].repeat(1, 3, 1, 1)

    def _structured_ctx(self, detail=0.02, static=False):
        return torch.stack([self._structured(0 if static else t * 0.4, detail)
                            for t in range(8)], 1)

    def _texture(self, ctx, t=0):
        # Same per-channel mean/contrast as context; texture is not a fade/flatten.
        y, x = torch.meshgrid(torch.arange(32), torch.arange(48), indexing="ij")
        noise = torch.cos(2 * math.pi * (x / 3 + y / 4) + t)[None, None]
        noise = noise / noise.std()
        return (ctx.mean() + ctx.flatten(3).std(-1).mean() * noise).repeat(1, 3, 1, 1)

    def test_structure_to_static_or_jittering_texture_is_flagged(self):
        ctx = self._structured_ctx()
        for jitter in (False, True):
            with self.subTest(jitter=jitter):
                mon = lh.HealthMonitor(patience=5).calibrate(ctx)
                for t in range(4):
                    mon.update(self._structured(t * 0.4))
                for t in range(8):
                    mon.update(self._texture(ctx, t * 1.7 if jitter else 0))
                self.assertEqual(mon.first.get("texture_collapse"), 4)
                self.assertNotIn("fade", mon.first)
                self.assertNotIn("flatten", mon.first)
                if jitter:
                    self.assertNotIn("freeze", mon.first)

    def test_structured_moving_static_and_high_detail_controls(self):
        for static, detail in ((False, 0.02), (True, 0.02), (False, 0.22)):
            with self.subTest(static=static, detail=detail):
                ctx = self._structured_ctx(detail, static)
                mon = lh.HealthMonitor(patience=5).calibrate(ctx)
                for t in range(30):
                    mon.update(self._structured(0 if static else t * 0.4, detail))
                self.assertTrue(mon.healthy, mon.first)

    def test_texture_uses_each_samples_own_structure_anchor(self):
        strong = self._structured_ctx()
        weak = 0.5 + 0.1 * (strong - 0.5)
        mon = lh.HealthMonitor(patience=3).calibrate(torch.cat((weak, strong)))
        # Identical detailed outputs preserve sample 0's weak coarse structure
        # but lose sample 1's strong structure: a batch-wide anchor is incorrect.
        for t in range(5):
            f = weak[:, t] + self._texture(strong, t * 1.7) - strong.mean()
            mon.update(f.repeat(2, 1, 1, 1))
        self.assertEqual(mon.first["texture_collapse"], 0)
        self.assertEqual(mon.first_sample["texture_collapse"], 1)
        self.assertEqual(mon.runs["texture_collapse"].tolist(), [0, 5])

    def test_texture_streaks_are_per_sample_and_resume_exactly(self):
        ctx = self._structured_ctx()
        clean, bad = self._structured(1), self._texture(ctx, 1)
        # Alternate bad samples: no cross-sample accumulation. Then sample 1
        # collapses, including a partial streak saved/restored before it fires.
        frames = [torch.cat((bad, clean) if t % 2 else (clean, bad))
                  for t in range(6)] + [torch.cat((clean, bad))] * 7
        full = lh.HealthMonitor(patience=5).calibrate(ctx.repeat(2, 1, 1, 1, 1))
        split = lh.HealthMonitor(patience=5).calibrate(ctx.repeat(2, 1, 1, 1, 1))
        for f in frames:
            full.update(f)
        for f in frames[:9]:
            split.update(f)
        self.assertNotIn("texture_collapse", split.first)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "texture-monitor.pt"
            torch.save(split.state_dict(), p)
            resumed = lh.HealthMonitor(patience=5).load_state_dict(
                torch.load(p, weights_only=True))
        for f in frames[9:]:
            resumed.update(f)
        self.assertEqual(full.first["texture_collapse"], 6)
        self.assertEqual(full.first_sample["texture_collapse"], 1)
        self.assertEqual(resumed.first, full.first)
        self.assertEqual(resumed.first_sample, full.first_sample)
        self.assertEqual(resumed.frame_index, full.frame_index)
        for k in full.runs:
            self.assertTrue(torch.equal(resumed.runs[k], full.runs[k]), k)
        for k in full.ref:
            self.assertTrue(torch.equal(resumed.ref[k], full.ref[k]), k)
        for settings in ({"structure_ratio": 0.1}, {"high_fraction": 0.75}):
            with self.assertRaises(SystemExit):
                lh.HealthMonitor(patience=5, **settings).load_state_dict(split.state_dict())

    def test_nonfinite_context_is_rejected_as_an_anchor(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                ctx = self._structured_ctx()
                ctx[0, 0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    lh.HealthMonitor().calibrate(ctx)

    def test_nonfinite_frame_flags_immediately_and_breaks_texture_streak(self):
        ctx = self._structured_ctx()
        mon = lh.HealthMonitor(patience=3).calibrate(ctx)
        bad = self._texture(ctx)
        mon.update(bad)
        mon.update(bad)
        invalid = bad.clone()
        invalid[0, 0, 0, 0] = float("nan")
        mon.update(invalid)
        self.assertEqual(mon.first, {"nonfinite": 2})
        mon.update(bad)
        mon.update(bad)
        self.assertNotIn("texture_collapse", mon.first)
        mon.update(bad)
        self.assertEqual(mon.first["texture_collapse"], 3)

    def test_tiny_flat_context_has_finite_stats_and_no_texture_claim(self):
        mon = lh.HealthMonitor(patience=2).calibrate(torch.full((1, 2, 3, 1, 1), 0.5))
        stats = mon.update(torch.full((1, 3, 1, 1), 0.5))
        self.assertTrue(all(math.isfinite(v) for v in stats.values()), stats)
        self.assertTrue(mon.healthy, mon.first)

    def test_legacy_state_cannot_silently_skip_the_new_screen(self):
        state = lh.HealthMonitor().calibrate(self._structured_ctx()).state_dict()
        state.pop("config")
        state["ref"].pop("coarse_power")
        with self.assertRaisesRegex(SystemExit, "context.*start fresh"):
            lh.HealthMonitor().load_state_dict(state)

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


class StreamCliTests(unittest.TestCase):
    def test_ckpt_ffn_width_is_rebuilt_exactly(self):
        """ffn_mult 63.984387 -> width 2048, but the 3-decimal 63.984 rebuilds
        2047; the stream CLI must read the width from the checkpoint."""
        torch.manual_seed(0)
        m = VideoPredictor(32, 1, 4, 4, 4, 4, "wave", causal=True, ffn_mult=63.984387,
                           kernel_version="dispersion", linear_pad=True)
        self.assertEqual(m.blocks[0].ffn.fc1.out_features, 2048)
        self.assertEqual(int(round(32 * 63.984)), 2047)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "model.pt")
            torch.save({"state": m.state_dict()}, p)
            res = lh.main(["stream", "--pole-param", "softplus", "--ckpt", p, "--grid", "4",
                           "--frames", "4", "--dim", "32", "--layers", "1", "--heads", "4",
                           "--stream-frames", "4", "--chunk", "4", "--batch", "1",
                           "--device", "cpu"])
        self.assertTrue(res["trained"])


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
