"""Run: python -m unittest discover -s tests -p 'test_waves.py' -v."""

import inspect
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data
import data_waves as waves


class WaveTests(unittest.TestCase):
    @staticmethod
    def coordinates(grid):
        y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float64),
                              torch.arange(grid, dtype=torch.float64), indexing="ij")
        return x * (2 * math.pi / grid), y * (2 * math.pi / grid)

    def test_2d_traveling_wave_closed_form(self):
        """All three unscaled channels must pass, not just a normalized picture."""
        kx, ky, speed, step = 2, 3, 1.15, 0.15
        omega = speed * math.sqrt(kx * kx + ky * ky)
        for grid in (16, 32, 64):
            with self.subTest(grid=grid):
                x, y = self.coordinates(grid)
                phase = kx * x + ky * y + 0.37
                actual = waves.spectral_evolve(phase.cos()[None], 64, speed=speed,
                                               time_step=step,
                                               initial_dt=(omega * phase.sin())[None])
                theta = phase[None] - omega * torch.arange(65, dtype=torch.float64)[:, None, None] * step
                expected = torch.stack((theta.cos(), omega * theta.sin(),
                                        -(kx * kx + ky * ky) * theta.cos()), dim=1)[None]
                error = (actual - expected).abs().amax(dim=(0, 1, 3, 4))
                print(f"traveling-wave grid={grid} max errors [field,dt,lap]={error.tolist()}")
                self.assertLess(error.max().item(), 1e-3)

    def test_advection_closed_form_and_conservation(self):
        x, y = self.coordinates(32)
        phase = 3 * x - 2 * y + 0.24
        vx, vy, step = 0.8, -0.4, 0.12
        frequency = 3 * vx - 2 * vy
        actual = waves.spectral_evolve(phase.cos()[None], 64, field="advection",
                                      velocity=(vx, vy), time_step=step)
        theta = phase[None] - frequency * step * torch.arange(65, dtype=torch.float64)[:, None, None]
        expected = torch.stack((theta.cos(), frequency * theta.sin(), -13 * theta.cos()), dim=1)[None]
        self.assertLess((actual - expected).abs().max().item(), 1e-10)
        energy = actual[0, :, 0].square().mean(dim=(-2, -1))
        self.assertLess((energy - energy[0]).abs().max().item(), 1e-12)

    def test_vortex_closed_form_and_navier_stokes(self):
        x, y = self.coordinates(32)
        mx, my, vx, vy, nu, step = 2, 3, 0.3, -0.2, 0.02, 0.15
        times = torch.arange(65, dtype=torch.float64)[:, None, None] * step
        ax, ay = mx * (x - vx * times), my * (y - vy * times)
        decay = torch.exp(-nu * (mx * mx + my * my) * times)
        u = ax.cos() * ay.cos() * decay
        ux, uy = -mx * ax.sin() * ay.cos() * decay, -my * ax.cos() * ay.sin() * decay
        lap = -(mx * mx + my * my) * u
        derivative = -vx * ux - vy * uy + nu * lap
        expected = torch.stack((u, derivative, lap), dim=1)[None]
        actual = waves.spectral_evolve(((mx * x).cos() * (my * y).cos())[None], 64,
                                      field="vortex", velocity=(vx, vy), viscosity=nu, time_step=step)
        self.assertLess((actual - expected).abs().max().item(), 1e-10)
        # With psi = omega / |k|^2, induced velocity=(psi_y,-psi_x).
        nonlinear = (uy / 13) * ux - (ux / 13) * uy
        self.assertLess(nonlinear.abs().max().item(), 1e-12)
        self.assertLess((actual[0, :, 1] + vx * ux + vy * uy + nonlinear - nu * lap).abs().max().item(), 1e-10)
        self.assertLess(actual[0, -1, 0].square().mean(), actual[0, 0, 0].square().mean())

    def test_supported_grids_horizon_range_and_motion(self):
        for field in waves.FIELDS:
            for grid in (16, 32, 64):
                with self.subTest(field=field, grid=grid):
                    out, meta, moving = waves.make_clip_batch(2, 64, grid, grid, seed=19,
                                                              field=field, return_meta=True,
                                                              return_moving=True)
                    self.assertEqual(out.shape, (2, 65, 3, grid, grid))
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertGreaterEqual(out.min().item(), 0)
                    self.assertLessEqual(out.max().item(), 1)
                    self.assertEqual(moving.shape, (2, 64, grid, grid))
                    self.assertEqual(moving.dtype, torch.bool)
                    self.assertTrue(torch.equal(moving, data.moving_mask(out)))
                    self.assertTrue(moving.any())
                    raw = (2 * out.double() - 1) * meta["scale"][:, None, :, None, None]
                    lap = torch.fft.ifft2(-self.k2(grid) * torch.fft.fft2(raw[:, :, 0])).real
                    self.assertLess((lap - raw[:, :, 2]).abs().max().item(), 1e-3)

    @staticmethod
    def k2(grid):
        k = torch.fft.fftfreq(grid, d=1 / grid, dtype=torch.float64)
        return k[:, None].square() + k[None, :].square()

    def test_seed_repeatability_prefix_and_global_rng(self):
        state = torch.random.get_rng_state().clone()
        for field in waves.FIELDS:
            with self.subTest(field=field):
                short, meta = waves.make_clip_batch(2, 3, 16, 16, field=field, seed=43, return_meta=True)
                long = waves.make_clip_batch(2, 64, 16, 16, field=field, seed=43)
                same = waves.make_clip_batch(2, 3, 16, 16, field=field, seed=43)
                other = waves.make_clip_batch(2, 3, 16, 16, field=field, seed=44)
                self.assertTrue(torch.equal(short, same))
                self.assertTrue(torch.equal(short, long[:, :4]))
                self.assertFalse(torch.equal(short, other))
                self.assertEqual(meta["seed"], 43)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_unspecified_seed_can_be_replayed(self):
        state = torch.random.get_rng_state().clone()
        a, meta = waves.make_clip_batch(1, 1, 16, 16, return_meta=True)
        b = waves.make_clip_batch(1, 1, 16, 16, seed=meta["seed"])
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_api_signature_and_optional_returns(self):
        old = inspect.signature(data.make_clip_batch).parameters
        new = inspect.signature(waves.make_clip_batch).parameters
        for name, parameter in old.items():
            self.assertEqual(parameter, new[name])
        baseline = waves.make_clip_batch(1, 4, 16, 32, seed=9)
        for want_meta, want_moving in ((False, False), (True, False), (False, True), (True, True)):
            result = waves.make_clip_batch(1, 4, 16, 32, seed=9, return_meta=want_meta,
                                          return_moving=want_moving, collisions=True, kicks=True)
            if want_meta or want_moving:
                self.assertEqual(len(result), 1 + want_meta + want_moving)
                self.assertTrue(torch.equal(result[0], baseline))
                if want_meta:
                    self.assertEqual(result[1]["field"], "wave")
                if want_moving:
                    self.assertTrue(torch.equal(result[-1], waves.moving_mask(baseline)))
            else:
                self.assertTrue(torch.equal(result, baseline))
        self.assertLessEqual(waves.moving_mask(baseline, 0.2).sum(), waves.moving_mask(baseline, 0.01).sum())

    def test_zero_horizon_zero_speed_and_dc(self):
        for field in waves.FIELDS:
            out, mask = waves.make_clip_batch(1, 0, 16, 16, field=field, return_moving=True)
            self.assertEqual(out.shape, (1, 1, 3, 16, 16))
            self.assertEqual(mask.shape, (1, 0, 16, 16))
            still = waves.make_clip_batch(1, 8, 16, 16, seed=4, field=field, speed=0, viscosity=0)
            self.assertTrue(torch.equal(still[:, 0], still[:, -1]))
            self.assertTrue(torch.equal(still[:, :, 1], torch.full_like(still[:, :, 1], 0.5)))
        initial = torch.full((1, 16, 16), 2.0, dtype=torch.float64)
        actual = waves.spectral_evolve(initial, 3, initial_dt=torch.full_like(initial, 0.3), time_step=0.2)
        expected = 2 + 0.3 * 0.2 * torch.arange(4, dtype=torch.float64)
        self.assertTrue(torch.allclose(actual[0, :, 0, 0, 0], expected))
        self.assertTrue(torch.allclose(actual[:, :, 1], torch.full_like(actual[:, :, 1], 0.3)))
        self.assertEqual(actual[:, :, 2].abs().max().item(), 0)

    def test_bad_parameters_fail_clearly(self):
        for kwargs in ({"field": "unknown"}, {"time_step": 0}, {"time_step": float("nan")},
                       {"speed": -1}, {"viscosity": -1}, {"velocity": [1, 2, 3]},
                       {"nb": 0}, {"move_thresh": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                waves.make_clip_batch(1, 2, 16, 16, **kwargs)
        for args in ((0, 2, 16, 16), (1, 65, 16, 16), (1, -1, 16, 16), (1, 2, 8, 8), (1, 2.5, 16, 16)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                waves.make_clip_batch(*args)


if __name__ == "__main__":
    unittest.main()
