"""R12 eval/render integration checks using deterministic predictor stubs."""
from argparse import Namespace
from unittest.mock import patch

import torch

import data
import data_waves
import render_rollout as rr
import train_compare as tc
from semantic_metrics import aggregate_semantic_metrics, semantic_frame_metrics


class FramePredictor(torch.nn.Module):
    """Black, persistence, or indexed oracle frames; no training or weights."""

    def __init__(self, mode, clip, args):
        super().__init__()
        self.mode, self.clip, self.args, self.calls = mode, clip, args, 0

    def forward(self, window):
        if self.mode == "black":
            return torch.zeros_like(window[:, -1])
        if self.mode == "copy":
            return window[:, -1]
        # rollout_eval iterates all horizons of one chunk before the next.
        chunk_number, horizon = divmod(self.calls, self.args.eval_rollout)
        self.calls += 1
        c0 = chunk_number * min(self.args.eval_chunk, self.args.eval_seeds)
        c1 = min(c0 + self.args.eval_chunk, self.args.eval_seeds)
        assert window.shape[0] == c1 - c0
        return self.clip[c0:c1, self.args.frames + horizon]


def _args(count, chunk=2):
    return Namespace(eval_seeds=3, eval_rollout=3, eval_chunk=chunk,
                     frames=3, grid=64, kicks=False, collisions=True,
                     radius=0.8, speed=2.30, n_balls=count)


def _truth(args):
    return data.make_clip_batch(args.eval_seeds, args.frames + args.eval_rollout - 1,
                                args.grid, args.grid, seed=90000,
                                kicks=args.kicks, collisions=args.collisions,
                                radius=args.radius, speed=args.speed,
                                nb=args.n_balls, return_meta=True)


def test_eval_semantic_wiring():
    with patch.object(tc, "DATA_SOURCE", "balls"):
        for count in (3, 8, 12):
            args = _args(count)
            clip, meta = _truth(args)
            for mode in ("black", "copy", "oracle"):
                expected_frames = []
                for horizon in range(args.eval_rollout):
                    gt_frame = args.frames + horizon
                    pred = (torch.zeros_like(clip[:, gt_frame]) if mode == "black" else
                            clip[:, args.frames - 1] if mode == "copy" else clip[:, gt_frame])
                    expected_frames.append(semantic_frame_metrics(
                        pred, meta["pos"][:, gt_frame], meta["col"], radius=args.radius))
                expected = aggregate_semantic_metrics(expected_frames, radius=args.radius)
                results = [tc.rollout_eval(FramePredictor(mode, clip, args), args, "cpu")]
                # Test uneven seed partitions too: 3 seeds / chunk 2 gives 2+1.
                for chunk in (1, 3):
                    chunk_args = _args(count, chunk=chunk)
                    results.append(tc.rollout_eval(FramePredictor(mode, clip, chunk_args), chunk_args, "cpu"))
                for result in results:
                    assert result["semantic"] == expected, (count, mode, result["eval_chunk"])
                    assert result["rollout_mse_curve"] == results[0]["rollout_mse_curve"]
                    semantic = result["semantic"]
                    assert semantic["expected_n"] == count
                    assert len(semantic["frames"]) == args.eval_rollout
                    for frame in semantic["frames"]:
                        assert len(frame["samples"]) == args.eval_seeds
                        assert all(sample["expected_n"] == count for sample in frame["samples"])
                    if mode == "black":
                        assert semantic["detected_n"] == semantic["matched_count"] == 0
                        assert semantic["mean_matched_position_error"] is None
                    elif mode == "copy":
                        assert semantic["matched_count"] < count
                    if mode == "oracle":
                        assert result["rollout_mse_curve"] == [0.0] * args.eval_rollout
                        # These deterministic g64/.8 scenes have resolved peaks.
                        assert semantic["detected_n"] == semantic["matched_count"] == count
                        assert semantic["mean_matched_position_error"] < 0.2
    print("OK eval semantic: black/copy/oracle, K=3/8/12, chunk=1/2/3 agree with direct frame metrics")


def test_wave_semantic_and_count():
    with patch.object(tc, "DATA_SOURCE", "waves"), patch.object(tc, "WAVE_FIELD", "advection"):
        common = dict(bs=2, T=4, H=16, W=16, seed=123)
        plain = tc.make_clip_batch(**common)
        crowded = tc.make_clip_batch(**common, nb=12, radius=0.8, speed=2.30, collisions=True)
        direct = data_waves.make_clip_batch(**common, field="advection")
        assert torch.equal(plain, crowded) and torch.equal(plain, direct)
        args = _args(12)
        args.grid = 16
        result = tc.rollout_eval(FramePredictor("black", None, args), args, "cpu")
        assert result["semantic"] is None
        assert result["mean_centroid_err"] is None and result["divergence_horizon"] is None
    print("OK waves: n-balls/ball geometry ignored; semantic and ball centroid fields null")


def test_render_truth_geometry():
    for count, grid, radius, speed in ((3, 16, 1.6, 1.15), (8, 32, 0.8, 2.30), (12, 64, 1.6, 2.30)):
        config = {"data_source": "balls", "grid": grid, "kicks": False, "collisions": True,
                  "n_balls": count, "radius": radius, "speed": speed}
        expected, meta = data.make_clip_batch(1, 7, grid, grid, seed=170001, nb=count,
                                               radius=radius, speed=speed, collisions=True,
                                               return_meta=True)
        pixels, _ = rr.make_truth(config.copy(), 8, 170001, torch)
        latent_pixels, colors, positions, _ = rr.make_latent_truth(config.copy(), 8, 170001, torch)
        assert torch.equal(pixels, expected) and torch.equal(latent_pixels, expected)
        assert torch.equal(colors, meta["col"]) and torch.equal(positions, meta["pos"])
        assert colors.shape == (1, count, 3) and positions.shape == (1, 8, count, 2)
    legacy = {"data_source": "balls", "grid": 16, "kicks": False, "collisions": True}
    expected = data.make_clip_batch(1, 7, 16, 16, seed=170001, collisions=True)
    pixels, _ = rr.make_truth(legacy.copy(), 8, 170001, torch)
    latent_pixels, colors, positions, _ = rr.make_latent_truth(legacy.copy(), 8, 170001, torch)
    assert torch.equal(pixels, expected) and torch.equal(latent_pixels, expected)
    assert colors.shape[1] == positions.shape[2] == data.N_BALLS
    waves = {"data_source": "waves", "grid": 16, "field": "wave"}
    pixels, _ = rr.make_truth(waves.copy(), 8, 170001, torch)
    latent_pixels, colors, positions, _ = rr.make_latent_truth(waves.copy(), 8, 170001, torch)
    assert torch.equal(pixels, latent_pixels) and colors is None and positions is None
    print("OK pixel/latent truth: saved count/radius/speed restored; legacy default frames byte-identical")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_eval_semantic_wiring()
    test_wave_semantic_and_count()
    test_render_truth_geometry()
