"""Run: python -m unittest discover -s tests -p 'test_video_vae.py' -v.

LONG_HORIZON.md phase 2 on CPU with tiny random VAEs: shapes, strides,
save/load, mp4 -> shards -> latent training -> decoded 5-minute-style stream.
Skips when the optional video deps (requirements-video.txt) are missing."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAVE = all(importlib.util.find_spec(m) for m in ("diffusers", "imageio", "imageio_ffmpeg"))

if HAVE:
    import encode_videos
    import long_horizon as lh
    import train_long
    from video_vae import PRESETS, LatentShards, VideoVAE, latent_steps, valid_frames
from wfvideo import VideoPredictor


@unittest.skipUnless(HAVE, "needs diffusers + imageio + imageio-ffmpeg (requirements-video.txt)")
class VideoVAETests(unittest.TestCase):
    def test_tiny_shapes_and_strides(self):
        x = torch.rand(1, 19, 3, 64, 64)
        for backend, (tl, hw) in {"ltx-tiny": (3, 2), "wan-tiny": (5, 8)}.items():
            with self.subTest(backend=backend):
                v = VideoVAE(backend)
                z = v.encode(x)
                self.assertEqual(tuple(z.shape), (1, tl, v.channels, hw, hw))
                self.assertEqual(v.latent_steps(19), tl)
                self.assertEqual(v.decode(z).shape[-2:], (64, 64))

    def test_frame_grid(self):
        self.assertEqual(valid_frames(7200, 8), 7193)       # 1 + 8k
        self.assertEqual(latent_steps(7193, 8), 900)       # 5 min at 24 fps through LTX
        self.assertEqual(latent_steps(33, 4), 9)
        self.assertEqual(valid_frames(10, 1), 10)

    def test_presets_match_the_budget_table(self):
        self.assertEqual((PRESETS["ltx"]["t_stride"], PRESETS["ltx"]["s_stride"],
                          PRESETS["ltx"]["channels"]),
                         (lh.VAES["ltx"]["t"], lh.VAES["ltx"]["s"], lh.VAES["ltx"]["c"]))
        self.assertEqual((PRESETS["wan"]["t_stride"], PRESETS["wan"]["s_stride"],
                          PRESETS["wan"]["channels"]),
                         (lh.VAES["wan21"]["t"], lh.VAES["wan21"]["s"], lh.VAES["wan21"]["c"]))

    def test_save_load_round_trip(self):
        v = VideoVAE("ltx-tiny")
        x = torch.rand(1, 9, 3, 64, 64)
        with tempfile.TemporaryDirectory() as d:
            v.save(d)
            w = VideoVAE("ltx-tiny", path=d)
        self.assertTrue(torch.allclose(v.encode(x), w.encode(x), atol=1e-6))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            VideoVAE("sora")
        with self.assertRaises(ValueError):
            VideoVAE("ltx-tiny").encode(torch.rand(1, 9, 3, 60, 64))   # not a /32 size
        with self.assertRaises(ValueError):
            VideoPredictor(16, 1, 2, 4, 4, 4, "wave", quat_color=True, in_ch=8)

    def test_videos_to_shards_to_training_to_decoded_stream(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            VideoVAE("ltx-tiny").save(d / "vae")
            idx = encode_videos.main(["--synthetic", "3", "--videos", str(d / "mp4"),
                                      "--vae", "ltx-tiny", "--vae-path", str(d / "vae"),
                                      "--height", "64", "--width", "64", "--max-frames", "33",
                                      "--out", str(d / "lat"), "--device", "cpu"])
            self.assertEqual(len(idx), 6)                         # 65 frames -> 33 + 25
            self.assertEqual([m["latent_steps"] for m in idx[:2]], [5, 4])
            # held-out split is by source video and independent of the window
            self.assertEqual(LatentShards(d / "lat", window=2).eval_srcs,
                             LatentShards(d / "lat", window=4).eval_srcs)
            self.assertEqual(len(LatentShards(d / "lat", window=2).eval_srcs), 1)
            segs = list(encode_videos.iter_segments(sorted((d / "mp4").glob("*.mp4"))[0],
                                                    24, 64, 64, 20))
            self.assertEqual([x.shape[0] for _, x, _ in segs], [20, 20, 20, 5])
            self.assertEqual([s0 for s0, _, _ in segs], [0, 20, 40, 60])
            sh = LatentShards(d / "lat", window=4)
            self.assertEqual(sh.batch(2, 4, torch.Generator().manual_seed(0)).shape,
                             (2, 4, 8, 2, 2))
            self.assertGreaterEqual(len(sh.eval), 1)
            res = train_long.main(["--latents", str(d / "lat"), "--seq-frames", "4",
                                   "--chunk", "2", "--dim", "16", "--layers", "1",
                                   "--heads", "2", "--steps", "2", "--batch", "2",
                                   "--eval-rollout", "2", "--out", str(d / "run")])
            self.assertEqual(res["data_source"], "latents")
            self.assertEqual(res["vae"]["backend"], "ltx-tiny")
            self.assertIn("latent_rollout_mse", res)
            out = lh.main(["stream", "--pole-param", "halflife", "--time-pos", "none",
                           "--ckpt", str(d / "run" / "model_wave.pt"), "--latents", str(d / "lat"),
                           "--vae", "ltx-tiny", "--vae-path", str(d / "vae"), "--frames", "2",
                           "--dim", "16", "--layers", "1", "--heads", "2", "--stream-frames", "8",
                           "--chunk", "4", "--batch", "1", "--device", "cpu"])
            self.assertEqual(out["vae"]["backend"], "ltx-tiny")
            self.assertEqual(out["log"][-1]["decoded_frames"], 2 * (1 + 8 * 3))   # 2 chunks of 4 steps
            json.dumps(out)                                        # result stays JSON-able


if __name__ == "__main__":
    unittest.main()
