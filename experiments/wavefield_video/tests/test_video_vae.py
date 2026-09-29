"""Run: python -m unittest discover -s tests -p 'test_video_vae.py' -v.

LONG_HORIZON.md phase 2 on CPU with tiny random VAEs: shapes, strides,
save/load, mp4 -> shards -> latent training -> decoded 5-minute-style stream.
Skips when the optional video deps (requirements-video.txt) are missing."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
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


def _write_shards(root, specs, C=2, h=2, w=2):
    """specs: [(src, latent_steps)] -> a minimal encode_videos-style shard dir."""
    import hashlib
    root.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (src, n) in enumerate(specs):
        z = torch.randn(n, C, h, w, generator=torch.Generator().manual_seed(i))
        torch.save({"latents": z}, root / f"s{i}.pt")
        index.append({"file": f"s{i}.pt", "src": src, "latent_steps": n, "vae": {"backend": "x"},
                      "sha256": hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]})
    (root / "index.json").write_text(json.dumps(index))


@unittest.skipUnless(HAVE, "needs diffusers + imageio + imageio-ffmpeg (requirements-video.txt)")
class HeldOutSplitTests(unittest.TestCase):
    def test_single_source_short_tail_still_holds_out_a_full_segment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "lat"
            _write_shards(root, [("a.mp4", 10)] * 5 + [("a.mp4", 3)])   # short final remainder
            for window in (4, 8):
                sh = LatentShards(root, window=window)
                self.assertEqual([z.shape[0] for z in sh.eval], [10])
                self.assertEqual(len(sh.train), 4)
            # the split itself does not depend on the window
            small = LatentShards(root, window=2)
            self.assertEqual(len(small.eval), 2)                       # full segment + tail
            self.assertEqual(len(small.train), 4)

    def test_rollout_horizon_from_longest_held_out_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "lat"
            _write_shards(root, [("a.mp4", 12), ("b.mp4", 12), ("c.mp4", 20), ("c.mp4", 4)])
            sh = LatentShards(root, window=4)
            self.assertEqual(sorted(z.shape[0] for z in sh.eval), [4, 20])
            m = VideoPredictor(16, 1, 2, 2, 2, 2, "wave", in_ch=2, time_pos="none",
                               causal=True, kernel_version="dispersion", linear_pad=True,
                               pole_param="halflife")
            a = type("A", (), dict(eval_rollout=64, chunk=2, eval_batch=2))()
            out = train_long.latent_eval(m, sh, a)
            self.assertEqual(out["latent_rollout_steps"], 18)          # 20 - chunk, not 4 - chunk - 1
            self.assertEqual(len(out["latent_rollout_mse"]), 18)


@unittest.skipUnless(HAVE, "needs diffusers + imageio + imageio-ffmpeg (requirements-video.txt)")
class StreamDecoderTests(unittest.TestCase):
    """Chunked stream decoding must give the frames of ONE decode of the whole
    stream (Codex review: a one-latent overlap resets the decoder at every chunk)."""

    def _check(self, backend, tol):
        from video_vae import StreamDecoder, measure_temporal_rf
        v = VideoVAE(backend, device="cpu")
        back, fwd = measure_temporal_rf(v, v.channels, 2, 2)
        g = torch.Generator().manual_seed(1)
        ctx = torch.randn(1, back + 2, v.channels, 2, 2, generator=g) * 0.5
        gen = torch.randn(1, 30, v.channels, 2, 2, generator=g) * 0.5
        full = v.decode(torch.cat([ctx, gen], 1))
        ref = full[:, 1 + (ctx.shape[1] - 1) * v.t_stride:]           # frames of `gen`

        def run(H, L, split=None):
            d = StreamDecoder(v, H, L)
            d.seed(ctx)
            out = []
            for i in range(0, 30, 7):
                if split is not None and i == split:                  # save / restore mid-stream
                    st = d.state_dict()
                    d = StreamDecoder(v, H, L)
                    d.load_state_dict(st)
                x = d.push(gen[:, i:i + 7])
                out += [x] if x is not None else []
            x = d.flush()
            out += [x] if x is not None else []
            return torch.cat(out, 1)
        exact = run(back + 1, fwd + 1)
        self.assertEqual(exact.shape, ref.shape)
        self.assertLess(float((exact - ref).abs().max()), tol)
        self.assertTrue(torch.equal(run(back + 1, fwd + 1, split=14), exact))
        old = run(1, 0)                                               # the previous scheme
        self.assertGreater(float((old - ref).abs().max()), 100 * tol)
        return back, fwd

    def test_restored_buffer_is_moved_to_cpu(self):
        """StreamSession.load() maps the checkpoint to the model device; a CUDA buffer
        would then meet push()'s CPU chunk in torch.cat. load_state_dict must move it
        (stand-in for a CUDA tensor on a CPU-only runner: records the .cpu() call)."""
        from video_vae import StreamDecoder
        v = VideoVAE("ltx-tiny", device="cpu")
        d = StreamDecoder(v, 2, 1)
        d.seed(torch.randn(1, 3, v.channels, 2, 2))
        d.push(torch.randn(1, 4, v.channels, 2, 2))
        st = d.state_dict()
        moved = []

        class OnDevice:
            def __init__(self, t):
                self.t = t

            def detach(self):
                return self

            def cpu(self):
                moved.append(True)
                return self.t
        st = {**st, "buf": OnDevice(st["buf"])}
        d2 = StreamDecoder(v, 2, 1)
        d2.load_state_dict(st)
        self.assertEqual(moved, [True])
        self.assertIsInstance(d2.buf, torch.Tensor)
        self.assertEqual(d2.buf.device.type, "cpu")
        self.assertIsNotNone(d2.push(torch.randn(1, 3, v.channels, 2, 2)))   # no device clash

    def test_ltx_tiny_matches_single_decode(self):
        back, fwd = self._check("ltx-tiny", 1e-5)
        self.assertGreater(fwd, 0)                                    # LTX decoder looks ahead

    def test_wan_tiny_matches_single_decode(self):
        back, fwd = self._check("wan-tiny", 5e-4)                     # causal: probe tolerance
        self.assertEqual(fwd, 0)


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
        with tempfile.TemporaryDirectory() as d:          # pipeline layout: VAE under vae/
            v.save(Path(d) / "vae")
            w = VideoVAE("ltx-tiny", path=d)
        self.assertTrue(torch.allclose(v.encode(x), w.encode(x), atol=1e-6))

    def test_fingerprint_covers_config_only_normalization(self):
        a, b = VideoVAE("wan-tiny"), VideoVAE("wan-tiny")
        self.assertEqual(a.fingerprint, b.fingerprint)
        b.std, b._fp = b.std * 2, None                   # latents_std changed, weights identical
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        from diffusers.configuration_utils import FrozenDict
        c = VideoVAE("wan-tiny")                         # config-only forward option changed
        c.model._internal_dict = FrozenDict({**dict(c.model.config), "scale_factor_temporal": 2})
        c._fp = None
        self.assertNotEqual(a.fingerprint, c.fingerprint)

    def test_fingerprint_is_independent_of_the_load_path(self):
        """The same weights loaded from two places (an HF id vs its local snapshot,
        in practice) are one latent space: loader metadata must not split them."""
        import shutil
        with tempfile.TemporaryDirectory() as d:
            VideoVAE("ltx-tiny").save(Path(d) / "a")
            shutil.copytree(Path(d) / "a", Path(d) / "b")
            fa = VideoVAE("ltx-tiny", path=Path(d) / "a").fingerprint
            fb = VideoVAE("ltx-tiny", path=Path(d) / "b").fingerprint
            self.assertEqual(fa, fb)
            self.assertEqual(fa, VideoVAE("ltx-tiny").fingerprint)     # in-memory, no path at all

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            VideoVAE("sora")
        with self.assertRaises(ValueError):
            VideoVAE("ltx-tiny").encode(torch.rand(1, 9, 3, 60, 64))   # not a /32 size
        with self.assertRaises(ValueError):
            VideoPredictor(16, 1, 2, 4, 4, 4, "wave", quat_color=True, in_ch=8)

    def test_hidden_and_appledouble_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            VideoVAE("ltx-tiny").save(d / "vae")
            args = ["--synthetic", "1", "--videos", str(d / "mp4"), "--vae", "ltx-tiny",
                    "--vae-path", str(d / "vae"), "--height", "64", "--width", "64",
                    "--max-frames", "33", "--device", "cpu"]
            encode_videos.main(args + ["--out", str(d / "lat0")])
            real = sorted((d / "mp4").glob("*.mp4"))[0]
            (d / "mp4" / f"._{real.name}").write_bytes(b"\x00\x05\x16\x07 AppleDouble")
            (d / "mp4" / ".hidden").mkdir()
            (d / "mp4" / ".hidden" / "x.mp4").write_bytes(b"junk")
            with contextlib.redirect_stdout(io.StringIO()):
                idx = encode_videos.main(["--synthetic", "0"] + args[2:] + ["--out", str(d / "lat")])
            self.assertEqual({Path(m["src"]).name for m in idx}, {real.name})

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
            self.assertEqual(len({m["sha256"] for m in idx}), len(idx))   # content digests
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
            # 8 latent steps decoded with a one-latent overlap: one continuous 8*8-frame
            # timeline (independent chunk decodes would give 2 * (1 + 8*3) = 50)
            self.assertEqual(out["log"][-1]["decoded_frames"], 8 * 8)
            self.assertEqual(out["latent_steps_generated"], 8)
            # force a flag at decoded frame 20: it must be reported as latent step 20 // 8 = 2
            orig = lh.HealthMonitor.update

            def update(mon, frame):
                if mon.frame_index == 20:
                    mon.first.setdefault("fade", 20)
                return orig(mon, frame)
            with unittest.mock.patch.object(lh.HealthMonitor, "update", update):
                out = lh.main(["stream", "--pole-param", "halflife", "--time-pos", "none",
                               "--ckpt", str(d / "run" / "model_wave.pt"), "--latents", str(d / "lat"),
                               "--vae", "ltx-tiny", "--vae-path", str(d / "vae"), "--frames", "2",
                               "--dim", "16", "--layers", "1", "--heads", "2", "--stream-frames", "8",
                               "--chunk", "4", "--batch", "1", "--device", "cpu"])
            self.assertEqual(out["collapse"]["fade"], 20)
            self.assertEqual(out["collapse_latent_step"]["fade"], 2)
            json.dumps(out)                                        # result stays JSON-able

    def test_resample_30_to_24_fps(self):
        with tempfile.TemporaryDirectory() as d:
            src = encode_videos.write_synthetic(d, 1, frames=60, size=32, fps=30)[0]
            segs = list(encode_videos.iter_segments(src, 24, 32, 32, 100))
            self.assertEqual(sum(x.shape[0] for _, x, _ in segs), 48)       # 2 s at 24 fps
            self.assertEqual(segs[0][2], 24)

    def test_sliced_latent_run_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            VideoVAE("ltx-tiny").save(d / "vae")
            encode_videos.main(["--synthetic", "3", "--videos", str(d / "mp4"), "--vae", "ltx-tiny",
                                "--vae-path", str(d / "vae"), "--height", "64", "--width", "64",
                                "--max-frames", "33", "--out", str(d / "lat"), "--device", "cpu"])
            common = ["--latents", str(d / "lat"), "--seq-frames", "4", "--chunk", "2", "--dim", "16",
                      "--layers", "1", "--heads", "2", "--batch", "2", "--eval-rollout", "2"]
            train_long.main(common + ["--steps", "4", "--out", str(d / "full")])
            train_long.main(common + ["--steps", "2", "--save-every", "2", "--out", str(d / "cut")])
            train_long.main(common + ["--steps", "4", "--resume", str(d / "cut" / "ckpt_wave.pt"),
                                      "--out", str(d / "cut")])
            a = torch.load(d / "full" / "model_wave.pt", weights_only=True)["state"]
            b = torch.load(d / "cut" / "model_wave.pt", weights_only=True)["state"]
            for k in a:
                self.assertTrue(torch.allclose(a[k], b[k], atol=1e-6), k)
            # a resume under training-changing flags is refused, naming them
            ck = str(d / "cut" / "ckpt_wave.pt")
            def with_flag(flag, value):
                args = list(common)
                if flag in args:
                    args[args.index(flag) + 1] = value
                else:
                    args += [flag] if value is None else [flag, value]
                return args
            for flag, value in (("--lr", "1e-3"), ("--dense", None), ("--tbptt-chunks", "2"),
                                ("--batch", "4")):
                err = io.StringIO()
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                    train_long.main(with_flag(flag, value) + ["--steps", "6", "--resume", ck,
                                                              "--out", str(d / "cut")])
                self.assertIn(flag, err.getvalue())
            # a target below the saved step would report a less-trained run
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                train_long.main(common + ["--steps", "1", "--resume", ck, "--out", str(d / "cut")])
            self.assertIn("already at step", err.getvalue())
            # free flags (steps, eval knobs, save cadence) may change
            train_long.main(with_flag("--eval-rollout", "1") + ["--steps", "5", "--save-every", "5",
                                                               "--resume", ck, "--out", str(d / "cut")])
            # checkpoints written before arguments were recorded cannot be verified
            old = torch.load(ck, weights_only=True)
            old.pop("args")
            torch.save(old, d / "cut" / "ckpt_noargs.pt")
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                train_long.main(common + ["--steps", "6", "--resume", str(d / "cut" / "ckpt_noargs.pt"),
                                          "--out", str(d / "cut")])
            self.assertIn("records no training arguments", err.getvalue())
    def test_encode_resumes_after_a_killed_slice(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            VideoVAE("ltx-tiny").save(d / "vae")
            args = ["--videos", str(d / "mp4"), "--vae", "ltx-tiny", "--vae-path", str(d / "vae"),
                    "--height", "64", "--width", "64", "--max-frames", "33", "--out", str(d / "lat"),
                    "--device", "cpu"]
            full = encode_videos.main(["--synthetic", "3"] + args)
            # simulate a slice killed inside video 2, after its first segment
            journal = d / "lat" / "progress.jsonl"
            rows = journal.read_text().splitlines()
            v2 = sorted({json.loads(r)["src"] for r in rows})[1]
            cut = next(i for i, r in enumerate(rows) if json.loads(r)["src"] == v2) + 1
            journal.write_text("\n".join(rows[:cut]) + "\n" + rows[cut][:10])   # + torn line
            (d / "lat" / "index.json").unlink()
            real_replace, replaced = encode_videos.os.replace, []

            def spy(src, dst):
                replaced.append(Path(dst).name)
                return real_replace(src, dst)
            with unittest.mock.patch.object(VideoVAE, "encode", autospec=True,
                                            side_effect=VideoVAE.encode) as enc, \
                    unittest.mock.patch.object(encode_videos.os, "replace", side_effect=spy):
                resumed = encode_videos.main(args)
            self.assertIn("progress.jsonl", replaced)           # torn tail repaired atomically
            for r in journal.read_text().splitlines():
                json.loads(r)                                   # no torn text left behind
            (d / "lat" / "index.json").unlink()
            replaced.clear()
            before = journal.read_bytes()
            with unittest.mock.patch.object(encode_videos.os, "replace", side_effect=spy):
                encode_videos.main(args)                        # intact journal, all done
            self.assertNotIn("progress.jsonl", replaced)        # never rewritten
            self.assertEqual(journal.read_bytes(), before)
            # a last line whose newline was cut is repaired, not glued to the next write
            journal.write_bytes(before.rstrip(b"\n"))
            (d / "lat" / "index.json").unlink()
            encode_videos.main(args)
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual(len(full), 6)                      # 3 videos x 2 segments
            self.assertEqual(enc.call_count, 3)                 # v2's 2nd segment + v3's two
            self.assertEqual(resumed, full)
            with self.assertRaises(SystemExit):                  # settings changed mid-encode
                encode_videos.main(args[:-6] + ["--height", "32", "--width", "32"] + args[-4:])
            # another corpus into the same --out is refused (videos root is in the identity)
            encode_videos.write_synthetic(d / "mp4b", 1, frames=65, size=64)
            (d / "lat" / "index.json").unlink()
            with self.assertRaises(SystemExit):
                encode_videos.main([str(d / "mp4b") if x == str(d / "mp4") else x for x in args])
            encode_videos.main(args)                             # original corpus: resumes
            # the same root spelled differently resumes (canonical keys), no re-encode
            rel = [x for x in args]
            rel[rel.index(str(d / "mp4"))] = str(d / "." / "mp4")
            with unittest.mock.patch.object(VideoVAE, "encode", autospec=True,
                                            side_effect=VideoVAE.encode) as enc:
                (d / "lat" / "index.json").unlink()
                encode_videos.main(rel)
            self.assertEqual(enc.call_count, 0)
            # same size, same name, different bytes: refused (content digest)
            vid = sorted((d / "mp4").glob("*.mp4"))[0]
            raw = bytearray(vid.read_bytes()); raw[-1] ^= 0xFF
            vid.write_bytes(bytes(raw))
            (d / "lat" / "index.json").unlink()
            with self.assertRaises(SystemExit):
                encode_videos.main(args)
            raw[-1] ^= 0xFF; vid.write_bytes(bytes(raw))                   # restore
            encode_videos.main(args)                             # identical again: resumes
            # a tampered shard no longer matches its recorded digest
            shard = torch.load(d / "lat" / "shard_00000.pt", weights_only=True)
            shard["latents"] = shard["latents"] + 1
            torch.save(shard, d / "lat" / "shard_00000.pt")
            with self.assertRaisesRegex(SystemExit, "does not match index.json"):
                LatentShards(d / "lat", window=2)
            (d / "lat" / "index.json").unlink()
            (d / "lat" / "progress.jsonl").unlink()
            (d / "lat" / "progress_config.json").unlink()
            encode_videos.main(args)                             # clean re-encode
            # different weights, same backend and shapes: a resumed encode and a stream
            # decoder must both refuse them
            other = VideoVAE("ltx-tiny")
            with torch.no_grad():
                next(other.model.parameters()).add_(1e-3)
            other.save(d / "vae2")
            self.assertNotEqual(VideoVAE("ltx-tiny", path=d / "vae2").fingerprint,
                                VideoVAE("ltx-tiny", path=d / "vae").fingerprint)
            (d / "lat" / "index.json").unlink()
            with self.assertRaises(SystemExit):
                encode_videos.main([x if x != str(d / "vae") else str(d / "vae2") for x in args])
            encode_videos.main(args)                             # restore a complete index
            train_long.main(["--latents", str(d / "lat"), "--seq-frames", "4", "--chunk", "2",
                             "--dim", "16", "--layers", "1", "--heads", "2", "--steps", "1",
                             "--batch", "2", "--eval-rollout", "2", "--out", str(d / "run")])
            def stream(lat, vae, *extra):
                return lh.main(["stream", "--pole-param", "halflife", "--time-pos", "none",
                                "--ckpt", str(d / "run" / "model_wave.pt"), "--latents", str(lat),
                                "--vae", "ltx-tiny", "--vae-path", str(vae), "--frames", "2",
                                "--dim", "16", "--layers", "1", "--heads", "2", "--chunk", "2",
                                "--batch", "1", "--device", "cpu", *extra])
            with self.assertRaisesRegex(SystemExit, "encoder that wrote"):   # decoder != encoder
                stream(d / "lat", d / "vae2", "--stream-frames", "4")
            # shards and decoder agree (vae2) but the predictor was trained on vae's latents
            encode_videos.main([x if x not in (str(d / "vae"), str(d / "lat")) else
                                {str(d / "vae"): str(d / "vae2"), str(d / "lat"): str(d / "lat2")}[x]
                                for x in args])
            with self.assertRaisesRegex(SystemExit, "was trained on"):
                stream(d / "lat2", d / "vae2", "--stream-frames", "4")
            # a checkpoint without a fingerprint (older format) is refused, not trusted
            saved = torch.load(d / "run" / "model_wave.pt", weights_only=True)
            saved["config"].pop("vae")
            torch.save(saved, d / "run" / "model_legacy.pt")
            with self.assertRaisesRegex(SystemExit, "records no VAE fingerprint"):
                lh.main(["stream", "--pole-param", "halflife", "--time-pos", "none",
                         "--ckpt", str(d / "run" / "model_legacy.pt"), "--latents", str(d / "lat"),
                         "--vae", "ltx-tiny", "--vae-path", str(d / "vae"), "--frames", "2",
                         "--dim", "16", "--layers", "1", "--heads", "2", "--chunk", "2",
                         "--batch", "1", "--device", "cpu", "--stream-frames", "4"])
            # a sliced screen (--checkpoint) equals an uninterrupted one
            whole = stream(d / "lat", d / "vae", "--stream-frames", "8")
            ck = str(d / "screen.state")
            stream(d / "lat", d / "vae", "--stream-frames", "4", "--checkpoint", ck)   # "killed"
            sliced = stream(d / "lat", d / "vae", "--stream-frames", "8", "--checkpoint", ck)
            self.assertEqual(sliced["steps_generated"], 8)
            self.assertEqual(sliced["log"][-1]["decoded_frames"], whole["log"][-1]["decoded_frames"])
            self.assertEqual(sliced["log"][-1]["last"], whole["log"][-1]["last"])
            self.assertEqual(len(sliced["log"]), len(whole["log"]))
            done = stream(d / "lat", d / "vae", "--stream-frames", "8", "--checkpoint", ck)
            self.assertEqual(done["steps_generated"], 8)                        # done: no-op
            self.assertTrue(done["state_bytes_constant"])                       # from saved log
            self.assertNotIn("log", torch.load(ck, weights_only=True)["extra"])  # sidecar only
            self.assertEqual(len(open(ck + ".log.jsonl").read().splitlines()),
                             len(whole["log"]))
            self.assertEqual(done["grid"], [2, 2])                              # latent h, w
            with self.assertRaisesRegex(SystemExit, "different context"):     # other stream
                stream(d / "lat", d / "vae", "--stream-frames", "8", "--checkpoint", ck,
                       "--batch", "2")
            with self.assertRaisesRegex(SystemExit, "monitored with"):        # other patience
                stream(d / "lat", d / "vae", "--stream-frames", "8", "--checkpoint", ck,
                       "--patience", "5")
            # a training resume onto shards from another VAE is refused
            common = ["--seq-frames", "4", "--chunk", "2", "--dim", "16", "--layers", "1",
                      "--heads", "2", "--batch", "2", "--eval-rollout", "2"]
            train_long.main(common + ["--latents", str(d / "lat"), "--steps", "1",
                                      "--save-every", "1", "--out", str(d / "tr")])
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                train_long.main(common + ["--latents", str(d / "lat2"), "--steps", "2",
                                          "--resume", str(d / "tr" / "ckpt_wave.pt"),
                                          "--out", str(d / "tr")])
            self.assertIn("was trained on VAE", err.getvalue())
            # a latent checkpoint without identity metadata is refused, not trusted
            legacy = torch.load(d / "tr" / "ckpt_wave.pt", weights_only=True)
            legacy.pop("data_fp")
            torch.save(legacy, d / "tr" / "ckpt_legacy.pt")
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                train_long.main(common + ["--latents", str(d / "lat"), "--steps", "2",
                                          "--resume", str(d / "tr" / "ckpt_legacy.pt"),
                                          "--out", str(d / "tr")])
            self.assertIn("missing VAE or dataset fingerprint", err.getvalue())
            # same VAE, different dataset (other segmentation of the same videos)
            encode_videos.main([x if x not in (str(d / "lat"), "33") else
                                {str(d / "lat"): str(d / "lat3"), "33": "41"}[x] for x in args])
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                train_long.main(common + ["--latents", str(d / "lat3"), "--steps", "2",
                                          "--resume", str(d / "tr" / "ckpt_wave.pt"),
                                          "--out", str(d / "tr")])
            self.assertIn("different latent dataset", err.getvalue())
    def test_run_latent_refuses_a_reused_out_with_other_settings(self):
        import os
        import subprocess
        here = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PY=sys.executable, VIDEOS=str(Path(d) / "none"), OUT=d,
                       STEPS="10")
            first = subprocess.run(["bash", str(here / "run_latent.sh")], env=env,
                                   capture_output=True, text=True)
            self.assertNotEqual(first.returncode, 2)               # got past the identity check
            self.assertIn("STEPS=10", (Path(d) / "run_config.txt").read_text())
            env["STEPS"] = "20"
            second = subprocess.run(["bash", str(here / "run_latent.sh")], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertIn("different settings", second.stderr)

if __name__ == "__main__":
    unittest.main()
