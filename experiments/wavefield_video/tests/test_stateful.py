"""Run: python -m unittest discover -s tests -p 'test_stateful.py' -v.

LONG_HORIZON.md phase 1: carried-state chunks, dense next-frame loss,
time-invariant positions, causal gate."""

import json
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

    def test_clean_write_blank_frames_write_nothing(self):
        """clean_write: no spatial table, no embed/pi bias -> a blank clip leaves
        every layer's state exactly zero (LONG_HORIZON.md 8.4)."""
        torch.manual_seed(0)
        m = VideoPredictor(DIM, 2, NH, TC, H, W, "wave", causal=True, kernel_version="dispersion",
                           linear_pad=True, pole_param="halflife", time_pos="none",
                           clean_write=True).eval()
        self.assertIsNone(m.posemb.py)
        with torch.no_grad():
            _, st = m(torch.zeros(1, TC, 3, H, W), states=[None, None])
        for layer, z in enumerate(st):              # every layer, not just the first
            self.assertEqual(z.abs().max().item(), 0.0, f"layer {layer}")

    def test_write_gate_keeps_chunk_full_stream_equivalence(self):
        for kw in (dict(write_gate=True), dict(write_gate=True, clean_write=True)):
            with self.subTest(**kw):
                long, ch = _pair("wave", dict(kernel_version="dispersion", linear_pad=True,
                                              pole_param="halflife", hl_max=64.0, **kw))
                for mm in (long, ch):                       # a non-trivial gate
                    torch.manual_seed(5)
                    for blk in mm.blocks:
                        torch.nn.init.normal_(blk.mix.wg.weight, std=0.5)
                fr = torch.rand(2, N, 3, H, W)
                with torch.no_grad():
                    full, _ = long(fr, states=[None, None], dense=True)
                    sts, outs = [None, None], []
                    for c in range(N // TC):
                        p, sts = ch(fr[:, c * TC:(c + 1) * TC], states=sts, dense=True)
                        outs.append(p)
                    st, so = ch.stream_init(2, "cpu"), []
                    for t in range(N):
                        o, st = ch.stream_step(fr[:, t], st, t)
                        so.append(o)
                self.assertLess((torch.cat(outs, 1) - full).abs().max().item(), 1e-4)
                self.assertLess((torch.stack(so, 1) - full).abs().max().item(), 1e-4)

    def test_write_options_need_the_plain_wave_arm(self):
        for kw in (dict(kind="ssm", write_gate=True), dict(kind="attn", clean_write=True)):
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    VideoPredictor(DIM, 1, NH, TC, H, W, **kw)
        with self.assertRaises(ValueError):                   # a time table writes every frame
            VideoPredictor(DIM, 1, NH, TC, H, W, "wave", kernel_version="dispersion",
                           linear_pad=True, clean_write=True)

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

    def test_train_long_checkpoint_loads_in_renderer(self):
        import tempfile
        import render_rollout
        import train_long
        with tempfile.TemporaryDirectory() as d:
            train_long.main(["--seq-frames", "8", "--chunk", "4", "--dim", "16", "--layers", "1",
                             "--heads", "2", "--grid", "16", "--steps", "1", "--batch", "2",
                             "--eval-rollout", "4", "--pole-param", "halflife", "--write-gate",
                             "--clean-write", "--out", d])
            args = render_rollout.parser().parse_args(["--ckpt", f"{d}/model_wave.pt", "--out", d])
            model, config, *_ = render_rollout.load_model(args, torch)   # strict state load
            for src in ("checkpoint:config", f"{d}/result_wave.json"):
                if not src.startswith("checkpoint"):
                    args.config = Path(src)
                    model, config, *_ = render_rollout.load_model(args, torch)
                self.assertEqual((config["frames"], config["time_pos"]), (4, "none"))
            mix = model.blocks[0].mix
            self.assertTrue(mix.write_gate and mix.pole_param == "halflife")
            self.assertNotIn("posemb.pt", model.state_dict())
    def test_grad_ckpt_and_micro_batch_keep_gradients(self):
        import argparse
        import train_long
        a = argparse.Namespace(seq_frames=8, chunk=4, dense=True, motion_loss=False, tbptt_chunks=1)
        clips = torch.rand(4, 9, 3, H, W)

        def grads(ckpt, mb):
            torch.manual_seed(0)
            m = VideoPredictor(DIM, 2, NH, 4, H, W, "wave", causal=True, kernel_version="dispersion",
                               linear_pad=True, pole_param="halflife", time_pos="none")
            torch.nn.init.normal_(m.head.weight, std=0.1)
            m.grad_ckpt = ckpt
            for i in range(0, 4, mb):
                train_long.sequence_loss(m, clips[i:i + mb], None, a, scale=mb / 4)
            return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

        ref = grads(False, 4)
        for ckpt, mb in ((True, 4), (False, 2), (True, 1)):
            with self.subTest(ckpt=ckpt, micro_batch=mb):
                self.assertTrue(torch.allclose(grads(ckpt, mb), ref, atol=1e-5, rtol=1e-4))
    def test_eval_only_uses_the_carried_state(self):
        import tempfile
        import eval_only
        import train_long
        common = ["--seq-frames", "16", "--chunk", "4", "--dim", "16", "--layers", "1",
                  "--heads", "2", "--grid", "16", "--steps", "1", "--batch", "2", "--eval-seeds", "1"]
        with tempfile.TemporaryDirectory() as d:
            train_long.main(common + ["--eval-rollout", "4", "--out", d])
            out = eval_only.main([f"{d}/model_wave.pt", f"{d}/result_wave.json",
                                  "--eval-rollout", "4", "--eval-seeds", "1"])
            self.assertEqual(out["rollout_path"], "stream_step (carried state)")
            train_long.main(common + ["--data-source", "occlusion", "--train-occ-start", "4",
                                      "--train-occ-end", "12", "--occ-start", "8",
                                      "--occ-end", "16", "--eval-rollout", "24", "--kind", "ssm",
                                      "--out", d])
            out = eval_only.main([f"{d}/model_ssm.pt", f"{d}/result_ssm.json", "--eval-seeds", "1",
                                  "--eval-rollout", "24"])
            self.assertIn("exit_direction_accuracy", out)             # occlusion evaluator
            args = __import__("render_rollout").parser().parse_args(
                ["--ckpt", f"{d}/model_ssm.pt", "--out", f"{d}/r", "--frames", "6", "--device", "cpu"])
            __import__("render_rollout").render(args)                 # auto mode
            meta = json.loads(Path(f"{d}/r/metrics.json").read_text())
            self.assertEqual(meta["rollout"]["mode"], "recurrent")    # SSM through its state
    def test_sigterm_checkpoints_and_resume_finishes(self):
        import os
        import signal
        import subprocess
        import tempfile
        here = Path(__file__).resolve().parents[1]
        args = [sys.executable, str(here / "train_long.py"), "--seq-frames", "8", "--chunk", "4",
                "--dim", "16", "--layers", "1", "--heads", "2", "--grid", "16", "--batch", "2",
                "--eval-rollout", "4", "--steps", "200", "--save-every-sec", "3600"]
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen(args + ["--out", d], cwd=here, stdout=subprocess.PIPE, text=True)
            for line in p.stdout:                        # wait until training is under way
                if line.startswith("STEP"):
                    break
            p.send_signal(signal.SIGTERM)
            out, _ = p.communicate(timeout=120)
            self.assertEqual(p.returncode, 143)
            self.assertIn("SIGTERM: saved step", out)
            ck = torch.load(Path(d) / "ckpt_wave.pt", weights_only=True)
            self.assertGreater(ck["step"], 0)
            self.assertFalse(any(n.endswith(".tmp") for n in os.listdir(d)))
            done = subprocess.run(args[:-4] + ["--steps", str(ck["step"] + 1), "--save-every-sec",
                                               "3600", "--resume", str(Path(d) / "ckpt_wave.pt"),
                                               "--out", d], cwd=here, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertTrue((Path(d) / "model_wave.pt").exists())

if __name__ == "__main__":
    unittest.main()
