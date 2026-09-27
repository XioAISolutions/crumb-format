"""Portable runner-contract tests for run_occlusion.sh; no torch/training needed.

Run with ``python test_occlusion_runner_r14.py``. A temporary fake PY executable
records the real shell runner's arguments and can simulate one failed arm.

Updated for the R15 arms (F/G/H) and the sliced-queue knobs (CONST_LR / RESUME /
SLICE_S) -- the file name is historical; it pins the live run_occlusion.sh
contract, so it grows with the suite.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("run_occlusion.sh")
# arm order per seed: (kind, label-in-tag, fuse, kernel_version, q_mix, quat_color)
ARMS = [("attn", "A_localattn", "none", "separable", False, False),
        ("ssm", "B_ssm", "none", "separable", False, False),
        ("wave", "C_wave", "none", "dispersion", False, False),
        ("ssm", "D_local_ssm", "local_ssm", "separable", False, False),
        ("wave", "E_local_wave", "local_wave", "dispersion", False, False),
        ("qssm", "F_qssm", "none", "separable", False, False),
        ("wave", "G_qwave", "none", "dispersion", True, False),
        ("wave", "H_qcolor", "none", "dispersion", False, True)]


def effective_options(arguments):
    """Parse observable options with argparse last-option-wins semantics."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("kind", "tag", "out", "data-source", "fuse", "kernel-version"):
        parser.add_argument("--" + name, default="none" if name == "fuse" else None)
    for name, default in (("steps", 2000), ("grid", 32), ("n-balls", 3), ("seed", 0),
                          ("target-params", 0), ("occ-start", 64), ("occ-end", 320),
                          ("eval-rollout", 256)):
        parser.add_argument("--" + name, type=int, default=default)
    for name in ("q-mix", "quat-color", "const-lr"):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument("--resume", default=None)
    parser.set_defaults(kernel_version="separable")
    return parser.parse_known_args(arguments[1:])[0]


def record_command():
    arguments = sys.argv[2:]
    options = effective_options(arguments)
    record_file = Path(os.environ["R14_TEST_RECORD"])
    first = not record_file.exists()
    record = {"args": arguments,
              "status": (Path(options.out) / "occlusion_status.txt").read_text().strip()}
    with record_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print("Recorded", options.kind, options.tag)
    if os.environ.get("R14_TEST_FAIL_ONE") and options.tag == "_C_wave_s0":
        return 7
    if os.environ.get("R14_TEST_RC_ONCE") and first:
        return int(os.environ["R14_TEST_RC_ONCE"])
    return 0


class OcclusionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r14_occlusion_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = self.root / "calls.jsonl"
        self.out = self.root / "out with spaces"
        self.fake = self.root / "fake python"
        self.fake.write_text('#!/usr/bin/env bash\n'
                             'exec "$R14_TEST_PYTHON" "$R14_TEST_MODULE" --record "$@"\n',
                             encoding="utf-8")
        self.fake.chmod(0o755)
        self.env = dict(os.environ, PY=str(self.fake), OUT=str(self.out), STEPS="2000",
                        R14_TEST_RECORD=str(self.record), R14_TEST_PYTHON=sys.executable,
                        R14_TEST_MODULE=str(Path(__file__).resolve()))
        for var in ("R14_TEST_FAIL_ONE", "R14_TEST_RC_ONCE", "RESUME", "CONST_LR", "SLICE_S"):
            self.env.pop(var, None)

    def run_matrix(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **overrides),
                              cwd=self.root, text=True, capture_output=True, timeout=120)

    def recorded(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_arms_over_seeds_equal_budget(self):
        run = self.run_matrix(SEEDS="0 1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(ARMS) * 2)             # arms x 2 seeds
        expected = [(seed,) + arm for seed in (0, 1) for arm in ARMS]
        for call, (seed, kind, label, fuse, kv, qmix, qcolor) in zip(calls, expected):
            opts = effective_options(call["args"])
            self.assertEqual(call["args"][0], "train_compare.py")
            self.assertEqual(call["status"], "RUNNING")
            self.assertEqual((opts.kind, opts.tag, opts.seed), (kind, f"_{label}_s{seed}", seed))
            self.assertEqual(opts.fuse, fuse)
            self.assertEqual(opts.kernel_version, kv)
            self.assertEqual(opts.q_mix, qmix)
            self.assertEqual(opts.quat_color, qcolor)
            self.assertIsNone(opts.resume)                      # classic mode: no resume
            self.assertFalse(opts.const_lr)
            self.assertEqual(opts.data_source, "occlusion")
            self.assertEqual((opts.grid, opts.occ_start, opts.occ_end), (64, 64, 320))
            self.assertEqual(opts.target_params, 4000000)       # equal ~4M budget
            self.assertEqual(opts.eval_rollout, 1024)
            self.assertEqual(opts.out, str(self.out))
            self.assertTrue((self.out / f"log_occlusion_{kind}_{label}_s{seed}.txt").is_file())
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertEqual(progress.count("command:"), len(ARMS) * 2)
        self.assertEqual(progress.count(" exit="), len(ARMS) * 2)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"DONE {len(ARMS) * 2}/{len(ARMS) * 2}\n")

    def test_default_seed_sweep_is_full_matrix(self):
        run = self.run_matrix()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(self.recorded()), len(ARMS) * 5)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"DONE {len(ARMS) * 5}/{len(ARMS) * 5}\n")

    def test_const_lr_flag_reaches_every_arm(self):
        run = self.run_matrix(SEEDS="0", CONST_LR="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(ARMS))
        for call in calls:
            self.assertTrue(effective_options(call["args"]).const_lr, call["args"][:4])

    def test_failed_arm_does_not_skip_remaining(self):
        run = self.run_matrix(SEEDS="0 1", R14_TEST_FAIL_ONE="1")
        self.assertEqual(run.returncode, 1, run.stderr)
        n = len(ARMS) * 2
        self.assertEqual(len(self.recorded()), n)               # all still attempted
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), f"FAILED 1/{n}\n")
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertIn("wave_C_wave_s0 exit=7", progress)
        self.assertIn("wave_E_local_wave_s1 exit=0", progress)

    def test_missing_interpreter_marks_failure(self):
        run = self.run_matrix(PY=str(self.root / "missing python"))
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.record.exists())
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), "FAILED exit=2\n")

    # ---- R15 sliced-queue mode -------------------------------------------------
    def test_resume_mode_skips_finished_and_resumes_from_ckpt(self):
        self.out.mkdir(parents=True)
        (self.out / "result_attn_A_localattn_s0.json").write_text("{}")
        (self.out / "ckpt_ssm_B_ssm_s0.pt").write_text("dummy-ckpt")
        run = self.run_matrix(SEEDS="0", RESUME="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(ARMS) - 1)             # A skipped
        first = effective_options(calls[0]["args"])
        self.assertEqual(first.tag, "_B_ssm_s0")
        self.assertEqual(first.resume, str(self.out / "ckpt_ssm_B_ssm_s0.pt"))
        for call in calls[1:]:
            self.assertIsNone(effective_options(call["args"]).resume)
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertEqual(progress.count("SKIP"), 1)
        self.assertIn("attn_A_localattn_s0 result-exists", progress)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"DONE {len(ARMS)}/{len(ARMS)}\n")

    def test_slice_boundary_stops_suite_and_reports_sliced(self):
        run = self.run_matrix(SEEDS="0 1", RESUME="1", R14_TEST_RC_ONCE="124")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(self.recorded()), 1)               # stopped at first arm
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertIn("attn_A_localattn_s0 SLICED", progress)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"SLICED 0/{len(ARMS) * 2}\n")

    def test_resume_second_slice_continues(self):
        # First slice: the fake python gives up after one arm (rc 124). Second
        # slice: normal exits -> the suite continues from the first incomplete
        # arm and reaches DONE with every unit attempted.
        run1 = self.run_matrix(SEEDS="0", RESUME="1", R14_TEST_RC_ONCE="124")
        self.assertEqual(run1.returncode, 0, run1.stderr)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"SLICED 0/{len(ARMS)}\n")
        run2 = self.run_matrix(SEEDS="0", RESUME="1")
        self.assertEqual(run2.returncode, 0, run2.stderr)
        self.assertEqual(len(self.recorded()), 1 + len(ARMS))   # re-attempt + rest
        self.assertEqual((self.out / "occlusion_status.txt").read_text(),
                         f"DONE {len(ARMS)}/{len(ARMS)}\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record"]:
        sys.exit(record_command())
    unittest.main()
