"""Portable runner-contract tests for run_occlusion.sh; no torch/training needed.

Run with ``python test_occlusion_runner_r14.py``. A temporary fake PY executable
records the real shell runner's arguments and can simulate one failed arm."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("run_occlusion.sh")
# arm order per seed: (kind, label-in-tag, fuse, kernel_version)
ARMS = [("attn", "A_localattn", "none", "separable"),
        ("ssm", "B_ssm", "none", "separable"),
        ("wave", "C_wave", "none", "dispersion"),
        ("ssm", "D_local_ssm", "local_ssm", "separable"),
        ("wave", "E_local_wave", "local_wave", "dispersion")]


def effective_options(arguments):
    """Parse observable options with argparse last-option-wins semantics."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("kind", "tag", "out", "data-source", "fuse", "kernel-version"):
        parser.add_argument("--" + name, default="none" if name == "fuse" else None)
    for name, default in (("steps", 2000), ("grid", 32), ("n-balls", 3), ("seed", 0),
                          ("target-params", 0), ("occ-start", 64), ("occ-end", 320),
                          ("eval-rollout", 256)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.set_defaults(kernel_version="separable")
    return parser.parse_known_args(arguments[1:])[0]


def record_command():
    arguments = sys.argv[2:]
    options = effective_options(arguments)
    record = {"args": arguments,
              "status": (Path(options.out) / "occlusion_status.txt").read_text().strip()}
    with Path(os.environ["R14_TEST_RECORD"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print("Recorded", options.kind, options.tag)
    if os.environ.get("R14_TEST_FAIL_ONE") and options.tag == "_C_wave_s0":
        return 7
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
        self.env.pop("R14_TEST_FAIL_ONE", None)

    def run_matrix(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **overrides),
                              cwd=self.root, text=True, capture_output=True, timeout=120)

    def recorded(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_five_arms_over_seeds_equal_budget(self):
        run = self.run_matrix(SEEDS="0 1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), 10)                        # 5 arms x 2 seeds
        expected = [(seed,) + arm for seed in (0, 1) for arm in ARMS]
        for call, (seed, kind, label, fuse, kv) in zip(calls, expected):
            opts = effective_options(call["args"])
            self.assertEqual(call["args"][0], "train_compare.py")
            self.assertEqual(call["status"], "RUNNING")
            self.assertEqual((opts.kind, opts.tag, opts.seed), (kind, f"_{label}_s{seed}", seed))
            self.assertEqual(opts.fuse, fuse)
            self.assertEqual(opts.kernel_version, kv)
            self.assertEqual(opts.data_source, "occlusion")
            self.assertEqual((opts.grid, opts.occ_start, opts.occ_end), (64, 64, 320))
            self.assertEqual(opts.target_params, 4000000)       # equal ~4M budget
            self.assertEqual(opts.eval_rollout, 1024)
            self.assertEqual(opts.out, str(self.out))
            self.assertTrue((self.out / f"log_occlusion_{kind}_{label}_s{seed}.txt").is_file())
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertEqual(progress.count("command:"), 10)
        self.assertEqual(progress.count(" exit="), 10)
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), "DONE 10/10\n")

    def test_default_seed_sweep_is_25_jobs(self):
        run = self.run_matrix()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(self.recorded()), 25)              # 5 arms x 5 seeds
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), "DONE 25/25\n")

    def test_failed_arm_does_not_skip_remaining(self):
        run = self.run_matrix(SEEDS="0 1", R14_TEST_FAIL_ONE="1")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(len(self.recorded()), 10)              # all still attempted
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), "FAILED 1/10\n")
        progress = (self.out / "occlusion_progress.txt").read_text()
        self.assertIn("wave_C_wave_s0 exit=7", progress)
        self.assertIn("wave_E_local_wave_s1 exit=0", progress)

    def test_missing_interpreter_marks_failure(self):
        run = self.run_matrix(PY=str(self.root / "missing python"))
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.record.exists())
        self.assertEqual((self.out / "occlusion_status.txt").read_text(), "FAILED exit=2\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record"]:
        sys.exit(record_command())
    unittest.main()
