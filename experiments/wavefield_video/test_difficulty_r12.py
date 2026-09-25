"""Portable runner contract tests; no torch, checkpoints or training required.

Run with ``python test_difficulty_r12.py``. A temporary fake PY executable records
the real shell runner's arguments and can simulate one failed arm.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("run_difficulty.sh")
ARMS = [("wave", "_D1"), ("wave", "_D2"), ("wave", "_D3"), ("attn", "_D3"),
        ("wave", "_D4"), ("wave", "_D5"), ("attn", "_D5"), ("wave", "_D6")]


def effective_options(arguments):
    """Parse observable task options with argparse's last-option-wins semantics."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("kind", "tag", "out", "data-source"):
        parser.add_argument("--" + name)
    for name, default in (("steps", 2000), ("grid", 32), ("n-balls", 3),
                          ("eval-rollout", 256), ("rollout-loss", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--speed", type=float, default=1.15)
    parser.add_argument("--radius", type=float, default=1.6)
    parser.add_argument("--collisions", action=argparse.BooleanOptionalAction, default=False)
    for name in ("resid-balanced", "motion-weighted", "rollout-ramp"):
        parser.add_argument("--" + name, action="store_true")
    return parser.parse_known_args(arguments[1:])[0]


def record_command():
    """Fake interpreter entry point: record argv/status, optionally exit 7."""
    arguments = sys.argv[2:]
    options = effective_options(arguments)
    record = {"args": arguments,
              "status": (Path(options.out) / "difficulty_status.txt").read_text().strip()}
    with Path(os.environ["R12_TEST_RECORD"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print("Recorded", options.kind, options.tag)
    if os.environ.get("R12_TEST_FAIL_ONE") and (options.kind, options.tag) == ("attn", "_D3"):
        return 7
    return 0


class DifficultyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r12_difficulty_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = self.root / "calls.jsonl"
        self.out = self.root / "output with spaces"
        self.fake = self.root / "fake python"
        self.fake.write_text('#!/usr/bin/env bash\n'
                             'exec "$R12_TEST_PYTHON" "$R12_TEST_MODULE" --record "$@"\n',
                             encoding="utf-8")
        self.fake.chmod(0o755)
        self.env = dict(os.environ, PY=str(self.fake), OUT=str(self.out), STEPS="2000",
                        RECIPE_EXTRA="", R12_TEST_RECORD=str(self.record),
                        R12_TEST_PYTHON=sys.executable, R12_TEST_MODULE=str(Path(__file__).resolve()))
        self.env.pop("R12_TEST_FAIL_ONE", None)

    def run_ladder(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **overrides),
                              cwd=self.root, text=True, capture_output=True, timeout=60)

    def recorded_options(self):
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        self.assertEqual(len(calls), 8)
        result = []
        for call, (kind, tag) in zip(calls, ARMS):
            arguments = call["args"]
            self.assertEqual(arguments[0], "train_compare.py")
            self.assertEqual(call["status"], "RUNNING")
            options = effective_options(arguments)
            self.assertEqual((options.kind, options.tag), (kind, tag))
            self.assertEqual(options.out, str(self.out))
            self.assertEqual(options.data_source, "balls")
            self.assertTrue((self.out / f"log_difficulty_{kind}{tag}.txt").is_file())
            result.append(options)
        progress = (self.out / "difficulty_progress.txt").read_text()
        self.assertEqual(progress.count("command:"), 8)
        self.assertEqual(progress.count(" exit="), 8)
        return result

    def test_default_eight_job_difficulty_matrix(self):
        run = self.run_ladder()
        self.assertEqual(run.returncode, 0, run.stderr)
        # grid, count, speed, radius, horizon: independent requested task matrix.
        tasks = [(32, 3, 2.30, 1.6, 256), (32, 8, 1.15, 1.6, 256),
                 (32, 12, 2.30, 1.6, 256), (32, 12, 2.30, 1.6, 256),
                 (32, 3, 1.15, .8, 256), (64, 8, 1.15, 1.6, 256),
                 (64, 8, 1.15, 1.6, 256), (32, 3, 1.15, 1.6, 1024)]
        for options, expected in zip(self.recorded_options(), tasks):
            self.assertEqual((options.grid, options.n_balls, options.speed,
                              options.radius, options.eval_rollout), expected)
            self.assertEqual(options.steps, 2000)
            self.assertEqual(options.rollout_loss, 1)
            self.assertTrue(options.collisions)
        self.assertEqual((self.out / "difficulty_status.txt").read_text(), "DONE\n")

    def test_recipe_overrides_and_rung_precedence(self):
        extra = ("--resid-balanced\n--motion-weighted --no-collisions --rollout-ramp "
                 "--speed 9.9 --radius 3.2 --n-balls 5 --grid 16 --eval-rollout 64 "
                 "--kind ssm --data-source waves --steps 99 --out ignored --tag ignored")
        run = self.run_ladder(RECIPE_EXTRA=extra, STEPS="9")
        self.assertEqual(run.returncode, 0, run.stderr)
        for options in self.recorded_options():
            self.assertEqual(options.steps, 9)
            self.assertFalse(options.collisions)
            self.assertTrue(options.resid_balanced and options.motion_weighted and options.rollout_ramp)
            self.assertEqual(options.speed, 2.3 if options.tag in ("_D1", "_D3") else 9.9)
            self.assertEqual(options.radius, .8 if options.tag == "_D4" else 3.2)
            self.assertEqual(options.n_balls, 12 if options.tag == "_D3" else
                             8 if options.tag in ("_D2", "_D5") else 5)
            self.assertEqual(options.grid, 64 if options.tag == "_D5" else
                             32 if options.tag == "_D6" else 16)
            self.assertEqual(options.eval_rollout, 1024 if options.tag == "_D6" else 64)
        self.assertEqual((self.out / "difficulty_status.txt").read_text(), "DONE\n")

    def test_failed_arm_does_not_skip_remaining_jobs(self):
        run = self.run_ladder(R12_TEST_FAIL_ONE="1")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.recorded_options()
        self.assertEqual((self.out / "difficulty_status.txt").read_text(), "FAILED 1/8\n")
        progress = (self.out / "difficulty_progress.txt").read_text()
        self.assertIn("attn_D3 exit=7", progress)
        self.assertIn("wave_D6 exit=0", progress)

    def test_missing_interpreter_marks_failure(self):
        run = self.run_ladder(PY=str(self.root / "missing python"))
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.record.exists())
        self.assertEqual((self.out / "difficulty_status.txt").read_text(), "FAILED exit=2\n")
        self.assertIn("difficulty aborted exit=2",
                      (self.out / "difficulty_progress.txt").read_text())

    def test_recipe_shell_expressions_and_globs_stay_literal(self):
        marker = self.root / "must-not-exist"
        run = self.run_ladder(RECIPE_EXTRA=f"--unused $(touch {marker}) *.py")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(marker.exists())
        self.recorded_options()
        first = json.loads(self.record.read_text().splitlines()[0])["args"]
        self.assertIn("$(touch", first)
        self.assertIn("*.py", first)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record"]:
        sys.exit(record_command())
    unittest.main()
