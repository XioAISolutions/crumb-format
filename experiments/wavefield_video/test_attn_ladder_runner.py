"""Portable runner-contract tests for run_attn_ladder.sh + run_attn_hybrid_retest.sh.

Run with ``python test_attn_ladder_runner.py`` (stdlib only, no torch). A
temporary fake PY executable records the real shell runners' arguments and can
simulate a failed arm or a slice-boundary timeout.

Contract pinned here (card t_d77fd4cd):
  - ladder units: rung-major [D6, D1, D2, D3, D4] x arms [attn, wave] x seeds,
    attn first so the D6 centerpiece lands first;
  - rung flags exactly as run_difficulty.sh (D1 speed 2.30, D2 n-balls 8,
    D3 n-balls 12 + speed, D4 radius 0.8, D5 grid 64 + 8 balls,
    D6 grid 32 + eval-rollout 1024);
  - unfreeze recipe: --const-lr ON by default, steps 8000 default;
  - hybrid five-way order local_wave, local_ssm, wave_only, attn_only, ssm_only;
  - RESUME mode: skip result-exists, resume ckpt, SLICE_S timeout -> SLICED.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LADDER = Path(__file__).resolve().with_name("run_attn_ladder.sh")
HYBRID = Path(__file__).resolve().with_name("run_attn_hybrid_retest.sh")

LADDER_UNITS = [("attn", "_D6"), ("wave", "_D6"),
                ("attn", "_D1"), ("wave", "_D1"),
                ("attn", "_D2"), ("wave", "_D2"),
                ("attn", "_D3"), ("wave", "_D3"),
                ("attn", "_D4"), ("wave", "_D4")]
HYBRID_UNITS = [("wave", "_local_wave", "local_wave", "dispersion"),
                ("ssm", "_local_ssm", "local_ssm", "dispersion"),
                ("wave", "_wave_only", "none", "dispersion"),
                ("attn", "_attn_only", "none", "dispersion"),
                ("ssm", "_ssm_only", "none", "dispersion")]

RUNG_FLAGS = {"D1": {"speed": 2.30}, "D2": {"n_balls": 8},
              "D3": {"n_balls": 12, "speed": 2.30}, "D4": {"radius": 0.8},
              "D5": {"grid": 64, "n_balls": 8},
              "D6": {"grid": 32, "eval_rollout": 1024}}


def effective_options(arguments):
    """Parse observable options with argparse last-option-wins semantics."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("kind", "tag", "out", "kernel-version", "resume"):
        parser.add_argument("--" + name, default=None)
    parser.add_argument("--fuse", default="none")
    for name, default in (("steps", 2000), ("grid", 32), ("n-balls", 3),
                          ("seed", 0), ("target-params", 0),
                          ("eval-rollout", 256), ("eval-seeds", 32),
                          ("telemetry-every", 25)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("radius", 1.6), ("speed", 2.30)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--const-lr", action="store_true")
    parser.set_defaults(kernel_version="separable")
    return parser.parse_known_args(arguments[1:])[0]


def record_command():
    arguments = sys.argv[2:]
    options = effective_options(arguments)
    record_file = Path(os.environ["AL_TEST_RECORD"])
    first = not record_file.exists()
    status = None
    out = Path(options.out) if options.out else None
    if out and out.is_dir():
        found = sorted(out.glob("*status*.txt"))
        if found:
            status = found[0].read_text().strip()
    record = {"args": arguments, "status": status}
    with record_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print("Recorded", options.kind, options.tag)
    fail_target = os.environ.get("AL_TEST_FAIL_TAG")
    if fail_target and ":" in fail_target:
        fail_kind, fail_tag = fail_target.split(":", 1)
        if options.kind == fail_kind and options.tag == fail_tag:
            return 7
    if os.environ.get("AL_TEST_RC_ONCE") and first:
        return int(os.environ["AL_TEST_RC_ONCE"])
    return 0


class RunnerHarness(unittest.TestCase):
    script = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="al_runner_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = self.root / "calls.jsonl"
        self.out = self.root / "out dir"  # spaces in the path on purpose
        self.fake = self.root / "fake python"
        self.fake.write_text(
            '#!/usr/bin/env bash\n'
            'exec "$AL_TEST_PYTHON" "$AL_TEST_MODULE" --record "$@"\n',
            encoding="utf-8")
        self.fake.chmod(0o755)
        self.env = dict(os.environ, PY=str(self.fake), OUT=str(self.out),
                        AL_TEST_RECORD=str(self.record),
                        AL_TEST_PYTHON=sys.executable,
                        AL_TEST_MODULE=str(Path(__file__).resolve()))
        for var in ("AL_TEST_FAIL_TAG", "AL_TEST_RC_ONCE", "RESUME",
                    "CONST_LR", "SLICE_S", "STEPS", "RUNGS", "SEEDS"):
            self.env.pop(var, None)

    def run_matrix(self, **overrides):
        return subprocess.run(["bash", str(self.script)],
                              env=dict(self.env, **overrides),
                              cwd=self.root, text=True, capture_output=True,
                              timeout=120)

    def recorded(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]


class LadderRunnerTests(RunnerHarness):
    script = LADDER

    def test_default_units_order_flags_and_unfreeze_recipe(self):
        run = self.run_matrix(SEEDS="0")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(LADDER_UNITS))
        for call, (kind, tag) in zip(calls, LADDER_UNITS):
            opts = effective_options(call["args"])
            self.assertEqual(call["args"][0], "train_compare.py")
            self.assertEqual(call["status"], "RUNNING")
            self.assertEqual((opts.kind, opts.tag, opts.seed), (kind, tag, 0))
            self.assertEqual(opts.steps, 8000)              # unfreeze budget
            self.assertTrue(opts.const_lr)                  # unfreeze schedule
            self.assertEqual(opts.out, str(self.out))
            # base rung values
            self.assertEqual(opts.eval_seeds, 16)
            self.assertEqual(opts.target_params, 4000000)
            self.assertIsNone(opts.resume)                  # classic mode
            # rung overrides
            rung = tag[1:]
            flags = RUNG_FLAGS.get(rung, {})
            self.assertEqual(opts.grid, flags.get("grid", 32), rung)
            self.assertEqual(opts.eval_rollout, flags.get("eval_rollout", 256), rung)
            self.assertAlmostEqual(opts.speed, flags.get("speed", 2.30), places=6, msg=rung)
            self.assertEqual(opts.n_balls, flags.get("n_balls", 3), rung)
            self.assertAlmostEqual(opts.radius, flags.get("radius", 1.6), places=6, msg=rung)
            self.assertTrue((self.out / f"log_attn_ladder_{kind}{tag}.txt").is_file())
        progress = (self.out / "attn_ladder_progress.txt").read_text()
        self.assertEqual(progress.count("command:"), len(LADDER_UNITS))
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"DONE {len(LADDER_UNITS)}/{len(LADDER_UNITS)}\n")

    def test_rungs_env_subset_with_d5(self):
        run = self.run_matrix(SEEDS="0", RUNGS="D1 D5")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual([(effective_options(c["args"]).kind,
                           effective_options(c["args"]).tag) for c in calls],
                         [("attn", "_D1"), ("wave", "_D1"),
                          ("attn", "_D5"), ("wave", "_D5")])
        d5 = effective_options(calls[2]["args"])
        self.assertEqual((d5.grid, d5.n_balls, d5.eval_rollout), (64, 8, 256))
        d1 = effective_options(calls[0]["args"])
        self.assertAlmostEqual(d1.speed, 2.30, places=6)
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         "DONE 4/4\n")

    def test_unknown_rung_fails_fast(self):
        run = self.run_matrix(RUNGS="D9")
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.record.exists())
        self.assertFalse((self.out / "attn_ladder_status.txt").exists())

    def test_resume_mode_skips_finished_and_resumes_from_ckpt(self):
        self.out.mkdir(parents=True)
        (self.out / "result_attn_D6.json").write_text("{}")
        (self.out / "ckpt_wave_D6.pt").write_text("dummy-ckpt")
        run = self.run_matrix(SEEDS="0", RESUME="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(LADDER_UNITS) - 1)     # attn_D6 skipped
        first = effective_options(calls[0]["args"])
        self.assertEqual(first.tag, "_D6")
        self.assertEqual(first.resume, str(self.out / "ckpt_wave_D6.pt"))
        for call in calls[1:]:
            self.assertIsNone(effective_options(call["args"]).resume)
        progress = (self.out / "attn_ladder_progress.txt").read_text()
        self.assertEqual(progress.count("SKIP"), 1)
        self.assertIn("attn_D6 result-exists", progress)
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"DONE {len(LADDER_UNITS)}/{len(LADDER_UNITS)}\n")

    def test_slice_boundary_stops_suite_and_reports_sliced(self):
        run = self.run_matrix(SEEDS="0", RESUME="1", AL_TEST_RC_ONCE="124")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(self.recorded()), 1)               # stopped at first arm
        progress = (self.out / "attn_ladder_progress.txt").read_text()
        self.assertIn("attn_D6 SLICED", progress)
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"SLICED 0/{len(LADDER_UNITS)}\n")

    def test_resume_second_slice_continues(self):
        run1 = self.run_matrix(SEEDS="0", RESUME="1", AL_TEST_RC_ONCE="124")
        self.assertEqual(run1.returncode, 0, run1.stderr)
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"SLICED 0/{len(LADDER_UNITS)}\n")
        run2 = self.run_matrix(SEEDS="0", RESUME="1")
        self.assertEqual(run2.returncode, 0, run2.stderr)
        self.assertEqual(len(self.recorded()), 1 + len(LADDER_UNITS))
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"DONE {len(LADDER_UNITS)}/{len(LADDER_UNITS)}\n")

    def test_failed_arm_does_not_skip_remaining(self):
        run = self.run_matrix(SEEDS="0", AL_TEST_FAIL_TAG="attn:_D1")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(len(self.recorded()), len(LADDER_UNITS))
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         f"FAILED 1/{len(LADDER_UNITS)}\n")
        progress = (self.out / "attn_ladder_progress.txt").read_text()
        self.assertIn("attn_D1 exit=7", progress)
        self.assertIn("wave_D1 exit=0", progress)
        self.assertIn("wave_D4 exit=0", progress)

    def test_missing_interpreter_marks_failure(self):
        run = self.run_matrix(PY=str(self.root / "missing python"))
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.record.exists())
        self.assertEqual((self.out / "attn_ladder_status.txt").read_text(),
                         "FAILED exit=2\n")


class HybridRunnerTests(RunnerHarness):
    script = HYBRID

    def test_five_arm_matrix_order_and_flags(self):
        run = self.run_matrix(SEEDS="0")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(HYBRID_UNITS))
        for call, (kind, tag, fuse, kv) in zip(calls, HYBRID_UNITS):
            opts = effective_options(call["args"])
            self.assertEqual((opts.kind, opts.tag), (kind, tag))
            self.assertEqual((opts.fuse, opts.kernel_version), (fuse, kv))
            self.assertEqual(opts.steps, 8000)
            self.assertTrue(opts.const_lr)
            self.assertEqual(opts.seed, 0)
            self.assertEqual(opts.grid, 32)
            self.assertEqual(opts.out, str(self.out))
            self.assertTrue((self.out / f"log_hybrid_retest_{kind}{tag}.txt").is_file())
        self.assertEqual((self.out / "hybrid_retest_status.txt").read_text(),
                         f"DONE {len(HYBRID_UNITS)}/{len(HYBRID_UNITS)}\n")

    def test_resume_skips_and_completes(self):
        self.out.mkdir(parents=True)
        (self.out / "result_ssm_local_ssm.json").write_text("{}")
        run = self.run_matrix(SEEDS="0", RESUME="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = self.recorded()
        self.assertEqual(len(calls), len(HYBRID_UNITS) - 1)
        self.assertEqual(effective_options(calls[0]["args"]).tag, "_local_wave")
        progress = (self.out / "hybrid_retest_progress.txt").read_text()
        self.assertEqual(progress.count("SKIP"), 1)
        self.assertEqual((self.out / "hybrid_retest_status.txt").read_text(),
                         f"DONE {len(HYBRID_UNITS)}/{len(HYBRID_UNITS)}\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record"]:
        sys.exit(record_command())
    unittest.main()
