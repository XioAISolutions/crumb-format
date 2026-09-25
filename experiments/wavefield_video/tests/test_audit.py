"""Fast regression tests for audit isolation and accurate failure reporting."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


AUDIT = Path(__file__).resolve().parents[1] / "audit_wavefield.sh"


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-wavefield-audit-")
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()
        shutil.copyfile(AUDIT, self.root / AUDIT.name)
        (self.root / "example.py").write_text("VALUE = 1\n")
        for version in (3, 4):
            (self.root / f"run_smoke_v{version}.sh").write_text(
                '#!/usr/bin/env bash\nset -e\ncd "$(dirname "$0")"\n'
                '[ -z "$CUDA_VISIBLE_DEVICES" ]\n'
                f'[ "$OUT" = "smoke_v{version}" ]\n'
                'mkdir -p "$OUT"\necho fresh > "$OUT/created.txt"\n'
            )
        self.output = Path(self.temp.name) / "audit output"

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self):
        env = os.environ.copy()
        env.update(PY=sys.executable, OUT="inherited-output-must-be-ignored")
        result = subprocess.run(
            ["bash", str(self.root / AUDIT.name), str(self.output)],
            env=env, capture_output=True, text=True, check=False,
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        return result, manifest, {c["name"]: c for c in manifest["checks"]}

    def test_pass_isolated_and_nested_python_compiled(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "another.py").write_text("VALUE = 2\n")
        old = self.root / "smoke_v4"
        old.mkdir()
        (old / "created.txt").write_text("keep original\n")
        result, manifest, checks = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(checks["py_compile"]["files"], 2)
        self.assertEqual((old / "created.txt").read_text(), "keep original\n")
        self.assertEqual((self.output / "source/smoke_v4/created.txt").read_text(), "fresh\n")
        self.assertFalse((self.root / "__pycache__").exists())
        self.assertTrue((self.output / "pycache/nested/another.pyc").is_file())
        self.assertIn("PASS/FAIL manifest", result.stdout)

    def test_compile_failure_does_not_skip_smokes(self):
        (self.root / "broken.py").write_text("def broken(:\n")
        result, manifest, checks = self.run_audit()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(manifest["status"], "FAIL")
        self.assertEqual(checks["py_compile"]["status"], "FAIL")
        self.assertEqual(checks["smoke_v3"]["status"], "PASS")
        self.assertEqual(checks["smoke_v4"]["status"], "PASS")

    def test_pipeline_failure_propagates_and_v4_still_runs(self):
        with (self.root / "run_smoke_v3.sh").open("a") as script:
            script.write("(exit 7) | tail -4\n")
        result, _, checks = self.run_audit()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(checks["smoke_v3"]["exit_code"], 7)
        self.assertEqual(checks["smoke_v4"]["status"], "PASS")

    def test_sanity_failure_text_propagates_without_nonzero_exit(self):
        with (self.root / "run_smoke_v3.sh").open("a") as script:
            script.write("echo 'SANITY FAILURES-PRESENT'\n")
        result, _, checks = self.run_audit()
        self.assertEqual(result.returncode, 1)
        self.assertTrue(checks["smoke_v3"]["reported_failure"])

    def test_existing_output_is_refused(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel"
        sentinel.write_text("keep\n")
        result = subprocess.run(
            ["bash", str(self.root / AUDIT.name), str(self.output)],
            env={**os.environ, "PY": sys.executable},
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(), "keep\n")
        self.assertIn("FAIL audit_setup", result.stderr)


if __name__ == "__main__":
    unittest.main()
