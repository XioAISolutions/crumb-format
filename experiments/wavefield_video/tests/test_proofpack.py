"""Integrity and portability checks; no torch dependency or real training run."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from proofpack import create_proofpack
from verify_proofpack import ProofpackError, sha256_file, verify_proofpack


class ProofpackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wavefield-proofpack-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.run = self.base / "training run"
        self.run.mkdir()
        (self.run / "result_wave_test.json").write_text('{"eval_mse":0.1}\n')
        (self.run / "model_wave_test.pt").write_bytes(b"opaque-checkpoint\x00\xff")
        (self.run / "nested").mkdir()
        (self.run / "nested" / "log.txt").write_text("training log\n")
        self.config = self.base / "configuration.json"
        self.config.write_text('{"kind":"wave","steps":1}\n')
        self.output = self.base / "proof bundle"
        self.command = "python3 train_compare.py --kind wave --steps 1 --out 'training run'"

    def build(self, **kwargs):
        options = dict(config=self.config, repro_commands=[self.command], repro_cwd=str(SOURCE))
        options.update(kwargs)
        return create_proofpack(self.run, self.output, **options)

    def rewrite_manifest(self, change):
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def test_bundle_roundtrip_keeps_bytes_and_exact_commands(self):
        original = {path.relative_to(self.run).as_posix(): sha256_file(path)
                    for path in self.run.rglob("*") if path.is_file()}
        output = self.build()
        manifest = verify_proofpack(output)
        self.assertEqual(manifest["reproduction"]["commands"], [self.command])
        for relative, digest in original.items():
            self.assertEqual(sha256_file(output / "run" / relative), digest)
            self.assertEqual(sha256_file(self.run / relative), digest)
        self.assertEqual((output / "config" / self.config.name).read_bytes(), self.config.read_bytes())
        self.assertIn(self.command, (output / "VERIFY.md").read_text())
        self.assertEqual({entry["role"] for entry in manifest["files"]},
                         {"result", "checkpoint", "config", "artifact", "instructions", "verifier"})

    def test_copied_verifier_is_standalone_after_original_inputs_removed(self):
        self.build()
        self.config.unlink()
        (self.run / "model_wave_test.pt").unlink()
        process = subprocess.run([sys.executable, "-I", "verify_proofpack.py", "."],
                                 cwd=self.output, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("PASS proofpack", process.stdout)

    def test_cli_creation_and_verification(self):
        process = subprocess.run(
            [sys.executable, str(SOURCE / "proofpack.py"), str(self.run), str(self.output),
             "--config", str(self.config), "--repro-command", self.command],
            capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn(sha256_file(self.output / "manifest.json"), process.stdout)
        (self.output / "run" / "model_wave_test.pt").write_bytes(b"damaged")
        failed = subprocess.run([sys.executable, str(SOURCE / "verify_proofpack.py"), str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("SHA-256 mismatch", failed.stderr)

    def test_same_length_checkpoint_tampering_fails_hash(self):
        self.build()
        checkpoint = self.output / "run" / "model_wave_test.pt"
        checkpoint.write_bytes(b"x" * checkpoint.stat().st_size)
        with self.assertRaisesRegex(ProofpackError, "SHA-256 mismatch"):
            verify_proofpack(self.output)

    def test_missing_artifact_fails(self):
        self.build()
        (self.output / "run" / "result_wave_test.json").unlink()
        with self.assertRaisesRegex(ProofpackError, "inventory mismatch.*missing="):
            verify_proofpack(self.output)

    def test_extra_artifact_fails(self):
        self.build()
        (self.output / "unlisted.txt").write_text("extra")
        with self.assertRaisesRegex(ProofpackError, "extra=.*unlisted"):
            verify_proofpack(self.output)

    def test_hashed_instructions_tampering_fails(self):
        self.build()
        (self.output / "VERIFY.md").write_text("other commands")
        with self.assertRaisesRegex(ProofpackError, "SHA-256 mismatch: VERIFY.md"):
            verify_proofpack(self.output)

    def test_symlink_inputs_and_bundle_files_are_rejected(self):
        (self.run / "linked.txt").symlink_to(self.config)
        with self.assertRaisesRegex(ProofpackError, "symlink"):
            self.build()
        (self.run / "linked.txt").unlink()
        self.build()
        checkpoint = self.output / "run" / "model_wave_test.pt"
        checkpoint.unlink()
        checkpoint.symlink_to(self.run / checkpoint.name)
        with self.assertRaisesRegex(ProofpackError, "symlink"):
            verify_proofpack(self.output)

    def test_schema_rejects_unsafe_paths_duplicates_and_invalid_hashes(self):
        self.build()
        pristine = (self.output / "manifest.json").read_text()
        mutations = [
            lambda m: m.update(schema_version=2),
            lambda m: m.update(schema_version=True),
            lambda m: m["files"][0].update(path="../escaped"),
            lambda m: m["files"][0].update(path="/absolute"),
            lambda m: m["files"][0].update(path="run/./aliased"),
            lambda m: m["files"][0].update(path="run\\windows"),
            lambda m: m["files"].append(dict(m["files"][0])),
            lambda m: m["files"][0].update(sha256="z" * 64),
            lambda m: m["files"][0].update(size_bytes=True),
            lambda m: m["files"][0].update(role=[]),
            lambda m: m["reproduction"].update(commands=[]),
            lambda m: m.update(files=[]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                (self.output / "manifest.json").write_text(pristine)
                self.rewrite_manifest(mutation)
                with self.assertRaises(ProofpackError):
                    verify_proofpack(self.output)

    def test_duplicate_json_keys_and_malformed_json_are_rejected(self):
        self.build()
        for text in ('{"format":"a","format":"b"}', '{"format":NaN}', '{broken'):
            with self.subTest(text=text):
                (self.output / "manifest.json").write_text(text)
                with self.assertRaises(ProofpackError):
                    verify_proofpack(self.output)

    def test_config_and_exact_reproduction_are_required(self):
        with self.assertRaisesRegex(ProofpackError, "no config"):
            self.build(config=None)
        with self.assertRaisesRegex(ProofpackError, "exact --repro-command"):
            self.build(repro_commands=None)
        self.assertFalse(self.output.exists())

    def test_auto_config_and_recorded_commands(self):
        (self.run / "config.json").write_text(json.dumps({"steps": 1, "repro_commands": [self.command]}))
        self.build(config=None, repro_commands=None)
        manifest = verify_proofpack(self.output)
        config_entries = [entry for entry in manifest["files"] if entry["role"] == "config"]
        self.assertEqual([entry["path"] for entry in config_entries], ["run/config.json"])
        self.assertEqual(manifest["reproduction"]["commands"], [self.command])

    def test_never_executes_recorded_commands(self):
        marker = self.base / "should-not-exist"
        self.build(repro_commands=[f"touch '{marker}'", "printf '```'", "exit 77"])
        verify_proofpack(self.output)
        self.assertFalse(marker.exists())
        instructions = (self.output / "VERIFY.md").read_text()
        self.assertIn("````sh", instructions)
        self.assertIn("exit 77", instructions)

    def test_existing_output_and_nested_output_refused(self):
        self.build()
        with self.assertRaisesRegex(ProofpackError, "already exists"):
            self.build()
        with self.assertRaisesRegex(ProofpackError, "outside"):
            create_proofpack(self.run, self.run / "bundle", config=self.config,
                             repro_commands=[self.command])

    def test_missing_checkpoint_or_invalid_result_refused(self):
        checkpoint = self.run / "model_wave_test.pt"
        checkpoint.unlink()
        with self.assertRaisesRegex(ProofpackError, "checkpoint"):
            self.build()
        checkpoint.write_bytes(b"opaque")
        (self.run / "result_wave_test.json").write_text("[]")
        with self.assertRaisesRegex(ProofpackError, "result JSON must be an object"):
            self.build()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
