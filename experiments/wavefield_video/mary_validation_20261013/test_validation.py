"""CPU contract/negative fixtures, never evidence about pretrained weights/GPU quality."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from validation import (FairProtocol, GateError, assert_frozen, decoded_health_report,
                        pixel_metrics, probe_decoder_chunking, probe_raw_future_causality,
                        sha256_file, verify_local_manifest)
from run_preflight import main as run_preflight


class FixtureCodec(torch.nn.Module):
    """CPU arithmetic fixture; deliberately not a production VAE."""
    t_stride = 4

    def __init__(self, acausal=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.acausal = acausal
        self.eval()

    def encode(self, frames):
        out = frames[:, ::self.t_stride].clone()*self.scale
        return out + frames.mean(1, keepdim=True) if self.acausal else out

    def decode(self, latents):
        return torch.cat((latents[:, :1], latents[:, 1:].repeat_interleave(4, dim=1)), 1)


class FixtureFeatures(torch.nn.Module):
    """Arithmetic feature fixture to test fixed anchoring, not pretrained features."""
    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, frames):
        return frames.mean((-1, -2))


def raw_frames():
    return .2+.6*torch.rand(1, 33, 3, 8, 8, generator=torch.Generator().manual_seed(22))


def full_run_rows(protocol):
    base = {"vae_fingerprint": "pinned_vae", "dataset_fingerprint": "heldout_split_v1",
            "raw_context_frames": 33, "raw_hidden_gap": 256, "raw_observed_frames": 289,
            "training_updates": protocol.training_updates, "loss_id": "same_loss_v1",
            "heldout_source_ids": ["source_b", "source_c"], "t_stride": 4,
            "target_raw_interval": [289, 293], "decoder_lookahead_latents": 0,
            "heldout_raw_generation_frames": protocol.heldout_raw_generation_frames,
            "completed": True, "completed_updates": protocol.training_updates}
    return [{**base, "arm": arm, "initialization_seed": seed}
            for arm in protocol.required_arms for seed in protocol.initialization_seeds]


class ValidationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_raw_future_and_longer_context_pass_for_causal_arithmetic_fixture(self):
        frames = raw_frames()
        original = frames.clone()
        report = probe_raw_future_causality(FixtureCodec(), frames, (1, 9, 17))
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["rows"]), 9)
        self.assertTrue(torch.equal(frames, original))
        self.assertTrue(all(row["poison_delta_max"] == 0 for row in report["rows"]))

    def test_full_clip_mean_and_future_pooled_encoder_are_rejected(self):
        report = probe_raw_future_causality(FixtureCodec(acausal=True), raw_frames(), (9, 17))
        self.assertFalse(report["pass"])
        self.assertTrue(any(row["longer_context_delta_max"] > .01 for row in report["rows"]))
        self.assertTrue(any(row["poison_delta_max"] > .01 for row in report["rows"]))

    def test_short_probe_cannot_certify_length_triggered_future_pooling_at_training_horizon(self):
        class TiledFuturePool(FixtureCodec):
            def encode(self, frames):
                out = super().encode(frames)
                return out+frames.mean(1, keepdim=True) if frames.shape[1] > 128 else out
        short = probe_raw_future_causality(TiledFuturePool(), raw_frames(), (9, 17))
        self.assertTrue(short["pass"])
        self.assertFalse(short["covers_required_context"])
        long = .2+.6*torch.rand(1, 293, 3, 8, 8, generator=torch.Generator().manual_seed(27))
        actual_horizon = probe_raw_future_causality(TiledFuturePool(), long, (33, 129, 289),
                                                   required_context_frames=289)
        self.assertFalse(actual_horizon["pass"])
        self.assertTrue(actual_horizon["covers_required_context"])
        with self.assertRaisesRegex(GateError, "entire learned"):
            probe_raw_future_causality(FixtureCodec(), raw_frames(), (9, 17), required_context_frames=289)

    def test_invalid_raw_grid_cannot_silently_drop_last_observations(self):
        with self.assertRaisesRegex(GateError, "no trimming"):
            probe_raw_future_causality(FixtureCodec(), torch.rand(1, 288, 3, 8, 8), (33,))
        with self.assertRaisesRegex(GateError, "valid observed prefix"):
            probe_raw_future_causality(FixtureCodec(), raw_frames(), (8,))

    def test_trainable_or_training_mode_encoder_fails_before_encoding(self):
        codec = FixtureCodec()
        codec.scale.requires_grad_(True)
        with self.assertRaisesRegex(GateError, "every parameter frozen"):
            probe_raw_future_causality(codec, raw_frames(), (9,))
        codec.scale.requires_grad_(False)
        codec.train()
        with self.assertRaises(GateError):
            assert_frozen(codec)

    def test_production_manifest_pin_and_file_integrity_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": 16}))
            (root / "diffusion_pytorch_model.safetensors").write_bytes(b"contract fixture, not real weights")
            manifest = {"backend": "wan", "repository": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        "revision": "a"*40, "source_kind": "official_huggingface_snapshot",
                        "files": {name: sha256_file(root/name) for name in
                                  ("config.json", "diffusion_pytorch_model.safetensors")}}
            path = root / "provenance.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(GateError, "approved manifest"):
                verify_local_manifest(root, path, "", "wan")
            proof = verify_local_manifest(root, path, sha256_file(path), "wan")
            self.assertEqual(proof["revision"], "a"*40)
            self.assertIn("hashes alone do not prove training", proof["provenance_scope"])
            (root / "diffusion_pytorch_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(GateError, "artifact changed"):
                verify_local_manifest(root, path, sha256_file(path), "wan")

    def test_tiny_home_encoder_or_renamed_tiny_config_cannot_pass_selection(self):
        for backend in ("wan-tiny", "ltx-tiny", "conv"):
            with self.assertRaisesRegex(GateError, "no conv/tiny"):
                verify_local_manifest("/nonexistent", "/nonexistent", "a"*64, backend)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": 4}))
            (root / "weights.safetensors").write_bytes(b"tiny fixture")
            manifest = {"backend": "wan", "repository": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        "revision": "a"*40, "source_kind": "official_huggingface_snapshot",
                        "files": {name: sha256_file(root/name) for name in ("config.json", "weights.safetensors")}}
            path = root / "provenance.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(GateError, "tiny/random"):
                verify_local_manifest(root, path, sha256_file(path), "wan")

    def test_weight_shard_index_cannot_escape_the_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": 16}))
            (root / "unused.safetensors").write_bytes(b"fixture")
            name = "diffusion_pytorch_model.safetensors.index.json"
            (root / name).write_text(json.dumps({"weight_map": {"encoder.weight": "../unhashed.safetensors"}}))
            manifest = {"backend": "wan", "repository": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        "revision": "a"*40, "source_kind": "official_huggingface_snapshot",
                        "files": {file: sha256_file(root/file) for file in ("config.json", "unused.safetensors", name)}}
            path = root / "provenance.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(GateError, "unhashed or escaping"):
                verify_local_manifest(root, path, sha256_file(path), "wan")

    def test_raw_256_gap_is_identical_across_all_registered_arms_and_seeds(self):
        protocol = FairProtocol()
        for stride in (4, 8):
            mapping = protocol.mapping(stride)
            self.assertEqual(mapping["observed_raw_frames_per_arm"], 289)
            self.assertEqual(mapping["hidden_raw_interval"], [33, 289])
            self.assertEqual(mapping["first_unseen_raw_index"], 289)
            self.assertFalse(mapping["attention_training_gap_capped"])
        rows = full_run_rows(protocol)
        protocol.assert_run_rows(rows)
        rows[2] = {**rows[2], "raw_hidden_gap": 24}
        with self.assertRaisesRegex(GateError, "common budget"):
            protocol.assert_run_rows(rows)

    def test_missing_seeds_and_shipping_comparators_block_a_fairness_claim(self):
        protocol = FairProtocol()
        with self.assertRaisesRegex(GateError, "all three"):
            protocol.assert_run_rows(full_run_rows(protocol)[:-1])
        with self.assertRaisesRegex(GateError, "shipping comparators"):
            replace(protocol, required_arms=("wave", "generic_ssm", "bounded_sdpa")).validate(4)

    def test_invalid_stride_and_unfinished_or_empty_heldout_run_rows_are_rejected(self):
        protocol = FairProtocol()
        for stride in (-4, 0, 4.0, True):
            with self.assertRaisesRegex(GateError, "positive integers"):
                protocol.mapping(stride)
        for changes in ({"completed": False}, {"completed_updates": 0}, {"heldout_source_ids": []}):
            rows = [{**row, **changes} for row in full_run_rows(protocol)]
            with self.assertRaises(GateError):
                protocol.assert_run_rows(rows)

    def test_native_pixel_first_collapse_uses_fixed_external_context_and_per_sample_flags(self):
        context = raw_frames()[:, :5].expand(2, -1, -1, -1, -1).clone()
        generated = context[:, :4].clone()
        generated[0].zero_()
        report = decoded_health_report(context, generated, fps=24, stride=4, patience=2)
        self.assertEqual(report["native_resolution_hw"], [8, 8])
        self.assertEqual(report["anchor_source"], "separate_observed_clean_context")
        self.assertEqual(report["first_collapse"]["fade"]["decoded_frame"], 0)
        self.assertEqual(report["first_collapse"]["fade"]["sample"], 0)
        self.assertEqual(report["first_collapse"]["fade"]["latent_step"], 0)
        self.assertEqual(report["semantic_quality"], "unverified")

    def test_nonpretrained_feature_fallback_rejected_and_fixture_anchor_cannot_renormalize_drift(self):
        context = torch.zeros(1, 5, 3, 8, 8)
        context[:, :, 0] = .8
        generated = torch.zeros_like(context)
        generated[:, :, 1] = .8
        with self.assertRaisesRegex(GateError, "no pixel/random fallback"):
            decoded_health_report(context, generated, features=FixtureFeatures(), patience=2)
        report = decoded_health_report(context, generated, features=FixtureFeatures(), patience=2,
                                       fixture_features=True)
        feature = report["frozen_feature_screen"]
        self.assertEqual(feature["status"], "fixture_only_not_pretrained")
        self.assertEqual(feature["reference"], "fixed_clean_context")
        self.assertEqual(feature["first_drift"]["decoded_frame"], 0)
        self.assertTrue(all(row[0] < .1 for row in feature["cosine_per_frame_sample"]))

    def test_decoded_pixel_metric_is_not_latent_error(self):
        reference = raw_frames()
        self.assertTrue(pixel_metrics(reference, reference)["exact_match"])
        self.assertGreater(pixel_metrics(torch.zeros_like(reference), reference)["mse"], .1)
        with self.assertRaises(GateError):
            pixel_metrics(torch.zeros(1, 2, 4, 2, 2), reference)

    def test_actual_native_decoder_stream_timeline_and_resume_gate_with_arithmetic_codec(self):
        report = probe_decoder_chunking(FixtureCodec(), 3, 4, 4, max_rf_latents=40)
        self.assertTrue(report["pass"])
        self.assertTrue(report["resume_tested"])
        self.assertEqual(report["rf_forward_latents"], 0)
        self.assertEqual([row["decoded_frames"] for row in report["rows"]], [112, 112])
        self.assertTrue(all(row["delta_max"] == 0 for row in report["rows"]))
        self.assertFalse(report["native_resolution_quality_evidence"])

    def test_decoder_with_full_clip_normalization_has_no_bounded_receptive_field(self):
        class FutureDecoder(FixtureCodec):
            def decode(self, latents):
                return super().decode(latents)+latents.mean(1, keepdim=True)
        with self.assertRaisesRegex(GateError, "receptive-field probe rejected"):
            probe_decoder_chunking(FutureDecoder(), 3, 4, 4, max_rf_latents=40)

    def test_cli_smoke_and_missing_production_artifacts_cannot_claim_pretrained_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            smoke = Path(temporary) / "smoke"
            self.assertEqual(run_preflight(["--smoke", "--out", str(smoke)]), 0)
            report = json.loads((smoke / "receipt.json").read_text())
            self.assertFalse(report["pretrained_evidence"])
            self.assertFalse(report["GPU_evidence"])
            self.assertTrue(report["decoder_fixture"]["pass"])
            self.assertFalse(report["acausal_negative_fixture"]["pass"])
            blocked = Path(temporary) / "blocked"
            self.assertEqual(run_preflight(["--out", str(blocked)]), 1)
            failure = json.loads((blocked / "receipt.json").read_text())
            self.assertEqual(failure["status"], "preflight_failed")
            self.assertFalse(failure["pretrained_evidence"])
            self.assertFalse(failure["GPU_evidence"])
            with self.assertRaises(FileExistsError):
                run_preflight(["--smoke", "--out", str(smoke)])

    def test_latent_health_cannot_be_mislabeled_as_decoded_rgb_health(self):
        with self.assertRaises(GateError):
            decoded_health_report(torch.zeros(1, 4, 16, 8, 8), torch.zeros(1, 4, 16, 8, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
