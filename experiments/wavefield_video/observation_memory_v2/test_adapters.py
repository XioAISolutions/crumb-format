"""Native compatibility/causality tests; the tiny optimizer test proves no long recall."""
import inspect
from pathlib import Path
import unittest

import torch

from adapters import (FeatureExtractor, MemoryHead, NATIVE_ROOT, VideoPredictor,
                      decode_prediction, extract_features)
from observation import ObservationConfig


def observations(gap=8, seed=14, h=4, w=4):
    generator = torch.Generator().manual_seed(seed)
    background = 0.25+0.5*torch.rand(1, 3, h, w, generator=generator)
    event = torch.zeros_like(background)
    event[:, :, 1, 2] = torch.tensor([.18, -.14, .12])
    frames = background[:, None].expand(-1, 32+gap, -1, -1, -1).clone()
    frames[:, 31] += event
    return frames, background, event


def tensor_states_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        case.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        case.assertEqual(a.keys(), b.keys())
        for key in a:
            tensor_states_equal(case, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            tensor_states_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class AdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_features_are_observed_signed_and_never_include_latch_or_targets(self):
        frames, _, event = observations()
        record = extract_features(frames.requires_grad_())
        self.assertFalse(record.features.requires_grad)
        self.assertLess(float(record.features.min()), -.1)
        self.assertGreater(float(record.features.max()), .1)
        self.assertTrue(torch.allclose(record.features[:, 31], event, atol=.01))
        self.assertLess(float(record.features[:, 32:].abs().max()), 1e-6)
        self.assertFalse(hasattr(record, "event_atlas"))
        self.assertFalse(hasattr(record, "target"))
        self.assertEqual(record.total_observed_frames, 40)
        self.assertEqual(record.features.shape[1], 40)

    def test_frontend_true_chunks_match_whole_prefix_and_suffix_poison_cannot_change_past(self):
        frames, _, _ = observations(gap=40)
        prefix, future = frames[:, :40], frames[:, 40:]
        poisoned = torch.full_like(future, .1)
        extractor = FeatureExtractor()
        first = extractor.extract(prefix[:, :32])
        second = extractor.extract(prefix[:, 32:])
        before = extractor.state_dict()
        record_a = extract_features(torch.cat((prefix, future), 1))
        record_b = extract_features(torch.cat((prefix, poisoned), 1))
        self.assertTrue(torch.equal(record_a.features[:, :40], record_b.features[:, :40]))
        self.assertTrue(torch.equal(torch.cat((first.features, second.features), 1), record_a.features[:, :40]))
        extractor.extract(poisoned)
        # Earlier packet tensors and snapshots are owned, never altered by a later chunk.
        fresh = FeatureExtractor()
        fresh.extract(prefix[:, :32])
        fresh.extract(prefix[:, 32:])
        tensor_states_equal(self, before, fresh.state_dict())
        self.assertEqual(first.frame_offset, 0)
        self.assertEqual(second.frame_offset, 32)
        self.assertEqual(second.total_observed_frames, 40)

    def test_native_actual_classes_and_wave_configuration(self):
        for kind in ("wave", "ssm", "attn"):
            head = MemoryHead(kind, 4, 4)
            self.assertIsInstance(head.native, VideoPredictor)
            self.assertEqual(Path(inspect.getfile(type(head.native))).resolve(), NATIVE_ROOT / "wfvideo.py")
            self.assertEqual(head.native.time_pos, "none")
            self.assertEqual(type(head.native.blocks[0].mix).__name__,
                             {"wave": "WaveMix3D", "ssm": "SSMLite", "attn": "AttnMix"}[kind])
        wave = MemoryHead("wave", 4, 4).native
        mix = wave.blocks[0].mix
        self.assertEqual(mix.kernel_version, "dispersion")
        self.assertEqual(mix.pole_param, "halflife")
        self.assertTrue(mix.linear_pad)
        self.assertIsNone(wave.embed.bias)
        self.assertIsNone(mix.pi.bias)
        self.assertIsNone(wave.posemb.py)

    def test_all_three_native_backward_finite_and_actual_parameter_update(self):
        frames, _, target_event = observations(gap=8)
        record = extract_features(frames)
        for kind in ("wave", "ssm", "attn"):
            with self.subTest(kind=kind):
                torch.manual_seed(42)
                head = MemoryHead(kind, 4, 4)
                before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
                optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0)
                predicted, state = head.ingest(record.features)
                self.assertEqual(state.processed_frames, 40)
                self.assertTrue(bool(torch.isfinite(predicted).all()))
                # Target remains solely in the test loss; extract/head APIs never receive it.
                loss = (predicted-target_event).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                mixer_gradients = [p.grad for p in head.native.blocks[0].mix.parameters() if p.grad is not None]
                self.assertGreater(sum(float(g.abs().sum()) for g in mixer_gradients), 0)
                optimizer.step()
                changed = {name: float((parameter.detach()-before[name]).abs().sum())
                           for name, parameter in head.named_parameters()}
                self.assertGreater(sum(changed.values()), 0)
                self.assertGreater(sum(value for name, value in changed.items() if ".mix." in name), 0)
                self.assertEqual(head.native.T, 32)
                self.assertEqual(head.native.blocks[0].mix.T, 32)

    def test_recurrent_true_partial_chunks_match_native_frame_steps(self):
        features = extract_features(observations()[0]).features
        for kind in ("wave", "ssm"):
            with self.subTest(kind=kind):
                torch.manual_seed(9)
                head = MemoryHead(kind, 4, 4)
                prediction, chunk_state = head.predict(features)
                frame_state = head.native.stream_init(1, "cpu")
                with torch.no_grad():
                    for index in range(features.shape[1]):
                        streamed, frame_state = head.native.stream_step(features[:, index], frame_state, index)
                self.assertTrue(torch.allclose(prediction, streamed, atol=2e-5, rtol=2e-5))
                for actual, expected in zip(chunk_state.recurrent, frame_state):
                    self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_public_native_frame_step_prefix_predictions_ignore_suffix_poisoning(self):
        prefix = torch.zeros(1, 32, 3, 4, 4)
        prefix[:, 3, :, 1, 2] = torch.tensor([.18, -.14, .12])
        poisoned = prefix.clone()
        poisoned[:, 12:] = torch.randn(1, 20, 3, 4, 4, generator=torch.Generator().manual_seed(31))
        for kind in ("wave", "ssm", "attn"):
            with self.subTest(kind=kind):
                torch.manual_seed(42)
                head = MemoryHead(kind, 4, 4)
                with torch.no_grad():
                    clean, state = head.prefix_predictions(prefix)
                    changed, poisoned_state = head.prefix_predictions(poisoned)
                self.assertEqual(state.processed_frames, 32)
                self.assertEqual(poisoned_state.processed_frames, 32)
                self.assertTrue(torch.equal(clean[:, :12], changed[:, :12]))
                for length in (1, 4, 12):
                    causal = head.predict(prefix[:, :length])[0]
                    self.assertTrue(torch.allclose(causal, clean[:, length-1], atol=3e-5, rtol=3e-5))

    def test_optional_native_checkpoint_partial_chunk_fails_before_ingestion(self):
        features = extract_features(observations()[0]).features
        for kind in ("wave", "ssm"):
            with self.subTest(kind=kind):
                head = MemoryHead(kind, 4, 4)
                head.native.grad_ckpt = True
                with self.assertRaisesRegex(ValueError, "partial.*grad_ckpt=False"):
                    head(features)
                self.assertEqual(head.native.T, 32)
                self.assertEqual(head.native.blocks[0].mix.T, 32)
                # Full true-length chunks retain the existing checkpoint path.
                loss = head(features[:, :32]).square().mean()
                loss.backward()
                self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                                    for p in head.parameters() if p.grad is not None))

    def test_wave_scope_rejects_unverified_multilayer_fft_chunk_path(self):
        for layers in (2, 4):
            with self.assertRaisesRegex(ValueError, "wave.*layers=1.*parity"):
                MemoryHead("wave", 4, 4, layers=layers)

    def test_attention_consumes_full_256_gap_but_cannot_keep_old_cue(self):
        frames, _, _ = observations(gap=256)
        features = extract_features(frames).features
        head = MemoryHead("attn", 4, 4)
        prediction, state = head.predict(features)
        zero_prediction, zero_state = head.predict(torch.zeros_like(features))
        self.assertEqual(state.processed_frames, 288)
        self.assertEqual(state.window.shape[1], 32)
        self.assertEqual(state.window.untyped_storage().nbytes(), state.window.numel()*state.window.element_size())
        self.assertTrue(torch.equal(state.window, zero_state.window))
        self.assertTrue(torch.equal(prediction, zero_prediction))
        self.assertFalse(prediction.requires_grad)
        self.assertFalse(state.window.requires_grad)

    def test_no_gradient_predict_and_prefix_state_is_immutable_under_future_ingest(self):
        features = extract_features(observations(gap=40)[0]).features.requires_grad_()
        for kind in ("wave", "ssm", "attn"):
            with self.subTest(kind=kind):
                head = MemoryHead(kind, 4, 4)
                past, state = head.predict(features[:, :32])
                past_copy = past.clone()
                if state.recurrent is not None:
                    saved = [x.clone() for x in state.recurrent]
                else:
                    saved = state.window.clone()
                _, future_state = head.predict(features[:, 32:], state)
                self.assertEqual(future_state.processed_frames, 72)
                self.assertTrue(torch.equal(past, past_copy))
                tensor_states_equal(self, saved, state.recurrent if state.recurrent is not None else state.window)
                self.assertFalse(past.requires_grad)
                tensors = state.recurrent if state.recurrent is not None else [state.window]
                self.assertTrue(all(not x.requires_grad for x in tensors))

    def test_decode_uses_own_reference_observed_pose_and_photometry_without_clamp(self):
        frames, background, _ = observations()
        frames[:, 1:] = frames[:, 1:] * 1.2 + .02
        extractor = FeatureExtractor(ObservationConfig(allow_vertical=False))
        record = extractor.extract(frames)
        zero = decode_prediction(torch.zeros_like(background), record)
        self.assertAlmostEqual(float(record.gains[0]), 1.2, delta=.025)
        self.assertAlmostEqual(float(record.offsets[0]), .02, delta=.015)
        self.assertTrue(torch.allclose(zero.rgb, 1.2*background+.02, atol=.01))
        signed = torch.full_like(background, -2.0, requires_grad=True)
        decoded = decode_prediction(signed, record)
        self.assertLess(float(decoded.rgb.detach().min()), 0)
        decoded.rgb.sum().backward()
        self.assertIsNotNone(signed.grad)
        self.assertTrue(bool(torch.isfinite(signed.grad).all()))
        self.assertTrue(torch.equal(decoded.reference_canonical, record.reference_canonical))

    def test_decode_fractional_forecast_view_matches_independent_camera_render(self):
        world = .25+.5*torch.rand(1, 3, 4, 12, generator=torch.Generator().manual_seed(83))
        first = world[:, :, :, :4]
        shifted = .5*world[:, :, :, :4]+.5*world[:, :, :, 1:5]
        observed = torch.stack((first, 1.2*shifted+.02, 1.2*shifted+.02), 1)
        record = extract_features(observed, ObservationConfig(allow_vertical=False))
        self.assertAlmostEqual(float(record.forecast_positions[0, 0]), .5, delta=.125)
        event = torch.zeros_like(first)
        event[:, :, 1, 2] = torch.tensor([.18, -.14, .12])
        padded = torch.cat((event, torch.zeros(1, 3, 4, 1)), -1)
        expected_view_event = .5*padded[:, :, :, :4]+.5*padded[:, :, :, 1:5]
        decoded = decode_prediction(event, record)
        self.assertTrue(torch.allclose(decoded.view_event, expected_view_event, atol=.01))
        self.assertTrue(torch.allclose(decoded.rgb, 1.2*(shifted+expected_view_event)+.02, atol=.01))


if __name__ == "__main__":
    unittest.main(verbosity=2)
