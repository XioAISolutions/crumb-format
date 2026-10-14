"""Observed-data contracts and counterexamples, independent of implementation details."""
import unittest

import torch

from observation import ObservationConfig, ObservationMemory, observe


def scene(seed=17, h=32, w=40):
    return 0.2 + 0.6 * torch.rand(3, h, w, generator=torch.Generator().manual_seed(seed))


def crop(world, x=8.0, y=8.0, h=16, w=16):
    """Independent finite-world bilinear camera renderer, used only by tests."""
    ix, iy = int(x), int(y)
    fx, fy = x-ix, y-iy
    return ((1-fy) * ((1-fx)*world[:, iy:iy+h, ix:ix+w] + fx*world[:, iy:iy+h, ix+1:ix+w+1]) +
            fy * ((1-fx)*world[:, iy+1:iy+h+1, ix:ix+w] + fx*world[:, iy+1:iy+h+1, ix+1:ix+w+1]))


def equal_state(case, a, b):
    case.assertEqual(a.keys(), b.keys())
    for key in a:
        if isinstance(a[key], torch.Tensor):
            case.assertTrue(torch.equal(a[key], b[key]), key)
        else:
            case.assertEqual(a[key], b[key], key)


class ObservationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_known_subpixel_translation_in_two_dimensions(self):
        world = scene()
        memory = ObservationMemory()
        memory.step(crop(world))
        shifted = crop(world, 8.75, 8.5)
        result = memory.step(shifted)
        self.assertTrue(result.accepted, result.reason)
        self.assertLessEqual(abs(result.position[0] - 0.75), 0.125)
        self.assertLessEqual(abs(result.position[1] - 0.5), 0.125)
        self.assertGreater(result.confidence, 0.5)
        self.assertGreaterEqual(result.coverage, 0.7)
        error = (result.normalized-result.reference).abs() * result.known
        self.assertLess(float(error.sum() / (3*result.known.sum())), 0.015)

    def test_gain_and_offset_normalize_both_brightness_directions(self):
        frame = crop(scene(34))
        for gain, offset in ((1.2, 0.03), (0.8, -0.025)):
            with self.subTest(gain=gain):
                memory = ObservationMemory()
                memory.step(frame)
                result = memory.step(gain*frame + offset)
                self.assertTrue(result.accepted, result.reason)
                self.assertAlmostEqual(result.gain, gain, delta=0.025)
                self.assertAlmostEqual(result.offset, offset, delta=0.015)
                self.assertLess(float((result.normalized-frame).abs().max()), 0.015)
                self.assertFalse(bool(result.accepted_write_mask.any()))
                self.assertLess(float((memory.forecast()["rgb"]-(gain*frame+offset)).abs().max()), 0.015)

    def test_full_occlusion_freezes_retained_state_and_existing_latch(self):
        frame = crop(scene())
        memory = ObservationMemory()
        memory.step(frame)
        changed = frame.clone()
        changed[:, 5:7, 6:8] += torch.tensor([0.16, -0.12, 0.10])[:, None, None]
        self.assertTrue(memory.step(changed).accepted)
        before = memory.state_dict()
        forecast = memory.read()["rgb"]
        result = memory.step(torch.full_like(frame, 0.1))
        self.assertTrue(result.hold)
        self.assertFalse(bool(result.accepted_write_mask.any()))
        equal_state(self, before, memory.state_dict())
        self.assertTrue(torch.equal(forecast, memory.read()["rgb"]))

    def test_repeating_translation_pattern_is_rejected(self):
        tile = crop(scene(5))[:, :, :2]
        frame = tile.repeat(1, 1, 8)
        memory = ObservationMemory()
        memory.step(frame)
        before = memory.state_dict()
        result = memory.step(frame.clone())
        self.assertTrue(result.hold)
        self.assertEqual(result.reason, "ambiguous_alignment")
        equal_state(self, before, memory.state_dict())

    def test_genuine_late_change_writes_and_survives_absence(self):
        frame = crop(scene(91))
        memory = ObservationMemory()
        memory.step(frame)
        for _ in range(12):
            self.assertTrue(memory.step(frame).accepted)
        changed = frame.clone()
        delta = torch.tensor([0.18, -0.14, 0.12])[:, None, None]
        changed[:, 6:8, 7:9] += delta
        result = memory.step(changed)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(bool(result.accepted_write_mask[:, 6:8, 7:9].all()))
        self.assertEqual(int(result.accepted_write_mask.sum()), 4)
        for _ in range(256):
            self.assertTrue(memory.step(frame).accepted)
        prediction = memory.forecast()["rgb"]
        erased = memory.forecast(use_latch=False)["rgb"]
        self.assertLess(float((prediction[:, 6:8, 7:9]-changed[:, 6:8, 7:9]).abs().max()), 0.015)
        self.assertLess(float((erased-frame).abs().max()), 0.015)
        self.assertGreater(float((prediction-erased).abs().sum()), 1.5)

    def test_partial_occlusion_does_not_latch_a_broad_patch(self):
        frame = crop(scene(82))
        memory = ObservationMemory()
        memory.step(frame)
        self.assertTrue(memory.step(frame).accepted)
        before = memory.state_dict()
        occluded = frame.clone()
        occluded[:, :, :8] = 0.1
        result = memory.step(occluded)
        self.assertTrue(result.hold)
        equal_state(self, before, memory.state_dict())

    def test_subpixel_event_write_recovers_world_contrast_without_double_blur(self):
        world = scene(28)
        memory = ObservationMemory(ObservationConfig(allow_vertical=False))
        memory.step(crop(world))
        shifted = crop(world, 8.5)
        self.assertTrue(memory.step(shifted).accepted)
        changed_world = world.clone()
        delta = torch.tensor([0.18, -0.14, 0.12])[:, None, None]
        changed_world[:, 14:16, 15:17] += delta
        changed = crop(changed_world, 8.5)
        result = memory.step(changed)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(bool(result.accepted_write_mask.any()))
        self.assertLess(float((result.canonical_innovation[:, 6:8, 7:9]-delta).abs().max()), 0.015)
        self.assertTrue(memory.step(shifted).accepted)
        recalled = memory.read((0.5, 0.0))["rgb"]
        self.assertLess(float((recalled-changed).abs().max()), 0.015)

    def test_rejected_result_uses_reported_retained_transform(self):
        world = scene(37)
        memory = ObservationMemory(ObservationConfig(allow_vertical=False,
                                                      min_inlier_fraction=0.65,
                                                      max_change_fraction=0.08))
        first = crop(world)
        memory.step(first)
        frame = 1.2*crop(world, 8.5)+0.02
        frame[:, 4:9, 5:10] += torch.tensor([.12, -.12, .12])[:, None, None]
        result = memory.step(frame)
        self.assertTrue(result.hold)
        self.assertEqual(result.reason, "broad_change")
        self.assertTrue(torch.allclose(result.normalized, (frame-result.offset)/result.gain))
        self.assertTrue(torch.equal(result.reference, memory.read(result.position, use_latch=False,
                                                                  observed_units=False)["rgb"]))

    def test_causal_bootstrap_unknown_extension_and_finite_world(self):
        world = scene(70)
        memory = ObservationMemory(ObservationConfig(allow_vertical=False))
        first = memory.step(crop(world))
        self.assertEqual(first.reason, "bootstrap")
        self.assertEqual(memory.read()["coverage"], 1)
        ahead = memory.read((3.0, 0.0))
        self.assertGreater(int(ahead["unknown"].sum()), 0)
        for x in (9.25, 10.25, 11.25):
            result = memory.step(crop(world, x))
            self.assertTrue(result.accepted, (x, result.reason))
        reconstructed = memory.read((3.25, 0), use_latch=False, observed_units=False)
        self.assertGreater(reconstructed["coverage"], 0.8)
        error = (reconstructed["rgb"]-crop(world, 11.25)).abs()*reconstructed["known"]
        self.assertLess(float(error.sum()/(3*reconstructed["known"].sum())), 0.025)

    def test_four_pixel_window_recovers_quarter_shift_and_brightness(self):
        world = scene(11, h=4, w=16)
        memory = ObservationMemory(ObservationConfig(allow_vertical=False))
        memory.step(world[:, :, :4])
        for x in (1.25, 2.25, 3.25):
            ix, fraction = int(x), x-int(x)
            frame = (1-fraction)*world[:, :, ix:ix+4] + fraction*world[:, :, ix+1:ix+5]
            result = memory.step(1.2*frame)
            self.assertTrue(result.accepted, result.reason)
            self.assertAlmostEqual(result.position[0], x, delta=0.125)
            self.assertAlmostEqual(result.gain, 1.2, delta=0.025)

    def test_suffix_poisoning_cannot_change_any_prefix_output_or_state(self):
        world = scene(31)
        frame = crop(world)
        changed = frame.clone()
        changed[:, 2:4, 5:7] += 0.12
        prefix = torch.stack((frame, changed, crop(world, 9.25), crop(world, 10.25)))
        suffix_a = crop(world, 10.25).expand(2, -1, -1, -1).clone()
        suffix_b = torch.full_like(suffix_a, 0.1)
        a = ObservationMemory()
        b = ObservationMemory()
        outputs_a, outputs_b = [], []
        saved_a, saved_b = [], []
        for current in torch.cat((prefix, suffix_a)):
            outputs_a.append(a.step(current))
            saved_a.append(a.state_dict())
        for current in torch.cat((prefix, suffix_b)):
            outputs_b.append(b.step(current))
            saved_b.append(b.state_dict())
        for i in range(len(prefix)):
            equal_state(self, saved_a[i], saved_b[i])
            for name in ("normalized", "reference", "known", "innovation", "accepted_write_mask", "canonical_innovation"):
                self.assertTrue(torch.equal(getattr(outputs_a[i], name), getattr(outputs_b[i], name)), name)
            self.assertEqual(outputs_a[i].position, outputs_b[i].position)
            self.assertEqual(outputs_a[i].accepted, outputs_b[i].accepted)
        batched_a = observe(torch.cat((prefix, suffix_a))[None])
        batched_b = observe(torch.cat((prefix, suffix_b))[None])
        self.assertTrue(torch.equal(batched_a["innovations"][:, :len(prefix)],
                                    batched_b["innovations"][:, :len(prefix)]))

    def test_inputs_results_and_saved_states_do_not_alias(self):
        frame = crop(scene(9))
        original = frame.clone()
        memory = ObservationMemory()
        result = memory.step(frame)
        self.assertTrue(torch.equal(frame, original))
        result.reference.zero_()
        result.normalized.zero_()
        saved = memory.state_dict()
        saved["atlas"].zero_()
        self.assertLess(float((memory.read()["rgb"]-original).abs().max()), 1e-6)
        frame.zero_()
        self.assertLess(float((memory.read()["rgb"]-original).abs().max()), 1e-6)

    def test_memory_interventions_do_not_replace_reference_or_pose(self):
        base = crop(scene(3))
        donor = ObservationMemory()
        receiver = ObservationMemory()
        for memory, signed in ((donor, 1), (receiver, -1)):
            memory.step(base)
            changed = base.clone()
            changed[:, 7:9, 8:10] += signed*torch.tensor([.15, -.12, .1])[:, None, None]
            self.assertTrue(memory.step(changed).accepted)
            memory.step(base)
        old = receiver.state_dict()
        receiver.replace_memory_from(donor)
        new = receiver.state_dict()
        self.assertTrue(torch.equal(old["atlas"], new["atlas"]))
        self.assertEqual(old["position"], new["position"])
        self.assertTrue(torch.equal(receiver.event_atlas, donor.event_atlas))
        receiver.clear_memory()
        self.assertFalse(bool(receiver.event_seen.any()))
        self.assertTrue(torch.equal(receiver.atlas, old["atlas"]))
        restored = ObservationMemory()
        restored.load_state_dict(donor.state_dict())
        equal_state(self, donor.state_dict(), restored.state_dict())
        self.assertTrue(torch.equal(restored.forecast()["rgb"], donor.forecast()["rgb"]))

    def test_explicit_errors_before_state_mutation(self):
        memory = ObservationMemory()
        with self.assertRaises(RuntimeError):
            memory.read()
        for frame in (torch.zeros(4, 8, 8), torch.zeros(3, 8, 8, dtype=torch.long),
                      torch.full((3, 8, 8), float("nan"))):
            with self.assertRaises(ValueError):
                memory.step(frame)
        memory.step(crop(scene()))
        before = memory.state_dict()
        with self.assertRaises(ValueError):
            memory.step(torch.zeros(3, 10, 10))
        equal_state(self, before, memory.state_dict())
        invalid = memory.state_dict()
        invalid["event_seen"] = torch.zeros(1, 2, 2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            memory.load_state_dict(invalid)
        equal_state(self, before, memory.state_dict())
        invalid = memory.state_dict()
        invalid["gain"] = float("nan")
        with self.assertRaises(ValueError):
            memory.load_state_dict(invalid)
        equal_state(self, before, memory.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
