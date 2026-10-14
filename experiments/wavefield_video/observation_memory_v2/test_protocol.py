"""CPU scorer/fixture contract checks, including explicit semantic mutants."""

import copy
import unittest

import torch

try:
    from .protocol import CONDITIONS, FAMILIES, OPTIONAL_CONDITIONS, controls, make_case, score, score_memory
except ImportError:
    from protocol import CONDITIONS, FAMILIES, OPTIONAL_CONDITIONS, controls, make_case, score, score_memory


class ProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_raw_timeline_and_frame_only_contract(self):
        case = make_case(0, "clean", "chromatic")
        self.assertEqual(tuple(case["frames"].shape), (288, 3, 12, 12))
        self.assertEqual(case["metadata"]["hidden_interval"], [32, 288])
        self.assertEqual(case["metadata"]["last_cue_index"], 31)
        self.assertEqual(case["metadata"]["target_index"], 288)
        self.assertEqual(case["metadata"]["seed_role"], "data_only")
        self.assertTrue(torch.equal(case["frames"][31], case["target"]))
        self.assertTrue(torch.equal(case["frames"][32:], case["background"][None].expand(256, -1, -1, -1)))
        poisoned = copy.deepcopy(case)
        poisoned["target"].fill_(float("nan"))
        poisoned["metadata"]["camera_path"] = [999.] * 289
        poisoned["metadata"]["cue_xy"] = [-999, -999]
        poisoned["masks"]["known"].zero_()
        self.assertTrue(torch.equal(case["frames"], poisoned["frames"]))

    def test_reproducible_seeds_and_arbitrary_interior_locations(self):
        coordinates = set()
        for seed in (0, 1, 2):
            for family in FAMILIES:
                a = make_case(seed, "clean", family)
                b = make_case(seed, "clean", family)
                self.assertTrue(torch.equal(a["frames"], b["frames"]))
                x, y = a["metadata"]["cue_xy"]
                self.assertTrue(2 <= x <= 8 and 2 <= y <= 8)
                coordinates.add((x, y))
        self.assertGreater(len(coordinates), 1)

    def test_oracle_every_condition_family_and_subpixel_ties(self):
        for condition in CONDITIONS + OPTIONAL_CONDITIONS:
            for family in FAMILIES:
                for seed in (0, 1, 2):
                    with self.subTest(condition=condition, family=family, seed=seed):
                        case = make_case(seed, condition, family)
                        s = score(controls(case)["privileged_oracle"], case)
                        self.assertTrue(s["finite"])
                        self.assertEqual(s["full_mse"], 0.)
                        if not s["event_present"]:
                            self.assertIsNone(s["reconstruction_skill_unconditional"])
                            self.assertFalse(s["false_presence"])
                        else:
                            self.assertEqual(s["reconstruction_skill_unconditional"], 1.)
                            self.assertEqual(s["joint_hit"], 1.)

    def test_wrong_identity_location_and_omission_fail(self):
        for condition in ("clean", "subpixel_quarter", "subpixel_half"):
            for family in FAMILIES:
                case = make_case(0, condition, family)
                variants = controls(case)
                for name in ("wrong_location", "wrong_identity", "privileged_background_only", "frozen_last_context"):
                    with self.subTest(condition=condition, family=family, control=name):
                        s = score(variants[name], case)
                        self.assertEqual(s["joint_hit"], 0.)
                        self.assertLessEqual(s["reconstruction_skill_unconditional"], 1e-6)
                self.assertEqual(score(variants["wrong_location"], case)["identity_hit"], 1.)
                self.assertEqual(score(variants["wrong_identity"], case)["location_hit"], 1.)

    def test_freeze_is_actual_last_context_constant(self):
        case = make_case(0, "brightness_step", "achromatic")
        frozen = controls(case)["frozen_last_context"]
        self.assertTrue(torch.equal(frozen, case["frames"][-1]))
        curve = frozen[None].expand(5, -1, -1, -1)
        self.assertTrue(torch.equal(curve[0], curve[-1]))
        self.assertFalse(torch.equal(frozen, case["target"]))

    def test_observability_and_occlusions_end_before_last_input(self):
        case = make_case(0, "write_occlusion", "achromatic")
        self.assertFalse(case["metadata"]["cue_observable"])
        self.assertFalse(bool(case["masks"]["event_observable"].any()))
        self.assertFalse(score(case["target"], case)["event_scored"])
        self.assertTrue(torch.equal(case["frames"][28:32], torch.full((4, 3, 12, 12), .1)))
        for condition in ("late_half_occlusion", "full_occlusion"):
            a = make_case(1, condition, "chromatic")
            begin, end = a["metadata"]["occlusion_interval"]
            self.assertGreaterEqual(begin, 32)
            self.assertLess(end, len(a["frames"]))
            self.assertTrue(torch.equal(a["frames"][-1], a["background"]))
        ambiguous = make_case(2, "ambiguous_texture", "chromatic")
        self.assertTrue(ambiguous["metadata"]["cue_observable"])
        self.assertFalse(ambiguous["metadata"]["alignment_identifiable"])

    def test_known_unknown_errors_separate_and_no_wrap(self):
        case = make_case(0, "partial_disocclusion", "chromatic")
        self.assertAlmostEqual(float(case["masks"]["unknown"].float().mean()), .5)
        pred = case["target"].clone()
        pred += case["masks"]["unknown"].float() * .2
        s = score(pred, case)
        self.assertEqual(s["known_mse"], 0.)
        self.assertAlmostEqual(s["unknown_mse"], .04, places=6)
        self.assertTrue(all(0 <= x <= 10 for x in case["metadata"]["camera_path"]))
        ordinary = score(make_case(0, "clean", "achromatic")["target"], make_case(0, "clean", "achromatic"))
        self.assertIsNone(ordinary["unknown_mse"])

    def test_no_event_false_presence_and_invalid_prediction(self):
        case = make_case(0, "static_noevent", "chromatic")
        self.assertFalse(score(case["frames"][-1], case)["false_presence"])
        pred = case["target"].clone()
        pred[:, 3:5, 3:5] += .18
        self.assertTrue(score(pred, case)["false_presence"])
        bad = score(torch.full_like(case["target"], float("nan")), case)
        self.assertFalse(bad["finite"])
        self.assertIsNone(bad["full_mse"])
        with self.assertRaises(ValueError):
            score(case["target"][None], case)

    def test_fractional_write_pose_and_small_occluder_boundary(self):
        case = make_case(0, "subpixel_write", "chromatic")
        self.assertEqual(case["metadata"]["camera_path"][15], 4.)
        self.assertEqual(case["metadata"]["camera_path"][16], 4.5)
        self.assertEqual(case["metadata"]["camera_path"][31], 4.5)
        self.assertTrue(torch.equal(case["frames"][31], case["target"]))
        self.assertEqual(score(case["target"], case)["joint_hit"], 1.)
        negative = make_case(0, "small_occluder_static", "achromatic")
        self.assertTrue(negative["metadata"]["object_occluder_ambiguous"])
        self.assertFalse(negative["metadata"]["event_present"])
        self.assertFalse(score(negative["target"], negative)["false_presence"])
        self.assertFalse(torch.equal(negative["frames"][31], negative["target"]))

    def test_background_error_does_not_change_event_roi_skill(self):
        case = make_case(0, "clean", "achromatic")
        intact = score(case["target"], case)
        bad_background = case["target"] + (~case["masks"]["event_roi"]).float() * .2
        changed = score(bad_background, case)
        self.assertEqual(intact["event_roi_skill"], changed["event_roi_skill"])
        self.assertLess(changed["reconstruction_skill_unconditional"], intact["reconstruction_skill_unconditional"])
        self.assertGreater(changed["background_mse"], intact["background_mse"])

    def test_memory_delta_cancels_shared_background_error(self):
        for family in FAMILIES:
            case = make_case(1, "brightness_step", family)
            reference, intact = case["background"], case["target"]
            base = score_memory(intact, reference, case)
            added = torch.full_like(reference, .25)
            changed = score_memory(intact + added, reference + added, case)
            self.assertAlmostEqual(base["memory_delta_skill"], changed["memory_delta_skill"], places=6)
            self.assertEqual(base["memory_delta_joint_hit"], 1.)
            self.assertEqual(changed["memory_delta_joint_hit"], 1.)
            for name in ("wrong_location", "wrong_identity", "privileged_background_only"):
                self.assertEqual(score_memory(controls(case)[name], reference, case)["memory_delta_joint_hit"], 0.)
        static = make_case(1, "static_noevent", "achromatic")
        s = score_memory(static["target"], static["background"], static)
        self.assertFalse(s["memory_delta_false_presence"])
        self.assertIsNone(s["memory_delta_skill"])

    def test_counterfactual_donor_changes_only_cue_and_event_truth(self):
        for condition in ("clean", "brightness_step", "subpixel_write"):
            for family in FAMILIES:
                original = make_case(2, condition, family, event_sign=1.)
                donor = make_case(2, condition, family, event_sign=-1.)
                self.assertTrue(torch.equal(original["background"], donor["background"]))
                self.assertEqual(original["metadata"]["camera_path"], donor["metadata"]["camera_path"])
                self.assertEqual(original["metadata"]["brightness"], donor["metadata"]["brightness"])
                self.assertEqual(original["metadata"]["cue_xy"], donor["metadata"]["cue_xy"])
                self.assertTrue(torch.equal(original["frames"][:28], donor["frames"][:28]))
                self.assertTrue(torch.equal(original["frames"][32:], donor["frames"][32:]))
                self.assertFalse(torch.equal(original["frames"][28:32], donor["frames"][28:32]))
                self.assertTrue(torch.equal(original["event"], -donor["event"]))
                self.assertTrue(torch.equal(donor["target"], donor["background"] + donor["event"]))
                self.assertEqual(donor["metadata"]["event_sign"], -1.)
                self.assertEqual(score(donor["target"], donor)["joint_hit"], 1.)
                self.assertEqual(score(donor["target"], original)["identity_hit"], 0.)
        for invalid in (0., .5, float("nan"), float("inf"), True, "-1"):
            with self.assertRaises(ValueError):
                make_case(0, "clean", "achromatic", event_sign=invalid)


if __name__ == "__main__":
    unittest.main()
