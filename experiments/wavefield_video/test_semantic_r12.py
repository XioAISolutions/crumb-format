"""Synthetic semantic-metric contracts; run with python test_semantic_r12.py."""

import json
import math
import unittest

import torch

from data import PALETTE_ANCHORS
from semantic_metrics import (aggregate_semantic_metrics, detect_blobs,
                              semantic_frame_metrics, semantic_summary, _matched_errors)


def frame(grid, positions, color_ids, radius=1.0, brightness=0.95):
    """Independent synthetic Gaussian renderer with known (x,y) centers."""
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    image = torch.zeros(3, grid, grid)
    for (x, y), color_id in zip(positions, color_ids):
        blob = torch.exp(-((xs - x).square() + (ys - y).square()) / (2 * radius ** 2))
        image += brightness * PALETTE_ANCHORS[color_id, :, None, None] * blob
    return image.clamp(0, 1).unsqueeze(0)


def score(image, positions, color_ids, radius=1.0):
    return semantic_frame_metrics(image, torch.tensor([positions], dtype=torch.float32),
                                  PALETTE_ANCHORS[color_ids].unsqueeze(0), radius=radius)[0]


class SemanticMetricTests(unittest.TestCase):
    def test_native_grid_and_known_position_error(self):
        for grid in (16, 32, 64):
            positions = [(grid * .25, grid * .25), (grid * .75, grid * .25),
                         (grid * .5, grid * .75)]
            colors = [0, 1, 2]
            with self.subTest(grid=grid, mode="perfect"):
                result = score(frame(grid, positions, colors), positions, colors)
                self.assertEqual((result["expected_n"], result["detected_n"], result["matched_count"]),
                                 (3, 3, 3))
                self.assertLess(result["mean_matched_position_error"], .05)
            with self.subTest(grid=grid, mode="shifted"):
                shifted = [(x + 1, y + 1) for x, y in positions]
                result = score(frame(grid, shifted, colors), positions, colors)
                self.assertEqual(result["matched_count"], 3)
                self.assertAlmostEqual(result["mean_matched_position_error"], math.sqrt(2), delta=.05)

    def test_missing_extra_and_displaced_blobs(self):
        positions, colors = [(6, 6), (24, 6), (15, 24)], [0, 1, 2]
        missing = score(frame(32, positions[:2], colors[:2]), positions, colors)
        self.assertEqual((missing["expected_n"], missing["detected_n"], missing["matched_count"]), (3, 2, 2))
        extra = score(frame(32, positions + [(25, 25)], colors + [0]), positions, colors)
        self.assertEqual((extra["expected_n"], extra["detected_n"], extra["matched_count"]), (3, 4, 3))
        displaced = score(frame(32, positions[:2] + [(25, 25)], colors), positions, colors)
        self.assertEqual((displaced["detected_n"], displaced["matched_count"]), (3, 2))
        self.assertLess(displaced["mean_matched_position_error"], .05)

    def test_darkness_and_flat_brightness_do_not_create_blobs(self):
        positions, colors = [(6, 6), (24, 6), (15, 24)], [0, 1, 2]
        truth = frame(32, positions, colors)
        for image in (torch.zeros_like(truth), truth * .1, torch.full_like(truth, .5)):
            result = score(image, positions, colors)
            self.assertEqual((result["expected_n"], result["detected_n"], result["matched_count"]), (3, 0, 0))
            self.assertIsNone(result["mean_matched_position_error"])
            json.dumps(result, allow_nan=False)

    def test_repeated_palette_large_counts_and_small_radius(self):
        for grid, radius in ((16, .8), (32, 1.6), (64, 1.6)):
            positions = [(grid * (col + .5) / 4, grid * (row + .5) / 3)
                         for row in range(3) for col in range(4)]
            for count in (8, 12):
                with self.subTest(grid=grid, radius=radius, count=count):
                    centers = positions[:count]
                    colors = [index % 3 for index in range(count)]
                    result = score(frame(grid, centers, colors, radius), centers, colors, radius)
                    self.assertEqual((result["expected_n"], result["detected_n"], result["matched_count"]),
                                     (count, count, count))
                    self.assertLess(result["mean_matched_position_error"], .25)

    def test_color_identity_is_required(self):
        positions, colors = [(6, 6), (24, 6)], [0, 1]
        result = score(frame(32, positions, colors[::-1]), positions, colors)
        self.assertEqual(result["detected_n"], 2)
        self.assertEqual(result["matched_count"], 0)
        self.assertIsNone(result["mean_matched_position_error"])

    def test_flat_blob_has_one_detection_and_unmixed_overlap_keeps_two(self):
        image = torch.zeros(1, 3, 32, 32)
        image[:, :, 10:21, 10:21] = PALETTE_ANCHORS[0][None, :, None, None] * .9
        self.assertEqual(len(detect_blobs(image)[0]), 1)
        positions, colors = [(15, 15), (15, 15)], [0, 1]
        result = score(frame(32, positions, colors, brightness=.5), positions, colors)
        self.assertEqual((result["detected_n"], result["matched_count"]), (2, 2))
        self.assertLess(result["mean_matched_position_error"], .05)

    def test_assignment_uses_maximum_cardinality_before_distance(self):
        # Greedy nearest-first can consume the only feasible partner of GT #1.
        detections = [{"position": [1.1, 0], "color_index": 0},
                      {"position": [-1.5, 0], "color_index": 0}]
        errors = _matched_errors([[0, 0], [2.5, 0]], [0, 0], detections, 2.0)
        self.assertEqual(len(errors), 2)
        self.assertAlmostEqual(sum(errors), 2.9)

    def test_aggregation_retains_samples_and_weights_only_real_matches(self):
        a = {"expected_n": 3, "detected_n": 3, "matched_count": 3, "mean_matched_position_error": 1.0}
        b = {"expected_n": 3, "detected_n": 1, "matched_count": 1, "mean_matched_position_error": 2.0}
        black = {"expected_n": 3, "detected_n": 0, "matched_count": 0, "mean_matched_position_error": None}
        result = aggregate_semantic_metrics([[a, b], [black, black]], radius=.8)
        self.assertEqual(result["expected_n"], 3)
        self.assertEqual(result["detected_n"], 1)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["mean_matched_position_error"], 1.25)
        self.assertEqual(result["match_tolerance_px"], 1.6)
        self.assertEqual(result["frames"][0]["samples"], [a, b])
        self.assertIsNone(result["frames"][1]["mean_matched_position_error"])
        self.assertIn("pos_err=1.250px", semantic_summary(result))
        json.dumps(result, allow_nan=False)

    def test_detector_does_not_consult_expected_count(self):
        image = frame(32, [(6, 6), (24, 6)], [0, 0])
        self.assertEqual(len(detect_blobs(image)[0]), 2)
        # Identical pixels with one vs two GT objects must report the same count.
        one = score(image, [(6, 6)], [0])
        two = score(image, [(6, 6), (24, 6)], [0, 0])
        self.assertEqual(one["detected_n"], two["detected_n"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
