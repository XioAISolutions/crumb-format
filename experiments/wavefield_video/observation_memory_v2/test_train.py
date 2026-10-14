"""Retrieval campaign contracts: balance, actual state interventions and artifact replay."""
from collections import Counter
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from adapters import HeadState, MemoryHead
from train import (BOUNDARY, CONDITIONS, FAMILIES, dataset, dataset_fingerprint,
                   evaluate, intervention_predictions, run_arm, hash_file)


class IdealObservedLatch:
    """A privileged-capacity positive control, receiving observed features only.

    The nonzero readout bias proves erased state is not zero-output reference.
    This is a test instrument, never a learned architecture contender.
    """
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def predict(self, features, state=None):
        self.calls.append((features.clone(), state))
        memory = torch.zeros_like(features[:, 0]) if state is None else state.recurrent[0].clone()
        for frame in features.unbind(1):
            changed = frame.abs().mean(1, keepdim=True) > .055
            memory = torch.where(changed, frame, memory)
        processed = features.shape[1] + (0 if state is None else state.processed_frames)
        return memory + .0001, HeadState("wave", processed, recurrent=[memory])


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.examples = dataset(701000, 8, 8)
        cls.static = dataset(702000, 2, 8, ("static_noevent",))

    def test_complete_factorial_and_disjoint_heldout_worlds(self):
        cells = Counter((e.case["metadata"]["condition"], e.case["metadata"]["family"])
                        for e in self.examples)
        self.assertEqual(set(cells), {(c, f) for c in CONDITIONS for f in FAMILIES})
        self.assertTrue(all(value == 1 for value in cells.values()))
        train_seeds = {e.case["metadata"]["seed"] for e in self.examples}
        self.assertEqual(len(train_seeds), 8)
        self.assertFalse(train_seeds & {e.case["metadata"]["seed"] for e in self.static})
        with self.assertRaisesRegex(ValueError, "complete condition"):
            dataset(0, 2, 8)

    def test_opposite_donors_change_only_observed_cue_and_truth(self):
        for example in self.examples:
            self.assertTrue(torch.equal(example.case["frames"][:28], example.donor_case["frames"][:28]))
            self.assertTrue(torch.equal(example.case["frames"][32:], example.donor_case["frames"][32:]))
            self.assertTrue(torch.equal(example.case["background"], example.donor_case["background"]))
            self.assertTrue(torch.equal(example.case["event"], -example.donor_case["event"]))
            self.assertTrue(torch.equal(example.record.reference_canonical, example.donor_record.reference_canonical))
            self.assertEqual(example.case["metadata"]["camera_path"], example.donor_case["metadata"]["camera_path"])

    def test_actual_256_frame_erasure_and_donor_keep_recipient_tail_and_decoder(self):
        example = dataset(703000, 2, 256, ("clean",))[0]
        self.assertEqual(example.record.features.shape[1], 288)
        head = IdealObservedLatch()
        before = example.record.reference_canonical.clone()
        predictions, accounting = intervention_predictions(head, example, torch.device("cpu"))
        reference = predictions["reference_only"]
        self.assertTrue(torch.allclose(predictions["intact"]-reference, example.case["event"], atol=.002))
        self.assertTrue(torch.allclose(predictions["swapped_state"]-reference, -example.case["event"], atol=.002))
        self.assertFalse(torch.equal(predictions["erased_state"], reference))
        self.assertLess(float((predictions["erased_state"]-reference).abs().max()), .0002)
        self.assertEqual(accounting["intervention_boundary"], BOUNDARY)
        self.assertEqual(accounting["recipient_tail_frames"], 256)
        self.assertEqual(accounting["processed_frames"], 288)
        self.assertTrue(torch.equal(before, example.record.reference_canonical))
        for index in (1, 2, 4):
            self.assertTrue(torch.equal(head.calls[index][0], example.record.features[:, 32:]))
        self.assertFalse(bool(head.calls[2][1].recurrent[0].any()))
        self.assertGreater(float(head.calls[1][1].recurrent[0].abs().sum()), 0)
        # Opposite observed RGB cues differ by subtraction/inverse float rounding.
        opposite_error = float((head.calls[1][1].recurrent[0] +
                                head.calls[4][1].recurrent[0]).abs().max())
        self.assertLess(opposite_error, 2e-6)

    def test_inference_interventions_ignore_poisoned_targets_masks_and_metadata(self):
        original = self.examples[0]
        case = copy.deepcopy(original.case)
        case["target"].fill_(float("nan"))
        case["masks"]["known"].zero_()
        case["metadata"]["camera_path"] = [999.]
        poisoned = dataclasses.replace(original, target=torch.full_like(original.target, float("nan")), case=case)
        a, _ = intervention_predictions(IdealObservedLatch(), original, torch.device("cpu"))
        b, _ = intervention_predictions(IdealObservedLatch(), poisoned, torch.device("cpu"))
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_positive_control_validates_gates_and_expired_evaluation_is_incomplete(self):
        evaluation, artifacts = evaluate(IdealObservedLatch(), self.examples+self.static, torch.device("cpu"))
        self.assertTrue(evaluation["complete"])
        self.assertTrue(evaluation["descriptive_gate_pass"], evaluation["checks"])
        self.assertEqual(evaluation["positive_examples"], 8)
        self.assertEqual(evaluation["static_examples"], 2)
        self.assertEqual(len(artifacts), 10)
        self.assertIn("privileged_oracle", evaluation["examples"][0]["controls"])
        self.assertIn("frame_ema", evaluation["examples"][0]["controls"])
        self.assertIn("frozen_last_context", evaluation["examples"][0]["controls"])
        partial, _ = evaluate(IdealObservedLatch(), self.examples, torch.device("cpu"), time.monotonic()-1)
        self.assertFalse(partial["complete"])
        self.assertFalse(partial["descriptive_gate_pass"])

    def test_fingerprint_binds_decoder_context_not_only_features_target(self):
        example = self.examples[0]
        modified = dataclasses.replace(example.record, reference_canonical=example.record.reference_canonical+.1)
        other = dataclasses.replace(example, record=modified)
        self.assertNotEqual(dataset_fingerprint([example]), dataset_fingerprint([other]))

    def test_actual_native_update_saved_predictions_replay_and_artifact_hashes(self):
        heldout = [self.examples[0], self.static[0]]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run_arm("wave", 3, self.examples, heldout, 1, torch.device("cpu"), output, time.monotonic()+60)
            self.assertTrue(result["completed"])
            self.assertEqual(result["completed_updates"], 1)
            self.assertTrue(any(".mix." in key for key in result["changed_state_tensors"]))
            self.assertEqual(result["checkpoint_sha256"], hash_file(output/result["checkpoint_file"]))
            self.assertEqual(result["predictions_sha256"], hash_file(output/result["predictions_file"]))
            checkpoint = torch.load(output/result["checkpoint_file"], map_location="cpu", weights_only=True)
            model = MemoryHead("wave", 12, 12)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            artifacts = torch.load(output/result["predictions_file"], map_location="cpu", weights_only=True)
            for example, saved in zip(heldout, artifacts):
                replay, _ = intervention_predictions(model, example, torch.device("cpu"))
                for key, value in replay.items():
                    self.assertTrue(torch.equal(value, saved[key]), key)
            saved_json = json.loads((output/"wave-s3.json").read_text())
            self.assertEqual(len(saved_json["evaluation"]["examples"]), 2)
            self.assertGreater(result["evaluation"]["examples"][0]["metrics"]["boundary_state_tensor_bytes"], 0)

    def test_expired_training_never_reports_comparable_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_arm("wave", 3, self.examples, [self.examples[0]], 1,
                             torch.device("cpu"), Path(folder), time.monotonic()-1)
            self.assertFalse(result["completed"])
            self.assertEqual(result["completed_updates"], 0)
            self.assertIsNone(result["evaluation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
