"""Staged final-only RGB retrieval campaign, or an explicit tiny CPU smoke.

All heads receive identical observed-only features. Actual state interventions
occur after frame31, then consume the identical recipient hidden tail. This is
not a generator, pretrained-VAE test or matched-resource architecture comparison.
CUDA work belongs only on the existing owned/preflighted GPU scheduler.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import torch

from adapters import HeadState, MemoryHead, decode_prediction, extract_features
from observation import ObservationConfig, ObservationMemory
from protocol import controls, make_case, score, score_memory
from run import raw_controls

ROOT = Path(__file__).resolve().parent
KINDS = ("wave", "ssm", "attn")
CONDITIONS = ("clean", "brightness_step", "subpixel_half", "subpixel_write")
FAMILIES = ("chromatic", "achromatic")
CONFIG = ObservationConfig(allow_vertical=False)
BOUNDARY = 32
GATES = {"joint": .8, "memory_delta_skill": .5, "erase_drop": .5,
         "donor_joint": .75, "static_false_presence": .05}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_bytes(value):
    """Actual tensor payload; excludes optimizer, activations and object overhead."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if dataclasses.is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


@dataclasses.dataclass
class Example:
    record: object
    target: torch.Tensor
    case: dict
    donor_record: object
    donor_case: dict
    baseline_predictions: dict
    frontend_state_bytes: int


def dataset(seed, count, gap, conditions=CONDITIONS):
    """Complete condition x family cells; unique world seed per example."""
    cells = len(conditions) * len(FAMILIES)
    if count < cells or count % cells:
        raise ValueError("count must contain complete condition x family cells")
    records = []
    with torch.no_grad():
        for i in range(count):
            condition = conditions[i % len(conditions)]
            family = FAMILIES[(i // len(conditions)) % len(FAMILIES)]
            case = make_case(seed+i, condition, family, gap=gap)
            donor = make_case(seed+i, condition, family, gap=gap, event_sign=-1.)
            record = extract_features(case["frames"][None], CONFIG, chunk_size=32)
            donor_record = extract_features(donor["frames"][None], CONFIG, chunk_size=32)
            memory = ObservationMemory(CONFIG)
            for frame in case["frames"]:
                memory.step(frame)
            predictions = raw_controls(case["frames"]) | controls(case)
            predictions |= {"aligned_reference_only": memory.forecast(use_latch=False)["rgb"],
                            "aligned_reference_latch": memory.forecast()["rgb"]}
            records.append(Example(record, case["target"][None], case, donor_record, donor,
                                   predictions, tensor_bytes(memory.state_dict())))
    return records


def erase_head_state(state):
    """Erase actual learned state, preserving time and recipient decoder context."""
    return HeadState(state.kind, state.processed_frames,
                     None if state.recurrent is None else [torch.zeros_like(v) for v in state.recurrent],
                     None if state.window is None else torch.zeros_like(state.window))


def copy_head_state(state):
    return HeadState(state.kind, state.processed_frames,
                     None if state.recurrent is None else [v.clone() for v in state.recurrent],
                     None if state.window is None else state.window.clone())


@torch.no_grad()
def intervention_predictions(head, example, device):
    record, donor = example.record.to(device), example.donor_record.to(device)
    _, boundary = head.predict(record.features[:, :BOUNDARY])
    tail = record.features[:, BOUNDARY:]
    if tail.shape[1] < 1:
        raise ValueError("evaluation requires a hidden post-cue tail")
    intact_event, final_state = head.predict(tail, copy_head_state(boundary))
    erased_event, _ = head.predict(tail, erase_head_state(boundary))
    _, donor_boundary = head.predict(donor.features[:, :BOUNDARY])
    if donor_boundary.processed_frames != boundary.processed_frames:
        raise ValueError("donor and recipient boundary times must agree")
    swapped_event, _ = head.predict(tail, copy_head_state(donor_boundary))
    # Every branch uses the recipient's appearance/pose/photometry packet.
    predictions = {
        "intact": decode_prediction(intact_event, record).rgb[0].cpu(),
        "erased_state": decode_prediction(erased_event, record).rgb[0].cpu(),
        "swapped_state": decode_prediction(swapped_event, record).rgb[0].cpu(),
        "reference_only": decode_prediction(torch.zeros_like(intact_event), record).rgb[0].cpu(),
    }
    accounting = {"boundary_state_tensor_bytes": tensor_bytes(boundary),
                  "final_state_tensor_bytes": tensor_bytes(final_state),
                  "processed_frames": final_state.processed_frames,
                  "intervention_boundary": BOUNDARY, "recipient_tail_frames": tail.shape[1]}
    return predictions, accounting


def evaluate(head, records, device, deadline=None):
    rows, artifacts = [], []
    head.eval()
    for index, example in enumerate(records):
        if deadline is not None and time.monotonic() >= deadline:
            break
        predictions, accounting = intervention_predictions(head, example, device)
        reference = predictions["reference_only"]
        intact = score(predictions["intact"], example.case) | score_memory(predictions["intact"], reference, example.case)
        erased = score_memory(predictions["erased_state"], reference, example.case)
        swapped = score_memory(predictions["swapped_state"], reference, example.donor_case)
        intact |= {"erase_drop": intact["memory_delta_joint_hit"] - erased["memory_delta_joint_hit"],
                   "erased_state_joint_hit": erased["memory_delta_joint_hit"],
                   "erased_state_skill": erased["memory_delta_skill"],
                   "donor_joint_hit": swapped["memory_delta_joint_hit"],
                   "donor_skill": swapped["memory_delta_skill"], **accounting}
        meta = example.case["metadata"]
        rows.append({"example_index": index, "data_seed": meta["seed"], "condition": meta["condition"],
                     "family": meta["family"], "raw_frame_gap": meta["gap"], "metrics": intact,
                     "controls": {name: score(pred, example.case)
                                  for name, pred in example.baseline_predictions.items()}})
        artifacts.append({"example_index": index, "data_seed": meta["seed"], **predictions,
                          "target": example.case["target"], "event": example.case["event"],
                          "donor_event": example.donor_case["event"]})
    positive = [r["metrics"] for r in rows if r["condition"] != "static_noevent"]
    static = [r["metrics"] for r in rows if r["condition"] == "static_noevent"]
    keys = ("full_mse", "joint_hit", "memory_delta_skill", "memory_delta_joint_hit", "erase_drop", "donor_joint_hit")
    metrics = {k: sum(float(m[k]) for m in positive)/len(positive) if positive else None for k in keys}
    checks = {
        "joint_recovery": bool(positive) and all(m["memory_delta_joint_hit"] >= GATES["joint"] for m in positive),
        "memory_component_skill": bool(positive) and all(m["memory_delta_skill"] >= GATES["memory_delta_skill"] for m in positive),
        "actual_state_erase_dependence": bool(positive) and all(m["erase_drop"] >= GATES["erase_drop"] for m in positive),
        "opposite_cue_donor": bool(positive) and all(m["donor_joint_hit"] >= GATES["donor_joint"] for m in positive),
        "static_false_presence": bool(static) and sum(bool(m["memory_delta_false_presence"]) for m in static)/len(static) <= GATES["static_false_presence"],
    }
    complete = len(rows) == len(records)
    return {"complete": complete, "metrics": metrics, "checks": checks,
            "descriptive_gate_pass": complete and all(checks.values()), "examples": rows,
            "positive_examples": len(positive), "static_examples": len(static)}, artifacts


def dataset_fingerprint(records):
    identity = hashlib.sha256()
    for example in records:
        for packet in (example.record, example.donor_record):
            for field in dataclasses.fields(packet):
                value = getattr(packet, field.name)
                identity.update(field.name.encode())
                if isinstance(value, torch.Tensor):
                    identity.update(str((tuple(value.shape), value.dtype)).encode())
                    identity.update(value.cpu().contiguous().numpy().tobytes())
                else:
                    identity.update(str(value).encode())
        identity.update(example.target.numpy().tobytes())
        identity.update(json.dumps(example.case["metadata"], sort_keys=True).encode())
    return identity.hexdigest()


def run_arm(kind, init_seed, records, heldout, updates, device, output, deadline):
    torch.manual_seed(init_seed)
    head = MemoryHead(kind, 12, 12, dim=16, layers=1, heads=2, window=32).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, cpu_started = time.monotonic(), time.process_time()
    losses = []
    head.train()
    for step in range(updates):
        if time.monotonic() >= deadline:
            break
        example = records[step % len(records)]
        record = example.record.to(device)
        optimizer.zero_grad(set_to_none=True)
        event = head(record.features)
        loss = torch.nn.functional.mse_loss(decode_prediction(event, record).rgb, example.target.to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite {kind} loss at step{step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"nonfinite {kind} gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step+1) % 100 == 0:
            print(json.dumps({"kind": kind, "initialization_seed": init_seed,
                              "updates": step+1, "loss": losses[-1]}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed, cpu_elapsed = time.monotonic()-started, time.process_time()-cpu_started
    changed = [k for k, v in head.state_dict().items()
               if k in before and not torch.equal(v.detach().cpu(), before[k])]
    training_complete = len(losses) == updates
    evaluation, predictions = evaluate(head, heldout, device, deadline) if training_complete else (None, [])
    complete = training_complete and evaluation["complete"]
    checkpoint = output/f"{kind}-s{init_seed}.pt"
    torch.save({"state_dict": head.state_dict(), "kind": kind, "seed": init_seed,
                "window": 32, "dim": 16, "layers": 1, "heads": 2,
                "objective": "observed_only_signed_RGB_retrieval",
                "completed_updates": len(losses)}, checkpoint)
    artifact = output/f"{kind}-s{init_seed}-predictions.pt"
    torch.save(predictions, artifact)
    result = {"kind": kind, "initialization_seed": init_seed,
              "requested_updates": updates, "completed_updates": len(losses),
              "training_completed": training_complete, "completed": complete,
              "training_wall_seconds": elapsed, "training_cpu_seconds": cpu_elapsed,
              "parameter_count": sum(p.numel() for p in head.parameters()),
              "parameter_tensor_bytes": tensor_bytes(list(head.parameters())),
              "registered_buffer_tensor_bytes": tensor_bytes(list(head.buffers())),
              "shared_frontend_state_bytes": sorted({e.frontend_state_bytes for e in heldout}),
              "changed_state_tensors": changed, "last_loss": losses[-1] if losses else None,
              "context_frames_seen": records[0].record.total_observed_frames,
              "bounded_attention_window": 32 if kind == "attn" else None,
              "native_variant": "halflife_clean_write" if kind == "wave" else "native_bias_spatial_embed",
              "metrics": None if evaluation is None else evaluation["metrics"],
              "evaluation": evaluation, "checkpoint_file": checkpoint.name,
              "checkpoint_sha256": hash_file(checkpoint), "predictions_file": artifact.name,
              "predictions_sha256": hash_file(artifact),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    save_json(output/f"{kind}-s{init_seed}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if min(args.updates, args.seconds) < 1:
        raise ValueError("positive update/time budgets required")
    if args.out.exists():
        raise FileExistsError("Use a fresh output identity")
    device = torch.device("cpu" if args.smoke else "cuda")
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: campaign not started; no silent CPU fallback")
    if device.type == "cuda":
        assert torch.ones(1, device=device).item() == 1
    torch.set_num_threads(1)
    seeds = [0] if args.smoke else [0, 1, 2]
    gap, count, updates = (8, 8, 1) if args.smoke else (256, 16, args.updates)
    args.out.mkdir(parents=True)
    sources = {p.name: hash_file(p) for p in ROOT.glob("*.py")}
    native = {p.name: hash_file(p) for p in (ROOT.parent/"wfvideo.py", ROOT.parent/"ssm_lite.py")}
    spec = {"smoke_only": args.smoke, "device": str(device),
            "initialization_seeds": seeds, "raw_frame_gap": gap,
            "context_frames_every_arm": 32+gap, "updates_each_arm": updates,
            "kinds": KINDS, "training_data_seeds": [980000+i for i in range(count)],
            "heldout_data_seeds": [990000+i for i in range(count)],
            "static_heldout_data_seeds": [990100, 990101],
            "conditions": CONDITIONS, "families": FAMILIES, "examples_per_condition_family": count//8,
            "example_order": "same sequential cycling for every arm",
            "shared_features": True, "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": 0.0},
            "input_representation": "signed observed-only canonical RGB innovations",
            "frozen_pretrained_VAE_used": False, "training_target_is_not_inference_input": True,
            "matched_parameter_or_compute_budget": False,
            "model_variant_confound": "Wave clean-write removes biases/spatial table; stock SSM/attention retain them",
            "required_before_superiority_claim": ["matched bias/write variants", "matched resource comparison",
                "factorized attention", "sink-plus-window baseline", "pretrained VAE causality and decoded video"],
            "state_intervention": {"boundary_frame": 32, "erasure": "actual recurrent tensors or attention window zeroed",
                "donor": "opposite observed cue sign; same world/pose; actual donor head state at32",
                "subsequent_input": "identical recipient hidden tail", "decoder_context": "recipient reference/pose retained"},
            "descriptive_gates_each_seed_and_case": GATES, "reference_only_is_not_state_erasure": True,
            "compute_scope": "Final-only retrieval loss/readout; attention buffers the whole prefix but evaluates only its final32-frame window",
            "training_memory_scope": "Full BPTT across all true chunks; no constant training memory claim",
            "tensor_state_accounting_excludes": ["optimizer", "activations", "FFT workspace", "Python overhead"],
            "deadline_scope": "global cooperative cap including preparation/training/evaluation; checked at update/example boundaries",
            "source_sha256": sources, "native_source_sha256": native,
            "observation_config": dataclasses.asdict(CONFIG), "time_budget_seconds": args.seconds,
            "benchmark_valid": False, "promotion_allowed": False}
    save_json(args.out/"preregistration.json", spec)
    start, cpu_start = time.monotonic(), time.process_time()
    deadline = start+args.seconds
    records = dataset(980000, count, gap)
    heldout = dataset(990000, count, gap) + dataset(990100, 2, gap, ("static_noevent",))
    data_file = args.out/"evaluation-inputs.pt"
    torch.save([{"features": e.record.features, "donor_features": e.donor_record.features,
                 "reference_canonical": e.record.reference_canonical, "known": e.record.known,
                 "forecast_positions": e.record.forecast_positions, "gains": e.record.gains,
                 "offsets": e.record.offsets, "frames": e.case["frames"], "target": e.case["target"],
                 "event": e.case["event"], "donor_event": e.donor_case["event"],
                 "metadata": e.case["metadata"], "masks": e.case["masks"]} for e in heldout], data_file)
    results = []
    for seed in seeds:
        for kind in KINDS:
            if time.monotonic() >= deadline:
                break
            results.append(run_arm(kind, seed, records, heldout, updates, device, args.out, deadline))
        if time.monotonic() >= deadline:
            break
    unchanged = sources == {p.name: hash_file(p) for p in ROOT.glob("*.py")}
    native_unchanged = native == {p.name: hash_file(p) for p in (ROOT.parent/"wfvideo.py", ROOT.parent/"ssm_lite.py")}
    if not unchanged or not native_unchanged:
        raise RuntimeError("Source changed; reject campaign evidence")
    complete = len(results) == len(seeds)*len(KINDS) and all(r["completed"] for r in results)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_json(args.out/"receipt.json", {"status": "smoke_completed" if complete and args.smoke else
              "campaign_completed" if complete else "partial_not_comparable", "complete": complete,
              "smoke_only": args.smoke, "training_data_fingerprint": dataset_fingerprint(records),
              "heldout_data_fingerprint": dataset_fingerprint(heldout),
              "evaluation_inputs_file": data_file.name, "evaluation_inputs_sha256": hash_file(data_file),
              "preregistration_sha256": hash_file(args.out/"preregistration.json"),
              "source_unchanged": unchanged, "native_source_unchanged": native_unchanged,
              "wall_seconds": time.monotonic()-start, "cpu_seconds": time.process_time()-cpu_start,
              "peak_process_rss_bytes": peak if platform.system() == "Darwin" else peak*1024,
              "torch": torch.__version__, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "offline_runner_retains_full_examples": True, "constant_process_memory_demonstrated": False,
              "results": results, "long_memory_claim_allowed": False,
              "architecture_superiority_claim_allowed": False, "generated_video_quality_claim_allowed": False})
    print(json.dumps({"complete": complete, "smoke_only": args.smoke,
                      "arms": len(results), "new_generated_videos": 0}))


if __name__ == "__main__":
    main()
