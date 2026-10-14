"""Frozen CPU engineering comparison; no training and no oracle inputs to contenders."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import torch

from observation import ObservationConfig, ObservationMemory
from protocol import CONDITIONS, OPTIONAL_CONDITIONS, controls, make_case, score, score_memory

ROOT = Path(__file__).resolve().parent
PRIMARY = ("clean", "brightness_step", "subpixel_quarter", "subpixel_half",
           "subpixel_write", "late_half_occlusion", "full_occlusion")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def infer(frames, integer=False):
    """Contender boundary: receives frames only, never a case or its labels."""
    config = ObservationConfig(allow_vertical=False)
    if integer:
        config = ObservationConfig(allow_vertical=False, search_step=1.0, refinement_step=1.0)
    memory = ObservationMemory(config)
    trace = []
    for frame in frames:
        result = memory.step(frame)
        trace.append({"accepted": result.accepted, "reason": result.reason,
                      "confidence": result.confidence, "coverage": result.coverage,
                      "write_pixels": int(result.accepted_write_mask.sum())})
    return memory.forecast()["rgb"], memory, trace


def raw_controls(frames):
    """Unregistered all-time latch, reference and frame EMA; no cue-timing oracle."""
    reference = frames[0].clone()
    latch = torch.zeros_like(reference)
    ema = reference.clone()
    for frame in frames[1:]:
        delta = frame - reference
        changed = delta.abs().mean(0, keepdim=True) > .055
        # Broad changes cannot be safely interpreted as a remembered object.
        if float(changed.float().mean()) <= .25:
            latch = torch.where(changed, delta, latch)
        ema = .95 * ema + .05 * frame
    return {"unregistered_all_time_latch": reference + latch,
            "observed_reference": reference, "frame_ema": ema}


def payload_bytes(memory):
    return sum(v.numel() * v.element_size() for v in memory.state_dict().values()
               if isinstance(v, torch.Tensor))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["arm"])].append(row)
    result = []
    keys = ("joint_hit", "full_mse", "background_mse", "memory_delta_joint_hit",
            "memory_delta_skill", "alignment_acceptance", "false_presence")
    for (condition, arm), group in sorted(groups.items()):
        entry = {"condition": condition, "arm": arm, "count": len(group)}
        for key in keys:
            values = [float(r[key]) for r in group if r.get(key) is not None]
            entry[key] = sum(values)/len(values) if values else None
        result.append(entry)
    return result


def gates(rows, seeds):
    per_seed = []
    for seed in seeds:
        main = [r for r in rows if r["seed"] == seed and r["arm"] == "aligned_reference_latch"
                and r["condition"] in PRIMARY]
        static = [r for r in rows if r["seed"] == seed and r["arm"] == "aligned_reference_latch"
                  and r["condition"] == "static_noevent"]
        checks = {
            "joint_recovery": bool(main) and all(r["memory_delta_joint_hit"] >= .8 for r in main),
            "memory_component_skill": bool(main) and all(r["memory_delta_skill"] >= .5 for r in main),
            "erase_dependence": bool(main) and all(r["erase_drop"] >= .5 for r in main),
            "donor_identity": bool(main) and all(r["donor_joint_hit"] >= .75 for r in main),
            "static_false_presence": bool(static) and sum(bool(r["memory_delta_false_presence"])
                                                        for r in static)/len(static) <= .05,
        }
        per_seed.append({"data_seed": seed, "seed_role": "fixture_only",
                         "checks": checks, "pass": all(checks.values())})
    return {"per_data_seed": per_seed, "pass": all(r["pass"] for r in per_seed),
            "interpretation": "Engineering RGB-reference retention only; no learned-memory/video claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[730101, 730102, 730103])
    parser.add_argument("--gap", type=int, default=256)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS) + list(OPTIONAL_CONDITIONS))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory; old evidence is never overwritten")
    if not set(PRIMARY + ("static_noevent",)).issubset(args.conditions):
        raise ValueError("The registered primary/static conditions are mandatory")
    if args.gap != 256 or len(set(args.seeds)) < 3:
        raise ValueError("Registered run requires a256 raw-frame gap and at least3 data seeds")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True)
    source = {p.name: digest(p) for p in ROOT.glob("*.py")}
    prereg = {
        "version": "observation_reference_v2", "data_seeds": args.seeds,
        "initialization_seeds": [], "trained_weights": False,
        "raw_frame_gap": args.gap, "last_cue": 31,
        "hidden_interval": [32, 288], "predict_before_ingesting": 288,
        "latent_step_gap": None, "conditions": args.conditions,
        "primary_conditions": list(PRIMARY), "families": ["chromatic", "achromatic"],
        "contender_input": ["frames"], "primary_arm": "aligned_reference_latch",
        "baseline_scope": "integer-only ablation shares photometry and all-time latch; not original E2",
        "gates": {"joint": .8, "memory_delta_skill": .5, "erase_drop": .5,
                  "donor_joint": .75, "static_false_presence": .05},
        "privileged_controls": ["privileged_oracle", "privileged_background_only",
                                "wrong_location", "wrong_identity"],
        "source_sha256": source, "benchmark_valid": False,
        "claim_scope": "Causal screen-space observed RGB memory engineering; not natural-video quality.",
    }
    json_write(args.out/"preregistration.json", prereg)
    started, cpu_started = time.monotonic(), time.process_time()
    rows, examples = [], []
    for seed in args.seeds:
        for condition in args.conditions:
            for family in ("chromatic", "achromatic"):
                case = make_case(seed, condition, family, gap=args.gap)
                donor_case = make_case(seed, condition, family, gap=args.gap, event_sign=-1.0)
                t0 = time.monotonic()
                with torch.no_grad():
                    prediction, memory, trace = infer(case["frames"])
                    base = memory.forecast(use_latch=False)["rgb"]
                    intact = memory.state_dict()
                    memory.clear_memory()
                    erased = memory.forecast()["rgb"]
                    memory.load_state_dict(intact)
                    _, donor, _ = infer(donor_case["frames"])
                    memory.replace_memory_from(donor)
                    swapped = memory.forecast()["rgb"]
                    memory.load_state_dict(intact)
                    integer_pred, integer_memory, integer_trace = infer(case["frames"], integer=True)
                primary = score(prediction, case) | score_memory(prediction, base, case)
                erased_metrics = score_memory(erased, base, case)
                donor_metrics = score_memory(swapped, base, donor_case)
                primary.update({"erase_drop": primary["memory_delta_joint_hit"] -
                                erased_metrics["memory_delta_joint_hit"],
                                "donor_joint_hit": donor_metrics["memory_delta_joint_hit"],
                                "alignment_acceptance": sum(t["accepted"] for t in trace)/len(trace),
                                "minimum_coverage": min(t["coverage"] for t in trace),
                                "hold_frames": sum(not t["accepted"] for t in trace),
                                "accepted_write_pixels": sum(t["write_pixels"] for t in trace),
                                "retained_tensor_bytes": payload_bytes(memory),
                                "wall_seconds_with_interventions": time.monotonic() - t0})
                predictions = raw_controls(case["frames"]) | controls(case)
                predictions["aligned_reference_latch"] = prediction
                predictions["aligned_reference_only"] = base
                predictions["integer_only_aligned_latch"] = integer_pred
                predictions["erased_latch"] = erased
                for arm, pred in predictions.items():
                    metrics = primary if arm == "aligned_reference_latch" else score(pred, case)
                    if arm == "integer_only_aligned_latch":
                        metrics |= score_memory(pred, integer_memory.forecast(use_latch=False)["rgb"], case)
                        metrics["alignment_acceptance"] = sum(t["accepted"] for t in integer_trace)/len(integer_trace)
                    rows.append({"seed": seed, "family": family, "condition": condition,
                                 "arm": arm, "raw_frame_gap": args.gap, **metrics})
                if seed == args.seeds[0] and family == "chromatic" and condition in PRIMARY:
                    examples.append({"condition": condition, "cue": case["frames"][31],
                                     "last_context": case["frames"][-1], "target": case["target"],
                                     "prediction": prediction, "erased": erased,
                                     "integer": integer_pred})
                print(json.dumps({"seed": seed, "condition": condition, "family": family,
                                  "joint": primary["memory_delta_joint_hit"],
                                  "skill": round(primary["memory_delta_skill"], 6)
                                  if primary["memory_delta_skill"] is not None else None}), flush=True)
    after = {p.name: digest(p) for p in ROOT.glob("*.py")}
    if after != source:
        raise RuntimeError("Source changed during evaluation; do not accept this run")
    with (args.out/"metrics.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    json_write(args.out/"summary.json", aggregate(rows))
    verdict = gates(rows, args.seeds)
    json_write(args.out/"verdict.json", verdict)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json_write(args.out/"receipt.json", {"wall_seconds": time.monotonic()-started,
               "cpu_seconds": time.process_time()-cpu_started,
               "peak_process_rss_bytes": peak if platform.system() == "Darwin" else peak*1024,
               "torch": torch.__version__, "device": "cpu", "threads": 1,
               "source_hashes_unchanged": True, "source_sha256": after,
               "case_count": len(args.seeds)*len(args.conditions)*2, "metric_rows": len(rows),
               "frames_retained_by_offline_runner": True,
               "constant_process_memory_demonstrated": False, "training_updates": 0,
               "preregistration_sha256": digest(args.out/"preregistration.json")})
    torch.save(examples, args.out/"examples.pt")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
