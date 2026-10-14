"""Bounded, frame-only recall fixtures and privileged evaluation controls.

The last observed cue is raw frame 31. Frames [32, 32 + gap) hide the
event, and the unobserved target is frame 32 + gap. This is a synthetic
recall protocol, not a claim that transient physical objects should reappear.
Inference receives only ``case['frames']``. All remaining fields are scorer
truth or protocol metadata; camera offsets and event coordinates are never
frontend inputs. A seed chooses data, not model initialization or training.
"""

from __future__ import annotations

import math
from typing import Any

import torch


CONDITIONS = (
    "clean", "brightness_step", "subpixel_quarter", "subpixel_half", "subpixel_write",
    "late_half_occlusion", "full_occlusion", "ambiguous_texture",
    "write_occlusion", "static_noevent",
)
OPTIONAL_CONDITIONS = ("partial_disocclusion", "small_occluder_static")
FAMILIES = ("chromatic", "achromatic")
HEIGHT = WIDTH = 12
CUE_START = 28
CUE_STOP = 32
LAST_CUE_INDEX = CUE_STOP - 1
_ALIASES = {
    "late_occlusion": "late_half_occlusion",
    "ambiguous_pattern": "ambiguous_texture",
    "static": "static_noevent",
    "subpixelquarter": "subpixel_quarter",
    "subpixelhalf": "subpixel_half",
    "latehalfocclusion": "late_half_occlusion",
    "fullocclusion": "full_occlusion",
    "writeocclusion": "write_occlusion",
    "static_no_event": "static_noevent",
}


def _crop(world: torch.Tensor, offset: float, width: int = WIDTH) -> torch.Tensor:
    """Finite horizontal bilinear crop, with no wrap or padding oracle."""
    left = math.floor(offset)
    fraction = offset - left
    if left < 0 or left + width + (fraction > 0) > world.shape[-1]:
        raise ValueError("camera crop leaves the finite world")
    crop = world[..., left:left + width]
    if fraction:
        crop = crop * (1 - fraction) + world[..., left + 1:left + width + 1] * fraction
    return crop.clone()


def _family_name(family: str | int) -> str:
    if isinstance(family, int):
        if family not in (0, 1):
            raise ValueError("family must be chromatic/achromatic or 0/1")
        return FAMILIES[family]
    aliases = {"chroma": "chromatic", "chro": "chromatic", "achro": "achromatic"}
    family = aliases.get(family, family)
    if family not in FAMILIES:
        raise ValueError(f"unknown event family: {family}")
    return family


def make_case(seed: int, condition: str, family: str | int, gap: int = 256, *,
              event_sign: float = 1.0) -> dict[str, Any]:
    """Return one reproducible RGB clip and its never-ingested target.

    ``frames`` is [T,3,12,12], with T=32+gap; target/background/event are
    [3,12,12]. Masks are [1,12,12]. ``partial_disocclusion`` and
    ``small_occluder_static`` are optional failure-boundary controls.
    ``event_sign=-1`` flips only the observed cue and corresponding event truth,
    preserving the seed's background, geometry, gain and camera trajectory.
    """
    condition = _ALIASES.get(condition, condition)
    if condition not in CONDITIONS + OPTIONAL_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if not isinstance(gap, int) or gap < 8:
        raise ValueError("gap must be an integer >= 8")
    if (isinstance(event_sign, bool) or not isinstance(event_sign, (int, float))
            or event_sign not in (-1., 1.)):
        raise ValueError("event_sign must be +1 or -1")
    family = _family_name(family)
    generator = torch.Generator().manual_seed(int(seed))
    target_index = CUE_STOP + gap
    h, w, origin = HEIGHT, WIDTH, 4
    world = .2 + .4 * torch.rand(3, h, 3 * w + 8, generator=generator)
    if condition == "ambiguous_texture":
        # A two-column period makes a later two-pixel displacement unidentifiable.
        tile = world[..., :2].clone()
        world = tile.repeat(1, 1, math.ceil(world.shape[-1] / 2))[..., :world.shape[-1]]
    event_world = torch.zeros_like(world)
    cue_y = int(torch.randint(2, h - 3, (), generator=generator))
    cue_x = int(torch.randint(2, w - 3, (), generator=generator))
    sign = 1 if int(torch.randint(2, (), generator=generator)) else -1
    vector = torch.tensor([1., -.8, .35] if family == "chromatic" else [1., 1., 1.])
    vector = vector * (.18 * sign * event_sign)
    event_present = condition not in ("static_noevent", "small_occluder_static")
    if event_present:
        event_world[:, cue_y:cue_y + 2, origin + cue_x:origin + cue_x + 2] = vector[:, None, None]
    path = [float(origin)] * (target_index + 1)
    nuisance_start = CUE_STOP + max(2, gap // 8)
    if condition in ("subpixel_quarter", "subpixel_half", "ambiguous_texture"):
        shift = {"subpixel_quarter": .25, "subpixel_half": .5, "ambiguous_texture": 2.}[condition]
        path[nuisance_start:] = [origin + shift] * (len(path) - nuisance_start)
    elif condition == "subpixel_write":
        # The cue is written at a fractional pose, not only read after a shift.
        path[16:] = [origin + .5] * (len(path) - 16)
    elif condition == "partial_disocclusion":
        # A next-frame-only crop exposes independently random, never-seen texture.
        path[-1] = float(origin + w // 2)
    brightness = [1.] * (target_index + 1)
    if condition in ("brightness_step", "static_noevent"):
        brightness[nuisance_start:] = [1.2] * (len(brightness) - nuisance_start)
    frames = torch.stack([_crop(world, path[t]) * brightness[t] for t in range(target_index)])
    for t in range(CUE_START, CUE_STOP):
        frames[t] += _crop(event_world, path[t]) * brightness[t]
    visibility = torch.ones(target_index, 1, h, w, dtype=torch.bool)
    occlusion_interval = None
    if condition == "late_half_occlusion":
        begin, end = target_index - 7, target_index - 2
        frames[begin:end, :, :, :w // 2] = .1
        visibility[begin:end, :, :, :w // 2] = False
        occlusion_interval = (begin, end)
    elif condition == "full_occlusion":
        index = min(target_index - 3, CUE_STOP + max(2, gap // 3))
        frames[index] = .1
        visibility[index] = False
        occlusion_interval = (index, index + 1)
    elif condition == "write_occlusion":
        frames[CUE_START:CUE_STOP] = .1
        visibility[CUE_START:CUE_STOP] = False
        occlusion_interval = (CUE_START, CUE_STOP)
    elif condition == "small_occluder_static":
        # RGB alone cannot always distinguish a transient cover from an object
        # write. This exposes that limit and is not a performance pass gate.
        frames[CUE_START:CUE_STOP, :, cue_y:cue_y + 2, cue_x:cue_x + 2] = .1
        visibility[CUE_START:CUE_STOP, :, cue_y:cue_y + 2, cue_x:cue_x + 2] = False
        occlusion_interval = (CUE_START, CUE_STOP)
    background = _crop(world, path[-1]) * brightness[-1]
    event = _crop(event_world, path[-1]) * brightness[-1]
    target = background + event
    event_roi = event.square().sum(0, keepdim=True) > 1e-10
    # Known appearance means the finite world coordinate contributed to at least
    # one unobscured input pixel. Fractional crops require both contributing cells.
    world_seen = torch.zeros(1, h, world.shape[-1], dtype=torch.bool)
    for t, offset in enumerate(path[:-1]):
        left = math.floor(offset)
        world_seen[..., left:left + w] |= visibility[t]
        if offset != left:
            world_seen[..., left + 1:left + w + 1] |= visibility[t]
    left = math.floor(path[-1])
    known = world_seen[..., left:left + w].clone()
    if path[-1] != left:
        known &= world_seen[..., left + 1:left + w + 1]
    cue_observable = bool(event_present and condition != "write_occlusion")
    alignment_identifiable = condition != "ambiguous_texture"
    return {
        "frames": frames.contiguous(),
        "target": target,
        "background": background,
        "event": event,
        "masks": {
            "known": known,
            "unknown": ~known,
            "event_roi": event_roi,
            "event_observable": event_roi & cue_observable,
            "event_scored": event_roi & cue_observable & alignment_identifiable,
        },
        "metadata": {
            "protocol": "raw_frame_recall_v2",
            "seed": int(seed), "seed_role": "data_only",
            "condition": condition, "family": family,
            "event_sign": float(event_sign),
            "gap": gap, "cue_interval": [CUE_START, CUE_STOP],
            "last_cue_index": LAST_CUE_INDEX,
            "hidden_interval": [CUE_STOP, target_index],
            "target_index": target_index,
            "event_present": event_present, "cue_observable": cue_observable,
            "target_event_present": bool(event_roi.any()),
            "alignment_identifiable": alignment_identifiable,
            "object_occluder_ambiguous": condition == "small_occluder_static",
            "cue_xy": [cue_x, cue_y],
            "camera_path": path, "brightness": brightness,
            "occlusion_interval": occlusion_interval,
            "inference_fields": ["frames"],
            "benchmark_valid": False,
        },
    }


def controls(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Explicit privileged oracle and semantic negative controls; no contenders."""
    background, event = case["background"], case["event"]
    shifted = torch.zeros_like(event)
    shifted[..., 3:] = event[..., :-3]
    return {
        "privileged_oracle": case["target"].clone(),
        "frozen_last_context": case["frames"][-1].clone(),
        "privileged_background_only": background.clone(),
        "wrong_location": background + shifted,
        "wrong_identity": background - event,
    }


def _masked_mse(error: torch.Tensor, mask: torch.Tensor) -> float | None:
    expanded = mask.expand_as(error)
    return float(error[expanded].square().mean()) if bool(expanded.any()) else None


def score(pred: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    """Score full RGB reconstruction, ROI recovery and semantic residual controls.

    Full-field reconstruction skill is divided by event energy for scale, but
    combines background and event errors: it is not a memory-component metric.
    Event ROI skill uses known, observable, unambiguous truth ROI, never detected
    or matched-only pixels. Background/camera/ROI are privileged scorer truth.
    Use ``score_memory`` with the arm's own reference to isolate memory delta.
    """
    target, background, event = (case[k] for k in ("target", "background", "event"))
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} != target {tuple(target.shape)}")
    pred = pred.detach().float().cpu()
    target, background, event = (v.detach().float().cpu() for v in (target, background, event))
    masks = {k: v.detach().bool().cpu() for k, v in case["masks"].items()}
    finite = bool(torch.isfinite(pred).all())
    event_energy = float(event.square().sum())
    has_event = event_energy > 1e-10
    result: dict[str, Any] = {
        "finite": finite, "event_present": has_event,
        "cue_observable": bool(case["metadata"]["cue_observable"]),
        "alignment_identifiable": bool(case["metadata"]["alignment_identifiable"]),
        "event_scored": bool(masks["event_scored"].any()),
        "known_fraction": float(masks["known"].float().mean()),
        "unknown_fraction": float(masks["unknown"].float().mean()),
        "event_energy": event_energy,
    }
    if not finite:
        return result | {"full_mse": None, "known_mse": None, "unknown_mse": None,
                         "event_roi_mse": None, "event_roi_mse_unconditional": None,
                         "background_mse": None, "event_roi_skill": None,
                         "reconstruction_skill_unconditional": None, "presence": False,
                         "presence_rate": 0., "location_hit": 0., "identity_hit": 0.,
                         "identity_cosine": None, "location_error_pixels": None,
                         "joint_hit": 0., "false_presence": None if has_event else True}
    error, residual = pred - target, pred - background
    energy = residual.square().sum(0)
    presence = bool(energy.max() > 3 * .06 ** 2)
    location_hit = identity_hit = 0.
    location_error = None
    identity_cosine = None
    skill = None
    if has_event:
        skill = 1. - float(error.square().sum()) / event_energy
        predicted_peak = int(energy.argmax())
        truth_energy = event.square().sum(0).flatten()
        # Equal-energy locations are all valid, including subpixel ties.
        location_hit = float(truth_energy[predicted_peak] >= truth_energy.max() * (1 - 1e-5))
        actual_vector = residual.sum((1, 2))
        truth_vector = event.sum((1, 2))
        identity_cosine = float(torch.nn.functional.cosine_similarity(actual_vector, truth_vector, dim=0))
        identity_hit = float(identity_cosine >= .9)
        h, w = energy.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        truth = event.square().sum(0)
        if float(energy.sum()) > 1e-10:
            pxy = torch.stack([(xx * energy).sum(), (yy * energy).sum()]) / energy.sum()
            txy = torch.stack([(xx * truth).sum(), (yy * truth).sum()]) / truth.sum()
            location_error = float((pxy - txy).norm())
    result.update({
        "full_mse": float(error.square().mean()),
        "known_mse": _masked_mse(error, masks["known"]),
        "unknown_mse": _masked_mse(error, masks["unknown"]),
        "event_roi_mse": _masked_mse(error, masks["event_scored"] & masks["known"]),
        "event_roi_mse_unconditional": _masked_mse(error, masks["event_roi"]),
        "background_mse": _masked_mse(error, ~masks["event_roi"]),
        "reconstruction_skill_unconditional": skill,
        "presence": presence, "presence_rate": float(presence),
        "location_hit": location_hit, "identity_hit": identity_hit,
        "identity_cosine": identity_cosine, "location_error_pixels": location_error,
        "joint_hit": float(presence) * location_hit * identity_hit,
        "false_presence": None if has_event else presence,
    })
    roi = (masks["event_scored"] & masks["known"]).expand_as(error)
    roi_energy = float(event[roi].square().sum())
    result["event_roi_skill"] = (1. - float(error[roi].square().sum()) / roi_energy
                                 if roi_energy > 1e-10 else None)
    return result


def score_memory(pred: torch.Tensor, reference_pred: torch.Tensor,
                 case: dict[str, Any]) -> dict[str, Any]:
    """Score an arm's intact-minus-own-reference RGB delta against event truth.

    The reference must be that arm's actual erased-memory/reference forecast,
    in the same observed brightness/pose units. It is never an oracle background
    inserted into inference. This measures component attribution, not a separate
    generative-video result. Shared reconstruction errors cancel in the delta.
    """
    if pred.shape != reference_pred.shape or pred.shape != case["event"].shape:
        raise ValueError("intact/reference predictions must both match event shape")
    component_case = dict(case)
    component_case["background"] = torch.zeros_like(case["event"])
    component_case["target"] = case["event"]
    component = score(pred.detach() - reference_pred.detach(), component_case)
    mappings = {
        "finite": "memory_delta_finite",
        "full_mse": "memory_delta_mse",
        "known_mse": "memory_delta_known_mse",
        "unknown_mse": "memory_delta_unknown_mse",
        "reconstruction_skill_unconditional": "memory_delta_skill",
        "event_roi_skill": "memory_delta_roi_skill",
        "event_roi_mse": "memory_delta_roi_mse",
        "presence": "memory_delta_presence",
        "presence_rate": "memory_delta_presence_rate",
        "location_hit": "memory_delta_location_hit",
        "identity_hit": "memory_delta_identity_hit",
        "joint_hit": "memory_delta_joint_hit",
        "false_presence": "memory_delta_false_presence",
        "identity_cosine": "memory_delta_identity_cosine",
        "location_error_pixels": "memory_delta_location_error_pixels",
    }
    result = {new: component[old] for old, new in mappings.items()}
    return result | {k: component[k] for k in (
        "event_present", "cue_observable", "alignment_identifiable", "event_scored")}
