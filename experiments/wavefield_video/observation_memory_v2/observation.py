"""Causal, observed-only translation/reference/RGB-latch control.

This is an explicit image memory, not a trained model. ``step`` receives one
RGB frame, never targets, transforms, masks, clip statistics, or future frames.
Positions are (x, y), in pixels relative to the first observation. Photometry
is ``observed = gain * canonical_reference + offset``. Normalized frames and
innovations use canonical units; ``read``/``forecast`` default to the latest
accepted observed units. Unknown pixels are returned with an explicit mask.

A small observed change is latched against the first observed reference.
Its subsequent disappearance does not erase that historical RGB change.
That behavior is a memory control, not a claim about current object presence.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ObservationConfig:
    atlas_padding: int = 16
    max_shift: float = 2.0
    search_step: float = 0.25
    refinement_step: float = 0.0625
    allow_vertical: bool = True
    min_coverage: float = 0.5
    min_inlier_fraction: float = 0.70
    max_alignment_error: float = 0.035
    inlier_threshold: float = 0.045
    ambiguity_margin: float = 0.012
    ambiguity_distance: float = 0.65
    min_texture: float = 0.02
    innovation_threshold: float = 0.055
    max_change_fraction: float = 0.25
    min_gain: float = 0.65
    max_gain: float = 1.45
    max_offset: float = 0.25

    def __post_init__(self):
        if self.atlas_padding < 2 or self.max_shift <= 0:
            raise ValueError("atlas_padding >= 2 and max_shift > 0 required")
        if not 0 < self.refinement_step <= self.search_step <= self.max_shift:
            raise ValueError("invalid subpixel search resolution")
        if not 0 < self.min_coverage <= 1 or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError("coverage/inlier fractions must be in (0, 1]")
        if not 0 < self.min_gain <= 1 <= self.max_gain:
            raise ValueError("gain bounds must contain 1")
        if not 0 <= self.max_change_fraction < self.min_inlier_fraction:
            raise ValueError("change fraction must be below inlier fraction")


@dataclass
class ObservationResult:
    # known/unknown describe the reference support before this frame's causal
    # extension. coverage is that overlap; read() reports retained-map coverage.
    normalized: torch.Tensor
    reference: torch.Tensor
    known: torch.Tensor
    unknown: torch.Tensor
    innovation: torch.Tensor
    accepted_write_mask: torch.Tensor
    reference_write_mask: torch.Tensor
    accepted: bool
    hold: bool
    reason: str
    confidence: float
    coverage: float
    position: tuple
    shift: tuple
    gain: float
    offset: float
    canonical_innovation: Optional[torch.Tensor] = None


class ObservationMemory:
    """One stream, bounded CPU state; every returned tensor is an owned copy."""

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self.atlas = self.seen = self.event_atlas = self.event_seen = None
        self.position = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.gain, self.offset = 1.0, 0.0
        self.shape = None
        self._last_accepted_frame = None
        self._cached_quality = (0.0, 0.0)

    def _frame(self, frame):
        if not isinstance(frame, torch.Tensor) or frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("frame must be a floating RGB tensor of shape (3,H,W)")
        if not frame.is_floating_point() or min(frame.shape[-2:]) < 3:
            raise ValueError("floating frames with H,W >= 3 required")
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("frame contains nonfinite values")
        if self.shape is not None and tuple(frame.shape) != self.shape:
            raise ValueError("frame shape cannot change within a stream")
        return frame.detach().to(device="cpu", dtype=torch.float32).clone()

    def _sample(self, atlas, positions):
        _, h, w = self.shape
        ah, aw = atlas.shape[-2:]
        pos = torch.as_tensor(positions, dtype=torch.float32).reshape(-1, 2)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        x = xx[None] + pos[:, 0, None, None] + self.config.atlas_padding
        y = yy[None] + pos[:, 1, None, None] + self.config.atlas_padding
        grid = torch.stack((2 * x / (aw - 1) - 1, 2 * y / (ah - 1) - 1), -1)
        return F.grid_sample(atlas[None].expand(len(pos), -1, -1, -1), grid,
                             mode="bilinear", padding_mode="zeros", align_corners=True)

    def _views(self, positions):
        ref = self._sample(self.atlas, positions)
        known = self._sample(self.seen.float(), positions)[:, 0] >= 1 - 1e-5
        return ref, known

    def _fit(self, current, positions):
        """Robust global gain/offset from overlap; changes are not alignment anchors."""
        ref, eligible = self._views(positions)
        cfg = self.config
        n, c, h, w = ref.shape
        x = ref.reshape(n, c, -1)
        y = current.reshape(1, c, -1).expand(n, -1, -1)
        mask = eligible.flatten(1)
        weight = mask.clone()
        count = mask.sum(1)
        gain, offset = torch.ones(n), torch.zeros(n)
        for _ in range(4):
            z = weight[:, None].float()
            mass = (weight.sum(1) * c).clamp_min(1)
            xm, ym = (x * z).sum((1, 2)) / mass, (y * z).sum((1, 2)) / mass
            xc, yc = x - xm[:, None, None], y - ym[:, None, None]
            gain = ((xc * yc * z).sum((1, 2)) /
                    (xc.square() * z).sum((1, 2)).clamp_min(1e-8)).clamp(0.4, 2.0)
            offset = ym - gain * xm
            error = ((y - offset[:, None, None]) / gain[:, None, None] - x).abs().mean(1)
            ordered = error.masked_fill(~mask, float("inf")).sort(1).values
            index = ((count.float() * 0.75).long() - 1).clamp(min=0, max=h*w-1)
            cutoff = ordered.gather(1, index[:, None])[:, 0]
            weight = mask & (error <= cutoff[:, None] + 1e-7)
        score = (error * weight).sum(1) / weight.sum(1).clamp_min(1)
        coverage = count.float() / (h * w)
        inliers = (mask & (error <= cfg.inlier_threshold)).sum(1) / count.clamp_min(1)
        chmean = (x * mask[:, None]).sum(2) / count[:, None].clamp_min(1)
        texture = (((x - chmean[:, :, None]).square() * mask[:, None]).sum((1, 2)) /
                   (count * c).clamp_min(1)).sqrt()
        valid = ((coverage >= cfg.min_coverage) & (count >= 4) &
                 (gain >= cfg.min_gain) & (gain <= cfg.max_gain) &
                 (offset.abs() <= cfg.max_offset) & (texture >= cfg.min_texture))
        objective = (score + 0.002 * (1 - coverage)).masked_fill(~valid, float("inf"))
        return objective, score, gain, offset, coverage, inliers

    def _align(self, frame):
        cfg = self.config
        # Repeated observations reuse a previously verified registration. This
        # never bootstraps confidence on a flat/repeating frame or skips writes.
        if self._last_accepted_frame is not None and torch.equal(frame, self._last_accepted_frame):
            return ((self.position, self.gain, self.offset), "accepted_cached",
                    *self._cached_quality)
        axis = torch.arange(-cfg.max_shift, cfg.max_shift + cfg.search_step/2, cfg.search_step)
        vertical = axis if cfg.allow_vertical else torch.tensor([0.0])
        dy, dx = torch.meshgrid(vertical, axis, indexing="ij")
        delta = torch.stack((dx.flatten(), dy.flatten()), 1)
        positions = delta + torch.tensor(self.position)
        fit = self._fit(frame, positions)
        best = int(fit[0].argmin())
        if not math.isfinite(float(fit[0][best])):
            return None, "insufficient_texture_or_overlap", 0.0, 0.0
        # Resolve separate, plausible basins before refining one subpixel peak.
        distance = (positions - positions[best]).norm(dim=1)
        competitor = fit[0].masked_fill(distance < cfg.ambiguity_distance, float("inf")).min()
        margin = float(competitor - fit[0][best])
        if margin < cfg.ambiguity_margin:
            return None, "ambiguous_alignment", 0.0, float(fit[4][best])
        local_axis = torch.arange(-cfg.search_step, cfg.search_step + cfg.refinement_step/2,
                                  cfg.refinement_step)
        ly, lx = torch.meshgrid(local_axis if cfg.allow_vertical else torch.tensor([0.0]),
                                local_axis, indexing="ij")
        refined = positions[best] + torch.stack((lx.flatten(), ly.flatten()), 1)
        allowed = ((refined - torch.tensor(self.position)).abs() <= cfg.max_shift + 1e-6).all(1)
        refined = refined[allowed]
        fine = self._fit(frame, refined)
        j = int(fine[0].argmin())
        pos = tuple(float(v) for v in refined[j])
        score, gain, offset, coverage, inliers = [float(v[j]) for v in fine[1:]]
        confidence = (max(0.0, 1 - score / cfg.max_alignment_error) * inliers *
                      min(1.0, margin / (2 * cfg.ambiguity_margin)))
        if score > cfg.max_alignment_error or inliers < cfg.min_inlier_fraction:
            return None, "occluded_or_inconsistent_overlap", confidence, coverage
        if any(abs(pos[i] - self.position[i]) >= cfg.max_shift - 1e-6 for i in range(2)):
            return None, "search_boundary", confidence, coverage
        pad = cfg.atlas_padding
        if any(abs(pos[i]) > pad - 1 for i in range(2)):
            return None, "atlas_boundary", confidence, coverage
        return (pos, gain, offset), "accepted", confidence, coverage

    def _weights(self, position):
        """Four observation coefficients per view pixel, for causal atlas writes."""
        _, h, w = self.shape
        ah, aw = self.atlas.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        x = xx.float() + position[0] + self.config.atlas_padding
        y = yy.float() + position[1] + self.config.atlas_padding
        ix, iy = x.floor().long(), y.floor().long()
        fx, fy = x - ix, y - iy
        indices, weights = [], []
        for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
            indices.append(((iy + oy).clamp(0, ah - 1) * aw + (ix + ox).clamp(0, aw - 1)).flatten())
            weights.append(((fy if oy else 1-fy) * (fx if ox else 1-fx)).flatten())
        return torch.stack(indices, 1), torch.stack(weights, 1)

    def _extend_reference(self, normalized, position, changed):
        """Write only identifiable unseen texels supported by this observation.

        Inverting the bilinear observation prevents repeatedly blurring a
        subpixel reference. Underdetermined/ill-conditioned extensions hold
        their unknown mask rather than inventing hidden random pixels.
        """
        _, h, w = self.shape
        idx, weight = self._weights(position)
        flat_seen = self.seen.flatten()
        active = weight > 1e-7
        unknown = torch.unique(idx[active & ~flat_seen[idx]])
        writes = torch.zeros(h*w, dtype=torch.bool)
        if len(unknown) == 0:
            return writes.reshape(1, h, w)
        rows = ~changed.flatten()
        if int(rows.sum()) < len(unknown):
            return writes.reshape(1, h, w)
        a = torch.zeros(h*w, len(unknown))
        for k, cell in enumerate(unknown):
            a[:, k] = (weight * (idx == cell)).sum(1)
        ar = a[rows]
        singular = torch.linalg.svdvals(ar)
        if len(singular) < len(unknown) or float(singular[-1]) < 1e-4:
            return writes.reshape(1, h, w)
        if float(singular[0] / singular[-1]) > 30:
            return writes.reshape(1, h, w)
        known_weights = weight * flat_seen[idx]
        known_rgb = (self.atlas.flatten(1)[:, idx] * known_weights[None]).sum(2)
        rhs = normalized.flatten(1).T - known_rgb.T
        solution = torch.linalg.lstsq(ar, rhs[rows], driver="gels").solution.T
        if not bool(torch.isfinite(solution).all()) or bool(((solution < -0.05) | (solution > 1.05)).any()):
            return writes.reshape(1, h, w)
        if float((ar @ solution.T - rhs[rows]).abs().max()) > 0.025:
            return writes.reshape(1, h, w)
        self.atlas.flatten(1)[:, unknown] = solution.clamp(0, 1)
        flat_seen[unknown] = True
        writes = (a > 1e-7).any(1)
        return writes.reshape(1, h, w)

    def _write_events(self, residual, changed, position, known):
        """Invert identifiable local changes using observed zero-change anchors.

        A fractional write/read pair must not blur the latch twice. Sparsity
        here means only the support of visibly changed pixels can change; the
        surrounding *observed* residuals constrain its texels. A rank/fit failure
        leaves historical memory untouched, rather than selecting an inverse.
        """
        canonical = torch.zeros_like(residual)
        if not bool(changed.any()):
            return changed.clone(), canonical
        idx, weight = self._weights(position)
        cells = torch.unique(idx[(weight > 1e-7) & changed.flatten()[:, None]])
        rows = known.flatten()
        if int(rows.sum()) < len(cells):
            return torch.zeros_like(changed), canonical
        a = torch.stack([(weight * (idx == cell)).sum(1) for cell in cells], 1)
        ar = a[rows]
        singular = torch.linalg.svdvals(ar)
        if len(singular) < len(cells) or float(singular[-1]) < 1e-4:
            return torch.zeros_like(changed), canonical
        if float(singular[0] / singular[-1]) > 30:
            return torch.zeros_like(changed), canonical
        rhs = residual.flatten(1).T[rows]
        values = torch.linalg.lstsq(ar, rhs, driver="gels").solution.T
        reconstructed_rgb = self.atlas.flatten(1)[:, cells] + values
        if not bool(torch.isfinite(values).all()) or bool(((reconstructed_rgb < -0.05) |
                                                           (reconstructed_rgb > 1.05)).any()):
            return torch.zeros_like(changed), canonical
        if float((ar @ values.T - rhs).abs().max()) > 0.025:
            return torch.zeros_like(changed), canonical
        selected = values.abs().mean(0) > self.config.innovation_threshold
        self.event_atlas.flatten(1)[:, cells[selected]] = values[:, selected]
        self.event_seen.flatten()[cells[selected]] = True
        # The adapter write is in the first observation's fixed world viewport.
        delta_atlas = torch.zeros_like(self.event_atlas)
        delta_atlas.flatten(1)[:, cells[selected]] = values[:, selected]
        canonical = self._sample(delta_atlas, [(0.0, 0.0)])[0]
        pixel_writes = changed.flatten() & (a[:, selected] > 1e-7).any(1)
        return pixel_writes.reshape_as(changed), canonical

    def step(self, frame: torch.Tensor) -> ObservationResult:
        frame = self._frame(frame)
        if self.atlas is None:
            self.shape = tuple(frame.shape)
            _, h, w = self.shape
            p = self.config.atlas_padding
            self.atlas = torch.full((3, h+2*p, w+2*p), 0.5)
            self.seen = torch.zeros(1, h+2*p, w+2*p, dtype=torch.bool)
            self.event_atlas = torch.zeros_like(self.atlas)
            self.event_seen = torch.zeros_like(self.seen)
            self.atlas[:, p:p+h, p:p+w] = frame
            self.seen[:, p:p+h, p:p+w] = True
            known = torch.ones(1, h, w, dtype=torch.bool)
            return ObservationResult(frame.clone(), frame.clone(), known, ~known, torch.zeros_like(frame),
                                     torch.zeros_like(known), known.clone(), True, False, "bootstrap",
                                     1.0, 1.0, self.position, self.velocity, 1.0, 0.0, torch.zeros_like(frame))
        alignment, reason, confidence, coverage = self._align(frame)
        if alignment is None:
            reference, known = self._views([self.position])
            known = known[0, None]
            # Every retained appearance/memory/pose/photometry tensor stays frozen.
            return ObservationResult((frame-self.offset)/self.gain, reference[0].clone(), known.clone(),
                                     ~known, torch.zeros_like(frame), torch.zeros_like(known),
                                     torch.zeros_like(known), False, True, reason, confidence, coverage,
                                     self.position, (0.0, 0.0), self.gain, self.offset, torch.zeros_like(frame))
        position, gain, offset = alignment
        reference, known = self._views([position])
        reference, known = reference[0], known[0, None]
        normalized = (frame - offset) / gain
        residual = (normalized - reference) * known
        changed = known & (residual.abs().mean(0, keepdim=True) > self.config.innovation_threshold)
        fraction = float(changed.sum() / known.sum().clamp_min(1))
        # A broad unmodelled change is unsafe to latch, even if robust alignment fit.
        if fraction > self.config.max_change_fraction:
            retained_ref, retained_known = self._views([self.position])
            retained_known = retained_known[0, None]
            return ObservationResult((frame-self.offset)/self.gain, retained_ref[0].clone(),
                                     retained_known.clone(), ~retained_known,
                                     torch.zeros_like(frame), torch.zeros_like(known), torch.zeros_like(known),
                                     False, True, "broad_change", confidence, coverage, self.position,
                                     (0.0, 0.0), self.gain, self.offset, torch.zeros_like(frame))
        extension = self._extend_reference(normalized, position, changed)
        accepted_writes, canonical = self._write_events(residual, changed, position, known)
        shift = tuple(position[i]-self.position[i] for i in range(2))
        self.position, self.velocity, self.gain, self.offset = position, shift, gain, offset
        self._last_accepted_frame = frame.clone()
        self._cached_quality = (confidence, coverage)
        return ObservationResult(normalized.clone(), reference.clone(), known.clone(), ~known,
                                 (residual * accepted_writes).clone(), accepted_writes.clone(), extension.clone(),
                                 True, False, reason, confidence, coverage, position, shift, gain, offset,
                                 canonical.clone())

    def read(self, position=None, *, use_latch=True, observed_units=True):
        if self.atlas is None:
            raise RuntimeError("read requires an observed bootstrap frame")
        position = self.position if position is None else tuple(float(v) for v in position)
        if len(position) != 2 or not all(math.isfinite(v) for v in position):
            raise ValueError("position must contain two finite observed-derived coordinates")
        ref, known = self._views([position])
        value = ref[0]
        if use_latch:
            value = value + self._sample(self.event_atlas, [position])[0]
        if observed_units:
            value = value * self.gain + self.offset
        return {"rgb": value.clone(), "reference": ref[0].clone(), "known": known[0, None].clone(),
                "unknown": ~known[0, None], "position": position,
                "coverage": float(known[0].float().mean()), "gain": self.gain, "offset": self.offset}

    def forecast(self, *, use_latch=True, observed_units=True):
        """Explicit constant-displacement control, called before the next observation."""
        position = tuple(self.position[i] + self.velocity[i] for i in range(2))
        return self.read(position, use_latch=use_latch, observed_units=observed_units)

    def clear_memory(self):
        if self.event_atlas is not None:
            self.event_atlas.zero_()
            self.event_seen.zero_()

    def replace_memory_from(self, other):
        """Donor intervention changes historical RGB latch only, preserving appearance/pose."""
        if self.shape != other.shape or self.config.atlas_padding != other.config.atlas_padding:
            raise ValueError("donor memory must have matching frame/atlas shape")
        if self.event_atlas is None or other.event_atlas is None:
            raise RuntimeError("both donor and recipient must be bootstrapped")
        self.event_atlas.copy_(other.event_atlas)
        self.event_seen.copy_(other.event_seen)

    def state_dict(self):
        return {"config": asdict(self.config), "shape": self.shape, "position": self.position, "velocity": self.velocity,
                "gain": self.gain, "offset": self.offset,
                **{name: None if getattr(self, name) is None else getattr(self, name).clone()
                   for name in ("atlas", "seen", "event_atlas", "event_seen", "_last_accepted_frame")},
                "_cached_quality": self._cached_quality}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("config") != asdict(self.config):
            raise ValueError("state configuration does not match this stream")
        shape = state.get("shape")
        if shape is None:
            self.__init__(self.config)
            return
        if not isinstance(shape, (tuple, list)) or len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 3:
            raise ValueError("invalid state frame shape")
        p = self.config.atlas_padding
        expected = (3, shape[1]+2*p, shape[2]+2*p)
        tensors = {}
        for name in ("atlas", "event_atlas", "seen", "event_seen"):
            value = state.get(name)
            want = expected if name in ("atlas", "event_atlas") else (1, *expected[1:])
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != want:
                raise ValueError("invalid state tensor shape")
            if name in ("seen", "event_seen"):
                if value.dtype != torch.bool:
                    raise ValueError("state coverage tensors must be boolean")
            elif not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("state appearance tensors must be finite floating point")
            tensors[name] = value.detach().cpu().clone()
        coordinates = {}
        for name in ("position", "velocity", "_cached_quality"):
            value = state.get(name)
            if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(math.isfinite(v) for v in value):
                raise ValueError("state coordinates/quality must contain two finite values")
            coordinates[name] = tuple(float(v) for v in value)
        gain, offset = state.get("gain"), state.get("offset")
        if not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError("state gain must be finite and positive")
        if not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise ValueError("state offset must be finite")
        cached = state.get("_last_accepted_frame")
        if cached is not None and (not isinstance(cached, torch.Tensor) or tuple(cached.shape) != tuple(shape)
                                   or not cached.is_floating_point() or not bool(torch.isfinite(cached).all())):
            raise ValueError("invalid cached observed frame")
        # Validate everything before mutating the recipient stream.
        for name, value in tensors.items():
            setattr(self, name, value)
        self.shape = tuple(shape)
        self.position, self.velocity = coordinates["position"], coordinates["velocity"]
        self.gain, self.offset = float(gain), float(offset)
        self._last_accepted_frame = None if cached is None else cached.detach().cpu().clone()
        self._cached_quality = coordinates["_cached_quality"]


def observe(frames: torch.Tensor, config: Optional[ObservationConfig] = None):
    """Convenience causal replay; no pooling, future mean, or oracle metadata.

    Per-step arrays are in each observed view's coordinates. Final reference/
    known/prediction are the one-step forecast after processing the prefix.
    ``streams`` permits root-side erasure/donor controls without re-reading GT.
    """
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5 or min(frames.shape[:2]) < 1:
        raise ValueError("frames must have shape (N,T,3,H,W), N,T >= 1")
    streams, traces = [], []
    for sequence in frames:
        memory = ObservationMemory(config)
        traces.append([memory.step(frame) for frame in sequence])
        streams.append(memory)
    forecasts = [stream.forecast() for stream in streams]
    return {"innovations": torch.stack([torch.stack([r.innovation for r in trace]) for trace in traces]),
            "canonical_innovations": torch.stack([torch.stack([r.canonical_innovation for r in trace]) for trace in traces]),
            "accepted_write_masks": torch.stack([torch.stack([r.accepted_write_mask for r in trace]) for trace in traces]),
            "accepted": torch.tensor([[r.accepted for r in trace] for trace in traces]),
            "hold": torch.tensor([[r.hold for r in trace] for trace in traces]),
            "confidence": torch.tensor([[r.confidence for r in trace] for trace in traces]),
            "coverage": torch.tensor([[r.coverage for r in trace] for trace in traces]),
            "positions": [[r.position for r in trace] for trace in traces],
            "reasons": [[r.reason for r in trace] for trace in traces],
            "reference": torch.stack([f["reference"] for f in forecasts]),
            "prediction": torch.stack([f["rgb"] for f in forecasts]),
            "known": torch.stack([f["known"] for f in forecasts]),
            "valid": torch.tensor([trace[-1].accepted for trace in traces]), "streams": streams}
