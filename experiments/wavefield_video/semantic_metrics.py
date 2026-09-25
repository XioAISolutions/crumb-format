"""Count visible ball blobs and match their local centroids in pixel space.

The three palette anchors are linearly unmixed just as in the legacy
``centroids_by_color`` metric, but spatial peaks are detected BEFORE consulting
the ground truth. Repeated anchor colors therefore permit multiple detections;
an empty/dark frame has none. This is a visible-blob metric: fully merged balls
of the same color may be indistinguishable, so GT count comes from metadata.

Inputs are RGB pixels, including AE-decoded pixels, never latent channels.
Distances/thresholds are native grid cells, independent of display scaling.
"""

import math

import torch
import torch.nn.functional as F

from data import PALETTE_ANCHORS, RADIUS


PEAK_THRESHOLD = 0.25       # absolute unmixed density, never normalized per frame
PEAK_PROMINENCE = 0.10      # rejects flat bright fields as well as dark collapse
MATCH_RADIUS_MULT = 2.0


def _radius(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("semantic radius must be finite and positive")
    return value


def _peak_plateaus(mask, field, minima):
    """Collapse a connected flat maximum to one peak before spatial NMS."""
    if not (mask & ((field - minima) >= PEAK_PROMINENCE)).any():
        return []
    remaining = set(map(tuple, mask.nonzero().tolist()))
    output = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = [start], [start]
        while pending:
            y, x = pending.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
                        component.append(neighbor)
        baseline = min(float(minima[y, x]) for y, x in component)
        if float(field[start]) - baseline < PEAK_PROMINENCE:
            continue
        cy = sum(y for y, x in component) / len(component)
        cx = sum(x for y, x in component) / len(component)
        y, x = min(component, key=lambda p: ((p[0] - cy) ** 2 + (p[1] - cx) ** 2, p))
        output.append((y, x, baseline))
    return sorted(output, key=lambda p: (-float(field[p[0], p[1]]), p[0], p[1]))


def detect_blobs(frames, *, radius=RADIUS):
    """Return one list of detections per RGB frame ``[B,3,H,W]``.

    Detections contain ``position`` (x,y), ``color_index`` and ``peak``. Peaks
    must exceed both an absolute brightness and a local contrast threshold.
    A brightness-weighted local centroid is measured within 2 sigma of each
    peak; nearby same-color peaks partition their pixels by nearest peak.
    The detector accepts neither GT positions nor an expected object count.
    """
    radius = _radius(radius)
    if frames.ndim != 4 or frames.shape[1] != 3 or min(frames.shape[-2:]) < 1:
        raise ValueError("semantic frames must have shape [B,3,H,W]")
    pixels = frames.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise ValueError("semantic frames must be finite")
    anchors = PALETTE_ANCHORS.to(device="cpu", dtype=torch.float32)
    inverse = torch.linalg.inv(anchors.T)
    density = torch.einsum("nc,bchw->bnhw", inverse, pixels).clamp_min(0)
    # Suppress neighboring grid maxima of one Gaussian. At the .8-pixel rung,
    # even a subpixel-centered Gaussian remains well above PEAK_THRESHOLD.
    nms = max(1, math.ceil(radius))
    support = max(1, math.ceil(2 * radius))
    maxima = F.max_pool2d(density, 2 * nms + 1, stride=1, padding=nms)
    minima = -F.max_pool2d(-density, 2 * support + 1, stride=1, padding=support)
    candidates = (density >= PEAK_THRESHOLD) & (density == maxima)
    height, width = pixels.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    output = []
    for sample in range(len(pixels)):
        detections = []
        for color_index in range(len(anchors)):
            field = density[sample, color_index]
            points = _peak_plateaus(candidates[sample, color_index], field,
                                    minima[sample, color_index])
            # Equal-height plateaus can have several maxima. Deterministic NMS
            # makes those one detection without forcing a particular count.
            peaks = []
            for y, x, baseline in points:
                if all(max(abs(y - py), abs(x - px)) > nms for py, px, _ in peaks):
                    peaks.append((y, x, baseline))
            if not peaks:
                continue
            distances = torch.stack([(xx - x).square() + (yy - y).square()
                                     for y, x, _ in peaks])
            owner = distances.argmin(0)
            for index, (y, x, baseline) in enumerate(peaks):
                window = (distances[index] <= support * support) & (owner == index)
                weights = (field - baseline).clamp_min(0) * window
                mass = weights.sum()
                if mass <= 0:
                    continue
                position = [float((weights * xx).sum() / mass),
                            float((weights * yy).sum() / mass)]
                detections.append({"position": position, "color_index": color_index,
                                   "peak": float(field[y, x])})
        output.append(detections)
    return output


def _assignment(cost):
    """Minimum-cost rectangular assignment (rows <= columns), no SciPy needed."""
    rows = len(cost)
    if not rows:
        return []
    cols = len(cost[0])
    u, v = [0.0] * (rows + 1), [0.0] * (cols + 1)
    p, way = [0] * (cols + 1), [0] * (cols + 1)
    for row in range(1, rows + 1):
        p[0] = row
        col = 0
        best, used = [math.inf] * (cols + 1), [False] * (cols + 1)
        while True:
            used[col] = True
            current = p[col]
            delta, next_col = math.inf, 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                reduced = cost[current - 1][candidate - 1] - u[current] - v[candidate]
                if reduced < best[candidate]:
                    best[candidate], way[candidate] = reduced, col
                if best[candidate] < delta:
                    delta, next_col = best[candidate], candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    best[candidate] -= delta
            col = next_col
            if p[col] == 0:
                break
        while col:
            previous = way[col]
            p[col] = p[previous]
            col = previous
    result = [-1] * rows
    for col in range(1, cols + 1):
        if p[col]:
            result[p[col] - 1] = col - 1
    return result


def _matched_errors(positions, color_ids, detections, tolerance):
    """Maximize valid match count, then minimize total Euclidean distance."""
    count, predicted = len(positions), len(detections)
    if not count or not predicted:
        return []
    penalty = (count + 1) * (tolerance + 1)
    distances, costs = [], []
    for position, color_id in zip(positions, color_ids):
        row, cost = [], []
        for detection in detections:
            distance = math.dist(position, detection["position"])
            valid = color_id == detection["color_index"] and distance <= tolerance
            row.append(distance if valid else None)
            cost.append(distance if valid else 2 * penalty)
        distances.append(row)
        costs.append(cost + [penalty] * count)  # each GT may be unmatched
    assignments = _assignment(costs)
    return [distances[row][col] for row, col in enumerate(assignments)
            if col < predicted and distances[row][col] is not None]


def semantic_frame_metrics(frames, gt_positions, colors, *, radius=RADIUS):
    """Score a batch; expected counts come from ``[B,N,2]`` GT metadata.

    Matches are one-to-one, same palette color, and within ``2*radius`` cells.
    A missing match is reflected in matched_count, never a fabricated zero
    position error. Report count/recall alongside the conditional matched error.
    """
    radius = _radius(radius)
    positions = gt_positions.detach().to(device="cpu", dtype=torch.float32)
    colors = colors.detach().to(device="cpu", dtype=torch.float32)
    if (positions.ndim != 3 or positions.shape[-1] != 2
            or positions.shape[0] != frames.shape[0]
            or tuple(colors.shape) != (*positions.shape[:2], 3)):
        raise ValueError("semantic GT must be positions [B,N,2] and colors [B,N,3]")
    if not torch.isfinite(positions).all() or not torch.isfinite(colors).all():
        raise ValueError("semantic GT positions/colors must be finite")
    anchors = PALETTE_ANCHORS.to(device="cpu", dtype=torch.float32)
    normalized = F.normalize(colors, dim=-1)
    color_ids = (normalized @ F.normalize(anchors, dim=-1).T).argmax(-1)
    detected = detect_blobs(frames, radius=radius)
    result = []
    for sample, blobs in enumerate(detected):
        errors = _matched_errors(positions[sample].tolist(), color_ids[sample].tolist(),
                                 blobs, MATCH_RADIUS_MULT * radius)
        result.append({"expected_n": int(positions.shape[1]), "detected_n": len(blobs),
                       "matched_count": len(errors),
                       "mean_matched_position_error": sum(errors) / len(errors) if errors else None})
    return result


def _aggregate(samples):
    total = len(samples)
    matched = sum(sample["matched_count"] for sample in samples)
    error = sum(sample["mean_matched_position_error"] * sample["matched_count"]
                for sample in samples if sample["matched_count"])
    return {**{name: sum(sample[name] for sample in samples) / total
               for name in ("expected_n", "detected_n", "matched_count")},
            "mean_matched_position_error": error / matched if matched else None}


def aggregate_semantic_metrics(frame_samples, *, radius=RADIUS):
    """Aggregate [frame][sample] records, retaining every per-frame/per-seed count.

    Counts are per-sample means; position errors are weighted by matched count.
    Frame 0 is the first predicted frame. Returns None for no ball records.
    """
    radius = _radius(radius)
    if not frame_samples or not any(frame_samples):
        return None
    if not all(frame_samples):
        raise ValueError("each semantic frame must contain at least one sample")
    frames = [{"frame": index, **_aggregate(samples), "samples": samples}
              for index, samples in enumerate(frame_samples)]
    return {**_aggregate([sample for samples in frame_samples for sample in samples]),
            "frames": frames, "units": "grid_cells", "count_reduction": "mean_per_sample",
            "position_error_reduction": "mean_over_matched_blobs",
            "match_tolerance_px": MATCH_RADIUS_MULT * radius,
            "detection": {"palette": "data.PALETTE_ANCHORS", "peak_threshold": PEAK_THRESHOLD,
                          "peak_prominence": PEAK_PROMINENCE, "radius_px": radius,
                          "matching": "same-color maximum-cardinality minimum-distance assignment"}}


def semantic_summary(report):
    """Compact log line shared by training eval and both render paths."""
    if report is None:
        return "semantic=n/a"
    error = report["mean_matched_position_error"]
    position = "null" if error is None else f"{error:.3f}px"
    return (f"semantic expected={report['expected_n']:.2f} detected={report['detected_n']:.2f} "
            f"matched={report['matched_count']:.2f} pos_err={position}")
