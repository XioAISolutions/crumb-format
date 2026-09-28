#!/usr/bin/env python3
"""Head-to-head: wave / attention / SSM video predictor, same scaffold + budget.

v2 fairness + architecture suite:
  * residual prediction (start at copy-last), zero-init head           (--residual)
  * three trivial baselines in every result JSON + printed table
  * parameter matching across arms via ffn_mult bisection              (--target-params)
  * shared factorized (t,y,x) positional encoding for ALL arms (always on)
  * dispersion wave kernel, content gate, local path, linear-pad       (flags below)
  * multi-step rollout loss                                            (--rollout-loss)
  * divergence-horizon rollout eval with color-matched centroids       (--eval-rollout)
"""
import argparse, json, math, sys, time, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wfvideo import VideoPredictor
from data import make_clip_batch as _balls_make, RADIUS, SPEED, MOVE_THRESH, N_BALLS
import data_waves as _dw
import data_occlusion as _occ
from latent_ae import load_ae, DOWNSAMPLE
from semantic_metrics import semantic_frame_metrics, aggregate_semantic_metrics, semantic_summary

DATA_SOURCE = "balls"   # set in main() from --data-source
WAVE_FIELD = "wave"     # set in main() from --field


def make_clip_batch(bs, T, H, W, **kw):
    """Pluggable data source: balls (data.py), waves (data_waves.py) or the R14
    occlusion memory probe (data_occlusion.py)."""
    if DATA_SOURCE == "waves":
        kw.pop("kicks", None)
        kw.pop("collisions", None)
        kw.pop("radius", None)       # ball geometry; waves fields ignore it
        kw.pop("speed", None)
        kw.pop("nb", None)          # --n-balls must not change PDE Fourier modes
        return _dw.make_clip_batch(bs, T, H, W, field=WAVE_FIELD, **kw)
    if DATA_SOURCE == "occlusion":
        kw.pop("kicks", None)        # occlusion balls just bounce; no stochastic kicks
        return _occ.make_clip_batch(bs, T, H, W, **kw)
    return _balls_make(bs, T, H, W, **kw)

# --------------------------------------------------------------------------- #
# Anti-collapse context augmentations (RESEARCH_SWEEP_20260926 sec 2-4).
# Both act on CONTEXT frames ONLY -- targets stay clean/continuous -- and are
# gated by their flags so the default (off) numeric path is byte-identical: no
# tensors allocated, and (crucially) no draws from the global RNG stream that
# seeds clip generation. drift-pert uses a private torch.Generator for exactly
# that reason.
# --------------------------------------------------------------------------- #
HIST_DISC_LEVELS = 16        # "bits" mode: uniform quantization levels in [0,1]


def apply_drift_pert(frames, magnitude, seed):
    """Drift-perturbed augmentation (SurgVista-style; RESEARCH_SWEEP sec 3).

    Overlays a per-sample smooth low-frequency gain + bias + color field on the
    context frames, wandering slowly across the T frames. This is the SAME
    family as crumb_coherence run_m0.inject_drift (exposure ramp + brightness
    lift + low-freq spatial color cast) but (a) randomized per batch sample,
    (b) amplitude-scaled by `magnitude`, (c) drawn from a private generator so
    the global RNG stream is untouched. Anti-collapse rationale: a drifted
    history no longer maps onto the clean future under copy-last, so the model
    must infer real dynamics instead of echoing pixels.

    frames [B,T,C,H,W] -> same shape, clamped to [0,1]."""
    B, T, C, H, W = frames.shape
    dev = frames.device
    g = torch.Generator(device=dev).manual_seed(int(seed))
    def rnd(*shape):
        return torch.rand(*shape, generator=g, device=dev, dtype=frames.dtype)
    t  = torch.linspace(0, 1, T, device=dev, dtype=frames.dtype).view(1, T, 1, 1, 1)
    ys = torch.linspace(0, 1, H, device=dev, dtype=frames.dtype).view(1, 1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=dev, dtype=frames.dtype).view(1, 1, 1, 1, W)
    gslope = 2 * rnd(B, 1, 1, 1, 1) - 1                    # exposure ramp sign U(-1,1)
    bslope = 2 * rnd(B, 1, 1, 1, 1) - 1                    # brightness-lift sign
    gain = 1.0 + magnitude * gslope * t                   # 1 -> 1 +/- magnitude
    bias = magnitude * bslope * t                         # 0 -> +/- magnitude
    amp  = magnitude * rnd(B, 1, C, 1, 1)                 # per-channel color cast
    kx   = 1.0 + rnd(B, 1, 1, 1, 1)                        # low spatial freq [1,2] cycles
    ky   = 1.0 + rnd(B, 1, 1, 1, 1)
    ph0  = 2 * math.pi * rnd(B, 1, 1, 1, 1)               # random spatial phase
    phase = 2 * math.pi * 0.5 * t + ph0                   # slow temporal wander
    field = (amp
             * (0.5 + 0.5 * torch.sin(2 * math.pi * kx * xs + phase))
             * (0.5 + 0.5 * torch.cos(2 * math.pi * ky * ys + 0.7 * phase)))
    return (frames * gain + bias + field).clamp(0, 1)


def quantize_frames(frames, levels):
    """Discrete history representation (FramePack v1 History Discretization;
    RESEARCH_SWEEP sec 2/4). Round context frames to `levels` uniform bins in
    [0,1]; targets stay continuous. No grad path needed -- context is model
    input, not a loss term -- so plain rounding is fine. Anti-collapse
    rationale: coarsening the history removes the sub-bin precision the model
    would otherwise reuse to copy the last frame pixel-exactly."""
    return torch.round(frames.clamp(0, 1) * (levels - 1)) / (levels - 1)


# Fixed-before-running eval constants (item 10) -- do NOT move after seeing results.
DIV_THRESH = RADIUS          # 1.6 px: divergence when median centroid error exceeds this
DIV_CONSEC = 8               # ...for 8 consecutive rollout frames
ID_SURV_TOL = 2.0 * RADIUS   # a ball's identity "survives" if final centroid error < this


# ----------------------------------------------------------------------------- metrics
def centroids_by_color(frames, colors):
    """Color-matched per-ball centroids. frames [B,3,H,W], colors [B,nb,3] ->
    [B,nb,2] (x,y). Each pixel is projected onto each ball's unit color vector
    (relu), isolating that ball; the brightness-weighted centroid follows it
    through occlusion better than a single global centroid."""
    B, C, H, W = frames.shape
    nb = colors.shape[1]
    if nb == 3 and C == 3:
        # Exact linear unmix: rendered pixels are a linear mixture of the ball
        # colors, so solving A c = pixel recovers each ball's own density map
        # (replaces the projection, which bled ~4px between overlapping balls).
        A = colors.transpose(1, 2)
        try:
            minv = torch.linalg.inv(A)
        except Exception:
            minv = torch.linalg.pinv(A)
        proj = torch.einsum("bnc,bchw->bnhw", minv, frames).clamp(min=0)
    else:
        chat = colors / colors.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        proj = torch.einsum("bchw,bnc->bnhw", frames, chat).clamp(min=0)
    xs = torch.arange(W, device=frames.device).float()
    ys = torch.arange(H, device=frames.device).float()
    tot = proj.flatten(2).sum(-1).clamp(min=1e-6)                         # [B,nb]
    cx = (proj.sum(2) * xs).sum(-1) / tot
    cy = (proj.sum(3) * ys).sum(-1) / tot
    return torch.stack([cx, cy], -1)                                     # [B,nb,2]


def divergence_horizon(median_err, thresh, consec):
    """First rollout frame where median centroid error > thresh for ``consec``
    consecutive frames; R (never diverged) if that never happens."""
    R = median_err.numel()
    over = (median_err > thresh).tolist()
    run = 0
    for i, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= consec:
            return i - consec + 1
    return R


# ----------------------------------------------------------------------------- param match
def match_ffn_mult(build_count, target, tol=0.10):
    """Bisect ffn_mult so total params land within ``tol`` of target (params are
    monotonic in ffn_mult). Returns (ffn_mult, achieved_params)."""
    lo, hi = 0.05, 64.0
    best = None
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        n = build_count(mid)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid
        else:
            hi = mid
    return best


# ----------------------------------------------------------------------------- eval
def baseline_mses(ctx, tgt):
    """zero / copy-last / constant-velocity MSE against target frame."""
    last, prev = ctx[:, -1], ctx[:, -2]
    return (F.mse_loss(torch.zeros_like(tgt), tgt).item(),
            F.mse_loss(last, tgt).item(),
            F.mse_loss(last + (last - prev), tgt).item())


def motion_balanced_loss(pred, target, last, moving,
                         resid_balanced=False, motion_weighted=False, w_max=10.0):
    """DEEP_DIVE_astra sec 1 objective. ``moving`` is a [B,H,W] bool mask of the
    pixels that actually changed vs ``last`` (the ground-truth previous frame):

        L = 0.5*mean(E[moving]) + 0.5*mean(E[static]) + 0.25*MSE(pred-last, target-last)

    E = per-pixel squared error (mean over channels). Splitting moving vs static
    with equal weight makes the handful of moving pixels count as much as the
    entire static background, so copy-last is no longer a good optimum; the third
    term directly supervises the *motion residual* (what changed since last).

    Two scale-aware knobs (both default OFF -> the formula above is byte-identical):

      * ``resid_balanced`` (DEEP_DIVE_2 Opus leak #1): the 0.25 residual term is
        ``MSE(pred-last, target-last)`` == full-frame ``MSE(pred, target)`` (the
        ``last`` cancels), so it re-imports the copy-last shortcut the 0.5/0.5 split
        removed and its motion content dilutes ~= moving_frac as the canvas empties.
        When on, restrict that term to the moving pixels (E[moving]) so it stays
        grid-invariant instead of rewarding a static majority.

      * ``motion_weighted`` (DEEP_DIVE_2 Astra #1 soft variant): replace the hard
        moving-pixel mean with a soft per-pixel weighting by motion magnitude
        ``w = clamp(delta/(delta.mean()+eps), 1, w_max)`` where ``delta=|target-last|``
        (channel-mean). Static pixels clamp to w=1 (unchanged); fast pixels weigh up
        to w_max. Smoother than a binary mask when the mask is unstable at low
        moving_frac. e_stat/resid are unaffected."""
    E = (pred - target).pow(2).mean(1)                       # [B,H,W]
    static = ~moving
    if motion_weighted:
        delta = (target - last).abs().mean(1)                # [B,H,W]
        w = (delta / (delta.mean() + 1e-6)).clamp(1.0, w_max)
        e_move = (w * E).mean()
    else:
        e_move = E[moving].mean() if moving.any() else E.new_zeros(())
    e_stat = E[static].mean() if static.any() else E.new_zeros(())
    if resid_balanced:
        resid = E[moving].mean() if moving.any() else E.new_zeros(())
    else:
        resid = F.mse_loss(pred - last, target - last)
    return 0.5 * e_move + 0.5 * e_stat + 0.25 * resid


def rollout_eval(m, a, dev):
    """Divergence-horizon rollout: roll R frames from S unseen seeds, autoregressive.

    Seeds are processed in sub-batches of ``a.eval_chunk`` (DEEP_DIVE_2 OOM fix):
    each chunk is an INDEPENDENT R-frame rollout, so at most ``eval_chunk`` seeds
    are ever forwarded through ``m`` at once. Per-frame MSE is a seed-count-weighted
    mean across chunks (exact, since chunks partition the S seeds), and per-ball
    centroid errors are concatenated along the seed axis -> identical result to the
    old all-at-once path, just with a bounded activation footprint.

    Divergence thresholds scale with ``a.radius`` (default = data.RADIUS, so the
    reported ``div_thresh_px`` is unchanged unless the geometry control moves it)."""
    S, R = a.eval_seeds, a.eval_rollout
    chunk = max(1, min(a.eval_chunk, S))
    div_thresh = a.radius                                    # cells; == RADIUS by default
    id_tol = 2.0 * a.radius
    if DATA_SOURCE == "waves":
        clip = make_clip_batch(S, a.frames + R - 1, a.grid, a.grid, device=dev, seed=90000)
        meta, cols = None, None
    else:
        clip, meta = make_clip_batch(S, a.frames + R - 1, a.grid, a.grid, device=dev,
                                     seed=90000, kicks=a.kicks, collisions=a.collisions,
                                     radius=a.radius, speed=a.speed,
                                     nb=getattr(a, "n_balls", N_BALLS), return_meta=True)
        cols = meta["col"]                               # [S,nb,3]
    mse_sum = [0.0] * R                                  # seed-weighted MSE sum per frame
    persist_sum = [0.0] * R
    cerr_chunks = []                                     # each [R, cb, nb]
    semantic_samples = [[] for _ in range(R)]
    mem = {}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cb = c1 - c0
            win = clip[c0:c1, :a.frames].clone()
            ccols = cols[c0:c1] if cols is not None else None
            cerrs = []
            for k in range(R):
                with torch.autocast(device_type=dev, dtype=torch.bfloat16,
                                    enabled=(dev == "cuda" and getattr(a, "eval_amp", True))):
                    nxt = m(win).float().clamp(0, 1)
                gt = clip[c0:c1, a.frames + k]
                prev_gt = clip[c0:c1, a.frames + k - 1]
                mse_sum[k] += F.mse_loss(nxt, gt).item() * cb
                persist_sum[k] += F.mse_loss(prev_gt, gt).item() * cb
                if ccols is not None:
                    pc = centroids_by_color(nxt, ccols)
                    gc = meta["pos"][c0:c1, a.frames + k]
                    cerrs.append((pc - gc).norm(dim=-1))     # [cb,nb]
                    semantic_samples[k].extend(semantic_frame_metrics(
                        nxt, gc, ccols, radius=a.radius))
                win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], 1)
                if dev == "cuda" and (k + 1) in (32, 256):
                    mem[k + 1] = max(mem.get(k + 1, 0.0),
                                     round(torch.cuda.max_memory_allocated() / 1e9, 3))
            if cerrs:
                cerr_chunks.append(torch.stack(cerrs, 0))    # [R, cb, nb]
    dt = time.time() - t0
    step_mse = [s / S for s in mse_sum]
    step_persist = [s / S for s in persist_sum]
    mean_model = sum(step_mse) / len(step_mse)
    mean_persist = sum(step_persist) / len(step_persist)
    out = {
        "eval_rollout": R, "eval_seeds": S, "eval_chunk": chunk,
        "r_persist_over_model": round(mean_persist / (mean_model + 1e-9), 3),
        "div_thresh_px": round(div_thresh, 3), "div_consec": DIV_CONSEC,
        "rollout_fps": round(S * R / dt, 1),
        "mem_gb_at_32": mem.get(32), "mem_gb_at_256": mem.get(256),
        "rollout_mse_curve": [round(x, 5) for x in step_mse],
        "semantic": aggregate_semantic_metrics(semantic_samples, radius=a.radius) if cols is not None else None,
    }
    if cerr_chunks:
        cerr = torch.cat(cerr_chunks, 1)                  # [R,S,nb]
        median_cerr = cerr.reshape(R, -1).median(dim=1).values
        out.update({
            "mean_centroid_err": round(cerr.mean().item(), 3),
            "final_centroid_err": round(cerr[-1].mean().item(), 3),
            "identity_survival": round((cerr[-1] < id_tol).float().mean().item(), 3),
            "divergence_horizon": divergence_horizon(median_cerr, div_thresh, DIV_CONSEC),
        })
    else:
        out.update({"mean_centroid_err": None, "final_centroid_err": None,
                    "identity_survival": None, "divergence_horizon": None})
    return out


# ----------------------------------------------------------------------------- occlusion probe
def target_centroid(frames, dom):
    """Chroma-weighted centroid of the *target* ball only. ``dom`` is the target's
    dominant color channel; the weight ``relu(frames[:,dom] - luminance)`` is high
    only where that channel exceeds the frame's average -- so it ignores the gray
    occluder (R=G=B -> weight 0) and the other balls' hues (a different dominant
    channel -> weight <=0). When the model has *dropped* the target the weight
    vanishes and the centroid collapses to the origin, which correctly reads as a
    large tracking error. frames [B,3,H,W] -> [B,2] (x,y)."""
    B, C, H, W = frames.shape
    w = (frames[:, dom] - frames.mean(1)).clamp(min=0)          # [B,H,W]
    xs = torch.arange(W, device=frames.device).float()
    ys = torch.arange(H, device=frames.device).float()
    tot = w.flatten(1).sum(-1).clamp(min=1e-6)                  # [B]
    cx = (w.sum(1) * xs).sum(-1) / tot
    cy = (w.sum(2) * ys).sum(-1) / tot
    return torch.stack([cx, cy], -1)                            # [B,2]


def occlusion_rollout_eval(m, a, dev):
    """R14 long-memory occlusion rollout (the decisive experiment).

    Autoregressive rollout where the ENVIRONMENT occludes the target: at each step
    the model predicts the next frame, we occlude that prediction inside the hidden
    window before feeding it back, and we score the *raw* prediction's target
    centroid against the deterministic ground-truth trajectory. So the only way to
    track the target through the [occ_start, occ_end) gap is persistent state --
    the pixels fed back carry no target during occlusion.

      * recurrent arms (wave / ssm / local+ssm / local+wave) roll out with their
        O(1) ``stream_step`` recurrence (persistent state bridges the gap);
      * attention rolls out windowed -- ``m(win)`` over the last T occluded frames,
        which physically cannot see past the T-frame window (< the 256-frame gap).

    The headline metrics are exit-direction accuracy and position/velocity error at
    emergence, plus divergence horizon and STATE BYTES (reported by the caller)."""
    S, R = a.eval_seeds, a.eval_rollout
    chunk = max(1, min(a.eval_chunk, S))
    id_tol, div_thresh = 2.0 * a.radius, a.radius
    occ_start, occ_end = a.occ_start, a.occ_end
    e = occ_end - a.frames                                   # emergence index in rollout coords
    streaming = hasattr(m, "_streamable") and m._streamable()
    clip, meta = make_clip_batch(S, a.frames + R - 1, a.grid, a.grid, device=dev,
                                 seed=90000, radius=a.radius, speed=a.speed,
                                 nb=a.n_balls, collisions=a.collisions, return_meta=True)
    cols, tgt, rect = meta["col"], meta["target_idx"], meta["occluder"]
    dom = int(cols[0, tgt].argmax().item())                  # target's identity channel
    gpos = meta["pos"]                                       # [S, frames+R, nb, 2]
    tc_pred_chunks, tc_gt_chunks = [], []
    mse_sum = [0.0] * R
    persist_sum = [0.0] * R
    pmotion, gmotion, n_motion = 0.0, 0.0, 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ac = lambda: torch.autocast(device_type=dev, dtype=torch.bfloat16,
                                enabled=(dev == "cuda" and getattr(a, "eval_amp", True)))
    with torch.no_grad():
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cb = c1 - c0
            tc_pred, tc_gt = [], []
            frame = win = None
            if streaming:
                states = m.stream_init(cb, dev)
                for t in range(a.frames):                    # warm state on visible context
                    with ac():
                        frame, states = m.stream_step(clip[c0:c1, t], states, t)
                    frame = frame.float().clamp(0, 1)
            else:
                win = clip[c0:c1, :a.frames].clone()
            prev = prev_gt = None
            for k in range(R):
                abs_t = a.frames + k
                if streaming:
                    nxt = frame
                else:
                    with ac():
                        nxt = m(win).float().clamp(0, 1)
                gt = clip[c0:c1, abs_t]
                prev_gt_frame = clip[c0:c1, abs_t - 1]
                mse_sum[k] += F.mse_loss(nxt, gt).item() * cb
                persist_sum[k] += F.mse_loss(prev_gt_frame, gt).item() * cb
                if prev is not None:
                    pmotion += (nxt - prev).abs().mean().item() * cb
                    gmotion += (gt - prev_gt).abs().mean().item() * cb
                    n_motion += cb
                tc_pred.append(target_centroid(nxt, dom))                 # [cb,2]
                tc_gt.append(gpos[c0:c1, abs_t, tgt])                      # [cb,2]
                obs = _occ.occlude_observation(nxt, abs_t, rect, occ_start, occ_end)
                if streaming:
                    with ac():
                        frame, states = m.stream_step(obs, states, abs_t)
                    frame = frame.float().clamp(0, 1)
                else:
                    win = torch.cat([win[:, 1:], obs.unsqueeze(1)], 1)
                prev, prev_gt = nxt, gt
            tc_pred_chunks.append(torch.stack(tc_pred, 0))                 # [R, cb, 2]
            tc_gt_chunks.append(torch.stack(tc_gt, 0))
    dt = time.time() - t0
    tc_pred = torch.cat(tc_pred_chunks, 1)                                 # [R, S, 2]
    tc_gt = torch.cat(tc_gt_chunks, 1)
    cerr = (tc_pred - tc_gt).norm(dim=-1)                                  # [R, S] target error
    median_cerr = cerr.median(dim=1).values
    step_mse = [s / S for s in mse_sum]
    step_persist = [s / S for s in persist_sum]
    mean_model = sum(step_mse) / len(step_mse)
    mean_persist = sum(step_persist) / len(step_persist)
    motion_pres = round((pmotion / max(1, n_motion)) / ((gmotion / max(1, n_motion)) + 1e-9), 3)

    def emergence():
        if not (1 <= e < R):
            return None, None, None, None
        pv, gv = tc_pred[e] - tc_pred[e - 1], tc_gt[e] - tc_gt[e - 1]
        dir_ok = ((torch.sign(pv[:, 0]) == torch.sign(gv[:, 0]))
                  & (torch.sign(pv[:, 1]) == torch.sign(gv[:, 1]))).float().mean().item()
        return (cerr[e].mean().item(), (pv - gv).norm(dim=-1).mean().item(),
                dir_ok, (cerr[e:] < id_tol).float().mean().item())

    pos_err, vel_err, dir_ok, id_surv = emergence()
    rnd = lambda v: round(v, 3) if v is not None else None
    return {
        "mode": "occlusion_rollout", "streaming": streaming,
        "eval_rollout": R, "eval_seeds": S, "eval_chunk": chunk,
        "occ_start": occ_start, "occ_end": occ_end, "emergence_frame": e,
        "target_idx": tgt, "occluder": list(rect),
        "div_thresh_px": round(div_thresh, 3), "div_consec": DIV_CONSEC,
        "divergence_horizon": divergence_horizon(median_cerr, div_thresh, DIV_CONSEC),
        "r_persist_over_model": round(mean_persist / (mean_model + 1e-9), 3),
        "mean_target_centroid_err": round(cerr.mean().item(), 3),
        "final_target_centroid_err": round(cerr[-1].mean().item(), 3),
        "target_identity_survival": rnd(id_surv),
        "exit_direction_accuracy": rnd(dir_ok),
        "position_error_at_emergence": rnd(pos_err),
        "velocity_error_at_emergence": rnd(vel_err),
        "motion_preservation": motion_pres,
        "rollout_fps": round(S * R / dt, 1),
        "ms_per_frame": round(1000.0 * dt / (S * R), 4),
        "mem_gb_peak": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else None,
        "rollout_mse_curve": [round(x, 5) for x in step_mse],
        "target_centroid_curve": [round(x, 3) for x in cerr.mean(dim=1).tolist()],
        "semantic": None,
    }


# ----------------------------------------------------------------------------- streaming
def stream_test(m, a, dev):
    """Long streaming rollout (default 1024 frames), item (2).

      * wave  -> the O(1)-in-T ``step()`` recurrence (constant state, no window).
      * attn/ssm -> windowed rollout: keep the last T frames and re-run the full
        forward every frame (the only way an attention window can 'stream').

    Peak GPU memory (per-128-frame segment) + fps are logged every 128 frames to
    JSON, so the recurrence's flat cost is visible against the window's re-compute.
    Weights need not be trained -- this measures streaming cost, not accuracy."""
    Ntot, Bsz, T = a.stream_frames, a.stream_batch, a.frames
    clip = make_clip_batch(Bsz, T, a.grid, a.grid, device=dev, seed=70000,
                           kicks=a.kicks, collisions=a.collisions,
                           nb=getattr(a, "n_balls", N_BALLS), radius=a.radius, speed=a.speed)
    ctx = clip[:, :T].clone()                                   # [B,T,3,H,W]
    windowed = a.kind in ("attn", "ssm")
    if a.kind == "wave" and a.kernel_version != "dispersion":
        raise SystemExit("--stream-test with --kind wave requires --kernel-version dispersion")
    if a.kind == "wave" and (a.gate or a.local_fuse):
        print("STREAM WARN: step() covers the pure dispersion path; --gate (non-causal "
              "global pool) and --local-fuse are NOT threaded through streaming.", flush=True)
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    log = []
    t0 = seg_t0 = time.time()
    ac = torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))

    def checkpoint(k):
        nonlocal seg_t0
        now = time.time()
        mem = round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0
        seg = 128 * Bsz / max(now - seg_t0, 1e-9)
        cum = (k + 1) * Bsz / max(now - t0, 1e-9)
        log.append({"frame": k + 1, "peak_gb": mem,
                    "seg_fps": round(seg, 1), "cum_fps": round(cum, 1)})
        print(f"STREAM {a.kind}{a.tag} frame={k+1:5d} "
              f"peak_gb={mem} seg_fps={seg:.1f} cum_fps={cum:.1f}", flush=True)
        seg_t0 = now
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()                # per-segment peak

    with torch.no_grad():
        if windowed:                                            # attn/ssm: re-run T-window
            win = ctx.clone()
            for k in range(Ntot):
                with ac:
                    nxt = m(win).float().clamp(0, 1)
                win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], 1)
                if (k + 1) % 128 == 0:
                    checkpoint(k)
        else:                                                   # wave: O(1) step() recurrence
            states = m.stream_init(Bsz, dev)
            frame = None
            for t in range(T):                                  # warm state on context frames
                with ac:
                    frame, states = m.stream_step(ctx[:, t], states, t)
                frame = frame.float().clamp(0, 1)
            for k in range(Ntot):
                with ac:
                    frame, states = m.stream_step(frame, states, T + k)
                frame = frame.float().clamp(0, 1)
                if (k + 1) % 128 == 0:
                    checkpoint(k)

    res = {"mode": "stream_test", "kind": a.kind, "tag": a.tag,
           "path": "step()" if not windowed else "windowed", "window_T": T,
           "stream_frames": Ntot, "stream_batch": Bsz, "grid": a.grid,
           "kernel_version": a.kernel_version, "pole_param": a.pole_param,
           "checkpoints": log}
    return res


# ----------------------------------------------------------------------------- latent
class LatentCore(nn.Module):
    """Latent-space next-step predictor built ON TOP of a VideoPredictor.

    We reuse the predictor's *learned mixing stack* (posemb + blocks + norm) --
    the part that differs between wave / attn / ssm, i.e. the whole point of the
    comparison -- but swap the pixel I/O for cz-channel latent I/O:
        embed: Conv2d(3, dim)   -> Conv2d(cz, dim)
        head : Linear(dim, 3)   -> Linear(dim, cz)
    VideoPredictor.forward bakes in ``3`` output channels, so we can't call it
    directly (and we're not allowed to edit wfvideo.py); instead we re-run the
    identical block loop here with a cz-wide head. The zero-init residual head is
    preserved, so training still starts at 'copy last latent' == copy last frame.

    forward: latent window [B, T, cz, h, w] -> next latent [B, cz, h, w]."""

    def __init__(self, vp, dim, cz):
        super().__init__()
        self.vp, self.cz = vp, cz
        vp.embed = nn.Conv2d(cz, dim, 3, padding=1)
        vp.head = nn.Linear(dim, cz)
        if vp.residual:
            nn.init.zeros_(vp.head.weight)
            nn.init.zeros_(vp.head.bias)

    def forward(self, z):
        vp = self.vp
        B, T, C, H, W = z.shape
        f = z.reshape(B * T, C, H, W)
        e = vp.embed(f).reshape(B, T, vp.H * vp.W, -1).reshape(B, T * vp.H * vp.W, -1)
        x = vp.posemb(e)
        for blk in vp.blocks:
            if vp.use_ckpt and vp.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = vp.norm(x)
        last = x[:, (vp.T - 1) * vp.H * vp.W: vp.T * vp.H * vp.W]        # [B, hw, D]
        delta = vp.head(last).reshape(B, vp.H, vp.W, self.cz).permute(0, 3, 1, 2)
        return z[:, -1] + delta if vp.residual else delta               # [B, cz, h, w]


class LatentWrapper(nn.Module):
    """Pixel-in / pixel-out adapter so EVERY pixel-space eval path (rollout_eval,
    baseline_mses, centroids_by_color, single-step eval) runs unchanged.

    forward([B,T,3,H,W]) = decode( core( encode(context frames) ) ) -> [B,3,H,W].
    Each rollout step therefore does a full AE round-trip on the fed-back frame,
    so the reported pixel metrics honestly include the AE's reconstruction error
    -- exactly what 'the gate to real pixels' has to pay for."""

    def __init__(self, ae, core):
        super().__init__()
        self.ae, self.core = ae, core
        self.T, self.kind = core.vp.T, core.vp.kind

    def encode_clip(self, frames):                       # [B,T,3,H,W] -> [B,T,cz,h,w]
        B, T, C, H, W = frames.shape
        z = self.ae.encode(frames.reshape(B * T, C, H, W))
        return z.reshape(B, T, *z.shape[1:])

    def forward(self, frames):                           # [B,T,3,H,W] -> [B,3,H,W]
        zp = self.core(self.encode_clip(frames))
        return self.ae.decode(zp)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["wave", "attn", "ssm", "qssm"], required=True)
    # ---- R15 hypercomplex arms (HYPERCOMPLEX_STUDY.md sec 4-E2/E3) -----------
    ap.add_argument("--q-mix", action=argparse.BooleanOptionalAction, default=False,
                    help="[kind wave] E2 qwave: quaternion-structured pi/po (WaveQuatMix) "
                         "-- Hamilton maps at dh/4 slots per head; pair with "
                         "--target-params so the param saving is reinvested and the "
                         "arm stays at the same budget")
    ap.add_argument("--quat-color", action="store_true",
                    help="[kind wave] E3 qcolor: quaternion color stem (QuatEmbed) -- "
                         "pixels as (0,r,g,b), 3x3 Hamilton kernels, dim/4 quaternion "
                         "channels; ~9*dim vs 27*dim stem params")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed: seeds weight init AND offsets the per-step "
                         "training-clip seeds, so a multi-seed sweep gets independent "
                         "runs; the held-out eval seeds are fixed for a fair comparison")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--const-lr", action="store_true", help="constant LR (LambdaLR==1) instead of OneCycle")
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast (fp32 train+eval)")
    ap.add_argument("--no-decay-norm-head", action="store_true", help="exclude norm/bias/head from AdamW weight decay")
    # ---- anti-collapse arms (RESEARCH_SWEEP_20260926 sec 2-4) ---------------
    ap.add_argument("--var-reg", type=float, default=0.0,
                    help="VICReg-style variance-matching on per-step frame deltas: add "
                         "var_reg * relu(std(gt_delta) - std(pred_delta)) to the loss "
                         "(hinge fires when the prediction UNDER-moves = collapse). "
                         "0=off. Contribution logged in each step row as var_reg.")
    ap.add_argument("--drift-pert", type=float, default=0.0,
                    help="drift-perturbed training augmentation: smooth low-frequency "
                         "gain/bias/color field (magnitude=this) applied to CONTEXT "
                         "frames only (targets clean); same family as crumb_coherence "
                         "run_m0.inject_drift. 0=off.")
    ap.add_argument("--hist-disc", choices=["off", "bits"], default="off",
                    help=f"discrete history representation: 'bits' quantizes context "
                         f"frames to {HIST_DISC_LEVELS} uniform levels before feeding "
                         f"the model (targets stay continuous). off=continuous (default).")
    ap.add_argument("--telemetry-every", type=int, default=25, help="log head grad-norm + batch copy_ratio every N steps (0=off)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--target-params", type=int, default=0, help="auto-match ffn_mult to this")
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                    help="predict last_frame + delta, zero-init head (default ON for v2)")
    ap.add_argument("--kernel-version", choices=["separable", "dispersion"], default="separable")
    ap.add_argument("--gate", action="store_true", help="Hyena-style content gate")
    ap.add_argument("--pole-param", choices=["softplus", "halflife"], default="softplus",
                    help="dispersion pole parameterization (LONG_HORIZON.md): softplus = v2 "
                         "default (init half-life ~0.7 frame); halflife = half-life in frames, "
                         "log-spread over [--hl-min, --hl-max] -- the minutes-scale memory arm")
    ap.add_argument("--hl-min", type=float, default=2.0, help="shortest pole half-life (frames)")
    ap.add_argument("--hl-max", type=float, default=4096.0, help="longest pole half-life (frames)")
    ap.add_argument("--local-fuse", action="store_true", help="parallel 3x3 depthwise local path")
    ap.add_argument("--linear-pad", action=argparse.BooleanOptionalAction, default=True,
                    help="zero-padded (linear) conv over time AND space (default ON for v2)")
    ap.add_argument("--rollout-loss", type=int, default=1, help="K-step rollout loss")
    ap.add_argument("--rollout-ramp", action="store_true",
                    help="ramp rollout K from 3->8 linearly over the second half of training")
    ap.add_argument("--motion-loss", action="store_true",
                    help="motion-balanced loss: 0.5*E[moving]+0.5*E[static]+0.25*MSE(pred-last,tgt-last) "
                         "-- removes copy-last as a valid optimum (DEEP_DIVE_astra sec 1)")
    ap.add_argument("--move-thresh", type=float, default=MOVE_THRESH,
                    help="per-channel |Δ| threshold for the moving-pixel mask")
    ap.add_argument("--resid-balanced", action="store_true",
                    help="DEEP_DIVE_2 Opus leak #1: restrict the 0.25 residual term of "
                         "the motion-balanced loss to moving pixels (grid-invariant) "
                         "instead of full-frame (needs --motion-loss; default off)")
    ap.add_argument("--motion-weighted", action="store_true",
                    help="DEEP_DIVE_2 Astra #1 soft variant: weight the motion term by "
                         "w=clamp(|Δ|/(mean|Δ|+eps),1,w_max) instead of a hard moving "
                         "mask (needs --motion-loss; default off)")
    ap.add_argument("--motion-w-max", type=float, default=10.0,
                    help="upper clamp on the --motion-weighted per-pixel weight")
    ap.add_argument("--radius", type=float, default=RADIUS,
                    help="ball Gaussian sigma in grid cells (default = data.RADIUS); "
                         "scale with grid for the DEEP_DIVE_2 geometry control")
    ap.add_argument("--speed", type=float, default=SPEED,
                    help="ball speed in cells/frame (default = data.SPEED); "
                         "scale with grid for the DEEP_DIVE_2 geometry control")
    ap.add_argument("--n-balls", type=int, default=N_BALLS,
                    help="number of balls (default: 3); ignored for --data-source waves")
    ap.add_argument("--kicks", action="store_true")
    ap.add_argument("--collisions", action=argparse.BooleanOptionalAction, default=False,
                    help="ball collisions; --no-collisions can override a runner recipe")
    ap.add_argument("--eval-rollout", type=int, default=256)
    ap.add_argument("--eval-seeds", type=int, default=32)
    ap.add_argument("--eval-chunk", type=int, default=4,
                    help="rollout-eval seed sub-batch size: forward at most this many "
                         "seeds at once to bound eval memory (DEEP_DIVE_2 OOM fix)")
    # v3: streaming + hardening (all default to no-op; existing behavior unchanged)
    ap.add_argument("--data-source", choices=["balls", "waves", "occlusion"], default="balls",
                    help="balls = data.py; waves = data_waves.py PDE fields; "
                         "occlusion = data_occlusion.py long-memory probe (R14)")
    ap.add_argument("--fuse", "--fusion", dest="fuse",
                    choices=["none", "local_wave", "local_ssm"], default="none",
                    help="gated local+global fusion arm (R14 / DEEP_DIVE_3 Rank 1): "
                         "h = g*h_local + (1-g)*h_global, g = sigmoid(Linear([h_local;h_global])). "
                         "local causal attention fused with the structured wave state "
                         "(local_wave, needs --kind wave --kernel-version dispersion) or a "
                         "generic diagonal SSM (local_ssm, needs --kind ssm). --fusion is an "
                         "alias for --fuse (DEEP_DIVE_3 names the flag --fusion).")
    ap.add_argument("--occ-start", type=int, default=_occ.OCC_START,
                    help="first frame the occlusion probe hides the target (data-source occlusion)")
    ap.add_argument("--train-occ-start", type=int, default=None,
                    help="occlusion window for TRAINING clips only (default: --occ-start). "
                         "Training clips start at frame 0, so with the default window "
                         "(64..320) and --frames 17 no clip ever contains the occluder; "
                         "set [T-gap, T) so the target frame is the first frame after the "
                         "gap and the loss requires bridging it (LONG_HORIZON.md)")
    ap.add_argument("--train-occ-end", type=int, default=None,
                    help="end (exclusive) of the training-clip occlusion window (default: --occ-end)")
    ap.add_argument("--occ-end", type=int, default=_occ.OCC_END,
                    help="first frame the target is visible again (exclusive upper bound)")
    ap.add_argument("--field", choices=["wave", "advection", "vortex"], default="wave",
                    help="PDE style when --data-source waves")
    ap.add_argument("--auto-batch", action="store_true",
                    help="on CUDA OOM, halve batch + retry (max 3), for train and eval")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint {state,opt,step} every N steps (0=off)")
    ap.add_argument("--resume", default="", help="resume from a checkpoint .pt")
    ap.add_argument("--stream-test", action="store_true",
                    help="run the long streaming rollout benchmark and exit (no training)")
    ap.add_argument("--stream-frames", type=int, default=1024)
    ap.add_argument("--stream-batch", type=int, default=8)
    # R7: latent space -- freeze a conv AE, train the predictor on encoded tokens,
    # predict the next latent, decode for pixel-space eval (same JSON fields).
    ap.add_argument("--latent", action="store_true",
                    help="train/predict in a frozen conv-AE latent space (needs --ae-ckpt)")
    ap.add_argument("--ae-ckpt", default="ckpts/ae.pt",
                    help="frozen ConvAE checkpoint from latent_ae.py --train")
    a = ap.parse_args()
    global DATA_SOURCE, WAVE_FIELD
    DATA_SOURCE, WAVE_FIELD = a.data_source, a.field
    if a.n_balls < 1:
        ap.error("--n-balls must be a positive integer")
    if a.fuse != "none":
        want = {"local_ssm": "ssm", "local_wave": "wave"}[a.fuse]
        if a.kind != want:
            ap.error(f"--fuse {a.fuse} requires --kind {want} (got {a.kind})")
        if a.fuse == "local_wave" and a.kernel_version != "dispersion":
            ap.error("--fuse local_wave requires --kernel-version dispersion (streamable state)")
        if a.latent:
            ap.error("--fuse hybrids are not wired through the latent path")
    if (a.q_mix or a.quat_color) and a.kind != "wave":
        ap.error("--q-mix / --quat-color apply to --kind wave "
                 "(HYPERCOMPLEX_STUDY.md sec 4-E2/E3)")
    if a.q_mix and a.fuse != "none":
        ap.error("--q-mix is not supported together with --fuse hybrids")
    if a.q_mix and a.dim % (4 * a.heads):
        ap.error("--q-mix needs --dim divisible by 4*--heads (quaternion pi/po slots)")
    if a.quat_color and a.dim % 4:
        ap.error("--quat-color needs --dim divisible by 4 (quaternion channels)")
    if a.quat_color and a.latent:
        ap.error("--quat-color is not wired through the latent path")
    if a.kind == "qssm" and a.dim % (2 * a.heads):
        ap.error("--kind qssm needs --dim divisible by 2*--heads (dhq = dh // 2)")
    if a.data_source == "occlusion":
        # Wire the (seedable, overridable) occlusion window into data_occlusion so
        # every generated clip -- train and eval -- shares one window definition.
        if not (0 <= a.occ_start <= a.occ_end):
            ap.error("require 0 <= --occ-start <= --occ-end")
        a.train_occ_start = a.occ_start if a.train_occ_start is None else a.train_occ_start
        a.train_occ_end = a.occ_end if a.train_occ_end is None else a.train_occ_end
        if not (0 <= a.train_occ_start <= a.train_occ_end):
            ap.error("require 0 <= --train-occ-start <= --train-occ-end")
        # Training clips use the training window; eval switches to --occ-start/--occ-end
        # right before the rollout (identical when the train flags are not given).
        _occ.OCC_START, _occ.OCC_END = a.train_occ_start, a.train_occ_end
        if a.frames > a.occ_start:
            print(f"OCCLUSION WARN: --frames {a.frames} > --occ-start {a.occ_start}; "
                  "context already overlaps the hidden window", flush=True)
    if a.latent and a.motion_loss:
        raise SystemExit("--motion-loss operates on a pixel-space moving mask; "
                         "it is not defined for --latent training (use plain latent MSE)")
    if (a.resid_balanced or a.motion_weighted) and not a.motion_loss:
        raise SystemExit("--resid-balanced / --motion-weighted modify the motion-balanced "
                         "objective; they require --motion-loss")
    if a.latent and a.stream_test:
        raise SystemExit("--stream-test uses the wave-only O(1) step() recurrence, "
                         "which is not threaded through the AE latent path")
    if a.latent and a.grid % DOWNSAMPLE != 0:
        raise SystemExit(f"--latent needs --grid divisible by {DOWNSAMPLE} (got {a.grid})")
    if a.latent and (a.drift_pert > 0 or a.hist_disc != "off"):
        raise SystemExit("--drift-pert / --hist-disc are pixel-space context "
                         "augmentations; they are not wired through the --latent path "
                         "(--var-reg works in latent space and is allowed)")
    if a.var_reg < 0 or a.drift_pert < 0:
        ap.error("--var-reg and --drift-pert must be >= 0")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    seed_off = a.seed * 1_000_000            # decorrelates training-clip seeds per run
    base = pathlib.Path(a.out)
    base.mkdir(parents=True, exist_ok=True)

    # ---- R7: frozen AE + latent geometry (no-op unless --latent) -----------
    ae = None
    lat_grid, cz = a.grid, 3
    if a.latent:
        ae = load_ae(a.ae_ckpt, map_location=dev).to(dev)
        cz = ae.latent_ch
        lat_grid = a.grid // DOWNSAMPLE
        print(f"LATENT ae={a.ae_ckpt} latent_ch={cz} pixel_grid={a.grid} "
              f"latent_grid={lat_grid}", flush=True)

    def build(ffn_mult):
        # In latent mode the predictor runs at the smaller latent grid; the wave/
        # attn/ssm stack is identical, only the I/O channels change (via LatentCore).
        vp = VideoPredictor(a.dim, a.layers, a.heads, a.frames, lat_grid, lat_grid, a.kind,
                            causal=a.causal, residual=a.residual, ffn_mult=ffn_mult,
                            kernel_version=a.kernel_version, linear_pad=a.linear_pad,
                            gate=a.gate, local_fuse=a.local_fuse, fuse=a.fuse,
                            q_mix=a.q_mix, quat_color=a.quat_color,
                            pole_param=a.pole_param, hl_min=a.hl_min, hl_max=a.hl_max)
        return LatentCore(vp, a.dim, cz) if a.latent else vp

    ffn_mult = a.ffn_mult
    if a.target_params > 0:
        ffn_mult, achieved = match_ffn_mult(
            lambda mm: sum(p.numel() for p in build(mm).parameters()), a.target_params)
        off = abs(achieved - a.target_params) / a.target_params
        print(f"PARAM-MATCH kind={a.kind} target={a.target_params} "
              f"ffn_mult={ffn_mult:.3f} achieved={achieved} off={off:.1%}"
              + ("  WARN>10%" if off > 0.10 else ""), flush=True)

    m = build(ffn_mult).to(dev)
    # pmodel = the trainable predictor whose weights we save/load/optimize; in
    # latent mode m becomes the pixel-in/out wrapper so all eval code is reused.
    if a.latent:
        core = m
        core.vp.use_ckpt = a.ckpt
        nparam = sum(p.numel() for p in core.parameters())
        ae_params = sum(p.numel() for p in ae.parameters())
        m = LatentWrapper(ae, core).to(dev)
        pmodel = core
    else:
        m.use_ckpt = a.ckpt
        nparam = sum(p.numel() for p in m.parameters())
        pmodel = m

    # ---- item (2): streaming benchmark -> write JSON and exit (no training) --
    if a.stream_test:
        if a.resume:
            m.load_state_dict(torch.load(a.resume, map_location=dev)["state"])
            print(f"RESUME weights <- {a.resume}", flush=True)
        m.eval()
        res = stream_test(m, a, dev)
        (base / f"stream_{a.kind}{a.tag}.json").write_text(json.dumps(res, indent=1))
        print("STREAM-DONE " + json.dumps(res), flush=True)
        return

    if a.no_decay_norm_head:
        _dec, _nodec = [], []
        for _n, _p in pmodel.named_parameters():
            if not _p.requires_grad:
                continue
            _is_nodec = _n.endswith("bias") or ("norm" in _n) or (_n.split(".")[-2] == "head")
            (_nodec if _is_nodec else _dec).append(_p)
        opt = torch.optim.AdamW([{"params": _dec, "weight_decay": 0.01},
                                 {"params": _nodec, "weight_decay": 0.0}], lr=a.lr)
    else:
        opt = torch.optim.AdamW([p for p in pmodel.parameters() if p.requires_grad],
                                lr=a.lr, weight_decay=0.01)
    if a.const_lr:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    # Rollout depth. --rollout-ramp holds K=3 through the first half then ramps
    # linearly to 8 over the second half (curriculum: short horizons first, then
    # push long-horizon stability once the one-step map is decent). We must always
    # generate enough frames for the *max* K reached, so gen_frames uses Kmax.
    RAMP_MIN, RAMP_MAX = 3, 8
    Kmax = RAMP_MAX if a.rollout_ramp else max(1, a.rollout_loss)
    gen_frames = a.frames + Kmax - 1

    def k_at(step):
        if not a.rollout_ramp:
            return max(1, a.rollout_loss)
        half = a.steps // 2
        if step <= half:
            return RAMP_MIN
        frac = min(1.0, (step - half) / max(1, a.steps - half))
        return int(round(RAMP_MIN + (RAMP_MAX - RAMP_MIN) * frac))

    # ---- item (4): resume {state, opt, step}; fast-forward the LR schedule ---
    start_step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=True)
        pmodel.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        for _ in range(start_step):          # re-align OneCycleLR (requires same --steps)
            sched.step()
        print(f"RESUME <- {a.resume} at step={start_step} (of {a.steps})", flush=True)

    ckpt_path = base / f"ckpt_{a.kind}{a.tag}.pt"
    cur_batch = a.batch                       # item (3): auto-batch shrinks this on OOM
    log = []
    var_reg_val = 0.0                         # last var-reg loss contribution (for step-row logging)
    t0 = time.time()
    for step in range(start_step + 1, a.steps + 1):
        K = k_at(step)                       # constant, or ramped 3->8 in the 2nd half
        for attempt in range(4):             # 1 try + up to 3 OOM retries
            try:
                gen = make_clip_batch(cur_batch, gen_frames, a.grid, a.grid, device=dev,
                                      seed=1000 + step + seed_off, kicks=a.kicks, collisions=a.collisions,
                                      radius=a.radius, speed=a.speed, nb=a.n_balls,
                                      return_moving=a.motion_loss, move_thresh=a.move_thresh)
                clips, moving = (gen if a.motion_loss else (gen, None))
                ctx = clips[:, :a.frames].clone()
                # Anti-collapse CONTEXT augmentations (train only; eval stays clean
                # so cross-arm metrics remain comparable). Both are gated so the
                # default path is byte-identical -- see helper docstrings. drift is
                # applied first (continuous field), then discretized: the model sees
                # the drifted-then-coarsened history a real streaming decoder would.
                if a.drift_pert > 0:
                    ctx = apply_drift_pert(ctx, a.drift_pert, seed=2_000_000 + step + seed_off)
                if a.hist_disc == "bits":
                    ctx = quantize_frames(ctx, HIST_DISC_LEVELS)
                with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda" and not a.fp32)):
                    if a.latent:
                        # Encode the whole clip once with the frozen AE (no grad),
                        # then run the SAME K-step rollout loss but in latent space:
                        # predict next latent, MSE against the encoded target latent,
                        # feed the predicted latent back (detached). Gradients flow
                        # through the predictor only; the AE stays frozen.
                        with torch.no_grad():
                            z = m.encode_clip(clips).float()           # [B, gen+1, cz, h, w]
                        zwin = z[:, :a.frames]
                        loss = 0.0
                        var_term = 0.0
                        for k in range(K):
                            zpred = core(zwin).float()
                            lk = F.mse_loss(zpred, z[:, a.frames + k])
                            loss = loss + (1.0 if k == 0 else 0.25) * lk
                            if a.var_reg > 0:  # variance-match in latent space (sec 2)
                                pd = zpred - zwin[:, -1]
                                gd = z[:, a.frames + k] - z[:, a.frames + k - 1]
                                var_term = var_term + (1.0 if k == 0 else 0.25) * torch.relu(gd.std() - pd.std())
                            if k < K - 1:
                                zwin = torch.cat([zwin[:, 1:], zpred.detach().unsqueeze(1)], 1)
                        if a.var_reg > 0:
                            loss = loss + a.var_reg * var_term
                            var_reg_val = float((a.var_reg * var_term).detach())
                    else:
                        win = ctx
                        loss = 0.0
                        var_term = 0.0
                        for k in range(K):
                            pred = m(win).float()
                            tgt_k = clips[:, a.frames + k]
                            if a.motion_loss:
                                last_k = clips[:, a.frames + k - 1]        # GT persistence baseline
                                lk = motion_balanced_loss(pred, tgt_k, last_k,
                                                          moving[:, a.frames + k - 1],
                                                          resid_balanced=a.resid_balanced,
                                                          motion_weighted=a.motion_weighted,
                                                          w_max=a.motion_w_max)
                            else:
                                lk = F.mse_loss(pred, tgt_k)
                            loss = loss + (1.0 if k == 0 else 0.25) * lk
                            if a.var_reg > 0:
                                # Per-step delta = motion the model produced (pred vs
                                # its own last input frame) vs GT motion (clean next vs
                                # clean prev). SIGN: relu(std(gt)-std(pred)) fires when
                                # the model UNDER-moves = the freeze/collapse we fight.
                                pd = pred - win[:, -1].float()
                                gd = (tgt_k - clips[:, a.frames + k - 1]).float()
                                var_term = var_term + (1.0 if k == 0 else 0.25) * torch.relu(gd.std() - pd.std())
                            if k < K - 1:  # feed own prediction, detached (own-forward gradients only)
                                win = torch.cat([win[:, 1:], pred.detach().unsqueeze(1)], 1)
                        if a.var_reg > 0:
                            loss = loss + a.var_reg * var_term
                            var_reg_val = float((a.var_reg * var_term).detach())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.telemetry_every and step % a.telemetry_every == 0:
                    with torch.no_grad():
                        _hw = next((_p.grad.norm().item() for _n, _p in pmodel.named_parameters()
                                    if _n.split(".")[-2] == "head" and _p.grad is not None), float("nan"))
                        _tcr = ((pred.float() - ctx[:, -1]).norm() / ((tgt_k.float() - ctx[:, -1]).norm() + 1e-9)).item()
                        # Gate diagnostic (fused arms only): mean g per layer; g->1
                        # means the persistent global (wave/ssm) state is dead weight.
                        _gm = pmodel.gate_means() if hasattr(pmodel, "gate_means") else []
                    _gtxt = (" gate_g=[" + ",".join(f"{v:.3f}" for v in _gm) + "]") if _gm else ""
                    print(f"TELE step={step} lr={sched.get_last_lr()[0]:.2e} "
                          f"head_gn={_hw:.4e} copy_r={_tcr:.4f}{_gtxt}", flush=True)
                torch.nn.utils.clip_grad_norm_(pmodel.parameters(), 1.0)
                opt.step()
                sched.step()
                break
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                if not (a.auto_batch and dev == "cuda" and oom) or attempt == 3:
                    raise
                opt.zero_grad(set_to_none=True)
                clips = ctx = loss = None
                torch.cuda.empty_cache()
                nb = max(1, cur_batch // 2)
                print(f"AUTO-BATCH OOM step={step} attempt={attempt} batch {cur_batch}->{nb}",
                      flush=True)
                cur_batch = nb
        if a.save_every > 0 and step % a.save_every == 0:
            torch.save({"state": pmodel.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)
        if step % 25 == 0 or step == start_step + 1:
            row = {"step": step, "K": K, "loss": round(float(loss), 5),
                   "st_s": round((step - start_step) / (time.time() - t0), 2)}
            if a.var_reg > 0:                # anti-collapse hinge contribution this step
                row["var_reg"] = round(var_reg_val, 6)
            log.append(row)
            print(json.dumps(row), flush=True)
    if a.save_every > 0:                      # final checkpoint at the end of training
        torch.save({"state": pmodel.state_dict(), "opt": opt.state_dict(), "step": a.steps}, ckpt_path)
    train_sec = time.time() - t0
    train_sps = round(a.steps / train_sec, 2)

    # ---- single-step eval + baselines (item 3: OOM-hardened) ---------------
    m.eval()
    with torch.no_grad():
        se = zb = cb = vb = cr = 0.0
        eb = 64                               # eval sub-batch, halved on OOM
        for i in range(a.eval_batches):
            for attempt in range(4):
                try:
                    clips = make_clip_batch(eb, a.frames, a.grid, a.grid, device=dev,
                                            seed=90000 + i, kicks=a.kicks, collisions=a.collisions,
                                            radius=a.radius, speed=a.speed, nb=a.n_balls)
                    ctx, tgt = clips[:, :a.frames], clips[:, a.frames]
                    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda" and not a.fp32)):
                        pred = m(ctx)
                    p, last = pred.float(), ctx[:, -1]
                    se += F.mse_loss(p, tgt).item()
                    # copy_ratio = ||pred-last|| / ||target-last||: how far the model
                    # moved vs how far it should have (~0 collapsed to copy-last,
                    # ~1 correct magnitude, >>1 unstable). DEEP_DIVE_astra sec 1.
                    cr += ((p - last).norm() / ((tgt - last).norm() + 1e-9)).item()
                    z, c, v = baseline_mses(ctx, tgt)
                    zb += z; cb += c; vb += v
                    break
                except RuntimeError as e:
                    if not (a.auto_batch and dev == "cuda" and "out of memory" in str(e).lower()) or attempt == 3:
                        raise
                    clips = ctx = tgt = pred = None
                    torch.cuda.empty_cache()
                    eb = max(1, eb // 2)
                    print(f"AUTO-BATCH OOM eval i={i} attempt={attempt} -> eb={eb}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()      # between eval passes
        n = a.eval_batches
        eval_mse = se / n
        copy_ratio = cr / n
        baselines = {"zero": round(zb / n, 5), "copy_last": round(cb / n, 5),
                     "const_vel": round(vb / n, 5)}
        if dev == "cuda":
            torch.cuda.empty_cache()
        roll_fn = occlusion_rollout_eval if a.data_source == "occlusion" else rollout_eval
        if a.data_source == "occlusion":
            _occ.OCC_START, _occ.OCC_END = a.occ_start, a.occ_end
        for attempt in range(4):              # rollout eval also OOM-hardened
            try:
                roll = roll_fn(m, a, dev)
                break
            except RuntimeError as e:
                if not (a.auto_batch and dev == "cuda" and "out of memory" in str(e).lower()) or attempt == 3:
                    raise
                torch.cuda.empty_cache()
                # Shrink the seed sub-batch first (bounds memory, keeps all seeds ->
                # metric unchanged); only drop seeds once chunk is already 1.
                if a.eval_chunk > 1:
                    a.eval_chunk = max(1, a.eval_chunk // 2)
                    print(f"AUTO-BATCH OOM rollout attempt={attempt} -> eval_chunk={a.eval_chunk}", flush=True)
                else:
                    a.eval_seeds = max(1, a.eval_seeds // 2)
                    print(f"AUTO-BATCH OOM rollout attempt={attempt} -> eval_seeds={a.eval_seeds}", flush=True)

    # STATE BYTES -- the R14 killer metric. Persistent recurrent state is
    # horizon-independent; a finite-context arm must instead keep a whole T-frame
    # window. Both are reported at batch 1 so arms are directly comparable.
    psb = pmodel.persistent_state_bytes(1, "cpu") if hasattr(pmodel, "persistent_state_bytes") else 0
    keeps_window = (a.fuse != "none") or (a.kind == "attn")
    window_bytes = a.frames * lat_grid * lat_grid * a.dim * 4 if keeps_window else 0

    res = {"kind": a.kind, "fuse": a.fuse, "params": nparam, "ffn_mult": round(ffn_mult, 3),
           "steps": a.steps, "dim": a.dim, "layers": a.layers, "heads": a.heads,
           "grid": a.grid, "frames": a.frames, "batch": a.batch,
           "causal": a.causal, "residual": a.residual, "kernel_version": a.kernel_version,
           "q_mix": a.q_mix, "quat_color": a.quat_color,
           "pole_param": a.pole_param, "hl_min": a.hl_min, "hl_max": a.hl_max,
           "const_lr": a.const_lr, "seed": a.seed, "fp32": a.fp32,
           "no_decay_norm_head": a.no_decay_norm_head, "telemetry_every": a.telemetry_every,
           "gate": a.gate, "local_fuse": a.local_fuse, "linear_pad": a.linear_pad,
           "occ_start": a.occ_start, "occ_end": a.occ_end,
           "train_occ_start": getattr(a, "train_occ_start", None),
           "train_occ_end": getattr(a, "train_occ_end", None),
           "persistent_state_bytes": psb, "window_bytes": window_bytes,
           "state_bytes_total": psb + window_bytes,
           "rollout_loss_K": Kmax, "rollout_ramp": a.rollout_ramp,
           "var_reg": a.var_reg, "drift_pert": a.drift_pert, "hist_disc": a.hist_disc,
           "motion_loss": a.motion_loss, "move_thresh": a.move_thresh,
           "resid_balanced": a.resid_balanced, "motion_weighted": a.motion_weighted,
           "motion_w_max": a.motion_w_max if a.motion_weighted else None,
           "radius": a.radius, "speed": a.speed, "n_balls": a.n_balls, "eval_chunk": a.eval_chunk,
           "data_source": a.data_source, "field": a.field,
           "kicks": a.kicks, "collisions": a.collisions,
           "latent": a.latent,
           "latent_ch": cz if a.latent else None,
           "latent_grid": lat_grid if a.latent else None,
           "ae_ckpt": a.ae_ckpt if a.latent else None,
           "ae_params": ae_params if a.latent else None,
           "train_sec": round(train_sec, 1), "train_steps_s": train_sps,
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "eval_mse": round(eval_mse, 5), "baselines": baselines,
           "eval_mse_over_copylast": round(eval_mse / (baselines["copy_last"] + 1e-9), 3),
           "copy_ratio": round(copy_ratio, 3),
           **roll, "log_tail": log[-8:]}
    (base / f"result_{a.kind}{a.tag}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": pmodel.state_dict()}, base / f"model_{a.kind}{a.tag}.pt")

    # ---- printed comparison table (item 2) ---------------------------------
    b = baselines
    print("\n=== RESULT TABLE (MSE lower=better; r>1 beats persistence) ===")
    print(f"  arm={a.kind}{a.tag}  params={nparam}  ffn_mult={ffn_mult:.3f}")
    print(f"  baseline  zero       = {b['zero']:.5f}")
    print(f"  baseline  copy-last  = {b['copy_last']:.5f}")
    print(f"  baseline  const-vel  = {b['const_vel']:.5f}")
    print(f"  model     eval_mse   = {eval_mse:.5f}   (model/copy-last = {res['eval_mse_over_copylast']})")
    print(f"  model     copy_ratio = {copy_ratio:.3f}   "
          f"(~0 collapsed | ~1 correct motion | >>1 unstable)")
    if a.data_source == "occlusion":
        dh = float(roll["divergence_horizon"]) / max(1, psb) * 1000.0  # frames per KB of state
        print(f"  occlusion path={'stream' if roll['streaming'] else 'window'}  "
              f"div_horizon={roll['divergence_horizon']}/{roll['eval_rollout']}  "
              f"r=persist/model={roll['r_persist_over_model']}")
        print(f"  emergence exit_dir_acc={roll['exit_direction_accuracy']}  "
              f"pos_err={roll['position_error_at_emergence']}  "
              f"vel_err={roll['velocity_error_at_emergence']}  "
              f"id_survival={roll['target_identity_survival']}")
        print(f"  state_bytes persistent={psb} window={window_bytes}  "
              f"motion_pres={roll['motion_preservation']}  "
              f"div_horizon/KB_state={dh:.2f}  ms/frame={roll['ms_per_frame']}")
    else:
        print(f"  rollout   r=persist/model = {roll['r_persist_over_model']}  "
              f"div_horizon={roll['divergence_horizon']}/{roll['eval_rollout']}  "
              f"id_survival={roll['identity_survival']}")
        print("  " + semantic_summary(roll["semantic"]))
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
