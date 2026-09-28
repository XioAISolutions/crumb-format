#!/usr/bin/env python3
r"""Render a train_compare checkpoint without training or teacher forcing.

    python render_rollout.py --ckpt runs_v2_box/model_wave_w1.pt \
        --kind wave --kernel-version dispersion --grid 16 --frames 256 \
        --side-by-side --out /tmp/brainsnn-demo

Dependencies: torch, Pillow; ffmpeg on PATH, or imageio + imageio-ffmpeg.
--frames counts NEW predictions, not the checkpoint's context length. A matching
result_<tag>.json is discovered beside model_<tag>.pt / ckpt_<tag>.pt (also one
directory up for checkpoints in ckpt/). Otherwise provide --config JSON containing
the training arguments, or a checkpoint with a plain-dict config/args member.
Legacy results omit data_source/field: balls/wave are explicit, reported defaults;
use --data-source waves --field ... for those runs. No architecture resizing.

Auto mode uses recurrence only for ungated dispersion wave models without local
fusion. Other models use a fixed context window. CPU ground truth and metric
curves grow with the requested horizon; the constant-state claim is ONLY about
recurrent model inference at fixed grid, model size, precision and batch size.
Existing nonempty output directories are refused. No checkpoint is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import math


def ball_options(config):
    """Restore scene geometry from result metadata; legacy runs retain defaults."""
    from data import N_BALLS, RADIUS, SPEED
    config.setdefault("n_balls", N_BALLS)
    config.setdefault("radius", RADIUS)
    config.setdefault("speed", SPEED)
    if type(config["n_balls"]) is not int or config["n_balls"] < 1:
        raise ValueError("config n_balls must be a positive integer")
    for name in ("radius", "speed"):
        value = config[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (name == "radius" and value == 0)):
            raise ValueError(f"config {name} must be finite and {'positive' if name == 'radius' else 'nonnegative'}")
    return {"nb": config["n_balls"], "radius": config["radius"], "speed": config["speed"]}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint file or unambiguous run/ckpt directory")
    ap.add_argument("--kind", choices=("wave", "attn", "ssm"), help="assert checkpoint model kind")
    ap.add_argument("--kernel-version", choices=("separable", "dispersion"), help="assert trained kernel")
    ap.add_argument("--grid", type=positive_int, help="assert trained square grid (no resizing)")
    ap.add_argument("--frames", type=positive_int, default=256, help="number of generated frames (default: 256)")
    ap.add_argument("--out", required=True, type=Path, help="new or empty artifact directory")
    ap.add_argument("--config", type=Path, help="exact training config or matching result JSON")
    ap.add_argument("--mode", choices=("auto", "recurrent", "windowed"), default="auto")
    ap.add_argument("--data-source", choices=("balls", "waves"))
    ap.add_argument("--field", choices=("wave", "advection", "vortex"))
    ap.add_argument("--latent", action="store_true",
                    help="render a train_compare --latent checkpoint: roll the predictor "
                         "autoregressively at the latent grid, decode every predicted latent "
                         "through the frozen AE, and score in PIXEL space (needs --ae-ckpt)")
    ap.add_argument("--ae-ckpt", type=Path,
                    help="frozen ConvAE checkpoint (latent_ae.py --train); with --latent, "
                         "defaults to the ae_ckpt recorded in the predictor's result JSON")
    ap.add_argument("--seed", type=int, default=170001, help="procedural clip seed (default: 170001)")
    ap.add_argument("--side-by-side", action="store_true", help="also save aligned truth and comparison PNGs/MP4s")
    ap.add_argument("--fps", type=positive_int, default=24, help="playback FPS, independent of measured model FPS")
    ap.add_argument("--scale", type=positive_int, default=8, help="nearest-neighbor display scale; metrics stay at native grid")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return ap


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path):
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt"))
        # A training run commonly has both a resumable and weights-only save.
        resumes = [p for p in candidates if p.name.startswith("ckpt_")]
        if len(resumes) == 1 and all(p.stem.removeprefix("model_").removeprefix("ckpt_") ==
                                     resumes[0].stem.removeprefix("ckpt_") for p in candidates):
            candidates = resumes
        if len(candidates) != 1:
            raise ValueError(f"{path}: found {len(candidates)} checkpoints; pass an exact --ckpt file")
        path = candidates[0]
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    return path


def load_model(args, torch):
    from wfvideo import VideoPredictor

    checkpoint = resolve_checkpoint(args.ckpt)
    # Never fall back to unrestricted pickle loading.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = saved.get("state", saved.get("state_dict", saved))
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected train_compare {'state': state_dict} checkpoint")
    source = None
    config = None
    if args.config:
        source = args.config.expanduser().resolve()
    else:
        for name in ("config", "args"):
            if isinstance(saved.get(name), dict):
                config = dict(saved[name])
                source = f"checkpoint:{name}"
                break
        if config is None:
            tag = checkpoint.stem.removeprefix("model_").removeprefix("ckpt_")
            roots = [checkpoint.parent]
            if checkpoint.parent.name in ("ckpt", "checkpoints"):
                roots.append(checkpoint.parent.parent)
            for root in roots:
                matched = root / f"result_{tag}.json"
                if matched.is_file():
                    source = matched
                    break
            if source is None:
                matches = [root / "config.json" for root in roots if (root / "config.json").is_file()]
                if len(matches) == 1:
                    source = matches[0]
    if isinstance(source, Path):
        config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("no matching result/config JSON; supply --config with exact training arguments")
    config = dict(config)
    notes = []
    legacy = (config.get("architecture") == "v1" or
              ("drift" in config and "kernel_version" not in config and "posemb.pt" not in state))
    config["architecture"] = "v1" if legacy else "v2"
    for name in ("kind", "kernel_version", "grid"):
        explicit = getattr(args, name)
        if explicit is not None:
            if name in config and config[name] != explicit:
                raise ValueError(f"--{name.replace('_', '-')}={explicit} conflicts with config {config[name]!r}")
            config[name] = explicit
    missing = [name for name in ("kind", "dim", "layers", "heads", "frames", "grid") if name not in config]
    if missing:
        raise ValueError("training config missing: " + ", ".join(missing))
    for name in ("dim", "layers", "heads", "frames", "grid"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"config {name} must be a positive integer")
    if config["kind"] not in ("wave", "attn", "ssm") or config["dim"] % config["heads"]:
        raise ValueError("invalid kind or dim not divisible by heads")
    defaults = {"kernel_version": "separable", "causal": False, "residual": not legacy,
                "linear_pad": not legacy, "gate": False, "local_fuse": False,
                "kicks": False, "collisions": False, "data_source": "balls", "field": "wave"}
    for name, value in defaults.items():
        explicit = getattr(args, name, None) if name in ("data_source", "field") else None
        if explicit is not None:
            if name in config and config[name] != explicit:
                notes.append(f"Evaluation {name} override: {config[name]} -> {explicit}; not a training-data assertion.")
            config[name] = explicit
        elif name not in config:
            config[name] = value
            notes.append(f"Missing training metadata {name}; using {value!r}.")
    for name in ("causal", "residual", "linear_pad", "gate", "local_fuse", "kicks", "collisions"):
        if type(config[name]) is not bool:
            raise ValueError(f"config {name} must be a JSON boolean")
    if config["data_source"] == "occlusion":        # train_long.py: balls with a masked gap
        notes.append("Trained on occlusion clips; the renderer's truth is unoccluded balls.")
    ball_options(config)
    if config["kernel_version"] not in ("separable", "dispersion"):
        raise ValueError("invalid kernel_version")
    if (config["data_source"] not in ("balls", "waves", "occlusion")
            or config["field"] not in ("wave", "advection", "vortex")):
        raise ValueError("invalid data_source/field")
    # train_compare rounds matched ffn_mult to 3 decimals in result JSON. At large
    # dimensions that can reconstruct a different hidden width. Trust tensor size.
    key = "blocks.0.ffn.fc1.weight"
    if key not in state:
        raise ValueError("unsupported checkpoint architecture: missing blocks.0.ffn.fc1.weight")
    hidden, dim = state[key].shape
    if dim != config["dim"]:
        raise ValueError("checkpoint FFN width does not match config dim")
    config["ffn_mult_recorded"] = config.get("ffn_mult")
    config["ffn_mult"] = hidden / dim
    config["ffn_hidden"] = hidden
    dimensions = (config["dim"], config["layers"], config["heads"], config["frames"],
                  config["grid"], config["grid"], config["kind"])
    if legacy:
        from v1_backup.wfvideo import VideoPredictor as LegacyPredictor
        if (config["kernel_version"] != "separable" or config["residual"] or config["linear_pad"]
                or config["gate"] or config["local_fuse"] or hidden != 4 * dim):
            raise ValueError("v1 checkpoint requires its original separable, non-residual, 4x-FFN configuration")
        model = LegacyPredictor(*dimensions, causal=config["causal"], posemb="posemb.pt" in state)
        notes.append("Legacy v1 architecture loaded from v1_backup/wfvideo.py; current synthetic generator may differ from training.")
    else:
        time_pos = config.get("time_pos", "table")
        if time_pos not in ("table", "none"):
            raise ValueError("invalid time_pos")
        if time_pos == "table" and "posemb.pt" not in state:
            raise ValueError("checkpoint has no v2 positional embeddings; use the matching legacy result or explicit architecture='v1' config")
        # train_long.py (LONG_HORIZON.md phase 1) options; absent keys keep the v2 defaults.
        extra = {k: config[k] for k in ("pole_param", "hl_min", "hl_max", "write_gate", "clean_write", "head", "flow_steps")
                 if config.get(k) is not None}
        model = VideoPredictor(*dimensions, time_pos=time_pos, **extra,
                               **{k: config[k] for k in ("causal", "residual", "ffn_mult", "kernel_version",
                                                        "linear_pad", "gate", "local_fuse")})
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint, str(source), notes


def make_truth(config, count, seed, torch):
    """Generate CPU reference; never feed its future frames to the predictor."""
    grid = config["grid"]
    if config["data_source"] in ("balls", "occlusion"):          # occlusion: unmasked balls
        from data import make_clip_batch
        return make_clip_batch(1, count - 1, grid, grid, seed=seed,
                               kicks=config["kicks"], collisions=config["collisions"],
                               **ball_options(config)), "data.make_clip_batch"
    import data_waves
    if count <= 65:
        return data_waves.make_clip_batch(1, count - 1, grid, grid, seed=seed,
                                         field=config["field"]), "data_waves.make_clip_batch"
    # The public clip sampler caps T at 64; its exact spectral evolution does not.
    # Decode a seeded t=0 frame using its ORIGINAL scale, then evolve the same PDE.
    # Float32 encoding/decoding introduces small roundoff, disclosed in metadata.
    initial, meta = data_waves.make_clip_batch(1, 0, grid, grid, seed=seed,
                                              field=config["field"], return_meta=True)
    raw = (initial[:, 0].double() * 2 - 1) * meta["scale"][:, :, None, None]
    evolved = data_waves.spectral_evolve(raw[:, 0], count - 1, field=meta["field"],
                                        time_step=meta["time_step"], speed=meta["speed"],
                                        velocity=meta["velocity"], viscosity=meta["viscosity"],
                                        initial_dt=raw[:, 1] if meta["field"] == "wave" else None)
    clip = (0.5 + evolved / (2 * meta["scale"][:, None, :, None, None])).clamp(0, 1).float()
    return clip, "data_waves.spectral_evolve from decoded float32 seed; original fixed channel scales (roundoff differs from sampler)"


def encode_video(folder, destination, fps, count):
    """Read PNGs incrementally; use imageio if external ffmpeg is unavailable/fails."""
    error = "ffmpeg not found on PATH"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                   "-start_number", "0", "-i", str(folder / "%06d.png"), "-frames:v", str(count),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode == 0 and destination.is_file() and destination.stat().st_size:
                return "ffmpeg/libx264"
            error = completed.stderr.strip()[-1600:] or f"ffmpeg exit {completed.returncode}"
        except OSError as exc:
            error = str(exc)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(destination), format="FFMPEG", mode="I", fps=fps,
                                codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"]) as writer:
            for index in range(count):
                writer.append_data(imageio.imread(folder / f"{index:06d}.png"))
        return "imageio/ffmpeg/libx264"
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"MP4 encoding failed ({error}); imageio fallback: {exc}. "
                           "Install ffmpeg, or imageio and imageio-ffmpeg. PNGs are preserved.") from exc


def discover_config(checkpoint, args):
    """Locate the training config for a checkpoint: --config, an embedded
    config/args dict, or a sibling result_<tag>.json / config.json (also one
    directory up for ckpt/ layouts). Returns (config_dict, source)."""
    if args.config:
        source = args.config.expanduser().resolve()
        return json.loads(source.read_text(encoding="utf-8")), source
    tag = checkpoint.stem.removeprefix("model_").removeprefix("ckpt_")
    roots = [checkpoint.parent]
    if checkpoint.parent.name in ("ckpt", "checkpoints"):
        roots.append(checkpoint.parent.parent)
    for root in roots:
        matched = root / f"result_{tag}.json"
        if matched.is_file():
            return json.loads(matched.read_text(encoding="utf-8")), matched
    matches = [root / "config.json" for root in roots if (root / "config.json").is_file()]
    if len(matches) == 1:
        return json.loads(matches[0].read_text(encoding="utf-8")), matches[0]
    return None, None


def load_latent_model(args, torch):
    """Rebuild a train_compare ``--latent`` predictor (a ``LatentCore`` wrapping a
    ``VideoPredictor`` at the latent grid) plus its frozen ``ConvAE``.

    The predictor runs at ``grid/DOWNSAMPLE`` with cz-channel latent I/O. The
    exact FFN width is recovered from the tensor (result JSON rounds ffn_mult),
    so ``load_state_dict(strict=True)`` matches the trained 1.6M-param arms."""
    from wfvideo import VideoPredictor
    from latent_ae import load_ae, DOWNSAMPLE
    from train_compare import LatentCore

    checkpoint = resolve_checkpoint(args.ckpt)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = saved.get("state", saved.get("state_dict", saved))
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected train_compare {'state': state_dict} checkpoint")
    if not any(k.startswith("vp.") for k in state):
        raise ValueError("not a --latent checkpoint: state has no 'vp.*' keys "
                         "(LatentCore wraps a VideoPredictor under .vp); omit --latent for pixel models")

    config = None
    source = None
    for name in ("config", "args"):
        if isinstance(saved.get(name), dict):
            config, source = dict(saved[name]), f"checkpoint:{name}"
            break
    if config is None:
        config, source = discover_config(checkpoint, args)
    if not isinstance(config, dict):
        raise ValueError("no matching result/config JSON; supply --config with the latent training arguments")
    config = dict(config)
    if not config.get("latent"):
        raise ValueError("config is not a --latent run (latent!=true); render this checkpoint without --latent")

    notes = []
    for name in ("kind", "dim", "layers", "heads", "frames", "grid", "latent_ch"):
        if name not in config:
            raise ValueError(f"latent training config missing: {name}")
    grid = config["grid"]
    if grid % DOWNSAMPLE != 0:
        raise ValueError(f"latent grid {grid} not divisible by AE downsample {DOWNSAMPLE}")
    lat_grid = grid // DOWNSAMPLE
    if config.get("latent_grid") not in (None, lat_grid):
        raise ValueError(f"config latent_grid {config['latent_grid']} != grid/{DOWNSAMPLE}={lat_grid}")
    for name in ("kind", "grid"):  # honour render-time assertions like the pixel path
        explicit = getattr(args, name)
        if explicit is not None and config[name] != explicit:
            raise ValueError(f"--{name}={explicit} conflicts with config {config[name]!r}")
    if args.kernel_version is not None and config.get("kernel_version") != args.kernel_version:
        raise ValueError(f"--kernel-version={args.kernel_version} conflicts with config {config.get('kernel_version')!r}")

    defaults = {"kernel_version": "separable", "causal": False, "residual": True,
                "linear_pad": True, "gate": False, "local_fuse": False,
                "kicks": False, "collisions": False, "data_source": "balls", "field": "wave"}
    for name, value in defaults.items():
        explicit = getattr(args, name, None) if name in ("data_source", "field") else None
        if explicit is not None:
            if name in config and config[name] != explicit:
                notes.append(f"Evaluation {name} override: {config[name]} -> {explicit}; not a training-data assertion.")
            config[name] = explicit
        elif name not in config:
            config[name] = value
            notes.append(f"Missing training metadata {name}; using {value!r}.")
    if config["kind"] not in ("wave", "attn", "ssm") or config["dim"] % config["heads"]:
        raise ValueError("invalid kind or dim not divisible by heads")
    if config["data_source"] not in ("balls", "waves") or config["field"] not in ("wave", "advection", "vortex"):
        raise ValueError("invalid data_source/field")

    ball_options(config)

    key = "vp.blocks.0.ffn.fc1.weight"
    if key not in state:
        raise ValueError("unsupported latent checkpoint: missing vp.blocks.0.ffn.fc1.weight")
    hidden, dim = state[key].shape
    if dim != config["dim"]:
        raise ValueError("checkpoint FFN width does not match config dim")
    config["ffn_mult_recorded"] = config.get("ffn_mult")
    config["ffn_mult"] = hidden / dim
    config["ffn_hidden"] = hidden
    cz = config["latent_ch"]

    vp = VideoPredictor(config["dim"], config["layers"], config["heads"], config["frames"],
                        lat_grid, lat_grid, config["kind"],
                        causal=config["causal"], residual=config["residual"],
                        ffn_mult=config["ffn_mult"], kernel_version=config["kernel_version"],
                        linear_pad=config["linear_pad"], gate=config["gate"], local_fuse=config["local_fuse"])
    core = LatentCore(vp, config["dim"], cz)
    core.load_state_dict(state, strict=True)
    core.eval()
    if int(vp.posemb.pt.shape[0]) != config["frames"]:
        raise ValueError("checkpoint temporal positions do not match config frames")

    ae_path = args.ae_ckpt
    if ae_path is None:
        if not config.get("ae_ckpt"):
            raise ValueError("no --ae-ckpt given and config carries no ae_ckpt path")
        ae_path = Path(config["ae_ckpt"])
        notes.append(f"--ae-ckpt not given; using ae_ckpt from config: {ae_path}")
    ae_path = ae_path.expanduser()
    if not ae_path.is_file():
        raise ValueError(f"AE checkpoint does not exist: {ae_path}")
    ae = load_ae(str(ae_path))
    if ae.latent_ch != cz:
        raise ValueError(f"AE latent_ch {ae.latent_ch} != predictor latent_ch {cz}: mismatched AE")
    return core, ae, config, checkpoint, source, str(ae_path), notes


def make_latent_truth(config, count, seed, torch):
    """Pixel ground-truth clip + optional per-ball colors/positions for centroids.

    Balls (data.py) support arbitrary horizons directly, so we regenerate with
    ``return_meta`` to get colors/positions for the pixel-space centroid metric.
    Waves reuse ``make_truth`` (incl. its >64-frame spectral evolution) and carry
    no centroids -- exactly as train_compare's rollout_eval does for waves."""
    grid = config["grid"]
    if config["data_source"] in ("balls", "occlusion"):          # occlusion: unmasked balls
        from data import make_clip_batch
        clip, meta = make_clip_batch(1, count - 1, grid, grid, seed=seed,
                                     kicks=config["kicks"], collisions=config["collisions"],
                                     return_meta=True, **ball_options(config))
        return clip, meta["col"], meta["pos"], "data.make_clip_batch (+meta for centroids)"
    clip, method = make_truth(config, count, seed, torch)
    return clip, None, None, method


def render_latent(args):
    """Autoregressive latent rollout, scored in pixel space.

    Encode the seed context once, then repeatedly: predict the next latent with
    the frozen predictor, decode it to pixels for scoring/PNGs, and feed the
    predicted latent back at the latent grid. Metrics (mse, copy_last_mse,
    centroid error, divergence) are all computed on decoded pixels vs the pixel
    ground-truth clip -- so they honestly include the AE reconstruction error."""
    import torch
    from PIL import Image, ImageDraw
    from latent_ae import DOWNSAMPLE
    from train_compare import centroids_by_color, divergence_horizon, DIV_CONSEC
    from semantic_metrics import semantic_frame_metrics, aggregate_semantic_metrics, semantic_summary

    out = args.out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output must be new or empty: {out}")
    core, ae, config, checkpoint, source, ae_source, notes = load_latent_model(args, torch)

    if args.mode == "recurrent":
        raise ValueError("recurrent mode is not defined for --latent (the O(1) step() recurrence "
                         "is not threaded through the AE); latent rollout is windowed")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    core = core.float().to(device)
    ae = ae.float().to(device)
    grid = config["grid"]
    lat_grid = grid // DOWNSAMPLE
    context_frames = config["frames"]

    clip, cols, pos, truth_method = make_latent_truth(config, context_frames + args.frames, args.seed, torch)
    clip = clip.to(device)
    if cols is not None:
        cols, pos = cols.to(device), pos.to(device)
    context = clip[:, :context_frames].clone()
    fixed_last = clip[:, context_frames - 1].clone()          # frozen copy-last baseline frame

    def encode_clip(frames):                                  # [1,T,3,H,W] -> [1,T,cz,h,w]
        B, T, C, H, W = frames.shape
        z = ae.encode(frames.reshape(B * T, C, H, W))
        return z.reshape(B, T, *z.shape[1:])

    out.mkdir(parents=True, exist_ok=True)
    folders = {"prediction": out / "prediction"}
    if args.side_by_side:
        folders.update(ground_truth=out / "ground_truth", comparison=out / "comparison")
    for folder in folders.values():
        folder.mkdir()
    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)
    print(f"Rendering {args.frames} latent frames: windowed, {device}, "
          f"pixel {grid}x{grid} / latent {lat_grid}x{lat_grid}; {checkpoint.name}", flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def frame_image(tensor):
        pixels = tensor[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        return Image.fromarray(pixels).resize((grid * args.scale,) * 2, Image.Resampling.NEAREST)

    curves, baseline_curve, cerr_curve = [], [], []
    semantic_samples = []
    div_thresh = config["radius"]
    id_tol = 2.0 * div_thresh
    model_seconds = 0.0
    warmup_seconds = 0.0
    latent_bytes = None
    start = time.perf_counter()
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync()
        warmup_start = time.perf_counter()
        z_win = encode_clip(context)                          # [1,T,cz,h,w]
        latent_bytes = z_win.numel() * z_win.element_size()
        sync()
        warmup_seconds = time.perf_counter() - warmup_start
        for index in range(args.frames):
            sync()
            tick = time.perf_counter()
            z_next = core(z_win)                              # [1,cz,h,w] predicted next latent
            prediction = ae.decode(z_next).float().clamp(0, 1)  # [1,3,H,W]
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"non-finite prediction at generated frame {index}; partial PNGs preserved")
            z_win = torch.cat((z_win[:, 1:], z_next[:, None]), dim=1)   # feed latent back
            sync()
            model_seconds += time.perf_counter() - tick
            pred_cpu = prediction.cpu()
            target = clip[:, context_frames + index]
            curves.append(float((prediction - target).square().mean()))
            baseline_curve.append(float((fixed_last - target).square().mean()))
            if cols is not None:
                pc = centroids_by_color(prediction, cols)     # [1,nb,2]
                gc = pos[:, context_frames + index]           # [1,nb,2]
                cerr_curve.append((pc - gc).norm(dim=-1))     # [1,nb]
                semantic_samples.append(semantic_frame_metrics(
                    pred_cpu, gc, cols, radius=config["radius"]))
            image = frame_image(pred_cpu)
            image.save(folders["prediction"] / f"{index:06d}.png")
            if args.side_by_side:
                truth = frame_image(target.cpu())
                truth.save(folders["ground_truth"] / f"{index:06d}.png")
                pair = Image.new("RGB", (image.width * 2, image.height + 28), "#05070b")
                pair.paste(image, (0, 28))
                pair.paste(truth, (image.width, 28))
                draw = ImageDraw.Draw(pair)
                draw.text((4, 8), "Latent prediction", fill="#68eaff")
                draw.text((image.width + 4, 8), "Ground truth", fill="#947cff")
                pair.save(folders["comparison"] / f"{index:06d}.png")
            if (index + 1) % 32 == 0 or index == args.frames - 1:
                print(f"  {index + 1}/{args.frames} frames", flush=True)

    mean_mse = sum(curves) / len(curves)
    mean_baseline = sum(baseline_curve) / len(baseline_curve)
    semantic = aggregate_semantic_metrics(semantic_samples, radius=config["radius"]) if cols is not None else None
    centroid = {"mean_centroid_err": None, "final_centroid_err": None,
                "identity_survival": None, "divergence_horizon": None,
                "centroid_err_curve": None}
    if cerr_curve:
        cerr = torch.cat([c.reshape(1, -1) for c in cerr_curve], 0)   # [R, nb]
        median_cerr = cerr.median(dim=1).values                       # [R]
        centroid = {
            "mean_centroid_err": round(float(cerr.mean()), 3),
            "final_centroid_err": round(float(cerr[-1].mean()), 3),
            "identity_survival": round(float((cerr[-1] < id_tol).float().mean()), 3),
            "divergence_horizon": divergence_horizon(median_cerr.cpu(), div_thresh, DIV_CONSEC),
            "centroid_err_curve": [round(float(x), 3) for x in cerr.mean(dim=1)],
        }

    ae_report = {}
    ae_json = Path(ae_source).with_suffix(".json")
    if ae_json.is_file():
        try:
            ae_report = {"ae_eval_recon_mse": json.loads(ae_json.read_text()).get("eval_recon_mse")}
        except (ValueError, OSError):
            ae_report = {}

    claim = ("Latent rollout: the predictor is rolled autoregressively at the latent grid and EVERY "
             "predicted latent is decoded through the frozen AE for pixel-space scoring, so the reported "
             "MSE/centroid/divergence INCLUDE the AE reconstruction floor (see ae_eval_recon_mse). "
             "Single synthetic clip; no photorealism or long-horizon coherence guarantee. "
             "CPU reference clips, metric logs and exported media grow with horizon.")
    configuration = {key: config.get(key) for key in
                     ("kind", "kernel_version", "grid", "dim", "layers", "heads", "ffn_mult",
                      "ffn_mult_recorded", "ffn_hidden", "causal", "residual", "linear_pad",
                      "gate", "local_fuse", "data_source", "field", "kicks", "collisions",
                      "n_balls", "radius", "speed")}
    configuration.update({"context_frames": context_frames, "latent": True,
                          "latent_ch": config["latent_ch"], "latent_grid": lat_grid,
                          "ae_ckpt": ae_source, "ae_params": sum(p.numel() for p in ae.parameters()),
                          "predictor_params": sum(p.numel() for p in core.parameters())})
    report = {
        "schema_version": 1, "status": "encoding", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": configuration,
        "rollout": {"frames": args.frames, "seed": args.seed, "mode": "windowed", "batch_size": 1,
                    "device": device, "dtype": "float32", "fps": args.fps, "scale": args.scale,
                    "first_target_index": context_frames, "teacher_forcing": False,
                    "latent_feedback": True, "truth_generation": truth_method,
                    "temporal_embedding": "reset window positions"},
        "metrics": {"mse": mean_mse, "final_mse": curves[-1], "copy_last_mse": mean_baseline,
                    "copy_last_over_model": mean_baseline / mean_mse if mean_mse > 0 else None,
                    "model_fps": args.frames / model_seconds if model_seconds > 0 else None,
                    "model_seconds": model_seconds, "warmup_seconds": warmup_seconds,
                    "div_thresh_px": round(div_thresh, 3), "div_consec": DIV_CONSEC,
                    "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None,
                    "latent_window_bytes": latent_bytes,
                    "mse_curve": curves, "copy_last_mse_curve": baseline_curve,
                    **centroid, **ae_report, "semantic": semantic},
        "metric_notes": {"mse": "Pixel-space MSE of the DECODED predicted latent vs the pixel ground truth, "
                                "native grid float32 [0,1] after clamping, before PNG/MP4 quantization; includes AE recon error.",
                         "copy_last": "Open-loop baseline: frozen last seed frame (pixels) at EVERY horizon; no future truth fed back.",
                         "semantic": "Detected RGB blobs matched to GT positions; counts and matched-only error must be read together. "
                                     "Zero matches gives null error, never zero; pixel grid units, before display scaling; balls only.",
                         "centroid": "Color-matched per-ball centroid error (grid cells) on decoded pixels; balls only. "
                                     "divergence_horizon = first frame with median centroid err > div_thresh_px for div_consec consecutive frames.",
                         "model_fps": "Batch 1, synchronized core forward + AE decode + latent feedback; excludes encode warmup, "
                                      "reference generation, CPU transfer, metrics, PNG/MP4 encoding. Not serving throughput."},
        "files": {"prediction_video": None, "ground_truth_video": None, "comparison_video": None,
                  "png_dirs": {key: folder.name for key, folder in folders.items()}},
        "provenance": {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "ae_checkpoint": ae_source, "ae_sha256": sha256_file(Path(ae_source)),
                       "config_source": str(source) if source is not None else None,
                       "config_sha256": sha256_file(Path(str(source))) if source is not None and Path(str(source)).is_file() else None,
                       "torch_version": torch.__version__, "python_version": platform.python_version(),
                       "source_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
                                         ("render_rollout.py", "wfvideo.py", "data.py", "data_waves.py",
                                          "ssm_lite.py", "latent_ae.py", "train_compare.py", "semantic_metrics.py")}},
        "claim_boundary": claim, "notes": notes,
    }

    def save_report():
        (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    try:
        report["encoders"] = {}
        for name, folder in folders.items():
            destination = out / f"{name}.mp4"
            report["encoders"][name] = encode_video(folder, destination, args.fps, args.frames)
            report["files"][f"{name}_video"] = destination.name
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "encoding_failed"
        report["encoding_error"] = str(exc)
        raise
    finally:
        report["render_seconds"] = time.perf_counter() - start
        save_report()
    print(f"Saved {out / 'prediction.mp4'} and metrics.json; pixel MSE={mean_mse:.6g}, "
          f"copy-last={mean_baseline:.6g}, div_horizon={centroid['divergence_horizon']}; "
          + semantic_summary(semantic))
    return report


def render(args):
    if args.latent:
        return render_latent(args)
    import torch
    from PIL import Image, ImageDraw
    from semantic_metrics import semantic_frame_metrics, aggregate_semantic_metrics, semantic_summary

    out = args.out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output must be new or empty: {out}")
    model, config, checkpoint, source, notes = load_model(args, torch)
    # train_long.py checkpoints (time_pos='none', wave or SSM) were trained through
    # their carried state, so they render through stream_step too.
    eligible = (config["kind"] == "wave" and config["kernel_version"] == "dispersion" and not (
        config["gate"] or config["local_fuse"])) or config.get("time_pos") == "none"
    mode = ("recurrent" if eligible else "windowed") if args.mode == "auto" else args.mode
    if mode == "recurrent" and not eligible:
        raise ValueError("recurrent mode requires wave + dispersion with gate=false and local_fuse=false, "
                         "or a train_long.py (time_pos='none') checkpoint")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = model.float().to(device)
    context_frames = config["frames"]
    clip, cols, pos, truth_method = make_latent_truth(config, context_frames + args.frames, args.seed, torch)
    context = clip[:, :context_frames].clone().to(device)
    context_bytes = context.numel() * context.element_size()
    fixed_last = clip[:, context_frames - 1].clone()
    out.mkdir(parents=True, exist_ok=True)
    folders = {"prediction": out / "prediction"}
    if args.side_by_side:
        folders.update(ground_truth=out / "ground_truth", comparison=out / "comparison")
    for folder in folders.values():
        folder.mkdir()
    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)
    print(f"Rendering {args.frames} frames: {mode}, {device}, {config['grid']}x{config['grid']}; {checkpoint.name}", flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def frame_image(tensor):
        pixels = tensor[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
        return Image.fromarray(pixels).resize((config["grid"] * args.scale,) * 2, Image.Resampling.NEAREST)

    curves, baseline_curve, state_samples = [], [], []
    semantic_samples = []
    model_seconds = 0.0
    warmup_seconds = 0.0
    state_bytes = None
    start = time.perf_counter()
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        states = None
        if mode == "recurrent":
            sync()
            warmup_start = time.perf_counter()
            states = model.stream_init(1, device)
            # After ingesting the LAST context frame, prediction already is the
            # FIRST future frame. Do not call step again before saving frame 0.
            for index in range(context_frames - 1):
                _, states = model.stream_step(context[:, index], states, index)
            current = context[:, -1].clone()
            state_bytes = sum(s.numel() * s.element_size() for s in states)
            del context
            sync()
            warmup_seconds = time.perf_counter() - warmup_start
        for index in range(args.frames):
            sync()
            tick = time.perf_counter()
            if mode == "recurrent":
                prediction, states = model.stream_step(current, states, context_frames - 1 + index)
            else:
                prediction = model(context)
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"non-finite prediction at generated frame {index}; partial PNGs preserved")
            prediction = prediction.float().clamp(0, 1)
            if mode == "recurrent":
                current = prediction
            else:
                context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
            sync()
            model_seconds += time.perf_counter() - tick
            pred_cpu = prediction.cpu()
            target = clip[:, context_frames + index]
            curves.append(float((pred_cpu - target).square().mean()))
            baseline_curve.append(float((fixed_last - target).square().mean()))
            if cols is not None:
                semantic_samples.append(semantic_frame_metrics(
                    pred_cpu, pos[:, context_frames + index], cols, radius=config["radius"]))
            if index == 0 or (index + 1) % 32 == 0 or index == args.frames - 1:
                state_samples.append({"frame": index + 1,
                                      "state_bytes": sum(s.numel() * s.element_size() for s in states) if states is not None else None,
                                      "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None})
            image = frame_image(pred_cpu)
            image.save(folders["prediction"] / f"{index:06d}.png")
            if args.side_by_side:
                truth = frame_image(target)
                truth.save(folders["ground_truth"] / f"{index:06d}.png")
                pair = Image.new("RGB", (image.width * 2, image.height + 28), "#05070b")
                pair.paste(image, (0, 28))
                pair.paste(truth, (image.width, 28))
                draw = ImageDraw.Draw(pair)
                draw.text((4, 8), "Prediction", fill="#68eaff")
                draw.text((image.width + 4, 8), "Ground truth", fill="#947cff")
                pair.save(folders["comparison"] / f"{index:06d}.png")
            if (index + 1) % 32 == 0 or index == args.frames - 1:
                print(f"  {index + 1}/{args.frames} frames", flush=True)
    mean_mse = sum(curves) / len(curves)
    mean_baseline = sum(baseline_curve) / len(baseline_curve)
    semantic = aggregate_semantic_metrics(semantic_samples, radius=config["radius"]) if cols is not None else None
    claim = ("Research target: long-horizon coherence at constant recurrent inference state. "
             "State size is independent of generated horizon at fixed model, grid, precision and batch; "
             "this single synthetic clip does not establish coherence or photorealism. "
             "CPU reference clips, metric logs and exported media grow with horizon.")
    if mode == "windowed":
        claim = ("This run recomputes a fixed context window; it is not evidence for recurrent streaming. "
                 "Single synthetic clip, no photorealism or long-horizon coherence guarantee. "
                 "CPU reference clips, metric logs and exported media grow with horizon.")
    configuration = {key: config[key] for key in ("architecture", "kind", "kernel_version", "grid", "dim", "layers", "heads",
                     "ffn_mult", "ffn_mult_recorded", "ffn_hidden", "causal", "residual", "linear_pad",
                     "gate", "local_fuse", "data_source", "field", "kicks", "collisions",
                     "n_balls", "radius", "speed")}
    configuration["context_frames"] = context_frames
    report = {
        "schema_version": 1, "status": "encoding", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": configuration,
        "rollout": {"frames": args.frames, "seed": args.seed, "mode": mode, "batch_size": 1,
                    "device": device, "dtype": "float32", "fps": args.fps, "scale": args.scale,
                    "first_target_index": context_frames, "teacher_forcing": False,
                    "truth_generation": truth_method, "temporal_embedding": "clamp at context_frames-1" if mode == "recurrent" else "reset window positions"},
        "metrics": {"mse": mean_mse, "final_mse": curves[-1], "copy_last_mse": mean_baseline,
                    "copy_last_over_model": mean_baseline / mean_mse if mean_mse > 0 else None,
                    "model_fps": args.frames / model_seconds, "model_seconds": model_seconds,
                    "warmup_seconds": warmup_seconds,
                    "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None,
                    "state_bytes": state_bytes, "context_bytes": context_bytes,
                    "mse_curve": curves, "copy_last_mse_curve": baseline_curve,
                    "state_samples": state_samples, "semantic": semantic},
        "metric_notes": {"mse": "Native-grid float32 [0,1] after output clamping, before PNG/video quantization; lower is better.",
                         "copy_last": "Open-loop baseline: frozen last seed frame at EVERY horizon; no future truth is fed back.",
                         "semantic": "Detected RGB blobs matched to GT positions; counts and matched-only error must be read together. "
                                     "Zero matches gives null error, never zero; pixel grid units, before display scaling; balls only.",
                         "model_fps": "Batch 1, synchronized model calls plus clamp/finite check/feedback; excludes warmup, reference generation, CPU transfer, metrics, PNG/MP4 encoding. Not serving throughput.",
                         "memory": "State tensor bytes only, excludes weights/activations/allocator; context_bytes is input seed size (retained window only in windowed mode). CUDA peak covers model warmup and rollout, not CPU RAM; null on CPU."},
        "files": {"prediction_video": None, "ground_truth_video": None, "comparison_video": None,
                  "png_dirs": {key: folder.name for key, folder in folders.items()}},
        "provenance": {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "config_source": source, "config_sha256": sha256_file(Path(source)) if Path(source).is_file() else None,
                       "torch_version": torch.__version__, "python_version": platform.python_version(),
                       "source_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
                                         ("render_rollout.py", "wfvideo.py", "data.py", "data_waves.py", "ssm_lite.py", "semantic_metrics.py") +
                                         (("v1_backup/wfvideo.py",) if config["architecture"] == "v1" else ())}},
        "claim_boundary": claim, "notes": notes,
    }

    def save_report():
        (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    try:
        report["encoders"] = {}
        for name, folder in folders.items():
            destination = out / f"{name}.mp4"
            report["encoders"][name] = encode_video(folder, destination, args.fps, args.frames)
            report["files"][f"{name}_video"] = destination.name
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "encoding_failed"
        report["encoding_error"] = str(exc)
        raise
    finally:
        report["render_seconds"] = time.perf_counter() - start
        save_report()
    print(f"Saved {out / 'prediction.mp4'} and metrics.json; MSE={mean_mse:.6g}, "
          f"model FPS={args.frames / model_seconds:.2f}; " + semantic_summary(semantic))
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        render(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        print(f"render_rollout: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
