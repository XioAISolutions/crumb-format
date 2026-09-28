"""Encode a folder of videos into latent shards once, offline (LONG_HORIZON.md phase 2).

Training on latents should not pay the VAE encoder every step, so each video is
decoded, resized/center-cropped to --height x --width, resampled to --fps, cut
into segments of --max-frames (snapped to the VAE's 1 + t*k grid), encoded, and
saved as one shard per segment:

    OUT/shard_00000.pt = {"latents": fp16 [Tl, C, h, w], "src": path,
                          "start_frame": int, "n_frames": int, "fps": float,
                          "vae": VideoVAE.describe()}
    OUT/index.json     = list of the above minus the tensor

train_long.py --latents OUT samples windows from these shards.

    python encode_videos.py --videos clips/ --vae ltx --height 256 --width 448 --out latents/
    python encode_videos.py --synthetic 4 --videos /tmp/syn --vae ltx-tiny --out /tmp/lat   # smoke
"""
import argparse
import json
import pathlib

import torch
import torch.nn.functional as F

from video_vae import VideoVAE, valid_frames

EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def read_video(path, fps=None):
    """-> (frames [T,3,H,W] float in [0,1], fps actually used)."""
    import imageio.v2 as iio
    rd = iio.get_reader(str(path), "ffmpeg")
    src_fps = float(rd.get_meta_data().get("fps", 24.0))
    step = max(1, round(src_fps / fps)) if fps else 1
    frames = [torch.from_numpy(f).permute(2, 0, 1) for i, f in enumerate(rd) if i % step == 0]
    rd.close()
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    return torch.stack(frames).float() / 255.0, src_fps / step


def fit(frames, height, width):
    """Resize the short side to cover (height, width), then center-crop."""
    T, C, H, W = frames.shape
    s = max(height / H, width / W)
    nh, nw = max(height, round(H * s)), max(width, round(W * s))
    x = F.interpolate(frames, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    y0, x0 = (nh - height) // 2, (nw - width) // 2
    return x[:, :, y0:y0 + height, x0:x0 + width]


def write_synthetic(out_dir, n, frames=65, size=64, fps=24):
    """Moving-ball mp4s (data.py) for smoke tests of the whole pipeline."""
    import imageio.v2 as iio
    from data import make_clip_batch
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips = make_clip_batch(n, frames - 1, 16, 16, seed=123)            # [n, frames, 3, 16, 16]
    clips = F.interpolate(clips.flatten(0, 1), size=(size, size), mode="nearest")
    clips = clips.view(n, frames, 3, size, size)
    paths = []
    for i in range(n):
        p = out / f"synthetic_{i:03d}.mp4"
        arr = (clips[i].permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().numpy()
        iio.mimwrite(str(p), list(arr), fps=fps, macro_block_size=1)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder of videos (searched recursively)")
    ap.add_argument("--vae", default="ltx", help="ltx | wan | conv | ltx-tiny | wan-tiny")
    ap.add_argument("--vae-path", default=None, help="local weights dir / HF id / conv ckpt")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--max-frames", type=int, default=257, help="frames per shard (snapped to 1+t*k)")
    ap.add_argument("--synthetic", type=int, default=0, help="first write N synthetic mp4s into --videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.synthetic:
        write_synthetic(a.videos, a.synthetic, size=max(a.height, a.width))
    vae = VideoVAE(a.vae, path=a.vae_path, device=a.device)
    seg = valid_frames(a.max_frames, vae.t_stride)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in pathlib.Path(a.videos).rglob("*") if p.suffix.lower() in EXTS)
    if not vids:
        raise SystemExit(f"no videos under {a.videos}")
    index, k = [], 0
    for v in vids:
        frames, fps = read_video(v, a.fps)
        frames = fit(frames, a.height, a.width)
        for s0 in range(0, frames.shape[0], seg):
            n = valid_frames(min(seg, frames.shape[0] - s0), vae.t_stride)
            if n < 1 + vae.t_stride:                 # too short to give 2 latent steps
                continue
            z = vae.encode(frames[None, s0:s0 + n])[0]
            meta = {"src": str(v), "start_frame": s0, "n_frames": n, "fps": fps,
                    "latent_steps": z.shape[0], "latent_shape": list(z.shape[1:]),
                    "vae": vae.describe(), "file": f"shard_{k:05d}.pt"}
            torch.save({"latents": z.half(), **meta}, out / meta["file"])
            index.append(meta)
            k += 1
        print(f"ENCODED {v} -> {k} shards so far", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"DONE {len(index)} shards, {sum(m['latent_steps'] for m in index)} latent steps", flush=True)
    return index


if __name__ == "__main__":
    main()
