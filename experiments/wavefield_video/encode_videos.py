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
import hashlib
import json
import os
import pathlib

import torch
import torch.nn.functional as F

from video_vae import VideoVAE, valid_frames

EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def iter_segments(path, fps, height, width, seg, skip=0):
    """Stream a video: decode one frame at a time, drop frames to ~fps, resize +
    crop each frame immediately, and yield (start_frame, [n,3,H,W] float) segments
    of up to ``seg`` frames. Memory is one segment at the target size, however long
    or high-resolution the source is (a full 5-min 1080p decode would be ~180 GB).
    ``skip``: segments already encoded (resume); their frames are decoded and
    counted but not resized or yielded."""
    import imageio.v2 as iio
    rd = iio.get_reader(str(path), "ffmpeg")
    try:
        src_fps = float(rd.get_meta_data().get("fps", 24.0))
        # Keep the first source frame at or after each target timestamp k/fps, so
        # 30 -> 24 fps drops 1 frame in 5 (an integer skip would keep all 30).
        out_fps = min(fps, src_fps) if fps else src_fps
        buf, start, kept = [], 0, 0
        for i, f in enumerate(rd):
            if i / src_fps < kept / out_fps - 1e-6 * (1 / src_fps):
                continue
            if kept >= skip * seg:
                x = torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255.0
                buf.append(fit(x, height, width)[0])
            kept += 1
            if kept % seg == 0:
                if buf:
                    yield start, torch.stack(buf), out_fps
                start, buf = kept, []
        if buf:
            yield start, torch.stack(buf), out_fps
    finally:
        rd.close()


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
    root = pathlib.Path(a.videos)
    # skip hidden paths: macOS AppleDouble "._clip.mp4" sidecars (copied along by
    # tar/rsync/zip from a Mac) have video suffixes but are not videos, and crash ffmpeg
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not vids:
        raise SystemExit(f"no videos under {a.videos}")
    # Resumable: every encoded segment is journaled (progress.jsonl) with its shard,
    # and every finished video gets a "done" line, so a run killed by a slice
    # budget continues at the next unencoded segment -- inside a long video too.
    # A segment killed mid-encode is redone (its shard number is reused).
    # the corpus is part of the encode's identity: a resume with another --videos root
    # or changed files is refused instead of appending a second corpus
    root = pathlib.Path(a.videos).resolve()
    vids = [p.resolve() for p in vids]
    key = {p: p.relative_to(root).as_posix() for p in vids}   # canonical source key

    def digest(p):                                   # content, not just size / mtime
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()[:16]
    manifest = [[key[p], p.stat().st_size, digest(p)] for p in vids]
    cfg = {"vae": vae.describe(), "height": a.height, "width": a.width, "fps": a.fps, "seg": seg,
           "videos": str(root), "manifest": manifest}
    cfg_path, journal = out / "progress_config.json", out / "progress.jsonl"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise SystemExit(f"{out} holds a partial encode with different settings; delete it or "
                         "use another --out")
    if not cfg_path.exists():                        # atomic, and never rewritten
        cfg_path.with_suffix(".tmp").write_text(json.dumps(cfg))
        os.replace(cfg_path.with_suffix(".tmp"), cfg_path)
    lines = []
    if journal.exists():
        for x in journal.read_text().splitlines():
            try:
                lines.append(json.loads(x))
            except json.JSONDecodeError:              # a line cut by the slice kill: redo it
                break
        # rewrite without the torn tail so new lines never follow a broken one
        journal.write_text("".join(json.dumps(d) + "\n" for d in lines))
    finished = {d["src"] for d in lines if d.get("done")}
    segs_done = {}
    for d in lines:
        if "seg" in d:
            segs_done[d["src"]] = segs_done.get(d["src"], 0) + 1
    index = [d["shard"] for d in lines if d.get("shard")]
    k = len(index)
    if lines:
        print(f"RESUME {len(finished)} videos done, {k} shards already encoded", flush=True)

    def log(entry):
        with journal.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    for v in vids:
        if key[v] in finished:
            continue
        skip = segs_done.get(key[v], 0)
        for j, (s0, frames, fps) in enumerate(iter_segments(v, a.fps, a.height, a.width, seg,
                                                            skip=skip), start=skip):
            n = valid_frames(frames.shape[0], vae.t_stride)
            if n < 1 + vae.t_stride:                 # too short to give 2 latent steps
                log({"src": key[v], "seg": j, "shard": None})
                continue
            z = vae.encode(frames[None, :n])[0].cpu()          # shards load on any host
            zh = z.half()
            meta = {"src": key[v], "start_frame": s0, "n_frames": n, "fps": fps,
                    # content digest: the dataset fingerprint (index.json) changes when
                    # latents do, even at the same paths and lengths
                    "sha256": hashlib.sha256(zh.contiguous().view(torch.uint8).numpy()
                                             .tobytes()).hexdigest()[:16],
                    "latent_steps": z.shape[0], "latent_shape": list(z.shape[1:]),
                    "vae": vae.describe(), "file": f"shard_{k:05d}.pt"}
            torch.save({"latents": zh, **meta}, out / (meta["file"] + ".tmp"))
            os.replace(out / (meta["file"] + ".tmp"), out / meta["file"])
            log({"src": key[v], "seg": j, "shard": meta})
            index.append(meta)
            k += 1
        log({"src": key[v], "done": True})
        print(f"ENCODED {v} -> {k} shards so far", flush=True)
    tmp = out / "index.json.tmp"                     # atomic: index.json means "complete"
    tmp.write_text(json.dumps(index, indent=1))
    os.replace(tmp, out / "index.json")
    print(f"DONE {len(index)} shards, {sum(m['latent_steps'] for m in index)} latent steps", flush=True)
    return index


if __name__ == "__main__":
    main()
