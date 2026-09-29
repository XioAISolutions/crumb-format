"""Minutes of video on one RTX 4090 with LongLive 2.0 (LONGLIVE_4090.md).

LongLive (NVlabs, arXiv 2509.22622) is the strongest open recipe for streaming
long video: a causal Wan2.2-TI2V-5B with a rolling KV window plus a frame sink.
Its release inference script cannot make a 2-5 minute clip on a 24 GB card as
shipped. This wrapper removes each blocker without patching LongLive:

  1. RoPE table. The temporal RoPE table has 1024 rows indexed by absolute
     latent frame. At 24 fps with 4x temporal compression that is 4093 frames,
     2.84 minutes, and the absolute path slices past the table after it. The
     overlay turns on LongLive's own ``use_relative_rope`` (window-relative
     positions; the cache holds un-rotated keys), which has no length limit.
  2. Host RAM. The release path keeps the whole decoded clip as float32 on the
     CPU, then concatenates, rescales and copies it. At 704x1280 that is
     10.8 MB per frame and ~78 GB per copy for 5 minutes: several copies at the
     peak. The overlay saves latents only (1800 x 48 x 44 x 80 bf16 = 0.6 GB for
     5 minutes), and ``decode`` below streams them through the VAE with a
     carried causal cache straight into an mp4, holding one chunk at a time.
  3. streaming_vae. With ``vae_type: wan`` the release pipeline calls
     ``WanVAE_.cached_decode``, which the Wan2.2 VAE module does not define
     (only the LightVAE wrapper does). ``stream_decode`` implements that cached
     decode for the Wan2.2 VAE; it matches ``WanVAE_.decode`` frame for frame.
  4. VRAM. bf16 weights (9.3 GiB) plus the 32-frame KV cache (9.7 GiB) plus
     activations come to ~22 GiB by ``plan``'s estimate: under 2 GiB of
     headroom on a 24 GB card, before allocator fragmentation. FP8 weights
     (TorchAO rowwise; the 4090's sm_89 has FP8) cut ~4 GiB, and a smaller
     ``--window`` cuts the KV cache linearly, so fp8 is the default.
     ``generate`` refuses a config whose estimate is within 1 GiB of the card
     unless forced. LongLive offloads the 11 GB text encoder itself
     (DynamicSwap) whenever free VRAM is under 40 GB.

    python longlive_long.py plan --minutes 5
    python longlive_long.py generate --ll-root ~/LongLive --ckpt .../model_bf16.pt \
        --prompts prompts.txt --minutes 5 --out runs_ll/five_min
    python longlive_long.py decode --ll-root ~/LongLive --latents runs_ll/five_min/latents/rank0-0-0_regular.pt \
        --out runs_ll/five_min/rank0-0.mp4

``generate`` decodes every latent file it produces unless --no-decode is given;
both steps skip work whose output already exists, so a sliced box job resumes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FPS = 24
T_STRIDE = 4                      # Wan2.2 VAE temporal compression
S_STRIDE = 16                     # spatial compression incl. the VAE patchify
LATENT_HW = (44, 80)              # 704 x 1280
LATENT_CH = 48
BLOCK = 8                         # latent frames per autoregressive block
ROPE_ROWS = 1024
LAYERS, HEADS, HEAD_DIM = 30, 24, 128
PATCH_TOKENS = LATENT_HW[0] * LATENT_HW[1] // 4       # frame_seq_length = 880
GEN_PARAMS = 5.0e9
GIB = 1024 ** 3


# ---------------------------------------------------------------- planning
def pixel_frames(latent_frames):
    return 1 + T_STRIDE * (latent_frames - 1)


def latent_frames_for(seconds, fps=FPS):
    """Smallest block-aligned latent frame count covering ``seconds``."""
    need = max(1, int(round(seconds * fps)))
    lat = 1 + -(-(need - 1) // T_STRIDE)
    return -(-lat // BLOCK) * BLOCK


def vram_estimate(precision="fp8", window=32, sink=8, cfg=False):
    """Rough steady-state generator VRAM in GiB. Weights: bf16 2 B/param; fp8
    keeps ~10% of parameters (norms, embeddings, head) in bf16. KV: the cache is
    allocated at local_attn_size frames (the sink lives inside it)."""
    w = GEN_PARAMS * (2.0 if precision == "bf16" else 0.9 * 1 + 0.1 * 2)
    kv = LAYERS * 2 * window * PATCH_TOKENS * HEADS * HEAD_DIM * 2 * (2 if cfg else 1)
    act = BLOCK * PATCH_TOKENS * HEADS * HEAD_DIM * 2 * 40        # ~40 live block-sized tensors
    other = 1.5 * GIB                                             # CUDA context, text-encoder swap slot
    parts = dict(weights=w / GIB, kv_cache=kv / GIB, activations=act / GIB, other=other / GIB)
    parts["total"] = sum(parts.values())
    return parts


def plan(minutes, precision="fp8", window=32, sink=8):
    lat = latent_frames_for(minutes * 60)
    px = pixel_frames(lat)
    H, W = LATENT_HW[0] * S_STRIDE, LATENT_HW[1] * S_STRIDE
    return dict(
        minutes=minutes, latent_frames=lat, pixel_frames=px, seconds=px / FPS,
        blocks=lat // BLOCK, resolution=f"{W}x{H}",
        needs_relative_rope=lat > ROPE_ROWS,
        latent_gb=lat * LATENT_CH * LATENT_HW[0] * LATENT_HW[1] * 2 / GIB,
        release_decode_copy_gb=px * 3 * H * W * 4 / GIB,
        vram=vram_estimate(precision, window, sink),
    )


# ---------------------------------------------------------------- overlay
def build_overlay(base, *, latent_frames, prompts, ckpt, out_dir, window=32, sink=8,
                  seed=0, fp8=True, compile=False):
    """Return a copy of a LongLive inference config (a plain dict, as loaded by
    yaml) set up for one long latents-only run. Keys go where the release code
    reads them: grouped sections are flattened by ``normalize_config``, so a key
    set in its section and at top level must agree."""
    cfg = json.loads(json.dumps(base))                  # deep copy of plain data
    for key in ("num_output_frames",):
        cfg[key] = int(latent_frames)
    shape = list(cfg.setdefault("data", {}).get("image_or_video_shape", [1, BLOCK, LATENT_CH, *LATENT_HW]))
    shape[1] = int(latent_frames)
    cfg["data"]["image_or_video_shape"] = shape
    cfg["data"]["data_path"] = str(prompts)
    cfg.setdefault("model_kwargs", {})["local_attn_size"] = int(window)
    inf = cfg.setdefault("inference", {})
    inf["sink_size"] = int(sink)
    inf["streaming_vae"] = False                        # see module doc, blocker 3
    inf["async_vae"] = False
    inf["save_latents_only"] = True                     # blocker 2
    inf.pop("vae_device", None)
    cfg["use_relative_rope"] = True                     # blocker 1
    cfg["output_folder"] = str(Path(out_dir) / "latents")
    cfg["save_with_index"] = True
    cfg["num_samples"] = 1
    cfg.setdefault("checkpoints", {})["generator_ckpt"] = str(ckpt)
    cfg.setdefault("logging", {})["seed"] = int(seed)
    cfg["fp8_quant"] = bool(fp8)
    # LongLive validated torch.compile on one 8-frame block; a long run adds KV
    # shapes while the window fills (its README warns of recompiles), so eager
    # is the default and --compile opts back in to the config's own setting.
    if not compile:
        cfg["torch_compile"] = False
    if fp8:
        cfg.pop("model_quant", None)
    return cfg


def latent_name(idx):
    """File LongLive writes for prompt ``idx`` with save_with_index (no LoRA, no EMA)."""
    return f"rank0-{idx}-0_regular.pt"


def n_prompts_of(prompts):
    """Samples LongLive's MultiTextConcatDataset yields: non-empty lines of a txt
    file, or caption subfolders of a directory."""
    prompts = Path(prompts)
    if prompts.is_file():
        return sum(1 for line in prompts.read_text(encoding="utf-8").splitlines() if line.strip())
    cap = prompts / "caption" if (prompts / "caption").is_dir() else prompts
    return sum(1 for d in cap.iterdir() if d.is_dir())


def expected_stems(prompts):
    return [Path(latent_name(i)).stem for i in range(n_prompts_of(prompts))]


def _file_digest(path, block=8 << 20):
    """size + sha256 of every byte (streamed). ~30 s for the 10 GB checkpoint: once
    per generate call, cheap next to the generation it guards."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return f"{path.stat().st_size}:{h.hexdigest()[:16]}"


def _prompts_digest(prompts):
    prompts = Path(prompts)
    h = hashlib.sha256()
    files = [prompts] if prompts.is_file() else sorted(p for p in prompts.rglob("*") if p.is_file())
    for f in files:
        h.update(str(f.relative_to(prompts) if f != prompts else f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _git_head(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def vae_weights_path(ll_root, vae_path=None):
    return Path(vae_path) if vae_path else Path(ll_root) / "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"


_SRC_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".cu", ".cuh", ".cpp", ".c", ".h", ".txt", ".cfg", ".toml"}
_SRC_SKIP_DIRS = {".git", "__pycache__", "wan_models", "videos", "LongLive-2.0-5B", "outputs"}


def _source_digest(root, exclude=(), max_file=8 << 20):
    """sha256 over the LongLive source tree as it is on disk (code, configs; not
    weights or outputs, nor ``exclude`` dirs such as an --out placed inside it), so
    uncommitted edits and non-git checkouts are covered."""
    root = Path(root).resolve()
    skip = {Path(e).resolve() for e in exclude}
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SRC_SKIP_DIRS
                         and (Path(dirpath) / d).resolve() not in skip)
        for name in sorted(files):
            f = Path(dirpath) / name
            if f.suffix.lower() in _SRC_SUFFIXES and f.stat().st_size <= max_file:
                h.update(f.relative_to(root).as_posix().encode() + b"\0")
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _tree_digest(root):
    """Full-content sha256 of every file under ``root`` (paths included), or
    "missing". For wan_models/Wan2.2-TI2V-5B: the T5 text encoder, its tokenizer,
    the base DiT and configs -- everything that turns a prompt into conditioning.
    ~45 GB, so a few minutes per generate call; cheap next to minutes of video."""
    root = Path(root)
    if not root.exists():
        return "missing"
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode() + b"\0")
        h.update(_file_digest(f).encode())
    return h.hexdigest()[:16]


def run_identity(cfg, ckpt, prompts, ll_root, vae=None, out=None):
    """Everything that determines the outputs: the effective overlay, full-content
    digests of the generator checkpoint, of the Wan model files (text encoder,
    tokenizer, base DiT) and of the decoder VAE weights, the prompts and the
    LongLive checkout."""
    ident = dict(overlay=cfg, ckpt=_file_digest(ckpt), prompts=_prompts_digest(prompts),
                 wan_models=_tree_digest(Path(ll_root) / "wan_models" / "Wan2.2-TI2V-5B"),
                 longlive=_git_head(ll_root), longlive_src=_source_digest(ll_root, exclude=[out] if out else ()))
    if vae is not None:
        ident["vae"] = _file_digest(vae) if Path(vae).exists() else "missing"
    return ident


def check_identity(out, ident):
    """Refuse to reuse an --out holding outputs of a different generation. Latents
    or videos without a recorded identity are refused too (unknown provenance)."""
    out = Path(out)
    f = out / "run_identity.json"
    have_outputs = any((out / "latents").glob("*.pt")) or any(out.glob("*.mp4"))
    if f.exists():
        old = json.loads(f.read_text())
        if old != json.loads(json.dumps(ident)):
            diff = sorted(k for k in set(old) | set(ident) if old.get(k) != ident.get(k))
            raise SystemExit(f"{out} holds a generation with different {', '.join(diff)}; "
                             "use a new --out instead of mixing runs")
    elif have_outputs:
        raise SystemExit(f"{out} has latents/videos but no run_identity.json; use a new --out")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(ident, indent=2, sort_keys=True))
    os.replace(tmp, f)


# ---------------------------------------------------------------- decoding
def stream_decode(vae_model, latents, mean, std, emit, chunk=8, unpatchify=None,
                  device="cpu", dtype=None):
    """Decode ``latents`` [T, C, h, w] through a Wan2.2 ``WanVAE_`` with its
    causal feature cache carried across chunks of ``chunk`` latent frames, and
    hand each decoded piece to ``emit(frames)`` as uint8 [n, H, W, 3]. Mirrors
    ``WanVAE_.decode`` (per-latent-frame decoder calls sharing ``_feat_map``,
    ``first_chunk`` on frame 0 only) without holding more than one chunk.
    Returns the number of pixel frames emitted."""
    import torch
    if unpatchify is None:
        def unpatchify(x, patch_size):
            from einops import rearrange
            return rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=patch_size, r=patch_size)
    dtype = dtype or next(vae_model.parameters()).dtype
    mean = mean.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / std.to(device=device, dtype=dtype)).view(1, -1, 1, 1, 1)
    n_out = 0
    vae_model.clear_cache()
    try:
        with torch.inference_mode():
            for s in range(0, latents.shape[0], chunk):
                z = latents[s:s + chunk].to(device=device, dtype=dtype)
                z = z.permute(1, 0, 2, 3).unsqueeze(0)                  # [1, C, t, h, w]
                z = z / inv_std + mean
                x = vae_model.conv2(z)
                outs = []
                for i in range(x.shape[2]):
                    vae_model._conv_idx = [0]
                    outs.append(vae_model.decoder(
                        x[:, :, i:i + 1], feat_cache=vae_model._feat_map,
                        feat_idx=vae_model._conv_idx, first_chunk=(s + i == 0)))
                out = unpatchify(torch.cat(outs, 2), 2).float().clamp_(-1, 1)   # [1, 3, f, H, W]
                frames = ((out[0].permute(1, 2, 3, 0) * 0.5 + 0.5) * 255.0).round().to(torch.uint8)
                emit(frames.cpu().numpy())
                n_out += frames.shape[0]
    finally:
        vae_model.clear_cache()
    return n_out


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wan22_vae_stats(ll_root):
    """The latent mean/std lists of LongLive's ``WanVAEWrapper``, read from its
    source (importing utils.wan_5b_wrapper would pull in the text encoder and
    the diffusion model)."""
    import ast
    import torch
    src = (Path(ll_root) / "utils" / "wan_5b_wrapper.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "WanVAEWrapper":
            vals = {}
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                        and sub.targets[0].id in ("mean", "std")):
                    vals[sub.targets[0].id] = ast.literal_eval(sub.value)
            if set(vals) == {"mean", "std"} and len(vals["mean"]) == len(vals["std"]):
                return (torch.tensor(vals["mean"], dtype=torch.float32),
                        torch.tensor(vals["std"], dtype=torch.float32))
    raise RuntimeError(f"WanVAEWrapper mean/std not found in {ll_root}/utils/wan_5b_wrapper.py")


def load_wan22_vae(ll_root, vae_path=None, device="cuda", dtype="bfloat16"):
    """(model, mean, std, unpatchify) from a LongLive checkout; only the VAE
    module is imported."""
    import torch
    ll_root = Path(ll_root).resolve()
    vae = _load_module(ll_root / "wan_5b" / "modules" / "vae2_2.py", "_ll_vae2_2")
    path = vae_path or str(ll_root / "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    model = vae._video_vae(pretrained_path=path).eval().requires_grad_(False)
    model = model.to(device=device, dtype=getattr(torch, dtype))
    mean, std = wan22_vae_stats(ll_root)
    return model, mean, std, vae.unpatchify


LL_DECODER_SOURCES = ("wan_5b/modules/vae2_2.py", "utils/wan_5b_wrapper.py")


def _decoder_impl_digest(ll_root=None):
    """The decode path's code: this module's stream decode, stats reading, VAE loading
    and mp4 writing, plus the LongLive files they load (the Wan2.2 VAE module and the
    wrapper holding the latent mean/std). A change to any of it invalidates videos
    decoded before."""
    import inspect
    h = hashlib.sha256("".join(inspect.getsource(f) for f in (
        stream_decode, wan22_vae_stats, load_wan22_vae, decode_file)).encode())
    if ll_root is not None:
        for rel in LL_DECODER_SOURCES:
            f = Path(ll_root) / rel
            h.update(rel.encode() + b"\0" + (f.read_bytes() if f.exists() else b"<missing>"))
    return h.hexdigest()[:16]


def decode_file(latent_path, out_path, ll_root, vae_path=None, chunk=8, device="cuda",
                dtype="bfloat16", fps=FPS, _vae=None, vae_digest=None):
    """Latents file -> mp4, written to a temp name and renamed when complete.
    ``<mp4>.provenance.json`` records what the mp4 was decoded from (latents and VAE
    digests, dtype, fps). An existing mp4 is kept only when that record matches;
    one decoded from other inputs, or with no record, is refused."""
    import imageio.v2 as imageio
    import torch
    out_path = Path(out_path)
    if vae_digest is None:
        vae_digest = "injected" if _vae is not None else _file_digest(vae_weights_path(ll_root, vae_path))
    prov = dict(latents=_file_digest(latent_path), vae=vae_digest, dtype=dtype, fps=fps,
                decoder=_decoder_impl_digest(ll_root))
    prov_path = out_path.with_name(out_path.name + ".provenance.json")
    old = json.loads(prov_path.read_text()) if prov_path.exists() else None
    if out_path.exists():
        if old == prov:
            print(f"[decode] exists, same inputs, skipping: {out_path}")
            return None
        raise SystemExit(f"{out_path} exists but was decoded from "
                         f"{'other inputs' if old else 'unrecorded inputs'}; "
                         "delete it or choose another --out")
    lat = torch.load(latent_path, map_location="cpu")
    if lat.dim() == 5:
        lat = lat[0]
    model, mean, std, unpatch = _vae or load_wan22_vae(ll_root, vae_path, device, dtype)
    tmp = out_path.with_suffix(".partial.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error")
    try:
        n = stream_decode(model, lat, mean, std, lambda f: [writer.append_data(x) for x in f],
                          chunk=chunk, unpatchify=unpatch, device=device,
                          dtype=getattr(torch, dtype))
    finally:
        writer.close()
    # record first: a crash between the two renames leaves a record with no mp4,
    # which simply decodes again
    ptmp = prov_path.with_suffix(".tmp")
    ptmp.write_text(json.dumps(prov, sort_keys=True))
    os.replace(ptmp, prov_path)
    os.replace(tmp, out_path)
    print(f"[decode] {latent_path} -> {out_path}: {n} frames, {n / fps:.1f} s")
    return n


# ---------------------------------------------------------------- commands
def _gpu_gib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GIB
    except Exception:
        pass
    return None


def cmd_plan(a):
    p = plan(a.minutes, a.precision, a.window, a.sink)
    print(json.dumps(p, indent=2))


def cmd_generate(a):
    import yaml
    ll_root = Path(a.ll_root).resolve()
    base_path = Path(a.base_config) if a.base_config else ll_root / "configs" / (
        "fp8/inference_fp8.yaml" if a.precision == "fp8" else "inference.yaml")
    base = yaml.safe_load(base_path.read_text())
    lat = latent_frames_for(a.minutes * 60)
    p = plan(a.minutes, a.precision, a.window, a.sink)
    gib = _gpu_gib()
    print(f"[plan] {p['latent_frames']} latent frames = {p['pixel_frames']} frames = "
          f"{p['seconds'] / 60:.2f} min; est. VRAM {p['vram']['total']:.1f} GiB"
          + (f" of {gib:.1f}" if gib else ""))
    if gib and p["vram"]["total"] > gib - 1.0 and not a.force:
        raise SystemExit(f"estimated {p['vram']['total']:.1f} GiB exceeds this "
                         f"{gib:.1f} GiB card; use --precision fp8 / a smaller --window, or --force")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = Path(a.prompts).resolve()
    n_prompts = n_prompts_of(prompts)
    if n_prompts < 1:               # else inference_iter=-1 and "all latents present" holds vacuously
        raise SystemExit(f"no prompts in {prompts} (need non-empty lines, or caption subfolders)")
    cfg = build_overlay(base, latent_frames=lat, prompts=prompts, ckpt=Path(a.ckpt).resolve(),
                        out_dir=out.resolve(), window=a.window, sink=a.sink, seed=a.seed,
                        fp8=a.precision == "fp8", compile=a.compile)
    cfg["inference_iter"] = n_prompts - 1
    ident = None
    if not a.dry_run:
        vae = vae_weights_path(ll_root, a.vae_path)
        ident = run_identity(cfg, Path(a.ckpt).resolve(), prompts, ll_root, vae, out=out)
        check_identity(out, ident)
    lat_dir = out / "latents"
    have = [latent_name(i) for i in range(n_prompts) if (lat_dir / latent_name(i)).exists()]
    if len(have) == n_prompts:
        print("[generate] all latents present, skipping generation")
    else:
        if have:
            print(f"[generate] {len(have)} latent file(s) already present; LongLive "
                  "regenerates from prompt 0 (it has no per-prompt resume)")
        ov = out / "longlive_overlay.yaml"
        ov.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd = [sys.executable, "inference.py", "--config_path", str(ov.resolve())]
        print("[generate]", " ".join(cmd), f"(cwd {ll_root})", flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ll_root, check=True)
    if a.no_decode or a.dry_run:
        return
    for i in range(n_prompts):
        f = lat_dir / latent_name(i)
        if not f.exists():
            raise SystemExit(f"LongLive did not write {f}")
        decode_file(f, out / (f.stem + ".mp4"), ll_root, a.vae_path, a.decode_chunk,
                    a.decode_device, "float32" if a.decode_device == "cpu" else "bfloat16",
                    vae_digest=ident["vae"])


def cmd_decode(a):
    decode_file(a.latents, a.out, a.ll_root, a.vae_path, a.chunk, a.device,
                "float32" if a.device == "cpu" else "bfloat16")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "generate"):
        s = sub.add_parser(name)
        s.add_argument("--minutes", type=float, default=5.0)
        s.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
        s.add_argument("--window", type=int, default=32, help="local_attn_size, latent frames")
        s.add_argument("--sink", type=int, default=8, help="sink frames")
        if name == "generate":
            s.add_argument("--ll-root", required=True)
            s.add_argument("--ckpt", required=True, help="LongLive-2.0-5B model_bf16.pt")
            s.add_argument("--prompts", required=True, help="txt (one prompt per line) or caption dir")
            s.add_argument("--out", required=True)
            s.add_argument("--base-config", default=None)
            s.add_argument("--vae-path", default=None)
            s.add_argument("--decode-chunk", type=int, default=8)
            s.add_argument("--decode-device", default="cuda")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--no-decode", action="store_true")
            s.add_argument("--dry-run", action="store_true", help="write the overlay only")
            s.add_argument("--force", action="store_true")
            s.add_argument("--compile", action="store_true",
                           help="keep the base config's torch_compile setting (default: eager)")
    s = sub.add_parser("decode")
    s.add_argument("--ll-root", required=True)
    s.add_argument("--latents", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--vae-path", default=None)
    s.add_argument("--chunk", type=int, default=8)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("expected", help="print the output stems a prompts file/dir yields")
    s.add_argument("--prompts", required=True)
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "generate": cmd_generate, "decode": cmd_decode,
     "expected": lambda a: print("\n".join(expected_stems(a.prompts)))}[a.cmd](a)


if __name__ == "__main__":
    main()
