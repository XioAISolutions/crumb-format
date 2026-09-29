"""Pretrained video VAE adapter: pixels <-> latent sequences (LONG_HORIZON.md phase 2).

The home-made per-frame ConvAE (latent_ae.py) fades to black in both arms, and a
per-frame AE gives no temporal compression. Production video VAEs compress time
too: 5 minutes at 24 fps is 7,200 frames but only 900 LTX latent steps. The wave
predictor then runs on latent steps, not frames.

    vae = VideoVAE("ltx", path="Lightricks/LTX-Video")      # or a local dir
    z = vae.encode(frames)          # frames [B,T,3,H,W] in [0,1] -> [B,Tl,C,h,w]
    x = vae.decode(z)               # -> [B,T',3,H',W'] in [0,1]

Backends
  ltx   diffusers.AutoencoderKLLTXVideo  t x8, s x32, 128 ch   (T = 1 + 8k)
  wan   diffusers.AutoencoderKLWan       t x4, s x8,  16 ch    (T = 1 + 4k)
  conv  latent_ae.ConvAE (per frame)     t x1, s x4            (legacy track)
  *-tiny  random small configs of ltx/wan, for CPU tests only (never meaningful)

Latents are normalized with the VAE's own per-channel mean/std when it has them,
so the predictor sees roughly unit-scale inputs. Weights load from a local
directory or a Hugging Face id (``from_pretrained``); ``save`` writes a directory
that loads back identically, which is how tests pin the tiny random VAEs.
"""
import hashlib
import json
import pathlib

import torch
import torch.nn as nn

PRESETS = {
    "ltx": dict(t_stride=8, s_stride=32, channels=128, hf="Lightricks/LTX-Video",
                subfolder="vae"),
    "wan": dict(t_stride=4, s_stride=8, channels=16, hf="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                subfolder="vae"),
    "conv": dict(t_stride=1, s_stride=4, channels=None, hf=None, subfolder=None),
}


def _tiny(backend):
    torch.manual_seed(0)
    if backend == "ltx":
        from diffusers import AutoencoderKLLTXVideo
        return AutoencoderKLLTXVideo(latent_channels=8, block_out_channels=(8, 8, 8, 8),
                                     decoder_block_out_channels=(8, 8, 8, 8),
                                     layers_per_block=(1, 1, 1, 1, 1),
                                     decoder_layers_per_block=(1, 1, 1, 1, 1))
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                            latents_mean=[0.0] * 4, latents_std=[1.0] * 4)


def latent_steps(n_frames, t_stride):
    """Latent steps a causal video VAE produces for n valid frames (1 + (n-1)/t)."""
    return n_frames if t_stride == 1 else 1 + (n_frames - 1) // t_stride


def valid_frames(n_frames, t_stride):
    """Largest frame count <= n the VAE accepts (1 + t*k); extra frames are dropped."""
    return n_frames if t_stride == 1 else 1 + t_stride * ((n_frames - 1) // t_stride)


class VideoVAE(nn.Module):
    def __init__(self, backend, path=None, device="cpu", dtype=torch.float32):
        super().__init__()
        tiny = backend.endswith("-tiny")
        base = backend[:-5] if tiny else backend
        if base not in PRESETS:
            raise ValueError(f"unknown VAE backend {backend!r} (want {', '.join(PRESETS)}[-tiny])")
        self.backend, self.base = backend, base
        p = PRESETS[base]
        self.t_stride, self.s_stride = p["t_stride"], p["s_stride"]
        if base == "conv":
            from latent_ae import load_ae
            if not path:
                raise ValueError("conv backend needs path= a latent_ae.py checkpoint")
            self.model = load_ae(path, map_location=device)
            self.channels = self.model.latent_ch
        else:
            if tiny and not path:
                self.model = _tiny(base)
            else:
                self.model = self._load(self._cls(base), path or p["hf"], p["subfolder"], dtype)
            self.channels = self.model.config.latent_channels if base == "ltx" else self.model.config.z_dim
            if hasattr(self.model, "enable_tiling"):
                self.model.enable_tiling()           # bounded memory on long clips
        self.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype
        mean, std = self._stats()
        # on the VAE's device: callers pass CPU or GPU latents (normalize after moving)
        self.register_buffer("mean", mean.to(self.device), persistent=False)
        self.register_buffer("std", std.to(self.device), persistent=False)

    @staticmethod
    def _load(cls, src, subfolder, dtype):
        """src: a VAE directory, a pipeline directory (VAE in ``subfolder``), or an
        HF id of either kind. Local dirs are resolved by where config.json is; HF
        ids try the pipeline layout first, then the repository root."""
        local = pathlib.Path(src).expanduser()
        if local.is_dir():
            sub = None if (local / "config.json").is_file() else subfolder
            return cls.from_pretrained(str(local), subfolder=sub, torch_dtype=dtype)
        try:
            return cls.from_pretrained(src, subfolder=subfolder, torch_dtype=dtype)
        except (OSError, EnvironmentError, ValueError):
            return cls.from_pretrained(src, torch_dtype=dtype)

    @staticmethod
    def _cls(base):
        from diffusers import AutoencoderKLLTXVideo, AutoencoderKLWan
        return AutoencoderKLLTXVideo if base == "ltx" else AutoencoderKLWan

    def _stats(self):
        C = self.channels
        if self.base == "ltx":
            m, s = self.model.latents_mean, self.model.latents_std
        elif self.base == "wan":
            m = torch.tensor(self.model.config.latents_mean)
            s = torch.tensor(self.model.config.latents_std)
        else:
            m, s = torch.zeros(C), torch.ones(C)
        m, s = m.detach().float().cpu().view(C), s.detach().float().cpu().view(C)
        if not torch.isfinite(s).all() or (s <= 0).any():
            s = torch.ones(C)
        return m.view(1, 1, C, 1, 1), s.view(1, 1, C, 1, 1)

    def latent_steps(self, n_frames):
        return latent_steps(n_frames, self.t_stride)

    @torch.no_grad()
    def encode(self, frames):
        """frames [B,T,3,H,W] in [0,1] -> normalized latents [B,Tl,C,h,w] (float32).
        T is trimmed to the VAE's 1 + t*k grid; H, W must divide by s_stride."""
        B, T, C, H, W = frames.shape
        if H % self.s_stride or W % self.s_stride:
            raise ValueError(f"H, W must be multiples of {self.s_stride} (got {H}x{W})")
        T = valid_frames(T, self.t_stride)
        x = frames[:, :T].to(self.device, self.dtype)
        if self.base == "conv":
            z = self.model.encode(x.reshape(B * T, C, H, W))
            z = z.reshape(B, T, *z.shape[1:])
        else:
            z = self.model.encode((x * 2 - 1).permute(0, 2, 1, 3, 4)).latent_dist.mode()
            z = z.permute(0, 2, 1, 3, 4)                          # [B,Tl,C,h,w]
        return (z.float() - self.mean) / self.std

    @torch.no_grad()
    def decode(self, z):
        """normalized latents [B,Tl,C,h,w] -> frames [B,T,3,H,W] in [0,1]."""
        z = (z.to(self.device).float() * self.std + self.mean).to(self.dtype)
        B, Tl = z.shape[:2]
        if self.base == "conv":
            x = self.model.decode(z.reshape(B * Tl, *z.shape[2:]))
            x = x.reshape(B, Tl, *x.shape[1:])
        else:
            x = self.model.decode(z.permute(0, 2, 1, 3, 4)).sample.permute(0, 2, 1, 3, 4)
            x = (x + 1) / 2
        return x.float().clamp(0, 1)

    def save(self, path):
        if self.base == "conv":
            raise ValueError("conv backend is saved by latent_ae.py")
        self.model.save_pretrained(path)

    def describe(self):
        return {"backend": self.backend, "t_stride": self.t_stride, "s_stride": self.s_stride,
                "channels": self.channels, "fingerprint": self.fingerprint}

    @property
    def fingerprint(self):
        """sha256 over the weights, the latent normalization and the model config: shards,
        resumed encodes and the stream's decoder must all use the same latent space,
        and backend + shape alone cannot tell two checkpoints apart."""
        if getattr(self, "_fp", None) is None:
            h = hashlib.sha256()
            # the effective normalization too: Wan keeps it in config, not in weights
            items = sorted(self.model.state_dict().items()) + [("_norm_mean", self.mean),
                                                                ("_norm_std", self.std)]
            for k, v in items:
                h.update(f"{k}{tuple(v.shape)}{v.dtype}".encode())
                h.update(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
            # config-only forward options (e.g. LTX encoder_causal) change the latents too
            # ("_"-prefixed keys are loader metadata -- _name_or_path is the path the
            # weights were loaded FROM, so the same weights via an HF id and a local
            # snapshot would otherwise fingerprint differently)
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                pub = {k: v for k, v in dict(cfg).items() if not str(k).startswith("_")}
                h.update(json.dumps(pub, sort_keys=True, default=str).encode())
            self._fp = h.hexdigest()[:16]
        return self._fp


def _frame_start(j, stride):
    """First decoded frame of window-local latent j (a causal video VAE gives
    latent 0 one frame and every later latent `stride` frames)."""
    return 0 if j <= 0 else 1 + (j - 1) * stride


def _latent_of_frame(f, stride):
    return 0 if f == 0 else (f - 1) // stride + 1


def measure_temporal_rf(vae, C, h, w, n=40, max_n=320, tol=1e-4):
    """(back, fwd): how many latents before / after latent i the decoded frames of
    latent i depend on (above tol x the output scale), measured on the actual
    weights by perturbing one latent of an n-latent probe. The probe doubles until
    the dependency fits inside it; one that still reaches the probe's ends at max_n
    latents is refused (chunked decoding could not reproduce a single decode)."""
    while True:
        g = torch.Generator().manual_seed(0)
        z = torch.randn(1, n, C, h, w, generator=g) * 0.5
        m = n // 4                                  # causal decoders reach mostly forward in frames
        z2 = z.clone()
        z2[:, m] += 1.0
        with torch.no_grad():
            a, b = vae.decode(z), vae.decode(z2)
        d = (a - b).abs().flatten(2).amax(2)[0]                  # [frames]
        hit = (d > tol * max(float(a.abs().max()), 1e-6)).nonzero().flatten().tolist()
        if not hit:
            return 0, 0
        lo, hi = _latent_of_frame(hit[0], vae.t_stride), _latent_of_frame(hit[-1], vae.t_stride)
        if lo > 0 and hi < n - 1:
            return hi - m, m - lo
        if n >= max_n:
            raise SystemExit(f"decoder temporal receptive field exceeds a {n}-latent probe; "
                             "chunked decoding cannot match a single decode")
        n *= 2


class StreamDecoder:
    """Decode a latent stream chunk by chunk with the frames a single decode of the
    whole stream would give, up to the decoder's temporal receptive field: every
    decode sees `history` latents before and `lookahead` latents after the ones it
    emits (those are held back until their lookahead arrives; flush() emits them at
    the true end of the stream). Constant memory: the buffer holds at most
    history + lookahead + one chunk of latents.

    Latents are indexed globally: seed() context gets negative indices and is never
    emitted; generated latent g gives decoded frames [g*stride, (g+1)*stride)."""

    def __init__(self, vae, history, lookahead):
        if history < 1:
            raise ValueError("history must be >= 1 (the first window latent decodes to one frame)")
        self.vae, self.H, self.L = vae, int(history), int(lookahead)
        self.buf, self.start, self.emitted, self.total = None, 0, 0, 0

    def seed(self, ctx):
        self.buf = ctx[:, -self.H:].detach().cpu().float()
        self.start, self.emitted, self.total = -self.buf.shape[1], 0, 0

    def _emit(self, upto):
        if upto <= self.emitted:
            return None
        w0 = max(self.start, self.emitted - self.H)
        win = self.buf[:, w0 - self.start:self.total - self.start]
        x = self.vae.decode(win).cpu()
        s = self.vae.t_stride
        f0 = _frame_start(self.emitted - w0, s)
        f1 = _frame_start(upto - w0, s)
        self.emitted = upto
        keep = max(self.start, self.emitted - self.H)             # history for the next window
        self.buf, self.start = self.buf[:, keep - self.start:], keep
        return x[:, f0:f1]

    def push(self, chunk):
        chunk = chunk.detach().cpu().float()
        self.buf = chunk if self.buf is None else torch.cat([self.buf, chunk], 1)
        self.total += chunk.shape[1]
        return self._emit(self.total - self.L)

    def flush(self):
        return self._emit(self.total)

    def state_dict(self):
        return {"H": self.H, "L": self.L, "buf": self.buf, "start": self.start,
                "emitted": self.emitted, "total": self.total}

    def load_state_dict(self, d):
        if (d["H"], d["L"]) != (self.H, self.L):
            raise SystemExit(f"stream state was decoded with history/lookahead {d['H']}/{d['L']}, "
                             f"this decoder measures {self.H}/{self.L}")
        self.buf, self.start, self.emitted, self.total = d["buf"], d["start"], d["emitted"], d["total"]


class LatentShards:
    """Windows of consecutive latent steps from encode_videos.py shards. The last
    ~holdout fraction of SOURCE VIDEOS is held out for eval (whole videos, so no
    held-out frames leak into training through a sibling segment)."""

    def __init__(self, root, window, holdout=0.1, device="cpu"):
        root = pathlib.Path(root)
        raw = (root / "index.json").read_text()
        index = json.loads(raw)
        # identifies the dataset (every shard's source, span, encoder and content
        # digest, each digest re-verified below), so a resumed run cannot silently
        # continue on different videos
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if len(index) < 2:
            raise SystemExit(f"{root}: need >= 2 shards to hold one out (have {len(index)})")
        shards = []
        for m in index:             # each shard must still be what the index recorded
            z = torch.load(root / m["file"], weights_only=True)["latents"]
            got = hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
            if m.get("sha256") != got:
                raise SystemExit(f"{root / m['file']}: content does not match index.json "
                                 f"({m.get('sha256')} != {got}); re-encode {root}")
            shards.append(z)
        # Split BEFORE filtering by window, by source video: the held-out set is a
        # fixed tail of the index, so the trainer and the stream screen (which use
        # different windows) agree on it and no held-out segment is ever trained on.
        srcs = list(dict.fromkeys(m["src"] for m in index))
        n_eval_src = max(1, round(holdout * len(srcs))) if len(srcs) > 1 else 0
        eval_srcs = set(srcs[len(srcs) - n_eval_src:])
        if not eval_srcs:
            # one source video: hold out its tail, counted in FULL-length segments so
            # a short final remainder (which a window may filter out) is never the
            # whole held-out set. Still independent of the window.
            full = max(z.shape[0] for z in shards)
            want = max(1, round(holdout * sum(z.shape[0] == full for z in shards)))
            cut, seen = len(index), 0
            while cut > 1 and seen < want:
                cut -= 1
                seen += shards[cut].shape[0] == full
            is_eval = [i >= cut for i in range(len(index))]
        else:
            is_eval = [m["src"] in eval_srcs for m in index]
        self.eval_srcs = sorted({m["src"] for m, e in zip(index, is_eval) if e})
        self.train = [z for z, e in zip(shards, is_eval) if not e and z.shape[0] >= window]
        self.eval = [z for z, e in zip(shards, is_eval) if e and z.shape[0] >= window]
        if not self.train or not self.eval:
            raise SystemExit(f"{root}: need train and held-out shards with >= {window} latent "
                             f"steps (have {len(self.train)} / {len(self.eval)}); lower "
                             "--seq-frames or encode longer clips")
        self.C, self.h, self.w = shards[0].shape[1:]
        self.vae = index[0]["vae"]
        self.device = device

    def batch(self, bs, window, gen, split="train"):
        pool = [z for z in (self.train if split == "train" else self.eval) if z.shape[0] >= window]
        if not pool:
            raise SystemExit(f"no {split} shard has {window} latent steps")
        out = []
        for _ in range(bs):
            z = pool[int(torch.randint(len(pool), (1,), generator=gen))]
            s0 = int(torch.randint(z.shape[0] - window + 1, (1,), generator=gen))
            out.append(z[s0:s0 + window])
        return torch.stack(out).float().to(self.device)          # [bs, window, C, h, w]
