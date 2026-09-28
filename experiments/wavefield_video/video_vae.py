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
                cls = self._cls(base)
                src = path or p["hf"]
                sub = None if path else p["subfolder"]
                self.model = cls.from_pretrained(src, subfolder=sub, torch_dtype=dtype)
            self.channels = self.model.config.latent_channels if base == "ltx" else self.model.config.z_dim
            if hasattr(self.model, "enable_tiling"):
                self.model.enable_tiling()           # bounded memory on long clips
        self.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype
        mean, std = self._stats()
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

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
        m, s = m.float().view(C), s.float().view(C)
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
        return ((z.float() - self.mean) / self.std)

    @torch.no_grad()
    def decode(self, z):
        """normalized latents [B,Tl,C,h,w] -> frames [B,T,3,H,W] in [0,1]."""
        z = (z.float() * self.std + self.mean).to(self.device, self.dtype)
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
                "channels": self.channels}


class LatentShards:
    """Windows of consecutive latent steps from encode_videos.py shards. The last
    ~holdout fraction of SOURCE VIDEOS is held out for eval (whole videos, so no
    held-out frames leak into training through a sibling segment)."""

    def __init__(self, root, window, holdout=0.1, device="cpu"):
        root = pathlib.Path(root)
        index = json.loads((root / "index.json").read_text())
        if len(index) < 2:
            raise SystemExit(f"{root}: need >= 2 shards to hold one out (have {len(index)})")
        # Split BEFORE filtering by window, by source video: the held-out set is a
        # fixed tail of the index, so the trainer and the stream screen (which use
        # different windows) agree on it and no held-out segment is ever trained on.
        srcs = list(dict.fromkeys(m["src"] for m in index))
        n_eval_src = max(1, round(holdout * len(srcs))) if len(srcs) > 1 else 0
        eval_srcs = set(srcs[len(srcs) - n_eval_src:])
        if not eval_srcs:                       # one source video: hold out its tail shards
            cut = len(index) - max(1, round(holdout * len(index)))
            is_eval = [i >= cut for i in range(len(index))]
        else:
            is_eval = [m["src"] in eval_srcs for m in index]
        self.eval_srcs = sorted({m["src"] for m, e in zip(index, is_eval) if e})
        shards = [torch.load(root / m["file"], weights_only=True)["latents"] for m in index]
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
