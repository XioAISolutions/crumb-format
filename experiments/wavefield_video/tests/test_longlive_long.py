"""longlive_long: planning numbers, the overlay, and the constant-memory decoder.

The decoder is checked against a small causal VAE with the Wan2.2 ``WanVAE_``
calling convention (conv2, decoder(x, feat_cache, feat_idx, first_chunk),
_feat_map/_conv_idx, clear_cache): chunked streaming must equal one full
decode for every chunk size. (Against LongLive's real vae2_2 module with a
small random config the same check gives 0 uint8 difference; see
LONGLIVE_4090.md.)
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import longlive_long as L  # noqa: E402


def test_plan_numbers():
    p = L.plan(5)
    assert p["latent_frames"] % L.BLOCK == 0
    assert p["pixel_frames"] >= 5 * 60 * L.FPS
    assert p["pixel_frames"] - L.T_STRIDE * L.BLOCK < 5 * 60 * L.FPS   # at most one block over
    assert p["needs_relative_rope"]                                     # 1800 > 1024 rows
    assert p["resolution"] == "1280x704"
    # the absolute RoPE table ends at 1024 latent frames = 4093 frames ~ 2.84 min
    assert L.pixel_frames(L.ROPE_ROWS) == 4093
    assert not L.plan(2)["needs_relative_rope"]
    assert L.plan(3)["needs_relative_rope"]
    # one float32 copy of a 5-minute decoded clip is tens of GB (the release path)
    assert p["release_decode_copy_gb"] > 70
    assert p["latent_gb"] < 1


def test_vram_estimate():
    fp8, bf16 = L.vram_estimate("fp8"), L.vram_estimate("bf16")
    assert 9.0 < bf16["kv_cache"] < 10.5                  # 30 x 2 x 32 x 880 x 3072 x 2 B
    assert bf16["total"] - fp8["total"] > 3.5
    assert fp8["total"] < 19.0 and bf16["total"] > 21.0   # bf16: <2 GiB headroom on 24 GB
    assert L.vram_estimate("fp8", window=24)["total"] < L.vram_estimate("fp8")["total"]


BASE = {
    "model_kwargs": {"model_name": "Wan2.2-TI2V-5B", "num_frame_per_block": 8, "local_attn_size": 32},
    "num_output_frames": 8,
    "data": {"data_path": "example/long_example.txt", "image_or_video_shape": [1, 8, 48, 44, 80]},
    "inference": {"sampling_steps": 4, "sink_size": 8, "streaming_vae": True, "vae_device": "cuda:2"},
    "checkpoints": {"generator_ckpt": "/path/to/model_bf16.pt"},
    "fp8_quant": True,
    "logging": {"seed": 0},
}


def test_overlay_sets_every_blocker_fix(tmp_path):
    lat = L.latent_frames_for(300)
    cfg = L.build_overlay(BASE, latent_frames=lat, prompts="p.txt", ckpt="c.pt",
                          out_dir=tmp_path, window=24, sink=4, seed=3)
    assert cfg["num_output_frames"] == lat
    assert cfg["data"]["image_or_video_shape"] == [1, lat, 48, 44, 80]
    assert cfg["use_relative_rope"] is True
    assert L.build_overlay(BASE, latent_frames=lat, prompts="p.txt", ckpt="c.pt",
                           out_dir=tmp_path, window=24, sink=4, seed=3,
                           relative_rope=False)["use_relative_rope"] is False
    inf = cfg["inference"]
    assert inf["save_latents_only"] is True and inf["streaming_vae"] is False
    assert "vae_device" not in inf
    assert inf["sink_size"] == 4 and cfg["model_kwargs"]["local_attn_size"] == 24
    assert cfg["checkpoints"]["generator_ckpt"] == "c.pt" and cfg["logging"]["seed"] == 3
    assert cfg["output_folder"] == str(tmp_path / "latents") and cfg["save_with_index"]
    assert BASE["inference"]["streaming_vae"] is True       # input not mutated
    # normalize_config flattens sections onto the top level; nothing may conflict
    flat = {}
    for sec in ("data", "inference", "logging", "checkpoints"):
        for k, v in cfg[sec].items():
            assert k not in cfg or cfg[k] == v, k
            flat[k] = v
    bf16 = L.build_overlay(BASE, latent_frames=lat, prompts="p", ckpt="c", out_dir=tmp_path, fp8=False)
    assert bf16["fp8_quant"] is False
    assert cfg["torch_compile"] is False
    comp = L.build_overlay({**BASE, "torch_compile": "auto"}, latent_frames=lat, prompts="p",
                           ckpt="c", out_dir=tmp_path, compile=True)
    assert comp["torch_compile"] == "auto"


class _CausalConv(nn.Conv3d):
    """Temporal kernel 3, causal, with a 2-frame feature cache (Wan's CACHE_T)."""

    def __init__(self, cin, cout):
        super().__init__(cin, cout, (3, 3, 3), padding=(0, 1, 1))

    def forward(self, x, cache=None):
        pad = cache if cache is not None else x.new_zeros(*x.shape[:2], 2, *x.shape[3:])
        return super().forward(torch.cat([pad, x], 2))


class TinyCausalVAE(nn.Module):
    """Decoder with the WanVAE_ calling convention: one latent frame per call,
    the first call emits 1 frame and every later call emits 2 (temporal x2),
    output channels 3*2*2 for a patch-2 unpatchify."""

    def __init__(self, z=6, d=8):
        super().__init__()
        self.z_dim = z
        self.conv2 = nn.Conv3d(z, z, 1)
        self.c1, self.c2 = _CausalConv(z, d), _CausalConv(d, 12 * 2)
        self.clear_cache()

    def clear_cache(self):
        self._feat_map = [None, None]
        self._conv_idx = [0]

    def _step(self, conv, x, feat_cache, feat_idx):
        i = feat_idx[0]
        pad = feat_cache[i]
        if pad is None:
            pad = x.new_zeros(*x.shape[:2], 2, *x.shape[3:])
        feat_cache[i] = torch.cat([pad, x], 2)[:, :, -2:].clone()
        feat_idx[0] += 1
        return conv(x, pad)

    def decoder(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        h = torch.tanh(self._step(self.c1, x, feat_cache, feat_idx))
        h = self._step(self.c2, h, feat_cache, feat_idx)                 # [1, 24, 1, h, w]
        a, b = h[:, :12], h[:, 12:]
        return a if first_chunk else torch.cat([a, b], 2)

    def decode(self, z, scale):
        self.clear_cache()
        z = z / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(1, -1, 1, 1, 1)
        x = self.conv2(z)
        outs = []
        for i in range(x.shape[2]):
            self._conv_idx = [0]
            outs.append(self.decoder(x[:, :, i:i + 1], feat_cache=self._feat_map,
                                     feat_idx=self._conv_idx, first_chunk=i == 0))
        from einops import rearrange
        out = rearrange(torch.cat(outs, 2), "b (c r q) f h w -> b c f (h q) (w r)", q=2, r=2)
        self.clear_cache()
        return out


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 50])
def test_stream_decode_equals_full_decode(chunk):
    torch.manual_seed(0)
    m = TinyCausalVAE().eval()
    T = 7
    z = torch.randn(T, 6, 3, 4)
    mean, std = torch.randn(6) * 0.1, torch.rand(6) + 0.5
    with torch.no_grad():
        ref = m.decode(z.permute(1, 0, 2, 3)[None], [mean, 1 / std]).float().clamp(-1, 1)
    ref8 = ((ref[0].permute(1, 2, 3, 0) * 0.5 + 0.5) * 255).round().to(torch.uint8).numpy()
    got = []
    n = L.stream_decode(m, z, mean, std, got.append, chunk=chunk)
    g = np.concatenate(got)
    assert n == g.shape[0] == ref8.shape[0] == 1 + 2 * (T - 1)
    assert g.dtype == np.uint8 and g.shape[1:] == (6, 8, 3)
    assert np.array_equal(g, ref8)
    assert m._feat_map == [None, None]                 # cache cleared afterwards


def test_stream_decode_is_causal_not_reset_per_chunk():
    """Resetting the cache at every chunk (what a naive chunked decode does)
    changes the output; the carried cache must not."""
    torch.manual_seed(1)
    m = TinyCausalVAE().eval()
    z = torch.randn(6, 6, 3, 4)
    mean, std = torch.zeros(6), torch.ones(6)
    full = []
    L.stream_decode(m, z, mean, std, full.append, chunk=6)
    naive = []
    for s in range(0, 6, 2):
        L.stream_decode(m, z[s:s + 2], mean, std, naive.append, chunk=2)
    assert sum(x.shape[0] for x in naive) != np.concatenate(full).shape[0] or \
        not np.array_equal(np.concatenate(naive), np.concatenate(full))


def test_decode_file_writes_mp4_atomically(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    torch.manual_seed(2)
    m = TinyCausalVAE().eval()
    lat = torch.randn(5, 6, 8, 8)
    f = tmp_path / "lat.pt"
    torch.save(lat, f)
    out = tmp_path / "v.mp4"
    n = L.decode_file(f, out, ll_root=None, chunk=2, device="cpu", dtype="float32",
                      _vae=(m, torch.zeros(6), torch.ones(6), None))
    assert n == 9 and out.exists() and not (tmp_path / "v.partial.mp4").exists()
    assert L.decode_file(f, out, ll_root=None, chunk=2, device="cpu", dtype="float32",
                         _vae=(m, torch.zeros(6), torch.ones(6), None)) is None


def test_generate_dry_run_writes_overlay(tmp_path):
    yaml = pytest.importorskip("yaml")
    ll = tmp_path / "ll"
    (ll / "configs" / "fp8").mkdir(parents=True)
    (ll / "configs" / "fp8" / "inference_fp8.yaml").write_text(yaml.safe_dump(BASE))
    prompts = tmp_path / "p.txt"
    prompts.write_text("a cat\n\na dog\n")
    out = tmp_path / "run"
    L.main(["generate", "--ll-root", str(ll), "--ckpt", "c.pt", "--prompts", str(prompts),
            "--minutes", "3", "--out", str(out), "--dry-run"])
    cfg = yaml.safe_load((out / "longlive_overlay.yaml").read_text())
    assert cfg["use_relative_rope"] and cfg["inference"]["save_latents_only"]
    assert cfg["num_output_frames"] == L.latent_frames_for(180)
    assert cfg["inference_iter"] == 1


def test_latent_complete_rejects_truncated_and_misshaped(tmp_path):
    good = tmp_path / "good.pt"
    torch.save(torch.zeros(16, L.LATENT_CH, *L.LATENT_HW, dtype=torch.bfloat16), good)
    assert L.latent_complete(good, 16)
    batched = tmp_path / "batched.pt"
    torch.save(torch.zeros(1, 16, L.LATENT_CH, *L.LATENT_HW, dtype=torch.bfloat16), batched)
    assert L.latent_complete(batched, 16)
    assert not L.latent_complete(good, 24)                       # other length
    cut = tmp_path / "cut.pt"
    cut.write_bytes(good.read_bytes()[:1000])                    # interrupted save
    assert not L.latent_complete(cut, 16)
    assert not L.latent_complete(tmp_path / "missing.pt", 16)


def test_generate_refuses_empty_prompts(tmp_path):
    yaml = pytest.importorskip("yaml")
    ll = tmp_path / "ll"
    (ll / "configs" / "fp8").mkdir(parents=True)
    (ll / "configs" / "fp8" / "inference_fp8.yaml").write_text(yaml.safe_dump(BASE))
    empty = tmp_path / "p.txt"
    empty.write_text("\n  \n")
    with pytest.raises(SystemExit, match="no prompts"):
        L.main(["generate", "--ll-root", str(ll), "--ckpt", "c.pt", "--prompts", str(empty),
                "--minutes", "1", "--out", str(tmp_path / "run")])


def test_expected_stems_txt_and_dir(tmp_path):
    t = tmp_path / "p.txt"
    t.write_text("a\n\n b \nc\n")
    assert L.expected_stems(t) == ["rank0-0-0_regular", "rank0-1-0_regular", "rank0-2-0_regular"]
    d = tmp_path / "caps" / "caption"
    (d / "s0").mkdir(parents=True)
    (d / "s1").mkdir()
    (d / "notes.txt").write_text("x")
    assert L.n_prompts_of(tmp_path / "caps") == 2


def _ident(tmp_path, **over):
    vae = tmp_path / "vae.pth"
    if not vae.exists():
        vae.write_bytes(b"v" * 100)
    ck = tmp_path / "ck.pt"
    if not ck.exists():
        ck.write_bytes(b"w" * 1000)
    pr = tmp_path / "p.txt"
    if not pr.exists():
        pr.write_text("a cat\n")
    cfg = L.build_overlay(BASE, latent_frames=over.pop("lat", 16), prompts=pr, ckpt=ck,
                          out_dir=tmp_path / "run", seed=over.pop("seed", 0))
    ll = tmp_path / "ll"
    ll.mkdir(exist_ok=True)
    if not (ll / "inference.py").exists():
        (ll / "inference.py").write_text("# LongLive\n")
    t5 = ll / "wan_models" / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth"
    if not t5.exists():
        t5.parent.mkdir(parents=True)
        t5.write_bytes(b"t5" * 50)
    return L.run_identity(cfg, ck, pr, ll, vae)


def test_identity_guard(tmp_path):
    out = tmp_path / "run"
    (out / "latents").mkdir(parents=True)
    L.check_identity(out, _ident(tmp_path))                       # fresh: records it
    (out / "latents" / "rank0-0-0_regular.pt").write_bytes(b"x")
    L.check_identity(out, _ident(tmp_path))                       # same config: resume ok
    for change in (dict(lat=24), dict(seed=1)):
        with pytest.raises(SystemExit, match="overlay"):
            L.check_identity(out, _ident(tmp_path, **change))
    (tmp_path / "p.txt").write_text("a dog\n")                    # same path, new contents
    with pytest.raises(SystemExit, match="prompts"):
        L.check_identity(out, _ident(tmp_path))
    (tmp_path / "p.txt").write_text("a cat\n")
    (tmp_path / "ck.pt").write_bytes(b"v" * 1000)                 # same path + size, new weights
    with pytest.raises(SystemExit, match="ckpt"):
        L.check_identity(out, _ident(tmp_path))
    (tmp_path / "ck.pt").write_bytes(b"w" * 1000)
    ck = bytearray(b"w" * 1000)
    ck[500] = ord("x")                                            # one middle byte, same size
    (tmp_path / "ck.pt").write_bytes(bytes(ck))
    with pytest.raises(SystemExit, match="ckpt"):
        L.check_identity(out, _ident(tmp_path))
    (tmp_path / "ck.pt").write_bytes(b"w" * 1000)
    (tmp_path / "vae.pth").write_bytes(b"u" * 100)                # decoder weights replaced
    with pytest.raises(SystemExit, match="vae"):
        L.check_identity(out, _ident(tmp_path))
    (tmp_path / "vae.pth").write_bytes(b"v" * 100)
    L.check_identity(out, _ident(tmp_path))                       # restored: ok again
    (tmp_path / "ll" / "inference.py").write_text("# locally edited\n")   # dirty / non-git LongLive
    with pytest.raises(SystemExit, match="longlive_src"):
        L.check_identity(out, _ident(tmp_path))
    (tmp_path / "ll" / "inference.py").write_text("# LongLive\n")
    t5 = tmp_path / "ll" / "wan_models" / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth"
    t5.write_bytes(b"T5" * 50)                                    # text encoder replaced in place
    with pytest.raises(SystemExit, match="wan_models"):
        L.check_identity(out, _ident(tmp_path))
    t5.write_bytes(b"t5" * 50)
    L.check_identity(out, _ident(tmp_path))
    # decode provenance covers LongLive's own VAE sources, not just this module's code
    ll = tmp_path / "ll"
    (ll / "wan_5b" / "modules").mkdir(parents=True)
    (ll / "wan_5b" / "modules" / "vae2_2.py").write_text("# vae\n")
    a = L._decoder_impl_digest(ll)
    (ll / "wan_5b" / "modules" / "vae2_2.py").write_text("# vae, patched\n")
    assert L._decoder_impl_digest(ll) != a
    inside = tmp_path / "ll" / "run_inside"                       # an --out inside the checkout
    inside.mkdir()
    a = L._source_digest(tmp_path / "ll", exclude=[inside])
    (inside / "run_identity.json").write_text("{}")
    assert L._source_digest(tmp_path / "ll", exclude=[inside]) == a
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "old.mp4").write_bytes(b"x")                          # outputs of unknown provenance
    with pytest.raises(SystemExit, match="no run_identity"):
        L.check_identity(bare, _ident(tmp_path))


def test_checkpoint_digest_covers_the_middle(tmp_path):
    """Same size and same first/last 16 MiB, one byte changed in the middle."""
    f = tmp_path / "big.pt"
    n = 40 << 20
    with open(f, "wb") as fh:
        fh.truncate(n)
    a = L._file_digest(f)
    with open(f, "r+b") as fh:
        fh.seek(n // 2)
        fh.write(b"\x01")
    assert L._file_digest(f) != a


def test_decode_provenance_guard(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    torch.manual_seed(3)
    m = TinyCausalVAE().eval()
    vae = (m, torch.zeros(6), torch.ones(6), None)
    f = tmp_path / "lat.pt"
    torch.save(torch.randn(3, 6, 8, 8), f)
    out = tmp_path / "v.mp4"
    kw = dict(ll_root=None, chunk=2, device="cpu", dtype="float32", _vae=vae, vae_digest="A")
    assert L.decode_file(f, out, **kw) == 5
    assert L.decode_file(f, out, **kw) is None                   # same inputs: kept
    with pytest.raises(SystemExit, match="other inputs"):
        L.decode_file(f, out, **{**kw, "vae_digest": "B"})       # another VAE
    torch.save(torch.randn(3, 6, 8, 8), f)                        # other latents, same path
    with pytest.raises(SystemExit, match="other inputs"):
        L.decode_file(f, out, **kw)
    # inside generate() (identity verified) stale renders are replaced, not refused
    assert L.decode_file(f, out, **kw, replace_stale=True) == 5
    assert L.decode_file(f, out, **kw) is None                   # and now current
    (tmp_path / "v.mp4.provenance.json").unlink()
    with pytest.raises(SystemExit, match="unrecorded"):
        L.decode_file(f, out, **kw)
