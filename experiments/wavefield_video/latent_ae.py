#!/usr/bin/env python3
"""Small convolutional autoencoder -- the gate from token space to real pixels.

The head-to-head predictors in ``train_compare.py`` currently mix RGB frames
directly. Real video is far higher-resolution than the 32x32 toy grid, and the
mixer cost scales with the token count T*H*W. This AE compresses each frame
*spatially* (H,W -> H/4,W/4 via two stride-2 convs) while *expanding* channels
(3 -> 64), so downstream we predict a latent that has 16x fewer tokens but is
per-token richer. Train the AE once, freeze it, then predict next-latent and
decode for pixel-space evaluation (see ``train_compare.py --latent``).

Architecture (task R7):
    encoder  3 -> 32 -> 64   two stride-2 convs      (spatial /4)
    decoder  64 -> 32 -> 3   mirrored transpose convs (spatial x4)
    decode ends in sigmoid so reconstructions live in [0, 1] like the data.

APIs:
    ae.encode(x[B,3,H,W]) -> z[B,64,H/4,W/4]
    ae.decode(z)          -> x_hat[B,3,H,W] in [0,1]

Train / smoke (CPU-quick):
    python latent_ae.py --train --data-source balls --grid 32 --steps 300 --out ckpts/ae.pt
    python latent_ae.py --train --data-source waves --field wave --grid 32 --steps 300 --out ckpts/ae.pt
"""
import argparse, json, pathlib, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# The AE downsamples by exactly this factor (two stride-2 layers => 2**2 = 4).
DOWNSAMPLE = 4


class ConvAE(nn.Module):
    """Conv autoencoder for a single frame [B, 3, H, W].

    ``base`` sets the first hidden width; channels go 3 -> base -> 2*base, so the
    default base=32 reproduces the task's 3->32->64 encoder exactly. H and W must
    be divisible by ``DOWNSAMPLE`` (they are on the 16/32/64 grids we use)."""

    def __init__(self, base=32):
        super().__init__()
        c1, c2 = base, 2 * base
        self.base, self.latent_ch = base, c2
        # kernel 4 / stride 2 / pad 1 halves the spatial dims exactly and mirrors
        # cleanly under ConvTranspose2d with the same (k, s, p) -- no output-size
        # ambiguity, so encode/decode round-trip shapes for any /4-divisible grid.
        self.enc = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1),   # H   -> H/2
            nn.GELU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),  # H/2 -> H/4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),  # H/4 -> H/2
            nn.GELU(),
            nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1),   # H/2 -> H
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        # sigmoid keeps reconstructions in the data's [0,1] range; the latent
        # itself is left unbounded (linear) so the downstream residual predictor
        # can add/subtract deltas freely without saturating a bounded code.
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        return self.decode(self.encode(x))


def load_ae(path, map_location="cpu"):
    """Rebuild a frozen ConvAE from a checkpoint saved by ``--train``."""
    ck = torch.load(path, map_location=map_location, weights_only=True)
    ae = ConvAE(base=ck["config"]["base"])
    ae.load_state_dict(ck["state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ----------------------------------------------------------------------------- data
def _frame_batch(source, field, bs, grid, seed):
    """Flatten a short clip into a batch of independent single frames [N,3,H,W].

    The AE is per-frame, so we only need frames -- reusing the same pluggable
    data sources as train_compare (balls = data.py, waves = data_waves.py) keeps
    the AE's training distribution identical to what the predictors will see."""
    T = 8  # clip length; flattened to bs*(T+1) frames -- pure quantity, not dynamics
    if source == "waves":
        import data_waves as dw
        clip = dw.make_clip_batch(bs, T, grid, grid, seed=seed, field=field)
    else:
        from data import make_clip_batch
        clip = make_clip_batch(bs, T, grid, grid, seed=seed)
    B, Tp, C, H, W = clip.shape
    return clip.reshape(B * Tp, C, H, W)


# ----------------------------------------------------------------------------- train
def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if a.grid % DOWNSAMPLE != 0:
        raise SystemExit(f"--grid {a.grid} must be divisible by {DOWNSAMPLE}")
    ae = ConvAE(base=a.base).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=a.lr, weight_decay=0.0)
    nparam = sum(p.numel() for p in ae.parameters())
    print(f"AE base={a.base} latent_ch={ae.latent_ch} grid={a.grid} "
          f"latent_grid={a.grid // DOWNSAMPLE} params={nparam} dev={dev}", flush=True)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        x = _frame_batch(a.data_source, a.field, a.batch, a.grid, seed=1000 + step).to(dev)
        recon = ae(x)
        loss = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, a.steps // 10) == 0 or step == 1:
            print(json.dumps({"step": step, "recon_mse": round(float(loss), 6),
                              "st_s": round(step / (time.time() - t0), 2)}), flush=True)
    train_sec = time.time() - t0

    # ---- held-out recon MSE (unseen seed range) ----------------------------
    ae.eval()
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(8):
            x = _frame_batch(a.data_source, a.field, 64, a.grid, seed=90000 + i).to(dev)
            tot += F.mse_loss(ae(x), x).item()
            n += 1
        eval_recon_mse = tot / n

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"base": a.base, "latent_ch": ae.latent_ch, "downsample": DOWNSAMPLE,
              "grid": a.grid, "latent_grid": a.grid // DOWNSAMPLE,
              "data_source": a.data_source, "field": a.field}
    torch.save({"state": ae.state_dict(), "config": config}, out)
    report = {"eval_recon_mse": round(eval_recon_mse, 6), "params": nparam,
              "steps": a.steps, "train_sec": round(train_sec, 1), **config}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"AE-DONE recon_mse(held-out)={eval_recon_mse:.6f} -> {out}", flush=True)
    print("AE-RESULT " + json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Small conv AE for wavefield frames.")
    ap.add_argument("--train", action="store_true", help="train the AE and save a checkpoint")
    ap.add_argument("--data-source", choices=["balls", "waves"], default="balls")
    ap.add_argument("--field", choices=["wave", "advection", "vortex"], default="wave",
                    help="PDE style when --data-source waves")
    ap.add_argument("--grid", type=int, default=32, help="frame H=W (must be /4-divisible)")
    ap.add_argument("--base", type=int, default=32, help="first hidden width (3->base->2*base)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32, help="clips/step (x9 frames after flatten)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="ckpts/ae.pt")
    a = ap.parse_args()
    if not a.train:
        ap.error("nothing to do: pass --train (encode/decode are library APIs)")
    train(a)


if __name__ == "__main__":
    main()
