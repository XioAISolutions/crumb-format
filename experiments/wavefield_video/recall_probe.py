"""Delayed-recall probe: can the wave state carry frame 0 to frame D? (LONG_HORIZON.md)

Each clip shows one coloured blob at a random cell on frame 0, then D-1 black
frames; the target (frame D) is the frame-0 blob again. The only route from
input to answer is the recurrent/FFT wave path across D frames, so this isolates
the two things minutes of video need from the mixer:

  1. poles that *can* hold information for D frames (pole_param), and
  2. training clips at least D long, since gradients never see a dependency
     longer than the training window (the campaign trains at T=16-17).

The model is the normal VideoPredictor (causal dispersion wave, residual head),
trained with the FFT forward -- O(T log T) per clip, which is what makes long
training windows affordable at all. CPU-sized on purpose (grid 8).

Score: recall = 1 - MSE(model) / MSE(predict black). 1 = perfect recall,
0 = no better than forgetting. Pre-registered read: a pole family "holds D"
when mean recall >= 0.5 across seeds.

    python recall_probe.py --delays 16 64 256 --pole-param softplus halflife
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from wfvideo import VideoPredictor

PALETTE = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.3], [0.3, 0.4, 1.0]])


def make_batch(bs, D, grid, gen, sigma=0.8):
    """[bs, D+1, 3, grid, grid]: blob on frame 0, black frames 1..D-1, target = frame 0."""
    yy, xx = torch.meshgrid(torch.arange(grid).float(), torch.arange(grid).float(), indexing="ij")
    cy = torch.randint(0, grid, (bs,), generator=gen).float()
    cx = torch.randint(0, grid, (bs,), generator=gen).float()
    col = PALETTE[torch.randint(0, len(PALETTE), (bs,), generator=gen)]
    blob = torch.exp(-((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
                     / (2 * sigma ** 2))                                   # [bs,g,g]
    img = blob[:, None] * col[:, :, None, None]                            # [bs,3,g,g]
    clip = torch.zeros(bs, D + 1, 3, grid, grid)
    clip[:, 0] = img
    clip[:, D] = img
    return clip


def run_one(pole_param, D, seed, a):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = VideoPredictor(a.dim, a.layers, a.heads, D, a.grid, a.grid, "wave", causal=True,
                       kernel_version="dispersion", linear_pad=True,
                       pole_param=pole_param, hl_min=a.hl_min, hl_max=a.hl_max)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
    t0 = time.time()
    for _ in range(a.steps):
        clip = make_batch(a.batch, D, a.grid, gen)
        loss = F.mse_loss(m(clip[:, :D]), clip[:, D])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    train_s = time.time() - t0
    m.eval()
    egen = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        clip = make_batch(256, D, a.grid, egen)
        pred = m(clip[:, :D])
        mse = F.mse_loss(pred, clip[:, D]).item()
        base = F.mse_loss(torch.zeros_like(clip[:, D]), clip[:, D]).item()
        hit = (pred.sum(1).flatten(1).argmax(1) == clip[:, D].sum(1).flatten(1).argmax(1)).float().mean().item()
    return {"pole_param": pole_param, "delay": D, "seed": seed, "steps": a.steps,
            "recall": round(1 - mse / base, 4), "argmax_hit": round(hit, 4),
            "mse": round(mse, 6), "mse_forget": round(base, 6),
            "train_s": round(train_s, 1), "s_per_step": round(train_s / a.steps, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delays", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--pole-param", nargs="+", choices=["softplus", "halflife"],
                    default=["softplus", "halflife"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hl-min", type=float, default=2.0)
    ap.add_argument("--hl-max", type=float, default=4096.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = []
    for D in a.delays:
        for pp in a.pole_param:
            for s in a.seeds:
                r = run_one(pp, D, s, a)
                rows.append(r)
                print(json.dumps(r), flush=True)
    print("\nsummary (mean over seeds):")
    for D in a.delays:
        for pp in a.pole_param:
            rs = [r for r in rows if r["delay"] == D and r["pole_param"] == pp]
            rec = sum(r["recall"] for r in rs) / len(rs)
            hit = sum(r["argmax_hit"] for r in rs) / len(rs)
            sps = sum(r["s_per_step"] for r in rs) / len(rs)
            print(f"  D={D:4d} {pp:>9}: recall={rec:.3f} argmax_hit={hit:.3f} "
                  f"{'HOLDS' if rec >= 0.5 else 'forgets'}  ({sps:.3f} s/step)")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "rows": rows}, fh, indent=1)
    return rows


if __name__ == "__main__":
    main()
