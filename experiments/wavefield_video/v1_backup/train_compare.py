#!/usr/bin/env python3
"""Head-to-head: wave-field vs attention video predictor, same scaffold + budget."""
import argparse, json, sys, time, pathlib
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wfvideo import VideoPredictor
from data import make_clip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["wave", "attn"], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--drift-steps", type=int, default=16)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    base = pathlib.Path(a.out)
    base.mkdir(parents=True, exist_ok=True)

    m = VideoPredictor(a.dim, a.layers, a.heads, a.frames, a.grid, a.grid, a.kind).to(dev)
    m.use_ckpt = a.ckpt
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        clips = make_clip_batch(a.batch, a.frames, a.grid, a.grid, device=dev, seed=1000 + step)
        ctx, tgt = clips[:, :a.frames], clips[:, a.frames]
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = m(ctx)
            loss = F.mse_loss(pred.float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            row = {"step": step, "loss": round(loss.item(), 5), "st_s": round(step / (time.time() - t0), 2)}
            log.append(row)
            print(json.dumps(row), flush=True)
    train_sec = time.time() - t0

    m.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(a.eval_batches):
            clips = make_clip_batch(64, a.frames, a.grid, a.grid, device=dev, seed=90000 + i)
            ctx, tgt = clips[:, :a.frames], clips[:, a.frames]
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                pred = m(ctx)
            se += F.mse_loss(pred.float(), tgt).item()
        eval_mse = se / a.eval_batches
        clip = make_clip_batch(16, a.frames + a.drift_steps, a.grid, a.grid, device=dev, seed=777)
        win = clip[:, :a.frames].clone()
        drift = []
        for k in range(a.drift_steps):
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                nxt = m(win)
            drift.append(round(F.mse_loss(nxt.float(), clip[:, a.frames + k]).item(), 5))
            win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], 1)
        pred = m(clip[:, :a.frames]).float()
    gt = clip[:, a.frames]
    lt = clip[:, a.frames - 1]
    rows = [torch.cat([lt[i], gt[i], pred[i].clamp(0, 1)], 2) for i in range(4)]
    strip = torch.cat(rows, 1).clamp(0, 1)
    png = False
    try:
        from PIL import Image
        import numpy as np
        arr = (strip.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(str(base / f"strip_{a.kind}{a.tag}.png"))
        png = True
    except Exception as e:
        print("PNG-FAIL", e, flush=True)
    res = {"kind": a.kind, "params": nparam, "steps": a.steps, "dim": a.dim,
           "layers": a.layers, "heads": a.heads, "grid": a.grid, "frames": a.frames,
           "batch": a.batch, "train_sec": round(train_sec, 1),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else 0,
           "eval_mse": round(eval_mse, 5), "drift": drift, "png": png,
           "log_tail": log[-8:]}
    (base / f"result_{a.kind}{a.tag}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": m.state_dict()}, base / f"model_{a.kind}{a.tag}.pt")
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
