#!/usr/bin/env python3
"""Why is centroid error ~6px for every arm? Test: GT-render centroid vs sim pos,
then model rollout per-step centroid error. Usage: python debug_centroid.py model_wave_w1.pt"""
import sys, torch, torch.nn.functional as F
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wfvideo import VideoPredictor
from data import make_clip_batch
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("tc", "train_compare.py")
tc = importlib.util.module_from_spec(spec)
import types
# avoid running main: train_compare guards with __main__? check; load via exec with name != main
spec.loader.exec_module(tc)  # if this runs main we'll see; else fine

def main():
    dev = "cpu"
    import json as _json
    ck = torch.load(sys.argv[1], map_location="cpu")
    st = ck["state"]
    jj = _json.load(open(sys.argv[2]))
    cfg = {"kind": jj["kind"], "dim": jj["dim"], "layers": jj["layers"], "heads": jj["heads"],
           "grid": jj["grid"], "frames": jj["frames"], "kernel_version": jj.get("kernel_version", "separable"),
           "gate": jj.get("gate", False), "local_fuse": jj.get("local_fuse", False),
           "causal": jj.get("causal", False), "residual": True, "ffn_mult": jj.get("ffn_mult", 4.0)}
    print("cfg:", {k: cfg[k] for k in ("kind","dim","layers","heads","grid","frames","kernel_version","gate","local_fuse","causal") if k in cfg})
    m = VideoPredictor(cfg["dim"], cfg["layers"], cfg["heads"], cfg["frames"], cfg["grid"], cfg["grid"],
                       cfg["kind"], causal=cfg.get("causal", False), residual=cfg.get("residual", True),
                       kernel_version=cfg.get("kernel_version", "separable"), ffn_mult=cfg.get("ffn_mult", 4.0),
                       linear_pad=cfg.get("linear_pad", False), gate=cfg.get("gate", False),
                       local_fuse=cfg.get("local_fuse", False))
    m.load_state_dict(st); m.eval()
    S, R, G, T = 4, 16, cfg["grid"], cfg["frames"]
    clip, meta = make_clip_batch(S, T + R, G, G, device=dev, seed=90000, collisions=True, return_meta=True)
    cols, pos = meta["col"], meta["pos"]
    # (a) GT-render centroid vs sim position
    gc_gt = tc.centroids_by_color(clip[:, T], cols)
    err_gt = (gc_gt - pos[:, T]).norm(dim=-1)
    print("GT-render centroid err (frame T): mean %.2f px  per-ball %s" % (err_gt.mean(), [round(x,2) for x in err_gt[0].tolist()]))
    # (b) model rollout per-step median centroid err
    win = clip[:, :T].clone()
    with torch.no_grad():
        for k in range(8):
            nxt = m(win)
            pc = tc.centroids_by_color(nxt, cols)
            e = (pc - pos[:, T + k]).norm(dim=-1)
            print("k=%2d  median=%.2f  mean=%.2f  mse=%.4f" % (k, e.median(), e.mean(), F.mse_loss(nxt, clip[:, T + k]).item()))
            win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], 1)

if __name__ == "__main__":
    main()
