#!/usr/bin/env python3
"""M1 assessment — plugin vs trivial temporal baselines on the real rollout.

The plugin is only "good" if it stabilizes the low band (drift/flicker) while
KEEPING motion + detail. Dumb temporal filters (EMA, median over time) shrink
the low band too — but they blur everything with it. Compare on the same metrics.
"""
import importlib.util, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1", os.path.join(HERE, "m1_integration.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

frames, grid, png = m1.load_rollout_frames(m1.DEFAULT_ROLLOUT, 256, 16)
print(f"loaded {tuple(frames.shape)} grid={grid}")

CFG = dict(anchor_mode="magnitude", cutoff_frac=0.1, alpha=0.9, rho=0.995)
plugin = m1.run_engine(frames, **CFG)


def ema(x, a=0.2):
    out = torch.empty_like(x)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        out[t] = a * out[t - 1] + (1 - a) * x[t]
    return out


def tmed(x, k=5):
    T = x.shape[0]
    pad = k // 2
    xp = torch.cat([x[:1].repeat(pad, 1, 1, 1), x, x[-1:].repeat(pad, 1, 1, 1)], 0)
    out = []
    for t in range(T):
        out.append(xp[t:t + k].median(dim=0).values)
    return torch.stack(out, 0)


cands = {
    "raw (no-op)": frames,
    "PLUGIN": plugin,
    "temporal EMA a=0.2": ema(frames),
    "temporal median k=5": tmed(frames),
}
hd = f"{'candidate':22s} {'lowvar_drop%':>12s} {'cent_drop%':>10s} {'lowflick%':>9s} {'hf_ratio':>8s} {'hf_ssim':>7s} {'own_flicker':>11s}"
print(hd)
for name, c in cands.items():
    r = m1.evaluate(frames, c, 0.1)
    print(f"{name:22s} {r['lowband_mag_var_drop']:12.1f} {r['centroid_var_drop']:10.1f} "
          f"{r['low_flicker_drop']:9.1f} {r['high_flicker_ratio']:8.3f} {r['detail_hfssim']:7.4f} "
          f"{r['plugin_flicker']:11.2e}")
print()
print("Reading: hf_ratio ~1 and hf_ssim ~1 = motion/detail kept. A candidate that")
print("drops the low band AND tanks hf_ssim (e.g. EMA/median touching every band)")
print("is 'stabilizing' by blurring — the plugin should not look like that.")
