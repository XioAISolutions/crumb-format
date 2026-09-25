#!/usr/bin/env python3
"""V2 suite plots: rollout MSE curves + quality/efficiency bars (wave vs attn vs ssm)."""
import json, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs_v2_box")
arms, res, eeval = {}, {}, {}
def norm(x): return x.replace("result_", "").replace("eeval_", "")
for f in sorted(d.glob("result_*.json")):
    j = json.load(open(f)); res[norm(f.stem)] = j
for f in sorted(d.glob("eeval_*.json")):
    j = json.load(open(f)); eeval[norm(f.stem)] = j
names = sorted(res.keys())
print("keys", names, sorted(eeval.keys()))
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for n in names:
    c = res[n].get("rollout_mse_curve")
    if c: ax[0][0].plot(range(len(c)), c, label=n, lw=1.5)
ax[0][0].set(title="Rollout MSE over 256 frames (lower=better)", xlabel="rollout step", ylabel="MSE")
ax[0][0].legend(fontsize=8); ax[0][0].grid(alpha=.3)
ee = {n: eeval.get(n, {}) for n in names}
ax[0][1].bar(range(len(names)), [ee[n].get("mean_centroid_err", 0) for n in names], .5,
             label="mean px", color="#4C72B0")
ax[0][1].bar([x + .5 for x in range(len(names))], [ee[n].get("final_centroid_err", 0) for n in names], .5,
             label="final px", color="#DD8452")
ax[0][1].set(title="Ball centroid error (px) — fixed metric", xticks=range(len(names)), ylim=(0, 8))
ax[0][1].set_xticklabels(names, fontsize=8); ax[0][1].legend(fontsize=8); ax[0][1].grid(alpha=.3, axis="y")
q = [res[n].get("eval_mse_over_copylast", 0) for n in names]
ax[1][0].bar(range(len(names)), q, .55, color="#55A868")
ax[1][0].set(title="Quality: model MSE / copy-last MSE (lower=better)", xticks=range(len(names)), ylim=(0, 1.2))
ax[1][0].set_xticklabels(names, fontsize=8); ax[1][0].axhline(1, color="r", ls="--", lw=1)
ax[1][0].grid(alpha=.3, axis="y")
fps = [ee[n].get("rollout_fps", 0) for n in names]
mem = [ee[n].get("mem_gb_at_256", 0) or 0 for n in names]
a2 = ax[1][1]; a2.bar(range(len(names)), fps, .55, color="#8172B3")
a2.set(title="Rollout speed (fps)", xticks=range(len(names))); a2.set_xticklabels(names, fontsize=8)
for i, (f, m) in enumerate(zip(fps, mem)):
    a2.text(i, f + max(fps) * .02, f"{f:.0f} fps\n{m:.2f} GB", ha="center", fontsize=7)
plt.tight_layout()
out = d / "suite_plots.png"; plt.savefig(out, dpi=115); print("WROTE", out)
