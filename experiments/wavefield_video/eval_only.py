#!/usr/bin/env python3
"""Re-run pixel-space rollout/semantic eval using a checkpoint's scene metadata.
Usage: python eval_only.py model_<tag>.pt result_<tag>.json [out_extra.json]
Both pixel checkpoints and latent predictors + frozen AEs are supported.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import render_rollout as rr
import train_compare as tc
from semantic_metrics import semantic_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--eval-seeds", type=rr.positive_int)
    ap.add_argument("--eval-rollout", type=rr.positive_int)
    ap.add_argument("--eval-chunk", type=rr.positive_int)
    ap.add_argument("--ae-ckpt", type=Path)
    cli = ap.parse_args()
    config = json.loads(cli.config.read_text())
    args = rr.parser().parse_args([
        "--ckpt", str(cli.checkpoint), "--config", str(cli.config),
        "--out", ".", *(["--latent"] if config.get("latent") else [])])
    args.ae_ckpt = cli.ae_ckpt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.latent:
        core, ae, config, *_ = rr.load_latent_model(args, torch)
        model = tc.LatentWrapper(ae, core)
    else:
        model, config, *_ = rr.load_model(args, torch)
    model = model.float().to(dev).eval()
    tc.DATA_SOURCE, tc.WAVE_FIELD = config["data_source"], config["field"]
    a = SimpleNamespace(
        # Preserve this utility's fp32 evaluation (FFT/complex paths need it),
        # without replacing torch.autocast globally as the older script did.
        eval_amp=False,
        eval_seeds=cli.eval_seeds or config.get("eval_seeds", 16),
        eval_rollout=cli.eval_rollout or config.get("eval_rollout", 256),
        eval_chunk=cli.eval_chunk or config.get("eval_chunk", 4),
        **{key: config[key] for key in
           ("grid", "frames", "kicks", "collisions", "radius", "speed", "n_balls")})
    out = tc.rollout_eval(model, a, dev)
    print(config["kind"] + " " + semantic_summary(out["semantic"]))
    print("RESULT " + json.dumps(out, allow_nan=False))
    if cli.output:
        cli.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
