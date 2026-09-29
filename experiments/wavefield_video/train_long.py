"""Stateful chunked training: learn dependencies longer than one clip (LONG_HORIZON.md phase 1).

train_compare.py trains every clip from a zero state and supervises only its last
frame, so the model can never be trained on anything longer than the clip that
fits in memory. Here a long sequence of --seq-frames frames is walked in
--chunk-frame chunks; each block's recurrent state is carried from chunk to
chunk (VideoPredictor.forward(states=...)), and the loss is taken at every
position (dense=True). Gradients flow through the carried state for
--tbptt-chunks chunks, then it is detached (truncated BPTT):

  G = 1        constant memory: one chunk's graph, however long the sequence
  G = N/chunk  full backprop through the whole sequence (the upper bound)

Chunked == one long forward == stream_step, exactly (tests/test_stateful.py),
so a model trained this way streams with the same state it trained with.
time_pos="none" is used so no length-T position table limits the horizon.

Evaluation streams through the carried state: stream_rollout_eval (balls) and
train_compare's occlusion_rollout_eval (occlusion), with --chunk frames of
context. Result JSON keeps train_compare's field names.

    python train_long.py --data-source occlusion --grid 16 --seq-frames 256 --chunk 64 \\
        --tbptt-chunks 1 --dense --train-occ-start 96 --train-occ-end 160 \\
        --occ-start 96 --occ-end 160 --eval-rollout 128     # emergence at 160-64 = 96 < 128
"""
import argparse
import json
import os
import pathlib
import signal
import time

import torch
import torch.nn.functional as F

import data_occlusion as _occ
import train_compare as tc
from data import MOVE_THRESH, N_BALLS, RADIUS, SPEED
from video_vae import LatentShards
from wfvideo import VideoPredictor


def _save_atomic(obj, path):
    tmp = pathlib.Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def arch_kwargs(a, in_ch=3):
    """VideoPredictor keyword arguments; also saved as checkpoint metadata so
    render_rollout.load_model / eval_only.py rebuild the same model."""
    kw = dict(causal=True, residual=True, ffn_mult=a.ffn_mult, time_pos="none", in_ch=in_ch)
    if a.kind == "wave":
        kw.update(kernel_version="dispersion", linear_pad=True, pole_param=a.pole_param,
                  hl_min=a.hl_min, hl_max=a.hl_max, write_gate=a.write_gate,
                  clean_write=a.clean_write)
    elif a.write_gate or a.clean_write:
        raise SystemExit("--write-gate/--clean-write apply to --kind wave")
    return kw


def build(a, in_ch=3, H=None, W=None):
    return VideoPredictor(a.dim, a.layers, a.heads, a.chunk, H or a.grid, W or a.grid, a.kind,
                          **arch_kwargs(a, in_ch))


def model_config(a, data=None):
    """render_rollout-compatible config: frames is the chunk (the model's T).
    Latent runs record in_ch, the latent grid and data_source="latents"; the pixel
    renderer refuses those (long_horizon.py stream --latents decodes them)."""
    cfg = dict(kernel_version="separable", linear_pad=False, gate=False, local_fuse=False)
    cfg.update(arch_kwargs(a, data.C if data is not None else 3))
    cfg.update(kind=a.kind, dim=a.dim, layers=a.layers, heads=a.heads, frames=a.chunk,
               grid=a.grid, kicks=a.kicks, collisions=a.collisions, radius=a.radius,
               speed=a.speed, n_balls=a.n_balls, data_source=a.data_source)
    if data is not None:        # vae (with its weights fingerprint) ties the model to a latent space
        cfg.update(grid=data.h, grid_w=data.w, data_source="latents", vae=data.vae)
    return cfg


def latent_eval(m, data, a):
    """Teacher-forced next-latent MSE vs copy-last on held-out shards, plus an
    autoregressive rollout (stream_step from one chunk of real context)."""
    gen = torch.Generator().manual_seed(90000)
    # the longest held-out shard sets the horizon: batch() samples only shards long
    # enough for the requested window, so one short tail must not cap it
    R = min(a.eval_rollout, max(z.shape[0] for z in data.eval) - a.chunk)
    with torch.no_grad():
        x = data.batch(a.eval_batch, a.chunk + 1, gen, "eval")
        p = m(x[:, :a.chunk], states=[None] * len(m.blocks))[0].float()   # exact kernel
        se, cb = F.mse_loss(p, x[:, -1]).item(), F.mse_loss(x[:, -2], x[:, -1]).item()
        out = {"eval_mse": round(se, 6), "eval_mse_over_copylast": round(se / (cb + 1e-9), 4),
               "latent_rollout_steps": max(R, 0)}
        if R >= 1:
            clip = data.batch(a.eval_batch, a.chunk + R, gen, "eval")
            st = m.stream_init(clip.shape[0], clip.device)
            f = None
            for t in range(a.chunk):
                f, st = m.stream_step(clip[:, t], st, t)
            curve, copy = [], []
            for k in range(R):
                gt = clip[:, a.chunk + k] if a.chunk + k < clip.shape[1] else None
                if gt is None:
                    break
                curve.append(F.mse_loss(f.float(), gt).item())
                copy.append(F.mse_loss(clip[:, a.chunk - 1], gt).item())
                f, st = m.stream_step(f, st, a.chunk + k)
            out.update(latent_rollout_mse=[round(v, 5) for v in curve],
                       latent_rollout_copy_mse=[round(v, 5) for v in copy])
    return out


def detach_states(states):
    return [s.detach() if torch.is_tensor(s) else s for s in states]


def sequence_loss(m, clips, moving, a, scale=1.0):
    """Walk one [B, N+1, 3, H, W] batch in chunks with carried state. Returns the
    mean per-chunk loss (a float) after calling backward() every G chunks; the
    loss is multiplied by scale (micro-batch share of the full batch)."""
    B, n_chunks, T = clips.shape[0], a.seq_frames // a.chunk, a.chunk
    states = [None] * len(m.blocks)
    group, total = 0.0, 0.0
    for c in range(n_chunks):
        x = clips[:, c * T:(c + 1) * T]
        if a.dense:
            pred, states = m(x, states=states, dense=True)            # [B,T,3,H,W]
            tgt, last = clips[:, c * T + 1:(c + 1) * T + 1], x
            mv = moving[:, c * T:(c + 1) * T] if moving is not None else None
            pred, tgt, last = (t.reshape(B * T, *t.shape[2:]) for t in (pred, tgt, last))
            mv = mv.reshape(B * T, *mv.shape[2:]) if mv is not None else None
        else:
            pred, states = m(x, states=states)                         # [B,3,H,W]
            tgt, last = clips[:, (c + 1) * T], x[:, -1]
            mv = moving[:, (c + 1) * T - 1] if moving is not None else None
        pred = pred.float()
        if a.motion_loss:
            lc = tc.motion_balanced_loss(pred, tgt, last, mv)
        else:
            lc = F.mse_loss(pred, tgt)
        group = group + lc * (scale / n_chunks)
        if (c + 1) % a.tbptt_chunks == 0 or c == n_chunks - 1:
            group.backward()
            total += float(group.detach())
            group = 0.0
            states = detach_states(states)
    return total


def stream_rollout_eval(m, a, dev):
    """Ball rollout through stream_step, so the carried state IS what is evaluated
    (train_compare.rollout_eval re-runs m(window) from a zero state every frame,
    which would score a chunk-window model, not the carried-state one). Warm on
    a.frames real frames, then roll a.eval_rollout frames on the model's own output.
    Same metric names/definitions as rollout_eval."""
    S, R = a.eval_seeds, a.eval_rollout
    clip, meta = tc.make_clip_batch(S, a.frames + R - 1, a.grid, a.grid, device=dev, seed=90000,
                                    kicks=a.kicks, collisions=a.collisions, radius=a.radius,
                                    speed=a.speed, nb=a.n_balls, return_meta=True)
    cols = meta["col"]
    mse, persist, cerrs = [0.0] * R, [0.0] * R, []
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, S, max(1, a.eval_chunk)):
            c1 = min(S, c0 + max(1, a.eval_chunk))
            cb = c1 - c0
            st, f = m.stream_init(cb, dev), None
            for t in range(a.frames):
                f, st = m.stream_step(clip[c0:c1, t], st, t)
            ce = []
            for k in range(R):
                f = f.float().clamp(0, 1)
                gt = clip[c0:c1, a.frames + k]
                mse[k] += F.mse_loss(f, gt).item() * cb
                persist[k] += F.mse_loss(clip[c0:c1, a.frames + k - 1], gt).item() * cb
                pc = tc.centroids_by_color(f, cols[c0:c1])
                ce.append((pc - meta["pos"][c0:c1, a.frames + k]).norm(dim=-1))
                f, st = m.stream_step(f, st, a.frames + k)
            cerrs.append(torch.stack(ce, 0))                         # [R, cb, nb]
    mse = [x / S for x in mse]
    persist = [x / S for x in persist]
    cerr = torch.cat(cerrs, 1)
    med = cerr.reshape(R, -1).median(dim=1).values
    return {"eval_rollout": R, "eval_seeds": S, "rollout_path": "stream_step (carried state)",
            "r_persist_over_model": round(sum(persist) / (sum(mse) + 1e-9), 3),
            "div_thresh_px": round(a.radius, 3), "div_consec": tc.DIV_CONSEC,
            "rollout_fps": round(S * R / (time.time() - t0), 1),
            "rollout_mse_curve": [round(x, 5) for x in mse],
            "mean_centroid_err": round(cerr.mean().item(), 3),
            "final_centroid_err": round(cerr[-1].mean().item(), 3),
            "identity_survival": round((cerr[-1] < 2.0 * a.radius).float().mean().item(), 3),
            "divergence_horizon": tc.divergence_horizon(med, a.radius, tc.DIV_CONSEC)}


def pixel_eval(m, a, dev):
    """Single-step eval + copy-last baseline (train_compare.main()'s definitions),
    then train_compare's rollout / occlusion eval."""
    if a.data_source == "occlusion":
        # eval window (starts after the context and its target frame), so the
        # single-step target is ordinary motion, as in train_compare's arms
        _occ.OCC_START, _occ.OCC_END = a.occ_start, a.occ_end
    with torch.no_grad():
        se = cb = cr = 0.0
        n_eval = 8
        for i in range(n_eval):
            clips = tc.make_clip_batch(a.eval_batch, a.frames, a.grid, a.grid, device=dev, seed=90000 + i,
                                       kicks=a.kicks, collisions=a.collisions, radius=a.radius,
                                       speed=a.speed, nb=a.n_balls)
            ctx, tgt = clips[:, :a.frames], clips[:, a.frames]
            # stateful path with a zero state = the exact truncated kernel training uses
            p, last = m(ctx, states=[None] * len(m.blocks))[0].float(), ctx[:, -1]
            se += F.mse_loss(p, tgt).item()
            cb += F.mse_loss(last, tgt).item()
            cr += ((p - last).norm() / ((tgt - last).norm() + 1e-9)).item()
    single = {"eval_mse": round(se / n_eval, 6), "eval_mse_over_copylast": round(se / (cb + 1e-9), 4),
              "copy_ratio": round(cr / n_eval, 4)}
    if a.data_source == "occlusion":
        return single, tc.occlusion_rollout_eval(m, a, dev)
    return single, stream_rollout_eval(m, a, dev)


# Flags a --resume may change: the step target, where/how often it saves, exact
# recompute (grad ckpt), eval-only knobs (the eval runs on the final model with the
# flags it reports) and the --latents PATH (its content is checked by the dataset
# fingerprint). Every other flag shapes training and must match.
RESUME_FREE = {"steps", "resume", "out", "tag", "save_every", "save_every_sec", "grad_ckpt",
               "eval_batch", "eval_rollout", "eval_seeds", "eval_chunk", "latents"}


def train_args(a):
    return json.loads(json.dumps({k: v for k, v in sorted(vars(a).items())
                                  if k not in RESUME_FREE}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-source", choices=["balls", "occlusion"], default="balls")
    ap.add_argument("--latents", default="", help="encode_videos.py shard dir: train on video latents")
    ap.add_argument("--kind", choices=["wave", "ssm"], default="wave")
    ap.add_argument("--pole-param", choices=["softplus", "halflife"], default="halflife")
    ap.add_argument("--hl-min", type=float, default=2.0)
    ap.add_argument("--hl-max", type=float, default=4096.0)
    ap.add_argument("--seq-frames", type=int, default=64, help="frames per training sequence")
    ap.add_argument("--chunk", type=int, default=16, help="frames per chunk (= model window T)")
    ap.add_argument("--tbptt-chunks", type=int, default=1,
                    help="chunks of gradient through the carried state before detaching")
    ap.add_argument("--dense", action=argparse.BooleanOptionalAction, default=False,
                    help="next-frame loss at every position vs last frame only (default). "
                         "Opt-in per LONG_HORIZON.md 8.2: -3..-4%% eval MSE/copy-last and 2x "
                         "copy-ratio on 2/2 seeds, below the pre-registered 10%% bar")
    ap.add_argument("--motion-loss", action="store_true")
    ap.add_argument("--write-gate", action="store_true",
                    help="learned gate on what enters the wave state (LONG_HORIZON.md 8.4)")
    ap.add_argument("--clean-write", action="store_true",
                    help="blank input writes nothing: no spatial table, no embed/pi bias")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-balls", type=int, default=N_BALLS)
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--kicks", action="store_true")
    ap.add_argument("--collisions", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--train-occ-start", type=int, default=None)
    ap.add_argument("--train-occ-end", type=int, default=None)
    ap.add_argument("--occ-start", type=int, default=_occ.OCC_START)
    ap.add_argument("--occ-end", type=int, default=_occ.OCC_END)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-rollout", type=int, default=64)
    ap.add_argument("--eval-seeds", type=int, default=16)
    ap.add_argument("--eval-chunk", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=8,
                    help="single-step eval batch (8 x 8 batches); keep small on the GPU box")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--save-every-sec", type=float, default=0,
                    help="also checkpoint when this many seconds passed since the last save "
                         "(sliced jobs: keep it well under the slice budget)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="runs_long")
    ap.add_argument("--tag", default="")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split --batch into micro-batches of this size and accumulate "
                         "gradients (0 = whole batch at once). Exact for MSE; with "
                         "--motion-loss the moving/static means pool per micro-batch")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="recompute each block's activations in backward (keeps one layer's "
                         "FFT spectra live at a time; needed for chunk 128 x grid 32 on 24 GB)")
    a = ap.parse_args(argv)
    if a.data_source == "occlusion" and a.occ_end - a.chunk + 8 > a.eval_rollout:
        ap.error(f"eval emergence at frame {a.occ_end - a.chunk} of the rollout needs "
                 f"--eval-rollout >= {a.occ_end - a.chunk + 8} (got {a.eval_rollout}); "
                 "set --occ-start/--occ-end so the gap ends inside the rollout")
    if a.data_source == "occlusion" and a.occ_start <= a.chunk:
        ap.error(f"--occ-start {a.occ_start} <= --chunk {a.chunk}: the eval gap would cover the "
                 "warm-up context or the single-step target frame")
    if a.seq_frames % a.chunk:
        ap.error("--seq-frames must be a multiple of --chunk")
    if a.tbptt_chunks < 1:
        ap.error("--tbptt-chunks must be >= 1")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tc.DATA_SOURCE = a.data_source
    if a.data_source == "occlusion":
        a.train_occ_start = a.occ_start if a.train_occ_start is None else a.train_occ_start
        a.train_occ_end = a.occ_end if a.train_occ_end is None else a.train_occ_end
        if not 0 < a.train_occ_start < a.train_occ_end <= a.seq_frames:
            ap.error(f"training gap [{a.train_occ_start}, {a.train_occ_end}) must lie inside the "
                     f"{a.seq_frames}-frame sequence with the ball re-emerging by its last frame; "
                     "set --train-occ-start/--train-occ-end")
        _occ.OCC_START, _occ.OCC_END = a.train_occ_start, a.train_occ_end
    data = None
    if a.latents:
        if a.motion_loss:
            ap.error("--motion-loss uses a pixel moving mask; latents train with plain MSE")
        data = LatentShards(a.latents, a.seq_frames + 1, device=dev)
        m = build(a, in_ch=data.C, H=data.h, W=data.w).to(dev)
        dgen = torch.Generator().manual_seed(1000 + 100_000 * a.seed)
    else:
        m = build(a).to(dev)
    m.grad_ckpt = a.grad_ckpt
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
    base = pathlib.Path(a.out)
    base.mkdir(parents=True, exist_ok=True)
    ckpt_path = base / f"ckpt_{a.kind}{a.tag}.pt"
    start, prior_sec = 0, 0.0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=True)
        if data is not None:        # same latent space and dataset, or refuse (fail closed)
            fp_ck = (ck.get("vae") or {}).get("fingerprint")
            fp_now = (data.vae or {}).get("fingerprint")
            if fp_ck is None or ck.get("data_fp") is None or fp_now is None:
                ap.error(f"--resume {a.resume} / --latents {a.latents}: missing VAE or dataset "
                         "fingerprint (written before fingerprints); cannot verify, start fresh")
            if fp_ck != fp_now:
                ap.error(f"--resume {a.resume} was trained on VAE {fp_ck}; --latents {a.latents} "
                         f"was encoded with {fp_now}")
            if ck["data_fp"] != data.fingerprint:
                ap.error(f"--resume {a.resume} was trained on a different latent dataset "
                         f"({ck['data_fp']} != {data.fingerprint} for --latents {a.latents})")
        if "args" not in ck:
            ap.error(f"--resume {a.resume} records no training arguments (written before "
                     "they were saved); cannot verify it continues this run, start fresh")
        now = train_args(a)
        diff = sorted(k for k in set(ck["args"]) | set(now) if ck["args"].get(k) != now.get(k))
        if diff:
            ap.error(f"--resume {a.resume} was trained with different " + ", ".join(
                f"--{k.replace('_', '-')} ({ck['args'].get(k)!r} -> {now.get(k)!r})" for k in diff)
                + f"; only {', '.join(sorted(RESUME_FREE))} may change on resume")
        m.load_state_dict(ck["state"])
        if data is not None and "dgen" in ck:            # continue the latent sample stream
            dgen.set_state(ck["dgen"].cpu())
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        prior_sec = float(ck.get("train_sec", 0.0))
        print(f"RESUME <- {a.resume} at step={start} (of {a.steps})", flush=True)

    def resume_meta():          # latent runs: sampler position + what it samples from
        return {"dgen": dgen.get_state(), "vae": data.vae, "data_fp": data.fingerprint}

    def save_ckpt(step):        # atomic: a kill mid-write leaves the previous file intact
        _save_atomic({"state": m.state_dict(), "opt": opt.state_dict(), "step": step,
                      "train_sec": prior_sec + time.time() - t0, "args": train_args(a),
                      **(resume_meta() if data is not None else {})}, ckpt_path)

    # A slice budget ends with SIGTERM (GNU timeout): finish the current step, save,
    # exit 143 so the runner reports SLICED and the next slice resumes from here.
    stop = []
    if a.save_every or a.save_every_sec:
        signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    m.train()
    log, t0 = [], time.time()
    last_save = t0
    for step in range(start + 1, a.steps + 1):
        if data is not None:
            clips, moving = data.batch(a.batch, a.seq_frames + 1, dgen), None
        else:
            gen = tc.make_clip_batch(a.batch, a.seq_frames, a.grid, a.grid, device=dev,
                                     seed=1000 + step + 100_000 * a.seed, kicks=a.kicks,
                                     collisions=a.collisions, radius=a.radius, speed=a.speed,
                                     nb=a.n_balls, return_moving=a.motion_loss,
                                     move_thresh=MOVE_THRESH)
            clips, moving = gen if a.motion_loss else (gen, None)
        opt.zero_grad(set_to_none=True)
        loss, mb = 0.0, a.micro_batch or a.batch            # gradient accumulation
        for i in range(0, a.batch, mb):
            loss += sequence_loss(m, clips[i:i + mb], moving[i:i + mb] if moving is not None
                                  else None, a, scale=clips[i:i + mb].shape[0] / a.batch)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % max(1, a.steps // 10) == 0 or step == a.steps:
            row = {"step": step, "loss": round(loss, 6),                 # steps THIS slice ran
                   "st_s": round((step - start) / max(time.time() - t0, 1e-9), 3)}
            log.append(row)
            print("STEP", json.dumps(row), flush=True)
        due = (a.save_every and step % a.save_every == 0) or \
              (a.save_every_sec and time.time() - last_save >= a.save_every_sec)
        if due or stop:
            save_ckpt(step)
            last_save = time.time()
        if stop:
            print(f"SIGTERM: saved step {step} to {ckpt_path}; exiting to resume", flush=True)
            raise SystemExit(143)
    train_sec = prior_sec + time.time() - t0              # cumulative across resumed slices
    signal.signal(signal.SIGTERM, signal.SIG_DFL)         # evaluation: plain termination
    if a.save_every or a.save_every_sec:
        save_ckpt(a.steps)

    m.eval()
    a.frames = a.chunk                                   # eval context = one chunk
    if data is not None:
        single, roll = latent_eval(m, data, a), {}
    else:
        single, roll = pixel_eval(m, a, dev)
    cfg = model_config(a, data)
    res = {**cfg, "pole_param": a.pole_param if a.kind == "wave" else None,
           "seq_frames": a.seq_frames, "chunk": a.chunk, "tbptt_chunks": a.tbptt_chunks,
           "dense": a.dense, "time_pos": "none", "write_gate": a.write_gate,
           "clean_write": a.clean_write, "steps": a.steps, "seed": a.seed,
           "params": sum(p.numel() for p in m.parameters()), "dim": a.dim, "layers": a.layers,
           "heads": a.heads, "batch": a.batch, "micro_batch": a.micro_batch or a.batch,
           "grad_ckpt": a.grad_ckpt,
           "latents": a.latents or None, "vae": data.vae if data is not None else None,
           "latents_fp": data.fingerprint if data is not None else None,
           "motion_loss": a.motion_loss, "train_occ_start": a.train_occ_start,
           "train_occ_end": a.train_occ_end, "occ_start": a.occ_start, "occ_end": a.occ_end,
           "persistent_state_bytes": m.persistent_state_bytes(), "train_sec": round(train_sec, 1),
           "log_tail": log[-3:], **single, **roll}
    # model before result, both atomic: "both files exist" (the runners' done test)
    # then always means both are complete
    _save_atomic({"state": m.state_dict(), "config": cfg}, base / f"model_{a.kind}{a.tag}.pt")
    tmp = base / f"result_{a.kind}{a.tag}.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, base / f"result_{a.kind}{a.tag}.json")
    keys = ("eval_mse_over_copylast", "copy_ratio", "divergence_horizon", "exit_direction_accuracy",
            "position_error_at_emergence")
    print("RESULT", json.dumps({k: res.get(k) for k in keys}), flush=True)
    return res


if __name__ == "__main__":
    main()
