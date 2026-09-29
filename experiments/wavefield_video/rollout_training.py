"""Supervised self-rollout for train_long; not diffusion/DMD Self-Forcing++.

Ground truth supplies a prefix and targets, never a rollout input. This module
is opt-in: train_long's historical teacher-forced loss is left untouched.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import train_compare as tc


def training_objective(a):
    """Identity stored only for rollout checkpoints; old default files stay valid."""
    if not getattr(a, 'rollout_k', 0):
        return None
    identity = dict(version='supervised_self_rollout_v1', rollout_k=a.rollout_k,
                    seq_frames=a.seq_frames, chunk=a.chunk, tbptt_chunks=a.tbptt_chunks,
                    dense=a.dense, motion_loss=a.motion_loss, feedback='clamp_0_1')
    if a.motion_loss:
        # The moving/static means pool per micro-batch; changing that partition
        # changes the objective (unlike plain MSE's exact accumulation).
        identity.update(batch=a.batch, micro_batch=min(a.micro_batch or a.batch, a.batch))
    return identity


def check_resume_objective(checkpoint_data, a):
    """Do not silently turn a teacher-forced resume into a rollout experiment."""
    expected = training_objective(a)
    previous = checkpoint_data.get('training_objective')
    if previous != expected:
        raise ValueError('resume training objective mismatch; use the original rollout/sequence/'
                         'TBPTT/loss settings or a fresh OUT (legacy checkpoints are teacher-forced)')


def _step(m, frame, states, index):
    # stream_step mutates its state LIST. Give each forward/recomputation its own
    # list; checkpoint must never replay against a list changed by a later step.
    def forward(x, *st):
        pred, new = m.stream_step(x, list(st), index)
        return (pred, *new)

    if getattr(m, 'grad_ckpt', False) and torch.is_grad_enabled():
        result = checkpoint(forward, frame, *states, use_reentrant=False)
    else:
        result = forward(frame, *states)
    return result[0], list(result[1:])


def rollout_sequence_loss(m, clips, moving, a, scale=1.0):
    """Same supervision/normalization as sequence_loss, different conditioning.

    --rollout-k K reserves the FINAL K chunks for autoregressive generation;
    at least one preceding chunk is real context. Within each TBPTT group the
    feedback frames AND recurrent states remain differentiable. At a group
    boundary both detach, bounding the live graph by G * chunk frames, not K.
    No optimizer step happens here (micro-batches accumulate in train_long).
    """
    B, T = clips.shape[0], a.chunk
    n_chunks = a.seq_frames // T
    if not 0 < a.rollout_k < n_chunks:
        raise ValueError('rollout needs K >= 1 and at least one ground-truth context chunk')
    if clips.shape[1] != a.seq_frames + 1:
        raise ValueError('rollout clips must contain seq_frames + 1 frames')
    context_chunks = n_chunks - a.rollout_k
    states = [None] * len(m.blocks)
    frame = None
    group, total = 0.0, 0.0
    for c in range(n_chunks):
        start = c * T
        if c < context_chunks:
            pred, states = m(clips[:, start:start + T], states=states, dense=a.dense)
            frame = (pred[:, -1] if a.dense else pred).float().clamp(0, 1)
        else:
            generated = []
            for j in range(T):
                pred, states = _step(m, frame, states, start + j)
                # Only this feedback operation conditions the next frame. Never
                # replace it with clips[:, start+j+1] (the mutation test kills that).
                frame = pred.float().clamp(0, 1)
                if a.dense:
                    generated.append(pred)
            if a.dense:
                pred = torch.stack(generated, dim=1)

        if a.dense:
            tgt = clips[:, start + 1:start + T + 1]
            last = clips[:, start:start + T]
            mv = moving[:, start:start + T] if moving is not None else None
            pred, tgt, last = (t.reshape(B * T, *t.shape[2:]) for t in (pred, tgt, last))
            mv = mv.reshape(B * T, *mv.shape[2:]) if mv is not None else None
        else:
            tgt, last = clips[:, start + T], clips[:, start + T - 1]
            mv = moving[:, start + T - 1] if moving is not None else None
        pred = pred.float()  # raw prediction is supervised, clamp is feedback only
        lc = tc.motion_balanced_loss(pred, tgt, last, mv) if a.motion_loss else F.mse_loss(pred, tgt)
        group = group + lc * (scale / n_chunks)
        if (c + 1) % a.tbptt_chunks == 0 or c == n_chunks - 1:
            group.backward()
            total += float(group.detach())
            group = 0.0
            states = [s.detach() for s in states]
            frame = frame.detach()
    return total
