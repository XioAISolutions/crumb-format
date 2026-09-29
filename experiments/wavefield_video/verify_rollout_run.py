"""Read actual paired artifacts; include a FAIR frozen-context rollout baseline.

Never use train_long's legacy r_persist_over_model as a freeze comparison: it
sees fresh ground truth at every step. No quality threshold is asserted here.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from data import make_clip_batch


def verify(out, rollout_k, steps):
    out = Path(out)
    if rollout_k <= 0 or steps <= 0:
        raise ValueError('positive rollout-k and steps required')
    expected = {f'{prefix}_wave_k{k}.{ext}' for k in (0, rollout_k)
                for prefix, ext in (('result', 'json'), ('model', 'pt'), ('ckpt', 'pt'))}
    found = {p.name for prefix in ('result', 'model', 'ckpt')
             for p in out.glob(f'{prefix}_wave_*')}
    if found != expected:
        raise ValueError(f'artifact count/set mismatch: expected {sorted(expected)}, got {sorted(found)}')
    results, states, rows = [], [], []
    for k in (0, rollout_k):
        r = json.loads((out / f'result_wave_k{k}.json').read_text())
        model = torch.load(out / f'model_wave_k{k}.pt', map_location='cpu', weights_only=True)
        ck = torch.load(out / f'ckpt_wave_k{k}.pt', map_location='cpu', weights_only=True)
        assert ck['step'] == r['steps'] == steps, 'incomplete training'
        objective = r.get('training_objective')
        assert model.get('training_objective') == ck.get('training_objective') == objective
        assert (objective is None if k == 0 else objective['rollout_k'] == k)
        assert r['log_tail'] and r['log_tail'][-1]['step'] == steps
        assert all(math.isfinite(row['loss']) for row in r['log_tail']), 'nonfinite loss'
        state = model['state']
        assert state and all(torch.isfinite(v).all() for v in state.values()), 'nonfinite weights'
        assert state.keys() == ck['state'].keys()
        assert all(torch.equal(v, ck['state'][name]) for name, v in state.items()), 'export/ckpt mismatch'
        # Exact eval protocol/seed/geometry from stream_rollout_eval, but freeze
        # ONE last context frame for the entire continuation (no GT refreshing).
        assert r['data_source'] == 'balls', 'this diagnostic implements balls only'
        T, R, S = r['chunk'], r['eval_rollout'], r['eval_seeds']
        data = make_clip_batch(S, T + R - 1, r['grid'], r['grid'], seed=90000,
                               kicks=r['kicks'], collisions=r['collisions'], radius=r['radius'],
                               speed=r['speed'], nb=r['n_balls'])
        frozen = (data[:, T:] - data[:, T-1:T]).square().mean(dim=(0, 2, 3, 4))
        assert len(frozen) == len(r['rollout_mse_curve']) == R
        denom = float(frozen.sum())
        ratio = sum(r['rollout_mse_curve']) / denom if denom > 0 else None
        rows.append(dict(rollout_k=k, loss=r['log_tail'][-1]['loss'],
                         rollout_mse_over_frozen=ratio, frozen_mse_curve=frozen.tolist(),
                         copy_ratio=r['copy_ratio'], identity_survival=r['identity_survival'],
                         divergence_horizon=r['divergence_horizon'], train_sec=r['train_sec'],
                         persistent_state_bytes=r['persistent_state_bytes']))
        results.append(r)
        states.append(state)
    for key in ('seq_frames', 'chunk', 'tbptt_chunks', 'dense', 'motion_loss', 'seed',
                'steps', 'batch', 'micro_batch', 'grid', 'dim', 'layers', 'heads', 'grad_ckpt',
                'eval_rollout', 'eval_seeds', 'radius', 'speed', 'n_balls', 'kicks', 'collisions',
                'params', 'persistent_state_bytes', 'pole_param', 'clean_write', 'write_gate'):
        assert results[0][key] == results[1][key], f'unmatched control: {key}'
    assert states[0].keys() == states[1].keys()
    changed = sum(not torch.equal(v, states[1][name]) for name, v in states[0].items())
    assert changed > 0, 'rollout and teacher updates are identical: path may be disabled'
    receipt = dict(artifact_count=len(found), tensors_compared=len(states[0]),
                   tensors_changed=changed, arms=rows,
                   claim='execution/conditioning check only; no video quality claim')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'arms'}, sort_keys=True))
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out')
    ap.add_argument('--rollout-k', type=int, required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--stream-smoke', action='store_true')
    ap.add_argument('--require-cuda', action='store_true', help='fail if the paired trainer used CPU')
    a = ap.parse_args()
    verify(a.out, a.rollout_k, a.steps)
    if a.require_cuda:
        log_path = Path(a.out) / f'train_k{a.rollout_k}.log'
        configs = [json.loads(line.removeprefix('TRAINING_OBJECTIVE '))
                   for line in log_path.read_text().splitlines()
                   if line.startswith('TRAINING_OBJECTIVE ')]
        assert len(configs) == 1 and configs[0]['device'] == 'cuda', 'CUDA training required'
        print('CUDA trainer receipt verified')
    if a.stream_smoke:
        stream = json.loads((Path(a.out) / 'stream_smoke.json').read_text())
        assert stream['trained'] and stream['frames'] == 48
        assert stream['state_bytes_constant'] and len(stream['log']) == 4
        assert stream['log'][-1]['frame'] == 52  # four context frames + 48 generated
        print('STREAM RECEIPT: 48 generated frames, 4 chunks, constant carried state; '
              f'collapse={stream["collapse"]}')
