"""Opt-in rollout behavior, not import-only tests. Run this file explicitly.

Root CI is format-only. mutation_check_rollout.py must also kill disabled
rollout and teacher-leaking feedback mutants. Nonzero heads avoid copy-init.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_long as tl
from wfvideo import VideoPredictor


def args(**kw):
    values = dict(seq_frames=8, chunk=2, tbptt_chunks=1, rollout_k=2,
                  dense=True, motion_loss=False)
    values.update(kw)
    return argparse.Namespace(**values)


def clips(batch=2):
    return torch.linspace(0.05, 0.35, batch * 9 * 3 * 4 * 4).reshape(batch, 9, 3, 4, 4)


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.07))
        self.blocks = [None]
        self.grad_ckpt = False
        self.calls = []

    def step(self, x, state, mode, t):
        st = torch.zeros_like(x) if state is None else state
        ystate = 0.5 * st + 0.1 * x + 0.01 * self.weight
        pred = x + self.weight + 0.2 * ystate
        self.calls.append(dict(mode=mode, t=t, x=x.detach().clone(),
                               state=st.detach().clone(), pred=pred.detach().clone(),
                               input_grad=x.requires_grad, state_grad=st.requires_grad))
        return pred, ystate

    def forward(self, x, states, dense=False):
        st, out = states[0], []
        for frame in x.unbind(1):
            p, st = self.step(frame, st, 'teacher', len(self.calls))
            out.append(p)
        return (torch.stack(out, 1) if dense else p), [st]

    def stream_step(self, x, states, t_index):
        p, st = self.step(x, states[0], 'rollout', t_index)
        states[0] = st  # Match VideoPredictor's mutable list API.
        return p, states


def test_rollout_consumes_own_predictions_and_carries_state():
    m, data = Recorder(), clips()
    tl.sequence_loss(m, data, None, args())
    assert [c['mode'] for c in m.calls] == ['teacher'] * 4 + ['rollout'] * 4
    for i, call in enumerate(m.calls):
        if i < 4:
            torch.testing.assert_close(call['x'], data[:, i], rtol=0, atol=0)
        else:
            torch.testing.assert_close(call['x'], m.calls[i - 1]['pred'].clamp(0, 1),
                                       rtol=0, atol=0)
            assert not torch.allclose(call['x'], data[:, i]), 'teacher leakage'
            assert call['t'] == i
        if i:
            expected_state = (0.5 * m.calls[i - 1]['state'] + 0.1 * m.calls[i - 1]['x']
                              + 0.01 * m.weight.detach())
            torch.testing.assert_close(call['state'], expected_state)


def test_future_targets_never_feed_rollout_inputs_but_change_gradients():
    data = clips()
    changed = data.clone()
    changed[:, 4:] += 0.21
    a, b = Recorder(), Recorder()
    la = tl.sequence_loss(a, data, None, args())
    lb = tl.sequence_loss(b, changed, None, args())
    assert len(a.calls) == len(b.calls) == 8
    for ca, cb in zip(a.calls, b.calls):
        torch.testing.assert_close(ca['x'], cb['x'], rtol=0, atol=0)
    assert abs(la - lb) > 1e-4
    assert not torch.allclose(a.weight.grad, b.weight.grad)


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('group', [1, 2, 3, 4])
def test_matches_independent_scalar_recurrence_loss_and_gradient(dense, group):
    data, a = clips(), args(dense=dense, tbptt_chunks=group)
    m = Recorder()
    actual = tl.sequence_loss(m, data, None, a)
    w = torch.tensor(0.07, requires_grad=True)
    state, frame, loss, total = torch.zeros_like(data[:, 0]), None, 0.0, 0.0
    for t in range(8):
        x = data[:, t] if t < 4 else frame
        state = state * 0.5 + x * 0.1 + 0.01 * w
        prediction = x + w + state * 0.2
        frame = prediction.clamp(0, 1)
        if dense or t % 2 == 1:
            loss = loss + (prediction - data[:, t + 1]).square().mean() / (8 if dense else 4)
        if (t + 1) % (2 * group) == 0 or t == 7:
            loss.backward()
            total += loss.detach().item()
            loss = 0.0
            state, frame = state.detach(), frame.detach()
    assert actual == pytest.approx(total, abs=1e-7)
    torch.testing.assert_close(m.weight.grad, w.grad, atol=1e-6, rtol=1e-5)


def test_tbptt_detaches_both_feedback_and_state_not_each_frame():
    m = Recorder()
    tl.sequence_loss(m, clips(), None, args())
    assert [c['input_grad'] for c in m.calls[4:]] == [False, True, False, True]
    assert [c['state_grad'] for c in m.calls[4:]] == [False, True, False, True]


def test_motion_loss_uses_true_previous_frames_and_aligned_masks():
    data, m = clips(), Recorder()
    moving = torch.zeros(2, 8, 4, 4, dtype=torch.bool)
    moving[:, ::2, :2] = True
    seen = []
    def loss(pred, target, last, mask):
        seen.append((target.clone(), last.clone(), mask.clone()))
        return (pred - target).square().mean()
    with mock.patch.object(tl.tc, 'motion_balanced_loss', side_effect=loss):
        tl.sequence_loss(m, data, moving, args(motion_loss=True))
    assert len(seen) == 4
    for c, (target, last, mask) in enumerate(seen):
        torch.testing.assert_close(target, data[:, 2*c+1:2*c+3].reshape(4, 3, 4, 4))
        torch.testing.assert_close(last, data[:, 2*c:2*c+2].reshape(4, 3, 4, 4))
        assert torch.equal(mask, moving[:, 2*c:2*c+2].reshape(4, 4, 4))


def wave():
    torch.manual_seed(11)
    m = VideoPredictor(8, 1, 2, 2, 4, 4, 'wave', causal=True, residual=True,
                       kernel_version='dispersion', linear_pad=True,
                       pole_param='halflife', time_pos='none')
    nn.init.normal_(m.head.weight, std=0.03)
    return m


def gradients(m, a, data, mb):
    total = 0.0
    for i in range(0, len(data), mb):
        total += tl.sequence_loss(m, data[i:i+mb], None, a,
                                  scale=len(data[i:i+mb]) / len(data))
    grad = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert torch.isfinite(grad).all() and grad.norm() > 1e-6
    return total, grad


@pytest.mark.parametrize('group', [1, 3])
def test_real_wave_gradients_checkpoint_and_uneven_microbatch_match(group):
    data = torch.cat([clips(), clips(1)], 0)
    loss, grad = gradients(wave(), args(tbptt_chunks=group), data, 3)
    for checkpoint, mb in ((True, 3), (False, 2), (True, 1)):
        model = wave()
        model.grad_ckpt = checkpoint
        other_loss, other_grad = gradients(model, args(tbptt_chunks=group), data, mb)
        assert other_loss == pytest.approx(loss, abs=1e-6)
        torch.testing.assert_close(other_grad, grad, atol=2e-5, rtol=3e-4)


def test_rollout_updates_differ_from_teacher_forcing_on_real_wave():
    _, teacher_grad = gradients(wave(), args(rollout_k=0), clips(), 2)
    _, rollout_grad = gradients(wave(), args(), clips(), 2)
    assert (teacher_grad - rollout_grad).norm() > 1e-4


def test_explicit_zero_equals_legacy_missing_flag_bitwise():
    a = args(rollout_k=0)
    b = copy.copy(a)
    del b.rollout_k
    loss, grad = gradients(wave(), a, clips(), 2)
    old_loss, old_grad = gradients(wave(), b, clips(), 2)
    assert old_loss == loss
    assert torch.equal(old_grad, grad)


@pytest.mark.parametrize('extra,reason', [
    (['--rollout-k', '-1'], '--rollout-k must be >= 0'),
    (['--rollout-k', '4'], 'at least one ground-truth context chunk'),
    (['--chunk', '0'], '--chunk and --seq-frames must be positive'),
])
def test_invalid_rollout_cli_fails_before_model_or_data(extra, reason, capsys):
    with mock.patch.object(tl, 'build', side_effect=AssertionError('model built too soon')):
        with pytest.raises(SystemExit) as error:
            tl.main(['--seq-frames', '8', '--chunk', '2'] + extra)
    assert error.value.code == 2
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize('k,chunk', [(1, 2), (3, 2), (1, 1), (7, 1)])
def test_boundary_horizons_have_exactly_k_generated_chunks(k, chunk):
    m = Recorder()
    tl.sequence_loss(m, clips(), None, args(rollout_k=k, chunk=chunk))
    assert sum(c['mode'] == 'rollout' for c in m.calls) == k * chunk
    assert len(m.calls) == 8


@pytest.mark.parametrize('value,clamped', [(-2.0, 0.0), (2.0, 1.0)])
def test_feedback_clamps_but_loss_supervises_raw_predictions(value, clamped):
    m = Recorder()
    m.weight.data.fill_(value)
    seen = []
    def loss(pred, target):
        seen.append(pred.detach().clone())
        return (pred - target).square().mean()
    with mock.patch('rollout_training.F.mse_loss', side_effect=loss):
        tl.sequence_loss(m, clips(), None, args())
    assert len(seen) == 4
    for c in m.calls[4:]:
        assert torch.equal(c['x'], torch.full_like(c['x'], clamped))
    assert all((p < 0).all() if value < 0 else (p > 1).all() for p in seen)


@pytest.mark.parametrize('kind', ['wave', 'ssm'])
@pytest.mark.parametrize('dense', [False, True])
def test_two_layer_checkpoint_replay_preserves_loss_and_gradient(kind, dense):
    torch.manual_seed(8)
    m = VideoPredictor(8, 2, 2, 2, 4, 4, kind, causal=True, residual=True,
                       kernel_version='dispersion', linear_pad=True,
                       pole_param='halflife' if kind == 'wave' else 'softplus', time_pos='none')
    nn.init.normal_(m.head.weight, std=0.02)
    other = copy.deepcopy(m)
    other.grad_ckpt = True
    expected, eg = gradients(m, args(tbptt_chunks=3, dense=dense), clips(), 2)
    actual, ag = gradients(other, args(tbptt_chunks=3, dense=dense), clips(), 2)
    assert expected == pytest.approx(actual, abs=1e-6)
    torch.testing.assert_close(eg, ag, atol=2e-5, rtol=3e-4)


def test_resume_objective_guard_in_both_directions_and_motion_partition():
    from rollout_training import check_resume_objective, training_objective
    a = args(batch=3, micro_batch=2, motion_loss=True)
    ck = {'training_objective': training_objective(a)}
    check_resume_objective(ck, a)
    for changed in (args(rollout_k=0), args(batch=3, micro_batch=1, motion_loss=True),
                    args(batch=3, micro_batch=2, motion_loss=True, tbptt_chunks=2)):
        with pytest.raises(ValueError, match='resume training objective mismatch'):
            check_resume_objective(ck, changed)
    with pytest.raises(ValueError, match='legacy checkpoints are teacher-forced'):
        check_resume_objective({}, a)
    check_resume_objective({}, args(rollout_k=0))
    # MSE pooling is exact and need not freeze the micro-batch partition.
    check_resume_objective({'training_objective': training_objective(args(micro_batch=1))},
                           args(micro_batch=2))


def test_real_cli_resume_equals_uninterrupted_and_guard_runs(tmp_path, capsys):
    base = ['--seq-frames', '8', '--chunk', '2', '--grid', '4', '--dim', '8',
            '--layers', '1', '--heads', '2', '--batch', '2', '--micro-batch', '1',
            '--dense', '--grad-ckpt', '--eval-rollout', '4', '--eval-seeds', '2',
            '--eval-chunk', '1', '--eval-batch', '1', '--save-every', '1']
    full, split = tmp_path / 'full', tmp_path / 'split'
    tl.main(base + ['--rollout-k', '2', '--steps', '3', '--out', str(full)])
    tl.main(base + ['--rollout-k', '2', '--steps', '1', '--out', str(split)])
    ck = split / 'ckpt_wave.pt'
    tl.main(base + ['--rollout-k', '2', '--steps', '3', '--out', str(split), '--resume', str(ck)])
    f = torch.load(full / 'model_wave.pt', weights_only=True)
    s = torch.load(split / 'model_wave.pt', weights_only=True)
    assert f['training_objective'] == s['training_objective']
    assert f['state'] and f['state'].keys() == s['state'].keys()
    assert all(torch.equal(f['state'][k], s['state'][k]) for k in f['state'])
    for k in ('0', '1'):
        with pytest.raises(SystemExit) as error:
            tl.main(base + ['--rollout-k', k, '--steps', '4', '--out', str(split),
                            '--resume', str(ck)])
        assert error.value.code == 2
        assert 'resume training objective mismatch' in capsys.readouterr().err


@pytest.mark.parametrize('value', [-2.0, 2.0])
@pytest.mark.parametrize('dense', [False, True])
def test_signed_latent_feedback_matches_unclamped_recurrence_and_gradient(value, dense):
    data, a = clips() + value, args(latents='signed-shards', dense=dense)
    m = Recorder()
    actual = tl.sequence_loss(m, data, None, a)
    w = torch.tensor(0.07, requires_grad=True)
    state, frame, loss, total = torch.zeros_like(data[:, 0]), None, 0.0, 0.0
    for t in range(8):
        x = data[:, t] if t < 4 else frame
        state = state * 0.5 + x * 0.1 + 0.01 * w
        prediction = x + w + state * 0.2
        frame = prediction  # A VAE latent is signed and has no pixel interval.
        if dense or t % 2 == 1:
            loss = loss + (prediction - data[:, t + 1]).square().mean() / (8 if dense else 4)
        if (t + 1) % 2 == 0:
            loss.backward()
            total += loss.detach().item()
            loss = 0.0
            state, frame = state.detach(), frame.detach()
    for i, call in enumerate(m.calls[4:], start=4):
        torch.testing.assert_close(call['x'], m.calls[i - 1]['pred'], rtol=0, atol=0)
        assert (call['x'] < 0).all() if value < 0 else (call['x'] > 1).all()
    assert actual == pytest.approx(total, abs=1e-6)
    torch.testing.assert_close(m.weight.grad, w.grad, atol=1e-6, rtol=1e-5)


def test_latent_feedback_identity_rejects_pixel_and_old_clamped_objectives():
    from rollout_training import check_resume_objective, training_objective
    pixel = training_objective(args())
    assert pixel == dict(version='supervised_self_rollout_v1', rollout_k=2,
                         seq_frames=8, chunk=2, tbptt_chunks=1, dense=True,
                         motion_loss=False, feedback='clamp_0_1')
    latent_args = args(latents='signed-shards')
    latent = training_objective(latent_args)
    assert latent['representation'] == 'vae_latents'
    assert latent['feedback'] == 'unclamped_float32'
    assert latent['version'] != pixel['version']
    check_resume_objective({'training_objective': latent}, latent_args)
    for checkpoint, current in [(pixel, latent_args), (latent, args())]:
        with pytest.raises(ValueError, match='resume training objective mismatch'):
            check_resume_objective({'training_objective': checkpoint}, current)


def test_real_signed_latent_cli_saves_policy_and_resumes_exactly(tmp_path, capsys):
    root = tmp_path / 'latents'
    root.mkdir()
    index = []
    for i in range(2):
        z = torch.randn(12, 2, 4, 4, generator=torch.Generator().manual_seed(i)) * 2
        torch.save({'latents': z}, root / f's{i}.pt')
        index.append(dict(file=f's{i}.pt', src=f'clip{i}.mp4', latent_steps=12,
                          vae={'backend': 'test', 'fingerprint': 'signed-test-vae'},
                          sha256=hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]))
    (root / 'index.json').write_text(json.dumps(index))
    base = ['--latents', str(root), '--seq-frames', '8', '--chunk', '2', '--dim', '8',
            '--layers', '1', '--heads', '2', '--batch', '2', '--micro-batch', '1',
            '--dense', '--grad-ckpt', '--eval-rollout', '4', '--eval-batch', '1',
            '--save-every', '1', '--rollout-k', '2']
    full, split = tmp_path / 'full', tmp_path / 'split'
    with mock.patch.object(torch.cuda, 'is_available', return_value=False):
        tl.main(base + ['--steps', '3', '--out', str(full)])
        tl.main(base + ['--steps', '1', '--out', str(split)])
        ck_path = split / 'ckpt_wave.pt'
        tl.main(base + ['--steps', '3', '--out', str(split), '--resume', str(ck_path)])
        f = torch.load(full / 'model_wave.pt', weights_only=True)
        s = torch.load(split / 'model_wave.pt', weights_only=True)
        ck = torch.load(ck_path, weights_only=True)
        result = json.loads((split / 'result_wave.json').read_text())
        assert f['training_objective'] == s['training_objective'] == ck['training_objective'] == result['training_objective']
        assert s['training_objective']['feedback'] == 'unclamped_float32'
        assert s['config']['data_fp'] == ck['data_fp'] == result['latents_fp']
        assert s['config']['vae']['fingerprint'] == ck['vae']['fingerprint'] == 'signed-test-vae'
        assert all(torch.equal(f['state'][k], s['state'][k]) for k in f['state'])
        assert all(torch.isfinite(v).all() for v in s['state'].values())
        # The previously accepted latent/clamped objective must never silently resume.
        from rollout_training import training_objective
        ck['training_objective'] = training_objective(args())
        torch.save(ck, ck_path)
        with pytest.raises(SystemExit) as error:
            tl.main(base + ['--steps', '4', '--out', str(split), '--resume', str(ck_path)])
        assert error.value.code == 2
        assert 'resume training objective mismatch' in capsys.readouterr().err
