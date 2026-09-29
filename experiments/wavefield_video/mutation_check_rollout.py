"""Prove rollout tests catch disabled rollout / teacher leakage / lost state.

Mutate disposable copies only, never the working tree. Fails closed unless the
real code passes and EVERY mutant fails the named behavioral assertion (not an
import error, missing fixture, or arbitrary exception).
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
TEST = 'test_rollout_consumes_own_predictions_and_carries_state'
MUTANTS = [
    ('disabled_dispatch', 'train_long.py',
     'if getattr(a, "rollout_k", 0):', 'if False:'),
    ('teacher_feedback', 'rollout_training.py',
     '_step(m, frame, states, start + j)', '_step(m, clips[:, start + j], states, start + j)'),
    ('reset_recurrent_state', 'rollout_training.py',
     '_step(m, frame, states, start + j)',
     '_step(m, frame, [torch.zeros_like(s) for s in states], start + j)'),
]


def run(root, env):
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '--rootdir', str(root),
                           str(root / 'tests/test_rollout_training.py') + '::' + TEST],
                          cwd=root, env=env, text=True, capture_output=True, timeout=120)


def main():
    env = {**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    control = run(ROOT, env)
    if control.returncode != 0 or '1 passed' not in control.stdout:
        raise RuntimeError('unmutated control failed:\n' + control.stdout + control.stderr)
    print('CONTROL: 1 passed', flush=True)
    scratch = ROOT / 'runs_rollout_mutations'
    scratch.mkdir(exist_ok=True)
    for name, filename, old, new in MUTANTS:
        with tempfile.TemporaryDirectory(prefix='mutant-', dir=scratch) as tmp:
            target = Path(tmp)
            (target / 'tests').mkdir()
            # Some legacy modules prepend their own directory to sys.path.
            # Copy their siblings too, otherwise a mutant can accidentally import
            # the ORIGINAL rollout_training and produce a vacuous green test.
            for source in ROOT.glob('*.py'):
                shutil.copy2(source, target / source.name)
            shutil.copy2(ROOT / 'tests/test_rollout_training.py',
                         target / 'tests/test_rollout_training.py')
            file = target / filename
            text = file.read_text()
            if text.count(old) != 1:
                raise RuntimeError(f'{name}: mutation anchor not unique')
            file.write_text(text.replace(old, new))
            result = run(target, env)
            if (result.returncode != 1 or '1 failed' not in result.stdout
                    or 'AssertionError' not in result.stdout or f'FAILED tests/test_rollout_training.py::{TEST}'
                    not in result.stdout or 'ERROR collecting' in result.stdout):
                raise RuntimeError(f'{name}: did not fail for expected assertion:\n'
                                   + result.stdout + result.stderr)
            print(f'KILLED {name}: {TEST} AssertionError', flush=True)
    print(f'MUTATION PASS: {len(MUTANTS)}/{len(MUTANTS)} killed; original tree unchanged', flush=True)


if __name__ == '__main__':
    main()
