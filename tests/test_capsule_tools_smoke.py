import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cli'))

from capsule_tools import build_capsule, build_relay


EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'


def test_build_capsule_smoke():
    capsule = build_capsule(str(EXAMPLES / 'task-bug-fix.crumb'))
    assert capsule['kind'] == 'task'
    assert capsule['encoded_tokens'] > 0


def test_build_relay_smoke():
    relay = build_relay(str(EXAMPLES))
    assert relay['count'] >= 1
    assert isinstance(relay['events'], list)
