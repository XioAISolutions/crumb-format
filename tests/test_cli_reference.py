"""Drift check for the auto-generated CLI reference.

``docs/CLI.md`` is regenerated from the live argparse tree by
``tools/generate_cli_reference.py``. Any change to the CLI surface
(new subcommand, renamed flag, etc.) that forgets to regenerate the
doc will fail this test. Fix::

    python tools/generate_cli_reference.py

and commit the resulting ``docs/CLI.md``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "CLI.md"
GENERATOR_PATH = REPO_ROOT / "tools" / "generate_cli_reference.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_cli_reference", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_cli_reference"] = module
    spec.loader.exec_module(module)
    return module


def test_cli_reference_exists():
    assert DOC_PATH.is_file(), f"expected auto-generated doc at {DOC_PATH}"


def test_cli_reference_has_no_drift():
    generator = _load_generator()
    generated = generator.generate()
    existing = DOC_PATH.read_text(encoding="utf-8")
    assert existing == generated, (
        f"{DOC_PATH} is out of date. "
        "Rerun `python tools/generate_cli_reference.py` and commit."
    )


def test_cli_reference_lists_major_commands():
    text = DOC_PATH.read_text(encoding="utf-8")
    for subcommand in ("new", "validate", "inspect", "palace", "pack", "lint", "metalk"):
        assert f"`crumb {subcommand}`" in text, (
            f"docs/CLI.md should mention `crumb {subcommand}`"
        )
