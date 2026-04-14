#!/usr/bin/env python3
"""Bootstrap updater for local CRUMB checkouts.

Run this from any clone of the repo when the installed `crumb` command is too old
or broken to update itself yet.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    print(f"Using repo: {repo_root}")
    try:
        run(["git", "pull", "--ff-only"], cwd=repo_root)
        run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        print(f"Bootstrap update failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print("Bootstrap update complete.")
    print("Now try:")
    print("  crumb version")
    print("  crumb update --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
