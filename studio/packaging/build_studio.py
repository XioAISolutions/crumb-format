from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    static_dir = root / "studio" / "static"
    separator = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        "CRUMB-Studio",
        "--add-data",
        f"{static_dir}{separator}studio/static",
        str(root / "studio" / "app.py"),
    ]
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
