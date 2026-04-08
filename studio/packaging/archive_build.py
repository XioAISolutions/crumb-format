from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _platform_suffix() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return sys.platform


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    dist_dir = root / "dist"
    artifact_dir = root / "dist-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    app_name = "CRUMB-Studio"
    if sys.platform == "darwin":
        bundle = dist_dir / f"{app_name}.app"
        if not bundle.exists():
            raise SystemExit(f"Missing build output: {bundle}")

        staging = artifact_dir / f"{app_name}.app"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(bundle, staging)

        archive_base = artifact_dir / f"{app_name}-{_platform_suffix()}"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=artifact_dir, base_dir=staging.name)
        shutil.rmtree(staging)
    else:
        bundle_dir = dist_dir / app_name
        if not bundle_dir.exists():
            raise SystemExit(f"Missing build output: {bundle_dir}")

        archive_base = artifact_dir / f"{app_name}-{_platform_suffix()}"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=dist_dir, base_dir=app_name)

    print(archive_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
