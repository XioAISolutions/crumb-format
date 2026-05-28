"""Console entry point for the ``crumb`` command.

The core parser lives in ``cli/crumb.py``. This wrapper keeps operator
commands that sit outside the CRUMB grammar itself, such as local version and
editable-checkout update handling.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

# Allow importing from cli/ without package restructuring.
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from crumb import main as crumb_main  # noqa: E402

PACKAGE_NAME = "crumb-format"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
BREAD = "\N{BREAD}"
ASCII_LOGO = r"""
 ██████╗██████╗ ██╗   ██╗███╗   ███╗██████╗
██╔════╝██╔══██╗██║   ██║████╗ ████║██╔══██╗
██║     ██████╔╝██║   ██║██╔████╔██║██████╔╝
██║     ██╔══██╗██║   ██║██║╚██╔╝██║██╔══██╗
╚██████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║██████╔╝
 ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝
"""


def get_cli_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "dev"


def print_extra_commands() -> None:
    print()
    print("Extra commands:")
    print("  crumb version          Show installed version")
    print("  crumb update --check   Check for a newer release")
    print("  crumb update           Upgrade the installed package or repo checkout")


def print_banner() -> None:
    print()
    print(f"{BREAD}  CRUMB {get_cli_version()} — AI handoff with bread crumbs")
    print()
    print(ASCII_LOGO)
    print(f"              {BREAD}  CRUMB  {BREAD}")
    print()


def _fallback_version_key(raw: str) -> tuple[object, ...]:
    parts = re.findall(r"\d+|[A-Za-z]+", raw)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def is_newer_version(latest: str, current: str) -> bool:
    try:
        from packaging.version import Version

        return Version(latest) > Version(current)
    except Exception:
        return _fallback_version_key(latest) > _fallback_version_key(current)


def _git_run(args: list[str], cwd: Path, check: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout.strip()


def get_editable_repo_root() -> Path | None:
    try:
        dist = distribution(PACKAGE_NAME)
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return None
        payload = json.loads(direct_url)
        if not payload.get("dir_info", {}).get("editable"):
            return None
        raw_url = payload.get("url")
        if not raw_url:
            return None
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme != "file":
            return None
        path = urllib.request.url2pathname(parsed.path)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return Path(path).resolve()
    except Exception:
        return None


def get_latest_version(timeout: float = 2.5) -> str | None:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"User-Agent": f"{PACKAGE_NAME}-cli-update-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return payload.get("info", {}).get("version") or None
    except (
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None


def get_editable_update_status(repo_root: Path) -> tuple[str | None, int | None, int | None]:
    try:
        _git_run(["fetch", "--quiet"], cwd=repo_root, check=True)
        upstream = _git_run(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_root,
            check=True,
        )
        ahead = int(_git_run(["rev-list", "--count", "@{u}..HEAD"], cwd=repo_root, check=True) or "0")
        behind = int(_git_run(["rev-list", "--count", "HEAD..@{u}"], cwd=repo_root, check=True) or "0")
        return upstream, ahead, behind
    except Exception:
        return None, None, None


def cmd_version() -> int:
    print(get_cli_version())
    return 0


def build_update_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crumb update",
        description="Check for or install a newer crumb-format release.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether a newer release or repo update is available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For editable installs, force a PyPI upgrade instead of updating the repo checkout.",
    )
    return parser


def run_editable_update(repo_root: Path) -> int:
    print(f"Editable install detected at: {repo_root}", flush=True)
    upstream, ahead, behind = get_editable_update_status(repo_root)
    if upstream is None or ahead is None or behind is None:
        print("Could not determine remote repo update status.", file=sys.stderr)
        return 1
    if ahead and behind:
        print(
            f"Repo checkout has diverged from {upstream}: ahead {ahead}, behind {behind}.",
            file=sys.stderr,
        )
        print("Resolve git history first, then run: crumb update", file=sys.stderr)
        return 1

    try:
        subprocess.check_call(["git", "-C", str(repo_root), "pull", "--ff-only"])
    except subprocess.CalledProcessError as exc:
        print(f"git pull failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_root)])
    except subprocess.CalledProcessError as exc:
        print(f"Editable reinstall failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print("Repo update complete.")
    return 0


def cmd_update(argv: list[str]) -> int:
    parser = build_update_parser()
    args = parser.parse_args(argv)

    repo_root = get_editable_repo_root()
    if repo_root and not args.force:
        upstream, ahead, behind = get_editable_update_status(repo_root)
        if args.check:
            if upstream is None or ahead is None or behind is None:
                print("Could not determine remote repo update status.")
                print(f"Repo path: {repo_root}")
                return 1
            if ahead and behind:
                print(f"Repo checkout has diverged from {upstream}: ahead {ahead}, behind {behind}")
            elif behind:
                print(f"Repo update available: behind {upstream} by {behind} commit(s)")
            else:
                print(f"Repo checkout is up to date with {upstream}")
            return 0
        return run_editable_update(repo_root)

    current = get_cli_version()
    latest = get_latest_version()
    if args.check:
        if not latest:
            print("Could not reach the package index to check for updates.")
            return 1
        if current != "dev" and is_newer_version(latest, current):
            print(f"Update available: {current} -> {latest}")
        else:
            print(f"Already up to date: {current}")
        return 0

    if latest and current != "dev" and not is_newer_version(latest, current):
        print(f"Already up to date: {current}")
        return 0

    print("Updating crumb-format...", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME])
    except subprocess.CalledProcessError as exc:
        print(f"Update failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print("Update complete.")
    print(f"Installed version: {get_cli_version()}")
    return 0


def run_core(argv: list[str]) -> int:
    try:
        result = crumb_main(argv)
        return result if isinstance(result, int) else 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "version":
        return cmd_version()
    if argv and argv[0] == "update":
        return cmd_update(argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print_banner()
        code = run_core(["--help"])
        print_extra_commands()
        return code
    return run_core(argv)


if __name__ == "__main__":
    raise SystemExit(main())
