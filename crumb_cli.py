"""Entry point for `crumb` console script installed via pip.

This re-exports main() from cli/crumb.py so that `pip install crumb-format`
creates a `crumb` command in the user's PATH.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

# Allow importing from cli/ without package restructuring
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from crumb import main as crumb_main  # noqa: E402

PACKAGE_NAME = "crumb-format"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
UPDATE_CACHE_PATH = Path.home() / ".crumb" / "update-check.json"
UPDATE_CHECK_INTERVAL_SECONDS = 60 * 60 * 24

BREAD = "🍞"
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


def print_banner() -> None:
    v = get_cli_version()
    print()
    print(f"{BREAD}  CRUMB {v} — AI handoff with bread crumbs")
    print()
    print(ASCII_LOGO)
    print(f"              {BREAD}  CRUMB  {BREAD}")
    print()


def print_wrapper_commands() -> None:
    print("Extra commands:")
    print("  crumb version          Show installed version")
    print("  crumb update --check   Check for a newer release")
    print("  crumb update           Upgrade the installed package")
    print()


def _fallback_version_key(raw: str):
    parts = re.findall(r"\d+|[A-Za-z]+", raw)
    normalized = []
    for part in parts:
        normalized.append(int(part) if part.isdigit() else part.lower())
    return tuple(normalized)


def is_newer_version(latest: str, current: str) -> bool:
    try:
        from packaging.version import Version

        return Version(latest) > Version(current)
    except Exception:
        return _fallback_version_key(latest) > _fallback_version_key(current)


def get_latest_version(timeout: float = 2.5) -> str | None:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"User-Agent": f"{PACKAGE_NAME}-cli-update-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        candidate = payload.get("info", {}).get("version")
        return candidate or None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def is_editable_install() -> bool:
    try:
        dist = distribution(PACKAGE_NAME)
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return False
        payload = json.loads(direct_url)
        return bool(payload.get("dir_info", {}).get("editable"))
    except Exception:
        return False


def load_update_cache() -> dict:
    try:
        return json.loads(UPDATE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_update_cache(payload: dict) -> None:
    try:
        UPDATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def maybe_notify_update(argv: list[str]) -> None:
    if os.environ.get("CRUMB_NO_UPDATE_CHECK") == "1":
        return
    if not sys.stderr.isatty():
        return
    if argv and argv[0] in {"update", "version", "--version", "-V"}:
        return

    cache = load_update_cache()
    now = int(__import__("time").time())
    last_checked = int(cache.get("last_checked", 0) or 0)
    latest = cache.get("latest_version")

    if now - last_checked >= UPDATE_CHECK_INTERVAL_SECONDS or not latest:
        latest = get_latest_version()
        save_update_cache(
            {
                "last_checked": now,
                "latest_version": latest,
            }
        )

    current = get_cli_version()
    if latest and current != "dev" and is_newer_version(latest, current):
        print(
            f"{BREAD} Update available: {current} → {latest}  (run: crumb update)",
            file=sys.stderr,
        )


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
        help="Only check whether a newer release is available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow pip upgrade even when this is an editable install.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Reserved for non-interactive workflows; currently not required.",
    )
    return parser


def cmd_update(argv: list[str]) -> int:
    parser = build_update_parser()
    args = parser.parse_args(argv)

    current = get_cli_version()
    latest = get_latest_version()

    if args.check:
        if not latest:
            print("Could not reach the package index to check for updates.")
            return 1
        if current == "dev":
            print(f"Installed version: {current}")
            print(f"Latest published version: {latest}")
            return 0
        if is_newer_version(latest, current):
            print(f"Update available: {current} → {latest}")
        else:
            print(f"Already up to date: {current}")
        return 0

    if is_editable_install() and not args.force:
        print("Editable install detected.")
        print("Run:")
        print("  git pull")
        print("  pip install -e .")
        print("Or run:")
        print("  crumb update --force")
        print("to replace the editable install with the published package.")
        return 0

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        PACKAGE_NAME,
    ]
    print("Updating crumb-format...")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        print(f"Update failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print("Update complete.")
    installed = get_cli_version()
    if installed != "dev":
        print(f"Installed version: {installed}")
    return 0


def run_core(argv: list[str]) -> int:
    try:
        result = crumb_main(argv)
        if isinstance(result, int):
            return result
        return 0
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        print_banner()
        code = run_core(["--help"])
        print_wrapper_commands()
        return code

    if argv[0] in {"version", "--version", "-V"}:
        return cmd_version()

    if argv[0] == "update":
        return cmd_update(argv[1:])

    if argv[0] in {"-h", "--help"}:
        print_banner()
        code = run_core(["--help"])
        print_wrapper_commands()
        return code

    maybe_notify_update(argv)
    return run_core(argv)


if __name__ == "__main__":
    raise SystemExit(main())
