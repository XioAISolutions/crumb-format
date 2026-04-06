"""Entry point for `crumb` console script installed via pip.

This re-exports main() from cli/crumb.py so that `pip install crumb-format`
creates a `crumb` command in the user's PATH.
"""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Allow importing from cli/ without package restructuring
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from crumb import main as crumb_main  # noqa: E402

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
        return version("crumb-format")
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        print_banner()
        return crumb_main(["--help"])

    if argv[0] in {"-h", "--help"}:
        print_banner()

    return crumb_main(argv)


if __name__ == "__main__":
    main()
