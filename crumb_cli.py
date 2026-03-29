"""Entry point for `crumb` console script installed via pip.

This re-exports main() from cli/crumb.py so that `pip install crumb-format`
creates a `crumb` command in the user's PATH.
"""

import sys
from pathlib import Path

# Allow importing from cli/ without package restructuring
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from crumb import main  # noqa: E402

if __name__ == "__main__":
    main()
