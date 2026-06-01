"""Standalone PyPI package configuration generator for crumb-wavelm.

Run this to generate a standalone pyproject.toml for publishing the wave-field
language model (`crumb_wavelm`) as its own PyPI package, separate from both
crumb-format and the CrumbLLM analysis product.

    python -m crumb_wavelm.setup_standalone --output /tmp/crumb-wavelm-pkg

The generated package:
  - Imports crumb_wavelm as the top-level package
  - Depends on torch, numpy
  - Optionally depends on crumb-format (for crumb_adapter)
  - Includes all configs, the CLI entry point, and docs

NOTE: this is the physics/wave-field research model. The `crumb-llm` name now
belongs to the CrumbLLM analysis product (XioAISolutions/CrumbLLM).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PYPROJECT = """\
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "crumb-wavelm"
version = "0.2.0"
description = "Crumb WaveLM: O(N log N) language modeling via physics-based wave equations. Replaces transformer attention with scatter → FFT-convolve → gather."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{{ name = "XIO AI Solutions" }}]
keywords = ["llm", "wave-field", "fft", "attention", "physics", "transformer", "O(N-log-N)"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
]

[project.optional-dependencies]
crumb = ["crumb-format"]
serve = ["crumb-format"]
all = ["crumb-format"]

[project.urls]
Homepage = "https://github.com/XioAISolutions/crumb-format"
Repository = "https://github.com/XioAISolutions/crumb-format"
Documentation = "https://github.com/XioAISolutions/crumb-format/blob/main/CRUMB_LLM.md"

[project.scripts]
crumb-wavelm = "crumb_wavelm.__main__:main"

[tool.setuptools]
packages = ["crumb_wavelm", "crumb_wavelm.configs"]

[tool.setuptools.package-data]
"crumb_wavelm.configs" = ["*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


def generate_standalone(output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # pyproject.toml
    (out / "pyproject.toml").write_text(PYPROJECT)

    # Copy the crumb_wavelm package
    src = Path(__file__).resolve().parent
    dst = out / "crumb_wavelm"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # README (use CRUMB_LLM.md from repo root)
    readme_src = src.parent / "CRUMB_LLM.md"
    if readme_src.exists():
        shutil.copy2(readme_src, out / "README.md")
    else:
        (out / "README.md").write_text("# crumb-llm\n\nSee https://github.com/XioAISolutions/crumb-format\n")

    # LICENSE
    license_src = src.parent / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, out / "LICENSE")

    # Tests
    tests_src = src.parent / "tests"
    tests_dst = out / "tests"
    if tests_dst.exists():
        shutil.rmtree(tests_dst)
    tests_dst.mkdir()
    for f in tests_src.glob("test_crumb_llm_*.py"):
        shutil.copy2(f, tests_dst / f.name)

    print(f"[standalone] generated at {out}/")
    print(f"  pip install {out}/")
    print(f"  python -m build {out}/")
    print(f"  twine upload {out}/dist/*")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate standalone crumb-wavelm PyPI package.")
    ap.add_argument("--output", default="/tmp/crumb-wavelm-pkg")
    args = ap.parse_args(argv)
    generate_standalone(args.output)


if __name__ == "__main__":
    main()
