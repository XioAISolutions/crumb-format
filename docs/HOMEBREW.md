# Homebrew installation

CRUMB now includes a maintained Homebrew formula source file in [`Formula/crumb-format.rb`](../Formula/crumb-format.rb). The intended distribution model is a standard third-party Homebrew tap backed by GitHub.

## Recommended user install flow

Once the tap exists publicly, users should be able to install CRUMB directly with:

```bash
brew install XioAISolutions/tap/crumb-format
```

Homebrew will automatically tap the repository during installation.

If users prefer to add the tap explicitly first, they can run:

```bash
brew tap XioAISolutions/tap
brew install crumb-format
```

## Repository layout

Homebrew recommends placing custom formulae in a `Formula/` subdirectory inside the tap repository. This repository now includes a formula file that can serve as the source of truth for that tap-managed formula.

```text
Formula/
  crumb-format.rb
```

## Formula strategy

The CRUMB formula is implemented as a Python application formula that uses Homebrew's virtual environment helper:

- It depends on `python@3.13`.
- It uses `include Language::Python::Virtualenv`.
- It installs with `virtualenv_install_with_resources`.
- It includes a deterministic smoke test that validates a minimal `.crumb` file.

This approach is aligned with Homebrew's guidance for Python CLI applications.

## Current source URL

The current formula uses the published PyPI source distribution for `crumb-format` `0.2.0`. That gives the formula an immutable source archive and stable checksum.

## How to publish the tap

The cleanest setup is a separate public repository named something like:

```text
XioAISolutions/homebrew-tap
```

Then copy or sync `Formula/crumb-format.rb` into that repository and commit it there. Users will then be able to install with the short tap syntax.

## How to update the formula for a new release

When cutting a new CRUMB release:

1. Publish the new package version to PyPI.
2. Obtain the new sdist URL and SHA256.
3. Update `url` and `sha256` in `Formula/crumb-format.rb`.
4. If needed, update the Python dependency version to match current Homebrew support.
5. Commit the updated formula into the tap repository.
6. Create the release tag in the main CRUMB repository.

## Caveat

This repository can hold the formula definition and documentation, but end users will get the smoothest Homebrew experience only after the formula is published in a proper tap repository. Until then, advanced users can still install from a checked-out formula file manually.
