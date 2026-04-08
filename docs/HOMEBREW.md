# Homebrew installation

CRUMB now includes a maintained tap-ready Homebrew formula source file in [`Formula/crumb.rb`](../Formula/crumb.rb). The intended distribution model is a standard third-party Homebrew tap backed by GitHub.

## Recommended user install flow

If Homebrew itself is not installed yet, install it first from the official site at [brew.sh](https://brew.sh/).

Once Homebrew is available, the shortest practical install flow is:

```bash
brew tap XioAISolutions/tap
brew install crumb
```

After installation, verify the packaged CLI is working:

```bash
crumb --help
crumb validate examples/task_rebuild_auth.crumb
```

In many cases users may also be able to install directly with the fully qualified tap path:

```bash
brew install XioAISolutions/tap/crumb
```

## Repository layout

Homebrew recommends placing custom formulae in a `Formula/` subdirectory inside the tap repository. This repository now includes a formula file that can serve as the source of truth for that tap-managed formula.

```text
Formula/
  crumb.rb
```

## Formula strategy

The CRUMB formula is implemented as a Python application formula that uses Homebrew's virtual environment helper:

- It depends on `python@3.13`.
- It uses `include Language::Python::Virtualenv`.
- It installs with `virtualenv_install_with_resources`.
- It includes a deterministic smoke test that validates a minimal `.crumb` file.

This approach is aligned with Homebrew's guidance for Python CLI applications.

## Current source URL

The checked-in formula currently points at the latest published PyPI source distribution available when the file was last updated. After each new release, update the `url` and `sha256` to match the new PyPI sdist before syncing the tap.

## How to publish the tap

The cleanest setup is a separate public repository named something like:

```text
XioAISolutions/homebrew-tap
```

Then copy or sync `Formula/crumb.rb` into that repository and commit it there. Users will then be able to install with the short tap syntax.

## How to update the formula for a new release

When cutting a new CRUMB release, the intended flow is:

1. Publish the new package version to PyPI.
2. Compute the new sdist checksum from the published artifact.
3. Update `Formula/crumb.rb` in the tap repository with the new `url` and `sha256`.
4. Push the tap update to `XioAISolutions/homebrew-tap`.
5. If you later add tap automation, keep this manual flow as the fallback path.

## Caveat

This repository can hold the formula definition and documentation, but end users will get the smoothest `brew install crumb` experience only after the formula is published in a proper tap repository and installed after `brew tap XioAISolutions/tap`. Until then, advanced users can still install from a checked-out formula file manually.
