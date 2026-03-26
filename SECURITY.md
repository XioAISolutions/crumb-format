# Security policy

## Scope

CRUMB is currently a text-first open format spec plus reference tooling.

## Reporting a vulnerability

If you find a security issue in the validators, CLI, or reference files, please open a private report through GitHub security reporting if available, or contact the maintainer directly before posting full exploit details publicly.

## Notes

- Do not put secrets into example crumbs
- Treat `.crumb` files as untrusted input in production systems
- Validate before consuming when building integrations
