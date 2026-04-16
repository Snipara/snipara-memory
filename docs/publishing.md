# Publishing

## Current State

The repository is prepared for PyPI publication, but publication still depends
on PyPI-side project setup.

What is already in the repo:

- package metadata in `pyproject.toml`
- CI workflow for lint, tests, and build
- trusted-publishing workflow for releases
- `twine check` validation in CI

## Local Release Validation

```bash
pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

## GitHub Actions

The repository includes:

- `.github/workflows/ci.yml`
- `.github/workflows/publish.yml`

The publish workflow expects PyPI trusted publishing to be configured for the
GitHub repository.

## First PyPI Release Checklist

1. Create the `snipara-memory` project on PyPI if it does not exist yet.
2. Add `Snipara/snipara-memory` as a trusted publisher on PyPI.
3. Merge the release-ready package state to `main`.
4. Trigger the publish workflow or push a version bump/tag.
5. Verify the new release on PyPI.

## Why Trusted Publishing

Trusted publishing removes the need to store a long-lived PyPI API token in
GitHub secrets for normal releases.
