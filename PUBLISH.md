# Publishing Guide

This project supports publishing to PyPI, crates.io, and a configured Maven
repository. The planned public version is `0.3.0`. The root `Cargo.toml` is the
version source for Rust, Python (via maturin) and the JVM (via Gradle).

## Release prerequisites

Run `make build` and `make test` on the release commit and review its CI results.
Run `make python-support-test` for the installed-wheel/runtime and source-archive
gates. Review the measured accuracy limits linked from `README.md`; regression
checks do not establish that every accuracy target passes.

The original `tdigest_core` workspace crate remains at `0.2.0`. Its matching
registry package must be available before publishing `gr-tdigest`: the local path
dependency alone is not sufficient for crates.io consumers. Neither the Cargo
release workflow nor `make publish-cargo` publishes that dependency.

Before tagging, move the pending release notes from `## [Unreleased]` into a
`## [0.3.0]` section and leave a new empty Unreleased section. Merge the release
commit into `master`; the tag workflows require it to be reachable from master.
Set `TDIGEST_SOURCE_SHA` to the source commit when building a local release JAR
to record its identity alongside the native payload SHA-256.

## GitHub Actions release workflows
Release workflows are in `.github/workflows/`:
- `release_pypi.yml`
- `release_cargo.yml`
- `release_maven.yml`

They trigger on tags matching `v*`.

### 1. Configure GitHub environments and secrets

PyPI (`pypi` environment):
- Configure PyPI Trusted Publisher for this repo/workflow.

Cargo (`crates-io` environment):
- Secret: `CARGO_REGISTRY_TOKEN`

Maven (`maven` environment):
- Secrets: `MAVEN_REPOSITORY_URL`, `MAVEN_USERNAME`, `MAVEN_PASSWORD`
- If signing is required: `MAVEN_SIGNING_KEY`, `MAVEN_SIGNING_PASSWORD`

### 2. Create and push a release tag
- Ensure `Cargo.toml` version matches the tag without `v`.
- Example:

```bash
git tag v0.3.0
git push origin v0.3.0
```

## Local publish
Use `make publish` when publishing from local credentials after completing the
release prerequisites above.

Dry run:

```bash
PUBLISH_DRY_RUN=1 make publish
```

Real publish:

```bash
MATURIN_PYPI_TOKEN=... \
CARGO_REGISTRY_TOKEN=... \
MAVEN_REPOSITORY_URL=... \
MAVEN_USERNAME=... \
MAVEN_PASSWORD=... \
make publish
```

Optional signing variables:
- `MAVEN_SIGNING_KEY`
- `MAVEN_SIGNING_PASSWORD`

## Repository rulesets
Recommended repository protection and ruleset setup is documented in `.github/REPO_SETTINGS.md`.
