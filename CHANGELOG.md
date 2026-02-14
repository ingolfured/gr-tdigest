# Changelog

All notable changes to this project are documented in this file.

This changelog was reconstructed from git history (commits + version transitions in `Cargo.toml`) and starts at `0.1.1` as requested.

## [Unreleased]

### Added
- Open-source governance and community docs:
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
- Minimal GitHub bug-reporting templates:
  - `.github/ISSUE_TEMPLATE/bug_report.yml`
  - `.github/ISSUE_TEMPLATE/config.yml`
- Minimal GitHub Actions CI workflow:
  - `.github/workflows/ci.yml`

### Changed
- `CODE_OF_CONDUCT.md` simplified to a minimal, practical policy.
- Security reporting flow moved to GitHub private vulnerability reporting (`security/advisories/new`).
- `README.md` now includes links to community and support docs.
- Removed old mixed CI/release workflow (`.github/workflows/publish_to_pypi.yml`) in favor of a clean CI-only start.

## [0.2.0] - 2026-02-14

### Added
- Rust weighted ingestion APIs:
  - `TDigest::add_weighted(value, weight)`
  - `TDigest::add_weighted_many(values, weights)`
  - `TDigest::merge_weighted_unsorted(values, weights)`
  - `TDigest::from_weighted_unsorted(values, weights, max_size)`
- Frontend weighted-ingest APIs:
  - Python: `TDigest.add_weighted(values, weights)`
  - Java/JNI: `TDigest.addWeighted(...)`
  - Polars plugin: `add_weighted_values(...)`
- Explicit precision casting in core/frontend paths:
  - `TDigest::cast_precision::<f32|f64>()`
  - `FrontendDigest::cast_precision(...)`
- Public cast-precision APIs on all user surfaces:
  - Python `TDigest.cast_precision(...)`
  - Java `TDigest.castPrecision(...)`
  - Polars plugin `cast_precision(...)`
- Explicit wire-version encode controls:
  - Python `to_bytes(version=...)`
  - Java `toBytes(version)`
  - Polars plugin `to_bytes(..., version=...)`
- TDIG v3 wire format support in codec:
  - default encoder now writes v3
  - decoder supports v1/v2/v3
  - v3 adds flags + header length + explicit payload precision code + optional 4-byte checksum
  - v2/v3 preserve fractional centroid weights and centroid kind

### Changed
- Release/version transition from `0.1.13` to `0.2.0`.
- Design docs split and deepened (`api_design.md`, `tdigest_design.md`, `comparison_design.md`) with stricter contract/current/comparison separation.
- `comparison_design.md` gap analysis updated to reflect cast API completion and TDIG v3/version-controls progress.
- Repository process/tooling now enforces changelog updates via pre-commit (`scripts/check_changelog_update.sh` + `.pre-commit-config.yaml`).
- `make setup` now installs pre-commit hooks with repo-local caches (`.pre-commit-cache`, `.uv-cache`) to avoid host cache permission issues.
- Agent workflow guidance tightened (`AGENTS.md`) to require changelog updates, design-doc review/updates, and README maintenance.
- `Makefile` now defaults `UV_CACHE_DIR` to repo-local `.uv-cache` for more reliable sandboxed/dev runs.
- Core digest merge switched from concat+sort run materialization to heap-stream k-way merge in `KWayCentroidMerge::from_runs`, reducing peak merge allocations and improving heavy merge throughput.
- Raw ingest merge (`MergeByMean::from_centroids_and_values`) switched to a streaming two-way iterator, removing prebuilt merged-buffer materialization in `merge_sorted`.

### Testing / Contracts
- Cross-surface contract/coherence test suite expanded and consolidated.
- Added wire migration tests for mixed v1/v2 blobs and explicit versioned encoding on Python/Java paths.
- Added merge strategy proof tests:
  - randomized correctness parity (`heap` vs historical `concat+sort`)
  - ignored heavy benchmark/proof test for CPU + memory + precision comparison (current profile: `n=10M`, `shards=40`, `max_size=1000`)
  - historical concat+sort comparator retained as test-only baseline (`#[cfg(test)]`)
- Full project gates continue to pass (`make build`, `make test`).

## [0.1.13] - 2026-02-09

### Changed
- Version bump from `0.1.12` to `0.1.13` (no functional delta in bump commit).

## [0.1.12] - 2026-02-09

### Added
- Java API support for digest merge and serialization interop.
- Java value-merge APIs and Gradle/Make wiring for Java-side tests.

### Changed
- Release/version transition from `0.1.11` to `0.1.12`.

## [0.1.11] - 2025-12-06

### Added
- Python support for `merge` and `merge_all` digest operations.

## [0.1.10] - 2025-11-24

### Added
- Digest encoding/decoding support (cross-surface byte serialization path introduced).

## [0.1.9] - 2025-11-07

### Added
- Polars behavior support for empty group-by scenarios.

## [0.1.8] - 2025-11-04

### Changed
- Introduced shared `frontends.rs` layer and dried up duplicated binding logic.
- Improved CDF handling around atomic-centroid semantics.

### Docs
- README improvements.

## [0.1.7] - 2025-11-03

### Testing / Contracts
- Introduced unified API coherence test direction across surfaces.

## [0.1.6] - 2025-10-31

### Changed
- Tightened non-finite handling and edge-policy behavior.

## [0.1.5] - 2025-10-30

### Added
- End-to-end float precision behavior (`f32`/`f64`) across surfaces.

## [0.1.4] - 2025-10-30

### Added
- List-oriented CDF/quantile expression support in Polars workflows.

## [0.1.3] - 2025-10-30

### Changed
- Large refactor pass across Gradle/bindings/docs/build setup.

## [0.1.2] - 2024-06-14

### Changed
- Library naming and packaging adjustments (`polars_tdigest` era alignment).

## [0.1.1] - 2024-06-13

### Added
- First `0.1.x` release line established.
- Initial packaging/release plumbing for early distribution.
