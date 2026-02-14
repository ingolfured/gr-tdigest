# AGENTS.md

This file is a working guide for coding agents in this repository. It is intentionally specific to this codebase and should be treated as the default operating manual for making safe changes.

## 1. Project purpose and shape

`gr-tdigest` is a single Rust TDigest core with four user-facing surfaces:

- Rust library API (`src/tdigest/*`)
- Rust CLI (`src/bin/tdigest_cli.rs`)
- Python package + Polars plugin (`bindings/python/*`, `src/py.rs`, `src/polars_expr.rs`)
- Java API via JNI (`bindings/java/*`, `src/jni.rs`)

Cross-surface behavior is enforced by integration tests in `integration/api_coherence/`.

## 2. Repository map (high signal)

- `src/lib.rs`: crate entry, feature gates, module exports, Python module registration.
- `src/tdigest/`: core algorithm and codecs.
- `src/error.rs`: common error types and user-facing error messages.
- `src/py.rs`: PyO3-native class/method bindings (thin adapter over shared Rust frontend service).
- `src/polars_expr.rs`: Polars plugin UDFs (`tdigest`, `add_values`, `cdf`, `quantile`, `median`, merge, byte conversions).
- `src/jni.rs`: JNI exports used by Java bridge.
- `src/bin/tdigest_cli.rs`: CLI used heavily by coherence tests.
- `bindings/python/gr_tdigest/__init__.py`: Python public API + Polars plugin wiring (keep runtime patching minimal).
- `bindings/java/src/gr/tdigest/`: Java API, native loader, JNI method declarations.
- `integration/api_coherence/`: behavior contracts across CLI/Python/Polars/Java.
- `integration/api_coherence/test_wire_python_polars.py` and `integration/api_coherence/test_wire_java_interop.py`: cross-language byte-wire compatibility.
- `Makefile`: canonical dev/build/test/lint/release commands.

## 3. Core architecture and invariants

### 3.1 Core digest model

- Main type: `TDigest<F>` where `F` is `f32` or `f64` (`src/tdigest/tdigest.rs`).
- Memory precision is controlled by `F`; accumulation fields `sum` and `count` stay `f64`.
- Centroids have kind `Atomic` or `Mixed` (`src/tdigest/centroids.rs`).
- Compression pipeline lives in `src/tdigest/compressor.rs`:
  1. Normalize
  2. Slice (policy)
  3. Merge (k-limit by scale)
  4. Cap
  5. Assemble
  6. Post-process

### 3.2 Semantics that must stay coherent

The integration suite treats these as contract-level behavior:

- Training input:
  - non-finite values (`NaN`, `+/-inf`) must error.
  - Polars training nulls must error.
- Empty digest:
  - CDF returns `NaN` for probes (including `+/-inf`).
  - Quantile for finite `q` yields `NaN` in native core; wrappers map this per surface (e.g. Polars `None`).
- Probe validation:
  - `quantile(q)` is strict: `q` must be finite and in `[0, 1]`.
  - `cdf(NaN)` returns `NaN`.
  - For non-empty digest, `cdf(-inf)=0` and `cdf(+inf)=1` on Python/Java/Polars surfaces.
- Precision coherence:
  - F32/F64 behaviors must remain predictable and explicit.
  - Mixed precision blobs in a single Polars `from_bytes` column are rejected.
  - Null blobs in Polars `from_bytes` are strict errors.
- Capability parity baseline:
  - All surfaces must expose build/train, add (single + multiple), merge multiple digests, `quantile`, `cdf`, `median`.
  - Naming may differ by surface (`add_values`/`merge_tdigests` on Polars), but capability coverage must match.

When changing behavior intentionally, update `integration/api_coherence/` tests together.

### 3.3 Wire format

- Canonical binary format is TDIG (`src/tdigest/wire.rs`).
- Header + centroid payload; payload width determines wire precision (`f32` or `f64` means).
- `wire_precision(...)` is used by Python/Polars logic; avoid silent coercions.
- Any wire layout change is high impact across Python, Polars, and Java serialization interop tests.

## 4. Build and test workflow

Use `Makefile` targets by default.

### 4.0 Non-negotiable project gates

- `make build` must pass from repo root.
- `make test` must pass from repo root.
- Treat these two commands as the canonical "whole project is healthy" contract.
- If you run narrower commands while iterating, still finish by running `make build` and `make test`.
- Any PR/change that breaks either target is incomplete.

## 4.1 Setup/build

- `make setup`
- `make build`
- `make build-rust`
- `make build-python`
- `make build-java`

## 4.2 Test

- Primary full-project test gate: `make test`
- `make test`
- `cargo test -- --quiet`
- `./.venv/bin/pytest -q`
- `./.venv/bin/pytest -q integration/api_coherence/test_contract_behavior.py`
- `./.venv/bin/pytest -q integration/api_coherence/test_contract_probe_validation.py`
- `./.venv/bin/pytest -q integration/api_coherence/test_wire_python_polars.py`
- `./.venv/bin/pytest -q integration/api_coherence/test_wire_java_interop.py`
- `bindings/java/gradlew --no-daemon --console=plain -p bindings/java test`

## 4.3 Lint and docs

- `make lint`
  - `cargo fmt`
  - `cargo clippy --fix --allow-dirty --allow-staged`
  - `ruff check --fix`
  - `ruff format`
  - `mypy`
  - `cargo doc` with warnings denied

## 5. Surface-specific implementation notes

### 5.1 Rust core (`src/tdigest/*`)

- Prefer keeping numeric behavior centralized in core modules (`quantile.rs`, `cdf.rs`, `compressor.rs`).
- Keep centroid ordering strict by mean and preserve weight accounting.
- If changing centroid kind behavior, validate CDF/quantile interpolation paths and serialization decode heuristics.
- In-place convenience API now exists on `TDigest<F>`:
  - `add(value)`
  - `add_many(values)`
  - `merge(&other)`
  - `merge_many(&[others...])`

### 5.2 CLI (`src/bin/tdigest_cli.rs`)

- CLI is used as a behavior reference in coherence tests.
- Precision flag is accepted (`f32|f64|auto`) but current CLI build path is f64-backed.
- Keep output format stability for tests (`csv`, header/no-header semantics).

### 5.3 Python native binding (`src/py.rs`)

- Route behavior through `src/tdigest/frontends.rs` (`FrontendDigest`) whenever possible.
- Keep this layer focused on Python argument/container conversion and exception translation.
- Do not re-implement merge/config/probe/training rules here if they already exist in Rust frontend helpers.

### 5.4 Python package wrapper (`bindings/python/gr_tdigest/__init__.py`)

- Keep this file mostly declarative: argument normalization, Polars expression wrappers, and ergonomic aliases.
- Polars plugin entry points are here (`tdigest`, `add_values`, `cdf`, `quantile`, `median`, `merge_tdigests`, `to_bytes`, `from_bytes`).
- Prefer pushing semantics into Rust (`src/py.rs` + shared frontend service) over Python-side monkey-patching.

### 5.5 Polars plugin (`src/polars_expr.rs`)

- Schema/dtype behavior is contract-sensitive:
  - Float32 input -> compact digest (f32-backed struct)
  - Float64/other numeric -> f64-backed struct
- `tdigest` training rejects null/non-finite inputs.
- `add_values` rejects null/non-finite values and supports scalar/numeric/list inputs.
- Probe nulls for CDF are strict errors.
- `from_bytes` enforces precision consistency across column values.

### 5.6 Java/JNI (`bindings/java/*`, `src/jni.rs`)

- Prefer Rust/JNI as the single source for semantic validation and merge/serde/query behavior.
- Java API should stay thin and ergonomic; avoid duplicating CDF/quantile special-case logic in Java when Rust already enforces it.
- JNI methods throw `IllegalArgumentException` for invalid user input.
- `Natives.java` load order:
  1. `-Dgr.tdigest.native=...`
  2. bundled JAR resource
  3. `System.loadLibrary("gr_tdigest")`
  4. local `target/release`
- Keep Java and JNI behavior aligned with integration contracts.

## 6. When touching behavior, run this minimum matrix

For behavior-affecting changes (validation, quantile/cdf semantics, precision, serialization):

1. `make build`
2. `make test`
3. `cargo test -- --quiet`
4. `./.venv/bin/pytest -q bindings/python/tests/test_api_python.py`
5. `./.venv/bin/pytest -q bindings/python/tests/test_api_polars.py`
6. `./.venv/bin/pytest -q integration/api_coherence/test_contract_behavior.py`
7. `./.venv/bin/pytest -q integration/api_coherence/test_contract_probe_validation.py`
8. `./.venv/bin/pytest -q integration/api_coherence/test_wire_python_polars.py`
9. `./.venv/bin/pytest -q integration/api_coherence/test_wire_java_interop.py`
10. `bindings/java/gradlew --no-daemon --console=plain -p bindings/java test`

If only one surface changed, still run the relevant coherence subset.

## 7. Simplification roadmap (heavy refactor)

### 7.1 Refactor north star

- Keep Rust as single source of truth for behavior and errors.
- Keep bindings thin adapters (argument conversion + error translation only).
- Eliminate duplicated validation/business logic across `py.rs`, `jni.rs`, CLI, and Python wrapper patches.

### 7.2 Current direction

- Shared Rust frontend service layer lives in `src/tdigest/frontends.rs` (`FrontendDigest`, config/probe/training/merge helpers).
- Continue migrating binding logic to this layer before adding new per-binding behavior.

- Phase 2: Centralize validation and error mapping.
- Move all probe/training/config validation into shared Rust helpers.
- Keep one canonical set of user-facing messages for each error class.

- Phase 3: Shrink Python wrapper patch surface.
- Prefer native Rust/PyO3 behavior over Python monkey-patch behavior where practical.
- Leave `__init__.py` focused on ergonomics and Polars plugin wiring, not core semantics.

- Phase 4: Consolidate merge/add patterns.
- Unify `add`/`merge` code paths in Rust core and have all surfaces use those APIs.
- Ensure parity tests remain the contract for capabilities.

- Phase 5: Polars adapter simplification.
- Extract repeated dtype/null/non-finite handling into helper functions or shared mini-modules.
- Keep expression functions declarative and short.

- Phase 6: Test architecture cleanup.
- Keep API coherence tests as capability/semantics contract.
- Minimize duplicated assertions by using shared fixtures/helpers and matrix-driven tests.

### 7.3 Refactor acceptance criteria

- `make build` passes.
- `make test` passes.
- No capability regressions in parity/coherence suites.
- Bindings (`src/py.rs`, `src/jni.rs`, `bindings/python/gr_tdigest/__init__.py`) are measurably smaller and have less duplicated validation logic.

## 8. Common pitfalls

- Do not silently relax strict validation rules without updating integration contracts.
- Do not introduce implicit precision promotion/demotion in wire decode paths.
- Do not change CLI CSV formatting lightly; tests parse exact columns.
- `README` examples can drift from implementation; treat tests + code as source of truth.
- For edge-policy changes (`edges` / protected edges), verify all four surfaces, not just core.

## 9. Commit hygiene for agents

- Keep changes scoped to the target surface plus necessary coherence tests.
- Prefer small, behaviorally explicit patches.
- If a change can alter cross-surface semantics, document it in test names/messages.
- When in doubt, preserve existing user-facing error wording that tests assert on.
- Use real multi-line commit messages when needed (actual newlines in the commit body; do not embed literal `\n` escape text).
- Every commit must include an update to `CHANGELOG.md` (at minimum under `## [Unreleased]`).
- Do not push a release tag while release notes still live under `## [Unreleased]`; move them into a versioned section first.

## 10. Documentation discipline

- Before starting a new work item, read both `api_design.md` and `tdigest_design.md`.
- When behavior or internals change, update `api_design.md` and/or `tdigest_design.md` in the same change.
- Keep `README.md` aligned with the current user-facing APIs and behavior for every change.
- If external-comparison/tradeoff context changes, also update `comparison_design.md`.
