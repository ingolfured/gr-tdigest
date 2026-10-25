# api_design.md

This is the **target-state API contract** for cross-surface behavior (Rust, Python, Polars, Java).

- It defines what behavior we want after refactor completion.
- It may differ from current implementation.
- If current behavior and this target contract diverge, this document wins.
- For external implementation/paper comparisons and tradeoff positioning, see `comparison_design.md`.

## 1. Scope boundary

This file defines:
- Public capabilities and semantic contracts.
- Validation, merge, serialization, and error behavior across surfaces.

This file does not define:
- TDigest algorithm internals or compression mechanics.
- External implementation comparisons.

For current implementation internals, see `tdigest_design.md`.

## 2. Locked decisions

1. Rust-first behavior ownership.
- Validation, merge policy, query rules, and decode policy live in Rust.
- Bindings stay thin (argument conversion + error mapping).

2. Strict null policy on Polars surfaces.
- Null probes and null add-values are errors.
- Null blobs in `from_bytes` are errors.

3. Explicit cast only.
- No implicit precision/config coercion in merge or decode.
- Polars struct decoding rejects delta configurations unless the scale is K2Norm and singleton policy is Off, matching binary decoding and training validation.
- Precision cast helpers are explicit API calls.

## 3. Capability parity contract

### JVM build and runtime support

The Java binding targets Java 11 bytecode, with Java/JNI tests, JAR smoke tests
and compiled-class version checks on JDK 11, 17, 21 and 25. The default
Gradle 8.14.5 build can itself run on JDK 11; JDK 25 builds select Gradle 9.1.0
through the verified distribution override. Internal build environments
can select verified JDK/Gradle archives and an exclusive Maven dependency
repository through the documented `TDIGEST_*` environment settings. Offline mode
uses pre-staged tools/dependencies and never attempts a public fallback. It
applies to Gradle and the native Cargo build. Publishing coordinates and the
numerical API are unchanged.

All surfaces must support:
- Build/train from numeric values.
- Add one value.
- Add multiple values.
- Add weighted value(s) (`values[i]` + `weights[i]`).
- Merge one digest.
- Merge multiple digests.
- `quantile`
- `cdf`
- `median`
- Explicit precision cast (`f32`/`f64`).
- `to_bytes` / `from_bytes`

Notes:
- Polars naming can stay surface-specific (`add_values`, `merge_tdigests`).
- New APIs should prefer common cross-surface names unless a surface constraint requires otherwise.
- CLI maps query capabilities to subcommands: `tdigest quantile`, `tdigest cdf`, `tdigest median`.
- CLI build/serialization flow is explicit: `tdigest build --to-digest ...`, then query via `--from-digest` (with optional additional training/merge input).

## 4. Normative semantics (target)

### 4.1 Training and add

- Empty training input is valid and returns an empty digest.
- Any non-finite value (`NaN`, `+inf`, `-inf`) is an error.
- Any null value (where representable) is an error.
- Adding an empty vector/list is a no-op success.

### 4.1.1 Existing Python interface

`tdigest_rs.TDigest` preserves the original public API and numerical behavior,
not merely approximate accuracy or passing upstream tolerances. It uses the
pinned original Rust arithmetic and unmodified Python wrapper. Existing calls,
dtype inference, uint32 weights, immutable merge/update, dictionary state, and
copy/pickle reconstruction retain their upstream behavior, including quirks.
The current review baseline keeps those calculations unchanged. Small numerical
changes can be considered when they have a concrete benefit and measured impact;
report absolute/relative differences and rank error on the affected workloads,
including tails and repeated merges. There is no universal absolute tolerance:
`1e-6` has different meaning for data near zero and data near a billion. Keep the
independent upstream oracle and document deliberate differences rather than
regenerating fixtures to hide them. The modern engine is explicitly selected
through `gr_tdigest`; matching delta alone does not make its answers identical. The original interface has no new delta
property, cdf method, or wire-state format. Focused regression tests cover immutable merge/update, dtype, copy/pickle, and the original dictionary state.
The unchanged 91-case upstream suite, 70 exact original-output cases, and pinned-source checksums are required by `make test` and installed-wheel CI.

### 4.1.2 Original Rust interface

The `tdigest_core` workspace package exposes the original root `TDigest<T>` for
f32/f64 with public `means: Vec<T>` and `weights: Vec<u32>` fields. Its seven
methods and `anyhow::Result` signatures retain the pinned original semantics,
including permissive probes and inherited malformed-input behavior. The modern
`gr-tdigest` API remains separate. New Rust checks cover both precisions and the
same independently generated original-output fixture as the Python gate; the
pinned upstream revision did not contain a Rust test suite.

### 4.1.3 Python runtime and installation

The Python distribution supports standard GIL builds of CPython 3.10–3.15 through a py310
ABI3 wheel, with NumPy >=1.26.4,<3 on 3.10–3.12, >=2.3.4,<3 on
3.13–3.14, and >=2.5.4,<3 on 3.15. The base installation exposes both NumPy numerical
interfaces without requiring the Polars Python package. The optional `polars`
extra accepts Polars >=1.0.0 without an upper cap. Installing either the base
package or its extra preserves an already installed compatible Polars version.
Development remains pinned to 1.44.2. The installed-artifact gate exercises
the same compiled wheel against 1.0.0, 1.20.0, 1.38.1, 1.44.2 and 2.0.0 across
all supported Python versions, including consumer cardinality/schema and streaming regressions.
The open dependency range permits newer releases; compatibility with unreleased
versions cannot be guaranteed before testing. Existing numerical imports
remain lazy, while expression operations require Polars to be installed.

The installed-artifact gate reuses one real wheel across the tested Polars
versions and all six Python versions, using both NumPy lines on 3.10–3.12 and
NumPy 2 on 3.13–3.15. It also
tests genuinely Polars-free installations on every runtime. The default gate
builds that ABI3 wheel once from the verified source archive with Python 3.10,
reusing Cargo dependency artifacts, then installs it on every supported runtime.
The 3.15 cell uses SciPy >=1.18.1 and pytest >=9.1.1 for compatibility validation;
these are test dependencies. Standard CPython is covered; free-threaded runtimes
and PyPy need separate qualification.
The bounded Python metadata is narrower than upstream's open-ended >=3.10
declaration. The development tool environment remains Python 3.12.

Modern NumPy training inputs are copied into owned Rust vectors. Compatible aligned/native-endian/contiguous Float32/Float64 arrays use a bulk copy; other layouts/dtypes retain their generic conversion and validation behavior. Constructors and updates neither retain nor mutate input storage.

### 4.1.4 Modern Python constructor concurrency

`gr_tdigest.TDigest.from_array` and `from_means_weights` convert their Python inputs to owned Rust vectors before releasing the GIL for native construction. Each call builds a private digest and retains existing validation and numerical order. Independent calls may overlap in a caller-managed pool; input conversion remains serialized by the GIL. This does not add a shared-object mutation guarantee or change the original `tdigest_rs` interface.

### 4.2 Queries

Quantile:
- Probe `q` must be finite and in `[0,1]`.
- `q=NaN`, `q=+/-inf`, `q<0`, `q>1` are errors.
- Empty digest + valid `q`:
  - Rust/Python/Java: `NaN`
  - Polars: `None`

CDF:
- Empty digest + any probe returns `NaN`.
- Non-empty digest:
  - finite probe -> finite value in `[0,1]`
  - `NaN` probe -> `NaN`
  - `-inf` probe -> `0.0`
  - `+inf` probe -> `1.0`
- Null probe(s) in Polars are errors.

Median:
- Empty digest:
  - Rust/Python/Java: `NaN`
  - Polars: `None`
- Non-empty modern digest returns `quantile(0.5)` using centroid weights, including fractional weights.

Shape behavior:
- Scalar probe in -> scalar out.
- Vector/list probe in -> vector/list out (surface-appropriate shape preservation).

### 4.3 Merge

- Precision mismatch merge is an error (`f32` vs `f64`).
- Config mismatch merge is an error:
  - `max_size`
  - `scale`
  - `singleton_policy`
  - `delta`
- No implicit coercion in merge.

Canonical empty for `merge_all([])`:
- precision: `f64`
- `max_size=1000`
- `scale=K2`
- `singleton_policy=Use`
- `delta=None`

### 4.4 Serialization / `from_bytes`

- TDIG written by one surface must decode on all surfaces.
- `to_bytes` supports explicit version selection (`1|2|3`) where surfaced.
- Wire precision (`f32`/`f64`) and delta must round-trip in default TDIG v3.
- Delta-mode digests require v3; wrappers reject explicit v1/v2 export rather than losing configuration.
- V3 uses flag bit1 and an 8-byte header extension for delta, and flag bit2 for full-width edge pins above 255. Extensions follow flag order and are checksum-covered; older readers reject unknown flags.
- Large edge pins require v3; explicit v1/v2 exports fail instead of truncating configuration.
- V1 requires positive integer centroid weights and a total within u64 range; fractional export fails rather than rounding away mass. V1's historical centroid-kind inference remains lossy for some weighted data; default v3 preserves kind.
- Invalid encoded weights return decode errors instead of panicking in centroid constructors.
- Polars structs retain centroid atomic/mixed kind, scale, singleton policy, pins, and delta.
  Older structs remain readable with their historical default configuration and kind inference.
- Python pickle/deepcopy use TDIG bytes.
- Empty digest to/from bytes is valid.
- Empty blob bytes (`b""`) are invalid TDIG and must error.
- Null blobs (Polars) are errors.
- Mixed `f32`/`f64` blobs in one logical deserialize operation are errors unless the user explicitly normalizes first.

## 5. Error model (target)

Rust defines canonical behavior/errors. Bindings map to native exception classes.

Canonical classes:
- `InvalidTrainingData`
- `InvalidProbe`
- `InvalidScale`
- `IncompatibleMerge`
- `DecodeError`
- `ClosedHandle` (Java/JNI lifecycle)
- `InvariantViolation`

Surface mapping:
- Rust: `Result<T, TdError>`
- Python: `ValueError` / `TypeError`
- Java: `IllegalArgumentException` / `IllegalStateException`
- Polars: `ComputeError`

## 6. Architecture target

- Keep `src/tdigest/frontends.rs` as the shared semantic layer.
- Route all public operation paths through shared Rust behavior.
- Continue removing duplicated validation/business logic from:
  - `src/py.rs`
  - `src/jni.rs`
  - `src/polars_expr.rs`
  - `bindings/python/gr_tdigest/__init__.py`
- Expose explicit precision-cast APIs across surfaces.

## 7. Acceptance criteria

- `make build` passes.
- `make test` passes.
- Capability/coherence/serialization suites pass.
- No documented cross-surface contract drift.
- Binding layers are thinner with less duplicated validation logic.

## Empirical accuracy contract

Quality acceptance uses separate rank, percentile-value and midpoint-CDF measurements against sorted observed data, with tie intervals and actual precision casts. The original API remains numerically unchanged and has no fabricated CDF comparison. The fast suite runs in `make test`; extended results and explicit unresolved boundary controls are documented in `tests/quality/README.md`. A passing known-outcome regression check must not be presented as passing accuracy. These are measured operating envelopes, not formal guarantees or universal superiority over the original engine.

## Checked modern state

Modern builders, reconstruction and precision conversions return checked results. Merges validate matching configuration even for empty contributors and replace the receiver only after validating the prepared result.

State validation checks centroid mass, ordering, finite statistics, aggregate-moment consistency and configuration. Fully atomic state must have extrema matching its observed points; protected-edge counts must be positive. Empty additions/merges and single-item frontend merges still validate existing state. Export validation uses the same contract. Struct reconstruction distinguishes a missing centroid list from an empty list and accepts either complete modern configuration or the legacy schema with no configuration fields. Binary reconstruction rejects trailing payload on empty digests and integer fields outside the platform range.

The raw Float64 sum remains authoritative across compression and precision conversion. Moment checks compare normalized quantities, with a rounding envelope that allows prior Float32 storage even after upcasting, plus subnormal quantization. These checks reject detectable contradictions; they cannot authenticate the original observations or prove statistical accuracy from a compressed summary.

Legacy TDIG v1 export rejects mixed unit-weight summaries when its historical kind heuristic would reconstruct atomic centroids inconsistent with the stored extrema; v2/v3 preserve the centroid kinds.

Positive transformations prepare and validate independent state before commit; failures preserve the receiver. Exact atomic points may retain a rounded zero moment without losing their known location or mass.

Polars rejects missing digest rows, checks training representability, and requires a binary precision hint. CDF broadcasting follows input lengths; per-row List expressions retain List output, while Python sequences provide a flat probe vector.

## Modern Python object API

The modern constructor accepts explicit positive weights. Copy and pickle preserve independent full state. A one-dimensional NumPy query returns an owned Float64 array even when it has one element; zero-dimensional probes return scalars, lists and tuples retain their containers, and multidimensional probes are rejected. See bindings/python/OBJECT_API.md.

## Polars expression cardinality

CDF chooses aligned rows or one-digest broadcasting from input shape. Missing states and incompatible contributors are errors. Column-based from_bytes requires explicit f32/f64 precision and reduces each logical group, including empty groups. Planned schema must equal the runtime schema. See bindings/python/POLARS_API.md.

## Modern Java ownership and serialization

Java additions and fluent merges mutate only the receiver after validation. Serialized Java state contains TDIG bytes, never a process-local native handle; restoration owns a fresh handle. Closed handles fail predictably. Catchable native failures become Java exceptions. JNI selects the system allocator; independent digests may be used concurrently, but shared-handle mutation requires caller coordination.

## Scala and Kotlin consumer examples

Public examples compile Scala 2.12.16 and Kotlin consumers against the shared Java JAR. The local authenticated Maven fixture exercises selected dependency resolution and prepared-cache offline use. Actual Spark driver/executor, serializer and checkpoint qualification remains a consumer-environment check.

Kotlin example compilation tracks its source and application classpath so incremental builds recompile after either changes.

## Installed Python discovery

The installed package includes native method docstrings, public type stubs and py.typed metadata. The object and Polars guides describe defaults, shapes, mutation, ownership and errors separately.

## Source-bound delivery gates

Build checks reject an incompatible selected Rust/Cargo toolchain before native compilation. Installed-wheel qualification records exact runtime cells and the source-archive build.
