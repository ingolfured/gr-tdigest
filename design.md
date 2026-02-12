# design.md

This is a **target-state** design document.

- It describes what we want the project to become.
- It may intentionally differ from current implementation.
- During refactor, when current behavior and target behavior differ, target behavior wins.

## 1. Locked decisions

These are now decided:

1. Rust-first architecture:
- Push as much behavior/validation/error logic as possible into Rust.
- Bindings should be thin adapters.

2. Polars null policy:
- Fail on null probes/null add-values (strict).
- Fail on null binary blobs in `from_bytes` (strict).
- Empty digests are valid and must be supported.

3. Explicit cast scope:
- Implement precision cast helpers first (`f32 <-> f64`).
- Reconfigure-cast (`scale`/`max_size`/`policy`) can come later as a separate explicit API.

## 2. Product goals

- Capability parity across Rust, Python, Polars, Java.
- One behavior contract across all faces.
- One error contract rooted in Rust.
- Cross-face wire compatibility via TDIG.

## 3. Capability parity contract

All faces must support:

- Build/train from numeric values.
- Add one value.
- Add multiple values.
- Merge one digest.
- Merge multiple digests.
- `quantile`
- `cdf`
- `median`
- `to_bytes` / `from_bytes` interoperability.

Polars existing names stay as-is (e.g. `add_values`, `merge_tdigests`).
For new APIs added during refactor, prefer the common cross-face names unless there is a Polars-specific constraint.

## 4. Query and validation semantics (target)

## 4.1 Training/add values

- Reject non-finite values: `NaN`, `+inf`, `-inf`.
- Reject nulls on surfaces that carry nulls (Polars).

## 4.2 Quantile probe (strict)

- `q` must be finite and in `[0,1]`.
- Error on:
  - `q=NaN`
  - `q=+/-inf`
  - `q<0` or `q>1` (example: `quantile(1.5)` must error)

## 4.3 Empty digest behavior

- `cdf(any probe)` -> `NaN`
- `quantile(valid q)`:
  - Rust/Python/Java: `NaN`
  - Polars: `None`
- `median()`:
  - Rust/Python/Java: `NaN`
  - Polars: `None`

## 4.4 Non-empty digest probe behavior

- `cdf(NaN)` -> `NaN`
- `cdf(-inf)` -> `0.0`
- `cdf(+inf)` -> `1.0`

## 4.5 NaN/null/empty matrix (normative)

This section is the source of truth for special-value behavior.

Build/train:

- Empty training input: allowed -> returns empty digest.
- Any NaN/`+inf`/`-inf` in training values: error.
- Any null training value (where representable, e.g. Polars): error.

Add values:

- Add scalar NaN/`+inf`/`-inf`: error.
- Add vector with any NaN/`+inf`/`-inf`: error.
- Add null values (where representable): error.
- Add empty vector/list: no-op success.

Merge:

- Empty + non-empty: success, same semantic result as non-empty digest.
- Empty + empty: success, empty digest.
- `merge_all([])`: returns canonical empty digest defaults (see section 9).
- Mixed precision or config mismatch: error unless user explicitly casts/rebuilds first.

CDF:

- Empty digest + any probe (finite, NaN, `+/-inf`): `NaN`.
- Non-empty + finite probe: finite value in `[0,1]`.
- Non-empty + NaN probe: `NaN`.
- Non-empty + `-inf`: `0.0`.
- Non-empty + `+inf`: `1.0`.
- Null probe(s) (Polars): strict error.

Quantile:

- Probe `q` must be finite and in `[0,1]`.
- `q=NaN`, `q=+/-inf`, `q<0`, `q>1`: error.
- Empty digest + valid `q`:
  - Rust/Python/Java: `NaN`
  - Polars: `None`

Median:

- Empty digest:
  - Rust/Python/Java: `NaN`
  - Polars: `None`
- Non-empty digest: finite number.

Serialization / `from_bytes`:

- Empty digest to/from bytes: valid.
- Mixed f32/f64 precision blobs in one logical deserialize operation: strict error.
- Null blobs (Polars column):
  - Any null blob row is a hard error.
  - No implicit skipping/ignoring of null blob rows.
  - All-null blob column is also a hard error (not treated as empty digest).
- Empty blob bytes (`b""`) are invalid TDIG and must error as decode failure (not treated as null and not treated as empty digest).

## 5. Merge semantics (target)

- Reject mixed precision merges (`f32` + `f64`) by default.
- Reject config mismatch merges across faces:
  - `max_size`
  - `scale`
  - `singleton_policy`
- No implicit coercion during merge.

Explicit path is allowed:

- Precision cast up/down is allowed only via explicit user action (never implicit in merge).
- Error messages must explain the fix path.

Error message style example:

- `"tdigest merge: incompatible digests (precision f32 vs f64). Cast explicitly before merge (e.g. cast_precision('f64'))."`
- `"tdigest merge: incompatible configs (scale k2 vs k3). Rebuild or cast to a shared configuration before merge."`

## 6. Serialization semantics (target)

- TDIG bytes written by one face must be readable by all others.
- Precision on wire (`f32`/`f64`) must round-trip correctly.
- Cross-face query behavior on deserialized digests must remain coherent.
- Mixed-precision blobs in a single logical deserialize operation are strict errors unless user explicitly normalizes/casts first.

## 7. Error model (Rust-first)

Rust defines canonical errors. Bindings only map to native exception types.

Canonical classes:

- `InvalidTrainingData`
- `InvalidProbe`
- `IncompatibleMerge`
- `DecodeError`
- `ClosedHandle` (Java/JNI lifecycle)
- `InvariantViolation`

Surface mapping:

- Rust: `Result<T, TdError>`
- Python: `ValueError`/`TypeError` with Rust-origin message
- Java: `IllegalArgumentException`/`IllegalStateException`
- Polars: `ComputeError`

## 8. Architecture target

- Introduce shared Rust frontend service module (internal).
- All operation paths (`build`, `add`, `merge`, `quantile`, `cdf`, `median`, `to/from_bytes`) call that module.
- Reduce duplicated logic in:
  - `src/py.rs`
  - `src/jni.rs`
  - `src/polars_expr.rs`
  - `bindings/python/gr_tdigest/__init__.py`
- Add explicit precision cast helpers in Rust and expose equivalents in other faces.

## 9. Resolved design decisions

These were explicitly chosen and are now target requirements.

1. Merge mismatch policy
- Strict fail on precision/config mismatch.
- No implicit merge coercion.
- Explicit cast/reconfigure is the only fix path.

2. Shape behavior for `cdf`/vector queries
- Preserve input style:
  - scalar in -> scalar out
  - list/array in -> list/array out (shape-preserving where applicable)
- This behavior should be consistent in spirit across faces.

Examples:

- Python:
  - `d.cdf(1.5)` -> `float`
  - `d.cdf([0.0, 1.0])` -> `list[float]`
- Java:
  - `d.cdf(1.5)` (scalar overload target) -> `double`
  - `d.cdf(new double[]{0.0, 1.0})` -> `double[]`
- Rust:
  - scalar helper target `cdf_scalar(1.5) -> f64`
  - batch `cdf(&[...]) -> Vec<f64>`
- Polars:
  - scalar probe expr -> scalar output column
  - list probe expr -> list output column

3. `merge_all([])` defaults
- One canonical default everywhere:
  - precision: `f64`
  - `max_size=1000`
  - `scale=K2`
  - `singleton_policy=Use`

4. Mixed precision in Polars `from_bytes`
- Strict error (no implicit upcast).

Examples:

- Input blob column contains:
  - row 0: f32 TDIG
  - row 1: f64 TDIG
- `td.from_bytes("blob", precision="auto")` -> error.

Error message style example:

- `"tdigest.from_bytes: mixed f32/f64 blobs in one column. Normalize precision first (e.g. decode+cast all to f64, then re-encode)."`

Valid cases:

- All f32 blobs + `precision="f32"` -> success.
- All f64 blobs + `precision="f64"` -> success.
- All f32 blobs + `precision="auto"` -> success (f32 result).
- All f64 blobs + `precision="auto"` -> success (f64 result).

5. Cast API scope (initial)
- Initial explicit cast API supports precision cast only:
  - `cast_precision("f32")`
  - `cast_precision("f64")`
- No implicit cast during merge/query/decode.
- Future optional API may support explicit config-rebuild cast.

## 10. Refactor acceptance criteria

- `make build` passes.
- `make test` passes.
- Capability/coherence/serialization tests pass.
- Binding layers are measurably smaller and thinner.
- Rust owns the core behavior and error logic.
