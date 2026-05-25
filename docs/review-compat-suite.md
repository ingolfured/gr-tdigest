# Upstream Compatibility Suite

This repository includes a separate copied review suite from the old upstream
`tdigest-rs` project. It is not mixed into the normal tests because the suite
checks legacy Python API behavior, including behavior that intentionally differs
from the stricter `gr_tdigest` public API.

## Location

Copied tests:

```text
compat/tdigest-rs-upstream/bindings/python/tests/
```

Copied benchmark:

```text
compat/tdigest-rs-upstream/bindings/python/benchmarks/run.py
```

Provenance and checksums are recorded in
`compat/tdigest-rs-upstream/PROVENANCE.md`.

## Strict Adapter

The copied suite imports `tdigest_rs`. The strict review target points that name
at `compat/tdigest-rs-strict-python/tdigest_rs/`, which re-exports the current
`gr_tdigest` package. This is intentionally different from the old top-level
pure-Python shim: the strict target exercises the real Rust/Python extension
and records behavior differences instead of hiding them.

Current reviewer-facing support includes:

- `TDigest.from_array(arr, delta=...)`;
- `TDigest.from_means_weights(...)`;
- `means` / `weights` arrays;
- legacy `merge(..., delta=...)` call compatibility;
- `trimmed_mean`;
- `to_dict` / `from_dict`;
- pickle and deepcopy support.

The strict adapter must not relax the strict validation rules documented in
`api_design.md`.

## Running

Run the exact copied test suite:

```bash
make legacy-strict-test
```

Run the strict manual benchmark smoke:

```bash
make legacy-bench-smoke
```

Run the larger manual benchmark:

```bash
make compat-bench
```

The benchmark is intentionally not part of default `make test` because the
full-size run is large.

## First Strict Result

The first strict run against the real `gr_tdigest` extension produced:

```text
70 passed, 21 failed
```

See `docs/legacy-failure-report.md` for the exact failure inventory.

## Obsolete 1/4 Branch

The abandoned 1/4 branch is obsolete for this strategy. Do not base future work
on it. Only the manually ported pieces are carried forward here:

- real delta mode;
- Python delta constructor path;
- pickle/deepcopy support;
- strict legacy test harness;
- manual legacy benchmark runner.

## Duplication Scan

After the strict inventory, the first non-legacy scan found overlap in these
areas:

- Python API smoke tests and copied legacy tests both cover `quantile`, `median`,
  merge, `from_means_weights`, pickle/deepcopy, and validation shape.
- Rust core tests now cover the old K2 delta cluster rule directly in
  `src/tdigest/compressor.rs`.
- Cross-surface coherence tests overlap with Python/Polars behavior tests for
  validation, CDF/quantile/median, precision, and wire round-trips.

Keep the cross-surface coherence tests even where they overlap; they prove a
stronger contract than the copied Python-only legacy tests. Candidate future
dedupe should be limited to repeated single-surface Python assertions once the
legacy failures are reviewed.

## Review Rule

Do not edit files under the copied `tests/` or copied `benchmarks/run.py` unless
the source snapshot is intentionally refreshed. Add wrappers, fixtures, or
documentation outside the copied tree.
