# Legacy tdigest-rs compatibility

The wheel ships two Python interfaces over the same Rust t-digest core:

- `tdigest_rs.TDigest` preserves the original G-Research Python API: default
  `delta=300`, float32/float64 dtype inference, NaN filtering, infinity endpoint
  masses, uint32 weights, and return-new-digest `merge` and `update`.
- `gr_tdigest.TDigest` retains the modern API: explicit precision/configuration,
  strict finite training input, and in-place merge/add operations.

The legacy interface delegates compression, merging, and numerical queries to
Rust. Its Python code converts arguments and NumPy arrays. The former test-only
import shim has been removed; tests import the package that ships in the wheel.

## Required validation

The unchanged upstream Python suite contains **91 cases**. All 91 pass on Linux
after the compatibility fixes. No upstream assertion, tolerance, or test case
has been weakened, skipped, or marked as an expected failure.

```bash
make build
make test                 # Rust, Java, Python/coherence, and all 91 upstream cases
make legacy-strict-test   # just the upstream cases, rebuilding the extension first
make legacy-wheel-test    # build and install a wheel in a clean environment, then run all 91
```

The Linux CI build-and-test job runs `make test` and `make legacy-wheel-test`;
either suite failing fails the job. The installed-wheel check verifies that both
the Python interface and native extension resolve inside the temporary virtual
environment, with no source-checkout import shim.

Source: G-Research `tdigest-rs` commit
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The six Python test/support files and
original benchmark remain byte-for-byte unchanged; see
[`PROVENANCE.md`](tdigest-rs-upstream/PROVENANCE.md).

## Defects corrected

- Median uses the shared weighted `quantile(0.5)` calculation. Means `[0, 10]`
  with weights `[90, 10]` now return median `0`, including through Polars and Java.
- Merge normalization preserves whether a centroid represents a distribution
  or an exact point mass. Repeated merges no longer turn compressed centroids
  into artificial point masses that distort quantile/CDF results.
- Fractional atomic weights retain their mass and use weight-aware interpolation.
- The legacy interface honors requested merge/update delta and leaves operands
  unchanged. Copies and pickles preserve precision, delta, and infinity weights.
- Default TDIG v3 serialization preserves delta. Explicit v1/v2 encoding of
  delta-mode digests is rejected because those versions have no delta field.
- Polars structs preserve centroid kind and compression configuration. Readers
  still accept older structs, using their historical defaults and kind heuristic.

## Compatibility details

`from_array` and `from_means_weights` on `tdigest_rs.TDigest` accept one-dimensional
NumPy float32 or float64 arrays. Mixed-precision merges raise `TypeError`.
Legacy weights are positive integers with a total bounded by uint32 capacity;
invalid/overflowing totals raise errors. NaNs are discarded, while negative and
positive infinities are retained as endpoint masses outside the strict finite
core. Modern training rules are unchanged.

The legacy `to_dict`/`from_dict` representation retains its original `means` and
`weights` keys and default-delta reconstruction behavior. Pickle uses a richer
state to preserve the full configuration, and can also read the old dictionary
pickle state. `update(buffer, delta, merge_delta)` applies the separate build and
merge deltas, returning a new digest.

The earlier diagnostic baseline was 70 passed / 21 failed. Those failures were
unresolved defects and API differences, not evidence that tolerances should be
relaxed. Passing the suite establishes the tested legacy Python contract; it
does not imply byte-identical centroids or results for every possible input.

## Benchmark scope

`make legacy-bench-smoke` runs a small custom threaded construction/merge/query
workload; `make compat-bench` uses the same runner's larger defaults. These targets
rebuild the extension. The copied original benchmark is retained for reference
and is not executed by these targets. A smoke result is not an accuracy or
performance comparison against the original implementation.

## Test ownership

Keep the copied tests and original benchmark unchanged unless intentionally updating the pinned snapshot and checksums. Add new regressions in the project test directories. Retain cross-surface coherence coverage even when an upstream Python test covers similar behavior: it verifies the same contract through Polars, Java, and the CLI.
