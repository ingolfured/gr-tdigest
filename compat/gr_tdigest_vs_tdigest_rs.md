# gr-tdigest vs. tdigest-rs: compatibility status

Preserving the old `tdigest-rs` Python API is a goal that still has unresolved
failures. The strict harness runs the upstream assertions against the compiled
`gr_tdigest` extension through an import-only `tdigest_rs` shim. It does not
adapt numerical behavior or change the assertions.

## Reproduce the diagnostic baseline

After the normal development environment setup, run from the repository root:

```bash
make legacy-strict-test
make legacy-bench-smoke
```

Both commands rebuild the Python extension first. `legacy-strict-test` is an
alias for `compat-test` and propagates pytest failures as a nonzero exit status.
It is currently an opt-in diagnostic, outside `make test` and CI. Green standard
tests do not establish legacy compatibility.

The copied tests and original benchmark are unchanged from G-Research
`tdigest-rs` commit `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`; see
[`PROVENANCE.md`](tdigest-rs-upstream/PROVENANCE.md) for the source and checksums.
The benchmark Make targets execute the separate custom runner in
`tdigest-rs-strict-python/benchmarks/run.py`. `legacy-bench-smoke` uses a small
workload; `compat-bench` uses that runner's larger defaults. Neither target runs
the copied upstream benchmark, and a smoke pass is not an accuracy comparison.

## Verified baseline: 70 passed, 21 failed

Validation on 2026-10-14 used Linux under WSL, Python 3.12.12, Rust 1.90.0,
Java 21, and Python dependencies from the committed `uv.lock`. The library
implementation tested was PR #3 commit
`dc330c0fb0c7661252065b97a47a69de58702913`; the subsequent harness/documentation
correction does not change it. `make build`, `make test`, and
`make legacy-bench-smoke` passed. The strict legacy suite collected 91 cases:

| Failure group | Cases | Status |
| --- | ---: | --- |
| Median accuracy / equality | 13 | Defect and legacy API mismatch requiring fixes |
| Quantile accuracy | 1 | Outside an existing upstream tolerance; investigate |
| NaN / infinity input handling | 5 | Current strict contract differs from legacy behavior |
| Input dtype / merge dtype handling | 2 | Current precision and validation rules differ from legacy behavior |

These are 21 failing test cases, not necessarily 21 independent defects. The
failures have not been accepted as compatible behavior. Preserve the upstream
assertions and tolerances while investigating; compatibility acceptance still
requires the old suite to pass.

## Median and quantile failures

Twelve Gaussian cases fail median accuracy checks, and
`test_median_random_data` fails the old `median() == quantile(0.5)` contract.
A deterministic weighted example also demonstrates a median defect:

```python
from gr_tdigest import TDigest

d = TDigest.from_means_weights([0.0, 10.0], [90.0, 10.0], delta=300)
print(d.median())       # 5.0 at the baseline
print(d.quantile(0.5))  # 0.0 at the baseline
```

The digest retains those two means and weights. The represented population has
90 zeros and 10 tens, whose median is 0. This discrepancy needs a regression
test and a library fix; it is not justified by ordinary estimation error.

`test_uniform[1.0--2.0-10000]` also fails: at `q=0.75`, the estimate is about
0.274090 while NumPy gives 0.220019, outside the test's existing tolerance.
This requires investigation rather than a blanket claim that all estimates
remain within tolerance.

## Input and precision differences requiring resolution

The modern API rejects non-finite training values. The old tests expect NaN
filtering and permit infinities in some inputs. Five cases fail:
`test_only_nan_values`, `test_single_nan_value`, `test_only_inf_values`,
`test_non_inf_extremas`, and `test_merge_with_infs`. Any compatibility solution
must account for the modern cross-surface validation contract as well as the
legacy API requirement.

The backend currently defaults to f64 regardless of input array dtype, with f32
selected through `precision="f32"`. Float16 input is accepted, and float32 and
float64 arrays both produce f64 digests by default. Consequently,
`test_raises_invalid_input_type` and `test_raises_merge_different_types` do not
raise the `TypeError` expected by the old suite. These differences require an
explicit compatibility decision and implementation; they are not resolved by
describing them as intentional.

## Coverage beyond the copied tests

Additional runtime probes found issues that need follow-up regression tests:

- `TDigest.update` is absent from the current Python class.
- `merge` mutates and returns its receiver; compare this with the old API's
  return-new-digest behavior.
- Pickle, deepcopy, and byte-wire round-trips lose `delta`. A restored
  `delta=20` digest fails to merge with its original because configuration
  validation sees `delta=None` versus `delta=20`.

Passing the copied suite is the required baseline, but these additional API
and serialization checks are also needed before claiming a drop-in replacement.
Library fixes and their regression tests belong in follow-up changes; this
harness records the failures so they can be addressed reproducibly.
