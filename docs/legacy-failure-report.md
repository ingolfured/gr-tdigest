# Legacy Strict Test Failure Report

Strict target:

```text
make legacy-strict-test
```

Run date: 2026-05-25.

Harness note: the target runs from `compat/tdigest-rs-upstream/bindings/python` with
`compat/tdigest-rs-strict-python` first on `PYTHONPATH`, so `tdigest_rs` imports the
current `gr_tdigest` extension rather than the old pure-Python shim.

Result:

```text
70 passed, 21 failed
```

Benchmark smoke:

```text
make legacy-bench-smoke
tdigest-rs strict legacy benchmark passed | n=256 n_arrays=16 iterations=1 delta=100 workers=2 elapsed=0.003s checksum=-12.9975
```

No copied upstream test assertions were modified for this run.

## Failures

| Test | Old expectation | Actual new behavior/error | Classification |
| --- | --- | --- | --- |
| `tests/test_distributions.py::test_gaussian[1.0-0.0-1000]` | `tdigest.median()` close to `loc=0.0` with `rel_tol=0.1`, `abs_tol=0.02` | median `0.0643961536073634` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[1.0-0.0-10000]` | median close to `0.0` within old tolerance | median `-0.04015447938763013` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[1.0--0.1-1000]` | median close to `-0.1` within old tolerance | median `-0.03560384611872988` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[1.0--0.1-10000]` | median close to `-0.1` within old tolerance | median `-0.14015447945387044` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[1.0-0.1-1000]` | median close to `0.1` within old tolerance | median `0.16439615330468355` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[1.0-0.1-10000]` | median close to `0.1` within old tolerance | median `0.05984552055739676` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5-0.0-1000]` | median close to `0.0` within old tolerance | median `0.0321980768036817` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5-0.0-10000]` | median close to `0.0` within old tolerance | median `-0.020077239693815065` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5--0.1-1000]` | median close to `-0.1` within old tolerance | median `-0.06780192328759849` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5--0.1-10000]` | median close to `-0.1` within old tolerance | median `-0.1200772398997334` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5-0.1-1000]` | median close to `0.1` within old tolerance | median `0.1321980768518353` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_gaussian[0.5-0.1-10000]` | median close to `0.1` within old tolerance | median `0.07992276029376975` | Intended/new median behavior difference to review |
| `tests/test_distributions.py::test_uniform[1.0--2.0-10000]` | all checked quantiles close to NumPy with `rel_tol=0.1`, `abs_tol=0.05` | at `q=0.75`, NumPy `0.22001904`, digest `0.2740897487383336` | Needs review as possible accuracy delta |
| `tests/test_tdigest.py::test_median_random_data` | `tdigest.quantile(0.5) == tdigest.median()` | quantile `0.011473670828134623`, median `0.014026947250942398` | Intended new median contract difference |
| `tests/test_tdigest.py::test_only_nan_values` | NaNs are dropped; empty digest with no means/weights | `ValueError`: non-finite values are not allowed | Intended new strict training validation |
| `tests/test_tdigest.py::test_single_nan_value` | NaN is dropped and finite means remain | `ValueError`: non-finite values are not allowed | Intended new strict training validation |
| `tests/test_tdigest.py::test_only_inf_values` | infinities are accepted as digest means | `ValueError`: non-finite values are not allowed | Intended new strict training validation |
| `tests/test_tdigest.py::test_non_inf_extremas` | infinities are accepted and preserved at extrema | `ValueError`: non-finite values are not allowed | Intended new strict training validation |
| `tests/test_tdigest.py::test_merge_with_infs` | infinities are accepted and merged; expected `len(tdigest) == 6` | `ValueError`: non-finite values are not allowed while building the first digest | Intended new strict training validation |
| `tests/test_tdigest.py::test_raises_invalid_input_type` | `float16` input raises `TypeError` matching `TDigest is not implemented for arr with type` | no exception; input is accepted through numeric conversion | Intended current API difference to review |
| `tests/test_tdigest.py::test_raises_merge_different_types` | merging a `float32`-built digest with a `float64`-built digest raises `TypeError` matching `has a different type` | no exception; both default to the current f64 backend unless `precision='f32'` is explicit | Intended current precision-default difference to review |

## Next Review Decision

The likely keep/adapt/comment split is:

- Keep runnable as-is: the 70 passing copied tests.
- Comment old expectation and add new-contract tests: non-finite training behavior, `median()` vs `quantile(0.5)`, dtype/default-precision semantics.
- Review before deciding: the single uniform quantile tolerance miss and the gaussian median tolerance misses, because they mix old stochastic/tolerance assumptions with the new median/query behavior.
