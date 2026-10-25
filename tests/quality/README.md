# Accuracy tests

`make test` includes the fast accuracy profiles, independent-reference controls,
and weighted tests. `make quality-fast` runs just these tests and retains a
measurement report. `make quality-extended` adds the slow matrix across storage
budgets, input orders and merge paths, and retains its report. `make quality-regression`
is the CI entry point for the extended tests and the source-bound Rust boundary audit.

The measurements compare the modern and preserved original engines against sorted
input after precision casting. Rank, tail, value and CDF errors stay separate;
the original engine is matched by logical centroid field bytes without selecting
for accuracy. This is a comparison budget, not a heap-memory claim. Every metric
is checked against the existing frozen baseline with the same tolerances.

Ten extended profiles exceed the absolute accuracy envelope. They appear as
pytest expected failures only after invariants and baseline regression assertions
pass and the observed limits match `known-limits.json`. New failures, worse
baseline metrics and changed inventories fail normally; reports retain the
accuracy status as `FAIL` for known limits.

Run one profile through pytest's usual selection options, for example:

```bash
.venv/bin/python -m pytest tests/quality \
  -k 'accuracy and fast and bimodal and f64'
```

Add `--quality-extended -m slow` to collect and select the extended matrix.
`--quality-report build/quality` writes `<mode>/results.json`, including source
hashes, comparison summaries and failed cases. A filtered or interrupted run is
reported as `INCOMPLETE` and cannot generate a candidate baseline.

To generate candidate frozen measurements deliberately, run the complete matrix
with a new output directory:

```bash
.venv/bin/python -m pytest tests/quality \
  --quality-extended --quality-report build/quality \
  --quality-baseline-output build/quality-candidates
```

This writes candidate `baseline-fast.json.gz` and `baseline-extended.json.gz`
without replacing the checked-in files or bypassing assertions. Review the
measurement changes and provenance before replacing a baseline. The recorded
input fixtures and their hashes are unchanged; tests never download data.

## Numerical boundary controls

`make quality-boundaries` runs 15 hand-checkable Rust controls. After the numerical fixes in #55, 14 pass; subnormal aggregate-mean loss remains the one **failure**. The tiny-capacity mass, residual tail, cancellation and fractional-weight median controls now pass in both precisions. These are numerical measurements, not skipped tests.

The boundary inventory was updated from [CI run 153](https://github.com/ingolfured/gr-tdigest/actions/runs/40185398411) after checking that its native source hashes match the reviewed source and that the 12 changed controls all improved from FAIL to PASS without changing their expected values. The unresolved subnormal failure remains explicit.


## What each measurement means

- **Rank error:** distance from the requested probability to the observed-data rank interval of the returned value. Correct answers at repeated values get an interval, rather than an artificial tie penalty. Mean, maximum, median and tail errors stay separate.
- **Value error:** difference from a declared type7 sample percentile, both in original units and divided by the middle-half data spread. If that spread is zero, use the full range, then1 for constant data. This is a specified percentile convention, not the only possible convention for discrete samples.
- **CDF error:** difference from the exact empirical midpoint probability at135 fixed percentile-based probes (deduplicated). This is a sampled check, not a supremum/KS guarantee. The original G-Research API has no CDF method, so its CDF fields are unavailable, never fabricated.
- **State checks:** finite/positive weights, preserved total mass and weighted contribution, ordered centroids/queries, input bounds, actual storage precision and exact state round trips. Original rounding/order diagnostics are recorded separately and do not remove inconvenient comparison outcomes.

## Fair comparisons and limits

Both engines receive identical observations after the same precision cast. The original source/wrapper remain checksum-bound. A count-only search chooses the largest observed original summary within the modern summary's logical field-byte budget. It never chooses a parameter by accuracy. The search trace and actual sizes are recorded; it is not a proof of a globally optimal original setting.

Modern logical bytes count mean(4/8)+weight8+kind1 per centroid; original bytes count mean(4/8)+weight4. This excludes headers, padding, spare capacity, allocation and RSS. Modern wire length is recorded separately. Original states filling less than90% or exceeding100% of that field budget remain in the results but are excluded from matched-budget win/loss totals. Accuracy-limit failures **remain** in those totals when both measurements exist.

These are empirical fixtures on two related domains, not a universal accuracy guarantee. Reversed single-build cases can be identical, and many profiles share data. The counts are test coverage, not independent statistical trials or confidence levels. No throughput ranking is inferred from test duration.
