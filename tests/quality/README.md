# Accuracy tests

For private artifacts, actual clusters and representative servers, see the
[owner checklist](OWNER_CHECKS.md). Numerical regression and approximate accuracy
remain separate acceptance questions.

`make test` includes the fast accuracy profiles, independent-reference controls,
and weighted tests. `make quality-fast` runs just these tests and retains a
measurement report. `make quality-extended` adds the slow matrix across storage
budgets, input orders and merge paths, and retains its report. `make quality-regression`
is the CI entry point for the extended tests.

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


## Reading the results

Rank error measures how far an answer falls from the requested position in the
sorted data, allowing for repeated values. Value error measures distance from a
sample percentile. CDF error compares the reported fraction below a value with
an exact count at selected points.

Both engines receive the same values and precision. Comparisons use a similar
amount of storage for centroid fields; this does not measure total memory use.
These datasets provide regression coverage, not a guarantee for every input.

The Rust tests in `tests/numerical_foundation.rs` cover small capacities, lost
weight, cancellation and fractional-weight medians as part of `make test`.
A known limitation remains: the mean of the smallest positive f64 value with
weight 0.1 rounds to zero.

The existing package tests check that source archives include these Python
tests, their saved reference results and the compressed input data. Tests use
the included data and never download it.

Quality runs verify that the source and installed native binary remain unchanged during measurement. Changed inputs invalidate the retained report, fail pytest teardown and prevent candidate baseline generation. Historical comparisons in [the recorded results](../../compat/quality-20261020/RESULTS.md) retain their original source identities and known accuracy limits.

## Wider capacity/scale diagnostics

`make quality-budgets` is a separate slower4140-profile measurement:3276on the
established full-data corpus plus864on12additional deterministic shapes. It tests
K2capacities16/32/64/128/200/512/1000and selectedQuad/K1sizes, actualf32/f64,
input reversal and40-way merge paths. Modern/original comparisons atK2/200and1000
reuse the actual logical-byte budget search. Original CDF remains unavailable.

The current source has114failed diagnostic profiles across the complete matrix,
including newly exposed small-capacity/shape failures. The runner exits nonzero;
these are not waived, skipped or disguised as completed accuracy fixes. The old
756-profile frozen baseline and its ten-case inventory are unchanged. Profiles
share data and are not independent trials. See the consumer accuracy report for
per-metric tradeoffs, original win/loss counts and reproduction instructions.
