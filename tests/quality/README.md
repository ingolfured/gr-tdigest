# Accuracy tests

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
