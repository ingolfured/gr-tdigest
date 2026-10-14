# Weighted point-mass audit evidence

This bounded independent audit checks **uncompressed weighted point data**, not
general t-digest approximation accuracy or performance. The default run uses seed
`202610140639`, 12 random sets of 12 integer-valued samples with duplicate values,
eight mass scales from `1e-300` to `1e300`, both mean precisions, original/reversed/
shuffled orders, sequential additions, and selected 40-way merges. With the fixed
implementation it covers 800 digests, 13,568 CDF checks, 19,968 quantile rank checks,
and 304 exact integer-mass checks. Runtime was below one second on the recorded
machine; this is not a benchmark claim.

Run with a Python environment containing the built candidate `gr_tdigest` package:

```sh
python audit_weighted_points.py --output audit-result.json --source-dir /path/to/gr-tdigest
```

The script imports the installed package; `--source-dir` only records source
provenance. It does not compile or install anything. Standard-library modules and
the package's existing NumPy dependency suffice. Python version, native library
path/hash, script hash and optional checkout hashes are included in the output.
These hashes identify artifacts; a source path alone does not prove that the
loaded library was built from that checkout.

**Exit code 0 means the ordinary gate passed; exit code 1 means it failed.**
Numerical mismatches, changed atomic means, lost mass and PyO3 panic exceptions
are recorded as failures. The script continues after those failures to collect
evidence but returns a failing exit code. KeyboardInterrupt/SystemExit propagate.
Failure detail is bounded, while all failure counters determine the exit status.
An unexpected harness exception also fails the process. Do not ignore exit codes.

Optional controls:

```sh
python audit_weighted_points.py --seed 202610140639 --trials 12 --output repeat.json
python audit_weighted_points.py --scales 0.1,1,16777216,1e9 --output narrower.json
python audit_weighted_points.py --include-extremes --output with-known-limits.json
```

Changing controls changes coverage and may change the random shuffle sequence.
Use defaults to compare with the recorded summaries. Merge probes are selected
at trials 0 and 5 and scales `1e-46`, `0.1`, `1`, and `1e9`.

## Reference and tolerances

`Fraction` represents the exact binary value of each supplied float weight;
duplicate values are coalesced independently of the library. With capacity 1000
and at most 11 distinct integer values, every centroid should remain an atomic
point at its original value. CDF uses midpoint ECDF over ties:
`(weight strictly below x + weight equal to x / 2) / total`.

Quantiles must lie in the exact weighted rank interval for the returned value.
Between observed values, that interval is a single cumulative boundary. Rank/CDF
tolerance is `5e-14` absolute for normal floating accumulation; individual mass
tolerance is `2e-14` relative. Integer totals through `2**53` are also checked
exactly against stored weights and the serialized count. This avoids specifying
an arbitrary interpolation value at an exact point-mass boundary.

## Recorded comparison

`recorded_results/before.json` and `after.json` preserve the historical counters
from the original scratch harness. `source_provenance.json` records its script
and source hashes. The portable script retains the same default dataset and
reference, adding controls, explicit failure status and artifact provenance.
`minimal_counterexamples.json` contains concise ordinary-range reproducers.

Before the single-centroid preservation fix, the audit recorded 57 mean-drift,
61 CDF, 121 quantile-rank and 13 mass reports, plus 5 merge/add panics. Reports can
overlap on a single digest. After the fix, all 800 digests completed with no
ordinary-range failures. The bug also existed in the baseline engine: debug
builds panic on duplicate output means; optimized builds can silently retain them.

## Known limits outside the passing gate

`--include-extremes` separately reports minimum-subnormal weights and overflowing
aggregate masses in both precisions. These are **known failing edge cases**, not
evidence of correctness and not part of the passing gate. Nonfinite results are
written as JSON strings. The flag does not excuse any ordinary gate failure.

At weights `5e-324`, half-weight/query rank calculations can underflow. Finite
individual weights `[1e308,1e308]` overflow the total, producing invalid CDFs or
a duplicate-value constructor panic. The retained source fixes neither limit.
The ordinary gate also does not establish compressed-data accuracy, preserve
historically lost mass, prove every fractional addition survives f64 rounding,
or replace canonical build/test and installed-wheel validation.

The portable gate was also checked against the baseline installed engine using
`--trials 1 --scales 0.1`: it reported failures and exited with status 1. Its
negative-control result is retained separately and uses narrower coverage than
the historical before/after comparison. `recorded_results/portable_after.json`
is the successful full default gate; `separate_extreme_probes.json` demonstrates
the optional known-limit reporting with a smaller passing ordinary sample.
