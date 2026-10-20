# What the broader quality checks found

The modern engine usually gives better percentile positions on the recorded datasets. It also usually has the smaller average value error there. The original often gives the smaller worst value error. There is no overall winner on every measure.

This change adds stronger tests and evidence. It leaves the calculations in both engines unchanged.

## Recorded data

We used every row from two public datasets:17,379 hourly bike-rental records and19,735 household-energy records. Nine measured columns cover counts, energy use, temperature, humidity and wind. Artificial random columns were excluded. The checked-in compressed data total about215KiB and need no internet during testing.

[Attribution, licenses and exact source hashes](../../tests/quality/data/README.md) are included with the data.

There are324 recorded-data profiles. Each varies the stored number precision, summary size, order or method of merging40 parts. All324 fit the stated comparison budget and pass the chosen empirical accuracy limits.

| Measurement | Modern better | Equal | Original better |
| --- | ---: | ---: | ---: |
| Average percentile-position error | 279 | 0 | 45 |
| Worst percentile-position error | 264 | 23 | 37 |
| Median-position error | 190 | 110 | 24 |
| Worst position error near the ends | 295 | 17 | 12 |
| Average value error | 260 | 0 | 64 |
| Worst value error | 119 | 0 | 205 |

“Position” asks where the answer falls among the actual observations. “Value” measures the difference in the returned number. Both matter: a small position error across a large gap can still mean a large value error.

These are related profiles sharing the same data. They are coverage counts, not independent trials or a probability that one engine will win on new data.

## Wider coverage and remaining failures

The complete study has756 profiles, including12 deliberately different distributions, both precisions, three sizes, reversed input and three build/merge paths. Of these,702 meet the matched-budget rule. The other54 remain in the files and are excluded from the matched-budget totals because the original uses too little or too much of that budget.

The difficult synthetic cases change the result: across all702 matched profiles, modern average position error wins479 and loses223. Average value error wins345 and loses357. Worst value error wins194 and loses508. The complete tables, including all losses, are in the measured files.

Ten profiles **fail** the accuracy limits: eight repeated-merge cases with two separated groups, plus two cases with a broad range of positive and negative values. Each uses the smallest tested summary size,32. Worst position error reaches about14.54 percentage points in this group. These failed profiles stay in the win/loss totals.

A separate set of15 small exact Rust controls has13 numerical failures. It exposes lost input weight at capacities1/2, a dropped tiny tail contribution, cancellation, asymmetric fractional-weight medians and a subnormal aggregate mean that becomes zero. The two equal-fraction controls pass. These defects are reproducible and unresolved; they are not skipped or counted as passing accuracy.

Four original f32 real-data profiles also exceed the weighted-contribution diagnostic. Their query measurements remain in the comparison. The preserved original implementation has not been altered to improve its scores.

## Fast enough to keep running

`make test` includes90 small reference/weighted checks and42 fast comparison profiles. The extended study is a single bounded run, rather than many large slow tests. On this machine it took roughly9 seconds with the release wheel and51 seconds with the development build. These are observed test costs, not a throughput comparison between engines.

The checks reject a corrupted dataset before use. A deliberate bad-reference test triggered38 quality-regression findings, confirming the acceptance gate can fail.

The installed release wheel also ran482 ordinary tests plus the42 fast and756 extended profiles on each of Python3.12,3.14,3.15. The ordinary tests passed; the same ten accuracy-limit failures remained explicit on every runtime. Package and source-archive identities are recorded in [validation.json](validation.json).

## How the comparison stays honest

Both engines receive the same observations at the same input precision. The original's parameter is chosen using size alone. We compare logical bytes used by centroid fields and record the actual sizes and parameter search; this is not a measurement of total memory use.

The reference answers come from sorted observed data. Repeated values have a valid position interval. Value comparisons use a declared sample-percentile convention. Cumulative probabilities are checked separately against exact midpoint probabilities at selected observations. The original API has no cumulative-probability method, so no result is invented for it.

`make quality-extended` and `make quality-boundaries` currently fail their strict accuracy checks. `make quality-regression` can pass only by matching the explicit known findings and frozen per-profile limits, while retaining `accuracy_status=FAIL`. A passing regression check means the recorded situation has not worsened; it does not mean these defects are fixed.

Use the recorded results to choose a size and merge strategy for the workload. The evidence supports better typical percentile positions on these real datasets, alongside clear limits for small summaries, extreme arithmetic and worst value error. It does not establish a universal guarantee or a ranking against other libraries.

## Reproduce and inspect

- [Test methods and commands](../../tests/quality/README.md)
- [Complete fast measurements](fast-results.json.gz)
- [Complete extended measurements](extended-results.json.gz), with every profile, error, size and original-parameter search
- [Known accuracy-limit cases](../../tests/quality/known-limits.json)
- [Exact boundary outcomes](../../tests/quality/known-boundaries.json)
- [Validation and artifact identities](validation.json)

The final build checks also passed on Java11 and17 using only the JDK/Gradle archive environment settings, with `JAVA_HOME` unset and host Java commands disabled. Gradle and direct cross-language test processes now select the same verified JDK. The final source archive includes that wiring, and its Python3.15 rebuild reproduces the already qualified Python/native package contents.
