# Release comparison at committed baseline 756d221

The full 30-case portable matrix completed on clean commit `756d221d35bd6d382cdfb8705598bb445a665997` with no source changes during compilation or timing. [Metadata](results/2026-10-14-756d221-metadata.json) identifies compiler, CPU, source and binary hashes; [raw results](results/2026-10-14-756d221.jsonl) retain every timing batch and accuracy field.

This is a comparison of the original G-Research core with the opt-in modern core. It excludes Python/Java/Polars overhead. The compatibility interface keeps original arithmetic; its result-vector compaction is measured separately. It is not evidence of global speed/accuracy leadership.

## Setup and attribution

Rust 1.90.0 release optimization, Linux/WSL on Intel Core Ultra 9 288V, one pinned CPU, delta=300, both f32/f64; 2026-10-14T21:26:50.255169+00:00 to 2026-10-14T21:30:07.833511+00:00. Independent original source is pinned to `8d9cf1fe82c5c9254e5dc9593f01717e317f4333` and verified before compilation. Each operation has warmup/calibration plus seven timed batches. Inputs are generated outside the timed section; required API copying and destruction are included.

The earlier intermediate run and its source replay patch remain preserved in [RESULTS.md](RESULTS.md). Current measurements include the committed first-batch fixes. Scalar-CDF optimization remains only a scratch proposal and is not included here.

## Construction, p50 and merging

Times are microseconds. Merge speed ratios are original / modern, so values above 1 favor modern. The modern k-way column is a different all-at-once operation; its result need not equal repeated pairwise merging.

| Case | Precision | Centroids old/new | Build old/new µs | p50 old/new µs | Sequential merge ratio | K-way speed ratio (old sequential/new) |
|---|---|---:|---:|---:|---:|---:|
| uniform n=1 | f64 | 1/1 | 0.0815/0.09558 | 0.002108/0.00138 | — | — |
| uniform n=1 | f32 | 1/1 | 0.08025/0.09531 | 0.002146/0.001346 | — | — |
| uniform n=2 | f64 | 2/2 | 0.09653/0.1055 | 0.002909/0.004922 | — | — |
| uniform n=2 | f32 | 2/2 | 0.09408/0.1108 | 0.00292/0.005743 | — | — |
| uniform n=10 | f64 | 10/10 | 0.2212/0.2753 | 0.005536/0.006715 | — | — |
| uniform n=10 | f32 | 10/10 | 0.1953/0.3624 | 0.005815/0.008184 | — | — |
| uniform n=1,000 | f64 | 128/128 | 15.31/18.56 | 0.04182/0.04803 | 0.52 | 2.82 |
| uniform n=1,000 | f32 | 128/128 | 18.96/21.47 | 0.06425/0.0573 | 0.34 | 2.42 |
| uniform n=100,000 | f64 | 142/142 | 3739/3026 | 0.06994/0.05508 | 0.61 | 1.53 |
| uniform n=100,000 | f32 | 142/142 | 3362/2954 | 0.07113/0.06843 | 0.47 | 1.24 |
| uniform n=1,000,000 | f64 | 146/146 | 6.596e+04/5.282e+04 | 0.0683/0.05622 | 0.60 | 1.74 |
| uniform n=1,000,000 | f32 | 146/146 | 5.487e+04/3.562e+04 | 0.07163/0.06464 | 0.46 | 1.46 |
| uniform n=10,000,000 | f64 | 148/148 | 2.896e+06/6.343e+05 | 0.07071/0.05656 | 0.58 | 1.80 |
| uniform n=10,000,000 | f32 | 147/147 | 2.186e+06/4.274e+05 | 0.07542/0.06696 | 0.45 | 1.58 |
| normal n=100,000 | f64 | 142/142 | 3432/3138 | 0.07285/0.0565 | 0.61 | 1.50 |
| normal n=100,000 | f32 | 142/142 | 3614/3009 | 0.07424/0.06409 | 0.48 | 1.40 |
| lognormal n=100,000 | f64 | 142/142 | 3212/2794 | 0.06543/0.05168 | 0.58 | 1.48 |
| lognormal n=100,000 | f32 | 142/142 | 3349/2755 | 0.07045/0.06438 | 0.50 | 1.43 |
| duplicates n=100,000 | f64 | 142/10 | 1248/611.7 | 0.0653/0.00546 | 9.79 | 22.88 |
| duplicates n=100,000 | f32 | 142/10 | 1227/645.7 | 0.0753/0.007124 | 6.85 | 18.10 |
| alternating_extremes n=100,000 | f64 | 142/142 | 4409/3390 | 0.07775/0.05955 | 0.62 | 1.60 |
| alternating_extremes n=100,000 | f32 | 142/2 | 854.4/449.5 | 0.06835/0.005797 | 23.12 | 77.10 |
| sorted n=100,000 | f64 | 142/142 | 796.5/1040 | 0.06572/0.05347 | 0.50 | 1.03 |
| sorted n=100,000 | f32 | 142/142 | 1121/1429 | 0.09384/0.08539 | 0.33 | 0.90 |
| reverse n=100,000 | f64 | 142/142 | 846.4/1081 | 0.06677/0.05215 | 0.52 | 0.97 |
| reverse n=100,000 | f32 | 142/142 | 630.7/1071 | 0.06916/0.06056 | 0.37 | 0.86 |

## Accuracy

The 1001-point quantile grid reports distance outside the empirical rank interval `[fraction below estimate, fraction at or below estimate]`; ties are allowed. This is a sampled maximum, not a worst-case guarantee. Type-7 quantiles are a disclosed value reference with different interpolation conventions on some discrete/tiny samples. Trimmed means use exact fractional-mass 10–90% sample means.

| Case | Precision | Maximum sampled rank error old/new | Absolute type-7 p50 error old/new | Absolute trimmed-mean error old/new |
|---|---|---:|---:|---:|
| uniform n=1 | f64 | 0 / 0 | 0 / 0 | 0 / 0 |
| uniform n=1 | f32 | 0 / 0 | 0 / 0 | 0 / 0 |
| uniform n=2 | f64 | 0.249 / 0 | 0 / 0 | 0 / 0 |
| uniform n=2 | f32 | 0.249 / 0 | 0 / 0 | 0 / 0 |
| uniform n=10 | f64 | 0.049 / 0 | 0 / 0 | 0 / 0 |
| uniform n=10 | f32 | 0.049 / 0 | 0 / 0 | 2.42144e-08 / 0 |
| uniform n=1,000 | f64 | 0.004 / 0.004 | 0.0034699 / 0.0034699 | 6.80656e-06 / 6.80656e-06 |
| uniform n=1,000 | f32 | 0.004 / 0.004 | 0.00346997 / 0.00346993 | 6.81437e-06 / 6.80836e-06 |
| uniform n=100,000 | f64 | 0.00066 / 0.00066 | 0.000466815 / 0.000466815 | 2.29366e-06 / 2.29366e-06 |
| uniform n=100,000 | f32 | 0.00066 / 0.00064 | 0.000467718 / 0.000472341 | 2.42794e-06 / 2.34204e-06 |
| uniform n=1,000,000 | f64 | 0.000229 / 0.000229 | 5.9855e-05 / 5.9855e-05 | 2.37475e-05 / 2.37475e-05 |
| uniform n=1,000,000 | f32 | 0.000229 / 0.000219 | 6.0156e-05 / 6.29182e-05 | 2.06449e-05 / 3.6608e-05 |
| uniform n=10,000,000 | f64 | 6.54e-05 / 6.54e-05 | 1.63354e-07 / 1.63354e-07 | 8.16961e-06 / 8.16961e-06 |
| uniform n=10,000,000 | f32 | 0.0020821 / 8.06e-05 | 0.000427961 / 2.74103e-06 | 3.07792e-05 / 1.21066e-05 |
| normal n=100,000 | f64 | 0.00095 / 0.00095 | 0.000146159 / 0.000146159 | 1.77215e-05 / 1.77215e-05 |
| normal n=100,000 | f32 | 0.00095 / 0.00095 | 0.000145839 / 0.000144552 | 1.80659e-05 / 1.79002e-05 |
| lognormal n=100,000 | f64 | 0.00249 / 0.00249 | 0.00903245 / 0.00903245 | 0.0230797 / 0.0230797 |
| lognormal n=100,000 | f32 | 0.00249 / 0.00249 | 0.00910333 / 0.00927353 | 0.0230371 / 0.0228555 |
| duplicates n=100,000 | f64 | 0.05466 / 0 | 0.427259 / 0 | 0.000233115 / 0 |
| duplicates n=100,000 | f32 | 0.05466 / 0 | 0.426861 / 0 | 0.000232764 / 0 |
| alternating_extremes n=100,000 | f64 | 0.045 / 0.045 | 1.03922e+08 / 1.0815e+08 | 4.11329e-06 / 4.27961e-06 |
| alternating_extremes n=100,000 | f32 | 0.499 / 0 | 1.04832e+08 / 0 | 312.934 / 0 |
| sorted n=100,000 | f64 | 0 / 0 | 9.4369e-16 / 5.55112e-17 | 4.26156e-06 / 4.26156e-06 |
| sorted n=100,000 | f32 | 0 / 0 | 2.68221e-07 / 4.70063e-09 | 4.46356e-06 / 4.26482e-06 |
| reverse n=100,000 | f64 | 0 / 0 | 7.77156e-16 / 1.11022e-16 | 4.26156e-06 / 4.26156e-06 |
| reverse n=100,000 | f32 | 0 / 0 | 2.38419e-07 / 4.70063e-09 | 4.29833e-06 / 4.26801e-06 |
| weighted_two_piles n=100 | f64 | 0.449 / 0 | 1 / 0 | 0 / 0 |
| weighted_ramp n=99,837 | f64 | 0.00138703 / 0.00138703 | 0.509112 / 0.509112 | 0.000574557 / 0.000574557 |
| weighted_two_piles n=100 | f32 | 0.449 / 0 | 1 / 0 | 0 / 0 |
| weighted_ramp n=99,837 | f32 | 0.00138703 / 0.00138703 | 0.509094 / 0.509112 | 0.000621435 / 0.00057556 |

## Compaction and memory

The independent original core over-reserves result vectors. Running the same arithmetic followed by `shrink_to_fit()` preserves the means and weights exactly in every measured compaction case. It reduces reserved vector capacity, not necessarily resident RSS by the same amount. Modern logical digest state remains larger than compact original state for many tiny/continuous cases.

| Uniform n | Precision | Original reserved bytes → compacted bytes | Original/compacted build ratio | Compact original / modern logical object+data bytes |
|---|---|---:|---:|---:|
| 1 | f64 | 12 | — | 60 / 128 |
| 1 | f32 | 8 | — | 56 / 108 |
| 2 | f64 | 24 | — | 72 / 152 |
| 2 | f32 | 16 | — | 64 / 120 |
| 10 | f64 | 120 | — | 168 / 344 |
| 10 | f32 | 80 | — | 128 / 216 |
| 1,000 | f64 | 12,000 → 1,536 | 0.986 | 1,584 / 3,176 |
| 1,000 | f32 | 8,000 → 1,024 | 1.042 | 1,072 / 1,632 |
| 100,000 | f64 | 1,200,000 → 1,704 | 1.080 | 1,752 / 3,512 |
| 100,000 | f32 | 800,000 → 1,136 | 1.061 | 1,184 / 1,800 |
| 1,000,000 | f64 | 12,000,000 → 1,752 | 1.005 | 1,800 / 3,608 |
| 1,000,000 | f32 | 8,000,000 → 1,168 | 0.972 | 1,216 / 1,848 |
| 10,000,000 | f64 | 120,000,000 → 1,776 | 1.017 | 1,824 / 3,656 |
| 10,000,000 | f32 | 80,000,000 → 1,176 | 0.969 | 1,224 / 1,860 |

## Comparison with the earlier run

14 of 30 cases have exactly identical non-timing JSON fields to the earlier intermediate run (accuracy, centroid counts, query estimates and memory capacities). This does not replace regression tests for edge cases absent from the matrix. Changes in observed speed across runs are not attributed to a code optimization: the runs were sequential, with possible CPU/thermal/background variation, and have no confidence intervals.

The 16 differing cases change only modern numerical results or derived error summaries; every recorded original-core output, centroid count and memory-capacity field is unchanged. Overflow-safe interpolation changes floating-point evaluation order. The largest difference among the stored actual quantile estimates is **1.19209289551e-07** (about one f64 ULP near ±1e9); the largest stored median difference is **2.98023223877e-08**. Derived error-summary differences reach 2.38418579102e-7. These are observed differences in stored probes, not a universal error bound. [The complete field comparison](results/2026-10-14-756d221-comparison.json) identifies each before/after value.

| Selected workload | Prior old/new µs | Committed old/new µs |
|---|---:|---:|
| uniform n=1,000 f64 | 18.23/21.16 | 15.31/18.56 |
| uniform n=10,000,000 f64 | 3.215e+06/8.085e+05 | 2.896e+06/6.343e+05 |
| uniform n=10,000,000 f32 | 2.473e+06/5.294e+05 | 2.186e+06/4.274e+05 |
| sorted n=100,000 f32 | 787.3/1121 | 1121/1429 |

## Limits and next work

- **Modern f32 mass accounting remains unresolved outside this matrix.** The largest unit-weight input is 10M, below 2²⁴. These results do not establish exact mass conservation at larger totals or with arbitrary fractional/scaled f32 weights. This limitation belongs to the modern engine; it must not be hidden by favorable accuracy or speed figures.
- One machine, one seed per distribution, no representative external competitors, and no confidence intervals. Production decisions need repeated runs, multiple seeds, matched memory/accuracy budgets and real workload traces.
- Modern is not uniformly faster: tiny/sorted construction and repeated pairwise merging can lose. K-way merging has different recompression semantics and must retain its separate accuracy evidence.
- Large gaps can produce large type-7 value errors even where many medians are valid; inspect empirical rank and value metrics together. Both cores still show difficult gapped-distribution cases.
- Scalar CDF, q99, trimmed-mean timing, merge accuracy and complete batch samples are retained in raw JSONL. The candidate scalar-CDF allocation optimization has not been applied or measured.
- Future measurements should target a new committed source revision and preserve this run rather than overwrite it.
