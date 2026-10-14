# Release comparison after weight and single-centroid mean fixes

The unchanged 30-case matrix completed against clean commit `22867bd55739a96a6fab3c8c48b27406ecb32197`, with no source changes during compilation or timing. [Metadata](results/2026-10-14-22867bd-metadata.json) records exact core, harness and binary hashes; [raw results](results/2026-10-14-22867bd.jsonl) preserve every batch and numerical field. The complete [comparison against 756d221](results/2026-10-14-22867bd-comparison.json) includes all non-timing differences.

The original `tdigest_rs` compatibility path retains original arithmetic. These measurements compare an independent pinned original Rust core against the opt-in modern engine, with original result compaction reported separately. Python, Java and Polars overhead is excluded. They establish workload-specific tradeoffs, not a global speed or accuracy lead.

## Method and source boundary

Linux/WSL, Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0 release, delta=300, K2Norm and singleton policy Off for modern, f32/f64, seed 20261014. Run: 2026-10-14T22:09:40.413206+00:00 to 2026-10-14T22:12:54.076648+00:00. The independent original source is `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Warmup/calibration precedes seven timed batches; inputs are generated outside timing, while required input copying and result destruction are included.

This revision includes f64 centroid mass for f32 means, fractional atomic-boundary interpolation and exact preservation of a single input centroid mean/kind. It does not include the scratch scalar-CDF allocation optimization. The unchanged harness is deliberate: new mass-boundary regression tests belong to the correctness evidence, because this matrix stops at ten million unit samples, below 2²⁴.

## Construction, p50, CDF and merging

Times are microseconds; ratios are original/modern (above 1 favors modern). Modern k-way merge is a distinct all-at-once operation, with its own recorded accuracy, not an interchangeable numerical implementation of a sequential fold.

| Case | Precision | Centroids old/new | Build old/new µs | p50 old/new µs | Modern CDF µs | Sequential merge ratio | K-way ratio versus old sequential |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform n=1 | f64 | 1/1 | 0.08329/0.0912 | 0.002077/0.001286 | 0.02201 | — | — |
| uniform n=1 | f32 | 1/1 | 0.07832/0.09062 | 0.001883/0.001372 | 0.02136 | — | — |
| uniform n=2 | f64 | 2/2 | 0.0938/0.1028 | 0.002833/0.004216 | 0.02328 | — | — |
| uniform n=2 | f32 | 2/2 | 0.09208/0.1033 | 0.002824/0.004241 | 0.02257 | — | — |
| uniform n=10 | f64 | 10/10 | 0.2115/0.2655 | 0.005414/0.005155 | 0.02988 | — | — |
| uniform n=10 | f32 | 10/10 | 0.1868/0.2671 | 0.005593/0.005437 | 0.03017 | — | — |
| uniform n=1,000 | f64 | 128/128 | 15.71/17.86 | 0.05583/0.04726 | 0.2847 | 0.53 | 3.18 |
| uniform n=1,000 | f32 | 128/128 | 15.21/18 | 0.05732/0.04547 | 0.2961 | 0.46 | 2.90 |
| uniform n=100,000 | f64 | 142/142 | 3162/2749 | 0.06344/0.0544 | 0.3283 | 0.58 | 1.56 |
| uniform n=100,000 | f32 | 142/142 | 2971/2559 | 0.06732/0.05075 | 0.3327 | 0.56 | 1.48 |
| uniform n=1,000,000 | f64 | 146/146 | 5.93e+04/4.994e+04 | 0.06321/0.05088 | 0.3367 | 0.58 | 1.64 |
| uniform n=1,000,000 | f32 | 146/146 | 5.003e+04/3.309e+04 | 0.06559/0.05134 | 0.384 | 0.55 | 1.64 |
| uniform n=10,000,000 | f64 | 148/148 | 2.84e+06/6.871e+05 | 0.06587/0.05243 | 0.3374 | 0.58 | 1.84 |
| uniform n=10,000,000 | f32 | 147/147 | 2.186e+06/4.363e+05 | 0.06614/0.05064 | 0.413 | 0.54 | 1.75 |
| normal n=100,000 | f64 | 142/142 | 3096/2721 | 0.06291/0.0541 | 0.4584 | 0.62 | 1.65 |
| normal n=100,000 | f32 | 142/142 | 3543/2858 | 0.07175/0.05383 | 0.3694 | 0.56 | 1.46 |
| lognormal n=100,000 | f64 | 142/142 | 3134/2660 | 0.0632/0.05072 | 0.3361 | 0.58 | 1.51 |
| lognormal n=100,000 | f32 | 142/142 | 2863/2487 | 0.0647/0.04956 | 0.3217 | 0.55 | 1.43 |
| duplicates n=100,000 | f64 | 142/10 | 1124/578.4 | 0.06492/0.005227 | 0.03139 | 10.15 | 24.18 |
| duplicates n=100,000 | f32 | 142/10 | 962.3/537.3 | 0.06568/0.005597 | 0.03068 | 8.89 | 20.87 |
| alternating_extremes n=100,000 | f64 | 142/142 | 3071/2670 | 0.06612/0.0518 | 0.3535 | 0.58 | 1.53 |
| alternating_extremes n=100,000 | f32 | 142/2 | 877.1/474.9 | 0.06627/0.004405 | 0.02377 | 25.14 | 75.65 |
| sorted n=100,000 | f64 | 142/142 | 850.3/1109 | 0.07166/0.05677 | 0.4028 | 0.44 | 0.92 |
| sorted n=100,000 | f32 | 142/142 | 929.7/1257 | 0.08313/0.0631 | 0.417 | 0.41 | 0.88 |
| reverse n=100,000 | f64 | 142/142 | 815.5/1067 | 0.06329/0.05039 | 0.336 | 0.51 | 0.98 |
| reverse n=100,000 | f32 | 142/142 | 642.1/972.6 | 0.06547/0.04916 | 0.3224 | 0.45 | 0.86 |

## Sampled accuracy

Rank error is distance outside the empirical interval `[fraction below estimate, fraction at or below estimate]`, over 1001 probabilities. Type-7 linear quantiles are the disclosed value reference; tiny/discrete interpolation conventions can differ. Trimmed means use exact fractional-mass 10–90% sample means. CDF error uses midpoint empirical ranks at 257 probes. These are sampled errors, not worst-case guarantees.

| Case | Precision | Maximum rank error old/new | Absolute p50 value error old/new | Trimmed-mean error old/new | Modern CDF maximum error |
|---|---|---:|---:|---:|---:|
| uniform n=1 | f64 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| uniform n=1 | f32 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| uniform n=2 | f64 | 0.249 / 0 | 0 / 0 | 0 / 0 | 0 |
| uniform n=2 | f32 | 0.249 / 0 | 0 / 0 | 0 / 0 | 0 |
| uniform n=10 | f64 | 0.049 / 0 | 0 / 0 | 0 / 0 | 0 |
| uniform n=10 | f32 | 0.049 / 0 | 0 / 0 | 2.42144e-08 / 0 | 0 |
| uniform n=1,000 | f64 | 0.004 / 0.004 | 0.0034699 / 0.0034699 | 6.80656e-06 / 6.80656e-06 | 0.00388091 |
| uniform n=1,000 | f32 | 0.004 / 0.004 | 0.00346997 / 0.00346993 | 6.81437e-06 / 6.80836e-06 | 0.00388089 |
| uniform n=100,000 | f64 | 0.00066 / 0.00066 | 0.000466815 / 0.000466815 | 2.29366e-06 / 2.29366e-06 | 0.000623922 |
| uniform n=100,000 | f32 | 0.00066 / 0.00064 | 0.000467718 / 0.000472341 | 2.42794e-06 / 2.34204e-06 | 0.000620033 |
| uniform n=1,000,000 | f64 | 0.000229 / 0.000229 | 5.9855e-05 / 5.9855e-05 | 2.37475e-05 / 2.37475e-05 | 0.00015392 |
| uniform n=1,000,000 | f32 | 0.000229 / 0.000219 | 6.0156e-05 / 6.29182e-05 | 2.06449e-05 / 3.6608e-05 | 0.000152811 |
| uniform n=10,000,000 | f64 | 6.54e-05 / 6.54e-05 | 1.63354e-07 / 1.63354e-07 | 8.16961e-06 / 8.16961e-06 | 6.56717e-05 |
| uniform n=10,000,000 | f32 | 0.0020821 / 8.06e-05 | 0.000427961 / 2.74103e-06 | 3.07792e-05 / 1.21066e-05 | 7.34485e-05 |
| normal n=100,000 | f64 | 0.00095 / 0.00095 | 0.000146159 / 0.000146159 | 1.77215e-05 / 1.77215e-05 | 0.000880236 |
| normal n=100,000 | f32 | 0.00095 / 0.00095 | 0.000145839 / 0.000144552 | 1.80659e-05 / 1.79002e-05 | 0.000879868 |
| lognormal n=100,000 | f64 | 0.00249 / 0.00249 | 0.00903245 / 0.00903245 | 0.0230797 / 0.0230797 | 0.0023329 |
| lognormal n=100,000 | f32 | 0.00249 / 0.00249 | 0.00910333 / 0.00927353 | 0.0230371 / 0.0228555 | 0.00233841 |
| duplicates n=100,000 | f64 | 0.05466 / 0 | 0.427259 / 0 | 0.000233115 / 0 | 0 |
| duplicates n=100,000 | f32 | 0.05466 / 0 | 0.426861 / 0 | 0.000232764 / 0 | 0 |
| alternating_extremes n=100,000 | f64 | 0.045 / 0.045 | 1.03922e+08 / 1.0815e+08 | 4.11329e-06 / 4.27961e-06 | 0.04152 |
| alternating_extremes n=100,000 | f32 | 0.499 / 0 | 1.04832e+08 / 0 | 312.934 / 0 | 0 |
| sorted n=100,000 | f64 | 0 / 0 | 9.4369e-16 / 5.55112e-17 | 4.26156e-06 / 4.26156e-06 | 4.96094e-06 |
| sorted n=100,000 | f32 | 0 / 0 | 2.68221e-07 / 4.70063e-09 | 4.46356e-06 / 4.26482e-06 | 4.96013e-06 |
| reverse n=100,000 | f64 | 0 / 0 | 7.77156e-16 / 1.11022e-16 | 4.26156e-06 / 4.26156e-06 | 4.96094e-06 |
| reverse n=100,000 | f32 | 0 / 0 | 2.38419e-07 / 4.70063e-09 | 4.29833e-06 / 4.26801e-06 | 4.96025e-06 |
| weighted_two_piles n=100 | f64 | 0.449 / 0 | 1 / 0 | 0 / 0 | — |
| weighted_ramp n=99,837 | f64 | 0.00138703 / 0.00138703 | 0.509112 / 0.509112 | 0.000574557 / 0.000574557 | — |
| weighted_two_piles n=100 | f32 | 0.449 / 0 | 1 / 0 | 0 / 0 | — |
| weighted_ramp n=99,837 | f32 | 0.00138703 / 0.00138703 | 0.509094 / 0.509112 | 0.000621435 / 0.00057556 | — |

## Memory cost and compaction

Measured layouts from the logical object-plus-centroid totals confirm F32 centroids grew from 12 to 16 bytes, while F64 centroids remain 24 bytes. Digest objects remain 96/104 bytes for F32/F64. At unchanged centroid count the F32 cost is exactly 4 bytes per centroid. These lower bounds exclude allocator rounding and language objects; capacity reduction is not an RSS measurement.

| Uniform n | Precision | Centroids before/current | Modern logical bytes before/current | Compact original logical bytes | Original reserved vectors → compacted vectors | Wire bytes before/current |
|---|---|---:|---:|---:|---:|---:|
| 1 | f64 | 1/1 | 128/128 | 60 | 12 | 89/89 |
| 1 | f32 | 1/1 | 108/112 | 56 | 8 | 85/85 |
| 2 | f64 | 2/2 | 152/152 | 72 | 24 | 106/106 |
| 2 | f32 | 2/2 | 120/128 | 64 | 16 | 98/98 |
| 10 | f64 | 10/10 | 344/344 | 168 | 120 | 242/242 |
| 10 | f32 | 10/10 | 216/256 | 128 | 80 | 202/202 |
| 1,000 | f64 | 128/128 | 3,176/3,176 | 1,584 | 12,000 → 1,536 | 2,248/2,248 |
| 1,000 | f32 | 128/128 | 1,632/2,144 | 1,072 | 8,000 → 1,024 | 1,736/1,736 |
| 100,000 | f64 | 142/142 | 3,512/3,512 | 1,752 | 1,200,000 → 1,704 | 2,486/2,486 |
| 100,000 | f32 | 142/142 | 1,800/2,368 | 1,184 | 800,000 → 1,136 | 1,918/1,918 |
| 1,000,000 | f64 | 146/146 | 3,608/3,608 | 1,800 | 12,000,000 → 1,752 | 2,554/2,554 |
| 1,000,000 | f32 | 146/146 | 1,848/2,432 | 1,216 | 8,000,000 → 1,168 | 1,970/1,970 |
| 10,000,000 | f64 | 148/148 | 3,656/3,656 | 1,824 | 120,000,000 → 1,776 | 2,588/2,588 |
| 10,000,000 | f32 | 147/147 | 1,860/2,448 | 1,224 | 80,000,000 → 1,176 | 1,983/1,983 |

For example, at 142 centroids, modern F32 logical state rises from 1,800 to 2,368 bytes (+568 bytes, +31.6%); one million such digests adds 568 MB of logical payload, before allocator/language overhead. This is the explicit cost of preserving mass independently of mean precision. TDIGv3 already stores f64 weights, so a centroid layout increase need not increase wire length. See actual length changes below; the new wire capability flag is a separate compatibility contract.

Original compaction preserves centroid contents exactly. Compact original memory is the appropriate comparison for retained state; un-compacted capacity describes the original allocation behavior, not a fair matched-memory allowance.

## Numerical changes since 756d221

13 of 30 cases changed only the F32 logical-memory field. Every recorded numerical field (quantile/median estimates, CDF and trimmed-mean errors), centroid count and wire length is identical to 756d221 in this matrix. All stored original-core numerical/control fields are unchanged. There are 0 centroid-count changes and 0 wire-length changes. These counts do not describe byte-for-byte serialized identity.

No stored f32 quantile or median estimates changed.

No stored f64 quantile or median estimates changed.

These are observed probe changes, not universal bounds. The candidate combines weight representation, atomic-boundary and single-centroid mean fixes; the matrix alone does not isolate their individual effects. Differences versus the original core remain separate from error against exact samples.

## Timing comparison with the previous run

The earlier run was not paired or interleaved with this one. Ratios below describe observed medians and may include background, thermal or frequency variation. Original timing is retained as a control; a small cross-run difference is not evidence of a code speedup or regression.

| Workload | Original build before/current µs | Modern build before/current µs | Modern p50 before/current µs | Modern CDF before/current µs |
|---|---:|---:|---:|---:|
| uniform n=1,000 f64 | 15.31/15.71 | 18.56/17.86 | 0.04803/0.04726 | 0.295/0.2847 |
| uniform n=10,000,000 f64 | 2.896e+06/2.84e+06 | 6.343e+05/6.871e+05 | 0.05656/0.05243 | 0.3467/0.3374 |
| uniform n=10,000,000 f32 | 2.186e+06/2.186e+06 | 4.274e+05/4.363e+05 | 0.06696/0.05064 | 0.3778/0.413 |
| sorted n=100,000 f32 | 1121/929.7 | 1429/1257 | 0.08539/0.0631 | 0.4201/0.417 |

## Limits and next work

- A subsequent [compressed-state audit](../compressed-state-audit/README.md) found two strict-order failures for adjacent f64 means with large finite weights. The reproduction is outside this benchmark matrix and remains an open defect; the successful timing run is not a release correctness gate.
- Modern is not uniformly faster; tiny/sorted construction and repeated pairwise merging can lose. Scalar CDF remains an allocation/prefix-scan optimization opportunity, with no measured optimization included here.
- Large empty gaps can yield large type-7 value differences even when many medians are valid; inspect rank and value errors together. Both cores still have difficult gapped-distribution cases.
- F64 weights fix the previously demonstrated F32 representational loss, but this unchanged matrix does not itself exercise totals above 2²⁴ or prove arbitrary exact floating-point mass sums. Subnormal masses and overflowing totals remain separately disclosed limits.
- Same delta is not same memory. Next measurements should use multiple seeds, 40-way production-shaped shards, compact-original and modern matched-memory budgets, and held-out error-budget comparisons.
- One CPU, one seed per distribution, no external competitors and no confidence intervals. Language-adapter, allocation-fleet and real production-trace measurements remain distinct work.
- Historical evidence is preserved in [RESULTS-756d221.md](RESULTS-756d221.md) and [RESULTS.md](RESULTS.md); do not relabel their source boundaries.
