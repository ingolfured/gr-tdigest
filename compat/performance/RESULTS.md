# Release-mode numerical and performance audit

This is a scoped Linux/WSL comparison, not evidence of a global speed or accuracy lead. The shipped `tdigest_rs` compatibility path retains the original arithmetic; the modern `gr_tdigest` path is opt-in and can produce substantially different answers. These core measurements do not include Python, Java or Polars adapter overhead.

Source: `b89527187219c83768f50416fa8ac61fe3ca2f6d` plus the recorded dirty source hashes in [run metadata](results/2026-10-14-metadata.json). Original arithmetic: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Harness and raw samples: [harness source](harness.rs), [raw JSONL](results/2026-10-14.jsonl). Compiler: `rustc 1.90.0 (1159e78c4 2025-09-14)`.

These numbers describe the historical intermediate core identified by `measured-core.patch`, before later overflow-safe interpolation and wire-validation fixes. They are not fresh measurements of a later commit.

Both implementations are compiled together at release opt-level 3, use the same process allocator, and run on one CPU of an Intel Core Ultra 9 288V under WSL. Each timing has warmups, calibration, seven batches, and raw min/median/max. One deterministic seed per distribution. Same delta=300 and explicit f32/f64, with centroid count reported because equal delta is not equal memory or identical semantics.

## End-to-end construction, p50 and 40-way merging

Times below are microseconds per operation. Speed ratio is original / modern: above 1 favors modern. Sequential merges are directly comparable left folds; modern k-way merge uses a distinct all-at-once algorithm.

| Case | Precision | Centroids old/new | Build old/new (µs) | p50 old/new (µs) | Sequential merge speed ratio | K-way vs old sequential |
|---|---|---:|---:|---:|---:|---:|
| uniform n=1 | f64 | 1/1 | 0.0888/0.0978 | 0.00215/0.00139 | — | — |
| uniform n=1 | f32 | 1/1 | 0.0865/0.107 | 0.0021/0.00132 | — | — |
| uniform n=2 | f64 | 2/2 | 0.101/0.109 | 0.00295/0.00501 | — | — |
| uniform n=2 | f32 | 2/2 | 0.102/0.119 | 0.00306/0.00586 | — | — |
| uniform n=10 | f64 | 10/10 | 0.237/0.306 | 0.00664/0.0072 | — | — |
| uniform n=10 | f32 | 10/10 | 0.228/0.443 | 0.0069/0.00861 | — | — |
| uniform n=1,000 | f64 | 128/128 | 18.2/21.2 | 0.0458/0.0534 | 0.53 | 3.12 |
| uniform n=1,000 | f32 | 128/128 | 17/22 | 0.0705/0.064 | 0.33 | 2.34 |
| uniform n=100,000 | f64 | 142/142 | 4.95e+03/4.29e+03 | 0.103/0.0766 | 0.62 | 1.52 |
| uniform n=100,000 | f32 | 142/142 | 4.51e+03/3.78e+03 | 0.0957/0.0825 | 0.48 | 1.34 |
| uniform n=1,000,000 | f64 | 146/146 | 7.54e+04/6.15e+04 | 0.0823/0.0679 | 0.56 | 1.44 |
| uniform n=1,000,000 | f32 | 146/146 | 6.86e+04/4.38e+04 | 0.0867/0.0778 | 0.46 | 1.49 |
| uniform n=10,000,000 | f64 | 148/148 | 3.21e+06/8.08e+05 | 0.0794/0.06 | 0.61 | 2.06 |
| uniform n=10,000,000 | f32 | 147/147 | 2.47e+06/5.29e+05 | 0.0903/0.076 | 0.45 | 1.52 |
| normal n=100,000 | f64 | 142/142 | 3.75e+03/3.25e+03 | 0.0738/0.0559 | 0.62 | 1.54 |
| normal n=100,000 | f32 | 142/142 | 3.41e+03/2.96e+03 | 0.0775/0.0711 | 0.47 | 1.37 |
| lognormal n=100,000 | f64 | 142/142 | 4e+03/3.3e+03 | 0.0759/0.0588 | 0.61 | 1.57 |
| lognormal n=100,000 | f32 | 142/142 | 3.61e+03/3.01e+03 | 0.0779/0.0671 | 0.48 | 1.37 |
| duplicates n=100,000 | f64 | 142/10 | 1.39e+03/685 | 0.0733/0.00552 | 10.19 | 23.47 |
| duplicates n=100,000 | f32 | 142/10 | 1.25e+03/644 | 0.0778/0.00739 | 7.56 | 20.25 |
| alternating_extremes n=100,000 | f64 | 142/142 | 4.48e+03/4.29e+03 | 0.0922/0.0698 | 0.62 | 1.56 |
| alternating_extremes n=100,000 | f32 | 142/2 | 1.08e+03/595 | 0.0808/0.00618 | 25.38 | 79.96 |
| sorted n=100,000 | f64 | 142/142 | 1.01e+03/1.19e+03 | 0.0757/0.0576 | 0.47 | 0.95 |
| sorted n=100,000 | f32 | 142/142 | 787/1.12e+03 | 0.0757/0.0659 | 0.32 | 0.82 |
| reverse n=100,000 | f64 | 142/142 | 894/1.23e+03 | 0.0737/0.0573 | 0.53 | 0.94 |
| reverse n=100,000 | f32 | 142/142 | 804/1.18e+03 | 0.0787/0.069 | 0.34 | 0.79 |

## Accuracy against actual typed samples

Rank error measures how far requested q lies outside the empirical interval `[fraction below estimate, fraction at or below estimate]`; this handles ties without marking an exact point mass as wrong. The quantile grid has 1001 probabilities. Value errors use a type-7 linear quantile reference, whose interpolation convention differs from the modern digest for some tiny or discrete data. Trimmed means use exact fractional-mass sample trimming. No error is normalized by a possibly zero/near-zero true value.

| Case | Precision | Max rank error old/new | p50 absolute type-7 error old/new | Trimmed-mean absolute error old/new |
|---|---|---:|---:|---:|
| uniform n=1 | f64 | 0/0 | 0/0 | 0/0 |
| uniform n=1 | f32 | 0/0 | 0/0 | 0/0 |
| uniform n=2 | f64 | 0.249/0 | 0/0 | 0/0 |
| uniform n=2 | f32 | 0.249/0 | 0/0 | 0/0 |
| uniform n=10 | f64 | 0.049/0 | 0/0 | 0/0 |
| uniform n=10 | f32 | 0.049/0 | 0/0 | 2.421e-08/0 |
| uniform n=1,000 | f64 | 0.004/0.004 | 0.00347/0.00347 | 6.807e-06/6.807e-06 |
| uniform n=1,000 | f32 | 0.004/0.004 | 0.00347/0.00347 | 6.814e-06/6.808e-06 |
| uniform n=100,000 | f64 | 0.00066/0.00066 | 0.0004668/0.0004668 | 2.294e-06/2.294e-06 |
| uniform n=100,000 | f32 | 0.00066/0.00064 | 0.0004677/0.0004723 | 2.428e-06/2.342e-06 |
| uniform n=1,000,000 | f64 | 0.000229/0.000229 | 5.985e-05/5.985e-05 | 2.375e-05/2.375e-05 |
| uniform n=1,000,000 | f32 | 0.000229/0.000219 | 6.016e-05/6.292e-05 | 2.064e-05/3.661e-05 |
| uniform n=10,000,000 | f64 | 6.54e-05/6.54e-05 | 1.634e-07/1.634e-07 | 8.17e-06/8.17e-06 |
| uniform n=10,000,000 | f32 | 0.002082/8.06e-05 | 0.000428/2.741e-06 | 3.078e-05/1.211e-05 |
| normal n=100,000 | f64 | 0.00095/0.00095 | 0.0001462/0.0001462 | 1.772e-05/1.772e-05 |
| normal n=100,000 | f32 | 0.00095/0.00095 | 0.0001458/0.0001446 | 1.807e-05/1.79e-05 |
| lognormal n=100,000 | f64 | 0.00249/0.00249 | 0.009032/0.009032 | 0.02308/0.02308 |
| lognormal n=100,000 | f32 | 0.00249/0.00249 | 0.009103/0.009274 | 0.02304/0.02286 |
| duplicates n=100,000 | f64 | 0.05466/0 | 0.4273/0 | 0.0002331/0 |
| duplicates n=100,000 | f32 | 0.05466/0 | 0.4269/0 | 0.0002328/0 |
| alternating_extremes n=100,000 | f64 | 0.045/0.045 | 1.039e+08/1.081e+08 | 4.113e-06/4.28e-06 |
| alternating_extremes n=100,000 | f32 | 0.499/0 | 1.048e+08/0 | 312.9/0 |
| sorted n=100,000 | f64 | 0/0 | 9.437e-16/5.551e-17 | 4.262e-06/4.262e-06 |
| sorted n=100,000 | f32 | 0/0 | 2.682e-07/4.701e-09 | 4.464e-06/4.265e-06 |
| reverse n=100,000 | f64 | 0/0 | 7.772e-16/1.11e-16 | 4.262e-06/4.262e-06 |
| reverse n=100,000 | f32 | 0/0 | 2.384e-07/4.701e-09 | 4.298e-06/4.268e-06 |
| weighted_two_piles n=100 | f64 | 0.449/0 | 1/0 | 0/0 |
| weighted_ramp n=99,837 | f64 | 0.001387/0.001387 | 0.5091/0.5091 | 0.0005746/0.0005746 |
| weighted_two_piles n=100 | f32 | 0.449/0 | 1/0 | 0/0 |
| weighted_ramp n=99,837 | f32 | 0.001387/0.001387 | 0.5091/0.5091 | 0.0006214/0.0005756 |

## Retained memory

Original core vectors retain capacity equal to constructor input length; these are capacity bytes, not RSS, and exclude allocator rounding and the stack object. The new compatibility adapter separately compacts these vectors after calling the unchanged arithmetic. Therefore original capacity numbers below must not be attributed to the compacted adapter. The benchmark itself calls the independent original core directly.

Modern lower bounds include the digest object plus logical centroid storage; true allocator overhead is excluded. In delta mode the inspected final assembly allocates capacity equal to the final centroid count. Modern tiny digests have a larger object and centroid layout than the original format, a relevant cost at millions of digests.

| Uniform sample size | Precision | Original core vector capacity bytes | Original compact logical object+data bytes | Modern logical object+data bytes | Modern wire bytes |
|---|---|---:|---:|---:|---:|
| 1 | f64 | 12 | 60 | 128 | 89 |
| 1 | f32 | 8 | 56 | 108 | 85 |
| 2 | f64 | 24 | 72 | 152 | 106 |
| 2 | f32 | 16 | 64 | 120 | 98 |
| 10 | f64 | 120 | 168 | 344 | 242 |
| 10 | f32 | 80 | 128 | 216 | 202 |
| 1,000 | f64 | 12,000 | 1,584 | 3,176 | 2,248 |
| 1,000 | f32 | 8,000 | 1,072 | 1,632 | 1,736 |
| 100,000 | f64 | 1,200,000 | 1,752 | 3,512 | 2,486 |
| 100,000 | f32 | 800,000 | 1,184 | 1,800 | 1,918 |
| 1,000,000 | f64 | 12,000,000 | 1,800 | 3,608 | 2,554 |
| 1,000,000 | f32 | 8,000,000 | 1,216 | 1,848 | 1,970 |
| 10,000,000 | f64 | 120,000,000 | 1,824 | 3,656 | 2,588 |
| 10,000,000 | f32 | 80,000,000 | 1,224 | 1,860 | 1,983 |

### Numerically neutral compatibility compaction

A separate constructor measurement calls the original arithmetic and then `shrink_to_fit()` on its two result vectors, matching the new adapter operation. The harness asserts exact mean and weight equality before/after. This measures Rust-core compaction cost, not Python wrapper throughput. Ratios above 1 mean the compacted constructor happened to be faster in this run; small differences are timing noise and must not be promoted as speedups.

| Uniform n | Precision | Original capacity bytes → compacted | Original / compacted build timing |
|---|---|---:|---:|
| 1,000 | f64 | 12,000 → 1,536 | 1.017 |
| 1,000 | f32 | 8,000 → 1,024 | 0.987 |
| 100,000 | f64 | 1,200,000 → 1,704 | 1.044 |
| 100,000 | f32 | 800,000 → 1,136 | 1.207 |
| 1,000,000 | f64 | 12,000,000 → 1,752 | 1.057 |
| 1,000,000 | f32 | 8,000,000 → 1,168 | 0.965 |
| 10,000,000 | f64 | 120,000,000 → 1,776 | 1.017 |
| 10,000,000 | f32 | 80,000,000 → 1,176 | 0.993 |

## Limitations and next useful work

- Timing ranges and complete quantile estimates are retained in raw JSONL. Results on this laptop do not establish production throughput or statistically significant small speed differences.
- Construction includes copying required by the actual API, sorting, compression and result destruction; data generation is excluded. Sorted weighted construction uses the original sorted-mean API and modern weighted-unsorted API, so constructor scope differs.
- Scalar CDF builds a prefix array per call in the modern implementation. Original Python/Rust interface has no CDF to compare. Batch queries or a carefully memory-budgeted optimization warrant separate measurement.
- Modern delta compression evaluates a logarithmic scale limit per candidate centroid; original arithmetic updates an inverse limit when emitting a centroid. This is a plausible construction bottleneck to profile, not a demonstrated attribution from a CPU profiler.
- Modern sequential merge and one-shot k-way merge can have different accuracy and centroid counts; speedups between them should not be described as unchanged arithmetic.
- Broader competitiveness requires multiple seeds, more machines, matched memory/accuracy budgets, production-shaped workload traces, and current representative external implementations. No best-in-world claim is justified.
- At the beginning of this audit the three checked-in Criterion benches referenced obsolete generic/API forms. They are being repaired separately; measurements here come from the standalone harness against the real crate.
