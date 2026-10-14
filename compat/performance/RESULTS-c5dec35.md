# Release comparison after finite-mean and subnormal-query fixes

This unchanged 30-case matrix targets clean candidate commit `c5dec35059990b40fe6f036dc83590bf0cdf2a70`. Core sources remained unchanged throughout compilation and timing. [Raw samples](results/2026-10-14-c5dec35.jsonl), [metadata](results/2026-10-14-c5dec35-metadata.json) and the complete [comparison with 22867bd](results/2026-10-14-c5dec35-comparison.json) retain the exact source/harness/binary identity and every recorded metric.

The candidate includes bounded finite multi-centroid averaging and temporary power-of-two query normalization for positive subnormal total mass. It excludes the proposed transactional weight-scaling and packaging-hygiene changes. Historical runs remain separately attributed; a later restacked equivalent source must not silently replace this measured commit identity.

## Setup

Linux/WSL on Intel Core Ultra 9 288V, one pinned CPU, Rust 1.90.0 release, seed 20261014, delta 300 and both F32/F64. Modern uses K2Norm with singleton policy Off. Run: 2026-10-14T22:34:42.037485+00:00 to 2026-10-14T22:38:07.039162+00:00. Independent original core: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The unchanged harness warms up and calibrates each operation before seven timed batches; generation is excluded while required input copies and result destruction are included.

These are Rust-core measurements; Python, Java and Polars overhead is excluded. The original compatibility path keeps original arithmetic. Same delta does not mean same centroid count or memory, and one-shot k-way merging is numerically distinct from sequential folding. No global fastest/best claim follows.

## Construction and primary queries

| Case | Precision | Centroids old/new | Build old/new µs | p50 old/new µs | Modern scalar CDF µs |
|---|---|---:|---:|---:|---:|
| uniform n=1 | f64 | 1/1 | 0.0829/0.09285 | 0.00205/0.001361 | 0.02146 |
| uniform n=1 | f32 | 1/1 | 0.07896/0.09151 | 0.002047/0.001465 | 0.02343 |
| uniform n=2 | f64 | 2/2 | 0.09299/0.1025 | 0.002808/0.004344 | 0.02254 |
| uniform n=2 | f32 | 2/2 | 0.09265/0.1055 | 0.002867/0.004529 | 0.02334 |
| uniform n=10 | f64 | 10/10 | 0.2117/0.2746 | 0.005494/0.005637 | 0.02974 |
| uniform n=10 | f32 | 10/10 | 0.1965/0.28 | 0.005843/0.006166 | 0.03237 |
| uniform n=1,000 | f64 | 128/128 | 15.66/18.81 | 0.05581/0.04655 | 0.2883 |
| uniform n=1,000 | f32 | 128/128 | 14.51/18.48 | 0.06077/0.05311 | 0.3079 |
| uniform n=100,000 | f64 | 142/142 | 3164/2651 | 0.06465/0.05151 | 0.338 |
| uniform n=100,000 | f32 | 142/142 | 2826/2496 | 0.06661/0.05001 | 0.3259 |
| uniform n=1,000,000 | f64 | 146/146 | 5.912e+04/5.011e+04 | 0.06508/0.05388 | 0.333 |
| uniform n=1,000,000 | f32 | 146/146 | 5.968e+04/3.757e+04 | 0.07069/0.05396 | 0.3564 |
| uniform n=10,000,000 | f64 | 148/148 | 2.878e+06/7.891e+05 | 0.08344/0.06391 | 0.382 |
| uniform n=10,000,000 | f32 | 147/147 | 2.252e+06/4.873e+05 | 0.07821/0.06172 | 0.441 |
| normal n=100,000 | f64 | 142/142 | 3222/2630 | 0.06309/0.05254 | 0.374 |
| normal n=100,000 | f32 | 142/142 | 2924/2527 | 0.06445/0.04958 | 0.3598 |
| lognormal n=100,000 | f64 | 142/142 | 3062/2719 | 0.06313/0.05081 | 0.348 |
| lognormal n=100,000 | f32 | 142/142 | 2857/2457 | 0.06474/0.04894 | 0.3679 |
| duplicates n=100,000 | f64 | 142/10 | 1224/605.3 | 0.06514/0.005235 | 0.03051 |
| duplicates n=100,000 | f32 | 142/10 | 1081/559.8 | 0.0637/0.005591 | 0.03067 |
| alternating_extremes n=100,000 | f64 | 142/142 | 3095/2736 | 0.06313/0.05117 | 0.3217 |
| alternating_extremes n=100,000 | f32 | 142/2 | 782.9/441.1 | 0.06542/0.004634 | 0.02457 |
| sorted n=100,000 | f64 | 142/142 | 697.1/1053 | 0.06479/0.05096 | 0.3316 |
| sorted n=100,000 | f32 | 142/142 | 597.6/969.5 | 0.06488/0.04941 | 0.3296 |
| reverse n=100,000 | f64 | 142/142 | 704.7/1070 | 0.06462/0.05212 | 0.5279 |
| reverse n=100,000 | f32 | 142/142 | 969.4/1284 | 0.08183/0.05965 | 0.3694 |

## Forty-way merging

All cases below split the input into 40 contiguous shards. Shard construction is untimed. The k-way result has separately recorded accuracy in raw JSON.

| Case | Precision | Original sequential µs | Modern sequential µs | Modern k-way µs | Original sequential / modern k-way |
|---|---|---:|---:|---:|---:|
| uniform n=1,000 | f64 | 122.6 | 227.7 | 36.39 | 3.37 |
| uniform n=1,000 | f32 | 106.7 | 226.2 | 35.74 | 2.99 |
| uniform n=100,000 | f64 | 325 | 555.7 | 204.7 | 1.59 |
| uniform n=100,000 | f32 | 310.4 | 558 | 206.7 | 1.5 |
| uniform n=1,000,000 | f64 | 389.2 | 670.3 | 236.7 | 1.64 |
| uniform n=1,000,000 | f32 | 340.7 | 632.2 | 236 | 1.44 |
| uniform n=10,000,000 | f64 | 428.1 | 705.3 | 230.3 | 1.86 |
| uniform n=10,000,000 | f32 | 396.4 | 713.5 | 226.2 | 1.75 |
| normal n=100,000 | f64 | 328.5 | 547.9 | 216 | 1.52 |
| normal n=100,000 | f32 | 302.2 | 537.7 | 210.9 | 1.43 |
| lognormal n=100,000 | f64 | 316.7 | 543.2 | 207.8 | 1.52 |
| lognormal n=100,000 | f32 | 309.8 | 551.9 | 221.5 | 1.4 |
| duplicates n=100,000 | f64 | 223.9 | 22.84 | 9.487 | 23.6 |
| duplicates n=100,000 | f32 | 189.9 | 21.68 | 8.93 | 21.3 |
| alternating_extremes n=100,000 | f64 | 325.2 | 559.6 | 210.6 | 1.54 |
| alternating_extremes n=100,000 | f32 | 176.8 | 7.236 | 2.256 | 78.4 |
| sorted n=100,000 | f64 | 167.5 | 370.1 | 169.4 | 0.989 |
| sorted n=100,000 | f32 | 145.8 | 369.8 | 167.6 | 0.87 |
| reverse n=100,000 | f64 | 240.3 | 468.4 | 274.7 | 0.875 |
| reverse n=100,000 | f32 | 144.1 | 330.9 | 200.3 | 0.72 |

## Sampled accuracy

Quantile rank error is distance outside the empirical interval `[fraction below estimate, fraction at or below estimate]` over 1001 probabilities. Type-7 linear quantiles are a disclosed value reference, not a unique definition for every discrete sample. Trimmed means use exact fractional-mass 10–90% sample means. CDF is compared with midpoint empirical ranks at 257 probes. These sampled maxima are not worst-case bounds.

| Case | Precision | Max rank error old/new | Absolute p50 value error old/new | Trimmed-mean error old/new | Modern CDF max error |
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

## Comparison with 22867bd

0 of 30 cases changed a non-timing field (0 changed leaves). All stored original-core numerical/control fields are unchanged. The full machine-readable comparison includes count, logical memory and wire-length fields alongside numerical estimates and error metrics.

Every recorded quantile/median estimate, CDF/trimmed-mean error, centroid count, logical memory field and wire length is exactly unchanged in this matrix. This is a useful ordinary-workload control; the newly fixed adjacent-mean and subnormal cases are covered by separate regression/audit evidence, not by these mostly ordinary-scale samples.

Cross-run timings are descriptive, not paired measurements: CPU frequency, heat and background activity can change them. The original-core times serve as a control but are not silently used to normalize candidate times.

| Workload | Original build before/current µs | Modern build before/current µs | Modern p50 before/current µs | Modern CDF before/current µs |
|---|---:|---:|---:|---:|
| uniform n=1,000 f64 | 15.71/15.66 | 17.86/18.81 | 0.04726/0.04655 | 0.2847/0.2883 |
| uniform n=10,000,000 f64 | 2.84e+06/2.878e+06 | 6.871e+05/7.891e+05 | 0.05243/0.06391 | 0.3374/0.382 |
| uniform n=10,000,000 f32 | 2.186e+06/2.252e+06 | 4.363e+05/4.873e+05 | 0.05064/0.06172 | 0.413/0.441 |
| sorted n=100,000 f32 | 929.7/597.6 | 1257/969.5 | 0.0631/0.04941 | 0.417/0.3296 |

## Memory and remaining limits

Logical layouts remain 96-byte object + 16 bytes per F32 centroid, and 104-byte object + 24 bytes per F64 centroid. This retains the previously disclosed F32 memory increase from the f64-weight fix: at 142 centroids, 1,800 → 2,368 logical bytes (+31.6%). These are lower bounds, not RSS. Compact original state is smaller for many tiny/continuous digests; original reserved vector capacity and numerically neutral compaction are reported separately.

- Modern gains are workload-dependent. Small/sorted builds and repeated pairwise merges can lose. Report accuracy, speed and memory together.
- The matrix uses one machine and seed, without external competitors, matched-memory/error budgets or confidence intervals. More production-shaped seeds and 40-way shard layouts remain useful.
- Query normalization fixes positive subnormal-total queries without rewriting stored state. It cannot recover information already lost in compression or resolve every tiny component inside a normal total.
- Aggregate/product overflow, transactional rejection of invalid weight scaling, package ownership/real upgrade behavior and the full manylinux/platform release matrix remain separate scope. The weight-scaling and packaging-hygiene proposals are not measured here.
- Preserve [22867bd results](RESULTS-22867bd.md), [756d221 results](RESULTS-756d221.md), and [the initial intermediate run](RESULTS.md) as historical evidence.

Related correctness gates and installed-wheel provenance are in the [c5dec35 validation manifest](../validation/c5dec35-validation.json).
