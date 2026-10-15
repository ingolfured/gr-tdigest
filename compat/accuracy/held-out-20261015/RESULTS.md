# Held-out fixed-delta accuracy survey, 15 October 2026

All declared modern invariants passed, but the survey does **not** establish uniformly better accuracy. Separated narrow modes exposed substantial value errors and CDF errors up to 5.256 percentage points. Exact discrete ties improved strongly. Smooth-distribution results were broadly similar to the original, with both improvements and regressions. This is an accuracy study; it contains no operation timings or matched-memory ranking.

## Source and controls

Modern source: `52f6212008111b62fa07374278b850868b9ea77e`. Independent original: `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Both were exported from immutable Git commits into a fresh run directory. Subsequent Rust API restoration and packaging commits are outside this source claim. The old compatibility API is not replaced by the modern implementation in this comparison: `modern` means the opt-in modern Rust core, while `original` is the independently built original crate.

Both use delta 300; modern uses K2Norm with singleton policy Off. Five predeclared held-out seeds (13, 97, 271828, 314159, 999983), eight distributions, three sizes (32, 1,000, 100,000), and F32/F64 means produced 240 datasets. Each dataset has direct construction plus 40-shard balanced-IID, sorted-range and 90%-in-one-shard layouts. Modern sequential and k-way merges are separate routes; the original has sequential merges. Large datasets also reverse shard order. Empty shards are skipped equally and retained in the metadata. Every input has unit weight; fractional-weight correctness is covered by separate audits, not this survey.

Inputs are compared after casting to the declared precision. Both implementations receive the original API's representable query probability (including F32 rounding). Quantile rank error is distance to the empirical tie interval `[F(x-), F(x)]`; value error is absolute distance from Type-7 sample quantiles. CDF truth is midpoint ECDF, including atoms. These conventions are recorded, not interchangeable with all other published quantile metrics. The original has no CDF API; no old CDF was invented.

## Execution and validation

The protocol was frozen before held-out execution. Fourteen hand-computed rational oracle assertions, seven Rust reference fixtures / 763 numeric cells, and a 16-dataset / 176-state tiny smoke passed before the survey. The first smoke attempt failed to compile because the scratch harness called an unavailable F32 decoder method; its logs remain preserved. The corrected harness dispatches the public wire decoder's precision variant without converting F32 to F64. Library source was never changed.

The survey completed **3,360 states: 2,160 modern and 1,200 original**, with no driver or modern invariant failures. It saved 3,370,080 primary quantile results and 1,725,214 modern CDF probes. Every modern query was repeated after wire round-trip with bit-equal results. Count and centroid mass, finite positive centroid weights, true extrema, finite bounded monotone queries and CDFs all passed. Independent Python rational/reference checks covered 3,873,308 tiny-state numeric cells. This validates the recorded checks, not an unstated accuracy threshold.

The source, executable, Cargo.lock, frozen protocol and every compressed case file were independently rehashed after completion. The binary SHA256 is `97bd2b927610ff7ac87ca1f293f053dd1ddd6c4d08b64d41ddcddf33ecfc3781`. The run took about 77 seconds of dataset execution on Linux/WSL; that duration is budget metadata, not a library benchmark. Full per-case gzip records occupy 154,172,740 bytes and remain retained with their hashes.

## Median and CDF accuracy

The following are worst errors over the declared seeds, both precisions and all modern routes at each size. Rank/CDF values are **percentage points**, not percentages of the estimated value. Maxima are descriptive observations; they are not universal bounds.

| Distribution | p50 rank, n=1,000 | p50 rank, n=100,000 | Any queried rank, n=100,000 | CDF, n=100,000 |
|---|---:|---:|---:|---:|
| Uniform | 0.600 | 0.099 | 0.686 | 0.649 |
| Normal | 0.300 | 0.138 | 0.655 | 0.639 |
| Exponential | 0.600 | 0.095 | 0.678 | 0.647 |
| Lognormal, sigma 3 | 0.300 | 0.209 | 0.564 | 0.517 |
| Pareto, alpha 1.5 | 0.500 | 0.079 | 0.625 | 0.572 |
| Ten-point ties | 0 | 0 | 0 | <= 1.2e-14 |
| 90% zero plus lognormal tail | 0 | 0 | 0.397 | 0.386 |
| Narrow modes near -10 and +10 | **1.700** | **0.255** | **5.296** | **5.256** |

All n=32 modern states had zero recorded rank error and CDF error, with exact Type-7 p50. That does not imply every returned quantile equals Type-7: the rank-interval metric intentionally accepts any value with the requested empirical rank.

Across the 960 modern merged states at n=100,000, median per-state p50 rank error was 0.010 percentage points, its nearest-rank p95 was 0.209, and its maximum was 0.255. Median per-state maximum CDF error was 0.117 percentage points; p95 was 3.794 and maximum 4.775. These aggregate summaries give each tested state equal weight and include deliberately adversarial distributions.

The worst direct F32 separated-mode state (seed 999983, n=100,000) returned CDF **0.5525535** at x=9.9709721, versus midpoint ECDF **0.49999**. Its p50 was **0.7547161** versus Type-7 **9.9709764**, a value error of **9.2162602**, despite rank error only **0.00109**. The large gap between modes makes rank and value errors tell different stories. Both must remain visible.

The worst p50 rank case (seed 999983, n=1,000, direct F64) returned **8.8143463** versus Type-7 **9.9821127**, rank error **0.017**. The original returned exactly the same median. At n=100,000 in F64, both direct cores returned approximately **-0.5661057** versus Type-7 **9.9709765**. The gap issue is therefore partly inherited, while F32 centroid construction also changes the results. These are much larger than a 0.000001 value change and must not be described as negligible numerical drift in the opt-in modern API.

## Comparison with the original

Restricting comparisons to the same direct/sequential route gives 1,200 pairs. Modern p50 rank error was lower in 106, equal in 1,072, and higher in 22; the largest increase was 0.048 percentage points. The maximum queried rank error was lower in 609, equal in 526, and higher in 65, with a largest increase of 1.226 percentage points. These counts are descriptive, correlated across related routes, and not statistical significance tests. K-way comparisons remain in a separately labelled file and use the original sequential route as the reference.

For the exact ten-value distribution, modern p50/rank errors were zero throughout, while the original's worst rank error at n=100,000 reached 5.648 percentage points. The original also produced **4,208 below-minimum estimates in 488 states**, and those same 488 states had decreasing quantile order. There were no nonfinite original estimates and no above-maximum estimates in this run. These inherited observations remain in the raw records and did not censor the survey. They are not newly introduced compatibility failures.

## Retained storage and review implications

This is the same delta, **not the same memory budget**. Logical storage is object size plus live centroid payload; it excludes allocator overhead, spare modern vector capacity, temporary merge state and process RSS. The original's separately recorded vector capacity was measured after compaction. Modern F32 has 16-byte centroids and a 96-byte object; F64 has 24-byte centroids and a 104-byte object. Original payloads are 8/12 bytes per centroid plus a 48-byte object.

On continuous distributions, corresponding direct/sequential states generally retain about twice the logical bytes at the same delta. Ties are different: modern keeps ten centroids, using 256 bytes for F32 or 344 for F64; original retains many repeated means. Across the large modern states, logical storage ranged from 256 to 4,944 bytes in F32 and 344 to 7,376 in F64. K-way merges often retained fewer centroids than sequential merges, so their memory/accuracy tradeoff must be considered together.

The evidence supports exact tie preservation and the tested invariants. It does not support a blanket accuracy improvement, a global speed/accuracy ranking, or a promise that the opt-in modern API reproduces original values. The focused next investigation is compression and interpolation around sharply separated modes, followed by a separately frozen memory-budget comparison if needed. No parameter tuning or library fix was performed using these held-out results.

`notable-cases.json` contains exact queries, input fingerprints and case filenames. `state-metrics.jsonl`, `paired-drift.jsonl` and `summary.json` retain per-state and per-seed summaries. The protocol, full metadata and compressed-file hash index identify the preserved raw data. The frozen scripts permit a complete rerun from the historical source objects.
