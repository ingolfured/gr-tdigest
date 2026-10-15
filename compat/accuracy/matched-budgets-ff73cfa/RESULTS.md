# Accuracy at frozen logical-storage and rank-error policies

Modern source `ff73cfab31d6368d203c1a361b8efaaea29d735b`; independent original `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. These measurements compare the opt-in modern core configured with K2Norm and singleton policy Off against the default original algorithm. Compatibility API parity is assessed separately.

Training selected deltas globally before the three fresh held-out seeds were evaluated. All 576 held-out dataset-policy invocations and 8064 states completed, with no modern invariant/wire failures. The offline raw-data reducer verified hashes, paired query/reference identities, error arithmetic and policy outcomes in one pass; it did not independently regenerate every large-n reference sample. It evaluated 8088192 quantile estimates; modern CDF probe count is recorded below. No operation timing is included.

All 6048 storage-policy states met their selected cap, and all 2016 states under the selected 2.0 percentage-point rank policy met that threshold. The 0.5 and 1.0-point policies remain unavailable within the frozen delta 25–1200 grid for both cores and both precisions. Passing a storage cap does not imply small rank error: the worst modern storage-policy queried error was **43.5830 percentage points** (1024 B, f64, separated_modes, n=100000, skewed/reverse/sequential).

## All frozen policies

All errors in this table are worst observed percentage points across the queried grid and the core's supported routes. Modern coverage includes additional k-way routes. Rank-policy thresholds and storage caps are separate; logical bytes are object plus live centroid payload, not capacity, peak memory or RSS.

| Policy | Core | Means | Delta | States | Misses | Max bytes | Max rank (pp) | Max p50 rank (pp) | Modern CDF (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 B | original | f64 | 50 | 360 | 0 | 780 | 29.8640 | 9.3750 | - |
| 2048 B | original | f64 | 100 | 360 | 0 | 1452 | 14.5990 | 1.7000 | - |
| 4096 B | original | f64 | 300 | 360 | 0 | 3804 | 5.4540 | 1.7000 | - |
| 1024 B | original | f32 | 100 | 360 | 0 | 984 | 14.5990 | 1.7000 | - |
| 2048 B | original | f32 | 200 | 360 | 0 | 1776 | 7.1890 | 1.7000 | - |
| 4096 B | original | f32 | 300 | 360 | 0 | 2552 | 5.4540 | 1.7000 | - |
| 1024 B | modern | f64 | 25 | 648 | 0 | 896 | 43.5830 | 19.5310 | 43.5446 |
| 2048 B | modern | f64 | 50 | 648 | 0 | 1520 | 30.0370 | 6.0960 | 30.0429 |
| 4096 B | modern | f64 | 150 | 648 | 0 | 4016 | 10.3930 | 0.8000 | 10.3882 |
| 1024 B | modern | f32 | 25 | 648 | 0 | 624 | 41.2540 | 19.5290 | 41.0626 |
| 2048 B | modern | f32 | 100 | 648 | 0 | 1920 | 16.1590 | 1.4180 | 16.1682 |
| 4096 B | modern | f32 | 200 | 648 | 0 | 3504 | 7.0870 | 0.8000 | 7.0421 |
| 0.5 pp | original | f64 | unavailable | 0 | - | - | - | - | - |
| 1 pp | original | f64 | unavailable | 0 | - | - | - | - | - |
| 2 pp | original | f64 | 1200 | 360 | 0 | 13248 | 1.5500 | 0.2410 | - |
| 0.5 pp | original | f32 | unavailable | 0 | - | - | - | - | - |
| 1 pp | original | f32 | unavailable | 0 | - | - | - | - | - |
| 2 pp | original | f32 | 1200 | 360 | 0 | 8848 | 1.5500 | 0.2410 | - |
| 0.5 pp | modern | f64 | unavailable | 0 | - | - | - | - | - |
| 1 pp | modern | f64 | unavailable | 0 | - | - | - | - | - |
| 2 pp | modern | f64 | 1200 | 648 | 0 | 26504 | 1.1600 | 0.2000 | 1.1928 |
| 0.5 pp | modern | f32 | unavailable | 0 | - | - | - | - | - |
| 1 pp | modern | f32 | unavailable | 0 | - | - | - | - | - |
| 2 pp | modern | f32 | 1200 | 648 | 0 | 17760 | 1.0600 | 0.2000 | 1.0416 |

## What the matched policies show

The modern storage choices use smaller deltas at every common cap. At the 2.0-point rank policy all four choices use delta 1200, but modern f64/f32 reach 26504/17760 logical bytes versus original 13248/8848. Both meet the same illustrative threshold on these probes; they do not have equal actual error or equal memory use. None of this establishes an optimal parameter choice or a universal bound.

At that rank policy, worst modern p50 rank error is 0.2000 pp and worst modern CDF error is 1.1928 pp. CDF has its own midpoint-ECDF metric and no original API equivalent, so the 2.0-point quantile policy does not certify a CDF limit.

A small rank error can correspond to a large value error across the empty gap between separated modes. Conversely, a quantile at an exact tied value can have zero tie-aware rank error while differing from an interpolating Type 7 reference. Both metrics are retained for this reason. These modern-core results do not justify a universal tiny-epsilon change relative to the original; existing compatibility behavior is a separate contract.

## Distribution detail at the separate 2.0-point rank policy

The following rows combine the available routes/sizes within each distribution and precision, over the same three fresh seeds. Value error is in that distribution's input units and is not pooled across distributions. Detailed route/sizing and six fixed tail queries are in the CSV files.

| Distribution | Core | Means | Max rank (pp) | Max p50 rank (pp) | Max p50 value error | Max value error, all q | Modern CDF (pp) |
|---|---|---|---:|---:|---:|---:|---:|
| exponential | original | f64 | 1.5500 | 0.2000 | 0.00300811 | 1.21024 | - |
| exponential | original | f32 | 1.5500 | 0.2000 | 0.0030081 | 1.21018 | - |
| exponential | modern | f64 | 0.4000 | 0.2000 | 0.00300811 | 1.21024 | 0.4000 |
| exponential | modern | f32 | 0.4000 | 0.2000 | 0.00364086 | 2.42528 | 0.4000 |
| lognormal | original | f64 | 1.5500 | 0.1000 | 0.00481174 | 50791.7 | - |
| lognormal | original | f32 | 1.5500 | 0.1000 | 0.00481173 | 50790.4 | - |
| lognormal | modern | f64 | 0.2000 | 0.1000 | 0.00680802 | 50791.7 | 0.2630 |
| lognormal | modern | f32 | 0.2000 | 0.1000 | 0.00680801 | 101684 | 0.2630 |
| normal | original | f64 | 1.5500 | 0.1000 | 0.0016762 | 0.558878 | - |
| normal | original | f32 | 1.5500 | 0.1000 | 0.0016762 | 0.558878 | - |
| normal | modern | f64 | 0.2000 | 0.1000 | 0.0016762 | 1.10967 | 0.2646 |
| normal | modern | f32 | 0.2000 | 0.1000 | 0.0016762 | 1.10967 | 0.2646 |
| pareto | original | f64 | 1.5500 | 0.2000 | 0.00316122 | 553.965 | - |
| pareto | original | f32 | 1.5500 | 0.2000 | 0.00316137 | 553.951 | - |
| pareto | modern | f64 | 0.4000 | 0.2000 | 0.00316122 | 553.965 | 0.4000 |
| pareto | modern | f32 | 0.4000 | 0.2000 | 0.00316131 | 1109.03 | 0.4000 |
| separated_modes | original | f64 | 1.5500 | 0.2000 | 16.9116 | 16.9116 | - |
| separated_modes | original | f32 | 1.5500 | 0.2000 | 16.8921 | 16.8921 | - |
| separated_modes | modern | f64 | 1.1600 | 0.2000 | 16.9116 | 16.9116 | 1.1928 |
| separated_modes | modern | f32 | 1.0600 | 0.2000 | 11.0112 | 12.994 | 1.0416 |
| spike_tail | original | f64 | 1.5500 | 0.0000 | 0 | 2386.06 | - |
| spike_tail | original | f32 | 1.5500 | 0.0000 | 0 | 2385.94 | - |
| spike_tail | modern | f64 | 0.1050 | 0.0000 | 0 | 2386.06 | 0.0433 |
| spike_tail | modern | f32 | 0.1050 | 0.0000 | 0 | 4781.58 | 0.0433 |
| ties | original | f64 | 1.5500 | 0.2410 | 0.472966 | 0.771852 | - |
| ties | original | f32 | 1.5500 | 0.2410 | 0.472966 | 0.771852 | - |
| ties | modern | f64 | 0.0000 | 0.0000 | 0 | 1.594 | 0.0000 |
| ties | modern | f32 | 0.0000 | 0.0000 | 0 | 1.594 | 0.0000 |
| uniform | original | f64 | 1.5500 | 0.2000 | 0.00151986 | 0.0557623 | - |
| uniform | original | f32 | 1.5500 | 0.2000 | 0.00151986 | 0.0557623 | - |
| uniform | modern | f64 | 0.4000 | 0.2000 | 0.00151986 | 0.135496 | 0.4000 |
| uniform | modern | f32 | 0.4000 | 0.2000 | 0.00164201 | 0.135496 | 0.4000 |

## Routes, references and limits

There are 2880 same-route original/modern comparisons and 2304 separately labelled modern-k-way/original-sequential comparisons. The CSV comparison counts are descriptive correlated state observations, not independent samples or significance tests. Direct, balanced-IID/range and skewed 40-way merge layouts remain available separately; reverse shard order is present only at n=100000.

Rank error measures distance from effective q to the empirical tie interval [F(x-), F(x)]. Value error uses Type 7 quantiles of post-cast samples. Modern CDF error uses the midpoint of the empirical jump. Both cores receive the same mean precision and representable q. Requested/effective tail queries and p50 estimates are retained in compact full results and selected notable states. No blanket epsilon or world-best claim follows from these inputs.

The independent original records 1152 out-of-bounds/nonfinite-quantile observation flags and 1152 decreasing-query flags. These are retained state occurrences, including repeated baseline evaluations across policies/routes, not 1152 independent bad datasets; the combined flag does not assert nonfinite values specifically. Original accuracy remains visible despite inherited semantics.

Fresh seeds are 15485863, 32452843 and 49979687; training used 1009, 10007 and 65537. Sizes are 32, 1000 and 100000, with eight distributions and unit weights. There is no claim about all production distributions, fractional weights, other scale/policy settings, equal RSS, or speed. The separate timing experiment uses n=1M and reports its own cap/error violations.

## Evidence and reproduction

[Policy outcomes](results/policy-outcomes.csv) includes every choice; [distribution/route accuracy](results/distribution-route-accuracy.csv) and [tail accuracy](results/tail-accuracy.csv) preserve per-distribution metrics; [route/size/seed groups](results/per-route-size-seed-group.csv) resolves each route, size and three-seed group; [paired route comparisons](results/paired-route-comparisons.csv) separates algorithms. [Notable states](results/notable-states.json) identifies selected extrema with raw filenames/hashes/record numbers. Compiled source identities and raw hashes remain in [reducer status](results/reducer-status.json) and the [raw manifest](results/raw-file-sha256.json).

[The bundle guide](README.md) labels historical preparation documents and describes the current evidence. `protocol/` contains the frozen runner/selector/reference sources and selection lock. Reproduction requires the recorded source commits and build, smoke, training, selection and held-out phases in order, then reporting. Bulk compressed raw data, complete compact state records and binaries remain outside Git at the paths in [full result locations](FULL_RESULTS_LOCATION.json); hashes verify identity, not availability. No bulk raw samples are included in this bundle.

Reduction status: complete; 1067044368 expanded bytes; 8088192 quantile estimates and 4134380 modern CDF probes. Reduction wall time 35.789 seconds is offline raw-data processing, not a library performance measurement.
