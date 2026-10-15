# Timing at training-selected storage/error policies

Opt-in modern K2Norm with singleton policy Off at `ff73cfab31d6368d203c1a361b8efaaea29d735b`, versus the independent default original algorithm at `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. These are personal-repository measurements using one native Rust release binary, the same jemalloc/dependencies, WSL Linux, CPU 0 and one thread. Fresh seed 86028121, n = 1,000,000, normal/ties/separated modes, F32/F64 means. No native code was executed by this analysis.

All 72 scheduled processes completed: 90 unique states repeated across two counterbalanced blocks, 468 timing rows and 36 unchanged state/leaf/query controls. **Eight unique states missed their selected policy** (16 block observations): the training-selected policy is not a guarantee at n = 1M. This extrapolates beyond the accuracy training/held-out maximum n = 100K.

For normal f64 at the 2 pp policy, both cores meet the queried rank threshold: modern build takes 0.825/0.868 times original and direct p50 takes 0.789/0.729 times original across the two blocks, while sequential merging takes 1.542/1.681 times original (about 54–68% slower). This policy does not match actual storage; the Python compatibility adapter is not timed.

## All 18 pairs

Each timing entry is **modern/original elapsed time, block 1/block 2**; below 1 means less modern time. Each block value is the median of its fixed batches, not a pooled estimate or significance test. Original then modern ran in block 1; reversed core, case and operation order in block 2. Direct and sequential query states are distinct. The original control's and modern core's block 2/block 1 drift are retained for every row in the linked CSV.

| Pair | Policy | Input | Means | Delta O/M | Build M/O | Sequential merge M/O | Direct p50 M/O | Sequential p50 M/O | Policy misses |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 1024 B | normal | f32 | 100/25 | 0.694/0.708 | 0.546/0.564 | 0.366/0.367 | 0.193/0.191 | original seq 1040 B |
| 1 | 1024 B | normal | f64 | 50/25 | 0.897/0.895 | 0.937/0.933 | 0.646/0.586 | 0.545/0.280 | none |
| 2 | 1024 B | ties | f32 | 100/25 | 0.437/0.401 | 0.268/0.288 | 0.373/0.382 | 0.089/0.101 | none |
| 3 | 1024 B | ties | f64 | 50/25 | 0.455/0.461 | 0.435/0.446 | 0.626/0.591 | 0.218/0.242 | none |
| 4 | 1024 B | separated_modes | f32 | 100/25 | 0.483/0.490 | 0.522/0.557 | 0.400/0.378 | 0.220/0.199 | original seq 1040 B |
| 5 | 1024 B | separated_modes | f64 | 50/25 | 0.890/0.893 | 1.017/0.901 | 0.701/0.650 | 0.555/0.496 | none |
| 6 | 4096 B | normal | f32 | 300/200 | 0.754/0.713 | 1.157/1.177 | 0.582/0.583 | 0.459/0.524 | none |
| 7 | 4096 B | normal | f64 | 300/150 | 0.811/0.875 | 0.824/0.802 | 0.404/0.297 | 0.446/0.403 | original seq 4296 B; modern seq 4496 B |
| 8 | 4096 B | ties | f32 | 300/200 | 0.429/0.441 | 0.087/0.089 | 0.080/0.092 | 0.036/0.041 | none |
| 9 | 4096 B | ties | f64 | 300/150 | 0.501/0.352 | 0.080/0.086 | 0.079/0.087 | 0.038/0.039 | original seq 4224 B |
| 10 | 4096 B | separated_modes | f32 | 300/200 | 0.520/0.495 | 1.182/1.191 | 0.635/0.636 | 0.535/0.519 | none |
| 11 | 4096 B | separated_modes | f64 | 300/150 | 0.824/0.934 | 0.856/0.895 | 0.284/0.353 | 0.429/0.467 | original seq 4392 B; modern seq 4496 B |
| 12 | 2 pp | normal | f32 | 1200/1200 | 0.739/0.717 | 1.762/1.683 | 0.735/0.729 | 0.773/0.751 | none |
| 13 | 2 pp | normal | f64 | 1200/1200 | 0.825/0.868 | 1.542/1.681 | 0.789/0.729 | 0.824/0.773 | none |
| 14 | 2 pp | ties | f32 | 1200/1200 | 0.457/0.427 | 0.028/0.026 | 0.027/0.027 | 0.012/0.012 | none |
| 15 | 2 pp | ties | f64 | 1200/1200 | 0.433/0.418 | 0.023/0.025 | 0.024/0.028 | 0.011/0.011 | none |
| 16 | 2 pp | separated_modes | f32 | 1200/1200 | 0.490/0.482 | 1.725/1.757 | 0.851/0.898 | 0.765/0.767 | original direct 2.097002 pp |
| 17 | 2 pp | separated_modes | f64 | 1200/1200 | 0.906/0.868 | 1.611/1.638 | 0.782/0.736 | 0.778/0.774 | none |

## Operations, controls and actual accuracy

[Paired timings](results/paired-timings.csv) contains both absolute ns/operation timings, both M/O ratios and both cores' block 2/block 1 drift for every same-route operation. [Modern k-way](results/modern-kway.csv) includes absolute times and both k-way/modern-sequential and k-way/original-sequential ratios; these compare different merge routes and retain original control drift. [Modern CDF](results/modern-cdf.csv) gives absolute time only for direct/sequential/k-way states; the original API has no CDF comparison. [Integer batch timings](results/batch-timings.jsonl) retain every measured elapsed time and iteration count.

For a concrete tradeoff, normal f64 at the 2 pp policy uses delta 1200 for both cores: modern build time is 0.825/0.868 times original, while sequential merge costs 1.542/1.681 times original. Modern k-way instead takes 0.993/0.902 ms versus original sequential 1.460/1.390 ms; that is a different route with its own recorded state accuracy. Modern scalar CDF on the direct state takes 1.220/1.219 us per call; no original CDF ratio is available.

Block drift limits the size of any inferred gain. For normal f64 at 1 KiB, sequential-state p50 has M/O ratios 0.545/0.280, but the unchanged original control slows by 1.679 times between blocks while modern time changes by 0.863 times. Both observations are retained; the larger apparent advantage in block 2 cannot be attributed to a code change between those blocks.

[State accuracy/storage](results/state-accuracy-storage.csv) includes actual bytes, centroid counts, per-route p50 and maximum queried rank/value errors, CDF errors and all policy misses. [Tail accuracy](results/tail-accuracy.csv) retains six fixed requested/effective quantile points. [Policy evaluations](results/policy-evaluations-both-blocks.csv) preserve both block observations. Matching training policies does not mean equal actual error, equal retained capacity, or equal RSS. Comparisons labelled as satisfying a bound require both states to meet it; misses remain in this primary table.

Original vector capacity is measured from actual timed states, without compaction; the compatibility Python adapter's shrink-to-fit behavior is not timed. Modern capacity is unknown. Logical object/live-centroid bytes, original retained vector capacity, and modern wire bytes are different quantities. All numerical, leaf, capacity, input and timed-query controls were identical across blocks; this does not eliminate thermal/frequency/scheduler variation.

Build includes original internal copy/sort and modern slice-to-owned-vector copy. Merge includes reset/ownership clones and excludes prebuilding 40 leaves. Output destruction follows timed build/merge batches; the public one-element modern CDF includes result allocation/destruction. Fixed samples: build has 1 warmup plus 3 single builds; merges have 2 warmups plus 5 batches of 8 reductions; p50 and CDF have 2 warmup batches plus 5 measured batches of 4096 and 4095 calls respectively. No adaptive calibration or retuning.

## Limits and provenance

This has two process observations per core/cell on one machine, one fresh seed and three inputs; it is descriptive evidence, not a significance test or world-best result. Timing does not pick deltas. The 0.5/1.0-point error policies are unavailable within the frozen 25–1200 grid; 2 KiB policies were selected but deliberately outside this timing subset. All 24 selection decisions remain in the [selection lock](results/selection-lock.json). Range/skew/reversed shard layouts, other scales/policies/platforms and Python adapter speed are not measured here. Modern gains require opting into that core; this does not establish unchanged performance of the compatibility API or its adapter/compaction changes. Large-n Type 7 references are shared and identity-checked, not independently regenerated by this report.

[Analysis validation](results/analysis-validation.json), [build provenance](results/build.json), [full execution metadata](results/full-metadata.json), [smoke metadata](results/smoke-metadata.json), and [identity controls](results/identity-controls.jsonl) preserve sources, locks, binary, schedule and raw hashes. Full source exports, native binary and raw query/state arrays remain outside this compact bundle at [saved locations](FULL_RESULTS_LOCATION.json). [README](README.md) labels the preserved historical preparation documents. The full driver duration includes input/reference calculations and checks and is not a library operation timing.

The [independent native review](results/independent-native-review.json) separately verifies the saved full run, including numerical controls and every policy miss.
