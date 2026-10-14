# Controlled interleaved comparison: 22867bd versus c5dec35

All 32 processes completed. Exact original/modern query summaries, centroid
state, modern count/sum bits and complete modern wire bytes matched across
both versions and all repeats for each of the four cases. Source files stayed
unchanged, manifests/locks matched and retained executable hashes verify.

The run occupied 10.7 minutes including its release builds;
measured processes account for 10.6 minutes.

The earlier double-digit large-case slowdown was not consistent across the
counterbalanced blocks. The original F32 p50 control tracks the observed modern
p50 shift, so neither a blanket regression nor a new speedup is established.
Small sorted-input differences (roughly 0–4% across blocks) remain descriptive
observations, not proof of zero overhead. See `INTERPRETATION.md` for the specific
control shifts and unresolved host/cache/layout uncertainty.

## Descriptive timing comparison

Ratios are candidate/baseline; values above 1 mean the candidate took longer.
The main ratio compares the median of four process medians per version. Each
block ratio compares two process medians per version. These are descriptive
summaries, not independent-sample confidence intervals.

| Input | Operation | Baseline | Candidate | Ratio | ABBA block | BAAB block |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform 10,000,000 F64 | original_build | 2.923 s | 2.916 s | 0.998 | 1.024 | 0.985 |
| uniform 10,000,000 F64 | modern_build | 0.702 s | 0.650 s | 0.925 | 1.013 | 0.880 |
| uniform 10,000,000 F64 | original_p50 | 0.068 us | 0.066 us | 0.977 | 0.992 | 0.912 |
| uniform 10,000,000 F64 | modern_p50 | 0.054 us | 0.053 us | 0.992 | 0.990 | 0.925 |
| uniform 10,000,000 F32 | original_build | 2.180 s | 2.177 s | 0.999 | 0.979 | 1.049 |
| uniform 10,000,000 F32 | modern_build | 0.448 s | 0.448 s | 1.002 | 0.913 | 1.018 |
| uniform 10,000,000 F32 | original_p50 | 0.070 us | 0.067 us | 0.949 | 0.806 | 1.000 |
| uniform 10,000,000 F32 | modern_p50 | 0.053 us | 0.051 us | 0.954 | 0.812 | 0.999 |
| sorted 100,000 F64 | original_build | 735.649 us | 731.664 us | 0.995 | 0.950 | 0.978 |
| sorted 100,000 F64 | modern_build | 975.849 us | 981.215 us | 1.005 | 0.996 | 1.011 |
| sorted 100,000 F64 | original_p50 | 0.063 us | 0.063 us | 0.998 | 0.999 | 0.993 |
| sorted 100,000 F64 | modern_p50 | 0.051 us | 0.051 us | 1.011 | 1.018 | 1.016 |
| sorted 100,000 F32 | original_build | 643.097 us | 631.294 us | 0.982 | 1.001 | 1.010 |
| sorted 100,000 F32 | modern_build | 967.385 us | 980.740 us | 1.014 | 1.002 | 1.043 |
| sorted 100,000 F32 | original_p50 | 0.066 us | 0.065 us | 0.997 | 0.998 | 0.973 |
| sorted 100,000 F32 | modern_p50 | 0.050 us | 0.050 us | 1.001 | 0.999 | 0.997 |

## Method and limits

The immutable source commits are
`22867bd55739a96a6fab3c8c48b27406ecb32197` and
`c5dec35059990b40fe6f036dc83590bf0cdf2a70`. The independent original is
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The newer transactional-scaling/packaging
candidate was not a measured arm and must not be silently substituted in this record.

Release builds used Rust 1.90.0 and the same resolved dependencies/allocator.
Each process ran on Linux logical CPU 0 with RAYON_NUM_THREADS=1.
Round one used ABBA and original-first operation ordering; round two used
BAAB, reversed cases and modern-first ordering. Each operation retained three
warmups, 15 ms calibration and seven batches targeting 25 ms. The inherited
timer sorts its seven sample values, so their chronological order is unavailable.

The unchanged original core is an environmental control, not a correction
factor that can erase drift. The host was not thermally or physically isolated.
Only two blocks, one seed, four cases and one machine are covered. Consistency
across blocks warrants investigation; it does not identify the responsible
instruction or establish a general performance ranking. Full raw process
records and metadata are [committed under results](results/2026-10-15-interleaved-metadata.json).
The complete local run also retains source exports, binaries and logs; those are
not committed. See [reproduction and audit scope](interleaved/README.md).
