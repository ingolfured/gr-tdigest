# Measured accuracy and tradeoffs

These are historical measurements from **21 October 2026**, using Python
candidate `0.2.5.dev2026102102` at runtime source
`aaf7ab3956fed4269c1eaeef915a6faee78fa305`. They are not a fresh
qualification of the final 0.3.0 binaries or a throughput benchmark.

At the default settings, modern `gr_tdigest` reduced average percentile-position
error by **98.7% across 396 profiles** and **99.6% on the recorded-data subset**.
Numerical value errors and storage costs have different tradeoffs.

## Comparison setup

Both engines received the same observations at the same precision. Original
`tdigest_rs` used its **delta 300** default; modern used **capacity 1,000 / K2 /
singleton Use**. These settings do not imply equal summary sizes.

The default suite contains 33 data columns/distributions, both mean precisions,
forward and reverse input, and three routes: one build, 40 sequential merges
and a balanced 40-way merge. The 108 recorded-data profiles use all rows of the
[UCI bike-rental and household-energy fixtures](data/README.md). The remaining
profiles cover deterministic distributions. Profiles share data and are not
independent statistical trials; each profile receives equal weight in averages.

Position error measures how far an answer falls from the requested position
in sorted observations, allowing repeated values their full valid interval.
Value error measures numerical distance from a type-7 interpolated sample
percentile. A small position difference across a wide gap can produce a large
value difference, so the measures remain separate.

## Default-setting results

Position errors below are in **percentage points**, not relative percentages.

| Across all 396 profiles | Original | Modern | Reduction |
| --- | ---: | ---: | ---: |
| Average percentile-position error | 0.26876 | **0.00341** | **98.7%** |
| Average of each profile's worst tested position error | 1.34577 | **0.01995** | **98.5%** |
| Average median-position error | 0.26591 | **0.00426** | **98.4%** |

The recorded-data subset's average position error fell from **0.46645 to
0.00172 percentage points**. The largest position error anywhere on the tested
grid fell from **7.39151 to 0.28076 percentage points**.

| Per-profile measure | Modern better | Tied | Original better |
| --- | ---: | ---: | ---: |
| Average position error | **378** | 18 | 0 |
| Worst tested position error | **378** | 18 | 0 |
| Tail position error | **340** | 56 | 0 |
| Median position error | **218** | 172 | 6 |
| Average numerical value error | **272** | 18 | 106 |
| Worst numerical value error | 168 | 30 | **198** |

Counts use the established 1e-12 score-comparison resolution. At the sampled
tail probabilities (at or below 1% and at or above 99%), modern had no observed
rank error. This is a finite-grid observation, not a guarantee for every query.

Modern's maximum sampled CDF error was 0.27798 percentage points. The original
has no CDF API, so there is no original CDF comparison.

## Storage and comparable-storage results

At defaults, the median paired modern/original logical field-byte ratio was
**5.67** across all profiles and **0.71** on recorded data. Modern used less
field storage in 64 of 108 recorded-data profiles and more in 44. Repeated
values can collapse into exact point masses, while continuous distributions
use more of modern's larger capacity.

These sizes count centroid fields, excluding headers, spare capacity,
allocators and process memory. They do not establish equivalent RAM ratios.

The separate comparable-storage comparison contains all 234 eligible
established-corpus profiles at modern capacity 1,000. The original parameter
was selected by size, never accuracy, to occupy 90–100% of modern's logical
field-byte budget. This corpus differs from the default suite; the two tables
should not be subtracted from each other.

| At comparable logical storage | Modern better | Tied | Original better |
| --- | ---: | ---: | ---: |
| Average position error | **213** | 0 | 21 |
| Worst tested position error | **212** | 2 | 20 |
| Tail position error | **223** | 11 | 0 |
| Median position error | **121** | 100 | 13 |
| Average numerical value error | **135** | 0 | 99 |
| Worst numerical value error | 58 | 0 | **176** |

## Why some scores move the other way

**Observed points versus interpolation.** All 204 default profiles with any
loss beyond the existing comparison resolution were reconstructed. In each,
modern's largest value-error probe returned an observed value at a valid
percentile position, with zero rank error. The type-7 reference interpolates
between observations. Across wide gaps, those conventions produce different
values; this explains the largest value errors without turning them into wins.

For example, on natural-order Float64 bike-rental counts with a single build,
average value error fell from 0.3651 to 0.0561 rentals, while worst value error
increased from 2.9994 to 4.4268 rentals.

**Compression and merge order.** Of 2,117 probes where modern's value error was
larger, 1,709 had zero modern rank error and 1,999 had modern rank error no worse
than the original. The other 118 had locally worse rank estimates across 23
profiles. The largest increase was 0.24414 percentage points on reversed,
sequentially merged `signed_scale` data.

Two of the six strict median losses exceeded the existing one-observation
comparison resolution: the same 8,192-value separated-cluster case under both
precisions with sequential merging. Modern missed by two observations
(0.02441 percentage points); its single-build and balanced-merge routes had
zero median-position error on that case.

## Source evidence and current checks

The original implementation was pinned to
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The
[historical evidence snapshot](https://github.com/ingolfured/gr-tdigest/tree/7dff6b3b9ea3dc831811c2b59c0f8e447ec6e489/compat/accuracy-20261021)
retains the full scores, query explanations, configuration and artifact hashes.
The [historical default-comparison runner](https://github.com/ingolfured/gr-tdigest/blob/7dff6b3b9ea3dc831811c2b59c0f8e447ec6e489/scripts/compare_default_accuracy.py)
and reconstruction instructions remain available at that fixed revision.
The archived results and scripts are not needed to run the current test suite.

Current checks use the existing local fixtures and shared exact reference.
See [the quality guide](README.md) for `make quality-fast`,
`make quality-extended` and `make quality-budgets`. These commands measure
their documented configurations; they do not reproduce the archived default
comparison merely by using the same datasets.

This report changes no algorithm, default, fixture, tolerance, known-limit
inventory or regression gate. Strong position/tail results do not imply that
every numerical-value metric improves, or that every possible input is covered.
