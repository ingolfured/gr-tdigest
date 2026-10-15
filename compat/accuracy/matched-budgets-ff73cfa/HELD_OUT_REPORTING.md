# Reporting the completed storage/error study

Prepared while the held-out phase was active. This workflow has not run against its results. It never changes the selection lock or library sources.

The immutable selection lock is `300203022d0dfbaee3e0a5bcf947bee6f804687b927238cd2e1b0576f16bb562`, modern source is `ff73cfab31d6368d203c1a361b8efaaea29d735b`, and original source is `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. Training seeds are 1009, 10007 and 65537; fresh held-out seeds are 15485863, 32452843 and 49979687. No reporting step chooses a delta.

## Stage 1: completed metadata only

After root confirms completion, run `summarize_heldout_metadata.py --execute --run-complete-confirmed --study STUDY --output FRESH_REPORT_DIRECTORY`. Default invocation only prints its plan. The script reads the three phase metadata files, ledger, phase protocols, and selection/replay locks. It never opens compressed cases or invokes a native library. It verifies ledger-bound metadata, completed coverage, recorded reference/invariant success, frozen selections, fresh seeds, every expected route, and policy arithmetic.

The known selection implies 576 held-out dataset-policy invocations and 8064 state records: 2880 original and 5184 modern. There are 16 selected core/precision/policy cells and eight unavailable cells. Every selected cell covers all eight distributions, three sizes and three fresh seeds. Broader modern k-way coverage remains explicit.

`policy-summary.json` contains all 24 choices, violation counts, maximum logical bytes and storage-cap excesses, plus rank-policy maximum queried errors. `route-pairs.jsonl` preserves the same dataset/precision/seed/policy and original/modern deltas. It labels direct/sequential pairs as matching routes and k-way-versus-sequential pairs as different algorithms; these must not be pooled into one winner count. Unavailable policies remain unavailable. The chosen 2% policy is independent of the 1/2/4 KiB policies.

Metadata provides no p50, tail, value-error, or CDF summaries. The metadata-only report explicitly leaves those fields pending and does not imply rehashed raw/binary evidence. Inherited original observations are counted by recorded state occurrence, including repeat inputs evaluated under different policies; they are not counts of independent datasets.

## Stage 2: one bounded streaming pass, only after completion and allocation

Read each compressed case once for numerical aggregation, retaining a single case's states at a time. Hash compressed bytes and compare the complete filename/hash manifest, with metadata and selection hashes rechecked before and after. Bind the raw record's distribution, size, seed, precision, deltas, route and input fingerprint to its metadata case. Independently recompute policy violations from the same state records and require agreement with Stage 1. Retain original nonfinite/out-of-range quirks; nonfinite maxima must be counted and cannot become zero or be silently removed from maxima. Any failure makes the report incomplete/failed rather than a clean summary.

Write compact per-state metrics and aggregate them by policy, precision, core, route, distribution and size. For each group, report all three seed values or descriptive median/range and the worst identified seed. Do not treat thousands of sampled queries or repeated routes/policies as independent experimental replications. Do not average value errors across distributions with different scales.

Required fields:

| Quantity | Definition and display |
|---|---|
| Queried rank error | Distance from effective q to the empirical interval `[F(x-), F(x)]`; raw fraction, displayed in percentage points. Keep max across all 1003 queried q values. |
| p50 rank error | The same tie-aware metric at effective q=0.5. Separate from all-query maximum. |
| Tail rank/value errors | Preserve fixed requested q=0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999 with actual rounded effective q. Report both sides. |
| Value error | Absolute deviation from Type 7 quantile of post-cast input samples, in input units; separately by distribution. Preserve p50 estimates/reference and optional p50/IQR only when IQR>0. |
| Modern CDF error | Absolute deviation from midpoint empirical CDF `(F(x-)+F(x))/2`; raw probability fraction or percentage points. No original CDF comparison. |
| Logical bytes | Object plus live centroid payload. Report maximum and actual per-cell distribution; distinguish original compacted-vector capacity, unknown modern capacity, and serialized bytes. No RSS claim. |
| Policy violation | `logical_bytes > cap` for storage or nonfinite/max queried rank error `> threshold` for rank; equality passes. Every miss remains in primary rows. |
| Change vs original | Pair identical input fingerprints and effective queries. Report estimate drift separately from error relative to samples. A change is not itself an error. |

Copy only compact summaries, selection/protocol/phase provenance, code, and raw hash manifests into the review bundle. Retain full compressed output outside Git. Select a small, clearly identified set of worst-case raw rows for explanation, with source filename/hash and record identity; do not select them to imply a global average. Fresh held-out policy misses do not permit retuning.

The existing `analyze_held_out.py` is historical fixed-delta analysis and must not be run unchanged: it assumes 240 cases/five seeds and pools k-way comparisons. A new budget-aware streaming reducer is still required for Stage 2. No raw replay, timing, or stronger accuracy claim is part of Stage 1.
