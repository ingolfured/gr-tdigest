# Reviewing the replacement

Existing G-Research applications should keep using `tdigest_rs.TDigest`, with its
original Python API and numerical baseline. The modern `gr_tdigest` engine is an
explicit choice for richer operations and measured performance or accuracy gains.
These are two reviewable contracts; the original interface does not silently
switch algorithms when this stack is installed.

## The three changes

1. **Correct the modern engine and preserve its state (PR #6).** Start with the
   small counterexamples: weighted medians, scaled CDF tails, true extrema,
   overlapping protected edges, large f32 piles, and single-centroid clusters.
   A cluster containing one input must retain that input's mean bits and kind;
   recomputing `mean * weight / weight` can move an exact point mass and change
   the CDF at the observed value. Then review compression, Polars configuration,
   and the codecs that preserve this state.
2. **Keep existing applications working (PR #7).** Preserve the public import,
   constructors, delta parameters, dtype rules, uint32 weights, immutable
   merge/update, and original saved-state behavior. Pinned upstream arithmetic
   supplies the baseline. The adapter releases spare vector capacity without
   changing centroid contents, and NumPy-only imports defer Polars loading.
3. **Prove and explain the contract (PR #8).** Require the unchanged upstream
   suite, independent exact-output fixtures, cross-surface regressions, genuine
   published-package pickles, and an optimized Linux wheel installed outside the
   checkout. Keep performance measurements reproducible and tied to their actual
   source hashes.

## Preserve mass independently of mean precision

Modern centroid weights now stay in f64 for both mean precisions. Previously,
adding one observation to an f32 pile of weight `2**24` could advance the total
count while leaving that centroid's weight unchanged. With two equal piles,
adding on the left could even move the median to the right. The new representation
retains the increment through construction, updates, merge, scaling, precision
casts, and saved state. It cannot recover mass already lost in old stored data.

This has an explicit cost: on the tested Linux x86_64/Rust 1.90 target, an f32
centroid grows from **12 to 16 bytes**; the f64 centroid remains 24 bytes. Means
and extrema retain their selected precision. Rust `weight() -> F` remains a
potentially lossy compatibility view; `weight_f64()` is authoritative.

Polars now writes Float64 weights with either Float32 or Float64 means. It reads
old Float32-weight structs by exact promotion, and mutations report the new
schema. Consumers that assume the old schema need to account for that change.
TDIG v3 sets checksum-covered flag **0x08** only when f32 means accompany weights
that cannot round-trip through f32. It adds no bytes because payload weights were
already f64; older readers reject the flag instead of silently narrowing mass.
Explicit v1/v2 export rejects such digests. An explicit f64 mean cast can satisfy
this weight restriction where the other old-format restrictions also permit
export. Delta and full-width protected-edge extensions retain their own guards.

## Numerical change policy

Identical original results are the preferred baseline. A useful numerical change
can be acceptable if it is disclosed with inputs, old/new results, absolute and
relative differences, and accuracy against the underlying data. Cover both
precisions and merge patterns; one absolute epsilon is not meaningful at every
scale. Changed weighted or tail conventions are not floating-point noise.

For means `[0,10]` and weights `[90,10]`, the original median is `1`, while the
modern weighted-point interpretation gives `0`. For values `[1,2]`, the original
`q=0.1` estimate can be `0.7`; the modern result stays within the observed range.
These modern semantics do not change the original import's calculations.

Serialization of the same state should preserve its answers. Smaller storage
alone does not justify changing numbers; lossy storage needs its own accuracy
evidence and an explicit contract.

## Genuine saved-state migration

The [published-pickle gate](published-pickle/README.md) uses the actual PyPI
`tdigest-rs` **0.1.0** wheel with NumPy 1.26.4, distinct from the pinned source
advertising **0.2.0**. It checks both precisions, weighted data, construction and
merge deltas, and pickle protocols 4/5 against the old package's restored results.

The original pickle stores means/weights and reconstructs with default delta 300.
It can already recompress state: the delta-5000 example shrinks from 1833 to 134
centroids in the old package itself. Matching that restored result preserves the
existing behavior; it does not imply the original pickle was lossless.

Forward migration passed the recorded candidate wheel with NumPy 2.3.4. Reverse
protocol-4 migration passed the tested NumPy 1.26.4 environment; protocol 5 failed
because NumPy 2 writes `numpy._core.numeric`, absent there. This is a rollback
constraint, not numerical drift. Retain old state or use an explicitly tested
interoperable policy if rollback is required. Provenance identifies the historical
artifact; the final stack must pass its own installed-wheel gate.

## Performance evidence

The [retained 30-case comparison](performance/RESULTS-756d221.md) records the earlier source at `756d221`. A fresh run of the corrected engine is pending; no final speed claim is made here. The [weighted-point audit](weighted-point-audit/README.md) supplies a separate reproducible correctness check with exact rational input weights.

Compare optimized builds on the same machine and inputs. Separate construction,
merge, p50/CDF queries, retained capacity, and peak memory. Match or disclose delta,
centroid count, memory, and error budgets. Report sampled empirical rank error
alongside value error, including ties, skew, tails, tiny inputs, and roughly
40-way merges. Distinguish sequential and one-shot merge semantics.

Measurements at `756d221` predate the f64-weight and single-centroid fixes. Preserve
that evidence with its original labels; it cannot establish the final code's
performance. The larger f32 representation needs fresh timing and memory evidence.
Modern speedups do not apply to unchanged `tdigest_rs` imports. Adapter compaction
reduces reserved result capacity, not the original construction peak or measured
RSS. One upstream comparison on one machine cannot establish a global lead.

## Remaining release decisions

- Accumulation still uses f64. Extremely small subnormal masses, very disparate
  weights, and finite weights whose totals/products overflow remain numerical
  limits requiring explicit handling. The f32 storage fix does not establish
  correct queries or exact mass for every finite input.
- Agree distribution identity, Python/NumPy/Polars support, and upgrade/uninstall
  behavior. This checkout still packages `gr-tdigest` for Python 3.12; upstream
  supported a wider Python range and owns the `tdigest_rs` namespace.
- The original public Rust crate API is not restored by the Python compatibility
  tests. Confirm whether G-Research has direct Rust consumers and define that
  migration contract before replacing the upstream project.
- Validate the actual platform and manylinux release artifacts. A release Linux
  wheel test covers only its recorded environment, including its pickle limits.
- Original arithmetic retains uint32 overflow and other upstream quirks. Passing
  its tests is compatibility evidence, not a claim those behaviors are ideal.
- Refresh upstream PR #38 from the final validated tree and rewrite its migration
  story. Green private review PRs do not by themselves make it release-ready.

Review and merge #6, then rebase/retarget #7, then #8. The user owns these merge
decisions. Change source checksums and golden outputs only through a documented
numerical-contract decision.
