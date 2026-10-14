# Reviewing the replacement

Existing G-Research applications should keep using `tdigest_rs.TDigest`, with its
original Python API and numerical baseline. The modern `gr_tdigest` engine is an
explicit choice for richer operations and measured performance or accuracy gains.
These are two reviewable contracts; the original interface does not silently
switch algorithms when this stack is installed.

Keeping the old calculations preserves their answers and their performance
profile. Compatible optimizations, such as releasing spare vector capacity, are
still possible. The modern engine changes compression and query conventions, so
we cannot silently route old calls through it while promising both identical old
answers and its measured speedups. Any future switch needs measured differences
and an explicit numerical-contract decision.

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

## The latest boundary fixes

The historical compressed-state audit at `22867bd` found two strict-order
failures among 3,840 checked states. A finite weighted average had rounded beyond
its cluster's last input mean, colliding with the next centroid. Commit `1d9993f`
bounds the finite result to its input interval; single-input clusters still keep
their exact mean bits. The [original failing report](compressed-state-audit/README.md)
stays intact. Its fixed-source rerun is **PASS: 1,440 routes and 3,840 states, with zero failures** at
`c5dec35`; regression tests and an audit rerun are separate evidence.

Commit `8ce82fc` also handles positive subnormal total mass: weights so small that
halving them or multiplying them by a quantile can round to zero. CDF and quantile
use a temporary view scaled by two exact powers of two, with total mass in
`[1,2)`. Stored state and the original `tdigest_rs` arithmetic are untouched.
This repairs queries of the stored centroids; it cannot recover information
already lost during compression. Tiny components inside a normal total,
inconsistent headers and aggregate/product overflow remain separate limits.

The two fixes add **13 Rust tests and 10 parameterized Python/Polars cases**.
They passed in the full Linux suite: **Linux make build/test/lint passed, with 294 Python/coherence cases plus 91 unchanged upstream cases and passing Rust/Java tests; the optimized installed-wheel compatibility gate passed all 208 cases**. Normal-total
queries retain their arithmetic and allocation behavior, with an extra condition;
the rare subnormal path allocates a copy. The fresh performance report below
records the new source; cross-run timing changes are not a causal measurement
of the added compression checks.

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

The [recorded 30-case release comparison](performance/RESULTS-22867bd.md) measures the engine at `22867bd`, before the two boundary fixes above. On this WSL machine, ten-million-value uniform builds were about 4.1x faster for f64 and 5.0x for f32 than the independent original core. Some tiny/sorted builds and sequential merges were slower. At 142 centroids, f32 logical state grew from 1,800 to 2,368 bytes; wire lengths in the matrix were unchanged. The complete report preserves every measured tradeoff.

The [historical validation record](validation/22867bd-validation.json) identifies that exact release wheel: the original compatibility gate passed 208 cases, and an additional 40 modern Python/Polars cases passed outside the checkout. The [weighted-point audit](weighted-point-audit/README.md) supplies a separate reproducible correctness check with exact rational input weights; the same installed wheel passed all 800 digests, 33,536 CDF/rank checks, and 304 integer-mass checks.

Those results remain attached to `22867bd`; they are not new-source validation
or timing. Boundary-fix performance evidence: **The [fresh 30-case run at c5dec35](performance/RESULTS-c5dec35.md) retained every recorded numerical/count/memory/wire field. On this run, 10M uniform construction was about 3.6× faster for f64 and 4.6× for f32 than the original core. Small/sorted construction and sequential merges still lose in selected cases. Modern large-build medians rose versus the previous run; a focused interleaved comparison is required to determine the cause**.
Final release artifact and gate evidence: **[exact artifact validation](validation/c5dec35-validation.json)**.

Compare optimized builds on the same machine and inputs. Separate construction,
merge, p50/CDF queries, retained capacity, and peak memory. Match or disclose delta,
centroid count, memory, and error budgets. Report sampled empirical rank error
alongside value error, including ties, skew, tails, tiny inputs, and roughly
40-way merges. Distinguish sequential and one-shot merge semantics.

Measurements at `756d221` predate the f64-weight and single-centroid fixes. They
remain under their original labels; the new `22867bd` run records the larger f32
representation and its actual timing and memory costs.
Modern speedups do not apply to unchanged `tdigest_rs` imports. Adapter compaction
reduces reserved result capacity, not the original construction peak or measured
RSS. One upstream comparison on one machine cannot establish a global lead.

## Remaining release decisions

- **Upgrade packaging is still a release gate.** The published `tdigest-rs` and current private `gr-tdigest` wheels share `tdigest_rs` files. If both are installed, the old `tdigest_rs.abi3.so` takes precedence over the new Python shim; uninstalling either can also remove shared files. The upstream replacement must use a normal same-distribution `tdigest-rs` upgrade, keep one owner per namespace, and test the actual upgrade/uninstall path. See [package ownership evidence](package-ownership/README.md).
- Accumulation still uses f64. Subnormal compression can lose mean/sum
  information before queries; tiny components within a normal total and finite
  weights whose totals/products overflow remain numerical limits. The query
  fallback and f32 mass fix do not establish correct answers for every finite input.
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
