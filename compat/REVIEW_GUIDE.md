# Reviewing the replacement

The goal is to let existing G-Research applications keep using `tdigest_rs.TDigest`,
while making the modern engine a trustworthy option for better storage, richer
operations, and measured performance or accuracy improvements.

## The three changes

1. **Correct the modern engine and preserve its state (PR #6).** Weighted queries
   must use actual mass, CDF must remain monotone and bounded after weight scaling,
   and updates must retain true data extrema. Configuration must affect compression
   when data is ingested and survive Python, Polars, Java, and wire round trips.
   Inspect the small counterexamples before reading the compressor and codec changes.
2. **Keep existing applications working (PR #7).** Preserve the public import,
   constructors, delta parameters, dtype rules, immutable merge/update, and saved
   state. The pinned original arithmetic provides a conservative numerical baseline.
   The adapter releases unused vector capacity without modifying centroid values or
   weights. NumPy-only use no longer imports Polars.
3. **Prove and explain the contract (PR #8).** Require the untouched original suite,
   an independent numerical oracle, cross-surface regressions, and installation of
   an optimized Linux wheel outside the checkout. Keep performance tooling runnable
   and publish comparisons with their settings, provenance, and limitations.

## Numerical change policy

Identical original results are the preferred baseline, not a reason to reject a
useful improvement forever. A justified small difference can be acceptable, but
it must be visible: give the input, old result, new result, absolute/relative
difference, and error against the underlying data. Test both precisions and merge
patterns. Do not apply one absolute epsilon to all data scales or describe a
changed weighted/tail convention as floating-point noise.

For example, old means `[0,10]` with weights `[90,10]` yield median `1`, while the
modern weighted-point interpretation yields `0`. That is a substantive semantic
choice. For values `[1,2]`, the original low-tail extrapolation can return `0.7`
at `q=0.1`; the modern result stays within the observed range. These changes are
currently confined to the explicitly selected modern engine.

Serialization that preserves the same state should preserve answers. A smaller
format is not by itself a reason to change numbers; any lossy encoding would need
its own accuracy evidence and a clear opt-in contract.

## What the measurements must establish

Compare optimized builds on the same machine and inputs. Separate construction,
merge, p50/CDF/query costs, retained memory, and peak memory. State whether controls
are matched by delta, centroid count, memory budget, or error target. Report
empirical rank error alongside value error, including ties, skew, extreme tails,
tiny inputs, and roughly 40-way merges. Separate core measurements from Python
adapter overhead and record source hashes for each result.

The [performance report](performance/README.md) records the measured source, raw
samples, tradeoffs, and reproduction instructions.

The old-import path still uses original arithmetic. Modern-engine timings cannot
be advertised as a speedup for unchanged old imports. Releasing spare vector
capacity can improve retained memory without changing calculations, but it does
not remove the original construction peak. A comparison with one upstream engine
on one machine cannot establish world-leading speed or accuracy.

## Remaining release decisions

- The current distribution is `gr-tdigest`, with Python 3.12 and current NumPy/Polars
  dependencies. Upstream `tdigest-rs` supported a wider Python range. Agree the
  package identity, runtime support matrix, and upgrade/uninstall behavior before
  replacing an installed distribution that owns the same import namespace.
- These gates preserve the original Python contract. Direct compatibility with the
  original public Rust crate API is not yet implemented or verified; confirm whether
  any G-Research consumers need it before preparing the upstream replacement.
- The release Linux wheel gate does not cover every platform or manylinux policy.
  Validate the actual release artifacts and supported environment matrix.
- The modern f32 engine stores centroid weights in f32. Very large atomic piles
  can lose unit-weight increments above `2**24`; this is not a negligible absolute
  output drift. The default f64 engine and the original interface's uint32 weights
  have different limits. Do not claim exact mass preservation for arbitrary f32
  workloads until the representation/accumulation contract is resolved.
- The original arithmetic retains upstream quirks, including uint32 overflow and
  some malformed-input behavior. Passing its suite is compatibility evidence, not
  a claim that those inherited behaviors are ideal.
- Update upstream PR #38 from the validated final tree and refresh its description:
  its earlier snapshot removed delta and required migration, which is not the
  contract of this review stack. It is not ready merely because these private PRs
  are green.

Review and merge in order: #6, then rebase/retarget #7, then #8. The user owns those
merge decisions. Source checksums and golden outputs should change only through
an explicit, documented numerical-contract change.
