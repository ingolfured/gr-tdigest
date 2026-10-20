# One product first; review PRs afterwards

**Recommendation: merge none of the existing stack yet.** Keep draft
[PR17](https://github.com/ingolfured/gr-tdigest/pull/17) as the combined integration
candidate. Finish and qualify that product, then reorganize the changes into a
small review sequence. Existing PRs and historical evidence remain preserved.

This is not the final product. The current development version is
`0.2.5-dev2026102002` (Python `0.2.5.dev2026102002`). The previous artifact manifest,
coverage and report in this directory describe source `a9c368f`, version
`0.2.5-dev2026102001`. They remain historical evidence for those exact bytes.
They do not qualify the latest code, and no replacement canonical artifacts have
been declared yet.

## The product we are finishing

There are two explicit engines. Modern `gr_tdigest` uses the approved new API,
capacity1000 and newer dependencies, including Polars2.0. Original `tdigest_rs`
and `tdigest_core` retain the pinned original API and calculations. Supplying
delta selects modern configuration, not the original engine.

Modern operations must preserve positive statistical mass, full saved state and
point identity. Failed updates must leave receivers unchanged. Missing digests
and mixed configurations/capacities are errors. Python/NumPy, Polars and JVM
interfaces must expose and test their actual public behavior, including shapes,
group cardinality, ownership and serialization.

Builds must support declared internal sources and prepared-cache offline use.
Java11/17/21 runtime checks and Scala2.12.16 examples are distinct from the build
JDK. Python3.12/3.14/3.15 support is checked on actual wheel bytes and exact
NumPy combinations. Python glibc2.28 and JNI glibc2.17 remain separate targets.

## What the latest work resolves

- Seven newly introduced accuracy regression profiles are removed by preserving
  established finite same-sign averaging. Mixed-sign cancellation compensation
  remains. No quality reference or tolerance was changed.
- All15 exact numerical boundary controls pass unchanged expected values.
- Public modern Rust construction and restoration return checked results before
  exposing state. Invalid options return errors; oversized seeds retain their
  requested capacity. The original Rust API is unchanged.
- Float32 training/conversion rejects overflow and loss of nonzero values to zero.
  Valid Float32 rounding does not narrow statistical mass. Known exact points
  retain queries even when their rounded weighted moment is subnormal or zero.
- The exact-boundary regression audit accepts verified fixes against unchanged
  targets. Missing/duplicate/changed controls, misleading PASS labels and new or
  changed failures are rejected. Quality runs invalidate their evidence if source
  or the loaded native binary changes during measurement.

These are development/source checks. The latest uniquely versioned wheel and JAR
still need final packaging and the complete same-artifact runtime qualification.

## What prevents a final-product claim

The extended suite has ten established failed accuracy profiles: small summaries
with repeated merging on separated or widely varying values. They remain failures,
even when a known-regression check passes. The suite separates rank, value, CDF
and tail errors and preserves losses against the original. It does not establish
that the modern algorithm is better on every metric or domain.

A trial forced a compression partition at the median. It removed the ten envelope
failures but introduced290 regressed profiles; the worst median rank error grew
from0.00842 to0.08167. The trial was rejected. Its measured evidence is retained;
its implementation is absent from the candidate.

Mixed-observation weighted products that are too small to represent still fail
explicitly. That restriction is not a silently approved exclusion from the
requested domain. Any unresolved accuracy limit or changed domain needs a concrete
proposal and the user's decision before release acceptance.

After the algorithm and public contracts are settled, freeze one source revision
and build new uniquely identified artifacts. Rerun the combined build/test/lint,
quality gates, exact wheel/JAR runtime and minimum-ABI checks, installed docs/types,
source archive rebuilds, cross-language roundtrips and relevant workload comparisons.
Passing checks from earlier branches or earlier artifacts cannot fill these rows.

Private0.1.9/0.1.11 regression tests, original JVM0.1.16 provenance, real internal
repositories/trust, the exact Spark/serializer/checkpoint/executor matrix and
representative100-plus-CPU servers remain consumer qualification. They are not
expected to run on this development machine. The external SSD comparison remains
an explicitly pending TODO; preserve its branches and experiment material.

## Proposed review story after completion

1. Preserve the original engine and prove its compatibility with pinned sources
   and original-output fixtures.
2. Make the build and dependency/tool selection work for the supported runtimes
   and restricted internal environments.
3. Establish modern core/state/numerical contracts with their regression cases.
4. Expose those contracts through Python/NumPy and Polars, including migration,
   schema, cardinality, types and installed documentation.
5. Complete JVM public APIs, ownership/serialization/native loading and tested
   Scala/Kotlin examples.
6. Deliver quality/workload evidence, reproducible source and artifact manifests,
   and a clear passed/failed/untested acceptance report.

This is an intended review structure, not authorization to merge incomplete
chapters or a promise that each chapter is independent. Buildable/testable
boundaries and the final combined result determine the actual split. Performance
experiments with unaccepted tradeoffs remain separate; experimental CDF PR13 is
not included automatically. No existing PR will be closed or rewritten until its
useful changes and evidence have a clear replacement.
