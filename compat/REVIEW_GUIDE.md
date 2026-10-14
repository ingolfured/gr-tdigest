# Reviewing the replacement

The promise to existing G-Research applications is simple: keep importing
`tdigest_rs.TDigest`, keep the same calls, and retain the original numerical
behavior. The modern engine is selected explicitly through `gr_tdigest`.
It offers additional operations and measured gains, with different numerical
conventions and some explicit costs.

Preserving the original arithmetic also preserves its performance profile.
The modern engine's speedups do not automatically apply to old imports. Moving
existing applications onto that engine would need a separate, measured numerical
contract decision. Matching `delta` alone does not make the answers identical.

## Read the PRs in this order

| PR | Question to answer | Start here |
| --- | --- | --- |
| **#6: modern correctness and state** | Does weighted mass survive construction, updates, merging, queries and serialization? | Small counterexamples and regressions, then the shared Rust implementation and codecs. |
| **#7: original Python API** | Can an existing application keep its import, arguments, dtype rules, immutable operations and saved-state behavior? | Pinned original source and wrapper, the thin native adapter, and the provenance checks. |
| **#8: evidence and delivery** | Are these contracts checked against independently obtained originals and actual installed artifacts? | Untouched upstream tests, exact-output fixtures, published pickles, release-wheel gate, packaging evidence and performance reports. |

These branches are stacked: #7 is based on #6, and #8 on #7. Review and merge
#6 first, then rebase/retarget #7, then #8. The user owns those merge decisions.
Green private PRs are preparation for upstream PR #38, not permission to release.

## What changes in the modern engine

The regressions begin with ordinary mistakes: a weighted median must use the
weights; scaled CDF tails must use actual mass; true extrema and atomic point
masses must survive compression; protected edges must not overlap. Polars must
apply the requested configuration before compression and preserve that state.

Mean precision and mass precision are now independent. An f32 centroid keeps its
weight in f64, so adding one observation to a pile of weight `2**24` is retained.
Previously the digest count could increase while the centroid weight stayed
unchanged, even moving a two-pile median in the wrong direction. On the tested
Linux target, this grows an f32 centroid from **12 to 16 bytes**; f64 remains
24 bytes. It cannot recover mass already lost in old stored data.

A cluster containing one input preserves its exact mean bits and kind. For a
multi-input cluster, a finite weighted average is bounded to its input interval
if rounding puts it outside. This prevents adjacent clusters from producing
duplicate or out-of-order means. The historical audit's two failures remain
recorded; the fixed implementation passes the same 3,840-state matrix.

Queries of a digest whose entire mass is positive and subnormal use a temporary
view scaled by exact powers of two. Stored state stays unchanged. Separately,
weight scaling now validates every product and both mass totals before mutation:
underflow to zero or overflow returns an error and leaves the original digest
intact. Neither fix establishes correct results for all finite inputs; aggregate
input/merge overflow and weighted-value sum overflow remain open limits.

Modern saved state has explicit compatibility costs:

- Polars writes Float64 weights with either Float32 or Float64 means and reads
  old Float32 weights by exact promotion. Consumers assuming the old struct
  schema must account for that change.
- TDIG v3 flag `0x08` marks f32-mean digests whose weights require f64 precision.
  It adds no bytes to the existing payload; old readers reject it instead of
  silently losing mass. Explicit v1/v2 export rejects such digests.
- Delta mode, wide protected-edge settings, fractional v1 weights and invalid
  encoded mass have their own checked format restrictions. Default v3 preserves
  the modern state; older formats retain documented limitations.

## What stays identical for existing applications

`tdigest_rs.TDigest` retains the upstream Python wrapper and pinned original Rust
arithmetic: constructors, delta arguments, dtype inference, uint32 weights,
immutable merge/update, dictionary state and copy/pickle reconstruction. The
adapter can release spare vector capacity without changing centroid contents.
NumPy-only imports defer Polars loading. These do not restore the original
public Rust crate API or widen this checkout's Python support matrix.

The numerical distinction is substantive. For means `[0,10]` and weights
`[90,10]`, the original median is `1`; the modern weighted-point interpretation
gives `0`. For values `[1,2]`, the original `q=0.1` can be `0.7`, whereas the
modern answer stays inside the observed range. Those are not rounding noise.

A future numerical change needs a concrete benefit and measurements of absolute,
relative and rank error on affected scales, tails and repeated merges. There is
no universal `1e-6` tolerance. Serialization of unchanged state should preserve
its answers; smaller storage alone does not justify changing them.

## Evidence on the corrected source

At `52f6212`, Linux `make build`, `make test` and `make lint` passed: **314
Python/coherence cases plus 91 untouched upstream cases**, passing Rust and Java
tests, and three pre-existing ignored Rust tests. The optimized wheel installed
outside the checkout passed all **208 compatibility cases**, with no skips or
xfails, and **70 modern API/regression cases**.

The same wheel passed two independent checks: **800 weighted-point digests,
33,536 CDF/rank checks and 304 integer-mass checks**; and **1,440 compressed-state
routes covering 3,840 states**, with no failures. The compressed matrix checks
invariants, not exact rank accuracy after compression. The source hashes, wheel
hashes, commands and restack-equivalence proof are in the
[validation records](validation/README.md). Historical failing reports remain
available under their original source labels.

The published-pickle gate uses the authentic PyPI **0.1.0** wheel, distinct from
the pinned source advertising **0.2.0**. Its 32 fixtures compare against the old
package's restored state. The original pickle reconstructs with default delta
300 and can already recompress a digest; preserving that behavior is not a claim
that the old format was lossless. Reverse protocol-4 migration passed the tested
NumPy 1.26.4 environment; protocol 5 failed because NumPy 2 writes a namespace
absent there. See the [pickle provenance and rollback limits](published-pickle/README.md).

Packaging checks found and fixed included Python bytecode and a source-archive
README collision. The corrected wheel and source archive contain no Python
caches. Actual source rebuild and same-distribution upgrade evidence, including
its precise limits, is recorded in [package ownership](package-ownership/README.md).
The source-built name-only `tdigest-rs` prototype upgraded published 0.1.0 without
a manual uninstall: all 208 cases passed after upgrade and reinstall, with all
32 old-restored states exact. This used an explicit NumPy 1.26.4 to 2.3.4
transition on Python 3.12; it does not establish unconstrained dependency
resolution or other Python/platform support.

## Speed, accuracy and storage

The [30-case release comparison at c5dec35](performance/RESULTS-c5dec35.md)
predates only the transactional scaling and packaging changes above. On this
machine, ten-million-value uniform builds were about **3.6× faster for f64 and
4.6× for f32** than the independent original core. Some tiny/sorted builds and
sequential merges were slower. Every measured numerical/count/memory/wire field
matched the earlier `22867bd` run. The [controlled interleaved follow-up](performance/RESULTS-interleaved-20261015.md)
completed 32 processes with identical states and complete wire bytes in both
arms. The earlier double-digit large-case slowdown did not reproduce consistently
across blocks; shifts in the unchanged original control also prevent calling
the observed p50 movement a new modern speedup. Small sorted-case differences
remain descriptive. This does not prove zero overhead or benchmark `52f6212`.

These are matched-delta measurements, not a demonstrated win at every storage or
error budget. Retained capacity, construction peak memory and RSS are different
quantities. At 142 centroids, f32 logical state grew from 1,800 to 2,368 bytes.
Compare build, roughly 40-way merge, p50 and CDF costs separately, and measure
rank/value accuracy across ties, skew and tails. One original implementation,
one machine and a limited matrix cannot establish a global lead.

## What still prevents an upstream release

- Finalize distribution and namespace ownership. A real same-name upgrade is
  different from co-installing `tdigest-rs` and private `gr-tdigest`, which share
  files. The old native extension can shadow the new shim, and uninstalling one
  owner can damage the other. Bundling the modern namespace needs its own policy.
- Restore or explicitly agree the Python/dependency/platform support contract.
  This checkout targets Python 3.12 and newer dependencies; upstream supported
  Python 3.10 and a wider wheel matrix. A Linux wheel is not manylinux proof.
- Confirm direct Rust consumers and the original public crate API contract.
  Python compatibility tests do not establish Rust source compatibility.
- Retain and disclose numerical limits: compression underflow, tiny mass inside
  a normal total, aggregate/product overflow and malformed seeded state. Original
  arithmetic also retains its upstream uint32 overflow and other quirks.
- Refresh the implementation of upstream PR #38 from the final validated work.
  Its status text describes this stack, but its old implementation snapshot has
  not yet been replaced. Do not merge or publish based on the description alone.
