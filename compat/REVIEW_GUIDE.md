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
| **#8: evidence and delivery** | Are the original Rust API, runtime support and release artifacts ready, with honest accuracy and speed evidence? | Original Rust crate, runtime/packaging commits, untouched tests, exact fixtures, published pickles, installed artifacts and measurement reports. |

These branches are stacked: #7 is based on #6, and #8 on #7. Review and merge
#6 first, then rebase/retarget #7, then #8. The user owns those merge decisions.
Green private PRs are preparation for upstream PR #38, not permission to release.

PR #8 finishes the delivery contract in separate commits: `3b8fa69` restores the
original Rust package, `360ad5b` restores the tested Python runtime range and
optional Polars installation, `3d0182a` corrects its metadata-format check, and
the following evidence commit records the results. PR #6 and #7 stay at their
already validated bases.

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
NumPy-only imports defer Polars loading.

The original public Rust `tdigest_core` package is now a lightweight workspace
member using the preserved original source. Public fields, seven methods and
result types are covered by new tests and external packaged/path consumers.
Git dependencies can select that package by name; a dependency pointing at the
old repository root must instead point at `src/upstream`. The latter is an
explicit source-layout caveat, not universal Rust drop-in compatibility.

The Python package now supports the tested CPython 3.10–3.12 and NumPy 1/2
combinations. Polars is an optional `[polars]` extra pinned to tested 1.34.0.
Python 3.13+, other NumPy combinations and the full wheel platform matrix remain
unverified. This bounds the former unbounded upstream Python metadata.

The numerical distinction is substantive. For means `[0,10]` and weights
`[90,10]`, the original median is `1`; the modern weighted-point interpretation
gives `0`. For values `[1,2]`, the original `q=0.1` can be `0.7`, whereas the
modern answer stays inside the observed range. Those are not rounding noise.

A future numerical change needs a concrete benefit and measurements of absolute,
relative and rank error on affected scales, tails and repeated merges. There is
no universal `1e-6` tolerance. Serialization of unchanged state should preserve
its answers; smaller storage alone does not justify changing them.

## Evidence on the corrected source

The latest K2Norm correction at `ff73cfa` passes the full Linux build, test and
lint gates: **334 Python/coherence + 91 untouched upstream cases**, **121 Rust
tests**, and Java; three pre-existing Rust tests remain ignored. The direct
optimized wheel and a distinct wheel rebuilt from its unmodified normalized
source archive each pass **2,022 test instances** across seven runtime cells,
with no skips/xfails. Each full cell runs 208 compatibility + 90 modern cases;
the Polars-free cell runs 208 + 26. The direct wheel also passes both numerical
invariant audits below. These are repeated environment cases, not 2,022 distinct
test definitions. [Exact logs and identities](validation/README.md) connect each
claim to its source and artifact. Earlier delivery evidence follows separately.

The unchanged library code passed Linux `make build`, `make test` and `make lint`
at `360ad5b`; `3d0182a` only corrects metadata spacing/order assertions in the
new artifact gate. The canonical suite passed: **314
Python/coherence cases plus 91 untouched upstream cases**, **116 Rust tests**,
passing Java tests, and three pre-existing ignored Rust tests. One actual
optimized cp310 ABI3 wheel passed **1,902 test instances** across seven isolated
environments: six CPython 3.10/3.11/3.12 × NumPy 1/2 cells each ran all **208
compatibility + 70 modern cases**; a real Polars-free cell ran **208 + 26**.
No cases were skipped or xfailed. These are repeated tests across environments,
not 1,902 distinct tests.

An actual normalized source archive was rebuilt without editing its extracted
files or lock. That distinct wheel passed its own same seven-cell gate. The
checks verify installed paths/native hashes, normal dependency checks, source
archive paths and active source bytes. They do not establish an unconstrained
dependency resolver or manylinux/cross-platform support.

The direct wheel also passed **800 weighted-point digests, 33,536 CDF/rank checks
and 304 integer-mass checks**, plus **1,440 compressed-state routes / 3,840
states**. These check bounded correctness/invariants, not a universal compressed
accuracy guarantee. Exact artifact hashes and commands are in the
[validation records](validation/README.md).

The independently exported original was rebuilt with its **own committed
Cargo.lock**. Both debug and release matched the 70-case fixture: **1,762 exact
bit comparisons plus 126 NaN-classification checks per profile**. Source, lock,
package-consumer evidence and reproduction instructions are in the
[original Rust evidence](numerical-review/original-rust/README.md). Current
upstream main `32a49203` changes only workflows relative to the numerical pin;
the original Rust/Python implementation and manifests are unchanged.

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
is historical evidence from before the later transactional scaling, K2Norm domain
and delivery changes. On this machine, ten-million-value uniform builds were about **3.6× faster for f64 and
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

## What the broader accuracy survey found

The [five-seed held-out survey](accuracy/held-out-20261015/RESULTS.md) tested
240 datasets and 3,360 states at historical modern source `52f6212`, against an
independent original. It preserves source-specific results; it is not a rerun
of the new runtime artifact and has no operation timings.

All tested modern invariants and wire round-trips passed. In 1,200 comparable
direct/sequential pairs, modern maximum queried-rank error was lower in 609,
equal in 526 and higher in 65. Exact discrete ties improved strongly. That is
not a blanket accuracy win: separated narrow modes produced CDF errors up to
**5.256 percentage points**, and medians far from Type-7 sample quantiles even
when rank error was small. Some large median errors also occur in the original.
The [independent reduced examples](numerical-review/separated-modes/README.md)
identify compression/interpolation loss across gaps, not a demonstrated mass
bug with a narrow arithmetic fix. No speculative fix was applied.

At the same delta, continuous data generally uses about twice the logical
retained storage of the original; ties can use less. Logical state excludes
spare capacity, temporary merge memory and RSS. The speed numbers above are
therefore not a matched-memory win. A separate predeclared storage/error-budget
study is being prepared; no result from that study is claimed here.

A separate small-input defect was confirmed in historical `52f6212` and the
preserved `3d0182a` wheel: at delta 1,200, two normalized means make the K2Norm
normalizer negative. For `[0, 0, 100]`, the modern median became about 33.333
instead of preserving the atomic piles. The current focused correction retains
the input centroids when that normalizer is zero or negative. It preserves
ordinary positive-domain calculations, including default delta 300, and leaves
the original compatibility path unchanged. Non-delta K2Norm protected interiors
use the same guard; their later cap stages still run. Retaining information may
use more centroids than the broken collapse.

The 20 new Python/Polars regressions fail in 14 expected cases against the
preserved wheel, with six passing controls, then all pass on the corrected
candidate. Four focused Rust integration tests and one private compressor test
pass in both debug and release. The installed-wheel matrix now includes those
20 cases (90 modern cases per full cell; 26 in the Polars-free cell). Full-project,
direct installed-wheel and unmodified source-rebuilt-wheel gates now pass at
`ff73cfa`, as recorded above. Historical `3d0182a` results retain their original
source and counts. No new performance or broad accuracy result is claimed for
this fix.

## What still prevents an upstream release

- Finalize distribution and namespace ownership. A real same-name upgrade is
  different from co-installing `tdigest-rs` and private `gr-tdigest`, which share
  files. The old native extension can shadow the new shim, and uninstalling one
  owner can damage the other. Bundling the modern namespace needs its own policy.
- Complete the remaining Python/dependency/platform contract: CPython 3.10–3.12
  and pinned NumPy 1/2 combinations now pass, but Python 3.13+, the wider wheel
  matrix and upstream's older Linux deployment floor remain unproven.
- Confirm the original Rust root-path consumer caveat and any registry release
  ordering. The standalone original crate is restored and tested; the root
  package is still the modern implementation.
- Retain and disclose numerical limits: compression underflow, tiny mass inside
  a normal total, aggregate/product overflow and malformed seeded state. Original
  arithmetic also retains its upstream uint32 overflow and other quirks.
- Refresh the implementation of upstream PR #38 from the final validated work.
  Its status text describes this stack, but its old implementation snapshot has
  not yet been replaced. Do not merge or publish based on the description alone.
