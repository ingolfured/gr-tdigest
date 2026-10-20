# Consumer handoff: first gap assessment

Assessment source: `1451f94c4d809dd43d851b91c7c777ad3c3ff2f3`, verified on
20 October 2026. This is an assessment of that revision, not a release approval
or qualification of a future combined candidate.

## Conclusion

This source is a useful integration starting point, but is **not ready for
consumer acceptance**. Passing ordinary tests coexist with ten failed accuracy
profiles and thirteen failed exact numerical boundary controls. In-place Python
multi-merge is not failure-atomic. The Java class lacks an array-only merge-all
overload, uses jemalloc on Linux and lacks the historical loader controls. Its
public API does not expose all modern configuration choices. The Python class
has no typing metadata and its native class documentation is absent.

The declared consumer combination (Python 3.12 / Polars 1.38.1 / NumPy 2.5.4 /
PyArrow 24.0.0) has not been qualified. The assessment source instead pins
Polars 2.0.0. Later user direction explicitly permits migrating the modern
engine to newer dependencies; this is a migration, not evidence for 1.38.1.

## Source and engine identity

The personal branch still resolves to the assessment SHA. Upstream PR #38 still
resolves to `57e7978d0811baed4a546ff2f08dccaeff03c321`, and remains open.
The handoff/archive ref resolves to `adaf1949d95e42161ca7c490cf8e980025e2275b`.
Exact refs and required binary baseline hashes are in `source-identity.json`.

`tdigest_rs.TDigest` and the `tdigest_core` crate use the preserved original
implementation at `8d9cf1fe82c5c9254e5dc9593f01717e317f4333`. The source guard,
unchanged upstream wrapper and 91 upstream cases, independent oracle, and 70
exact-output fixtures remain required. Dictionary/pickle restoration preserves
the original loader, including default-delta recompression. It is not lossless.

`gr_tdigest.TDigest`, Polars and `gr.tdigest.TDigest` select the modern engine.
Its `delta` option selects K2Norm/Off compression, **not** original arithmetic.
Historical prototypes and punctuation in distribution names do not identify an
engine. The required 0.1.9/0.1.11 wheels and JVM 0.1.16 receipt are not available
here. Their provenance and numerical comparisons remain explicitly unverified.

## Decisions received after the handoff

* The new modern API may migrate to newer dependencies and defaults, including
  capacity 1,000. Historical modern callers migrate; no extra compatibility
  adapter for 0.1.9/0.1.11 is requested.
* Legacy means the identical pinned original `tdigest_rs` implementation.
* Missing modern digests and mixed capacities are rejected.
* Spark 3.5 / Scala 2.12 is provisional. Exact Spark, Scala, serializer,
  checkpoint and executor settings remain internal confirmation items.
* No production publication or deployment is authorized.

## Exact public NumPy receipts

`numpy-receipts.json` contains paths, receipt hashes and individual cells.
The exact NumPy 2 releases recorded in the public runtime matrices are
**2.2.6, 2.3.4 and 2.5.4**. This does not establish support for every NumPy 2.x.

| Receipt/source family | CPython | NumPy | Polars |
|---|---|---|---|
| Earlier compatibility receipts, `runtime-3d0182a` and `runtime-ff73cfa` | 3.10 | 1.26.4 / 2.2.6 | 1.34.0 |
| Same | 3.11 / 3.12 | 1.26.4 / 2.3.4 | 1.34.0 |
| `python-315-20261020` and final quality runtime matrix | 3.10 | 1.26.4 / 2.2.6 | 2.0.0 |
| Same | 3.11 / 3.12 | 1.26.4 / 2.3.4 | 2.0.0 |
| Same | 3.13 | 2.3.4 | 2.0.0 |
| Same | 3.14 / 3.15 | 2.5.4 | 2.0.0 |
| Installed quality checks | 3.12 / 3.14 / 3.15 | 2.5.4 | 2.0.0 |

Polars-free checks also exist for 3.10/1.26.4 and 3.15/2.5.4 on the newer
stack. Earlier receipts include 3.10/1.26.4 without Polars. The user-reported
3.12.3/1.38.1/2.2.6 and sandbox 1.41.2/2.4.6 checks are separate consumer
evidence, not additional public candidate-wheel cells. PyArrow 24.0.0 and the
declared 1.38.1 combination remain untested in these public receipts. None of
these matrices qualifies a new candidate wheel before that exact wheel is run.

## Requirement coverage at the assessment revision

“Existing coverage” means source/tests or historical receipts exist. It does
not mean a new combined candidate or an internal deployment passed.

| Area | Assessment | Evidence / work required |
|---|---|---|
| Original Python/Rust arithmetic, dtype and immutable operations | Existing guarded coverage | `src/upstream/PROVENANCE.md`, `scripts/check_upstream_sources.py`, original fixtures; rerun on candidate |
| Exact original dictionary/pickle results | Existing coverage | Unchanged wrapper and independent original-output tests; do not replace with modern bytes |
| Python 3.10–3.15 | Bounded existing coverage | Exact matrix above; standard CPython only; requalify same candidate bytes |
| Polars 1.38.1 consumer target | Untested / migration | Source requires 2.0.0; newer dependencies now allowed, document migration |
| Python manylinux 2.28 floor | Unqualified | Local receipts use `linux_x86_64`, not a floor-qualified release wheel; build and inspect actual artifact |
| Java 11/17 | Existing runtime coverage | JUnit, bytecode and outside-checkout JAR checks; final candidate must repeat |
| Java 21 / Scala 2.12.16 / Kotlin | Untested | Actual runtime and language compiler checks required; Java-only is insufficient |
| JNI glibc 2.17 floor | Unqualified | Host builds do not prove old ABI; inspect symbols and run against chosen floor |
| Weighted construction and addition | Partly covered | `from_means_weights` and `add_weighted` work; `from_array(weights=...)` clearly rejects, does not ignore it |
| Mean precision independent of weight precision | Mostly implemented | Core/exports use f64 mass; legacy `Centroid::weight() -> F` still narrows and needs explicit documentation or modern API change |
| Mass and contribution preservation | **Fail** | Known boundary controls: small-capacity Use loses mass; residual contribution/cancellation/subnormal mean failures |
| Fractional weighted median | **Fail in known cases** | Four asymmetric boundary cases despite simple 90/10 test passing |
| Overflow/underflow handling | Incomplete | Some scaling checks exist; aggregate, cast and raw reconstruction paths need review |
| Modern failure atomicity | **Fail** | `src/py.rs::merge` commits each iterable item before later item validation; receiver changes on failure |
| Scalar/list/tuple queries | Existing coverage | `src/py.rs` preserves these containers |
| NumPy query shape/dtype | Explicit migration needed | New output is f64 1-D; multidimensional input errors; 0-D becomes scalar; integer probes already produce floating probabilities |
| Python pickle/copy | Implemented | Modern bytes-based reduction exists; prior consumer defect is not still present here |
| Query ownership/concurrency | Partly documented | Inputs copied, queries read-only; independent construction covered; shared native-handle mutation is not guaranteed |
| Copy/transform full state | Partly implemented | Bytes/copy/cast exist; means/weights reconstruction is weighted observation construction, not lossless state restore |
| Point/tie CDF and quantile endpoints | Existing coverage | Public exact-reference tests; rerun exact sanity cases on candidate |
| NaN/inf input and probes | Mostly covered | Strict frontend rejects invalid training/q; CDF NaN/infinity conventions differ intentionally from original |
| Polars CDF alignment | **Source defect** | `cdf_generic` broadcasts by number of nonempty rows instead of input length |
| Polars missing digest handling | **Source defect** | Lenient CDF parser substitutes empty state; user now selects explicit rejection |
| Polars reductions/cardinality/composition | Partial coverage | Dynamic/composed expressions exist; add one/many/empty planner/runtime, grouped/window/join tests |
| Polars struct validation | Partial coverage | Expanded state fields exist; rigorously test scalar types, metadata consistency and NaN means |
| Modern six-field old struct reads | Implemented with inference | Historical defaults/inference described below; exact consumer artifacts still need internal comparison |
| New struct writes | Schema migration | Extra config/kind fields and f64 weights change both schemas; retain readable six fields, document versioned migration |
| TDIG v1/v2/v3 | Existing coverage | Explicit v1 fractional/config limitations, v3 extensions, precision/kind preservation; malformed aggregate validation needs strengthening |
| Merge configuration | Inconsistent layers | Frontend is strict for populated state but bypasses config for empties; raw Rust/Polars differ; selected new policy must be explicit |
| Empty merge-all | Decision/documentation | Current modern empty is f64/1000/K2/Use; original remains untouched; Java historical empty-array behavior separately noted |
| Java public methods | Partial | Fluent add/merge, weighted add, queries, cast, scale and bytes exist; array-only mergeAll is missing; Java enum lacks K2Norm/delta |
| Java native lifecycle / serialization | Existing bounded coverage | Cleaner/AutoCloseable, byte/stream/object restoration exist; expand multiple objects/closed original/GC checks |
| Native loading | Gaps | Override requires absolute path; no extraction-directory control; resource load failure diagnostics incomplete |
| JNI panic boundary | Gap | No `catch_unwind`; expected errors handled but catchable Rust panic can cross JNI boundary |
| JVM allocator | Gap | Linux global allocator is jemalloc; historical system-allocator integration fix is not incorporated |
| JVM offline/internal authenticated build | Existing bounded coverage | JDK/Gradle URL+SHA selection, exclusive Maven mirror, staged offline tests; actual internal trust/services remain user checks |
| Rust/native toolchain bootstrap | Partial | Rust1.95 pinned; early Cargo/JDK diagnostics, cross-tool selection and source archive wrapper completeness need qualification |
| Installed Python docs/stubs | Gap | Native class has no docstring; no `.pyi` or `py.typed` shipped; aliases hide method signatures |
| Package ownership / upgrade | Historical coverage only | Original package-ownership receipts exist; repeat fresh install and upgrade for uniquely versioned candidate bytes |
| Quality reference/data coverage | Strong bounded evidence, **not acceptance** | 37,114 public real rows / 9 fields, synthetic data, 756 profiles; 10 failures and 13 exact boundary failures remain visible |
| Large workload performance | New measurement required | Prior measurements bind earlier sources; thousands groups/many columns/latency/RSS/bridges/concurrency must be measured anew |
| 100+ CPU and actual Spark cluster | Internal / untested | Local host has 16 physical / 32 logical CPUs; no extrapolated qualification |
| Unique one-source wheel/JAR/manifests | Missing for this candidate | Freeze combined source, build unique version, hash and test the same bytes; separate-branch passes are insufficient |

## Authoritative old format rules

The assessment source's `src/tdigest/wire.rs` decodes TDIG v1 kinds as atomic
only when weight equals one; all other v1 centroids are mixed. V2/v3 encode kind
explicitly. Missing v3 delta extension means max-size mode. F32 full-width mass
uses a v3 flag that older readers may reject. V1 fractional mass and v1/v2
unrepresentable configuration exports are rejected, not rounded or dropped.

The six-field Polars reader in `src/tdigest/codecs.rs` uses its existing K2 /
Use defaults and weight-based kind inference when extended fields are absent.
That historical inference cannot recover true point identity or configuration.
Preserve the decoder rule and test it; do not describe inferred information as
recovered. The older integer-weight `polars_tdigest` schema is outside the
approved legacy engine scope until explicitly requested.

Modern dictionary exports contain readable means/weights plus `wire_b64_v3`.
When present, the binary state is authoritative in the current loader; editing
readable fields does not edit that state. Without binary state, the loader
constructs weighted observations and cannot restore old extrema/statistics/kind.
This distinction needs prominent installed documentation.

## Candidate work and qualification boundary

Stage fixes and new evidence on `codex/consumer-candidate-20261020`, based on the
exact assessment SHA. Keep existing evidence and oracle immutable. Use a unique
development version, produce one source archive plus a Linux wheel and native
JAR, and record all source/build/ABI/artifact identities. Run build/test/lint,
original exact gates, modern hard contracts and quality tests on the combined
source; then exercise those exact installed artifacts. Failed numerical cases
stay failed until corrected, never waived through a golden-file update.

The user performs private 0.1.9/0.1.11 binary regressions, actual Spark executor
qualification, internal publication and deployment. Exact JVM/Scala/Spark and
Kotlin requirements, release namespace/scope, accuracy exclusions or changed
acceptance gates, and representative internal performance tradeoffs remain
sign-off decisions. A candidate with explicit failed/untested rows is reviewable
but is not approved for production.

The freshly rerun bounded Python development-build probes are in `baseline-probes.json`: 19 passed and 3 failed (weighted `from_array` is unsupported, a one-element NumPy query becomes a scalar, and iterable merge is not failure-atomic). This is a source-identified development extension probe, not an installed-wheel qualification.
