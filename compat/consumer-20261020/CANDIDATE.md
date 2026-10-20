# Historical version001 consumer candidate — 20 October 2026

This report is retained for source `a9c368f`, version001. See the
[current version003 artifact report](accuracy-investigation/ARTIFACTS.md) and
[integration status](INTEGRATION.md) for current fixes, measurements and limits.

**A reviewable candidate is ready. Acceptance and production release remain blocked.** The combined source is `a9c368f2e766e602e2b0269486de089297e66de1`, compared with `1451f94c4d809dd43d851b91c7c777ad3c3ff2f3`. The development version is `0.2.5-dev2026102001` on the JVM and `0.2.5.dev2026102001` in Python. Later evidence-only commits do not change the artifact source revision.

The exact wheel and JAR passed the installation, language and Linux compatibility checks below. The wider accuracy gate failed: ten existing approximation profiles still fail, and seven profiles have new regressions in fifteen measured metrics. No frozen tolerance or reference fixture was regenerated.

## Changes that are ready to review

Modern weighted construction now consumes the supplied weights. Multi-item updates prepare independent state and commit only after validation. Weight totals stay Float64 with Float32 means; copying, serialization, scaling and precision conversion preserve full state or reject an unrepresentable operation. Small-capacity compression no longer drops represented mass.

Python now documents its new capacity-1,000 API, scalar/list/tuple/one-dimensional Float64 NumPy results, and explicit rejection of multidimensional query arrays. The wheel includes installed docstrings, type stubs and typing metadata. The selected modern stack is Polars 2.0.0; this is the approved migration, not qualification of Polars 1.38.1.

Polars rejects missing digests and malformed state. CDF broadcasting follows input shape. Binary-column decoding requires explicit `f32` or `f64`, so lazy planning matches execution. Reductions, row alignment and full-frame schema migration are documented separately from the Python object API.

The JVM adds array merge-all, weighted builders and modern delta/K2Norm settings. A real restored-handle ownership bug was fixed: garbage collection of a temporary object could previously free a restored digest. The JAR uses the system allocator and supports controlled extraction, relative native overrides and payload diagnostics.

The original `tdigest_rs`/`tdigest_core` implementation, upstream wrapper, original tests and output fixtures are unchanged. Modern delta configuration does not select that engine. Original saved-state restoration retains its historical limitations.

## What passed

| Check | Result and scope |
|---|---|
| Combined source | `make build`, `make test`, `make lint`, warnings-denied Rust documentation |
| Installed wheel | Ten full runtime cells:208 original-compatibility +222 modern cases each; two Polars-free cells:208 +120 each |
| Source archive | Rebuilt on Python 3.12,3.14,3.15; each independently passed 430 installed cases. These are separate test artifacts. |
| Python Linux floor |430 cases passed with the canonical wheel on actual glibc 2.28 in Rocky8, network disabled |
| JVM runtimes | Same canonical JAR passed 18 JUnit cases on Java 11,17,21 and Java 11/glibc 2.17 in CentOS7 |
| JVM language consumers | Scala 2.12.16/sbt 1.12.11 and Kotlin 2.4.21/Gradle 8.14.5, build JDK 17; both also ran offline on Java 11 |
| Repository selection | Authenticated loopback fixture succeeded; bad credentials in a fresh cache failed; offline checks succeeded after stopping that fixture |
| JVM state and loading | Multiple object/stream restorations survive original close and collection; relative override/extraction controls and expected failures checked |
| Bounded JVM memory |32,768 digests,4,096 simultaneously live,4,194,304 observations under64 MiB Java heap; peak process RSS66,716 KiB |
| Python data interchange | Python 3.12.15 / NumPy 2.5.4 / Polars 2.0.0 / PyArrow 24.0.0 full-struct Arrow and Parquet round-trips; Arrow also executed on glibc 2.28 |
| Actual installed API | Native docs/signatures, shipped stubs/typing metadata, external mypy client and import ownership passed |
| Installation migration | Fresh install, public 0.2.4 upgrade/rollback, reinstall and removal of both owned module paths passed |
| Cross-language state | Exact wheel/JAR f32/f64 weighted toy data produced matching bytes and query results in both directions |

The runtime/source matrix executed6,246 passing test cases across those environments; these include repeated cases, not6,246 unique requirements. The source test suite still has three pre-existing ignored slow Rust story/proof tests. The separate JNI-feature unit run passed 62 tests, including the panic guard, with those three ignored. No live-JVM panic-injection hook was added.

Exact NumPy support cells follow. Broader NumPy 2.x combinations remain untested.

| Python | NumPy | Polars |
|---|---|---|
| 3.10.22 | 1.26.4 | 2.0.0 |
| 3.10.22 | 2.2.6 | 2.0.0 |
| 3.11.16 | 1.26.4 | 2.0.0 |
| 3.11.16 | 2.3.4 | 2.0.0 |
| 3.12.15 | 1.26.4 | 2.0.0 |
| 3.12.15 | 2.3.4 | 2.0.0 |
| 3.12.15 | 2.5.4 | 2.0.0 |
| 3.13.16 | 2.3.4 | 2.0.0 |
| 3.14.8 | 2.5.4 | 2.0.0 |
| 3.15.0 | 2.5.4 | 2.0.0 |
| 3.10.22 | 1.26.4 | not installed |
| 3.15.0 | 2.5.4 | not installed |

## Accuracy: improvements and remaining failures

| Study | Result |
|---|---|
| Fast quality |90 reference/weighted checks and 42 comparison profiles pass |
| Extended quality |756 profiles;10 existing accuracy-envelope failures;15 metric regressions across7 real-data profiles |
| Exact boundaries |14 pass;1 subnormal weighted-product case is explicitly unsupported. Previous baseline:2 pass and 13 fail. |

Rejecting an unrepresentable positive product is safer than returning malformed state, but it is not a numerical pass or approval to exclude that domain. A separate advanced raw Rust builder can still seed unchecked state; its caller must validate before queries. A fallible restoration/build route remains work.

The real-data study covers all 37,114 rows and nine fields from the public bike-rental and household-energy datasets. It varies Float32/Float64 means, capacities32/128/512, input order and direct/sequential/balanced40-way merging. Storage comparisons use recorded centroid-field bytes, not total process memory.

| Real-data comparison,324 matched profiles | Modern better | Equal | Original better |
|---|---:|---:|---:|
| Average position error | 279 | 0 | 45 |
| Worst position error | 268 | 21 | 35 |
| Median position error | 191 | 109 | 24 |
| Worst tail position error | 294 | 17 | 13 |
| Average value error | 260 | 0 | 64 |
| Worst value error | 119 | 0 | 205 |

The modern engine often improves position error; the original often has the smaller worst value error. CDF errors are measured separately against exact midpoint probabilities. The original API has no CDF method, so no original CDF result is invented. All poor profiles remain in the totals. Four original Float32 profiles also exceed the weighted-contribution diagnostic; original arithmetic was not changed.

The seven new failing regression profiles are listed with all fifteen metrics and unchanged limits in [quality-differences.json](evidence/quality-differences.json). Known-failure matching is not numerical acceptance.

## Workload comparison

The baseline was freshly built from exact revision 1451f94 with the same pinned tools and Python glibc 2.28 target. This is a controlled public source comparison, not either unavailable internal 0.1.9/0.1.11 binary. Each scenario ran in a fresh process in baseline/candidate/candidate/baseline order. No heavy builds overlapped timings; light orchestration continued.

This host has16 physical/32 logical CPUs and one NUMA node; affinity exposes all 32 logical CPUs. The CPU quota interface was unavailable. Pools were fixed to Rayon 1, Polars 4 and BLAS/OpenMP 1. These are descriptive local measurements, not100-plus-CPU qualification or an agreed performance gate.

| Scenario / operation | Baseline median,ms | Candidate median,ms | Time change; positive means slower |
|---|---:|---:|---:|
| bridge / roundtrip | 3.395 | 3.411 | +0.46% |
| columns128 / cdf | 376.990 | 390.450 | +3.57% |
| columns128 / fragmented_cdf | 2183.439 | 2182.757 | -0.03% |
| columns128 / median | 1559.845 | 1568.948 | +0.58% |
| columns128 / training | 1433.571 | 1393.494 | -2.80% |
| columns16 / cdf | 47.890 | 48.242 | +0.74% |
| columns16 / fragmented_cdf | 278.923 | 281.381 | +0.88% |
| columns16 / median | 240.372 | 239.946 | -0.18% |
| columns16 / training | 171.709 | 172.051 | +0.20% |
| columns64 / cdf | 190.543 | 192.017 | +0.77% |
| columns64 / fragmented_cdf | 1086.685 | 1086.878 | +0.02% |
| columns64 / median | 801.553 | 805.175 | +0.45% |
| columns64 / training | 728.247 | 721.435 | -0.94% |
| fanin40 / merge | 1531.434 | 1600.915 | +4.54% |
| stream / construction | 37.435 | 42.444 | +13.38% |

Column cases use2,048 groups and 32 observations per group. Fan-in uses 1,024 groups,16 digest columns and 40 shards. Streaming constructs1,048,576 values in16,384-value chunks. The bridge case includes2,048 digests, bytes, allocation/copy and NumPy query results. Training/merge/stream medians pool six timed samples per artifact; query/bridge medians pool24. P95/P99 and per-process peak-memory ranges are in [performance-summary.json](evidence/performance-summary.json); small-sample tail estimates are descriptive.

Streaming construction took 13.4% longer and 40-way merging4.5% longer. Most column query timings were close, with 128-column CDF3.6% slower. These costs remain visible; no broad speedup claim is made.

| Independent construction workers | Candidate values/second | Throughput relative to1 worker |
|---|---:|---:|
| 1 | 43,211,529 | 1.00x |
| 2 | 85,821,271 | 1.99x |
| 4 | 162,982,508 | 3.77x |
| 8 | 266,720,643 | 6.17x |
| 16 | 401,386,703 | 9.29x |

Each concurrency batch submits64 independent digests of 32,768 values. There are six timed batches per artifact/worker setting. Queue-inclusive task P95/P99 remain separate from batch throughput. This does not support concurrent mutation of a single digest.

## Artifact and source identity

Full hashes, tool versions, targets, allocator and ABI checks are in [artifact-manifest.json](artifact-manifest.json). The canonical artifacts are in this task’s `candidate-20261020/artifacts/` directory and delivery bundle. They have not been published to package repositories.

| Artifact | SHA-256 prefix |
|---|---|
| `gr-tdigest-0.2.5-dev2026102001.jar` | `056da58e404485ee` |
| `gr-tdigest-a9c368f-source.tar.gz` | `7de3c671228498bc` |
| `gr_tdigest-0.2.5.dev2026102001-cp310-abi3-manylinux_2_28_x86_64.whl` | `6b1056894004da81` |
| `gr_tdigest-0.2.5.dev2026102001.tar.gz` | `d1dd15379f797616` |
| `libgr_tdigest-jni-glibc 217.so` | `3e437a2657e256f9` |
| `libgr_tdigest-python-glibc 228.so` | `5b14c191f75bae49` |

The canonical wheel and JAR are the bytes that were installed and tested. Source-rebuild wheels have their own hashes and are evidence variants; they do not replace these artifacts under the same version. Reproduction starts by checking out the artifact source SHA, then following [BUILD.md](BUILD.md). Evidence-only report commits may follow it.

## PR review order and remaining work

Review the original implementation/wrapper in [#7](https://github.com/ingolfured/gr-tdigest/pull/7), the oracle/fixtures/source guards in [#8](https://github.com/ingolfured/gr-tdigest/pull/8), and the small setup fix in [#10](https://github.com/ingolfured/gr-tdigest/pull/10). Their overlapping packaging portions still need the combined view. No current stacked PR is recommended as a definitely safe independent merge into master.

Do not redo the work or discard history. Once the modern failures are resolved, restack preserved commits into original compatibility, modern correctness, Python/Polars, JVM and qualification evidence reviews. Keep experimental CDF [#13](https://github.com/ingolfured/gr-tdigest/pull/13) separate. No existing PR was merged, rewritten or closed for this candidate.

Before release:

- Resolve the seven new regression profiles and ten existing approximation failures, or obtain explicit approval for any changed accuracy/domain limits. The subnormal exclusion is not approved.
- Close the unchecked raw Rust builder/restoration gap.
- Run the exact internal 0.1.9/0.1.11 binary/private-fixture comparisons. Historical JVM 0.1.16 provenance remains unrecovered.
- Confirm the exact Spark/Scala/serializer/checkpoint/executor matrix and exercise real driver/executor serialization, shuffles and broadcasts. Spark 3.5/Scala 2.12 remains provisional.
- Confirm the minimum Kotlin compiler, real internal repositories/trust, representative-server tradeoffs, integration head, package ownership and release version.

Only ignored failed setup outputs in the task-created worktree were removed. Historical branches, original oracle, measured artifacts and staged offline inputs remain. The external SSD source/evidence comparison remains on the [pending TODO](TODO.md); its source, branches and experiment material were not cleaned.

[Requirement-by-requirement coverage](coverage.json) distinguishes PASS, FAIL and UNTESTED. Missing private data is an internal qualification boundary, not invented external evidence.
