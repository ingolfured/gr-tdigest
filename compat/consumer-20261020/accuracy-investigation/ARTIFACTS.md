# Current review artifacts and checks

**The integration bugs are fixed, and the accuracy tradeoffs are measured. This
is a review candidate, not a finished or approved release.** The remaining
small-summary accuracy failures prevent a final-product claim. No accuracy target
was relaxed, no reference was regenerated, and no trial compressor was adopted.

The [accuracy report](REPORT.md) explains the numerical recommendation and shows
where the modern engine wins and loses against the original. Keep K2 and the
approved default capacity 1,000. Do not merge the existing PR stack yet; agree the
remaining accuracy contract or improve compression safely, then split the
combined product into reviewable chapters.

## One identified source, one set of review bytes

Source: `60b295140d5eed81df24a8aed05f89767646c7d8`.
Version: JVM/Rust `0.2.5-dev2026102003`; Python `0.2.5.dev2026102003`.
Later documentation/evidence commits do not change that artifact source.

The [artifact manifest](artifacts-v3.json) records the source tree, every hash,
build tools, Linux floors, exact runtime cells and limits. The separate
[coverage report](coverage-v3.json) labels passed, failed and untested requirements.
Raw results, logs and measured harnesses are preserved under `evidence/` and
`tools/`, with hashes in [manifest.json](manifest.json). The supplied binaries are
retained in the task's `accuracy-20261020/artifacts-v3/` directory.

The Python wheel and JVM JAR have different native payloads built from this same
source. Each payload has its own hash. Source rebuilds are separately identified
variants; they do not replace the canonical review wheel.

## Fixed and verified

- CI was still expecting 222 modern cases after three tests were added. It now
  requires all 225, derives the combined total correctly, exposes failure reasons
  and retains runtime receipts.
- A custom build-time native filename produced a JAR the loader could not use.
  Packaging now assigns the canonical name, removes stale generated payloads and
  checks the finished JAR. A renamed-input standalone smoke check runs in CI.
- Installed isolated-mode quality runs now load their explicitly named helper.
  A nonzero exit counts as a measured accuracy failure only after the full report,
  unchanged inputs and loaded-native identity have been verified.
- Earlier core fixes remain: seven new accuracy regressions removed, all 15 exact
  boundary controls passing, checked Rust construction and failure-atomic updates.
  The original `tdigest_rs`/`tdigest_core` source, wrapper, oracle and fixtures remain
  unchanged from their pinned compatibility implementation.

## Checks on these exact artifacts

| Check | Result and tested scope |
|---|---|
| Combined source | `make build`, `make test`, `make lint`, warnings-denied Rust documentation, all-target compilation passed |
| Original compatibility | Pinned-source guard; original Rust tests; unchanged 91 upstream and 70 exact-output cases; 208 installed compatibility cases per full cell passed |
| Same Python wheel | Ten full runtime cells, two without Polars; three additional source-archive rebuilds passed |
| Python Linux floor | 433 installed tests on actual glibc 2.28, with network disabled; symbol requirements checked |
| Same JVM JAR | 20 JUnit cases each on Java 11, 17 and 21; another 20 on actual glibc 2.17 / Java 11 with network disabled |
| JVM packaging/loading | Class version 55; correct native payload; custom native input name; bundled, relative override, extraction directory and useful failures passed |
| Scala and Kotlin | Scala 2.12.16 / sbt 1.12.11; Kotlin 2.4.21 / Gradle 8.14.5; JDK 17 build and Java 11 execution passed |
| Restricted repositories | Authenticated staged-only local Maven fixture; wrong credentials failed with HTTP 401 in fresh cache; server-stopped offline reruns passed |
| Python installation | Installed docs/signatures/stubs, external type-checking example, public-baseline upgrade/rollback, import ownership and removal passed |
| State interchange | Float32/Float64 exact toy wheel/JAR bytes and queries matched; PyArrow 24 / NumPy 2.5.4 / Polars 2 full-struct IPC/Parquet on glibc 2.28 passed |
| Bounded JVM memory | 32,768 digests, 4,096 live at once, 4,194,304 observations, 64 MiB Java heap; peak whole-process memory 69,668 KiB |
| Exact controls | All 15 passed unchanged expected values |
| Frozen approximate suite | 756 profiles; zero new metric regressions, **ten accuracy failures remain** |
| Wider diagnostic | 4,140 profiles; **114 failed rank/CDF profiles**, zero construction/state/query exceptions; installed wheel reproduced all scores |

The runtime/source-build matrix contains **6,285 passing test executions**, including
repeats across environments. They are not 6,285 unique requirements. The ordinary
Rust suite still has three pre-existing ignored slow proof/story tests; the JNI
unit run has 62 passes and those same three ignored. No live-JVM panic-injection
hook or real Spark execution is claimed.

## Exact Python combinations

Full cells use Polars 2.0.0. These are tested combinations, not a blanket NumPy 2.x
guarantee. Patch versions and dependency versions appear in the raw matrix.

| Python | NumPy | Cases per full cell |
|---|---|---:|
| 3.10 | 1.26.4 | 433 |
| 3.10 | 2.2.6 | 433 |
| 3.11 | 1.26.4 | 433 |
| 3.11 | 2.3.4 | 433 |
| 3.12 | 1.26.4 | 433 |
| 3.12 | 2.3.4 | 433 |
| 3.12 | 2.5.4 | 433 |
| 3.13 | 2.3.4 | 433 |
| 3.14 | 2.5.4 | 433 |
| 3.15 | 2.5.4 | 433 |

The Polars-free cells are Python 3.10 / NumPy 1.26.4 and Python 3.15 / NumPy 2.5.4,
328 cases each. Rebuilding the verified source archive on Python 3.12, 3.14 and
3.15 adds 433 passing cases each. All three rebuilt native hashes match the
canonical wheel's native payload; whole-wheel hashes differ and remain recorded.

## Current workload comparison

These new results use the canonical version003 wheel against a separately
identified wheel built from public starting source
`1451f94c4d809dd43d851b91c7c777ad3c3ff2f3`. They do not substitute for the private
0.1.9/0.1.11 binary comparisons. Both environments use NumPy 2.5.4 and Polars 2.0.0.

Fresh processes run in baseline/candidate/candidate/baseline order. Polars uses
four threads; Rayon, BLAS and OpenMP use one. No task-owned heavy build or other
timing overlaps these measurements. Ordinary desktop services remain active.
The host has 16 physical / 32 logical cores and one NUMA node, with affinity
0–31. All available CPU-bandwidth limits on the process/ancestor chain read
`max 100000`; no finite cgroup quota is observed there. This does not reserve
those CPUs against other desktop work.

Positive time changes mean slower; negative changes mean faster. These are
measured tradeoffs, with no approved speed or memory acceptance threshold.

| Operation | Baseline median ms | Candidate median ms | Time change |
|---|---:|---:|---:|
| 16 columns: build | 186.087 | 190.415 | +2.33% |
| 64 columns: repeated CDF | 188.098 | 191.054 | +1.57% |
| 128 columns: build | 1487.586 | 1509.279 | +1.46% |
| 128 columns: repeated CDF | 379.533 | 380.685 | +0.30% |
| 128 columns: repeated median | 1555.885 | 1557.387 | +0.10% |
| 128 columns: fragmented CDF | 2191.849 | 2184.712 | -0.33% |
| 40-way group merge | 1533.134 | 1623.737 | +5.91% |
| Bounded-memory streaming | 37.811 | 42.465 | +12.31% |
| Saved-state + NumPy bridge | 3.348 | 3.389 | +1.21% |

Column workloads use 2,048 groups, 32 observations per group and explicit capacity
200. Merge uses 1,024 groups, 16 columns and 40 input summaries per group.
Streaming processes 1,048,576 values in chunks of 16,384 at capacity 1,000.

Independent-digest construction scales as follows. Units are millions of input
values per second; each worker owns separate state.

| Workers | Baseline million values/s | Candidate million values/s |
|---:|---:|---:|
| 1 | 49.1 | 43.1 |
| 2 | 98.0 | 85.2 |
| 4 | 184.7 | 161.6 |
| 8 | 331.3 | 300.0 |
| 16 | 420.6 | 390.6 |

Full raw timings include throughput, sampled p95/p99 completion latency and peak
process memory. Training/merge/stream have six pooled measurements; repeated
queries/bridge have 24. These small-sample tails are descriptive, not population
latency guarantees. No 100-plus-CPU or representative-server qualification follows
from this host.

## Remaining acceptance work

The ten frozen and 114 wider failed accuracy profiles remain visible. Larger
summaries help on these inputs, but do not establish a universal error bound.
Modern wins many rank/tail comparisons; the original often wins worst value error.
Some extreme mixed-observation weighted products remain explicitly unsupported
when they cannot be represented. No new domain exclusion has been approved.

Private consumer regressions, the historical working 0.1.16 JAR receipt, the exact
Spark/serializer/checkpoint/executor configuration, real internal repositories
and representative servers remain owner-run checks. Spark 3.5 / Scala 2.12 is
provisional. The minimum required Kotlin compiler still needs confirmation.
The external SSD source/evidence comparison remains a pending TODO; its branches
and experiment material have been preserved.

Historical version001 and failed version002 bytes, receipts and measurements are
retained. Their successes are not substituted for current-artifact checks.
No branch was merged, release published or historical source removed.

## Reproduce and verify

Use source `60b2951` and the exact hashed artifacts. See [the build guide](../BUILD.md)
for staged toolchain/repository configuration and separate component, installed
and distributed checks. The source archive includes the full Gradle wrapper.

```sh
make build
make test
make lint
python3 scripts/test_python_support.py --wheel /path/to/candidate.whl \
  --sdist /path/to/candidate.tar.gz --output-dir /path/to/new-matrix
make quality-regression  # regression PASS can coexist with accuracy FAIL
make quality-budgets     # intentionally exits nonzero for observed accuracy losses
```

Artifacts were built with Rust/Cargo 1.95.0, Zig 0.12.1, maturin 1.15.0 and Gradle
8.14.5. Build JDK was 17.0.20.1+1; runtime checks used Java 11/17/21. Python retains
the manylinux/glibc 2.28 floor; JNI retains the separate glibc 2.17 floor. JNI uses
the system allocator. Native required symbols top out at glibc 2.28 and 2.16,
respectively, and both advertised floors were tested by actual execution.

## Exact artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `gr-tdigest-0.2.5-dev2026102003.jar` | `c9d8841f6beadd65c68325ac82dd0e0faaaa4ce668304b7a5179181433f77122` |
| `gr-tdigest-source-60b2951.tar.gz` | `6fb4aa3f1b5928565c614cae0fd4dcfd6192aee0fd7d6a4f04d3be0a7be2dc0f` |
| `gr_tdigest-0.2.5.dev2026102003-cp310-abi3-manylinux_2_28_x86_64.whl` | `f142db78ea211c1fdfff690d6976e8d316a89e5e66ce8f2be1a18ad67ee1151f` |
| `gr_tdigest-0.2.5.dev2026102003.tar.gz` | `2c339d67851dfa4225333ee836badbfd6096a9eb080da3f28eac72887b9fa3a6` |
| `libgr_tdigest-jni-glibc217.so` | `68b81960d52e11be245049a359e2e0787274dbe5595a07b9e5f11b5369f0047d` |
| `libgr_tdigest-python-glibc228.so` | `f89b96ebf0d72adee3699599416a1c4e0daeb988921502dfaee6797052bdada6` |
