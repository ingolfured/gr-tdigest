# gr-tdigest

Fast, mergeable distribution summaries for **Python/NumPy, Polars and the JVM**,
powered by one Rust core. Estimate percentiles, medians and cumulative
probabilities without retaining every observation. Weighted data, streaming
updates and shared saved state work across the modern interfaces.

The separate original-compatible `tdigest_rs` interface keeps the upstream API
and calculations unchanged.

## ✨ Features
- 🦀 Single Rust core shared across Rust, Polars, Python, and Java.
- 🚀 Mergeable digests for large and streaming data with fast union and consistent accuracy.
- 🔁 Cross-surface coherence: consistent, verified behavior across all bindings.
- ⚡ Quantile, CDF, and median with optimized evaluation loops using half-weight bracketing and singleton-aware interpolation.
- 🧠 Heap-stream k-way digest merge in Rust core for lower peak memory on large digest unions.
- 🧵 Streaming two-way raw-ingest merge path in Rust core (centroids + values) to avoid extra merge buffers.
- 🧊 Mean/value precision as `f64` or `f32`, with `auto` selection where supported; modern centroid weights stay `f64` in both modes to preserve accumulated mass.
- ⚖️ Weighted ingest across Rust/Python/Polars/Java (`add_weighted`, `add_weighted_values`, Java weighted adds).
- 🔄 Explicit precision casting across surfaces (`cast_precision` / `castPrecision`).
- 📦 TDIG v3 wire default (flags + header length + precision code + checksum), with v1/v2 decode compatibility.
- 🧭 Explicit wire-version encode controls (`to_bytes(version=1|2|3)`, `toBytes(version)`).
- 🖥️ Rust CLI subcommands (`build`, `quantile`, `cdf`, `median`) with `text|csv|json|ndjson` ingestion.
- 🎚️ Scale families: `Quad`, `K1`, `K2`, `K3`.
- 🔩 Singleton handling policy: edge-precision (`keep N`), respect singletons, or uniform merge.

## Measured accuracy

**98.7% lower average percentile-position error at the default settings** across
396 recorded-data and deterministic profiles; **99.6% lower on recorded bike-rental
and household-energy data**. Modern also improved the worst tested position
error in 378 profiles, tied in 18, and lost in none.

![Measured default accuracy](compat/accuracy-20261021/default-accuracy.svg)

These measurements compare original `delta=300` with modern capacity `1000`,
using the same observations and precision. Summary sizes differ. In the separate
comparable-storage comparison, modern improved average position error in
**213 of 234 profiles** and tail position error in **223**, with 11 tail ties
and no tail losses. Numerical value errors have different tradeoffs.
See [all scores, storage costs and explanations](tests/quality/ACCURACY.md).

## Examples

**Python**
```python
import gr_tdigest as td

d = td.TDigest.from_array([0, 1, 2, 3], max_size=100, scale="k2")
print("p50 =", d.quantile(0.5))
print("cdf =", d.cdf([0.0, 1.5, 3.0]))
```

**Polars**
```python
import polars as pl
from gr_tdigest import tdigest, quantile

out = (
    pl.DataFrame({"g": ["a"] * 5, "x": [0, 1, 2, 3, 4]})
    .lazy()
    .group_by("g")
    .agg(tdigest("x", max_size=100, scale="k2").alias("td"))
    .select(quantile("td", 0.5))
    .collect()
)
print(out)
```

**Rust CLI**
```bash
# 1) Load numbers from CSV and save a digest
# numbers.csv
# value
# 0
# 1
# 2
# 3
target/release/tdigest build \
  --input numbers.csv \
  --input-format csv \
  --input-column value \
  --to-digest model.tdig

# 2) Read the digest and query quantiles
target/release/tdigest quantile \
  --from-digest model.tdig \
  --p 0.5,0.9,0.99 \
  --no-header
```

**Java (AutoCloseable)**
```java
import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;

public class Example {
  public static void main(String[] args) {
    try (TDigest digest = TDigest.builder()
        .maxSize(100)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[]{0, 1, 2, 3})) {
      double p50 = digest.quantile(0.5);
      double[] cdf = digest.cdf(new double[]{0.0, 1.5, 3.0});
      System.out.println(p50 + " " + cdf.length);
    }
  }
}
```

## Project layout
```text
├── src/                                  # Rust core, CLI, algorithm modules
│   ├── bin/                              # tdigest CLI
│   └── tdigest/                          # Core T-Digest implementation
├── bindings/
│   ├── python/                           # Python package + tests
│   └── java/                             # Java API (Gradle) + JNI bridge
├── integration/
│   └── api_coherence/                    # Cross-surface contract tests
├── crates/
│   └── testdata/                         # Fixtures
└── dist/                                 # Built artifacts after release
```

## Quick Development Start
```bash
make setup
make build
make test
```

Setup supports linked Git worktrees. If the checkout path contains spaces,
set `CARGO_TARGET_DIR` to a build directory without spaces: the bundled
jemalloc build rejects spaces in its installation prefix. The CLI tests and Java
packager respect `CARGO_TARGET_DIR` directly.

## Publishing
- Publishing and release workflows are documented in `PUBLISH.md`.

## Versions and compatibility
- Weighted medians use centroid mass. TDIG v3 and Polars round-trips retain compression configuration and centroid kind.
- Delta-mode export requires TDIG v3; older formats cannot retain delta.
- Rust: 1.95.0 (pinned; edition 2021)
- Python: CPython 3.10–3.15 (ABI3 wheels built with maturin); development tools default to 3.12
- NumPy: 1.26.4 or the 2.x line; tested combinations are recorded in `scripts/test_python_support.py`
- Polars: optional; install `gr-tdigest[polars]` for the tested 2.0.0 integration

Existing G-Research applications keep `from tdigest_rs import TDigest` and
`TDigest.from_array(values, delta=300)`. This interface uses the unmodified
original numerical implementation and Python wrapper, including float32
arithmetic, uint32 weights, and the original copy/pickle behavior. The new engine
is opt-in through `from gr_tdigest import TDigest`; using `delta` there does not
promise the same numbers. See [source provenance](src/upstream/PROVENANCE.md).

The unchanged upstream Python suite runs in `make test` and Linux CI. `make python-support-test` builds one release Linux wheel and tests it outside the checkout on Python 3.10–3.15 (both NumPy lines on 3.10–3.12), plus source-archive rebuilds on 3.12, 3.14 and 3.15. See [legacy compatibility](compat/gr_tdigest_vs_tdigest_rs.md).

## Changelog
- See `CHANGELOG.md`.

## Future improvements
- Auto suggest a scaling function based on distribution.

## Community
- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Issues: https://github.com/ingolfured/gr-tdigest/issues

## License
- Apache-2.0


### Original Rust crate

`tdigest_core::TDigest<f32>` and `TDigest<f64>` retain the original G-Research
API and arithmetic, including public `means` and `u32` `weights` vectors. The
workspace package lives at `src/upstream`; Git dependencies select the package
by its original name, while local path dependencies must point to that directory.
The repository root remains the separate modern `gr-tdigest` package. Registry
publication requires publishing the matching `tdigest_core` version first; this
workspace layout does not itself publish either package.

When a query returns an ndarray, it is an independent writable Float64 snapshot with ordinary NumPy ownership and resizing behavior. Centroid means and weights are also independent Float64 snapshots. One-dimensional NumPy probes retain an array result, including one element; zero-dimensional probes return scalars. Multidimensional probes are rejected.

The NumPy output conversion measurements and their scope are recorded in [the validation note](https://github.com/ingolfured/gr-tdigest/blob/5801881075c6e31fec746c8c62c6a25782b6c13f/compat/performance/numpy-output-20261015/README.md).

Polars digest structs with delta require K2Norm scale and singleton policy Off, as binary digests do. Incompatible structs return a configuration error when read for queries or serialization.

Modern `TDigest.from_array` and `TDigest.from_means_weights` release the Python GIL during Rust construction after copying input into owned vectors. Independent constructors can therefore run in a caller-managed thread pool. Python input conversion still holds the GIL, and the library does not create a separate constructor pool. This change applies to the opt-in `gr_tdigest` interface.

Modern training and updates copy compatible aligned, native-endian contiguous one-dimensional NumPy Float32/Float64 buffers directly into owned Rust vectors. Read-only arrays are supported; strided, unaligned, byte-swapped, scalar and other-dtype inputs retain their generic conversion path. Input storage is neither retained nor mutated.

## Quality checks and build support

`make test` includes fast recorded-data, weighted and exact-state checks.
Extended suites cover full datasets, both precisions, repeated merges and small
summary stress cases. [The quality guide](tests/quality/README.md) separates hard
correctness, frozen regression checks and descriptive approximation scores.
An explained rank/value/storage tradeoff does not require every metric to win
on every dataset; unexpected accuracy loss and broken state still need fixing.

Java builds support JDK 11 and 17, including checksum-verified JDK/Gradle
bootstrap from internal URLs or local archives. Configure `TDIGEST_JDK_URL`,
`TDIGEST_GRADLE_URL` and an exclusive `TDIGEST_MAVEN_REPOSITORY_URL`; use
`TDIGEST_OFFLINE=1` with prepared caches. See the
[Java setup guide](bindings/java/README.md) for environment settings and runtime
checks. The same prebuilt candidate JAR has Java 11/17/21 checks.

The configured JDK also supplies the direct Java compiler/runtime subprocesses used by `make test`; an environment-selected JDK does not require a separate system Java installation.

## Consumer APIs and delivery

Use the [Python/NumPy guide](bindings/python/OBJECT_API.md), [Polars guide](bindings/python/POLARS_API.md), and [Java build and runtime guide](bindings/java/README.md). Scala/sbt, Kotlin/Gradle and a separate-executor Spark check are in [the JVM examples](bindings/java/examples/README.md).

Private consumer artifacts and representative cluster/server checks remain
[owner-run](tests/quality/OWNER_CHECKS.md), with scripts to retain exact artifact,
host and workload evidence inside the consuming environment.

Modern Rust construction returns a checked result, for example `TDigest::<f64>::builder().max_size(1000).build()?`. Modern defaults use capacity 1,000; the separate original `tdigest_rs` and `tdigest_core` interfaces retain their pinned API and calculations.

The upcoming release is `0.2.5-rc.1` (`0.2.5rc1` for Python). Use the guides above for installation and API examples.

## Measured summary-size tradeoffs

`make quality-budgets` separately measures4140capacity/scale/data/merge profiles,
including explicit200/1000sizes and additional deterministic shapes. It is a
slower diagnostic and returns failure for the observed accuracy failures. Small
summary limits remain documented approximation tradeoffs; changing the scale
moves accuracy between regions rather than universally improving every score. The default remainsK2/1000 and the original engine is
unchanged. See the [accuracy investigation and recommendation](https://github.com/ingolfured/gr-tdigest/blob/5801881075c6e31fec746c8c62c6a25782b6c13f/compat/consumer-20261020/accuracy-investigation/REPORT.md).

JVM builds accept a selected native file with any basename through
`TDIGEST_NATIVE_FILE`. Packaging renames it to the native loader's canonical
resource path, removes stale generated payloads and verifies the finished JAR.
CI exercises a renamed override and runs that JAR outside the checkout.

Capacity tuning counts trial clusters without building discarded centroids; final compression and public numerical settings retain their existing behavior.
