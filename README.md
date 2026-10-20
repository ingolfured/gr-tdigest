# tdigest-rs

Java builds support JDK11 and17, including an optional checksum-verified JDK and
Gradle bootstrap from internal URLs or local archives. Configure
`TDIGEST_JDK_URL`, `TDIGEST_GRADLE_URL` and an exclusive
`TDIGEST_MAVEN_REPOSITORY_URL`; use `TDIGEST_OFFLINE=1` with prepared caches.
See [Java setup and offline builds](bindings/java/README.md) for all environment
settings and actual Java11 bytecode/runtime checks.

T-Digest provides a mergeable summary of a distribution, enabling approximate quantiles and CDF with strong tail accuracy.
`gr-tdigest` ships one Rust core with Rust, Python, Polars, and Java surfaces.

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
- Allow scaling of weights and guard against centroid weight overflow.
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

When a query returns an ndarray, it is an independent writable Float64 snapshot with ordinary NumPy ownership and resizing behavior. Centroid means and weights are also independent Float64 snapshots. Scalar results, including the existing one-element NumPy query fast path, retain their current behavior.

The NumPy output conversion measurements and their scope are recorded in [the validation note](compat/performance/numpy-output-20261015/README.md).

Polars digest structs with delta require K2Norm scale and singleton policy Off, as binary digests do. Incompatible structs return a configuration error when read for queries or serialization.

Modern `TDigest.from_array` and `TDigest.from_means_weights` release the Python GIL during Rust construction after copying input into owned vectors. Independent constructors can therefore run in a caller-managed thread pool. Python input conversion still holds the GIL, and the library does not create a separate constructor pool. This change applies to the opt-in `gr_tdigest` interface.

Modern training and updates copy compatible aligned, native-endian contiguous one-dimensional NumPy Float32/Float64 buffers directly into owned Rust vectors. Read-only arrays are supported; strided, unaligned, byte-swapped, scalar and other-dtype inputs retain their generic conversion path. Input storage is neither retained nor mutated.

## Accuracy on recorded data

`make test` now includes fast quality checks using recorded bike-rental and household-energy data, varied distributions and exact weighted references. `make quality-extended` measures full datasets, both precisions, several summary sizes and40-way merges. It retains failing profiles and reports percentile position, value and cumulative-probability errors separately.

The modern engine usually improves percentile position and average value error on these real-data profiles; the original often has the better worst value error. Small summaries and numerical edge cases still have explicit failures. See the [readable findings](compat/quality-20261020/RESULTS.md) and [test methods](tests/quality/README.md). `make quality-regression` protects the recorded outcomes while preserving their failed accuracy status.

The configured JDK also supplies the direct Java compiler/runtime subprocesses used by `make test`; an environment-selected JDK does not require a separate system Java installation.

## Consumer candidate assessment (20 October 2026)

The [source-specific gap assessment](compat/consumer-20261020/ASSESSMENT.md) records the consumer handoff and subsequent decisions. The modern API may migrate to new defaults/dependencies; the original `tdigest_rs` engine remains pinned and numerically identical. Historical validation does not qualify a new combined candidate or a production release.

Modern consumer candidate guides: [Python/NumPy](bindings/python/OBJECT_API.md), [Polars](bindings/python/POLARS_API.md), and [source/build/qualification procedure](compat/consumer-20261020/BUILD.md). This is a development candidate with explicit remaining numerical and internal qualification limits, not a production release.

The [combined candidate report](compat/consumer-20261020/CANDIDATE.md) identifies the exact wheel/JAR source and bytes, passed runtime checks, failed accuracy gates, workload costs and remaining acceptance work. Read it alongside the historical assessment before reviewing or merging the modern stack.
The [current integration status and proposed review story](compat/consumer-20261020/INTEGRATION.md)
supersede that report for ongoing work; [new source-check evidence](compat/consumer-20261020/integration-v2/receipt.json)
is separate from the earlier exact-artifact qualification.

Current integration work makes modern Rust construction checked:
`TDigest::<f64>::builder().max_size(1000).build()?` returns valid state or an
error. The original `tdigest_core` API is unchanged. Existing candidate artifacts
remain bound to the source SHA in their manifest; implementation updates require
a newly identified and qualified artifact set before acceptance. Current work also
preserves exact tiny point masses, rejects Float32 training underflow and retains
the unchanged targets of the numerical boundary checks.

The JVM candidate includes [Scala/sbt and Kotlin/Gradle examples](bindings/java/examples/README.md), with configured repository credentials and offline checks. The coverage report distinguishes build JDK, runtime JDKs and actual cluster checks.

## Measured summary-size tradeoffs

`make quality-budgets` separately measures4140capacity/scale/data/merge profiles,
including explicit200/1000sizes and additional deterministic shapes. It is a
slower diagnostic and returns failure for the observed accuracy failures. Small
summaries remain a numerical acceptance issue; changing the scale does not
universally fix them. The default remainsK2/1000 and the original engine is
unchanged. See the [accuracy investigation and recommendation](compat/consumer-20261020/accuracy-investigation/REPORT.md).

JVM builds accept a selected native file with any basename through
`TDIGEST_NATIVE_FILE`. Packaging renames it to the native loader's canonical
resource path, removes stale generated payloads and verifies the finished JAR.
CI exercises a renamed override and runs that JAR outside the checkout.
