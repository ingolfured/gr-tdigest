# tdigest-rs
T-Digest provides a mergeable summary of a distribution, enabling approximate quantiles and CDF with strong tail accuracy.
`gr-tdigest` ships one Rust core with Rust, Python, Polars, and Java surfaces.

## ✨ Features
- 🦀 Single Rust core shared across Rust, Polars, Python, and Java.
- 🚀 Mergeable digests for large and streaming data with fast union and consistent accuracy.
- 🔁 Cross-surface coherence: consistent, verified behavior across all bindings.
- ⚡ Quantile, CDF, and median with optimized evaluation loops using half-weight bracketing and singleton-aware interpolation.
- 🧠 Heap-stream k-way digest merge in Rust core for lower peak memory on large digest unions.
- 🧵 Streaming two-way raw-ingest merge path in Rust core (centroids + values) to avoid extra merge buffers.
- 🧊 TDigest precision as `f64` or `f32`, with `auto` precision selection where supported.
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

## Publishing
- Publishing and release workflows are documented in `PUBLISH.md`.

## Versions and compatibility
- Rust: stable (edition 2021)
- Python: CPython 3.12 (built with maturin)
- Polars: 1.x line (Python)

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
