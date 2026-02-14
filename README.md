# 🌀 tdigest-rs
T-Digest provides a mergeable summary of a distribution, enabling **approximate quantiles and CDF** with strong tail accuracy. **tdigest-rs** delivers a production-ready Rust core with Python and Polars APIs plus Java (JNI), combining high performance, stable accuracy, and minimal memory overhead.



## ✨ Features
- 🦀 Single Rust core shared across Rust, Polars, Python, and Java
- 🚀 Mergeable digests for large / streaming data — fast union with consistent accuracy and guaranteed unique centroids
- 🔁 Cross-surface coherence: Consistent, verified behavior across all bindings
- ⚡ Quantile & CDF — optimized evaluation loops with half-weight bracketing and singleton-aware interpolation
- 🧠 Heap-stream k-way digest merge in Rust core for lower peak memory on large digest unions
- 🧵 Streaming two-way raw-ingest merge path in Rust core (centroids + values) to avoid extra merge buffers
- 🧊 TDigest Precision: Centroids as `f64` or `f32` — **auto-selected by input dtype**
- ⚖️ Weighted ingest across Rust/Python/Polars/Java (`add_weighted`, `add_weighted_values`, Java weighted adds)
- 🔄 Explicit precision casting across surfaces (`cast_precision` / `castPrecision`)
- 📦 TDIG v3 wire default (flags + header length + precision code + checksum), with v1/v2 decode compatibility
- 🧭 Explicit wire-version encode controls (`to_bytes(version=1|2|3)`, `toBytes(version)`)
- 🎚️ Scale families: `Quad`, `K1`, `K2`, `K3`
- 🔩 Singleton handling policy: **edge-precision (keep _N_)**, **respect singletons**, or **uniform merge**


## 📜 License
Apache-2.0

## 🤝 Community
- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Issue tracker: https://github.com/ingolfured/gr-tdigest/issues

## ⚡ Quick start
```bash
make setup    # toolchains + Python deps
make build    # Rust lib+CLI, Python ext, Java classes (dev)
make test     # Rust + Python tests
make release  # release CLI + wheel + JARs
```

## 🚀 Release automation
Release workflows are in `.github/workflows/` and trigger on tags matching `v*`:

- `release_pypi.yml`
- `release_cargo.yml`
- `release_maven.yml`

Minimum GitHub setup:

1. PyPI (`release_pypi.yml`):
- Create GitHub environment `pypi`.
- Configure PyPI Trusted Publisher for this repo/workflow in PyPI.

2. Cargo (`release_cargo.yml`):
- Create GitHub environment `crates-io`.
- Add secret `CARGO_REGISTRY_TOKEN`.

3. Maven (`release_maven.yml`):
- Create GitHub environment `maven`.
- Add secrets `MAVEN_REPOSITORY_URL`, `MAVEN_USERNAME`, `MAVEN_PASSWORD`.
- Add `MAVEN_SIGNING_KEY` and `MAVEN_SIGNING_PASSWORD` if your Maven repository requires signed artifacts.

4. Release tag:
- Ensure `Cargo.toml` version equals the release tag without `v` (for example `v0.2.2`).
- Push tag: `git tag v0.2.2 && git push origin v0.2.2`

5. Repository protection (recommended):
- Apply rulesets from version-controlled specs:
  - `./scripts/apply_github_rulesets.sh`
  - details: `.github/REPO_SETTINGS.md`

## 📤 Local publish command
`make publish` publishes to PyPI, crates.io, and Maven from local credentials.

Dry run (recommended first):

```bash
PUBLISH_DRY_RUN=1 make publish
```

Real publish:

```bash
MATURIN_PYPI_TOKEN=... \
CARGO_REGISTRY_TOKEN=... \
MAVEN_REPOSITORY_URL=... \
MAVEN_USERNAME=... \
MAVEN_PASSWORD=... \
make publish
```

Optional Maven signing variables:

- `MAVEN_SIGNING_KEY`
- `MAVEN_SIGNING_PASSWORD`

## 🧪 Usage


**Python**
```python
import gr_tdigest as td
d = td.TDigest.from_array([0,1,2,3], max_size=100, scale="k2")
print("p50 =", d.quantile(0.5))
print("cdf  =", d.cdf([0.0, 1.5, 3.0]))
d.add_weighted([10.0, 20.0], [2.0, 3.0])
blob_v1 = d.to_bytes(version=1)
d32 = d.cast_precision("f32")
```

**Polars**
```python
import polars as pl
from gr_tdigest import tdigest, quantile

df = pl.DataFrame({"g": ["a"]*5, "x": [0,1,2,3,4]})
out = (
    df.lazy()
      .group_by("g")
      .agg(tdigest(pl.col("x"), max_size=100, scale="k2").alias("td"))
      .select(quantile("td", 0.5))
      .collect()
)
print(out)
```

**Rust CLI**
```bash
echo '0 1 2 3' | target/release/tdigest --stdin --cmd quantile --p 0.5 --no-header
```

**Java (AutoCloseable)**
```java
import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;

import java.util.Arrays;

public class Example {
  public static void main(String[] args) {
    try (TDigest digest = TDigest.builder()
        .maxSize(100)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.EDGES).keep(4)
        .precision(Precision.F32)
        .build(new float[]{0, 1, 2, 3})) {
      double[] c = digest.cdf(new double[]{0.0, 1.5, 3.0});
      double p50 = digest.quantile(0.5);
    }
  }
}
```


## 🗂️ Project layout
```
├── src/                                  # Rust core, CLI entrypoint, algorithm modules
│   ├── bin/                              # Command-line app (tdigest CLI)
│   ├── tdigest/                          # Core T-Digest implementation (centroids, merge, scale)
│   └── quality/                          # Accuracy helpers & scoring utilities
├── bindings/                             # Language bindings
│   ├── python/                           # Python wheel (maturin)
│   │   ├── gr_tdigest/                   # Python package (abi3 native extension)
│   │   └── tests/                        # Python API + Polars tests
│   └── java/                             # Java API (Gradle project) + JNI shims
│       └── src/
│           └── gr/
│               └── tdigest/              # Public Java API + native bridge
├── integration/
│   └── api_coherence/                    # Cross-API contract tests (CLI ↔ Python ↔ Polars ↔ Java)
├── benches/                              # Rust benchmarks (quantile/CDF/codecs)
├── crates/
│   └── testdata/                         # Small datasets & fixtures for tests/benches
└── dist/                                 # Build artifacts (wheels/JARs) after release
```

## 🧩 Versions & compatibility
- **Rust**: stable (2021 edition)
- **Python**: CPython 3.12; packaged with **maturin**
- **Polars**: current 1.x (Python); Rust crate versions tracked in `Cargo.toml`

## 🧾 Changelog
- See `CHANGELOG.md` for release notes and unreleased changes.

## 🔮 Future improvements
- Allow scaling of weights and guard against centroid weight overflow
- Auto suggest a scaling function based on distribution
