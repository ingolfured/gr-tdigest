# 🌀 tdigest-rs
T-Digest provides a mergeable summary of a distribution, enabling **approximate quantiles and CDF** with strong tail accuracy. **tdigest-rs** delivers a production-ready Rust core with Python and Polars APIs plus Java (JNI), emphasizing compact memory, stable merge behavior, and easy adoption in data pipelines.



## ✨ Features
- 🚀 Mergeable quantiles for large / streaming data
- 🦀 Single Rust core shared across Rust, Polars, Python, and Java
- 🧊 Precision modes: canonical `f64` or compact `f32`
- 🎚️ Scale families: `Quad`, `K1`, `K2`, `K3`
- 🔩 Singleton handling policy: **edge–precision (keep _N_)**, **respect singletons**, or **uniform merge**

## 📜 License
Apache-2.0

## ⚡ Quick start
```bash
make setup    # toolchains + Python deps
make build    # Rust lib+CLI, Python ext, Java classes (dev)
make test     # Rust + Python tests
make release  # release CLI + wheel + JARs
```

## 🧪 Usage

**Rust CLI**
```bash
echo '0 1 2 3' | target/release/tdigest --stdin --cmd quantile --p 0.5 --no-header
```

**Python**
```python
import gr_tdigest as td
d = td.TDigest.from_array([0,1,2,3], max_size=100, scale="k2")
print("p50 =", d.quantile(0.5))
print("cdf  =", d.cdf([0.0, 1.5, 3.0]).tolist())
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

**Java (AutoCloseable)**
```java
import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;

import java.util.Arrays;

public class TestRun {
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

**Compile & run the Java example**
```bash
make release-jar

# Assume:
#   API JAR:      target/tdigest-rs-api.jar
#   Native libs:  target/release (contains libtdigest_rs.*)
# Adjust paths if your build uses different names/locations.

mkdir -p target/java-hello
javac -cp target/tdigest-rs-api.jar -d target/java-hello TestRun.java
java --enable-native-access=ALL-UNNAMED      -Djava.library.path=target/release      -cp target/tdigest-rs-api.jar:target/java-hello      TestRun
```

## 🗂️ Project layout
```
.
├── src/                # Rust core + CLI + bindings (Polars exprs, Python, JNI)
│   ├── bin/tdigest_cli.rs
│   ├── polars_expr.rs
│   ├── py.rs
│   ├── jni.rs
│   └── tdigest/…       # algorithm & internals
├── bindings/
│   ├── python/         # wheel via maturin
│   └── java/src/…      # Java API + JNI shims
├── gr_tdigest/         # Python package (abi3 extension & __init__)
├── tests/              # Python tests
├── benches/            # Rust benches
├── dist/               # Built wheels/JARs
└── Makefile
```

## 🧩 Versions & compatibility
- **Rust**: stable (2021 edition)
- **Python**: CPython 3.12; packaged with **maturin**
- **Polars**: current 1.x (Python); Rust crate versions tracked in `Cargo.toml`

## 🔮 Future improvements
- Guard against centroid weight overflow
- Ensure no leaks in CDF and quantile paths
