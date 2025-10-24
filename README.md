# 🌀 tdigest-rs

A **Rust t-digest** with **Polars integration** and extensions for **Python** and **Java/JNI**.
Fast, compact, and built for real-world analytics pipelines.

- 🚀 Mergeable quantiles for large / streaming data
- 🦀 Single Rust core shared across Rust, Polars, Python, and Java
- 🧊 Precision modes: canonical `f64` or compact `f32`
- 🎚️ Scale families: `Quad`, `K1`, `K2`, `K3`
- 🔩 Singleton handling policy: **edge–precision**, **respect singletons (keep _N_)**, or **uniform merge**

---

## 🔢 Versions & compatibility (tested)
Rust and Python use **different** Polars versions — that’s expected.

| Layer | Package | Version |
|---|---|---|
| **Python** | CPython | 3.12.x |
|  | polars | 1.34.0 |
|  | polars-runtime-32 | 1.34.0 |
|  | numpy | 2.3.4 |
|  | maturin | 1.9.6 |
|  | pytest | 8.4.2 |
|  | ruff | 0.14.1 |
|  | ziglang (manylinux) | 0.12.1 |
| **Rust** | rustc | ≥ 1.81 (2021 edition) |
|  | polars (crates) | 0.51.x |
|  | jemallocator | 0.5.4 |

---

## 🧰 Quick start (Makefile-first)
Clone the repo, then let the **Makefile** drive:

```bash
make help      # discover the main targets
make setup     # toolchain + Python env (uv)
make build     # Rust core + CLI, Python extension, Java classes
make test      # Rust tests + CLI smoke + Python tests + Java demo
```

Packaging (when you need it):
```bash
make wheel     # build manylinux wheel(s) into ./dist
make jar       # build Java API + native jar for the current platform
```

---

## 🧪 Usage examples

### Rust CLI
Compute a median from stdin:
```bash
echo "1 2 3 4 5" | target/release/tdigest --stdin --quantile 0.5
# quantile,value
# 0.5,3.0
```

From a file to JSON Lines:
```bash
target/release/tdigest --file data.txt --cmd cdf --output jsonl > out.jsonl
```

### Python (Polars expression)
```python
import polars as pl
from tdigest_rs import tdigest, quantile, StorageSchema, ScaleFamily

df = pl.DataFrame({"g": ["a"] * 2000, "x": list(range(2000))})
out = (
    df.lazy()
      .group_by("g")
      .agg(
          tdigest(
              pl.col("x"),
              storage=StorageSchema.F64,   # or StorageSchema.F32
              scale=ScaleFamily.QUAD,      # Quad/K1/K2/K3
              max_size=512
          ).alias("td")
      )
      .select(quantile("td", 0.5))
      .collect()
)
print(out)
```

### Java (AutoCloseable builder)
```java
import gr.tdigest_rs.TDigest;
import gr.tdigest_rs.TDigest.Precision;
import gr.tdigest_rs.TDigest.Scale;
import gr.tdigest_rs.TDigest.SingletonPolicy;
import java.util.Arrays;

public class TestRun {
  public static void main(String[] args) {
    System.out.println("AutoCloseable example:");
    try (TDigest digest = TDigest.builder()
        .maxSize(100)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.EDGES).keep(4)
        .precision(Precision.F32)             // internal f32 sketch; API uses double[] probes
        .build(new float[]{0, 1, 2, 3})) {    // build(float[])
      System.out.println(Arrays.toString(digest.cdf(new double[]{0.0, 1.5, 3.0})));
      System.out.println("p50 = " + digest.quantile(0.5));
    }
  }
}
```

---

## 🧱 Project layout
```
.
├── Makefile
├── src/                 # Rust core + CLI
│   ├── bin/tdigest_cli.rs
│   ├── polars_expr.rs   # Polars expressions (Rust side)
│   ├── py.rs            # Python bindings (maturin/pyo3)
│   └── tdigest/         # algorithm & internals
├── src-java/            # Java demo / JNI client
├── tdigest_rs/          # Python package (abi3 extension)
└── tests/               # Python tests
```

---

## 🤝 Contributing
This repo ships with a **pre-commit** setup (fail-fast) to keep things tidy.

Install & run:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

See the pinned hooks in `.pre-commit-config.yaml`.

---

## 🧬 License
Licensed under **Apache-2.0**.
