# tdigest_core

The original G-Research Rust API and arithmetic, retained at the pinned source
revision documented in PROVENANCE.md. The package exposes `tdigest_core::TDigest`
with public `means: Vec<T>` and `weights: Vec<u32>` fields for `f32`/`f64` use.
Its constructors, queries, immutable merge, result types, and inherited edge
behavior remain unchanged. The modern implementation is a separate `gr-tdigest`
package; matching its delta does not imply identical outputs.

```rust
use tdigest_core::TDigest;
let digest = TDigest::<f64>::from_array(&[1.0, 2.0, 3.0], 300.0)?;
let median: f64 = digest.median()?;
# Ok::<(), anyhow::Error>(())
```

The new Rust tests check the public API; the pinned upstream revision did not
contain Rust test cases. Independent numerical reference fixtures are checked
from the workspace root. See the repository's compatibility review guide for
the unchanged original Python suite and exact-output provenance.

Licensed under Apache-2.0; the original LICENSE is included.
