# Original numerical implementation

From G-Research/tdigest-rs commit
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333` (Apache-2.0).
The numerical calculations are unchanged. Three historical namespace imports
remain adapted: `use crate::{` to `use self::{` in lib.rs; `use crate::{` to
`use super::{` in core.rs; `use crate::traits::FloatConst;` to
`use super::traits::FloatConst;` in scale.rs. These resolve identically when this
directory is built as the standalone `tdigest_core` crate. The trait definitions,
license, and public Python wrapper are byte-for-byte unchanged. The original
dependency constraints are in this directory's Cargo.toml; the Python feature
depends on this shared-source crate. Dependency versions remain part of the
numerical provenance and must pass the independent exact-output fixtures.
`src/upstream_py.rs` adapts only PyO3/NumPy syntax and module registration.

`tdigest_rs.TDigest` must preserve the existing API and numerical behavior.
Do not replace the calculations, normalize centroids, widen float32 arithmetic,
convert integer weights to floats, or silently correct upstream quirks.
The modern engine is opt-in via `gr_tdigest`.

Hashes below are of the original GitHub blobs, after reversing ONLY the three
namespace changes above. `scripts/check_upstream_sources.py` verifies this rule.
Keep the public Python wrapper out of automatic formatting.

```text
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  src/upstream/LICENSE
0ed15e8d3d18e2d9e82b4a0c8e1cdbace3de315dc79649566e35c28bb9446aad  bindings/python/tdigest_rs/lib.py
1753d6c48728561f77dd076c1a063ee9abe25251b1ed94a798b520173b7d3412  src/upstream/core.rs
eeb21b517557df106eb57c7d03dde1b8f26fe4b1745ee6f4b30cb604ddcaf63b  src/upstream/lib.rs
034cf845a9e283d51fe076a6b22956acb21aa31738094f18ed4d6e08c3240558  src/upstream/scale.rs
74324e7ec5fc5e9e8d448b88fa35c8a0525354eef10eb7fead694cf2d9754868  src/upstream/traits.rs
```
