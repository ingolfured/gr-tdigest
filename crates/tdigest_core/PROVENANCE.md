# Original numerical implementation

Copied byte-for-byte from G-Research/tdigest-rs commit
`8d9cf1fe82c5c9254e5dc9593f01717e317f4333` (Apache-2.0).
The four Rust source files, license, and Python wrapper are unmodified.
`Cargo.toml` keeps the original dependency constraints and removes the upstream
Python workspace member; the native adapter in `src/upstream_py.rs` only updates
the PyO3/NumPy binding syntax and module registration for our extension.

`tdigest_rs.TDigest` must preserve the existing API and numerical behavior.
Do not replace these calculations with the modern compressor, normalize
centroids, widen float32 arithmetic, convert integer weights to floats, or
silently correct upstream quirks. Modern behavior is opt-in via `gr_tdigest`.
Keep these files out of automatic formatting. Hashes are relative to repo root:

```text
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  crates/tdigest_core/LICENSE
0ed15e8d3d18e2d9e82b4a0c8e1cdbace3de315dc79649566e35c28bb9446aad  bindings/python/tdigest_rs/lib.py
1753d6c48728561f77dd076c1a063ee9abe25251b1ed94a798b520173b7d3412  crates/tdigest_core/src/core.rs
eeb21b517557df106eb57c7d03dde1b8f26fe4b1745ee6f4b30cb604ddcaf63b  crates/tdigest_core/src/lib.rs
034cf845a9e283d51fe076a6b22956acb21aa31738094f18ed4d6e08c3240558  crates/tdigest_core/src/scale.rs
74324e7ec5fc5e9e8d448b88fa35c8a0525354eef10eb7fead694cf2d9754868  crates/tdigest_core/src/traits.rs
```
