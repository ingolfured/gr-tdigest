# 0.3.0 release notes

**Status: unreleased.** This describes the changes planned for 0.3.0 compared
with the 0.2.x release line.

0.3.0 is a substantial correctness, compatibility and usability update to
gr-tdigest. It improves weighted statistics and saved state, makes invalid
operations fail safely, and brings the Python, Polars and JVM interfaces into
closer agreement with the Rust core. It also includes API and schema changes
that existing modern-engine applications need to account for when upgrading.

The modern Rust crate, Python package and JVM artifact use version 0.3.0.
The separate original G-Research Rust crate, `tdigest_core`, remains at 0.2.0.

## More reliable weighted statistics

Weighted medians and quantiles now consistently account for observation weights,
including fractional weights. Fixes cover interpolation near the tails, exact
single-value summaries, very small total weights, and compression that previously
lost a trailing contribution at small capacities. More careful summation also
improves calculations when large positive and negative values cancel.

Modern summaries retain weights, total weight and weighted sums in Float64,
even when means are stored in Float32. This prevents weights from being rounded
away merely because Float32 storage was selected. It increases storage for
Float32 centroids; it cannot restore information already lost in an older digest.

## Safer updates and saved state

Construction, reconstruction, merging, precision conversion and positive scaling
check that the resulting digest is valid. Unsupported arithmetic and inconsistent
saved state produce errors. A failed modern update, merge or scaling operation
leaves the existing digest unchanged, including an iterable merge whose later
input is invalid.

Modern Python copies, pickle restoration and TDIG binary round trips preserve
independent full state: weights, extrema, weighted sum, compression settings and
the distinction between exact points and compressed groups. Rebuilding from
`means` and `weights` alone still constructs weighted observations; it is not a
full-state copy.

TDIG v3 retains the added configuration and weight information. Existing v1/v2
data remains readable, but older formats cannot express every modern state.
Exports that would lose unsupported state are rejected. Older readers may also
reject newly used v3 flags, so check readers before exchanging new saved data.

## Original G-Research compatibility

The Python `tdigest_rs` and Rust `tdigest_core` interfaces use the pinned original
engine. They preserve its supported calculations and API behavior, including
its historical copy/pickle behavior. Overflowing uint32 totals and invalid
arithmetic results from finite inputs now raise errors instead of wrapping,
panicking or producing corrupted summaries.

Use `gr_tdigest` for the modern Python engine. Setting `delta` on a modern digest
does not switch to the original engine or promise identical results. The two
interfaces deliberately retain different behavior.

## Python and Polars improvements

Modern Python construction now honors supplied weights. Copies own independent
state, and installed type declarations and method documentation make the API
easier to use from editors and type checkers.

Compatible NumPy input and output buffers use bulk copies. Independent modern
constructors release the Python interpreter lock during native construction,
allowing a caller's thread pool to build separate digests concurrently. Input
conversion still holds the lock; sharing one mutable digest between threads is
not supported.

Polars is optional. Numerical Python use does not require it, and installing
`gr-tdigest[polars]` retains an already installed compatible Polars version.
Compatibility checks cover selected Polars 1.x releases and 2.0.0. CDF operations
now match digests and probes by row count, preserve List-shaped results, and keep
valid empty digests distinct from missing rows.

## JVM delivery

Java gains completed weighted/delta builders and fluent operations. Deserialized
objects own independent native handles, fixing premature cleanup of restored
state. Catchable Rust panics are translated to Java exceptions. Packaged JARs
load their native library outside a development checkout and provide clearer
loading diagnostics. Scala, Kotlin and separate-executor Spark examples cover
use of the shared Java artifact.

JVM builds support configured toolchain/dependency sources and offline operation
after dependencies have been staged. They target Java 11 bytecode.

## Upgrading from 0.2.x

| Area | What to check in your application |
| --- | --- |
| Modern Rust API | Builders, seeded construction, merges and precision conversions return checked `Result` values. Handle errors, for example with `builder().build()?`. The configuration name is `delta`, replacing `legacy_delta`. |
| Defaults and delta | The modern default capacity is consistently 1,000. Set it explicitly if your application relied on an older Python default. Modern delta mode uses K2Norm with singleton policy Off; incompatible explicit settings are rejected. |
| NumPy query outputs | Scalar and zero-dimensional probes return scalars. One-dimensional arrays return independent writable Float64 arrays, including single-element inputs. Lists and tuples retain their containers. Flatten and reshape multidimensional probes in your application. |
| Polars schemas | New structs include compression settings and centroid kind; weights are Float64 in both mean precisions. Review explicit schemas and persisted struct columns. Older six-field structs are read using their historical defaults. |
| Polars missing rows and CDF | Missing digest rows are errors. Supply a genuine empty digest when that is intended. CDF broadcasting follows row counts, and List probes retain List output. |
| Binary decoding | Give column-based Polars `from_bytes` an explicit `precision="f32"` or `"f64"`. This checks the encoded precision; it does not cast the digest. |
| Saved state | Use full-state copies or TDIG v3 for faithful modern restoration. Delta and other newly represented settings may require v3. Check compatibility with older consumers before writing new state. |
| Invalid operations | Expect errors for malformed state, unsupported numeric ranges, or merges with differing precision or compression settings, including empty contributors. |

See the [Python/NumPy guide](bindings/python/OBJECT_API.md),
[Polars guide](bindings/python/POLARS_API.md),
[Rust API contract](api_design.md) and [Java guide](bindings/java/README.md)
for detailed behavior and examples.

## Runtime and build support

- Python: standard CPython 3.10–3.15. NumPy requirements depend on the Python
  version; Python 3.13 and later require NumPy 2.x. See the
  [package requirements](bindings/python/pyproject.toml).
- Polars: optional, with a declared minimum of 1.0.0. The installed-package checks
  cover 1.0.0, 1.20.0, 1.38.1, 1.44.2 and 2.0.0. An open dependency range does not
  establish compatibility with every future version.
- JVM: build/runtime coverage for JDK 11, 17, 21 and 25, targeting Java 11 bytecode.
- Rust source builds: the repository pins Rust 1.95.0.
- Linux Python wheels: target glibc 2.28 or later.

## Accuracy and validation

The test suite now includes independent original-engine reference outputs,
weighted numerical controls, recorded-data accuracy cases, cross-language saved
state checks and installed-package tests. Source archives and standalone JARs
are checked as consumer artifacts as well as testing the checkout.

T-digest remains an approximation. Accuracy depends on the data, summary size,
scale and merge pattern; known difficult cases remain, particularly with small
summaries. Modern does not win every accuracy measure against the original
engine. The [comparison guide](comparison_design.md) explains those tradeoffs,
and the [quality test guide](tests/quality/README.md) describes the reproducible
checks. Historical development timings are not a general speed or memory
guarantee for 0.3.0.

Publication and final release validation are tracked in the
[publishing guide](PUBLISH.md).
