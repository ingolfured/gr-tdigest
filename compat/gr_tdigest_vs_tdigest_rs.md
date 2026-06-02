# gr-tdigest vs. tdigest-rs

`gr-tdigest` is the successor to the original `tdigest-rs` Python library. Its
legacy Python surface is designed as a **drop-in replacement**: the old
constructors and serialization APIs keep working unchanged, including

```python
from gr_tdigest import TDigest, DEFAULT_DELTA  # DEFAULT_DELTA == 300.0

d = TDigest.from_array(values, delta=DEFAULT_DELTA)   # delta-mode, as before
d = TDigest.from_means_weights(means, weights, delta=DEFAULT_DELTA)
blob = d.to_dict(); TDigest.from_dict(blob)           # and pickle round-trips
```

Passing `delta=` selects the same compression behavior as the old library
(`scale="k2norm"`, singletons off). gr-tdigest additionally offers a
`max_size`-based mode, explicit `precision`, scale families, and singleton
policies — none of which the old library exposed.

This document records the **deliberate behavioral differences** a project
migrating from `tdigest-rs` may observe. These are intentional design choices,
not bugs.

## 1. Non-finite input is rejected, not silently dropped

The old library silently discarded `NaN` values and accepted `±inf` as
centroid means. gr-tdigest rejects any non-finite training value up front:

```python
TDigest.from_array([1.0, float("nan"), 2.0])
# ValueError: tdigest: non-finite values are not allowed ...
#             hint: clean your data or drop NaN/±inf before building the digest
```

**Why:** silently dropping data, or letting `±inf` poison quantile estimates,
hides data-quality problems. Failing loudly is safer for a summary structure
whose whole purpose is faithful distribution estimation.

**Migration:** if you previously relied on `NaN` being dropped, filter the
input yourself, e.g. `arr = arr[np.isfinite(arr)]`.

## 2. `median()` is not guaranteed to equal `quantile(0.5)`

In the old library `median()` was an alias for `quantile(0.5)`. In gr-tdigest
they are computed independently and may differ slightly:

```python
d.median()        # e.g. -0.04979
d.quantile(0.5)   # e.g. -0.06035
```

**Why:** the two follow gr-tdigest's interpolation rules separately. The
difference is within normal t-digest estimation error.

**Migration:** do not assume `median() == quantile(0.5)`. Pick one and use it
consistently; use `quantile(0.5)` if you specifically need the 0.5-quantile
estimate.

## 3. Quantile and median estimates differ within tolerance

gr-tdigest uses revised interpolation (half-weight bracketing, singleton-aware
edges). For the same input and `delta`, individual quantile/median values can
differ from the old library by small amounts — well inside typical t-digest
accuracy bounds, but enough to break tests that pinned exact values or used
very tight tolerances.

**Migration:** compare against expected quantiles with realistic tolerances
rather than exact equality.

## 4. Precision is chosen explicitly, not inferred from input dtype

The old library inferred its backend precision from the input array's dtype, so
a `float32`-built digest and a `float64`-built digest were considered different
types and could not be merged.

gr-tdigest decouples precision from input dtype. The backend defaults to **f64**
regardless of the input array's dtype, and you opt into 32-bit explicitly:

```python
TDigest.from_array(np.array([1, 2, 3], dtype=np.float32)).inner_kind()  # "f64"
TDigest.from_array(values, precision="f32").inner_kind()                # "f32"
```

As a result, two digests built from differently-typed inputs (without an
explicit `precision`) are both f64 and merge cleanly. Merging digests of
genuinely different precision is still rejected, with a clear instruction:

```python
f32 = TDigest.from_array(values, precision="f32")
f64 = TDigest.from_array(values, precision="f64")
f32.merge(f64)
# ValueError: tdigest merge: incompatible digests (precision f32 vs f64).
#             Cast explicitly before merge (e.g. cast_precision('f64')).
```

**Migration:** set `precision="f32"` where you previously relied on a
`float32` input to select the 32-bit backend, and use `cast_precision(...)`
to align two digests before merging.

## 5. Non-`float64` numeric input is accepted

The old library raised `TypeError` for unsupported input dtypes (e.g.
`float16`). gr-tdigest converts numeric input to its backend precision instead
of raising:

```python
TDigest.from_array(np.array([1, 2, 3], dtype=np.float16))  # accepted (f64 backend)
```

**Migration:** no action needed; previously-rejected numeric dtypes now work.

---

The vendored upstream test suite under `tdigest-rs-upstream/` exercises these
differences directly; see [`tdigest-rs-upstream/PROVENANCE.md`](tdigest-rs-upstream/PROVENANCE.md)
for its source, and `make legacy-strict-test` to run it against the current
extension.
