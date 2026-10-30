# Modern Polars API: consumer candidate

Install `gr-tdigest` normally when Polars is already installed. Its expressions
use that existing Polars runtime. The optional `gr-tdigest[polars]` extra installs
Polars if needed and requires only `polars>=1.0.0`, without an upper cap; it also
retains an existing compatible version. There is no need to pick a separate
package or native build for Polars 1.x versus 2.x. Development stays pinned to
1.44.2. Original `tdigest_rs` objects remain a separate API.

`make python-support-test` installs the same compiled wheel in isolated cells
for Polars 1.0.0, 1.20.0, 1.38.1, 1.44.2 and 2.0.0 across CPython 3.10–3.15.
Every cell starts with Polars installed, installs the base package and then the
extra with normal dependency resolution, and verifies both preserve its version
and the native library hash. The public API and consumer regressions cover
grouped lazy joins, chunked inputs, streaming execution, empty/missing states,
list schemas and binary precision. Polars-free baseline installs remain covered.
The open dependency range permits newer Polars releases without a package
metadata change; unreleased versions still need verification before compatibility
can be guaranteed.

Use the collection API of the installed Polars release (`streaming=True` on
early 1.x, `engine="streaming"` on newer releases). In grouped queries, train a
digest first, then encode it in a subsequent expression: early 1.x planners
reject nested reductions such as `to_bytes(tdigest("x"))` inside one `agg`.

On CPython 3.15, Polars 1.44.2 has an upstream Series namespace dispatch
limitation: `series.struct.field(...)` can return `None` even for ordinary
structs. Use `frame.select(pl.col("d").struct.field("count"))` for field
extraction on that combination. The expression API and plugin are exercised
by the compatibility gate; this package does not patch Polars namespace methods.

```python
import polars as pl
import gr_tdigest as gr
rows = pl.DataFrame({"g": ["a", "a", "b", "b"], "x": [0., 10., 100., 200.]})
groups = rows.lazy().group_by("g").agg(gr.tdigest("x", max_size=1000).alias("d"))
result = rows.lazy().join(groups, on="g").with_columns(gr.cdf("d", pl.col("x")).alias("p")).collect()
```

## Cardinality and precision

| Expression | Rows produced per logical group |
|---|---|
| `tdigest(values)` | One reduced digest |
| `merge_tdigests(digest)` | One reduced digest |
| `quantile(digest, q)`, `median(digest)` | One query of the reduced digest |
| add/weighted-add, scale, precision cast | One transformed reduced digest |
| `to_bytes(digest)` | One encoding of the reduced digest |
| `from_bytes(blob)` | One reduced digest; precision inferred from materialized binary inputs or supplied for column expressions |
| `cdf(digest, values)` | One digest broadcast over probes, or aligned digest/probe rows; list probes produce list results |

CDF broadcasting depends on input lengths, not how many digest rows contain
data. Aligned empty digests keep NaN results. A single selected empty digest
broadcasts NaN. Unsupported lengths raise an error. A column expression is a
supported dynamic CDF probe; quantile probabilities remain scalar configuration.
Select the first digest explicitly when that is what the application intends.

Default capacity is 1,000. Automatic precision follows the input dtype: Float32
training produces Float32 mean/extrema state; other numeric inputs use Float64.
Mass and sums stay Float64. CDF output follows floating probe precision; integer
probes produce floating probabilities. Explicit casts transform precision;
consistency checks do not silently cast state.

## Missing, empty and malformed state

Null training values and null CDF probes are errors. Missing digest rows are
also errors, for both queries and reductions. Filter or impute at the application
layer only after choosing its intended data policy. A valid empty digest is not
missing: it has no centroids, zero sum/count and NaN extrema. Nonempty NaN means,
invalid weights, negative capacity, nonfinite metadata and inconsistent mass
are rejected rather than converted into plausible empty results.

Merges require the same capacity, scale, singleton policy, pins, delta and mean
precision, including empty contributors. A larger capacity cannot recover lost
detail. Empty modern reduction uses the documented canonical empty state.

## Struct and binary migration

New writes retain readable `centroids`, `sum`, `min`, `max`, `count`, `max_size`
and add configuration/kind fields. Centroid weights are Float64, including for
Float32 means. This changes the schema from historical six-field writes. Old
six-field reads retain the historical K2/Use defaults and infer atomic kind only
for weight one. Missing information is inferred, not recovered. The older
integer `polars_tdigest` schema is not automatically supported.

Full-state TDIG v3 bytes preserve precision, configuration, extrema and kind.
V1/v2 limitations are explicit. Null or malformed blobs and mixed precision in
one decode operation are rejected. Schema migration and a Polars major upgrade
must be exercised against real consumer frames before internal deployment.

Fast public tests include aligned empty/healthy rows, strict missing-state
rejection, metadata corruption and lazy grouped joins with dynamic CDF probes,
Float32 casts and clipping. These bounded checks do not qualify the user's
unconfirmed Spark or private consumer matrix.

`from_bytes` uses `precision="auto"` by default for materialized `bytes`, `bytearray`, `memoryview` and nonempty Polars `Binary` Series. The adapter inspects every header and supplies the detected precision to the plugin before planning. It never samples, casts state, drops nulls, or collects a lazy expression. Mixed-precision Series, nulls and non-Binary Series are rejected. A Series is used as literal input; pass a column expression for per-group decoding of frame rows.

Column names and expressions still require `precision="f32"` or `"f64"`. This checks encoded precision; it does not convert it. Planning and execution use the same schema, including an empty reduction. An empty Series has no header and needs an explicit precision. An encoded v3 empty digest does carry precision and supports auto. Empty v1/v2 states have no precision code and require the existing explicit legacy decoding path; use v3 to preserve the precision of an empty f32 digest.

```python
blob = gr.TDigest.from_array([0., 10.], precision="f32").to_bytes()
decoded = pl.select(gr.from_bytes(blob).alias("d"))  # remains f32
blobs = pl.Series("blob", [blob], dtype=pl.Binary)
decoded_series = pl.select(gr.from_bytes(blobs).alias("d"))
decoded_column = pl.DataFrame({"blob": blobs}).lazy().select(
    gr.from_bytes("blob", precision="f32").alias("d")
).collect()
```
