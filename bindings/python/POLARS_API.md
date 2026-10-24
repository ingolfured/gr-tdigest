# Modern Polars API: consumer candidate

The modern integration supports Polars 1.x and 2.x. The tested releases are
1.44.2 (latest 1.x) and 2.0.0. The optional `polars` extra accepts
`polars>=1.44.2,<3`; development stays pinned to 1.44.2. Install an exact release
with `pip install 'gr-tdigest[polars]' 'polars==2.0.0'` when selecting 2.x.
Original `tdigest_rs` objects are a separate API.

`make python-support-test` installs the same compiled wheel in isolated cells
for both Polars releases across CPython 3.10–3.15. It checks installed dependency
versions and native library hashes, then runs the public API and consumer
regressions, including grouped lazy joins, chunked inputs, streaming execution,
empty/missing states, list schemas and binary precision. Polars-free baseline
installs remain covered. Future releases must pass this gate before updating
the tested pins; the dependency range alone is not compatibility evidence.

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
| `from_bytes(blob)` | One reduced digest from decoded blobs, with explicit consistent precision |
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

Column-based `from_bytes` now requires `precision="f32"` or `"f64"`. This checks encoded precision; it does not convert it. Inspect eager bytes first with `wire_precision` or `infer_column_precision` when needed. Planning and execution then use the same schema, including an empty reduction.
