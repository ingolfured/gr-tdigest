# tdigest_design.md

This document is the **current-state implementation design** for the Rust TDigest core.

- Scope: internals in `src/tdigest/*` and how frontends use them.
- It describes what the code does today.
- For external implementation/paper comparisons and tradeoff positioning, see `comparison_design.md`.

For target cross-surface API contracts, see `api_design.md`.

## 1. Document boundary

This file covers:
- Core data model and module responsibilities.
- Ingest/merge/compression/query/wire internals.
- Current frontend-service layering.

Out of scope for this file:
- Target-state product/API policy (see `api_design.md`).
- External library or paper comparisons.

## 2. Core architecture

### 2.1 Main modules

- `src/tdigest/tdigest.rs`: digest type, builders, add/merge entry points, bytes entry points.
- `src/tdigest/merges.rs`: stream-merging helpers + stage-1 normalization.
  - Includes a heap-based streaming k-way centroid merge used by digest-to-digest merge.
- `src/tdigest/compressor.rs`: staged compressor (`compress_into`) and policy logic.
- `src/tdigest/centroids.rs`: centroid representation and `Atomic` vs `Mixed` behavior.
- `src/tdigest/quantile.rs`: quantile and median internals.
- `src/tdigest/cdf.rs`: CDF internals.
- `src/tdigest/wire.rs`: TDIG codec.
- `src/tdigest/frontends.rs`: shared frontend validation/config/merge/decode helpers.

### 2.2 Digest and centroid model

`TDigest<F>` where `F` is `f32` or `f64`:
- Centroid means are stored in `F`; centroid weights are always `f64`.
- Digest stats `sum` and `count` are `f64`.
- Optional `delta` selects K2Norm compression with singleton policy Off, bypassing the max-size cap.
- Digest min/max are stored in `F` and surfaced as `f64`.

Centroid kind is explicit:
- `Atomic`: exact identical-value mass (unit or pile).
- `Mixed`: merged/interpolated mass.

`is_singleton()` is kept as compatibility naming and maps to `is_atomic()`.

Mean precision must not narrow mass: at an f32 weight of 2^24, adding one sample
could previously leave the centroid weight unchanged while the digest count
advanced. For two equal piles, adding a value to the left could then move the
median to the right. We retain weights in f64 throughout normalization, compression,
merge, scaling, and precision casts. This fixes the storage loss for exact integer
totals through 2^53; ordinary f64 rounding and aggregate overflow still require
care for larger or extremely disparate weights. It cannot recover mass already
lost in an older stored digest.

On the tested Linux x86_64 target with Rust 1.90, a modern `Centroid<f32>` occupies
16 bytes instead of 12; `Centroid<f64>` remains 24 bytes. The f32 digest shell
remains 96 bytes. This costs four bytes per allocated f32 centroid (one third more
centroid storage), and its speed/cache impact must be measured. Means and extrema
retain their chosen precision. Public Rust `weight() -> F` remains a potentially
lossy compatibility accessor; `weight_f64()` exposes the authoritative mass.
Python weights remain Float64 arrays and Java weighted inputs remain double[].
The separate original `tdigest_rs` implementation retains its uint32 weights.

## 3. How new values are merged

### 3.1 Add path (`add` / `add_many`)

`add_many(values)` is a thin wrapper:
1. Empty input: no-op.
2. Otherwise call `merge_unsorted(values)`.
3. Replace `self` with the merged result.

`add(value)` delegates to `add_many([value])`.

Weighted add path:
- `add_weighted(value, weight)` and `add_weighted_many(values, weights)` are supported in Rust core.
- Inputs must satisfy:
  - same length for values/weights
  - finite values
  - finite, strictly positive weights
- Weighted values are converted to atomic centroids, then merged through the same digest merge/compression path.
- Frontend surfaces route weighted add through `FrontendDigest::add_weighted_f64(...)`, so Python/JNI/Polars share the same validation and merge semantics.

Weight scaling is transactional. `scale_weights` validates the positive finite
factor, prepares centroid products locally, and rejects any nonpositive or
nonfinite product before calling checked constructors. It also rejects a
nonfinite accumulated product total or nonpositive/nonfinite scaled stored
count. Only then does it replace centroids and scale count/sum. Errors use the
existing `InvalidWeight` variant and preserve the original digest bytes. Empty
digests remain no-ops after factor validation; valid positive subnormal products
remain supported. Successful count and sum arithmetic retains its prior order.
This validation is limited to weight scaling: aggregate input/merge validation,
weighted-value product overflow, and the raw-sum overflow policy are separate.

### 3.2 Raw-value ingest (`merge_unsorted` / `merge_sorted`)

`merge_unsorted`:
1. Reject non-finite values.
2. Sort ascending.
3. Call `merge_sorted`.

`merge_sorted`:
1. Reject non-finite values.
2. Empty input: return clone.
3. Build a result shell (`new_result_for_values`) with updated count/min/max placeholders.
4. Build a sorted stream with `MergeByMean::from_centroids_and_values`:
- Existing centroids are reused as-is.
- Each new scalar value becomes an atomic unit centroid (`weight=1`).
- This is a streaming two-way merge iterator (no prebuilt merged buffer).
5. Run the full compressor pipeline with `compress_into`.

Important: compressor stage 1 recomputes digest metadata (`count`, `sum`, `min`, `max`) from the normalized stream and writes those back to the result.

Weighted constructor/merge helpers:
- `from_weighted_unsorted(values, weights, max_size)` builds from weighted input directly.
- `merge_weighted_unsorted(values, weights)` merges weighted input into an existing digest.

### 3.3 Digest-to-digest merge (`merge_digests`)

`TDigest::merge_digests(digests)`:
1. Scan digests and keep only non-empty runs (`count>0` and non-empty centroid vector).
2. Pick the first non-empty digest as the config source (`max_size`, `scale`, `policy`, `delta`).
3. Perform a streaming k-way merge of sorted centroid runs via `KWayCentroidMerge::from_runs`.
  - Implementation uses a min-heap over run heads (`O(total_len * log(k))`).
  - It avoids materializing one full concat+sort buffer before compression.
  - Historical concat+sort run merge is retained only as a `#[cfg(test)]` baseline helper.
4. Recompress with `compress_into` using the chosen config.
5. If all digests are empty, return `TDigest::default()`.

Note: core merge is permissive by design. Strict precision/config checks are enforced in `FrontendDigest::merge_in_place`.

## 4. Compression pipeline internals

`compress_into` runs stages 1->6.

### 4.1 Stage 1: normalize

`normalize_stream` does three jobs:
1. Validate non-decreasing mean order (panic on decreasing mean).
2. Coalesce adjacent equal-mean items; the result is atomic only when every input was atomic. Mixed provenance and fractional weights are preserved.
3. Accumulate:
- `total_w`
- `total_mw = sum(mean * weight)`
- `min`/`max` mean

Output at this stage is strictly increasing by mean.

All later compressor paths preserve a one-input cluster's mean bits and
Atomic/Mixed kind. The shared accumulator emits that input without recomputing
`mean * weight / weight`; even one rounding step would move an exact point mass,
change CDF results at the observed value, and risk duplicate means on a later
merge. K-limit passes, equal-weight bucketization and single-bucket collapse use
the same emission rule. Clusters containing multiple inputs remain Mixed.
Their weighted-average arithmetic is unchanged while its result lies within the
cluster's first/last input means. If finite accumulation rounds that result
outside the interval, clamp it to the nearest endpoint. Consecutive disjoint
input clusters then retain disjoint mean intervals after compression and f32
rounding, preventing duplicate or out-of-order centroids. The transient
accumulator tracks the last input mean; centroid storage and wire layouts do not
change. The bound applies only when the weight sum, weighted sum, and quotient
are all finite. It must not hide overflow or repair invalid aggregate state.

### 4.2 Stage 2: policy slice

Policy is derived from `SingletonPolicy` and current `max_size`.

`Off`:
- `left=[]`, `right=[]`, `interior=all`.
- `core_cap = max_size`.
- Fast paths:
  - if `len <= max_size`, passthrough.
  - if `max_size == 1`, collapse all to one mixed centroid.
  - if `max_size == 2`, bucketize directly into 2.

`Use`:
- Preserve up to one centroid on each side (first and last when present).
- `core_cap = max_size - left_len - right_len` (saturating).
- Interior excludes the preserved endpoints.

`UseWithProtectedEdges(k)`:
- Protect up to `k` consecutive atomic centroids from the left and right edges.
- Edge scanning stops at the first mixed centroid.
- Protected edges are outside interior cap.
- `core_cap = max_size` (applies only to interior).

### 4.3 Stage 3: k-limit merge

`klimit_merge(items, d=max_size, family)` greedily builds interior clusters.

Definitions:
- `total_w = sum(item.weight)`.
- `q_to_k(q, d, family)` maps quantile position to k-space.
- Keep cluster growth while `delta_k <= 1 + KLIMIT_TOL` (`KLIMIT_TOL = 1e-12`).

Loop sketch:
1. Track completed cluster mass `c_acc`.
2. For candidate item with weight `w_next`:
- `q_r = (c_acc + acc_cluster_weight + w_next) / total_w`
- `k_right = q_to_k(q_r, d, family)`
- Compare vs current `k_left`.
3. If within limit, absorb item.
4. Else flush current cluster and start a new one.

Cluster kind emission:
- Single-item cluster stays atomic only if its head item was atomic.
- Multi-item cluster is always emitted as mixed.

Delta mode uses the same k-limit merge with K2Norm and zero tolerance. Its normalizer uses the number of centroids in the current pass; quantiles use cumulative weight. Delta mode bypasses Stage 4 and Stage 6.

The K2Norm merge pass requires a positive normalizer `4*ln(n/delta)+24`.
When it is zero or negative, the input has very few normalized centroids
relative to the requested delta. Stage 3 retains those centroids unchanged,
including their weights and Atomic/Mixed kinds, instead of applying a reversed
scale that merges every input. The same guard applies when K2Norm is selected
with max-size mode; its later cap/policy stages still run. Positive-domain
arithmetic is unchanged, including the `+infinity` normalizer reached when a
tiny positive delta overflows `n/delta` and the scale factor becomes zero.
This changes the modern engine only; the original `tdigest_rs`/`tdigest_core`
calculation path remains pinned.


### 4.4 Stage 4: cap

If interior count exceeds `core_cap`, run a second k-limit merge on the Stage-3 output:
1. Keep scale family fixed (`K1`/`K2`/`K3`/`Quad`).
2. Binary-search a smaller `d'` in `(0, d]` where `d` is the configured `max_size`.
3. Pick the largest `d'` such that `klimit_merge(..., d')` yields `<= core_cap`.
4. Use that result as the capped interior.

This keeps Stage 4 aligned with Stage 3 geometry (same `q_to_k` family + `Δk<=1` rule), while still enforcing the cap.

Safety fallback:
- If rare plateau behavior prevents converging under cap, Stage 4 tightens `d'` further for a bounded number of retries.
- If still over cap, it falls back to order-preserving equal-weight bucketization as a last-resort guard.

### 4.5 Stage 5: assemble

Concatenate without mutation:
- `left + core_capped + right`.

### 4.6 Stage 6: post-policy finalize

`Use` policy only:
- If assembled length still exceeds `max_size`, bucketize whole assembled vector to enforce global cap.

`Off` and `UseWithProtectedEdges(k)`:
- No additional post cap.

### 4.7 How `max_size` actually applies

`max_size` is policy-dependent in effect:
- `Off`: hard cap on total centroid count.
- `Use`: tries to preserve endpoints but still enforces final total `<= max_size`.
- `UseWithProtectedEdges(k)`: hard cap on interior only; total can exceed `max_size` by protected edges.

`max_size` also sets `d` for `q_to_k`, so it influences cluster admissibility in stage 3 even before stage-4/6 capping.

## 5. Query internals

### 5.1 CDF (`src/tdigest/cdf.rs`)

`TDigest::cdf(vals)`:
1. Empty digest -> vector of `NaN`.
2. Build one prefix-weight array (`prefix[i] = cumulative weight before centroid i`).
3. For each probe:
- `NaN` probe -> `NaN`.
- Else evaluate via `cdf_at_val_fast` (binary search by centroid mean).

`cdf_at_val_fast` cases:

Exact centroid hit (`idx`):
- Return midpoint mass: `(prefix[idx] + 0.5 * w_idx) / total_w`.

Left of first centroid:
- `< min` -> `0.0`.
- Between `min` and first mean: guarded linear ramp using first centroid half-weight.

Right of last centroid:
- `> max` -> `1.0`.
- Between last mean and `max`: symmetric guarded ramp.

Between centroids `li` and `ri`:
- `gap = mean_r - mean_l`.
- Exclusions:
  - `left_excl = 0.5 * w_l` if left centroid is atomic else `0`.
  - `right_excl = 0.5 * w_r` if right centroid is atomic else `0`.
- Span mass:
  - `dw_center = 0.5 * (w_l + w_r)`
  - `dw_span = dw_center - left_excl - right_excl`
- Base mass:
  - `base = prefix[li] + 0.5 * w_l + left_excl`
- Output:
  - `(base + dw_span * frac) / total_w`, `frac=(x-mean_l)/gap`.

If both neighbors are atomic, `dw_span = 0`, so this becomes a strict step.

Performance:
- Rayon path is used when probe count `>= PAR_MIN` where `PAR_MIN = 32768`.

### 5.2 Quantile and median (`src/tdigest/quantile.rs`)

`TDigest::quantile(q)`:
1. `q=NaN` -> `NaN`.
2. Empty digest -> `NaN`.
3. Finite `q` is clamped to `[0,1]` in core.
4. Convert to target weight index: `index = q * total_weight`.
5. Edge handling:
- `q=0` -> `min`; `q=1` -> `max`.
- Outside the first/last centroid centers, retain atomic mass or interpolate toward data extrema.
- All boundaries use actual weights, including totals below one.
6. Find bracketing centroids using half-weight center-to-center spans.
7. Interpolate with singleton-aware rules.

Singleton-aware interpolation rules:

Atomic pile snap (any positive weight):
- If `index` is strictly inside the atomic mass, return its mean exactly.

Otherwise interpolate with dead zones:
- `dead_left = weight_left / 2` when left is atomic, else `0`.
- `dead_right = weight_right / 2` when right is atomic, else `0`.
- Interpolate in remaining span.

This keeps quantile behavior symmetric with CDF atomic exclusions.

`median()`:
- Empty -> `NaN`.
- Non-empty -> `quantile(0.5)`; there is no even-count shortcut that ignores centroid weights.

### 5.3 Strictness split for quantile probes

Current split is intentional:
- Core `quantile` clamps finite probes.
- Frontend layer (`FrontendDigest::quantile_strict`) rejects non-finite and out-of-range probes.

### 5.4 Under-capacity exactness (current behavior)

When total training cardinality `N` is below `max_size` (and no later operation forces an over-capacity recompression), exactness is pinned down as follows:
- `cdf(x)` at training values is exact under midpoint ECDF semantics over ties: `(#<x + 0.5*#=x) / N`.
- `quantile(q)` is exact at mid-ranks `q = (i + 0.5) / N`, returning the exact order statistic at index `i`.

Non-goal:
- Exactness for arbitrary quantile probes is not guaranteed; the exactness contract is mid-rank-specific.

These semantics are asserted by focused tests in:
- `src/tdigest/cdf.rs`
- `src/tdigest/quantile.rs`

### 5.5 Positive subnormal total mass

Half-weight arithmetic and `q * total` can underflow when the entire stored
centroid mass is positive and subnormal. These queries use a temporary digest
view whose weights first multiply by `2^1022` (`1 / f64::MIN_POSITIVE`). A
second exact power-of-two scale, derived from that normal total's exponent,
places the total in `[1, 2)`. The two factors stay separate because their
combined reciprocal could overflow. Every positive weight becomes normal, and
the total no longer makes a tiny positive `q` smaller before tail interpolation.
Means, kinds, extrema, and configuration stay unchanged, and queries never
mutate the stored digest.

CDF uses the actual centroid total already computed for its prefix array.
Quantile first checks its stored total, then computes actual mass only in the
subnormal branch. The temporary count is the scaled actual mass, so recursion
terminates after one copy even if a directly seeded subnormal header differs
from its subnormal centroid total. Empty/NaN handling, quantile endpoints, and
weight-independent single-centroid quantiles retain their existing branches.
Normal-total queries keep the previous summation order, kernels, and allocation
behavior, with only an extra condition. The rare fallback allocates a query copy.

This evaluates the stored centroid distribution; it cannot recover mean or sum
information already lost during compression. A normal total with a
minimum-subnormal component remains on the ordinary path and may still lose
that component's tiny half-weight or rank boundary. Malformed normal header
counts hiding subnormal actual mass are not repaired by the quantile prefilter.
Aggregate overflow and malformed-state validation remain separate concerns.

## 6. TDIG wire format

Implemented in `src/tdigest/wire.rs`.

Versioning:
- Encoder default writes TDIG **v3**.
- Decoder supports **v1**, **v2**, and **v3**.
- Explicit versioned encoding is available via:
  - Rust core: `to_bytes_with_version(WireVersion::V1|V2|V3)`
  - Frontend service: `to_bytes_with_version(...)`
  - Python/Java/Polars surfaces: `to_bytes(version=...)` / `toBytes(version)`

Payload layouts:
- v1:
  - `f32 mean + u64 weight` per centroid, or
  - `f64 mean + u64 weight` per centroid.
- v2:
  - `f32 mean + f64 weight + kind(u8)` per centroid, or
  - `f64 mean + f64 weight + kind(u8)` per centroid.
- v3:
  - Header adds `flags`, `header_len`, and explicit `payload_precision` code.
  - Header includes optional 4-byte checksum (CRC32) controlled by flags.
  - Payload layout is the same as v2 (`mean + f64 weight + kind`), but width is explicit in header.

Decode behavior:
- v1/v2: payload length determines wire precision.
- v3: payload precision is explicit in header and payload length is validated against it.
- v1 keeps legacy heuristic (`w==1` => atomic unit, otherwise mixed).
- v2 decodes centroid kind explicitly from payload and preserves atomic-vs-mixed identity.
- v2/v3 preserve fractional centroid weights on wire.
- v3 checksum is verified when present.
- v3 flag bit3 (0x08) means an f32-mean digest contains weights that require f64
  precision. It adds no bytes: the payload already stores f64 weights. The flag
  is checksum-covered, and older readers reject it instead of narrowing mass.
  It is valid only with f32 mean precision. Existing representable-weight blobs
  keep the same layout and flag bytes. Explicit v1/v2 export rejects these
  f32-mean digests; callers may explicitly cast to f64 before older-format export.
  New readers still accept old v1/v2 payloads and retain their encoded weights.
- v3 flag bit2 carries an optional u64 `pin_per_side` for edge settings above 255.
  It follows the delta extension when present; otherwise it starts at byte 64.
  The fixed pin byte is 255, the edge policy is required, and header length/checksum
  cover the full extension. Older versions reject export of these settings.
- v1 export rejects fractional/out-of-range weights; its historical kind heuristic
  can still change weighted-query behavior, so use v3 for faithful state transfer.
  Decoders reject nonpositive or nonfinite weights before constructing centroids.
- v3 flag bit1 stores an optional positive finite delta at header bytes 64..72;
  the extension is covered by the checksum. Absent extension means max-size mode.
- Explicit v1/v2 export cannot represent delta mode and is rejected. The existing
  infallible Rust encoder asserts this precondition; language wrappers return errors.
- Polars struct codecs carry scale, policy, pin count, optional delta, and atomic
  flags. New structs always store centroid weights as Float64; mean/min/max
  dtypes still identify f32 or f64 precision. Readers accept older Float32 weights
  with Float32 means and promote those weights exactly. Older structs also retain
  their original defaults and weight-based kind inference. Mutation/serialization
  upgrades to the current schema without changing mean precision. Single-digest
  serialization avoids recompression.

## 7. Frontend service role

`src/tdigest/frontends.rs` centralizes behavior used by Python/JNI/Polars/CLI:
- Parse/normalize scale, policy, precision hints.
- Validate training values.
- Enforce strict quantile probes (`finite` + `[0,1]`).
- Enforce strict merge compatibility (precision + config).
- Provide explicit precision casting (`cast_precision`).
- Provide precision-aware decode checks (`from_bytes_with_expected`).

Adapters (`src/py.rs`, `src/jni.rs`, `src/polars_expr.rs`) should remain thin around this layer.

## 8. Known current-state gaps

1. Legacy wire compatibility mode.
- Decoder supports v1, v2, and v3.
- Default re-encoding writes v3; old formats cannot store delta mode.

2. Strict merge location.
- Strict compatibility checks are frontend-layer behavior, not core merge behavior.

3. Doc split discipline.
- Keep target behavior only in `api_design.md` and current internals only here.

## Original G-Research numerical compatibility

`tdigest_rs.TDigest` uses the pinned `tdigest_core` workspace package at
`src/upstream` through `src/upstream_py.rs`. The original Rust package and Python
wrapper share these source files; a private crate alias preserves the adapter
path without duplicating arithmetic. Both f32 arithmetic and u32 centroid weights are retained.
The Python wrapper is unchanged from upstream, including its cached arrays and
default-delta reconstruction during copy/pickle. This deliberate separate path
preserves existing application outputs; `gr_tdigest`, Polars, Java, and the CLI
continue using the modern shared core. Numerical fixes there must not silently
change the original interface. See the vendored source provenance and exact
output regression tests.
