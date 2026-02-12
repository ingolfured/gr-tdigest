#![cfg(feature = "python")]
#![allow(clippy::unused_unit)]

use std::any::TypeId;

use ordered_float::FloatCore;
use polars::prelude::ListChunked;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::tdigest::codecs::WireOf;
use crate::tdigest::frontends::{parse_precision_hint, parse_singleton_policy_str, PrecisionHint};
use crate::tdigest::precision::FloatLike;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{wire_precision, WireDecodedDigest, WirePrecision};
use crate::tdigest::{ScaleFamily, TDigest};

/* ==================== input dtypes ==================== */

#[inline]
fn guard_no_upcast_f32_to_f64<F: 'static>(values: &Series) -> PolarsResult<()> {
    if matches!(values.dtype(), DataType::Float32) && TypeId::of::<F>() == TypeId::of::<f64>() {
        return Err(PolarsError::ComputeError(
            "tdigest: precision=\"f64\" is not allowed for Float32 input. \
             Use precision=\"f32\" to match the input dtype or upcast your column to Float64."
                .into(),
        ));
    }
    Ok(())
}

/* ==================== planning helpers ==================== */

fn cdf_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    let vdt = inputs.get(1).map(|f| f.dtype()).unwrap_or(&DataType::Null);
    let out = match vdt {
        DataType::List(inner) => match inner.as_ref() {
            DataType::Float32 => DataType::List(Box::new(DataType::Float32)),
            _ => DataType::List(Box::new(DataType::Float64)),
        },
        DataType::Float32 => DataType::Float32,
        _ => DataType::Float64,
    };
    Ok(Field::new("".into(), out))
}

// The digest struct is compact if its "min" field is Float32.
fn is_compact_digest_dtype(dt: &DataType) -> bool {
    if let DataType::Struct(fields) = dt {
        for f in fields {
            if f.name() == "min" {
                return matches!(f.dtype(), DataType::Float32);
            }
        }
    }
    false
}

// NOTE: pyo3-polars (current) does not pass kwargs to output_type_func.
fn tdigest_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    let in_dt = inputs
        .get(0)
        .map(|f| f.dtype())
        .cloned()
        .unwrap_or(DataType::Null);
    let dt = if matches!(in_dt, DataType::Float32) {
        TDigest::<f32>::polars_dtype()
    } else {
        TDigest::<f64>::polars_dtype()
    };
    Ok(Field::new("tdigest".into(), dt))
}

// merge keeps the incoming digest precision
fn merge_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    let dt = inputs
        .get(0)
        .map(|f| f.dtype())
        .cloned()
        .unwrap_or(DataType::Null);
    Ok(Field::new("tdigest".into(), dt))
}

// quantile returns f32 if input digest column is compact; else f64
fn quantile_output_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let dt = input_fields
        .get(0)
        .map(|f| f.dtype())
        .unwrap_or(&DataType::Null);
    let out = if is_compact_digest_dtype(dt) {
        DataType::Float32
    } else {
        DataType::Float64
    };
    Ok(Field::new("".into(), out))
}

/* ==================== UDFs ==================== */

#[polars_expr(output_type_func = cdf_output_dtype)]
fn cdf(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(inputs.len() == 2, ComputeError: "`cdf` expects (digest, values)");
    let digest_s = &inputs[0];
    let values_s = &inputs[1];

    if is_compact_digest_dtype(digest_s.dtype()) {
        cdf_generic::<f32>(digest_s, values_s)
    } else {
        cdf_generic::<f64>(digest_s, values_s)
    }
}

#[polars_expr(output_type_func = quantile_output_dtype)]
fn quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    quantile_impl(inputs, kwargs.q)
}

#[polars_expr(output_type = Float64)]
fn median(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_and_merge::<f64>(inputs);
    if td.is_empty() {
        Ok(Series::new("".into(), [None::<f64>]))
    } else {
        Ok(Series::new("".into(), [Some(td.median())]))
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    if is_compact_digest_dtype(inputs[0].dtype()) {
        let td = parse_and_merge::<f32>(inputs);
        td.to_series(inputs[0].name())
    } else {
        let td = parse_and_merge::<f64>(inputs);
        td.to_series(inputs[0].name())
    }
}

#[polars_expr(output_type_func = merge_output_dtype)]
fn add_values(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "`add_values` expects (digest, values)"
    );
    let digest_s = &inputs[0];
    let values_s = &inputs[1];

    if is_compact_digest_dtype(digest_s.dtype()) {
        add_values_generic::<f32>(digest_s, values_s)
    } else {
        add_values_generic::<f64>(digest_s, values_s)
    }
}

#[polars_expr(output_type_func = tdigest_output_dtype)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_from_array_helper(inputs, &kwargs)
}

// --- plain helper (no macro attribute), can live anywhere NOT immediately after a #[polars_expr] ---
fn upcast_tdigest<From, To>(src: &TDigest<From>) -> TDigest<To>
where
    From: Copy + 'static + FloatCore + FloatLike,
    To: Copy + 'static + FloatCore + FloatLike,
{
    use crate::tdigest::centroids::Centroid;
    use crate::tdigest::tdigest::DigestStats;

    let cents_to: Vec<Centroid<To>> = src
        .centroids()
        .iter()
        .map(|c| {
            if c.is_atomic_unit() {
                Centroid::<To>::new_atomic_unit_f64(c.mean_f64())
            } else if c.is_atomic() {
                Centroid::<To>::new_atomic_f64(c.mean_f64(), c.weight_f64())
            } else {
                Centroid::<To>::new_mixed_f64(c.mean_f64(), c.weight_f64())
            }
        })
        .collect();

    TDigest::<To>::builder()
        .max_size(src.max_size())
        .scale(src.scale())
        .singleton_policy(src.singleton_policy())
        .with_centroids_and_stats(
            cents_to,
            DigestStats {
                data_sum: src.sum(),
                total_weight: src.count(),
                data_min: src.min(),
                data_max: src.max(),
            },
        )
        .build()
}

#[polars_expr(output_type = Binary)]
fn to_bytes(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "`to_bytes` expects a single TDigest struct column"
    );
    let s = &inputs[0];

    // Use compact vs full precision based on digest dtype
    let td_bytes: Vec<u8> = if is_compact_digest_dtype(s.dtype()) {
        let td = parse_and_merge::<f32>(inputs);
        td.to_bytes()
    } else {
        let td = parse_and_merge::<f64>(inputs);
        td.to_bytes()
    };

    let mut builder = BinaryChunkedBuilder::new("".into(), 1);
    builder.append_value(&td_bytes);
    Ok(builder.finish().into_series())
}

// New: from_bytes output type — uses a precision *hint* column when present.
// Default with no hint: compact f32 schema (matches python_f32→polars_f32 tests).
fn from_bytes_output_dtype(inputs: &[Field]) -> PolarsResult<Field> {
    // inputs[0]: blob (Binary)
    // inputs[1]: optional precision hint (Float32 or Float64)
    let hint_dt = inputs
        .get(1)
        .map(|f| f.dtype())
        // default to Float32 (compact) when no hint is provided
        .unwrap_or(&DataType::Float32);

    let dt = if matches!(hint_dt, DataType::Float32) {
        TDigest::<f32>::polars_dtype()
    } else {
        TDigest::<f64>::polars_dtype()
    };

    Ok(Field::new("tdigest".into(), dt))
}

#[polars_expr(output_type_func = from_bytes_output_dtype)]
fn from_bytes(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() >= 1,
        ComputeError: "`from_bytes` expects a binary column"
    );
    let s = &inputs[0];

    if s.len() == 0 {
        let td = TDigest::<f64>::builder().build();
        return td.to_series("tdigest");
    }

    // Optional precision hint: second arg (Float32 → expect f32; otherwise f64).
    let hint_is_f32 = inputs
        .get(1)
        .map(|hint| matches!(hint.dtype(), DataType::Float32))
        .unwrap_or(false);

    // First, access the binary column to potentially sniff precision.
    let bin_sniff = s.binary().map_err(|e| {
        PolarsError::ComputeError(format!("from_bytes: expected Binary column, got {e}").into())
    })?;

    // Determine expected precision:
    //
    // - If a hint column is present: trust it (Float32 → F32, anything else → F64).
    // - If no hint: sniff the first non-null blob using wire_precision(...).
    let mut expected_prec: Option<WirePrecision> = if inputs.len() > 1 {
        if hint_is_f32 {
            Some(WirePrecision::F32)
        } else {
            Some(WirePrecision::F64)
        }
    } else {
        None
    };

    if expected_prec.is_none() {
        for opt in bin_sniff.clone().into_iter() {
            if let Some(bytes) = opt {
                match wire_precision(bytes) {
                    Ok(p) => {
                        expected_prec = Some(p);
                        break;
                    }
                    Err(e) => {
                        polars_bail!(ComputeError: "tdigest.from_bytes: {e}");
                    }
                }
            }
        }
    }

    // No non-null blobs (all nulls): fall back to an empty f64 digest column.
    let expected_prec = match expected_prec {
        Some(p) => p,
        None => {
            let td = TDigest::<f64>::builder().build();
            return td.to_series("tdigest");
        }
    };

    // Decode and merge, enforcing a uniform precision across the column.
    let bin = s.binary().map_err(|e| {
        PolarsError::ComputeError(format!("from_bytes: expected Binary column, got {e}").into())
    })?;

    match expected_prec {
        WirePrecision::F32 => {
            // All blobs must be f32; decode and merge
            let mut digests: Vec<TDigest<f32>> = Vec::new();

            for opt in bin.into_iter() {
                if let Some(bytes) = opt {
                    match wire_precision(bytes) {
                        Ok(WirePrecision::F32) => {
                            let decoded =
                                crate::tdigest::wire::decode_digest(bytes).map_err(|e| {
                                    PolarsError::ComputeError(
                                        format!("from_bytes decode error: {e}").into(),
                                    )
                                })?;
                            match decoded {
                                WireDecodedDigest::F32(td32) => digests.push(td32),
                                WireDecodedDigest::F64(_) => {
                                    polars_bail!(ComputeError:
                                        "tdigest.from_bytes: mixed f32/f64 blobs in column (expected f32)"
                                    );
                                }
                            }
                        }
                        Ok(WirePrecision::F64) => {
                            polars_bail!(ComputeError:
                                "tdigest.from_bytes: mixed f32/f64 blobs in column (expected f32)"
                            );
                        }
                        Err(e) => {
                            polars_bail!(ComputeError: "tdigest.from_bytes: {e}");
                        }
                    }
                }
            }

            let merged = TDigest::<f32>::merge_digests(digests);
            merged.to_series("tdigest")
        }
        WirePrecision::F64 => {
            // All blobs must be f64; decode and merge
            let mut digests: Vec<TDigest<f64>> = Vec::new();

            for opt in bin.into_iter() {
                if let Some(bytes) = opt {
                    match wire_precision(bytes) {
                        Ok(WirePrecision::F64) => {
                            let decoded =
                                crate::tdigest::wire::decode_digest(bytes).map_err(|e| {
                                    PolarsError::ComputeError(
                                        format!("from_bytes decode error: {e}").into(),
                                    )
                                })?;
                            match decoded {
                                WireDecodedDigest::F64(td64) => digests.push(td64),
                                WireDecodedDigest::F32(_) => {
                                    polars_bail!(ComputeError:
                                        "tdigest.from_bytes: mixed f32/f64 blobs in column (expected f64)"
                                    );
                                }
                            }
                        }
                        Ok(WirePrecision::F32) => {
                            polars_bail!(ComputeError:
                                "tdigest.from_bytes: mixed f32/f64 blobs in column (expected f64)"
                            );
                        }
                        Err(e) => {
                            polars_bail!(ComputeError: "tdigest.from_bytes: {e}");
                        }
                    }
                }
            }

            let merged = TDigest::<f64>::merge_digests(digests);
            merged.to_series("tdigest")
        }
    }
}

#[polars_expr(output_type = String)]
fn tdigest_summary(inputs: &[Series]) -> PolarsResult<Series> {
    // Always summarize in f64 for stable formatting (upcast if needed)
    let td = if is_compact_digest_dtype(inputs[0].dtype()) {
        upcast_tdigest::<f32, f64>(&parse_and_merge::<f32>(inputs))
    } else {
        parse_and_merge::<f64>(inputs)
    };

    let s = if td.is_empty() {
        None
    } else {
        Some(format!(
            "TDigest(n={:.0}, k={}, min={:.6}, p50={:.6}, max={:.6}, centroids={})",
            td.count(),
            td.max_size(),
            td.min(),
            td.quantile(0.5),
            td.max(),
            td.centroids().len(),
        ))
    };
    Ok(Series::new("".into(), [s]))
}

/* ==================== kwargs & output types ==================== */

#[derive(Debug, Deserialize, Clone)]
struct QuantileKwargs {
    /// Scalar probability in [0,1]
    q: f64,
}

#[inline]
fn default_max_size() -> usize {
    1000
}
#[inline]
fn default_scale() -> ScaleFamily {
    ScaleFamily::K2
}
#[inline]
fn default_singleton_mode() -> String {
    "use".to_string()
}
#[inline]
fn default_precision() -> String {
    "auto".to_string()
}

#[derive(Debug, Deserialize, Clone)]
struct TDigestKwargs {
    #[serde(default = "default_max_size")]
    max_size: usize,
    #[serde(default = "default_scale")]
    scale: ScaleFamily,
    #[serde(default = "default_singleton_mode")]
    singleton_mode: String,
    #[serde(
        default,
        alias = "edges_to_preserve",
        alias = "edges_per_side",
        alias = "edges",
        alias = "pin_total"
    )]
    pin_per_side: Option<usize>,
    #[serde(default = "default_precision")]
    precision: String, // "auto" | "f32" | "f64"
}

enum PrecisionKw {
    Auto,
    F32,
    F64,
}

fn parse_precision_kw(prec_raw: &str) -> PolarsResult<PrecisionKw> {
    match parse_precision_hint(prec_raw) {
        Ok(PrecisionHint::Auto) => Ok(PrecisionKw::Auto),
        Ok(PrecisionHint::F32) => Ok(PrecisionKw::F32),
        Ok(PrecisionHint::F64) => Ok(PrecisionKw::F64),
        Err(e) => Err(PolarsError::ComputeError(e.to_string().into())),
    }
}

/* ==================== core helpers (all generic) ==================== */

fn tdigest_from_array_helper(inputs: &[Series], kwargs: &TDigestKwargs) -> PolarsResult<Series> {
    let policy = parse_singleton_policy_str(Some(&kwargs.singleton_mode), kwargs.pin_per_side)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // Inference rule (matches planner): Float32 input → compact
    let input_is_f32 = matches!(inputs[0].dtype(), DataType::Float32);

    // If user provided `precision`, enforce it unless "auto".
    match parse_precision_kw(&kwargs.precision)? {
        PrecisionKw::Auto => {}
        PrecisionKw::F32 => {
            if !input_is_f32 {
                polars_bail!(ComputeError:
                    "precision=\"f32\" conflicts with input dtype {:?}. \
                     Schema is inferred from input values: Float32→compact(f32), otherwise f64.",
                    inputs[0].dtype());
            }
        }
        PrecisionKw::F64 => {
            if input_is_f32 {
                polars_bail!(ComputeError:
                    "precision=\"f64\" conflicts with input dtype {:?}. \
                     Schema is inferred from input values: Float32→compact(f32), otherwise f64.",
                    inputs[0].dtype());
            }
        }
    }

    if input_is_f32 {
        build_from_values::<f32>(&inputs[0], kwargs.max_size, kwargs.scale, policy)
            .and_then(|td| td.to_series(inputs[0].name()))
    } else {
        build_from_values::<f64>(&inputs[0], kwargs.max_size, kwargs.scale, policy)
            .and_then(|td| td.to_series(inputs[0].name()))
    }
}

fn build_from_values<F>(
    values: &Series,
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
) -> PolarsResult<TDigest<F>>
where
    F: FloatLike + FloatCore + WireOf,
{
    guard_no_upcast_f32_to_f64::<F>(values)?;

    // 1) Hard fail on ANY nulls (your test requires this semantics)
    if values.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "tdigest: input contains nulls; nulls are not allowed in training data".into(),
        ));
    }

    // 2) Extract numeric values AND reject non-finite (NaN/±inf)
    let vf: Vec<F> = match values.dtype() {
        DataType::Float64 => {
            let ca = values.f64()?;
            if ca
                .into_iter()
                .any(|opt| opt.map(|v| !v.is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            let ca = values.f64()?;
            ca.into_no_null_iter().map(F::from_f64).collect()
        }
        DataType::Float32 => {
            let ca = values.f32()?;
            if ca
                .into_iter()
                .any(|opt| opt.map(|v| !(v as f64).is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            let ca = values.f32()?;
            ca.into_no_null_iter()
                .map(|x| F::from_f64(x as f64))
                .collect()
        }
        dt if dt.is_numeric() => {
            let casted = values.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            if ca
                .into_iter()
                .any(|opt| opt.map(|v| !v.is_finite()).unwrap_or(false))
            {
                return Err(PolarsError::ComputeError(
                    "tdigest: input contains non-finite values (NaN or ±inf)".into(),
                ));
            }
            let ca = casted.f64()?;
            ca.into_no_null_iter().map(F::from_f64).collect()
        }
        other => {
            return Err(PolarsError::ComputeError(
                format!("tdigest: unsupported dtype {other:?}; expected a numeric column").into(),
            ))
        }
    };

    // 3) Empty training input → return an empty digest carrying the chosen params.
    //    Downstream:
    //      - quantile(...) → None (via quantile_impl)
    //      - cdf(...)      → NaN  (via cdf_or_nan)
    if vf.is_empty() {
        let td = TDigest::<F>::builder()
            .max_size(max_size)
            .scale(scale)
            .singleton_policy(policy)
            .build(); // empty digest
        return Ok(td);
    }

    let td0: TDigest<F> = TDigest::<F>::from_unsorted(&vf, max_size)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    if td0.scale() == scale && td0.singleton_policy() == policy {
        Ok(td0)
    } else {
        use crate::tdigest::centroids::Centroid;
        use crate::tdigest::tdigest::DigestStats;

        let cents: Vec<Centroid<F>> = td0
            .centroids()
            .iter()
            .map(|c| {
                if c.is_atomic_unit() {
                    Centroid::<F>::new_atomic_unit_f64(c.mean_f64())
                } else if c.is_atomic() {
                    Centroid::<F>::new_atomic_f64(c.mean_f64(), c.weight_f64())
                } else {
                    Centroid::<F>::new_mixed_f64(c.mean_f64(), c.weight_f64())
                }
            })
            .collect();

        Ok(TDigest::<F>::builder()
            .max_size(max_size)
            .scale(scale)
            .singleton_policy(policy)
            .with_centroids_and_stats(
                cents,
                DigestStats {
                    data_sum: td0.sum(),
                    total_weight: td0.count(),
                    data_min: td0.min(),
                    data_max: td0.max(),
                },
            )
            .build())
    }
}

fn parse_and_merge<F>(inputs: &[Series]) -> TDigest<F>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let s = &inputs[0];
    let parsed = TDigest::<F>::from_series(s).unwrap_or_default();
    TDigest::<F>::merge_digests(parsed)
}

fn add_values_generic<F>(digest_s: &Series, values_s: &Series) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let parsed = TDigest::<F>::from_series(digest_s).unwrap_or_default();
    let mut td = TDigest::<F>::merge_digests(parsed);
    let vals = extract_add_values::<F>(values_s)?;
    if !vals.is_empty() {
        td.add_many(vals)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }
    td.to_series(digest_s.name())
}

fn extract_add_values<F>(values: &Series) -> PolarsResult<Vec<F>>
where
    F: Copy + 'static + FloatCore + FloatLike,
{
    match values.dtype() {
        DataType::List(_) => {
            let lc = values.list()?;
            if lc.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values list contains nulls; nulls are not allowed");
            }

            let mut out: Vec<F> = Vec::new();
            for i in 0..lc.len() {
                let sub = lc
                    .get_as_series(i)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));
                if sub.null_count() > 0 {
                    polars_bail!(ComputeError:
                        "tdigest.add_values: values list contains nulls; nulls are not allowed");
                }
                let vals_f64 = subseries_to_f64_vec(&sub)?;
                if vals_f64.iter().any(|v| !v.is_finite()) {
                    polars_bail!(ComputeError:
                        "tdigest.add_values: values contain non-finite values (NaN or ±inf)");
                }
                out.extend(vals_f64.into_iter().map(F::from_f64));
            }
            Ok(out)
        }
        dt if dt.is_numeric() => {
            if values.null_count() > 0 {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values contain nulls; nulls are not allowed");
            }
            let vals_f64: Vec<f64> = if values.dtype() == &DataType::Float64 {
                values.f64()?.into_no_null_iter().collect()
            } else if values.dtype() == &DataType::Float32 {
                values
                    .f32()?
                    .into_no_null_iter()
                    .map(|x| x as f64)
                    .collect()
            } else {
                values
                    .cast(&DataType::Float64)?
                    .f64()?
                    .into_no_null_iter()
                    .collect()
            };
            if vals_f64.iter().any(|v| !v.is_finite()) {
                polars_bail!(ComputeError:
                    "tdigest.add_values: values contain non-finite values (NaN or ±inf)");
            }
            Ok(vals_f64.into_iter().map(F::from_f64).collect())
        }
        other => polars_bail!(ComputeError:
            "tdigest.add_values: unsupported values dtype {other:?}; expected numeric or list"),
    }
}

/* --------- tiny helper to build typed Series from native float families ---- */

trait OutFloat: PolarsNumericType {
    fn from_f64(x: f64) -> <Self as PolarsNumericType>::Native;
    fn series_from_vec(name: &str, v: Vec<<Self as PolarsNumericType>::Native>) -> Series;
}

impl OutFloat for Float32Type {
    #[inline]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn series_from_vec(name: &str, v: Vec<f32>) -> Series {
        Float32Chunked::from_vec(name.into(), v).into_series()
    }
}

impl OutFloat for Float64Type {
    #[inline]
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn series_from_vec(name: &str, v: Vec<f64>) -> Series {
        Float64Chunked::from_vec(name.into(), v).into_series()
    }
}

/* --------- generic CDF and quantile that specialize only by F -------------- */

fn cdf_generic<F>(digest_s: &Series, values_s: &Series) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let digests: Vec<TDigest<F>> = TDigest::<F>::from_series(digest_s).unwrap_or_default();
    let digest_len = digest_s.len();

    // Detect broadcast: exactly one non-empty digest across all rows
    let mut non_empty_iter = digests.iter().filter(|td| !td.is_empty());
    let single_non_empty = match (non_empty_iter.next(), non_empty_iter.next()) {
        (Some(td0), None) => Some(td0.clone()),
        _ => None,
    };

    match values_s.dtype() {
        DataType::List(inner) => {
            if matches!(inner.as_ref(), DataType::Float32) {
                cdf_on_list_out::<F, Float32Type>(
                    &digests,
                    digest_len,
                    values_s.list()?,
                    single_non_empty,
                )
            } else {
                cdf_on_list_out::<F, Float64Type>(
                    &digests,
                    digest_len,
                    values_s.list()?,
                    single_non_empty,
                )
            }
        }
        DataType::Float32 => {
            cdf_on_scalar_out::<F, Float32Type>(&digests, values_s, single_non_empty)
        }
        _ => cdf_on_scalar_out::<F, Float64Type>(&digests, values_s, single_non_empty),
    }
}

#[inline]
fn subseries_to_f64_vec(sub: &Series) -> PolarsResult<Vec<f64>> {
    if sub.dtype() == &DataType::Float64 {
        Ok(sub.f64()?.into_no_null_iter().collect())
    } else {
        let cast_sub = sub.cast(&DataType::Float64)?;
        Ok(cast_sub.f64()?.into_no_null_iter().collect())
    }
}

/// Scalar values column → output Series of the *Out* float family (f32/f64).
/// Strict policy: any NULL in the probe column → ERROR.
fn cdf_on_scalar_out<F, Out>(
    digests: &[TDigest<F>],
    values_s: &Series,
    single_non_empty: Option<TDigest<F>>,
) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike,
    Out: PolarsNumericType + OutFloat,
{
    // Strict null probe check
    if values_s.null_count() > 0 {
        polars_bail!(ComputeError: "tdigest.cdf: probe column contains nulls; nulls are not allowed");
    }

    // Convert to f64 vec (no nulls after the check above)
    let vals_f64: Vec<f64> = if values_s.dtype() == &DataType::Float64 {
        values_s.f64()?.into_no_null_iter().collect()
    } else if values_s.dtype() == &DataType::Float32 {
        values_s
            .f32()?
            .into_no_null_iter()
            .map(|x| x as f64)
            .collect()
    } else {
        values_s
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect()
    };

    // Compute using unified empty→NaN rule
    let mut out: Vec<<Out as PolarsNumericType>::Native> = Vec::with_capacity(values_s.len());

    if let Some(td0) = single_non_empty {
        for x in vals_f64 {
            out.push(Out::from_f64(td0.cdf_or_nan(&[x])[0]));
        }
    } else {
        for (i, x) in vals_f64.into_iter().enumerate() {
            let td = digests.get(i).cloned().unwrap_or_default();
            out.push(Out::from_f64(td.cdf_or_nan(&[x])[0]));
        }
    }

    Ok(Out::series_from_vec("", out))
}

/// List values column → output Series of lists with *Out* item dtype (f32/f64).
/// Strict policy: any NULL list cell or inner NULL → ERROR.
fn cdf_on_list_out<F, Out>(
    digests: &[TDigest<F>],
    digest_len: usize,
    lc: &ListChunked,
    single_non_empty: Option<TDigest<F>>,
) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike,
    Out: PolarsNumericType + OutFloat,
{
    // If any list element is null → ERROR (strict probe policy)
    if lc.null_count() > 0 {
        polars_bail!(ComputeError: "tdigest.cdf: probe list contains null values; nulls are not allowed");
    }

    // Fast-path: single digest row + single list row → flat vector output (not a list)
    if digest_len == 1 && lc.len() == 1 {
        let td = digests.get(0).cloned().unwrap_or_default();
        let sub = lc
            .get_as_series(0)
            .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

        if sub.null_count() > 0 {
            polars_bail!(ComputeError: "tdigest.cdf: probe list contains nulls; nulls are not allowed");
        }

        let vals = subseries_to_f64_vec(&sub)?;
        let out_native: Vec<<Out as PolarsNumericType>::Native> = td
            .cdf_or_nan(&vals)
            .into_iter()
            .map(Out::from_f64)
            .collect();
        return Ok(Out::series_from_vec("", out_native));
    }

    // General case: build list-per-row
    let mut rows: Vec<Series> = Vec::with_capacity(lc.len());
    for i in 0..lc.len() {
        let td = match &single_non_empty {
            Some(td0) => td0.clone(),
            None => digests.get(i).cloned().unwrap_or_default(),
        };
        let sub = lc
            .get_as_series(i)
            .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

        if sub.null_count() > 0 {
            polars_bail!(ComputeError: "tdigest.cdf: probe list contains nulls; nulls are not allowed");
        }

        let vals = subseries_to_f64_vec(&sub)?;
        let out_native: Vec<<Out as PolarsNumericType>::Native> = td
            .cdf_or_nan(&vals)
            .into_iter()
            .map(Out::from_f64)
            .collect();
        rows.push(Out::series_from_vec("", out_native));
    }

    let out_lc: ListChunked = rows.into_iter().collect();
    Ok(out_lc.into_series())
}

fn quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    if !q.is_finite() {
        polars_bail!(ComputeError: "q must be a finite number in [0, 1]");
    }
    if !(0.0..=1.0).contains(&q) {
        polars_bail!(ComputeError: "q must be in [0, 1]. Example: quantile(q=0.95) for the 95th percentile");
    }

    let want_f32 = is_compact_digest_dtype(inputs[0].dtype());

    if want_f32 {
        let td = parse_and_merge::<f32>(inputs);
        let is_empty = td.is_effectively_empty();
        if is_empty {
            return Ok(Series::new("".into(), [None::<f32>]));
        }
        let val = td.quantile(q);
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f32>]))
        } else {
            Ok(Series::new("".into(), [Some(val as f32)]))
        }
    } else {
        let td = parse_and_merge::<f64>(inputs);
        let is_empty = td.is_effectively_empty();
        if is_empty {
            return Ok(Series::new("".into(), [None::<f64>]));
        }
        let val = td.quantile(q);
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f64>]))
        } else {
            Ok(Series::new("".into(), [Some(val)]))
        }
    }
}
