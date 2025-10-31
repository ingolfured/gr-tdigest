#![cfg(feature = "python")]
#![allow(clippy::unused_unit)]

use std::any::TypeId;

use ordered_float::FloatCore;
use polars::prelude::ListChunked;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::tdigest::codecs::WireOf;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest};

/* ==================== input dtypes ==================== */

#[inline]
fn guard_no_upcast_f32_to_f64<F: 'static>(values: &Series) -> PolarsResult<()> {
    // If the input column is Float32 *and* the requested storage is f64,
    // reject with a crisp, actionable error.
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
// So we infer output schema from the *input value dtype*:
//   - Float32 input → TDigest<f32> schema (compact)
//   - otherwise     → TDigest<f64> schema
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

#[polars_expr(output_type_func = tdigest_output_dtype)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_from_array_helper(inputs, &kwargs)
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

// Defaults: max_size=1000, scale=K2, singleton_mode="use"
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
    // accept "auto" from callers (Python wrapper uses this)
    "auto".to_string()
}

#[derive(Debug, Deserialize, Clone)]
struct TDigestKwargs {
    #[serde(default = "default_max_size")]
    max_size: usize,
    #[serde(default = "default_scale")]
    scale: ScaleFamily,
    // accept forgiving string; we map it to SingletonPolicy ourselves
    #[serde(default = "default_singleton_mode")]
    singleton_mode: String,
    // consolidated public name; accept legacy wire aliases for compatibility
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

/// parse "auto" | "f32" | "f64"
fn parse_precision_kw(prec_raw: &str) -> PolarsResult<PrecisionKw> {
    let p = prec_raw.trim().to_lowercase();
    match p.as_str() {
        "auto" => Ok(PrecisionKw::Auto),
        "f32" => Ok(PrecisionKw::F32),
        "f64" => Ok(PrecisionKw::F64),
        _ => Err(PolarsError::ComputeError(
            format!("unknown precision={prec_raw:?}; expected 'auto', 'f32' or 'f64'").into(),
        )),
    }
}

/* ==================== core helpers (all generic) ==================== */

fn tdigest_from_array_helper(inputs: &[Series], kwargs: &TDigestKwargs) -> PolarsResult<Series> {
    let policy = parse_singleton_policy(&kwargs.singleton_mode, kwargs.pin_per_side)?;

    // Inference rule (matches planner): Float32 input → compact
    let input_is_f32 = matches!(inputs[0].dtype(), DataType::Float32);

    // If user provided `precision`, enforce it unless "auto".
    match parse_precision_kw(&kwargs.precision)? {
        PrecisionKw::Auto => {
            // no-op, accept planner inference
        }
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
    // Forbid Float32 input when F == f64
    guard_no_upcast_f32_to_f64::<F>(values)?;

    // Materialize input as Vec<F>
    let vf: Vec<F> = match values.dtype() {
        DataType::Float64 => {
            let ca = values.f64()?;
            ca.into_no_null_iter().map(F::from_f64).collect()
        }
        DataType::Float32 => {
            let ca = values.f32()?;
            ca.into_no_null_iter()
                .map(|x| F::from_f64(x as f64))
                .collect()
        }
        dt if dt.is_numeric() => {
            // cast to f64 first for integers/others
            let casted = values.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            ca.into_no_null_iter().map(F::from_f64).collect()
        }
        other => {
            return Err(PolarsError::ComputeError(
                format!("tdigest: unsupported dtype {other:?}; expected a numeric column").into(),
            ))
        }
    };

    // Build with the requested max_size, then reconfigure scale/policy via builder
    let td0: TDigest<F> = TDigest::<F>::from_unsorted(&vf, max_size)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // If defaults already match, keep it; else rebuild with desired options
    if td0.scale() == scale && td0.singleton_policy() == policy {
        Ok(td0)
    } else {
        use crate::tdigest::centroids::Centroid;
        use crate::tdigest::tdigest::DigestStats;

        let cents: Vec<Centroid<F>> = td0
            .centroids()
            .iter()
            .map(|c| Centroid::<F>::new(c.mean_f64(), c.weight_f64()))
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

/// Parse a TDigest struct column and merge all rows into a single TDigest<F>.
fn parse_and_merge<F>(inputs: &[Series]) -> TDigest<F>
where
    F: Copy + 'static + FloatCore + FloatLike + WireOf,
{
    let s = &inputs[0];
    let parsed = TDigest::<F>::from_series(s).unwrap_or_default();
    TDigest::<F>::merge_digests(parsed)
}

/// Upcast TDigest<From> → TDigest<To> via centroids/stats.
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
        .map(|c| Centroid::<To>::new(c.mean_f64(), c.weight_f64()))
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

/* --------- tiny helper to build typed Series from native float families ---- */

trait OutFloat: PolarsNumericType {
    fn from_f64(x: f64) -> <Self as PolarsNumericType>::Native;
    fn series_from_options(
        name: &str,
        v: Vec<Option<<Self as PolarsNumericType>::Native>>,
    ) -> Series;
    fn series_from_vec(name: &str, v: Vec<<Self as PolarsNumericType>::Native>) -> Series;
}

impl OutFloat for Float32Type {
    #[inline]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn series_from_options(name: &str, v: Vec<Option<f32>>) -> Series {
        Float32Chunked::from_iter_options(name.into(), v.into_iter()).into_series()
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
    fn series_from_options(name: &str, v: Vec<Option<f64>>) -> Series {
        Float64Chunked::from_iter_options(name.into(), v.into_iter()).into_series()
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
fn cdf_on_scalar_out<F, Out>(
    digests: &[TDigest<F>],
    values_s: &Series,
    single_non_empty: Option<TDigest<F>>,
) -> PolarsResult<Series>
where
    F: Copy + 'static + FloatCore + FloatLike,
    Out: PolarsNumericType + OutFloat,
{
    // Evaluate all queries in f64; cast to Out::Native at the end.
    let vals_f64: Vec<Option<f64>> = if values_s.dtype() == &DataType::Float64 {
        values_s.f64()?.into_iter().collect()
    } else if values_s.dtype() == &DataType::Float32 {
        values_s
            .f32()?
            .into_iter()
            .map(|o| o.map(|x| x as f64))
            .collect()
    } else {
        values_s
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .collect()
    };

    let mut out: Vec<Option<<Out as PolarsNumericType>::Native>> =
        Vec::with_capacity(values_s.len());

    if let Some(td0) = single_non_empty {
        for vx in vals_f64 {
            out.push(vx.map(|x| Out::from_f64(td0.cdf(&[x])[0])));
        }
    } else {
        for (i, vx) in vals_f64.into_iter().enumerate() {
            let td = digests.get(i).cloned().unwrap_or_default();
            out.push(vx.map(|x| Out::from_f64(td.cdf(&[x])[0])));
        }
    }

    Ok(Out::series_from_options("", out))
}

/// List values column → output Series of lists with *Out* item dtype (f32/f64).
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
    // Fast-path: single digest row + single list row → flat vector output (not a list)
    if digest_len == 1 && lc.len() == 1 {
        let td = digests.get(0).cloned().unwrap_or_default();
        let sub = lc
            .get_as_series(0)
            .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));
        let vals = subseries_to_f64_vec(&sub)?;
        let out_native: Vec<<Out as PolarsNumericType>::Native> =
            td.cdf(&vals).into_iter().map(Out::from_f64).collect();
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
        let vals = subseries_to_f64_vec(&sub)?;
        let out_native: Vec<<Out as PolarsNumericType>::Native> =
            td.cdf(&vals).into_iter().map(Out::from_f64).collect();
        rows.push(Out::series_from_vec("", out_native));
    }

    let out_lc: ListChunked = rows.into_iter().collect();
    Ok(out_lc.into_series())
}

fn quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    if !(0.0..=1.0).contains(&q) {
        polars_bail!(ComputeError: "q must be in [0, 1]. Example: quantile(q=0.95) for the 95th percentile");
    }

    let want_f32 = is_compact_digest_dtype(inputs[0].dtype());

    if want_f32 {
        let td = parse_and_merge::<f32>(inputs);
        if td.is_empty() {
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
        if td.is_empty() {
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

/* ==================== policy parsing ==================== */

fn parse_singleton_policy(mode_raw: &str, k_opt: Option<usize>) -> PolarsResult<SingletonPolicy> {
    let m = mode_raw
        .trim()
        .to_lowercase()
        .replace('_', "")
        .replace(' ', "");
    match m.as_str() {
        "off" => Ok(SingletonPolicy::Off),
        "use" => Ok(SingletonPolicy::Use),
        "edges" | "usewithprotectededges" => {
            let k = k_opt.ok_or_else(|| {
                PolarsError::ComputeError(
                    "singleton_mode='edges' requires 'pin_per_side' (per side, >= 1)".into(),
                )
            })?;
            if k < 1 {
                polars_bail!(ComputeError: "pin_per_side must be >= 1 when singleton_mode='edges'");
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(k))
        }
        _ => Err(PolarsError::ComputeError(
            format!("unknown singleton_mode={mode_raw:?}; expected 'off'|'use'|'edges'").into(),
        )),
    }
}
