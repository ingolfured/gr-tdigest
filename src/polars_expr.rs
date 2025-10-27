#![cfg(feature = "python")]
#![allow(clippy::unused_unit)]

use polars::prelude::ListChunked;
use polars::prelude::*;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use serde::Deserialize;

use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest};

const SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float64,
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

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

#[polars_expr(output_type_func = cdf_output_dtype)]
fn cdf(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(inputs.len() == 2, ComputeError: "`cdf` expects (digest, values)");
    let digest_s = &inputs[0];
    let values_s = &inputs[1];

    // Decode one TDigest per row (codec handles canonical/compact)
    let digests = TDigest::from_series(digest_s, false)
        .or_else(|_| TDigest::from_series(digest_s, true))
        .unwrap_or_default();

    let digest_len = digest_s.len();

    // Detect broadcast: exactly one non-empty digest across all rows
    let mut non_empty_iter = digests.iter().filter(|td| !td.is_empty());
    let single_non_empty = match (non_empty_iter.next(), non_empty_iter.next()) {
        (Some(td0), None) => Some(td0.clone()), // exactly one non-empty
        _ => None,
    };

    match values_s.dtype() {
        // Per-row list of query points → per-row list of CDFs
        DataType::List(inner) => {
            let inner_is_f32 = matches!(inner.as_ref(), DataType::Float32);
            let lc = values_s.list()?;

            // Special-case: single digest row + single list row → flat vector output
            if digest_len == 1 && lc.len() == 1 {
                let td = digests.get(0).cloned().unwrap_or_default();
                let sub: Series = lc
                    .get_as_series(0)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

                let vals_f64: Vec<f64> = if sub.dtype() == &DataType::Float64 {
                    sub.f64()?.into_no_null_iter().collect()
                } else {
                    let cast_sub: Series = sub.cast(&DataType::Float64)?;
                    cast_sub.f64()?.into_no_null_iter().collect()
                };

                let out_f64 = td.cdf(&vals_f64);
                return if inner_is_f32 {
                    Ok(Series::new(
                        "".into(),
                        out_f64.into_iter().map(|x| x as f32).collect::<Vec<f32>>(),
                    ))
                } else {
                    Ok(Series::new("".into(), out_f64))
                };
            }

            // General case: list-per-row output
            let mut rows: Vec<Series> = Vec::with_capacity(lc.len());
            for i in 0..lc.len() {
                let td = match &single_non_empty {
                    Some(td0) => td0.clone(),
                    None => digests.get(i).cloned().unwrap_or_default(),
                };

                let sub: Series = lc
                    .get_as_series(i)
                    .unwrap_or_else(|| Series::new("".into(), Vec::<f64>::new()));

                let vals_f64: Vec<f64> = if sub.dtype() == &DataType::Float64 {
                    sub.f64()?.into_no_null_iter().collect()
                } else {
                    let cast_sub: Series = sub.cast(&DataType::Float64)?;
                    cast_sub.f64()?.into_no_null_iter().collect()
                };

                let out_f64 = td.cdf(&vals_f64);
                if inner_is_f32 {
                    rows.push(Series::new(
                        "".into(),
                        out_f64.into_iter().map(|x| x as f32).collect::<Vec<f32>>(),
                    ));
                } else {
                    rows.push(Series::new("".into(), out_f64));
                }
            }

            let out_lc: ListChunked = rows.into_iter().collect();
            Ok(out_lc.into_series())
        }

        // Scalar Float32 column (or broadcasted literal) → Float32 output
        DataType::Float32 => {
            let vals = values_s.f32()?;
            let mut out: Vec<Option<f32>> = Vec::with_capacity(values_s.len());

            if let Some(td0) = single_non_empty {
                // Reuse the single non-empty digest for all rows (handles broadcast)
                for vx in vals {
                    out.push(vx.map(|x| td0.cdf(&[x as f64])[0] as f32));
                }
            } else {
                for (i, vx) in vals.into_iter().enumerate() {
                    let td = digests.get(i).cloned().unwrap_or_default();
                    out.push(vx.map(|x| td.cdf(&[x as f64])[0] as f32));
                }
            }
            Ok(Series::new("".into(), out))
        }

        // Scalar (default → Float64) column (or broadcasted literal) → Float64 output
        _ => {
            let cast: Series = values_s.cast(&DataType::Float64)?;
            let vals = cast.f64()?;
            let mut out: Vec<Option<f64>> = Vec::with_capacity(values_s.len());

            if let Some(td0) = single_non_empty {
                for vx in vals {
                    out.push(vx.map(|x| td0.cdf(&[x])[0]));
                }
            } else {
                for (i, vx) in vals.into_iter().enumerate() {
                    let td = digests.get(i).cloned().unwrap_or_default();
                    out.push(vx.map(|x| td.cdf(&[x])[0]));
                }
            }
            Ok(Series::new("".into(), out))
        }
    }
}

#[polars_expr(output_type_func = quantile_output_dtype)]
fn quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    quantile_impl(inputs, kwargs.q)
}

#[polars_expr(output_type = Float64)]
fn median(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    if td.is_empty() {
        Ok(Series::new("".into(), [None::<f64>]))
    } else {
        Ok(Series::new("".into(), [Some(td.median())]))
    }
}

/// Merge incoming digest rows → canonical (f64) schema for stability.
#[polars_expr(output_type_func = tdigest_output_f64)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    td.to_series(inputs[0].name(), /*compact=*/ false)
}

/// Public constructor symbol (canonical f64 codec) expected by tests as `tdigest`.
#[polars_expr(output_type_func = tdigest_output_f64)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_from_array_helper(inputs, &kwargs)
}

/// Public constructor symbol (compact f32 codec) expected by tests as `_tdigest_f32`.
#[polars_expr(output_type_func = tdigest_output_f32)]
fn _tdigest_f32(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_from_array_f32_helper(inputs, &kwargs)
}

#[polars_expr(output_type = String)]
fn tdigest_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
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

#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    /// Scalar probability in [0,1]
    q: f64,
}

// Defaults: max_size=1000, scale=K2, singleton_policy=Use
#[inline]
fn default_max_size() -> usize {
    1000
}
#[inline]
fn default_scale() -> ScaleFamily {
    ScaleFamily::K2
}
#[inline]
fn default_singleton_policy() -> SingletonPolicy {
    SingletonPolicy::Use
}

#[derive(Debug, Deserialize)]
struct TDigestKwargs {
    #[serde(default = "default_max_size")]
    max_size: usize,
    #[serde(default = "default_scale")]
    scale: ScaleFamily,
    #[serde(default = "default_singleton_policy")]
    singleton_policy: SingletonPolicy,
}

fn tdigest_output_f64(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "tdigest".into(),
        DataType::Struct(tdigest_fields_f64()),
    ))
}

fn tdigest_output_f32(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "tdigest".into(),
        DataType::Struct(tdigest_fields_f32()),
    ))
}

/* ==================== schema builders ==================== */

fn tdigest_fields_cfg(
    mean_dt: DataType,
    weight_dt: DataType,
    minmax_dt: DataType,
    sum_dt: DataType,
) -> Vec<Field> {
    vec![
        Field::new(
            "centroids".into(),
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new("mean".into(), mean_dt),
                Field::new("weight".into(), weight_dt),
            ]))),
        ),
        Field::new("sum".into(), sum_dt),
        Field::new("min".into(), minmax_dt.clone()),
        Field::new("max".into(), minmax_dt),
        Field::new("count".into(), DataType::Float64),
        Field::new("max_size".into(), DataType::Int64),
    ]
}

fn tdigest_fields_f64() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float64,
        DataType::Float64,
        DataType::Float64,
        DataType::Float64,
    )
}
fn tdigest_fields_f32() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Float64,
    )
}

/* ==================== compact detection for planning ==================== */

/// Decide compactness using only the "min" field’s dtype.
/// If it's Float32 → compact; otherwise canonical.
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

/// Output dtype for `quantile`: Float32 if input digest is compact, else Float64.
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

/* ==================== core helpers ==================== */

fn tdigest_from_array_helper(inputs: &[Series], kwargs: &TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(
        inputs,
        kwargs.max_size,
        kwargs.scale,
        kwargs.singleton_policy,
        /*compact=*/ false,
    )
}

fn tdigest_from_array_f32_helper(
    inputs: &[Series],
    kwargs: &TDigestKwargs,
) -> PolarsResult<Series> {
    tdigest_impl(
        inputs,
        kwargs.max_size,
        kwargs.scale,
        kwargs.singleton_policy,
        /*compact=*/ true,
    )
}

fn tdigest_impl(
    inputs: &[Series],
    max_size: usize,
    scale: ScaleFamily,
    singleton_policy: SingletonPolicy,
    compact: bool,
) -> PolarsResult<Series> {
    let mut td = tdigest_from_series(inputs, max_size, scale, singleton_policy)?;
    if td.is_empty() {
        td = TDigest::builder()
            .max_size(max_size)
            .scale(scale)
            .singleton_policy(singleton_policy)
            .build();
    }
    td.to_series(inputs[0].name(), compact)
}

fn tdigest_from_series(
    inputs: &[Series],
    max_size: usize,
    scale: ScaleFamily,
    singleton_policy: SingletonPolicy,
) -> PolarsResult<TDigest> {
    let series = &inputs[0];

    if !SUPPORTED_TYPES.contains(series.dtype()) {
        polars_bail!(InvalidOperation: "tdigest_from_array: only numerical types are supported");
    }

    let series_casted: Series = if series.dtype() == &DataType::Float64 {
        series.clone()
    } else {
        series.cast(&DataType::Float64)?
    };

    let values = series_casted.f64()?;
    let chunks: Vec<TDigest> = POOL.install(|| {
        values
            .downcast_iter()
            .par_bridge()
            .map(|chunk: &Float64Array| {
                let t = TDigest::builder()
                    .max_size(max_size)
                    .scale(scale)
                    .singleton_policy(singleton_policy)
                    .build();
                t.merge_unsorted(chunk.non_null_values_iter().collect())
            })
            .collect()
    });

    Ok(TDigest::merge_digests(chunks))
}

/// Parse a TDigest struct column via unified strict API:
/// try canonical first, then compact (so callers needn’t know the on-wire precision).
fn parse_tdigest(inputs: &[Series]) -> TDigest {
    let s = &inputs[0];
    let parsed = TDigest::from_series(s, /*compact=*/ false)
        .or_else(|_| TDigest::from_series(s, /*compact=*/ true))
        .unwrap_or_default();
    TDigest::merge_digests(parsed)
}

fn quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);

    // Decide requested dtype from the digest’s dtype
    let digest_dt = inputs[0].dtype();
    let want_f32 = is_compact_digest_dtype(digest_dt);

    if td.is_empty() {
        if want_f32 {
            return Ok(Series::new("".into(), [None::<f32>]));
        } else {
            return Ok(Series::new("".into(), [None::<f64>]));
        }
    }

    let val = td.quantile(q);
    if want_f32 {
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f32>]))
        } else {
            Ok(Series::new("".into(), [Some(val as f32)]))
        }
    } else {
        if val.is_nan() {
            Ok(Series::new("".into(), [None::<f64>]))
        } else {
            Ok(Series::new("".into(), [Some(val)]))
        }
    }
}
