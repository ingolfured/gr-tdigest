#![cfg(feature = "python")]
#![allow(clippy::unused_unit)]

use polars::prelude::*;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use serde::Deserialize;

use crate::tdigest::{ScaleFamily, TDigest};

const SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

/* ==================== user-facing exprs ==================== */

#[polars_expr(output_type = Float64)]
fn estimate_cdf(inputs: &[Series], kwargs: CDFKwargs) -> PolarsResult<Series> {
    estimate_cdf_impl(inputs, kwargs.xs)
}

#[polars_expr(output_type = Float64)]
fn estimate_median(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    if td.is_empty() {
        Ok(Series::new("".into(), [None::<f64>]))
    } else {
        Ok(Series::new("".into(), [Some(td.estimate_median())]))
    }
}

#[polars_expr(output_type = Float64)]
fn estimate_quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    estimate_quantile_impl(inputs, kwargs.quantile)
}

// merge → canonical (f64) schema for stability
#[polars_expr(output_type_func = tdigest_output_f64)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    td.try_to_series(inputs[0].name())
}

// canonical f64 storage
#[polars_expr(output_type_func = tdigest_output_f64)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(inputs, kwargs.max_size, kwargs.scale, Storage::F64)
}

// compact f32 storage (internal; called from Python by name)
#[polars_expr(output_type_func = tdigest_output_f32)]
fn _tdigest_f32(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(inputs, kwargs.max_size, kwargs.scale, Storage::F32)
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
            td.estimate_quantile(0.5),
            td.max(),
            td.centroids().len(),
        ))
    };
    Ok(Series::new("".into(), [s]))
}

/* ==================== kwargs & output types ==================== */

#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    quantile: f64,
}

#[derive(Debug, Deserialize)]
struct CDFKwargs {
    xs: Vec<f64>,
}

// Defaults: max_size=1000, scale=K2
#[inline]
fn default_max_size() -> usize {
    1000
}
#[inline]
fn default_scale() -> ScaleFamily {
    ScaleFamily::K2
}

#[derive(Debug, Deserialize)]
struct TDigestKwargs {
    #[serde(default = "default_max_size")]
    max_size: usize,
    #[serde(default = "default_scale")]
    scale: ScaleFamily,
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

/* ==================== core helpers ==================== */

#[derive(Copy, Clone)]
enum Storage {
    F64,
    F32,
}

fn tdigest_impl(
    inputs: &[Series],
    max_size: usize,
    scale: ScaleFamily,
    storage: Storage,
) -> PolarsResult<Series> {
    let mut td = tdigest_from_series(inputs, max_size, scale)?;
    if td.is_empty() {
        td = TDigest::new(Vec::new(), 0.0, 0.0, 0.0, 0.0, 0);
    }
    match storage {
        Storage::F64 => td.try_to_series(inputs[0].name()),
        Storage::F32 => td.try_to_series_compact(inputs[0].name()),
    }
}

fn tdigest_from_series(
    inputs: &[Series],
    max_size: usize,
    scale: ScaleFamily,
) -> PolarsResult<TDigest> {
    let series = &inputs[0];
    let series_casted: Series = if series.dtype() == &DataType::Float64 {
        series.clone()
    } else {
        if !SUPPORTED_TYPES.contains(series.dtype()) {
            polars_bail!(InvalidOperation: "tdigest: only supported for numerical types");
        }
        series.cast(&DataType::Float64)?
    };

    let values = series_casted.f64()?;
    let chunks: Vec<TDigest> = POOL.install(|| {
        values
            .downcast_iter()
            .par_bridge()
            .map(|chunk: &Float64Array| {
                let t = TDigest::new_with_size_and_scale(max_size, scale);
                t.merge_unsorted(chunk.non_null_values_iter().collect())
            })
            .collect()
    });

    Ok(TDigest::merge_digests(chunks))
}

/// Parse a TDigest struct column via public strict APIs:
///   try canonical first, then compact.
fn parse_tdigest(inputs: &[Series]) -> TDigest {
    let s = &inputs[0];
    let parsed = TDigest::try_from_series(s)
        .or_else(|_| TDigest::try_from_series_compact(s))
        .unwrap_or_default();
    TDigest::merge_digests(parsed)
}

fn estimate_quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    let out = if td.is_empty() {
        [None]
    } else {
        let val = td.estimate_quantile(q);
        if val.is_nan() {
            [None]
        } else {
            [Some(val)]
        }
    };
    Ok(Series::new("".into(), out))
}

fn estimate_cdf_impl(inputs: &[Series], xs: Vec<f64>) -> PolarsResult<Series> {
    let td = parse_tdigest(&inputs[..1]);
    if td.is_empty() {
        return Ok(Series::new("".into(), [None::<f64>]));
    }
    let out = td.estimate_cdf(&xs);
    Ok(Series::new("".into(), out))
}
