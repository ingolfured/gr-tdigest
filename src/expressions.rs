#![allow(clippy::unused_unit)]

use polars::prelude::*;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use serde::Deserialize;

use crate::tdigest::codecs::{parse_tdigests_any, tdigest_to_series, tdigest_to_series_32};
use crate::tdigest::{ScaleFamily, TDigest};

const SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

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

#[polars_expr(output_type_func = tdigest_output)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    merge_tdigests_impl(inputs)
}

#[polars_expr(output_type_func = tdigest_output)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(inputs, kwargs.max_size, kwargs.scale)
}

#[polars_expr(output_type_func = tdigest_output_32)]
fn tdigest_32(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_32_impl(inputs, kwargs.max_size, kwargs.scale)
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

#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    quantile: f64,
}

#[derive(Debug, Deserialize)]
struct CDFKwargs {
    xs: Vec<f64>,
}

// Defaults: max_size=1000, scale=K2 (serde accepts "quad","k1","k2","k3" thanks to #[serde(rename_all="lowercase")] on ScaleFamily)
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

fn tdigest_fields() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float64,
        DataType::Float64,
        DataType::Float64,
        DataType::Float64,
    )
}

fn tdigest_fields_32() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Float64,
    )
}

fn tdigest_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "tdigest".into(),
        DataType::Struct(tdigest_fields()),
    ))
}

fn tdigest_output_32(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "tdigest".into(),
        DataType::Struct(tdigest_fields_32()),
    ))
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

fn parse_tdigest(inputs: &[Series]) -> TDigest {
    let tdigests: Vec<TDigest> = parse_tdigests_any(&inputs[0]);
    TDigest::merge_digests(tdigests)
}

fn tdigest_impl_generic<F>(
    inputs: &[Series],
    max_size: usize,
    scale: ScaleFamily,
    to_series: F,
) -> PolarsResult<Series>
where
    F: FnOnce(TDigest, &str) -> Series,
{
    let mut td = tdigest_from_series(inputs, max_size, scale)?;
    if td.is_empty() {
        td = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0);
    }
    Ok(to_series(td, inputs[0].name()))
}

fn tdigest_impl(inputs: &[Series], max_size: usize, scale: ScaleFamily) -> PolarsResult<Series> {
    tdigest_impl_generic(inputs, max_size, scale, tdigest_to_series)
}

fn tdigest_32_impl(inputs: &[Series], max_size: usize, scale: ScaleFamily) -> PolarsResult<Series> {
    tdigest_impl_generic(inputs, max_size, scale, tdigest_to_series_32)
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

fn merge_tdigests_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    Ok(tdigest_to_series(td, inputs[0].name()))
}
