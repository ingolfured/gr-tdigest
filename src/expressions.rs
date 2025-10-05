#![allow(clippy::unused_unit)]
#![cfg_attr(all(test, not(feature = "python")), allow(dead_code))]
#![cfg(any(test, feature = "python"))]

#[cfg(feature = "python")]
use pyo3_polars::derive::polars_expr;

use crate::tdigest::codecs::parse_tdigests_any;
use crate::tdigest::codecs::tdigest_to_series_32;
use crate::tdigest::{codecs::tdigest_to_series, TDigest};
use polars::prelude::*;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use serde::Deserialize;

const SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

#[cfg_attr(feature = "python", polars_expr(output_type = Float64))]
fn estimate_cdf(inputs: &[Series], kwargs: CDFKwargs) -> PolarsResult<Series> {
    estimate_cdf_impl(inputs, kwargs.xs)
}

#[cfg_attr(feature = "python", polars_expr(output_type = Float64))]
fn estimate_median(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    if td.is_empty() {
        Ok(Series::new("".into(), [None::<f64>]))
    } else {
        Ok(Series::new("".into(), [Some(td.estimate_median())]))
    }
}

#[cfg_attr(feature = "python", polars_expr(output_type = Float64))]
fn estimate_quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    estimate_quantile_impl(inputs, kwargs.quantile)
}

#[cfg_attr(feature = "python", polars_expr(output_type_func = tdigest_output))]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    merge_tdigests_impl(inputs)
}

#[cfg_attr(feature = "python", polars_expr(output_type_func = tdigest_output))]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(inputs, kwargs.max_size)
}

#[cfg_attr(feature = "python", polars_expr(output_type_func = tdigest_output_32))]
fn tdigest_32(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_32_impl(inputs, kwargs.max_size)
}

#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    quantile: f64,
}

#[derive(Debug, Deserialize)]
struct CDFKwargs {
    xs: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct TDigestKwargs {
    max_size: usize,
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
        Field::new("count".into(), DataType::Int64),
        Field::new("max_size".into(), DataType::Int64),
    ]
}

fn tdigest_fields() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float64,
        DataType::Int64,
        DataType::Float64,
        DataType::Float64,
    )
}

fn tdigest_fields_32() -> Vec<Field> {
    tdigest_fields_cfg(
        DataType::Float32,
        DataType::UInt32,
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

fn tdigest_from_series(inputs: &[Series], max_size: usize) -> PolarsResult<TDigest> {
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
                let t = TDigest::new_with_size(max_size);
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

fn tdigest_impl(inputs: &[Series], max_size: usize) -> PolarsResult<Series> {
    let mut td = tdigest_from_series(inputs, max_size)?;
    if td.is_empty() {
        td = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0)
    }
    Ok(tdigest_to_series(td, inputs[0].name()))
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

fn tdigest_32_impl(inputs: &[Series], max_size: usize) -> PolarsResult<Series> {
    let mut td = tdigest_from_series(inputs, max_size)?;
    if td.is_empty() {
        td = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0);
    }
    Ok(tdigest_to_series_32(td, inputs[0].name()))
}

fn merge_tdigests_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    Ok(tdigest_to_series(td, inputs[0].name()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::test_helpers::{assert_abs_close, assert_exact, assert_rel_close};

    #[test]
    fn expr_impl_smoke_minimal() {
        let inputs = [Series::new("n".into(), [1, 2, 3])
            .cast(&DataType::Int32)
            .unwrap()];
        let td_ser = super::tdigest_impl(&inputs, 100).unwrap();
        let q_ser = super::estimate_quantile_impl(&[td_ser.clone()], 0.5).unwrap();
        let q = q_ser.f64().unwrap().get(0).unwrap();
        assert_exact("estimate_quantile_impl median = {q}, expected 2.0", 2.0, q);
        let cdf_ser = super::estimate_cdf_impl(&[td_ser], vec![2.0_f64]).unwrap();
        let cdf = cdf_ser.f64().unwrap().get(0).unwrap();
        assert_exact("estimate_cdf_impl CDF(2.0) = {cdf}, expected 0.5", 0.5, cdf);
    }

    #[test]
    fn expr_impl_empty_returns_none() {
        let empty_inputs = [Series::new("n".into(), Vec::<i32>::new())
            .cast(&DataType::Int32)
            .unwrap()];
        let td_ser = super::tdigest_impl(&empty_inputs, 100).unwrap();
        let q_ser = super::estimate_quantile_impl(&[td_ser], 0.5).unwrap();
        assert!(
            q_ser.f64().unwrap().get(0).is_none(),
            "expected None for empty digest"
        );
    }

    #[test]
    fn f32_vs_f64_parser_equivalence() {
        let inputs = [Series::new("n".into(), (1..=1000).collect::<Vec<i32>>())
            .cast(&DataType::Int32)
            .unwrap()];
        let s64 = super::tdigest_impl(&inputs, 100).unwrap();
        let s32 = super::tdigest_32_impl(&inputs, 100).unwrap();
        let td64 = parse_tdigests_any(&s64).into_iter().next().unwrap();
        let td32 = parse_tdigests_any(&s32).into_iter().next().unwrap();
        let tol = 1e-6;
        assert_rel_close(
            "p50 f32 vs f64",
            td64.estimate_quantile(0.5),
            td32.estimate_quantile(0.5),
            tol,
        );
        assert_rel_close(
            "CDF(500) f32 vs f64",
            td64.estimate_cdf(&[500.0])[0],
            td32.estimate_cdf(&[500.0])[0],
            tol,
        );
    }

    #[test]
    fn mixed_origin_core_merge() {
        let inputs = [Series::new("n".into(), (1..=1000).collect::<Vec<i32>>())
            .cast(&DataType::Int32)
            .unwrap()];
        let td64 = parse_tdigests_any(&super::tdigest_impl(&inputs, 100).unwrap())
            .into_iter()
            .next()
            .unwrap();
        let td32 = parse_tdigests_any(&super::tdigest_32_impl(&inputs, 100).unwrap())
            .into_iter()
            .next()
            .unwrap();
        let merged = TDigest::merge_digests(vec![td64, td32]);
        let p50 = merged.estimate_quantile(0.5);
        assert_abs_close("mixed_origin_core_merge p50", 500.0, p50, 100.0);
        let cdf_500 = merged.estimate_cdf(&[500.0])[0];
        assert_rel_close("mixed_origin_core_merge cdf_500", 0.5, cdf_500, 0.2);
    }
}
