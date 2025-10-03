#![allow(clippy::unused_unit)]
use polars::prelude::*;

use crate::tdigest::{codecs::parse_tdigests, codecs::tdigest_to_series, TDigest};

use polars_core::export::rayon::prelude::*;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

static SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

// TODO: get rid of serde completely
#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    quantile: f64,
}

#[derive(Debug, Deserialize)]
struct TDigestKwargs {
    max_size: usize,
}

fn tdigest_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("tdigest", DataType::Struct(tdigest_fields())))
}

fn tdigest_fields() -> Vec<Field> {
    vec![
        Field::new(
            "centroids",
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new("mean", DataType::Float64),
                Field::new("weight", DataType::Int64),
            ]))),
        ),
        Field::new("sum", DataType::Float64),
        Field::new("min", DataType::Float64),
        Field::new("max", DataType::Float64),
        Field::new("count", DataType::Int64),
        Field::new("max_size", DataType::Int64),
    ]
}

// ------------------------------- core helpers -------------------------------

fn tdigest_from_series(inputs: &[Series], max_size: usize) -> PolarsResult<TDigest> {
    let series = &inputs[0];
    let series_casted: &Series = if series.dtype() == &DataType::Float64 {
        series
    } else {
        if !SUPPORTED_TYPES.contains(series.dtype()) {
            polars_bail!(InvalidOperation: "only supported for numerical types");
        }
        let cast_result = series.cast(&DataType::Float64);
        if cast_result.is_err() {
            polars_bail!(InvalidOperation: "only supported for numerical types");
        }
        &cast_result.unwrap()
    };

    let values = series_casted.f64()?;
    let chunks: Vec<TDigest> = POOL.install(|| {
        values
            .downcast_iter()
            .par_bridge()
            .map(|chunk| {
                let t = TDigest::new_with_size(max_size);
                let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                t.merge_unsorted(array.non_null_values_iter().collect())
            })
            .collect::<Vec<TDigest>>()
    });

    Ok(TDigest::merge_digests(chunks))
}

fn parse_tdigest(inputs: &[Series]) -> TDigest {
    let tdigests: Vec<TDigest> = parse_tdigests(&inputs[0]);
    TDigest::merge_digests(tdigests)
}

// ------------------------------- testable impls ------------------------------

/// Plain, testable implementation: returns the **Series** form of the digest.
pub(crate) fn tdigest_impl(inputs: &[Series], max_size: usize) -> PolarsResult<Series> {
    let mut td = tdigest_from_series(inputs, max_size)?;
    if td.is_empty() {
        // avoid NaN default
        td = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0)
    }
    Ok(tdigest_to_series(td, inputs[0].name()))
}

/// Plain, testable implementation of estimate_quantile expr.
pub(crate) fn estimate_quantile_impl(inputs: &[Series], q: f64) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    let out = if td.is_empty() {
        [None]
    } else {
        [Some(td.estimate_quantile(q))]
    };
    Ok(Series::new("", out))
}

/// Plain, testable implementation of estimate_cdf expr.
pub(crate) fn estimate_cdf_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(&inputs[..1]);
    if td.is_empty() {
        return Ok(Series::new("", [None::<f64>]));
    }

    let x_series = &inputs[1];
    let x_values = x_series.f64()?;
    let xs: Vec<Option<f64>> = x_values.into_iter().collect();

    let out: Vec<Option<f64>> = if xs.iter().any(|v| v.is_none()) {
        xs.iter()
            .map(|o| o.map(|v| td.estimate_cdf(&[v])[0]))
            .collect()
    } else {
        let vals: Vec<f64> = xs.iter().map(|o| o.unwrap()).collect();
        td.estimate_cdf(&vals).into_iter().map(Some).collect()
    };

    Ok(Series::new("", out))
}

// --------------------------- proc-macro forwarders ---------------------------

#[polars_expr(output_type_func=tdigest_output)]
pub(crate) fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    tdigest_impl(inputs, kwargs.max_size)
}

#[polars_expr(output_type_func=tdigest_output)]
pub(crate) fn tdigest_cast(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    let td = tdigest_from_series(inputs, kwargs.max_size)?;
    Ok(tdigest_to_series(td, inputs[0].name()))
}

#[polars_expr(output_type_func=tdigest_output)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    Ok(tdigest_to_series(td, inputs[0].name()))
}

#[polars_expr(output_type=Float64)]
pub(crate) fn estimate_quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    estimate_quantile_impl(inputs, kwargs.quantile)
}

#[polars_expr(output_type=Float64)]
pub(crate) fn estimate_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    estimate_cdf_impl(inputs)
}

#[polars_expr(output_type=Float64)]
pub(crate) fn estimate_median(inputs: &[Series]) -> PolarsResult<Series> {
    let td = parse_tdigest(inputs);
    if td.is_empty() {
        Ok(Series::new("", [None::<f64>]))
    } else {
        Ok(Series::new("", [Some(td.estimate_median())]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal smoke: build digest (impl), then query via quantile + cdf (impls).
    #[test]
    fn expr_impl_smoke_minimal() {
        // Keep it lean: one supported dtype.
        let inputs = [Series::new("n", [1, 2, 3]).cast(&DataType::Int32).unwrap()];

        // Build digest via impl
        let td_ser = tdigest_impl(&inputs, 100).unwrap();

        // Quantile impl: median should be exactly 2.0
        let q_ser = estimate_quantile_impl(&[td_ser.clone()], 0.5).unwrap();
        let q = q_ser.f64().unwrap().get(0).unwrap();
        assert!(
            (q - 2.0).abs() < 1e-12,
            "estimate_quantile_impl median = {q}, expected 2.0"
        );

        // CDF impl: midpoint convention ⇒ CDF(2.0) = 0.5
        let cdf_ser = estimate_cdf_impl(&[td_ser, Series::new("", &[2.0_f64])]).unwrap();
        let cdf = cdf_ser.f64().unwrap().get(0).unwrap();
        assert!(
            (cdf - 0.5).abs() < 1e-12,
            "estimate_cdf_impl CDF(2.0) = {cdf}, expected 0.5"
        );
    }

    /// Empty input ⇒ empty digest ⇒ quantile impl returns None.
    #[test]
    fn expr_impl_empty_returns_none() {
        let empty_inputs = [Series::new("n", Vec::<i32>::new())
            .cast(&DataType::Int32)
            .unwrap()];
        let td_ser = tdigest_impl(&empty_inputs, 100).unwrap();
        let q_ser = estimate_quantile_impl(&[td_ser], 0.5).unwrap();
        assert!(
            q_ser.f64().unwrap().get(0).is_none(),
            "expected None for empty digest"
        );
    }
}
