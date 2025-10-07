use crate::tdigest::{Centroid, TDigest};
use polars::prelude::*;

// --------------------- field name constants ----------------------------------
const F_CENTROIDS: &str = "centroids";
const F_MEAN: &str = "mean";
const F_WEIGHT: &str = "weight";
const F_SUM: &str = "sum";
const F_MIN: &str = "min";
const F_MAX: &str = "max";
const F_COUNT: &str = "count";
const F_MAX_SIZE: &str = "max_size";

// --------------------- shared helpers: centroid pack/unpack ------------------

#[inline]
fn centroid_struct(mean_s: &Series, weight_s: &Series) -> PolarsResult<Series> {
    StructChunked::from_series(
        "centroid".into(),
        mean_s.len(),
        [mean_s, weight_s].into_iter(), // yields &Series (not &&Series)
    )
    .map(|sc| sc.into_series())
}

/// Build `List<Struct{mean, weight}>` with f64 inner fields
fn pack_centroids_f64(cents: &[Centroid]) -> Series {
    let mut means: Vec<f64> = Vec::with_capacity(cents.len());
    let mut wgts: Vec<f64> = Vec::with_capacity(cents.len());
    for c in cents {
        means.push(c.mean());
        wgts.push(c.weight());
    }
    let mean_s = Series::new(F_MEAN.into(), means);
    let weight_s = Series::new(F_WEIGHT.into(), wgts);
    let inner = centroid_struct(&mean_s, &weight_s).expect("centroid struct (f64)");
    Series::new(F_CENTROIDS.into(), &[inner])
}

/// Build `List<Struct{mean, weight}>` with f32 inner fields (compact)
fn pack_centroids_f32(cents: &[Centroid]) -> Series {
    let mut means: Vec<f32> = Vec::with_capacity(cents.len());
    let mut wgts: Vec<f32> = Vec::with_capacity(cents.len());
    for c in cents {
        means.push(c.mean() as f32);
        wgts.push(c.weight() as f32);
    }
    let mean_s = Series::new(F_MEAN.into(), means);
    let weight_s = Series::new(F_WEIGHT.into(), wgts);
    let inner = centroid_struct(&mean_s, &weight_s).expect("centroid struct (f32)");
    Series::new(F_CENTROIDS.into(), &[inner])
}

/// Parse one row’s `List<Struct{mean, weight}>` into `Vec<Centroid>` (as f64)
fn unpack_centroids_to_f64(centroids_list: &ListChunked, row: usize) -> Vec<Centroid> {
    let mut out = Vec::new();
    let Some(row_ser) = centroids_list.get_as_series(row) else {
        return out;
    };
    let Ok(sc) = row_ser.struct_() else {
        return out;
    };

    let m_ser = match sc
        .field_by_name(F_MEAN)
        .ok()
        .and_then(|s| s.cast(&DataType::Float64).ok())
    {
        Some(s) => s,
        None => return out,
    };
    let w_ser = match sc
        .field_by_name(F_WEIGHT)
        .ok()
        .and_then(|s| s.cast(&DataType::Float64).ok())
    {
        Some(s) => s,
        None => return out,
    };

    let m = m_ser.f64().unwrap();
    let w = w_ser.f64().unwrap();

    // zip() keeps the shorter; filter out Nones cleanly
    for (mm, ww) in m.into_iter().zip(w.into_iter()) {
        if let (Some(mv), Some(wv)) = (mm, ww) {
            out.push(Centroid::new(mv, wv));
        }
    }
    out
}

// --------------------- shared helpers: scalars & outer struct ----------------

#[inline]
fn any_to_f64(v: AnyValue<'_>) -> Option<f64> {
    v.try_extract::<f64>()
        .ok()
        .or_else(|| v.try_extract::<f32>().ok().map(|x| x as f64))
}

#[inline]
fn any_to_count(v: AnyValue<'_>) -> Option<f64> {
    v.try_extract::<f64>()
        .ok()
        .or_else(|| v.try_extract::<i64>().ok().map(|x| x as f64))
        .or_else(|| v.try_extract::<u64>().ok().map(|x| x as f64))
}

/// Build the single-row outer struct from pre-built centroids + scalar fields.
/// `min`/`max` are passed as Series to preserve exact dtype (f64 for canonical, f32 for compact).
fn build_outer_struct(
    name: &str,
    centroids_list: &Series,
    sum: f64,
    min: Series,
    max: Series,
    count: f64,
    max_size: i64,
) -> Series {
    let sum_s = Series::new(F_SUM.into(), [sum]);
    let count_s = Series::new(F_COUNT.into(), [count]);
    let max_size_s = Series::new(F_MAX_SIZE.into(), [max_size]);

    StructChunked::from_series(
        name.into(),
        1,
        [centroids_list, &sum_s, &min, &max, &count_s, &max_size_s].into_iter(),
    )
    .expect("tdigest outer struct")
    .into_series()
}

// --------------------- writers (single dispatcher) ---------------------------

enum Precision {
    F32,
    F64,
}

fn tdigest_to_series_with(td: TDigest, name: &str, p: Precision) -> Series {
    let cents = td.centroids();

    match p {
        Precision::F64 => {
            let centroids_list = pack_centroids_f64(cents);
            build_outer_struct(
                name,
                &centroids_list,
                td.sum(),
                Series::new(F_MIN.into(), [td.min()]),
                Series::new(F_MAX.into(), [td.max()]),
                td.count(),
                td.max_size() as i64,
            )
        }
        Precision::F32 => {
            let centroids_list = pack_centroids_f32(cents);
            build_outer_struct(
                name,
                &centroids_list,
                td.sum(),
                Series::new(F_MIN.into(), [td.min() as f32]),
                Series::new(F_MAX.into(), [td.max() as f32]),
                td.count(),
                td.max_size() as i64,
            )
        }
    }
}

/// Canonical: f64 means/weights, f64 min/max/sum/count, i64 max_size
pub fn tdigest_to_series(td: TDigest, name: &str) -> Series {
    tdigest_to_series_with(td, name, Precision::F64)
}

/// Compact: f32 means/weights, f32 min/max, f64 sum/count, i64 max_size
pub(crate) fn tdigest_to_series_32(td: TDigest, name: &str) -> Series {
    tdigest_to_series_with(td, name, Precision::F32)
}

// --------------------- parser (re-uses the same helpers) ---------------------

pub(crate) fn parse_tdigests_any(input: &Series) -> Vec<TDigest> {
    let Ok(s) = input.struct_() else {
        return Vec::new();
    };

    let (Ok(centroids_col), Ok(sum_col), Ok(min_col), Ok(max_col), Ok(count_col), Ok(max_size_col)) = (
        s.field_by_name(F_CENTROIDS),
        s.field_by_name(F_SUM),
        s.field_by_name(F_MIN),
        s.field_by_name(F_MAX),
        s.field_by_name(F_COUNT),
        s.field_by_name(F_MAX_SIZE),
    ) else {
        return Vec::new();
    };

    let Ok(centroids_list) = centroids_col.list() else {
        return Vec::new();
    };

    let n = input.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let sum = sum_col.get(i).ok().and_then(any_to_f64).unwrap_or(0.0);
        let min = min_col.get(i).ok().and_then(any_to_f64).unwrap_or(0.0);
        let max = max_col.get(i).ok().and_then(any_to_f64).unwrap_or(0.0);
        let count = count_col.get(i).ok().and_then(any_to_count).unwrap_or(0.0);
        let max_sz = max_size_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<i64>().ok())
            .unwrap_or(0) as usize;

        let cents = unpack_centroids_to_f64(centroids_list, i);
        out.push(TDigest::new(cents, sum, count, min, max, max_sz));
    }

    out
}
