use crate::tdigest::{Centroid, TDigest};
use polars::prelude::*;
use polars::series::Series;

/// Build an f32 tdigest struct from a TDigest by downcasting the f64 struct.
/// - centroids.mean/weight → Float32
/// - min/max → Float32
/// - sum stays Float64; count/max_size unchanged
pub(crate) fn tdigest_to_series_f32(td: TDigest, name: &str) -> Series {
    // Start from the canonical f64 struct to avoid touching TDigest internals here.
    let s64 = tdigest_to_series(td, name);
    let sc = s64
        .struct_()
        .expect("tdigest_to_series returned non-struct");

    let centroids_col = sc.field_by_name("centroids").expect("centroids");
    let sum_col = sc.field_by_name("sum").expect("sum");
    let min_col = sc.field_by_name("min").expect("min");
    let max_col = sc.field_by_name("max").expect("max");
    let count_col = sc.field_by_name("count").expect("count");
    let max_size_col = sc.field_by_name("max_size").expect("max_size");

    // centroids is a 1-row List whose inner dtype is Struct{mean,weight}
    let list_av = centroids_col.get(0).expect("centroids row 0");
    let centroids_list_series = match list_av {
        AnyValue::List(ls) => ls,
        _ => Series::new("centroids", &[] as &[Series]), // empty list fallback
    };

    // Cast struct fields to f32
    let scents = centroids_list_series
        .struct_()
        .expect("centroids inner struct");
    let mean_f32 = scents
        .field_by_name("mean")
        .expect("mean")
        .cast(&DataType::Float32)
        .unwrap();
    let weight_f32 = scents
        .field_by_name("weight")
        .expect("weight")
        .cast(&DataType::Float32)
        .unwrap();

    let centroids_struct_f32 = StructChunked::new("centroids", &[mean_f32, weight_f32])
        .unwrap()
        .into_series();

    // Wrap single struct as a one-row List
    let centroids_list_f32 = Series::new("centroids", &[centroids_struct_f32]);

    // min/max to f32; keep sum/count/max_size as-is
    let min_f32 = min_col.cast(&DataType::Float32).unwrap();
    let max_f32 = max_col.cast(&DataType::Float32).unwrap();

    StructChunked::new(
        name,
        &[
            centroids_list_f32,
            sum_col.clone(),
            min_f32,
            max_f32,
            count_col.clone(),
            max_size_col.clone(),
        ],
    )
    .unwrap()
    .into_series()
}

// NEW: tolerant parser that accepts either f64 or f32 tdigest payloads.

/// Accepts both "classic" f64 digests and compact f32 digests.
/// Tolerant casts:
/// - centroids.mean:  f64 or f32  -> f64
/// - centroids.weight: f64 or f32 or i64 -> f64
/// - min/max:         f64 or f32  -> f64
/// - count:           i64 or u64  -> f64
pub(crate) fn parse_tdigests_any(input: &Series) -> Vec<TDigest> {
    let Ok(struct_ca) = input.struct_() else {
        return Vec::new();
    };

    let Ok(centroids_col) = struct_ca.field_by_name("centroids") else {
        return Vec::new();
    };
    let Ok(sum_col) = struct_ca.field_by_name("sum") else {
        return Vec::new();
    };
    let Ok(min_col) = struct_ca.field_by_name("min") else {
        return Vec::new();
    };
    let Ok(max_col) = struct_ca.field_by_name("max") else {
        return Vec::new();
    };
    let Ok(count_col) = struct_ca.field_by_name("count") else {
        return Vec::new();
    };
    let Ok(max_size_col) = struct_ca.field_by_name("max_size") else {
        return Vec::new();
    };

    let n = input.len();
    let mut out = Vec::with_capacity(n);

    // centroids is a List column; use list() and get_as_series(i) on Polars 0.40
    let Ok(centroids_list) = centroids_col.list() else {
        return Vec::new();
    };

    for i in 0..n {
        // Scalars
        let sum = sum_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<f64>().ok())
            .unwrap_or(0.0);

        let min = min_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<f64>()
                    .ok()
                    .or_else(|| v.try_extract::<f32>().ok().map(|x| x as f64))
            })
            .unwrap_or(0.0);

        let max = max_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<f64>()
                    .ok()
                    .or_else(|| v.try_extract::<f32>().ok().map(|x| x as f64))
            })
            .unwrap_or(0.0);

        let count = count_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<i64>()
                    .ok()
                    .or_else(|| v.try_extract::<u64>().ok().map(|x| x as i64))
            })
            .unwrap_or(0) as f64;

        let max_size = max_size_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<i64>().ok())
            .unwrap_or(0) as usize;

        // Centroids row: a Series (dtype=Struct) of length = #centroids
        let mut cents: Vec<Centroid> = Vec::new();
        if let Some(centroids_row) = centroids_list.get_as_series(i) {
            let sc = centroids_row.struct_().unwrap();

            let mean_s = sc
                .field_by_name("mean")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            let weight_s = sc
                .field_by_name("weight")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();

            let mean_ca = mean_s.f64().unwrap();
            let wgt_ca = weight_s.f64().unwrap();

            let len = mean_ca.len().min(wgt_ca.len());
            for idx in 0..len {
                if let (Some(m), Some(w)) = (mean_ca.get(idx), wgt_ca.get(idx)) {
                    cents.push(Centroid::new(m, w));
                }
            }
        }

        out.push(TDigest::new(cents, sum, count, min, max, max_size));
    }

    out
}

// TODO: error handling w/o panic
pub fn parse_tdigests(input: &Series) -> Vec<TDigest> {
    input
        .struct_()
        .into_iter()
        .flat_map(|chunk| {
            let count_series = chunk.field_by_name("count").unwrap();
            let count_it = count_series.i64().unwrap().into_iter();
            let max_series = chunk.field_by_name("max").unwrap();
            let min_series = chunk.field_by_name("min").unwrap();
            let sum_series = chunk.field_by_name("sum").unwrap();
            let max_size_series = chunk.field_by_name("max_size").unwrap();
            let centroids_series = chunk.field_by_name("centroids").unwrap();
            let mut max_it = max_series.f64().unwrap().into_iter();
            let mut min_it = min_series.f64().unwrap().into_iter();
            let mut max_size_it = max_size_series.i64().unwrap().into_iter();
            let mut sum_it = sum_series.f64().unwrap().into_iter();
            let mut centroids_it = centroids_series.list().unwrap().into_iter();

            count_it
                .map(|c| {
                    let centroids = centroids_it.next().unwrap().unwrap();
                    let mean_series = centroids.struct_().unwrap().field_by_name("mean").unwrap();
                    let mean_it = mean_series.f64().unwrap().into_iter();
                    let weight_series = centroids
                        .struct_()
                        .unwrap()
                        .field_by_name("weight")
                        .unwrap();
                    let mut weight_it = weight_series.i64().unwrap().into_iter();
                    let centroids_res = mean_it
                        .map(|m| {
                            Centroid::new(m.unwrap(), weight_it.next().unwrap().unwrap() as f64)
                        })
                        .collect::<Vec<_>>();
                    TDigest::new(
                        centroids_res,
                        sum_it.next().unwrap().unwrap(),
                        c.unwrap() as f64,
                        max_it.next().unwrap().unwrap(),
                        min_it.next().unwrap().unwrap(),
                        max_size_it.next().unwrap().unwrap() as usize,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

pub fn tdigest_to_series(tdigest: TDigest, name: &str) -> Series {
    let mut means: Vec<f64> = vec![];
    let mut weights: Vec<i64> = vec![];
    tdigest.centroids().iter().for_each(|c| {
        weights.push(c.weight() as i64);
        means.push(c.mean());
    });

    let centroids_series = DataFrame::new(vec![
        Series::new("mean", means),
        Series::new("weight", weights),
    ])
    .unwrap()
    .into_struct("centroids")
    .into_series();

    DataFrame::new(vec![
        Series::new("centroids", [Series::new("centroids", centroids_series)]),
        Series::new("sum", [tdigest.sum()]),
        Series::new("min", [tdigest.min()]),
        Series::new("max", [tdigest.max()]),
        Series::new("count", [tdigest.count() as i64]),
        Series::new("max_size", [tdigest.max_size() as i64]),
    ])
    .unwrap()
    .into_struct(name)
    .into_series()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_tdigest_deserializstion() {
        let json_str = "[{\"tdigest\":{\"centroids\":[{\"mean\":4.0,\"weight\":1},{\"mean\":5.0,\"weight\":1},{\"mean\":6.0,\"weight\":1}],\"sum\":15.0,\"min\":4.0,\"max\":6.0,\"count\":3,\"max_size\":100}},{\"tdigest\":{\"centroids\":[{\"mean\":1.0,\"weight\":1},{\"mean\":2.0,\"weight\":1},{\"mean\":3.0,\"weight\":1}],\"sum\":6.0,\"min\":1.0,\"max\":3.0,\"count\":3,\"max_size\":100}}]";
        let cursor = Cursor::new(json_str);
        let df = JsonReader::new(cursor).finish().unwrap();
        let series = df.column("tdigest").unwrap();
        let res = parse_tdigests(series);
        let expected = vec![
            TDigest::new(
                vec![
                    Centroid::new(4.0, 1.0),
                    Centroid::new(5.0, 1.0),
                    Centroid::new(6.0, 1.0),
                ],
                15.0,
                3.0,
                6.0,
                4.0,
                100,
            ),
            TDigest::new(
                vec![
                    Centroid::new(1.0, 1.0),
                    Centroid::new(2.0, 1.0),
                    Centroid::new(3.0, 1.0),
                ],
                6.0,
                3.0,
                3.0,
                1.0,
                100,
            ),
        ];
        assert!(res == expected);
    }

    #[test]
    fn test_tdigest_serialization() {
        let tdigest = TDigest::new(
            vec![
                Centroid::new(10.0, 1.0),
                Centroid::new(20.0, 2.0),
                Centroid::new(30.0, 3.0),
            ],
            60.0,
            3.0,
            30.0,
            10.0,
            300,
        );
        let res = tdigest_to_series(tdigest, "n");

        let cs = DataFrame::new(vec![
            Series::new("mean", [10.0, 20.0, 30.0]),
            Series::new("weight", [1, 2, 3]),
        ])
        .unwrap()
        .into_struct("centroids")
        .into_series();

        let expected = DataFrame::new(vec![
            Series::new("centroids", [Series::new("a", cs)]),
            Series::new("sum", [60.0]),
            Series::new("min", [10.0]),
            Series::new("max", [30.0]),
            Series::new("count", [3.0]),
            Series::new("max_size", [300_i64]),
        ])
        .unwrap()
        .into_struct("n")
        .into_series();

        assert!(res == expected);
    }
}
