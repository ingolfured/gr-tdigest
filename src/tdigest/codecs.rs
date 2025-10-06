use crate::tdigest::TDigest;
use polars::prelude::*;

pub(crate) fn tdigest_to_series_32(tdv: TDigest, name: &str) -> Series {
    // 1) centroids -> mean: f32, weight: f32
    let len = tdv.centroids().len();
    let mut means = Vec::with_capacity(len);
    let mut weights = Vec::with_capacity(len);
    for c in tdv.centroids() {
        means.push(c.mean() as f32);
        weights.push(c.weight() as f32);
    }

    let mean_s = Series::new("mean".into(), means);
    let weight_s = Series::new("weight".into(), weights);

    // Struct(mean: f32, weight: f32) — width = number of centroids
    let centroid_struct: Series = StructChunked::from_series(
        "centroid".into(),
        mean_s.len(), // <-- row count (NOT number of fields)
        [&mean_s, &weight_s].into_iter(),
    )
    .expect("build centroid struct")
    .into_series();

    // List<Struct{...}> with a single element (row 0)
    let centroids_list: Series = Series::new("centroids".into(), &[centroid_struct]);

    // 2) other fields
    let sum_s = Series::new("sum".into(), [tdv.sum()]);
    let min_s = Series::new("min".into(), [tdv.min() as f32]);
    let max_s = Series::new("max".into(), [tdv.max() as f32]);
    let count_s = Series::new("count".into(), [tdv.count()]);
    let max_size_s = Series::new("max_size".into(), [tdv.max_size() as i64]);

    // 3) outer Struct — width = 1 (single row)
    StructChunked::from_series(
        name.into(),
        1, // <-- single row
        [
            &centroids_list,
            &sum_s,
            &min_s,
            &max_s,
            &count_s,
            &max_size_s,
        ]
        .into_iter(),
    )
    .expect("build outer tdigest struct")
    .into_series()
}

/// Canonical (lossless) tdigest as a single-row `Series`.
/// - centroids: List<Struct{ mean: Float64, weight: Float64 }>
/// - sum/min/max/count: Float64
/// - max_size: Int64
pub fn tdigest_to_series(td: TDigest, name: &str) -> Series {
    let len = td.centroids().len();
    let mut means = Vec::with_capacity(len);
    let mut weights = Vec::with_capacity(len);
    for c in td.centroids() {
        means.push(c.mean());
        weights.push(c.weight());
    }

    let mean_s = Series::new("mean".into(), means);
    let weight_s = Series::new("weight".into(), weights);

    // Struct(mean: f64, weight: f64) — width = number of centroids
    let centroid_struct: Series = StructChunked::from_series(
        "centroid".into(),
        mean_s.len(), // <-- row count
        [&mean_s, &weight_s].into_iter(),
    )
    .expect("build centroid struct")
    .into_series();

    let centroids_list: Series = Series::new("centroids".into(), &[centroid_struct]);

    StructChunked::from_series(
        name.into(),
        1, // <-- single row
        [
            &centroids_list,
            &Series::new("sum".into(), [td.sum()]),
            &Series::new("min".into(), [td.min()]),
            &Series::new("max".into(), [td.max()]),
            &Series::new("count".into(), [td.count()]),
            &Series::new("max_size".into(), [td.max_size() as i64]),
        ]
        .into_iter(),
    )
    .expect("build outer tdigest struct")
    .into_series()
}

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

    let Ok(centroids_list) = centroids_col.list() else {
        return Vec::new();
    };

    for i in 0..n {
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
                v.try_extract::<f64>()
                    .ok()
                    .or_else(|| v.try_extract::<i64>().ok().map(|x| x as f64))
                    .or_else(|| v.try_extract::<u64>().ok().map(|x| x as f64))
            })
            .unwrap_or(0.0);

        let max_size = max_size_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<i64>().ok())
            .unwrap_or(0) as usize;

        // Parse centroids row -> Vec<Centroid>
        let mut cents: Vec<crate::tdigest::Centroid> = Vec::new();
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
                    cents.push(crate::tdigest::Centroid::new(m, w));
                }
            }
        }

        out.push(TDigest::new(cents, sum, count, min, max, max_size));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::Centroid;

    #[test]
    fn test_tdigest_roundtrip_scaling_and_parse() {
        let tdigest = TDigest::new(
            vec![
                crate::tdigest::Centroid::new(10.0, 1.0e34),
                crate::tdigest::Centroid::new(20.0, 2.0e34),
                crate::tdigest::Centroid::new(30.0, 3.0e34),
            ],
            6.0e34,
            6.0e34,
            30.0,
            10.0,
            300,
        );

        // Baseline quantile from original (unscaled) digest.
        let p50_before = tdigest.estimate_quantile(0.5);

        // Compact writer (may scale internally).
        let ser = tdigest_to_series_32(tdigest.clone(), "n");

        // Quick sanity: compact weights are finite f32.
        {
            let sc = ser.struct_().expect("struct");
            let centroids_col = sc.field_by_name("centroids").expect("centroids");
            let list = centroids_col.list().expect("list");
            let row0 = list.get_as_series(0).expect("row0");
            let inner = row0.struct_().expect("inner");
            let w_series = inner.field_by_name("weight").expect("weight");
            let w_f32 = w_series.f32().expect("f32 weights");
            for i in 0..w_f32.len() {
                if let Some(w) = w_f32.get(i) {
                    assert!(w.is_finite());
                }
            }
        }

        // Parse back and verify invariants insensitive to global scaling.
        let parsed = parse_tdigests_any(&ser);
        assert_eq!(parsed.len(), 1);
        let td2 = parsed[0].clone();

        // Centroid count must match.
        assert_eq!(td2.centroids().len(), tdigest.centroids().len());

        // Quantiles preserved.
        let p50_after = td2.estimate_quantile(0.5);
        assert!(
            (p50_before - p50_after).abs() <= 1e-6,
            "p50 changed: before={p50_before}, after={p50_after}"
        );

        // Weight ratios preserved (allow global scale).
        let w1: Vec<f64> = tdigest.centroids().iter().map(|c| c.weight()).collect();
        let w2: Vec<f64> = td2.centroids().iter().map(|c| c.weight()).collect();
        let s1: f64 = w1.iter().sum();
        let s2: f64 = w2.iter().sum();
        for (a, b) in w1.iter().zip(w2.iter()) {
            let ra = *a / s1;
            let rb = *b / s2;
            assert!((ra - rb).abs() <= 1e-7, "weight ratio changed");
        }
    }
}
