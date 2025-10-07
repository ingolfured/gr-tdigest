use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars::prelude::*;
use polars_tdigest::tdigest::{Centroid, TDigest};

fn synth_digest(k: usize, shift: f64) -> TDigest {
    // make non-trivial centroids to exercise packing
    let cents: Vec<Centroid> = (0..k)
        .map(|i| {
            let m = (i as f64 + 0.5 + shift) * 1e-3;
            let w = 1.0 + ((i as f64 + shift) % 3.0);
            Centroid::new(m, w)
        })
        .collect();
    // sum/min/max/count are arbitrary but consistent
    let sum: f64 = cents.iter().map(|c| c.mean() * c.weight()).sum();
    let count: f64 = cents.iter().map(|c| c.weight()).sum();
    let min = cents.first().map(|c| c.mean()).unwrap_or(0.0);
    let max = cents.last().map(|c| c.mean()).unwrap_or(0.0);
    TDigest::new(cents, sum, count, min, max, 200)
}

fn make_series(rows: usize, k: usize, f32_mode: bool) -> Series {
    let mut s = if f32_mode {
        tdigest_to_series_32(synth_digest(k, 0.0), "td")
    } else {
        tdigest_to_series(synth_digest(k, 0.0), "td")
    };
    for r in 1..rows {
        let td = synth_digest(k, r as f64 * 0.123);
        let next = if f32_mode {
            tdigest_to_series_32(td, "td")
        } else {
            tdigest_to_series(td, "td")
        };
        s.append(&next).unwrap();
    }
    s
}

fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("codecs/write");
    g.warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(3));

    for &(rows, k) in &[(10, 200), (100, 200), (1000, 200), (100, 50), (100, 1000)] {
        for &f32_mode in &[false, true] {
            let id = BenchmarkId::from_parameter(format!(
                "rows={rows},k={k},type={}",
                if f32_mode { "f32" } else { "f64" }
            ));
            g.throughput(Throughput::Elements((rows * k) as u64));
            g.bench_function(id, |b| {
                b.iter(|| {
                    // build fresh each iter to include packing cost
                    let s = make_series(rows, k, f32_mode);
                    black_box(s.len())
                });
            });
        }
    }
    g.finish();
}

fn bench_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("codecs/read");
    g.warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(3));

    for &(rows, k) in &[(10, 200), (100, 200), (1000, 200), (100, 50), (100, 1000)] {
        for &f32_mode in &[false, true] {
            let s = make_series(rows, k, f32_mode);
            let id = BenchmarkId::from_parameter(format!(
                "rows={rows},k={k},type={}",
                if f32_mode { "f32" } else { "f64" }
            ));
            g.throughput(Throughput::Elements((rows * k) as u64));
            g.bench_with_input(id, &s, |b, series| {
                b.iter(|| {
                    let v = parse_tdigests_strict(black_box(series));
                    black_box(v.len())
                });
            });
        }
    }
    g.finish();
}

criterion_group!(codecs, bench_write, bench_read);
criterion_main!(codecs);
