//! Criterion benchmarks for core t-digest operations.
//!
//! Run with: `cargo bench --bench tdigest_core`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_tdigest::tdigest::{ScaleFamily, TDigest};
use tdigest_testdata::{gen_dataset, DistKind};

/* ------------------------ BUILD HELPER ------------------------ */

/// Build a TDigest from sorted data. Sorting cost is included in the benchmark.
fn build_digest(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    seed: u64,
) -> TDigest {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap()); // guard if NaNs ever sneak in
    TDigest::new_with_size_and_scale(max_size, scale).merge_sorted(data)
}

#[derive(Clone, Copy)]
struct Params {
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    seed: u64,
    kind: DistKind,
}

/* ------------------------ BENCH: BUILD ------------------------ */

fn bench_build(c: &mut Criterion) {
    let cases = [
        Params {
            n: 100_000,
            max_size: 100,
            scale: ScaleFamily::K2,
            kind: DistKind::Mixture,
            seed: 42,
        },
        Params {
            n: 1_000_000,
            max_size: 200,
            scale: ScaleFamily::K2,
            kind: DistKind::Mixture,
            seed: 42,
        },
    ];

    let mut g = c.benchmark_group("build_digest");
    g.warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(6))
        .sample_size(30);

    for p in cases {
        let id =
            BenchmarkId::from_parameter(format!("n={},k={},scale={:?}", p.n, p.max_size, p.scale));
        g.bench_function(id, |b| {
            b.iter(|| {
                let td = build_digest(p.kind, p.n, p.max_size, p.scale, p.seed);
                black_box(td);
            });
        });
    }
    g.finish();
}

/* --------------------- BENCH: QUANTILE ------------------------ */

fn bench_quantile(c: &mut Criterion) {
    let p = Params {
        n: 1_000_000,
        max_size: 200,
        scale: ScaleFamily::K2,
        kind: DistKind::Mixture,
        seed: 123,
    };
    let td = build_digest(p.kind, p.n, p.max_size, p.scale, p.seed);

    // Single quantile (median)
    let mut g_single = c.benchmark_group("estimate_quantile/single");
    g_single
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(4));
    g_single.bench_function("q=0.5", |b| {
        b.iter(|| {
            let r = td.estimate_quantile(black_box(0.5));
            black_box(r);
        });
    });
    g_single.finish();

    // Batched quantiles (throughput)
    let qs: Vec<f64> = (1..1000).map(|i| (i as f64) / 1000.0).collect();
    let mut g_batch = c.benchmark_group("estimate_quantile/batch_1000");
    g_batch
        .throughput(Throughput::Elements(qs.len() as u64))
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(4));
    g_batch.bench_function("grid", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &q in &qs {
                acc += td.estimate_quantile(black_box(q));
            }
            black_box(acc);
        });
    });
    g_batch.finish();
}

/* ------------------------ BENCH: CDF -------------------------- */

fn bench_cdf(c: &mut Criterion) {
    let p = Params {
        n: 1_000_000,
        max_size: 200,
        scale: ScaleFamily::K2,
        kind: DistKind::Mixture,
        seed: 999,
    };
    let td = build_digest(p.kind, p.n, p.max_size, p.scale, p.seed);

    // Single x
    let mut g_single = c.benchmark_group("estimate_cdf/single");
    g_single
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(4));
    g_single.bench_function("x=0.5", |b| {
        b.iter(|| {
            let out = td.estimate_cdf(black_box(&[0.5]));
            black_box(out);
        });
    });
    g_single.finish();

    // Batched xs
    let xs: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
    let mut g_batch = c.benchmark_group("estimate_cdf/batch_1000");
    g_batch
        .throughput(Throughput::Elements(xs.len() as u64))
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(4));
    g_batch.bench_function("grid", |b| {
        b.iter(|| {
            let out = td.estimate_cdf(black_box(&xs));
            black_box(out);
        });
    });
    g_batch.finish();
}

/* --------------- BONUS: SCALE-FAMILY COMPARE ------------------ */

fn bench_scales(c: &mut Criterion) {
    let n = 500_000usize;
    let seed = 777u64;
    let kind = DistKind::Mixture;
    let max_size = 200usize;

    let mut g = c.benchmark_group("estimate_quantile/scale_compare_q95");
    g.warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(4));

    for scale in [
        ScaleFamily::Quad,
        ScaleFamily::K1,
        ScaleFamily::K2,
        ScaleFamily::K3,
    ] {
        let td = build_digest(kind, n, max_size, scale, seed);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{scale:?}")),
            &td,
            |b, td| {
                b.iter(|| {
                    let r = td.estimate_quantile(black_box(0.95));
                    black_box(r);
                });
            },
        );
    }
    g.finish();
}

fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(5))
        .sample_size(40)
        .with_plots()
}

criterion_group!(
    name = tdigest_benches;
    config = configure();
    targets = bench_build, bench_quantile, bench_cdf, bench_scales
);
criterion_main!(tdigest_benches);
