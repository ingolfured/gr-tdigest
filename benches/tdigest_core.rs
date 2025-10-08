//! Criterion benchmarks for core t-digest operations.
//!
//! Stable run tips:
//!   export RUSTFLAGS="-C target-cpu=native"
//!   export RAYON_NUM_THREADS=8       # or 1 for max stability
//!
//! Discover benches:
//!   cargo bench --bench tdigest_core -- --list
//!
//! Save a baseline across all groups in this bench:
//!   cargo bench --bench tdigest_core -- --save-baseline cdf_base
//!
//! Compare specific groups to that baseline later:
//!   cargo bench --bench tdigest_core -- --baseline cdf_base "estimate_cdf/sizes_prepared"
//!   cargo bench --bench tdigest_core -- --baseline cdf_base "estimate_cdf/sizes_nocache"

use std::hint::black_box;
use std::sync::Once;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_tdigest::tdigest::{CdfCachePolicy, ScaleFamily, TDigest};
use rayon::ThreadPoolBuilder;
use tdigest_testdata::{gen_dataset, DistKind};

/* ------------------------ RAYON INIT (once) ------------------------ */

static RAYON_INIT: Once = Once::new();

fn init_rayon() {
    RAYON_INIT.call_once(|| {
        let builder = match std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(n) => ThreadPoolBuilder::new().num_threads(n),
            None => ThreadPoolBuilder::new(),
        };
        let _ = builder.build_global(); // ignore Err if already built
    });
}

/* ------------------------ BUILD HELPERS ------------------------ */

/// Build a TDigest from sorted data. Sorting cost is included.
fn build_digest(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    seed: u64,
) -> TDigest {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap()); // guard against NaNs
    TDigest::new_with_size_and_scale(max_size, scale).merge_sorted(data)
}

/// Same as `build_digest`, but with prepared CDF arrays (cache policy = Prepared).
fn build_digest_prepared(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    seed: u64,
) -> TDigest {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    TDigest::new_with_size_scale_cache(max_size, scale, CdfCachePolicy::Prepared).merge_sorted(data)
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
    init_rayon();

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
    for p in cases {
        let id =
            BenchmarkId::from_parameter(format!("n={},k={},scale={:?}", p.n, p.max_size, p.scale));
        g.bench_function(id, |b| {
            b.iter(|| black_box(build_digest(p.kind, p.n, p.max_size, p.scale, p.seed)));
        });
    }
    g.finish();
}

/* --------------------- BENCH: QUANTILE ------------------------ */

fn bench_quantile(c: &mut Criterion) {
    init_rayon();

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
    g_single.bench_function("q=0.5", |b| {
        b.iter(|| black_box(td.estimate_quantile(black_box(0.5))));
    });
    g_single.finish();

    // Batched (1000 qs)
    let qs: Vec<f64> = (1..1000).map(|i| (i as f64) / 1000.0).collect();
    let mut g_batch = c.benchmark_group("estimate_quantile/batch_1000");
    g_batch.throughput(Throughput::Elements(qs.len() as u64));
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
    init_rayon();

    let p = Params {
        n: 1_000_000,
        max_size: 200,
        scale: ScaleFamily::K2,
        kind: DistKind::Mixture,
        seed: 999,
    };
    let td = build_digest_prepared(p.kind, p.n, p.max_size, p.scale, p.seed);

    // Single x
    let mut g_single = c.benchmark_group("estimate_cdf/single");
    g_single.bench_function("x=0.5", |b| {
        b.iter(|| black_box(td.estimate_cdf(black_box(&[0.5]))));
    });
    g_single.finish();

    // Batched xs (1000)
    let xs: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
    let mut g_batch = c.benchmark_group("estimate_cdf/batch_1000");
    g_batch.throughput(Throughput::Elements(xs.len() as u64));
    g_batch.bench_function("grid", |b| {
        b.iter(|| black_box(td.estimate_cdf(black_box(&xs))));
    });
    g_batch.finish();
}

/* --------------- BENCH: SCALE-FAMILY COMPARE (q=0.95) ------------------ */

fn bench_scales(c: &mut Criterion) {
    init_rayon();

    let n = 500_000usize;
    let seed = 777u64;
    let kind = DistKind::Mixture;
    let max_size = 200usize;

    let mut g = c.benchmark_group("estimate_quantile/scale_compare_q95");
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
                b.iter(|| black_box(td.estimate_quantile(black_box(0.95))));
            },
        );
    }
    g.finish();
}

/* ------------------------ BENCH: CDF SIZE SWEEPS ------------------------ */

fn bench_cdf_sizes_prepared(c: &mut Criterion) {
    init_rayon();

    // Fixed digest: K2 scale, max_size=1000, uniform data.
    let n_build = 100_000;
    let max_size = 1_000;
    let mut data = gen_dataset(DistKind::Uniform, n_build, 4242);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let td =
        TDigest::new_with_size_scale_cache(max_size, ScaleFamily::K2, CdfCachePolicy::Prepared)
            .merge_sorted(data);

    // Sweep from 1 → 10_000_000
    let sizes = [
        1usize, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
    ];
    let mut group = c.benchmark_group("estimate_cdf/sizes_prepared");

    for &m in &sizes {
        let xs: Vec<f64> = (0..m).map(|i| (i as f64) / (m.max(1) as f64)).collect();
        // Repeat small sizes to reduce noise while keeping per-iter work realistic.
        let repeat = if m <= 1_000 {
            10_000
        } else if m <= 10_000 {
            1_000
        } else {
            1
        };

        group.throughput(Throughput::Elements((m as u64) * (repeat as u64)));
        group.bench_with_input(BenchmarkId::from_parameter(m), &xs, |b, xs| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for _ in 0..repeat {
                    let out = td.estimate_cdf(black_box(xs));
                    // read a middle element to prevent optimizer from discarding the result
                    acc += out[out.len().saturating_sub(1) / 2];
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

fn bench_cdf_sizes_nocache(c: &mut Criterion) {
    init_rayon();

    let n_build = 100_000;
    let max_size = 1_000;
    let mut data = gen_dataset(DistKind::Uniform, n_build, 4242);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let td = TDigest::new_with_size_and_scale(max_size, ScaleFamily::K2).merge_sorted(data);

    let sizes = [
        1usize, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
    ];
    let mut group = c.benchmark_group("estimate_cdf/sizes_nocache");

    for &m in &sizes {
        let xs: Vec<f64> = (0..m).map(|i| (i as f64) / (m.max(1) as f64)).collect();
        let repeat = if m <= 1_000 {
            10_000
        } else if m <= 10_000 {
            1_000
        } else {
            1
        };

        group.throughput(Throughput::Elements((m as u64) * (repeat as u64)));
        group.bench_with_input(BenchmarkId::from_parameter(m), &xs, |b, xs| {
            b.iter(|| {
                let mut acc = 0.0_f64;
                for _ in 0..repeat {
                    let out = td.estimate_cdf(black_box(xs));
                    acc += out[out.len().saturating_sub(1) / 2];
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

/* ------------------------ CONFIG (hard-coded timing) ------------------------ */

fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1)) // hard-coded
        .measurement_time(Duration::from_secs(2)) // hard-coded
        .sample_size(30) // hard-coded
        .without_plots()
}

/* ------------------------ GROUP REGISTRATION ------------------------ */

criterion_group!(
    name = tdigest_benches;
    config = configure();
    targets =
        bench_build,
        bench_quantile,
        bench_cdf,
        bench_scales,
        bench_cdf_sizes_prepared,
        bench_cdf_sizes_nocache
);

criterion_main!(tdigest_benches);
