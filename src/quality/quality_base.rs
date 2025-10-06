//! Shared quality/diagnostics helpers for the t-digest implementation.
//!
//! - Synthetic dataset generators (Uniform, Normal, LogNormal, Mixture)
//! - QualityReport + scoring heuristic
//! - Helpers: expected_quantile (order-stat interpolation), exact_ecdf_for_sorted
//! - Precision + build helpers to standardize digest construction for tests.
//!
//! Engineering diagnostics, not publication-grade stats.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tdigest::{ScaleFamily, TDigest};

/// Compact report printed in tests/benches.
#[derive(Debug, Clone, Copy)]
pub struct QualityReport {
    pub n: usize,
    /// KS-like error (max absolute error on the chosen grid).
    pub ks: f64,
    /// Mean absolute error on the grid.
    pub mae: f64,
    /// A single scalar for rough comparison (higher is better).
    pub score: f64,
}

impl QualityReport {
    #[inline]
    pub fn from_metrics(n: usize, ks: f64, mae: f64) -> Self {
        // Same heuristic everywhere so numbers are comparable.
        let score = (-((1200.0 * mae) + (18.0 * ks))).exp();
        QualityReport { n, ks, mae, score }
    }
}

/// Which synthetic distribution to generate.
#[derive(Debug, Clone, Copy)]
pub enum DistKind {
    Uniform,
    Normal,
    LogNormal { sigma: f64 },
    Mixture,
}

/// Simulate building digests with different numeric precision.
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F64,
    /// “F32 mode”: round inputs to f32 first, then back to f64 before building the digest.
    /// This isolates input precision from internal math, giving a realistic lower-precision signal.
    F32Inputs,
}

/// Pretty banner for section headings in story-style tests.
pub fn print_banner(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("{title}");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
}

/// Subsection header (indented a touch) for size/scale groups.
pub fn print_section(title: &str) {
    println!("  ── {title} ────────────────────────────────────────────");
}

/// Small, shared print helper used in tests/benches.
pub fn print_report(tag: &str, r: QualityReport) {
    println!(
        "{} -> QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
        tag, r.n, r.ks, r.mae, r.score
    );
}

#[inline]
fn box_muller_normal<R: Rng>(rng: &mut R) -> f64 {
    // Clamp u1 to avoid ln(0).
    let u1: f64 = rng.random::<f64>().clamp(1e-12, 1.0);
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Generate `n` samples for the chosen distribution, clamped/squashed into [0,1].
pub fn gen_dataset(kind: DistKind, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);

    match kind {
        DistKind::Uniform => {
            for _ in 0..n {
                out.push(rng.random::<f64>());
            }
        }
        DistKind::Normal => {
            for _ in 0..n {
                let x = 0.5 + 0.2 * box_muller_normal(&mut rng);
                out.push(x.clamp(0.0, 1.0));
            }
        }
        DistKind::LogNormal { sigma } => {
            for _ in 0..n {
                let z = box_muller_normal(&mut rng);
                let x = (sigma * z).exp();
                let y = x / (1.0 + x);
                out.push(y.clamp(0.0, 1.0));
            }
        }
        DistKind::Mixture => {
            for _ in 0..n {
                let bucket: u32 = rng.random_range(0..100);
                let v = match bucket {
                    0..=29 => {
                        let center = match rng.random_range(0..3) {
                            0 => 0.10,
                            1 => 0.50,
                            _ => 0.90,
                        };
                        center + 1e-3 * rng.random_range(-1.0..1.0)
                    }
                    30..=69 => rng.random::<f64>(),
                    _ => {
                        let exp = rng.random_range(3.0..9.0);
                        if rng.random_bool(0.5) {
                            let u = rng.random::<f64>().clamp(1e-12, 1.0);
                            u.powf(exp)
                        } else {
                            let u = rng.random::<f64>().clamp(1e-12, 1.0);
                            1.0 - u.powf(exp)
                        }
                    }
                };
                out.push(v.clamp(0.0, 1.0));
            }
        }
    }
    out
}

/// Interpolate between order statistics to get an expected value at quantile `q`.
pub fn expected_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return sorted[n - 1];
    }
    let t = q * (n as f64 - 1.0);
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let alpha = t - lo as f64;
        (1.0 - alpha) * sorted[lo] + alpha * sorted[hi]
    }
}

/// Exact ECDF on a sorted sample using the midpoint convention on ties.
pub fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0;
        let f = mid / nf;
        for _ in i..j {
            out.push(f);
        }
        i = j;
    }
    out
}

/// Build a TDigest for sorted `data` with a chosen `max_size`, `scale` and `precision` model.
pub fn build_digest_sorted(
    mut data: Vec<f64>,
    max_size: usize,
    scale: ScaleFamily,
    precision: Precision,
) -> TDigest {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let data = match precision {
        Precision::F64 => data,
        Precision::F32Inputs => data.into_iter().map(|x| (x as f32) as f64).collect(),
    };
    let base = TDigest::new_with_size_and_scale(max_size, scale);
    base.merge_sorted(data)
}
