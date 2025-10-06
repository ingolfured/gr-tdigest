//! Quality checks for `TDigest::estimate_cdf(&[x])`.

use super::quality_base::{
    build_digest_sorted, exact_ecdf_for_sorted, gen_dataset, DistKind, Precision, QualityReport,
};
use crate::tdigest::{ScaleFamily, TDigest};

/// Compute empirical CDF at the grid `xs` using the midpoint-ties ECDF.
fn empirical_cdf_at_grid(sorted: &[f64], ecdf_sorted: &[f64], xs: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n == 0 {
        return vec![f64::NAN; xs.len()];
    }
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        // count of values <= x
        let idx = match sorted.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(mut j) => {
                while j + 1 < n && sorted[j + 1] <= x {
                    j += 1;
                }
                j + 1
            }
            Err(j) => j,
        };
        let f = if idx == 0 { 0.0 } else { ecdf_sorted[idx - 1] };
        out.push(f);
    }
    out
}

/// Returns (ks_like, mae) between digest-estimated CDF and empirical CDF on a dense grid.
fn cdf_grid_errors(td: &TDigest, sorted: &[f64]) -> (f64, f64) {
    let ecdf_sorted = exact_ecdf_for_sorted(sorted);
    let steps = 1000usize;
    let xs: Vec<f64> = (0..=steps).map(|i| (i as f64) / (steps as f64)).collect();
    let est: Vec<f64> = td.estimate_cdf(&xs);
    let exp: Vec<f64> = empirical_cdf_at_grid(sorted, &ecdf_sorted, &xs);

    let mut ks_like: f64 = 0.0;
    let mut mae: f64 = 0.0;
    for (e, a) in est.iter().zip(exp.iter()) {
        let err = (*e - *a).abs();
        mae += err;
        if err > ks_like {
            ks_like = err;
        }
    }
    mae /= (steps + 1) as f64;
    (ks_like, mae)
}

/// Build, score, and wrap as `QualityReport` with configurable scale/precision.
pub fn assess_cdf_with(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    precision: Precision,
    seed: u64,
) -> QualityReport {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let td = build_digest_sorted(data.clone(), max_size, scale, precision);
    let (ks, mae) = cdf_grid_errors(&td, &data);
    QualityReport::from_metrics(n, ks, mae)
}

/// Back-compat simple entry (Quad + F64).
pub fn assess_cdf(kind: DistKind, n: usize, max_size: usize, seed: u64) -> QualityReport {
    assess_cdf_with(kind, n, max_size, ScaleFamily::Quad, Precision::F64, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Bring the printers into *this* submodule explicitly.
    use crate::print_report; // re-exported in lib.rs
    use crate::quality::quality_base::{print_banner, print_section};

    /// ========= 1) REGRESSION: pin a single “truthy” scenario =========
    #[test]
    fn cdf_regression_max1000_quad_f64() {
        const SEED: u64 = 4242;
        const N: usize = 100_000;
        const MAX_SIZE: usize = 1000;
        const SCALE: ScaleFamily = ScaleFamily::Quad;
        const PREC: Precision = Precision::F64;

        // ---- BLESSED FROM 2025-10-06 21:26 RUN ----
        const BASE_KS: f64 = 1.206064e-3;
        const BASE_MAE: f64 = 4.157326e-5;
        const BASE_SCORE: f64 = 0.9309059239725813;

        // Keep tolerances strict-ish; score tolerance separate due to exp heuristic sensitivity.
        const KS_TOL: f64 = 5e-4;
        const MAE_TOL: f64 = 5e-6;
        const SCORE_TOL: f64 = 1e-3;

        let r = assess_cdf_with(DistKind::Mixture, N, MAX_SIZE, SCALE, PREC, SEED);
        print_report("REG/F[Mixture, k=1000, Quad, F64]", r);

        assert!(
            (r.ks - BASE_KS).abs() <= KS_TOL,
            "KS changed: {} vs {}",
            r.ks,
            BASE_KS
        );
        assert!(
            (r.mae - BASE_MAE).abs() <= MAE_TOL,
            "MAE changed: {} vs {}",
            r.mae,
            BASE_MAE
        );
        assert!(
            (r.score - BASE_SCORE).abs() <= SCORE_TOL,
            "score changed: {} vs {}",
            r.score,
            BASE_SCORE
        );
    }

    /// ========= 2) STORY: readable full matrix =========
    #[test]
    fn cdf_story_matrix() {
        use DistKind::*;
        const SEED: u64 = 4242;
        const N: usize = 100_000;

        let sizes = [100usize, 1000usize];
        let scales = [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K3,
        ];
        let precs = [Precision::F64, Precision::F32Inputs];
        let dists = [Uniform, Normal, LogNormal { sigma: 1.0 }, Mixture];

        print_banner(&format!(
            "CDF STORY MATRIX — full sweep (n={N}, seed={SEED})"
        ));

        for &dist in &dists {
            print_banner(&format!("DIST: {:?}", dist));
            for &k in &sizes {
                print_section(&format!("max_size = {k}"));
                for &scale in &scales {
                    println!("    SCALE: {:?} ▼", scale);
                    for &prec in &precs {
                        let tag = format!("      [prec={:?}] →", prec);
                        let r = assess_cdf_with(dist, N, k, scale, prec, SEED);
                        print_report(&tag, r);
                    }
                    println!();
                }
                println!();
            }
        }

        println!("(Hint: For same scale+precision, score(1000) should usually ≥ score(100).)");
        println!("═══════════════════════════════════════════════════════════════════════════");
    }
}
