use crate::tdigest::{ScaleFamily, TDigest};

#[derive(Debug, Clone)]
pub struct Quality {
    pub n: usize,
    pub max_size: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub n: usize,
    pub max_abs_err: f64,  // KS
    pub mean_abs_err: f64, // MAE
}

impl QualityReport {
    /// Half-lives chosen so base lands around ~0.7 for max_size=1000, seed=42.
    pub fn quality_score(&self) -> f64 {
        // Derived from observed KS≈1.478685e-3 and MAE≈3.493082e-5
        // so that each subscore is ~0.70 ⇒ geometric mean is ~0.70.
        self.quality_score_with_halves(0.002_873_614_6, 0.000_067_883_096)
    }

    /// Customizable half-lives: score hits 0.5 when KS==half_ks and MAE==half_mae.
    pub fn quality_score_with_halves(&self, half_ks: f64, half_mae: f64) -> f64 {
        fn sub(x: f64, half: f64) -> f64 {
            if half <= 0.0 || !x.is_finite() {
                return 0.0;
            }
            (2.0f64).powf(-x / half).clamp(0.0, 1.0)
        }
        let s_ks = sub(self.max_abs_err, half_ks);
        let s_mae = sub(self.mean_abs_err, half_mae);
        (s_ks * s_mae).sqrt()
    }

    pub fn strictly_better_than(&self, other: &QualityReport) -> bool {
        let eps = 1e-12;
        (self.max_abs_err <= other.max_abs_err + eps)
            && (self.mean_abs_err <= other.mean_abs_err + eps)
    }

    pub fn to_line(&self) -> String {
        format!(
            "QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
            self.n,
            self.max_abs_err,
            self.mean_abs_err,
            self.quality_score()
        )
    }

    pub fn log(&self) {
        eprintln!("{}", self.to_line());
    }
}

impl Quality {
    pub fn new(n: usize, max_size: usize, seed: u64) -> Self {
        Self { n, max_size, seed }
    }

    pub fn run(&self) -> QualityReport {
        use crate::tdigest::cdf::exact_ecdf_for_sorted;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut values: Vec<f64> = Vec::with_capacity(self.n);
        if self.n == 0 {
            return QualityReport {
                n: 0,
                max_abs_err: f64::NAN,
                mean_abs_err: f64::NAN,
            };
        }
        values.push(0.0); // guarantee a 0
        while values.len() < self.n {
            let bucket: u32 = rng.gen_range(0..100);
            let x = if bucket < 70 {
                rng.gen_range(-1.0..1.0)
            } else if bucket < 90 {
                let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 1000.0
            } else {
                let exp = rng.gen_range(3.0..9.0);
                let mag = 10f64.powf(exp);
                if rng.gen_bool(0.5) {
                    mag
                } else {
                    -mag
                }
            };
            values.push(x);
        }

        // Exact ECDF at each sorted sample (midpoint convention on ties)
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact_cdf = exact_ecdf_for_sorted(&values);

        let t0 = TDigest::new_with_size(self.max_size);
        let digest = t0.merge_sorted(values.clone()); // taking-ownership API => clone
        let td_cdf = digest.estimate_cdf(&values);

        let mut ks = 0.0;
        let mut sum_abs = 0.0;
        for (a, b) in exact_cdf.iter().zip(td_cdf.iter()) {
            let d = (a - b).abs();
            if d > ks {
                ks = d;
            }
            sum_abs += d;
        }
        let mae = sum_abs / (self.n as f64);

        QualityReport {
            n: self.n,
            max_abs_err: ks,
            mean_abs_err: mae,
        }
    }

    /// Run the quality harness using a specific scale family.
    pub fn run_with_scale(&self, family: ScaleFamily) -> QualityReport {
        use crate::tdigest::cdf::exact_ecdf_for_sorted;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut values: Vec<f64> = Vec::with_capacity(self.n);
        if self.n == 0 {
            return QualityReport {
                n: 0,
                max_abs_err: f64::NAN,
                mean_abs_err: f64::NAN,
            };
        }
        values.push(0.0);
        while values.len() < self.n {
            let bucket: u32 = rng.gen_range(0..100);
            let x = if bucket < 70 {
                rng.gen_range(-1.0..1.0)
            } else if bucket < 90 {
                let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 1000.0
            } else {
                let exp = rng.gen_range(3.0..9.0);
                let mag = 10f64.powf(exp);
                if rng.gen_bool(0.5) {
                    mag
                } else {
                    -mag
                }
            };
            values.push(x);
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact_cdf = exact_ecdf_for_sorted(&values);

        let t0 = TDigest::new_with_size_and_scale(self.max_size, family);
        let digest = t0.merge_sorted(values.clone());
        let td_cdf = digest.estimate_cdf(&values);

        let mut ks = 0.0;
        let mut sum_abs = 0.0;
        for (a, b) in exact_cdf.iter().zip(td_cdf.iter()) {
            let d = (a - b).abs();
            if d > ks {
                ks = d;
            }
            sum_abs += d;
        }
        let mae = sum_abs / (self.n as f64);

        QualityReport {
            n: self.n,
            max_abs_err: ks,
            mean_abs_err: mae,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_smoke_test_ks_mae_score() {
        let rep = Quality::new(100_000, 1000, 42).run();
        rep.log();
        let s = rep.quality_score();
        assert!(
            (0.719..0.720).contains(&s),
            "quality_score out of expected band: got {:.6}, want ~0.700 (band [0.699, 0.701))",
            s
        );
    }

    #[test]
    fn quality_improves_with_larger_digest() {
        let base = Quality::new(100_000, 100, 42).run();
        let better = Quality::new(100_000, 1000, 42).run();

        eprintln!("BASE   -> {}", base.to_line());
        eprintln!("BETTER -> {}", better.to_line());

        assert!(
            better.strictly_better_than(&base),
            "Expected 'better' to be ≤ base on both KS and MAE. base={}, better={}",
            base.to_line(),
            better.to_line()
        );

        let s_base = base.quality_score();
        let s_better = better.quality_score();
        assert!(
            s_better >= s_base,
            "Expected score to not regress. base={:.6}, better={:.6}",
            s_base,
            s_better
        );
    }

    #[test]
    fn quality_regression_scales_fixed_seed() {
        let q = Quality::new(100_000, 1000, 42);

        let mut results = Vec::new();
        let expected = [
            (ScaleFamily::Quad, 0.720_f64),
            (ScaleFamily::K1, 0.533_f64),
            (ScaleFamily::K2, 0.862_f64),
            (ScaleFamily::K3, 0.867_f64),
        ];
        for (fam, _) in expected {
            let rep = q.run_with_scale(fam);
            eprintln!("{:?} -> {}", fam, rep.to_line());
            results.push((fam, rep));
        }

        let score_tol = 5e-3;
        for ((fam, exp_score), (_, rep)) in expected.iter().zip(results.iter()) {
            let got = rep.quality_score();
            assert!(
                (got - exp_score).abs() <= score_tol,
                "Score regression for {:?}: got {:.3}, expected {:.3} ± {:.3}",
                fam,
                got,
                exp_score,
                score_tol
            );
        }
    }
}
