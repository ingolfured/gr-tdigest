// src/quality.rs
use crate::tdigest::TDigest;

#[derive(Debug, Clone)]
pub struct Quality {
    pub n: usize,
    pub max_size: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub n: usize,
    pub max_abs_err: f64, // KS
    pub mean_abs_err: f64, // MAE
}

impl QualityReport {
    /// Half-lives chosen so base lands around ~0.7
    pub fn quality_score(&self) -> f64 {
        self.quality_score_with_halves(0.028, 0.0008)
    }

    /// Customizable half-lives: score hits 0.5 when KS==half_ks and MAE==half_mae.
    pub fn quality_score_with_halves(&self, half_ks: f64, half_mae: f64) -> f64 {
        fn sub(x: f64, half: f64) -> f64 {
            if half <= 0.0 || !x.is_finite() { return 0.0; }
            (2.0f64).powf(-x / half).clamp(0.0, 1.0)
        }
        let s_ks  = sub(self.max_abs_err, half_ks);
        let s_mae = sub(self.mean_abs_err, half_mae);
        (s_ks * s_mae).sqrt()
    }

    pub fn strictly_better_than(&self, other: &QualityReport) -> bool {
        let eps = 1e-12;
        (self.max_abs_err <= other.max_abs_err + eps) &&
        (self.mean_abs_err <= other.mean_abs_err + eps)
    }

    pub fn to_line(&self) -> String {
        format!(
            "QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
            self.n, self.max_abs_err, self.mean_abs_err, self.quality_score()
        )
    }
    pub fn log(&self) { eprintln!("{}", self.to_line()); }
}

impl Quality {
    pub fn new(n: usize, max_size: usize, seed: u64) -> Self {
        Self { n, max_size, seed }
    }

    pub fn run(&self) -> QualityReport {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut values: Vec<f64> = Vec::with_capacity(self.n);
        if self.n == 0 {
            return QualityReport { n: 0, max_abs_err: f64::NAN, mean_abs_err: f64::NAN };
        }
        values.push(0.0); // guarantee a 0
        while values.len() < self.n {
            let bucket: u32 = rng.gen_range(0..100);
            let x = if bucket < 70 {
                // 70% uniform in [-1, 1]
                rng.gen_range(-1.0..1.0)
            } else if bucket < 90 {
                // 20% normal(0, 1000) via Box–Muller
                let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 1000.0
            } else {
                // 10% very large magnitudes: 10^U(3,9) with random sign
                let exp = rng.gen_range(3.0..9.0);
                let mag = 10f64.powf(exp);
                if rng.gen_bool(0.5) { mag } else { -mag }
            };
            values.push(x);
        }

        // Exact ECDF at each sorted sample (midpoint convention on ties)
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact_cdf = Self::exact_ecdf_for_sorted(&values);

        let t0 = TDigest::new_with_size(self.max_size);
        let digest = t0.merge_sorted(values.clone()); // taking-ownership API => clone
        let td_cdf = digest.estimate_cdf(&values);

        let mut ks = 0.0;
        let mut sum_abs = 0.0;
        for (a, b) in exact_cdf.iter().zip(td_cdf.iter()) {
            let d = (a - b).abs();
            if d > ks { ks = d; }
            sum_abs += d;
        }
        let mae = sum_abs / (self.n as f64);

        QualityReport { n: self.n, max_abs_err: ks, mean_abs_err: mae }
    }

    pub(crate) fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
        let n = sorted.len();
        let nf = n as f64;
        let mut out = vec![0.0; n];
        let mut i = 0usize;
        while i < n {
            let mut j = i + 1;
            while j < n && sorted[j] == sorted[i] { j += 1; }
            let mid = (i + j) as f64 / 2.0;
            let val = mid / nf;
            for k in i..j { out[k] = val; }
            i = j;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_smoke_test_ks_mae_score() {
        let rep = Quality::new(100_000, 100, 42).run();
        rep.log();
        let s = rep.quality_score();
        eprintln!("quality_score = {:.3}", s);
        assert!(
            (0.70..0.71).contains(&s),
            "quality_score out of expected band: got {:.6}, want ~0.705 (band [0.70, 0.71))",
            s
        );
    }

    #[test]
    fn quality_improves_with_larger_digest() {
        let base   = Quality::new(100_000, 100, 42).run();
        let better = Quality::new(100_000, 1000, 42).run();

        eprintln!("BASE   -> {}", base.to_line());
        eprintln!("BETTER -> {}", better.to_line());

        assert!(
            better.strictly_better_than(&base),
            "Expected 'better' to be ≤ base on both KS and MAE. base={}, better={}",
            base.to_line(), better.to_line()
        );

        let s_base   = base.quality_score();
        let s_better = better.quality_score();
        assert!(
            s_better >= s_base,
            "Expected score to not regress. base={:.6}, better={:.6}",
            s_base, s_better
        );
    }
}
