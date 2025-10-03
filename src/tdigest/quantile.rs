use super::TDigest;

impl TDigest {
    /// To estimate the value located at `q` quantile
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() {
            return 0.0;
        }

        let count_: f64 = self.count.into_inner();
        let rank: f64 = q * count_;

        let mut pos: usize;
        let mut t: f64;
        if q > 0.5 {
            if q >= 1.0 {
                return self.max();
            }

            pos = 0;
            t = count_;

            for (k, centroid) in self.centroids.iter().enumerate().rev() {
                t -= centroid.weight();

                if rank >= t {
                    pos = k;
                    break;
                }
            }
        } else {
            if q <= 0.0 {
                return self.min();
            }

            pos = self.centroids.len() - 1;
            t = 0.0;

            for (k, centroid) in self.centroids.iter().enumerate() {
                if rank < t + centroid.weight() {
                    pos = k;
                    break;
                }

                t += centroid.weight();
            }
        }

        let mut delta = 0.0;
        let mut min: f64 = self.min.into_inner();
        let mut max: f64 = self.max.into_inner();

        if self.centroids.len() > 1 {
            if pos == 0 {
                delta = self.centroids[pos + 1].mean() - self.centroids[pos].mean();
                max = self.centroids[pos + 1].mean();
            } else if pos == (self.centroids.len() - 1) {
                delta = self.centroids[pos].mean() - self.centroids[pos - 1].mean();
                min = self.centroids[pos - 1].mean();
            } else {
                delta = (self.centroids[pos + 1].mean() - self.centroids[pos - 1].mean()) / 2.0;
                min = self.centroids[pos - 1].mean();
                max = self.centroids[pos + 1].mean();
            }
        }

        let value =
            self.centroids[pos].mean() + ((rank - t) / self.centroids[pos].weight() - 0.5) * delta;
        super::TDigest::clamp(value, min, max)
    }

    fn find_median_between_centroids(&self) -> Option<f64> {
        if (self.count.into_inner() as i64) % 2 != 0 {
            return None;
        }
        let mut target = (self.count.into_inner() as i64) / 2;
        for (idx, c) in self.centroids.iter().enumerate() {
            target -= c.weight() as i64;
            if target == 0 {
                let m1 = c.mean();
                let m2 = self.centroids[idx + 1].mean();
                return Option::Some((m1 + m2) / 2.0);
            }
            if target < 0 {
                return Option::None;
            }
        }
        Option::None
    }

    pub fn estimate_median(&self) -> f64 {
        /*
         * If the number of elements is even, median is average of two adjacent observation.
         * Interpolation algorithm used in `estimate_quantile` often positions estimated median too far away from the middle point.
         * So let's detect the case when the median is exactly between two centroids.
         */
        self.find_median_between_centroids()
            .unwrap_or(self.estimate_quantile(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::TDigest;

    // =============================== Helpers (for quantile tests) ===============================
    mod helpers {
        pub fn assert_rel_close(label: &str, expected: f64, got: f64, rtol: f64) {
            let denom = expected.abs().max(1e-300);
            let rel = ((expected - got).abs()) / denom;
            assert!(
                rel < rtol,
                "{}: expected ~= {:.9}, got {:.9}, rel_err={:.6e}, rtol={:.6e}",
                label,
                expected,
                got,
                rel,
                rtol
            );
        }
        pub fn assert_abs_close(label: &str, expected: f64, got: f64, atol: f64) {
            let abs = (expected - got).abs();
            assert!(
                abs <= atol,
                "{}: expected ~= {:.9}, got {:.9}, abs_err={:.6e}, atol={:.6e}",
                label,
                expected,
                got,
                abs,
                atol
            );
        }

        // Type-7 (NumPy/R default): interpolate between order stats at r = q*(n-1)
        pub fn bracket(values: &[f64], q: f64) -> (f64, f64, usize, usize) {
            assert!(!values.is_empty(), "bracket() requires non-empty values");
            let n = values.len();
            let q = q.clamp(0.0, 1.0);
            let r = q * (n.saturating_sub(1) as f64);

            let i_lo = r.floor() as usize;
            let i_hi = r.ceil() as usize;

            (values[i_lo], values[i_hi], i_lo, i_hi)
        }

        pub fn assert_in_bracket(label: &str, x: f64, lo: f64, hi: f64, i_lo: usize, i_hi: usize) {
            assert!(
                x >= lo && x <= hi,
                "{label}: {x} not in bracket [{lo}, {hi}] (i_lo={i_lo}, i_hi={i_hi})"
            );
        }

        pub fn assert_monotone_chain(label: &str, xs: &[f64]) {
            for i in 1..xs.len() {
                assert!(
                    xs[i] >= xs[i - 1],
                    "{label}: non-monotone at i={i}: {} < {}",
                    xs[i],
                    xs[i - 1]
                );
            }
        }
    }
    use helpers::*;

    // =============================== Quantiles ===============================
    #[test]
    fn centroid_addition_regression_pr_1() {
        // https://github.com/MnO2/t-digest/pull/1
        let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let mut t = TDigest::new_with_size(10);
        for v in vals {
            t = t.merge_unsorted(vec![v]);
        }
        assert_rel_close("median", 1.0, t.estimate_quantile(0.5), 0.01);
        assert_rel_close("q=0.95", 2.0, t.estimate_quantile(0.95), 0.01);
    }

    /// n=10, max_size=10 — edge clamps, median bracket, monotone grid.
    #[test]
    fn quantiles_small_max10_smalln() {
        let mut values = vec![-10.0, -1.0, 0.0, 0.0, 2e-10, 1.0, 2.0, 10.0, 1e9, -1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigest::new_with_size(10).merge_sorted(values.clone());

        assert_abs_close(
            "Q(0)",
            *values.first().unwrap(),
            t.estimate_quantile(0.0),
            0.0,
        );
        assert_abs_close(
            "Q(1)",
            *values.last().unwrap(),
            t.estimate_quantile(1.0),
            0.0,
        );

        let (lo_m, hi_m, _, _) = bracket(&values, 0.5);
        let med = t.estimate_quantile(0.5);
        assert_in_bracket("median", med, lo_m, hi_m, 4, 5);

        let grid = [
            t.estimate_quantile(0.01),
            t.estimate_quantile(0.10),
            t.estimate_quantile(0.25),
            med,
            t.estimate_quantile(0.75),
            t.estimate_quantile(0.90),
            t.estimate_quantile(0.99),
        ];
        assert_monotone_chain("quantiles grid", &grid);
    }

    /// n=100, max_size=10 — quantile lies between bracketing order stats.
    #[test]
    fn quantiles_small_max10_medn_100() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect(); // 100 values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigest::new_with_size(10).merge_sorted(values.clone());

        for &(q, label) in &[
            (0.01_f64, "Q(0.01)"),
            (0.10_f64, "Q(0.10)"),
            (0.25_f64, "Q(0.25)"),
            (0.50_f64, "Q(0.50)"),
            (0.75_f64, "Q(0.75)"),
            (0.90_f64, "Q(0.90)"),
            (0.99_f64, "Q(0.99)"),
        ] {
            let (lo, hi, i_lo, i_hi) = bracket(&values, q);
            let x = t.estimate_quantile(q);
            assert_in_bracket(label, x, lo, hi, i_lo, i_hi);
        }
    }

    /// Even-count median special-case (kept from original).
    #[test]
    fn median_between_centroids_even_count() {
        // median of [-1, -1, ..., 1, 1] should be ~0
        let mut quantile_didnt_work: bool = false;
        for num in [1, 2, 3, 10, 20] {
            let mut t = TDigest::new_with_size(100);
            for _ in 1..=num {
                t = t.merge_sorted(vec![-1.0]);
            }
            for _ in 1..=num {
                t = t.merge_sorted(vec![1.0]);
            }

            if t.estimate_quantile(0.5).abs() > 0.1 {
                quantile_didnt_work = true;
            }
            assert_abs_close("median()", 0.0, t.estimate_median(), 0.01);
        }
        assert!(quantile_didnt_work);
    }
}
