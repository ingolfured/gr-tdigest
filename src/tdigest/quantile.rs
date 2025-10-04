use super::TDigest;

impl TDigest {
    /// Estimate the value located at quantile `q` (0..=1).
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() {
            return 0.0;
        }

        let count_: f64 = self.count.into_inner();
        let rank: f64 = q * count_;

        // Early path for exact median on identical-value piles (even total only):
        // If both middle order-stat ranks fall strictly inside a *single* centroid
        // that is a same-mean pile (`singleton=true`), return its mean.
        if q == 0.5 && (count_ as i64) % 2 == 0 {
            let r_lo = (count_ as i64) / 2; // lower middle (1-based)
            let r_hi = r_lo + 1; // upper middle (1-based)

            let mut prev = 0.0;
            for c in self.centroids.iter() {
                let next = prev + c.weight();
                let inside = (prev < r_lo as f64) && ((r_hi as f64) <= next);
                if inside && c.is_singleton() {
                    return c.mean();
                }
                prev = next;
            }
        }

        // Locate the containing centroid using the rank walk.
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

        // If we landed inside a pile of identical values, don't interpolate inside it.
        if self.centroids[pos].is_singleton() && self.centroids[pos].weight() > 1.0 {
            return self.centroids[pos].mean();
        }

        // Otherwise, compute a local slope `delta` from neighbors and interpolate,
        // clamping to adjacent means to keep monotonicity and avoid overshoot.
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

    /// Find the pair of centroid indices that bracket q=0.5 using the same rank logic
    /// as `estimate_quantile`, but returning neighbors so we can average their means
    /// for even totals.
    fn bracket_centroids_for_median(&self) -> (usize, usize) {
        debug_assert!(!self.centroids.is_empty());
        if self.centroids.len() == 1 {
            return (0, 0);
        }

        let count_ = self.count();
        let rank = 0.5 * count_;

        // Walk from the left (q <= 0.5 branch) to locate the containing cell.
        let mut t = 0.0;
        for (k, c) in self.centroids.iter().enumerate() {
            if rank < t + c.weight() {
                // Inside centroid k → bracket is (k-1, k) if k>0, else (0,1).
                if k == 0 {
                    return (0, 1);
                } else {
                    return (k - 1, k);
                }
            }
            t += c.weight();
        }

        // If we fall off the end due to numeric wiggle, use the last two.
        let m = self.centroids.len();
        (m - 2, m - 1)
    }

    /// Median with an even-count special case to avoid over-interpolation.
    /// - If total count is even: return the average of the two *neighboring centroid means*
    ///   around q=0.5 (independent of centroid weights).
    /// - If odd: fall back to `estimate_quantile(0.5)`.
    pub fn estimate_median(&self) -> f64 {
        let total = self.count().round() as i64;
        if total <= 0 {
            return 0.0;
        }
        if total % 2 != 0 {
            return self.estimate_quantile(0.5);
        }
        let (il, ir) = self.bracket_centroids_for_median();
        (self.centroids[il].mean() + self.centroids[ir].mean()) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use crate::tdigest::test_helpers::*;
    use crate::tdigest::TDigest;

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
        let means: Vec<f64> = t.centroids.iter().map(|c| c.mean.into_inner()).collect();
        assert_eq!(
            means.len(),
            2,
            "expected exactly two centroids, got {:?}",
            means
        );
        assert!(
            means.contains(&1.0) && means.contains(&2.0),
            "expected centroid means [1.0, 2.0], got {:?}",
            means
        );
    }

    /// n=10, max_size=10 — edge clamps, median bracket, monotone grid.
    #[test]
    fn quantiles_small_max10_smalln() {
        let mut values = vec![-10.0, -1.0, 0.0, 0.0, 2e-10, 1.0, 2.0, 10.0, 1e9, -1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigest::new_with_size(10).merge_sorted(values.clone());

        assert_exact("Q(0)", *values.first().unwrap(), t.estimate_quantile(0.0));
        assert_exact("Q(1)", *values.last().unwrap(), t.estimate_quantile(1.0));

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

    /// Even-count median special-case (independent of centroid weights).
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
            assert_exact("median()", 0.0, t.estimate_median());
        }
        assert!(quantile_didnt_work);
    }
}
