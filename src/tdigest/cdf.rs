//! CDF (cumulative distribution function) evaluation for `TDigest`.
//!
//! Allocation-lean kernel that operates directly on the centroid slice.
//! We keep a single `prefix` vector (running weight sums) per call and
//! avoid snapshotting means/weights into separate arrays.
//!
//! # Semantics
//! - **Outside support**: clamp to `{0, 1}` (strictly below `min` → `0`,
//!   strictly above `max` → `1`).
//! - **Left/right tails**: guarded linear ramps between `min↔mean[0]` and
//!   `mean[last]↔max`, where the adjacent edge centroid contributes **half weight**.
//! - **Exact centroid hit**: return midpoint mass, i.e. `(prefix + 0.5·w) / N`.
//! - **Between centroids**: center-to-center interpolation that **excludes
//!   atomic half-mass** from the interpolation span. Two atomic neighbors
//!   produce a **step** (no smearing).
//!
//! # Guarantees
//! - Output is in **[0, 1]** (except for explicit NaN propagation / empty digest).
//! - Output is **non-decreasing** in the query value.
//! - With sufficient capacity (no compression), the result matches the
//!   **midpoint ECDF** over ties.
//!
//! # Edge cases
//! - **Empty digest** → CDF returns a vector of **NaN** (one per input probe).
//! - **NaN probe**    → that output element is **NaN**.
//!
//! # Performance
//! - Per query is **O(log n)** due to a binary search on centroid means.
//! - A single `prefix` array is built **once per call** (O(n)) and reused.
//! - For very large batches, we switch to Rayon parallelism.

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::TDigest;
use ordered_float::FloatCore;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// Crossover for parallel evaluation with Rayon.
///
/// Keep this conservative: Rayon setup has a fixed cost. Below this size a
/// scalar loop is typically faster; above it, parallelism wins.
const PAR_MIN: usize = 32_768;

impl<F: FloatLike + FloatCore> TDigest<F> {
    /// Estimate the CDF at each `vals[i]`, returning values in **[0, 1]**.
    ///
    /// ## Semantics
    /// Mirrors the reference `MergingDigest.cdf()` with explicit handling of:
    /// - **Outside support**: clamps using `self.min()`/`self.max()`.
    /// - **Tails**: linear ramps `min↔mean[0]` and `mean[last]↔max` with
    ///   **half-weight** at the adjacent edge centroid.
    /// - **Exact mean hit**: midpoint mass `(prefix + 0.5·w)/N`.
    /// - **Between means**: center-to-center interpolation that **excludes**
    ///   `w/2` from any **atomic** neighbor (units and piles).
    ///
    /// ## Edge cases
    /// - Empty digest (`m = 0`) → a vector of `NaN` with `vals.len()` entries.
    /// - Empty `vals` → returns an empty vector.
    /// - **NaN probe** → that output element is `NaN`.
    pub fn cdf(&self, vals: &[f64]) -> Vec<f64> {
        let cents = self.centroids();
        let n = cents.len();

        // Degenerate cases
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        if vals.is_empty() {
            return Vec::new();
        }

        let mut prefix: Vec<f64> = Vec::with_capacity(n);
        let mut run: f64 = 0.0;
        for c in cents {
            prefix.push(run);
            run += c.weight_f64();
        }
        let count_f64 = run;
        if count_f64 > 0.0 && count_f64 < f64::MIN_POSITIVE {
            // Half-weight arithmetic can underflow even when every stored
            // weight is positive. Only this rare path allocates a query copy.
            return self
                .normalized_subnormal_mass_for_queries(count_f64)
                .cdf(vals);
        }

        let min_v = self.min();
        let max_v = self.max();

        // Sorted, sufficiently dense batches can share one forward cursor.
        // Keep short/sparse/parallel batches and NaNs on the existing path.
        if vals.len() >= 64
            && vals.len() < PAR_MIN
            && n <= vals.len()
            && vals.windows(2).all(|w| w[0] <= w[1])
        {
            let mut idx = 0;
            return vals
                .iter()
                .map(|&v| {
                    while idx < n && cents[idx].mean_f64() < v {
                        idx += 1;
                    }
                    let position = if idx < n && cents[idx].mean_f64() == v {
                        // Preserve the existing binary-search choice even for
                        // externally supplied duplicate centroid means.
                        cents.binary_search_by(|c| c.mean_f64().partial_cmp(&v).unwrap())
                    } else {
                        Err(idx)
                    };
                    cdf_at_position(v, cents, &prefix, count_f64, min_v, max_v, position)
                })
                .collect();
        }

        // Main path: parallel for big batches, scalar otherwise.
        if vals.len() >= PAR_MIN {
            vals.par_iter()
                .with_min_len(4096)
                .map(|&v| {
                    if v.is_nan() {
                        f64::NAN
                    } else {
                        cdf_at_val_fast(v, cents, &prefix, count_f64, min_v, max_v)
                    }
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                if v.is_nan() {
                    out.push(f64::NAN);
                } else {
                    out.push(cdf_at_val_fast(v, cents, &prefix, count_f64, min_v, max_v));
                }
            }
            out
        }
    }
}

// A difference between finite opposite-sign endpoints can overflow. Halving
// first preserves the fraction without introducing an infinite denominator.
fn fraction_between(value: f64, left: f64, right: f64) -> f64 {
    let span = right - left;
    if span.is_infinite() {
        (value * 0.5 - left * 0.5) / (right * 0.5 - left * 0.5)
    } else {
        (value - left) / span
    }
}

/* ------------------------- PRIVATE KERNEL ------------------------- */

/// Optimized evaluation kernel (reference semantics).
///
/// Exact hit → (prefix[idx] + 0.5*w)/N.
/// Left/right tails → guarded ramps using min/max and edge centroid half-weights.
/// Between centroids → center-to-center interpolation with **atomic** exclusion.
#[inline(always)]
fn cdf_at_val_fast<F: FloatLike + FloatCore>(
    val: f64,
    cents: &[Centroid<F>],
    prefix: &[f64],
    count_f64: f64,
    min_v: f64,
    max_v: f64,
) -> f64 {
    let position = cents.binary_search_by(|c| c.mean_f64().partial_cmp(&val).unwrap());
    cdf_at_position(val, cents, prefix, count_f64, min_v, max_v, position)
}

#[inline(always)]
fn cdf_at_position<F: FloatLike + FloatCore>(
    val: f64,
    cents: &[Centroid<F>],
    prefix: &[f64],
    count_f64: f64,
    min_v: f64,
    max_v: f64,
    position: Result<usize, usize>,
) -> f64 {
    let n = cents.len();

    match position {
        // Exact centroid hit: midpoint semantics (half-weight).
        Ok(idx) => (prefix[idx] + 0.5 * cents[idx].weight_f64()) / count_f64,

        Err(idx) => {
            // Left of first centroid mean
            if idx == 0 {
                if val < min_v {
                    return 0.0;
                }
                let m0 = cents[0].mean_f64();
                let w0 = cents[0].weight_f64();
                let gap = m0 - min_v;
                if gap > 0.0 {
                    // A mixed edge carries half its mass between the data minimum
                    // and its mean. No unit sample can be assumed for weighted data:
                    // fixed masses here break monotonicity after scaling weights.
                    return fraction_between(val, min_v, m0) * (0.5 * (w0 / count_f64));
                } else {
                    // Degenerate (all equal); idx==0 implies val<=min.
                    return 0.0;
                }
            }

            // Right of last centroid mean
            if idx == n {
                if val > max_v {
                    return 1.0;
                }
                let mn = cents[n - 1].mean_f64();
                let wn = cents[n - 1].weight_f64();
                let gap = max_v - mn;
                if gap > 0.0 {
                    // Symmetric half-mass ramp, independent of the weight units.
                    let dq = (1.0 - fraction_between(val, mn, max_v)) * (0.5 * (wn / count_f64));
                    return 1.0 - dq;
                } else {
                    return 1.0;
                }
            }

            // Between centroids idx-1 and idx
            let li = idx - 1;
            let ri = idx;
            let ml = cents[li].mean_f64();
            let mr = cents[ri].mean_f64();
            let wl = cents[li].weight_f64();
            let wr = cents[ri].weight_f64();

            let gap = mr - ml;
            if gap <= 0.0 {
                // Pathological/too-close: fall back to midpoint mass to preserve monotonicity.
                let dw = 0.5 * (wl + wr);
                return (prefix[li] + dw) / count_f64;
            }

            // Atomic-aware exclusion: subtract w/2 for ANY atomic neighbor.
            let left_excl = if cents[li].is_atomic() { wl * 0.5 } else { 0.0 };
            let right_excl = if cents[ri].is_atomic() { wr * 0.5 } else { 0.0 };

            // Two atomics ⇒ dw_span == 0 ⇒ pure step.
            let dw_center = 0.5 * (wl + wr);
            let dw_span = dw_center - left_excl - right_excl;

            // Base mass at left center plus any left exclusion.
            let base = prefix[li] + wl * 0.5 + left_excl;
            let frac = fraction_between(val, ml, mr);

            (base + dw_span * frac) / count_f64
        }
    }
}

/* --------- Reference exact ECDF for tests (midpoint semantics over ties) --------- */

#[cfg(test)]
mod tests {
    use crate::tdigest::{tdigest::TDigestBuilder, ScaleFamily};

    fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
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
            let mid = (i + j) as f64 * 0.5;
            let val = mid / nf;
            out.extend(std::iter::repeat_n(val, j - i));
            i = j;
        }
        out
    }

    #[test]
    fn cdf_small_max10_medn_100_simple() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        values[0] = -1e9; // extreme left tail
        values[99] = 1e9; // extreme right tail
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = TDigestBuilder::new()
            .max_size(10)
            .scale(ScaleFamily::Quad)
            .build()
            .merge_sorted(values.clone())
            .expect("no NaNs");
        let approx = td.cdf(&values);
        assert_eq!(approx.len(), values.len());

        // 1) bounds + 2) monotone
        for i in 0..approx.len() {
            let p = approx[i];
            assert!((0.0..=1.0).contains(&p), "cdf[{}]={} out of [0,1]", i, p);
            if i > 0 {
                assert!(approx[i] + 1e-12 >= approx[i - 1], "non-monotone at {}", i);
            }
        }

        // 3) tails hit ~1%/~99% (allow tiny slack for compression)
        let eps = 1e-6;
        assert!(
            approx[0] <= 0.01 + eps,
            "left tail too large: {}",
            approx[0]
        );
        assert!(
            approx[99] >= 0.99 - eps,
            "right tail too small: {}",
            approx[99]
        );

        // 4) middle rank (~median) should be roughly around 0.5
        let mid = 50;
        assert!(
            (0.49..=0.51).contains(&approx[mid]),
            "median sanity: {}",
            approx[mid]
        );
    }

    #[test]
    fn cdf_exact_with_enough_capacity() {
        use crate::tdigest::test_helpers::assert_exact;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        const N: usize = 9_999;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vals: Vec<f64> = (0..N)
            .map(|_| rng.random_range(0..N as u64) as f64)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let exact = exact_ecdf_for_sorted(&vals);
        let approx = TDigestBuilder::new()
            .max_size(N + 1)
            .build()
            .merge_sorted(vals.clone())
            .expect("no NaNs")
            .cdf(&vals);

        for (i, (&e, &a)) in exact.iter().zip(&approx).enumerate() {
            assert_exact(&format!("CDF[{i}]"), e, a);
        }
    }

    #[test]
    fn cdf_midpoint_ecdf_is_exact_at_training_values_under_capacity() {
        use crate::tdigest::test_helpers::assert_exact;

        // Sorted training data with ties; N is intentionally far below max_size.
        let vals = vec![-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 3.0, 7.0, 7.0];
        let n = vals.len();
        let td = TDigestBuilder::new()
            .max_size(1000)
            .build()
            .merge_sorted(vals.clone())
            .expect("no NaNs");

        assert!(n < td.max_size());

        let expected = exact_ecdf_for_sorted(&vals); // midpoint over ties
        let got = td.cdf(&vals);
        for (i, (&e, &g)) in expected.iter().zip(&got).enumerate() {
            assert_exact(&format!("CDF(midpoint exact @ training value) [{i}]"), e, g);
        }
    }

    #[test]
    fn cdf_between_two_atomic_centroids_is_flat_step() {
        use crate::tdigest::ScaleFamily;

        // Build two atomic centroids:
        // - left:  five identical zeros → atomic with w=5 at mean=0.0
        // - right: seven identical tens → atomic with w=7 at mean=10.0
        // N = 12 total
        let mut values: Vec<f64> = Vec::new();
        values.extend(std::iter::repeat(0.0).take(5));
        values.extend(std::iter::repeat(10.0).take(7));
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = TDigestBuilder::new()
            .max_size(64)
            .scale(ScaleFamily::K2)
            .build()
            .merge_sorted(values)
            .expect("no NaNs");

        // Sanity: compressor should coalesce equal means into two centroids.
        assert_eq!(td.centroids().len(), 2, "expected two centroids (0 and 10)");
        assert!((td.centroids()[0].mean_f64() - 0.0).abs() < 1e-12);
        assert!((td.centroids()[1].mean_f64() - 10.0).abs() < 1e-12);
        assert!((td.centroids()[0].weight_f64() - 5.0).abs() < 1e-12);
        assert!((td.centroids()[1].weight_f64() - 7.0).abs() < 1e-12);

        // Probes strictly between the two means.
        let probes = [1.0, 5.0, 9.0];

        // Expected: with correct atomic handling, there is **no** interpolative mass
        // contributed by either atomic neighbor (left_excl = wl/2, right_excl = wr/2),
        // so the between-mean CDF is a **flat step** at prefix_left + wl (all of left mass).
        // Here: prefix_left = 0, wl = 5 ⇒ expected = 5/12.
        let expected_between = 5.0 / 12.0;

        let out = td.cdf(&probes);
        assert_eq!(out.len(), probes.len());

        for (i, &p) in out.iter().enumerate() {
            // This assertion FAILS with current code (which only excludes 0.5 for units),
            // and will PASS after changing CDF span exclusion to subtract w/2 for any atomic.
            assert!(
                (p - expected_between).abs() <= 1e-9,
                "cdf({}) = {}, expected flat step {} between atomic centroids",
                probes[i],
                p,
                expected_between
            );
        }

        // Optional extra sanity: exact hit at the left centroid mean should be midpoint mass (2.5/12),
        // while any value just above the left mean should jump to 5/12 (the flat step).
        let exact = td.cdf(&[0.0])[0];
        assert!(
            (exact - (2.5 / 12.0)).abs() <= 1e-12,
            "exact hit at left mean should be midpoint mass"
        );
    }
    fn sorted_batch_reference<
        F: crate::tdigest::precision::FloatLike + ordered_float::FloatCore,
    >(
        digest: &crate::tdigest::TDigest<F>,
        probes: &[f64],
    ) {
        let mut prefix = Vec::new();
        let mut count = 0.0;
        for c in digest.centroids() {
            prefix.push(count);
            count += c.weight_f64();
        }
        let actual = digest.cdf(probes);
        if count > 0.0 && count < f64::MIN_POSITIVE {
            let scaled = digest.normalized_subnormal_mass_for_queries(count);
            sorted_batch_reference(&scaled, probes);
            let expected = scaled.cdf(probes);
            for (&a, &b) in actual.iter().zip(&expected) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            return;
        }
        for (&v, &got) in probes.iter().zip(&actual) {
            let expected = if v.is_nan() {
                f64::NAN
            } else {
                super::cdf_at_val_fast(
                    v,
                    digest.centroids(),
                    &prefix,
                    count,
                    digest.min(),
                    digest.max(),
                )
            };
            if expected.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), expected.to_bits(), "probe={v}");
            }
        }
    }

    #[test]
    fn sorted_batch_matches_binary_search_across_probe_shapes() {
        use crate::tdigest::singleton_policy::SingletonPolicy;
        for cap in [1, 2, 16, 128] {
            for policy in [SingletonPolicy::Off, SingletonPolicy::Use] {
                let vals: Vec<f64> = (0..2048)
                    .map(|i| ((i * 67 % 257) as f64 - 128.0) / 8.0)
                    .collect();
                let d64 = TDigestBuilder::<f64>::new()
                    .max_size(cap)
                    .singleton_policy(policy)
                    .build()
                    .merge_unsorted(vals.clone())
                    .unwrap();
                let d32 = TDigestBuilder::<f32>::new()
                    .max_size(cap)
                    .singleton_policy(policy)
                    .build()
                    .merge_unsorted(vals.into_iter().map(|v| v as f32).collect())
                    .unwrap();
                for n in [1, 63, 64, 128, 1024, 32768] {
                    let mut probes: Vec<f64> =
                        (0..n).map(|i| (i as f64 - n as f64 / 2.0) / 8.0).collect();
                    if n > 1 {
                        probes[0] = f64::NEG_INFINITY;
                        probes[n - 1] = f64::INFINITY;
                    }
                    sorted_batch_reference(&d64, &probes);
                    sorted_batch_reference(&d32, &probes);
                    probes.reverse();
                    sorted_batch_reference(&d64, &probes);
                    sorted_batch_reference(&d32, &probes);
                    probes[n / 2] = f64::NAN;
                    sorted_batch_reference(&d64, &probes);
                    sorted_batch_reference(&d32, &probes);
                }
                for v in [-16.0, -0.0, 0.0, 16.0] {
                    sorted_batch_reference(&d64, &vec![v; 128]);
                    sorted_batch_reference(&d32, &vec![v; 128]);
                }
            }
        }
    }
    #[test]
    fn sorted_batch_keeps_sparse_and_subnormal_controls() {
        use crate::tdigest::singleton_policy::SingletonPolicy;
        let dense = TDigestBuilder::<f64>::new()
            .max_size(2048)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .merge_sorted((0..1024).map(|v| v as f64).collect())
            .unwrap();
        let sparse: Vec<f64> = (0..64).map(|v| v as f64 * 16.0 + 0.25).collect();
        assert!(dense.centroids().len() > sparse.len());
        sorted_batch_reference(&dense, &sparse);
        let mut almost: Vec<f64> = (0..2048).map(|v| v as f64 / 2.0).collect();
        almost.swap(2046, 2047);
        sorted_batch_reference(&dense, &almost);
        let tiny = f64::from_bits(1);
        let d64 = TDigestBuilder::<f64>::new()
            .max_size(128)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .merge_weighted_unsorted(&[-1.0, 0.0, 2.0], &[tiny; 3])
            .unwrap();
        let d32 = TDigestBuilder::<f32>::new()
            .max_size(128)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .merge_weighted_unsorted(&[-1.0, 0.0, 2.0], &[tiny; 3])
            .unwrap();
        let probes: Vec<f64> = (0..128).map(|v| v as f64 / 16.0 - 4.0).collect();
        sorted_batch_reference(&d64, &probes);
        sorted_batch_reference(&d32, &probes);
    }
}
