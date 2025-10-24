//! CDF (cumulative distribution function) evaluation for `TDigest`.
//!
//! This module provides a fast, allocation-lean evaluator for
//! `TDigest::estimate_cdf`, including an optimized scalar kernel and an
//! auto-switch to Rayon for large query batches.
//!
//! # Semantics
//! This implementation aims for robust, predictable CDF behavior under streaming and compressed data:
//! - **Outside support**: clamp to `{0, 1}` (strictly below `min` → `0`, strictly
//!   above `max` → `1`).
//! - **Left/right tails**: guarded linear ramps between `min↔mean[0]` and
//!   `mean[last]↔max`, where the edge centroid contributes **half weight**.
//! - **Exact centroid hit**: return midpoint mass, i.e. `(prefix + 0.5·w) / N`.
//! - **Between centroids**: center-to-center interpolation that **excludes
//!   singleton half-mass** from the interpolation span. Two adjacent singletons
//!   form a discrete step.
//!
//! # Guarantees
//! - Output is in **[0, 1]**.
//! - Output is **non-decreasing** in the query value.
//! - With sufficient capacity (no compression), the result matches the
//!   **midpoint ECDF** over ties.
//!
//! # Performance
//! - Per query is **O(log n)** due to a binary search on centroid means.
//! - A single temporary “light” view (`means`, `weights`, `prefix`, `count`) is
//!   built **once per call** (O(n)) and reused for every query value.
//! - For large batches the evaluation switches to **Rayon** parallelism
//!   (see [`PAR_MIN`]) to amortize setup costs across threads.
//!
//! # Numerical notes
//! - Tail ramps use explicit formulas that place exactly **0.5/N** at `min` and
//!   `1−0.5/N` at `max` when the extremal centroid has weight 1.
//! - When two adjacent centroids are **both singletons**, the interpolation span
//!   collapses to a **step** over the left singleton (discrete mass).
//! - When centroids are pathologically close (`gap ≤ 0`), the kernel falls back
//!   to midpoint mass to preserve monotonicity.
//!
//! # Example
//! ```rust
//! use tdigest_rs::tdigest::{TDigest, tdigest::TDigestBuilder, ScaleFamily};
//!
//! // Build a small digest and evaluate its CDF on a sorted grid
//! let values: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
//! let td = TDigestBuilder::new()
//!     .max_size(64)
//!     .scale(ScaleFamily::Quad)
//!     .build()
//!     .merge_sorted(values.clone());
//!
//! let cdf = td.estimate_cdf(&values);
//! assert_eq!(cdf.len(), values.len());
//! assert!(cdf.first().unwrap() >= &0.0 && cdf.last().unwrap() <= &1.0);
//! for w in cdf.windows(2) {
//!     assert!(w[1] + 1e-12 >= w[0]); // monotone
//! }
//! ```
//!
//! # Concurrency
//! The API takes `&self` and performs a read-only snapshot into temporary
//! vectors. It is safe to call concurrently from multiple threads.
//!
//! [`PAR_MIN`]: constant.PAR_MIN

use crate::tdigest::TDigest;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// Crossover for parallel evaluation with Rayon.
///
/// Keep this conservative: Rayon setup has a fixed cost. Below this size a
/// scalar loop is typically faster; above it, parallelism wins.
///
/// Heuristic, tuned to avoid overhead dominating small batches.
const PAR_MIN: usize = 32_768;

impl TDigest {
    /// Estimate the CDF at each `vals[i]`, returning values in **[0, 1]**.
    ///
    /// ## Semantics
    /// Mirrors the reference `MergingDigest.cdf()` with explicit handling of:
    /// - **Outside support**: clamp to `{0, 1}` using `self.min()`/`self.max()`.
    /// - **Tails**: linear ramps `min↔mean[0]` and `mean[last]↔max` with
    ///   **half-weight** at the adjacent edge centroid.
    /// - **Exact centroid mean**: midpoint mass `(prefix + 0.5·w)/N`.
    /// - **Between centroids**: center-to-center interpolation that **excludes**
    ///   singleton half-mass from the interpolation span (atomic piles do not
    ///   “smear”).
    ///
    /// ## Complexity
    /// - Build “light arrays” once: **O(m)** where `m = #centroids`.
    /// - Per query: **O(log m)** from binary search on centroid means.
    /// - For `vals.len() ≥ PAR_MIN`, queries are processed with **Rayon**.
    ///
    /// ## Edge cases
    /// - Empty digest (`m = 0`) → a vector of `NaN` with `vals.len()` entries.
    /// - Empty `vals` → returns an empty vector.
    ///
    /// ## Monotonicity & bounds
    /// The result is guaranteed to be within **[0, 1]** and non-decreasing in
    /// each query value.
    pub fn estimate_cdf(&self, vals: &[f64]) -> Vec<f64> {
        // Degenerate cases
        let n = self.centroids().len();
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        if vals.is_empty() {
            return Vec::new();
        }

        // Build "light arrays" once per call (means, weights, prefix, N).
        let CdfLight {
            means,
            weights,
            prefix,
            count_f64,
        } = build_arrays_light(self);

        let min_v = self.min();
        let max_v = self.max();

        // Tiny fast path: ≤ 8 queries → scalar loop. Common for point lookups.
        if vals.len() <= 8 {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(
                    v, &means, &weights, &prefix, count_f64, min_v, max_v,
                ));
            }
            return out;
        }

        // Main path: parallel for big batches, scalar otherwise.
        if vals.len() >= PAR_MIN {
            vals.par_iter()
                .with_min_len(4096)
                .map(|&v| cdf_at_val_fast(v, &means, &weights, &prefix, count_f64, min_v, max_v))
                .collect()
        } else {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(
                    v, &means, &weights, &prefix, count_f64, min_v, max_v,
                ));
            }
            out
        }
    }
}

/* ------------------------- PRIVATE HELPERS ------------------------- */

/// Lightweight snapshot of the digest state needed by the CDF kernel.
///
/// This avoids repeated virtual dispatch or accessor overhead inside the hot
/// loop and ensures tight, cache-friendly arrays.
struct CdfLight {
    means: Vec<f64>,
    weights: Vec<f64>,
    prefix: Vec<f64>,
    count_f64: f64,
}

#[inline]
fn build_arrays_light(td: &TDigest) -> CdfLight {
    let n = td.centroids().len();

    let mut means = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for c in td.centroids() {
        means.push(c.mean());
        weights.push(c.weight());
    }

    // prefix[i] = sum(weights[0..i])
    let mut prefix = Vec::with_capacity(n);
    let mut run = 0.0;
    for &w in &weights {
        prefix.push(run);
        run += w;
    }

    CdfLight {
        means,
        weights,
        prefix,
        count_f64: td.count(), // equals `run`, but use the accessor for clarity
    }
}

/* --------- Optimized evaluation kernel (reference semantics) ---------
Exact hit → (prefix[idx] + 0.5*w)/N.
Left/right tails → guarded ramps using min/max and edge centroid half-weights.
Between centroids → center-to-center interpolation with singleton exclusion. */
#[inline(always)]
fn cdf_at_val_fast(
    val: f64,
    means: &[f64],
    weights: &[f64],
    prefix: &[f64],
    count_f64: f64,
    min_v: f64,
    max_v: f64,
) -> f64 {
    let n = means.len();

    match means.binary_search_by(|m| m.partial_cmp(&val).unwrap()) {
        // Exact centroid hit: midpoint semantics (half-weight).
        Ok(idx) => (prefix[idx] + 0.5 * weights[idx]) / count_f64,

        Err(idx) => {
            // Left of first centroid mean
            if idx == 0 {
                if val < min_v {
                    return 0.0;
                }
                let m0 = means[0];
                let w0 = weights[0];
                let gap = m0 - min_v;
                if gap > 0.0 {
                    if val == min_v {
                        return 0.5 / count_f64;
                    }
                    // Guarded ramp from min → mean[0] with half-weight at edge centroid.
                    return (1.0 + (val - min_v) / gap * (w0 / 2.0 - 1.0)) / count_f64;
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
                let mn = means[n - 1];
                let wn = weights[n - 1];
                let gap = max_v - mn;
                if gap > 0.0 {
                    if val == max_v {
                        return 1.0 - 0.5 / count_f64;
                    }
                    // Guarded ramp from mean[last] → max with half-weight at edge centroid.
                    let dq = (1.0 + (max_v - val) / gap * (wn / 2.0 - 1.0)) / count_f64;
                    return 1.0 - dq;
                } else {
                    return 1.0;
                }
            }

            // Between centroids idx-1 and idx
            let li = idx - 1;
            let ri = idx;
            let ml = means[li];
            let mr = means[ri];
            let wl = weights[li];
            let wr = weights[ri];

            let gap = mr - ml;
            if gap <= 0.0 {
                // Pathological/too-close: fall back to midpoint mass to preserve monotonicity.
                let dw = 0.5 * (wl + wr);
                return (prefix[li] + dw) / count_f64;
            }

            // Two singletons bracketing → no interpolation: step over left singleton.
            if wl == 1.0 && wr == 1.0 {
                return (prefix[li] + 1.0) / count_f64;
            }

            // Singleton exclusion (atomic piles do not contribute 0.5 to the span).
            let left_excl = if wl == 1.0 { 0.5 } else { 0.0 };
            let right_excl = if wr == 1.0 { 0.5 } else { 0.0 };

            let dw = 0.5 * (wl + wr);
            let dw_no_singleton = dw - left_excl - right_excl;
            // Safety (mirrors reference invariants):
            // assert dw_no_singleton > dw / 2.0 && gap > 0.0

            // Base mass at left half-weight plus any left exclusion.
            let base = prefix[li] + wl / 2.0 + left_excl;
            let frac = (val - ml) / gap;
            (base + dw_no_singleton * frac) / count_f64
        }
    }
}

/* --------- Reference exact ECDF for tests (midpoint semantics over ties) --------- */

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use crate::tdigest::{tdigest::TDigestBuilder, ScaleFamily};

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
            .merge_sorted(values.clone());
        let approx = td.estimate_cdf(&values);
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

        let exact = super::exact_ecdf_for_sorted(&vals);
        let approx = TDigestBuilder::new()
            .max_size(N + 1)
            .build()
            .merge_sorted(vals.clone())
            .estimate_cdf(&vals);

        for (i, (&e, &a)) in exact.iter().zip(&approx).enumerate() {
            assert_exact(&format!("CDF[{i}]"), e, a);
        }
    }
}
