// src/tdigest/cdf.rs
use crate::tdigest::TDigest;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// Crossover for parallel evaluation with Rayon.
/// Keep conservative to avoid overhead dominating small batches.
const PAR_MIN: usize = 32_768;

impl TDigest {
    /// Estimate the CDF at each `vals[i]`, returning values in [0, 1].
    ///
    /// Implementation notes:
    /// - Single implementation path using on-the-fly "light arrays" (means, weights, prefix).
    /// - Tiny fast path for very small query batches (≤ 8) to avoid Rayon and bounds checks noise.
    /// - Parallelized for large `vals` via Rayon with a conservative crossover.
    pub fn estimate_cdf(&self, vals: &[f64]) -> Vec<f64> {
        // Degenerate cases
        let n = self.centroids().len();
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        if vals.is_empty() {
            return Vec::new();
        }

        // Build "light arrays" once per call.
        let CdfLight {
            means,
            weights,
            prefix,
            count_f64,
        } = build_arrays_light(self);

        // Tiny fast path: ≤ 8 queries, scalar loop (hot for point lookups).
        if vals.len() <= 8 {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(v, &means, &weights, &prefix, count_f64));
            }
            return out;
        }

        // Main path: parallel for big batches, scalar otherwise.
        if vals.len() >= PAR_MIN {
            vals.par_iter()
                .with_min_len(4096)
                .map(|&v| cdf_at_val_fast(v, &means, &weights, &prefix, count_f64))
                .collect()
        } else {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(v, &means, &weights, &prefix, count_f64));
            }
            out
        }
    }
}

/* ------------------------- PRIVATE HELPERS ------------------------- */

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

/* --------- Optimized evaluation kernel ---------
Exact/snap/halfway paths divide by count (to match exact ECDF rounding),
interpolation path does a single division by multiplying once by inv_count. */
#[inline(always)]
fn cdf_at_val_fast(
    val: f64,
    means: &[f64],
    weights: &[f64],
    prefix: &[f64],
    count_f64: f64,
) -> f64 {
    let n = means.len();

    match means.binary_search_by(|m| m.partial_cmp(&val).unwrap()) {
        // Exact centroid hit.
        Ok(idx) => (prefix[idx] + 0.5 * weights[idx]) / count_f64,

        Err(idx) => {
            if idx == 0 {
                return 0.0;
            }
            if idx == n {
                return 1.0;
            }

            let cl_mean = means[idx - 1];
            let cr_mean = means[idx];
            let cl_w = weights[idx - 1];
            let cr_w = weights[idx];

            // Snap to unit steps — STRICT '< 0.5' so midpoint doesn't snap.
            if cl_w == 1.0 && (val - cl_mean).abs() < 0.5 {
                return (prefix[idx - 1] + 0.5) / count_f64;
            }
            if cr_w == 1.0 && (val - cr_mean).abs() < 0.5 {
                return (prefix[idx] + 0.5) / count_f64;
            }

            // Symmetric halfway between two unit steps → average with division.
            let mid = 0.5 * (cl_mean + cr_mean);
            if cl_w == 1.0 && cr_w == 1.0 && (val - mid).abs() < 1e-12 {
                let left = (prefix[idx - 1] + 0.5) / count_f64;
                let right = (prefix[idx] + 0.5) / count_f64;
                return 0.5 * (left + right);
            }

            // Linear interpolation in the gap.
            let inv_count = 1.0 / count_f64;
            let gap = cr_mean - cl_mean;
            // denom = 0.5 * (cl_w + cr_w); slope m = gap/denom → use mul = denom/gap == 1/m
            let denom = 0.5 * (cl_w + cr_w);
            let mul = denom / gap;
            let x = (val - cl_mean) * mul;
            let base = prefix[idx] - 0.5 * cl_w; // left-mid prefix
            (base + x) * inv_count
        }
    }
}

/* --------- Reference exact ECDF for tests --------- */

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

/* ------------------------------ TESTS ------------------------------ */

#[cfg(test)]
mod tests {
    use crate::tdigest::scale::q_to_k;
    use crate::tdigest::test_helpers::*;
    use crate::tdigest::SingletonPolicy;
    use crate::tdigest::TDigest;

    fn assert_cdf_oob_clamps(label: &str, td: &TDigest, below: f64, above: f64) {
        let v = td.estimate_cdf(&[below, above]);
        assert!((v[0] - 0.0).abs() == 0.0, "{label}/min-ε: {}", v[0]);
        assert!((v[1] - 1.0).abs() == 0.0, "{label}/max+ε: {}", v[1]);
    }

    #[test]
    fn cdf_small_max10_smalln() {
        let mut values = vec![-1e9, -5.0, -2.0, 0.0, 2.0, 5.0, 1e-10, 2e-10, 2e-10, 1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = TDigest::new_with_size(10).merge_sorted(values.clone());

        assert_cdf_oob_clamps(
            "cdf_small_max10_smalln",
            &td,
            values[0] - 1.0,
            values[values.len() - 1] + 1.0,
        );

        let trio = td.estimate_cdf(&[1e-10, 2e-10, 5.0]);
        assert_strict_increasing("cdf trio", &trio);
        assert_all_in_unit_interval("cdf trio bounds", &trio);
    }

    #[test]
    fn cdf_small_max10_medn_100() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        values[0] = -1e9;      // Extreme low value
        values[1] = -30.0;     // Boundary condition
        values[98] = 1e-10;    // Very small value
        values[99] = 1e9;      // Extreme high value
        values.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort values

        let exact = super::exact_ecdf_for_sorted(&values);

        // Compress the values into TDigest with max size 10
        let td = TDigest::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)  // Force merging of all centroids
            .build()
            .merge_sorted(values.clone());

        // Debugging: Print centroids after merging
        eprintln!("Centroids after merging (max_size=10, SingletonPolicy::Off):");
        for (i, centroid) in td.centroids().iter().enumerate() {
            eprintln!("Centroid {}: mean = {:.6e}, weight = {:.6e}", i, centroid.mean(), centroid.weight());
        }

        // Estimate CDF using TDigest
        let approx = td.estimate_cdf(&values);

        // Calculate KS and MAE
        let (ks, mae) = ks_mae(&exact, &approx);

        // Debugging outputs to inspect the failure
        eprintln!("Exact ECDF: {:?}", exact);
        eprintln!("Approximate CDF: {:?}", approx);
        eprintln!("KS statistic: {:.6e}", ks);
        eprintln!("MAE: {:.6e}", mae);

        // More debugging info
        eprintln!("==================== Debugging Compression Process =====================");
        let result = td.centroids();
        for (i, centroid) in result.iter().enumerate() {
            eprintln!("Centroid {}: mean = {:.6e}, weight = {:.6e}", i, centroid.mean(), centroid.weight());
        }

        // More debugging on the k-space and merging thresholds
        eprintln!("==================== Inspecting k-space Logic =====================");
        let total_weight: f64 = result.iter().map(|c| c.weight()).sum();
        eprintln!("Total weight of all centroids: {:.6e}", total_weight);

        // Print out the k-space parameters for the first few centroids
        for i in 0..result.len().min(5) {
            let centroid = &result[i];
            let q_r = centroid.weight() * total_weight; // Simplified for demonstration
            let k_r = q_to_k(q_r, 1.0, td.scale());  // Substitute actual scale factor
            eprintln!("Centroid {}: k_r = {:.6e}, q_r = {:.6e}", i, k_r, q_r);
        }

        // Check how many centroids were generated and how they are distributed
        eprintln!("Total number of centroids: {}", result.len());
        
        // Assert CDF KS and MAE are within expected bounds
        assert!(ks < 0.035, "CDF KS too large: {:.6e}", ks);
        assert!(mae < 0.003, "CDF MAE too large: {:.6e}", mae);

        // Assert that the estimated CDF values are within the unit interval [0, 1]
        assert_all_in_unit_interval("cdf(100) bounds", &approx);

        // Assert that the estimated CDF is monotonic
        assert_monotone_chain("cdf(100) monotone", &approx);
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
        let approx = TDigest::new_with_size(N + 1)
            .merge_sorted(vals.clone())
            .estimate_cdf(&vals);

        for (i, (&e, &a)) in exact.iter().zip(&approx).enumerate() {
            assert_exact(&format!("CDF[{i}]"), e, a);
        }
    }
}
