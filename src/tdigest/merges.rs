//! Merge helpers for TDigest centroids.
//!
//! This module provides two utilities used by the TDigest builder/merge path:
//! - `MergeByMean<F>`: merges a slice of centroids with a sorted slice of values
//!   into a single centroid stream ordered by mean.
//! - `KWayCentroidMerge<F>`: k-way merge of multiple centroid runs, ordered by mean.
//! - `normalize_stream`: tiny normalization pass used by the compressor (kept here
//!   for minimal churn), coalescing equal-mean runs into singleton piles and
//!   validating sort order.
//!
//! The API uses `mean_f64()` / `weight_f64()` to keep math in `f64`,
//! and constructs new centroids via `Centroid::<F>::new_singleton_f64(...)`
//! when materializing scalars.

use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;

/* =============================================================================
 * MergeByMean
 * ============================================================================= */

pub struct MergeByMean<F: FloatLike + FloatCore> {
    data: Vec<Centroid<F>>,
}

impl<F: FloatLike + FloatCore> MergeByMean<F> {
    /// Merge two sorted sources by mean:
    /// - `centroids`: already-sorted by mean (TDigest invariant)
    /// - `values_sorted`: raw scalar values sorted ascending
    pub fn from_centroids_and_values(centroids: &[Centroid<F>], values_sorted: &[F]) -> Self {
        // Fast paths
        if values_sorted.is_empty() {
            return Self {
                data: centroids.to_vec(),
            };
        }
        if centroids.is_empty() {
            let mut out: Vec<Centroid<F>> = Vec::with_capacity(values_sorted.len());
            for &v in values_sorted {
                out.push(Centroid::<F>::new_singleton_f64(v.to_f64(), 1.0));
            }
            return Self { data: out };
        }

        // General case: two-way merge.
        let mut out: Vec<Centroid<F>> = Vec::with_capacity(centroids.len() + values_sorted.len());

        let mut i = 0usize;
        let mut j = 0usize;

        // (Removed optional min/max sanity block to avoid Option<f64> -> f64 mismatch.)

        while i < centroids.len() && j < values_sorted.len() {
            let cm = centroids[i].mean_f64();
            let vm = values_sorted[j].to_f64();

            if cm <= vm {
                out.push(centroids[i]);
                i += 1;
            } else {
                out.push(Centroid::<F>::new_singleton_f64(vm, 1.0));
                j += 1;
            }
        }

        while i < centroids.len() {
            out.push(centroids[i]);
            i += 1;
        }
        while j < values_sorted.len() {
            out.push(Centroid::<F>::new_singleton_f64(
                values_sorted[j].to_f64(),
                1.0,
            ));
            j += 1;
        }

        Self { data: out }
    }
}

impl<F: FloatLike + FloatCore> IntoIterator for MergeByMean<F> {
    type Item = Centroid<F>;
    type IntoIter = std::vec::IntoIter<Centroid<F>>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/* =============================================================================
 * K-way merge of centroid runs
 * ============================================================================= */

pub struct KWayCentroidMerge<F: FloatLike + FloatCore> {
    data: Vec<Centroid<F>>,
}

impl<F: FloatLike + FloatCore> KWayCentroidMerge<F> {
    /// Build a single sorted stream by concatenating + sorting small runs.
    /// For typical TDigest usage runs are already sorted; this is a simple
    /// and predictable implementation without a heap.
    pub fn from_runs(runs: &[&[Centroid<F>]]) -> Self {
        let mut all: Vec<Centroid<F>> = Vec::new();
        for r in runs {
            all.extend_from_slice(r);
        }
        // Ensure global ordering by mean (fix borrow on total_cmp).
        all.sort_by(|a, b| a.mean_f64().total_cmp(b.mean_f64()));
        Self { data: all }
    }
}

impl<F: FloatLike + FloatCore> IntoIterator for KWayCentroidMerge<F> {
    type Item = Centroid<F>;
    type IntoIter = std::vec::IntoIter<Centroid<F>>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/* =============================================================================
 * Normalization shim used by compressor (Stage 1)
 * ============================================================================= */

/// Output of `normalize_stream`.
pub struct Normalized<F: FloatLike + FloatCore> {
    pub out: Vec<Centroid<F>>,
    pub total_w: f64,
    pub total_mw: f64,
    pub min: f64,
    pub max: f64,
}

/// Normalize an input stream of centroids:
/// - **validate** non-decreasing means (panic on violation to match tests)
/// - **coalesce** adjacent equal-mean centroids into a **singleton pile**
/// - **accumulate** total weight/sum and min/max
pub fn normalize_stream<F, I>(items: I) -> Normalized<F>
where
    F: FloatLike + FloatCore,
    I: IntoIterator<Item = Centroid<F>>,
{
    let mut it = items.into_iter();
    let mut out: Vec<Centroid<F>> = Vec::new();

    let mut total_w = 0.0_f64;
    let mut total_mw = 0.0_f64;

    let first = match it.next() {
        Some(c) => c,
        None => {
            return Normalized {
                out,
                total_w,
                total_mw,
                min: 0.0,
                max: 0.0,
            }
        }
    };

    let mut cur_mean = first.mean_f64();
    let mut cur_w = first.weight_f64();

    let min_mean = cur_mean;
    let mut max_mean = cur_mean;

    // Fold contiguous equal-means; validate ordering.
    for c in it {
        let m = c.mean_f64();
        let w = c.weight_f64();

        if m < cur_mean {
            panic!("compress_into requires non-decreasing means");
        }

        if m == cur_mean {
            cur_w += w;
        } else {
            // Flush previous run as a **singleton** pile.
            out.push(Centroid::<F>::new_singleton_f64(cur_mean, cur_w));
            total_w += cur_w;
            total_mw += cur_mean * cur_w;

            // Start new run
            cur_mean = m;
            cur_w = w;
        }

        if m > max_mean {
            max_mean = m;
        }
    }

    // Flush last run
    out.push(Centroid::<F>::new_singleton_f64(cur_mean, cur_w));
    total_w += cur_w;
    total_mw += cur_mean * cur_w;

    Normalized {
        out,
        total_w,
        total_mw,
        min: min_mean,
        max: max_mean,
    }
}
