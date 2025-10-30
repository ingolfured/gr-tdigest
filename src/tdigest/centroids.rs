// src/tdigest/centroids.rs
use core::cmp::Ordering;
use ordered_float::{FloatCore, OrderedFloat};
use serde::{Deserialize, Serialize};

use crate::tdigest::precision::FloatLike;

/// A centroid describes a weighted point in the digest.
/// `mean` and `weight` are stored in `F`. `mean_ord` mirrors `mean` for total ordering.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Centroid<F: FloatLike + FloatCore> {
    mean: F,
    weight: F,
    /// Whether this centroid represents a singleton/pile originating from raw values.
    singleton: bool,
    /// Cached ordered view for total ordering by mean; not serialized.
    #[serde(skip_serializing, skip_deserializing)]
    mean_ord: OrderedFloat<F>,
}

// `Ord` requires `Eq`. Our equality is structural, so this is OK.
impl<F: FloatLike + FloatCore> Eq for Centroid<F> {}

impl<F: FloatLike + FloatCore> Centroid<F> {
    /// Construct a centroid (mixed by default).
    #[inline]
    pub fn new(mean_f64: f64, weight_f64: f64) -> Self {
        let m = F::from_f64(mean_f64);
        let w = F::from_f64(weight_f64);
        Self {
            mean: m,
            weight: w,
            singleton: false,
            mean_ord: OrderedFloat::from(m),
        }
    }

    #[inline]
    pub fn new_singleton_f64(mean: f64, weight: f64) -> Self {
        let m = F::from_f64(mean);
        let w = F::from_f64(weight);
        Self {
            mean: m,
            weight: w,
            singleton: true,
            mean_ord: OrderedFloat::from(m),
        }
    }

    #[inline]
    pub fn new_mixed_f64(mean: f64, weight: f64) -> Self {
        let m = F::from_f64(mean);
        let w = F::from_f64(weight);
        Self {
            mean: m,
            weight: w,
            singleton: false,
            mean_ord: OrderedFloat::from(m),
        }
    }

    #[inline]
    pub fn mark_singleton(&mut self, yes: bool) {
        self.singleton = yes;
    }

    #[inline]
    pub fn mean_f64(&self) -> f64 {
        self.mean.to_f64()
    }
    #[inline]
    pub fn weight_f64(&self) -> f64 {
        self.weight.to_f64()
    }
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.singleton
    }

    #[inline]
    pub fn mean(&self) -> F {
        self.mean
    }
    #[inline]
    pub fn weight(&self) -> F {
        self.weight
    }

    /// Used by normalization to coalesce same-mean piles.
    #[inline]
    pub fn add_weight_f64(&mut self, w: f64) {
        let nw = self.weight.to_f64() + w;
        self.weight = F::from_f64(nw);
    }
}

impl<F: FloatLike + FloatCore> Ord for Centroid<F> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.mean_ord.cmp(&other.mean_ord)
    }
}

impl<F: FloatLike + FloatCore> PartialOrd for Centroid<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Utility: verify strictly increasing means (no ties) after compression.
#[inline]
pub fn is_sorted_strict_by_mean<F: FloatLike + FloatCore>(cs: &[Centroid<F>]) -> bool {
    cs.windows(2).all(|w| w[0].mean_f64() < w[1].mean_f64())
}
