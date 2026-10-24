//! Read-only inputs shared by CDF and quantile evaluation.
//!
//! Ordinary queries borrow centroids. Only positive subnormal total mass needs
//! an owned, normalized centroid slice; raw sums and configuration are irrelevant
//! to these kernels and are never copied or transformed.

use std::borrow::Cow;

use super::centroids::Centroid;
use super::precision::FloatLike;
use super::TDigest;
use ordered_float::FloatCore;

pub(super) struct QueryView<'a, F: FloatLike + FloatCore> {
    centroids: Cow<'a, [Centroid<F>]>,
    total_weight: f64,
    min: f64,
    max: f64,
}

impl<'a, F: FloatLike + FloatCore> QueryView<'a, F> {
    /// The caller chooses the existing query's total: CDF uses its prefix sum,
    /// whereas the ordinary quantile path uses the stored digest count.
    pub(super) fn new(digest: &'a TDigest<F>, total_weight: f64) -> Self {
        Self {
            centroids: Cow::Borrowed(digest.centroids()),
            total_weight,
            min: digest.min(),
            max: digest.max(),
        }
    }

    /// Two exact power-of-two scales make each weight normal and place the
    /// actual total in [1, 2). Keep the factors separate to avoid overflow.
    /// This cannot restore information already lost during compression.
    pub(super) fn normalize_subnormal_mass(mut self, actual_mass: f64) -> Self {
        debug_assert!(actual_mass > 0.0 && actual_mass < f64::MIN_POSITIVE);
        const FIRST_SCALE: f64 = 1.0 / f64::MIN_POSITIVE;
        let first_total = actual_mass * FIRST_SCALE;
        // first_total is normal in [2^-52, 1), so the second exponent is 1..52.
        let exponent = ((first_total.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let second_scale = f64::from_bits(((1023 - exponent) as u64) << 52);
        self.centroids = Cow::Owned(
            self.centroids
                .iter()
                .map(|c| c.rescaled(1.0, FIRST_SCALE).rescaled(1.0, second_scale))
                .collect(),
        );
        self.total_weight = first_total * second_scale;
        debug_assert!(self.total_weight >= 1.0 && self.total_weight < 2.0);
        self
    }

    #[inline]
    pub(super) fn centroids(&self) -> &[Centroid<F>] {
        &self.centroids
    }

    #[inline]
    pub(super) fn count(&self) -> f64 {
        self.total_weight
    }

    #[inline]
    pub(super) fn min(&self) -> f64 {
        self.min
    }

    #[inline]
    pub(super) fn max(&self) -> f64 {
        self.max
    }
}
