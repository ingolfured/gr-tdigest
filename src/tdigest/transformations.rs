//! Precision casts and value/weight transformations for the digest.
//! Weight scaling prepares and validates the full result before committing it.

use super::{DigestStats, TDigest};
use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::FloatLike;
use crate::{TdError, TdResult};
use ordered_float::FloatCore;

#[inline]
fn ensure_positive_finite_scale_factor(factor: f64, context: &'static str) -> TdResult<()> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(TdError::InvalidScaleFactor { context });
    }
    Ok(())
}

#[inline]
fn cast_centroid<From, To>(c: &Centroid<From>) -> Centroid<To>
where
    From: FloatLike + FloatCore,
    To: FloatLike + FloatCore,
{
    let mean = c.mean_f64();
    let weight = c.weight_f64();
    if c.is_atomic_unit() {
        Centroid::<To>::new_atomic_unit_f64(mean)
    } else if c.is_atomic() {
        Centroid::<To>::new_atomic_f64(mean, weight)
    } else {
        Centroid::<To>::new_mixed_f64(mean, weight)
    }
}

impl<F: FloatLike + FloatCore> TDigest<F> {
    /// Cast this digest to a different mean-storage precision.
    ///
    /// This preserves configuration, f64 centroid weights and digest-level stats
    /// while converting centroid means and extrema to the target float type.
    pub fn cast_precision<T>(&self) -> TDigest<T>
    where
        T: FloatLike + FloatCore,
    {
        let cents: Vec<Centroid<T>> = self
            .centroids()
            .iter()
            .map(|c| cast_centroid::<F, T>(c))
            .collect();
        let stats = DigestStats {
            data_sum: self.sum(),
            total_weight: self.count(),
            data_min: self.min(),
            data_max: self.max(),
        };
        let mut builder = TDigest::<T>::builder()
            .max_size(self.max_size())
            .scale(self.scale())
            .singleton_policy(self.singleton_policy());
        if let Some(delta) = self.delta() {
            builder = builder.delta(delta);
        }
        builder.with_centroids_and_stats(cents, stats).build()
    }

    /// Scale all centroid weights by `factor`.
    ///
    /// Effects:
    /// - centroid weights are multiplied by `factor`
    /// - digest `count` and `sum` are multiplied by `factor`
    /// - `min`/`max` are unchanged
    ///
    /// This preserves quantile/cdf/median shape while re-weighting the digest,
    /// subject to floating-point rounding. If a scaled centroid weight becomes
    /// zero/nonfinite or either mass total becomes nonfinite, return an error without
    /// changing the digest. This does not validate weighted-value sum overflow.
    pub fn scale_weights(&mut self, factor: f64) -> TdResult<&mut Self> {
        ensure_positive_finite_scale_factor(factor, "weight scale factor must be finite and > 0")?;
        if self.centroids.is_empty() {
            return Ok(self);
        }

        let mut scaled_centroids = Vec::with_capacity(self.centroids.len());
        let mut scaled_mass = 0.0;
        for centroid in &self.centroids {
            let weight = centroid.weight_f64() * factor;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(TdError::InvalidWeight {
                    context: "scaled centroid weight must be finite and > 0",
                });
            }
            scaled_mass += weight;
            if !scaled_mass.is_finite() {
                return Err(TdError::InvalidWeight {
                    context: "scaled total weight must be finite and > 0",
                });
            }
            // The product is validated before a checked centroid constructor.
            scaled_centroids.push(centroid.rescaled(1.0, factor));
        }
        let scaled_count = self.count * factor;
        if !scaled_count.is_finite() || scaled_count <= 0.0 {
            return Err(TdError::InvalidWeight {
                context: "scaled total weight must be finite and > 0",
            });
        }

        // Commit only after all products and both total representations pass.
        self.centroids = scaled_centroids;
        self.count = scaled_count;
        self.sum *= factor;
        Ok(self)
    }

    /// Scale centroid means by `factor` (value-axis scaling).
    ///
    /// Effects:
    /// - centroid means are multiplied by `factor`
    /// - digest `min`/`max` and `sum` are multiplied by `factor`
    /// - centroid weights and digest `count` are unchanged
    ///
    /// Note: this requires a strictly positive factor to preserve monotone ordering.
    pub fn scale_values(&mut self, factor: f64) -> TdResult<&mut Self> {
        ensure_positive_finite_scale_factor(factor, "value scale factor must be finite and > 0")?;
        if self.centroids.is_empty() || self.count == 0.0 {
            return Ok(self);
        }

        self.centroids = self
            .centroids
            .iter()
            .map(|c| c.rescaled(factor, 1.0))
            .collect();

        self.set_min(self.min() * factor);
        self.set_max(self.max() * factor);
        self.sum *= factor;
        Ok(self)
    }
}
