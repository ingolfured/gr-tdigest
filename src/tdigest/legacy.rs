//! Legacy Python input and merge policy over the shared t-digest compressor.
//!
//! Non-finite endpoint masses are kept outside the strict finite core. NaNs are
//! discarded only here; modern frontends retain their strict training contract.

use super::frontends::{
    validate_quantile_probe, validate_trimmed_mean_bounds, DigestConfig, DigestPrecision,
    FrontendDigest, FrontendError,
};
use super::precision::FloatLike;
use super::singleton_policy::SingletonPolicy;
use super::{DigestStats, ScaleFamily, TDigest};
use ordered_float::FloatCore;

#[derive(Debug, Clone)]
pub struct LegacyDigest {
    pub(crate) finite: FrontendDigest,
    pub(crate) negative_infinity: f64,
    pub(crate) positive_infinity: f64,
}

fn config(delta: f64) -> DigestConfig {
    DigestConfig {
        max_size: 1000,
        scale: ScaleFamily::K2Norm,
        policy: SingletonPolicy::Off,
        delta: Some(delta),
    }
}

fn reconfigure<F: FloatLike + FloatCore>(td: &TDigest<F>, delta: f64) -> TDigest<F> {
    TDigest::builder()
        .max_size(1000)
        .scale(ScaleFamily::K2Norm)
        .singleton_policy(SingletonPolicy::Off)
        .delta(delta)
        .with_centroids_and_stats(
            td.centroids().to_vec(),
            DigestStats {
                data_sum: td.sum(),
                total_weight: td.count(),
                data_min: td.min(),
                data_max: td.max(),
            },
        )
        .build()
}

impl LegacyDigest {
    pub fn from_weighted(
        values: Vec<f64>,
        weights: Vec<f64>,
        delta: f64,
        precision: DigestPrecision,
    ) -> Result<Self, FrontendError> {
        let mut finite = FrontendDigest::from_values(Vec::new(), config(delta), precision)?;
        if values.len() != weights.len() {
            return Err(FrontendError::InvalidTrainingData(
                "means and weights must have the same length".into(),
            ));
        }
        let mut finite_values = Vec::with_capacity(values.len());
        let mut finite_weights = Vec::with_capacity(values.len());
        let mut negative_infinity = 0.0;
        let mut positive_infinity = 0.0;
        let mut total = 0.0;
        for (value, weight) in values.into_iter().zip(weights) {
            if !weight.is_finite() || weight <= 0.0 || weight.fract() != 0.0 {
                return Err(FrontendError::InvalidTrainingData(
                    "legacy weights must be positive integers".into(),
                ));
            }
            if value.is_nan() {
                continue;
            }
            total += weight;
            if total > u32::MAX as f64 {
                return Err(FrontendError::InvalidTrainingData(
                    "legacy total weight exceeds uint32 capacity".into(),
                ));
            }
            if value == f64::NEG_INFINITY {
                negative_infinity += weight;
            } else if value == f64::INFINITY {
                positive_infinity += weight;
            } else {
                finite_values.push(value);
                finite_weights.push(weight);
            }
        }
        finite.add_weighted_f64(finite_values, finite_weights)?;
        Ok(Self {
            finite,
            negative_infinity,
            positive_infinity,
        })
    }

    pub fn delta(&self) -> f64 {
        self.finite
            .config()
            .delta
            .expect("legacy delta is always set")
    }

    pub fn count(&self) -> f64 {
        self.finite_count() + self.negative_infinity + self.positive_infinity
    }

    fn finite_count(&self) -> f64 {
        match &self.finite {
            FrontendDigest::F32(td) => td.count(),
            FrontendDigest::F64(td) => td.count(),
        }
    }

    pub fn merge(&self, other: &Self, delta: f64) -> Result<Self, FrontendError> {
        let empty =
            FrontendDigest::from_values(Vec::new(), config(delta), self.finite.precision())?;
        if self.finite.precision() != other.finite.precision() {
            return Err(FrontendError::IncompatibleMerge(
                "self has a different type to other".into(),
            ));
        }
        if self.count() + other.count() > u32::MAX as f64 {
            return Err(FrontendError::InvalidTrainingData(
                "legacy total weight exceeds uint32 capacity".into(),
            ));
        }
        let finite = if self.finite_count() + other.finite_count() == 0.0 {
            empty
        } else {
            match (&self.finite, &other.finite) {
                (FrontendDigest::F32(a), FrontendDigest::F32(b)) => {
                    FrontendDigest::F32(TDigest::merge_digests(vec![
                        reconfigure(a, delta),
                        reconfigure(b, delta),
                    ]))
                }
                (FrontendDigest::F64(a), FrontendDigest::F64(b)) => {
                    FrontendDigest::F64(TDigest::merge_digests(vec![
                        reconfigure(a, delta),
                        reconfigure(b, delta),
                    ]))
                }
                _ => unreachable!("precision checked above"),
            }
        };
        Ok(Self {
            finite,
            negative_infinity: self.negative_infinity + other.negative_infinity,
            positive_infinity: self.positive_infinity + other.positive_infinity,
        })
    }

    pub fn means(&self) -> Vec<f64> {
        let mut values = self.finite.centroid_means_f64();
        if self.negative_infinity > 0.0 {
            values.insert(0, f64::NEG_INFINITY);
        }
        if self.positive_infinity > 0.0 {
            values.push(f64::INFINITY);
        }
        values
    }

    pub fn weights(&self) -> Vec<f64> {
        let mut weights = self.finite.centroid_weights_f64();
        if self.negative_infinity > 0.0 {
            weights.insert(0, self.negative_infinity);
        }
        if self.positive_infinity > 0.0 {
            weights.push(self.positive_infinity);
        }
        weights
    }

    pub fn len(&self) -> usize {
        self.finite.centroid_len()
            + usize::from(self.negative_infinity > 0.0)
            + usize::from(self.positive_infinity > 0.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn quantile(&self, q: f64) -> Result<f64, FrontendError> {
        validate_quantile_probe(q).map_err(|e| FrontendError::InvalidProbe(e.into()))?;
        let rank = q * self.count();
        if self.negative_infinity > 0.0 && rank < self.negative_infinity {
            return Ok(f64::NEG_INFINITY);
        }
        if self.positive_infinity > 0.0 && rank >= self.count() - self.positive_infinity {
            return Ok(f64::INFINITY);
        }
        let finite_count = self.finite_count();
        if finite_count == 0.0 {
            return Ok(f64::NAN);
        }
        self.finite
            .quantile_strict(((rank - self.negative_infinity) / finite_count).clamp(0.0, 1.0))
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if self.count() == 0.0 || x.is_nan() {
            return f64::NAN;
        }
        if x == f64::NEG_INFINITY {
            return 0.5 * self.negative_infinity / self.count();
        }
        if x == f64::INFINITY {
            return 1.0 - 0.5 * self.positive_infinity / self.count();
        }
        let finite_mass = if self.finite_count() == 0.0 {
            0.0
        } else {
            self.finite.cdf(&[x])[0] * self.finite_count()
        };
        (self.negative_infinity + finite_mass) / self.count()
    }

    pub fn trimmed_mean(&self, lower: f64, upper: f64) -> Result<f64, FrontendError> {
        validate_trimmed_mean_bounds(lower, upper)
            .map_err(|e| FrontendError::InvalidProbe(e.into()))?;
        if self.count() == 0.0 || lower == upper {
            return Ok(f64::NAN);
        }
        let lo = lower * self.count();
        let hi = upper * self.count();
        let mut start = 0.0;
        let mut sum = 0.0;
        for (mean, weight) in self.means().into_iter().zip(self.weights()) {
            let end = start + weight;
            let take = hi.min(end) - lo.max(start);
            if take > 0.0 {
                sum += mean * take;
            }
            start = end;
        }
        Ok(sum / (hi - lo))
    }
}
