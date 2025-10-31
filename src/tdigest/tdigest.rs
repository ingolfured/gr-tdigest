// src/tdigest/tdigest.rs
use crate::{TdError, TdResult};
use ordered_float::{FloatCore, OrderedFloat};

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::compressor::compress_into;
use crate::tdigest::merges::{KWayCentroidMerge, MergeByMean};
use crate::tdigest::precision::{FloatLike, Precision};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;

/// TDigest orchestration + public API, generic over centroid float storage `F` (`f32` or `f64`).
///
/// - Centroids (means/weights) and `min/max` are stored in `F` to control **in-memory size**.
/// - `sum` and `count` remain `f64` for **stable accumulation** and to match the on-wire schema.
/// - The compressor/merges do their internal math in `f64` and convert at the boundaries.
#[derive(Debug, PartialEq, Clone)]
pub struct TDigest<F: FloatLike + FloatCore> {
    centroids: Vec<Centroid<F>>,
    max_size: usize,
    sum: f64,   // ∑x over raw data (f64 for stability / wire)
    count: f64, // ∑w over raw data (f64 for stability / wire)
    max: OrderedFloat<F>,
    min: OrderedFloat<F>,
    scale: ScaleFamily,
    policy: SingletonPolicy,
}

pub type TDigestF64 = TDigest<f64>;
pub type TDigestF32 = TDigest<f32>;

impl<F: FloatLike + FloatCore> Default for TDigest<F> {
    fn default() -> Self {
        Self {
            centroids: Vec::new(),
            max_size: 1000,
            sum: 0.0,
            count: 0.0,
            max: OrderedFloat::from(F::from_f64(f64::NAN)),
            min: OrderedFloat::from(F::from_f64(f64::NAN)),
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
        }
    }
}

/* =============================================================================
 * Input adapters
 * ============================================================================= */

/// Accepts containers already typed as `F` and yields a `Vec<F>`.
pub trait IntoVecF<F: FloatLike> {
    fn into_vec_f(self) -> Vec<F>;
}
impl<F: FloatLike> IntoVecF<F> for Vec<F> {
    #[inline]
    fn into_vec_f(self) -> Vec<F> {
        self
    }
}
impl<F: FloatLike> IntoVecF<F> for &[F] {
    #[inline]
    fn into_vec_f(self) -> Vec<F> {
        self.to_vec()
    }
}

#[inline]
fn ensure_no_nan_values<F: FloatLike + FloatCore>(values: &[F]) -> TdResult<()> {
    if values.iter().any(|v| v.is_nan()) {
        return Err(TdError::NaNInput {
            context: "sample value",
        });
    }
    Ok(())
}

/* =============================================================================
 * Options / Builder
 * ============================================================================= */

#[derive(Debug, Clone, Copy)]
pub struct DigestOptions<F: FloatLike> {
    pub max_size: usize,
    pub scale: ScaleFamily,
    pub singleton_policy: SingletonPolicy,
    _marker: std::marker::PhantomData<F>,
}
impl<F: FloatLike> Default for DigestOptions<F> {
    fn default() -> Self {
        Self {
            max_size: 1000,
            scale: ScaleFamily::K2,
            singleton_policy: SingletonPolicy::Use,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Builder for [`TDigest<F>`].
///
/// Use the builder when you want to:
/// - construct an empty digest with chosen parameters, or
/// - seed a digest with *existing centroids* and *data-level stats*.
#[derive(Debug, Clone)]
pub struct TDigestBuilder<F: FloatLike + FloatCore> {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    // optional seeds
    init_centroids: Option<Vec<Centroid<F>>>,
    init_stats: Option<DigestStats>,
    override_max_size: Option<usize>,
}
impl<F: FloatLike + FloatCore> Default for TDigestBuilder<F> {
    fn default() -> Self {
        Self {
            max_size: 1000,
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
            init_centroids: None,
            init_stats: None,
            override_max_size: None,
        }
    }
}
impl<F: FloatLike + FloatCore> TDigestBuilder<F> {
    /// Create a new builder with defaults.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression parameter (`max_size`).
    #[inline]
    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }

    /// Choose the scale family used by the k-limit (Stage **3**).
    #[inline]
    pub fn scale(mut self, s: ScaleFamily) -> Self {
        self.scale = s;
        self
    }

    /// Set the singleton/edge policy influencing Stages **2** and **6**.
    #[inline]
    pub fn singleton_policy(mut self, p: SingletonPolicy) -> Self {
        self.policy = p;
        self
    }

    /// Seed with centroids + data-level stats.
    pub fn with_centroids_and_stats(
        mut self,
        centroids: Vec<Centroid<F>>,
        stats: DigestStats,
    ) -> Self {
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self
    }

    /// Seed with centroids and raw stats (convenience mirror of [`TDigest::new`]).
    pub fn with_centroids(
        mut self,
        centroids: Vec<Centroid<F>>,
        sum: f64,
        count: f64,
        max: F,
        min: F,
        max_size_override: usize,
    ) -> Self {
        let stats = DigestStats {
            data_sum: sum,
            total_weight: count,
            data_min: min.to_f64(),
            data_max: max.to_f64(),
        };
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self.override_max_size = Some(max_size_override);
        self
    }

    /// Build the digest, seeding if seeds were provided.
    pub fn build(self) -> TDigest<F> {
        // If seeded, construct directly from the provided centroids and stats.
        if let (Some(cents), Some(st)) = (self.init_centroids, self.init_stats) {
            let max_size = self.override_max_size.unwrap_or(self.max_size);
            TDigest {
                centroids: cents,
                max_size,
                sum: st.data_sum,
                count: st.total_weight,
                min: OrderedFloat::from(F::from_f64(st.data_min)),
                max: OrderedFloat::from(F::from_f64(st.data_max)),
                scale: self.scale,
                policy: self.policy,
            }
        } else {
            TDigest {
                centroids: Vec::new(),
                max_size: self.max_size,
                sum: 0.0,
                count: 0.0,
                max: OrderedFloat::from(F::from_f64(f64::NAN)),
                min: OrderedFloat::from(F::from_f64(f64::NAN)),
                scale: self.scale,
                policy: self.policy,
            }
        }
    }
}

/* =============================================================================
 * Digest
 * ============================================================================= */

/// Data-level stats for seeding a digest (∑x/∑w/min/max of the **raw data**, not centroids).
#[derive(Debug, Clone, Copy)]
pub struct DigestStats {
    /// Sum of raw data values (∑x), not sum of centroid means.
    pub data_sum: f64,
    /// Total weight (∑w). For raw samples this equals the sample count.
    pub total_weight: f64,
    /// Minimum observed raw value (in f64; converted to F in builder).
    pub data_min: f64,
    /// Maximum observed raw value (in f64; converted to F in builder).
    pub data_max: f64,
}

impl<F: FloatLike + FloatCore> TDigest<F> {
    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner().to_f64()
    }
    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner().to_f64()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
    #[inline]
    pub fn centroids(&self) -> &[Centroid<F>] {
        &self.centroids
    }
    /// Entry point for fluent construction.
    #[inline]
    pub fn builder() -> TDigestBuilder<F> {
        TDigestBuilder::default()
    }

    /// Report the precision implied by the type parameter.
    #[inline]
    pub fn precision(&self) -> Precision {
        Precision::of_type::<F>()
    }

    // --- internal metadata setters (small & inlined) ---
    #[inline]
    pub(crate) fn set_sum(&mut self, s: f64) {
        self.sum = s;
    }
    #[inline]
    pub(crate) fn set_count(&mut self, c: f64) {
        self.count = c;
    }
    #[inline]
    pub(crate) fn set_min(&mut self, v: f64) {
        self.min = OrderedFloat::from(F::from_f64(v));
    }
    #[inline]
    pub(crate) fn set_max(&mut self, v: f64) {
        self.max = OrderedFloat::from(F::from_f64(v));
    }

    /// Build from **unsorted** values (convenience over [`TDigest::merge_unsorted`]).
    pub fn from_unsorted(values: &[F], max_size: usize) -> TdResult<TDigest<F>> {
        ensure_no_nan_values(values)?;
        let base = Self::builder().max_size(max_size).build();
        base.merge_unsorted(values.to_vec())
    }

    /// The configured scale family used by the compressor’s k-limit.
    #[inline]
    pub fn scale(&self) -> ScaleFamily {
        self.scale
    }

    /// The configured singleton/edge policy.
    #[inline]
    pub fn singleton_policy(&self) -> SingletonPolicy {
        self.policy
    }

    /// Ingest **unsorted** values; behavior matches [`TDigest::merge_sorted`] after sorting.
    pub fn merge_unsorted(&self, mut unsorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        ensure_no_nan_values(&unsorted_values)?;
        unsorted_values.sort_by(|a, b| a.total_cmp(*b));
        self.merge_sorted(unsorted_values)
    }

    /// Ingest **sorted** values by interleaving with existing centroids and running the pipeline.
    pub fn merge_sorted(&self, sorted_values: Vec<F>) -> TdResult<TDigest<F>> {
        ensure_no_nan_values(&sorted_values)?;
        if sorted_values.is_empty() {
            return Ok(self.clone());
        }
        let mut result = self.new_result_for_values(&sorted_values);

        let stream = MergeByMean::from_centroids_and_values(&self.centroids, &sorted_values);

        // Pipeline: Normalize → Slice → Merge(k-limit) → Cap → Assemble → Post
        let compressed: Vec<Centroid<F>> = compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge"
        );
        Ok(result)
    }

    /// Merge multiple digests by k-way merging their centroid runs and sending the result through
    /// the same pipeline used for raw values.
    pub fn merge_digests(digests: Vec<TDigest<F>>) -> TDigest<F> {
        // Decide defaults by first non-empty digest to keep semantics stable.
        let mut chosen: Option<(usize, ScaleFamily, SingletonPolicy)> = None;
        let mut runs: Vec<&[Centroid<F>]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0_f64;
        let mut min = OrderedFloat::from(F::from_f64(f64::INFINITY));
        let mut max = OrderedFloat::from(F::from_f64(f64::NEG_INFINITY));

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if chosen.is_none() {
                    chosen = Some((d.max_size, d.scale, d.policy));
                }
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }
        if total_count == 0.0 {
            return TDigest::default();
        }
        let (chosen_max_size, chosen_scale, chosen_policy) = chosen.unwrap();

        let mut result = TDigest::<F> {
            centroids: Vec::new(),
            max_size: chosen_max_size,
            sum: 0.0,
            count: total_count,
            max,
            min,
            scale: chosen_scale,
            policy: chosen_policy,
        };

        // Producer: k-way merge of centroid runs (no extra coalescing beyond equal-mean heads).
        let merged_stream = KWayCentroidMerge::from_runs(&runs);

        // Same pipeline as values.
        let compressed: Vec<Centroid<F>> =
            compress_into(&mut result, chosen_max_size, merged_stream);
        result.centroids = compressed;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge_digests"
        );
        result
    }

    /* ===========================
     * Small utilities
     * =========================== */

    /// Construct a digest from raw centroids + stats.
    ///
    /// If `centroids.len() > max_size`, this creates a temporary large digest (via the builder)
    /// and then merges down through the pipeline to respect capacity. This mirrors the case where
    /// a producer generates more than `max_size` centroids and requires compression.
    pub fn new(
        centroids: Vec<Centroid<F>>,
        sum: f64,
        count: f64,
        max: F,
        min: F,
        max_size: usize,
    ) -> Self {
        if centroids.len() <= max_size {
            TDigest {
                centroids,
                max_size,
                sum,
                count,
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
                scale: ScaleFamily::K2,
                policy: SingletonPolicy::Use,
            }
        } else {
            let sz = centroids.len();
            let digests = vec![
                TDigest::<F>::builder().max_size(100).build(),
                TDigest::<F>::builder()
                    .with_centroids(centroids, sum, count, max, min, sz)
                    .build(),
            ];
            Self::merge_digests(digests)
        }
    }

    /// Mean of the represented distribution (`∑x / ∑w`) in **f64** for stability.
    #[inline]
    pub fn mean(&self) -> f64 {
        let n = self.count;
        if n > 0.0 {
            self.sum / n
        } else {
            0.0
        }
    }

    /// Sum of raw values (∑x).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Total weight (∑w). For raw samples, this equals the sample count.
    #[inline]
    pub fn count(&self) -> f64 {
        self.count
    }

    /// The configured compression parameter.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Borrow the internal centroids.
    #[inline]
    pub fn centroids_ref(&self) -> &[Centroid<F>] {
        &self.centroids
    }

    /// Prepare a result shell when ingesting a batch of **sorted** raw values.
    ///
    /// Sets `count` to existing `∑w + values.len()`, resets `sum` to be filled by Stage **1**,
    /// and updates min/max by comparing current bounds with the new batch.
    fn new_result_for_values(&self, values: &[F]) -> TDigest<F> {
        let mut r = TDigest::<F> {
            centroids: Vec::new(),
            max_size: self.max_size(),
            sum: 0.0,
            count: self.count() + (values.len() as f64),
            max: OrderedFloat::from(F::from_f64(f64::NAN)),
            min: OrderedFloat::from(F::from_f64(f64::NAN)),
            scale: self.scale,
            policy: self.policy,
        };

        let vmin = OrderedFloat::from(values[0]);
        let vmax = OrderedFloat::from(values[values.len() - 1]);
        if self.count() > 0.0 {
            r.min = std::cmp::min(self.min, vmin);
            r.max = std::cmp::max(self.max, vmax);
        } else {
            r.min = vmin;
            r.max = vmax;
        }
        r
    }

    /* ===========================
     * Convenience constructors
     * =========================== */

    /// Convenience: build from an arbitrary numeric container already typed as `F`, using options.
    pub fn from_array_with<A: IntoVecF<F>>(arr: A, opts: DigestOptions<F>) -> TdResult<TDigest<F>> {
        let vals = arr.into_vec_f();
        ensure_no_nan_values(&vals)?;
        let base = TDigest::<F>::builder()
            .max_size(opts.max_size)
            .scale(opts.scale)
            .singleton_policy(opts.singleton_policy)
            .build();
        base.merge_unsorted(vals)
    }

    /// Convenience: defaults (max_size=1000, scale=K2, singleton=Use).
    pub fn from_array<A: IntoVecF<F>>(arr: A) -> TdResult<TDigest<F>> {
        Self::from_array_with(arr, DigestOptions::<F>::default())
    }
}
