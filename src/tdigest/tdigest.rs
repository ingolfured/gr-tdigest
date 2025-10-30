use ordered_float::OrderedFloat;

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::compressor::compress_into;
use crate::tdigest::merges::{KWayCentroidMerge, MergeByMean};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;
// ⬇️ bring Precision into scope (re-exported from tdigest/mod.rs)
use crate::tdigest::Precision;

use serde::{Deserialize, Serialize};

/// TDigest orchestration + public API.
///
/// # Design: “single pipeline, many producers”
///
/// All ingestion paths (raw values, existing digests) are funneled into the **same**
/// compression pipeline in `compressor::compress_into`. The pipeline has a *fixed*,
/// easy-to-reason-about sequence of stages:
///
/// 1) **Normalize** — verify non-decreasing means, coalesce adjacent equal means into a
///    *singleton pile*, and accumulate running totals for metadata.
/// 2) **Slice (edges vs. interior)** — derive edge protection/quotas from [`SingletonPolicy`].
/// 3) **Merge (k-limit)** — greedily merge under the selected [`ScaleFamily`] rule (Δk ≤ 1).
/// 4) **Cap (equal-weight bucketization)** — if needed, compress interior to capacity by
///    order-preserving equal-weight grouping (may emit fewer than `N` for stability).
/// 5) **Assemble** — concatenate left edges + interior + right edges without mutation.
/// 6) **Post (policy finalization)** — policy-specific final pass (e.g., total cap for `Use`).
///
/// This structure keeps behavior deterministic and testable while making it obvious **where**
/// to change things (scale families only affect stage 3; tail policies only affect stages 2/6).
///
/// ## Public ingestion paths
/// - [`TDigest::merge_sorted`] / [`TDigest::merge_unsorted`]: raw values via a *merge-by-mean* producer.
/// - [`TDigest::merge_digests`]: multiple digests via a *k-way centroid merge*.
/// - All paths end in the **same** compression pipeline (see `compress_into`).
///
/// ### Invariants preserved
/// - Strictly increasing centroid means (no duplicates).
/// - Total weight conservation (up to expected float rounding).
/// - Clear ownership of semantics:
///   - “equal means ⇒ pile” belongs to **Normalize**.
///   - edge/tail protection belongs to **Slice**/**Post**.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    sum: OrderedFloat<f64>,
    count: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    precision: Precision,
}

impl Default for TDigest {
    fn default() -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size: 1000, // default compression parameter
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: ScaleFamily::K2,       // default scale
            policy: SingletonPolicy::Use, // default singleton policy
            precision: Precision::F64,    // default in-memory precision
        }
    }
}

// Accept both f32 and f64
pub trait IntoF64Vec {
    fn into_f64_vec(self) -> Vec<f64>;
}
impl IntoF64Vec for &[f64] {
    fn into_f64_vec(self) -> Vec<f64> {
        self.to_vec()
    }
}
impl IntoF64Vec for Vec<f64> {
    fn into_f64_vec(self) -> Vec<f64> {
        self
    }
}
impl IntoF64Vec for &[f32] {
    fn into_f64_vec(self) -> Vec<f64> {
        self.iter().map(|&x| x as f64).collect()
    }
}
impl IntoF64Vec for Vec<f32> {
    fn into_f64_vec(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DigestOptions {
    pub max_size: usize,
    pub scale: ScaleFamily,
    pub f32_mode: bool,
    pub singleton_policy: SingletonPolicy,
}
impl Default for DigestOptions {
    fn default() -> Self {
        Self {
            max_size: 1000,
            scale: ScaleFamily::K2,
            f32_mode: false,
            singleton_policy: SingletonPolicy::Use,
        }
    }
}

impl TDigest {
    #[inline]
    pub fn precision(&self) -> Precision {
        self.precision
    }
    #[inline]
    pub fn is_f32_mode(&self) -> bool {
        matches!(self.precision, Precision::F32)
    }

    pub fn from_array<A: IntoF64Vec>(arr: A) -> TDigest {
        Self::from_array_with(arr, DigestOptions::default())
    }
    pub fn from_array_with<A: IntoF64Vec>(arr: A, opts: DigestOptions) -> TDigest {
        let base = TDigest::builder()
            .max_size(opts.max_size)
            .scale(opts.scale)
            .singleton_policy(opts.singleton_policy)
            // NOTE: if/when you wire true f32 storage, map opts.f32_mode here
            .precision(if opts.f32_mode {
                Precision::F32
            } else {
                Precision::F64
            })
            .build();
        base.merge_unsorted(arr.into_f64_vec())
    }
}

/// Builder for [`TDigest`].
///
/// Use the builder when you want to:
/// - construct an empty digest with chosen parameters, or
/// - seed a digest with *existing centroids* and *data-level stats*.
///
/// Seeding helpers:
/// - [`TDigestBuilder::with_centroids_and_stats`] — pass centroids + [`DigestStats`].
/// - [`TDigestBuilder::with_centroids`] — convenience to pass raw fields with optional
///   `max_size_override` (mirrors how [`TDigest::new`] may temporarily exceed capacity and
///   then compress down).
#[derive(Debug, Clone)]
pub struct TDigestBuilder {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    // optional seeds
    init_centroids: Option<Vec<Centroid>>,
    init_stats: Option<DigestStats>,
    override_max_size: Option<usize>,
    precision: Precision,
}

impl Default for TDigestBuilder {
    fn default() -> Self {
        TDigestBuilder {
            max_size: 1000,
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
            init_centroids: None,
            init_stats: None,
            override_max_size: None,
            precision: Precision::F64,
        }
    }
}

impl TDigestBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression parameter (`max_size`).
    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }

    /// Choose the scale family used by the k-limit (Stage **3**).
    pub fn scale(mut self, s: ScaleFamily) -> Self {
        self.scale = s;
        self
    }

    /// Set the singleton/edge policy influencing Stages **2** and **6**.
    pub fn singleton_policy(mut self, p: SingletonPolicy) -> Self {
        self.policy = p;
        self
    }

    /// Seed with centroids + data-level stats.
    pub fn with_centroids_and_stats(
        mut self,
        centroids: Vec<Centroid>,
        stats: DigestStats,
    ) -> Self {
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self
    }

    /// Seed with centroids and raw stats (convenience mirror of [`TDigest::new`]).
    pub fn with_centroids(
        mut self,
        centroids: Vec<Centroid>,
        sum: f64,
        count: f64,
        max: f64,
        min: f64,
        max_size_override: usize,
    ) -> Self {
        let stats = DigestStats {
            data_sum: sum,
            total_weight: count,
            data_min: min,
            data_max: max,
        };
        self.init_centroids = Some(centroids);
        self.init_stats = Some(stats);
        self.override_max_size = Some(max_size_override);
        self
    }

    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    /// Build the digest, seeding if seeds were provided.
    pub fn build(self) -> TDigest {
        // If seeded, construct directly from the provided centroids and stats.
        if let (Some(cents), Some(st)) = (self.init_centroids, self.init_stats) {
            let max_size = self.override_max_size.unwrap_or(self.max_size);
            TDigest {
                centroids: cents,
                max_size,
                sum: OrderedFloat(st.data_sum),
                count: OrderedFloat(st.total_weight),
                min: OrderedFloat(st.data_min),
                max: OrderedFloat(st.data_max),
                scale: self.scale,
                policy: self.policy,
                precision: self.precision,
            }
        } else {
            TDigest {
                centroids: Vec::new(),
                max_size: self.max_size,
                sum: 0.0.into(),
                count: 0.0.into(),
                max: f64::NAN.into(),
                min: f64::NAN.into(),
                scale: self.scale,
                policy: self.policy,
                precision: self.precision,
            }
        }
    }
}

/// Data-level stats for seeding a digest (∑x/∑w/min/max of the **raw data**, not centroids).
#[derive(Debug, Clone, Copy)]
pub struct DigestStats {
    /// Sum of raw data values (∑x), not sum of centroid means.
    pub data_sum: f64,
    /// Total weight (∑w). For raw samples this equals the sample count.
    pub total_weight: f64,
    /// Minimum observed raw value.
    pub data_min: f64,
    /// Maximum observed raw value.
    pub data_max: f64,
}

impl TDigest {
    /// Entry point for fluent construction.
    pub fn builder() -> TDigestBuilder {
        TDigestBuilder::default()
    }

    // --- internal metadata setters (small & inlined) ---
    #[inline]
    pub(crate) fn set_sum(&mut self, s: f64) {
        self.sum = OrderedFloat(s);
    }
    #[inline]
    pub(crate) fn set_count(&mut self, c: f64) {
        self.count = OrderedFloat(c);
    }
    #[inline]
    pub(crate) fn set_min(&mut self, v: f64) {
        self.min = OrderedFloat(v);
    }
    #[inline]
    pub(crate) fn set_max(&mut self, v: f64) {
        self.max = OrderedFloat(v);
    }

    /// Build from **unsorted** values (convenience over [`TDigest::merge_unsorted`]).
    pub fn from_unsorted(values: &[f64], max_size: usize) -> TDigest {
        let base = TDigest::builder().max_size(max_size).build();
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
    pub fn merge_unsorted(&self, mut unsorted_values: Vec<f64>) -> TDigest {
        unsorted_values.sort_by(|a, b| a.total_cmp(b));
        self.merge_sorted(unsorted_values)
    }

    /// Ingest **sorted** values by interleaving with existing centroids and running the pipeline.
    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }
        let mut result = self.new_result_for_values(&sorted_values);

        // Producer: ordered stream of centroids + value runs (no coalescing here).
        let stream = MergeByMean::from_centroids_and_values(
            &self.centroids,
            &sorted_values,
            // mark value runs as singletons unless policy says Off
            !matches!(self.policy, SingletonPolicy::Off),
        );

        // Pipeline: Normalize → Slice → Merge(k-limit) → Cap → Assemble → Post
        let compressed = compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;

        // Strong invariant: strictly increasing means (no duplicates).
        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge"
        );
        result
    }

    /// Merge multiple digests by k-way merging their centroid runs and sending the result through
    /// the same pipeline used for raw values.
    pub fn merge_digests(digests: Vec<TDigest>) -> TDigest {
        // Decide defaults by first non-empty digest to keep semantics stable.
        let mut chosen = TDigest::default();
        let mut runs: Vec<&[Centroid]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if runs.is_empty() {
                    chosen.max_size = d.max_size;
                    chosen.scale = d.scale;
                    chosen.policy = d.policy;
                    chosen.precision = d.precision;
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

        let mut result = TDigest {
            centroids: Vec::new(),
            max_size: chosen.max_size,
            sum: 0.0.into(),
            count: total_count.into(),
            max,
            min,
            scale: chosen.scale,
            policy: chosen.policy,
            precision: chosen.precision,
        };

        // Producer: k-way merge of centroid runs (no extra coalescing beyond equal-mean heads).
        let merged_stream = KWayCentroidMerge::new(runs);

        // Same pipeline as values.
        let compressed = compress_into(&mut result, chosen.max_size, merged_stream);
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
        centroids: Vec<Centroid>,
        sum: f64,
        count: f64,
        max: f64,
        min: f64,
        max_size: usize,
    ) -> Self {
        if centroids.len() <= max_size {
            TDigest {
                centroids,
                max_size,
                sum: OrderedFloat::from(sum),
                count: OrderedFloat::from(count),
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
                scale: ScaleFamily::K2,
                policy: SingletonPolicy::Use,
                precision: Precision::F64,
            }
        } else {
            let sz = centroids.len();
            let digests = vec![
                TDigest::builder().max_size(100).build(),
                TDigest::builder()
                    .with_centroids(centroids, sum, count, max, min, sz)
                    .build(),
            ];
            Self::merge_digests(digests)
        }
    }

    /// Mean of the represented distribution (`∑x / ∑w`).
    #[inline]
    pub fn mean(&self) -> f64 {
        let n = self.count.into_inner();
        if n > 0.0 {
            self.sum.into_inner() / n
        } else {
            0.0
        }
    }

    /// Sum of raw values (∑x).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.into_inner()
    }

    /// Total weight (∑w). For raw samples, this equals the sample count.
    #[inline]
    pub fn count(&self) -> f64 {
        self.count.into_inner()
    }

    /// Maximum observed raw value.
    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner()
    }

    /// Minimum observed raw value.
    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner()
    }

    /// Whether the digest currently has no centroids.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// The configured compression parameter.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Borrow the internal centroids.
    #[inline]
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    /// Prepare a result shell when ingesting a batch of **sorted** raw values.
    ///
    /// Sets `count` to existing `∑w + values.len()`, resets `sum` to be filled by Stage **1**,
    /// and updates min/max by comparing current bounds with the new batch.
    fn new_result_for_values(&self, values: &[f64]) -> TDigest {
        let mut r = TDigest {
            centroids: Vec::new(),
            max_size: self.max_size(),
            sum: 0.0.into(),
            count: (self.count() + values.len() as f64).into(),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: self.scale,
            policy: self.policy,
            precision: self.precision,
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
}
