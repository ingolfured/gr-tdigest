use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A centroid summarizes a cluster in the digest.
///
/// `singleton` is a **data flag**:
/// - `true` for an atomic ECDF jump (a raw singleton with `weight==1`, or a pile of
///   identical values with `weight>=2`),
/// - `false` for a mixed cluster (a blend of multiple distinct means).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Centroid {
    mean: OrderedFloat<f64>,
    weight: OrderedFloat<f64>,
    singleton: bool,
}

impl PartialOrd for Centroid {
    fn partial_cmp(&self, other: &Centroid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Centroid {
    fn cmp(&self, other: &Centroid) -> Ordering {
        // We never output duplicate means; ordering by mean alone is fine.
        self.mean.cmp(&other.mean)
    }
}

impl Default for Centroid {
    fn default() -> Self {
        Centroid {
            mean: OrderedFloat::from(0.0),
            weight: OrderedFloat::from(1.0),
            singleton: true,
        }
    }
}

impl Centroid {
    /// Legacy constructor kept for backward compatibility with older call sites.
    ///
    /// Defaults to a **mixed** centroid (not a data-true singleton).
    #[inline]
    pub fn new(mean: f64, weight: f64) -> Self {
        Self::new_mixed(mean, weight)
    }

    /// Explicit constructor for a **data-true singleton** (raw or same-mean pile).
    #[inline]
    pub fn new_singleton(mean: f64, weight: f64) -> Self {
        debug_assert!(weight >= 1.0);
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
            singleton: true,
        }
    }

    /// Explicit constructor for a **mixed** centroid (not a singleton).
    #[inline]
    pub fn new_mixed(mean: f64, weight: f64) -> Self {
        debug_assert!(weight > 0.0);
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
            singleton: false,
        }
    }

    /// Generic constructor setting `singleton` explicitly.
    #[inline]
    pub fn from_parts(mean: f64, weight: f64, singleton: bool) -> Self {
        if singleton {
            Self::new_singleton(mean, weight)
        } else {
            Self::new_mixed(mean, weight)
        }
    }

    /// Centroid mean.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean.into_inner()
    }
    /// Centroid weight.
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight.into_inner()
    }
    /// Whether this centroid is a data-true singleton/pile.
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.singleton
    }

    /// Mutably mark/unmark as data-true singleton.
    ///
    /// Note: the compressor decides whether a cluster *remains* a pile or becomes mixed.
    #[inline]
    pub fn mark_singleton(&mut self, v: bool) {
        self.singleton = v;
    }

    /// Fold a batch `(sum, weight)` into this centroid and return the contributed sum.
    ///
    /// Does **not** mutate `singleton`. The compressor (Stage **3**) decides whether the result
    /// is still a pile or mixed.
    #[inline]
    pub fn add(&mut self, sum: f64, weight: f64) -> f64 {
        let w0 = self.weight.into_inner();
        let m0 = self.mean.into_inner();
        let new_sum = sum + w0 * m0;
        let new_w = w0 + weight;
        self.weight = OrderedFloat::from(new_w);
        self.mean = OrderedFloat::from(new_sum / new_w);
        new_sum
    }
}

/* ===========================
 * Small helpers
 * =========================== */

/// Strictly increasing by mean (no duplicates). Used to assert pipeline invariants.
#[inline]
pub fn is_sorted_strict_by_mean(cs: &[Centroid]) -> bool {
    cs.windows(2).all(|w| w[0].mean() < w[1].mean())
}

/// Non-strictly increasing by mean (duplicates allowed).
#[inline]
pub fn is_sorted_by_mean(cs: &[Centroid]) -> bool {
    cs.windows(2).all(|w| w[0] <= w[1])
}

/// Merge *adjacent* centroids that have the exact same mean into a single centroid.
///
/// Keeps/sets `singleton=true` and sums weights (because equal-mean runs are piles).
/// This is used by Stage **1**.
pub fn coalesce_adjacent_equal_means(values: Vec<Centroid>) -> Vec<Centroid> {
    if values.len() <= 1 {
        return values;
    }
    let mut out: Vec<Centroid> = Vec::with_capacity(values.len());
    let mut acc = values[0];
    for c in values.into_iter().skip(1) {
        if c.mean() == acc.mean() {
            let w = acc.weight() + c.weight();
            acc = Centroid::new_singleton(acc.mean(), w);
        } else {
            out.push(acc);
            acc = c;
        }
    }
    out.push(acc);
    out
}

/// Same as [`coalesce_adjacent_equal_means`] but never marks the merged result as a singleton.
/// Useful for producer variants that want to force mixed semantics.
pub fn coalesce_adjacent_equal_means_no_singleton(values: Vec<Centroid>) -> Vec<Centroid> {
    if values.len() <= 1 {
        return values;
    }
    let mut out: Vec<Centroid> = Vec::with_capacity(values.len());
    let mut acc = values[0];
    for c in values.into_iter().skip(1) {
        if c.mean() == acc.mean() {
            let w = acc.weight() + c.weight();
            acc = Centroid::new_mixed(acc.mean(), w);
        } else {
            out.push(acc);
            acc = c;
        }
    }
    out.push(acc);
    out
}
