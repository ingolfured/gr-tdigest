// src/tdigest/mod.rs
pub mod cdf;
pub mod codecs;
pub mod quantile;
pub mod test_helpers;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Peekable;

#[allow(unused_macros)]
macro_rules! ttrace {
    ($($arg:tt)*) => {
        if std::env::var("TDIGEST_TRACE").is_ok() {
            eprintln!($($arg)*);
        }
    }
}

/* ===========================
 * Core types
 * =========================== */

/// A centroid summarizes a cluster in the digest.
///
/// `singleton` is a *data* flag:
/// - `true` for an atomic ECDF jump (a raw singleton with `weight==1`, or a pile of
///   identical values with `weight>=2`),
/// - `false` for a mixed cluster.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Centroid {
    mean: OrderedFloat<f64>,
    weight: OrderedFloat<f64>,
    singleton: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // accept "quad","k1","k2","k3" from Python kwargs / serde
pub enum ScaleFamily {
    /// Piecewise-quadratic tail-friendly scale (legacy default).
    Quad,
    /// k1: arcsine scale; stricter tails than linear.
    K1,
    /// k2: logistic scale; forces unit centroids at extremes.
    K2,
    /// k3: double-log scale; ultra-strict tails.
    K3,
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

impl Centroid {
    pub fn new(mean: f64, weight: f64) -> Self {
        // Default heuristic: weight==1 => singleton; weight>1 is not, unless explicitly marked.
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
            singleton: (weight == 1.0),
        }
    }

    /// Construct a centroid known to be a pile of identical values.
    /// This sets `singleton = true` even if `weight > 1`.
    pub fn new_singleton_pile(mean: f64, weight: f64) -> Self {
        debug_assert!(weight >= 1.0);
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
            singleton: true,
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean.into_inner()
    }
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight.into_inner()
    }
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.singleton
    }

    /// Fold a batch `(sum, weight)` into this centroid and return the contributed sum.
    /// Note: This does *not* mutate `singleton`; the compressor decides whether we remained
    /// a same-mean pile or got mixed with other means.
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

impl Default for Centroid {
    fn default() -> Self {
        Centroid {
            mean: OrderedFloat::from(0.0),
            weight: OrderedFloat::from(1.0),
            singleton: true,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    /// Protect this many *raw items* on each tail during compression.
    /// First/last `extra_tail_keep` items are never merged by the compressor.
    extra_tail_keep: usize,
    sum: OrderedFloat<f64>,
    count: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
    scale: ScaleFamily,
}

impl Default for TDigest {
    fn default() -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size: 100,
            extra_tail_keep: 0,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: ScaleFamily::Quad, // preserve old behavior
        }
    }
}

impl TDigest {
    pub fn new_with_size(max_size: usize) -> Self {
        Self::new_with_size_and_scale(max_size, ScaleFamily::Quad)
    }

    /// Pick the scale family explicitly.
    pub fn new_with_size_and_scale(max_size: usize, scale: ScaleFamily) -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size,
            extra_tail_keep: 0,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale,
        }
    }

    /// Minimal-API way to enable/adjust tail protection without touching existing call sites.
    #[inline]
    pub fn with_protected_tails(mut self, n: usize) -> Self {
        self.extra_tail_keep = n;
        self
    }

    /// Build from unsorted values (convenience over `merge_unsorted`).
    pub fn from_unsorted(values: &[f64], max_size: usize) -> TDigest {
        let base = TDigest::new_with_size(max_size);
        base.merge_unsorted(values.to_vec())
    }

    /// Convenience: estimate the q-quantile via your existing method (defined in `quantile.rs`).
    pub fn quantile(&self, q: f64) -> f64 {
        self.estimate_quantile(q)
    }

    /// Convenience: estimate CDF(x) via your existing method (defined in `cdf.rs`).
    pub fn cdf(&self, x: &[f64]) -> Vec<f64> {
        self.estimate_cdf(x)
    }

    /// Convenience: median == quantile(0.5).
    pub fn median(&self) -> f64 {
        self.estimate_quantile(0.5)
    }

    #[inline]
    pub fn scale(&self) -> ScaleFamily {
        self.scale
    }

    pub fn merge_unsorted(&self, mut unsorted_values: Vec<f64>) -> TDigest {
        unsorted_values.sort_by(|a, b| a.total_cmp(b));
        self.merge_sorted(unsorted_values)
    }

    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }
        let mut result = self.new_result_for_values(&sorted_values);

        let stream = MergeByMean::from_centroids_and_values(&self.centroids, &sorted_values);
        let compressed = Self::compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;
        result
    }

    pub fn merge_digests(digests: Vec<TDigest>) -> TDigest {
        // TODO: scale different down and have one max_size
        let max_size = digests.first().map(|d| d.max_size).unwrap_or(100);
        let mut chosen_scale = ScaleFamily::Quad;
        let mut chosen_extra = 0usize;

        // Pre-reserve runs to avoid churn.
        let mut runs: Vec<&[Centroid]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if runs.is_empty() {
                    chosen_scale = d.scale; // adopt first non-empty's scale
                    chosen_extra = d.extra_tail_keep;
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
            max_size,
            extra_tail_keep: chosen_extra,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(total_count),
            max,
            min,
            scale: chosen_scale,
        };

        let merged_stream = KWayCentroidMerge::new(runs);
        let compressed = Self::compress_into(&mut result, max_size, merged_stream);
        result.centroids = compressed;

        result
    }

    /* ===========================
     * Small utilities
     * =========================== */

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
                extra_tail_keep: 0,
                sum: OrderedFloat::from(sum),
                count: OrderedFloat::from(count),
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
                scale: ScaleFamily::Quad,
            }
        } else {
            let sz = centroids.len();
            let digests = vec![
                TDigest::new_with_size(100),
                TDigest::new(centroids, sum, count, max, min, sz),
            ];
            Self::merge_digests(digests)
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        let n = self.count.into_inner();
        if n > 0.0 {
            self.sum.into_inner() / n
        } else {
            0.0
        }
    }
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.into_inner()
    }
    #[inline]
    pub fn count(&self) -> f64 {
        self.count.into_inner()
    }
    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner()
    }
    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }
    #[inline]
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    fn new_result_for_values(&self, values: &[f64]) -> TDigest {
        let mut r = TDigest::new_with_size_and_scale(self.max_size(), self.scale)
            .with_protected_tails(self.extra_tail_keep);

        r.count = OrderedFloat::from(self.count() + values.len() as f64);
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

    /// Family-aware q -> k mapping. `d` is the scale denominator (≈ max_size).
    fn q_to_k(q: f64, d: f64, family: ScaleFamily) -> f64 {
        use std::f64::consts::{LN_2, PI};
        let eps = 1e-15;
        let qq = TDigest::clamp(q, eps, 1.0 - eps);
        match family {
            // Piecewise-quadratic
            ScaleFamily::Quad => {
                // Inverse of: r=k/d; q = 2r^2 (r<0.5), else 1-2(1-r)^2
                let r = if qq < 0.5 {
                    (qq * 0.5).sqrt()
                } else {
                    1.0 - ((1.0 - qq) * 0.5).sqrt()
                };
                d * r
            }
            // k1: arcsine scale
            ScaleFamily::K1 => {
                let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
                (d / (2.0 * PI)) * s
            }
            // k2: logistic scale
            ScaleFamily::K2 => {
                let s = (qq / (1.0 - qq)).ln();
                (d / (4.0 * LN_2)) * s
            }
            // k3: double-log
            ScaleFamily::K3 => {
                let a = (1.0 / (1.0 - qq)).ln(); // ln(1/(1-q))
                let b = (1.0 / qq).ln(); // ln(1/q)
                let ratio = (a / b).max(eps);
                (d / 4.0) * ratio.ln()
            }
        }
    }

    #[inline]
    fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
        v.max(lo).min(hi)
    }

    /* ===========================
     * Core compressor (shared)
     * =========================== */
    fn compress_into<I>(result: &mut TDigest, max_size: usize, items: I) -> Vec<Centroid>
    where
        I: IntoIterator<Item = Centroid>,
    {
        let mut it = items.into_iter();
        let first = match it.next() {
            Some(c) => c,
            None => return Vec::with_capacity(0),
        };

        let total_n = result.count();
        let mut comp = Compressor::new(result, max_size, total_n, first);
        for next in it {
            comp.take(next);
        }
        comp.finish()
    }
}

struct Compressor<'a> {
    // output & accounting
    result: &'a mut TDigest,
    out: Vec<Centroid>,

    // scale parameters
    d: f64,
    n_inv: f64, // 1.0 / n cached
    family: ScaleFamily,

    // current centroid under construction
    curr: Centroid,
    curr_same_mean_only: bool,
    batch_sum: f64,
    batch_weight: f64,
    curr_w: f64,

    processed_left: f64,

    // cached left boundary in k-space: k(q_l) where q_l = processed_left * n_inv
    k_left: f64,

    // tail protection
    protect: f64, // items per side (clamped)
    n_total: f64,
}

impl<'a> Compressor<'a> {
    fn new(result: &'a mut TDigest, max_size: usize, total_n: f64, first: Centroid) -> Self {
        let d = max_size as f64;
        let n = total_n.max(1.0);
        let n_inv = 1.0 / n;
        let family = result.scale;
        let curr_w = first.weight();
        let k_left = TDigest::q_to_k(0.0, d, family);

        // Clamp protection to at most half the mass to avoid degenerate "all protected".
        let protect = (result.extra_tail_keep as f64).min(n * 0.5);

        Self {
            result,
            out: Vec::with_capacity(max_size.saturating_mul(2)),
            d,
            n_inv,
            family,
            curr: first,
            curr_same_mean_only: true,
            batch_sum: 0.0,
            batch_weight: 0.0,
            curr_w,
            processed_left: 0.0,
            k_left,
            protect,
            n_total: n,
        }
    }

    #[inline]
    fn can_absorb(&self, next_w: f64) -> bool {
        if self.protect > 0.0 {
            let left = self.processed_left;
            let curr_end = left + self.curr_w;
            let next_end = curr_end + next_w;
            let right_start = self.n_total - self.protect;

            // Still within the left protected region → never absorb.
            if curr_end < self.protect {
                ttrace!(
                    "LEFT-GUARD: block absorb  curr_end={:.0} < protect={:.0}",
                    curr_end,
                    self.protect
                );
                return false;
            }

            // Don't *cross* out of the left protected region in one step.
            // If any part of the current run sits before `protect` and absorbing would jump past it,
            // freeze here and start a fresh run.
            if left < self.protect && next_end > self.protect {
                ttrace!(
                    "LEFT-GUARD: block crossing  left={:.0} .. next_end={:.0} crosses protect={:.0}",
                    left, next_end, self.protect
                );
                return false;
            }

            // Adding `next` would land inside the right protected region → don't absorb.
            if next_end > right_start {
                ttrace!(
                    "RIGHT-GUARD: block absorb  next_end={:.0} > right_start={:.0}",
                    next_end,
                    right_start
                );
                return false;
            }
        }

        // Normal k-space width rule.
        let q_r = (self.processed_left + self.curr_w + next_w) * self.n_inv;
        let k_r = TDigest::q_to_k(q_r, self.d, self.family);
        (k_r - self.k_left) <= 1.0 - 1e-12
    }

    #[inline]
    fn take(&mut self, next: Centroid) {
        ttrace!(
            "take: curr(m={:.6},w={:.0})  next(m={:.6},w={:.0})  processed_left={:.0}",
            self.curr.mean(),
            self.curr_w,
            next.mean(),
            next.weight(),
            self.processed_left
        );

        if self.can_absorb(next.weight()) {
            ttrace!("  -> ABSORB");
            // extend current run
            self.curr_same_mean_only &= next.mean() == self.curr.mean();
            self.batch_sum += next.mean() * next.weight();
            self.batch_weight += next.weight();
            self.curr_w += next.weight();
        } else {
            ttrace!("  -> FLUSH & NEW");
            // finalize current, then advance left and refresh k_left
            self.flush_curr();
            self.processed_left += self.curr_w;
            self.k_left = TDigest::q_to_k(self.processed_left * self.n_inv, self.d, self.family);

            // start new run
            self.curr = next;
            self.curr_same_mean_only = true;
            self.batch_sum = 0.0;
            self.batch_weight = 0.0;
            self.curr_w = self.curr.weight();
        }
    }

    #[inline]
    fn flush_curr(&mut self) {
        let contributed = self.curr.add(self.batch_sum, self.batch_weight);

        // keep atomic jump if run was same-mean only
        self.curr.singleton = self.curr.weight() == 1.0 || self.curr_same_mean_only;

        // account into result sum
        self.result.sum = (self.result.sum.into_inner() + contributed).into();

        // clear batch and emit
        self.batch_sum = 0.0;
        self.batch_weight = 0.0;
        self.out.push(self.curr);
    }

    fn finish(mut self) -> Vec<Centroid> {
        self.flush_curr();

        // After building, fuse adjacent equal means so weights aren't fragmented.
        let fused = coalesce_adjacent_equal_means(std::mem::take(&mut self.out));
        debug_assert!(is_sorted_by_mean(&fused));

        // Enforce unit singletons at edges if necessary (safe, preserves sum).
        let min = self.result.min();
        let max = self.result.max();
        let mut out = enforce_edge_unit_singletons(fused, min, max);
        out.shrink_to_fit();
        out
    }
}

fn is_sorted_by_mean(cs: &[Centroid]) -> bool {
    cs.windows(2).all(|w| w[0] <= w[1])
}

/// Merge adjacent centroids that have the exact same mean into a single centroid.
/// Keeps `singleton=true` and sums weights.
fn coalesce_adjacent_equal_means(xs: Vec<Centroid>) -> Vec<Centroid> {
    if xs.len() <= 1 {
        return xs;
    }
    let mut out: Vec<Centroid> = Vec::with_capacity(xs.len());
    let mut acc = xs[0];
    for c in xs.into_iter().skip(1) {
        if c.mean() == acc.mean() {
            let w = acc.weight() + c.weight();
            acc = Centroid::new_singleton_pile(acc.mean(), w);
        } else {
            out.push(acc);
            acc = c;
        }
    }
    out.push(acc);
    out
}

fn enforce_edge_unit_singletons(xs: Vec<Centroid>, min: f64, max: f64) -> Vec<Centroid> {
    let n = xs.len();
    if n == 0 {
        return xs;
    }

    if n == 1 {
        let c = xs[0];
        let left_split = c.weight() > 1.0 && c.mean() > min;
        let right_split = c.weight() > 1.0 && c.mean() < max;

        if !left_split && !right_split {
            return xs; // fast path
        }

        let mut out = Vec::with_capacity(3);

        // Left split if needed
        let mut core = c;
        if left_split {
            out.push(Centroid::new_singleton_pile(min, 1.0));
            let w_rem = c.weight() - 1.0;
            let m_rem = (c.weight() * c.mean() - min) / w_rem;
            core = Centroid {
                mean: OrderedFloat::from(m_rem),
                weight: OrderedFloat::from(w_rem),
                singleton: false,
            };
        }

        // Right split if needed
        if right_split {
            let w_rem = core.weight() - 1.0;
            let m_rem = (core.weight() * core.mean() - max) / w_rem;
            out.push(Centroid {
                mean: OrderedFloat::from(m_rem),
                weight: OrderedFloat::from(w_rem),
                singleton: false,
            });
            out.push(Centroid::new_singleton_pile(max, 1.0));
        } else {
            out.push(core);
        }

        debug_assert!(is_sorted_by_mean(&out));
        return out;
    }

    // n >= 2
    let left_split = xs[0].weight() > 1.0 && xs[0].mean() > min;
    let right_split = xs[n - 1].weight() > 1.0 && xs[n - 1].mean() < max;

    if !left_split && !right_split {
        return xs; // fast path
    }

    // Allocate exactly what we need: original n plus up to 2 singletons.
    let mut out = Vec::with_capacity(n + (left_split as usize) + (right_split as usize));

    // ---- Left edge
    let mut first = xs[0];
    if left_split {
        out.push(Centroid::new_singleton_pile(min, 1.0));
        let w_rem = first.weight() - 1.0;
        let m_rem = (first.weight() * first.mean() - min) / w_rem;
        first = Centroid {
            mean: OrderedFloat::from(m_rem),
            weight: OrderedFloat::from(w_rem),
            singleton: false,
        };
    }
    out.push(first);

    // ---- Middle slice (Centroid is Copy → memcpy)
    if n > 2 {
        out.extend_from_slice(&xs[1..n - 1]);
    }

    // ---- Right edge
    let prev_mean = out.last().map(|c| c.mean()).unwrap_or(xs[n - 1].mean());
    if right_split {
        let w_rem = xs[n - 1].weight() - 1.0;
        let m_rem = (xs[n - 1].weight() * xs[n - 1].mean() - max) / w_rem;
        let remainder = Centroid {
            mean: OrderedFloat::from(m_rem),
            weight: OrderedFloat::from(w_rem),
            singleton: false,
        };

        if remainder.mean() < prev_mean {
            // Extremely rare numeric wobble: keep order by swapping with the last
            let last_idx = out.len() - 1;
            let prev = out[last_idx];
            out[last_idx] = remainder;
            out.push(prev);
        } else {
            out.push(remainder);
        }
        out.push(Centroid::new_singleton_pile(max, 1.0));
    } else {
        // No right split → append original last
        if xs[n - 1].mean() >= prev_mean {
            out.push(xs[n - 1]);
        }
    }

    debug_assert!(is_sorted_by_mean(&out), "edge enforcement broke sort order");
    out
}

struct MergeByMean<'a> {
    centroids: Peekable<std::slice::Iter<'a, Centroid>>,
    values: Peekable<std::slice::Iter<'a, f64>>,
    /// When pulling from values, group consecutive identical values into a single centroid
    /// so those piles become `singleton=true` even with weight>1.
    pending_value_run: Option<(f64, usize)>,
}

impl<'a> MergeByMean<'a> {
    fn from_centroids_and_values(centroids: &'a [Centroid], values: &'a [f64]) -> Self {
        Self {
            centroids: centroids.iter().peekable(),
            values: values.iter().peekable(),
            pending_value_run: None,
        }
    }

    /// Drain a run of identical raw values into one centroid.
    fn next_values_run(&mut self) -> Option<Centroid> {
        if let Some((val, len)) = self.pending_value_run.take() {
            return Some(Centroid::new_singleton_pile(val, len as f64));
        }
        let first = *self.values.next()?; // at least one value exists
        let mut len = 1usize;
        while let Some(&&v) = self.values.peek() {
            if v == first {
                self.values.next();
                len += 1;
            } else {
                break;
            }
        }
        Some(Centroid::new_singleton_pile(first, len as f64))
    }
}

impl Iterator for MergeByMean<'_> {
    type Item = Centroid;
    fn next(&mut self) -> Option<Self::Item> {
        // Always flush any stashed equal-mean value run before peeking again.
        if self.pending_value_run.is_some() {
            return self.next_values_run();
        }

        match (self.centroids.peek(), self.values.peek()) {
            (Some(c), Some(v)) => {
                if c.mean() < **v {
                    Some(*self.centroids.next().unwrap())
                } else if c.mean() > **v {
                    self.next_values_run()
                } else {
                    // Equal means: emit the existing centroid now, stash the *whole* values run.
                    let target = **v;
                    let mut len = 0usize;
                    while let Some(&&vv) = self.values.peek() {
                        if vv == target {
                            self.values.next();
                            len += 1;
                        } else {
                            break;
                        }
                    }
                    self.pending_value_run = Some((target, len));
                    Some(*self.centroids.next().unwrap())
                }
            }
            (Some(_), None) => Some(*self.centroids.next().unwrap()),
            (None, Some(_)) => self.next_values_run(),
            (None, None) => None,
        }
    }
}

struct KWayCentroidMerge<'a> {
    runs: Vec<&'a [Centroid]>,
    pos: Vec<usize>,
    heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>,
}

impl<'a> KWayCentroidMerge<'a> {
    fn new(runs: Vec<&'a [Centroid]>) -> Self {
        let mut heap = BinaryHeap::with_capacity(runs.len());
        let pos = vec![0; runs.len()];
        for (i, r) in runs.iter().enumerate() {
            if let Some(c) = r.first() {
                heap.push((Reverse(c.mean), i));
            }
        }
        Self { runs, pos, heap }
    }
}

impl Iterator for KWayCentroidMerge<'_> {
    type Item = Centroid;
    fn next(&mut self) -> Option<Self::Item> {
        let (Reverse(_m), run) = self.heap.pop()?;
        let i = self.pos[run];
        let slice = self.runs[run];
        let c = slice[i];
        self.pos[run] += 1;
        if self.pos[run] < slice.len() {
            let n = slice[self.pos[run]];
            self.heap.push((Reverse(n.mean), run));
        }
        Some(c)
    }
}

/* ===========================
 * Tests living here validate merge & structural invariants.
 * (Quantile/CDF tests live in respective modules as well.)
 * =========================== */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::test_helpers::{assert_exact, assert_rel_close};

    /// Identical values become singleton piles (weight>1 with `singleton=true`);
    /// distinct values are weight==1 singletons.
    #[test]
    fn singletons_grouped_runs() {
        // Values: three 1s, one 2, two 3s
        let vals = vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0];
        let t = TDigest::new_with_size(64).merge_sorted(vals);

        let cs = t.centroids();
        assert!(
            cs.windows(2).all(|w| w[0].mean() != w[1].mean()),
            "duplicate centroid means found"
        );

        assert_eq!(cs.len(), 3);

        assert_eq!(cs[0].mean(), 1.0);
        assert_eq!(cs[0].weight(), 3.0);
        assert!(cs[0].is_singleton());

        assert_eq!(cs[1].mean(), 2.0);
        assert_eq!(cs[1].weight(), 1.0);
        assert!(cs[1].is_singleton());

        assert_eq!(cs[2].mean(), 3.0);
        assert_eq!(cs[2].weight(), 2.0);
        assert!(cs[2].is_singleton());
    }

    #[test]
    fn merge_digests_uniform_100x1000() {
        let mut digests: Vec<TDigest> = Vec::with_capacity(100);
        for _ in 1..=100 {
            let t = TDigest::new_with_size(100).merge_sorted((1..=1_000).map(f64::from).collect());
            digests.push(t);
        }
        let t = TDigest::merge_digests(digests);

        assert_exact("Q(1.00)", 1000.0, t.estimate_quantile(1.0));
        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));

        let q99 = t.estimate_quantile(0.99);
        assert_rel_close("Q(0.99)", 990.0, q99, 7e-4);

        let q01 = t.estimate_quantile(0.01);
        assert_rel_close("Q(0.01)", 10.0, q01, 2e-1);

        let q50 = t.estimate_quantile(0.5);
        assert_rel_close("Q(0.50)", 500.0, q50, 2e-3);
    }

    #[test]
    fn merge_unsorted_smoke_small() {
        let vals = vec![5.0, 1.0, 3.0, 4.0, 2.0, 2.0, 9.0, 7.0];
        let t = TDigest::new_with_size(64).merge_unsorted(vals.clone());

        assert_exact("count", vals.len() as f64, t.count());
        assert_exact("min", 1.0, t.min());
        assert_exact("max", 9.0, t.max());

        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 9.0, t.estimate_quantile(1.0));

        let expected_mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        assert_rel_close("mean", expected_mean, t.mean(), 1e-9);
    }

    #[test]
    fn merge_digests_two_blocks_smoke() {
        let d1 =
            TDigest::new_with_size(128).merge_sorted((1..=5).map(f64::from).collect::<Vec<_>>());
        let d2 =
            TDigest::new_with_size(128).merge_sorted((6..=10).map(f64::from).collect::<Vec<_>>());

        let t = TDigest::merge_digests(vec![d1, d2]);

        assert_exact("count", 10.0, t.count());
        assert_exact("min", 1.0, t.min());
        assert_exact("max", 10.0, t.max());

        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 10.0, t.estimate_quantile(1.0));
        assert_rel_close("Q(0.50)", 5.5, t.estimate_quantile(0.5), 1e-9);
    }

    #[test]
    fn compact_scaling_prevents_overflow_and_preserves_shape() {
        use crate::tdigest::test_helpers::assert_rel_close;

        // Three centroids with astronomical weights to force scaling for f32.
        let c0 = Centroid::new(0.0, 1.0e36);
        let c1 = Centroid::new(1.0, 2.0e36);
        let c2 = Centroid::new(2.0, 3.0e36);
        let cents = vec![c0, c1, c2];

        // Keep TDigest "as-is" (no compression): set max_size comfortably above len.
        let sum = c0.mean() * c0.weight() + c1.mean() * c1.weight() + c2.mean() * c2.weight(); // 8e36
        let count = c0.weight() + c1.weight() + c2.weight(); // 6e36
        let td = TDigest::new(cents, sum, count, 2.0, 0.0, 512);

        // Baseline quantile
        let p50_before = td.estimate_quantile(0.5);

        // Write compact (f32/f32) — this will auto-scale internally.
        let ser = td.clone().try_to_series_compact("n").unwrap();

        // Sanity: compact centroids' weights must be finite f32 and well within range.
        {
            let sc = ser.struct_().expect("struct");
            let centroids_col = sc.field_by_name("centroids").expect("centroids");
            let list = centroids_col.list().expect("list");
            let row0 = list.get_as_series(0).expect("row0");
            let inner = row0.struct_().expect("inner");
            let w_series = inner.field_by_name("weight").expect("weight");
            let w_f32 = w_series.f32().expect("f32 weights");

            for i in 0..w_f32.len() {
                if let Some(wf) = w_f32.get(i) {
                    let wf: f32 = wf;
                    assert!(wf.is_finite(), "compact weight must be finite");
                    assert!(wf.abs() < f32::MAX / 4.0, "compact weight too large: {wf}");
                }
            }
        }

        // Parse back and verify shape invariants.
        let parsed = TDigest::try_from_series_compact(&ser).unwrap();
        assert_eq!(parsed.len(), 1, "one digest expected");
        let td2 = parsed.into_iter().next().unwrap();

        // 1) Quantiles should be preserved under uniform re-scaling of weights.
        let p50_after = td2.estimate_quantile(0.5);
        assert_rel_close("p50 preserved after scaling", p50_before, p50_after, 1e-6);

        // 2) Relative weights should match (only a global factor differs).
        let w1: Vec<f64> = td.centroids().iter().map(|c| c.weight()).collect();
        let w2: Vec<f64> = td2
            .centroids()
            .iter()
            .map(|c: &crate::tdigest::Centroid| c.weight())
            .collect();
        assert_eq!(w1.len(), w2.len(), "centroid count must match");

        let s1: f64 = w1.iter().sum();
        let s2: f64 = w2.iter().sum();
        for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
            let ra = *a / s1;
            let rb = *b / s2;
            assert_rel_close(&format!("weight ratio[{i}]"), ra, rb, 1e-7);

            // 3) Optional: CDF invariants at a few points.
            let xs = [0.5, 1.0, 1.5];
            let cdf1 = td.estimate_cdf(&xs);
            let cdf2 = td2.estimate_cdf(&xs);
            for i in 0..xs.len() {
                assert_rel_close(&format!("CDF({}) preserved", xs[i]), cdf1[i], cdf2[i], 1e-6);
            }
        }
    }

    #[test]
    fn edge_tail_protection_small_max() {
        use crate::tdigest::test_helpers::assert_exact;

        // Tiny core, protect 3 raw items per tail.
        let base = TDigest::new_with_size(4).with_protected_tails(3);

        // Merge three batches:
        // - middle bulk: 1..=20
        // - extend right tail: 21..=30
        // - distinct left tail: -3, -2, -1  (ensures 3 separate edge centroids are protected)
        let t1 = base.merge_sorted((1..=20).map(f64::from).collect());
        let t2 = t1.merge_sorted((21..=30).map(f64::from).collect());
        let t3 = t2.merge_sorted(vec![-3.0, -2.0, -1.0]);

        // Global invariants
        assert_exact("count", 33.0, t3.count());
        assert_exact("min", -3.0, t3.min());
        assert_exact("max", 30.0, t3.max());

        let cs = t3.centroids();

        // Sorted by mean (non-decreasing) and strictly increasing (no duplicate centroids).
        assert!(
            cs.windows(2).all(|w| w[0] <= w[1]),
            "centroids not sorted by mean"
        );
        assert!(
            cs.windows(2).all(|w| w[0].mean() < w[1].mean()),
            "duplicate centroid means found"
        );

        // All centroids: weight must be finite and > 0.
        for (i, c) in cs.iter().enumerate() {
            assert!(
                c.weight().is_finite() && c.weight() > 0.0,
                "invalid weight at {i}"
            );
            assert!(c.mean().is_finite(), "invalid mean at {i}");
        }

        // Must have at least the 3+3 protected edge centroids plus some middle.
        assert!(
            cs.len() >= 6,
            "need ≥6 centroids (3 per edge), got {}",
            cs.len()
        );

        // ---- Left edge: first 3 must be protected (singleton or pile) at -3, -2, -1.
        assert!(cs[0].is_singleton() && cs[1].is_singleton() && cs[2].is_singleton());
        assert_exact("left[0] mean", -3.0, cs[0].mean());
        assert_exact("left[1] mean", -2.0, cs[1].mean());
        assert_exact("left[2] mean", -1.0, cs[2].mean());

        // No cross-boundary absorption: next centroid (if any) strictly greater than -1.
        if cs.len() > 3 {
            assert!(
                cs[3].mean() > -1.0,
                "merged across left protected boundary (cs[3]={})",
                cs[3].mean()
            );
        }

        // ---- Right edge: last 3 must be protected (singleton or pile) at 28, 29, 30.
        let n = cs.len();
        assert!(cs[n - 3].is_singleton() && cs[n - 2].is_singleton() && cs[n - 1].is_singleton());
        assert_exact("right[-3] mean", 28.0, cs[n - 3].mean());
        assert_exact("right[-2] mean", 29.0, cs[n - 2].mean());
        assert_exact("right[-1] mean", 30.0, cs[n - 1].mean());

        // ---- Extreme quantiles exact due to protected edges.
        assert_exact("Q(0.00)", -3.0, t3.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 30.0, t3.estimate_quantile(1.0));

        // ---- Quantiles near tails: inside reasonable bands with tiny core.
        let q10 = t3.estimate_quantile(0.10);
        assert!(
            (-3.0..=1.0).contains(&q10),
            "Q(0.10) expected in [-3,1], got {}",
            q10
        );

        let q90 = t3.estimate_quantile(0.90);
        assert!(
            (25.0..=30.0).contains(&q90),
            "Q(0.90) expected in [25,30], got {}",
            q90
        );

        // Sanity: centroid count remains bounded with tiny core + protected tails.
        assert!(
            cs.len() <= 12,
            "too many centroids for tiny capacity with protected tails; got {}",
            cs.len()
        );
    }
}
