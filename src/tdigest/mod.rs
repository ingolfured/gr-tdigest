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
pub enum ScaleFamily {
    /// Your current piecewise-quadratic tail-friendly scale (default to preserve behavior).
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
            singleton: weight == 1.0,
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    sum: OrderedFloat<f64>,
    count: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
    scale: ScaleFamily, // <— NEW
}
impl Default for TDigest {
    fn default() -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size: 100,
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
        TDigest {
            centroids: Vec::new(),
            max_size,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: ScaleFamily::Quad, // default
        }
    }

    /// New: pick the scale family explicitly.
    pub fn new_with_size_and_scale(max_size: usize, scale: ScaleFamily) -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale,
        }
    }

    #[inline]
    pub fn scale(&self) -> ScaleFamily {
        self.scale
    }

    pub fn merge_unsorted(&self, unsorted_values: Vec<f64>) -> TDigest {
        let mut v: Vec<OrderedFloat<f64>> = unsorted_values
            .into_iter()
            .map(OrderedFloat::from)
            .collect();
        v.sort();
        let v = v.into_iter().map(|f| f.into_inner()).collect();
        self.merge_sorted(v)
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
        let max_size = digests.first().map(|d| d.max_size).unwrap_or(100);
        let mut chosen_scale = ScaleFamily::Quad;

        // Pre-reserve runs to avoid churn.
        let mut runs: Vec<&[Centroid]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if runs.is_empty() {
                    chosen_scale = d.scale; // <— adopt first non-empty's scale
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
        let mut result = TDigest::new_with_size_and_scale(max_size, chosen_scale);

        result.count = OrderedFloat::from(total_count);
        result.min = min;
        result.max = max;

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
        let mut r = TDigest::new_with_size_and_scale(self.max_size(), self.scale);
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
            // Your existing piecewise-quadratic (kept as-is)
            ScaleFamily::Quad => {
                // Inverse of: r=k/d; q = 2r^2 (r<0.5), else 1-2(1-r)^2
                // Solve for r: r = sqrt(q/2) for q<0.5; r = 1 - sqrt((1-q)/2) otherwise.
                let r = if qq < 0.5 {
                    (qq * 0.5).sqrt()
                } else {
                    1.0 - ((1.0 - qq) * 0.5).sqrt()
                };
                d * r
            }
            // k1: arcsine scale
            ScaleFamily::K1 => {
                // k = δ/(2π) * asin(2q-1); here d plays the role of δ.
                let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
                (d / (2.0 * PI)) * s
            }
            // k2: logistic scale
            ScaleFamily::K2 => {
                // k = δ/(4 ln 2) * ln(q/(1-q))
                let s = (qq / (1.0 - qq)).ln();
                (d / (4.0 * LN_2)) * s
            }
            // k3: double-log (ultra-strict tails)
            ScaleFamily::K3 => {
                // k = δ/4 * ln( ln(1/(1-q)) / ln(1/q) )
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
    n: f64,
    family: ScaleFamily,

    // current centroid under construction
    curr: Centroid,
    curr_same_mean_only: bool,
    batch_sum: f64,
    batch_weight: f64,
    curr_w: f64,

    processed_left: f64,
}

impl<'a> Compressor<'a> {
    fn new(result: &'a mut TDigest, max_size: usize, total_n: f64, first: Centroid) -> Self {
        let d = max_size as f64;
        let curr_w = first.weight();
        let family = result.scale; // <-- read before taking &mut in the struct

        Self {
            result,
            out: Vec::with_capacity(max_size),
            d,
            n: total_n.max(1.0),
            family, // <-- use the local copy
            curr: first,
            curr_same_mean_only: true,
            batch_sum: 0.0,
            batch_weight: 0.0,
            curr_w,
            processed_left: 0.0,
        }
    }

    #[inline]
    fn can_absorb(&self, next_w: f64) -> bool {
        let q_l = self.processed_left / self.n;
        let q_r = (self.processed_left + self.curr_w + next_w) / self.n;

        let k_l = TDigest::q_to_k(q_l, self.d, self.family);
        let k_r = TDigest::q_to_k(q_r, self.d, self.family);
        (k_r - k_l) <= 1.0 - 1e-12
    }

    #[inline]
    fn take(&mut self, next: Centroid) {
        if self.can_absorb(next.weight()) {
            // Absorb into the current centroid (defer arithmetic via batch buffers).
            self.curr_same_mean_only &= next.mean() == self.curr.mean();
            self.batch_sum += next.mean() * next.weight();
            self.batch_weight += next.weight();
            self.curr_w += next.weight();
        } else {
            // Finalize current centroid and move the "left" boundary forward.
            self.flush_curr();
            self.processed_left += self.curr_w;

            // Start a new centroid with `next`.
            self.curr = next;
            self.curr_same_mean_only = true;
            self.batch_sum = 0.0;
            self.batch_weight = 0.0;
            self.curr_w = self.curr.weight();
        }
    }

    #[inline]
    fn flush_curr(&mut self) {
        // Fold the batch into `curr`.
        let contributed = self.curr.add(self.batch_sum, self.batch_weight);

        // If everything that merged shared the same mean, keep it an atomic jump.
        self.curr.singleton = self.curr.weight() == 1.0 || self.curr_same_mean_only;

        // Account into the result sum.
        self.result.sum = (self.result.sum.into_inner() + contributed).into();

        // Clear batch and emit.
        self.batch_sum = 0.0;
        self.batch_weight = 0.0;
        self.out.push(self.curr);
    }

    fn finish(mut self) -> Vec<Centroid> {
        self.flush_curr();

        // After building, fuse adjacent equal means so weights aren't fragmented.
        let fused = coalesce_adjacent_equal_means(std::mem::take(&mut self.out));
        debug_assert!(is_sorted_by_mean(&fused));

        // This is safe and sum-preserving, and it won't introduce duplicate means.
        let min = self.result.min();
        let max = self.result.max();
        let enforced = enforce_edge_unit_singletons(fused, min, max);

        let mut out = enforced;
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

fn enforce_edge_unit_singletons(mut xs: Vec<Centroid>, min: f64, max: f64) -> Vec<Centroid> {
    if xs.is_empty() {
        return xs;
    }

    // Left edge
    {
        let first_mean = xs[0].mean();
        let first_w = xs[0].weight();
        if first_w > 1.0 && first_mean > min {
            let w_rem = first_w - 1.0;
            // New remainder mean that preserves total sum on the left centroid
            let m_rem = (first_w * first_mean - min) / w_rem;

            // Replace the original first centroid with the remainder (not a singleton by default)
            xs[0] = Centroid {
                mean: OrderedFloat::from(m_rem),
                weight: OrderedFloat::from(w_rem),
                singleton: false,
            };
            // Insert a 1-weight singleton at the global min in front
            xs.insert(0, Centroid::new_singleton_pile(min, 1.0));
        }
    }

    // Right edge
    {
        let last_idx = xs.len() - 1;
        let last_mean = xs[last_idx].mean();
        let last_w = xs[last_idx].weight();
        if last_w > 1.0 && last_mean < max {
            let w_rem = last_w - 1.0;
            let m_rem = (last_w * last_mean - max) / w_rem;

            // Replace the original last centroid with the remainder
            xs[last_idx] = Centroid {
                mean: OrderedFloat::from(m_rem),
                weight: OrderedFloat::from(w_rem),
                singleton: false,
            };
            // Append a 1-weight singleton at the global max
            xs.push(Centroid::new_singleton_pile(max, 1.0));
        }
    }

    debug_assert!(is_sorted_by_mean(&xs), "edge enforcement broke sort order");
    xs
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
}
