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
}

impl TDigest {
    /* ===========================
     * Public API
     * =========================== */

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

        // Pre-reserve runs to avoid churn.
        let mut runs: Vec<&[Centroid]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }
        if total_count == 0.0 {
            return TDigest::default();
        }

        let mut result = TDigest::new_with_size(max_size);
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

    pub fn new_with_size(max_size: usize) -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
        }
    }

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
        let mut r = TDigest::new_with_size(self.max_size());
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

    /// Piecewise-quadratic tail-friendly scale:
    /// r = k/d; q = 2r² for r<0.5; q = 1 − 2(1 − r)² otherwise.
    fn k_to_q(k: f64, d: f64) -> f64 {
        let r = k / d;
        if r >= 0.5 {
            let b = 1.0 - r;
            1.0 - 2.0 * b * b
        } else {
            2.0 * r * r
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

        // Read before borrowing `result` mutably.
        let total_count = result.count();

        let mut comp = Compressor::new(result, max_size, first, total_count);
        for next in it {
            comp.take(next);
        }
        comp.finish()
    }
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
        }
    }
}

/* ===========================
 * Compressor
 * =========================== */

struct Compressor<'a> {
    result: &'a mut TDigest,
    out: Vec<Centroid>,
    scale: ScaleCursor,
    curr: Centroid,
    processed: f64,
    batch_sum: f64,
    batch_weight: f64,
    /// True if everything batched into `curr` had the same mean as `curr`.
    curr_same_mean_only: bool,
}

impl<'a> Compressor<'a> {
    fn new(result: &'a mut TDigest, max_size: usize, first: Centroid, total_count: f64) -> Self {
        Self {
            result,
            out: Vec::with_capacity(max_size),
            scale: ScaleCursor::new(max_size, total_count),
            curr: first,
            processed: first.weight(),
            batch_sum: 0.0,
            batch_weight: 0.0,
            curr_same_mean_only: true, // so far just `first`
        }
    }

    #[inline]
    fn take(&mut self, next: Centroid) {
        self.processed += next.weight();
        if self.processed <= self.scale.limit() {
            // Merge `next` into `curr` via batch buffers.
            self.curr_same_mean_only &= next.mean() == self.curr.mean();
            self.batch_sum += next.mean() * next.weight();
            self.batch_weight += next.weight();
        } else {
            // Finalize `curr` with whatever we batched.
            self.flush_curr();
            self.scale.advance();
            // Start a new current centroid.
            self.curr = next;
            self.curr_same_mean_only = true;
        }
    }

    #[inline]
    fn flush_curr(&mut self) {
        let contributed = self.curr.add(self.batch_sum, self.batch_weight);
        self.curr.singleton = self.curr.weight() == 1.0 || self.curr_same_mean_only;

        self.result.sum = (self.result.sum.into_inner() + contributed).into();
        self.batch_sum = 0.0;
        self.batch_weight = 0.0;
        self.out.push(self.curr);
    }

    fn finish(mut self) -> Vec<Centroid> {
        self.flush_curr();
        debug_assert!(is_sorted_by_mean(&self.out));

        // Fuse adjacent equal-mean neighbors into piles so weights aren't fragmented.
        let mut fused = coalesce_adjacent_equal_means(std::mem::take(&mut self.out));
        fused.shrink_to_fit();
        fused
    }
}

/* ===========================
 * Scale + merge streams
 * =========================== */

struct ScaleCursor {
    k: f64,
    d: f64,
    n: f64,
    current_limit: f64,
}
impl ScaleCursor {
    fn new(max_size: usize, total_count: f64) -> Self {
        let mut s = Self {
            k: 1.0,
            d: max_size as f64,
            n: total_count,
            current_limit: 0.0,
        };
        s.current_limit = TDigest::k_to_q(s.k, s.d) * s.n;
        s.k += 1.0;
        s
    }
    #[inline]
    fn limit(&self) -> f64 {
        self.current_limit
    }
    #[inline]
    fn advance(&mut self) {
        self.current_limit = TDigest::k_to_q(self.k, self.d) * self.n;
        self.k += 1.0;
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
