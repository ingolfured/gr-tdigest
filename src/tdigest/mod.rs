/*
 * Original version created by by Paul Meng and distributed under Apache-2.0 license.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * https://github.com/MnO2/t-digest
 *
 */

pub mod codecs;

use ordered_float::OrderedFloat;
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

/// Centroid implementation to the cluster mentioned in the paper.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct Centroid {
    mean: OrderedFloat<f64>,
    weight: OrderedFloat<f64>,
}

impl PartialOrd for Centroid {
    fn partial_cmp(&self, other: &Centroid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Centroid {
    fn cmp(&self, other: &Centroid) -> Ordering {
        self.mean.cmp(&other.mean)
    }
}

impl Centroid {
    pub fn new(mean: f64, weight: f64) -> Self {
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
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

    pub fn add(&mut self, sum: f64, weight: f64) -> f64 {
        let weight_: f64 = self.weight.into_inner();
        let mean_: f64 = self.mean.into_inner();

        let new_sum: f64 = sum + weight_ * mean_;
        let new_weight: f64 = weight_ + weight;
        self.weight = OrderedFloat::from(new_weight);
        self.mean = OrderedFloat::from(new_sum / new_weight);
        new_sum
    }
}

impl Default for Centroid {
    fn default() -> Self {
        Centroid {
            mean: OrderedFloat::from(0.0),
            weight: OrderedFloat::from(1.0),
        }
    }
}

/// T-Digest to be operated on.
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
            let digests: Vec<TDigest> = vec![
                TDigest::new_with_size(100),
                TDigest::new(centroids, sum, count, max, min, sz),
            ];

            Self::merge_digests(digests)
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        let count_: f64 = self.count.into_inner();
        let sum_: f64 = self.sum.into_inner();

        if count_ > 0.0 {
            sum_ / count_
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
    pub fn centroids(&self) -> &Vec<Centroid> {
        &self.centroids
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

impl TDigest {
    fn k_to_q(k: f64, d: f64) -> f64 {
        let k_div_d = k / d;
        if k_div_d >= 0.5 {
            let base = 1.0 - k_div_d;
            1.0 - 2.0 * base * base
        } else {
            2.0 * k_div_d * k_div_d
        }
    }

    fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
        if v > hi {
            hi
        } else if v < lo {
            lo
        } else {
            v
        }
    }

    // See
    // https://github.com/protivinsky/pytdigest/blob/main/pytdigest/tdigest.c#L300-L336
    pub fn estimate_cdf(&self, vals: &[f64]) -> Vec<f64> {
        let n = self.centroids.len();
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        let mut means: Vec<f64> = Vec::with_capacity(n);
        let mut weights: Vec<f64> = Vec::with_capacity(n);
        for c in &self.centroids {
            means.push(c.mean());
            weights.push(c.weight());
        }
        // Precompute running total weights
        let mut running_total_weights: Vec<f64> = Vec::with_capacity(n);
        let mut total = 0.0;
        for &w in &weights {
            running_total_weights.push(total);
            total += w;
        }
        let count = self.count();
        vals.iter()
            .map(|&val| {
                // Binary search for the first centroid with mean >= val
                match means.binary_search_by(|m| m.partial_cmp(&val).unwrap()) {
                    Ok(mut centroid_index) => {
                        // Find the first centroid with mean == val (in case of duplicates)
                        while centroid_index > 0 && means[centroid_index - 1] == val {
                            centroid_index -= 1;
                        }
                        // Sum all centroids with this mean
                        let mut weight_at_value = weights[centroid_index];
                        let running_total_start = running_total_weights[centroid_index];
                        let mut i = centroid_index + 1;
                        while i < n && means[i] == val {
                            weight_at_value += weights[i];
                            i += 1;
                        }
                        (running_total_start + (weight_at_value / 2.0)) / count
                    }
                    Err(centroid_index) => {
                        if centroid_index == 0 {
                            0.0
                        } else if centroid_index >= n {
                            1.0
                        } else {
                            let cr_mean = means[centroid_index];
                            let cl_mean = means[centroid_index - 1];
                            let cr_weight = weights[centroid_index];
                            let cl_weight = weights[centroid_index - 1];
                            let mut running_total_weight = running_total_weights[centroid_index];
                            running_total_weight -= cl_weight / 2.0;
                            // If both are weight 1 and val is exactly halfway, return mean of their CDF steps
                            if cl_weight == 1.0
                                && cr_weight == 1.0
                                && ((val - (cl_mean + cr_mean) / 2.0).abs() < 1e-12)
                            {
                                let cdf_left =
                                    (running_total_weights[centroid_index - 1] + 0.5) / count;
                                let cdf_right =
                                    (running_total_weights[centroid_index] + 0.5) / count;
                                return 0.5 * (cdf_left + cdf_right);
                            }
                            // Check if within 0.5 weighted distance of a centroid of weight 1
                            if cl_weight == 1.0 && (val - cl_mean).abs() <= 0.5 {
                                // Step at cl_mean
                                return (running_total_weights[centroid_index - 1] + 0.5) / count;
                            }
                            if cr_weight == 1.0 && (val - cr_mean).abs() <= 0.5 {
                                // Step at cr_mean
                                return (running_total_weights[centroid_index] + 0.5) / count;
                            }
                            let m = (cr_mean - cl_mean) / (cl_weight / 2.0 + cr_weight / 2.0);
                            let x = (val - cl_mean) / m;
                            (running_total_weight + x) / count
                        }
                    }
                }
            })
            .collect()
    }

    pub fn merge_unsorted(&self, unsorted_values: Vec<f64>) -> TDigest {
        let mut sorted_values: Vec<OrderedFloat<f64>> = unsorted_values
            .into_iter()
            .map(OrderedFloat::from)
            .collect();
        sorted_values.sort();
        let sorted_values = sorted_values.into_iter().map(|f| f.into_inner()).collect();

        self.merge_sorted(sorted_values)
    }

    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }

        let mut result = TDigest::new_with_size(self.max_size());
        result.count = OrderedFloat::from(self.count() + (sorted_values.len() as f64));

        let maybe_min = OrderedFloat::from(*sorted_values.first().unwrap());
        let maybe_max = OrderedFloat::from(*sorted_values.last().unwrap());

        if self.count() > 0.0 {
            result.min = std::cmp::min(self.min, maybe_min);
            result.max = std::cmp::max(self.max, maybe_max);
        } else {
            result.min = maybe_min;
            result.max = maybe_max;
        }

        let mut compressed: Vec<Centroid> = Vec::with_capacity(self.max_size);

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_count: f64 =
            Self::k_to_q(k_limit, self.max_size as f64) * result.count.into_inner();
        k_limit += 1.0;

        let mut iter_centroids = self.centroids.iter().peekable();
        let mut iter_sorted_values = sorted_values.iter().peekable();

        let mut curr: Centroid = if let Some(c) = iter_centroids.peek() {
            let curr = **iter_sorted_values.peek().unwrap();
            if c.mean() < curr {
                iter_centroids.next().unwrap().clone()
            } else {
                Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
            }
        } else {
            Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
        };

        let mut weight_so_far: f64 = curr.weight();

        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        while iter_centroids.peek().is_some() || iter_sorted_values.peek().is_some() {
            let next: Centroid = if let Some(c) = iter_centroids.peek() {
                if iter_sorted_values.peek().is_none()
                    || c.mean() < **iter_sorted_values.peek().unwrap()
                {
                    iter_centroids.next().unwrap().clone()
                } else {
                    Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
                }
            } else {
                Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
            };

            let next_sum: f64 = next.mean() * next.weight();
            weight_so_far += next.weight();

            if weight_so_far <= q_limit_times_count {
                sums_to_merge += next_sum;
                weights_to_merge += next.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum.into_inner() + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;

                compressed.push(curr.clone());
                q_limit_times_count = Self::k_to_q(k_limit, self.max_size as f64) * result.count();
                k_limit += 1.0;
                curr = next;
            }
        }

        result.sum =
            OrderedFloat::from(result.sum.into_inner() + curr.add(sums_to_merge, weights_to_merge));
        compressed.push(curr);
        compressed.shrink_to_fit();
        compressed.sort();

        result.centroids = compressed;
        result
    }

    fn external_merge(centroids: &mut [Centroid], first: usize, middle: usize, last: usize) {
        let mut result: Vec<Centroid> = Vec::with_capacity(centroids.len());

        let mut i = first;
        let mut j = middle;

        while i < middle && j < last {
            match centroids[i].cmp(&centroids[j]) {
                Ordering::Less => {
                    result.push(centroids[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(centroids[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    result.push(centroids[i].clone());
                    i += 1;
                }
            }
        }

        while i < middle {
            result.push(centroids[i].clone());
            i += 1;
        }

        while j < last {
            result.push(centroids[j].clone());
            j += 1;
        }

        i = first;
        for centroid in result.into_iter() {
            centroids[i] = centroid;
            i += 1;
        }
    }

    // Merge multiple T-Digests
    pub fn merge_digests(digests: Vec<TDigest>) -> TDigest {
        let n_centroids: usize = digests.iter().map(|d| d.centroids.len()).sum();
        if n_centroids == 0 {
            return TDigest::default();
        }

        let max_size = digests.first().unwrap().max_size;
        let mut centroids: Vec<Centroid> = Vec::with_capacity(n_centroids);
        let mut starts: Vec<usize> = Vec::with_capacity(digests.len());

        let mut count: f64 = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        let mut start: usize = 0;
        for digest in digests.into_iter() {
            starts.push(start);

            let curr_count: f64 = digest.count();
            if curr_count > 0.0 {
                min = std::cmp::min(min, digest.min);
                max = std::cmp::max(max, digest.max);
                count += curr_count;
                for centroid in digest.centroids {
                    centroids.push(centroid);
                    start += 1;
                }
            }
        }

        let mut digests_per_block: usize = 1;
        while digests_per_block < starts.len() {
            for i in (0..starts.len()).step_by(digests_per_block * 2) {
                if i + digests_per_block < starts.len() {
                    let first = starts[i];
                    let middle = starts[i + digests_per_block];
                    let last = if i + 2 * digests_per_block < starts.len() {
                        starts[i + 2 * digests_per_block]
                    } else {
                        centroids.len()
                    };

                    debug_assert!(first <= middle && middle <= last);
                    Self::external_merge(&mut centroids, first, middle, last);
                }
            }

            digests_per_block *= 2;
        }

        let mut result = TDigest::new_with_size(max_size);
        let mut compressed: Vec<Centroid> = Vec::with_capacity(max_size);

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_count: f64 = Self::k_to_q(k_limit, max_size as f64) * count;

        let mut iter_centroids = centroids.iter_mut();
        let mut curr = iter_centroids.next().unwrap();
        let mut weight_so_far: f64 = curr.weight();
        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        for centroid in iter_centroids {
            weight_so_far += centroid.weight();

            if weight_so_far <= q_limit_times_count {
                sums_to_merge += centroid.mean() * centroid.weight();
                weights_to_merge += centroid.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum.into_inner() + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;
                compressed.push(curr.clone());
                q_limit_times_count = Self::k_to_q(k_limit, max_size as f64) * count;
                k_limit += 1.0;
                curr = centroid;
            }
        }

        result.sum =
            OrderedFloat::from(result.sum.into_inner() + curr.add(sums_to_merge, weights_to_merge));
        compressed.push(curr.clone());
        compressed.shrink_to_fit();
        compressed.sort();

        result.count = OrderedFloat::from(count);
        result.min = min;
        result.max = max;
        result.centroids = compressed;
        result
    }

    /// To estimate the value located at `q` quantile
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() {
            return 0.0;
        }

        let count_: f64 = self.count.into_inner();
        let rank: f64 = q * count_;

        let mut pos: usize;
        let mut t: f64;
        if q > 0.5 {
            if q >= 1.0 {
                return self.max();
            }

            pos = 0;
            t = count_;

            for (k, centroid) in self.centroids.iter().enumerate().rev() {
                t -= centroid.weight();

                if rank >= t {
                    pos = k;
                    break;
                }
            }
        } else {
            if q <= 0.0 {
                return self.min();
            }

            pos = self.centroids.len() - 1;
            t = 0.0;

            for (k, centroid) in self.centroids.iter().enumerate() {
                if rank < t + centroid.weight() {
                    pos = k;
                    break;
                }

                t += centroid.weight();
            }
        }

        let mut delta = 0.0;
        let mut min: f64 = self.min.into_inner();
        let mut max: f64 = self.max.into_inner();

        if self.centroids.len() > 1 {
            if pos == 0 {
                delta = self.centroids[pos + 1].mean() - self.centroids[pos].mean();
                max = self.centroids[pos + 1].mean();
            } else if pos == (self.centroids.len() - 1) {
                delta = self.centroids[pos].mean() - self.centroids[pos - 1].mean();
                min = self.centroids[pos - 1].mean();
            } else {
                delta = (self.centroids[pos + 1].mean() - self.centroids[pos - 1].mean()) / 2.0;
                min = self.centroids[pos - 1].mean();
                max = self.centroids[pos + 1].mean();
            }
        }

        let value =
            self.centroids[pos].mean() + ((rank - t) / self.centroids[pos].weight() - 0.5) * delta;
        Self::clamp(value, min, max)
    }

    fn find_median_between_centroids(&self) -> Option<f64> {
        if (self.count.into_inner() as i64) % 2 != 0 {
            return None;
        }
        let mut target = (self.count.into_inner() as i64) / 2;
        for (idx, c) in self.centroids.iter().enumerate() {
            target -= c.weight() as i64;
            if target == 0 {
                let m1 = c.mean();
                let m2 = self.centroids[idx + 1].mean();
                return Option::Some((m1 + m2) / 2.0);
            }
            if target < 0 {
                return Option::None;
            }
        }
        Option::None
    }

    pub fn estimate_median(&self) -> f64 {
        /*
         * If the number of elements is even, median is average of two adjacent observation.
         * Interpolation algorithm used in `estimate_quantile` often positions estimated median too far away from the middle point.
         * So let's detect the case when the median is exactly between two centroids.
         */
        self.find_median_between_centroids()
            .unwrap_or(self.estimate_quantile(0.5))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_centroid_addition_regression() {
        //https://github.com/MnO2/t-digest/pull/1

        let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let mut t = TDigest::new_with_size(10);

        for v in vals {
            t = t.merge_unsorted(vec![v]);
        }

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 1.0;
        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.95);
        let expected: f64 = 2.0;
        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_merge_sorted_against_uniform_distro() {
        let t = TDigest::new_with_size(100);
        let values: Vec<f64> = (1..=1_000_000).map(f64::from).collect();

        let t = t.merge_sorted(values);

        let ans = t.estimate_quantile(1.0);
        let expected: f64 = 1_000_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.99);
        let expected: f64 = 990_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.01);
        let expected: f64 = 10_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.0);
        let expected: f64 = 1.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 500_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_merge_unsorted_against_uniform_distro() {
        let t = TDigest::new_with_size(100);
        let values: Vec<f64> = (1..=1_000_000).map(f64::from).collect();

        let t = t.merge_unsorted(values);

        let ans = t.estimate_quantile(1.0);
        let expected: f64 = 1_000_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.99);
        let expected: f64 = 990_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.01);
        let expected: f64 = 10_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.0);
        let expected: f64 = 1.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 500_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_merge_sorted_against_skewed_distro() {
        let t = TDigest::new_with_size(100);
        let mut values: Vec<f64> = (1..=600_000).map(f64::from).collect();
        for _ in 0..400_000 {
            values.push(1_000_000.0);
        }

        let t = t.merge_sorted(values);

        let ans = t.estimate_quantile(0.99);
        let expected: f64 = 1_000_000.0;
        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.01);
        let expected: f64 = 10_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 500_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_merge_unsorted_against_skewed_distro() {
        let t = TDigest::new_with_size(100);
        let mut values: Vec<f64> = (1..=600_000).map(f64::from).collect();
        for _ in 0..400_000 {
            values.push(1_000_000.0);
        }

        let t = t.merge_unsorted(values);

        let ans = t.estimate_quantile(0.99);
        let expected: f64 = 1_000_000.0;
        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.01);
        let expected: f64 = 10_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 500_000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_merge_digests() {
        let mut digests: Vec<TDigest> = Vec::new();

        for _ in 1..=100 {
            let t = TDigest::new_with_size(100);
            let values: Vec<f64> = (1..=1_000).map(f64::from).collect();
            let t = t.merge_sorted(values);
            digests.push(t)
        }

        let t = TDigest::merge_digests(digests);

        let ans = t.estimate_quantile(1.0);
        let expected: f64 = 1000.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.99);
        let expected: f64 = 990.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.01);
        let expected: f64 = 10.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.2);

        let ans = t.estimate_quantile(0.0);
        let expected: f64 = 1.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);

        let ans = t.estimate_quantile(0.5);
        let expected: f64 = 500.0;

        let percentage: f64 = (expected - ans).abs() / expected;
        assert!(percentage < 0.01);
    }

    #[test]
    fn test_median_between_centroids() {
        // median of [-1, -1, ..., 1, 1] should be ~0
        let mut quantile_didnt_work: bool = false;
        for num in [1, 2, 3, 10, 20] {
            let mut t = TDigest::new_with_size(100);
            for _ in 1..=num {
                t = t.merge_sorted(vec![-1.0]);
            }
            for _ in 1..=num {
                t = t.merge_sorted(vec![1.0]);
            }

            if t.estimate_quantile(0.5).abs() > 0.1 {
                quantile_didnt_work = true;
            }

            assert!(t.estimate_median().abs() < 0.01);
        }
        assert!(quantile_didnt_work);
    }

    #[test]
    fn test_cdf() {
        let t = TDigest::new_with_size(100);
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = t.merge_sorted(values);

        let cdf_vals = t.estimate_cdf(&[3.0, 1.0, 5.0, 0.0, 6.0]);
        assert!(
            (cdf_vals[0] - 0.5).abs() < 0.0001,
            "CDF(3.0) deviates from 0.5"
        );
        assert!(
            (cdf_vals[1] - 0.1).abs() < 0.0001,
            "CDF(1.0) deviates from 0.1"
        );
        assert!(
            (cdf_vals[2] - 0.9).abs() < 0.0001,
            "CDF(5.0) deviates from 0.9"
        );
        assert_eq!(cdf_vals[3], 0.0, "CDF(0.0) should be 0.0");
        assert_eq!(cdf_vals[4], 1.0, "CDF(6.0) should be 1.0");
    }

    #[test]
    fn test_cdf_out_of_bounds() {
        let t = TDigest::new_with_size(100);
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = t.merge_sorted(values);

        let cdf_vals = t.estimate_cdf(&[0.0, 6.0]);
        // Test when the value is less than the minimum element
        assert_eq!(cdf_vals[0], 0.0, "CDF(0.0) should be 0.0");
        // Test when the value is greater than the maximum element
        assert_eq!(cdf_vals[1], 1.0, "CDF(6.0) should be 1.0");
    }

    #[test]
    fn test_cdf_all_same_value() {
        // All values are the same, CDF should step from 0 to 1 at that value
        let t = TDigest::new_with_size(10);
        let t = t.merge_sorted(vec![2.0, 2.0, 2.0, 2.0, 2.0]);
        let cdf_vals = t.estimate_cdf(&[1.0, 2.0, 3.0]);
        println!("cdf_vals: {:?}", cdf_vals);
        assert_eq!(cdf_vals[0], 0.0, "CDF below all values should be 0.0");
        assert!(
            (cdf_vals[1] - 0.5).abs() < 0.1,
            "CDF at the value should be close to 0.5"
        );
        assert_eq!(cdf_vals[2], 1.0, "CDF above all values should be 1.0");
    }

    #[test]
    fn test_cdf_duplicate_centroids() {
        // Insert values to force duplicate centroids (same mean)
        let t = TDigest::new_with_size(100);
        let t = t.merge_sorted(vec![1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]);
        let cdf_vals = t.estimate_cdf(&[2.0, 1.0, 3.0]);
        // The exact values may depend on compression, but should be monotonic and between 0 and 1
        assert!(
            cdf_vals[0] > cdf_vals[1] && cdf_vals[2] > cdf_vals[0],
            "CDF monotonic with duplicates"
        );
        assert!(cdf_vals[0] > 0.0 && cdf_vals[0] < 1.0, "CDF in (0,1)");
    }

    #[test]
    fn test_cdf_large_and_one_small() {
        // Many large values, one very small value
        let mut values = vec![1e9; 100];
        values.push(-1e9);
        let t = TDigest::new_with_size(100);
        let t = t.merge_sorted(values);
        let cdf_vals = t.estimate_cdf(&[-1e9, 0.0, 1e9, 2e9]);
        assert!(cdf_vals[0] < 0.02, "CDF at smallest value");
        assert!(cdf_vals[1] < 0.05, "CDF at 0.0 should be small");
        assert_eq!(cdf_vals[3], 1.0, "CDF above all values");
    }

    #[test]
    fn test_cdf_small_numbers() {
        // Very small numbers, check for precision
        let t = TDigest::new_with_size(10);
        let t = t.merge_sorted(vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10]);
        let cdf_vals = t.estimate_cdf(&[2e-10, 3e-10, 6e-10]);
        assert!(cdf_vals[0] > 0.1 && cdf_vals[0] < 0.5, "CDF at 2e-10");
        assert!(cdf_vals[1] > cdf_vals[0], "CDF at 3e-10 > CDF at 2e-10");
        assert_eq!(cdf_vals[2], 1.0, "CDF above all values");
    }

    #[test]
    fn test_cdf_negative_values() {
        // Negative values and zero
        let t = TDigest::new_with_size(10);
        let t = t.merge_sorted(vec![-5.0, -2.0, 0.0, 2.0, 5.0]);
        let cdf_vals = t.estimate_cdf(&[-10.0, -2.0, 0.0, 3.0, 10.0]);
        assert_eq!(cdf_vals[0], 0.0, "CDF below all values");
        assert!(cdf_vals[1] > 0.0 && cdf_vals[1] < 0.5, "CDF at -2.0");
        assert!(cdf_vals[2] > cdf_vals[1] && cdf_vals[2] < 0.7, "CDF at 0.0");
        assert!(cdf_vals[3] > cdf_vals[2] && cdf_vals[3] < 1.0, "CDF at 3.0");
        assert_eq!(cdf_vals[4], 1.0, "CDF above all values");
    }

    #[test]
    fn test_cdf_empty_input() {
        // Empty input vector for vals
        let t = TDigest::new_with_size(10);
        let t = t.merge_sorted(vec![1.0, 2.0, 3.0]);
        let cdf_vals = t.estimate_cdf(&[]);
        assert_eq!(cdf_vals.len(), 0, "CDF of empty input should be empty");
    }
}

// ==== Quality harness: KS + MAE, single score + no-regression guard =========

#[derive(Debug, Clone)]
pub struct Quality {
    pub n: usize,
    pub max_size: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub n: usize,
    pub max_abs_err: f64, // KS
    pub mean_abs_err: f64, // MAE
}

impl QualityReport {
    /// Half-lives chosen so base lands around ~0.7
    pub fn quality_score(&self) -> f64 {
        self.quality_score_with_halves(0.028, 0.0008)
    }

    /// Customizable half-lives: score hits 0.5 when KS==half_ks and MAE==half_mae.
    pub fn quality_score_with_halves(&self, half_ks: f64, half_mae: f64) -> f64 {
        fn sub(x: f64, half: f64) -> f64 {
            if half <= 0.0 || !x.is_finite() { return 0.0; }
            (2.0f64).powf(-x / half).clamp(0.0, 1.0)
        }
        let s_ks  = sub(self.max_abs_err, half_ks);
        let s_mae = sub(self.mean_abs_err, half_mae);
        (s_ks * s_mae).sqrt()
    }

    pub fn strictly_better_than(&self, other: &QualityReport) -> bool {
        let eps = 1e-12;
        (self.max_abs_err <= other.max_abs_err + eps) &&
        (self.mean_abs_err <= other.mean_abs_err + eps)
    }

    pub fn to_line(&self) -> String {
        format!(
            "QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
            self.n, self.max_abs_err, self.mean_abs_err, self.quality_score()
        )
    }
    pub fn log(&self) { eprintln!("{}", self.to_line()); }
}

impl Quality {
    pub fn new(n: usize, max_size: usize, seed: u64) -> Self {
        Self { n, max_size, seed }
    }

    pub fn run(&self) -> QualityReport {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut values: Vec<f64> = Vec::with_capacity(self.n);
        if self.n == 0 {
            return QualityReport { n: 0, max_abs_err: f64::NAN, mean_abs_err: f64::NAN };
        }
        values.push(0.0); // guarantee a 0
        while values.len() < self.n {
            let bucket: u32 = rng.gen_range(0..100);
            let x = if bucket < 70 {
                // 70% uniform in [-1, 1]
                rng.gen_range(-1.0..1.0)
            } else if bucket < 90 {
                // 20% normal(0, 1000) via Box–Muller
                let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 1000.0
            } else {
                // 10% very large magnitudes: 10^U(3,9) with random sign
                let exp = rng.gen_range(3.0..9.0);
                let mag = 10f64.powf(exp);
                if rng.gen_bool(0.5) { mag } else { -mag }
            };
            values.push(x);
        }

        // 2) Exact ECDF at each sorted sample (midpoint convention on ties)
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact_cdf = Self::exact_ecdf_for_sorted(&values);

        let t0 = TDigest::new_with_size(self.max_size);
        let digest = t0.merge_sorted(values.clone()); // taking-ownership API => clone
        let td_cdf = digest.estimate_cdf(&values);

        let mut ks = 0.0;
        let mut sum_abs = 0.0;
        for (a, b) in exact_cdf.iter().zip(td_cdf.iter()) {
            let d = (a - b).abs();
            if d > ks { ks = d; }
            sum_abs += d;
        }
        let mae = sum_abs / (self.n as f64);

        QualityReport { n: self.n, max_abs_err: ks, mean_abs_err: mae }
    }

    fn exact_ecdf_for_sorted(sorted: &Vec<f64>) -> Vec<f64> {
        let n = sorted.len();
        let nf = n as f64;
        let mut out = vec![0.0; n];
        let mut i = 0usize;
        while i < n {
            let mut j = i + 1;
            while j < n && sorted[j] == sorted[i] { j += 1; }
            let mid = (i + j) as f64 / 2.0;
            let val = mid / nf;
            for k in i..j { out[k] = val; }
            i = j;
        }
        out
    }
}


#[test]
fn quality_smoke_test_ks_mae_score() {
    let rep = Quality::new(100_000, 100, 42).run();
    rep.log();

    let s = rep.quality_score();
    eprintln!("quality_score = {:.3}", s);

    assert!(s > 0.70 && s < 0.71); // expect ~0.7 with given halves
}

#[test]
fn quality_improves_with_larger_digest() {
    let base   = Quality::new(100_000, 100, 42).run();
    let better = Quality::new(100_000, 1000, 42).run();

    let s_base   = base.quality_score();
    let s_better = better.quality_score();

    eprintln!("BASE   -> {}", base.to_line());
    eprintln!("BETTER -> {}", better.to_line());

    assert!(better.strictly_better_than(&base));

    assert!(s_better >= s_base);
}