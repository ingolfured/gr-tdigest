//! Compressor: runs the TDigest pipeline stages **1→6** on an input stream of centroids.
//!
//! ### Pipeline (referenced by numbers throughout)
//! 1. **Normalize**: validate non-decreasing means; coalesce adjacent equal-means into a single
//!    centroid preserving mixed provenance (weight adds up); compute (∑w, ∑w·mean, min, max).
//! 2. **Slice**: partition `(left edges, interior, right edges)` and compute interior capacity
//!    from the active [`SingletonPolicy`].
//! 3. **Merge (k-limit)**: greedily merge interior under Δk ≤ 1 for the selected [`ScaleFamily`].
//! 4. **Cap**: if interior exceeds its budget, run a second k-limit merge with tuned `d'`
//!    (via binary search) to honor the cap while preserving scale-family geometry.
//! 5. **Assemble**: concatenate `left + core + right`.
//! 6. **Post**: optional policy finalization (e.g., total-cap for `Use`).
//!
//! #### Invariants
//! - Output centroids are strictly increasing by mean.
//! - Total weight is preserved (up to floating rounding).
//! - Stage ownership is explicit: same-mean runs are coalesced only in **1**; edge rules only in **2/6**.

use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::merges::normalize_stream;
use crate::tdigest::precision::FloatLike;
use crate::tdigest::scale::{q_to_k, ScaleFamily};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::TDigest;

const KLIMIT_TOL: f64 = 1e-12;
const CAP_SEARCH_ITERS: usize = 28;
const CAP_FALLBACK_ITERS: usize = 12;
const CAP_SEARCH_LO: f64 = 1e-6;

/* =============================================================================
 * Internal utilities
 * ============================================================================= */

#[derive(Clone, Copy, Debug, Default)]
struct WeightedStats {
    w_sum: f64,
    mw_sum: f64,
    first_mean: f64,
    last_mean: f64,
    first_atomic: bool,
    items: usize,
}

impl WeightedStats {
    #[inline]
    fn add<F: FloatLike + FloatCore>(&mut self, centroid: &Centroid<F>) {
        if self.items == 0 {
            self.first_mean = centroid.mean_f64();
            self.first_atomic = centroid.is_atomic();
        }
        self.last_mean = centroid.mean_f64();
        self.items += 1;
        self.w_sum += centroid.weight_f64();
        self.mw_sum += centroid.weight_f64() * centroid.mean_f64();
    }
    #[inline]
    fn mean(&self) -> f64 {
        if self.items == 1 {
            // A pass-through centroid has not been averaged. Recomputing m*w/w
            // can move its mean by an ulp, changing point-mass CDFs and making
            // later merges produce duplicate or out-of-order means.
            self.first_mean
        } else if self.w_sum > 0.0 {
            let mean = self.mw_sum / self.w_sum;
            // Weighted averaging can round beyond both input means by an ulp.
            // Disjoint sorted clusters must retain disjoint mean intervals.
            // Do not hide aggregate/weighted-sum overflow behind this bound.
            if self.w_sum.is_finite() && self.mw_sum.is_finite() && mean.is_finite() {
                if self.first_mean <= self.last_mean {
                    mean.clamp(self.first_mean, self.last_mean)
                } else {
                    // E18 can visit a sorted cluster from its right edge.
                    mean.clamp(self.last_mean, self.first_mean)
                }
            } else {
                mean
            }
        } else {
            0.0
        }
    }
    #[inline]
    fn centroid<F: FloatLike + FloatCore>(&self) -> Centroid<F> {
        if self.items == 1 && self.first_atomic {
            Centroid::<F>::new_atomic_f64(self.mean(), self.w_sum)
        } else {
            Centroid::<F>::new_mixed_f64(self.mean(), self.w_sum)
        }
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.w_sum <= 0.0
    }
    #[inline]
    fn clear(&mut self) {
        self.w_sum = 0.0;
        self.mw_sum = 0.0;
        self.items = 0;
    }
}

/// Run the full pipeline (**1→6**) on `items` and write digest-level metadata to `result`.
///
/// Stages:
/// 1) Normalize → `normalize_stream`
/// 2) Slice     → policy-dependent edge/interior partition
/// 3) Merge     → `klimit_merge`
/// 4) Cap       → `cap_core` / `cap_with_second_klimit`
/// 5) Assemble  → `assemble_with_edges`
/// 6) Post      → policy-specific finalization
pub(crate) fn compress_into<F, I>(
    result: &mut TDigest<F>,
    max_size: usize,
    items: I,
) -> Vec<Centroid<F>>
where
    F: FloatLike + FloatCore,
    I: IntoIterator<Item = Centroid<F>>,
{
    // -- 1) Normalize ---------------------------------------------------------
    let norm = normalize_stream::<F, _>(items);
    if norm.out.is_empty() {
        return norm.out;
    }

    // Centroid mass and weighted sums can be recomputed, but centroid means
    // do not retain the true extrema of mixed clusters. Keep the data bounds
    // carried by the add/merge result shell; use the stream for an unset bound.
    result.set_count(norm.total_w);
    result.set_sum(norm.total_mw);
    result.set_min(result.min().min(norm.min));
    result.set_max(result.max().max(norm.max));

    // -- 2) Slice -------------------------------------------------------------
    let policy = policy_from(result.singleton_policy(), max_size);
    let delta = result.delta();
    if delta.is_none() {
        if let Some(v) = policy.fast_path(&norm.out) {
            debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&v));
            return v;
        }
    }
    let slices = policy.slice(&norm.out);

    // -- 3) Merge (k-limit) ---------------------------------------------------
    let family: ScaleFamily = result.scale();
    let d: f64 = result.max_size() as f64;
    let core = if let Some(delta) = delta {
        // delta-mode is contract-validated upstream (scale = K2Norm, policy = Off);
        // Use the strict delta-mode k-limit; this is not a byte-identity guarantee.
        debug_assert_eq!(family, ScaleFamily::K2Norm);
        klimit_merge::<F>(slices.interior, delta, family, 0.0)
    } else {
        directional_klimit_merge::<F>(slices.interior, d, family, KLIMIT_TOL)
    };

    // -- 4) Cap ---------------------------------------------------------------
    // Second k-limit cap pass if the interior exceeds its budget.
    // Fallback to equal-weight bucketization remains as a safety net in cap_core.
    let core_capped = if delta.is_some() {
        core
    } else {
        cap_core::<F>(core, slices.caps.core_cap, d, family)
    };

    // -- 5) Assemble ----------------------------------------------------------
    let assembled = assemble_with_edges::<F>(slices.left, core_capped, slices.right);

    // -- 6) Post --------------------------------------------------------------
    let compressed = if delta.is_some() {
        assembled
    } else {
        policy.post(assembled)
    };

    debug_assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(
        &compressed
    ));
    compressed
}

/* =============================================================================
 * Policy (Stages 2 & 6)
 * ============================================================================= */

#[derive(Clone, Copy, Debug)]
enum Policy {
    Off { max: usize },
    Use { max: usize },
    UseProt { max: usize, k: usize },
}

fn policy_from(sp: SingletonPolicy, max: usize) -> Policy {
    match sp {
        SingletonPolicy::Off => Policy::Off { max },
        SingletonPolicy::Use => Policy::Use { max },
        SingletonPolicy::UseWithProtectedEdges(k) => Policy::UseProt { max, k },
    }
}

impl Policy {
    /// Fast-path for trivial/exact cases (part of **2**). Returns `Some` when no further work is needed.
    fn fast_path<F: FloatLike + FloatCore>(&self, out: &[Centroid<F>]) -> Option<Vec<Centroid<F>>> {
        match *self {
            Policy::Off { max } => {
                if out.len() <= max {
                    return Some(out.to_vec());
                }
                if max == 1 {
                    return Some(if out.len() == 1 {
                        out.to_vec()
                    } else {
                        vec![weighted_collapse(out)]
                    });
                }
                if max == 2 {
                    return Some(bucketize_equal_weight(out, 2));
                }
                None
            }
            Policy::Use { .. } | Policy::UseProt { .. } => None,
        }
    }

    /// **(2) Slice** — compute `(left, interior, right)` and interior capacity.
    fn slice<'a, F: FloatLike + FloatCore>(&self, out: &'a [Centroid<F>]) -> EdgeSlices<'a, F> {
        match *self {
            // No edge protection; core capacity equals `max`.
            Policy::Off { max } => EdgeSlices {
                left: &[],
                interior: out,
                right: &[],
                caps: EdgeCaps { core_cap: max },
            },
            // Preserve endpoints; core capacity excludes the two edges.
            Policy::Use { max } => {
                let n = out.len();
                let l = usize::from(n >= 1);
                let r = usize::from(n >= 2);

                let left = &out[..l];
                let right = if n > l { &out[n - r..] } else { &[] };
                let interior = if n > (l + r) { &out[l..n - r] } else { &[] };
                let core_cap = max.saturating_sub(l + r);

                EdgeSlices {
                    left,
                    interior,
                    right,
                    caps: EdgeCaps { core_cap },
                }
            }
            // Protect up to `k` **atomic** centroids per edge; interior has independent cap.
            Policy::UseProt { max, k } => {
                let n = out.len();
                let l_prot = edge_run_len(out, k, true);
                let r_prot = edge_run_len(out, k, false);

                let left_end = l_prot.min(n);
                // Protected ranges may touch or overlap on small digests. Keep
                // their union exactly once instead of discarding the right tail.
                let right_start = n.saturating_sub(r_prot).max(left_end);

                let left = &out[..left_end];
                let right = &out[right_start..];
                let interior = &out[left_end..right_start];

                EdgeSlices {
                    left,
                    interior,
                    right,
                    caps: EdgeCaps { core_cap: max },
                }
            }
        }
    }

    /// **(6) Post** — finalize per policy (e.g., total-cap enforcement for `Use`).
    fn post<F: FloatLike + FloatCore>(&self, assembled: Vec<Centroid<F>>) -> Vec<Centroid<F>> {
        match *self {
            Policy::Use { max } => {
                if assembled.len() <= max {
                    assembled
                } else if max == 0 {
                    Vec::new()
                } else {
                    bucketize_equal_weight(&assembled, max)
                }
            }
            Policy::Off { .. } | Policy::UseProt { .. } => assembled,
        }
    }
}

/* =============================================================================
 * Core operations (Stages 3, 4, 5)
 * ============================================================================= */

/// **(3) Merge (k-limit)** — greedy interior merge under Δk ≤ 1.
///
/// Guarantees:
/// - Order and total weight are preserved.
/// - Single-item clusters remain **atomic** only when the head was data-true atomic.
/// - Multi-item clusters are **mixed**.
///
/// `tol` is added to the `Δk ≤ 1` check (pass [`KLIMIT_TOL`] for the default
/// tolerant behaviour, `0.0` for the strict delta-mode bound).
fn klimit_merge<F: FloatLike + FloatCore>(
    items: &[Centroid<F>],
    d: f64,
    family: ScaleFamily,
    tol: f64,
) -> Vec<Centroid<F>> {
    klimit_merge_ordered(items.iter(), d, family, tol)
}

/// E18's frozen heuristic: at the first unequal mirrored weight, visit the
/// lighter side first. Equal profiles retain the original forward pass.
fn prefer_reverse<F: FloatLike + FloatCore>(items: &[Centroid<F>]) -> bool {
    for (left, right) in items.iter().zip(items.iter().rev()).take(items.len() / 2) {
        if left.weight_f64() < right.weight_f64() {
            return false;
        }
        if left.weight_f64() > right.weight_f64() {
            return true;
        }
    }
    false
}

fn directional_klimit_merge<F: FloatLike + FloatCore>(
    items: &[Centroid<F>],
    d: f64,
    family: ScaleFamily,
    tol: f64,
) -> Vec<Centroid<F>> {
    if prefer_reverse(items) {
        let mut out = klimit_merge_ordered(items.iter().rev(), d, family, tol);
        out.reverse();
        out
    } else {
        klimit_merge(items, d, family, tol)
    }
}

fn klimit_merge_ordered<'a, F, I>(
    items: I,
    d: f64,
    family: ScaleFamily,
    tol: f64,
) -> Vec<Centroid<F>>
where
    F: FloatLike + FloatCore + 'a,
    I: Iterator<Item = &'a Centroid<F>> + Clone + ExactSizeIterator,
{
    if items.len() == 0 {
        return Vec::new();
    }
    let total_w: f64 = items.clone().map(|c| c.weight_f64()).sum();
    // n in the canonical Dunning K2 normalizer is the centroid count of the
    // current merge pass; for non-K2Norm families this argument is ignored.
    let n = items.len() as f64;

    // K2Norm increases with q only when its normalizer is positive. At a
    // sufficiently large delta relative to the normalized input count, a
    // nonpositive normalizer would reverse the scale and accept every merge.
    // There is ample capacity in this regime: retain the existing means,
    // weights and Atomic/Mixed kinds rather than discard information.
    // Positive infinity is deliberately NOT caught: a tiny positive delta
    // can overflow n/d and keeps its existing aggressive-compression path.
    if family == ScaleFamily::K2Norm && (n / d).ln().mul_add(4.0, 24.0) <= 0.0 {
        return items.copied().collect();
    }

    // C = mass of completed clusters already emitted.
    let mut c_acc = 0.0_f64; // running capacity accumulator

    let mut clusters: Vec<Centroid<F>> = Vec::with_capacity(items.len());
    let mut acc = WeightedStats::default();

    let mut q_l = c_acc / total_w;
    let mut k_left = q_to_k(q_l, d, family, n);

    for c in items {
        let w_next = c.weight_f64();
        let q_r = (c_acc + acc.w_sum + w_next) / total_w;
        let k_right = q_to_k(q_r, d, family, n);

        if (k_right - k_left) <= 1.0 + tol {
            acc.add(c);
        } else {
            flush_cluster::<F>(&mut clusters, &mut acc);
            if let Some(last) = clusters.last() {
                c_acc += last.weight_f64();
            }
            q_l = c_acc / total_w;
            k_left = q_to_k(q_l, d, family, n);

            acc.add(c);
        }
    }
    flush_cluster::<F>(&mut clusters, &mut acc);

    clusters
}

/// **(4) Cap** — reduce to at most `core_cap` by running a second k-limit pass.
///
/// We binary-search `d'` in `(0, d]` and choose the largest value that yields `<= core_cap`
/// centroids under the same k-limit merge rule used by Stage 3.
///
/// If plateau behavior prevents strict cap convergence, we tighten `d'` a few times and
/// finally fall back to equal-weight bucketization as a last-resort safety net.
fn cap_core<F: FloatLike + FloatCore>(
    core: Vec<Centroid<F>>,
    core_cap: usize,
    d: f64,
    family: ScaleFamily,
) -> Vec<Centroid<F>> {
    if core.len() <= core_cap {
        return core;
    }
    if core_cap == 0 {
        return Vec::new();
    }
    cap_with_second_klimit(&core, core_cap, d, family)
}

/// Stage-4 cap strategy: second k-limit merge with tuned `d'`.
fn cap_with_second_klimit<F: FloatLike + FloatCore>(
    core: &[Centroid<F>],
    core_cap: usize,
    d: f64,
    family: ScaleFamily,
) -> Vec<Centroid<F>> {
    debug_assert!(core_cap > 0);
    if core.len() <= core_cap {
        return core.to_vec();
    }

    // Find the largest d' that satisfies the cap to preserve as much detail as possible.
    let mut lo = CAP_SEARCH_LO;
    let mut hi = d.max(CAP_SEARCH_LO);
    let mut best = lo;
    let mut found = false;

    for _ in 0..CAP_SEARCH_ITERS {
        let mid = 0.5 * (lo + hi);
        let len_mid = klimit_merge(core, mid, family, KLIMIT_TOL).len();
        if len_mid <= core_cap {
            found = true;
            best = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mut out = klimit_merge(
        core,
        if found { best } else { CAP_SEARCH_LO },
        family,
        KLIMIT_TOL,
    );

    // Guard against rare non-monotone/plateau effects.
    let mut guard = 0usize;
    let mut tuned = if found { best } else { CAP_SEARCH_LO };
    while out.len() > core_cap && guard < CAP_FALLBACK_ITERS {
        tuned *= 0.95;
        out = klimit_merge(core, tuned, family, KLIMIT_TOL);
        guard += 1;
    }

    if out.len() > core_cap {
        return bucketize_equal_weight(core, core_cap);
    }

    out
}

/// **(4) Cap** — order-preserving equal-weight bucketization into `buckets` groups.
///
/// - Emits up to `buckets` outputs; a tiny trailing remainder may be skipped (see [`cap_core`]).
/// - Multi-item buckets are **mixed**; a one-item bucket retains its input kind.
fn bucketize_equal_weight<F: FloatLike + FloatCore>(
    cs: &[Centroid<F>],
    buckets: usize,
) -> Vec<Centroid<F>> {
    debug_assert!(buckets > 0);
    if cs.is_empty() {
        return Vec::new();
    }
    if buckets == 1 {
        return vec![weighted_collapse(cs)];
    }

    let total_w: f64 = cs.iter().map(|c| c.weight_f64()).sum();
    if total_w <= 0.0 {
        return cs.iter().take(buckets).copied().collect();
    }

    let target = total_w / buckets as f64;
    let mut out = Vec::with_capacity(buckets);
    let mut acc = WeightedStats::default();

    for c in cs {
        acc.add(c);
        if acc.w_sum >= target && acc.w_sum > 0.0 {
            out.push(acc.centroid::<F>());
            acc.clear();
            if out.len() == buckets {
                break;
            }
        }
    }
    if !acc.is_empty() && out.len() < buckets {
        out.push(acc.centroid::<F>());
    }
    out
}

/// **(5) Assemble** — concatenate `left`, `core`, `right` without element mutation.
fn assemble_with_edges<F: FloatLike + FloatCore>(
    left: &[Centroid<F>],
    core: Vec<Centroid<F>>,
    right: &[Centroid<F>],
) -> Vec<Centroid<F>> {
    let mut v = Vec::with_capacity(left.len() + core.len() + right.len());
    v.extend_from_slice(left);
    v.extend(core);
    v.extend_from_slice(right);
    v
}

/* =============================================================================
 * Edge helpers (used by Stage 2)
 * ============================================================================= */

#[derive(Debug, Clone, Copy)]
struct EdgeCaps {
    core_cap: usize,
}

#[derive(Debug)]
struct EdgeSlices<'a, F: FloatLike + FloatCore> {
    left: &'a [Centroid<F>],
    interior: &'a [Centroid<F>],
    right: &'a [Centroid<F>],
    caps: EdgeCaps,
}

/// Count up to `k` consecutive **atomic** centroids on one edge.
///
/// Stops at the first **mixed** centroid. When `from_left=false`, scans from the right.
fn edge_run_len<F: FloatLike + FloatCore>(cs: &[Centroid<F>], k: usize, from_left: bool) -> usize {
    if k == 0 || cs.is_empty() {
        return 0;
    }
    let mut cnt = 0usize;
    if from_left {
        for c in cs {
            if c.is_singleton() && cnt < k {
                cnt += 1;
            } else {
                break;
            }
        }
    } else {
        for c in cs.iter().rev() {
            if c.is_singleton() && cnt < k {
                cnt += 1;
            } else {
                break;
            }
        }
    }
    cnt
}

/* =============================================================================
 * Small helpers
 * ============================================================================= */

/// Emit the current cluster (if any) and reset accumulation (used by **3**).
fn flush_cluster<F: FloatLike + FloatCore>(
    clusters: &mut Vec<Centroid<F>>,
    acc: &mut WeightedStats,
) {
    if acc.is_empty() {
        return;
    }
    clusters.push(acc.centroid::<F>());
    acc.clear();
}

/// Collapse multiple centroids into one mixed cluster; preserve a lone input.
fn weighted_collapse<F: FloatLike + FloatCore>(slice: &[Centroid<F>]) -> Centroid<F> {
    let mut acc = WeightedStats::default();
    for c in slice {
        acc.add(c);
    }
    acc.centroid::<F>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::{singleton_policy::SingletonPolicy, ScaleFamily, TDigest};

    #[test]
    fn nonpositive_k2norm_domain_retains_existing_centroid_kinds_and_bits() {
        macro_rules! assert_retained {
            ($float:ty) => {{
                let items = [
                    Centroid::<$float>::new_mixed_f64(0.125, 0.25),
                    Centroid::<$float>::new_atomic_f64(100.0, 0.5),
                ];
                assert_eq!(
                    klimit_merge(&items, 1200.0, ScaleFamily::K2Norm, 0.0),
                    items.to_vec()
                );
            }};
        }
        assert_retained!(f32);
        assert_retained!(f64);
    }

    #[test]
    fn finite_cluster_average_stays_inside_its_input_mean_interval() {
        let left = 1.0000000000000009_f64;
        let right = 1.000000000000001_f64;
        let weights = [2e100, 3.0000000000000002e100];
        let raw_mean = (left * weights[0] + right * weights[1]) / (weights[0] + weights[1]);
        assert!(
            raw_mean > right,
            "fixture must exercise averaging overshoot"
        );
        for sign in [1.0, -1.0] {
            let mut inputs = [
                Centroid::<f64>::new_atomic_f64(left * sign, weights[0]),
                Centroid::<f64>::new_atomic_f64(right * sign, weights[1]),
            ];
            inputs.sort_by(|a, b| f64::total_cmp(&a.mean_f64(), &b.mean_f64()));
            let mut stats = WeightedStats::default();
            for centroid in &inputs {
                stats.add(centroid);
            }
            assert!(stats.mean() >= inputs[0].mean_f64());
            assert!(stats.mean() <= inputs[1].mean_f64());
            assert_eq!(
                stats.centroid::<f64>().weight_f64(),
                weights.iter().sum::<f64>()
            );
            assert!(!stats.centroid::<f64>().is_atomic());
        }
    }

    #[test]
    fn cluster_interval_bound_does_not_hide_nonfinite_accumulators() {
        for (mean, weight) in [(2.0, 1e308), (0.0, 1e308)] {
            let mut stats = WeightedStats::default();
            for _ in 0..2 {
                stats.add(&Centroid::<f64>::new_atomic_f64(mean, weight));
            }
            assert!(!stats.w_sum.is_finite());
            assert_eq!(stats.mean().is_nan(), (stats.mw_sum / stats.w_sum).is_nan());
            if !stats.mean().is_nan() {
                assert_eq!(stats.mean(), stats.mw_sum / stats.w_sum);
            }
        }
        let mut weighted_sum_overflow = WeightedStats::default();
        for _ in 0..2 {
            weighted_sum_overflow.add(&Centroid::<f64>::new_atomic_f64(1e308, 2.0));
        }
        assert!(weighted_sum_overflow.w_sum.is_finite());
        assert!(weighted_sum_overflow.mean().is_infinite());
    }

    #[test]
    fn all_compressor_paths_preserve_single_input_mean_bits_and_kind() {
        for centroid in [
            Centroid::<f64>::new_atomic_f64(-3.0, 0.1),
            Centroid::<f64>::new_mixed_f64(-3.0, 0.1),
            Centroid::<f64>::new_atomic_f64(-0.0, 0.1),
            Centroid::<f64>::new_mixed_f64(-0.0, 0.1),
        ] {
            let input = [centroid];
            let outputs = [
                klimit_merge(&input, 1000.0, ScaleFamily::K2, KLIMIT_TOL),
                bucketize_equal_weight(&input, 1),
                bucketize_equal_weight(&input, 2),
                vec![weighted_collapse(&input)],
                cap_core(input.to_vec(), 1, 1000.0, ScaleFamily::K2),
            ];
            for output in outputs {
                assert_eq!(output.len(), 1);
                assert_eq!(
                    output[0].mean_f64().to_bits(),
                    centroid.mean_f64().to_bits()
                );
                assert_eq!(
                    output[0].weight_f64().to_bits(),
                    centroid.weight_f64().to_bits()
                );
                assert_eq!(output[0].kind(), centroid.kind());
            }
        }
    }

    fn assert_direction_bits<F: FloatLike + FloatCore>(a: &[Centroid<F>], b: &[Centroid<F>]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.mean_f64().to_bits(), b.mean_f64().to_bits());
            assert_eq!(a.weight_f64().to_bits(), b.weight_f64().to_bits());
            assert_eq!(a.kind(), b.kind());
        }
    }

    fn direction_equal_profile_cases<F: FloatLike + FloatCore>() {
        for n in [0usize, 1, 2, 3, 11, 67] {
            for profile in 0..3 {
                let input: Vec<_> = (0..n)
                    .map(|i| {
                        let weight = match profile {
                            0 => 1.0,
                            1 => 0.125,
                            _ => (i.min(n - 1 - i) % 5 + 1) as f64 / 8.0,
                        };
                        let mean = if i == n / 2 {
                            -0.0
                        } else {
                            (i as f64 - (n / 2) as f64) * 3.0
                        };
                        if i % 4 == 0 {
                            Centroid::<F>::new_mixed_f64(mean, weight)
                        } else {
                            Centroid::<F>::new_atomic_f64(mean, weight)
                        }
                    })
                    .collect();
                assert!(!prefer_reverse(&input));
                for scale in [
                    ScaleFamily::Quad,
                    ScaleFamily::K1,
                    ScaleFamily::K2,
                    ScaleFamily::K2Norm,
                    ScaleFamily::K3,
                ] {
                    for d in [1e-6, 3.0, 17.0, 1000.0] {
                        for tol in [0.0, KLIMIT_TOL] {
                            assert_direction_bits(
                                &directional_klimit_merge(&input, d, scale, tol),
                                &baseline_forward_klimit(&input, d, scale, tol),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn direction_equal_profiles_retain_frozen_forward_bits_f32_and_f64() {
        direction_equal_profile_cases::<f32>();
        direction_equal_profile_cases::<f64>();
    }

    #[test]
    fn direction_first_unequal_mirrored_weight_chooses_lighter_edge() {
        let inputs: Vec<_> = [1.0, 2.0, 1.0, 1.0]
            .iter()
            .enumerate()
            .map(|(i, &w)| Centroid::<f64>::new_atomic_f64(i as f64, w))
            .collect();
        assert!(prefer_reverse(&inputs));
        let reversed: Vec<_> = inputs.iter().rev().copied().collect();
        assert!(!prefer_reverse(&reversed));
        let single = [Centroid::<f64>::new_mixed_f64(-0.0, 0.125)];
        assert!(!prefer_reverse(&single));
        assert_direction_bits(
            &directional_klimit_merge(&single, 10.0, ScaleFamily::K2, KLIMIT_TOL),
            &single,
        );
    }

    fn direction_reflection_cases<F: FloatLike + FloatCore>() {
        for magnitude in [1e-20, 1.0, 1e20] {
            let input: Vec<_> = (0..67)
                .map(|i| {
                    let mean = (i as f64 * i as f64 - 500.0) * magnitude;
                    let weight = (67 - i) as f64 / 8.0;
                    if i % 5 == 0 {
                        Centroid::<F>::new_mixed_f64(mean, weight)
                    } else {
                        Centroid::<F>::new_atomic_f64(mean, weight)
                    }
                })
                .collect();
            let mirror: Vec<_> = input
                .iter()
                .rev()
                .map(|c| {
                    if c.is_atomic() {
                        Centroid::<F>::new_atomic_f64(-c.mean_f64(), c.weight_f64())
                    } else {
                        Centroid::<F>::new_mixed_f64(-c.mean_f64(), c.weight_f64())
                    }
                })
                .collect();
            assert!(prefer_reverse(&input));
            assert!(!prefer_reverse(&mirror));
            for scale in [
                ScaleFamily::Quad,
                ScaleFamily::K1,
                ScaleFamily::K2,
                ScaleFamily::K2Norm,
                ScaleFamily::K3,
            ] {
                for d in [3.0, 17.0, 1000.0] {
                    let output = directional_klimit_merge(&input, d, scale, KLIMIT_TOL);
                    let reflected = directional_klimit_merge(&mirror, d, scale, KLIMIT_TOL);
                    assert_eq!(output.len(), reflected.len());
                    for (a, b) in output.iter().zip(reflected.iter().rev()) {
                        if a.mean_f64() == 0.0 {
                            assert_eq!(b.mean_f64(), 0.0);
                        } else {
                            assert_eq!(a.mean_f64().to_bits(), (-b.mean_f64()).to_bits());
                        }
                        assert_eq!(a.weight_f64().to_bits(), b.weight_f64().to_bits());
                        assert_eq!(a.kind(), b.kind());
                    }
                    assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&output));
                    assert_eq!(
                        output.iter().map(Centroid::weight_f64).sum::<f64>(),
                        input.iter().map(Centroid::weight_f64).sum::<f64>()
                    );
                    // The non-adaptive route still serves delta and Stage4.
                    assert_direction_bits(
                        &klimit_merge(&input, d, scale, 0.0),
                        &baseline_forward_klimit(&input, d, scale, 0.0),
                    );
                }
            }
        }
    }

    #[test]
    fn direction_weighted_reflections_preserve_mass_kind_and_mean_symmetry() {
        direction_reflection_cases::<f32>();
        direction_reflection_cases::<f64>();
    }

    fn direction_policy_cases<F: FloatLike + FloatCore>() {
        let input: Vec<_> = (0..67)
            .map(|i| {
                let mean = i as f64 - 33.0;
                let weight = (67 - i) as f64 / 8.0;
                if i > 2 && i < 64 && i % 5 == 0 {
                    Centroid::<F>::new_mixed_f64(mean, weight)
                } else {
                    Centroid::<F>::new_atomic_f64(mean, weight)
                }
            })
            .collect();
        for scale in [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K2Norm,
            ScaleFamily::K3,
        ] {
            for policy in [
                SingletonPolicy::Off,
                SingletonPolicy::Use,
                SingletonPolicy::UseWithProtectedEdges(2),
            ] {
                let mut td = TDigest::<F>::builder()
                    .max_size(7)
                    .scale(scale)
                    .singleton_policy(policy)
                    .build();
                let output = compress_into(&mut td, 7, input.clone());
                assert_eq!(
                    td.count(),
                    input.iter().map(Centroid::weight_f64).sum::<f64>()
                );
                assert_eq!(
                    td.sum(),
                    input
                        .iter()
                        .map(|c| c.mean_f64() * c.weight_f64())
                        .sum::<f64>()
                );
                assert_eq!(
                    output.iter().map(Centroid::weight_f64).sum::<f64>(),
                    td.count()
                );
                assert_eq!(td.min().to_bits(), input[0].mean_f64().to_bits());
                assert_eq!(
                    td.max().to_bits(),
                    input.last().unwrap().mean_f64().to_bits()
                );
                assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&output));
                match policy {
                    SingletonPolicy::Off => assert!(output.len() <= 7),
                    SingletonPolicy::Use => {
                        assert!(output.len() <= 7);
                        assert_direction_bits(&output[..1], &input[..1]);
                        assert_direction_bits(
                            &output[output.len() - 1..],
                            &input[input.len() - 1..],
                        );
                    }
                    SingletonPolicy::UseWithProtectedEdges(_) => {
                        assert!(output.len() <= 11);
                        assert_direction_bits(&output[..2], &input[..2]);
                        assert_direction_bits(
                            &output[output.len() - 2..],
                            &input[input.len() - 2..],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn direction_all_scales_and_policies_keep_mass_bounds_and_protected_edges() {
        direction_policy_cases::<f32>();
        direction_policy_cases::<f64>();
    }

    // Frozen pre-E18 forward loop: independent control for iteration refactoring.
    fn baseline_forward_klimit<F: FloatLike + FloatCore>(
        items: &[Centroid<F>],
        d: f64,
        family: ScaleFamily,
        tol: f64,
    ) -> Vec<Centroid<F>> {
        if items.is_empty() {
            return Vec::new();
        }
        let total_w: f64 = items.iter().map(|c| c.weight_f64()).sum();
        // n in the canonical Dunning K2 normalizer is the centroid count of the
        // current merge pass; for non-K2Norm families this argument is ignored.
        let n = items.len() as f64;

        // K2Norm increases with q only when its normalizer is positive. At a
        // sufficiently large delta relative to the normalized input count, a
        // nonpositive normalizer would reverse the scale and accept every merge.
        // There is ample capacity in this regime: retain the existing means,
        // weights and Atomic/Mixed kinds rather than discard information.
        // Positive infinity is deliberately NOT caught: a tiny positive delta
        // can overflow n/d and keeps its existing aggressive-compression path.
        if family == ScaleFamily::K2Norm && (n / d).ln().mul_add(4.0, 24.0) <= 0.0 {
            return items.to_vec();
        }

        // C = mass of completed clusters already emitted.
        let mut c_acc = 0.0_f64; // running capacity accumulator

        let mut clusters: Vec<Centroid<F>> = Vec::with_capacity(items.len());
        let mut acc = WeightedStats::default();

        let mut q_l = c_acc / total_w;
        let mut k_left = q_to_k(q_l, d, family, n);

        for c in items {
            let w_next = c.weight_f64();
            let q_r = (c_acc + acc.w_sum + w_next) / total_w;
            let k_right = q_to_k(q_r, d, family, n);

            if (k_right - k_left) <= 1.0 + tol {
                acc.add(c);
            } else {
                flush_cluster::<F>(&mut clusters, &mut acc);
                if let Some(last) = clusters.last() {
                    c_acc += last.weight_f64();
                }
                q_l = c_acc / total_w;
                k_left = q_to_k(q_l, d, family, n);

                acc.add(c);
            }
        }
        flush_cluster::<F>(&mut clusters, &mut acc);

        clusters
    }

    fn direction_nondyadic_controls<F: FloatLike + FloatCore>(magnitude: f64) {
        // Pairwise/reflected control is independent of the new reverse iterator:
        // apply the frozen ascending loop to explicitly negated reflected inputs.
        // The second profile spans400 decimal orders without mean*weight overflow.
        for weights in [
            [0.7, 0.1, 0.2, 0.3, 1.1, 0.9, 0.4, 0.6, 0.8, 1.3, 0.1],
            [
                1e200, 0.1, 1e-200, 1e100, 0.3, 1e-100, 0.9, 1e180, 0.7, 1e-180, 1e-200,
            ],
        ] {
            let means = [
                -9.0, -4.0, -2.0, -1.0, -1e-10, 0.0, 1e-10, 1.0, 2.0, 4.0, 9.0,
            ];
            let input: Vec<_> = means
                .iter()
                .zip(weights)
                .enumerate()
                .map(|(i, (&mean, weight))| {
                    if i % 3 == 0 {
                        Centroid::<F>::new_mixed_f64(mean * magnitude, weight)
                    } else {
                        Centroid::<F>::new_atomic_f64(mean * magnitude, weight)
                    }
                })
                .collect();
            assert!(prefer_reverse(&input));
            let mirror: Vec<_> = input
                .iter()
                .rev()
                .map(|c| {
                    if c.is_atomic() {
                        Centroid::<F>::new_atomic_f64(-c.mean_f64(), c.weight_f64())
                    } else {
                        Centroid::<F>::new_mixed_f64(-c.mean_f64(), c.weight_f64())
                    }
                })
                .collect();
            for scale in [
                ScaleFamily::Quad,
                ScaleFamily::K1,
                ScaleFamily::K2,
                ScaleFamily::K2Norm,
                ScaleFamily::K3,
            ] {
                for d in [3.0, 7.0, 17.0, 1000.0] {
                    let output = directional_klimit_merge(&input, d, scale, KLIMIT_TOL);
                    let control = baseline_forward_klimit(&mirror, d, scale, KLIMIT_TOL);
                    assert_eq!(output.len(), control.len());
                    for (actual, reflected) in output.iter().zip(control.iter().rev()) {
                        if actual.mean_f64() == 0.0 {
                            assert_eq!(reflected.mean_f64(), 0.0);
                        } else {
                            assert_eq!(
                                actual.mean_f64().to_bits(),
                                (-reflected.mean_f64()).to_bits()
                            );
                        }
                        assert_eq!(
                            actual.weight_f64().to_bits(),
                            reflected.weight_f64().to_bits()
                        );
                        assert_eq!(actual.kind(), reflected.kind());
                        assert!(actual.mean_f64().is_finite());
                    }
                    assert!(crate::tdigest::centroids::is_sorted_strict_by_mean(&output));
                    let forward_mass: f64 = input.iter().map(Centroid::weight_f64).sum();
                    let output_mass: f64 = output.iter().map(Centroid::weight_f64).sum();
                    // Positive-sum rounding bound depends on this mass scale and
                    // term count; it is not a quantile/rank accuracy tolerance.
                    let bound = forward_mass * f64::EPSILON * (4 * input.len()) as f64;
                    assert!((output_mass - forward_mass).abs() <= bound);
                    assert_direction_bits(
                        &klimit_merge(&input, d, scale, 0.0),
                        &baseline_forward_klimit(&input, d, scale, 0.0),
                    );
                }
                let mut td = TDigest::<F>::builder()
                    .max_size(7)
                    .scale(scale)
                    .singleton_policy(SingletonPolicy::Off)
                    .build();
                let _ = compress_into(&mut td, 7, input.clone());
                assert_eq!(
                    td.count().to_bits(),
                    input
                        .iter()
                        .map(Centroid::weight_f64)
                        .sum::<f64>()
                        .to_bits()
                );
                assert_eq!(
                    td.sum().to_bits(),
                    input
                        .iter()
                        .map(|c| c.mean_f64() * c.weight_f64())
                        .sum::<f64>()
                        .to_bits()
                );
            }
        }
    }

    #[test]
    fn direction_nondyadic_and_extreme_weights_match_reflected_frozen_control() {
        direction_nondyadic_controls::<f32>(1e20);
        direction_nondyadic_controls::<f64>(1e100);
    }

    // ---- tiny helpers ----
    fn c<F: FloatLike + FloatCore>(m: f64, w: f64, singleton: bool) -> Centroid<F> {
        if singleton {
            Centroid::<F>::new_singleton_f64(m, w)
        } else {
            Centroid::<F>::new_mixed_f64(m, w)
        }
    }
    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs() + b.abs())
    }

    type Fp = f64; // run these tests in f64 storage; f32 is covered elsewhere

    // ---------- streaming/coalescing pass ----------

    #[test]
    fn coalescing_equal_means_preserves_mixed_provenance() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(100)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.0, 2.5, false),
            c::<Fp>(2.0, 3.0, true),
        ];

        let out = super::compress_into(&mut td, 100, input);
        assert_eq!(out.len(), 2);
        assert!(
            !out[0].is_singleton(),
            "a mixed centroid cannot become an exact point mass"
        );
        assert!(approx(out[0].mean_f64(), 1.0, 1e-12));
        assert!(approx(out[0].weight_f64(), 3.5, 1e-12));
        assert!(out[1].is_singleton());
        assert!(approx(out[1].mean_f64(), 2.0, 1e-12));
        assert!(approx(out[1].weight_f64(), 3.0, 1e-12));
    }

    #[test]
    #[should_panic(expected = "compress_into requires non-decreasing means")]
    fn compress_into_panics_on_unsorted_means() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        // `normalize_stream` should panic if means are decreasing.
        let input = vec![c::<Fp>(1.0, 1.0, true), c::<Fp>(0.9, 1.0, true)];
        let _ = super::compress_into(&mut td, 10, input);
    }

    // ---------- bucketizer ----------

    #[test]
    fn bucketize_equal_weight_preserves_total_weight_and_order() {
        let cs = vec![
            c::<Fp>(0.0, 1.0, false),
            c::<Fp>(2.0, 1.0, false),
            c::<Fp>(4.0, 1.0, false),
        ];
        let out = super::bucketize_equal_weight(&cs, 2);
        assert_eq!(out.len(), 2);

        let w_in: f64 = cs.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        assert!(
            out[0].mean_f64() <= out[1].mean_f64(),
            "means remain ordered"
        );
    }

    #[test]
    fn bucketize_equal_weight_single_bucket_collapses_to_mixed() {
        let cs = vec![c::<Fp>(1.0, 2.0, false), c::<Fp>(3.0, 2.0, false)];
        let out = super::bucketize_equal_weight(&cs, 1);
        assert_eq!(out.len(), 1);
        assert!(approx(
            out[0].mean_f64(),
            (1.0 * 2.0 + 3.0 * 2.0) / 4.0,
            1e-12
        ));
        assert!(approx(out[0].weight_f64(), 4.0, 1e-12));
        assert!(!out[0].is_singleton(), "collapse yields mixed centroid");
    }

    #[test]
    fn cap_core_second_klimit_respects_cap_and_preserves_weight_order() {
        let core: Vec<_> = (0..200).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::cap_core(core.clone(), 32, 1000.0, ScaleFamily::K2);

        assert!(out.len() <= 32, "cap must be respected");

        let w_in: f64 = core.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        for w in out.windows(2) {
            assert!(w[0].mean_f64() <= w[1].mean_f64(), "means remain ordered");
        }
    }

    // ---------- k-limit merge core ----------

    #[test]
    fn klimit_merge_preserves_weight_order_and_flags() {
        let items = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(1.1, 1.0, true),
            c::<Fp>(3.0, 2.0, false),
        ];
        let out = super::klimit_merge(&items, 10.0, ScaleFamily::K2, super::KLIMIT_TOL);
        assert!(!out.is_empty());

        let w_in: f64 = items.iter().map(|x| x.weight_f64()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        for w in out.windows(2) {
            assert!(w[0].mean_f64() <= w[1].mean_f64(), "means non-decreasing");
        }

        if out.len() == 2 {
            assert!(out[0].is_singleton(), "single-item cluster remains atomic");
            assert!(!out[1].is_singleton(), "multi-item cluster must be mixed");
        }
    }

    #[test]
    fn delta_merge_matches_old_k2_rule_for_weighted_input() {
        // delta-mode merges via klimit_merge with K2Norm + strict (tol=0). This
        // must reproduce the old tdigest-rs `compute()` merge byte-for-byte:
        // q_limit derived from the canonical Dunning K2 (paper eq 8), with
        // n = len(items) and q = cumulative weights / total weight.
        let items = vec![
            c::<Fp>(0.0, 100.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
            c::<Fp>(3.0, 1.0, true),
            c::<Fp>(4.0, 1.0, true),
        ];
        let delta = 20.0;
        let out = super::klimit_merge(&items, delta, ScaleFamily::K2Norm, 0.0);

        // Hand-roll the old tdigest-rs q_limit loop using the same K2Norm
        // formula via q_to_k, so the reference computation goes through the
        // single source of truth.
        let total_w: f64 = items.iter().map(|x| x.weight_f64()).sum();
        let n = items.len() as f64;
        let q_limit_at = |q0: f64| -> f64 {
            // q_limit = inv(k(q0) + 1). We compute it numerically by solving
            // q_to_k(q, delta, K2Norm, n) = q_to_k(q0, ..) + 1 via bisection
            // — accurate enough for the equality assertion below.
            let target = crate::tdigest::scale::q_to_k(q0, delta, ScaleFamily::K2Norm, n) + 1.0;
            let mut lo = q0;
            let mut hi = 1.0 - 1e-15;
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                let k = crate::tdigest::scale::q_to_k(mid, delta, ScaleFamily::K2Norm, n);
                if k < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };

        let mut count = 0usize;
        let mut cumulative = 0.0_f64;
        let mut sigma = items[0].weight_f64();
        let mut q_limit = q_limit_at(0.0);
        for c in items.iter().skip(1) {
            let w = c.weight_f64();
            let q = (cumulative + sigma + w) / total_w;
            if q <= q_limit {
                sigma += w;
            } else {
                count += 1;
                cumulative += sigma;
                q_limit = q_limit_at(cumulative / total_w);
                sigma = w;
            }
        }
        count += 1;

        assert_eq!(out.len(), count);
        let w_out: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(total_w, w_out, 1e-12), "weight preserved");
    }

    #[test]
    fn delta_mode_bypasses_off_fast_path_and_runs_merge() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(100)
            .scale(ScaleFamily::K2Norm)
            .singleton_policy(SingletonPolicy::Off)
            .delta(20.0)
            .build();

        let input = vec![
            c::<Fp>(0.0, 100.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
            c::<Fp>(3.0, 1.0, true),
            c::<Fp>(4.0, 1.0, true),
        ];
        let out = super::compress_into(&mut td, 100, input);
        assert_eq!(
            out.len(),
            4,
            "legacy mode must still merge when len <= max_size"
        );
    }

    // ---------- edge run length ----------

    #[test]
    fn edge_run_len_counts_atomic_only_on_requested_side() {
        let cs = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(0.1, 1.0, true),
            c::<Fp>(0.2, 1.0, false),
            c::<Fp>(0.3, 1.0, true),
            c::<Fp>(0.4, 1.0, true),
        ];
        assert_eq!(
            super::edge_run_len(&cs, 3, true),
            2,
            "left run stops at first mixed"
        );
        assert_eq!(super::edge_run_len(&cs, 1, true), 1);
        assert_eq!(
            super::edge_run_len(&cs, 2, false),
            2,
            "right run counts trailing atomic"
        );
        assert_eq!(super::edge_run_len(&cs, 10, false), 2);
    }

    // ---------- compress_into policies ----------

    #[test]
    fn policy_off_passthrough_under_capacity() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(0.0, 1.0, true),
            c::<Fp>(1.0, 1.0, true),
            c::<Fp>(2.0, 1.0, true),
        ];
        let out = super::compress_into(&mut td, 10, input.clone());
        assert_eq!(out.len(), 3);
        for (i, o) in out.iter().enumerate() {
            assert!(o.is_singleton());
            assert!(approx(o.mean_f64(), i as f64, 1e-12));
        }
    }

    #[test]
    fn policy_off_respects_max_size_via_bucketize() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(3)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input: Vec<_> = (0..10).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);
        assert!(
            out.len() <= 3,
            "must not exceed max_size (got {})",
            out.len()
        );
        let total_w: f64 = out.iter().map(|x| x.weight_f64()).sum();
        assert!(approx(total_w, 10.0, 1e-12), "weight preserved");
    }

    #[test]
    fn policy_use_protects_one_edge_each_and_caps_total() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(3) // total cap in this policy (edges included)
            .singleton_policy(SingletonPolicy::Use)
            .build();

        let input: Vec<_> = (0..8).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);

        assert!(out.len() <= 3, "Use policy keeps total <= max_size");
        assert!(
            approx(out.first().unwrap().mean_f64(), 0.0, 1e-12),
            "leftmost preserved"
        );
        assert!(
            approx(out.last().unwrap().mean_f64(), 7.0, 1e-12),
            "rightmost preserved"
        );
        if out.len() == 3 {
            assert!(
                (out[1].mean_f64() > 0.0) && (out[1].mean_f64() < 7.0),
                "interior stays interior"
            );
        }
    }

    #[test]
    fn policy_use_with_protected_edges_keeps_k_atomic_edges_extra() {
        // Protect k=2 atomic at each side; core cap = max_size independently of edges.
        let mut td = TDigest::<Fp>::builder()
            .max_size(2) // core capacity only
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build();

        let mut input: Vec<_> = (0..12).map(|i| c::<Fp>(i as f64, 1.0, true)).collect();
        input[5] = c::<Fp>(5.0, 2.0, false);

        let out = super::compress_into(&mut td, 2, input);

        assert!(out.len() >= 4 && out.len() <= 6);
        assert!(out[0].is_singleton() && approx(out[0].mean_f64(), 0.0, 1e-12));
        assert!(out[1].is_singleton() && approx(out[1].mean_f64(), 1.0, 1e-12));
        assert!(
            out[out.len() - 1].is_singleton() && approx(out[out.len() - 1].mean_f64(), 11.0, 1e-12)
        );
        assert!(
            out[out.len() - 2].is_singleton() && approx(out[out.len() - 2].mean_f64(), 10.0, 1e-12)
        );

        let core_count = out.len() - 4;
        assert!(core_count <= 2, "core must respect max_size");
    }

    // ---------- metadata on result ----------

    #[test]
    fn result_metadata_count_sum_min_max_are_set() {
        let mut td = TDigest::<Fp>::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![
            c::<Fp>(-3.0, 1.0, true),
            c::<Fp>(-1.0, 2.0, true),
            c::<Fp>(4.0, 3.0, true),
        ];
        let w_sum: f64 = input.iter().map(|x| x.weight_f64()).sum();
        let mw_sum: f64 = input.iter().map(|x| x.weight_f64() * x.mean_f64()).sum();

        let out = super::compress_into(&mut td, 10, input);
        assert!(!out.is_empty());
        assert!(approx(td.count(), w_sum, 1e-12));
        assert!(approx(td.sum(), mw_sum, 1e-12));
        assert!(approx(td.min(), -3.0, 1e-12));
        assert!(approx(td.max(), 4.0, 1e-12));
    }
}
