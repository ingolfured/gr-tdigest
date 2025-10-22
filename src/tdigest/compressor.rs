//! Second-layer compressor for TDigest centroids (singleton propagation on).
//!
//! Contract:
//! - INPUT MUST BE NON-DECREASING BY MEAN. We coalesce adjacent equal means in one pass.
//!   If we see a decrease (mean[i] < mean[i-1]) we panic.
//! - Keep first/last as dedicated edges when shrinking.
//! - Interior merged by k-limit: Δk(q_l→q_r) ≤ 1 + tol with family q→k.
//! - If interior still exceeds capacity, equal-weight bucket the interior only (no padding).
//! - Preserve strict mean ordering and total weight.
//! - IMPORTANT: Maintain Centroid::is_singleton() as a *data* flag:
//!     * true  => atomic ECDF jump (raw singleton or same-mean pile only)
//!     * false => mixed cluster (spanned >1 distinct means or re-bucketed)

// core
use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::scale::{q_to_k, ScaleFamily};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::TDigest;

/* ---------- small helpers ---------- */

const KLIMIT_TOL: f64 = 1e-12;

/// Collapse a slice to a single **mixed** centroid (used when we knowingly mix).
#[inline]
fn weighted_collapse_mixed(slice: &[Centroid]) -> Centroid {
    let (mut w_sum, mut mw_sum) = (0.0_f64, 0.0_f64);
    for c in slice {
        let w = c.weight();
        w_sum += w;
        mw_sum += w * c.mean();
    }
    let mean = if w_sum > 0.0 { mw_sum / w_sum } else { 0.0 };
    Centroid::new_mixed(mean, w_sum)
}

/// Equal-weight bucketing of the already-merged interior clusters; returns ≤ `buckets`.
/// Buckets are **mixed** by definition (we re-slice mass to hit capacity).
#[inline]
fn bucketize_equal_weight(interior: &[Centroid], buckets: usize) -> Vec<Centroid> {
    debug_assert!(buckets > 0);
    if interior.is_empty() {
        return Vec::new();
    }
    if buckets == 1 {
        return vec![weighted_collapse_mixed(interior)];
    }

    let total_w: f64 = interior.iter().map(|c| c.weight()).sum();
    if total_w <= 0.0 {
        // Degenerate weights: copy through up to capacity, but mark as mixed
        return interior
            .iter()
            .take(buckets)
            .map(|c| Centroid::new_mixed(c.mean(), c.weight()))
            .collect();
    }

    let target = total_w / buckets as f64;
    let mut acc_w = 0.0_f64;
    let mut acc_mw = 0.0_f64;
    let mut out = Vec::with_capacity(buckets);
    let mut emitted = 0usize;

    for c in interior {
        let w = c.weight();
        acc_w += w;
        acc_mw += w * c.mean();

        if acc_w >= target && acc_w > 0.0 {
            out.push(Centroid::new_mixed(acc_mw / acc_w, acc_w));
            emitted += 1;
            acc_w = 0.0;
            acc_mw = 0.0;

            if emitted == buckets {
                break;
            }
        }
    }
    if acc_w > 0.0 && out.len() < buckets {
        out.push(Centroid::new_mixed(acc_mw / acc_w, acc_w));
    }

    out
}

/* ---------- public API ---------- */

/// Compress centroids into at most `max_size` using k-limit on the interior.
/// We *always* track/propagate the `singleton` bit:
/// - same-mean coalescing → singleton pile (true)
/// - span >1 mean or bucketized → mixed (false)
pub(crate) fn compress_into<I>(
    result: &mut TDigest,
    max_size: usize,
    items: I,
) -> Vec<Centroid>
where
    I: IntoIterator<Item = Centroid>,
{
    // We don't rely on the policy here other than to assert Off is allowed.
    debug_assert!(
        matches!(result.singleton_policy(), SingletonPolicy::Off)
            || matches!(result.singleton_policy(), SingletonPolicy::Use)
            || matches!(result.singleton_policy(), SingletonPolicy::UseWithProtectedEdges(_)),
        "Unexpected singleton policy"
    );

    // (A) Single pass: verify non-decreasing order AND coalesce adjacent equal means.
    //     Equal-mean runs become a single *singleton pile* (true).
    let mut out: Vec<Centroid> = Vec::new();
    let mut prev_mean = f64::NEG_INFINITY;

    for c in items {
        let m = c.mean();
        let w = c.weight();

        // order check
        if m < prev_mean {
            panic!(
                "compress_into requires non-decreasing means; saw {} after {}",
                m, prev_mean
            );
        }

        if let Some(last) = out.last_mut() {
            // exact equality ⇒ same-mean pile stays atomic (singleton=true)
            if last.mean() == m {
                let w_new = last.weight() + w;
                *last = Centroid::new_singleton(m, w_new);
                prev_mean = m;
                continue;
            }
        }
        out.push(c); // preserve the incoming centroid's singleton bit as-is
        prev_mean = m;
    }

    // (B) Fast paths
    let n = out.len();
    if n == 0 {
        return out;
    }
    if n <= max_size {
        debug_assert!(is_sorted_strict_by_mean(&out));
        return out;
    }
    if max_size == 0 {
        return Vec::new();
    }
    if max_size == 1 {
        // If we already have exactly one centroid, keep it (flag preserved).
        if out.len() == 1 {
            return out;
        }
        // Otherwise we collapse across many means ⇒ definitely mixed.
        return vec![weighted_collapse_mixed(&out)];
    }
    if max_size == 2 {
        // keep edge means; fold interior weight into edges ⇒ edges become MIXED
        let left = &out[0];
        let right = &out[n - 1];
        let interior_w: f64 = out[1..n - 1].iter().map(|c| c.weight()).sum();
        let w_left = left.weight() + interior_w * 0.5;
        let w_right = right.weight() + (interior_w - interior_w * 0.5);
        return vec![
            Centroid::new_mixed(left.mean(), w_left),
            Centroid::new_mixed(right.mean(), w_right),
        ];
    }

    // (C) Dedicated edges + k-limit interior
    let left_edge = out[0];      // keep original singleton bit
    let right_edge = out[n - 1]; // keep original singleton bit
    let interior = &out[1..n - 1];

    if interior.is_empty() {
        return vec![left_edge, right_edge];
    }

    // scale family mapping
    let family: ScaleFamily = result.scale();
    let d: f64 = result.max_size() as f64; // denominator for k-space

    // total weight across all centroids (for q)
    let total_w: f64 = left_edge.weight()
        + right_edge.weight()
        + interior.iter().map(|c| c.weight()).sum::<f64>();

    // k-limit forward pass over interior,
    // tracking whether the *current* cluster is atomic (singleton pile) or mixed.
    let mut clusters: Vec<Centroid> = Vec::with_capacity(interior.len());

    let mut C = left_edge.weight(); // consumed weight left of current interior cluster
    let mut sigma_w = 0.0_f64;
    let mut sigma_mw = 0.0_f64;
    let mut sigma_count: usize = 0;
    let mut sigma_singleton: bool = true; // will be set on first add

    let mut q_l = C / total_w;
    let mut k_left = q_to_k(q_l, d, family);

    for c in interior {
        let w_next = c.weight();
        let q_r = (C + sigma_w + w_next) / total_w;
        let k_right = q_to_k(q_r, d, family);

        if (k_right - k_left) <= 1.0 + KLIMIT_TOL {
            // merge into current cluster
            sigma_w += w_next;
            sigma_mw += w_next * c.mean();

            // singleton bookkeeping: first member takes its flag; adding a second member ⇒ mixed
            if sigma_count == 0 {
                sigma_singleton = c.is_singleton();
            } else {
                sigma_singleton = false; // span of >1 mean ⇒ mixed
            }
            sigma_count += 1;
        } else {
            // flush current cluster if any
            if sigma_w > 0.0 {
                let mean = sigma_mw / sigma_w;
                let out_c = if sigma_singleton && sigma_count == 1 {
                    // still atomic (single centroid carried through)
                    Centroid::new_singleton(mean, sigma_w)
                } else {
                    Centroid::new_mixed(mean, sigma_w)
                };
                clusters.push(out_c);
                C += sigma_w;
                sigma_w = 0.0;
                sigma_mw = 0.0;
                sigma_count = 0;
                sigma_singleton = true;
            }
            // start new with current
            sigma_w = w_next;
            sigma_mw = w_next * c.mean();
            sigma_count = 1;
            sigma_singleton = c.is_singleton();

            q_l = C / total_w;
            k_left = q_to_k(q_l, d, family);
        }
    }
    if sigma_w > 0.0 {
        let mean = sigma_mw / sigma_w;
        let out_c = if sigma_singleton && sigma_count == 1 {
            Centroid::new_singleton(mean, sigma_w)
        } else {
            Centroid::new_mixed(mean, sigma_w)
        };
        clusters.push(out_c);
    }

    // (D) enforce interior capacity (never pad)
    let interior_budget = max_size - 2;
    let interior_final = if clusters.len() <= interior_budget {
        clusters
    } else {
        // Re-bucketed clusters are *mixed* by construction.
        bucketize_equal_weight(&clusters, interior_budget)
    };

    // (E) assemble
    let mut compressed = Vec::with_capacity(2 + interior_final.len());
    compressed.push(left_edge);
    compressed.extend(interior_final.into_iter());
    compressed.push(right_edge);

    debug_assert!(is_sorted_strict_by_mean(&compressed));
    #[cfg(debug_assertions)]
    {
        let w_in: f64 = out.iter().map(|c| c.weight()).sum();
        let w_out: f64 = compressed.iter().map(|c| c.weight()).sum();
        debug_assert!((w_in - w_out).abs() < 1e-12, "total weight changed");
    }

    compressed
}

/* ------------------------------ TESTS ------------------------------ */

#[cfg(test)]
mod behavior_tests {
    use super::*;
    use crate::tdigest::centroids::Centroid;
    use crate::tdigest::singleton_policy::SingletonPolicy;
    use crate::tdigest::TDigest;

    const EPS: f64 = 1e-12;

    // unchanged tests except: the "simplified" one now passes sorted input

    #[test]
    fn max_zero_yields_empty() {
        let items = vec![Centroid::new(0.0, 1.0), Centroid::new(1.0, 1.0)];
        let mut td = TDigest::builder()
            .max_size(0)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let out = compress_into(&mut td, 0, items);
        assert!(out.is_empty());
    }

    #[test]
    fn max_one_collapses_to_weighted_mean() {
        let items = vec![Centroid::new(0.0, 1.0), Centroid::new(2.0, 3.0)];
        let mut td = TDigest::builder()
            .max_size(1)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let out = compress_into(&mut td, 1, items);
        assert_eq!(out.len(), 1);
        assert!((out[0].mean() - 1.5).abs() < EPS);
        assert!((out[0].weight() - 4.0).abs() < EPS);
    }

    #[test]
    fn under_capacity_sorts_and_coalesces_equal_means() {
        // non-decreasing with duplicate: OK (coalesced)
        let items = vec![
            Centroid::new(0.0, 1.0),
            Centroid::new(2.0, 1.0),
            Centroid::new(2.0, 2.0),
        ];
        let mut td = TDigest::builder()
            .max_size(5)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let out = compress_into(&mut td, 5, items);
        assert_eq!(out.len(), 2);
        assert!(out[0].mean() < out[1].mean());
        assert!((out[0].mean() - 0.0).abs() < EPS && (out[0].weight() - 1.0).abs() < EPS);
        assert!((out[1].mean() - 2.0).abs() < EPS && (out[1].weight() - 3.0).abs() < EPS);
    }

    #[test]
    fn interior_bucket_single() {
        // 0,1,2,3, max_size=3 ⇒ keep edges 0 & 3, merge interior {1,2} → (1.5,2)
        let items = vec![
            Centroid::new(0.0, 1.0),
            Centroid::new(1.0, 1.0),
            Centroid::new(2.0, 1.0),
            Centroid::new(3.0, 1.0),
        ];
        let mut td = TDigest::builder()
            .max_size(3)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let out = compress_into(&mut td, 3, items);
        assert_eq!(out.len(), 3);
        assert!(out[0].mean() < out[1].mean() && out[1].mean() < out[2].mean());
        assert!((out.first().unwrap().mean() - 0.0).abs() < EPS);
        assert!((out.last().unwrap().mean() - 3.0).abs() < EPS);
        assert!((out[1].mean() - 1.5).abs() < EPS && (out[1].weight() - 2.0).abs() < EPS);
    }

    #[test]
    fn preserves_sort_and_total_weight_simplified() {
        // now sorted (we panic on decreases)
        let items = vec![
            Centroid::new(1.0, 1.0),
            Centroid::new(2.0, 1.0),
            Centroid::new(3.0, 2.0),
            Centroid::new(4.0, 1.0),
        ];
        let mut td = TDigest::builder()
            .max_size(3)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let total_before: f64 = items.iter().map(|c| c.weight()).sum();
        let out = compress_into(&mut td, 3, items);

        for w in out.windows(2) {
            assert!(w[0].mean() < w[1].mean(), "means not strictly increasing");
        }
        let total_after: f64 = out.iter().map(|c| c.weight()).sum();
        assert!((total_before - total_after).abs() < EPS, "total weight changed");
    }

    #[test]
    fn extremes_are_preserved_with_off_simplified() {
        let items = vec![
            Centroid::new(0.0, 1.0),
            Centroid::new(5.0, 1.0),
            Centroid::new(10.0, 1.0),
        ];
        let mut td = TDigest::builder()
            .max_size(2)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let out = compress_into(&mut td, 2, items.clone());
        assert_eq!(out.len(), 2);
        assert!((out.first().unwrap().mean() - 0.0).abs() < EPS);
        assert!((out.last().unwrap().mean() - 10.0).abs() < EPS);
        assert!(out[0].mean() < out[1].mean());

        let total_before: f64 = items.iter().map(|c| c.weight()).sum();
        let total_after: f64 = out.iter().map(|c| c.weight()).sum();
        assert!((total_before - total_after).abs() < EPS, "total weight changed");
    }

    // --- 5) Big interior cluster: heavy middle, tiny edges (sorted means) ------
    #[test]
    fn big_middle_cluster_respects_klimit_and_preserves_edges() {
        let mut items: Vec<Centroid> = Vec::new();
        // left edge
        items.push(Centroid::new(-5.0, 1.0));
        // light interior up to just below zero
        for m in [-2.0, -1.0, -0.5, -0.25] {
            items.push(Centroid::new(m, 1.0));
        }
        // very heavy centroid exactly at zero (sorted position)
        items.push(Centroid::new(0.0, 1000.0));
        // light interior just above zero
        for m in [0.25, 0.5, 1.0, 2.0] {
            items.push(Centroid::new(m, 1.0));
        }
        // right edge
        items.push(Centroid::new(5.0, 1.0));

        let mut td = TDigest::builder()
            .max_size(7) // interior budget 5
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let total_before: f64 = items.iter().map(|c| c.weight()).sum();
        let out = super::compress_into(&mut td, 7, items);

        // edges preserved
        assert!((out.first().unwrap().mean() + 5.0).abs() < 1e-12);
        assert!((out.last().unwrap().mean() - 5.0).abs() < 1e-12);

        // strictly increasing & weight preserved
        for w in out.windows(2) {
            assert!(w[0].mean() < w[1].mean());
        }
        let total_after: f64 = out.iter().map(|c| c.weight()).sum();
        assert!((total_before - total_after).abs() < 1e-12);

        // expect an interior centroid near 0.0 with most of the mass
        let interior = &out[1..out.len()-1];
        let heavy_near_zero = interior.iter().any(|c| c.weight() >= 900.0 && c.mean().abs() < 1e-2);
        assert!(heavy_near_zero, "expected a heavy interior cluster near 0.0 carrying the mass");
        assert!(out.len() <= 7);
    }


    // --- 6) Astronomical extremes: very small/large means, finite results ----
    //
    // This catches any instability when edges are huge in magnitude.
    // We ensure:
    // - edges kept exactly,
    // - interior compressed to budget,
    // - no NaN/Inf in means/weights,
    // - total weight preserved.
    #[test]
    fn extreme_value_magnitudes_are_stable() {
        let items = vec![
            Centroid::new(-1.0e300, 1.0), // left edge
            Centroid::new(-1.0, 1.0),
            Centroid::new(0.0, 2.0),
            Centroid::new(1.0, 1.0),
            Centroid::new(1.0e300, 1.0),  // right edge
        ];

        let mut td = TDigest::builder()
            .max_size(4) // interior budget = 2
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let total_before: f64 = items.iter().map(|c| c.weight()).sum();
        let out = super::compress_into(&mut td, 4, items);

        // keep exact extreme means
        assert!((out.first().unwrap().mean() + 1.0e300).abs() < f64::INFINITY);
        assert_eq!(out.first().unwrap().mean(), -1.0e300);
        assert_eq!(out.last().unwrap().mean(), 1.0e300);

        // finite means/weights and sorted
        for c in &out {
            assert!(c.mean().is_finite(), "non-finite mean");
            assert!(c.weight().is_finite() && c.weight() > 0.0, "invalid weight");
        }
        for w in out.windows(2) {
            assert!(w[0].mean() < w[1].mean());
        }

        // weight preserved; size within budget (≤ 4)
        let total_after: f64 = out.iter().map(|c| c.weight()).sum();
        assert!((total_before - total_after).abs() < 1e-12, "total weight changed");
        assert!(out.len() <= 4);
    }

    // --- 7) Skewed interior: dust + one giant near zero (sorted means) ---------
    // With interior budget=2, the heavy mass at 0 can be split into 2 interior buckets.
    // We assert that most interior weight stays near 0, edges preserved, and totals match.
    #[test]
    fn skewed_weights_form_compact_center_cluster() {
        let mut items: Vec<Centroid> = Vec::new();
        // edges
        items.push(Centroid::new(-10.0, 1.0));
        // tiny dust up to just below zero
        for m in [-1.0, -0.5, -0.25] {
            items.push(Centroid::new(m, 0.1));
        }
        // giant at zero (sorted position)
        items.push(Centroid::new(0.0, 100.0));
        // tiny dust just above zero
        for m in [0.25, 0.5, 1.0] {
            items.push(Centroid::new(m, 0.1));
        }
        // right edge
        items.push(Centroid::new(10.0, 1.0));

        let mut td = TDigest::builder()
            .max_size(4) // interior budget = 2
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let total_before: f64 = items.iter().map(|c| c.weight()).sum();
        let out = super::compress_into(&mut td, 4, items);

        // edges preserved
        assert_eq!(out.first().unwrap().mean(), -10.0);
        assert_eq!(out.last().unwrap().mean(), 10.0);

        // sorted & weight preserved
        for w in out.windows(2) {
            assert!(w[0].mean() < w[1].mean());
        }
        let total_after: f64 = out.iter().map(|c| c.weight()).sum();
        assert!((total_before - total_after).abs() < 1e-12);

        // interior mass should remain concentrated near 0, even if split across 2 buckets
        let interior = &out[1..out.len()-1];
        assert!(interior.len() <= 2, "interior exceeds budget");

        // collect how much interior weight lies in a tight band around zero
        // (wide enough to catch two buckets straddling zero)
        let band = 0.6;
        let center_w: f64 = interior
            .iter()
            .filter(|c| c.mean().abs() <= band)
            .map(|c| c.weight())
            .sum();

        // we expect essentially the giant's weight to be retained near zero
        assert!(center_w >= 100.0 - 1e-9, "expected ≥100 weight near 0, got {}", center_w);
    }

    #[test]
    fn coalescing_equal_means_preserves_singleton_pile() {
        use crate::tdigest::{ScaleFamily, SingletonPolicy, TDigest};

        // Three 1.0s and two 2.0s — all atomic inputs.
        // With ample capacity, the compressor only coalesces equal means.
        // The result should be two centroids, both marked singleton=true (piles).
        let td = TDigest::builder()
            .max_size(16)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Use)
            .build();

        let vals = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let td = td.merge_sorted(vals);

        let cs = td.centroids();
        assert_eq!(cs.len(), 2, "should coalesce into two piles");

        // First pile: mean 1.0, weight 3, still atomic (singleton pile)
        assert_eq!(cs[0].mean(), 1.0);
        assert!((cs[0].weight() - 3.0).abs() < 1e-12);
        assert!(cs[0].is_singleton(), "same-mean pile must remain singleton");

        // Second pile: mean 2.0, weight 2, still atomic (singleton pile)
        assert_eq!(cs[1].mean(), 2.0);
        assert!((cs[1].weight() - 2.0).abs() < 1e-12);
        assert!(cs[1].is_singleton(), "same-mean pile must remain singleton");
    }

    #[test]
    fn merging_distinct_mean_singletons_yields_mixed() {
        use crate::tdigest::{ScaleFamily, SingletonPolicy, TDigest};

        // Four distinct atomic inputs with capacity 3:
        // keep edges (0 and 3), interior {1,2} gets merged by k-limit.
        // That interior must be mixed (spans multiple means).
        let td = TDigest::builder()
            .max_size(3)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Use)
            .build();

        let td = td.merge_sorted(vec![0.0, 1.0, 2.0, 3.0]);
        let cs = td.centroids();
        assert_eq!(cs.len(), 3);

        // Edges remain atomic singletons
        assert!(cs.first().unwrap().is_singleton());
        assert!(cs.last().unwrap().is_singleton());

        // Interior merged across distinct means ⇒ mixed=false
        let mid = &cs[1];
        assert!(!mid.is_singleton(), "interior across 1 and 2 must be mixed");
        assert!((mid.mean() - 1.5).abs() < 1e-12);
        assert!((mid.weight() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn merging_distinct_mean_singletons_is_mixed_even_with_policy_off() {
        use crate::tdigest::{ScaleFamily, SingletonPolicy, TDigest};

        let td = TDigest::builder()
            .max_size(3)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off) // <- Off on purpose
            .build();

        let td = td.merge_sorted(vec![0.0, 1.0, 2.0, 3.0]);
        let cs = td.centroids();
        assert_eq!(cs.len(), 3);

        // Interior {1,2} must be mixed regardless of policy.
        let mid = &cs[1];
        assert!(!mid.is_singleton(), "interior across 1 and 2 must be mixed (policy Off too)");
        assert!((mid.mean() - 1.5).abs() < 1e-12);
        assert!((mid.weight() - 2.0).abs() < 1e-12);
    }


}
