// ---------- tracing helper ----------
#[allow(unused_macros)]
macro_rules! ttrace {
    ($($arg:tt)*) => {
        if std::env::var("TDIGEST_TRACE").is_ok() {
            eprintln!($($arg)*);
        }
    }
}

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::scale::{q_to_k, ScaleFamily};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::TDigest;

const KLIMIT_TOL: f64 = 1e-12;

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

#[inline]
fn bucketize_equal_weight(cs: &[Centroid], buckets: usize) -> Vec<Centroid> {
    debug_assert!(buckets > 0);
    if cs.is_empty() {
        return Vec::new();
    }
    if buckets == 1 {
        return vec![weighted_collapse_mixed(cs)];
    }

    let total_w: f64 = cs.iter().map(|c| c.weight()).sum();
    if total_w <= 0.0 {
        return cs
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

    for c in cs {
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

#[inline]
fn klimit_merge(items: &[Centroid], d: f64, family: ScaleFamily) -> Vec<Centroid> {
    if items.is_empty() {
        return Vec::new();
    }
    let total_w: f64 = items.iter().map(|c| c.weight()).sum();

    let mut clusters: Vec<Centroid> = Vec::with_capacity(items.len());
    let mut C = 0.0_f64;
    let mut sigma_w = 0.0_f64;
    let mut sigma_mw = 0.0_f64;
    let mut sigma_count: usize = 0;
    let mut sigma_singleton = true;

    let mut q_l = C / total_w;
    let mut k_left = q_to_k(q_l, d, family);

    for c in items {
        let w_next = c.weight();
        let q_r = (C + sigma_w + w_next) / total_w;
        let k_right = q_to_k(q_r, d, family);

        if (k_right - k_left) <= 1.0 + KLIMIT_TOL {
            sigma_w += w_next;
            sigma_mw += w_next * c.mean();
            if sigma_count == 0 {
                sigma_singleton = c.is_singleton();
            } else {
                sigma_singleton = false;
            }
            sigma_count += 1;
        } else {
            if sigma_w > 0.0 {
                let mean = sigma_mw / sigma_w;
                let out_c = if sigma_singleton && sigma_count == 1 {
                    Centroid::new_singleton(mean, sigma_w)
                } else {
                    Centroid::new_mixed(mean, sigma_w)
                };
                clusters.push(out_c);
                C += sigma_w;
            }
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
    clusters
}

#[inline]
fn edge_run_len(cs: &[Centroid], k: usize, from_left: bool) -> usize {
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

/// Full compression honoring edge protection.
/// Contract for UseWithProtectedEdges(k): **core capacity is `max_size`, edges are extra.**
pub(crate) fn compress_into<I>(result: &mut TDigest, max_size: usize, items: I) -> Vec<Centroid>
where
    I: IntoIterator<Item = Centroid>,
{
    debug_assert!(
        matches!(result.singleton_policy(), SingletonPolicy::Off)
            || matches!(result.singleton_policy(), SingletonPolicy::Use)
            || matches!(
                result.singleton_policy(),
                SingletonPolicy::UseWithProtectedEdges(_)
            ),
        "Unexpected singleton policy"
    );

    // A) stream pass
    let mut out: Vec<Centroid> = Vec::new();
    let mut prev_mean = f64::NEG_INFINITY;
    let (mut agg_sum_w, mut agg_sum_mw) = (0.0_f64, 0.0_f64);
    let (mut data_min, mut data_max) = (f64::INFINITY, f64::NEG_INFINITY);

    for c in items {
        let m = c.mean();
        let w = c.weight();
        if m < prev_mean {
            panic!(
                "compress_into requires non-decreasing means; saw {} after {}",
                m, prev_mean
            );
        }
        agg_sum_w += w;
        agg_sum_mw += w * m;
        if data_min == f64::INFINITY {
            data_min = m;
        }
        data_max = m;

        if let Some(last) = out.last_mut() {
            if last.mean() == m {
                let w_new = last.weight() + w;
                *last = Centroid::new_singleton(m, w_new);
                prev_mean = m;
                continue;
            }
        }
        out.push(c);
        prev_mean = m;
    }

    if !out.is_empty() {
        result.set_count(agg_sum_w);
        result.set_sum(agg_sum_mw);
        result.set_min(data_min);
        result.set_max(data_max);
    }

    ttrace!(
        "A) after coalesce: n_in={}, min={}, max={}, total_w={}",
        out.len(),
        data_min,
        data_max,
        agg_sum_w
    );

    // B) trivial cases
    if out.is_empty() {
        return out;
    }
    if max_size == 0 {
        // Off/Use modes interpret this as "no core"; edges may still be added below.
        ttrace!("B) max_size==0 (core capacity zero)");
    }
    if matches!(result.singleton_policy(), SingletonPolicy::Off) && out.len() <= max_size {
        debug_assert!(is_sorted_strict_by_mean(&out));
        ttrace!("B) Off under-capacity passthrough: len={}", out.len());
        return out;
    }
    if matches!(result.singleton_policy(), SingletonPolicy::Off) && max_size == 1 {
        if out.len() == 1 {
            return out;
        }
        let v = vec![weighted_collapse_mixed(&out)];
        ttrace!(
            "B) Off max_size==1 collapse: len_in={}, len_out=1",
            out.len()
        );
        return v;
    }
    if matches!(result.singleton_policy(), SingletonPolicy::Off) && max_size == 2 {
        let v = bucketize_equal_weight(&out, 2);
        ttrace!("B) Off + size=2 bucketize-all: len_out=2");
        return v;
    }

    // C) policies
    let family: ScaleFamily = result.scale();
    let d: f64 = result.max_size() as f64;
    let policy = result.singleton_policy();

    let compressed = match policy {
        // ---------- OFF: total cap = max_size ----------
        SingletonPolicy::Off => {
            let clusters = klimit_merge(&out, d, family);
            let v = if clusters.len() <= max_size {
                clusters
            } else {
                bucketize_equal_weight(&clusters, max_size)
            };
            ttrace!("C-Off) clusters_in={}, clusters_out={}", out.len(), v.len());
            v
        }

        // ---------- USE: total cap = max_size (edges included) ----------
        SingletonPolicy::Use => {
            // Protect exactly 1 at each end (if present).
            let l_prot = (!out.is_empty()) as usize;
            let r_prot = (out.len() >= 2) as usize; // will be 1 if we actually have a right end
            let left = &out[..l_prot];
            let right = if out.len() > l_prot {
                &out[out.len() - r_prot..]
            } else {
                &[]
            };
            let interior = if out.len() > (l_prot + r_prot) {
                &out[l_prot..out.len() - r_prot]
            } else {
                &[]
            };

            let mut core = klimit_merge(interior, d, family);
            let total_cap = max_size;
            let core_cap = total_cap.saturating_sub(l_prot + r_prot);
            if core.len() > core_cap {
                ttrace!("C-Use) shrink core {} -> {}", core.len(), core_cap);
                core = bucketize_equal_weight(&core, core_cap.max(0));
            }

            let mut v = Vec::with_capacity(l_prot + core.len() + r_prot);
            v.extend_from_slice(left);
            v.extend(core);
            v.extend_from_slice(right);
            ttrace!(
                "C-Use) out_len={}, l={}, core={}, r={}",
                v.len(),
                l_prot,
                v.len().saturating_sub(l_prot + r_prot),
                r_prot
            );
            v
        }

        // ---------- USE WITH PROTECTED EDGES (k): core cap = max_size; edges are extra ----------
        SingletonPolicy::UseWithProtectedEdges(k) => {
            let k: usize = k;
            let l_prot = edge_run_len(&out, k, true);
            let r_prot = edge_run_len(&out, k, false);
            // Slice runs and interior
            let n = out.len();
            let left_slice_end = l_prot.min(n);
            let right_slice_start = n.saturating_sub(r_prot);

            let left_protected = &out[..left_slice_end];
            let right_protected = if right_slice_start > left_slice_end {
                &out[right_slice_start..]
            } else {
                &[]
            };
            let interior = if right_slice_start > left_slice_end {
                &out[left_slice_end..right_slice_start]
            } else {
                &[]
            };

            // Core budget is the *full* max_size
            let core_cap = max_size;

            ttrace!(
                "C-UseEdges) k={}, l_prot={}, r_prot={}, core_cap={}, interior_len={}",
                k,
                l_prot,
                r_prot,
                core_cap,
                interior.len()
            );

            // k-limit interior -> core, then cap to core_cap
            let mut core = klimit_merge(interior, d, family);
            ttrace!(
                "   k-limit interior: in={}, out={}",
                interior.len(),
                core.len()
            );

            if core.len() > core_cap {
                ttrace!(
                    "   shrink core {} -> {} via bucketize",
                    core.len(),
                    core_cap
                );
                core = bucketize_equal_weight(&core, core_cap.max(0));
            }

            // **Important**: Protected edges remain intact and **singleton**.
            // We never pour interior mass into edge centroids under this policy.
            let mut v =
                Vec::with_capacity(left_protected.len() + core.len() + right_protected.len());
            v.extend_from_slice(left_protected);
            v.extend(core);
            v.extend_from_slice(right_protected);

            ttrace!(
                "   assemble edges+core: left={}, core={}, right={}, out_len={}",
                left_protected.len(),
                v.len()
                    .saturating_sub(left_protected.len() + right_protected.len()),
                right_protected.len(),
                v.len()
            );
            v
        }
    };

    debug_assert!(is_sorted_strict_by_mean(&compressed));
    #[cfg(debug_assertions)]
    {
        let w_in: f64 = out.iter().map(|c| c.weight()).sum();
        let w_out: f64 = compressed.iter().map(|c| c.weight()).sum();
        debug_assert!((w_in - w_out).abs() < 1e-12, "total weight changed");
        ttrace!(
            "Z) DONE: len_in={}, len_out={}, w_in={}, w_out={}",
            out.len(),
            compressed.len(),
            w_in,
            w_out
        );
    }
    compressed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::{singleton_policy::SingletonPolicy, ScaleFamily, TDigest};

    // ---- tiny helpers ----
    fn c(m: f64, w: f64, singleton: bool) -> Centroid {
        if singleton {
            Centroid::new_singleton(m, w)
        } else {
            Centroid::new_mixed(m, w)
        }
    }
    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs() + b.abs())
    }
    fn cdf1(td: &TDigest, x: f64) -> f64 {
        td.cdf(&[x])[0]
    }

    // ---------- streaming/coalescing pass ----------

    #[test]
    fn coalesces_equal_means_into_singleton_pile() {
        let mut td = TDigest::builder()
            .max_size(100)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        // Same mean → must coalesce to ONE centroid and keep singleton *pile* semantics.
        let input = vec![
            c(1.0, 1.0, true),
            c(1.0, 2.5, false), // mixed but equal-mean; pile should remain "singleton" flag
            c(2.0, 3.0, true),
        ];

        let out = super::compress_into(&mut td, 100, input);
        assert_eq!(out.len(), 2);
        assert!(
            out[0].is_singleton(),
            "equal-mean run collapses to singleton pile"
        );
        assert!(approx(out[0].mean(), 1.0, 1e-12));
        assert!(approx(out[0].weight(), 3.5, 1e-12));
        assert!(out[1].is_singleton());
        assert!(approx(out[1].mean(), 2.0, 1e-12));
        assert!(approx(out[1].weight(), 3.0, 1e-12));
    }

    #[test]
    #[should_panic(expected = "compress_into requires non-decreasing means")]
    fn compress_into_panics_on_unsorted_means() {
        let mut td = TDigest::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();
        let input = vec![c(1.0, 1.0, true), c(0.9, 1.0, true)];
        let _ = super::compress_into(&mut td, 10, input);
    }

    // ---------- bucketizer ----------

    #[test]
    fn bucketize_equal_weight_preserves_total_weight_and_order() {
        let cs = vec![c(0.0, 1.0, false), c(2.0, 1.0, false), c(4.0, 1.0, false)];
        let out = super::bucketize_equal_weight(&cs, 2);
        assert_eq!(out.len(), 2);

        let w_in: f64 = cs.iter().map(|x| x.weight()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        assert!(out[0].mean() <= out[1].mean(), "means remain ordered");
    }

    #[test]
    fn bucketize_equal_weight_single_bucket_collapses_to_mixed() {
        let cs = vec![c(1.0, 2.0, false), c(3.0, 2.0, false)];
        let out = super::bucketize_equal_weight(&cs, 1);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0].mean(), (1.0 * 2.0 + 3.0 * 2.0) / 4.0, 1e-12));
        assert!(approx(out[0].weight(), 4.0, 1e-12));
        assert!(!out[0].is_singleton(), "collapse yields mixed centroid");
    }

    // ---------- k-limit merge core ----------

    #[test]
    fn klimit_merge_preserves_weight_order_and_flags() {
        // Expect first cluster to be single-item → stays singleton;
        // second cluster multi-item → becomes mixed.
        let items = vec![
            c(0.0, 1.0, true),
            c(1.0, 1.0, true),
            c(1.1, 1.0, true),
            c(3.0, 2.0, false),
        ];
        let out = super::klimit_merge(&items, 10.0, ScaleFamily::K2);
        assert!(!out.is_empty());

        let w_in: f64 = items.iter().map(|x| x.weight()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight()).sum();
        assert!(approx(w_in, w_out, 1e-12), "weight preserved");

        for w in out.windows(2) {
            assert!(w[0].mean() <= w[1].mean(), "means non-decreasing");
        }

        if out.len() == 2 {
            assert!(
                out[0].is_singleton(),
                "single-item cluster remains singleton"
            );
            assert!(!out[1].is_singleton(), "multi-item cluster must be mixed");
        }
    }

    // ---------- edge run length ----------

    #[test]
    fn edge_run_len_counts_singletons_only_on_requested_side() {
        let cs = vec![
            c(0.0, 1.0, true),
            c(0.1, 1.0, true),
            c(0.2, 1.0, false),
            c(0.3, 1.0, true),
            c(0.4, 1.0, true),
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
            "right run counts trailing singletons"
        );
        assert_eq!(super::edge_run_len(&cs, 10, false), 2);
    }

    // ---------- compress_into policies ----------

    #[test]
    fn policy_off_passthrough_under_capacity() {
        let mut td = TDigest::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![c(0.0, 1.0, true), c(1.0, 1.0, true), c(2.0, 1.0, true)];
        let out = super::compress_into(&mut td, 10, input.clone());
        assert_eq!(out.len(), 3);
        for (i, o) in out.iter().enumerate() {
            assert!(o.is_singleton());
            assert!(approx(o.mean(), i as f64, 1e-12));
        }
    }

    #[test]
    fn policy_off_respects_max_size_via_bucketize() {
        let mut td = TDigest::builder()
            .max_size(3)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input: Vec<_> = (0..10).map(|i| c(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);
        assert!(
            out.len() <= 3,
            "must not exceed max_size (got {})",
            out.len()
        );
        let total_w: f64 = out.iter().map(|x| x.weight()).sum();
        assert!(approx(total_w, 10.0, 1e-12), "weight preserved");
    }

    #[test]
    fn policy_use_protects_one_edge_each_and_caps_total() {
        let mut td = TDigest::builder()
            .max_size(3) // total cap in this policy (edges included)
            .singleton_policy(SingletonPolicy::Use)
            .build();

        let input: Vec<_> = (0..8).map(|i| c(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 3, input);

        assert!(out.len() <= 3, "Use policy keeps total <= max_size");
        assert!(
            approx(out.first().unwrap().mean(), 0.0, 1e-12),
            "leftmost preserved"
        );
        assert!(
            approx(out.last().unwrap().mean(), 7.0, 1e-12),
            "rightmost preserved"
        );
        if out.len() == 3 {
            assert!(
                (out[1].mean() > 0.0) && (out[1].mean() < 7.0),
                "interior stays interior"
            );
        }
    }

    #[test]
    fn policy_use_with_protected_edges_keeps_k_singletons_edges_extra() {
        // Protect k=2 singletons at each side; core cap = max_size independently of edges.
        let mut td = TDigest::builder()
            .max_size(2) // core capacity only
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build();

        let mut input: Vec<_> = (0..12).map(|i| c(i as f64, 1.0, true)).collect();
        // Interior mixed centroid to break singleton run
        input[5] = c(5.0, 2.0, false);

        let out = super::compress_into(&mut td, 2, input);

        // Expect 2 left + core(<=2) + 2 right; edges are extra
        assert!(out.len() >= 4 && out.len() <= 6);
        assert!(out[0].is_singleton() && approx(out[0].mean(), 0.0, 1e-12));
        assert!(out[1].is_singleton() && approx(out[1].mean(), 1.0, 1e-12));
        assert!(
            out[out.len() - 1].is_singleton() && approx(out[out.len() - 1].mean(), 11.0, 1e-12)
        );
        assert!(
            out[out.len() - 2].is_singleton() && approx(out[out.len() - 2].mean(), 10.0, 1e-12)
        );

        let core_count = out.len() - 4;
        assert!(core_count <= 2, "core must respect max_size");
    }

    // ---------- metadata on result ----------

    #[test]
    fn result_metadata_count_sum_min_max_are_set() {
        let mut td = TDigest::builder()
            .max_size(10)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input = vec![c(-3.0, 1.0, true), c(-1.0, 2.0, true), c(4.0, 3.0, true)];
        let w_sum: f64 = input.iter().map(|x| x.weight()).sum();
        let mw_sum: f64 = input.iter().map(|x| x.weight() * x.mean()).sum();

        let out = super::compress_into(&mut td, 10, input);
        assert!(!out.is_empty());
        assert!(approx(td.count(), w_sum, 1e-12));
        assert!(approx(td.sum(), mw_sum, 1e-12));
        assert!(approx(td.min(), -3.0, 1e-12));
        assert!(approx(td.max(), 4.0, 1e-12));
    }

    // ---------- tiny distribution sanity (via public API) ----------

    #[test]
    fn small_n_cdf_and_quantile_are_sane() {
        let td = TDigest::builder()
            .max_size(8)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let mut xs = vec![-2.0, -1.0, -1.0, 0.0, 0.5, 1.0, 3.0, 9.0];
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = td.merge_sorted(xs.clone());

        // Monotone and bounded CDF at a grid of points
        let grid = [-10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 6.0, 9.0, 10.0];
        let mut prev = -1.0;
        for &x in &grid {
            let p = cdf1(&td, x);
            assert!(
                p.is_finite() && (-1e-12..=1.0 + 1e-12).contains(&p),
                "CDF in [0,1]"
            );
            assert!(p + 1e-12 >= prev, "non-decreasing CDF");
            prev = p;
        }

        // Quantiles within observed range (scalar API assumed)
        for &p in &[0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let q = td.quantile(p);
            assert!(q >= xs[0] - 1e-12 && q <= *xs.last().unwrap() + 1e-12);
        }
    }
    // =========================
    // Additional small tests
    // =========================

    #[test]
    fn extremes_keep_cdf_bounded_and_monotone() {
        let td = TDigest::builder()
            .max_size(10)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        // Large interior + extreme tails
        let mut xs: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        xs[0] = -1e9;
        xs[1] = -30.0;
        xs[98] = 1e-10;
        xs[99] = 1e9;
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td = td.merge_sorted(xs);

        let grid = [-1e12, -1e9, -30.0, 0.0, 1e-10, 1.0, 1e9, 1e12];
        let mut prev = -1.0;
        for &x in &grid {
            let p = cdf1(&td, x);
            assert!(
                p.is_finite() && (-1e-12..=1.0 + 1e-12).contains(&p),
                "CDF in [0,1]"
            );
            assert!(p + 1e-12 >= prev, "non-decreasing CDF");
            prev = p;
        }
    }

    #[test]
    fn quantile_endpoints_match_observed_min_max() {
        let td = TDigest::builder()
            .max_size(16)
            .scale(ScaleFamily::Quad)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let mut xs = vec![3.0, 1.0, 2.0, 7.0, 4.0, 6.0, 5.0, 9.0, -2.0, 11.0];
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = td.merge_sorted(xs.clone());

        let q0 = td.quantile(0.0);
        let q1 = td.quantile(1.0);
        assert!(approx(q0, xs[0], 1e-12), "p=0 returns min");
        assert!(approx(q1, *xs.last().unwrap(), 1e-12), "p=1 returns max");
    }

    #[test]
    fn each_scale_family_basic_monotone_cdf_smoke() {
        let families = [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K3,
        ];
        for &fam in &families {
            let td = TDigest::builder()
                .max_size(32)
                .scale(fam)
                .singleton_policy(SingletonPolicy::Off)
                .build();
            let mut xs: Vec<f64> = (0..50)
                .map(|i| ((i as f64).sin() * 0.7) + (i as f64) * 0.1)
                .collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // clone before merge_sorted so we can still read xs later
            let td = td.merge_sorted(xs.clone());

            let test_x = [
                -10.0,
                xs[0],
                (xs[5] + xs[6]) * 0.5,
                xs[xs.len() / 2],
                xs[xs.len() - 5],
                *xs.last().unwrap(),
                999.0,
            ];

            let mut prev = -1.0;
            for &x in &test_x {
                let p = cdf1(&td, x);
                assert!(
                    p.is_finite() && (-1e-12..=1.0 + 1e-12).contains(&p),
                    "bounded CDF"
                );
                assert!(p + 1e-12 >= prev, "monotone CDF for family {:?}", fam);
                prev = p;
            }
        }
    }

    #[test]
    fn policy_usewithprotectededges_protected_edges_remain_singletons() {
        let mut td = TDigest::builder()
            .max_size(3) // core cap only
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(2))
            .build();

        // 2 singletons on each side qualify as protected; interior has a mixed centroid.
        let mut input: Vec<_> = (0..10).map(|i| c(i as f64, 1.0, true)).collect();
        input[5] = c(5.0, 2.0, false); // break singleton run in the middle

        let out = super::compress_into(&mut td, 3, input);

        // First two and last two must be singletons, unchanged.
        assert!(out.len() >= 4);
        assert!(out[0].is_singleton() && approx(out[0].mean(), 0.0, 1e-12));
        assert!(out[1].is_singleton() && approx(out[1].mean(), 1.0, 1e-12));
        assert!(out[out.len() - 2].is_singleton() && approx(out[out.len() - 2].mean(), 8.0, 1e-12));
        assert!(out[out.len() - 1].is_singleton() && approx(out[out.len() - 1].mean(), 9.0, 1e-12));
    }

    #[test]
    fn policy_usewithprotectededges_k_zero_is_same_core_budget_as_off() {
        // k=0 → no edges protected; acts like Off with the same core cap.
        let mut td = TDigest::builder()
            .max_size(4)
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(0))
            .build();

        let input: Vec<_> = (0..20).map(|i| c(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 4, input);
        assert!(out.len() <= 4, "no protected edges when k=0");
    }

    #[test]
    fn bucketize_two_buckets_is_reasonably_balanced() {
        // Not asserting exact split—just that both buckets get positive mass and totals match.
        let cs: Vec<_> = (0..10).map(|i| c(i as f64, 1.0, false)).collect();
        let out = super::bucketize_equal_weight(&cs, 2);
        assert_eq!(out.len(), 2);
        assert!(out[0].weight() > 0.0 && out[1].weight() > 0.0);
        let w_in: f64 = cs.iter().map(|x| x.weight()).sum();
        let w_out: f64 = out.iter().map(|x| x.weight()).sum();
        assert!(approx(w_in, w_out, 1e-12));
    }

    #[test]
    fn quantiles_stay_within_range_across_percentiles() {
        let td = TDigest::builder()
            .max_size(32)
            .scale(ScaleFamily::K3)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let mut xs: Vec<f64> = (0..200)
            .map(|i| (i as f64).ln_1p() * ((i % 7) as f64 + 1.0))
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = td.merge_sorted(xs.clone());

        for &p in &[0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 0.99, 1.0] {
            let q = td.quantile(p);
            assert!(
                q >= xs[0] - 1e-12 && q <= *xs.last().unwrap() + 1e-12,
                "q(p) in data range"
            );
        }
    }

    #[test]
    fn policy_off_max_size_two_gives_exactly_two() {
        // With many inputs, Off should reduce to exactly 2 via the "size==2" fast path.
        let mut td = TDigest::builder()
            .max_size(2)
            .singleton_policy(SingletonPolicy::Off)
            .build();

        let input: Vec<_> = (0..50).map(|i| c(i as f64, 1.0, true)).collect();
        let out = super::compress_into(&mut td, 2, input);
        assert_eq!(out.len(), 2);
    }
}
