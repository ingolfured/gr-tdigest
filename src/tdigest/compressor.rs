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
