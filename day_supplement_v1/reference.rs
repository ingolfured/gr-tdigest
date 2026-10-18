//! Scratch accuracy-only harness; never times operations. Not yet compiled/run.
use gr_tdigest::tdigest::centroids::Centroid;
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::{ScaleFamily, TDigest};
use serde_json::{json, Value};
use std::mem::size_of;
use tdigest_core::TDigest as Original;

fn random_bits(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn uniform(s: &mut u64) -> f64 {
    (random_bits(s) >> 11) as f64 / (1_u64 << 53) as f64
}
fn normal(s: &mut u64) -> f64 {
    (-2.0 * uniform(s).max(1.0 / (1_u64 << 53) as f64).ln()).sqrt()
        * (std::f64::consts::TAU * uniform(s)).cos()
}
fn generate(kind: &str, n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| match kind {
            "uniform" => uniform(&mut state),
            "normal" => normal(&mut state),
            "exponential" => -(1.0 - uniform(&mut state)).ln(),
            "lognormal" => (3.0 * normal(&mut state)).exp(),
            "pareto" => (1.0 - uniform(&mut state)).powf(-1.0 / 1.5),
            "ties" => (10.0 * uniform(&mut state)).floor(),
            "spike_tail" => {
                if uniform(&mut state) < 0.9 {
                    0.0
                } else {
                    (3.0 * normal(&mut state)).exp()
                }
            }
            "separated_modes" => {
                (if uniform(&mut state) < 0.5 {
                    -10.0
                } else {
                    10.0
                }) + 0.01 * normal(&mut state)
            }
            _ => panic!("unknown distribution"),
        })
        .collect()
}
fn type7(xs: &[f64], q: f64) -> f64 {
    let index = q * (xs.len() - 1) as f64;
    let left = index.floor() as usize;
    xs[left] * (1.0 - index.fract()) + xs[(left + 1).min(xs.len() - 1)] * index.fract()
}
fn empirical_interval(xs: &[f64], x: f64) -> (f64, f64) {
    (
        xs.partition_point(|&v| v < x) as f64 / xs.len() as f64,
        xs.partition_point(|&v| v <= x) as f64 / xs.len() as f64,
    )
}
fn quantile_metrics(xs: &[f64], queries: &[(f64, f64)], estimates: &[f64]) -> Value {
    let iqr = type7(xs, 0.75) - type7(xs, 0.25);
    let rows: Vec<_> = queries
        .iter()
        .zip(estimates)
        .map(|(&(requested, q), &y)| {
            let (lo, hi) = empirical_interval(xs, y);
            let rank_error = if y.is_finite() {
                (lo - q).max(q - hi).max(0.0)
            } else {
                f64::INFINITY
            };
            let truth = type7(xs, q);
            json!([requested, q, y, truth, rank_error, (y - truth).abs()])
        })
        .collect();
    let median = queries.iter().position(|&(_, q)| q == 0.5).unwrap();
    let median_error = (estimates[median] - type7(xs, 0.5)).abs();
    json!({"columns":["requested_q","effective_q","estimate","type7_reference","rank_error","abs_value_error"],
        "rows":rows,"p50_abs_error":median_error,"sample_iqr":iqr,
        "p50_error_over_iqr":if iqr > 0.0 {Some(median_error / iqr)} else {None}})
}
fn query_failures(xs: &[f64], values: &[f64]) -> Vec<String> {
    let mut issues = Vec::new();
    if values
        .iter()
        .any(|&x| !x.is_finite() || x < xs[0] || x > xs[xs.len() - 1])
    {
        issues.push("nonfinite or out-of-sample-bounds quantile".into());
    }
    if values.windows(2).any(|pair| pair[0] > pair[1]) {
        issues.push("quantile order decreased".into());
    }
    issues
}

macro_rules! precision_suite {
    ($run:ident, $F:ty, $variant:ident) => {
        fn $run(kind: &str, n: usize, seed: u64, old_delta: f64, modern_delta: f64, reverse: bool) {
            let values: Vec<$F> = generate(kind, n, seed).into_iter().map(|x| x as $F).collect();
            assert!(values.iter().all(|v| v.is_finite()), "generated nonfinite input");
            let mut sorted: Vec<f64> = values.iter().map(|&x| x as f64).collect();
            sorted.sort_by(f64::total_cmp);
            let mut fingerprint = 0xcbf29ce484222325_u64;
            for &value in &values {
                for byte in (value as f64).to_bits().to_le_bytes() {
                    fingerprint = (fingerprint ^ byte as u64).wrapping_mul(0x100000001b3);
                }
            }
            let mut requested: Vec<f64> = (0..=1000).map(|i| i as f64/1000.0).collect();
            requested.extend([0.0001, 0.9999]);
            requested.sort_by(f64::total_cmp);
            requested.dedup();
            // Both implementations receive the original API's representable query.
            let queries: Vec<_> = requested.iter().map(|&q| (q, q as $F as f64)).collect();
            let mut probes = Vec::new();
            for i in 0..=256 {
                let atom = sorted[i * (n-1) / 256];
                probes.extend([type7(&sorted, i as f64/256.0), atom,
                    (atom as $F).next_down() as f64, (atom as $F).next_up() as f64]);
                let idx = i * (n-1) / 256;
                if idx + 1 < n { probes.push(sorted[idx]/2.0 + sorted[idx+1]/2.0); }
            }
            probes.sort_by(f64::total_cmp);
            probes.dedup();
            let make_old = |xs: &[$F]| Original::<$F>::from_array(xs, old_delta as $F).unwrap();
            let make_modern = |xs: Vec<$F>| TDigest::<$F>::builder().delta(modern_delta)
                .scale(ScaleFamily::K2Norm).singleton_policy(SingletonPolicy::Off)
                .build().merge_unsorted(xs).unwrap();
            let mut any_failure = false;
            let mut emit = |route: &str, old: &Original<$F>, modern: Option<&TDigest<$F>>, shards: &[Vec<$F>], leaf_counts: &Value| {
                let old_estimates: Vec<_> = queries.iter().map(|&(_, q)| old.quantile(q as $F).unwrap() as f64).collect();
                let mut old_issues = query_failures(&sorted, &old_estimates);
                let old_mass: u64 = old.weights.iter().map(|&w| w as u64).sum();
                if old_mass != n as u64 { old_issues.push("count mass mismatch".into()); }
                let (core, state, issues) = if let Some(td) = modern {
                    let estimates: Vec<_> = queries.iter().map(|&(_, q)| td.quantile(q)).collect();
                    let mut issues = query_failures(&sorted, &estimates);
                    let mass: f64 = td.centroids().iter().map(|c| c.weight_f64()).sum();
                    if td.count() != n as f64 || mass != n as f64 { issues.push("count mass mismatch".into()); }
                    if td.centroids().iter().any(|c| !c.mean_f64().is_finite() || !c.weight_f64().is_finite() || c.weight_f64() <= 0.0) {
                        issues.push("invalid centroid".into());
                    }
                    if td.min() != sorted[0] || td.max() != sorted[n-1] { issues.push("extrema mismatch".into()); }
                    let cdfs: Vec<f64> = probes.iter().map(|&x| td.cdf(&[x])[0]).collect();
                    if cdfs.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) { issues.push("invalid CDF".into()); }
                    if cdfs.windows(2).any(|p| p[0] > p[1]) { issues.push("CDF order decreased".into()); }
                    let cdf_rows: Vec<_> = probes.iter().zip(&cdfs).map(|(&x, &y)| {
                        let (lo, hi) = empirical_interval(&sorted, x); let truth = (lo + hi)/2.0;
                        json!([x,y,truth,(y-truth).abs()])
                    }).collect();
                    let bytes = td.to_bytes();
                    let restored = match gr_tdigest::tdigest::wire::decode_digest(&bytes).unwrap() {
    gr_tdigest::tdigest::wire::WireDecodedDigest::$variant(td) => td,
    _ => panic!("wire decoder changed the stored mean precision"),
};
                    let wire_equal = !queries.iter().zip(&estimates).any(|(&(_,q),&y)| restored.quantile(q).to_bits()!=y.to_bits())
                        && !probes.iter().zip(&cdfs).any(|(&x,&y)| restored.cdf(&[x])[0].to_bits()!=y.to_bits());
                    if !wire_equal {
                        issues.push("wire round-trip query mismatch".into());
                    }
                    let drift: Vec<_> = estimates.iter().zip(&old_estimates).map(|(&a,&b)| a-b).collect();
                    ("modern", json!({"quantile":quantile_metrics(&sorted,&queries,&estimates),
                        "cdf":{"columns":["x","estimate","midpoint_ecdf","abs_error"],"rows":cdf_rows},
                        "drift_vs_original_at_same_query":drift,
                        "centroids":td.centroids().len(),"count":td.count(),"sum":td.sum(),
                        "logical_retained_bytes":size_of::<TDigest<$F>>()+td.centroids().len()*size_of::<Centroid<$F>>(),
                        "object_bytes":size_of::<TDigest<$F>>(),"centroid_bytes":size_of::<Centroid<$F>>(),
                        "vector_capacity_bytes":Value::Null,"wire_bytes":bytes.len(),"wire_queries_bit_equal":wire_equal}), issues)
                } else {
                    let mut compacted = Original::<$F> { means: old.means.clone(), weights: old.weights.clone() };
                    compacted.means.shrink_to_fit(); compacted.weights.shrink_to_fit();
                    assert_eq!(compacted.means,old.means); assert_eq!(compacted.weights,old.weights);
                    ("original", json!({"quantile":quantile_metrics(&sorted,&queries,&old_estimates),
                        "centroids":old.means.len(),"count":old_mass,
                        "logical_retained_bytes":size_of::<Original<$F>>()+old.means.len()*size_of::<$F>()+old.weights.len()*size_of::<u32>(),
                        "object_bytes":size_of::<Original<$F>>(),"centroid_bytes":size_of::<$F>()+size_of::<u32>(),
                        "vector_capacity_bytes":compacted.means.capacity()*size_of::<$F>()+compacted.weights.capacity()*size_of::<u32>()}), old_issues)
                };
                // Preserve inherited original behavior as evidence, not a modern
                // invariant failure or a reason to censor the dataset.
                any_failure |= core == "modern" && !issues.is_empty();
                println!("{}", json!({"kind":kind,"n":n,"seed":seed,"precision":stringify!($F),
                    "input_fnv1a64":format!("{fingerprint:016x}"),
                    "reference_samples":if n<=32 {Some(&sorted)} else {None},"original_delta":old_delta,"modern_delta":modern_delta,
                    "route":route,"core":core,"shard_sizes":shards.iter().map(|s|s.len()).collect::<Vec<_>>(),
                    "empty_shards_skipped":shards.iter().filter(|s|s.is_empty()).count(),"leaf_centroid_counts":leaf_counts,
                    "state":state,
                    "invariant_failures":if core=="modern" {issues.clone()} else {Vec::new()},
                    "inherited_original_observations":if core=="original" {issues} else {Vec::new()}}));
            };
            let direct_old = make_old(&values);
            let direct_modern = make_modern(values.clone());
            emit("direct", &direct_old, None, &[], &Value::Null);
            emit("direct", &direct_old, Some(&direct_modern), &[], &Value::Null);
            let mut shuffled = values.clone();
            let mut shuffle_seed = seed ^ 0xa0761d6478bd642f;
            for i in (1..n).rev() { let j=(random_bits(&mut shuffle_seed) % (i as u64+1)) as usize; shuffled.swap(i,j); }
            for layout in ["balanced_iid", "balanced_range", "skewed"] {
                let mut shards: Vec<Vec<$F>> = (0..40).map(|_|Vec::new()).collect();
                for i in 0..n {
                    let (slot, value) = match layout {
                        "balanced_iid" => (i % 40, shuffled[i]),
                        "balanced_range" => ((i*40/n).min(39), sorted[i] as $F),
                        "skewed" => (if i < n*9/10 {0} else {1+(i-n*9/10)%39}, shuffled[i]),
                        _ => unreachable!(),
                    };
                    shards[slot].push(value);
                }
                let old_parts: Vec<_> = shards.iter().filter(|s|!s.is_empty()).map(|s|make_old(s)).collect();
                let new_parts: Vec<_> = shards.iter().filter(|s|!s.is_empty()).map(|s|make_modern(s.clone())).collect();
                let leaf_counts=json!({"original":old_parts.iter().map(|d|d.means.len()).collect::<Vec<_>>(),
                    "modern":new_parts.iter().map(|d|d.centroids().len()).collect::<Vec<_>>()});
                for reversed in [false,true] {
                    if reversed && !reverse {continue;}
                    let mut order: Vec<usize>=(0..old_parts.len()).collect();
                    if reversed {order.reverse();}
                    let mut om=Original::<$F>{means:old_parts[order[0]].means.clone(), weights:old_parts[order[0]].weights.clone()};
                    let mut nm=new_parts[order[0]].clone();
                    for &i in &order[1..] {om=om.merge(&old_parts[i],old_delta as $F).unwrap();nm.merge(&new_parts[i]);}
                    let nk=TDigest::<$F>::merge_digests(order.iter().map(|&i|new_parts[i].clone()).collect());
                    let suffix=if reversed {"reverse"} else {"forward"};
                    emit(&format!("{layout}/{suffix}/sequential"),&om,None,&shards,&leaf_counts);
                    emit(&format!("{layout}/{suffix}/sequential"),&om,Some(&nm),&shards,&leaf_counts);
                    emit(&format!("{layout}/{suffix}/kway"),&om,Some(&nk),&shards,&leaf_counts);
                }
            }
            if any_failure {std::process::exit(2);}
        }
    };
}
precision_suite!(run32, f32, F32);
precision_suite!(run64, f64, F64);
// Independent driver checks these helper results against exact Python rationals.
fn reference_probe() {
    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("singleton", vec![1.0]),
        ("two", vec![1.0, 2.0]),
        ("ties", vec![0.0, 0.0, 10.0, 10.0]),
        ("negative", vec![-4.0, -2.0, 0.0, 2.0, 4.0]),
        ("adjacent_scale", vec![0.0, 2.0_f64.powi(-24), 1.0]),
        (
            "f32_collision",
            [16777216.0_f64, 16777217.0, 16777218.0]
                .iter()
                .map(|&v| v as f32 as f64)
                .collect(),
        ),
        ("all_equal", vec![3.0, 3.0, 3.0]),
    ];
    let qs = [0.0, 0.25, 0.5, 0.75, 1.0];
    let probes = [
        -5.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 11.0, 16777216.0, 16777218.0,
    ];
    for (name, xs) in fixtures {
        let quantile: Vec<_> = qs.iter().map(|&q| json!([q, type7(&xs, q)])).collect();
        let cdf: Vec<_> = probes
            .iter()
            .map(|&x| {
                let (lo, hi) = empirical_interval(&xs, x);
                json!([x, lo, hi, (lo + hi) / 2.0])
            })
            .collect();
        let mut ranks = Vec::new();
        for q in qs {
            for x in probes {
                // Exercise the actual reference metric path, not another local
                // reimplementation of its formula.
                let metric = quantile_metrics(&xs, &[(q, q), (0.5, 0.5)], &[x, x]);
                ranks.push(json!([q, x, metric["rows"][0][4]]));
            }
        }
        println!(
            "{}",
            json!({"fixture":name,"samples":xs,"quantile":quantile,"cdf":cdf,"rank":ranks})
        );
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--reference-probe" {
        reference_probe();
        return;
    }
    assert_eq!(
        args.len(),
        8,
        "kind n seed f32|f64 original_delta modern_delta forward|both-orders"
    );
    let n: usize = args[2].parse().unwrap();
    assert!(n > 0);
    let seed = args[3].parse().unwrap();
    let old_delta: f64 = args[5].parse().unwrap();
    let modern_delta: f64 = args[6].parse().unwrap();
    assert!(
        old_delta.is_finite() && old_delta > 0.0 && modern_delta.is_finite() && modern_delta > 0.0
    );
    let reverse = match args[7].as_str() {
        "forward" => false,
        "both-orders" => true,
        _ => panic!("unknown route order"),
    };
    match args[4].as_str() {
        "f32" => run32(&args[1], n, seed, old_delta, modern_delta, reverse),
        "f64" => run64(&args[1], n, seed, old_delta, modern_delta, reverse),
        _ => panic!("unknown precision"),
    }
}
