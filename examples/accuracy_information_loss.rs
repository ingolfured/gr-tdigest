//! Two populations can have identical complete compressed state and different CDFs.
//! This is an information-loss demonstration, not an accuracy acceptance test and
//! not a proof that a larger summary or another compressor cannot improve.
use gr_tdigest::tdigest::{singleton_policy::SingletonPolicy, TDigest};
use serde_json::json;

fn main() {
    let mut rows = Vec::new();
    macro_rules! demonstrate {
        ($float:ty, $precision:literal) => {{
            let base = TDigest::<$float>::builder()
                .max_size(1)
                .singleton_policy(SingletonPolicy::Off)
                .build()
                .unwrap();
            let a = base
                .merge_weighted_unsorted(&[0., 4.5, 8., 10.], &[1., 84., 14., 1.])
                .unwrap();
            let b = base
                .merge_weighted_unsorted(&[0., 2., 5.5, 10.], &[1., 14., 84., 1.])
                .unwrap();
            assert_eq!(a.to_bytes(), b.to_bytes());
            assert_eq!(a.count(), 100.0);
            assert_eq!(a.sum(), 500.0);
            rows.push(json!({
                "precision": $precision, "capacity": 1,
                "interface": "modern Rust (Python training frontend minimum capacity is10)",
                "population_a": {"values":[0.,4.5,8.,10.],"weights":[1.,84.,14.,1.]},
                "population_b": {"values":[0.,2.,5.5,10.],"weights":[1.,14.,84.,1.]},
                "same_complete_state": true, "wire_hex": a.to_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "count": a.count(), "sum": a.sum(), "min": a.min(), "max": a.max(),
                "centroid_mean": a.centroids()[0].mean_f64(),
                "query":5., "estimate_for_both": a.cdf(&[5.])[0],
                "truth_a":0.85, "truth_b":0.15,
                "meaning":"Queries cannot recover which population produced this identical saved state. This does not prove that capacity32cannot improve."
            }));
        }};
    }
    demonstrate!(f32, "f32");
    demonstrate!(f64, "f64");
    println!("{}", serde_json::to_string_pretty(&json!({"kind":"INFORMATION_LOSS_DEMONSTRATION_NOT_ACCURACY_ACCEPTANCE", "examples":rows})).unwrap());
}
