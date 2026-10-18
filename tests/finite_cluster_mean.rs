//! Frozen exact-input oracle for the experimental cold overflow fallback.

use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::wire::{decode_digest, WireDecodedDigest};
use gr_tdigest::tdigest::TDigest;

fn value(bits: &serde_json::Value) -> f64 {
    f64::from_bits(u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap())
}

fn ordered_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 != 0 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

#[test]
fn finite_cluster_mean_matches_exact_input_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/finite_cluster_mean_oracle.json")).unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let means: Vec<f64> = case["means_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(value)
            .collect();
        let weights: Vec<f64> = case["weights_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(value)
            .collect();
        let expected = value(&case["expected_bits"]);
        let precision = case["precision"].as_str().unwrap();
        let (actual, raw_sum, wire_roundtrip, cdf) = if precision == "f32" {
            let means: Vec<f32> = means.iter().map(|v| *v as f32).collect();
            let td = TDigest::<f32>::builder()
                .max_size(1)
                .singleton_policy(SingletonPolicy::Off)
                .build()
                .merge_weighted_unsorted(&means, &weights)
                .unwrap();
            let restored = match decode_digest(&td.to_bytes()).unwrap() {
                WireDecodedDigest::F32(td) => td,
                _ => panic!("precision changed"),
            };
            (
                td.median(),
                td.sum(),
                restored.to_bytes() == td.to_bytes(),
                td.cdf(&[td.median()])[0],
            )
        } else {
            let td = TDigest::<f64>::builder()
                .max_size(1)
                .singleton_policy(SingletonPolicy::Off)
                .build()
                .merge_weighted_unsorted(&means, &weights)
                .unwrap();
            let restored = match decode_digest(&td.to_bytes()).unwrap() {
                WireDecodedDigest::F64(td) => td,
                _ => panic!("precision changed"),
            };
            (
                td.median(),
                td.sum(),
                restored.to_bytes() == td.to_bytes(),
                td.cdf(&[td.median()])[0],
            )
        };
        let ulps = if precision == "f32" {
            let a = actual as f32;
            let e = expected as f32;
            let key = |x: f32| {
                let b = x.to_bits();
                if b >> 31 != 0 {
                    !b
                } else {
                    b | 1 << 31
                }
            };
            key(a).abs_diff(key(e)) as u64
        } else {
            ordered_bits(actual).abs_diff(ordered_bits(expected))
        };
        let passed = actual.is_finite()
            && ulps <= case["max_ulps"].as_u64().unwrap()
            && wire_roundtrip
            && cdf == 0.5
            && !raw_sum.is_finite();
        println!(
            "{}",
            serde_json::json!({"name":case["name"],"precision":precision,"actual_bits":format!("{:016x}",actual.to_bits()),"expected_bits":case["expected_bits"],"actual":if actual.is_finite(){Some(actual)}else{None},"expected":expected,"absolute_error":if actual.is_finite(){Some((actual-expected).abs())}else{None},"ulp_error":ulps,"max_ulps":case["max_ulps"],"raw_sum_class":if raw_sum.is_nan(){"nan"}else if raw_sum.is_infinite(){"infinite"}else{"finite"},"wire_roundtrip":wire_roundtrip,"passed":passed})
        );
        if !passed {
            failures.push(case["name"].as_str().unwrap().to_string());
        }
    }
    assert!(
        failures.is_empty(),
        "exact-input fallback gate failed: {failures:?}"
    );
}

#[test]
fn finite_cluster_mean_keeps_ordinary_and_singleton_arithmetic() {
    for (values, weights, mean) in [
        (vec![-4.0, 0.0, 8.0], vec![0.25, 0.5, 0.25], 1.0),
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 1.0], 2.0),
    ] {
        let mean: f64 = mean;
        let td = TDigest::<f64>::builder()
            .max_size(1)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .merge_weighted_unsorted(&values, &weights)
            .unwrap();
        assert_eq!(td.median().to_bits(), mean.to_bits());
        assert_eq!(td.sum().to_bits(), (mean * td.count()).to_bits());
    }
    let single = TDigest::<f64>::builder()
        .max_size(1)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .merge_weighted_unsorted(&[1e308], &[2.0])
        .unwrap();
    assert_eq!(single.median().to_bits(), 1e308_f64.to_bits());
    assert!(single.sum().is_infinite());
    assert!(
        single.mean().is_infinite(),
        "raw sum/mean overflow remains explicitly outside this repair"
    );
}
