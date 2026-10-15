//! New Rust checks against the independently generated, pinned original output.
//! The fixture is shared with the exact Python gate and is never regenerated here.
use serde_json::Value;
use tdigest_core::TDigest;

fn decode(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
}

fn assert_number(actual: f64, bits: &str, context: &str) {
    let expected = decode(bits);
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
    }
}

macro_rules! original_numerics {
    ($test:ident, $float:ty, $dtype:literal) => {
        #[test]
        fn $test() {
            fn dataset(name: &str) -> Vec<$float> {
                match name {
                    "singleton" => vec![3.0],
                    "two" => vec![1.0, 2.0],
                    "ties" => vec![3.0, 3.0, 1.0, 0.0, 0.0, -0.0],
                    "nonfinite" => vec![
                        <$float>::NEG_INFINITY,
                        -2.0,
                        0.0,
                        <$float>::INFINITY,
                        <$float>::NAN,
                    ],
                    "allnan" => vec![<$float>::NAN, <$float>::NAN],
                    "tiny" => vec![-<$float>::MIN_POSITIVE, -0.0, 0.0, <$float>::MIN_POSITIVE],
                    "weighted" => vec![0.0, 10.0],
                    "large-weight" | "uint32-max" => vec![1.0],
                    "uneven" => vec![-10.0, 0.0, 1.0, 3.0, 100.0],
                    "linear" => (0..1000).map(|i| i as $float / 10.0).collect(),
                    "compressed" => (0..1000)
                        .map(|i| ((i * 37 % 1000) as $float - 500.0) / 10.0)
                        .collect(),
                    _ => panic!("unknown pinned dataset {name}"),
                }
            }
            let reference: Value = serde_json::from_str(include_str!(
                "../bindings/python/tests/data/upstream_numerical_golden.json"
            ))
            .unwrap();
            assert_eq!(
                reference["upstream_commit"],
                "8d9cf1fe82c5c9254e5dc9593f01717e317f4333"
            );
            let mut checked = 0;
            for case in reference["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|case| case["dtype"] == $dtype)
            {
                checked += 1;
                let context = case["name"].as_str().unwrap();
                let params = &case["input"];
                let values = dataset(params["dataset"].as_str().unwrap());
                let delta = params["delta"].as_f64().unwrap() as $float;
                let operation = params["op"].as_str().unwrap();
                let digest = match operation {
                    "from_array" => TDigest::<$float>::from_array(&values, delta).unwrap(),
                    "from_means_weights" => {
                        let weights: Vec<u32> = params["weights"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
                            .collect();
                        TDigest::<$float>::from_means_weights(&values, &weights, delta).unwrap()
                    }
                    "merge" | "update" => {
                        // The fixture's Python update is build(right) + immutable merge.
                        let split = params["split"].as_u64().unwrap() as usize;
                        let left = TDigest::<$float>::from_array(&values[..split], delta).unwrap();
                        let right = TDigest::<$float>::from_array(
                            &values[split..],
                            params["other_delta"].as_f64().unwrap() as $float,
                        )
                        .unwrap();
                        let before: Vec<u64> =
                            left.means.iter().map(|x| (*x as f64).to_bits()).collect();
                        let weights = left.weights.clone();
                        let result = left
                            .merge(&right, params["merge_delta"].as_f64().unwrap() as $float)
                            .unwrap();
                        assert_eq!(
                            left.means
                                .iter()
                                .map(|x| (*x as f64).to_bits())
                                .collect::<Vec<_>>(),
                            before
                        );
                        assert_eq!(left.weights, weights);
                        result
                    }
                    "copy" | "deepcopy" | "pickle" | "dict" => {
                        // These fixture cases model original Python reconstruction via the Rust constructor.
                        let original = TDigest::<$float>::from_array(&values, delta).unwrap();
                        TDigest::<$float>::from_means_weights(
                            &original.means,
                            &original.weights,
                            300.0,
                        )
                        .unwrap()
                    }
                    _ => panic!("unknown pinned operation {operation}"),
                };
                let expected_means = case["means"].as_array().unwrap();
                assert_eq!(digest.means.len(), expected_means.len(), "{context}");
                for (actual, expected) in digest.means.iter().zip(expected_means) {
                    assert_number(*actual as f64, expected.as_str().unwrap(), context);
                }
                let weights: Vec<u32> = case["weights"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
                    .collect();
                assert_eq!(digest.weights, weights, "{context}");
                assert_eq!(
                    digest.n_zero_weights().unwrap(),
                    weights.iter().filter(|&&w| w == 0).count()
                );
                assert_number(
                    digest.median().unwrap() as f64,
                    case["median"].as_str().unwrap(),
                    context,
                );
                for query in case["quantiles"].as_array().unwrap() {
                    let q = decode(query[0].as_str().unwrap()) as $float;
                    assert_number(
                        digest.quantile(q).unwrap() as f64,
                        query[1].as_str().unwrap(),
                        context,
                    );
                }
                for query in case["trimmed_means"].as_array().unwrap() {
                    let lower = query[0].as_f64().unwrap() as $float;
                    let upper = query[1].as_f64().unwrap() as $float;
                    assert_number(
                        digest.trimmed_mean(lower, upper).unwrap() as f64,
                        query[2].as_str().unwrap(),
                        context,
                    );
                }
            }
            assert_eq!(checked, 35);
        }
    };
}

original_numerics!(independent_original_outputs_f32, f32, "float32");
original_numerics!(independent_original_outputs_f64, f64, "float64");
