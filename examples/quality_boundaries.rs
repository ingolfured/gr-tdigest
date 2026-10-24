//! Small independent boundary controls. Numerical failures are measurements,
//! never silently skipped tests. --require-clean returns nonzero on any failure.
use gr_tdigest::tdigest::{singleton_policy::SingletonPolicy, TDigest};
use serde_json::{json, Value};

fn record(rows: &mut Vec<Value>, id: &str, precision: &str, expected: f64, observed: f64) {
    rows.push(
        json!({"id": format!("{id}/{precision}"), "expected":expected,
                    "observed":observed, "status":if observed == expected {"PASS"} else {"FAIL"}}),
    );
}

fn main() {
    let mut rows = Vec::new();
    macro_rules! controls {
        ($float:ty, $precision:literal) => {{
            for cap in [1, 2] {
                let d = TDigest::<$float>::builder()
                    .max_size(cap)
                    .singleton_policy(SingletonPolicy::Use)
                    .build()
                    .unwrap()
                    .merge_unsorted(vec![0., 1., 2., 100.])
                    .unwrap();
                let mass: f64 = d.centroids_ref().iter().map(|c| c.weight_f64()).sum();
                record(
                    &mut rows,
                    &format!("tiny_use_mass_cap{cap}"),
                    $precision,
                    4.,
                    mass,
                );
            }
            let d = TDigest::<$float>::builder()
                .max_size(2)
                .singleton_policy(SingletonPolicy::Off)
                .build()
                .unwrap()
                .merge_weighted_unsorted(
                    &[0., 1., 2f64.powi(60) as $float],
                    &[0.5, 0.5, 2f64.powi(-60)],
                )
                .unwrap();
            let moment: f64 = d
                .centroids_ref()
                .iter()
                .map(|c| c.mean_f64() * c.weight_f64())
                .sum();
            // The two nonzero contributions are exactly 1/2 and 1.
            record(&mut rows, "residual_tail_moment", $precision, 1.5, moment);
            let d = TDigest::<$float>::builder()
                .max_size(1)
                .singleton_policy(SingletonPolicy::Off)
                .build()
                .unwrap()
                .merge_unsorted(vec![-1e16 as $float, 1., 1e16 as $float])
                .unwrap();
            record(
                &mut rows,
                "finite_cancellation_mean",
                $precision,
                1. / 3.,
                d.mean(),
            );
            for (i, weights) in [
                [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                [0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
                [0.1; 6],
            ]
            .iter()
            .enumerate()
            {
                let d = TDigest::<$float>::builder()
                    .max_size(128)
                    .singleton_policy(SingletonPolicy::Off)
                    .build()
                    .unwrap()
                    .merge_weighted_unsorted(&[0., 1., 2., 3., 4., 5.], weights)
                    .unwrap();
                // Pairwise-identical binary weights imply exact reflection.
                record(
                    &mut rows,
                    &format!("fractional_symmetric_median_{i}"),
                    $precision,
                    2.5,
                    d.median(),
                );
            }
        }};
    }
    controls!(f32, "f32");
    controls!(f64, "f64");
    let tiny = f64::from_bits(1);
    let d = TDigest::<f64>::builder()
        .max_size(128)
        .build()
        .unwrap()
        .merge_weighted_unsorted(&[tiny], &[0.1]);
    match d {
        Ok(d) => record(&mut rows, "subnormal_aggregate_mean", "f64", tiny, d.mean()),
        Err(error) => rows.push(json!({"id":"subnormal_aggregate_mean/f64", "expected":tiny,
            "observed":null, "status":"UNSUPPORTED", "error":error.to_string()})),
    }
    // An explicit unsupported-product error prevents silent corruption. It is
    // still not a successful numerical result for this historical profile.
    let failed = rows.iter().filter(|r| r["status"] != "PASS").count();
    println!("{}", serde_json::to_string_pretty(&json!({"accuracy_status":if failed==0 {"PASS"} else {"FAIL"}, "failed":failed, "controls":rows})).unwrap());
    if failed > 0 && std::env::args().any(|arg| arg == "--require-clean") {
        std::process::exit(1);
    }
}
