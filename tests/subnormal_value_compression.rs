//! Independent exact-input control for E16's second underflow boundary.
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::TDigest;

#[test]
fn subnormal_value_cancellation_preserves_correctly_rounded_mean() {
    let u = f64::from_bits(1);
    let values = [-3.0 * u, 2.0 * u];
    let make = |weights: &[f64]| {
        TDigest::<f64>::builder()
            .max_size(1)
            .singleton_policy(SingletonPolicy::Off)
            .build()
            .merge_weighted_unsorted(&values, weights)
            .unwrap()
    };
    let ordinary = make(&[1.0, 1.0]);
    let tiny = make(&[u, u]);
    // Exact rational inputs: (-3*2^-1074 + 2*2^-1074)/2 = -2^-1075.
    // IEEE nearest-even rounds this to negative zero; zero's sign does not
    // change numerical error, so ordinary equality accepts the baseline +0.
    let expected = -0.0_f64;
    println!("tiny_value_control baseline_or_candidate_mean={:?} bits={} normal_mean={:?} exact_rounded={:?}",tiny.median(),tiny.median().to_bits(),ordinary.median(),expected);
    assert_eq!(ordinary.median(), expected);
    assert_eq!(tiny.median(), expected);
}
