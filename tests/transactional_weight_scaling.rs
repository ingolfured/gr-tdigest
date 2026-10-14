//! Scaling validates all candidate mass before replacing the digest state.

use gr_tdigest::tdigest::centroids::Centroid;
use gr_tdigest::tdigest::precision::FloatLike;
use gr_tdigest::tdigest::TDigest;
use gr_tdigest::TdError;
use ordered_float::FloatCore;

fn rejected_products<F: FloatLike + FloatCore>() {
    for (label, weights, factor) in [
        ("minimum subnormal underflow", vec![f64::from_bits(1)], 0.5),
        ("single weight overflow", vec![1e308], 2.0),
        ("later weight underflow", vec![1e-100, 1e-300], 1e-100),
        (
            "finite products with overflowing total",
            vec![5e307, 5e307],
            2.0,
        ),
    ] {
        let values: Vec<F> = (0..weights.len()).map(|i| F::from_f64(i as f64)).collect();
        let mut digest = TDigest::<F>::from_weighted_unsorted(&values, &weights, 1000).unwrap();
        let before = digest.to_bytes();
        let error = digest
            .scale_weights(factor)
            .err()
            .expect("invalid weight scaling must fail");
        assert!(
            matches!(error, TdError::InvalidWeight { .. }),
            "{label}: {error}"
        );
        assert_eq!(
            digest.to_bytes(),
            before,
            "{label}: failed scaling mutated state"
        );
    }
}

#[test]
fn invalid_scaled_mass_returns_normal_errors_transactionally_in_both_precisions() {
    rejected_products::<f32>();
    rejected_products::<f64>();
}

fn rejected_seeded_count_overflow<F: FloatLike + FloatCore>() {
    // Public seeded states carry their own count. Keep all candidate centroid
    // products and their sum finite so this reaches the final stored-count guard.
    let zero = F::from_f64(0.0);
    let mut digest = TDigest::<F>::new(
        vec![Centroid::new_atomic_unit_f64(0.0)],
        0.0,
        1e308,
        zero,
        zero,
        1000,
    );
    let factor = 2.0;
    let candidate_mass: f64 = digest
        .centroids()
        .iter()
        .map(|centroid| centroid.weight_f64() * factor)
        .sum();
    assert_eq!(candidate_mass, 2.0);
    assert!((digest.count() * factor).is_infinite());
    let before = digest.to_bytes();
    let error = digest
        .scale_weights(factor)
        .err()
        .expect("invalid weight scaling must fail");
    assert!(matches!(error, TdError::InvalidWeight { .. }), "{error}");
    assert_eq!(
        digest.to_bytes(),
        before,
        "failed scaling mutated seeded state"
    );
}

#[test]
fn seeded_count_overflow_is_rejected_after_finite_centroid_products() {
    rejected_seeded_count_overflow::<f32>();
    rejected_seeded_count_overflow::<f64>();
}

fn valid_extremes<F: FloatLike + FloatCore>() {
    let mut tiny = TDigest::<F>::from_weighted_unsorted(
        &[F::from_f64(-2.0), F::from_f64(6.0)],
        &[1.0, 3.0],
        1000,
    )
    .unwrap();
    let original = tiny.clone();
    tiny.scale_weights(f64::from_bits(1)).unwrap();
    assert_eq!(tiny.count(), f64::from_bits(4));
    assert_eq!(tiny.centroids()[0].weight_f64(), f64::from_bits(1));
    assert_eq!(tiny.centroids()[1].weight_f64(), f64::from_bits(3));
    assert_eq!(tiny.cdf(&[-2.0, 2.0, 6.0]), original.cdf(&[-2.0, 2.0, 6.0]));
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(tiny.quantile(q), original.quantile(q));
    }
    let mut large =
        TDigest::<F>::from_weighted_unsorted(&[F::from_f64(0.0)], &[1e308], 1000).unwrap();
    large.scale_weights(0.5).unwrap();
    assert_eq!(large.count(), 5e307);
    assert_eq!(large.centroids()[0].weight_f64(), 5e307);
    assert!(large.centroids()[0].is_atomic());
    assert_eq!(large.cdf(&[0.0]), vec![0.5]);
}

#[test]
fn positive_subnormal_products_and_large_finite_totals_remain_supported() {
    valid_extremes::<f32>();
    valid_extremes::<f64>();
}
