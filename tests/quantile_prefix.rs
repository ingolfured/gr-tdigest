//! E17: exact reflection symmetry, CDF round trips, and ordinary controls.
use gr_tdigest::tdigest::centroids::Centroid;
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::TDigest;

macro_rules! symmetry {
    ($name:ident,$float:ty) => {
        #[test]
        fn $name() {
            // The exact binary input masses are pairwise equal under reflection.
            // Therefore q=1/2 falls exactly between values 2 and 3, independently
            // of decimal interpretation or a floating-point prefix implementation.
            for weights in [
                [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                [0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
            ] {
                let values: [$float; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
                let td = TDigest::<$float>::builder()
                    .max_size(6)
                    .singleton_policy(SingletonPolicy::Off)
                    .build()
                    .merge_weighted_unsorted(&values, &weights)
                    .unwrap();
                println!(
                    "symmetry weights={weights:?} count={:?} median={:?} cdf_gap={:?}",
                    td.count(),
                    td.median(),
                    td.cdf(&[2.5])[0]
                );
                assert_eq!(td.median(), 2.5, "weights={weights:?}");
                assert_eq!(td.quantile(0.0), 0.0);
                assert_eq!(td.quantile(1.0), 5.0);
            }
        }
    };
}
symmetry!(fractional_reflection_symmetry_f32, f32);
symmetry!(fractional_reflection_symmetry_f64, f64);

#[test]
fn atomic_gap_cdf_round_trip_is_preserved() {
    let td = TDigest::<f64>::builder()
        .max_size(6)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .merge_weighted_unsorted(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
        )
        .unwrap();
    let q = td.cdf(&[2.5])[0];
    println!("roundtrip q={q:?} returned={:?}", td.quantile(q));
    assert_eq!(td.quantile(q), 2.5);
}

#[test]
fn dyadic_atomic_and_mixed_controls() {
    let atomic = TDigest::<f64>::builder()
        .max_size(6)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .merge_weighted_unsorted(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.125, 0.25, 0.375, 0.375, 0.25, 0.125],
        )
        .unwrap();
    assert_eq!(atomic.median(), 2.5);
    let mixed = TDigest::<f64>::new(
        vec![
            Centroid::new_mixed_f64(2.0, 2.0),
            Centroid::new_mixed_f64(6.0, 2.0),
        ],
        16.0,
        4.0,
        8.0,
        0.0,
        2,
    );
    for (q, value) in [
        (0.0, 0.0),
        (0.125, 1.0),
        (0.25, 2.0),
        (0.5, 4.0),
        (0.75, 6.0),
        (0.875, 7.0),
        (1.0, 8.0),
    ] {
        assert_eq!(mixed.quantile(q), value);
        assert_eq!(mixed.cdf(&[value])[0], q);
    }
    assert!(mixed.quantile(f64::NAN).is_nan());
    assert_eq!(mixed.quantile(f64::NEG_INFINITY), 0.0);
    assert_eq!(mixed.quantile(f64::INFINITY), 8.0);
    assert!(TDigest::<f64>::builder()
        .max_size(4)
        .build()
        .median()
        .is_nan());
}
