use tdigest_core::TDigest;

macro_rules! overflow_tests {
    ($name:ident, $float:ty) => {
        #[test]
        fn $name() {
            let overfull = TDigest::<$float> {
                means: vec![1.0, 2.0],
                weights: vec![u32::MAX, 1],
            };
            assert!(overfull
                .quantile(0.5)
                .unwrap_err()
                .to_string()
                .contains("overflow"));
            assert!(overfull.median().is_err());
            assert!(overfull.trimmed_mean(0.0, 1.0).is_err());
            assert!(
                TDigest::<$float>::from_means_weights(&[1.0, 2.0], &[u32::MAX, 1], 300.0,).is_err()
            );

            let full = TDigest::<$float>::from_means_weights(&[1.0], &[u32::MAX], 300.0).unwrap();
            let one = TDigest::<$float>::from_array(&[2.0], 300.0).unwrap();
            assert!(full.merge(&one, 300.0).is_err());
            assert_eq!(full.means, vec![1.0]);
            assert_eq!(full.weights, vec![u32::MAX]);
            assert_eq!(full.quantile(0.5).unwrap(), 1.0);

            // Finite inputs must not expose an overflowed compressed mean or query.
            let max = <$float>::MAX;
            assert!(TDigest::<$float>::from_means_weights(
                &[max * 0.125, max * 0.25, max * 0.5, max],
                &[10, 10, 10, 10],
                1.0,
            )
            .is_err());
            let extreme = TDigest::<$float> {
                means: vec![max * 0.5, max],
                weights: vec![10, 10],
            };
            assert!(extreme.quantile(0.5).is_err());
            assert!(extreme.trimmed_mean(0.0, 1.0).is_err());
        }
    };
}

overflow_tests!(original_overflow_is_an_error_f32, f32);
overflow_tests!(original_overflow_is_an_error_f64, f64);
