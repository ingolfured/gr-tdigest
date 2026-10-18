//! A rounded bucket target must not omit a positive weighted-value contribution.

use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::TDigest;

macro_rules! residual_regression {
    ($name:ident, $float:ty) => {
        #[test]
        fn $name() {
            let large = 2.0_f64.powi(60) as $float;
            let tiny = 2.0_f64.powi(-60);
            let empty = || TDigest::<$float>::builder().max_size(2)
                .singleton_policy(SingletonPolicy::Off).build();
            let direct = empty().merge_weighted_unsorted(&[0.0, 1.0, large], &[0.5, 0.5, tiny]).unwrap();
            let shards = vec![
                empty().merge_weighted_unsorted(&[0.0], &[0.5]).unwrap(),
                empty().merge_weighted_unsorted(&[1.0], &[0.5]).unwrap(),
                empty().merge_weighted_unsorted(&[large], &[tiny]).unwrap(),
            ];
            let check = |digest: &TDigest<$float>, contribution_count: usize| {
                assert_eq!(digest.count(), 1.0);
                let expected_sum = 0.5 + contribution_count as f64;
                assert_eq!(digest.sum(), expected_sum);
                assert_eq!(digest.centroids().iter().map(|c| c.mean_f64() * c.weight_f64()).sum::<f64>(), expected_sum,
                    "mass-only checks cannot detect this dropped input: inspect weighted contribution");
                assert_eq!(digest.centroids().iter().map(|c| (c.mean_f64(), c.weight_f64())).collect::<Vec<_>>(),
                    vec![(0.0, 0.5), (1.0 + 2.0 * contribution_count as f64, 0.5)]);
                assert!(digest.centroids()[0].is_atomic());
                assert!(!digest.centroids()[1].is_atomic());
                assert_eq!(digest.min(), 0.0);
                assert_eq!(digest.max(), large as f64);
            };
            check(&direct, 1);
            check(&TDigest::merge_digests(shards), 1);
            let mut updated = direct;
            for contributions in 2..=32 {
                updated.add_weighted(large, tiny).unwrap();
                check(&updated, contributions);
            }
        }
    };
}

residual_regression!(f32_bucket_tail_keeps_weighted_value_contribution, f32);
residual_regression!(f64_bucket_tail_keeps_weighted_value_contribution, f64);
