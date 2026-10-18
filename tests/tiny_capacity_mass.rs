//! A tiny total cap cannot preserve two endpoints by discarding interior mass.

use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::{ScaleFamily, TDigest};

macro_rules! tiny_cap_regressions {
    ($direct:ident, $repeated:ident, $control:ident, $float:ty) => {
        #[test]
        fn $direct() {
            for scale in [
                ScaleFamily::Quad,
                ScaleFamily::K1,
                ScaleFamily::K2,
                ScaleFamily::K2Norm,
                ScaleFamily::K3,
            ] {
                for cap in [1, 2] {
                    let digest = TDigest::<$float>::builder()
                        .max_size(cap)
                        .scale(scale)
                        .singleton_policy(SingletonPolicy::Use)
                        .build()
                        .merge_unsorted(vec![0.0, 1.0, 2.0, 3.0])
                        .unwrap();
                    assert_eq!(digest.count(), 4.0);
                    assert_eq!(digest.sum(), 6.0);
                    assert_eq!(
                        digest
                            .centroids()
                            .iter()
                            .map(|c| c.weight_f64())
                            .sum::<f64>(),
                        4.0,
                        "cap={cap}, scale={scale:?}: every input must retain its mass"
                    );
                    let points: Vec<_> = digest
                        .centroids()
                        .iter()
                        .map(|c| (c.mean_f64(), c.weight_f64()))
                        .collect();
                    assert_eq!(
                        points,
                        if cap == 1 {
                            vec![(1.5, 4.0)]
                        } else {
                            vec![(0.5, 2.0), (2.5, 2.0)]
                        }
                    );
                    assert!(digest.centroids().iter().all(|c| !c.is_atomic()));
                    assert_eq!(digest.quantile(0.0), 0.0);
                    assert_eq!(digest.median(), 1.5);
                    assert_eq!(digest.quantile(1.0), 3.0);
                    assert_eq!(digest.cdf(&[1.5]), vec![0.5]);
                }
            }
        }

        #[test]
        fn $repeated() {
            for cap in [1, 2] {
                let empty = || {
                    TDigest::<$float>::builder()
                        .max_size(cap)
                        .singleton_policy(SingletonPolicy::Use)
                        .build()
                };
                // Powers of two make the mass assertions exact in both precisions.
                let values: [$float; 4] = [0.0, 2.0, 4.0, 6.0];
                let weights = [0.125, 0.25, 0.5, 0.125];
                let shard = empty().merge_weighted_unsorted(&values, &weights).unwrap();
                let check = |digest: &TDigest<$float>, expected: f64| {
                    assert_eq!(digest.count(), expected);
                    assert_eq!(
                        digest
                            .centroids()
                            .iter()
                            .map(|c| c.weight_f64())
                            .sum::<f64>(),
                        expected
                    );
                    assert!(!digest.centroids().is_empty() && digest.centroids().len() <= cap);
                    assert_eq!((digest.min(), digest.max()), (0.0, 6.0));
                    let cdf = digest.cdf(&[-1.0, 0.0, 1.0, 3.0, 5.0, 6.0, 7.0]);
                    assert!(cdf.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
                    assert!(cdf.windows(2).all(|pair| pair[0] <= pair[1]));
                };
                check(&shard, 1.0);
                let mut merged = shard.clone();
                let mut updated = shard.clone();
                for copies in 2..=32 {
                    merged.merge(&shard);
                    updated.add_weighted_many(&values, &weights).unwrap();
                    check(&merged, copies as f64);
                    check(&updated, copies as f64);
                }
                let all = TDigest::merge_digests(vec![shard; 40]);
                check(&all, 40.0);
            }
        }

        #[test]
        fn $control() {
            // Ordinary capacities keep their existing exact states.
            for cap in [3, 128] {
                let digest = TDigest::<$float>::builder()
                    .max_size(cap)
                    .singleton_policy(SingletonPolicy::Use)
                    .build()
                    .merge_unsorted((0..9).map(|i| i as $float).collect())
                    .unwrap();
                let points: Vec<_> = digest
                    .centroids()
                    .iter()
                    .map(|c| (c.mean_f64(), c.weight_f64(), c.is_atomic()))
                    .collect();
                let expected = if cap == 3 {
                    vec![(0.0, 1.0, true), (4.0, 7.0, false), (8.0, 1.0, true)]
                } else {
                    (0..9).map(|i| (i as f64, 1.0, true)).collect()
                };
                assert_eq!(points, expected);
                assert_eq!(digest.count(), 9.0);
                assert_eq!(digest.sum(), 36.0);
            }
            // A tiny digest already inside its cap remains atomic and exact.
            for cap in [1, 2] {
                let values: Vec<$float> = (0..cap).map(|i| i as $float).collect();
                let digest = TDigest::<$float>::builder()
                    .max_size(cap)
                    .singleton_policy(SingletonPolicy::Use)
                    .build()
                    .merge_unsorted(values)
                    .unwrap();
                assert_eq!(digest.centroids().len(), cap);
                assert!(digest
                    .centroids()
                    .iter()
                    .all(|c| c.is_atomic() && c.weight_f64() == 1.0));
            }
        }
    };
}

tiny_cap_regressions!(
    f32_tiny_use_caps_keep_interior_mass,
    f32_fractional_repeated_merges_keep_mass,
    f32_ordinary_use_caps_unchanged,
    f32
);
tiny_cap_regressions!(
    f64_tiny_use_caps_keep_interior_mass,
    f64_fractional_repeated_merges_keep_mass,
    f64_ordinary_use_caps_unchanged,
    f64
);
