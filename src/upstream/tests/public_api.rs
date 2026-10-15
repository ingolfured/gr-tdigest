//! New regression tests for the existing public API; upstream supplied none.
use tdigest_core::TDigest;

macro_rules! public_api {
    ($name:ident, $float:ty) => {
        #[test]
        fn $name() {
            let constructed: anyhow::Result<TDigest<$float>> =
                TDigest::from_array(&[1.0, 2.0, 3.0], 300.0);
            let digest = constructed.unwrap();
            let _: &Vec<$float> = &digest.means;
            let _: &Vec<u32> = &digest.weights;
            let median: anyhow::Result<$float> = digest.median();
            assert_eq!(median.unwrap(), 2.0);
            let quantile: anyhow::Result<$float> = digest.quantile(0.5);
            assert_eq!(quantile.unwrap(), 2.0);
            let trimmed: anyhow::Result<$float> = digest.trimmed_mean(0.0, 1.0);
            assert_eq!(trimmed.unwrap(), 2.0);

            let weighted: anyhow::Result<TDigest<$float>> =
                TDigest::from_means_weights(&[2.0], &[3], 300.0);
            let weighted = weighted.unwrap();
            assert_eq!(weighted.means, vec![2.0]);
            assert_eq!(weighted.weights, vec![3]);
            let before_means = digest.means.clone();
            let before_weights = digest.weights.clone();
            let merged: anyhow::Result<TDigest<$float>> = digest.merge(&weighted, 300.0);
            let merged = merged.unwrap();
            assert_eq!(merged.weights.iter().sum::<u32>(), 6);
            assert_eq!(digest.means, before_means);
            assert_eq!(digest.weights, before_weights);
            let zero_count: anyhow::Result<usize> = merged.n_zero_weights();
            assert_eq!(zero_count.unwrap(), 0);

            // Struct literals and mutable fields are part of the original API.
            let mut direct = TDigest::<$float> {
                means: vec![1.0, 2.0],
                weights: vec![1, 0],
            };
            assert_eq!(direct.n_zero_weights().unwrap(), 1);
            direct.means[0] = 3.0;
            direct.weights[1] = 2;
            let TDigest { means, weights }: TDigest<$float> = direct;
            assert_eq!(means, vec![3.0, 2.0]);
            assert_eq!(weights, vec![1, 2]);

            // Preserve inherited Rust behavior; do not add modern validation.
            let empty = TDigest::<$float> {
                means: vec![],
                weights: vec![],
            };
            assert_eq!(empty.quantile(0.5).unwrap(), 0.0);
            assert!(TDigest::<$float>::from_array(&[], 300.0).is_ok());
            let mismatch = std::panic::catch_unwind(|| {
                TDigest::<$float>::from_means_weights(&[1.0, 2.0], &[1], 300.0)
            });
            assert!(
                mismatch.is_err(),
                "original mismatched arrays panic rather than returning TdError"
            );
        }
    };
}

public_api!(original_public_api_f32, f32);
public_api!(original_public_api_f64, f64);
