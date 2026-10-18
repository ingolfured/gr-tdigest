//! Borrowed mutation routes must retain the owned merge API's exact state.

use gr_tdigest::tdigest::centroids::Centroid;
use gr_tdigest::tdigest::frontends::FrontendDigest;
use gr_tdigest::tdigest::precision::FloatLike;
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::{DigestStats, ScaleFamily, TDigest};
use ordered_float::FloatCore;

fn same_state<F: FloatLike + FloatCore>(actual: &TDigest<F>, expected: &TDigest<F>) {
    assert_eq!(actual.to_bytes(), expected.to_bytes());
    for q in [0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.0] {
        assert_eq!(actual.quantile(q).to_bits(), expected.quantile(q).to_bits());
    }
    let probes = [
        f64::NEG_INFINITY,
        -10.0,
        -4.0,
        -0.0,
        0.0,
        1.0,
        4.0,
        10.0,
        f64::INFINITY,
    ];
    for (got, want) in actual.cdf(&probes).into_iter().zip(expected.cdf(&probes)) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
}

fn shard<F: FloatLike + FloatCore>(
    index: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
    delta: Option<f64>,
    weight_unit: f64,
) -> TDigest<F> {
    let zero = if index % 2 == 0 { -0.0 } else { 0.0 };
    let means = [-4.0 + (index % 3) as f64 * 0.25, zero, 1.0, 4.0];
    let mut cents = Vec::new();
    let mut total_weight = 0.0;
    let mut data_sum = 0.0;
    for (i, mean) in means.into_iter().enumerate() {
        let weight = weight_unit * [0.1, 0.2, 0.3, 0.5][(index + i) % 4];
        total_weight += weight;
        data_sum += mean * weight;
        cents.push(if (index + i) % 2 == 0 {
            Centroid::<F>::new_atomic_f64(mean, weight)
        } else {
            Centroid::<F>::new_mixed_f64(mean, weight)
        });
    }
    let mut builder = TDigest::<F>::builder()
        .max_size(8)
        .scale(scale)
        .singleton_policy(policy);
    if let Some(delta) = delta {
        builder = builder.delta(delta);
    }
    builder
        .with_centroids_and_stats(
            cents,
            DigestStats {
                total_weight,
                data_sum,
                data_min: -5.0,
                data_max: 5.0,
            },
        )
        .build()
}

fn borrowed_routes_match_owned<F: FloatLike + FloatCore>() {
    let configurations = [
        (ScaleFamily::K2, SingletonPolicy::Use, None),
        (
            ScaleFamily::K1,
            SingletonPolicy::UseWithProtectedEdges(2),
            None,
        ),
        (ScaleFamily::Quad, SingletonPolicy::Off, None),
        (ScaleFamily::K3, SingletonPolicy::Off, None),
        (ScaleFamily::K2Norm, SingletonPolicy::Off, Some(25.0)),
        (ScaleFamily::K2Norm, SingletonPolicy::Off, Some(1200.0)),
    ];
    for (scale, policy, delta) in configurations {
        for weight_unit in [1.0, 1e-309] {
            let shards: Vec<_> = (0..40)
                .map(|index| shard::<F>(index, scale, policy, delta, weight_unit))
                .collect();
            let originals: Vec<_> = shards.iter().map(TDigest::to_bytes).collect();
            let mut borrowed = TDigest::<F>::default();
            let mut owned = borrowed.clone();
            for next in &shards {
                owned = TDigest::merge_digests(vec![owned, next.clone()]);
                borrowed.merge(next);
                same_state(&borrowed, &owned);
            }
            let mut many = shards[0].clone();
            many.merge_many(&shards[1..]);
            same_state(&many, &TDigest::merge_digests(shards.clone()));
            assert_eq!(
                originals,
                shards.iter().map(TDigest::to_bytes).collect::<Vec<_>>()
            );
        }
    }

    // Core empty rules differ intentionally from the frontend adoption rules.
    let empty = TDigest::<F>::builder()
        .max_size(17)
        .scale(ScaleFamily::K1)
        .build();
    let mut unchanged = empty.clone();
    unchanged.merge_many(&[]);
    same_state(&unchanged, &empty);
    let mut pair = empty.clone();
    pair.merge(&empty);
    same_state(
        &pair,
        &TDigest::merge_digests(vec![empty.clone(), empty.clone()]),
    );
    let nonempty = shard::<F>(0, ScaleFamily::K2, SingletonPolicy::Off, None, 1.0);
    let mut adopted = empty.clone();
    adopted.merge_many(&[empty.clone(), nonempty.clone(), empty.clone()]);
    same_state(
        &adopted,
        &TDigest::merge_digests(vec![empty.clone(), empty.clone(), nonempty, empty]),
    );
}

#[test]
fn borrowed_f32_routes_preserve_bytes_queries_and_sources() {
    borrowed_routes_match_owned::<f32>();
}

#[test]
fn borrowed_f64_routes_preserve_bytes_queries_and_sources() {
    borrowed_routes_match_owned::<f64>();
}

#[test]
fn frontend_borrowed_merge_preserves_checks_empty_rules_and_exact_state() {
    macro_rules! check {
        ($float:ty, $variant:ident) => {{
            let a = shard::<$float>(0, ScaleFamily::K2, SingletonPolicy::Off, None, 1.0);
            let b = shard::<$float>(1, ScaleFamily::K2, SingletonPolicy::Off, None, 1.0);
            let expected = TDigest::merge_digests(vec![a.clone(), b.clone()]);
            let rhs = FrontendDigest::$variant(b);
            let rhs_bytes = rhs.to_bytes();
            let mut lhs = FrontendDigest::$variant(a);
            lhs.merge_in_place(&rhs).unwrap();
            assert_eq!(lhs.to_bytes(), expected.to_bytes());
            assert_eq!(rhs.to_bytes(), rhs_bytes);

            let incompatible = FrontendDigest::$variant(shard::<$float>(
                0,
                ScaleFamily::K1,
                SingletonPolicy::Use,
                None,
                1.0,
            ));
            let before = lhs.to_bytes();
            assert!(lhs.merge_in_place(&incompatible).is_err());
            assert_eq!(lhs.to_bytes(), before);

            let empty = FrontendDigest::$variant(TDigest::<$float>::default());
            lhs.merge_in_place(&empty).unwrap();
            assert_eq!(lhs.to_bytes(), before);
            let mut adopt = empty;
            adopt.merge_in_place(&rhs).unwrap();
            assert_eq!(adopt.to_bytes(), rhs_bytes);
        }};
    }
    check!(f32, F32);
    check!(f64, F64);
}
