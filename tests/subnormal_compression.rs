//! Compression should not depend on choosing wholly subnormal mass units.
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::wire::{decode_digest, WireDecodedDigest};
use gr_tdigest::tdigest::{ScaleFamily, TDigest};

macro_rules! scale_regressions {
    ($name:ident,$float:ty,$variant:ident) => {
        #[test]
        fn $name() {
            let values: [$float; 6] = [-2.5, -1.0, 0.125, 0.5, 4.25, 8.0];
            let units = [1.0, 2.0, 1.0, 4.0, 3.0, 5.0];
            // Use caps1/2 have independent E01 mass loss; do not conflate it
            // with this independently based experiment.
            for (cap, policy, delta) in [
                (1, SingletonPolicy::Off, None),
                (2, SingletonPolicy::Off, None),
                (3, SingletonPolicy::Off, None),
                (3, SingletonPolicy::Use, None),
                (2, SingletonPolicy::UseWithProtectedEdges(1), None),
                (64, SingletonPolicy::Off, Some(1.0)),
            ] {
                let make = || {
                    let builder = TDigest::<$float>::builder()
                        .max_size(cap)
                        .singleton_policy(policy);
                    match delta {
                        Some(d) => builder.scale(ScaleFamily::K2Norm).delta(d).build(),
                        None => builder.build(),
                    }
                };
                for mass_unit in [
                    f64::from_bits(1),
                    f64::from_bits(1_u64 << 14),
                    f64::from_bits(1_u64 << 44),
                ] {
                    let weights: Vec<_> = units.iter().map(|w| w * mass_unit).collect();
                    let normal = make().merge_weighted_unsorted(&values, &units).unwrap();
                    let tiny = make().merge_weighted_unsorted(&values, &weights).unwrap();
                    let check = |actual: &TDigest<$float>, reference: &TDigest<$float>| {
                        assert_eq!(
                            actual.count().to_bits(),
                            (reference.count() * mass_unit).to_bits()
                        );
                        assert_eq!(
                            actual.centroids().len(),
                            reference.centroids().len(),
                            "cap={cap},policy={policy:?},unit={mass_unit:e}"
                        );
                        for (a, b) in actual.centroids().iter().zip(reference.centroids()) {
                            assert_eq!(
                                a.mean_f64().to_bits(),
                                b.mean_f64().to_bits(),
                                "cap={cap},policy={policy:?},unit={mass_unit:e}"
                            );
                            assert_eq!(
                                a.weight_f64().to_bits(),
                                (b.weight_f64() * mass_unit).to_bits()
                            );
                            assert_eq!(a.kind(), b.kind());
                        }
                        assert_eq!(
                            actual
                                .centroids()
                                .iter()
                                .map(|c| c.weight_f64())
                                .sum::<f64>()
                                .to_bits(),
                            actual.count().to_bits()
                        );
                        for q in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
                            assert_eq!(
                                actual.quantile(q).to_bits(),
                                reference.quantile(q).to_bits()
                            );
                        }
                        for (a, b) in actual
                            .cdf(&[-3.0, -2.5, -0.25, 0.5, 3.0, 8.0, 9.0])
                            .iter()
                            .zip(reference.cdf(&[-3.0, -2.5, -0.25, 0.5, 3.0, 8.0, 9.0]))
                        {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                    };
                    check(&tiny, &normal);
                    // Original raw metadata arithmetic remains visible.
                    let original_sum: f64 = values
                        .iter()
                        .zip(&weights)
                        .map(|(v, w)| (*v as f64) * w)
                        .sum();
                    assert_eq!(tiny.sum().to_bits(), original_sum.to_bits());
                    let restored = match decode_digest(&tiny.to_bytes()).unwrap() {
                        WireDecodedDigest::$variant(td) => td,
                        _ => panic!("precision changed"),
                    };
                    assert_eq!(tiny.to_bytes(), restored.to_bytes());
                    check(&restored, &normal);
                    let mut tiny_merged = tiny.clone();
                    let mut normal_merged = normal.clone();
                    for _ in 0..3 {
                        tiny_merged.merge(&tiny);
                        normal_merged.merge(&normal);
                        check(&tiny_merged, &normal_merged);
                    }
                    let mut tiny_added = tiny;
                    let mut normal_added = normal;
                    tiny_added.add_weighted_many(&values, &weights).unwrap();
                    normal_added.add_weighted_many(&values, &units).unwrap();
                    check(&tiny_added, &normal_added);
                }
            }
        }
    };
}
scale_regressions!(f32_subnormal_compression_matches_normal_units, f32, F32);
scale_regressions!(f64_subnormal_compression_matches_normal_units, f64, F64);

#[test]
fn ordinary_mass_and_exact_subnormal_passthrough_are_unchanged() {
    let make = || {
        TDigest::<f64>::builder()
            .max_size(1)
            .singleton_policy(SingletonPolicy::Off)
            .build()
    };
    let normal = make()
        .merge_weighted_unsorted(&[0.0, 1.0, 2.0, 3.0], &[1.0; 4])
        .unwrap();
    assert_eq!(normal.centroids()[0].mean_f64(), 1.5);
    assert_eq!(normal.centroids()[0].weight_f64(), 4.0);
    assert_eq!(normal.sum(), 6.0);
    let tiny = TDigest::<f64>::builder()
        .max_size(4)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .merge_weighted_unsorted(&[-0.0, 1.0], &[f64::from_bits(1), f64::from_bits(2)])
        .unwrap();
    assert_eq!(
        tiny.centroids()[0].mean_f64().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(tiny.centroids()[1].mean_f64(), 1.0);
    assert!(tiny.centroids().iter().all(|c| c.is_atomic()));
    assert_eq!(tiny.centroids()[0].weight_f64().to_bits(), 1);
    assert_eq!(tiny.centroids()[1].weight_f64().to_bits(), 2);
}

#[test]
fn normalized_compression_does_not_introduce_large_value_overflow() {
    let values = [f64::MAX * 0.75, f64::MAX];
    let td = TDigest::<f64>::builder()
        .max_size(1)
        .singleton_policy(SingletonPolicy::Off)
        .build()
        .merge_weighted_unsorted(&values, &[f64::from_bits(1), f64::from_bits(2)])
        .unwrap();
    assert!(td.median().is_finite());
    assert!((values[0]..=values[1]).contains(&td.median()));
    assert!(td.sum().is_finite());
    assert_eq!(td.count().to_bits(), 3);
}
