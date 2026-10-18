//! Consuming an owned merge input preserves state, configuration and errors.
use gr_tdigest::tdigest::frontends::{FrontendDigest, FrontendError};
use gr_tdigest::tdigest::{ScaleFamily, TDigest};

fn reference(digests: Vec<FrontendDigest>) -> Result<FrontendDigest, FrontendError> {
    if digests.is_empty() {
        return Ok(FrontendDigest::canonical_empty());
    }
    let first = digests[0].clone();
    let mut acc = first.clone();
    for digest in digests.iter().skip(1) {
        acc.merge_in_place(digest)?;
    }
    Ok(acc)
}

#[test]
fn owned_merge_keeps_empty_adoption_order_and_errors() {
    let a = FrontendDigest::F64(
        TDigest::<f64>::builder()
            .max_size(64)
            .build()
            .merge_weighted_unsorted(&[-2.0, 0.0, 1.0, 3.0], &[0.125, 0.5, 1.0, 0.25])
            .unwrap(),
    );
    let empty64 = FrontendDigest::F64(TDigest::<f64>::default());
    let empty32 = FrontendDigest::F32(TDigest::<f32>::default());
    let mismatched = FrontendDigest::F64(
        TDigest::<f64>::builder()
            .max_size(17)
            .scale(ScaleFamily::K1)
            .build()
            .merge_unsorted(vec![0.0, 1.0])
            .unwrap(),
    );
    let cases = vec![
        vec![],
        vec![a.clone()],
        vec![empty64.clone()],
        vec![empty64.clone(), a.clone()],
        vec![a.clone(), empty64.clone()],
        vec![a.clone(); 40],
        vec![a.clone(), mismatched.clone()],
        vec![mismatched, a],
        vec![empty32.clone(), empty64.clone()],
        vec![empty32; 3],
    ];
    for digests in cases {
        let expected = reference(digests.clone());
        let actual = FrontendDigest::merge_all(digests);
        match (actual, expected) {
            (Ok(got), Ok(want)) => assert_eq!(got.to_bytes(), want.to_bytes()),
            (Err(got), Err(want)) => assert_eq!(got.to_string(), want.to_string()),
            (got, want) => panic!("merge behavior changed: {got:?} vs {want:?}"),
        }
    }
}
