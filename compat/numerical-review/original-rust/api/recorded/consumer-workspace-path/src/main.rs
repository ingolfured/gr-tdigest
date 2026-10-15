use tdigest_core::TDigest;
fn main() -> anyhow::Result<()> {
    let left = TDigest::<f64>::from_array(&[1.0, 2.0, 3.0], 300.0)?;
    let right = TDigest::<f64>::from_means_weights(&[2.0], &[3_u32], 300.0)?;
    let joined = left.merge(&right, 300.0)?;
    let _: &Vec<f64> = &joined.means;
    let _: &Vec<u32> = &joined.weights;
    assert_eq!(joined.weights.iter().sum::<u32>(), 6);
    assert_eq!(joined.quantile(0.5)?, 2.0);
    assert_eq!(joined.median()?, 2.0);
    assert_eq!(joined.trimmed_mean(0.0, 1.0)?, 2.0);
    assert_eq!(joined.n_zero_weights()?, 0);
    println!("original Rust crate imports and all seven methods passed");
    Ok(())
}
