//! Overflow guards around the pinned original arithmetic. In-range calculations
//! still execute the original functions in their original precision and order.

use anyhow::{ensure, Result};
use num::Float;

use super::traits::{FloatConst, TotalOrd};

fn check_mass<'a>(weights: impl IntoIterator<Item = &'a u32>) -> Result<u32> {
    weights.into_iter().try_fold(0_u32, |total, weight| {
        total
            .checked_add(*weight)
            .ok_or_else(|| anyhow::anyhow!("tdigest: uint32 total weight overflow"))
    })
}

fn check_means<T: Float>(input_finite: bool, output: &[T]) -> Result<()> {
    // Explicit nonfinite observations retain the original API's semantics.
    ensure!(
        !input_finite || output.iter().all(|m| m.is_finite()),
        "tdigest: centroid arithmetic overflow"
    );
    Ok(())
}

fn check_clusters<T: Float>(
    input_finite: bool,
    input_mass: u32,
    means: &[T],
    weights: &[u32],
) -> Result<()> {
    check_means(input_finite, means)?;
    // The original kernel can drop its final cluster after inf + -inf becomes
    // NaN. A finite output list alone would miss that lost mass.
    ensure!(
        !input_finite || check_mass(weights)? == input_mass,
        "tdigest: centroid arithmetic overflow lost mass"
    );
    Ok(())
}

pub(super) fn create_clusters<T>(
    means: &[T],
    weights: &[u32],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst + TotalOrd<T>,
{
    let mass = check_mass(weights)?;
    let result = super::core::create_clusters(means, weights, delta)?;
    check_clusters(
        means.iter().all(|m| m.is_finite()),
        mass,
        &result.0,
        &result.1,
    )?;
    Ok(result)
}

pub(super) fn compute<T>(
    means: &[T],
    weights: &[u32],
    mask: &[bool],
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst,
{
    let mass = check_mass(weights)?;
    let result = super::core::compute(means, weights, mask, delta)?;
    check_clusters(
        means.iter().all(|m| m.is_finite()),
        mass,
        &result.0,
        &result.1,
    )?;
    Ok(result)
}

pub(super) fn merge_clusters<T>(
    means1: &Vec<T>,
    weights1: &Vec<u32>,
    means2: &Vec<T>,
    weights2: &Vec<u32>,
    delta: T,
) -> Result<(Vec<T>, Vec<u32>, Vec<bool>)>
where
    T: Float + FloatConst,
{
    let mass = check_mass(weights1.iter().chain(weights2))?;
    let result = super::core::merge_clusters(means1, weights1, means2, weights2, delta)?;
    check_clusters(
        means1.iter().chain(means2).all(|m| m.is_finite()),
        mass,
        &result.0,
        &result.1,
    )?;
    Ok(result)
}

pub(super) fn compute_quantile<T>(means: &[T], weights: &[u32], q: T) -> Result<T>
where
    T: Float + FloatConst,
{
    check_mass(weights)?;
    let result = super::core::compute_quantile(means, weights, q)?;
    check_means(
        q.is_finite() && !means.is_empty() && means.iter().all(|m| m.is_finite()),
        &[result],
    )?;
    Ok(result)
}

pub(super) fn compute_trimmed_mean<T>(means: &[T], weights: &[u32], lower: T, upper: T) -> Result<T>
where
    T: Float + FloatConst,
{
    check_mass(weights)?;
    let result = super::core::compute_trimmed_mean(means, weights, lower, upper)?;
    // Empty/zero-width selections historically return NaN. Preserve those and
    // permissive probes; guard the ordinary nonempty interval against overflow.
    let ordinary = lower.is_finite()
        && upper.is_finite()
        && lower >= T::zero()
        && lower < upper
        && upper <= T::one()
        && weights.iter().any(|w| *w > 0);
    check_means(ordinary && means.iter().all(|m| m.is_finite()), &[result])?;
    Ok(result)
}
