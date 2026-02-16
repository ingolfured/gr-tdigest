//! Experimental auto-selection of TDigest scale family.
//!
//! This module provides a deterministic, opt-in helper that picks one of the
//! existing scale families (`Quad`, `K1`, `K2`, `K3`) using a small pilot
//! sample and a workload-weighted quality objective.
//!
//! Design goals for phase 1:
//! - Keep runtime low enough for quick experimentation.
//! - Keep behavior deterministic for reproducibility.
//! - Bias objective toward CDF quality and p50 quality.
//! - Do not change defaults; callers must opt in explicitly.
#![allow(dead_code)]

use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest};

/// Candidate scale families considered by auto-selection.
pub(crate) const SCALE_CANDIDATES: [ScaleFamily; 4] = [
    ScaleFamily::Quad,
    ScaleFamily::K1,
    ScaleFamily::K2,
    ScaleFamily::K3,
];

/// Tuning knobs for the auto-scale selector.
#[derive(Debug, Clone, Copy)]
pub struct AutoScaleConfig {
    /// If sorted length is <= this, score on full data (no pilot subsample).
    pub full_eval_len: usize,
    /// Max number of sorted values used for pilot scoring.
    pub pilot_sample_size: usize,
    /// Number of linear probes for CDF error scoring.
    pub cdf_grid_points: usize,
    /// Weight on CDF MAE in objective.
    pub w_cdf_mae: f64,
    /// Weight on CDF worst-case absolute error in objective.
    pub w_cdf_max: f64,
    /// Weight on p50 absolute error in objective.
    pub w_q50_abs: f64,
    /// Weight on quantile-MAE in objective.
    pub w_q_mae: f64,
    /// Tie epsilon for objective comparisons.
    pub tie_epsilon: f64,
    /// Relative improvement required to switch away from `K2`.
    pub k2_guard_rel: f64,
    /// Absolute objective improvement required to switch away from `K2`.
    pub k2_guard_abs: f64,
}

impl Default for AutoScaleConfig {
    fn default() -> Self {
        Self {
            full_eval_len: 4096,
            pilot_sample_size: 1024,
            cdf_grid_points: 128,
            // CDF dominates, then worst-case CDF, then p50 and quantile MAE.
            w_cdf_mae: 0.45,
            w_cdf_max: 0.35,
            w_q50_abs: 0.15,
            w_q_mae: 0.05,
            tie_epsilon: 1e-12,
            // Guarded switching keeps auto from taking tiny/noisy wins away from K2.
            k2_guard_rel: 0.0,
            k2_guard_abs: 5e-5,
        }
    }
}

/// Detailed quality metrics for one scale under one dataset/config.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoScaleScore {
    pub objective: f64,
    pub cdf_mae: f64,
    pub cdf_max: f64,
    pub q_mae: f64,
    pub q50_abs: f64,
}

impl AutoScaleScore {
    #[inline]
    fn from_parts(
        cdf_mae: f64,
        cdf_max: f64,
        q_mae: f64,
        q50_abs: f64,
        cfg: AutoScaleConfig,
    ) -> Self {
        let objective = cfg.w_cdf_mae * cdf_mae
            + cfg.w_cdf_max * cdf_max
            + cfg.w_q50_abs * q50_abs
            + cfg.w_q_mae * q_mae;
        Self {
            objective,
            cdf_mae,
            cdf_max,
            q_mae,
            q50_abs,
        }
    }
}

/// Select a scale family from unsorted values with explicit selector config.
pub(crate) fn select_scale_for_values_with(
    values: &[f64],
    max_size: usize,
    cfg: AutoScaleConfig,
) -> ScaleFamily {
    if values.len() <= 2 || max_size <= 2 {
        return ScaleFamily::K2;
    }
    if values.iter().any(|x| !x.is_finite()) {
        return ScaleFamily::K2;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    select_scale_for_sorted_with(&sorted, max_size, cfg)
}

/// Select a scale family from sorted values with explicit selector config.
pub(crate) fn select_scale_for_sorted_with(
    sorted: &[f64],
    max_size: usize,
    cfg: AutoScaleConfig,
) -> ScaleFamily {
    if sorted.len() <= 2 || max_size <= 2 {
        return ScaleFamily::K2;
    }
    if sorted.iter().any(|x| !x.is_finite()) {
        return ScaleFamily::K2;
    }

    let eval_sorted: Vec<f64> = if sorted.len() <= cfg.full_eval_len.max(8) {
        sorted.to_vec()
    } else {
        evenly_spaced_sorted_sample(sorted, cfg.pilot_sample_size.max(8))
    };

    let mut best_scale = ScaleFamily::K2;
    let mut best_score = f64::INFINITY;
    let mut k2_score = f64::INFINITY;
    for &scale in &SCALE_CANDIDATES {
        let score = evaluate_scale_for_sorted_with(scale, &eval_sorted, max_size, cfg);
        if !score.objective.is_finite() {
            continue;
        }
        if scale == ScaleFamily::K2 {
            k2_score = score.objective;
        }
        let strictly_better = score.objective + cfg.tie_epsilon < best_score;
        let tie_and_preferred = (score.objective - best_score).abs() <= cfg.tie_epsilon
            && scale_preference_rank(scale) < scale_preference_rank(best_scale);
        if strictly_better || tie_and_preferred {
            best_scale = scale;
            best_score = score.objective;
        }
    }
    if best_scale != ScaleFamily::K2 && k2_score.is_finite() && best_score.is_finite() {
        let required_gain = (k2_score * cfg.k2_guard_rel.max(0.0)).max(cfg.k2_guard_abs.max(0.0));
        let observed_gain = k2_score - best_score;
        if observed_gain <= required_gain + cfg.tie_epsilon {
            return ScaleFamily::K2;
        }
    }
    best_scale
}

/// Score one fixed scale against sorted values.
pub(crate) fn evaluate_scale_for_sorted_with(
    scale: ScaleFamily,
    sorted: &[f64],
    max_size: usize,
    cfg: AutoScaleConfig,
) -> AutoScaleScore {
    if sorted.is_empty() {
        return AutoScaleScore::default();
    }
    if sorted.iter().any(|x| !x.is_finite()) {
        return AutoScaleScore {
            objective: f64::INFINITY,
            ..AutoScaleScore::default()
        };
    }

    let td = TDigest::<f64>::builder()
        .max_size(max_size.max(1))
        .scale(scale)
        .singleton_policy(SingletonPolicy::Use)
        .build()
        .merge_sorted(sorted.to_vec())
        .expect("auto-scale scoring expects finite sorted values");

    let q_grid: [f64; 13] = [
        0.01, 0.05, 0.10, 0.20, 0.35, 0.45, 0.50, 0.55, 0.65, 0.80, 0.90, 0.95, 0.99,
    ];

    let mut q_err_sum = 0.0;
    let mut q50_abs = 0.0;
    for &q in &q_grid {
        let est = td.quantile(q);
        let exp = expected_quantile(sorted, q);
        let e = (est - exp).abs();
        if q == 0.50 {
            q50_abs = e;
        }
        q_err_sum += e;
    }
    let q_mae = q_err_sum / q_grid.len() as f64;

    let probes = cdf_probe_grid(sorted, cfg.cdf_grid_points.max(16));
    let cdf_est = td.cdf(&probes);
    let mut cdf_abs_sum = 0.0;
    let mut cdf_max = 0.0;
    for (&x, &est) in probes.iter().zip(cdf_est.iter()) {
        let exp = exact_midpoint_cdf_at(sorted, x);
        let e = (est - exp).abs();
        cdf_abs_sum += e;
        if e > cdf_max {
            cdf_max = e;
        }
    }
    let cdf_mae = cdf_abs_sum / probes.len() as f64;

    AutoScaleScore::from_parts(cdf_mae, cdf_max, q_mae, q50_abs, cfg)
}

#[inline]
fn scale_preference_rank(scale: ScaleFamily) -> usize {
    // Tie-break prefers status quo first for stability.
    match scale {
        ScaleFamily::K2 => 0,
        ScaleFamily::K3 => 1,
        ScaleFamily::K1 => 2,
        ScaleFamily::Quad => 3,
    }
}

fn evenly_spaced_sorted_sample(sorted: &[f64], target: usize) -> Vec<f64> {
    if sorted.len() <= target {
        return sorted.to_vec();
    }
    if target <= 1 {
        return vec![sorted[0]];
    }
    let last = sorted.len() - 1;
    let mut out = Vec::with_capacity(target);
    for i in 0..target {
        let idx = i * last / (target - 1);
        out.push(sorted[idx]);
    }
    out
}

fn cdf_probe_grid(sorted: &[f64], grid_points: usize) -> Vec<f64> {
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    if lo == hi {
        return vec![lo; grid_points + 1];
    }
    (0..=grid_points)
        .map(|i| lo + (i as f64) * (hi - lo) / (grid_points as f64))
        .collect()
}

fn expected_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return sorted[n - 1];
    }
    let t = q * (n as f64 - 1.0);
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let alpha = t - lo as f64;
        (1.0 - alpha) * sorted[lo] + alpha * sorted[hi]
    }
}

fn exact_midpoint_cdf_at(sorted: &[f64], x: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let lt = sorted.partition_point(|v| *v < x);
    let le = sorted.partition_point(|v| *v <= x);
    let eq = le - lt;
    (lt as f64 + 0.5 * eq as f64) / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use testdata::{gen_dataset, DistKind};

    #[test]
    fn auto_scale_selection_is_deterministic_for_same_input() {
        let values = gen_dataset(DistKind::Mixture, 2_500, 4242);
        let cfg = AutoScaleConfig::default();
        let a = select_scale_for_values_with(&values, 1_000, cfg);
        let b = select_scale_for_values_with(&values, 1_000, cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn auto_scale_falls_back_to_k2_for_tiny_or_nonfinite_input() {
        let cfg = AutoScaleConfig::default();
        assert_eq!(
            select_scale_for_values_with(&[], 1_000, cfg),
            ScaleFamily::K2
        );
        assert_eq!(
            select_scale_for_values_with(&[1.0], 1_000, cfg),
            ScaleFamily::K2
        );
        assert_eq!(
            select_scale_for_values_with(&[1.0, f64::NAN, 2.0], 1_000, cfg),
            ScaleFamily::K2
        );
    }
}
