//! Fast quality experiments for the experimental auto-scale selector.
//!
//! This suite is intentionally lightweight so we can iterate quickly:
//! - many small scenarios,
//! - broad distribution shapes,
//! - deterministic seeds.
//!
//! It compares auto-selection against all fixed scales and asserts aggregate
//! quality improvements over current fixed default behavior (`K2`).

use super::quality_base::{gen_dataset, DistKind};
use crate::tdigest::auto_scale::{
    evaluate_scale_for_sorted_with, select_scale_for_values_with, AutoScaleConfig, AutoScaleScore,
};
use crate::tdigest::ScaleFamily;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
enum ScenarioKind {
    Uniform,
    Normal,
    LogNormal05,
    LogNormal10,
    LogNormal20,
    Mixture,
    DiscreteSteps,
    SparseTwoPoint,
    TinyWithOutliers,
    TriModalTicks,
}

#[derive(Debug, Clone, Copy)]
struct CaseOutcome {
    scenario: ScenarioKind,
    seed: u64,
    n: usize,
    max_size: usize,
    auto_scale: ScaleFamily,
    auto: AutoScaleScore,
    sel_quad: AutoScaleScore,
    sel_k1: AutoScaleScore,
    sel_k2: AutoScaleScore,
    sel_k3: AutoScaleScore,
    quad: AutoScaleScore,
    k1: AutoScaleScore,
    k2: AutoScaleScore,
    k3: AutoScaleScore,
}

impl CaseOutcome {
    fn fixed(&self, s: ScaleFamily) -> AutoScaleScore {
        match s {
            ScaleFamily::Quad => self.quad,
            ScaleFamily::K1 => self.k1,
            ScaleFamily::K2 => self.k2,
            ScaleFamily::K3 => self.k3,
        }
    }

    fn selected(&self, s: ScaleFamily) -> AutoScaleScore {
        match s {
            ScaleFamily::Quad => self.sel_quad,
            ScaleFamily::K1 => self.sel_k1,
            ScaleFamily::K2 => self.sel_k2,
            ScaleFamily::K3 => self.sel_k3,
        }
    }

    fn best_fixed(self) -> (ScaleFamily, AutoScaleScore) {
        let mut best_s = ScaleFamily::Quad;
        let mut best = self.quad;
        for s in [ScaleFamily::K1, ScaleFamily::K2, ScaleFamily::K3] {
            let sc = self.fixed(s);
            if sc.objective < best.objective {
                best_s = s;
                best = sc;
            }
        }
        (best_s, best)
    }
}

fn scenario_values(kind: ScenarioKind, n: usize, seed: u64) -> Vec<f64> {
    match kind {
        ScenarioKind::Uniform => gen_dataset(DistKind::Uniform, n, seed),
        ScenarioKind::Normal => gen_dataset(DistKind::Normal, n, seed),
        ScenarioKind::LogNormal05 => gen_dataset(DistKind::LogNormal { sigma: 0.5 }, n, seed),
        ScenarioKind::LogNormal10 => gen_dataset(DistKind::LogNormal { sigma: 1.0 }, n, seed),
        ScenarioKind::LogNormal20 => gen_dataset(DistKind::LogNormal { sigma: 2.0 }, n, seed),
        ScenarioKind::Mixture => gen_dataset(DistKind::Mixture, n, seed),
        ScenarioKind::DiscreteSteps => {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..n)
                .map(|_| {
                    let step: i32 = rng.random_range(0..=40);
                    step as f64 / 40.0
                })
                .collect()
        }
        ScenarioKind::SparseTwoPoint => {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..n)
                .map(|_| if rng.random_bool(0.97) { 0.0 } else { 1.0 })
                .collect()
        }
        ScenarioKind::TinyWithOutliers => {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..n)
                .map(|_| {
                    if rng.random_bool(0.02) {
                        if rng.random_bool(0.5) {
                            rng.random_range(10.0..30.0)
                        } else {
                            rng.random_range(-30.0..-10.0)
                        }
                    } else {
                        1e-7 + rng.random_range(-1e-9..1e-9)
                    }
                })
                .collect()
        }
        ScenarioKind::TriModalTicks => {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..n)
                .map(|_| {
                    let center: f64 = match rng.random_range(0..3) {
                        0 => 0.1_f64,
                        1 => 0.5_f64,
                        _ => 0.9_f64,
                    };
                    let jitter: f64 = rng.random_range(-5.0e-4_f64..5.0e-4_f64);
                    ((center + jitter) * 10_000.0).round() / 10_000.0
                })
                .collect()
        }
    }
}

fn run_case(
    values: &[f64],
    n: usize,
    max_size: usize,
    select_cfg: AutoScaleConfig,
    eval_cfg: AutoScaleConfig,
    scenario: ScenarioKind,
    seed: u64,
) -> CaseOutcome {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let auto_scale = select_scale_for_values_with(values, max_size, select_cfg);
    let sel_quad = evaluate_scale_for_sorted_with(ScaleFamily::Quad, &sorted, max_size, select_cfg);
    let sel_k1 = evaluate_scale_for_sorted_with(ScaleFamily::K1, &sorted, max_size, select_cfg);
    let sel_k2 = evaluate_scale_for_sorted_with(ScaleFamily::K2, &sorted, max_size, select_cfg);
    let sel_k3 = evaluate_scale_for_sorted_with(ScaleFamily::K3, &sorted, max_size, select_cfg);
    let auto = evaluate_scale_for_sorted_with(auto_scale, &sorted, max_size, eval_cfg);
    let quad = evaluate_scale_for_sorted_with(ScaleFamily::Quad, &sorted, max_size, eval_cfg);
    let k1 = evaluate_scale_for_sorted_with(ScaleFamily::K1, &sorted, max_size, eval_cfg);
    let k2 = evaluate_scale_for_sorted_with(ScaleFamily::K2, &sorted, max_size, eval_cfg);
    let k3 = evaluate_scale_for_sorted_with(ScaleFamily::K3, &sorted, max_size, eval_cfg);

    CaseOutcome {
        scenario,
        seed,
        n,
        max_size,
        auto_scale,
        auto,
        sel_quad,
        sel_k1,
        sel_k2,
        sel_k3,
        quad,
        k1,
        k2,
        k3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, Default)]
    struct Aggregate {
        total: usize,
        sum_auto: f64,
        sum_quad: f64,
        sum_k1: f64,
        sum_k2: f64,
        sum_k3: f64,
        wins_vs_k2: usize,
        non_worse_vs_k2: usize,
        regret_sum: f64,
        k3_best_count: usize,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct PolicyAggregate {
        total: usize,
        sum_policy: f64,
        sum_k2: f64,
        sum_quad: f64,
        sum_k1: f64,
        sum_k3: f64,
        wins_vs_k2: usize,
        non_worse_vs_k2: usize,
        regret_sum: f64,
        switches_from_k2: usize,
    }

    impl PolicyAggregate {
        fn include(
            &mut self,
            chosen_scale: ScaleFamily,
            chosen_eval: AutoScaleScore,
            r: &CaseOutcome,
        ) {
            let eps = 1e-12;
            self.total += 1;
            self.sum_policy += chosen_eval.objective;
            self.sum_k2 += r.k2.objective;
            self.sum_quad += r.quad.objective;
            self.sum_k1 += r.k1.objective;
            self.sum_k3 += r.k3.objective;
            if chosen_eval.objective + eps < r.k2.objective {
                self.wins_vs_k2 += 1;
            }
            if chosen_eval.objective <= r.k2.objective + eps {
                self.non_worse_vs_k2 += 1;
            }
            if chosen_scale != ScaleFamily::K2 {
                self.switches_from_k2 += 1;
            }
            let (_, best_fixed) = r.best_fixed();
            self.regret_sum += (chosen_eval.objective - best_fixed.objective).max(0.0);
        }

        fn avg_policy(&self) -> f64 {
            self.sum_policy / self.total as f64
        }

        fn avg_k2(&self) -> f64 {
            self.sum_k2 / self.total as f64
        }

        fn avg_best_fixed_avg(&self) -> f64 {
            let total = self.total as f64;
            let avg_quad = self.sum_quad / total;
            let avg_k1 = self.sum_k1 / total;
            let avg_k2 = self.sum_k2 / total;
            let avg_k3 = self.sum_k3 / total;
            avg_quad.min(avg_k1).min(avg_k2).min(avg_k3)
        }

        fn print_summary(&self, label: &str) {
            let total = self.total as f64;
            let avg_policy = self.sum_policy / total;
            let avg_k2 = self.sum_k2 / total;
            let avg_regret = self.regret_sum / total;
            println!(
                "{label}: cases={} avg_policy={:.6e} avg_k2={:.6e} wins_vs_k2={}/{} non_worse_vs_k2={}/{} avg_regret={:.6e} switches_from_k2={}/{}",
                self.total,
                avg_policy,
                avg_k2,
                self.wins_vs_k2,
                self.total,
                self.non_worse_vs_k2,
                self.total,
                avg_regret,
                self.switches_from_k2,
                self.total
            );
        }
    }

    #[inline]
    fn local_scale_rank(scale: ScaleFamily) -> usize {
        match scale {
            ScaleFamily::K2 => 0,
            ScaleFamily::K3 => 1,
            ScaleFamily::K1 => 2,
            ScaleFamily::Quad => 3,
        }
    }

    fn select_with_margins(r: &CaseOutcome, rel_margin: f64, abs_margin: f64) -> ScaleFamily {
        let mut best_scale = ScaleFamily::K2;
        let mut best_sel = r.sel_k2;
        for s in [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K3,
        ] {
            let sel = r.selected(s);
            let strictly_better = sel.objective < best_sel.objective;
            let tie_and_preferred = sel.objective == best_sel.objective
                && local_scale_rank(s) < local_scale_rank(best_scale);
            if strictly_better || tie_and_preferred {
                best_scale = s;
                best_sel = sel;
            }
        }

        if best_scale == ScaleFamily::K2 {
            return ScaleFamily::K2;
        }
        let k2_sel = r.sel_k2.objective;
        let required = (k2_sel * rel_margin).max(abs_margin);
        if k2_sel - best_sel.objective > required {
            best_scale
        } else {
            ScaleFamily::K2
        }
    }

    fn aggregate_policy(rows: &[CaseOutcome], rel_margin: f64, abs_margin: f64) -> PolicyAggregate {
        let mut out = PolicyAggregate::default();
        for r in rows {
            let chosen_scale = select_with_margins(r, rel_margin, abs_margin);
            let chosen_eval = r.fixed(chosen_scale);
            out.include(chosen_scale, chosen_eval, r);
        }
        out
    }

    impl Aggregate {
        fn include(&mut self, r: &CaseOutcome) {
            let eps = 1e-12;
            self.total += 1;
            self.sum_auto += r.auto.objective;
            self.sum_quad += r.quad.objective;
            self.sum_k1 += r.k1.objective;
            self.sum_k2 += r.k2.objective;
            self.sum_k3 += r.k3.objective;
            if r.auto.objective + eps < r.k2.objective {
                self.wins_vs_k2 += 1;
            }
            if r.auto.objective <= r.k2.objective + eps {
                self.non_worse_vs_k2 += 1;
            }
            let (best_s, best_fixed) = r.best_fixed();
            if best_s == ScaleFamily::K3 {
                self.k3_best_count += 1;
            }
            self.regret_sum += (r.auto.objective - best_fixed.objective).max(0.0);
        }

        fn print_summary(&self, label: &str) {
            if self.total == 0 {
                println!("{label}: total=0");
                return;
            }
            let total_f = self.total as f64;
            let avg_auto = self.sum_auto / total_f;
            let avg_quad = self.sum_quad / total_f;
            let avg_k1 = self.sum_k1 / total_f;
            let avg_k2 = self.sum_k2 / total_f;
            let avg_k3 = self.sum_k3 / total_f;
            let best_fixed_avg = avg_quad.min(avg_k1).min(avg_k2).min(avg_k3);
            let avg_regret = self.regret_sum / total_f;
            println!(
                "{label}: cases={} avg_auto={:.6e} avg_quad={:.6e} avg_k1={:.6e} avg_k2={:.6e} avg_k3={:.6e} best_fixed_avg={:.6e} wins_vs_k2={}/{} non_worse_vs_k2={}/{} avg_regret={:.6e} k3_best_count={}",
                self.total,
                avg_auto,
                avg_quad,
                avg_k1,
                avg_k2,
                avg_k3,
                best_fixed_avg,
                self.wins_vs_k2,
                self.total,
                self.non_worse_vs_k2,
                self.total,
                avg_regret,
                self.k3_best_count
            );
        }
    }

    fn run_matrix(
        scenarios: &[ScenarioKind],
        seeds: &[u64],
        ns: &[usize],
        max_sizes: &[usize],
        select_cfg: AutoScaleConfig,
        eval_cfg: AutoScaleConfig,
    ) -> Vec<CaseOutcome> {
        let mut rows = Vec::new();
        for &n in ns {
            for &scenario in scenarios {
                for &seed in seeds {
                    let values = scenario_values(scenario, n, seed);
                    for &max_size in max_sizes {
                        rows.push(run_case(
                            &values, n, max_size, select_cfg, eval_cfg, scenario, seed,
                        ));
                    }
                }
            }
        }
        rows
    }

    #[test]
    fn auto_scale_fast_matrix_beats_fixed_k2_and_tracks_best_static() {
        let scenarios = [
            ScenarioKind::Uniform,
            ScenarioKind::Normal,
            ScenarioKind::LogNormal05,
            ScenarioKind::LogNormal10,
            ScenarioKind::LogNormal20,
            ScenarioKind::Mixture,
            ScenarioKind::DiscreteSteps,
            ScenarioKind::SparseTwoPoint,
            ScenarioKind::TinyWithOutliers,
            ScenarioKind::TriModalTicks,
        ];
        let seeds = [7_u64, 29_u64, 113_u64, 997_u64];
        let ns = [2_200usize];
        let max_sizes = [64_usize, 128_usize, 1000_usize];

        let select_cfg = AutoScaleConfig {
            full_eval_len: 4096,
            pilot_sample_size: 768,
            cdf_grid_points: 256,
            ..AutoScaleConfig::default()
        };
        let eval_cfg = AutoScaleConfig {
            full_eval_len: ns[0],
            pilot_sample_size: ns[0],
            cdf_grid_points: 256,
            ..AutoScaleConfig::default()
        };

        let rows = run_matrix(&scenarios, &seeds, &ns, &max_sizes, select_cfg, eval_cfg);

        let total = rows.len() as f64;
        let mut sum_auto = 0.0;
        let mut sum_quad = 0.0;
        let mut sum_k1 = 0.0;
        let mut sum_k2 = 0.0;
        let mut sum_k3 = 0.0;
        let mut wins_vs_k2 = 0usize;
        let mut non_worse_vs_k2 = 0usize;
        let mut regret_sum = 0.0;
        let mut k3_best_count = 0usize;
        let eps = 1e-12;
        let mut hard_case_count = 0usize;
        let mut hard_wins_vs_k2 = 0usize;

        for r in &rows {
            sum_auto += r.auto.objective;
            sum_quad += r.quad.objective;
            sum_k1 += r.k1.objective;
            sum_k2 += r.k2.objective;
            sum_k3 += r.k3.objective;
            if r.auto.objective < r.k2.objective {
                wins_vs_k2 += 1;
            }
            if r.auto.objective <= r.k2.objective + eps {
                non_worse_vs_k2 += 1;
            }
            let (best_s, best_fixed) = r.best_fixed();
            if best_s == ScaleFamily::K3 {
                k3_best_count += 1;
            }
            regret_sum += (r.auto.objective - best_fixed.objective).max(0.0);
            let worst_fixed = r
                .quad
                .objective
                .max(r.k1.objective)
                .max(r.k2.objective)
                .max(r.k3.objective);
            let fixed_spread = worst_fixed - best_fixed.objective;
            if fixed_spread > 3e-4 {
                hard_case_count += 1;
                if r.auto.objective + eps < r.k2.objective {
                    hard_wins_vs_k2 += 1;
                }
            }

            println!(
                "[auto-scale case] scenario={:?} seed={} n={} k={} -> auto={:?} obj(auto)={:.6e} obj(best_fixed:{:?})={:.6e}",
                r.scenario,
                r.seed,
                r.n,
                r.max_size,
                r.auto_scale,
                r.auto.objective,
                best_s,
                best_fixed.objective
            );
        }

        let avg_auto = sum_auto / total;
        let avg_quad = sum_quad / total;
        let avg_k1 = sum_k1 / total;
        let avg_k2 = sum_k2 / total;
        let avg_k3 = sum_k3 / total;
        let best_fixed_avg = avg_quad.min(avg_k1).min(avg_k2).min(avg_k3);
        let avg_regret = regret_sum / total;

        println!(
            "auto-scale summary: cases={} avg_auto={:.6e} avg_quad={:.6e} avg_k1={:.6e} avg_k2={:.6e} avg_k3={:.6e} best_fixed_avg={:.6e} wins_vs_k2={}/{} non_worse_vs_k2={}/{} avg_regret={:.6e} k3_best_count={}",
            rows.len(),
            avg_auto,
            avg_quad,
            avg_k1,
            avg_k2,
            avg_k3,
            best_fixed_avg,
            wins_vs_k2,
            rows.len(),
            non_worse_vs_k2,
            rows.len(),
            avg_regret,
            k3_best_count
        );
        println!(
            "auto-scale hard-cases: threshold_spread>3e-4 count={} wins_vs_k2={}/{}",
            hard_case_count, hard_wins_vs_k2, hard_case_count
        );

        // "Before" baseline is fixed K2.
        assert!(
            avg_auto < avg_k2 * 0.97,
            "auto should improve avg objective vs fixed K2 (auto={avg_auto:.6e}, k2={avg_k2:.6e})"
        );
        assert!(
            wins_vs_k2 * 100 >= rows.len() * 20,
            "auto should produce strict wins over K2 in at least 20% of scenarios (got {wins_vs_k2}/{})",
            rows.len()
        );
        assert!(
            non_worse_vs_k2 * 100 >= rows.len() * 90,
            "auto should be non-worse than K2 in at least 90% of scenarios (got {non_worse_vs_k2}/{})",
            rows.len()
        );
        assert!(
            avg_auto < avg_k3 * 0.99,
            "auto should improve avg objective vs fixed K3 (auto={avg_auto:.6e}, k3={avg_k3:.6e})"
        );
        assert!(hard_case_count >= 1, "hard-case subset should be non-empty");
        assert!(
            hard_wins_vs_k2 * 100 >= hard_case_count * 40,
            "auto should beat K2 on at least 40% of hard cases (got {hard_wins_vs_k2}/{hard_case_count})"
        );

        // Critique the "always K3" assumption.
        assert!(
            k3_best_count * 100 <= rows.len() * 70,
            "K3 should not dominate all/most scenarios in this matrix (k3_best_count={k3_best_count}, total={})",
            rows.len()
        );

        // Auto can have regret vs oracle-best fixed, but keep it bounded.
        assert!(
            avg_auto <= best_fixed_avg * 1.08,
            "auto average objective drifted too far from best fixed average (auto={avg_auto:.6e}, best_fixed_avg={best_fixed_avg:.6e})"
        );
        assert!(
            avg_regret <= best_fixed_avg * 0.20,
            "auto average regret too high (avg_regret={avg_regret:.6e}, best_fixed_avg={best_fixed_avg:.6e})"
        );
    }

    #[test]
    #[ignore = "exploratory long-running quality sweep"]
    fn auto_scale_exploratory_large_matrix_report() {
        let scenarios = [
            ScenarioKind::Uniform,
            ScenarioKind::Normal,
            ScenarioKind::LogNormal05,
            ScenarioKind::LogNormal10,
            ScenarioKind::LogNormal20,
            ScenarioKind::Mixture,
            ScenarioKind::DiscreteSteps,
            ScenarioKind::SparseTwoPoint,
            ScenarioKind::TinyWithOutliers,
            ScenarioKind::TriModalTicks,
        ];
        let seed_count = std::env::var("AUTO_SCALE_SWEEP_SEEDS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(32);
        let seeds: Vec<u64> = (0..seed_count).map(|i| 10_000 + i).collect();
        println!("exploratory config: seed_count={seed_count}");
        let ns = [512_usize, 2_200_usize, 10_000_usize];
        let max_sizes = [
            32_usize, 64_usize, 128_usize, 256_usize, 512_usize, 1000_usize,
        ];

        let select_cfg = AutoScaleConfig {
            full_eval_len: 4_096,
            pilot_sample_size: 768,
            cdf_grid_points: 96,
            ..AutoScaleConfig::default()
        };
        let eval_cfg = AutoScaleConfig {
            full_eval_len: 10_000,
            pilot_sample_size: 10_000,
            cdf_grid_points: 256,
            ..AutoScaleConfig::default()
        };

        let rows = run_matrix(
            &scenarios,
            seeds.as_slice(),
            &ns,
            &max_sizes,
            select_cfg,
            eval_cfg,
        );
        let mut overall = Aggregate::default();
        for r in &rows {
            overall.include(r);
        }
        overall.print_summary("exploratory overall");

        for &n in &ns {
            let mut by_n = Aggregate::default();
            for r in rows.iter().filter(|r| r.n == n) {
                by_n.include(r);
            }
            by_n.print_summary(&format!("exploratory n={n}"));
        }

        for &k in &max_sizes {
            let mut by_k = Aggregate::default();
            for r in rows.iter().filter(|r| r.max_size == k) {
                by_k.include(r);
            }
            by_k.print_summary(&format!("exploratory max_size={k}"));
        }

        let rel_grid = [0.0_f64, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05];
        let abs_grid = [
            0.0_f64, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4,
        ];
        let mut best_cfg = (0.0_f64, 0.0_f64);
        let mut best_agg = aggregate_policy(&rows, best_cfg.0, best_cfg.1);
        for &rel in &rel_grid {
            for &abs in &abs_grid {
                let agg = aggregate_policy(&rows, rel, abs);
                let avg_policy = agg.avg_policy();
                let avg_k2 = agg.avg_k2();
                let non_worse_ok = agg.non_worse_vs_k2 * 100 >= agg.total * 90;
                let gain_ok = avg_policy < avg_k2 * 0.99;
                if non_worse_ok && gain_ok && avg_policy < best_agg.avg_policy() {
                    best_cfg = (rel, abs);
                    best_agg = agg;
                }
            }
        }
        best_agg.print_summary(&format!(
            "exploratory best_margin rel={:.3}% abs={:.1e}",
            best_cfg.0 * 100.0,
            best_cfg.1
        ));
        println!(
            "exploratory best_margin details: avg_best_fixed_avg={:.6e}",
            best_agg.avg_best_fixed_avg()
        );
    }
}
