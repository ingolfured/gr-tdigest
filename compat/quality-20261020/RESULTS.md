# Modern versus original G-Research: measured accuracy

### Which implementation wins?

- **Bike data: Modern wins all four error measures in every size/merge group below.**
- **Energy data: Modern wins average rank, tail rank and average value error. GR wins peak value error:** Modern's peak-value error is **1.81–1.86 times GR's**.
- **Synthetic data: GR wins all four measures on Gaussian, exponential, lognormal, Pareto, bimodal and signed-scale distributions when averaged across setups.** Modern wins all four on drift, skewed two-point and zero-inflated data. Rounded ties and uniform grids split by metric.

There is no single overall winner: Modern wins the real-data averages; GR handles Energy peak-value error and several synthetic distributions better. These are grouped results; individual cases can differ.

### How to read the numbers

**Every number is Modern error ÷ GR error. Lower error is better. GR is always the reference value 1.**

| Example | Meaning | Winner |
| --- | --- | --- |
| 0.25 | Modern has one quarter of GR's error | Modern |
| 1.00 | Equal error | Tie |
| 2.00 | Modern has twice GR's error | GR |

The tables name the winner explicitly, followed by this ratio. For example, **Modern (0.41)** means Modern has 41% of GR's error; **GR (1.84)** means Modern has 1.84 times GR's error.

### Real data: winner by size and merge setup

| Dataset | Size | Construction | Average rank error | Tail rank error | Average value error | Peak value error |
| --- | --- | --- | --- | --- | --- | --- |
| Bike | 32 | 40 sequential merges | Modern (0.45) | Modern (0.14) | Modern (0.38) | Modern (0.32) |
| Bike | 32 | 40-way balanced merge | Modern (0.46) | Modern (0.16) | Modern (0.41) | Modern (0.42) |
| Bike | 32 | Single build | Modern (0.41) | Modern (0.20) | Modern (0.38) | Modern (0.55) |
| Bike | 128 | 40 sequential merges | Modern (0.021) | Modern (0.047) | Modern (0.037) | Modern (0.34) |
| Bike | 128 | 40-way balanced merge | Modern (0.021) | Modern (0.030) | Modern (0.039) | Modern (0.35) |
| Bike | 128 | Single build | Modern (0.017) | Modern (0.030) | Modern (0.035) | Modern (0.35) |
| Bike | 512 | 40 sequential merges | Modern (0.010) | Modern (0) | Modern (0.029) | Modern (0.34) |
| Bike | 512 | 40-way balanced merge | Modern (0.010) | Modern (0) | Modern (0.028) | Modern (0.35) |
| Bike | 512 | Single build | Modern (0.008) | Modern (0) | Modern (0.028) | Modern (0.35) |
| Energy | 32 | 40 sequential merges | Modern (0.084) | Modern (0.013) | Modern (0.70) | GR (1.81) |
| Energy | 32 | 40-way balanced merge | Modern (0.078) | Modern (0.064) | Modern (0.67) | GR (1.81) |
| Energy | 32 | Single build | Modern (0.23) | Modern (0.12) | Modern (0.73) | GR (1.83) |
| Energy | 128 | 40 sequential merges | Modern (0.009) | Modern (0.002) | Modern (0.36) | GR (1.85) |
| Energy | 128 | 40-way balanced merge | Modern (0.009) | Modern (0.031) | Modern (0.35) | GR (1.84) |
| Energy | 128 | Single build | Modern (0.043) | Modern (0.037) | Modern (0.40) | GR (1.84) |
| Energy | 512 | 40 sequential merges | Modern (0.002) | Modern (0.00042) | Modern (0.32) | GR (1.86) |
| Energy | 512 | 40-way balanced merge | Modern (0.002) | Modern (0.004) | Modern (0.31) | GR (1.85) |
| Energy | 512 | Single build | Modern (0.008) | Modern (0.003) | Modern (0.37) | GR (1.84) |

Each row compares average per-case errors across the dataset's columns, both precisions and both input orders. “Peak value error” is the average of each case's maximum value error, not the single largest error across the group. Value errors are normalized by each column's IQR, with the test suite's fallback when IQR is zero. Tail rank measures probabilities ≤1% or ≥99%.

<details>
<summary>Real-data breakdown by f32/f64</summary>

Same ratio throughout: Modern error ÷ GR error; below 1 favors Modern, above 1 favors GR.

| Dataset | Precision | Size | Construction | Average rank error | Tail rank error | Average value error | Peak value error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bike | f32 | 32 | 40 sequential merges | Modern (0.48) | Modern (0.15) | Modern (0.40) | Modern (0.35) |
| Bike | f32 | 32 | 40-way balanced merge | Modern (0.48) | Modern (0.17) | Modern (0.43) | Modern (0.52) |
| Bike | f32 | 32 | Single build | Modern (0.42) | Modern (0.28) | Modern (0.39) | Modern (0.56) |
| Bike | f32 | 128 | 40 sequential merges | Modern (0.022) | Modern (0.048) | Modern (0.038) | Modern (0.35) |
| Bike | f32 | 128 | 40-way balanced merge | Modern (0.022) | Modern (0.032) | Modern (0.039) | Modern (0.36) |
| Bike | f32 | 128 | Single build | Modern (0.017) | Modern (0.030) | Modern (0.035) | Modern (0.34) |
| Bike | f32 | 512 | 40 sequential merges | Modern (0.011) | Modern (0) | Modern (0.029) | Modern (0.34) |
| Bike | f32 | 512 | 40-way balanced merge | Modern (0.011) | Modern (0) | Modern (0.029) | Modern (0.35) |
| Bike | f32 | 512 | Single build | Modern (0.008) | Modern (0) | Modern (0.029) | Modern (0.36) |
| Bike | f64 | 32 | 40 sequential merges | Modern (0.42) | Modern (0.14) | Modern (0.36) | Modern (0.29) |
| Bike | f64 | 32 | 40-way balanced merge | Modern (0.43) | Modern (0.14) | Modern (0.39) | Modern (0.35) |
| Bike | f64 | 32 | Single build | Modern (0.40) | Modern (0.16) | Modern (0.38) | Modern (0.54) |
| Bike | f64 | 128 | 40 sequential merges | Modern (0.021) | Modern (0.046) | Modern (0.036) | Modern (0.33) |
| Bike | f64 | 128 | 40-way balanced merge | Modern (0.021) | Modern (0.028) | Modern (0.038) | Modern (0.34) |
| Bike | f64 | 128 | Single build | Modern (0.017) | Modern (0.030) | Modern (0.034) | Modern (0.35) |
| Bike | f64 | 512 | 40 sequential merges | Modern (0.010) | Modern (0) | Modern (0.028) | Modern (0.34) |
| Bike | f64 | 512 | 40-way balanced merge | Modern (0.010) | Modern (0) | Modern (0.028) | Modern (0.34) |
| Bike | f64 | 512 | Single build | Modern (0.008) | Modern (0) | Modern (0.027) | Modern (0.34) |
| Energy | f32 | 32 | 40 sequential merges | Modern (0.091) | Modern (0.11) | Modern (0.75) | GR (1.80) |
| Energy | f32 | 32 | 40-way balanced merge | Modern (0.084) | Modern (0.076) | Modern (0.70) | GR (1.84) |
| Energy | f32 | 32 | Single build | Modern (0.23) | Modern (0.18) | Modern (0.76) | GR (1.86) |
| Energy | f32 | 128 | 40 sequential merges | Modern (0.010) | Modern (0.037) | Modern (0.38) | GR (1.85) |
| Energy | f32 | 128 | 40-way balanced merge | Modern (0.010) | Modern (0.045) | Modern (0.36) | GR (1.87) |
| Energy | f32 | 128 | Single build | Modern (0.045) | Modern (0.037) | Modern (0.42) | GR (1.86) |
| Energy | f32 | 512 | 40 sequential merges | Modern (0.002) | Modern (0.006) | Modern (0.34) | GR (1.85) |
| Energy | f32 | 512 | 40-way balanced merge | Modern (0.002) | Modern (0.006) | Modern (0.32) | GR (1.87) |
| Energy | f32 | 512 | Single build | Modern (0.009) | Modern (0.003) | Modern (0.39) | GR (1.86) |
| Energy | f64 | 32 | 40 sequential merges | Modern (0.077) | Modern (0.007) | Modern (0.65) | GR (1.83) |
| Energy | f64 | 32 | 40-way balanced merge | Modern (0.074) | Modern (0.056) | Modern (0.64) | GR (1.78) |
| Energy | f64 | 32 | Single build | Modern (0.22) | Modern (0.094) | Modern (0.70) | GR (1.81) |
| Energy | f64 | 128 | 40 sequential merges | Modern (0.008) | Modern (0.00087) | Modern (0.34) | GR (1.86) |
| Energy | f64 | 128 | 40-way balanced merge | Modern (0.008) | Modern (0.022) | Modern (0.34) | GR (1.82) |
| Energy | f64 | 128 | Single build | Modern (0.041) | Modern (0.037) | Modern (0.38) | GR (1.82) |
| Energy | f64 | 512 | 40 sequential merges | Modern (0.002) | Modern (0.00022) | Modern (0.30) | GR (1.87) |
| Energy | f64 | 512 | 40-way balanced merge | Modern (0.002) | Modern (0.003) | Modern (0.30) | GR (1.82) |
| Energy | f64 | 512 | Single build | Modern (0.008) | Modern (0.003) | Modern (0.35) | GR (1.82) |

</details>

### Synthetic data: winner by distribution

Averages across sizes 32/128/512, both precisions, both input orders and all three construction paths. Same ratio: Modern error ÷ GR error.

| Distribution | Average rank error | Tail rank error | Average value error | Peak value error |
| --- | --- | --- | --- | --- |
| bimodal | GR (1.25) | GR (1.20) | GR (1.37) | GR (1.16) |
| drift | Modern (0.80) | Modern (0) | Modern (0.80) | Modern (0.84) |
| exponential_stratified | GR (3.01) | GR (2.65) | GR (1.93) | GR (1.64) |
| gaussian_stratified | GR (3.10) | GR (3.04) | GR (2.02) | GR (1.64) |
| lognormal_stratified | GR (2.99) | GR (2.91) | GR (1.70) | GR (1.64) |
| pareto_stratified | GR (3.01) | GR (2.82) | GR (1.63) | GR (1.64) |
| rounded_ties | Modern (0.35) | Modern (0.23) | Modern (0.44) | GR (1.64) |
| signed_scale | GR (2.34) | GR (2.68) | GR (1.63) | GR (1.64) |
| skewed_two_point | Modern (0) | Modern (0) | Modern (0) | Modern (0) |
| uniform_grid | Modern (0.32) | Modern (0.61) | GR (1.01) | GR (1.64) |
| zero_inflated | Modern (0.004) | Modern (0.47) | Modern (0.007) | Modern (0.003) |

### Test basis

Results come from [CI run 104](https://github.com/ingolfured/gr-tdigest/actions/runs/39566448129) at `e6a14a2`; the [artifact contains every case and both engines' raw errors](https://github.com/ingolfured/gr-tdigest/actions/runs/39566448129/artifacts/12332712916). Both engines receive identical inputs and are compared at matched logical centroid-field byte budgets. All 324 real-data cases and 378/432 synthetic cases qualify; constant-data cases have no eligible comparison. The grouped ratios are ratios of average errors, not counts of wins. CDF has no GR comparison because its API does not expose CDF.

CI passes the regression gate, but 10 synthetic cases still exceed accuracy limits and 13/15 separate numerical boundary controls fail.
