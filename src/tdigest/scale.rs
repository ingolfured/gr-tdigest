use serde::{Deserialize, Serialize};

/// Scale families define the q→k mapping that controls compression density.
///
/// **Used in Stage 3 (k-limit merge).**
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // accept "quad","k1","k2","k2norm","k3"
#[derive(Default)]
pub enum ScaleFamily {
    /// Piecewise-quadratic tail-friendly scale.
    Quad,
    /// k1: arcsine scale.
    K1,
    /// k2: logistic scale, no-norm variant `d / (4·ln 2) · ln(q/(1−q))` (DEFAULT).
    ///
    /// Matches Dunning's Java `K_2_NO_NORM` — n-independent normalizer.
    #[default]
    K2,
    /// k2 normalized: canonical paper formula `δ / (4·ln(n/δ) + 24) · ln(q/(1−q))`.
    ///
    /// Matches Dunning's Java `K_2` and the paper's eq (8). Used by `delta`-mode
    /// to reproduce the old tdigest-rs merge behavior. Requires `n` (the number
    /// of centroids in the current merge pass), which must be passed by the
    /// caller of `q_to_k`.
    K2Norm,
    /// k3: double-log scale.
    K3,
}

#[inline]
pub(crate) fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Per-compression-pass scale constants. Preparing them never reassociates
/// the q-dependent expression or changes its clamp/rounding order.
pub(crate) struct PreparedScale {
    family: ScaleFamily,
    factor: f64,
}

impl PreparedScale {
    #[inline]
    pub(crate) fn new(d: f64, family: ScaleFamily, n: f64) -> Self {
        use std::f64::consts::{LN_2, PI};
        let factor = match family {
            ScaleFamily::Quad => d,
            ScaleFamily::K1 => d / (2.0 * PI),
            ScaleFamily::K2 => d / (4.0 * LN_2),
            ScaleFamily::K2Norm => d / ((n / d).ln().mul_add(4.0, 24.0)),
            ScaleFamily::K3 => d / 4.0,
        };
        Self { family, factor }
    }

    #[inline]
    pub(crate) fn q_to_k(&self, q: f64) -> f64 {
        let eps = 1e-15;
        let qq = clamp(q, eps, 1.0 - eps);
        match self.family {
            ScaleFamily::Quad => {
                let r = if qq < 0.5 {
                    (qq * 0.5).sqrt()
                } else {
                    1.0 - ((1.0 - qq) * 0.5).sqrt()
                };
                self.factor * r
            }
            ScaleFamily::K1 => {
                let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
                self.factor * s
            }
            ScaleFamily::K2 | ScaleFamily::K2Norm => {
                let s = (qq / (1.0 - qq)).ln();
                self.factor * s
            }
            ScaleFamily::K3 => {
                let a = (1.0 / (1.0 - qq)).ln();
                let b = (1.0 / qq).ln();
                let ratio = (a / b).max(eps);
                self.factor * ratio.ln()
            }
        }
    }
}

/// Family-aware `q → k` mapping. `d` is the scale denominator (≈ `max_size`,
/// or `δ` in delta-mode). `n` is the centroid count of the current merge pass —
/// only consulted by [`ScaleFamily::K2Norm`]; other families ignore it.
///
/// This mapping shapes cluster budget along the distribution:
/// more resolution near tails for some scales, more uniform for others.
/// Consumed by Stage **3** to evaluate the Δk ≤ 1 condition.
// Unchanged pre-hoist expression retained only as a differential test oracle.
#[cfg(test)]
#[inline]
pub(crate) fn q_to_k(q: f64, d: f64, family: ScaleFamily, n: f64) -> f64 {
    use std::f64::consts::{LN_2, PI};
    let eps = 1e-15;
    let qq = clamp(q, eps, 1.0 - eps);
    match family {
        // Piecewise-quadratic
        ScaleFamily::Quad => {
            // Inverse of: r=k/d; q = 2r^2 (r<0.5), else 1-2(1-r)^2
            let r = if qq < 0.5 {
                (qq * 0.5).sqrt()
            } else {
                1.0 - ((1.0 - qq) * 0.5).sqrt()
            };
            d * r
        }
        // k1: arcsine scale
        ScaleFamily::K1 => {
            let s = (2.0 * qq - 1.0).clamp(-1.0, 1.0).asin();
            (d / (2.0 * PI)) * s
        }
        // k2: logistic, no-norm — n-independent.
        ScaleFamily::K2 => {
            let s = (qq / (1.0 - qq)).ln();
            (d / (4.0 * LN_2)) * s
        }
        // k2_norm: canonical Dunning K2 with n-aware normalizer.
        // Paper eq (8): k(q) = δ / (4·ln(n/δ) + 24) · ln(q/(1-q)).
        ScaleFamily::K2Norm => {
            let factor = d / ((n / d).ln().mul_add(4.0, 24.0));
            factor * (qq / (1.0 - qq)).ln()
        }
        // k3: double-log
        ScaleFamily::K3 => {
            let a = (1.0 / (1.0 - qq)).ln(); // ln(1/(1-q))
            let b = (1.0 / qq).ln(); // ln(1/q)
            let ratio = (a / b).max(eps);
            (d / 4.0) * ratio.ln()
        }
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;

    #[test]
    fn prepared_scale_preserves_reference_bits_at_extremes_and_clamp_boundaries() {
        let qs = [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            f64::from_bits(1),
            1e-16,
            1e-15,
            0.01,
            0.1,
            0.49999999999999994,
            0.5,
            0.5000000000000001,
            0.9,
            0.99,
            1.0 - 1e-15,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let ds = [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1e-12,
            0.01,
            1.0,
            25.0,
            300.0,
            1200.0,
            1e6,
            f64::MAX,
        ];
        for family in [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K2Norm,
            ScaleFamily::K3,
        ] {
            for d in ds {
                for n in [1.0, 2.0, 8.0, 1000.0, 1e9] {
                    let prepared = PreparedScale::new(d, family, n);
                    for q in qs.into_iter().chain((0..=1024).map(|i| i as f64 / 1024.0)) {
                        let actual = prepared.q_to_k(q);
                        let expected = q_to_k(q, d, family, n);
                        if expected.is_nan() {
                            assert!(actual.is_nan());
                        } else {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "family={family:?}, d={d}, n={n}, q={q}"
                            );
                        }
                    }
                }
            }
        }
    }
}
