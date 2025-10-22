pub fn assert_exact(label: &str, expected: f64, got: f64) {
    assert!(
        expected == got,
        "{}: expected exactly {:.9}, got {:.9}",
        label,
        expected,
        got
    );
}

pub fn assert_rel_close(label: &str, expected: f64, got: f64, rtol: f64) {
    let denom = expected.abs().max(1e-300);
    let rel = ((expected - got).abs()) / denom;
    assert!(
        rel < rtol,
        "{}: expected ~= {:.9}, got {:.9}, rel_err={:.6e}, rtol={:.6e}",
        label,
        expected,
        got,
        rel,
        rtol
    );
}

pub fn assert_abs_close(label: &str, expected: f64, got: f64, atol: f64) {
    let abs = (expected - got).abs();
    assert!(
        abs <= atol,
        "{}: expected ~= {:.9}, got {:.9}, abs_err={:.6e}, atol={:.6e}",
        label,
        expected,
        got,
        abs,
        atol
    );
}

pub fn assert_monotone_chain(label: &str, xs: &[f64]) {
    for i in 1..xs.len() {
        assert!(
            xs[i] >= xs[i - 1],
            "{}: non-monotone at i={}: {} < {}",
            label,
            i,
            xs[i],
            xs[i - 1]
        );
    }
}

pub fn assert_strict_increasing(label: &str, xs: &[f64]) {
    for i in 1..xs.len() {
        assert!(
            xs[i] > xs[i - 1],
            "{}: not strictly increasing at i={}: {} !> {}",
            label,
            i,
            xs[i],
            xs[i - 1]
        );
    }
}

pub fn assert_all_in_unit_interval(label: &str, xs: &[f64]) {
    for (i, &x) in xs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&x),
            "{}: value out of [0,1] at i={}: {}",
            label,
            i,
            x
        );
    }
}

pub fn ks_mae(exact: &[f64], approx: &[f64]) -> (f64, f64) {
    assert_eq!(exact.len(), approx.len(), "KS/MAE length mismatch");
    let mut ks = 0.0;
    let mut sum = 0.0;
    for i in 0..exact.len() {
        let d = (exact[i] - approx[i]).abs();
        if d > ks {
            ks = d;
        }
        sum += d;
    }
    (ks, sum / exact.len() as f64)
}

pub fn assert_in_bracket(label: &str, x: f64, lo: f64, hi: f64, i_lo: usize, i_hi: usize) {
    assert!(
        x >= lo && x <= hi,
        "{}: {} not in bracket [{}, {}] (i_lo={}, i_hi={})",
        label,
        x,
        lo,
        hi,
        i_lo,
        i_hi
    );
}

pub fn bracket(values: &[f64], q: f64) -> (f64, f64, usize, usize) {
    assert!(!values.is_empty(), "bracket() requires non-empty values");
    let n = values.len();
    let q = q.clamp(0.0, 1.0);
    let r = q * (n.saturating_sub(1) as f64);

    let i_lo = r.floor() as usize;
    let i_hi = r.ceil() as usize;

    (values[i_lo], values[i_hi], i_lo, i_hi)
}
