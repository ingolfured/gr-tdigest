// src/tdigest/frontends.rs
//! Small, shared parsing & normalization helpers for all front-ends (Python, Polars, JNI, CLI).
//! Keep this dependency-light and free of PyO3/JNI/Polars types.

use std::fmt::{Display, Formatter};

use crate::tdigest::{scale::ScaleFamily, singleton_policy::SingletonPolicy};

#[derive(Debug, Clone)]
pub enum ParseError {
    InvalidScale(String),
    InvalidPolicy(String),
    MissingEdges,        // 'edges' requires pin_per_side
    InvalidEdges(usize), // pin_per_side < 1
    InvalidPrecision(String),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidScale(s) => write!(
                f,
                "invalid scale: {s} (expected 'quad', 'k1', 'k2', or 'k3')"
            ),
            ParseError::InvalidPolicy(s) => write!(
                f,
                "invalid singleton policy: {s} (expected 'off', 'use', or 'edges')"
            ),
            ParseError::MissingEdges => {
                write!(f, "singleton_policy='edges' requires pin_per_side (>= 1)")
            }
            ParseError::InvalidEdges(k) => write!(
                f,
                "pin_per_side must be >= 1 when singleton_policy='edges' (got {k})"
            ),
            ParseError::InvalidPrecision(s) => write!(
                f,
                "unknown precision={s:?}; expected 'auto', 'f32' or 'f64'"
            ),
        }
    }
}
impl std::error::Error for ParseError {}

/// Lowercase/normalize a free-form string by removing `_` and spaces.
#[inline]
fn norm(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace(['_', ' '], "")
}

/* ----------------------- scale helpers ----------------------- */

pub fn parse_scale_str(raw: Option<&str>) -> Result<ScaleFamily, ParseError> {
    match raw.map(|x| x.trim().to_ascii_lowercase()) {
        None => Ok(ScaleFamily::K2),
        Some(ref v) if v == "quad" => Ok(ScaleFamily::Quad),
        Some(ref v) if v == "k1" => Ok(ScaleFamily::K1),
        Some(ref v) if v == "k2" => Ok(ScaleFamily::K2),
        Some(ref v) if v == "k3" => Ok(ScaleFamily::K3),
        Some(v) => Err(ParseError::InvalidScale(v)),
    }
}

pub fn scale_to_str(s: ScaleFamily) -> &'static str {
    match s {
        ScaleFamily::Quad => "quad",
        ScaleFamily::K1 => "k1",
        ScaleFamily::K2 => "k2",
        ScaleFamily::K3 => "k3",
    }
}

/* ----------------- singleton policy helpers ----------------- */

/// Accepts: off | use | edges | (legacy) usewithprotectededges
pub fn parse_singleton_policy_str(
    kind: Option<&str>,
    pin_per_side: Option<usize>,
) -> Result<SingletonPolicy, ParseError> {
    match kind.map(norm) {
        None => Ok(SingletonPolicy::Use),
        Some(ref v) if v == "off" => Ok(SingletonPolicy::Off),
        Some(ref v) if v == "use" => Ok(SingletonPolicy::Use),
        Some(ref v) if v == "edges" || v == "usewithprotectededges" => {
            let k = pin_per_side.ok_or(ParseError::MissingEdges)?;
            if k < 1 {
                return Err(ParseError::InvalidEdges(k));
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(k))
        }
        Some(v) => Err(ParseError::InvalidPolicy(v)),
    }
}

/* ---------------------- precision helpers ---------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionHint {
    Auto,
    F32,
    F64,
}

pub fn parse_precision_hint(s: &str) -> Result<PrecisionHint, ParseError> {
    match norm(s).as_str() {
        "auto" => Ok(PrecisionHint::Auto),
        "f32" => Ok(PrecisionHint::F32),
        "f64" => Ok(PrecisionHint::F64),
        other => Err(ParseError::InvalidPrecision(other.to_string())),
    }
}

pub fn policy_from_code_edges(code: i32, edges: i32) -> Result<SingletonPolicy, ParseError> {
    match code {
        0 => Ok(SingletonPolicy::Off),
        1 => Ok(SingletonPolicy::Use),
        2 => {
            let k = edges.max(0) as usize;
            if k < 1 {
                Err(ParseError::InvalidEdges(k))
            } else {
                Ok(SingletonPolicy::UseWithProtectedEdges(k))
            }
        }
        other => Err(ParseError::InvalidPolicy(other.to_string())),
    }
}
