// src/tdigest/frontends.rs
//! Small, shared parsing & normalization helpers for all front-ends (Python, Polars, JNI, CLI).
//! Keep this dependency-light and free of PyO3/JNI/Polars types.

use std::fmt::{Display, Formatter};

use crate::tdigest::wire::{
    decode_digest, encode_digest, wire_precision, WireDecodedDigest, WirePrecision,
};
use crate::tdigest::TDigest;
use crate::tdigest::{scale::ScaleFamily, singleton_policy::SingletonPolicy};
use crate::{TdError, TdResult};

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

/* ---------------------- validation helpers ---------------------- */

/// Shared finite-data guard for training/addition paths across front-ends.
#[inline]
pub fn ensure_finite_training_values(values: &[f64]) -> TdResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(TdError::NonFiniteInput {
            context: "sample value (NaN or ±inf)",
        });
    }
    Ok(())
}

/// Strict quantile probe validation shared by JNI/Python wrappers.
#[inline]
pub fn validate_quantile_probe(q: f64) -> Result<(), &'static str> {
    if !q.is_finite() {
        return Err("q must be a finite number in [0,1]");
    }
    if !(0.0..=1.0).contains(&q) {
        return Err("q must be in [0,1]");
    }
    Ok(())
}

/* ---------------------- shared frontend service ---------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigestConfig {
    pub max_size: usize,
    pub scale: ScaleFamily,
    pub policy: SingletonPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestPrecision {
    F32,
    F64,
}

impl DigestPrecision {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            DigestPrecision::F32 => "f32",
            DigestPrecision::F64 => "f64",
        }
    }
}

impl From<WirePrecision> for DigestPrecision {
    fn from(value: WirePrecision) -> Self {
        match value {
            WirePrecision::F32 => DigestPrecision::F32,
            WirePrecision::F64 => DigestPrecision::F64,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FrontendError {
    InvalidTrainingData(String),
    InvalidProbe(String),
    IncompatibleMerge(String),
    DecodeError(String),
}

impl Display for FrontendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontendError::InvalidTrainingData(msg)
            | FrontendError::InvalidProbe(msg)
            | FrontendError::IncompatibleMerge(msg)
            | FrontendError::DecodeError(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FrontendError {}

impl From<TdError> for FrontendError {
    fn from(value: TdError) -> Self {
        FrontendError::InvalidTrainingData(value.to_string())
    }
}

#[derive(Debug, Clone)]
pub enum FrontendDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

impl FrontendDigest {
    #[inline]
    pub fn canonical_empty() -> Self {
        FrontendDigest::F64(
            TDigest::<f64>::builder()
                .max_size(1000)
                .scale(ScaleFamily::K2)
                .singleton_policy(SingletonPolicy::Use)
                .build(),
        )
    }

    pub fn from_values(
        values: Vec<f64>,
        config: DigestConfig,
        precision: DigestPrecision,
    ) -> Result<Self, FrontendError> {
        ensure_finite_training_values(&values).map_err(FrontendError::from)?;

        match precision {
            DigestPrecision::F32 => {
                let xs: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                let td = TDigest::<f32>::builder()
                    .max_size(config.max_size)
                    .scale(config.scale)
                    .singleton_policy(config.policy)
                    .build()
                    .merge_unsorted(xs)
                    .map_err(FrontendError::from)?;
                Ok(FrontendDigest::F32(td))
            }
            DigestPrecision::F64 => {
                let td = TDigest::<f64>::builder()
                    .max_size(config.max_size)
                    .scale(config.scale)
                    .singleton_policy(config.policy)
                    .build()
                    .merge_unsorted(values)
                    .map_err(FrontendError::from)?;
                Ok(FrontendDigest::F64(td))
            }
        }
    }

    #[inline]
    pub fn precision(&self) -> DigestPrecision {
        match self {
            FrontendDigest::F32(_) => DigestPrecision::F32,
            FrontendDigest::F64(_) => DigestPrecision::F64,
        }
    }

    #[inline]
    pub fn config(&self) -> DigestConfig {
        match self {
            FrontendDigest::F32(td) => DigestConfig {
                max_size: td.max_size(),
                scale: td.scale(),
                policy: td.singleton_policy(),
            },
            FrontendDigest::F64(td) => DigestConfig {
                max_size: td.max_size(),
                scale: td.scale(),
                policy: td.singleton_policy(),
            },
        }
    }

    #[inline]
    pub fn is_effectively_empty(&self) -> bool {
        match self {
            FrontendDigest::F32(td) => td.is_effectively_empty(),
            FrontendDigest::F64(td) => td.is_effectively_empty(),
        }
    }

    pub fn add_values_f64(&mut self, values: Vec<f64>) -> Result<(), FrontendError> {
        ensure_finite_training_values(&values).map_err(FrontendError::from)?;

        match self {
            FrontendDigest::F32(td) => {
                let xs: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                td.add_many(xs).map_err(FrontendError::from)?;
            }
            FrontendDigest::F64(td) => {
                td.add_many(values).map_err(FrontendError::from)?;
            }
        }
        Ok(())
    }

    pub fn merge_in_place(&mut self, other: &Self) -> Result<(), FrontendError> {
        if self.precision() != other.precision() {
            return Err(FrontendError::IncompatibleMerge(format!(
                "tdigest merge: incompatible digests (precision {} vs {}). \
Cast explicitly before merge (e.g. cast_precision('f64')).",
                self.precision().as_str(),
                other.precision().as_str()
            )));
        }

        // Empty digest merge is always safe for same precision.
        if self.is_effectively_empty() && !other.is_effectively_empty() {
            *self = other.clone();
            return Ok(());
        }
        if other.is_effectively_empty() {
            return Ok(());
        }

        let lhs_cfg = self.config();
        let rhs_cfg = other.config();
        if lhs_cfg != rhs_cfg {
            return Err(FrontendError::IncompatibleMerge(format!(
                "tdigest merge: incompatible configs (max_size {} vs {}, scale {:?} vs {:?}, singleton_policy {:?} vs {:?}). \
Rebuild or cast to a shared configuration before merge.",
                lhs_cfg.max_size,
                rhs_cfg.max_size,
                lhs_cfg.scale,
                rhs_cfg.scale,
                lhs_cfg.policy,
                rhs_cfg.policy
            )));
        }

        match (self, other) {
            (FrontendDigest::F32(a), FrontendDigest::F32(b)) => {
                *a = TDigest::<f32>::merge_digests(vec![a.clone(), b.clone()]);
            }
            (FrontendDigest::F64(a), FrontendDigest::F64(b)) => {
                *a = TDigest::<f64>::merge_digests(vec![a.clone(), b.clone()]);
            }
            _ => unreachable!("precision mismatch checked above"),
        }
        Ok(())
    }

    pub fn merge_all(digests: Vec<Self>) -> Result<Self, FrontendError> {
        if digests.is_empty() {
            return Ok(Self::canonical_empty());
        }

        let first = digests[0].clone();
        let mut acc = first.clone();
        for d in digests.iter().skip(1) {
            acc.merge_in_place(d)?;
        }
        Ok(acc)
    }

    pub fn quantile_strict(&self, q: f64) -> Result<f64, FrontendError> {
        validate_quantile_probe(q).map_err(|msg| FrontendError::InvalidProbe(msg.to_string()))?;
        Ok(match self {
            FrontendDigest::F32(td) => td.quantile(q),
            FrontendDigest::F64(td) => td.quantile(q),
        })
    }

    #[inline]
    pub fn median(&self) -> f64 {
        match self {
            FrontendDigest::F32(td) => td.median(),
            FrontendDigest::F64(td) => td.median(),
        }
    }

    #[inline]
    pub fn cdf(&self, xs: &[f64]) -> Vec<f64> {
        match self {
            FrontendDigest::F32(td) => td.cdf_or_nan(xs),
            FrontendDigest::F64(td) => td.cdf_or_nan(xs),
        }
    }

    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            FrontendDigest::F32(td) => encode_digest(td),
            FrontendDigest::F64(td) => encode_digest(td),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FrontendError> {
        match decode_digest(bytes) {
            Ok(WireDecodedDigest::F32(td32)) => Ok(FrontendDigest::F32(td32)),
            Ok(WireDecodedDigest::F64(td64)) => Ok(FrontendDigest::F64(td64)),
            Err(e) => Err(FrontendError::DecodeError(format!(
                "tdigest decode error: {e}"
            ))),
        }
    }

    pub fn from_bytes_with_expected(
        bytes: &[u8],
        expected: WirePrecision,
    ) -> Result<Self, FrontendError> {
        let actual = wire_precision(bytes)
            .map_err(|e| FrontendError::DecodeError(format!("tdigest.from_bytes: {e}")))?;

        if actual != expected {
            return Err(FrontendError::DecodeError(format!(
                "tdigest.from_bytes: mixed f32/f64 blobs in column (expected {})",
                match expected {
                    WirePrecision::F32 => "f32",
                    WirePrecision::F64 => "f64",
                }
            )));
        }
        Self::from_bytes(bytes)
    }

    #[inline]
    pub fn inner_kind(&self) -> &'static str {
        match self {
            FrontendDigest::F32(_) => "f32",
            FrontendDigest::F64(_) => "f64",
        }
    }
}
