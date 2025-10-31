// src/error.rs
use core::fmt;

/// Library-wide error for gr-tdigest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdError {
    /// User tried to insert NaN during digest construction.
    /// `context` pinpoints where it came from (e.g., "sample value", "sample weight", "wire mean").
    NaNInput { context: &'static str },

    /// Internal invariant violation (should never happen in release builds).
    Invariant { what: &'static str },
}

impl fmt::Display for TdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TdError::NaNInput { context } => write!(
                f,
                "tdigest: NaN values are not allowed (got {}). \
hint: clean your data or drop NaNs before building the digest",
                context
            ),
            TdError::Invariant { what } => {
                write!(f, "tdigest: internal invariant violation: {}", what)
            }
        }
    }
}

impl std::error::Error for TdError {}

pub type TdResult<T> = Result<T, TdError>;
