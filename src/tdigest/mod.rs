#![allow(clippy::module_inception)]
//! t-digest core types and public API.
//!
//! - [`TDigest`] — main data structure and builder.
//! - **Compression** — internal compressor enforces a k-limit ([`ScaleFamily`]) while
//!   preserving extremes and discrete *singleton piles*.
//! - **Scales** — see [`ScaleFamily`] for available k-mappings.
//! - **CDF** — see [`TDigest::cdf`] for exact semantics and guarantees.
//!
//! ### Notes on intra-doc links
//! If you refer to internal helpers, link to the **module**, not private items.
//! Example: prefer [`crate::tdigest::merges`] over linking to a private type.
//! This avoids `rustdoc::private_intra_doc_links` warnings.

pub mod cdf;
pub mod centroids;
pub mod codecs;
pub mod compressor;
pub mod merges;
pub mod precision;
pub mod quantile;
pub mod scale;
pub mod singleton_policy;

pub mod tdigest;

pub use self::precision::Precision;
pub use self::scale::ScaleFamily;
pub use self::tdigest::{DigestStats, TDigest, TDigestBuilder};

#[cfg(test)]
mod test_helpers;
