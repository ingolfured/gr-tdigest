use serde::{Deserialize, Serialize};

/// Controls how the compressor treats data-true singletons (weight==1 or *piles*) and edges.
///
/// Semantics:
/// - `Off`: treat everything uniformly. No tail protection; capacity applies to the **whole** digest.
/// - `Use`: respect data-true singletons/piles; capacity still applies to the **whole** digest,
///          including edges (final Stage **6** may bucketize the full result to `max_size`).
/// - `UseWithProtectedEdges(k)`: like `Use`, but protect up to `k` singletons (or piles at the same
///          mean) on **each** edge. Those edge items are excluded from the core capacity; only the
///          interior is capped (Stage **2**).
///
/// Note: “singleton” is a **data fact** (raw weight==1 or a same-mean *pile* with weight>1).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SingletonPolicy {
    /// Treat everything uniformly — no special casing for singletons or piles.
    Off,
    /// Respect data-true singletons/piles, no tail protection.
    #[default]
    Use,
    /// Respect data-true singletons/piles and preserve up to `k` of them at each edge,
    /// outside the core capacity.
    UseWithProtectedEdges(usize),
}
