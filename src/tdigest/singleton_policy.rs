use serde::{Deserialize, Serialize};

/// Controls how TDigest treats true singletons (weight==1 or piles) and edges.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SingletonPolicy {
    /// Treat everything uniformly — no special casing for singletons or piles.
    Off,
    /// Respect data-defined singletons (weight==1 or piles),
    /// but don't apply any tail protection.
    #[default]
    Use,
    /// Same as `Use`, but also preserve up to N raw singletons on each edge
    /// by preventing them from being merged across the boundary.
    UseWithProtectedEdges(usize),
}
