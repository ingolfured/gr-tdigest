use serde::{Deserialize, Serialize};

/// Controls how TDigest treats true singletons (weight==1 or piles) and edges.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SingletonPolicy {
    /// Treat everything uniformly — no special casing for singletons or piles.
    Off,
    /// Respect data-defined singletons (weight==1 or piles),
    /// but don't apply any tail protection.
    Use,
    /// Same as `Use`, but also preserve up to N raw singletons on each edge
    /// by preventing them from being merged across the boundary.
    UseWithProtectedEdges(usize),
}

impl Default for SingletonPolicy {
    fn default() -> Self {
        SingletonPolicy::Use
    }
}

#[inline]
pub(crate) fn protected_count(p: SingletonPolicy, n_total: f64) -> f64 {
    match p {
        SingletonPolicy::UseWithProtectedEdges(k) => (k as f64).min(n_total * 0.5),
        _ => 0.0,
    }
}
