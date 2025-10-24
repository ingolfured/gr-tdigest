pub mod cdf;
pub mod centroids;
pub mod codecs;
pub mod quantile;
pub mod test_helpers;

// Internal building blocks
mod compressor;
mod merges;
mod scale;
mod singleton_policy;
#[allow(clippy::module_inception)]
mod tdigest;

pub use self::centroids::Centroid;
pub use self::scale::ScaleFamily;
pub use self::singleton_policy::SingletonPolicy;
pub use self::tdigest::{DigestStats, TDigest, TDigestBuilder};

// Optional tracing macro (cheap unless env var set)
#[macro_export]
macro_rules! ttrace {
    ($($arg:tt)*) => {
        if std::env::var("TDIGEST_TRACE").is_ok() {
            eprintln!($($arg)*);
        }
    };
}
