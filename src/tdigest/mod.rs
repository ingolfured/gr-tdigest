pub mod centroids;
pub mod cdf;
pub mod codecs;
pub mod quantile;
pub mod test_helpers;

// Internal building blocks
mod tdigest;
mod scale;
mod singleton_policy;
mod compressor;
mod merges;

// Public surface
pub use centroids::Centroid;
pub use scale::ScaleFamily;
pub use singleton_policy::SingletonPolicy;
pub use tdigest::TDigest;

// Opt-in tracing (cheap unless env var set)
#[macro_export]
macro_rules! ttrace {
    ($($arg:tt)*) => {
        if std::env::var("TDIGEST_TRACE").is_ok() {
            eprintln!($($arg)*);
        }
    }
}
