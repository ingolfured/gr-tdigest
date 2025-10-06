pub use quality::cdf_quality::assess_cdf;
pub use quality::quality_base::{
    build_digest_sorted, exact_ecdf_for_sorted, expected_quantile, gen_dataset, print_report,
    DistKind, Precision, QualityReport,
};
pub use quality::quantile_quality::assess_quantiles_with;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

// Expose the quality submodules (folder: src/quality/)
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

pub mod tdigest;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

#[cfg(feature = "python")]
#[pymodule]
fn polars_tdigest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    py::register(m)?;
    Ok(())
}
