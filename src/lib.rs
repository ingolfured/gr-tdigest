#![cfg(any(test, feature = "python"))]
mod expressions;

pub mod quality;
pub mod tdigest;
mod utils;
pub use quality::QualityReport;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

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
