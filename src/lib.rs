mod expressions;
pub mod tdigest;

#[cfg(test)]
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

#[cfg(test)]
pub use crate::quality::quality_base::{print_banner, print_report, print_section};
// use pyo3_polars::export::export_polars_plugin;

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
