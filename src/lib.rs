mod expressions;
pub mod tdigest;

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
