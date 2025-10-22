// src/lib.rs
#![allow(non_snake_case)]

mod expressions;
pub mod tdigest;

#[cfg(feature = "java")]
pub mod jni;

// ---- test-only quality modules ------------------------------------------------
#[cfg(test)]
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

#[cfg(test)]
pub use crate::quality::quality_base::{print_banner, print_report, print_section};

// ---- jemalloc on linux (ok to keep) ------------------------------------------
#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

// ---- Python extension (behind feature) ---------------------------------------
#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

#[cfg(feature = "python")]
#[pymodule]
fn tdigest_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    py::register(m)?;
    Ok(())
}
