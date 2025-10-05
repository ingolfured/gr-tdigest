use crate::tdigest::TDigest;
use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes; // ← bring TDigest into scope

#[pyclass]
pub struct PyTDigest {
    inner: TDigest,
}

#[pymethods]
impl PyTDigest {
    // Return a PyO3 0.25-friendly type
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = config::standard();
        let bytes = encode_to_vec(&self.inner, cfg)
            .map_err(|e| PyValueError::new_err(format!("serialize error: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let cfg = config::standard();
        let (inner, _len): (TDigest, usize) = decode_from_slice(b.as_bytes(), cfg)
            .map_err(|e| PyValueError::new_err(format!("deserialize error: {e}")))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn len(&self) -> usize {
        // TDigest has no `len()`, use number of centroids
        self.inner.centroids().len()
    }

    #[getter]
    pub fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    #[getter]
    pub fn min(&self) -> Option<f64> {
        Some(self.inner.min())
    }

    #[getter]
    pub fn max(&self) -> Option<f64> {
        Some(self.inner.max())
    }

    fn __len__(&self) -> usize {
        self.inner.centroids().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "TDigest(len={}, max_size={}, min={:?}, max={:?})",
            self.inner.centroids().len(),
            self.inner.max_size(),
            self.inner.min(),
            self.inner.max()
        )
    }
}
