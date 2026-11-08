use crate::py::PyTDigest;
use crate::tdigest::{frontends::FrontendDigest, gpu::GpuDigest};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(name = "GpuDigest")]
pub(crate) struct PyGpuDigest(GpuDigest);

#[pymethods]
impl PyGpuDigest {
    #[new]
    #[pyo3(signature=(digest, device=0))]
    fn new(digest: &PyTDigest, device: i32) -> PyResult<Self> {
        let snapshot = match &digest.inner {
            FrontendDigest::F32(d) => GpuDigest::upload(d, device),
            FrontendDigest::F64(d) => GpuDigest::upload(d, device),
        };
        snapshot.map(Self).map_err(PyRuntimeError::new_err)
    }

    fn cdf<'py>(
        &mut self,
        py: Python<'py>,
        probes: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .0
            .cdf(probes.as_slice()?)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyArray1::from_slice(py, result))
    }
}
