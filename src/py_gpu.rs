use crate::py::PyTDigest;
use crate::tdigest::{
    frontends::FrontendDigest,
    gpu::{DeviceArray, GpuDigest, TrainingError},
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

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

    #[pyo3(signature=(values, weights=None))]
    fn add(
        &mut self,
        py: Python<'_>,
        values: &PyDeviceArray,
        weights: Option<&PyDeviceArray>,
    ) -> PyResult<()> {
        py.detach(|| self.0.add(&values.0, weights.map(|w| &w.0)))
            .map_err(training_error)
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, self.0.to_bytes())
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
    fn quantile<'py>(
        &mut self,
        py: Python<'py>,
        probabilities: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qs = probabilities.as_slice()?;
        let result = self.0.quantile(qs).map_err(query_error)?;
        Ok(PyArray1::from_slice(py, result))
    }

    fn median(&mut self) -> PyResult<f64> {
        self.0.median().map_err(PyRuntimeError::new_err)
    }
    fn cdf_device(&mut self, x: &PyDeviceArray) -> PyResult<PyDeviceArray> {
        self.0
            .cdf_device(&x.0)
            .map(PyDeviceArray)
            .map_err(PyRuntimeError::new_err)
    }
    fn quantile_device(&mut self, x: &PyDeviceArray) -> PyResult<PyDeviceArray> {
        self.0
            .quantile_device(&x.0)
            .map(PyDeviceArray)
            .map_err(query_error)
    }
}

#[pyclass(name = "DeviceArray")]
pub(crate) struct PyDeviceArray(DeviceArray);
#[pymethods]
impl PyDeviceArray {
    #[new]
    #[pyo3(signature=(values,device=0))]
    fn new(values: PyReadonlyArray1<'_, f64>, device: i32) -> PyResult<Self> {
        DeviceArray::upload(values.as_slice()?, device)
            .map(Self)
            .map_err(PyRuntimeError::new_err)
    }
    #[getter]
    fn device(&self) -> i32 {
        self.0.device()
    }
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(
            py,
            self.0.to_host().map_err(PyRuntimeError::new_err)?,
        ))
    }
}

fn query_error(message: String) -> PyErr {
    if message == crate::tdigest::gpu::BAD_Q {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

fn training_error(err: TrainingError) -> PyErr {
    match err {
        TrainingError::Input(s) => PyValueError::new_err(s),
        TrainingError::Device(s) => PyRuntimeError::new_err(s),
    }
}
