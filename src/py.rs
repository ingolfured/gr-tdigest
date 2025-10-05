use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::tdigest::TDigest as CoreTDigest;

use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};

#[pyclass(name = "TDigest")]
pub struct PyTDigest {
    inner: CoreTDigest,
}

#[pymethods]
impl PyTDigest {
    #[staticmethod]
    pub fn from_array(xs: Vec<f64>, max_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: CoreTDigest::from_unsorted(&xs, max_size),
        })
    }

    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.estimate_quantile(0.5))
    }

    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        Ok(self.inner.estimate_quantile(q))
    }

    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;
        let xs: Vec<f64> = arr.extract()?;
        let ys: Vec<f64> = self.inner.estimate_cdf(&xs);
        let out = np.call_method1("asarray", (ys,))?;
        Ok(out.unbind())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = config::standard();
        let bytes = encode_to_vec(&self.inner, cfg).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize error: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let cfg = config::standard();
        let (inner, _len): (CoreTDigest, usize) =
            decode_from_slice(b.as_bytes(), cfg).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("deserialize error: {e}"))
            })?;
        Ok(Self { inner })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    Ok(())
}
