//! Original tdigest_rs bindings, adapted only to the current PyO3/NumPy API.
//! Numerical operations use the pinned upstream kernels with overflow guards.

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;

use crate::upstream::TDigest;

// The original compressor reserves one output slot per input and retains that
// capacity. Release unused slots at the Python boundary without changing any
// centroid, integer weight, arithmetic, or query convention.
fn compact_digest<T>(mut digest: TDigest<T>) -> TDigest<T> {
    crate::compact_storage::compact_digest_storage(&mut digest.means, &mut digest.weights);
    digest
}

macro_rules! generate {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            pub inner: TDigest<$type>,
        }

        #[pymethods]
        impl $name {
            #[getter]
            fn means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<$type>>> {
                Ok(PyArray1::from_vec(py, self.inner.means.clone()))
            }

            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
                Ok(PyArray1::from_vec(py, self.inner.weights.clone()))
            }

            fn __len__(&self) -> PyResult<usize> {
                Ok(self.inner.means.len())
            }

            #[classmethod]
            fn from_array(
                _cls: &Bound<'_, PyType>,
                py: Python,
                arr: PyReadonlyArray1<$type>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.is_empty() {
                    return Err(PyValueError::new_err("Array must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                py.detach(|| {
                    Ok(Self {
                        inner: compact_digest(
                            TDigest::from_array(&arr, delta)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        ),
                    })
                })
            }

            #[classmethod]
            fn from_means_weights(
                _cls: &Bound<'_, PyType>,
                py: Python,
                arr: PyReadonlyArray1<$type>,
                weights: PyReadonlyArray1<u32>,
                delta: $type,
            ) -> PyResult<Self> {
                if arr.is_empty() {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                if weights.is_empty() {
                    return Err(PyValueError::new_err("Means must be non-empty!"));
                }
                let arr = arr.as_array().to_vec();
                let weights = weights.as_array().to_vec();

                py.detach(|| {
                    Ok(Self {
                        inner: compact_digest(
                            TDigest::from_means_weights(&arr, &weights, delta)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        ),
                    })
                })
            }

            fn quantile(&self, py: Python, x: $type) -> PyResult<$type> {
                py.detach(|| {
                    self.inner
                        .quantile(x)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
            }

            fn median(&self, py: Python) -> PyResult<$type> {
                py.detach(|| {
                    self.inner
                        .median()
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
            }

            fn trimmed_mean(&self, py: Python, lower: $type, upper: $type) -> PyResult<$type> {
                py.detach(|| {
                    self.inner
                        .trimmed_mean(lower, upper)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
            }

            fn merge(&self, py: Python, other: &Self, delta: $type) -> PyResult<Self> {
                py.detach(|| {
                    Ok(Self {
                        inner: compact_digest(
                            self.inner
                                .merge(&other.inner, delta)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        ),
                    })
                })
            }

            fn n_zero_weights(&self) -> PyResult<usize> {
                Ok(self.inner.n_zero_weights()?)
            }
        }
    };
}

generate!(_TDigestInternal32, f32);
generate!(_TDigestInternal64, f64);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_TDigestInternal32>()?;
    m.add_class::<_TDigestInternal64>()?;
    Ok(())
}
