use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat};

use crate::tdigest::frontends::{parse_scale_str, parse_singleton_policy_str};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{
    decode_digest, encode_digest, wire_precision, WireDecodedDigest, WirePrecision,
};
use crate::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

// ---------- strict arg parsers via shared helpers ----------

fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    parse_scale_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_policy(kind: Option<&str>, pin_per_side: Option<usize>) -> Result<SingletonPolicy, PyErr> {
    parse_singleton_policy_str(kind, pin_per_side).map_err(|e| PyValueError::new_err(e.to_string()))
}

// ---------- internal enum to track backend precision ----------

enum InnerDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

#[pyclass(name = "TDigest", subclass)]
pub struct PyTDigest {
    inner: InnerDigest,
}

// ---------- Python methods ----------

#[pymethods]
impl PyTDigest {
    #[staticmethod]
    #[pyo3(signature = (values, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None))]
    pub fn from_array(
        py: Python<'_>,
        values: PyObject,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin_per_side: Option<usize>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin_per_side)?;

        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (values,))?;
        let values_f64: Vec<f64> = arr.extract()?;

        // Strict: reject any non-finite training values (matches integration tests)
        if values_f64.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "tdigest: input contains non-finite values (NaN or ±inf)",
            ));
        }

        if f32_mode {
            // Compact backend: TDigest<f32>
            let values_f32: Vec<f32> = values_f64.iter().map(|v| *v as f32).collect();

            let base = TDigestBuilder::<f32>::new()
                .max_size(max_size)
                .scale(sc)
                .singleton_policy(policy)
                .build();

            let digest: TDigest<f32> = base
                .merge_unsorted(values_f32)
                .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

            Ok(Self {
                inner: InnerDigest::F32(digest),
            })
        } else {
            // Full-precision backend: TDigest<f64>
            let base = TDigestBuilder::<f64>::new()
                .max_size(max_size)
                .scale(sc)
                .singleton_policy(policy)
                .build();

            let digest: TDigest<f64> = base
                .merge_unsorted(values_f64)
                .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

            Ok(Self {
                inner: InnerDigest::F64(digest),
            })
        }
    }

    pub fn median(&self) -> PyResult<f64> {
        let m = match &self.inner {
            InnerDigest::F32(td) => td.median(),
            InnerDigest::F64(td) => td.median(),
        };
        Ok(m)
    }

    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        if !q.is_finite() {
            return Err(PyValueError::new_err("q must be a finite number in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err(
                "q must be in [0, 1]. Example: quantile(0.95) for the 95th percentile",
            ));
        }

        let val = match &self.inner {
            InnerDigest::F32(td) => td.quantile(q),
            InnerDigest::F64(td) => td.quantile(q),
        };
        Ok(val)
    }

    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;

        // Try scalar first (works for Python floats and 0-D numpy scalars)
        if let Ok(xf) = arr.extract::<f64>() {
            let p = match &self.inner {
                InnerDigest::F32(td) => td.cdf_or_nan(&[xf])[0],
                InnerDigest::F64(td) => td.cdf_or_nan(&[xf])[0],
            };
            let obj: PyObject = PyFloat::new(py, p).into_any().unbind();
            return Ok(obj);
        }

        // Otherwise: treat as a 1-D array / sequence
        let values: Vec<f64> = arr.extract()?;
        let ys: Vec<f64> = match &self.inner {
            InnerDigest::F32(td) => td.cdf_or_nan(&values),
            InnerDigest::F64(td) => td.cdf_or_nan(&values),
        };

        let out = np.call_method1("asarray", (ys,))?;
        Ok(out.unbind())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Canonical TDIG wire format — width follows backend precision:
        // - F32 → f32 means on the wire
        // - F64 → f64 means on the wire
        let bytes: Vec<u8> = match &self.inner {
            InnerDigest::F32(td) => encode_digest(td),
            InnerDigest::F64(td) => encode_digest(td),
        };
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        // Decode TDIG wire format and reconstruct the matching backend.
        match decode_digest(b.as_bytes()) {
            Ok(WireDecodedDigest::F32(td32)) => Ok(Self {
                inner: InnerDigest::F32(td32),
            }),
            Ok(WireDecodedDigest::F64(td64)) => Ok(Self {
                inner: InnerDigest::F64(td64),
            }),
            Err(e) => Err(PyValueError::new_err(format!("tdigest decode error: {e}"))),
        }
    }

    // ------ NEW: required by test_polars_f32_to_python_f32 ------
    pub fn inner_kind(&self) -> &'static str {
        match &self.inner {
            InnerDigest::F32(_) => "f32",
            InnerDigest::F64(_) => "f64",
        }
    }
}

#[pyfunction]
fn wire_precision_py(b: &Bound<PyAny>) -> PyResult<String> {
    // Ensure it's actually a bytes object
    let py_bytes: &Bound<PyBytes> = b.downcast()?;
    let bytes = py_bytes.as_bytes();

    match wire_precision(bytes) {
        Ok(WirePrecision::F32) => Ok("f32".to_string()),
        Ok(WirePrecision::F64) => Ok("f64".to_string()),
        Err(e) => Err(PyValueError::new_err(format!("wire_precision: {e}"))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(wire_precision_py, m)?)?;
    Ok(())
}
