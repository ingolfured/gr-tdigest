use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyFloat, PyList, PySequence, PyType};

use crate::tdigest::frontends::{parse_scale_str, parse_singleton_policy_str};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{
    decode_digest, encode_digest, wire_precision, WireDecodedDigest, WirePrecision,
};
use crate::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

// -------------------- helpers --------------------

fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    parse_scale_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_policy(kind: Option<&str>, pin: Option<usize>) -> Result<SingletonPolicy, PyErr> {
    parse_singleton_policy_str(kind, pin).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[derive(Clone)]
enum InnerDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

// config struct used to enforce matching parameters
#[derive(Clone, Debug, PartialEq)]
struct DigestConfig {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
}

fn extract_config<F: crate::tdigest::precision::FloatLike + ordered_float::FloatCore>(
    td: &TDigest<F>,
) -> DigestConfig {
    DigestConfig {
        max_size: td.max_size(),
        scale: td.scale(),
        policy: td.singleton_policy(),
    }
}

fn check_configs_match(a: &DigestConfig, b: &DigestConfig) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!(
            "tdigest merge: incompatible configs: \
             max_size {} vs {}, scale {:?} vs {:?}, policy {:?} vs {:?}",
            a.max_size, b.max_size, a.scale, b.scale, a.policy, b.policy
        )))
    } else {
        Ok(())
    }
}

#[pyclass(name = "TDigest", subclass)]
pub struct PyTDigest {
    inner: InnerDigest,
}

// -------------------- core Python API --------------------

#[pymethods]
impl PyTDigest {
    // ---------------- from_array ----------------
    #[staticmethod]
    #[pyo3(signature = (values, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, pin_per_side=None))]
    pub fn from_array(
        py: Python<'_>,
        values: PyObject,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        pin: Option<usize>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(PyValueError::new_err("max_size must be > 0"));
        }

        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin)?;

        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (values,))?;
        let vals_f64: Vec<f64> = arr.extract()?;

        if vals_f64.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "tdigest: input contains non-finite values",
            ));
        }

        if f32_mode {
            let vals_f32: Vec<f32> = vals_f64.iter().map(|v| *v as f32).collect();
            let base = TDigestBuilder::<f32>::new()
                .max_size(max_size)
                .scale(sc)
                .singleton_policy(policy)
                .build();

            let d = base
                .merge_unsorted(vals_f32)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            Ok(Self {
                inner: InnerDigest::F32(d),
            })
        } else {
            let base = TDigestBuilder::<f64>::new()
                .max_size(max_size)
                .scale(sc)
                .singleton_policy(policy)
                .build();

            let d = base
                .merge_unsorted(vals_f64)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            Ok(Self {
                inner: InnerDigest::F64(d),
            })
        }
    }

    // ---------------- quantile / median ----------------
    pub fn median(&self) -> PyResult<f64> {
        Ok(match &self.inner {
            InnerDigest::F32(td) => td.median(),
            InnerDigest::F64(td) => td.median(),
        })
    }

    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("q must be in [0,1]"));
        }
        Ok(match &self.inner {
            InnerDigest::F32(td) => td.quantile(q),
            InnerDigest::F64(td) => td.quantile(q),
        })
    }

    // ---------------- cdf ----------------
    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;

        // scalar?
        if let Ok(xf) = arr.extract::<f64>() {
            let v = match &self.inner {
                InnerDigest::F32(td) => td.cdf_or_nan(&[xf])[0],
                InnerDigest::F64(td) => td.cdf_or_nan(&[xf])[0],
            };
            return Ok(PyFloat::new(py, v).into_any().unbind());
        }

        let xs: Vec<f64> = arr.extract()?;
        let out = match &self.inner {
            InnerDigest::F32(td) => td.cdf_or_nan(&xs),
            InnerDigest::F64(td) => td.cdf_or_nan(&xs),
        };

        let arr_out = np.call_method1("asarray", (out,))?;
        Ok(arr_out.unbind())
    }

    // ---------------- bytes ↔ digest ----------------
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = match &self.inner {
            InnerDigest::F32(td) => encode_digest(td),
            InnerDigest::F64(td) => encode_digest(td),
        };
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        match decode_digest(b.as_bytes()) {
            Ok(WireDecodedDigest::F32(td)) => Ok(Self {
                inner: InnerDigest::F32(td),
            }),
            Ok(WireDecodedDigest::F64(td)) => Ok(Self {
                inner: InnerDigest::F64(td),
            }),
            Err(e) => Err(PyValueError::new_err(format!("decode error: {e}"))),
        }
    }

    pub fn inner_kind(&self) -> &'static str {
        match &self.inner {
            InnerDigest::F32(_) => "f32",
            InnerDigest::F64(_) => "f64",
        }
    }

    // -------------------------------------------------------------------------
    //                              MERGE (in-place)
    // -------------------------------------------------------------------------

    #[pyo3(signature = (other))]
    pub fn merge<'py>(
        &'py mut self,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, PyTDigest>> {
        // Try single digest
        if let Ok(one) = other.extract::<PyRef<PyTDigest>>() {
            self.merge_one(&one)?;
            return Ok(PyRefMut::from(self));
        }

        // Otherwise treat as sequence
        let seq = PySequence::try_from(other)?;
        let len = seq.len().unwrap_or(0);

        for i in 0..len {
            let item = seq.get_item(i)?;
            let d: PyRef<PyTDigest> = item.extract()?;
            self.merge_one(&d)?;
        }

        Ok(PyRefMut::from(self))
    }

    fn merge_one(&mut self, other: &PyRef<PyTDigest>) -> PyResult<()> {
        match (&mut self.inner, &other.inner) {
            (InnerDigest::F32(a), InnerDigest::F32(b)) => {
                check_configs_match(&extract_config(a), &extract_config(b))?;
                let merged = TDigest::merge_digests(vec![a.clone(), b.clone()]);
                *a = merged;
                Ok(())
            }
            (InnerDigest::F64(a), InnerDigest::F64(b)) => {
                check_configs_match(&extract_config(a), &extract_config(b))?;
                let merged = TDigest::merge_digests(vec![a.clone(), b.clone()]);
                *a = merged;
                Ok(())
            }
            _ => Err(PyValueError::new_err(
                "tdigest merge: cannot mix f32 and f64 digests",
            )),
        }
    }

    // -------------------------------------------------------------------------
    //                              MERGE_ALL
    // -------------------------------------------------------------------------

    #[classmethod]
    #[pyo3(signature = (digests))]
    pub fn merge_all(_cls: &Bound<PyType>, digests: &Bound<PyAny>) -> PyResult<Self> {
        // Direct single digest → clone
        if let Ok(single) = digests.extract::<PyRef<PyTDigest>>() {
            return Ok(single.clone());
        }

        let seq = PySequence::try_from(digests)?;
        let len = seq.len().unwrap_or(0);

        if len == 0 {
            let empty = TDigestBuilder::<f64>::new()
                .max_size(1)
                .scale(ScaleFamily::K2)
                .singleton_policy(SingletonPolicy::Off)
                .build();
            return Ok(Self {
                inner: InnerDigest::F64(empty),
            });
        }

        // Seed
        let first_any = seq.get_item(0)?;
        let first: PyRef<PyTDigest> = first_any.extract()?;

        match &first.inner {
            InnerDigest::F32(seed) => {
                let cfg = extract_config(seed);
                let mut acc = seed.clone();

                for i in 1..len {
                    let obj = seq.get_item(i)?;
                    let d: PyRef<PyTDigest> = obj.extract()?;

                    match &d.inner {
                        InnerDigest::F32(other) => {
                            check_configs_match(&cfg, &extract_config(other))?;
                            acc = TDigest::merge_digests(vec![acc, other.clone()]);
                        }
                        _ => {
                            return Err(PyValueError::new_err(
                                "tdigest merge_all: cannot mix f32 and f64 digests",
                            ))
                        }
                    }
                }

                Ok(Self {
                    inner: InnerDigest::F32(acc),
                })
            }

            InnerDigest::F64(seed) => {
                let cfg = extract_config(seed);
                let mut acc = seed.clone();

                for i in 1..len {
                    let obj = seq.get_item(i)?;
                    let d: PyRef<PyTDigest> = obj.extract()?;

                    match &d.inner {
                        InnerDigest::F64(other) => {
                            check_configs_match(&cfg, &extract_config(other))?;
                            acc = TDigest::merge_digests(vec![acc, other.clone()]);
                        }
                        _ => {
                            return Err(PyValueError::new_err(
                                "tdigest merge_all: cannot mix f32 and f64 digests",
                            ))
                        }
                    }
                }

                Ok(Self {
                    inner: InnerDigest::F64(acc),
                })
            }
        }
    }
}

// ---------------- wire_precision_py ----------------

#[pyfunction]
fn wire_precision_py(b: &Bound<PyAny>) -> PyResult<String> {
    let py_bytes: &Bound<PyBytes> = b.downcast()?;
    match wire_precision(py_bytes.as_bytes()) {
        Ok(WirePrecision::F32) => Ok("f32".to_string()),
        Ok(WirePrecision::F64) => Ok("f64".to_string()),
        Err(e) => Err(PyValueError::new_err(format!("{e}"))),
    }
}

pub(crate) fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(wire_precision_py, m)?)?;
    Ok(())
}
