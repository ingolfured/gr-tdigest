use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyFloat, PyIterator};

use crate::tdigest::frontends::{
    ensure_finite_training_values, parse_scale_str, parse_singleton_policy_str,
    validate_quantile_probe,
};
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

// ---------- config helpers for merge_all() / merge() ----------

#[derive(Clone, Debug, PartialEq)]
struct DigestConfig {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
}

fn extract_config_f32(td: &TDigest<f32>) -> DigestConfig {
    DigestConfig {
        max_size: td.max_size(),
        scale: td.scale(),
        policy: td.singleton_policy(),
    }
}

fn extract_config_f64(td: &TDigest<f64>) -> DigestConfig {
    DigestConfig {
        max_size: td.max_size(),
        scale: td.scale(),
        policy: td.singleton_policy(),
    }
}

fn check_configs_match(a: &DigestConfig, b: &DigestConfig) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "tdigest merge: incompatible configs: \
             max_size {} vs {}, scale {:?} vs {:?}, policy {:?} vs {:?}",
            a.max_size, b.max_size, a.scale, b.scale, a.policy, b.policy
        )));
    }
    Ok(())
}

// ---------- internal Rust-only merge helper ----------

impl PyTDigest {
    fn merge_inner(&mut self, other: &PyTDigest) -> PyResult<()> {
        match (&mut self.inner, &other.inner) {
            (InnerDigest::F32(a), InnerDigest::F32(b)) => {
                let cfg_a = extract_config_f32(a);
                let cfg_b = extract_config_f32(b);
                check_configs_match(&cfg_a, &cfg_b)?;
                let merged = TDigest::merge_digests(vec![a.clone(), b.clone()]);
                *a = merged;
                Ok(())
            }
            (InnerDigest::F64(a), InnerDigest::F64(b)) => {
                let cfg_a = extract_config_f64(a);
                let cfg_b = extract_config_f64(b);
                check_configs_match(&cfg_a, &cfg_b)?;
                let merged = TDigest::merge_digests(vec![a.clone(), b.clone()]);
                *a = merged;
                Ok(())
            }
            _ => Err(PyValueError::new_err(
                "tdigest merge: cannot mix f32 and f64 digests",
            )),
        }
    }
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

        ensure_finite_training_values(&values_f64)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
        validate_quantile_probe(q).map_err(|msg| PyValueError::new_err(msg.to_string()))?;

        let val = match &self.inner {
            InnerDigest::F32(td) => td.quantile(q),
            InnerDigest::F64(td) => td.quantile(q),
        };
        Ok(val)
    }

    pub fn add(&mut self, py: Python<'_>, values: PyObject) -> PyResult<()> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (values,))?;
        let ndim = arr
            .getattr("ndim")
            .ok()
            .and_then(|x| x.extract::<usize>().ok())
            .unwrap_or(1);

        // Scalar fast-path only for 0-D numpy arrays / scalar inputs.
        if ndim == 0 {
            let v: f64 = arr.extract()?;
            ensure_finite_training_values(&[v])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            match &mut self.inner {
                InnerDigest::F32(td) => {
                    td.add(v as f32)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }
                InnerDigest::F64(td) => {
                    td.add(v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }
            }
            return Ok(());
        }

        // Sequence/ndarray path.
        let values_f64: Vec<f64> = arr.extract()?;
        ensure_finite_training_values(&values_f64)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        match &mut self.inner {
            InnerDigest::F32(td) => {
                let values_f32: Vec<f32> = values_f64.iter().map(|v| *v as f32).collect();
                td.add_many(values_f32)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            InnerDigest::F64(td) => {
                td.add_many(values_f64)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        Ok(())
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

    // ------ required by test_polars_f32_to_python_f32 ------
    pub fn inner_kind(&self) -> &'static str {
        match &self.inner {
            InnerDigest::F32(_) => "f32",
            InnerDigest::F64(_) => "f64",
        }
    }

    // -------------------------------------------------------------------------
    //                           MERGE (IN-PLACE)
    // -------------------------------------------------------------------------
    ///
    /// Python:
    ///   td.merge(other)
    /// where `other` is either:
    ///   - a TDigest, or
    ///   - any iterable of TDigest instances.
    ///
    /// Mutates `td` in-place. Returns None.
    ///
    pub fn merge(&mut self, other: Bound<'_, PyAny>) -> PyResult<()> {
        // Single TDigest
        if let Ok(one) = other.extract::<PyRef<PyTDigest>>() {
            self.merge_inner(&one)?;
            return Ok(());
        }

        // Otherwise: treat as iterable of TDigest
        let iter = PyIterator::from_object(&other).map_err(|_| {
            PyValueError::new_err("tdigest merge: expected a TDigest or an iterable of TDigest")
        })?;

        for item_res in iter {
            let item = item_res?;
            let d: PyRef<PyTDigest> = item.extract().map_err(|_| {
                PyValueError::new_err("tdigest merge: iterable must contain only TDigest objects")
            })?;
            self.merge_inner(&d)?;
        }

        Ok(())
    }
}

// -------------------------------------------------------------------------
//   Module-level helpers (wire_precision + merge_all)
// -------------------------------------------------------------------------

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

#[pyfunction]
fn merge_all(digests: Bound<'_, PyAny>) -> PyResult<PyTDigest> {
    // Case 1: a single TDigest → clone
    if let Ok(single) = digests.extract::<PyRef<PyTDigest>>() {
        return Ok(PyTDigest {
            inner: match &single.inner {
                InnerDigest::F32(td) => InnerDigest::F32(td.clone()),
                InnerDigest::F64(td) => InnerDigest::F64(td.clone()),
            },
        });
    }

    // Case 2: iterable of TDigest
    let iter = PyIterator::from_object(&digests).map_err(|_| {
        PyValueError::new_err("tdigest merge_all: expected a TDigest or an iterable of TDigest")
    })?;

    let mut f32_vec: Vec<TDigest<f32>> = Vec::new();
    let mut f64_vec: Vec<TDigest<f64>> = Vec::new();

    for item_res in iter {
        let item = item_res?;
        let d: PyRef<PyTDigest> = item.extract().map_err(|_| {
            PyValueError::new_err("tdigest merge_all: iterable must contain only TDigest objects")
        })?;

        match &d.inner {
            InnerDigest::F32(td) => f32_vec.push(td.clone()),
            InnerDigest::F64(td) => f64_vec.push(td.clone()),
        }
    }

    // Empty iterable → empty f64 digest with sane defaults
    if f32_vec.is_empty() && f64_vec.is_empty() {
        let empty = TDigestBuilder::<f64>::new()
            .max_size(1000)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Use)
            .build();
        return Ok(PyTDigest {
            inner: InnerDigest::F64(empty),
        });
    }

    // Mixed precisions are disallowed
    if !f32_vec.is_empty() && !f64_vec.is_empty() {
        return Err(PyValueError::new_err(
            "tdigest merge_all: cannot mix f32 and f64 digests",
        ));
    }

    if !f32_vec.is_empty() {
        // Enforce consistent config
        let cfg0 = extract_config_f32(&f32_vec[0]);
        for td in &f32_vec[1..] {
            let cfg = extract_config_f32(td);
            check_configs_match(&cfg0, &cfg)?;
        }
        let merged = TDigest::merge_digests(f32_vec);
        return Ok(PyTDigest {
            inner: InnerDigest::F32(merged),
        });
    }

    // Only f64 digests
    let cfg0 = extract_config_f64(&f64_vec[0]);
    for td in &f64_vec[1..] {
        let cfg = extract_config_f64(td);
        check_configs_match(&cfg0, &cfg)?;
    }
    let merged = TDigest::merge_digests(f64_vec);
    Ok(PyTDigest {
        inner: InnerDigest::F64(merged),
    })
}

// -------------------------------------------------------------------------

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(wire_precision_py, m)?)?;
    m.add_function(wrap_pyfunction!(merge_all, m)?)?;
    Ok(())
}
