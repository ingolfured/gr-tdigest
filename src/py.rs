use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyFloat;

use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};
use serde::{Deserialize, Serialize};

use crate::tdigest::centroids::Centroid;
use crate::tdigest::frontends::{parse_scale_str, parse_singleton_policy_str, scale_to_str};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

// ---------- strict arg parsers via shared helpers ----------
fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    parse_scale_str(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_policy(kind: Option<&str>, pin_per_side: Option<usize>) -> Result<SingletonPolicy, PyErr> {
    parse_singleton_policy_str(kind, pin_per_side).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "TDigest", subclass)]
pub struct PyTDigest {
    inner: TDigest<f64>,
}

// ---------- stable bytes blob (SerDigest) ----------
#[derive(Serialize, Deserialize)]
struct SerCentroid {
    mean: f64,
    weight: f64,
}

#[derive(Serialize, Deserialize)]
struct SerPolicy {
    kind: u8,
    edges_per_side: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct SerDigest {
    max_size: usize,
    scale: String, // "quad"|"k1"|"k2"|"k3"
    policy: SerPolicy,
    sum: f64,
    min: f64,
    max: f64,
    count: f64,
    centroids: Vec<SerCentroid>,
}

fn scale_to_string(s: ScaleFamily) -> String {
    scale_to_str(s).to_string()
}

fn string_to_scale(s: &str) -> Result<ScaleFamily, PyErr> {
    parse_scale(Some(s))
}

fn policy_to_ser(p: &SingletonPolicy) -> SerPolicy {
    match *p {
        SingletonPolicy::Off => SerPolicy {
            kind: 0,
            edges_per_side: None,
        },
        SingletonPolicy::Use => SerPolicy {
            kind: 1,
            edges_per_side: None,
        },
        SingletonPolicy::UseWithProtectedEdges(k) => SerPolicy {
            kind: 2,
            edges_per_side: Some(k),
        },
    }
}

// -- in ser_to_policy, error strings only say 'edges' --
fn ser_to_policy(p: &SerPolicy) -> Result<SingletonPolicy, PyErr> {
    Ok(match p.kind {
        0 => SingletonPolicy::Off,
        1 => SingletonPolicy::Use,
        2 => {
            let k = p.edges_per_side.ok_or_else(|| {
                PyValueError::new_err("serialized digest missing edges_per_side for 'edges' policy")
            })?;
            if k < 1 {
                return Err(PyValueError::new_err(
                    "edges_per_side must be >= 1 for 'edges' policy",
                ));
            }
            SingletonPolicy::UseWithProtectedEdges(k)
        }
        _ => return Err(PyValueError::new_err("invalid serialized policy code")),
    })
}

fn td_to_ser(td: &TDigest<f64>) -> SerDigest {
    let centroids = td
        .centroids()
        .iter()
        .map(|c: &Centroid<f64>| SerCentroid {
            mean: c.mean(),
            weight: c.weight(),
        })
        .collect();

    SerDigest {
        max_size: td.max_size(),
        scale: scale_to_string(td.scale()),
        policy: policy_to_ser(&td.singleton_policy()),
        sum: td.sum(),
        min: td.min(),
        max: td.max(),
        count: td.count(),
        centroids,
    }
}

fn ser_to_td(sd: SerDigest) -> Result<TDigest<f64>, PyErr> {
    let scale = string_to_scale(&sd.scale)?;
    let policy = ser_to_policy(&sd.policy)?;

    // Serialized form has no kind flag; preserve atomic unit iff weight == 1.0
    let cents: Vec<Centroid<f64>> = sd
        .centroids
        .into_iter()
        .map(|c| {
            if c.weight == 1.0 {
                Centroid::<f64>::new_atomic_unit_f64(c.mean)
            } else {
                Centroid::<f64>::new_mixed_f64(c.mean, c.weight)
            }
        })
        .collect();

    Ok(TDigest::<f64>::builder()
        .max_size(sd.max_size)
        .scale(scale)
        .singleton_policy(policy)
        .with_centroids_and_stats(
            cents,
            crate::tdigest::DigestStats {
                data_sum: sd.sum,
                total_weight: sd.count,
                data_min: sd.min,
                data_max: sd.max,
            },
        )
        .build())
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_size must be > 0",
            ));
        }
        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, pin_per_side)?;

        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (values,))?;
        let values: Vec<f64> = arr.extract()?;
        // Strict: reject any non-finite training values (matches integration tests)
        if values.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "tdigest: input contains non-finite values (NaN or ±inf)",
            ));
        }

        let base = TDigestBuilder::<f64>::new()
            .max_size(max_size)
            .scale(sc)
            .singleton_policy(policy)
            .build();

        let digest: TDigest<f64> = base
            .merge_unsorted(values)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        let inner = if f32_mode {
            quantize_digest_to_f32(&digest)
        } else {
            digest
        };

        Ok(Self { inner })
    }

    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.median())
    }

    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        if !q.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must be a finite number in [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&q) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must be in [0, 1]. Example: quantile(0.95) for the 95th percentile",
            ));
        }
        Ok(self.inner.quantile(q))
    }

    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;

        // --- Try scalar first (works for Python floats and 0-D numpy scalars) ---
        if let Ok(xf) = arr.extract::<f64>() {
            let p = self.inner.cdf_or_nan(&[xf])[0];
            let obj: PyObject = PyFloat::new(py, p).into_any().unbind();
            return Ok(obj);
        }

        // --- Otherwise: treat as a 1-D array / sequence ---
        let values: Vec<f64> = arr.extract()?;
        let ys: Vec<f64> = self.inner.cdf_or_nan(&values);

        let out = np.call_method1("asarray", (ys,))?;
        Ok(out.unbind())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = config::standard();
        let ser = td_to_ser(&self.inner);
        let bytes = encode_to_vec(&ser, cfg).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize error: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let cfg = config::standard();
        let (ser, _len): (SerDigest, usize) =
            decode_from_slice(b.as_bytes(), cfg).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("deserialize error: {e}"))
            })?;
        Ok(Self {
            inner: ser_to_td(ser)?,
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    Ok(())
}

// ---------- helpers ----------

fn quantize_digest_to_f32(td: &TDigest<f64>) -> TDigest<f64> {
    // Preserve kind when possible during quantization.
    let cents_q: Vec<Centroid<f64>> = td
        .centroids()
        .iter()
        .map(|c| {
            let m = c.mean() as f32 as f64;
            let w = c.weight() as f32 as f64;
            if c.is_atomic_unit() {
                Centroid::<f64>::new_atomic_unit_f64(m)
            } else if c.is_atomic() {
                Centroid::<f64>::new_atomic_f64(m, w)
            } else {
                Centroid::<f64>::new_mixed_f64(m, w)
            }
        })
        .collect();

    TDigest::<f64>::builder()
        .max_size(td.max_size())
        .scale(td.scale())
        .singleton_policy(td.singleton_policy())
        .with_centroids_and_stats(
            cents_q,
            crate::tdigest::DigestStats {
                data_sum: td.sum(),
                total_weight: td.count(),
                data_min: td.min(),
                data_max: td.max(),
            },
        )
        .build()
}
