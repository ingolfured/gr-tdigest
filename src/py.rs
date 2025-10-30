use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};

// Pull types from the crate's tdigest public surface
use crate::tdigest::centroids::Centroid;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    match s.map(|t| t.to_ascii_lowercase()) {
        None => Ok(ScaleFamily::K2), // library-wide default
        Some(ref v) if v == "quad" => Ok(ScaleFamily::Quad),
        Some(ref v) if v == "k1" => Ok(ScaleFamily::K1),
        Some(ref v) if v == "k2" => Ok(ScaleFamily::K2),
        Some(ref v) if v == "k3" => Ok(ScaleFamily::K3),
        Some(v) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid scale: {v} (expected 'quad', 'k1', 'k2', or 'k3')"
        ))),
    }
}

fn parse_policy(kind: Option<&str>, edges: Option<usize>) -> Result<SingletonPolicy, PyErr> {
    match kind.map(|k| k.to_ascii_lowercase()) {
        None => Ok(SingletonPolicy::Use), // library default
        Some(ref v) if v == "off" => Ok(SingletonPolicy::Off),
        Some(ref v) if v == "use" => Ok(SingletonPolicy::Use),
        Some(ref v) if v == "edges" || v == "usewithprotectededges" => {
            Ok(SingletonPolicy::UseWithProtectedEdges(edges.unwrap_or(3)))
        }
        Some(v) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid singleton_policy: {v} (expected 'off', 'use', or 'edges')"
        ))),
    }
}

#[pyclass(name = "TDigest", subclass)]
pub struct PyTDigest {
    inner: TDigest,
}

#[pymethods]
impl PyTDigest {
    /// Build from a Python array-like of floats (float32 or float64).
    ///
    /// Example:
    ///   TDigest.from_array(values, max_size=200, scale="k2",
    ///                      f32_mode=True,
    ///                      singleton_policy="edges", edges=4)
    ///
    /// Notes:
    /// - `f32_mode=True` quantizes centroids to 32-bit precision (means & weights).
    /// - Binary serialization (to_bytes/from_bytes) remains canonical, independent of `f32_mode`.
    #[staticmethod]
    #[pyo3(signature = (values, max_size=1000, scale=None, f32_mode=false, singleton_policy=None, edges=None))]
    pub fn from_array(
        py: Python<'_>,
        values: PyObject,
        max_size: usize,
        scale: Option<&str>,
        f32_mode: bool,
        singleton_policy: Option<&str>,
        edges: Option<usize>,
    ) -> PyResult<Self> {
        if max_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_size must be > 0",
            ));
        }
        let sc = parse_scale(scale)?;
        let policy = parse_policy(singleton_policy, edges)?;

        // Coerce to NumPy float64 and extract
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (values,))?;
        let values: Vec<f64> = arr.extract()?;

        // Build/ingest
        let base = TDigestBuilder::new()
            .max_size(max_size)
            .scale(sc)
            .singleton_policy(policy)
            .build();
        let digest = base.merge_unsorted(values);

        // If f32_mode, quantize centroids to ~32-bit precision
        let inner = if f32_mode {
            quantize_digest_to_f32(&digest)
        } else {
            digest
        };

        Ok(Self { inner })
    }

    /// Median (p=0.5)
    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.median())
    }

    /// Quantile for scalar probability p in [0,1]
    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        Ok(self.inner.quantile(q))
    }

    /// CDF evaluated at array-like x (returns a NumPy array)
    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;
        let values: Vec<f64> = arr.extract()?;
        let ys: Vec<f64> = self.inner.cdf(&values);
        let out = np.call_method1("asarray", (ys,))?;
        Ok(out.unbind())
    }

    /// Serialize digest to bytes (canonical bincode of the TDigest itself).
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = config::standard();
        let bytes = encode_to_vec(&self.inner, cfg).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize error: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Deserialize digest from bytes produced by `to_bytes`.
    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let cfg = config::standard();
        let (inner, _len): (TDigest, usize) =
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

/* ==================== private helpers ==================== */

/// Rebuild a digest with centroids quantized to f32 (means & weights).
/// Keeps sum/count/min/max as-is to preserve global stats.
fn quantize_digest_to_f32(td: &TDigest) -> TDigest {
    let cents_q: Vec<Centroid> = td
        .centroids()
        .iter()
        .map(|c| {
            let m = c.mean() as f32 as f64;
            let w = c.weight() as f32 as f64;
            Centroid::new(m, w)
        })
        .collect();

    TDigest::builder()
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
