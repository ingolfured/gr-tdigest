//! Immutable digest snapshots for batched HIP queries.
use super::{precision::FloatLike, summation::Sum, TDigest};
use ordered_float::FloatCore;
use std::{
    ffi::{c_char, c_void, CStr},
    ptr,
};

#[repr(C)]
struct Row {
    mean: f64,
    weight: f64,
    prefix: f64,
    atomic: u64,
    center: f64,
    boundary: f64,
    raw_weight: f64,
}
unsafe extern "C" {
    fn gr_gpu_upload(
        device: i32,
        rows: *const Row,
        n: usize,
        total: f64,
        count: f64,
        min: f64,
        max: f64,
        out: *mut *mut c_void,
    ) -> i32;
    fn gr_gpu_drop(s: *mut c_void);
    fn gr_gpu_query(
        s: *mut c_void,
        quantile: bool,
        probes: *const f64,
        out: *mut *const f64,
        n: usize,
    ) -> i32;
    fn gr_gpu_error(code: i32) -> *const c_char;
}
pub(crate) const BAD_Q: &str = "quantile probabilities must be finite and in [0,1]";
fn check(code: i32) -> Result<(), String> {
    if code == -1 {
        return Err(BAD_Q.into());
    }
    if code == 0 {
        return Ok(());
    }
    // HIP provides a static, nul-terminated string for an error code.
    Err(unsafe { CStr::from_ptr(gr_gpu_error(code)) }
        .to_string_lossy()
        .into_owned())
}

/// Owns the uploaded rows and reuses device scratch space. Queries require exclusive access.
pub struct GpuDigest {
    handle: ptr::NonNull<c_void>,
    device: i32,
    wire: Vec<u8>,
    config: training::Config,
    stats: training::Stats,
}
// The shim selects the snapshot's device on each call; all scratch access requires &mut self.
unsafe impl Send for GpuDigest {}
unsafe impl Sync for GpuDigest {}
impl Drop for GpuDigest {
    fn drop(&mut self) {
        unsafe {
            gr_gpu_drop(self.handle.as_ptr());
        }
    }
}
impl GpuDigest {
    pub fn upload<F: FloatLike + FloatCore>(d: &TDigest<F>, device: i32) -> Result<Self, String> {
        d.validate_state().map_err(|e| e.to_string())?;
        let raw = d;
        let normalized;
        let total: f64 = raw.centroids().iter().map(|c| c.weight_f64()).sum();
        let d = if total > 0. && total < f64::MIN_POSITIVE {
            normalized = raw.normalized_subnormal_mass_for_queries(total);
            &normalized
        } else {
            raw
        };
        let total: f64 = d.centroids().iter().map(|c| c.weight_f64()).sum();
        let mut prefix = 0.;
        let mut center = 0.;
        let mut previous = 0.;
        let mut boundary = Sum::default();
        let rows: Vec<_> = d
            .centroids()
            .iter()
            .zip(raw.centroids())
            .map(|(c, original)| {
                let weight = c.weight_f64();
                center += (previous + weight) * 0.5;
                previous = weight;
                boundary.add(weight);
                let row = Row {
                    mean: c.mean_f64(),
                    weight: c.weight_f64(),
                    prefix,
                    atomic: u64::from(c.is_atomic()),
                    center,
                    raw_weight: original.weight_f64(),
                    boundary: boundary.total(),
                };
                prefix += row.weight;
                row
            })
            .collect();
        let mut handle = ptr::null_mut();
        // The shim copies rows before returning; only a successful call transfers ownership.
        let status = unsafe {
            gr_gpu_upload(
                device,
                rows.as_ptr(),
                rows.len(),
                total,
                d.count(),
                d.min(),
                d.max(),
                &mut handle,
            )
        };
        if let Err(error) = check(status) {
            if !handle.is_null() {
                unsafe {
                    gr_gpu_drop(handle);
                }
            }
            return Err(error);
        }
        Ok(Self {
            handle: ptr::NonNull::new(handle).ok_or("HIP returned no snapshot")?,
            device,
            wire: raw.to_bytes(),
            config: training::Config::of(raw),
            stats: training::Stats::of(raw),
        })
    }
    pub fn to_bytes(&self) -> &[u8] {
        &self.wire
    }

    /// The completed host result remains valid until the next query.
    pub fn cdf(&mut self, probes: &[f64]) -> Result<&[f64], String> {
        self.query(probes, false)
    }

    pub fn quantile(&mut self, probabilities: &[f64]) -> Result<&[f64], String> {
        if probabilities.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(BAD_Q.into());
        }
        self.query(probabilities, true)
    }

    pub fn median(&mut self) -> Result<f64, String> {
        Ok(self.quantile(&[0.5])?[0])
    }

    fn query(&mut self, probes: &[f64], quantile: bool) -> Result<&[f64], String> {
        if probes.is_empty() {
            return Ok(&[]);
        }
        let mut out = ptr::null();
        check(unsafe {
            gr_gpu_query(
                self.handle.as_ptr(),
                quantile,
                probes.as_ptr(),
                &mut out,
                probes.len(),
            )
        })?;
        // The shim returns synchronized mapped memory owned by self; the borrow
        // prevents another query (or destruction) while the caller reads it.
        Ok(unsafe { std::slice::from_raw_parts(out, probes.len()) })
    }
}

mod array;
pub use array::DeviceArray;

mod training;
pub use training::TrainingError;
