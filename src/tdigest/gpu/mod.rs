//! Immutable digest snapshots for batched HIP queries.
use super::{precision::FloatLike, TDigest};
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
}
unsafe extern "C" {
    fn gr_gpu_upload(
        device: i32,
        rows: *const Row,
        n: usize,
        total: f64,
        min: f64,
        max: f64,
        out: *mut *mut c_void,
    ) -> i32;
    fn gr_gpu_drop(s: *mut c_void);
    fn gr_gpu_cdf(s: *mut c_void, probes: *const f64, out: *mut *const f64, n: usize) -> i32;
    fn gr_gpu_error(code: i32) -> *const c_char;
}
fn check(code: i32) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    // HIP provides a static, nul-terminated string for an error code.
    Err(unsafe { CStr::from_ptr(gr_gpu_error(code)) }
        .to_string_lossy()
        .into_owned())
}

/// Owns the uploaded rows and reuses device scratch space. Queries require exclusive access.
pub struct GpuDigest(ptr::NonNull<c_void>);
// The shim selects the snapshot's device on each call; all scratch access requires &mut self.
unsafe impl Send for GpuDigest {}
unsafe impl Sync for GpuDigest {}
impl Drop for GpuDigest {
    fn drop(&mut self) {
        unsafe {
            gr_gpu_drop(self.0.as_ptr());
        }
    }
}
impl GpuDigest {
    pub fn upload<F: FloatLike + FloatCore>(d: &TDigest<F>, device: i32) -> Result<Self, String> {
        let total: f64 = d.centroids().iter().map(|c| c.weight_f64()).sum();
        if total > 0. && total < f64::MIN_POSITIVE {
            return Self::upload(&d.normalized_subnormal_mass_for_queries(total), device);
        }
        let mut prefix = 0.;
        let rows: Vec<_> = d
            .centroids()
            .iter()
            .map(|c| {
                let row = Row {
                    mean: c.mean_f64(),
                    weight: c.weight_f64(),
                    prefix,
                    atomic: u64::from(c.is_atomic()),
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
        Ok(Self(
            ptr::NonNull::new(handle).ok_or("HIP returned no snapshot")?,
        ))
    }
    /// The completed host result remains valid until the next query.
    pub fn cdf(&mut self, probes: &[f64]) -> Result<&[f64], String> {
        if probes.is_empty() {
            return Ok(&[]);
        }
        let mut out = ptr::null();
        check(unsafe { gr_gpu_cdf(self.0.as_ptr(), probes.as_ptr(), &mut out, probes.len()) })?;
        // The shim returns synchronized mapped memory owned by self; the borrow
        // prevents another query (or destruction) while the caller reads it.
        Ok(unsafe { std::slice::from_raw_parts(out, probes.len()) })
    }
}
