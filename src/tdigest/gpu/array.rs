//! Owned contiguous f64 device buffers. Transfers are explicit.
use super::{check, GpuDigest};
use std::{ffi::c_void, ptr};

unsafe extern "C" {
    fn gr_gpu_alloc(device: i32, input: *const f64, n: usize, out: *mut *mut f64) -> i32;
    fn gr_gpu_free(device: i32, p: *mut f64);
    fn gr_gpu_download(device: i32, p: *const f64, out: *mut f64, n: usize) -> i32;
    fn gr_gpu_resident_query(
        s: *mut c_void,
        q: bool,
        p: *const f64,
        out: *mut f64,
        n: usize,
    ) -> i32;
}
pub struct DeviceArray {
    ptr: *mut f64,
    len: usize,
    device: i32,
}
// Buffers are immutable after construction and the shim selects the owner device.
unsafe impl Send for DeviceArray {}
unsafe impl Sync for DeviceArray {}
impl Drop for DeviceArray {
    fn drop(&mut self) {
        unsafe { gr_gpu_free(self.device, self.ptr) }
    }
}
impl DeviceArray {
    fn alloc(n: usize, input: *const f64, device: i32) -> Result<Self, String> {
        let mut a = Self {
            ptr: ptr::null_mut(),
            len: n,
            device,
        };
        check(unsafe { gr_gpu_alloc(device, input, n, &mut a.ptr) })?;
        Ok(a)
    }
    pub fn upload(xs: &[f64], device: i32) -> Result<Self, String> {
        Self::alloc(xs.len(), xs.as_ptr(), device)
    }
    pub fn to_host(&self) -> Result<Vec<f64>, String> {
        let mut xs = vec![0.; self.len];
        check(unsafe { gr_gpu_download(self.device, self.ptr, xs.as_mut_ptr(), self.len) })?;
        Ok(xs)
    }
}
impl GpuDigest {
    fn query_device(&mut self, xs: &DeviceArray, q: bool) -> Result<DeviceArray, String> {
        if self.1 != xs.device {
            return Err("buffer and digest must be on the same device".into());
        }
        let out = DeviceArray::alloc(xs.len, ptr::null(), self.1)?;
        check(unsafe { gr_gpu_resident_query(self.0.as_ptr(), q, xs.ptr, out.ptr, xs.len) })?;
        Ok(out)
    }
    pub fn cdf_device(&mut self, x: &DeviceArray) -> Result<DeviceArray, String> {
        self.query_device(x, false)
    }
    pub fn quantile_device(&mut self, x: &DeviceArray) -> Result<DeviceArray, String> {
        self.query_device(x, true)
    }
}
