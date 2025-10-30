// src/jni.rs
//
// JNI bindings for package `gr.tdigest` (class TDigestNative).
// Bind via symbol names (no RegisterNatives). No class lookups in JNI_OnLoad.
//
// Enabled behind feature "java".

#![cfg(feature = "java")]

use jni::objects::{JByteArray, JClass, JDoubleArray, JFloatArray, JObject, JString, JValue};
use jni::sys::{
    jarray, jboolean, jbyte, jbyteArray, jdouble, jdoubleArray, jint, jlong, JNI_VERSION_1_8,
};
use jni::{JNIEnv, JavaVM};
use std::cmp::Ordering;
use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::OnceLock;

// Optionally retain the VM (useful if you later attach threads for callbacks).
static JVM: OnceLock<JavaVM> = OnceLock::new();
pub fn java_vm() -> Option<&'static JavaVM> {
    JVM.get()
}

#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: JavaVM, _reserved: *mut c_void) -> jint {
    let _ = JVM.set(vm);
    JNI_VERSION_1_8
}

#[no_mangle]
pub extern "system" fn JNI_OnUnload(_vm: JavaVM, _reserved: *mut c_void) {}

// ----------------------------- Minimal native digest -----------------------------

struct NativeDigest {
    xs: Vec<f64>, // sorted ascending
}

impl NativeDigest {
    fn new(mut xs: Vec<f64>) -> Self {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Self { xs }
    }

    fn quantile(&self, p: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 {
            return f64::NAN;
        }
        let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        if n == 1 {
            return self.xs[0];
        }
        let pos = (n - 1) as f64 * p;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            self.xs[lo]
        } else {
            let w = pos - lo as f64;
            self.xs[lo] * (1.0 - w) + self.xs[hi] * w
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 {
            return f64::NAN;
        }
        if x <= self.xs[0] {
            return 0.0;
        }
        if x >= self.xs[n - 1] {
            return 1.0;
        }
        // upper-bound binary search
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = self.xs[hi] - self.xs[lo];
        let t = if span <= 0.0 {
            0.0
        } else {
            (x - self.xs[lo]) / span
        };
        (lo as f64 + t) / (n - 1) as f64
    }
}

// Handle helpers
fn into_handle<T>(b: Box<T>) -> jlong {
    Box::into_raw(b) as jlong
}
unsafe fn from_handle<'a, T>(h: jlong) -> &'a mut T {
    &mut *(h as *mut T)
}

// ----------------------------- Helpers: bytes <-> f64 -----------------------------

fn f64s_to_le_bytes(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * size_of::<f64>());
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn le_bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    if b.len() % size_of::<f64>() != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(b.len() / size_of::<f64>());
    for chunk in b.chunks_exact(8) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        out.push(f64::from_le_bytes(arr));
    }
    out
}

// ----------------------------- JNI exports (TDigestNative) -----------------------------

/// Java: `static native long fromBytes(byte[] bytes);`

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_fromBytes(
    env: JNIEnv,
    _cls: JClass,
    bytes: JByteArray,
) -> jlong {
    let len = env.get_array_length(&bytes).unwrap_or(0) as usize;
    let mut buf = vec![0i8; len];
    if len > 0 {
        env.get_byte_array_region(&bytes, 0, &mut buf)
            .expect("get_byte_array_region");
    }
    // reinterpret i8 -> u8
    let ubuf: &[u8] = unsafe { slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let xs = le_bytes_to_f64s(ubuf);
    into_handle(Box::new(NativeDigest::new(xs)))
}

/// Java: `static native byte[] toBytes(long handle);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_toBytes(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jbyteArray {
    if handle == 0 {
        return env
            .new_byte_array(0)
            .expect("new_byte_array(empty)")
            .into_raw();
    }
    let d = unsafe { from_handle::<NativeDigest>(handle) };
    let bytes = f64s_to_le_bytes(&d.xs);
    let arr = env
        .new_byte_array(bytes.len() as jint)
        .expect("new_byte_array");
    if !bytes.is_empty() {
        // cast &[u8] -> &[i8]
        let ptr = bytes.as_ptr() as *const jbyte;
        let slice_i8 = unsafe { slice::from_raw_parts(ptr, bytes.len()) };
        env.set_byte_array_region(&arr, 0, slice_i8)
            .expect("set_byte_array_region");
    }
    arr.into_raw()
}

/// Java: `static native long fromArray(Object values, int maxSize, String scale, int policyCode, int edges, boolean f32mode);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_fromArray(
    mut env: JNIEnv,
    _cls: JClass,
    values_obj: JObject,
    _max_size: jint,
    _scale: JString,
    _policy_code: jint,
    _edges: jint,
    _f32mode: jboolean,
) -> jlong {
    let mut xs: Vec<f64> = Vec::new();

    // Primitive arrays: double[] "[D" / float[] "[F"
    if env.is_instance_of(&values_obj, "[D").unwrap_or(false) {
        let arr_obj = env.new_local_ref(&values_obj).expect("new_local_ref");
        let raw: jarray = arr_obj.as_raw();
        let darr = unsafe { JDoubleArray::from_raw(raw) };
        let len = env.get_array_length(&darr).unwrap_or(0) as usize;
        if len > 0 {
            let mut buf = vec![0f64; len];
            env.get_double_array_region(&darr, 0, &mut buf)
                .expect("get_double_array_region");
            xs = buf;
        }
        return into_handle(Box::new(NativeDigest::new(xs)));
    }

    if env.is_instance_of(&values_obj, "[F").unwrap_or(false) {
        let arr_obj = env.new_local_ref(&values_obj).expect("new_local_ref");
        let raw: jarray = arr_obj.as_raw();
        let farr = unsafe { JFloatArray::from_raw(raw) };
        let len = env.get_array_length(&farr).unwrap_or(0) as usize;
        if len > 0 {
            let mut buf_f = vec![0f32; len];
            env.get_float_array_region(&farr, 0, &mut buf_f)
                .expect("get_float_array_region");
            xs = buf_f.into_iter().map(|v| v as f64).collect();
        }
        return into_handle(Box::new(NativeDigest::new(xs)));
    }

    // Best-effort: java.util.List<Double>
    if env
        .is_instance_of(&values_obj, "java/util/List")
        .unwrap_or(false)
    {
        let size = env
            .call_method(&values_obj, "size", "()I", &[])
            .and_then(|v| v.i())
            .unwrap_or(0);
        let mut out = Vec::with_capacity(size as usize);
        for i in 0..size {
            let elem = env
                .call_method(
                    &values_obj,
                    "get",
                    "(I)Ljava/lang/Object;",
                    &[JValue::from(i)],
                )
                .ok()
                .and_then(|v| v.l().ok());
            if let Some(obj) = elem {
                if let Ok(dval) = env.call_method(obj, "doubleValue", "()D", &[]) {
                    if let Ok(d) = dval.d() {
                        out.push(d);
                    }
                }
            }
        }
        xs = out;
    }

    into_handle(Box::new(NativeDigest::new(xs)))
}

/// Java: `static native void free(long handle);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_free(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut NativeDigest)) };
    }
}

/// Java: `static native double[] cdf(long handle, double[] values);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_cdf(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
) -> jdoubleArray {
    let n = env.get_array_length(&values).unwrap_or(0) as usize;
    let out = env.new_double_array(n as jint).expect("new_double_array");
    if handle == 0 || n == 0 {
        return out.into_raw();
    }
    let d = unsafe { from_handle::<NativeDigest>(handle) };

    // Read input doubles
    let mut in_buf = vec![0f64; n];
    env.get_double_array_region(&values, 0, &mut in_buf)
        .expect("get_double_array_region");

    // Compute cdf for each x
    let mut out_buf = Vec::with_capacity(n);
    for &x in &in_buf {
        out_buf.push(d.cdf(x));
    }

    env.set_double_array_region(&out, 0, &out_buf)
        .expect("set_double_array_region");
    out.into_raw()
}

/// Java: `static native double quantile(long handle, double q);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_quantile(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    q: jdouble,
) -> jdouble {
    if handle == 0 {
        return f64::NAN;
    }
    let d = unsafe { from_handle::<NativeDigest>(handle) };
    d.quantile(q as f64)
}

/// Java: `static native double median(long handle);`
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_median(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jdouble {
    if handle == 0 {
        return f64::NAN;
    }
    let d = unsafe { from_handle::<NativeDigest>(handle) };
    d.quantile(0.5)
}
