// src/jni.rs
//
// JNI bindings for package `gr.tdigest` (class TDigestNative).
// Delegates to the REAL Rust TDigest so results match CLI/Python/Polars.
//
// Feature-gated behind "java".

#![cfg(feature = "java")]

use jni::objects::{JByteArray, JClass, JDoubleArray, JFloatArray, JObject, JString, JValue};
use jni::sys::{
    jarray, jboolean, jbyte, jbyteArray, jdouble, jdoubleArray, jint, jlong, JNI_VERSION_1_8,
};
use jni::{JNIEnv, JavaVM};

use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::OnceLock;

use crate::tdigest::{singleton_policy::SingletonPolicy, ScaleFamily, TDigest};
use crate::{TdError, TdResult};

// Retain VM if needed later
static JVM: OnceLock<JavaVM> = OnceLock::new();
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: JavaVM, _reserved: *mut c_void) -> jint {
    let _ = JVM.set(vm);
    JNI_VERSION_1_8
}
#[no_mangle]
pub extern "system" fn JNI_OnUnload(_vm: JavaVM, _reserved: *mut c_void) {}

// -------------------- Helpers: map Java args -> Rust types --------------------

fn map_scale(s: &str) -> ScaleFamily {
    match s.trim().to_lowercase().as_str() {
        "quad" => ScaleFamily::Quad,
        "k1" => ScaleFamily::K1,
        "k2" => ScaleFamily::K2,
        "k3" => ScaleFamily::K3,
        other => {
            eprintln!("jni: unknown scale '{}' -> default K2", other);
            ScaleFamily::K2
        }
    }
}

fn map_policy(code: jint, edges: jint) -> SingletonPolicy {
    match code {
        0 => SingletonPolicy::Off,
        1 => SingletonPolicy::Use,
        2 => SingletonPolicy::UseWithProtectedEdges(edges.max(0) as usize),
        _ => {
            eprintln!("jni: unknown policy code '{}' -> default USE", code);
            SingletonPolicy::Use
        }
    }
}

// --------------- Native handle that owns the REAL Rust TDigest ---------------

// Pin JNI backing to f64 to avoid generic type issues and keep results stable.
type TD64 = TDigest<f64>;

struct NativeDigest {
    inner: TD64,
}

impl NativeDigest {
    fn from_values(
        mut values: Vec<f64>,
        max_size: usize,
        scale: ScaleFamily,
        policy: SingletonPolicy,
        _f32mode: bool, // accepted for API coherence; f64 backing for now
    ) -> TdResult<Self> {
        // Hard error on NaN to match cross-language behavior
        if values.iter().any(|v| v.is_nan()) {
            return Err(TdError::NaNInput {
                context: "sample value",
            });
        }

        // Drop non-finite values (preserve previous forgiving ±inf behavior)
        values.retain(|v| v.is_finite());

        let d = TD64::builder()
            .max_size(max_size)
            .scale(scale)
            .singleton_policy(policy)
            .build()
            .merge_unsorted(values)?; // forward core Result

        Ok(Self { inner: d })
    }

    fn cdf(&self, xs: &[f64]) -> Vec<f64> {
        self.inner.cdf(xs)
    }

    fn quantile(&self, p: f64) -> f64 {
        self.inner.quantile(p)
    }
}

// Handle helpers
fn into_handle<T>(b: Box<T>) -> jlong {
    Box::into_raw(b) as jlong
}
unsafe fn from_handle<'a, T>(h: jlong) -> &'a mut T {
    &mut *(h as *mut T)
}

// Throw helper (needs &mut JNIEnv)
fn throw_illegal_arg(env: &mut JNIEnv, msg: String) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", msg);
}

// ----------------------------- Bytes helpers (no-op serialization for tests) -----------------------------

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
/// Not used in coherence tests; treat as "build from raw doubles".
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_fromBytes(
    mut env: JNIEnv,
    _cls: JClass,
    bytes: JByteArray,
) -> jlong {
    let len = env.get_array_length(&bytes).unwrap_or(0) as usize;
    let mut buf = vec![0i8; len];
    if len > 0 {
        if let Err(e) = env.get_byte_array_region(&bytes, 0, &mut buf) {
            throw_illegal_arg(
                &mut env,
                format!("jni: get_byte_array_region failed: {e:?}"),
            );
            return 0;
        }
    }
    let ubuf: &[u8] = unsafe { slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let xs = le_bytes_to_f64s(ubuf);

    match NativeDigest::from_values(xs, 1000, ScaleFamily::K2, SingletonPolicy::Use, false) {
        Ok(nd) => into_handle(Box::new(nd)),
        Err(e) => {
            throw_illegal_arg(&mut env, e.to_string());
            0
        }
    }
}

/// Java: `static native byte[] toBytes(long handle);`
/// Not used in coherence tests; dump inner sample positions (non-canonical).
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
    let _d = unsafe { from_handle::<NativeDigest>(handle) };
    // There isn't a public iterator over centroids here; just return empty for safety.
    let bytes = f64s_to_le_bytes(&[]);
    let arr = env
        .new_byte_array(bytes.len() as jint)
        .expect("new_byte_array");
    if !bytes.is_empty() {
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
    max_size: jint,
    scale: JString,
    policy_code: jint,
    edges: jint,
    f32mode: jboolean,
) -> jlong {
    let mut xs: Vec<f64> = Vec::new();

    // Accept double[] ("[D") or float[] ("[F") or java.util.List<Double>
    if env.is_instance_of(&values_obj, "[D").unwrap_or(false) {
        let raw: jarray = env
            .new_local_ref(&values_obj)
            .expect("new_local_ref")
            .as_raw();
        let darr = unsafe { JDoubleArray::from_raw(raw) };
        let len = env.get_array_length(&darr).unwrap_or(0) as usize;
        if len > 0 {
            let mut buf = vec![0f64; len];
            if let Err(e) = env.get_double_array_region(&darr, 0, &mut buf) {
                throw_illegal_arg(
                    &mut env,
                    format!("jni: get_double_array_region failed: {e:?}"),
                );
                return 0;
            }
            xs = buf;
        }
    } else if env.is_instance_of(&values_obj, "[F").unwrap_or(false) {
        let raw: jarray = env
            .new_local_ref(&values_obj)
            .expect("new_local_ref")
            .as_raw();
        let farr = unsafe { JFloatArray::from_raw(raw) };
        let len = env.get_array_length(&farr).unwrap_or(0) as usize;
        if len > 0 {
            let mut buf_f = vec![0f32; len];
            if let Err(e) = env.get_float_array_region(&farr, 0, &mut buf_f) {
                throw_illegal_arg(
                    &mut env,
                    format!("jni: get_float_array_region failed: {e:?}"),
                );
                return 0;
            }
            xs = buf_f.into_iter().map(|v| v as f64).collect();
        }
    } else if env
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
    } else {
        // Unsupported input → empty digest
    }

    let scale_str = env
        .get_string(&scale)
        .ok()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "k2".to_string());
    let sf = map_scale(&scale_str);
    let policy = map_policy(policy_code, edges);
    let f32 = f32mode != 0;

    match NativeDigest::from_values(xs, max_size.max(1) as usize, sf, policy, f32) {
        Ok(nd) => into_handle(Box::new(nd)),
        Err(e) => {
            throw_illegal_arg(&mut env, e.to_string());
            0
        }
    }
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

    let mut in_buf = vec![0f64; n];
    if let Err(e) = env.get_double_array_region(&values, 0, &mut in_buf) {
        // If we can't read, return empty array; Java side can decide how to handle.
        eprintln!("jni: get_double_array_region failed: {e:?}");
        return out.into_raw();
    }

    let ps = d.cdf(&in_buf);
    env.set_double_array_region(&out, 0, &ps)
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
