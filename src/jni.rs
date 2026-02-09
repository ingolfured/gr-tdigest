// src/jni.rs
#![cfg(feature = "java")]

use jni::objects::{JByteArray, JClass, JDoubleArray, JFloatArray, JObject, JString, JValue};
use jni::sys::{
    jarray, jboolean, jbyte, jbyteArray, jdouble, jdoubleArray, jint, jlong, JNI_VERSION_1_8,
};
use jni::{JNIEnv, JavaVM};

use std::ffi::c_void;
use std::slice;
use std::sync::OnceLock;

use crate::tdigest::frontends::parse_scale_str;
use crate::tdigest::frontends::policy_from_code_edges;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::{decode_digest, encode_digest, WireDecodedDigest};
use crate::tdigest::{ScaleFamily, TDigest};
use crate::{TdError, TdResult};

// Retain VM
static JVM: OnceLock<JavaVM> = OnceLock::new();
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: JavaVM, _reserved: *mut c_void) -> jint {
    let _ = JVM.set(vm);
    JNI_VERSION_1_8
}
#[no_mangle]
pub extern "system" fn JNI_OnUnload(_vm: JavaVM, _reserved: *mut c_void) {}

// -------------------- Helpers --------------------

fn map_scale(s: &str) -> ScaleFamily {
    parse_scale_str(Some(s)).unwrap_or(ScaleFamily::K2)
}

fn map_policy(code: jint, edges: jint) -> Result<SingletonPolicy, String> {
    policy_from_code_edges(code as i32, edges as i32).map_err(|e| e.to_string())
}

fn ensure_non_null_handle(handle: jlong) -> Result<(), &'static str> {
    if handle == 0 {
        Err("TDigest handle is null")
    } else {
        Ok(())
    }
}

fn read_jdouble_array(env: &JNIEnv, arr: &JDoubleArray) -> Result<Vec<f64>, String> {
    let len = env.get_array_length(arr).unwrap_or(0) as usize;
    let mut out = vec![0f64; len];
    if len > 0 {
        env.get_double_array_region(arr, 0, &mut out)
            .map_err(|e| format!("jni: get_double_array_region failed: {e:?}"))?;
    }
    Ok(out)
}

fn read_jfloat_array(env: &JNIEnv, arr: &JFloatArray) -> Result<Vec<f32>, String> {
    let len = env.get_array_length(arr).unwrap_or(0) as usize;
    let mut out = vec![0f32; len];
    if len > 0 {
        env.get_float_array_region(arr, 0, &mut out)
            .map_err(|e| format!("jni: get_float_array_region failed: {e:?}"))?;
    }
    Ok(out)
}

// --------------- Native handle ---------------

#[derive(Clone)]
enum NativeInner {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

#[derive(Clone)]
struct NativeDigest {
    inner: NativeInner,
}

impl NativeDigest {
    fn from_values(
        values: Vec<f64>,
        max_size: usize,
        scale: ScaleFamily,
        policy: SingletonPolicy,
        f32mode: bool,
    ) -> TdResult<Self> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(TdError::NonFiniteInput {
                context: "sample value (NaN or ±inf)",
            });
        }

        if f32mode {
            // Compact backend: TDigest<f32>
            let xs32: Vec<f32> = values.iter().map(|v| *v as f32).collect();
            let d = TDigest::<f32>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .build()
                .merge_unsorted(xs32)?; // forward core Result
            Ok(Self {
                inner: NativeInner::F32(d),
            })
        } else {
            // Full-precision backend: TDigest<f64>
            let d = TDigest::<f64>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .build()
                .merge_unsorted(values)?; // forward core Result
            Ok(Self {
                inner: NativeInner::F64(d),
            })
        }
    }

    fn cdf(&self, xs: &[f64]) -> Vec<f64> {
        match &self.inner {
            NativeInner::F32(td) => td.cdf_or_nan(xs),
            NativeInner::F64(td) => td.cdf_or_nan(xs),
        }
    }

    fn quantile(&self, p: f64) -> Result<f64, &'static str> {
        if !p.is_finite() {
            return Err("q must be a finite number in [0,1]");
        }
        if !(0.0..=1.0).contains(&p) {
            return Err("q must be in [0,1]");
        }
        let v = match &self.inner {
            NativeInner::F32(td) => td.quantile(p),
            NativeInner::F64(td) => td.quantile(p),
        };
        Ok(v)
    }

    fn merge_f64_values(&mut self, values: Vec<f64>) -> TdResult<()> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(TdError::NonFiniteInput {
                context: "sample value (NaN or ±inf)",
            });
        }
        self.inner = match &self.inner {
            NativeInner::F32(td) => {
                let xs32: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                NativeInner::F32(td.merge_unsorted(xs32)?)
            }
            NativeInner::F64(td) => NativeInner::F64(td.merge_unsorted(values)?),
        };
        Ok(())
    }

    fn merge_digest(&mut self, other: &NativeDigest) -> Result<(), String> {
        self.inner = match (&self.inner, &other.inner) {
            (NativeInner::F32(a), NativeInner::F32(b)) => {
                NativeInner::F32(TDigest::<f32>::merge_digests(vec![a.clone(), b.clone()]))
            }
            (NativeInner::F64(a), NativeInner::F64(b)) => {
                NativeInner::F64(TDigest::<f64>::merge_digests(vec![a.clone(), b.clone()]))
            }
            _ => {
                return Err(
                    "tdigest: cannot merge digests of different precision (f32 vs f64)".to_string(),
                )
            }
        };
        Ok(())
    }
}

// Handle helpers
fn into_handle<T>(b: Box<T>) -> jlong {
    Box::into_raw(b) as jlong
}
unsafe fn from_handle<'a, T>(h: jlong) -> &'a mut T {
    &mut *(h as *mut T)
}
unsafe fn from_handle_const<'a, T>(h: jlong) -> &'a T {
    &*(h as *const T)
}

// Throw helper
fn throw_illegal_arg(env: &mut JNIEnv, msg: String) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", msg);
}

// ----------------------------- JNI exports -----------------------------

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

    let nd_res: Result<NativeDigest, String> = match decode_digest(ubuf) {
        Ok(WireDecodedDigest::F32(td32)) => Ok(NativeDigest {
            inner: NativeInner::F32(td32),
        }),
        Ok(WireDecodedDigest::F64(td64)) => Ok(NativeDigest {
            inner: NativeInner::F64(td64),
        }),
        Err(e) => Err(format!("jni: decode TDigest failed: {e}")),
    };

    match nd_res {
        Ok(nd) => into_handle(Box::new(nd)),
        Err(msg) => {
            throw_illegal_arg(&mut env, msg);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_toBytes(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jbyteArray {
    // Canonical TDIG wire format — width follows backend precision:
    // - F32 → f32 means on the wire
    // - F64 → f64 means on the wire
    let bytes: Vec<u8> = if handle == 0 {
        Vec::new()
    } else {
        let d = unsafe { from_handle::<NativeDigest>(handle) };
        match &d.inner {
            NativeInner::F32(td) => encode_digest(td),
            NativeInner::F64(td) => encode_digest(td),
        }
    };

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

#[allow(unused_mut)] // some builds warn; we *do* pass &mut env on error paths
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

    if env.is_instance_of(&values_obj, "[D").unwrap_or(false) {
        let raw: jarray = env
            .new_local_ref(&values_obj)
            .expect("new_local_ref")
            .as_raw();
        let darr = unsafe { JDoubleArray::from_raw(raw) };
        match read_jdouble_array(&env, &darr) {
            Ok(v) => xs = v,
            Err(msg) => {
                throw_illegal_arg(&mut env, msg);
                return 0;
            }
        }
    } else if env.is_instance_of(&values_obj, "[F").unwrap_or(false) {
        let raw: jarray = env
            .new_local_ref(&values_obj)
            .expect("new_local_ref")
            .as_raw();
        let farr = unsafe { JFloatArray::from_raw(raw) };
        match read_jfloat_array(&env, &farr) {
            Ok(v) => xs = v.into_iter().map(|x| x as f64).collect(),
            Err(msg) => {
                throw_illegal_arg(&mut env, msg);
                return 0;
            }
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
    let policy = match map_policy(policy_code, edges) {
        Ok(p) => p,
        Err(msg) => {
            throw_illegal_arg(&mut env, msg);
            return 0;
        }
    };
    let f32 = f32mode != 0;

    match NativeDigest::from_values(xs, max_size.max(1) as usize, sf, policy, f32) {
        Ok(nd) => into_handle(Box::new(nd)),
        Err(e) => {
            throw_illegal_arg(&mut env, e.to_string());
            0
        }
    }
}

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

#[allow(unused_mut)]
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_mergeArrayF64(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
) {
    if let Err(msg) = ensure_non_null_handle(handle) {
        throw_illegal_arg(&mut env, msg.to_string());
        return;
    }
    let xs = match read_jdouble_array(&env, &values) {
        Ok(v) => v,
        Err(msg) => {
            throw_illegal_arg(&mut env, msg);
            return;
        }
    };

    let d = unsafe { from_handle::<NativeDigest>(handle) };
    if let Err(e) = d.merge_f64_values(xs) {
        throw_illegal_arg(&mut env, e.to_string());
    }
}

#[allow(unused_mut)]
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_mergeArrayF32(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JFloatArray,
) {
    if let Err(msg) = ensure_non_null_handle(handle) {
        throw_illegal_arg(&mut env, msg.to_string());
        return;
    }
    let xs32 = match read_jfloat_array(&env, &values) {
        Ok(v) => v,
        Err(msg) => {
            throw_illegal_arg(&mut env, msg);
            return;
        }
    };
    let xs: Vec<f64> = xs32.into_iter().map(|v| v as f64).collect();

    let d = unsafe { from_handle::<NativeDigest>(handle) };
    if let Err(e) = d.merge_f64_values(xs) {
        throw_illegal_arg(&mut env, e.to_string());
    }
}

#[allow(unused_mut)]
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_mergeDigest(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    other_handle: jlong,
) {
    if let Err(msg) = ensure_non_null_handle(handle) {
        throw_illegal_arg(&mut env, msg.to_string());
        return;
    }
    if let Err(msg) = ensure_non_null_handle(other_handle) {
        throw_illegal_arg(&mut env, msg.to_string());
        return;
    }

    let other_owned = unsafe { from_handle_const::<NativeDigest>(other_handle).clone() };
    let d = unsafe { from_handle::<NativeDigest>(handle) };
    if let Err(e) = d.merge_digest(&other_owned) {
        throw_illegal_arg(&mut env, e);
    }
}

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
        eprintln!("jni: get_double_array_region failed: {e:?}");
        return out.into_raw();
    }

    let ps = d.cdf(&in_buf);
    env.set_double_array_region(&out, 0, &ps)
        .expect("set_double_array_region");
    out.into_raw()
}

#[allow(unused_mut)] // mutable only needed on the exceptional path
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_TDigestNative_quantile(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    q: jdouble,
) -> jdouble {
    if handle == 0 {
        return f64::NAN;
    }
    let d = unsafe { from_handle::<NativeDigest>(handle) };
    match d.quantile(q as f64) {
        Ok(v) => v,
        Err(msg) => {
            let mut env2 = env;
            throw_illegal_arg(&mut env2, msg.to_string());
            f64::NAN
        }
    }
}

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
    d.quantile(0.5).unwrap_or(f64::NAN)
}
