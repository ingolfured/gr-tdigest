use std::ptr::NonNull;

use jni::objects::{JByteArray, JClass, JDoubleArray, JFloatArray, JObject, JString};
use jni::sys::{jboolean, jbyteArray, jdouble, jdoubleArray, jint, jlong};
use jni::JNIEnv;

use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};

use crate::tdigest::centroids::Centroid;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

/* ---------------- error helpers ---------------- */

fn throw_illegal_arg(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", msg);
}
fn throw_illegal_state(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/IllegalStateException", msg);
}

/* ---------------- handle helper ---------------- */

fn from_handle<'a>(env: &mut JNIEnv<'a>, handle: jlong) -> Option<&'a mut TDigest> {
    if handle == 0 {
        let _ = env.throw_new("java/lang/NullPointerException", "TDigest handle is null");
        return None;
    }
    let ptr = handle as *mut TDigest;
    NonNull::new(ptr)
        .map(|mut nn| unsafe { nn.as_mut() })
        .or_else(|| {
            throw_illegal_state(env, "TDigest handle was invalid");
            None
        })
}

/* ---------------- parse helpers ---------------- */

fn parse_scale(env: &mut JNIEnv, jscale: JString) -> Option<ScaleFamily> {
    let s: String = match env.get_string(&jscale) {
        Ok(js) => js.into(),
        Err(_) => {
            throw_illegal_arg(env, "scale must be a valid UTF-8 string");
            return None;
        }
    };
    let v = s.trim().to_ascii_lowercase();
    Some(match v.as_str() {
        "quad" => ScaleFamily::Quad,
        "k1" => ScaleFamily::K1,
        "k2" => ScaleFamily::K2,
        "k3" => ScaleFamily::K3,
        _ => {
            throw_illegal_arg(env, "scale must be one of: quad, k1, k2, k3");
            return None;
        }
    })
}

fn parse_policy_code(code: jint, edges: jint) -> SingletonPolicy {
    match code {
        0 => SingletonPolicy::Off,
        1 => SingletonPolicy::Use,
        2 => {
            let keep = if edges > 0 { edges as usize } else { 3 };
            SingletonPolicy::UseWithProtectedEdges(keep)
        }
        _ => SingletonPolicy::Use,
    }
}

/* ---------------- f32 quantization (like Python helper) ---------------- */

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

/* ================= JNI exports ================= */

/* Build from bytes (bincode) */

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_fromBytes(
    mut env: JNIEnv,
    _cls: JClass,
    bytes: JByteArray,
) -> jlong {
    let n = match env.get_array_length(&bytes) {
        Ok(n) => n as usize,
        Err(_) => {
            throw_illegal_arg(&mut env, "bytes length not accessible");
            return 0;
        }
    };
    // JNI byte is i8
    let mut buf_i8 = vec![0i8; n];
    if env.get_byte_array_region(&bytes, 0, &mut buf_i8).is_err() {
        throw_illegal_arg(&mut env, "unable to read bytes");
        return 0;
    }
    // Convert to u8 for bincode
    let buf_u8: Vec<u8> = buf_i8.iter().map(|b| *b as u8).collect();

    let cfg = config::standard();
    let (digest, _len): (TDigest, usize) = match decode_from_slice(&buf_u8, cfg) {
        Ok(v) => v,
        Err(e) => {
            throw_illegal_arg(&mut env, &format!("deserialize error: {e}"));
            return 0;
        }
    };
    Box::into_raw(Box::new(digest)) as jlong
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_toBytes(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jbyteArray {
    let Some(d) = from_handle(&mut env, handle) else {
        return std::ptr::null_mut();
    };
    let cfg = config::standard();
    let bytes_u8 = match encode_to_vec(d, cfg) {
        Ok(b) => b,
        Err(e) => {
            throw_illegal_state(&mut env, &format!("serialize error: {e}"));
            return std::ptr::null_mut();
        }
    };
    // Convert to i8 for JNI
    let bytes_i8: Vec<i8> = bytes_u8.iter().map(|b| *b as i8).collect();

    let arr = match env.new_byte_array(bytes_i8.len() as i32) {
        Ok(a) => a,
        Err(_) => return std::ptr::null_mut(),
    };
    let _ = env.set_byte_array_region(&arr, 0, &bytes_i8);
    arr.into_raw()
}

/* Single entrypoint: fromArray(Object values, ...) supports double[] and float[] */

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_fromArray(
    mut env: JNIEnv,
    _cls: JClass,
    values: JObject,
    max_size: jint,
    scale: JString,
    policy_code: jint,
    edges: jint,
    f32_mode: jboolean,
) -> jlong {
    let Some(scale) = parse_scale(&mut env, scale) else {
        return 0;
    };
    let ms = if max_size > 0 { max_size as usize } else { 0 };
    if ms == 0 {
        throw_illegal_arg(&mut env, "max_size must be > 0");
        return 0;
    }
    let policy = parse_policy_code(policy_code, edges);
    let want_f32 = f32_mode != 0;

    // Build base digest
    let base = TDigestBuilder::new()
        .max_size(ms)
        .scale(scale)
        .singleton_policy(policy)
        .build();

    // Detect values type: double[] -> "[D", float[] -> "[F"
    // Avoid aliasing mutable borrows of env by splitting the calls.
    let cls_double = match env.find_class("[D") {
        Ok(c) => c,
        Err(_) => {
            throw_illegal_state(&mut env, "unable to resolve double[] class");
            return 0;
        }
    };
    let cls_float = match env.find_class("[F") {
        Ok(c) => c,
        Err(_) => {
            throw_illegal_state(&mut env, "unable to resolve float[] class");
            return 0;
        }
    };
    let is_double_array = env.is_instance_of(&values, cls_double).unwrap_or(false);
    let is_float_array = env.is_instance_of(&values, cls_float).unwrap_or(false);

    let digest = if is_double_array {
        let values: JDoubleArray = JDoubleArray::from(values);
        let n = env.get_array_length(&values).unwrap_or(0) as usize;
        let mut buf = vec![0.0f64; n];
        if env.get_double_array_region(&values, 0, &mut buf).is_err() {
            throw_illegal_arg(&mut env, "unable to read double[]");
            return 0;
        }
        let d = base.merge_unsorted(buf);
        if want_f32 {
            quantize_digest_to_f32(&d)
        } else {
            d
        }
    } else if is_float_array {
        let values: JFloatArray = JFloatArray::from(values);
        let n = env.get_array_length(&values).unwrap_or(0) as usize;
        let mut tmp = vec![0.0f32; n];
        if env.get_float_array_region(&values, 0, &mut tmp).is_err() {
            throw_illegal_arg(&mut env, "unable to read float[]");
            return 0;
        }
        let buf: Vec<f64> = tmp.into_iter().map(|v| v as f64).collect();
        let d = base.merge_unsorted(buf);
        // If caller handed us f32 data, f32_mode true/false still controls centroid quantization.
        if want_f32 {
            quantize_digest_to_f32(&d)
        } else {
            d
        }
    } else {
        throw_illegal_arg(&mut env, "values must be double[] or float[]");
        return 0;
    };

    Box::into_raw(Box::new(digest)) as jlong
}

/* Free */

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_free(
    _env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let _ = unsafe { Box::from_raw(handle as *mut TDigest) };
}

/* Queries: cdf / quantile / median */

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_cdf(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
) -> jdoubleArray {
    let Some(d) = from_handle(&mut env, handle) else {
        return std::ptr::null_mut();
    };
    let n = env.get_array_length(&values).unwrap_or(0) as usize;
    let mut buf = vec![0.0f64; n];
    if env.get_double_array_region(&values, 0, &mut buf).is_err() {
        throw_illegal_arg(&mut env, "unable to read values array");
        return std::ptr::null_mut();
    }
    let out = d.cdf(&buf);
    let arr = match env.new_double_array(out.len() as i32) {
        Ok(a) => a,
        Err(_) => return std::ptr::null_mut(),
    };
    let _ = env.set_double_array_region(&arr, 0, &out);
    arr.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_quantile(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    q: jdouble,
) -> jdouble {
    let Some(d) = from_handle(&mut env, handle) else {
        return f64::NAN;
    };
    if !(0.0..=1.0).contains(&q) {
        throw_illegal_arg(&mut env, "q must be in [0, 1]");
        return f64::NAN;
    }
    d.quantile(q)
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_median(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jdouble {
    let Some(d) = from_handle(&mut env, handle) else {
        return f64::NAN;
    };
    d.median()
}
