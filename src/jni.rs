use std::ptr::NonNull;

use jni::objects::{JClass, JDoubleArray, JString};
use jni::sys::{jdouble, jdoubleArray, jint, jlong};
use jni::JNIEnv;

use crate::tdigest::{ScaleFamily, TDigest};

fn throw_illegal_arg(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", msg);
}

fn throw_illegal_state(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/IllegalStateException", msg);
}

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

fn parse_scale(env: &mut JNIEnv, jscale: JString) -> Option<ScaleFamily> {
    let s: String = match env.get_string(&jscale) {
        Ok(js) => js.into(),
        Err(_) => {
            throw_illegal_arg(env, "scale must be a valid UTF-8 string");
            return None;
        }
    };
    let s = s.trim().to_ascii_lowercase();
    let scale = match s.as_str() {
        "quad" => ScaleFamily::Quad,
        "k1" => ScaleFamily::K1,
        "k2" => ScaleFamily::K2,
        "k3" => ScaleFamily::K3,
        _ => {
            throw_illegal_arg(env, "scale must be one of: quad, k1, k2, k3");
            return None;
        }
    };
    Some(scale)
}
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_create(
    mut env: JNIEnv,
    _cls: JClass,
    max_size: jint,
    scale: JString,
) -> jlong {
    let Some(scale) = parse_scale(&mut env, scale) else {
        return 0;
    };
    let ms = if max_size > 0 { max_size as usize } else { 0 };
    if ms == 0 {
        throw_illegal_arg(&mut env, "maxSize must be > 0");
        return 0;
    }
    let digest = TDigest::builder().max_size(ms).scale(scale).build();
    let boxed = Box::new(digest);
    Box::into_raw(boxed) as jlong
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_createFromValues(
    mut env: JNIEnv,
    _cls: JClass,
    values: JDoubleArray,
    max_size: jint,
    scale: JString,
) -> jlong {
    let Some(scale) = parse_scale(&mut env, scale) else {
        return 0;
    };
    let ms = if max_size > 0 { max_size as usize } else { 0 };
    if ms == 0 {
        throw_illegal_arg(&mut env, "maxSize must be > 0");
        return 0;
    }

    let n = match env.get_array_length(&values) {
        Ok(n) => n as usize,
        Err(_) => {
            throw_illegal_arg(&mut env, "values array length not accessible");
            return 0;
        }
    };

    let mut buf = vec![0.0f64; n];
    if env.get_double_array_region(&values, 0, &mut buf).is_err() {
        throw_illegal_arg(&mut env, "unable to read values array");
        return 0;
    }

    // If you have an unsorted merge, replace sort+merge_sorted with that.
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let digest = TDigest::builder()
        .max_size(ms)
        .scale(scale)
        .build()
        .merge_sorted(buf);
    let boxed = Box::new(digest);
    Box::into_raw(boxed) as jlong
}

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

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_estimateCdf(
    mut env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    xs: JDoubleArray,
) -> jdoubleArray {
    let Some(d) = from_handle(&mut env, handle) else {
        return std::ptr::null_mut();
    };

    let n = env.get_array_length(&xs).unwrap_or(0) as usize;
    let mut buf = vec![0.0f64; n];
    if env.get_double_array_region(&xs, 0, &mut buf).is_err() {
        throw_illegal_arg(&mut env, "unable to read xs array");
        return std::ptr::null_mut();
    }

    let out = d.estimate_cdf(&buf);
    let arr = env.new_double_array(out.len() as i32).unwrap();
    let _ = env.set_double_array_region(&arr, 0, &out);
    arr.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_gr_tdigest_1rs_TDigestNative_estimateQuantile(
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
    d.estimate_quantile(q)
}
