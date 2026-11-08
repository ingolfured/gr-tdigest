use super::{catch_native, read_jdouble_array};
use crate::tdigest::gpu::{DeviceArray, GpuDigest, TrainingError};
use jni::{
    objects::{JByteArray, JClass, JDoubleArray, JObjectArray},
    sys::{jboolean, jbyteArray, jdoubleArray, jint},
    JNIEnv,
};
use TrainingError::{Device, Input};

fn run<T: Default>(mut env: JNIEnv, op: impl FnOnce(&mut JNIEnv) -> Result<T, TrainingError>) -> T {
    match catch_native(|| op(&mut env))
        .map_err(Device)
        .and_then(|v| v)
    {
        Ok(v) => v,
        Err(error) => {
            let (kind, msg) = match error {
                Input(msg) => ("java/lang/IllegalArgumentException", msg),
                Device(msg) => ("java/lang/IllegalStateException", msg),
            };
            if !env.exception_check().unwrap_or(true) {
                let _ = env.throw_new(kind, msg);
            }
            T::default()
        }
    }
}
fn vm(error: jni::errors::Error) -> TrainingError {
    Device(error.to_string())
}
fn upload(env: &JNIEnv, bytes: &JByteArray, device: i32) -> Result<GpuDigest, TrainingError> {
    GpuDigest::from_bytes(&env.convert_byte_array(bytes).map_err(vm)?, device)
}
fn download(env: &JNIEnv, d: &GpuDigest) -> Result<jbyteArray, TrainingError> {
    Ok(env
        .byte_array_from_slice(d.to_bytes())
        .map_err(vm)?
        .into_raw())
}
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_Gpu_query(
    env: JNIEnv,
    _: JClass,
    device: jint,
    bytes: JByteArray,
    xs: JDoubleArray,
    q: jboolean,
) -> jdoubleArray {
    run(env, |env| {
        let mut digest = upload(env, &bytes, device)?;
        let xs = read_jdouble_array(env, &xs).map_err(Device)?;
        if q != 0 && xs.iter().any(|x| !(0.0..=1.0).contains(x)) {
            return Err(Input(crate::tdigest::gpu::BAD_Q.into()));
        }
        let values = if q != 0 {
            digest.quantile(&xs)
        } else {
            digest.cdf(&xs)
        }
        .map_err(Device)?;
        let out = env.new_double_array(values.len() as i32).map_err(vm)?;
        env.set_double_array_region(&out, 0, values).map_err(vm)?;
        Ok(out.into_raw())
    })
}
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_Gpu_train(
    env: JNIEnv,
    _: JClass,
    device: jint,
    seed: JByteArray,
    xs: JDoubleArray,
    ws: JDoubleArray,
) -> jbyteArray {
    run(env, |env| {
        let mut digest = upload(env, &seed, device)?;
        let buffer = |array: &JDoubleArray| {
            DeviceArray::upload(&read_jdouble_array(env, array).map_err(Device)?, device)
                .map_err(Device)
        };
        let values = buffer(&xs)?;
        let weights = if ws.is_null() {
            None
        } else {
            Some(buffer(&ws)?)
        };
        digest.add(&values, weights.as_ref())?;
        download(env, &digest)
    })
}
#[no_mangle]
pub extern "system" fn Java_gr_tdigest_Gpu_merge(
    env: JNIEnv,
    _: JClass,
    device: jint,
    children: JObjectArray,
) -> jbyteArray {
    run(env, |env| {
        let mut digests = Vec::new();
        for i in 0..env.get_array_length(&children).map_err(vm)? {
            let child = env.get_object_array_element(&children, i).map_err(vm)?;
            let bytes = JByteArray::from(child);
            let digest = upload(env, &bytes, device)?;
            env.delete_local_ref(bytes).map_err(vm)?;
            digests.push(digest);
        }
        let refs = digests.iter().collect::<Vec<_>>();
        download(env, &GpuDigest::merge_all(&refs, Some(device))?)
    })
}
