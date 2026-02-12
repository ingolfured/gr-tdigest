// src/tdigest/wire.rs
//
// Canonical TDigest binary wire codec ("TDIG" format).
//
// Layout (little-endian):
//
//   header (56 bytes):
//     0..4   : magic = b"TDIG"
//     4      : version
//     5      : scale_code   (u8)
//     6      : policy_code  (u8)
//     7      : pin_per_side (u8)
//     8..16  : max_size      (u64)
//    16..24  : total_weight
//    24..32  : min           (f64)
//    32..40  : max           (f64)
//    40..48  : centroid_count (u64)
//    48..56  : data_sum      (f64)
//
//   centroids (payload):
//     - let N = centroid_count
//     - version 1:
//         - F32 wire: each centroid = mean(f32) + weight(u64)
//         - F64 wire: each centroid = mean(f64) + weight(u64)
//     - version 2:
//         - F32 wire: each centroid = mean(f32) + weight(f64) + kind(u8)
//         - F64 wire: each centroid = mean(f64) + weight(f64) + kind(u8)
//       else:
//           error
//
// Decode always returns TDigest<f64> (safe superset).

use std::fmt;

use ordered_float::FloatCore;

use crate::tdigest::centroids::Centroid;
use crate::tdigest::precision::{FloatLike, Precision};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::{DigestStats, TDigest};

const MAGIC: &[u8; 4] = b"TDIG";
const VERSION_V1: u8 = 1;
const VERSION_V2: u8 = 2;
const ENCODE_VERSION: u8 = VERSION_V2;
const HEADER_LEN: usize = 56;

#[derive(Debug)]
pub enum WireDecodedDigest {
    F32(TDigest<f32>),
    F64(TDigest<f64>),
}

#[derive(Debug)]
pub enum WireError {
    InvalidMagic,
    UnsupportedVersion(u8),
    InvalidScale(u8),
    InvalidPolicy(u8),
    InvalidHeader(&'static str),
    InvalidPayload(&'static str),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use WireError::*;
        match self {
            InvalidMagic => write!(f, "invalid TDIG magic header"),
            UnsupportedVersion(v) => write!(f, "unsupported TDIG version: {v}"),
            InvalidScale(c) => write!(f, "invalid TDIG scale code: {c}"),
            InvalidPolicy(c) => write!(f, "invalid TDIG policy code: {c}"),
            InvalidHeader(msg) => write!(f, "invalid TDIG header: {msg}"),
            InvalidPayload(msg) => write!(f, "invalid TDIG payload: {msg}"),
        }
    }
}

impl std::error::Error for WireError {}

pub type WireResult<T> = Result<T, WireError>;

#[derive(Debug, Clone, Copy)]
enum WireFloatWidth {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WirePrecision {
    F32,
    F64,
}

pub fn wire_precision(bytes: &[u8]) -> WireResult<WirePrecision> {
    if bytes.len() < HEADER_LEN {
        return Err(WireError::InvalidHeader("buffer too small"));
    }
    if &bytes[0..4] != MAGIC {
        return Err(WireError::InvalidMagic);
    }
    let version = bytes[4];
    if version != VERSION_V1 && version != VERSION_V2 {
        return Err(WireError::UnsupportedVersion(version));
    }

    // Layout reminder:
    // 0..4   : magic
    // 4      : version
    // 5      : scale_code
    // 6      : policy_code
    // 7      : pin_per_side
    // 8..16  : max_size
    // 16..24 : total_weight
    // 24..32 : min (f64)
    // 32..40 : max (f64)
    // 40..48 : centroid_count (u64)
    // 48..56 : data_sum (f64)
    //
    // So we can read centroid_count directly from offset 40.
    let mut offset = 40;
    let centroid_count_u64 = read_u64(bytes, &mut offset)?; // reuses your existing helper
    let centroid_count = centroid_count_u64 as usize;

    let payload_len = bytes.len().saturating_sub(HEADER_LEN);

    // Empty digest → no payload; by convention treat as F64 (same as decode path).
    if centroid_count == 0 {
        return Ok(WirePrecision::F64);
    }

    // Same logic as `decode_digest` to decide width from payload size.
    let (f32_stride, f64_stride) = if version == VERSION_V1 {
        (4 + 8, 8 + 8)
    } else {
        (4 + 8 + 1, 8 + 8 + 1)
    };
    let f32_len = centroid_count
        .checked_mul(f32_stride)
        .ok_or(WireError::InvalidPayload(
            "overflow computing f32 payload size",
        ))?;
    let f64_len = centroid_count
        .checked_mul(f64_stride)
        .ok_or(WireError::InvalidPayload(
            "overflow computing f64 payload size",
        ))?;

    if payload_len == f32_len {
        Ok(WirePrecision::F32)
    } else if payload_len == f64_len {
        Ok(WirePrecision::F64)
    } else {
        Err(WireError::InvalidPayload(
            "payload length does not match f32 or f64 layout",
        ))
    }
}

/* ============================
 * Small helpers
 * ============================ */

#[inline]
fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

#[inline]
fn read_u64(bytes: &[u8], offset: &mut usize) -> WireResult<u64> {
    if *offset + 8 > bytes.len() {
        return Err(WireError::InvalidHeader("truncated u64"));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    Ok(u64::from_le_bytes(arr))
}

#[inline]
fn read_f64(bytes: &[u8], offset: &mut usize) -> WireResult<f64> {
    if *offset + 8 > bytes.len() {
        return Err(WireError::InvalidHeader("truncated f64"));
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    Ok(f64::from_le_bytes(arr))
}

#[inline]
fn read_u8(bytes: &[u8], offset: &mut usize) -> WireResult<u8> {
    if *offset + 1 > bytes.len() {
        return Err(WireError::InvalidPayload("truncated u8"));
    }
    let v = bytes[*offset];
    *offset += 1;
    Ok(v)
}

fn scale_to_code(s: ScaleFamily) -> u8 {
    match s {
        ScaleFamily::Quad => 0,
        ScaleFamily::K1 => 1,
        ScaleFamily::K2 => 2,
        ScaleFamily::K3 => 3,
    }
}

fn code_to_scale(c: u8) -> WireResult<ScaleFamily> {
    match c {
        0 => Ok(ScaleFamily::Quad),
        1 => Ok(ScaleFamily::K1),
        2 => Ok(ScaleFamily::K2),
        3 => Ok(ScaleFamily::K3),
        _ => Err(WireError::InvalidScale(c)),
    }
}

fn policy_to_code(p: &SingletonPolicy) -> (u8, u8) {
    match *p {
        SingletonPolicy::Off => (0, 0),
        SingletonPolicy::Use => (1, 0),
        SingletonPolicy::UseWithProtectedEdges(k) => {
            let k_u8 = if k > 255 { 255 } else { k as u8 };
            (2, k_u8)
        }
    }
}

fn code_to_policy(code: u8, pin_per_side: u8) -> WireResult<SingletonPolicy> {
    match code {
        0 => Ok(SingletonPolicy::Off),
        1 => Ok(SingletonPolicy::Use),
        2 => {
            if pin_per_side == 0 {
                return Err(WireError::InvalidHeader(
                    "edges policy requires pin_per_side >= 1",
                ));
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(
                pin_per_side as usize,
            ))
        }
        _ => Err(WireError::InvalidPolicy(code)),
    }
}

#[inline]
fn kind_to_code<F: FloatLike + FloatCore>(c: &Centroid<F>) -> u8 {
    if c.is_atomic() {
        0
    } else {
        1
    }
}

#[inline]
fn code_is_atomic(code: u8) -> WireResult<bool> {
    match code {
        0 => Ok(true),
        1 => Ok(false),
        _ => Err(WireError::InvalidPayload("invalid centroid kind code")),
    }
}

/* ============================
 * Encode
 * ============================ */

pub fn encode_digest<F>(td: &TDigest<F>) -> Vec<u8>
where
    F: FloatLike + FloatCore,
{
    let n = td.centroids().len() as u64;
    let prec = Precision::of_type::<F>();
    let width = match prec {
        Precision::F32 => WireFloatWidth::F32,
        _ => WireFloatWidth::F64,
    };
    let version = ENCODE_VERSION;

    let per_centroid_len: usize = match width {
        WireFloatWidth::F32 => {
            if version == VERSION_V1 {
                4 + 8 // mean(f32) + weight(u64)
            } else {
                4 + 8 + 1 // mean(f32) + weight(f64) + kind(u8)
            }
        }
        WireFloatWidth::F64 => {
            if version == VERSION_V1 {
                8 + 8 // mean(f64) + weight(u64)
            } else {
                8 + 8 + 1 // mean(f64) + weight(f64) + kind(u8)
            }
        }
    };

    let mut buf = Vec::with_capacity(HEADER_LEN + per_centroid_len * (n as usize));

    // magic + version
    buf.extend_from_slice(MAGIC);
    buf.push(version);

    // scale / policy / pin_per_side
    let scale_code = scale_to_code(td.scale());
    let (policy_code, pin_per_side) = policy_to_code(&td.singleton_policy());
    buf.push(scale_code);
    buf.push(policy_code);
    buf.push(pin_per_side);

    // max_size
    write_u64(&mut buf, td.max_size() as u64);

    let cents = td.centroids();
    if version == VERSION_V1 {
        // Backward-compatible integerized weight header.
        let mut total_weight_u64: u64 = 0;
        for c in cents {
            let w = c.weight_f64();
            debug_assert!(
                w >= 0.0,
                "centroid weight must be non-negative for wire encoding"
            );
            let w_rounded = w.round();
            debug_assert!(
                (w - w_rounded).abs() <= 1e-6,
                "centroid weight must be close to integer for wire encoding"
            );
            let w_u64 = if w_rounded <= 0.0 {
                0
            } else if w_rounded > u64::MAX as f64 {
                u64::MAX
            } else {
                w_rounded as u64
            };
            total_weight_u64 = total_weight_u64.saturating_add(w_u64);
        }
        write_u64(&mut buf, total_weight_u64);
    } else {
        // Version 2 stores exact f64 total_weight.
        write_f64(&mut buf, td.count());
    }

    // min / max / centroid_count / data_sum
    write_f64(&mut buf, td.min());
    write_f64(&mut buf, td.max());
    write_u64(&mut buf, n);
    write_f64(&mut buf, td.sum());

    debug_assert_eq!(buf.len(), HEADER_LEN);

    // payload: centroids
    for c in cents {
        let mean_f64 = c.mean_f64();
        let w = c.weight_f64();

        match width {
            WireFloatWidth::F32 => {
                write_f32(&mut buf, mean_f64 as f32);
                if version == VERSION_V1 {
                    let w_rounded = w.round();
                    let w_u64 = if w_rounded <= 0.0 {
                        0
                    } else if w_rounded > u64::MAX as f64 {
                        u64::MAX
                    } else {
                        w_rounded as u64
                    };
                    write_u64(&mut buf, w_u64);
                } else {
                    write_f64(&mut buf, w);
                    write_u8(&mut buf, kind_to_code(c));
                }
            }
            WireFloatWidth::F64 => {
                write_f64(&mut buf, mean_f64);
                if version == VERSION_V1 {
                    let w_rounded = w.round();
                    let w_u64 = if w_rounded <= 0.0 {
                        0
                    } else if w_rounded > u64::MAX as f64 {
                        u64::MAX
                    } else {
                        w_rounded as u64
                    };
                    write_u64(&mut buf, w_u64);
                } else {
                    write_f64(&mut buf, w);
                    write_u8(&mut buf, kind_to_code(c));
                }
            }
        }
    }

    buf
}

/* ============================
 * Decode
 * ============================ */
pub fn decode_digest(bytes: &[u8]) -> WireResult<WireDecodedDigest> {
    if bytes.len() < HEADER_LEN {
        return Err(WireError::InvalidHeader("buffer too small"));
    }

    // magic
    if &bytes[0..4] != MAGIC {
        return Err(WireError::InvalidMagic);
    }

    let version = bytes[4];
    if version != VERSION_V1 && version != VERSION_V2 {
        return Err(WireError::UnsupportedVersion(version));
    }

    let scale_code = bytes[5];
    let policy_code = bytes[6];
    let pin_per_side = bytes[7];

    let mut offset = 8;

    let max_size_u64 = read_u64(bytes, &mut offset)?;
    let total_weight = if version == VERSION_V1 {
        read_u64(bytes, &mut offset)? as f64
    } else {
        let tw = read_f64(bytes, &mut offset)?;
        if !tw.is_finite() || tw < 0.0 {
            return Err(WireError::InvalidHeader("invalid total_weight"));
        }
        tw
    };
    let min = read_f64(bytes, &mut offset)?;
    let max = read_f64(bytes, &mut offset)?;
    let centroid_count_u64 = read_u64(bytes, &mut offset)?;
    let data_sum = read_f64(bytes, &mut offset)?;

    if offset != HEADER_LEN {
        return Err(WireError::InvalidHeader("header length mismatch"));
    }

    let max_size = max_size_u64 as usize;
    let centroid_count = centroid_count_u64 as usize;

    let scale = code_to_scale(scale_code)?;
    let policy = code_to_policy(policy_code, pin_per_side)?;

    let payload_len = bytes.len().saturating_sub(HEADER_LEN);
    let width = if centroid_count == 0 {
        // No centroids → width irrelevant, treat as f64 for simplicity
        WireFloatWidth::F64
    } else {
        let (f32_stride, f64_stride) = if version == VERSION_V1 {
            (4 + 8, 8 + 8)
        } else {
            (4 + 8 + 1, 8 + 8 + 1)
        };
        let f32_len = centroid_count
            .checked_mul(f32_stride)
            .ok_or(WireError::InvalidPayload(
                "overflow computing f32 payload size",
            ))?;
        let f64_len = centroid_count
            .checked_mul(f64_stride)
            .ok_or(WireError::InvalidPayload(
                "overflow computing f64 payload size",
            ))?;

        if payload_len == f32_len {
            WireFloatWidth::F32
        } else if payload_len == f64_len {
            WireFloatWidth::F64
        } else {
            return Err(WireError::InvalidPayload(
                "payload length does not match f32 or f64 layout",
            ));
        }
    };

    let stats = DigestStats {
        data_sum,
        total_weight,
        data_min: min,
        data_max: max,
    };

    match width {
        WireFloatWidth::F32 => {
            let mut cents: Vec<Centroid<f32>> = Vec::with_capacity(centroid_count);
            let mut p = HEADER_LEN;

            for _ in 0..centroid_count {
                if p + 4 > bytes.len() {
                    return Err(WireError::InvalidPayload("truncated f32 mean"));
                }
                let mut arr_m = [0u8; 4];
                arr_m.copy_from_slice(&bytes[p..p + 4]);
                p += 4;
                let mean_f32 = f32::from_le_bytes(arr_m);
                let mean_f64 = mean_f32 as f64;

                let c = if version == VERSION_V1 {
                    if p + 8 > bytes.len() {
                        return Err(WireError::InvalidPayload("truncated weight u64"));
                    }
                    let mut arr_w = [0u8; 8];
                    arr_w.copy_from_slice(&bytes[p..p + 8]);
                    p += 8;
                    let w_u64 = u64::from_le_bytes(arr_w);
                    let w_f64 = w_u64 as f64;
                    if w_u64 == 1 {
                        Centroid::<f32>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f32>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    if !w_f64.is_finite() || w_f64 <= 0.0 {
                        return Err(WireError::InvalidPayload("invalid centroid weight"));
                    }
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if (w_f64 - 1.0).abs() <= f64::EPSILON {
                            Centroid::<f32>::new_atomic_unit_f64(mean_f64)
                        } else {
                            Centroid::<f32>::new_atomic_f64(mean_f64, w_f64)
                        }
                    } else {
                        Centroid::<f32>::new_mixed_f64(mean_f64, w_f64)
                    }
                };
                cents.push(c);
            }

            let td = TDigest::<f32>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .with_centroids_and_stats(cents, stats)
                .build();

            Ok(WireDecodedDigest::F32(td))
        }
        WireFloatWidth::F64 => {
            let mut cents: Vec<Centroid<f64>> = Vec::with_capacity(centroid_count);
            let mut p = HEADER_LEN;

            for _ in 0..centroid_count {
                if p + 8 > bytes.len() {
                    return Err(WireError::InvalidPayload("truncated f64 mean"));
                }
                let mut arr_m = [0u8; 8];
                arr_m.copy_from_slice(&bytes[p..p + 8]);
                p += 8;
                let mean_f64 = f64::from_le_bytes(arr_m);

                let c = if version == VERSION_V1 {
                    if p + 8 > bytes.len() {
                        return Err(WireError::InvalidPayload("truncated weight u64"));
                    }
                    let mut arr_w = [0u8; 8];
                    arr_w.copy_from_slice(&bytes[p..p + 8]);
                    p += 8;
                    let w_u64 = u64::from_le_bytes(arr_w);
                    let w_f64 = w_u64 as f64;
                    if w_u64 == 1 {
                        Centroid::<f64>::new_atomic_unit_f64(mean_f64)
                    } else {
                        Centroid::<f64>::new_mixed_f64(mean_f64, w_f64)
                    }
                } else {
                    let w_f64 = read_f64(bytes, &mut p)?;
                    if !w_f64.is_finite() || w_f64 <= 0.0 {
                        return Err(WireError::InvalidPayload("invalid centroid weight"));
                    }
                    let kind = read_u8(bytes, &mut p)?;
                    if code_is_atomic(kind)? {
                        if (w_f64 - 1.0).abs() <= f64::EPSILON {
                            Centroid::<f64>::new_atomic_unit_f64(mean_f64)
                        } else {
                            Centroid::<f64>::new_atomic_f64(mean_f64, w_f64)
                        }
                    } else {
                        Centroid::<f64>::new_mixed_f64(mean_f64, w_f64)
                    }
                };
                cents.push(c);
            }

            let td = TDigest::<f64>::builder()
                .max_size(max_size)
                .scale(scale)
                .singleton_policy(policy)
                .with_centroids_and_stats(cents, stats)
                .build();

            Ok(WireDecodedDigest::F64(td))
        }
    }
}
