use super::{check, DeviceArray, GpuDigest};
use crate::tdigest::{
    centroids::Centroid, frontends::FrontendDigest, precision::FloatLike,
    singleton_policy::SingletonPolicy, DigestStats, ScaleFamily, TDigest,
};
use ordered_float::FloatCore;
use std::{ffi::c_void, ptr};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Cent {
    mean: f64,
    weight: f64,
    atomic: u64,
}
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Default)]
pub(super) struct Config {
    cap: usize,
    pins: usize,
    family: i32,
    policy: i32,
    narrow: i32,
    delta: f64,
}
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(super) struct Stats {
    sum: f64,
    count: f64,
    min: f64,
    max: f64,
    rows: usize,
}
unsafe extern "C" {
    fn gr_gpu_train(
        s: *mut c_void,
        prior: Stats,
        cfg: Config,
        xs: *const f64,
        ws: *const f64,
        n: usize,
        out: *mut Cent,
        limit: usize,
        stats: *mut Stats,
    ) -> i32;
}

#[derive(Debug)]
pub enum TrainingError {
    Device(String),
    Input(String),
}
use TrainingError::{Device, Input};

impl Config {
    pub(super) fn of<F: FloatLike + FloatCore>(d: &TDigest<F>) -> Self {
        let (policy, pins) = match d.singleton_policy() {
            SingletonPolicy::Off => (0, 0),
            SingletonPolicy::Use => (1, 0),
            SingletonPolicy::UseWithProtectedEdges(k) => (2, k),
        };
        let family = match d.scale() {
            ScaleFamily::Quad => 0,
            ScaleFamily::K1 => 1,
            ScaleFamily::K2 => 2,
            ScaleFamily::K2Norm => 3,
            ScaleFamily::K3 => 4,
        };
        Self {
            cap: d.max_size(),
            pins,
            family,
            policy,
            narrow: d.precision().is_f32().into(),
            delta: d.delta().unwrap_or(0.),
        }
    }
}
impl Stats {
    pub(super) fn of<F: FloatLike + FloatCore>(d: &TDigest<F>) -> Self {
        Self {
            sum: d.sum(),
            count: d.count(),
            min: d.min(),
            max: d.max(),
            rows: d.centroids().len(),
        }
    }
}
fn buffer_size<F: FloatLike + FloatCore>(
    d: &TDigest<F>,
    extra: usize,
) -> Result<(Config, usize), TrainingError> {
    let cfg = Config::of(d);
    let total = d
        .centroids()
        .len()
        .checked_add(extra)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| Input("HIP operation exceeds the supported batch length".into()))?;
    let limit = if d.delta().is_some() {
        total
    } else {
        total.min(cfg.cap.saturating_add(cfg.pins.saturating_mul(2)))
    };
    Ok((cfg, limit))
}
fn restore<F: FloatLike + FloatCore>(
    d: &TDigest<F>,
    rows: &[Cent],
    stats: Stats,
    device: i32,
) -> Result<GpuDigest, TrainingError> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|c| !c.mean.is_finite() || !c.weight.is_finite() || c.weight <= 0.)
    {
        return Err(Input(
            "invalid training data or unrepresentable aggregate".into(),
        ));
    }
    let centroids = rows
        .iter()
        .map(|c| {
            if c.atomic != 0 {
                Centroid::new_atomic_f64(c.mean, c.weight)
            } else {
                Centroid::new_mixed_f64(c.mean, c.weight)
            }
        })
        .collect();
    let mut builder = TDigest::<F>::builder()
        .max_size(d.max_size())
        .scale(d.scale())
        .singleton_policy(d.singleton_policy());
    if let Some(delta) = d.delta() {
        builder = builder.delta(delta);
    }
    let stats = DigestStats {
        data_sum: stats.sum,
        total_weight: stats.count,
        data_min: stats.min,
        data_max: stats.max,
    };
    let result = builder
        .with_centroids_and_stats(centroids, stats)
        .build()
        .map_err(|e| Input(e.to_string()))?;
    GpuDigest::upload(&result, device).map_err(Device)
}
impl GpuDigest {
    /// Add an observation batch atomically. Only the compressed state crosses back to the host.
    pub fn add(&mut self, xs: &DeviceArray, ws: Option<&DeviceArray>) -> Result<(), TrainingError> {
        if xs.device != self.device
            || ws.is_some_and(|w| w.device != self.device || w.len != xs.len)
        {
            return Err(Input(
                "values and weights must share the digest device and batch length".into(),
            ));
        }
        if xs.len == 0 {
            return Ok(());
        }
        let original =
            FrontendDigest::from_bytes(self.to_bytes()).map_err(|e| Input(e.to_string()))?;
        *self = match original {
            FrontendDigest::F32(d) => self.trained(&d, xs, ws)?,
            FrontendDigest::F64(d) => self.trained(&d, xs, ws)?,
        };
        Ok(())
    }
    fn trained<F: FloatLike + FloatCore>(
        &self,
        d: &TDigest<F>,
        xs: &DeviceArray,
        ws: Option<&DeviceArray>,
    ) -> Result<Self, TrainingError> {
        let (cfg, limit) = buffer_size(d, xs.len)?;
        let mut rows = vec![Cent::default(); limit];
        let mut stats = Stats::default();
        check(unsafe {
            gr_gpu_train(
                self.handle.as_ptr(),
                Stats::of(d),
                cfg,
                xs.ptr,
                ws.map_or(ptr::null(), |w| w.ptr),
                xs.len,
                rows.as_mut_ptr(),
                limit,
                &mut stats,
            )
        })
        .map_err(Device)?;
        restore(d, &rows[..stats.rows], stats, self.device)
    }
}

mod merge;

mod groups;
