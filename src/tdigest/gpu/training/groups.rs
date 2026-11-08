use super::*;

#[repr(C)]
struct Child {
    rows: *const super::super::Row,
    stats: Stats,
}
#[repr(C)]
struct Group {
    first: usize,
    count: usize,
    scratch: usize,
    stride: usize,
    output: usize,
    limit: usize,
    cfg: Config,
}
unsafe extern "C" {
    fn gr_gpu_rows(s: *const c_void) -> *const super::super::Row;
    fn gr_gpu_groups(
        device: i32,
        groups: *const Group,
        n: usize,
        children: *const Child,
        count: usize,
        rounds: usize,
        scratch: usize,
        slots: usize,
        stats: *mut Stats,
        rows: *mut Cent,
    ) -> i32;
}
impl GpuDigest {
    /// Merge each group in input order. Independent groups share kernel launches.
    pub fn merge_groups(
        groups: &[Vec<&Self>],
        device: Option<i32>,
    ) -> Result<Vec<Self>, TrainingError> {
        let device = device
            .or_else(|| groups.iter().flatten().next().map(|d| d.device))
            .unwrap_or(0);
        let mut outputs = Vec::with_capacity(groups.len());
        while outputs.len() < groups.len() {
            let start = outputs.len();
            let end = groups.len().min(start + 4096);
            outputs.extend(Self::merge_chunk(&groups[start..end], device)?);
        }
        Ok(outputs)
    }
    fn merge_chunk(groups: &[Vec<&Self>], device: i32) -> Result<Vec<Self>, TrainingError> {
        let mut plans = Vec::with_capacity(groups.len());
        let mut children = Vec::new();
        let (mut scratch, mut slots, mut rounds) = (0usize, 0usize, 0usize);
        let overflow = || Input("batch exceeds supported workspace size".into());
        for group in groups {
            let cfg = group.first().map_or_else(Config::default, |d| d.config);
            let first = children.len();
            let mut largest = 0;
            let mut total = 0usize;
            for child in group {
                if child.device != device {
                    return Err(Input("digests must be on the selected device".into()));
                }
                if child.config != cfg {
                    return Err(Input(
                        "incompatible merge configuration or precision".into(),
                    ));
                }
                largest = largest.max(child.stats.rows);
                total = total.checked_add(child.stats.rows).ok_or_else(overflow)?;
                children.push(Child {
                    rows: unsafe { gr_gpu_rows(child.handle.as_ptr()) },
                    stats: child.stats,
                });
            }
            let limit = if cfg.delta > 0. {
                total
            } else {
                total.min(largest.max(cfg.cap.saturating_add(cfg.pins.saturating_mul(2))))
            };
            let stride = limit.checked_add(largest).ok_or_else(overflow)?;
            let next_scratch = scratch
                .checked_add(stride.checked_mul(3).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
            // Bound temporary device storage across groups; a single larger group
            // can still run when the device has room.
            if !plans.is_empty() && next_scratch > (128 << 20) / std::mem::size_of::<Cent>() {
                children.truncate(first);
                break;
            }
            plans.push(Group {
                first,
                count: group.len(),
                scratch,
                stride,
                output: slots,
                limit,
                cfg,
            });
            scratch = next_scratch;
            slots = slots.checked_add(limit).ok_or_else(overflow)?;
            rounds = rounds.max(group.len());
        }
        if scratch > isize::MAX as usize / std::mem::size_of::<Cent>() {
            return Err(overflow());
        }
        let mut rows = vec![Cent::default(); slots];
        let mut stats = vec![Stats::default(); plans.len()];
        if slots > 0 {
            check(unsafe {
                gr_gpu_groups(
                    device,
                    plans.as_ptr(),
                    plans.len(),
                    children.as_ptr(),
                    children.len(),
                    rounds,
                    scratch,
                    slots,
                    stats.as_mut_ptr(),
                    rows.as_mut_ptr(),
                )
            })
            .map_err(Device)?;
        }
        groups
            .iter()
            .zip(plans)
            .zip(stats)
            .map(|((group, g), s)| {
                if s.rows > g.limit {
                    return Err(Input("invalid intermediate merge state".into()));
                }
                let Some(first) = group.first() else {
                    return Self::merge_all(&[], Some(device));
                };
                if s.rows == 0 || group.len() == 1 {
                    return Self::from_bytes(first.to_bytes(), device);
                }
                let cs = &rows[g.output..g.output + s.rows];
                match FrontendDigest::from_bytes(first.to_bytes())
                    .map_err(|e| Input(e.to_string()))?
                {
                    FrontendDigest::F32(d) => restore(&d, cs, s, device),
                    FrontendDigest::F64(d) => restore(&d, cs, s, device),
                }
            })
            .collect()
    }
}
