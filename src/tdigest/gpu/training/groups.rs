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
    fn gr_gpu_group_limits(device: i32, resident: *mut usize, free_bytes: *mut usize) -> i32;
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
        allocation_oom: *mut bool,
    ) -> i32;
}
enum Chunk {
    Merged(Vec<GpuDigest>),
    RetryWithBytes(usize),
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
        let mut retry_budget = usize::MAX;
        let mut resident = 0;
        while outputs.len() < groups.len() {
            let start = outputs.len();
            let mut free_bytes = 0;
            check(unsafe { gr_gpu_group_limits(device, &mut resident, &mut free_bytes) })
                .map_err(Device)?;
            let end = groups.len().min(start.saturating_add(resident));
            // Leave headroom for returned snapshots and other GPU users.
            let budget = retry_budget.min(free_bytes / 4);
            match Self::merge_chunk(&groups[start..end], device, budget)? {
                Chunk::Merged(chunk) => outputs.extend(chunk),
                Chunk::RetryWithBytes(bytes) => retry_budget = bytes,
            }
        }
        Ok(outputs)
    }
    fn merge_chunk(
        groups: &[Vec<&Self>],
        device: i32,
        workspace_bytes: usize,
    ) -> Result<Chunk, TrainingError> {
        let mut plans = Vec::with_capacity(groups.len());
        let mut children = Vec::new();
        let (mut scratch, mut slots, mut rounds) = (0usize, 0usize, 0usize);
        let mut planned_bytes = 0;
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
            let next_slots = slots.checked_add(limit).ok_or_else(overflow)?;
            let bytes = next_scratch
                .saturating_add(next_slots)
                .saturating_mul(size_of::<Cent>())
                .saturating_add(children.len().saturating_mul(size_of::<Child>()))
                .saturating_add(
                    (plans.len() + 1).saturating_mul(size_of::<Group>() + size_of::<Stats>()),
                );
            // An indivisible group may use the headroom if it fits by itself.
            if !plans.is_empty() && bytes > workspace_bytes {
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
            slots = next_slots;
            planned_bytes = bytes;
            rounds = rounds.max(group.len());
        }
        if scratch > isize::MAX as usize / std::mem::size_of::<Cent>() {
            return Err(overflow());
        }
        let mut rows = vec![Cent::default(); slots];
        let mut stats = vec![Stats::default(); plans.len()];
        if slots > 0 {
            let mut allocation_oom = false;
            let status = unsafe {
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
                    &mut allocation_oom,
                )
            };
            // A free-memory reading can race another allocation. Only retry
            // failures from scratch allocation, before the first kernel launch.
            if allocation_oom && plans.len() > 1 {
                return Ok(Chunk::RetryWithBytes(planned_bytes / 2));
            }
            check(status).map_err(Device)?;
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
            .collect::<Result<_, _>>()
            .map(Chunk::Merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_workspace_budget_counts_every_device_buffer() {
        let device = std::env::var("GR_TDIGEST_TEST_DEVICE").map_or(0, |s| s.parse().unwrap());
        let input = TDigest::<f64>::from_unsorted(&[1.], 100).unwrap();
        let d = GpuDigest::upload(&input, device).unwrap();
        let groups = vec![vec![&d, &d]; 4];
        let reference = GpuDigest::merge_all(&groups[0], Some(device)).unwrap();
        // Each group reserves nine scratch centroids, two output slots,
        // two child descriptors, one group descriptor and one result header.
        let bytes = 11 * size_of::<Cent>()
            + 2 * size_of::<Child>()
            + size_of::<Group>()
            + size_of::<Stats>();
        for (budget, count) in [(0, 1), (2 * bytes - 1, 1), (2 * bytes, 2), (4 * bytes, 4)] {
            let Chunk::Merged(out) = GpuDigest::merge_chunk(&groups, device, budget).unwrap()
            else {
                panic!("unexpected allocation failure")
            };
            assert_eq!(out.len(), count);
            assert!(out.iter().all(|d| d.to_bytes() == reference.to_bytes()));
        }
    }
}
