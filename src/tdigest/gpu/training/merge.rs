use super::*;

unsafe extern "C" {
    fn gr_gpu_merge(
        left: *mut c_void,
        a: Stats,
        right: *mut c_void,
        b: Stats,
        cfg: Config,
        output: *mut Cent,
        limit: usize,
        result: *mut Stats,
    ) -> i32;
}
impl GpuDigest {
    pub fn from_bytes(bytes: &[u8], device: i32) -> Result<Self, TrainingError> {
        let digest = FrontendDigest::from_bytes(bytes).map_err(|e| Input(e.to_string()))?;
        match digest {
            FrontendDigest::F32(d) => Self::upload(&d, device),
            FrontendDigest::F64(d) => Self::upload(&d, device),
        }
        .map_err(Device)
    }
    /// Merge two independent snapshots without changing either input.
    pub fn merged(&self, other: &Self) -> Result<Self, TrainingError> {
        if self.1 != other.1 {
            return Err(Input("digests must be on the same device".into()));
        }
        let decode =
            |s: &Self| FrontendDigest::from_bytes(s.to_bytes()).map_err(|e| Input(e.to_string()));
        let (a, b) = (decode(self)?, decode(other)?);
        if a.config() != b.config() || a.precision() != b.precision() {
            return Err(Input(
                "incompatible merge configuration or precision".into(),
            ));
        }
        match (a, b) {
            (FrontendDigest::F32(a), FrontendDigest::F32(b)) => self.merge_rows(&a, other, &b),
            (FrontendDigest::F64(a), FrontendDigest::F64(b)) => self.merge_rows(&a, other, &b),
            _ => unreachable!("precision checked"),
        }
    }
    fn merge_rows<F: FloatLike + FloatCore>(
        &self,
        a: &TDigest<F>,
        other: &Self,
        b: &TDigest<F>,
    ) -> Result<Self, TrainingError> {
        if a.is_effectively_empty() {
            return Self::upload(b, self.1).map_err(Device);
        }
        if b.is_effectively_empty() {
            return Self::upload(a, self.1).map_err(Device);
        }
        let (cfg, limit) = buffer_size(a, b.centroids().len())?;
        let mut rows = vec![Cent::default(); limit];
        let mut result = Stats::default();
        check(unsafe {
            gr_gpu_merge(
                self.0.as_ptr(),
                Stats::of(a),
                other.0.as_ptr(),
                Stats::of(b),
                cfg,
                rows.as_mut_ptr(),
                limit,
                &mut result,
            )
        })
        .map_err(Device)?;
        restore(a, &rows[..result.rows], result, self.1)
    }
    /// Match the frontend's left fold, including a canonical empty when no inputs are supplied.
    pub fn merge_all(inputs: &[&Self], device: Option<i32>) -> Result<Self, TrainingError> {
        let Some((first, tail)) = inputs.split_first() else {
            return Self::from_bytes(
                &FrontendDigest::canonical_empty().to_bytes(),
                device.unwrap_or(0),
            );
        };
        if device.is_some_and(|id| id != first.1) {
            return Err(Input("digests must be on the selected device".into()));
        }
        let mut acc = Self::from_bytes(first.to_bytes(), first.1)?;
        for next in tail {
            acc = acc.merged(next)?;
        }
        Ok(acc)
    }
}
