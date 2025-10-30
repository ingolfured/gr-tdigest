use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "f32" | "f64" when serialized
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    #[inline]
    pub fn is_f32(self) -> bool {
        matches!(self, Precision::F32)
    }
}
