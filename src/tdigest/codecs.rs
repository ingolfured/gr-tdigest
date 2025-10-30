//! Unified Polars ↔ TDigest codecs.
//!
//! Two precision modes:
//! - **Canonical** – f64 centroids; f64 min/max/sum/count
//! - **Compact**   – f32 centroids; f32 min/max; f64 sum/count
//!
//! Front-end API:
//!   td.to_series(name, compact=false)          → Series
//!   td.to_series_with_default(name, f32_mode)  → Series
//!   TDigest::from_series(&series, compact)     → `Vec<TDigest>`
//!
//! Internal strict variants still available:
//!   try_to_series\[_compact], try_from_series\[_compact]

use std::fmt;

use polars::prelude::*;

use crate::tdigest::{centroids::Centroid, DigestStats, TDigest};

// --------------------- field name constants ----------------------------------

pub const F_CENTROIDS: &str = "centroids";
pub const F_MEAN: &str = "mean";
pub const F_WEIGHT: &str = "weight";
pub const F_SUM: &str = "sum";
pub const F_MIN: &str = "min";
pub const F_MAX: &str = "max";
pub const F_COUNT: &str = "count";
pub const F_MAX_SIZE: &str = "max_size";

// --------------------- precision mode ----------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum PrecisionMode {
    Canonical, // f64 centroids, f64 min/max
    Compact,   // f32 centroids, f32 min/max
}

// --------------------- public API on TDigest ---------------------------------

impl TDigest {
    /// Canonical encoding: f64 centroids and f64 min/max/sum/count.
    pub fn try_to_series(&self, name: &str) -> PolarsResult<Series> {
        tdigest_to_series_with(self, name, PrecisionMode::Canonical)
    }

    /// Compact encoding: f32 centroids and f32 min/max (sum/count stay f64).
    pub fn try_to_series_compact(&self, name: &str) -> PolarsResult<Series> {
        tdigest_to_series_with(self, name, PrecisionMode::Compact)
    }

    /// Strict parse of canonical schema.
    pub fn try_from_series(input: &Series) -> Result<Vec<TDigest>, CodecError> {
        parse_tdigests_strict(input, PrecisionMode::Canonical)
    }

    /// Strict parse of compact schema.
    pub fn try_from_series_compact(input: &Series) -> Result<Vec<TDigest>, CodecError> {
        parse_tdigests_strict(input, PrecisionMode::Compact)
    }

    /// Canonical Polars dtype for a TDigest row (struct of 6 fields).
    pub fn polars_dtype() -> DataType {
        canonical_dtype()
    }

    /// Compact Polars dtype for a TDigest row (struct of 6 fields).
    pub fn polars_dtype_compact() -> DataType {
        compact_dtype()
    }
}

// --------------------- unified front-end wrappers ---------------------------

impl TDigest {
    /// Unified Polars codec interface (used by all language fronts).
    ///
    /// `compact = false` → Canonical (f64 centroids)
    /// `compact = true`  → Compact (f32 centroids)
    pub fn to_series(&self, name: &str, compact: bool) -> PolarsResult<Series> {
        if compact {
            self.try_to_series_compact(name)
        } else {
            self.try_to_series(name)
        }
    }

    /// Convenience variant that follows the digest's in-memory precision flag.
    ///
    /// Call this from language bindings if you have an `f32_mode` boolean.
    pub fn to_series_with_default(&self, name: &str, f32_mode: bool) -> PolarsResult<Series> {
        if f32_mode {
            self.try_to_series_compact(name)
        } else {
            self.try_to_series(name)
        }
    }

    /// Unified strict decoder. `compact=false` → canonical; `true` → compact.
    pub fn from_series(input: &Series, compact: bool) -> Result<Vec<TDigest>, CodecError> {
        if compact {
            Self::try_from_series_compact(input)
        } else {
            Self::try_from_series(input)
        }
    }
    pub fn to_series_default(&self, name: &str) -> PolarsResult<Series> {
        self.to_series_with_default(name, self.is_f32_mode())
    }
}

// --------------------- error type (strict) -----------------------------------

#[non_exhaustive]
#[derive(Debug)]
pub enum CodecError {
    NotAStruct {
        series_name: String,
    },
    MissingField(&'static str),
    WrongDtype {
        field: &'static str,
        found: Box<DataType>,
        expected: Box<DataType>,
    },
    BadCentroidLayout {
        expected_mean: Box<DataType>,
        expected_weight: Box<DataType>,
        found: Box<DataType>,
    },
    NullField {
        field: &'static str,
        row: usize,
    },
    NullCentroid {
        row: usize,
        index: usize,
    },
    Polars(PolarsError),
}

impl From<PolarsError> for CodecError {
    fn from(e: PolarsError) -> Self {
        CodecError::Polars(e)
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CodecError::*;
        match self {
            NotAStruct { series_name } =>
                write!(f, "series '{series_name}' is not a struct column"),
            MissingField(field) =>
                write!(f, "missing field '{field}'"),
            WrongDtype { field, found, expected } =>
                write!(f, "field '{field}' has dtype {found:?}, expected {expected:?}"),
            BadCentroidLayout { expected_mean, expected_weight, found } =>
                write!(f, "centroids must be List<Struct{{mean: {expected_mean:?}, weight: {expected_weight:?}}}>, found {found:?}"),
            NullField { field, row } =>
                write!(f, "null in scalar field '{field}' at row {row}"),
            NullCentroid { row, index } =>
                write!(f, "null in centroids at row {row}, index {index}"),
            Polars(e) => write!(f, "polars error: {e}"),
        }
    }
}

// --------------------- default codec selector --------------------------------

#[derive(Clone, Copy, Debug)]
pub enum DefaultCodec {
    Canonical,
    Compact,
}

impl DefaultCodec {
    #[inline]
    pub fn for_f32_mode(f32_mode: bool) -> Self {
        if f32_mode {
            Self::Compact
        } else {
            Self::Canonical
        }
    }
}

// --------------------- schema builders & validators --------------------------

fn canonical_dtype() -> DataType {
    DataType::Struct(vec![
        Field::new(
            F_CENTROIDS.into(),
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new(F_MEAN.into(), DataType::Float64),
                Field::new(F_WEIGHT.into(), DataType::Float64),
            ]))),
        ),
        Field::new(F_SUM.into(), DataType::Float64),
        Field::new(F_MIN.into(), DataType::Float64),
        Field::new(F_MAX.into(), DataType::Float64),
        Field::new(F_COUNT.into(), DataType::Float64),
        Field::new(F_MAX_SIZE.into(), DataType::Int64),
    ])
}

fn compact_dtype() -> DataType {
    DataType::Struct(vec![
        Field::new(
            F_CENTROIDS.into(),
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new(F_MEAN.into(), DataType::Float32),
                Field::new(F_WEIGHT.into(), DataType::Float32),
            ]))),
        ),
        Field::new(F_SUM.into(), DataType::Float64),
        Field::new(F_MIN.into(), DataType::Float32),
        Field::new(F_MAX.into(), DataType::Float32),
        Field::new(F_COUNT.into(), DataType::Float64),
        Field::new(F_MAX_SIZE.into(), DataType::Int64),
    ])
}

fn validate_schema_strict(s: &StructChunked, mode: PrecisionMode) -> Result<(), CodecError> {
    let expected = match mode {
        PrecisionMode::Canonical => canonical_dtype(),
        PrecisionMode::Compact => compact_dtype(),
    };

    let expected_fields = match &expected {
        DataType::Struct(v) => v,
        _ => unreachable!(),
    };

    for field in expected_fields {
        let name = field.name();
        let expected_dt = field.dtype();
        let ser = s.field_by_name(name).map_err(|_| {
            CodecError::MissingField(match name.as_str() {
                F_CENTROIDS => F_CENTROIDS,
                F_SUM => F_SUM,
                F_MIN => F_MIN,
                F_MAX => F_MAX,
                F_COUNT => F_COUNT,
                F_MAX_SIZE => F_MAX_SIZE,
                _ => "unknown",
            })
        })?;

        let found_dt = ser.dtype();
        if found_dt != expected_dt {
            if name.as_str() == F_CENTROIDS {
                return Err(CodecError::BadCentroidLayout {
                    expected_mean: Box::new(match mode {
                        PrecisionMode::Canonical => DataType::Float64,
                        PrecisionMode::Compact => DataType::Float32,
                    }),
                    expected_weight: Box::new(match mode {
                        PrecisionMode::Canonical => DataType::Float64,
                        PrecisionMode::Compact => DataType::Float32,
                    }),
                    found: Box::new(found_dt.clone()),
                });
            } else {
                return Err(CodecError::WrongDtype {
                    field: match name.as_str() {
                        F_SUM => F_SUM,
                        F_MIN => F_MIN,
                        F_MAX => F_MAX,
                        F_COUNT => F_COUNT,
                        F_MAX_SIZE => F_MAX_SIZE,
                        _ => "unknown",
                    },
                    found: Box::new(found_dt.clone()),
                    expected: Box::new(expected_dt.clone()),
                });
            }
        }
    }
    Ok(())
}

// --------------------- private writer core -----------------------------------

fn tdigest_to_series_with(td: &TDigest, name: &str, mode: PrecisionMode) -> PolarsResult<Series> {
    let cents = td.centroids();
    match mode {
        PrecisionMode::Canonical => {
            let centroids_list = pack_centroids_f64(cents)?;
            build_outer_struct(
                name,
                &centroids_list,
                td.sum(),
                Series::new(F_MIN.into(), [td.min()]),
                Series::new(F_MAX.into(), [td.max()]),
                td.count(),
                td.max_size() as i64,
            )
        }
        PrecisionMode::Compact => {
            let centroids_list = pack_centroids_f32(cents)?;
            build_outer_struct(
                name,
                &centroids_list,
                td.sum(),
                Series::new(F_MIN.into(), [td.min() as f32]),
                Series::new(F_MAX.into(), [td.max() as f32]),
                td.count(),
                td.max_size() as i64,
            )
        }
    }
}

// --------------------- pack/unpack helpers -----------------------------------

#[inline]
fn centroid_struct(mean_s: &Series, weight_s: &Series) -> PolarsResult<Series> {
    StructChunked::from_series(
        "centroid".into(),
        mean_s.len(),
        [mean_s, weight_s].into_iter(),
    )
    .map(|sc| sc.into_series())
}

/// Build `List<Struct{mean, weight}>` with f64 inner fields
fn pack_centroids_f64(cents: &[Centroid]) -> PolarsResult<Series> {
    let means: Vec<f64> = cents.iter().map(|c| c.mean()).collect();
    let wgts: Vec<f64> = cents.iter().map(|c| c.weight()).collect();
    let mean_s = Series::new(F_MEAN.into(), means);
    let weight_s = Series::new(F_WEIGHT.into(), wgts);
    let inner = centroid_struct(&mean_s, &weight_s)?;
    Ok(Series::new(F_CENTROIDS.into(), &[inner]))
}

/// Build `List<Struct{mean, weight}>` with f32 inner fields (compact)
fn pack_centroids_f32(cents: &[Centroid]) -> PolarsResult<Series> {
    let means: Vec<f32> = cents.iter().map(|c| c.mean() as f32).collect();
    let wgts: Vec<f32> = cents.iter().map(|c| c.weight() as f32).collect();
    let mean_s = Series::new(F_MEAN.into(), means);
    let weight_s = Series::new(F_WEIGHT.into(), wgts);
    let inner = centroid_struct(&mean_s, &weight_s)?;
    Ok(Series::new(F_CENTROIDS.into(), &[inner]))
}

/// Build the single-row outer struct from pre-built centroids + scalar fields.
fn build_outer_struct(
    name: &str,
    centroids_list: &Series,
    sum: f64,
    min: Series,
    max: Series,
    count: f64,
    max_size: i64,
) -> PolarsResult<Series> {
    let sum_s = Series::new(F_SUM.into(), [sum]);
    let count_s = Series::new(F_COUNT.into(), [count]);
    let max_size_s = Series::new(F_MAX_SIZE.into(), [max_size]);

    StructChunked::from_series(
        name.into(),
        1,
        [centroids_list, &sum_s, &min, &max, &count_s, &max_size_s].into_iter(),
    )
    .map(|sc| sc.into_series())
}

/// Parse one row’s centroids into `Vec<Centroid>` with strict null/shape checks.
fn unpack_centroids_strict(
    centroids_list: &ListChunked,
    row: usize,
    mode: PrecisionMode,
) -> Result<Vec<Centroid>, CodecError> {
    let Some(row_ser) = centroids_list.get_as_series(row) else {
        return Ok(Vec::new());
    };
    let sc = row_ser
        .struct_()
        .map_err(|_| CodecError::BadCentroidLayout {
            expected_mean: Box::new(match mode {
                PrecisionMode::Canonical => DataType::Float64,
                PrecisionMode::Compact => DataType::Float32,
            }),
            expected_weight: Box::new(match mode {
                PrecisionMode::Canonical => DataType::Float64,
                PrecisionMode::Compact => DataType::Float32,
            }),
            found: Box::new(row_ser.dtype().clone()),
        })?;

    match mode {
        PrecisionMode::Canonical => {
            let m_ser = sc
                .field_by_name(F_MEAN)
                .map_err(|_| CodecError::MissingField(F_MEAN))?;
            let w_ser = sc
                .field_by_name(F_WEIGHT)
                .map_err(|_| CodecError::MissingField(F_WEIGHT))?;
            let m = m_ser.f64().map_err(|_| CodecError::WrongDtype {
                field: F_MEAN,
                found: Box::new(m_ser.dtype().clone()),
                expected: Box::new(DataType::Float64),
            })?;
            let w = w_ser.f64().map_err(|_| CodecError::WrongDtype {
                field: F_WEIGHT,
                found: Box::new(w_ser.dtype().clone()),
                expected: Box::new(DataType::Float64),
            })?;

            let len = m.len().min(w.len());
            let mut out = Vec::with_capacity(len);
            for (idx, (mm, ww)) in m.into_iter().zip(w.into_iter()).enumerate() {
                let (Some(mv), Some(wv)) = (mm, ww) else {
                    return Err(CodecError::NullCentroid { row, index: idx });
                };
                out.push(Centroid::new(mv, wv));
            }
            Ok(out)
        }
        PrecisionMode::Compact => {
            let m_ser = sc
                .field_by_name(F_MEAN)
                .map_err(|_| CodecError::MissingField(F_MEAN))?;
            let w_ser = sc
                .field_by_name(F_WEIGHT)
                .map_err(|_| CodecError::MissingField(F_WEIGHT))?;
            let m = m_ser.f32().map_err(|_| CodecError::WrongDtype {
                field: F_MEAN,
                found: Box::new(m_ser.dtype().clone()),
                expected: Box::new(DataType::Float32),
            })?;
            let w = w_ser.f32().map_err(|_| CodecError::WrongDtype {
                field: F_WEIGHT,
                found: Box::new(w_ser.dtype().clone()),
                expected: Box::new(DataType::Float32),
            })?;

            let len = m.len().min(w.len());
            let mut out = Vec::with_capacity(len);
            for (idx, (mm, ww)) in m.into_iter().zip(w.into_iter()).enumerate() {
                let (Some(mv), Some(wv)) = (mm, ww) else {
                    return Err(CodecError::NullCentroid { row, index: idx });
                };
                out.push(Centroid::new(mv as f64, wv as f64));
            }
            Ok(out)
        }
    }
}

// --------------------- strict parser entry -----------------------------------

fn parse_tdigests_strict(input: &Series, mode: PrecisionMode) -> Result<Vec<TDigest>, CodecError> {
    let s = input.struct_().map_err(|_| CodecError::NotAStruct {
        series_name: input.name().to_string(),
    })?;

    validate_schema_strict(s, mode)?;

    // Bind each field to avoid temporary-drop borrow issues.
    let c_field = s
        .field_by_name(F_CENTROIDS)
        .map_err(|_| CodecError::MissingField(F_CENTROIDS))?;
    let centroids_list = c_field.list().map_err(|_| CodecError::WrongDtype {
        field: F_CENTROIDS,
        found: Box::new(c_field.dtype().clone()),
        expected: Box::new(DataType::List(Box::new(DataType::Struct(vec![])))), // message only
    })?;

    let sum_field = s
        .field_by_name(F_SUM)
        .map_err(|_| CodecError::MissingField(F_SUM))?;
    let count_field = s
        .field_by_name(F_COUNT)
        .map_err(|_| CodecError::MissingField(F_COUNT))?;
    let maxsz_field = s
        .field_by_name(F_MAX_SIZE)
        .map_err(|_| CodecError::MissingField(F_MAX_SIZE))?;
    let min_field = s
        .field_by_name(F_MIN)
        .map_err(|_| CodecError::MissingField(F_MIN))?;
    let max_field = s
        .field_by_name(F_MAX)
        .map_err(|_| CodecError::MissingField(F_MAX))?;

    let sum_col = sum_field.f64().map_err(|_| CodecError::WrongDtype {
        field: F_SUM,
        found: Box::new(sum_field.dtype().clone()),
        expected: Box::new(DataType::Float64),
    })?;
    let count_col = count_field.f64().map_err(|_| CodecError::WrongDtype {
        field: F_COUNT,
        found: Box::new(count_field.dtype().clone()),
        expected: Box::new(DataType::Float64),
    })?;
    let maxsz_col = maxsz_field.i64().map_err(|_| CodecError::WrongDtype {
        field: F_MAX_SIZE,
        found: Box::new(maxsz_field.dtype().clone()),
        expected: Box::new(DataType::Int64),
    })?;

    let n = input.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let sum = sum_col.get(i).ok_or(CodecError::NullField {
            field: F_SUM,
            row: i,
        })?;
        let count = count_col.get(i).ok_or(CodecError::NullField {
            field: F_COUNT,
            row: i,
        })?;
        let max_size_i64 = maxsz_col.get(i).ok_or(CodecError::NullField {
            field: F_MAX_SIZE,
            row: i,
        })?;
        let max_size = max_size_i64 as usize;

        let min = match mode {
            PrecisionMode::Canonical => {
                let col = min_field.f64().map_err(|_| CodecError::WrongDtype {
                    field: F_MIN,
                    found: Box::new(min_field.dtype().clone()),
                    expected: Box::new(DataType::Float64),
                })?;
                col.get(i).ok_or(CodecError::NullField {
                    field: F_MIN,
                    row: i,
                })?
            }
            PrecisionMode::Compact => {
                let col = min_field.f32().map_err(|_| CodecError::WrongDtype {
                    field: F_MIN,
                    found: Box::new(min_field.dtype().clone()),
                    expected: Box::new(DataType::Float32),
                })?;
                col.get(i).ok_or(CodecError::NullField {
                    field: F_MIN,
                    row: i,
                })? as f64
            }
        };

        let max = match mode {
            PrecisionMode::Canonical => {
                let col = max_field.f64().map_err(|_| CodecError::WrongDtype {
                    field: F_MAX,
                    found: Box::new(max_field.dtype().clone()),
                    expected: Box::new(DataType::Float64),
                })?;
                col.get(i).ok_or(CodecError::NullField {
                    field: F_MAX,
                    row: i,
                })?
            }
            PrecisionMode::Compact => {
                let col = max_field.f32().map_err(|_| CodecError::WrongDtype {
                    field: F_MAX,
                    found: Box::new(max_field.dtype().clone()),
                    expected: Box::new(DataType::Float32),
                })?;
                col.get(i).ok_or(CodecError::NullField {
                    field: F_MAX,
                    row: i,
                })? as f64
            }
        };

        let cents = unpack_centroids_strict(centroids_list, i, mode)?;
        out.push(
            TDigest::builder()
                .max_size(max_size)
                .with_centroids_and_stats(
                    cents,
                    DigestStats {
                        data_sum: sum,
                        total_weight: count,
                        data_min: min,
                        data_max: max,
                    },
                )
                .build(),
        );
    }

    Ok(out)
}
