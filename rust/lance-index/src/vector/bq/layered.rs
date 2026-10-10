// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixed prefix layouts for opt-in layered RaBitQ indices.

use lance_core::{Error, Result};

/// Plane widths for splitting a native full-precision code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RQLayout {
    pub high_bits: u8,
    pub low_bits: u8,
}

impl RQLayout {
    /// Resolve the only supported layered layouts: 1+2+2, 1+4+2 and 1+4+4.
    pub fn try_new(num_bits: u8) -> Result<Self> {
        let (high_bits, low_bits) = match num_bits {
            5 => (2, 2),
            7 => (4, 2),
            9 => (4, 4),
            _ => {
                return Err(Error::invalid_input(format!(
                    "IVF_RQ layered requires num_bits=5, 7 or 9, got {num_bits}"
                )));
            }
        };
        Ok(Self {
            high_bits,
            low_bits,
        })
    }
}

use super::ex_dot::{blocked_ex_code_bytes, pack_blocked_row};
use super::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN, RABIT_CODE_COLUMN,
};
use super::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_ADD_FACTORS_FIELD,
    EX_SCALE_FACTORS_COLUMN, EX_SCALE_FACTORS_FIELD, SCALE_FACTORS_COLUMN,
};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use std::sync::Arc;

pub const HIGH_ADD_FACTORS_COLUMN: &str = "__add_factors_ex_hi";
pub const HIGH_SCALE_FACTORS_COLUMN: &str = "__scale_factors_ex_hi";
pub const HIGH_BOUNDS_COLUMN: &str = "__rq_bounds_hi";
pub const FULL_BOUNDS_COLUMN: &str = "__rq_bounds_full";

// Each level stores ||w_level-w_sign||, |add_level-add_sign| and a
// coefficient for floating-point/LUT error. These bound the stored estimator,
// rather than its error relative to the unquantized vector.
fn bounds_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
        true,
    )
}

pub(crate) fn estimator_bounds(
    residuals: &[f32],
    codes: &[u8],
    dim: usize,
    bits: u8,
    binary: &super::transform::RabitRawQueryFactors,
    level: &super::transform::RabitRawQueryFactors,
) -> Result<ArrayRef> {
    let adds = level
        .ex_add_factors
        .as_ref()
        .ok_or_else(|| Error::internal("missing level add factors"))?;
    let scales = level
        .ex_scale_factors
        .as_ref()
        .ok_or_else(|| Error::internal("missing level scale factors"))?;
    let code_scale = (1u32 << bits) as f64;
    let bias = -(code_scale - 0.5);
    let mut bounds = Vec::with_capacity(residuals.len() / dim * 3);
    for (row, (residual, codes)) in residuals
        .chunks_exact(dim)
        .zip(codes.chunks_exact(dim))
        .enumerate()
    {
        let binary_scale = binary.scale_factors.value(row) as f64;
        let scale = scales.value(row) as f64;
        let norm = residual
            .iter()
            .zip(codes)
            .map(|(&r, &c)| {
                let sign = f64::from(u8::from(r.is_sign_positive()));
                let delta =
                    (code_scale * sign + f64::from(c) + bias) * scale - (sign - 0.5) * binary_scale;
                delta * delta
            })
            .sum::<f64>()
            .sqrt();
        for value in [
            norm,
            (f64::from(adds.value(row)) - f64::from(binary.add_factors.value(row))).abs(),
            binary_scale.abs() + code_scale * scale.abs(),
        ] {
            let rounded = value as f32;
            bounds.push(if f64::from(rounded) < value {
                rounded.next_up()
            } else {
                rounded
            });
        }
    }
    Ok(Arc::new(FixedSizeListArray::try_new_from_values(
        Float32Array::from(bounds),
        3,
    )?))
}

pub(crate) fn storage_fields(
    dim: usize,
    num_bits: u8,
    mut fields: Vec<Field>,
) -> Result<Vec<Field>> {
    let layout = RQLayout::try_new(num_bits)?;
    for (name, bits) in [
        (RABIT_BLOCKED_EX_CODE_COLUMN, layout.high_bits),
        (RABIT_BLOCKED_EX_CODE_LO_COLUMN, layout.low_bits),
    ] {
        fields.push(Field::new(
            name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                blocked_ex_code_bytes(dim, bits) as i32,
            ),
            true,
        ));
    }
    fields.extend([
        EX_ADD_FACTORS_FIELD.clone(),
        EX_SCALE_FACTORS_FIELD.clone(),
        Field::new(HIGH_ADD_FACTORS_COLUMN, DataType::Float32, true),
        Field::new(HIGH_SCALE_FACTORS_COLUMN, DataType::Float32, true),
        bounds_field(HIGH_BOUNDS_COLUMN),
        bounds_field(FULL_BOUNDS_COLUMN),
    ]);
    Ok(fields)
}

/// Split already-quantized full codes. No second quantization or scale search.
pub(crate) fn split_codes(
    values: &[u8],
    dim: usize,
    layout: RQLayout,
) -> Result<(ArrayRef, ArrayRef, Vec<u8>)> {
    let rows = values.len() / dim;
    let hi_bytes = blocked_ex_code_bytes(dim, layout.high_bits);
    let lo_bytes = blocked_ex_code_bytes(dim, layout.low_bits);
    let mut hi = vec![0; rows * hi_bytes];
    let mut lo = vec![0; rows * lo_bytes];
    let hi_values: Vec<u8> = values.iter().map(|v| v >> layout.low_bits).collect();
    let mask = ((1u16 << layout.low_bits) - 1) as u8;
    let mut low_values = vec![0; dim];
    for row in 0..rows {
        for (dst, src) in low_values
            .iter_mut()
            .zip(&values[row * dim..(row + 1) * dim])
        {
            *dst = src & mask;
        }
        pack_blocked_row(
            &hi_values[row * dim..(row + 1) * dim],
            layout.high_bits,
            &mut hi[row * hi_bytes..(row + 1) * hi_bytes],
        );
        pack_blocked_row(
            &low_values,
            layout.low_bits,
            &mut lo[row * lo_bytes..(row + 1) * lo_bytes],
        );
    }
    Ok((
        Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(hi),
            hi_bytes as i32,
        )?),
        Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(lo),
            lo_bytes as i32,
        )?),
        hi_values,
    ))
}

/// Query precision on a layered IVF_RQ index. Other layouts accept only `Full`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RQPrecision {
    /// Scan sign codes with the binary estimator.
    Sign,
    /// Scan sign and high codes with the high-level factor pair.
    High,
    /// Evaluate all stored bits.
    #[default]
    Full,
}

impl std::str::FromStr for RQPrecision {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "sign" => Ok(Self::Sign),
            "high" => Ok(Self::High),
            "full" => Ok(Self::Full),
            _ => Err(Error::invalid_input(format!(
                "rq_precision must be sign, high or full, got {value}"
            ))),
        }
    }
}

use arrow_array::RecordBatch;
use lance_core::cache::{CacheCodec, CacheKey};
use lance_core::deepsize::{Context, DeepSizeOf};
use std::borrow::Cow;

/// A plane and its factors share one persistent cache entry.
#[derive(Debug)]
pub struct PlaneBatch(pub RecordBatch);
impl DeepSizeOf for PlaneBatch {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// Plane entry holding a layered partition's [`HIGH_BOUNDS_COLUMN`] and
/// [`FULL_BOUNDS_COLUMN`] under [`SignBounds::Lazy`]. Only High precision,
/// and full precision on a file without error factors, prune with them.
pub const SIGN_BOUNDS_PLANE: u8 = 3;

/// Memory priority of the sign plane. Priority-aware tiers keep the sign
/// plane over the high plane over the low plane.
const SIGN_PLANE_MEMORY_PRIORITY: u8 = 3;

/// Appended to the key of a sign, high or low plane entry that holds its
/// codes alone ([`EntryColumns::Codes`]).
const CODE_ONLY_KEY_SUFFIX: &str = "-codes";

/// Where a layered index's cache keeps the estimator bounds columns
/// ([`HIGH_BOUNDS_COLUMN`], [`FULL_BOUNDS_COLUMN`]). The placement decides
/// what the sign plane entry holds, so it is part of that entry's key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SignBounds {
    /// In their own plane entry, [`SIGN_BOUNDS_PLANE`], read only by the
    /// scans that prune with them. Full-precision scans prune with the
    /// native error factors instead, so they read them only on a file
    /// without error factors.
    #[default]
    Lazy,
    /// In the sign plane entry, so every scan reads them with the sign
    /// codes. There is no bounds plane.
    Eager,
}

impl SignBounds {
    /// The spelling of `LANCE_RQ_SIGN_BOUNDS`: `lazy` or `eager`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lazy => "lazy",
            Self::Eager => "eager",
        }
    }
}

impl std::fmt::Display for SignBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which columns an IVF_RQ index's cache entries hold: a layered
/// partition's plane entries and a native flat partition's entry. The
/// composition decides what an entry holds, so it is part of its key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EntryColumns {
    /// Every column a read of the plane or partition returns.
    #[default]
    All,
    /// Only the columns the index's resident store does not keep: the codes
    /// and the estimator bounds (see [`plane_entry_columns`]). Every read
    /// attaches the store's rows of the others, views of the store for a
    /// whole plane or partition and copies for gathered rows, so the scored
    /// batch is the one an `All` entry holds. Only an index whose small
    /// columns are resident keeps such entries.
    Codes,
}

impl EntryColumns {
    /// The spelling of `LANCE_RQ_ENTRY_COLUMNS`: `all` or `codes`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Codes => "codes",
        }
    }

    /// What an index that asks for `self` keeps in its entries: `Codes`
    /// needs the resident store, so without it the entries hold `All`.
    pub fn resolve(self, resident: bool) -> Self {
        match self {
            Self::Codes if resident => Self::Codes,
            _ => Self::All,
        }
    }
}

impl std::fmt::Display for EntryColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Plane keys live under the immutable index UUID namespace.
pub struct PlaneKey {
    pub partition: usize,
    pub plane: u8,
    /// The index's bounds placement. Only the sign plane's key depends on
    /// it: the other planes hold the same columns under either placement.
    pub sign_bounds: SignBounds,
    /// What the index's entries hold. The keys of planes 0-2 depend on it;
    /// the bounds plane holds file columns alone either way.
    pub entry_columns: EntryColumns,
}
impl CacheKey for PlaneKey {
    type ValueType = PlaneBatch;
    fn key(&self) -> Cow<'_, str> {
        // Persisted entries outlive the process, so each composition of an
        // entry needs its own key. The sign plane with bounds keeps the key
        // its entries had before the bounds plane existed, and a full entry
        // the key it had before code-only entries existed.
        let codes = if self.entry_columns == EntryColumns::Codes && self.plane < SIGN_BOUNDS_PLANE {
            CODE_ONLY_KEY_SUFFIX
        } else {
            ""
        };
        if self.plane == 0 && self.sign_bounds == SignBounds::Lazy {
            format!("{}:0-bounds{codes}", self.partition).into()
        } else {
            format!("{}:{}{codes}", self.partition, self.plane).into()
        }
    }
    fn type_name() -> &'static str {
        "RQPlane"
    }
    fn codec_for_key(&self) -> Option<CacheCodec> {
        Self::codec().map(|codec| {
            codec
                .with_plane_tag(self.plane)
                .with_memory_priority(plane_memory_priority(self.plane))
        })
    }
    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PlaneBatch>())
    }
}

/// The bounds plane is read with the high plane, by High precision, so it
/// ranks with it.
fn plane_memory_priority(plane: u8) -> u8 {
    match plane {
        SIGN_BOUNDS_PLANE => SIGN_PLANE_MEMORY_PRIORITY - 1,
        _ => SIGN_PLANE_MEMORY_PRIORITY.saturating_sub(plane),
    }
}

/// The sign plane's columns under [`SignBounds::Eager`]: those it keeps
/// under [`SignBounds::Lazy`], then the bounds plane's.
static SIGN_PLANE_WITH_BOUNDS_COLUMNS: [&str; 7] = [
    lance_core::ROW_ID,
    RABIT_CODE_COLUMN,
    ADD_FACTORS_COLUMN,
    SCALE_FACTORS_COLUMN,
    ERROR_FACTORS_COLUMN,
    HIGH_BOUNDS_COLUMN,
    FULL_BOUNDS_COLUMN,
];
/// Where the bounds columns start in [`SIGN_PLANE_WITH_BOUNDS_COLUMNS`].
const BOUNDS_COLUMNS_START: usize = 5;

/// Column projection of a layered partition's plane entry, including its
/// factors: sign (0), high (1), low (2) or [`SIGN_BOUNDS_PLANE`], with the
/// bounds columns placed by `sign_bounds`. Empty for a plane the placement
/// does not have.
pub fn plane_columns(plane: u8, sign_bounds: SignBounds) -> &'static [&'static str] {
    match (plane, sign_bounds) {
        (0, SignBounds::Lazy) => &SIGN_PLANE_WITH_BOUNDS_COLUMNS[..BOUNDS_COLUMNS_START],
        (0, SignBounds::Eager) => &SIGN_PLANE_WITH_BOUNDS_COLUMNS,
        (1, _) => &[
            RABIT_BLOCKED_EX_CODE_COLUMN,
            HIGH_ADD_FACTORS_COLUMN,
            HIGH_SCALE_FACTORS_COLUMN,
        ],
        (2, _) => &[
            RABIT_BLOCKED_EX_CODE_LO_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
        ],
        (SIGN_BOUNDS_PLANE, SignBounds::Lazy) => {
            &SIGN_PLANE_WITH_BOUNDS_COLUMNS[BOUNDS_COLUMNS_START..]
        }
        _ => &[],
    }
}

/// Columns a plane entry of `plane` holds under `entry_columns`, in
/// [`plane_columns`] order: every column of the plane, or under
/// [`EntryColumns::Codes`] only those a resident store does not keep.
pub fn plane_entry_columns(
    plane: u8,
    sign_bounds: SignBounds,
    entry_columns: EntryColumns,
) -> Vec<&'static str> {
    let columns = plane_columns(plane, sign_bounds).iter().copied();
    match entry_columns {
        EntryColumns::All => columns.collect(),
        EntryColumns::Codes => columns
            .filter(|name| super::resident::is_file_column(name))
            .collect(),
    }
}

/// A quantized candidate tied to an immutable index and its physical partition.
///
/// Distributed executors may perform a global distance cut on these values, then
/// return the survivors to the same index for full-code reranking with the same query.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantizedCandidate {
    pub index_uuid: uuid::Uuid,
    pub partition_id: usize,
    pub row_offset: u32,
    pub row_id: u64,
    pub distance: f32,
    pub centroid_distance: f32,
    /// Original-partition binary LUT value, including its SIMD rounding policy.
    pub binary_inner_product: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::bq::{
        RQBuildParams, builder::RabitQuantizer, storage::RabitQuantizationStorage,
        transform::RQTransformer,
    };
    use crate::vector::quantizer::{Quantization, QuantizerStorage};
    use crate::vector::storage::{
        DistCalculator, DistanceCalculatorOptions, QueryResidual, VectorStore,
    };
    use crate::vector::transform::Transformer;
    use crate::vector::{ApproxMode, CENTROID_DIST_COLUMN, PART_ID_COLUMN};
    use arrow_array::types::Float32Type;
    use arrow_array::{Float32Array, UInt32Array, UInt64Array, cast::AsArray};
    use lance_arrow::RecordBatchExt;
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    #[rstest]
    #[case::rq5_l2(5, DistanceType::L2)]
    #[case::rq7_l2(7, DistanceType::L2)]
    #[case::rq9_l2(9, DistanceType::L2)]
    #[case::rq5_dot(5, DistanceType::Dot)]
    #[case::rq7_dot(7, DistanceType::Dot)]
    #[case::rq9_dot(9, DistanceType::Dot)]
    fn layered_roundtrip_and_levels(
        #[case] bits: u8,
        #[case] distance_type: DistanceType,
        #[values(64, 72)] dim: usize,
        #[values(0.0, 0.25)] centroid_value: f32,
    ) {
        const ROWS: usize = 37;
        let values: Vec<f32> = (0..dim * ROWS)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 50.)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), dim as i32)
                .unwrap();
        let rq = RabitQuantizer::build(
            &vectors,
            distance_type,
            &RQBuildParams::new(bits).with_layered(true),
        )
        .unwrap();
        let norms = Float32Array::from(
            values
                .chunks(dim)
                .map(|r| r.iter().map(|v| v * v).sum::<f32>())
                .collect::<Vec<_>>(),
        );
        let input = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(vectors.clone()) as ArrayRef),
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from((0..ROWS as u64).collect::<Vec<_>>())) as ArrayRef,
            ),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0; ROWS])) as ArrayRef,
            ),
            (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef),
        ])
        .unwrap();
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![centroid_value; dim]),
            dim as i32,
        )
        .unwrap();
        let rotated_centroid = rq
            .quantize_split(&centroids)
            .unwrap()
            .rotated_residuals
            .unwrap();
        let batch = RQTransformer::new(rq.clone(), distance_type, centroids.clone(), "vector")
            .unwrap()
            .transform(&input)
            .unwrap();
        let full = RabitQuantizationStorage::try_from_batch(
            batch.clone(),
            rq.metadata_ref(),
            distance_type,
            None,
        )
        .unwrap();
        let mut metadata = rq.metadata_ref().clone();
        metadata.layered = false;
        let native_rq = RabitQuantizer::try_from(
            RabitQuantizer::from_metadata(&metadata, distance_type).unwrap(),
        )
        .unwrap();
        let layered_codes = rq.quantize_split(&vectors).unwrap();
        let native_codes = native_rq.quantize_split(&vectors).unwrap();
        assert_eq!(
            layered_codes.binary_codes.as_ref(),
            native_codes.binary_codes.as_ref()
        );
        assert_eq!(layered_codes.ex_code_values, native_codes.ex_code_values);
        assert_eq!(
            layered_codes.ex_res_dot_dists,
            native_codes.ex_res_dot_dists
        );
        let single_batch = RQTransformer::new(native_rq, distance_type, centroids, "vector")
            .unwrap()
            .transform(&input)
            .unwrap();
        let single =
            RabitQuantizationStorage::try_from_batch(single_batch, &metadata, distance_type, None)
                .unwrap();
        let dist_q_c = match distance_type {
            DistanceType::L2 => values[..dim]
                .iter()
                .map(|v| (v - centroid_value).powi(2))
                .sum(),
            DistanceType::Dot => 1.0 - values[..dim].iter().sum::<f32>() * centroid_value,
            _ => unreachable!(),
        };
        let query: ArrayRef = Arc::new(Float32Array::from(values[..dim].to_vec()));
        assert_eq!(
            full.dist_calculator(query.clone(), dist_q_c)
                .distance_all(ROWS),
            single
                .dist_calculator(query.clone(), dist_q_c)
                .distance_all(ROWS)
        );
        for invalid in [f32::NAN, -1.0] {
            let hints = FixedSizeListArray::try_new_from_values(
                Float32Array::from(vec![invalid; ROWS * 3]),
                3,
            )
            .unwrap();
            let invalid_batch = batch
                .replace_column_by_name(HIGH_BOUNDS_COLUMN, Arc::new(hints))
                .unwrap();
            let store = RabitQuantizationStorage::try_from_batch_at_precision(
                invalid_batch,
                rq.metadata_ref(),
                distance_type,
                None,
                RQPrecision::High,
            )
            .unwrap();
            let calc = store.dist_calculator(query.clone(), dist_q_c);
            assert_eq!(
                calc.distance_all(ROWS),
                full.dist_calculator_with_scratch(
                    query.clone(),
                    dist_q_c,
                    None,
                    &mut Vec::new(),
                    DistanceCalculatorOptions {
                        approx_mode: ApproxMode::Normal,
                        rq_precision: RQPrecision::High,
                    },
                )
                .distance_all(ROWS)
            );
            for (row, ip) in calc.binary_inner_products().into_iter().enumerate() {
                assert_eq!(calc.raw_query_lower_bound(row, ip), Some(f32::NEG_INFINITY));
            }
        }
        for precision in [RQPrecision::Sign, RQPrecision::High, RQPrecision::Full] {
            let projected = RabitQuantizationStorage::try_from_batch_at_precision(
                batch.clone(),
                rq.metadata_ref(),
                distance_type,
                None,
                precision,
            )
            .unwrap();
            for centered in [false, true] {
                let context = || {
                    centered.then_some(QueryResidual::RabitRawQuery {
                        rotated_centroid: Some(&rotated_centroid),
                        query: None,
                    })
                };
                for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                    let mut scratch = Vec::new();
                    let mut projected_scratch = Vec::new();
                    let calc = full.dist_calculator_with_scratch(
                        query.clone(),
                        dist_q_c,
                        context(),
                        &mut scratch,
                        DistanceCalculatorOptions {
                            approx_mode,
                            rq_precision: precision,
                        },
                    );
                    let projected_calc = projected.dist_calculator_with_scratch(
                        query.clone(),
                        dist_q_c,
                        context(),
                        &mut projected_scratch,
                        DistanceCalculatorOptions {
                            approx_mode,
                            rq_precision: RQPrecision::Full,
                        },
                    );
                    assert_eq!(calc.distance_all(ROWS), projected_calc.distance_all(ROWS));
                    if precision == RQPrecision::Full {
                        let mut native_scratch = Vec::new();
                        let native = single.dist_calculator_with_scratch(
                            query.clone(),
                            dist_q_c,
                            context(),
                            &mut native_scratch,
                            DistanceCalculatorOptions {
                                approx_mode,
                                rq_precision: RQPrecision::Full,
                            },
                        );
                        let binary = calc.binary_inner_products();
                        assert_eq!(binary, native.binary_inner_products());
                        assert_eq!(calc.distance_all(ROWS), native.distance_all(ROWS));
                        for (row, &ip) in binary.iter().enumerate() {
                            assert_eq!(
                                calc.raw_query_lower_bound(row, ip),
                                native.raw_query_lower_bound(row, ip),
                                "full pruning differs from native at row {row}, mode={approx_mode:?}",
                            );
                        }
                    }
                    if precision == RQPrecision::High && approx_mode == ApproxMode::Accurate {
                        let binary = calc.binary_inner_products();
                        let distances = calc.distance_all(ROWS);
                        for (row, &ip) in binary.iter().enumerate() {
                            let bound = calc.raw_query_lower_bound(row, ip).unwrap();
                            assert!(
                                bound <= distances[row],
                                "invalid bound at row {row}: {bound} > {}",
                                distances[row]
                            );
                        }
                        let best = distances.iter().copied().min_by(f32::total_cmp).unwrap();
                        assert!(
                            binary
                                .iter()
                                .enumerate()
                                .any(|(row, &ip)| calc.raw_query_lower_bound(row, ip).unwrap()
                                    > best),
                            "fixture must exercise pruning"
                        );
                    }
                    // Exercise SIMD plus tail rows, both context forms, and range cuts.
                    for (lower, upper) in [(None, None), (Some(-0.5), Some(20.0))] {
                        let distances = calc.distance_all(ROWS);
                        let mut expected: Vec<_> = distances
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|(_, d)| {
                                lower.is_none_or(|v| *d >= v) && upper.is_none_or(|v| *d < v)
                            })
                            .map(|(id, d)| (id as u64, d))
                            .collect();
                        expected.sort_by(|a, b| a.1.total_cmp(&b.1));
                        expected.truncate(3);
                        let mut heap = std::collections::BinaryHeap::new();
                        calc.accumulate_topk_with_scratch(
                            3,
                            lower,
                            upper,
                            u64::from,
                            &mut heap,
                            &mut Vec::new(),
                            &mut Vec::new(),
                            &mut Vec::new(),
                            &mut Vec::new(),
                        );
                        let actual: Vec<_> = heap
                            .into_sorted_vec()
                            .into_iter()
                            .map(|n| (n.id, n.dist.0))
                            .collect();
                        let case = format!(
                            "bits={bits} precision={precision:?} dim={dim} centered={centered} mode={approx_mode:?} range={lower:?}..{upper:?}"
                        );
                        if precision == RQPrecision::Full {
                            let mut native_scratch = Vec::new();
                            let native = single.dist_calculator_with_scratch(
                                query.clone(),
                                dist_q_c,
                                context(),
                                &mut native_scratch,
                                DistanceCalculatorOptions {
                                    approx_mode,
                                    rq_precision: RQPrecision::Full,
                                },
                            );
                            let mut native_heap = std::collections::BinaryHeap::new();
                            native.accumulate_topk_with_scratch(
                                3,
                                lower,
                                upper,
                                u64::from,
                                &mut native_heap,
                                &mut Vec::new(),
                                &mut Vec::new(),
                                &mut Vec::new(),
                                &mut Vec::new(),
                            );
                            let native_actual: Vec<_> = native_heap
                                .into_sorted_vec()
                                .into_iter()
                                .map(|n| (n.id, n.dist.0))
                                .collect();
                            assert_eq!(actual, native_actual, "{case}");
                        }
                        // Normal mode (and Full in both modes) prunes with native's
                        // statistical confidence bound, which a random rotation can violate
                        // for a true top-k row. Pruned top-k is only guaranteed exact when
                        // every expected row's bound holds.
                        let binary = calc.binary_inner_products();
                        let bounds_hold = expected.iter().all(|&(id, d)| {
                            calc.raw_query_lower_bound(id as usize, binary[id as usize])
                                .is_none_or(|bound| bound <= d)
                        });
                        if bounds_hold {
                            // Equal distances have no defined order in the heap output.
                            let by_distance_then_id = |a: &(u64, f32), b: &(u64, f32)| {
                                a.1.total_cmp(&b.1).then(a.0.cmp(&b.0))
                            };
                            let mut actual = actual;
                            actual.sort_by(by_distance_then_id);
                            expected.sort_by(by_distance_then_id);
                            assert_eq!(actual, expected, "{case}");
                        }
                    }
                }
            }
        }
        let missing = batch.drop_column(RABIT_BLOCKED_EX_CODE_LO_COLUMN).unwrap();
        assert!(
            RabitQuantizationStorage::try_from_batch(
                missing,
                rq.metadata_ref(),
                distance_type,
                None
            )
            .unwrap_err()
            .to_string()
            .contains("missing column")
        );
        assert_eq!(
            batch[HIGH_ADD_FACTORS_COLUMN]
                .as_primitive::<Float32Type>()
                .len(),
            ROWS
        );
    }

    #[test]
    fn plane_key_codec_tags_plane_and_priority() {
        for entry_columns in [EntryColumns::All, EntryColumns::Codes] {
            for (plane, priority) in [(0, 3), (1, 2), (2, 1), (SIGN_BOUNDS_PLANE, 2)] {
                let codec = PlaneKey {
                    partition: 7,
                    plane,
                    sign_bounds: SignBounds::Lazy,
                    entry_columns,
                }
                .codec_for_key()
                .unwrap();
                assert_eq!(codec.plane_tag(), Some(plane));
                assert_eq!(codec.memory_priority(), priority);
                assert!(codec.supports_row_selection());
            }
        }
        assert_eq!(PlaneKey::codec().unwrap().plane_tag(), None);
    }

    /// Only the sign plane's key depends on the bounds placement, and the
    /// sign plane with bounds keeps its original key.
    #[test]
    fn plane_key_separates_sign_plane_compositions() {
        let key = |plane, sign_bounds| {
            let plane_key = PlaneKey {
                partition: 7,
                plane,
                sign_bounds,
                entry_columns: EntryColumns::All,
            };
            plane_key.key().into_owned()
        };
        assert_eq!(key(0, SignBounds::Eager), "7:0");
        assert_ne!(key(0, SignBounds::Lazy), key(0, SignBounds::Eager));
        for plane in [1, 2, SIGN_BOUNDS_PLANE] {
            assert_eq!(key(plane, SignBounds::Lazy), format!("7:{plane}"));
            assert_eq!(key(plane, SignBounds::Eager), format!("7:{plane}"));
        }
    }

    /// The bounds plane holds exactly the columns the lazy placement leaves
    /// out of the sign plane, in the eager sign plane's order.
    #[test]
    fn plane_columns_place_bounds_by_sign_bounds() {
        let lazy_sign = plane_columns(0, SignBounds::Lazy);
        let bounds = plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy);
        assert_eq!(bounds, [HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN]);
        assert_eq!(
            [lazy_sign, bounds].concat(),
            plane_columns(0, SignBounds::Eager)
        );
        assert!(!lazy_sign.iter().any(|name| bounds.contains(name)));
        assert!(plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Eager).is_empty());
        for plane in [1, 2] {
            assert_eq!(
                plane_columns(plane, SignBounds::Lazy),
                plane_columns(plane, SignBounds::Eager)
            );
            assert_eq!(plane_columns(plane, SignBounds::Lazy).len(), 3);
        }
        assert!(plane_columns(SIGN_BOUNDS_PLANE + 1, SignBounds::Lazy).is_empty());
    }

    /// Code-only entries of the sign, high and low planes take their own
    /// keys; every other key is the one its entries had before, so full
    /// entries persisted by an earlier build keep being found.
    #[test]
    fn plane_key_separates_entry_columns() {
        let key = |plane, sign_bounds, entry_columns| {
            PlaneKey {
                partition: 7,
                plane,
                sign_bounds,
                entry_columns,
            }
            .key()
            .into_owned()
        };
        let full_keys = [
            (0, SignBounds::Lazy, "7:0-bounds"),
            (0, SignBounds::Eager, "7:0"),
            (1, SignBounds::Lazy, "7:1"),
            (1, SignBounds::Eager, "7:1"),
            (2, SignBounds::Lazy, "7:2"),
            (2, SignBounds::Eager, "7:2"),
            (SIGN_BOUNDS_PLANE, SignBounds::Lazy, "7:3"),
            (SIGN_BOUNDS_PLANE, SignBounds::Eager, "7:3"),
        ];
        for (plane, sign_bounds, full) in full_keys {
            assert_eq!(key(plane, sign_bounds, EntryColumns::All), full);
            let codes = key(plane, sign_bounds, EntryColumns::Codes);
            if plane == SIGN_BOUNDS_PLANE {
                assert_eq!(codes, full);
            } else {
                assert_eq!(codes, format!("{full}-codes"));
            }
        }
    }

    /// A code-only entry holds the plane's code and bounds columns, in the
    /// plane's order; the bounds plane holds the same columns either way.
    #[test]
    fn plane_entry_columns_keep_file_columns() {
        for sign_bounds in [SignBounds::Lazy, SignBounds::Eager] {
            for plane in [0, 1, 2, SIGN_BOUNDS_PLANE] {
                assert_eq!(
                    plane_entry_columns(plane, sign_bounds, EntryColumns::All),
                    plane_columns(plane, sign_bounds)
                );
            }
        }
        let codes =
            |plane, sign_bounds| plane_entry_columns(plane, sign_bounds, EntryColumns::Codes);
        assert_eq!(codes(0, SignBounds::Lazy), [RABIT_CODE_COLUMN]);
        assert_eq!(
            codes(0, SignBounds::Eager),
            [RABIT_CODE_COLUMN, HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN]
        );
        assert_eq!(codes(1, SignBounds::Lazy), [RABIT_BLOCKED_EX_CODE_COLUMN]);
        assert_eq!(
            codes(2, SignBounds::Eager),
            [RABIT_BLOCKED_EX_CODE_LO_COLUMN]
        );
        assert_eq!(
            codes(SIGN_BOUNDS_PLANE, SignBounds::Lazy),
            [HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN]
        );
        assert!(codes(SIGN_BOUNDS_PLANE, SignBounds::Eager).is_empty());
    }

    #[test]
    fn entry_columns_need_a_resident_store() {
        assert_eq!(EntryColumns::Codes.resolve(true), EntryColumns::Codes);
        assert_eq!(EntryColumns::Codes.resolve(false), EntryColumns::All);
        assert_eq!(EntryColumns::All.resolve(true), EntryColumns::All);
        assert_eq!(EntryColumns::All.resolve(false), EntryColumns::All);
        assert_eq!(EntryColumns::default(), EntryColumns::All);
        assert_eq!(EntryColumns::Codes.to_string(), "codes");
        assert_eq!(EntryColumns::All.to_string(), "all");
    }

    /// A layered store needs the full bounds only to prune full precision on
    /// a file without error factors. Without the high bounds, a High scan of
    /// a layered store scores every row instead of pruning with the error
    /// factors, which bound only the full-precision estimator.
    #[rstest]
    fn layered_bounds_required_only_when_read(
        #[values(DistanceType::L2, DistanceType::Dot)] distance_type: DistanceType,
    ) {
        const ROWS: usize = 37;
        const DIM: usize = 64;
        let values: Vec<f32> = (0..DIM * ROWS)
            .map(|i| ((i * 13 % 97) as f32 - 48.) / 48.)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), DIM as i32)
                .unwrap();
        let rq = RabitQuantizer::build(
            &vectors,
            distance_type,
            &RQBuildParams::new(7).with_layered(true),
        )
        .unwrap();
        let norms = Float32Array::from(
            values
                .chunks(DIM)
                .map(|r| r.iter().map(|v| v * v).sum::<f32>())
                .collect::<Vec<_>>(),
        );
        let input = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(vectors) as ArrayRef),
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from((0..ROWS as u64).collect::<Vec<_>>())) as ArrayRef,
            ),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0; ROWS])) as ArrayRef,
            ),
            (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef),
        ])
        .unwrap();
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; DIM]), DIM as i32)
                .unwrap();
        let batch = RQTransformer::new(rq.clone(), distance_type, centroids, "vector")
            .unwrap()
            .transform(&input)
            .unwrap();
        let metadata = rq.metadata_ref();
        let open = |batch: RecordBatch| {
            RabitQuantizationStorage::try_from_batch(batch, metadata, distance_type, None)
        };
        let lean_batch = batch
            .drop_column(HIGH_BOUNDS_COLUMN)
            .unwrap()
            .drop_column(FULL_BOUNDS_COLUMN)
            .unwrap();
        let bounded = open(batch.clone()).unwrap();
        let lean = open(lean_batch.clone()).unwrap();
        let query: ArrayRef = Arc::new(Float32Array::from(values[..DIM].to_vec()));
        let dist_q_c = match distance_type {
            DistanceType::L2 => values[..DIM].iter().map(|v| v * v).sum(),
            _ => 1.0,
        };
        fn lower_bounds(calc: &super::super::storage::RabitDistCalculator<'_>) -> Vec<Option<u32>> {
            calc.binary_inner_products()
                .into_iter()
                .enumerate()
                .map(|(row, ip)| calc.raw_query_lower_bound(row, ip).map(f32::to_bits))
                .collect()
        }
        for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
            for rq_precision in [RQPrecision::High, RQPrecision::Full] {
                let options = DistanceCalculatorOptions {
                    approx_mode,
                    rq_precision,
                };
                let case = format!("{distance_type:?} {approx_mode:?} {rq_precision:?}");
                let (mut bounded_scratch, mut lean_scratch) = (Vec::new(), Vec::new());
                let bounded_calc = bounded.dist_calculator_with_scratch(
                    query.clone(),
                    dist_q_c,
                    None,
                    &mut bounded_scratch,
                    options,
                );
                let lean_calc = lean.dist_calculator_with_scratch(
                    query.clone(),
                    dist_q_c,
                    None,
                    &mut lean_scratch,
                    options,
                );
                assert_eq!(
                    lean_calc.distance_all(ROWS),
                    bounded_calc.distance_all(ROWS),
                    "{case}"
                );
                let bounded_bounds = lower_bounds(&bounded_calc);
                assert!(bounded_bounds.iter().all(Option::is_some), "{case}");
                if rq_precision == RQPrecision::Full {
                    assert_eq!(lower_bounds(&lean_calc), bounded_bounds, "{case}");
                } else {
                    assert!(
                        lower_bounds(&lean_calc).iter().all(Option::is_none),
                        "{case}"
                    );
                }
            }
        }

        let error = open(lean_batch.drop_column(ERROR_FACTORS_COLUMN).unwrap()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains("missing column"), "{error}");
        assert!(message.contains(FULL_BOUNDS_COLUMN), "{error}");
        open(
            batch
                .drop_column(ERROR_FACTORS_COLUMN)
                .unwrap()
                .drop_column(HIGH_BOUNDS_COLUMN)
                .unwrap(),
        )
        .unwrap();
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(6)]
    #[case(8)]
    fn rejects_unsupported_layout(#[case] bits: u8) {
        assert!(
            RQLayout::try_new(bits)
                .unwrap_err()
                .to_string()
                .contains("requires num_bits=5, 7 or 9")
        );
    }
}
