// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! RaBitQ sign bounds and scoring of separately gathered ex rows.
//! These primitives preserve the existing full scan's LUT, kernel and heap
//! semantics. Query scheduling and I/O selection belong to the caller.

use super::*;

/// Full-precision ex rows gathered for a stage-2 rerank: blocked native codes
/// or high and low plane codes, plus the full ex factors, all
/// indexed by the survivor's ex index. Whole planes keep the sign-plane
/// offset as the ex index; a compact gather uses the position in the gather.
#[derive(Debug, Clone, Copy)]
pub struct ExRows<'a> {
    pub(super) hi: &'a [u8],
    pub(super) hi_row_bytes: usize,
    pub(super) lo: Option<&'a [u8]>,
    pub(super) ex_add: &'a [f32],
    pub(super) ex_scale: &'a [f32],
}

impl<'a> ExRows<'a> {
    /// Borrow the rows of a high-plane batch (plane 1: high codes) and a
    /// low-plane batch (plane 2: low codes and the full ex factors) with the
    /// same row order, validating code widths as the eager loader does.
    pub fn from_plane_batches(
        hi: &'a RecordBatch,
        lo: &'a RecordBatch,
        rotated_dim: usize,
        num_bits: u8,
    ) -> Result<Self> {
        let column = |batch: &'a RecordBatch, name: &str| {
            batch.column_by_name(name).ok_or_else(|| {
                Error::invalid_input(format!("RabitQ gathered ex rows require column {name}"))
            })
        };
        let hi_codes = column(hi, RABIT_BLOCKED_EX_CODE_COLUMN)?;
        let lo_codes = column(lo, RABIT_BLOCKED_EX_CODE_LO_COLUMN)?;
        let factors = |name: &str| -> Result<&'a [f32]> {
            let values = column(lo, name)?
                .as_primitive_opt::<Float32Type>()
                .ok_or_else(|| {
                    Error::invalid_input(format!("RabitQ column {name} must be float32"))
                })?
                .values();
            Ok(values.as_ref())
        };
        let ex_add = factors(EX_ADD_FACTORS_COLUMN)?;
        let ex_scale = factors(EX_SCALE_FACTORS_COLUMN)?;
        let rows = hi.num_rows();
        if lo.num_rows() != rows || ex_add.len() != rows || ex_scale.len() != rows {
            return Err(Error::invalid_input(format!(
                "RabitQ gathered ex rows disagree in length: high plane {rows}, low plane {}, ex add factors {}, ex scale factors {}",
                lo.num_rows(),
                ex_add.len(),
                ex_scale.len()
            )));
        }
        let planes = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                Field::new(
                    RABIT_BLOCKED_EX_CODE_COLUMN,
                    hi_codes.data_type().clone(),
                    true,
                ),
                Field::new(
                    RABIT_BLOCKED_EX_CODE_LO_COLUMN,
                    lo_codes.data_type().clone(),
                    true,
                ),
            ])),
            vec![hi_codes.clone(), lo_codes.clone()],
        )?;
        load_ex_code_planes(planes, rotated_dim, num_bits)?;
        // Validated above as uint8 fixed-size lists of the layout's widths.
        let hi_codes = hi_codes.as_fixed_size_list();
        let lo_codes = lo_codes.as_fixed_size_list();
        Ok(Self {
            hi: hi_codes.values().as_primitive::<UInt8Type>().values(),
            hi_row_bytes: hi_codes.value_length() as usize,
            lo: Some(lo_codes.values().as_primitive::<UInt8Type>().values()),
            ex_add,
            ex_scale,
        })
    }

    /// Borrow current native blocked ex codes and their full add/scale factors.
    /// The absent low plane selects the same single-plane kernel as native Full.
    /// `rotated_dim` is the padded dimension and `num_bits` is the total width
    /// (2 through 9). Null codes or factors and mismatched widths are rejected.
    ///
    /// ```
    /// # use arrow_array::RecordBatch;
    /// # use lance_core::Result;
    /// # use lance_index::vector::bq::storage::ExRows;
    /// fn gathered_rows(batch: &RecordBatch) -> Result<ExRows<'_>> {
    ///     ExRows::from_native_batch(batch, 2048, 7)
    /// }
    /// ```
    pub fn from_native_batch(
        batch: &'a RecordBatch,
        rotated_dim: usize,
        num_bits: u8,
    ) -> Result<Self> {
        let ex_bits = rabit_ex_bits(num_bits)?;
        if ex_bits == 0 {
            return Err(Error::invalid_input(
                "native ex rows require num_bits greater than 1",
            ));
        }
        let codes = batch
            .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
            .and_then(|column| column.as_fixed_size_list_opt())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "native ex rows require fixed-size-list column {RABIT_BLOCKED_EX_CODE_COLUMN}"
                ))
            })?;
        let row_bytes = blocked_ex_code_bytes(rotated_dim, ex_bits);
        if codes.value_type() != DataType::UInt8
            || codes.value_length() as usize != row_bytes
            || codes.null_count() != 0
            || codes.values().null_count() != 0
        {
            return Err(Error::invalid_input(format!(
                "native ex rows require {row_bytes} non-null uint8 codes per row for rotated_dim={rotated_dim}, num_bits={num_bits}"
            )));
        }
        let factors = |name: &str| -> Result<&'a [f32]> {
            let values = batch
                .column_by_name(name)
                .and_then(|column| column.as_primitive_opt::<Float32Type>())
                .ok_or_else(|| {
                    Error::invalid_input(format!("native ex rows require float32 column {name}"))
                })?;
            if values.null_count() != 0 || values.len() != codes.len() {
                return Err(Error::invalid_input(format!(
                    "native ex rows require {} non-null {name} factors, got {} rows and {} nulls",
                    codes.len(),
                    values.len(),
                    values.null_count()
                )));
            }
            Ok(values.values().as_ref())
        };
        Ok(Self {
            hi: codes.values().as_primitive::<UInt8Type>().values(),
            hi_row_bytes: row_bytes,
            lo: None,
            ex_add: factors(EX_ADD_FACTORS_COLUMN)?,
            ex_scale: factors(EX_SCALE_FACTORS_COLUMN)?,
        })
    }

    /// Number of gathered rows.
    pub fn len(&self) -> usize {
        self.ex_add.len()
    }

    /// Whether this view has no rows.
    pub fn is_empty(&self) -> bool {
        self.ex_add.is_empty()
    }
}

/// Stage-1 output of a lazy full-precision scan over every row of the
/// original sign plane: binary inner products in the partition's own
/// SIMD/tail layout and the native lower bounds derived from them.
#[derive(Debug, Default, Clone)]
pub struct SignStage {
    pub binary_ips: Vec<f32>,
    pub lower_bounds: Vec<f32>,
}

/// A stage-1 survivor handed to [`RabitDistCalculator::accumulate_survivor_rows`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvivorRow {
    /// Row position in the [`ExRows`] the stage-2 calculator was built with.
    pub ex_index: u32,
    pub row_id: u64,
    /// Stage-1 binary inner product of the row's original sign-plane offset.
    pub binary_ip: f32,
    /// Stage-1 lower bound of the row's original sign-plane offset.
    pub lower_bound: f32,
}

/// Rows a stage-1 selection classified, in the categories of the eager
/// per-row step (see [`RabitPruneCounters`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StagePruneCounts {
    /// Accepted rows: every row without a prefilter, otherwise the rows the
    /// prefilter selects.
    pub candidates: usize,
    pub pruned_upper_bound: usize,
    pub pruned_heap: usize,
}

/// Select the stage-1 survivors of a partition in ascending offset order.
///
/// Row `i` survives iff it is accepted, `lower_bounds[i] < upper`, and
/// `lower_bounds[i] < threshold`, with the upper-bound check first as in the
/// eager per-row step. `upper` is the query's upper distance bound (none
/// means `f32::MAX`, as in the eager scan). `threshold` must be the top of a
/// full-score heap that already held `k` rows — never an estimate — and
/// `None` while the heap is not full. A NaN bound or threshold compares false
/// and keeps the row. Replaces the contents of `survivors` and `counts`.
pub fn select_full_survivors(
    lower_bounds: &[f32],
    accept: Option<&[bool]>,
    upper: Option<f32>,
    threshold: Option<f32>,
    survivors: &mut Vec<u32>,
    counts: &mut StagePruneCounts,
) {
    debug_assert!(accept.is_none_or(|accept| accept.len() == lower_bounds.len()));
    let upper = upper.unwrap_or(f32::MAX);
    survivors.clear();
    *counts = StagePruneCounts::default();
    for (offset, &lower_bound) in lower_bounds.iter().enumerate() {
        if accept.is_some_and(|accept| !accept[offset]) {
            continue;
        }
        counts.candidates += 1;
        if lower_bound >= upper {
            counts.pruned_upper_bound += 1;
        } else if threshold.is_some_and(|threshold| lower_bound >= threshold) {
            counts.pruned_heap += 1;
        } else {
            survivors.push(offset as u32);
        }
    }
}

impl RabitQuantizationStorage {
    /// Open the sign columns of a current native or layered multi-bit partition.
    ///
    /// This does not decode or allocate ex codes. The result supplies bounds
    /// through [`RabitDistCalculator::full_sign_stage_with_scratch`] and scores
    /// gathered rows through [`Self::dist_calculator_with_ex_rows`]. It must
    /// not be passed to a whole-partition distance scan.
    pub fn try_from_sign_plane_for_full(
        batch: RecordBatch,
        metadata: &RabitQuantizationMetadata,
        distance_type: DistanceType,
    ) -> Result<Self> {
        if metadata.layered {
            if batch.column_by_name(ERROR_FACTORS_COLUMN).is_none() {
                return Err(Error::invalid_input(format!(
                    "sign-plane full-precision storage requires the {ERROR_FACTORS_COLUMN} column for native lower-bound pruning"
                )));
            }
            return Self::from_batch_with_remapper(
                batch,
                metadata,
                distance_type,
                None,
                false,
                PlaneScope::SignOnlyForFull,
            );
        }
        if !(2..=9).contains(&metadata.num_bits)
            || !metadata.packed
            || metadata.query_estimator != RabitQueryEstimator::RawQuery
        {
            return Err(Error::invalid_input(format!(
                "sign preparation requires packed native multi-bit raw-query storage: num_bits={}, packed={}, estimator={:?}",
                metadata.num_bits, metadata.packed, metadata.query_estimator
            )));
        }
        let dim = metadata.rotated_dim();
        if dim == 0 || !dim.is_multiple_of(8) {
            return Err(Error::invalid_input(format!(
                "sign preparation requires a positive rotated_dim divisible by 8, got {dim}"
            )));
        }
        let row_ids = batch
            .column_by_name(ROW_ID)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .filter(|column| column.null_count() == 0)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "sign preparation requires non-null uint64 column {ROW_ID}"
                ))
            })?
            .clone();
        let codes = batch.column_by_name(RABIT_CODE_COLUMN)
            .and_then(|column| column.as_fixed_size_list_opt())
            .filter(|column| column.value_type() == DataType::UInt8
                && column.value_length() as usize == metadata.binary_code_bytes()
                && column.null_count() == 0 && column.values().null_count() == 0)
            .ok_or_else(|| Error::invalid_input(format!(
                "sign preparation requires {} non-null uint8 codes per row in {RABIT_CODE_COLUMN} for rotated_dim={dim}",
                metadata.binary_code_bytes()
            )))?.clone();
        let factors = |name: &str| {
            batch
                .column_by_name(name)
                .and_then(|column| column.as_primitive_opt::<Float32Type>())
                .filter(|column| column.null_count() == 0)
                .cloned()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "sign preparation requires non-null float32 column {name}"
                    ))
                })
        };
        let add_factors = factors(ADD_FACTORS_COLUMN)?;
        let scale_factors = factors(SCALE_FACTORS_COLUMN)?;
        let error_factors = Some(factors(ERROR_FACTORS_COLUMN)?);
        let distance_type = if distance_type == DistanceType::Cosine {
            DistanceType::L2
        } else {
            distance_type
        };
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            codes,
            add_factors,
            scale_factors,
            error_factors,
            ex_codes: None,
            packed_ex_codes: None,
            ex_add_factors: None,
            ex_scale_factors: None,
            ex_codes_lo: None,
            high_add_factors: None,
            high_scale_factors: None,
            stage1_only: true,
        })
    }

    /// Build the stage-2 calculator of a lazy full-precision scan: the same
    /// query preparation and constructor as
    /// [`VectorStore::dist_calculator_with_scratch`] (so query factors, the
    /// ex-dot kernel and the add-factor transform are bit-identical to the
    /// eager scan), scoring survivors against `ex_rows` by ex index. Pass the
    /// same `query`, `dist_q_c`, `residual` and `options` as stage 1.
    pub fn dist_calculator_with_ex_rows<'a>(
        &'a self,
        query: Arc<dyn Array>,
        dist_q_c: f32,
        residual: Option<QueryResidual<'a>>,
        f32_scratch: &'a mut Vec<f32>,
        options: DistanceCalculatorOptions,
        ex_rows: ExRows<'a>,
    ) -> Result<RabitDistCalculator<'a>> {
        if self.metadata.query_estimator != RabitQueryEstimator::RawQuery
            || self.metadata.num_bits <= 1
            || options.rq_precision != crate::vector::bq::layered::RQPrecision::Full
            || options.approx_mode == ApproxMode::Fast
        {
            return Err(Error::invalid_input(format!(
                "gathered ex rows require a full-precision gated scan, got layered={} rq_precision={:?} approx_mode={:?}",
                self.metadata.layered, options.rq_precision, options.approx_mode
            )));
        }
        let code_dim = self.code_dim();
        let (hi_row_bytes, lo_row_bytes) = if self.metadata.layered {
            let layout = crate::vector::bq::layered::RQLayout::try_new(self.metadata.num_bits)?;
            (
                blocked_ex_code_bytes(code_dim, layout.high_bits),
                blocked_ex_code_bytes(code_dim, layout.low_bits),
            )
        } else {
            (
                blocked_ex_code_bytes(code_dim, rabit_ex_bits(self.metadata.num_bits)?),
                0,
            )
        };
        let rows = ex_rows.len();
        if ex_rows.hi_row_bytes != hi_row_bytes
            || ex_rows.hi.len() != rows * hi_row_bytes
            || ex_rows.lo.is_some() != self.metadata.layered
            || ex_rows.lo.map_or(0, <[u8]>::len) != rows * lo_row_bytes
        {
            return Err(Error::invalid_input(format!(
                "gathered ex rows do not match rotated_dim={code_dim} num_bits={}: {rows} rows, high plane {} bytes at {} per row (expected {hi_row_bytes}), low plane {} bytes (expected {lo_row_bytes} per row)",
                self.metadata.num_bits,
                ex_rows.hi.len(),
                ex_rows.hi_row_bytes,
                ex_rows.lo.map_or(0, <[u8]>::len),
            )));
        }
        Ok(self.dist_calculator_with_scratch_and_ex(
            query,
            dist_q_c,
            residual,
            f32_scratch,
            options,
            Some(ex_rows),
        ))
    }
}

impl<'a> RabitDistCalculator<'a> {
    /// Binary inner products in the original partition's SIMD/tail layout.
    /// Carry these through a candidate cut so reranking does not change LUT rounding
    /// when a SIMD row becomes a scalar remainder in a smaller candidate batch.
    pub fn binary_inner_products(&self) -> Vec<f32> {
        let code_len = rabit_binary_code_bytes(self.dim);
        let n = self.codes.len() / code_len;
        let mut dists = Vec::new();
        let mut u16_scratch = Vec::new();
        let mut u8_scratch = Vec::new();
        let mut u32_scratch = Vec::new();
        self.binary_distances_with_scratch(
            n,
            code_len,
            &mut dists,
            &mut u16_scratch,
            &mut u8_scratch,
            &mut u32_scratch,
        );
        dists
    }

    /// Stage 1 of a lazy full-precision scan: the binary inner products and
    /// native lower bounds of every row of the original sign plane, computed
    /// exactly as the eager dense scan computes them (same SIMD/tail split and
    /// LUT quantization, exact fallback, and Accurate high-accuracy LUT).
    ///
    /// Returns `Err(reason)` without touching `out` when this calculator
    /// cannot gate on the lower bound (the eager scan then scores every row,
    /// so the caller must take the eager path for the partition).
    pub fn full_sign_stage_with_scratch(
        &self,
        out: &mut SignStage,
        quantized_dists: &mut Vec<u16>,
        quantized_dists_table: &mut Vec<u8>,
        hacc_quantized_dists: &mut Vec<u32>,
    ) -> std::result::Result<(), &'static str> {
        if let Some(reason) = self.raw_query_lower_bound_gating_disabled_reason() {
            return Err(reason);
        }
        let Some(error_factors) = self.pruning_errors() else {
            return Err("missing_error_factors");
        };
        let code_len = rabit_binary_code_bytes(self.dim);
        let n = self.codes.len() / code_len;
        out.lower_bounds.clear();
        if n == 0 {
            out.binary_ips.clear();
            return Ok(());
        }
        self.binary_distances_with_scratch(
            n,
            code_len,
            &mut out.binary_ips,
            quantized_dists,
            quantized_dists_table,
            hacc_quantized_dists,
        );
        // Same expression as `raw_query_lower_bound`, so the survivors'
        // bounds match the eager scan's (and its prune masks) bit for bit.
        out.lower_bounds
            .extend(out.binary_ips.iter().enumerate().map(|(id, &binary_ip)| {
                self.raw_query_binary_distance(id, binary_ip) - error_factors[id] * self.query_error
            }));
        Ok(())
    }
    /// The loop-invariant exact-rerank inputs shared by the eager top-k scans
    /// and the lazy survivor rerank.
    pub(super) fn raw_query_ex_context(
        &self,
        n: usize,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
    ) -> RawQueryTopkContext<'_> {
        RawQueryTopkContext {
            n,
            k,
            ex_bits: self.num_bits - 1,
            ex_codes: self
                .ex_codes
                .expect("raw-query multi-bit RQ requires ex codes"),
            ex_add_factors: self
                .ex_add_factors
                .expect("raw-query multi-bit RQ requires ex add factors"),
            ex_scale_factors: self
                .ex_scale_factors
                .expect("raw-query multi-bit RQ requires ex scale factors"),
            query_lower_bound: lower_bound.unwrap_or(f32::MIN),
            query_upper_bound: upper_bound.unwrap_or(f32::MAX),
        }
    }

    /// Stage 2 of a lazy full-precision scan for one partition. `rows` are
    /// the partition's stage-1 survivors in ascending sign-plane offset order;
    /// each goes through the eager scan's per-row step (bound checks against
    /// the live heap, exact rerank, range check, heap update) with its
    /// stage-1 inner product and bound. `stage1` carries the rows stage 1
    /// already pruned so the partition reports the same prune counters as the
    /// eager scan; they are returned together with the stage-2 tallies.
    ///
    /// `on_first_full` receives the heap top once, when this partition first
    /// fills the heap to `k` rows.
    ///
    /// The calculator must come from
    /// [`RabitQuantizationStorage::dist_calculator_with_ex_rows`] with the
    /// rows the survivors' `ex_index` refers to.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_survivor_rows(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        rows: &[SurvivorRow],
        stage1: StagePruneCounts,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        mut on_first_full: impl FnMut(f32),
    ) -> RabitPruneCounters {
        let mut counters = RabitPruneCounters {
            candidates: stage1.candidates,
            pruned_upper_bound: stage1.pruned_upper_bound,
            pruned_heap: stage1.pruned_heap,
            ..Default::default()
        };
        let n = self.codes.len() / rabit_binary_code_bytes(self.dim);
        // The eager scans return before reporting for these, too.
        if k == 0 || n == 0 {
            return counters;
        }
        let ctx = self.raw_query_ex_context(n, k, lower_bound, upper_bound);
        let mut max_dist = res.peek().map(|node| node.dist);
        let mut is_full = res.len() >= k;
        for row in rows {
            self.accumulate_raw_query_multi_bit_row(
                &ctx,
                row.ex_index as usize,
                row.row_id,
                row.binary_ip,
                row.lower_bound,
                res,
                &mut max_dist,
                &mut counters,
            );
            if !is_full && res.len() >= k {
                is_full = true;
                if let Some(top) = res.peek() {
                    on_first_full(top.dist.0);
                }
            }
        }
        record_rabit_prune_stats(&counters);
        counters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use rstest::rstest;
    #[rstest]
    #[case::valid_slice(7, 48, false, false, None)]
    #[case::one_bit(1, 48, false, false, Some("greater than 1"))]
    #[case::wrong_width(7, 40, false, false, Some("48 non-null uint8 codes"))]
    #[case::null_code(7, 48, true, false, Some("48 non-null uint8 codes"))]
    #[case::null_factor(7, 48, false, true, Some("non-null __add_factors_ex"))]
    fn test_native_ex_rows_validate_sliced_batch(
        #[case] num_bits: u8,
        #[case] width: i32,
        #[case] null_code: bool,
        #[case] null_factor: bool,
        #[case] error_text: Option<&str>,
    ) {
        let codes = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(
                (0..width * 3)
                    .map(|i| (!(null_code && i == width)).then_some(i as u8))
                    .collect::<Vec<_>>(),
            ),
            width,
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter([
            (RABIT_BLOCKED_EX_CODE_COLUMN, Arc::new(codes) as ArrayRef),
            (
                EX_ADD_FACTORS_COLUMN,
                Arc::new(Float32Array::from(vec![
                    Some(1.0),
                    (!null_factor).then_some(2.0),
                    Some(3.0),
                ])),
            ),
            (
                EX_SCALE_FACTORS_COLUMN,
                Arc::new(Float32Array::from(vec![4.0, 5.0, 6.0])),
            ),
        ])
        .unwrap()
        .slice(1, 1);
        let result = ExRows::from_native_batch(&batch, 64, num_bits);
        if let Some(error_text) = error_text {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains(error_text), "{error}");
        } else {
            let rows = result.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows.hi, (48..96).collect::<Vec<u8>>());
            assert_eq!(rows.ex_add, &[2.0]);
            assert_eq!(rows.ex_scale, &[5.0]);
        }
    }
}
