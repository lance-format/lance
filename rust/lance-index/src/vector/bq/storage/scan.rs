// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Native RaBitQ sign bounds and scoring of separately gathered ex rows.
//! These primitives preserve the existing full scan's LUT, kernel and heap
//! semantics. Query scheduling and I/O selection belong to the caller.

use super::*;

/// Borrowed blocked native ex codes and their full-score factors.
#[derive(Debug, Clone, Copy)]
pub struct ExRows<'a> {
    codes: &'a [u8],
    row_bytes: usize,
    ex_add: &'a [f32],
    ex_scale: &'a [f32],
}

impl<'a> ExRows<'a> {
    /// Borrow a projected or gathered native ex batch without repacking.
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
            codes: codes.values().as_primitive::<UInt8Type>().values(),
            row_bytes,
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
    /// Open the sign columns of a current native multi-bit partition.
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
        })
    }

    /// Prepare the existing native full-score kernel with gathered ex rows.
    /// Query preparation is shared with [`VectorStore::dist_calculator_with_scratch`].
    /// The caller must retain each row's original sign inner product and row id.
    pub fn dist_calculator_with_ex_rows<'a>(
        &'a self,
        query: Arc<dyn Array>,
        dist_q_c: f32,
        residual: Option<QueryResidual<'a>>,
        f32_scratch: &'a mut Vec<f32>,
        options: DistanceCalculatorOptions,
        ex_rows: ExRows<'a>,
    ) -> Result<RabitDistCalculator<'a>> {
        if options.approx_mode == ApproxMode::Fast
            || self.metadata.query_estimator != RabitQueryEstimator::RawQuery
            || self.metadata.num_bits <= 1
        {
            return Err(Error::invalid_input(format!(
                "gathered ex rows require a gated native multi-bit scan: num_bits={}, estimator={:?}, approx_mode={:?}",
                self.metadata.num_bits, self.metadata.query_estimator, options.approx_mode
            )));
        }
        let expected = blocked_ex_code_bytes(self.code_dim(), self.metadata.num_bits - 1);
        if ex_rows.row_bytes != expected || ex_rows.codes.len() != ex_rows.len() * expected {
            return Err(Error::invalid_input(format!(
                "gathered ex row width {} does not match rotated_dim={} num_bits={} expected {expected}",
                ex_rows.row_bytes,
                self.code_dim(),
                self.metadata.num_bits
            )));
        }
        let mut calc =
            self.dist_calculator_with_scratch(query, dist_q_c, residual, f32_scratch, options);
        calc.ex_codes = Some(ex_rows.codes);
        calc.ex_code_len = ex_rows.row_bytes;
        calc.ex_add_factors = Some(ex_rows.ex_add);
        calc.ex_scale_factors = Some(ex_rows.ex_scale);
        calc.packed_ex_codes = None;
        Ok(calc)
    }
}

impl<'a> RabitDistCalculator<'a> {
    /// Calculate the original partition's binary inner products with the
    /// same SIMD blocks, tail and LUT precision as the full scan.
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

    /// Compute sign inner products and native lower bounds in partition order.
    /// Returns the full scan's gating reason without changing `out` when
    /// Fast mode, the estimator or missing error factors prevent pruning.
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
        let Some(error_factors) = self.error_factors else {
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
    fn raw_query_ex_context(
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

    /// Score survivors against the live full-score heap in original row order.
    /// `rows` must reference the ex batch supplied to
    /// [`RabitQuantizationStorage::dist_calculator_with_ex_rows`]. A threshold
    /// used to select them may be stale, but must never be tighter than that
    /// of a previously full heap. `on_first_full` receives the first heap top.
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
            assert_eq!(rows.codes, (48..96).collect::<Vec<u8>>());
            assert_eq!(rows.ex_add, &[2.0]);
            assert_eq!(rows.ex_scale, &[5.0]);
        }
    }
}
