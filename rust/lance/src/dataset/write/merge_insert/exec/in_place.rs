// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, UInt8Array, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::{
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{
        DisplayAs, ExecutionPlan, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::{StreamExt, stream};
use lance_core::{ROW_ADDR, ROW_ID};

use crate::Dataset;
use crate::dataset::transaction::UpdateMode::RewriteColumns;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::write::merge_insert::assign_action::Action;
use crate::dataset::write::merge_insert::{
    MERGE_ACTION_COLUMN, MERGE_SOURCE_SENTINEL, MergeInsertJob, MergeInsertParams, MergeStats,
    PatchedFragments, SourceDedupeBehavior, create_duplicate_row_error, resolve_target_bases,
};

use super::MergeInsertMetrics;

/// Patches the source columns into the existing fragments instead of rewriting
/// whole rows.
///
/// This is the v2 counterpart of the legacy in-place write path: the columns
/// present in the source are written as new data files attached to the
/// fragments that already hold the matched rows, and the old versions of those
/// columns are tombstoned. Columns absent from the source are never read or
/// written, which is what makes a narrow update of a wide table cheap.
///
/// Compared to [`super::FullSchemaMergeInsertExec`] this node:
/// - consumes only the source data columns plus `_rowaddr` / `_rowid` /
///   `__action` (the target's other columns never enter the plan, so the
///   target scan does not read them either)
/// - produces no deletion vectors and keeps fragment ids stable
/// - commits [`Operation::Update`] with [`RewriteColumns`]
///
/// Row placement is resolved by [`MergeInsertJob::update_fragments`], which
/// already sorts by row address, groups by fragment, and fills the rows a
/// fragment did not have an update for. Reusing it keeps a single in-place
/// column-write implementation rather than adding a second one.
#[derive(Debug)]
pub struct InPlaceMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    /// Duplicates the source stream dropped before the join, in `FirstSeen`
    /// mode. Counted there rather than here, so it has to be folded into the
    /// stats this node reports.
    source_skipped_duplicates: Arc<AtomicU64>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    merge_stats: Arc<Mutex<Option<MergeStats>>>,
    transaction: Arc<Mutex<Option<Transaction>>>,
}

impl InPlaceMergeInsertExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
        source_skipped_duplicates: Arc<AtomicU64>,
    ) -> DFResult<Self> {
        let empty_schema = Arc::new(Schema::empty());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(empty_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));

        Ok(Self {
            input,
            dataset,
            params,
            source_skipped_duplicates,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            merge_stats: Arc::new(Mutex::new(None)),
            transaction: Arc::new(Mutex::new(None)),
        })
    }

    /// Takes the merge statistics if the execution has completed.
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge_stats
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Takes the transaction if the execution has completed.
    pub fn transaction(&self) -> Option<Transaction> {
        self.transaction
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Locates the control columns and the source data columns in the input.
    ///
    /// The output stream carries `_rowaddr` followed by the data columns, which
    /// is the schema [`MergeInsertJob::update_fragments`] expects. `_rowid` is
    /// read for duplicate detection but not forwarded. Data columns are ordered
    /// by the dataset schema so the written file layout does not depend on the
    /// order the source happened to provide.
    fn prepare_stream_schema(
        &self,
        input_schema: &SchemaRef,
    ) -> DFResult<(usize, usize, usize, Vec<usize>, SchemaRef)> {
        let index_of = |name: &str| {
            input_schema
                .column_with_name(name)
                .map(|(i, _)| i)
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "Expected {name} column in in-place merge insert input"
                    ))
                })
        };
        let rowaddr_idx = index_of(ROW_ADDR)?;
        let rowid_idx = index_of(ROW_ID)?;
        let action_idx = index_of(MERGE_ACTION_COLUMN)?;

        let mut by_name = std::collections::HashMap::new();
        for (idx, field) in input_schema.fields().iter().enumerate() {
            if idx == rowaddr_idx || idx == rowid_idx || idx == action_idx {
                continue;
            }
            let name = field.name().as_str();
            if name == ROW_ADDR
                || name == ROW_ID
                || name == MERGE_ACTION_COLUMN
                || name == MERGE_SOURCE_SENTINEL
            {
                continue;
            }
            by_name.insert(name, idx);
        }

        let mut data_column_indices = Vec::with_capacity(by_name.len());
        // `_rowaddr` is nullable because that is the schema
        // `update_fragments` expects; this node only ever emits non-null
        // addresses (see the `Action::UpdateAll` arm in `create_patch_stream`).
        let mut output_fields = vec![Arc::new(arrow_schema::Field::new(
            ROW_ADDR,
            arrow_schema::DataType::UInt64,
            true,
        ))];
        for dataset_field in self.dataset.schema().fields.iter() {
            if let Some(idx) = by_name.remove(dataset_field.name.as_str()) {
                data_column_indices.push(idx);
                output_fields.push(Arc::new(input_schema.field(idx).clone()));
            }
        }

        if !by_name.is_empty() {
            let mut unknown: Vec<&str> = by_name.into_keys().collect();
            unknown.sort_unstable();
            return Err(DataFusionError::Internal(format!(
                "In-place merge insert input carries column(s) {unknown:?} that are not \
                 dataset fields"
            )));
        }
        if data_column_indices.is_empty() {
            return Err(DataFusionError::Internal(
                "No data columns found in in-place merge insert input".to_string(),
            ));
        }

        Ok((
            rowaddr_idx,
            rowid_idx,
            action_idx,
            data_column_indices,
            Arc::new(Schema::new(output_fields)),
        ))
    }

    /// Drops the rows that must not be written and projects the rest down to
    /// `_rowaddr` + data columns.
    fn create_patch_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        metrics: &MergeInsertMetrics,
    ) -> DFResult<SendableRecordBatchStream> {
        let (rowaddr_idx, rowid_idx, action_idx, data_column_indices, output_schema) =
            self.prepare_stream_schema(&input_stream.schema())?;

        let dedupe = self.params.source_dedupe_behavior;
        let on_columns = self.params.on.clone();
        let updated_rows = metrics.num_updated_rows.clone();
        let skipped_duplicates = metrics.num_skipped_duplicates.clone();
        let mut seen_row_ids = HashSet::new();

        let schema = output_schema.clone();
        let stream = input_stream.map(move |batch_result| -> DFResult<RecordBatch> {
            let batch = batch_result?;
            let row_addrs = downcast_u64(&batch, rowaddr_idx, ROW_ADDR)?;
            let row_ids = downcast_u64(&batch, rowid_idx, ROW_ID)?;
            let actions = batch
                .column(action_idx)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "Expected UInt8Array for {MERGE_ACTION_COLUMN} column"
                    ))
                })?;

            let mut keep_rows: Vec<u32> = Vec::with_capacity(batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                let action = Action::try_from(actions.value(row_idx)).map_err(|e| {
                    DataFusionError::Internal(format!(
                        "Invalid action code {}: {}",
                        actions.value(row_idx),
                        e
                    ))
                })?;
                match action {
                    Action::UpdateAll => {
                        if row_addrs.is_null(row_idx) {
                            return Err(DataFusionError::Internal(
                                "In-place merge insert produced an update without a row address"
                                    .to_string(),
                            ));
                        }
                        if !seen_row_ids.insert(row_ids.value(row_idx)) {
                            match dedupe {
                                SourceDedupeBehavior::Fail => {
                                    return Err(create_duplicate_row_error(
                                        &batch,
                                        row_idx,
                                        &on_columns,
                                    ));
                                }
                                SourceDedupeBehavior::FirstSeen => {
                                    skipped_duplicates.add(1);
                                    continue;
                                }
                            }
                        }
                        updated_rows.add(1);
                        keep_rows.push(row_idx as u32);
                    }
                    // Rows the update condition rejected: the target keeps its
                    // current values, so nothing is written for them. They still
                    // claim their target row, so a second source row matching the
                    // same target is caught by `source_dedupe_behavior` even when
                    // the condition excludes one of them.
                    Action::MatchedNoOp => {
                        if !row_addrs.is_null(row_idx)
                            && !seen_row_ids.insert(row_ids.value(row_idx))
                        {
                            match dedupe {
                                SourceDedupeBehavior::Fail => {
                                    return Err(create_duplicate_row_error(
                                        &batch,
                                        row_idx,
                                        &on_columns,
                                    ));
                                }
                                SourceDedupeBehavior::FirstSeen => {
                                    skipped_duplicates.add(1);
                                }
                            }
                        }
                    }
                    // Rows that matched no target row.
                    Action::Nothing => {}
                    Action::Fail => {
                        return Err(DataFusionError::Execution(format!(
                            "Merge insert failed: found matching row with key values: {}",
                            crate::dataset::write::merge_insert::format_key_values_on_columns(
                                &batch,
                                row_idx,
                                &on_columns
                            )
                        )));
                    }
                    // Eligibility keeps inserts and deletes off this path, so
                    // reaching here means the routing and this node disagree.
                    Action::Insert | Action::Delete => {
                        return Err(DataFusionError::Internal(format!(
                            "In-place merge insert cannot handle action {action:?}"
                        )));
                    }
                }
            }

            project_kept_rows(
                &batch,
                keep_rows,
                rowaddr_idx,
                &data_column_indices,
                schema.clone(),
            )
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }
}

fn downcast_u64<'a>(batch: &'a RecordBatch, idx: usize, name: &str) -> DFResult<&'a UInt64Array> {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal(format!("Expected UInt64Array for {name} column")))
}

fn project_kept_rows(
    batch: &RecordBatch,
    keep_rows: Vec<u32>,
    rowaddr_idx: usize,
    data_column_indices: &[usize],
    output_schema: SchemaRef,
) -> DFResult<RecordBatch> {
    let mut source_indices = Vec::with_capacity(data_column_indices.len() + 1);
    source_indices.push(rowaddr_idx);
    source_indices.extend_from_slice(data_column_indices);

    if keep_rows.is_empty() {
        let empty = output_schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_empty_array(field.data_type()))
            .collect::<Vec<_>>();
        return RecordBatch::try_new(output_schema, empty).map_err(DataFusionError::from);
    }

    let indices = arrow_array::UInt32Array::from(keep_rows);
    let taken = arrow_select::take::take_record_batch(batch, &indices)?;
    let columns = source_indices
        .iter()
        .map(|&idx| taken.column(idx).clone())
        .collect::<Vec<_>>();
    RecordBatch::try_new(output_schema, columns).map_err(DataFusionError::from)
}

impl DisplayAs for InPlaceMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                let when_matched = match &self.params.when_matched {
                    crate::dataset::WhenMatched::UpdateAll => "UpdateAll".to_string(),
                    crate::dataset::WhenMatched::UpdateIf(condition) => {
                        format!("UpdateIf({})", condition)
                    }
                    crate::dataset::WhenMatched::UpdateIfExpr(expr) => {
                        format!("UpdateIf({})", expr.human_display())
                    }
                    other => format!("{:?}", other),
                };
                write!(
                    f,
                    "InPlaceMergeInsert: on=[{}], when_matched={}, mode=RewriteColumns",
                    self.params.on.join(", "),
                    when_matched
                )
            }
            datafusion::physical_plan::DisplayFormatType::TreeRender => {
                write!(f, "InPlaceMergeInsert[{}]", self.dataset.uri())
            }
        }
    }
}

impl ExecutionPlan for InPlaceMergeInsertExec {
    fn name(&self) -> &str {
        "InPlaceMergeInsertExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::empty())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "InPlaceMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            source_skipped_duplicates: self.source_skipped_duplicates.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
            merge_stats: self.merge_stats.clone(),
            transaction: self.transaction.clone(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn required_input_distribution(&self) -> Vec<datafusion_physical_expr::Distribution> {
        vec![datafusion_physical_expr::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let metrics = MergeInsertMetrics::new(&self.metrics, partition);

        let input_stream = self.input.execute(partition, context)?;
        let patch_stream = self.create_patch_stream(input_stream, &metrics)?;

        let dataset = self.dataset.clone();
        let params = self.params.clone();
        let merge_stats_holder = self.merge_stats.clone();
        let transaction_holder = self.transaction.clone();
        let compacted_sstables = self.params.compacted_sstables.clone();
        let source_skipped_duplicates = self.source_skipped_duplicates.clone();

        let result_stream = stream::once(async move {
            let target_bases_info = resolve_target_bases(&dataset, &params).await?;
            // A guess: a compatible transaction can commit before this one, in
            // which case the real commit version is later. `matched_offsets`
            // below is what lets `build_manifest` correct the stamp.
            let current_version = dataset.manifest.version + 1;
            let PatchedFragments {
                updated_fragments,
                new_fragments,
                fields_modified,
                matched_offsets,
            } = MergeInsertJob::update_fragments(
                dataset.clone(),
                patch_stream,
                current_version,
                target_bases_info,
            )
            .await?;

            // Eligibility forbids inserts and the join is therefore an inner
            // join, so every row carries a target address and no row is routed
            // to a new fragment.
            debug_assert!(
                new_fragments.is_empty(),
                "in-place merge insert produced {} new fragment(s)",
                new_fragments.len()
            );

            // Only the files this operation wrote count toward the metrics: an
            // updated fragment keeps its pre-existing data files and carries the
            // patch as the last one.
            for fragment in &updated_fragments {
                if let Some(data_file) = fragment.files.last()
                    && let Some(size) = data_file.file_size_bytes.get()
                {
                    metrics.bytes_written.add(u64::from(size) as usize);
                }
                metrics.num_files_written.add(1);
            }

            let operation = Operation::Update {
                removed_fragment_ids: Vec::new(),
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                // In-place patches leave every row where it was, so no index's
                // fragment bitmap needs extending.
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(RewriteColumns),
                inserted_rows_filter: None,
                // Which rows were patched, so `build_manifest` re-stamps their
                // `_row_last_updated_at_version` with the version this commit
                // actually lands on rather than the one guessed above.
                updated_fragment_offsets: Some(matched_offsets),
            };
            let transaction = Transaction::new(dataset.manifest.version, operation, None);

            if let Ok(mut guard) = transaction_holder.lock() {
                guard.replace(transaction);
            }
            // `FirstSeen` drops duplicate source rows before the join, so fold
            // that count in — this node only sees what survived.
            let mut stats = MergeStats::from(&metrics);
            stats.num_skipped_duplicates = stats
                .num_skipped_duplicates
                .checked_add(source_skipped_duplicates.load(Ordering::Relaxed))
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "merge insert skipped duplicate count overflowed u64".to_string(),
                    )
                })?;
            if let Ok(mut guard) = merge_stats_holder.lock() {
                guard.replace(stats);
            }

            Ok(RecordBatch::new_empty(Arc::new(Schema::empty())))
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(Schema::empty()),
            result_stream,
        )))
    }
}
