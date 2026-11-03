// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction task execution and rewrite result construction.
//!
//! A task only writes replacement fragments and returns [`RewriteResult`];
//! publication is exclusively a commit responsibility. Tests live in
//! `optimize/tests/execution.rs` and `optimize/tests/lineage.rs`.

use std::borrow::Cow;
use std::ops::AddAssign;
use std::sync::Arc;

use super::binary_copy;
use super::planning::{
    CompactionMode, CompactionOptions, CompactionPlan, TaskData, has_eager_remappable_index,
};
use crate::Dataset;
use crate::dataset::blob::{BlobV2BatchRewritePlan, SidecarRewrite};
use crate::dataset::index::load_indices_for_remapping;
use crate::dataset::rowids::{
    RowLineage, RowLineagePlan, RowLineageSpill, RowVersionKind, inline_row_lineage_max_bytes,
    load_row_id_sequences, load_row_version_sequence, place_row_lineage, plan_row_lineage_spill,
};
use crate::dataset::utils::{CapturedRowIds, make_rowid_capture_stream};
use crate::dataset::versions;
use crate::dataset::{
    WriteMode, WriteParams, cleanup_data_fragments, write::write_fragments_internal_impl,
};
use crate::io::commit::migrate_fragments;
use crate::io::deletion::read_dataset_deletion_file;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::{BlobHandling, Field as LanceField};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_table::feature_flags::managed_blobs_enabled;
use lance_table::format::{Fragment, RowDatasetVersionSequence};
use lance_table::rowids::RowIdSequence;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

/// Metrics returned by [`crate::dataset::optimize::compact_files`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// The number of fragments that have been overwritten.
    pub fragments_removed: usize,
    /// The number of new fragments that have been added.
    pub fragments_added: usize,
    /// The number of files that have been removed, including deletion files.
    pub files_removed: usize,
    /// The number of files that have been added, which is always equal to the
    /// number of fragments.
    pub files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Determine if page-level binary copy can safely merge the provided fragments.
///
/// Preconditions:
/// - Compaction mode is not `Reencode`
/// - Dataset storage format is non-legacy
/// - Fragment list is non-empty and every fragment has at least one data file
/// - All data files share identical Lance file versions
/// - No fragment has a deletion file
/// - No fragment has data overlays
/// - All data files use an identical schema mapping (`fields`, `column_indices`) in dataset schema
///   order, with the current schema's expected physical-column count
/// - Input data files must not contain extra global buffers (beyond schema / file descriptor)
pub(super) async fn can_use_binary_copy(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> bool {
    let version = options.write_version(dataset);
    versions::can_use_binary_copy(version, dataset, options, fragments)
        .await
        .unwrap_or_else(|err| {
            log::warn!("Binary copy disabled due to error: {}", err);
            false
        })
}

fn first_binary_copy_version_mismatch(
    fragments: &[Fragment],
    target: ConcreteFileVersion,
) -> Result<Option<(String, ConcreteFileVersion)>> {
    for fragment in fragments {
        for data_file in fragment.referenced_lance_files() {
            let actual = data_file.file_version()?;
            if actual != target {
                return Ok(Some((data_file.path.clone(), actual)));
            }
        }
    }
    Ok(None)
}

pub async fn can_use_binary_copy_current(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> Result<bool> {
    use lance_file::reader::FileReader as LFReader;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

    if matches!(options.compaction_mode(), CompactionMode::Reencode) {
        log::debug!("Binary copy disabled: compaction mode is Reencode");
        return Ok(false);
    }

    let has_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob());
    if has_blob_columns {
        log::debug!("Binary copy disabled: dataset contains blob columns");
        return Ok(false);
    }

    if fragments.is_empty() {
        log::debug!("Binary copy disabled: no fragments to compact");
        return Ok(false);
    }

    if let Some(fragment) = fragments.iter().find(|fragment| fragment.files.is_empty()) {
        log::debug!(
            "Binary copy disabled: fragment {} has no data files",
            fragment.id
        );
        return Ok(false);
    }
    let ref_fields = &fragments[0].files[0].fields;
    let ref_cols = &fragments[0].files[0].column_indices;
    let version = dataset.manifest.data_storage_format.lance_file_format();
    let (schema_fields, schema_column_indices) =
        lance_file::versions::data_file_columns(version, dataset.schema());
    if ref_fields.as_ref() != schema_fields.as_slice()
        || ref_cols.as_ref() != schema_column_indices.as_slice()
    {
        log::debug!(
            "Binary copy disabled: data files do not use the dataset schema's physical column order"
        );
        return Ok(false);
    }
    let expected_column_count: usize = dataset
        .schema()
        .fields
        .iter()
        .map(|field| lance_file::versions::physical_column_count(version, field))
        .sum();
    for fragment in fragments {
        // Binary copy only reads base files; overlays must be materialized by the scanner.
        if !fragment.overlays.is_empty() {
            log::debug!(
                "Binary copy disabled: fragment {} has data overlays",
                fragment.id
            );
            return Ok(false);
        }
        if fragment.deletion_file.is_some() {
            log::debug!(
                "Binary copy disabled: fragment {} has a deletion file",
                fragment.id
            );
            return Ok(false);
        }

        for data_file in &fragment.files {
            if data_file.fields != *ref_fields || data_file.column_indices != *ref_cols {
                return Ok(false);
            }

            // check file global buffer
            let object_store = match data_file.base_id {
                Some(base_id) => dataset.object_store(Some(base_id)).await?,
                None => dataset.object_store.clone(),
            };
            let full_path = dataset
                .data_file_dir(data_file)?
                .clone()
                .join(data_file.path.as_str());
            let scan_scheduler = ScanScheduler::new(
                object_store.clone(),
                SchedulerConfig::max_bandwidth(&object_store),
            );
            let file_scheduler = scan_scheduler
                .open_file_with_priority(&full_path, 0, &data_file.file_size_bytes)
                .await?;
            let file_meta = LFReader::read_all_metadata(&file_scheduler).await?;
            if file_meta.column_infos.len() != expected_column_count {
                log::debug!(
                    "Binary copy disabled: data file '{}' in fragment {} has {} physical columns, expected {} for the current schema",
                    data_file.path,
                    fragment.id,
                    file_meta.column_infos.len(),
                    expected_column_count
                );
                return Ok(false);
            }
            // Binary copy only preserves page and column-buffer bytes; the
            // output footer (including global buffers) is re-generated. Extra
            // global buffers are allowed only when every one of them is a
            // recognized zone write seed, which the copy then carries or
            // drops by explicit per-column rules. Anything unrecognized keeps
            // disqualifying the file, so no unknown buffer is ever silently
            // dropped.
            if file_meta.file_buffers.len() > 1
                && let Err(e) = binary_copy::seed_carry::extract_source_seeds(
                    &file_meta,
                    &file_scheduler,
                    data_file,
                )
                .await
            {
                log::debug!(
                    "Binary copy disabled: extra global buffers cannot be handled safely: {}",
                    e
                );
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Build a scan reader for rewrite and optionally capture row IDs.
///
/// Parameters:
/// - `dataset`: Dataset handle used to create the scanner.
/// - `fragments`: When `with_frags` is true, restrict the scan to these old fragments
///   and preserve insertion order.
/// - `batch_size`: Optional batch size; if provided, set it on the scanner to control
///   read batching.
/// - `io_buffer_size`: Optional I/O buffer size in bytes; if provided, set it on the
///   scanner to control how much data is queued during reads.
/// - `with_frags`: Whether to scan only the specified old fragments and force
///   in-order reading.
/// - `capture_row_ids`: When index remapping is needed, include and capture the
///   `_rowid` column from the stream.
///
/// Returns:
/// - `SendableRecordBatchStream`: The batch stream (with `_rowid` removed if captured)
///   to feed the rewrite path.
/// - `Option<Receiver<CapturedRowIds>>`: A receiver to obtain captured row IDs after the
///   stream completes; `None` if not capturing.
/// - `bool`: Whether the dataset has blob v2 columns and the stream includes `_rowaddr`.
async fn prepare_reader(
    dataset: &Dataset,
    fragments: &[Fragment],
    batch_size: Option<usize>,
    io_buffer_size: Option<u64>,
    with_frags: bool,
    capture_row_ids: bool,
) -> Result<(
    SendableRecordBatchStream,
    Option<std::sync::mpsc::Receiver<CapturedRowIds>>,
    bool,
)> {
    let mut scanner = dataset.scan();
    let has_legacy_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob() && !field.is_blob_v2());
    if has_legacy_blob_columns {
        scanner.blob_handling(BlobHandling::AllBinary);
    }
    let has_blob_v2_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob_v2());
    if has_blob_v2_columns {
        scanner.with_row_address();
    }
    if let Some(bs) = batch_size {
        scanner.batch_size(bs);
    }
    if let Some(io_buffer_size) = io_buffer_size {
        scanner.io_buffer_size(io_buffer_size);
    }
    if with_frags {
        scanner
            .with_fragments(fragments.to_vec())
            .scan_in_order(true);
    }
    if capture_row_ids {
        scanner.with_row_id();
        let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
        let (data_no_row_ids, rx) =
            make_rowid_capture_stream(data, dataset.manifest.uses_stable_row_ids())?;
        Ok((data_no_row_ids, Some(rx), has_blob_v2_columns))
    } else {
        Ok((
            SendableRecordBatchStream::from(scanner.try_into_stream().await?),
            None,
            has_blob_v2_columns,
        ))
    }
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result.
    ///
    /// This result should be later passed to
    /// [`crate::dataset::optimize::commit_compaction`] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let dataset = if dataset.manifest.version == self.read_version {
            Cow::Borrowed(dataset)
        } else {
            Cow::Owned(dataset.checkout_version(self.read_version).await?)
        };
        rewrite_files(dataset, self.task.clone(), &self.options).await
    }
}

impl CompactionPlan {
    /// Retrieve standalone tasks that be be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }
}

/// The result of a single compaction task.
///
/// This should be passed to [`crate::dataset::optimize::commit_compaction`] to
/// commit the operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
    /// Serialized `RoaringTreemap` of the row addresses from the original
    /// fragments that were read during compaction.
    ///
    /// - `None` when nothing consumes them; with stable row IDs, whenever
    ///   neither `defer_index_remap` nor an address-domain index applies.
    /// - `Some` then these addresses are either (1) written to storage for
    ///   deferred index remap post-processing, or (2) used with reserved
    ///   fragment IDs to build old-to-new mappings.
    pub row_addrs: Option<Vec<u8>>,
}

/// The old row addresses of `fragments`' live (non-deleted) rows, reconstructed
/// from fragment metadata and deletion files alone -- no data scan. Callers
/// (the deferred/FRI path and the eager address-domain remap path) pair these
/// addresses, sorted, with the rewritten rows, which keep task order, so the
/// task's fragments must be in ascending id order.
pub(super) async fn live_row_addrs(
    dataset: &Dataset,
    fragments: &[Fragment],
    new_fragments: &[Fragment],
) -> Result<RoaringTreemap> {
    if let Some(pair) = fragments.windows(2).find(|pair| pair[0].id >= pair[1].id) {
        return Err(Error::invalid_input(format!(
            "Cannot record the row-address moves of a compaction task whose fragments are not \
             in ascending id order: fragment {} precedes fragment {}",
            pair[0].id, pair[1].id
        )));
    }
    let addrs = live_source_row_addrs(dataset, fragments).await?;
    let rewritten_rows = new_fragments
        .iter()
        .map(|fragment| fragment.physical_rows.unwrap_or_default() as u64)
        .sum::<u64>();
    if addrs.len() != rewritten_rows {
        return Err(Error::internal(format!(
            "Cannot record the row-address moves of compacting fragments {:?}: their metadata \
             and deletion files leave {} live rows, but the rewrite wrote {rewritten_rows}",
            fragments
                .iter()
                .map(|fragment| fragment.id)
                .collect::<Vec<_>>(),
            addrs.len(),
        )));
    }
    Ok(addrs)
}

/// Every live row address of `fragments`: their physical rows minus deletions,
/// read from fragment metadata and deletion files.
pub async fn live_source_row_addrs(
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<RoaringTreemap> {
    let mut addrs = RoaringTreemap::new();
    for fragment in fragments {
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::internal(format!(
                "Fragment id {} is outside the row-address range",
                fragment.id
            ))
        })?;
        let physical_rows = fragment.physical_rows.ok_or_else(|| {
            Error::internal(format!(
                "Fragment {} is missing physical_rows after migration",
                fragment.id
            ))
        })?;
        let start = u64::from(RowAddress::first_row(fragment_id));
        addrs.insert_range(start..start + physical_rows as u64);
        if let Some(deletion_file) = &fragment.deletion_file {
            let deletions = read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
            for offset in deletions.to_sorted_iter() {
                addrs.remove(u64::from(RowAddress::new_from_parts(fragment_id, offset)));
            }
        }
    }
    Ok(addrs)
}

/// Refuse a task whose output clustering id can never resolve, before the
/// task rewrites anything: an id at or below the manifest's
/// `max_clustering_id` that the manifest no longer holds, since definition ids
/// are never reused. An id above the watermark may have been allocated after
/// this snapshot was read and is left to the commit, which checks every id
/// against the landing manifest; that check is the one that counts, this one
/// only saves a planner that holds a stale id from rewriting every task's
/// data to have the result refused.
pub(super) fn refuse_stale_output_clustering(
    manifest: &lance_table::format::Manifest,
    task: &TaskData,
) -> Result<()> {
    let Some(id) = task.output_clustering_id else {
        return Ok(());
    };
    if id > manifest.max_clustering_id
        || manifest
            .clustering
            .iter()
            .any(|definition| definition.id == id)
    {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "compaction task outputs would carry clustering definition {id}, which the table \
         no longer defines"
    )))
}

/// Rewrite the files in a single task.
///
/// This assumes that the dataset is the correct read version to be compacted.
pub(super) async fn rewrite_files(
    dataset: Cow<'_, Dataset>,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();
    let write_version = options.write_version(dataset.as_ref());
    versions::validate_write_version(
        dataset.manifest.data_storage_format.lance_file_format(),
        write_version,
    )?;
    refuse_stale_output_clustering(&dataset.manifest, &task)?;

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            read_version: dataset.manifest.version,
            original_fragments: task.fragments,
            row_addrs: None,
        });
    }

    let previous_writer_version = &dataset.manifest.writer_version;
    // The versions of Lance prior to when we started writing the writer version
    // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
    // make sure to recompute them.
    // See: https://github.com/lance-format/lance/issues/1531
    let recompute_stats = previous_writer_version.is_none();

    // It's possible the fragments are old and don't have physical rows or
    // num deletions recorded. If that's the case, we need to grab and set that
    // information.
    let fragments = migrate_fragments(dataset.as_ref(), &task.fragments, recompute_stats).await?;
    let num_rows = fragments
        .iter()
        .map(|f| f.physical_rows.unwrap() as u64)
        .sum::<u64>();
    let uses_stable_row_ids = dataset.manifest.uses_stable_row_ids();
    // A scan captures stable row ids, not row addresses; derive old addresses
    // from fragment metadata instead, whenever something will consume them.
    let records_from_fragment_metadata = uses_stable_row_ids
        && (options.defer_index_remap || {
            // The planner precomputes this once for the whole plan (see
            // `CompactionOptions::has_eager_remappable_index`); fall back to
            // loading it here for options built outside the planner.
            match options.has_eager_remappable_index {
                Some(value) => value,
                None => has_eager_remappable_index(dataset.as_ref()).await?,
            }
        });
    // Capturing row addresses is only useful if something will consume them:
    // an index to remap now, or a deferred remap through the FRI.
    let capture_row_addrs = !uses_stable_row_ids
        && (options.defer_index_remap
            || load_indices_for_remapping(dataset.as_ref())
                .await?
                .is_some());
    let mut new_fragments: Vec<Fragment>;
    let task_id = uuid::Uuid::new_v4();
    log::info!(
        "Compaction task {}: Begin compacting {} rows across {} fragments",
        task_id,
        num_rows,
        fragments.len()
    );
    let mode = options.compaction_mode();
    let can_binary_copy = can_use_binary_copy(dataset.as_ref(), options, &fragments).await;
    if !can_binary_copy && matches!(mode, CompactionMode::ForceBinaryCopy) {
        if let Some((path, actual)) = first_binary_copy_version_mismatch(&fragments, write_version)?
        {
            return Err(Error::not_supported_source(
                format!(
                    "compaction task {task_id}: binary copy target is {write_version}, but data file '{path}' uses {actual}"
                )
                .into(),
            ));
        }
        return Err(Error::not_supported_source(
            format!("compaction task {}: binary copy is not supported", task_id).into(),
        ));
    }
    let mut row_ids_rx: Option<std::sync::mpsc::Receiver<CapturedRowIds>> = None;
    let mut reader: Option<SendableRecordBatchStream> = None;

    let mut sidecars = None;
    if !can_binary_copy {
        let (prepared_reader, rx_initial, has_blob_v2_columns) = prepare_reader(
            dataset.as_ref(),
            &fragments,
            options.batch_size,
            options.io_buffer_size,
            true,
            capture_row_addrs,
        )
        .await?;
        row_ids_rx = rx_initial;

        let mut rows_read = 0;
        let schema = prepared_reader.schema();
        let reader_with_progress = prepared_reader.inspect_ok(move |batch| {
            rows_read += batch.num_rows();
            log::info!(
                "Compaction task {}: Read progress {}/{}",
                task_id,
                rows_read,
                num_rows,
            );
        });

        if has_blob_v2_columns {
            let dataset_arc = Arc::new(dataset.as_ref().clone());
            let supports_prepared = matches!(
                write_version,
                ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3
            );
            let mut rewrite_plan = BlobV2BatchRewritePlan::try_new(
                dataset.schema(),
                schema.as_ref(),
                false,
                supports_prepared,
            )?;
            if supports_prepared && !managed_blobs_enabled(dataset.schema())? {
                let context = Arc::new(SidecarRewrite::default());
                rewrite_plan = rewrite_plan.with_sidecar_rewrite(context.clone());
                sidecars = Some(context);
            }
            let rewrite_plan = Arc::new(rewrite_plan);
            let transformed_schema = rewrite_plan.output_schema().clone();
            let transformed = reader_with_progress.then(move |batch_result| {
                let dataset = dataset_arc.clone();
                let rewrite_plan = rewrite_plan.clone();
                async move {
                    let batch = batch_result?;
                    rewrite_plan
                        .transform_batch(&dataset, batch)
                        .await
                        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
                }
            });
            reader = Some(Box::pin(RecordBatchStreamAdapter::new(
                transformed_schema,
                transformed,
            )));
        } else {
            reader = Some(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                reader_with_progress,
            )));
        }
    }

    let surviving_rows = fragments.iter().try_fold(0_u64, |total, fragment| {
        let fragment_rows = fragment.num_rows().ok_or_else(|| {
            Error::internal(format!(
                "Fragment {} is missing row count metadata after migration",
                fragment.id
            ))
        })?;
        total.checked_add(fragment_rows as u64).ok_or_else(|| {
            Error::internal("Compaction task surviving row count overflowed u64".to_string())
        })
    })?;

    // Planner-sized tasks may exceed the target, but should remain one output
    // instead of producing a target-sized fragment plus a stranded tail. For
    // genuinely oversized tasks, choose a target-scale output count and spread
    // the tail across those outputs.
    let target_rows_per_fragment = options.target_rows_per_fragment as u64;
    let output_fragment_count = surviving_rows
        .checked_div(target_rows_per_fragment)
        .unwrap_or(1)
        .max(1);
    let output_fragment_count_usize = usize::try_from(output_fragment_count).map_err(|_| {
        Error::internal(format!(
            "Compaction output fragment count {output_fragment_count} does not fit in usize"
        ))
    })?;
    let base_rows_per_file = surviving_rows / output_fragment_count;
    let larger_file_count =
        usize::try_from(surviving_rows % output_fragment_count).map_err(|_| {
            Error::internal("Compaction larger output fragment count does not fit in usize")
        })?;
    let file_row_counts = if surviving_rows == 0 {
        Vec::new()
    } else {
        (0..output_fragment_count_usize)
            .map(|file_index| {
                let file_rows = base_rows_per_file + u64::from(file_index < larger_file_count);
                usize::try_from(file_rows).map_err(|_| {
                    Error::internal(format!(
                        "Compaction output row count {file_rows} does not fit in usize"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    };
    let max_rows_per_file = file_row_counts.first().copied().unwrap_or(1);

    let file_writer_options = FileWriterOptions {
        data_cache_bytes: options.data_cache_bytes,
        ..Default::default()
    };

    let mut params = WriteParams {
        max_rows_per_file,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        // External blobs may reference URIs outside the dataset's base_paths
        // (e.g. absolute file:// URIs with base_id == 0). Without this flag
        // the writer would reject such blobs.
        allow_external_blob_outside_bases: true,
        data_storage_version: Some(write_version.to_selector()),
        file_writer_options: Some(file_writer_options),
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        params.max_bytes_per_file = max_bytes_per_file;
    }

    if dataset.manifest.uses_stable_row_ids() {
        params.enable_stable_row_ids = true;
    }

    let lineage_plan = if can_binary_copy {
        new_fragments = versions::rewrite_files_binary_copy(
            write_version,
            dataset.as_ref(),
            &fragments,
            &params,
            options.binary_copy_read_batch_bytes,
        )
        .await?;

        if new_fragments.is_empty() && matches!(mode, CompactionMode::ForceBinaryCopy) {
            return Err(Error::not_supported_source(
                format!("compaction task {}: binary copy is not supported", task_id).into(),
            ));
        }

        if capture_row_addrs {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut addrs = RoaringTreemap::new();
            for frag in &fragments {
                let frag_id = frag.id as u32;
                let count = u64::try_from(frag.physical_rows.unwrap_or(0)).map_err(|_| {
                    Error::internal(format!(
                        "Fragment {} has too many physical rows to represent as row addresses",
                        frag.id
                    ))
                })?;
                let start = u64::from(lance_core::utils::address::RowAddress::first_row(frag_id));
                addrs.insert_range(start..start + count);
            }
            let captured = CapturedRowIds::AddressStyle(addrs);
            let _ = tx.send(captured);
            row_ids_rx = Some(rx);
        }
        None
    } else {
        let mut stream = reader.expect("reader must be prepared for non-binary-copy path");
        let mut write_schema = dataset.schema().clone();
        // On a table that can spill, the output fragments' lineage is computed
        // before anything is written: the row ids and versions carry over from
        // the inputs, so a sequence that leaves the manifest can ride along as
        // a hidden column of the file being written rather than need a file of
        // its own. Every other table carries its lineage over after the write,
        // cut at the row counts actually written, and holds none of it while
        // writing. Binary copy cannot add columns to the files it copies, so
        // it always takes that path.
        let spill_budget = if dataset.manifest.uses_stable_row_ids() {
            inline_row_lineage_max_bytes(dataset.as_ref())?
        } else {
            None
        };
        let lineage_plan = match spill_budget {
            Some(limit) => {
                let planned_rows = file_row_counts
                    .iter()
                    .map(|rows| *rows as u64)
                    .collect::<Vec<_>>();
                let mut lineages =
                    compute_row_lineage(dataset.as_ref(), &fragments, &planned_rows).await?;
                let plan = plan_row_lineage_spill(limit, &lineages);
                if let RowLineagePlan::InFile(spill) = plan
                    && spill.any()
                {
                    let fields = spill.schema_fields()?;
                    let columns = spill.take_columns(&mut lineages);
                    stream = append_row_lineage_columns(stream, &fields, columns);
                    // The writer stores only the columns its schema lists.
                    write_schema.fields.extend(fields);
                }
                Some(PlannedRowLineage {
                    lineages,
                    planned_rows,
                    plan,
                })
            }
            None => None,
        };
        let (frags, _) = write_fragments_internal_impl(
            write_version,
            Some(dataset.as_ref()),
            dataset.object_store.clone(),
            &dataset.base,
            write_schema,
            stream,
            params,
            None,
            Some(file_row_counts),
            None,
            false,
            false,
            None,
            sidecars.clone(),
        )
        .await?;
        new_fragments = frags;
        lineage_plan
    };

    log::info!("Compaction task {}: file written", task_id);

    // Wrap in an async block so `?` returns into `row_addrs_result` and we can
    // run cleanup before propagating the error.
    let row_addrs_result: Result<Option<Vec<u8>>> = async {
        if uses_stable_row_ids {
            log::info!("Compaction task {}: rechunking stable row ids", task_id);
            match lineage_plan {
                Some(planned) => {
                    place_planned_row_lineage(dataset.as_ref(), &mut new_fragments, planned).await?
                }
                None => {
                    rechunk_row_lineage(dataset.as_ref(), &mut new_fragments, &fragments).await?
                }
            }
        }
        let mut row_addrs = if let Some(row_ids_rx) = row_ids_rx {
            let captured_ids = row_ids_rx
                .try_recv()
                .map_err(|err| Error::internal(format!("Failed to receive row ids: {}", err)))?;
            captured_ids.row_addrs(None)?.into_owned()
        } else if records_from_fragment_metadata {
            live_row_addrs(dataset.as_ref(), &fragments, &new_fragments).await?
        } else {
            return Ok(None);
        };
        // Compaction reads whole fragments, so the captured addresses are
        // dense per-fragment ranges; run containers (standard roaring
        // format) shrink the persisted blob from O(rows) to O(runs) bytes.
        row_addrs.optimize();
        let mut serialized = Vec::with_capacity(row_addrs.serialized_size());
        row_addrs.serialize_into(&mut serialized)?;
        Ok(Some(serialized))
    }
    .await;

    let row_addrs = match row_addrs_result {
        Ok(v) => v,
        Err(e) => {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            return Err(e);
        }
    };

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    log::info!("Compaction task {}: completed", task_id);

    for fragment in &mut new_fragments {
        fragment.clustering_id = task.output_clustering_id;
        fragment.metadata = task.output_metadata.clone();
    }

    Ok(RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments: fragments,
        row_addrs,
    })
}

/// Carry the stable row ids and per-row versions of `old_fragments` over to
/// output fragments of `chunk_sizes` rows each, in order, which hold the same
/// live rows in the same order.
async fn compute_row_lineage(
    dataset: &Dataset,
    old_fragments: &[Fragment],
    chunk_sizes: &[u64],
) -> Result<Vec<RowLineage>> {
    let mut old_sequences = load_row_id_sequences(dataset, old_fragments)
        .try_collect::<Vec<_>>()
        .await?;
    // Should sort them back into original order.
    old_sequences.sort_by_key(|(frag_id, _)| {
        old_fragments
            .iter()
            .position(|frag| frag.id as u32 == *frag_id)
            .expect("Fragment not found")
    });

    // Load old per-row version sequences, defaulting the way readers do:
    // created-at to version 1, last-updated-at to created-at.
    let mut old_created_at_sequences = Vec::with_capacity(old_fragments.len());
    let mut old_last_updated_sequences = Vec::with_capacity(old_fragments.len());
    for (frag, (_, row_ids)) in old_fragments.iter().zip(old_sequences.iter()) {
        let created_at =
            match load_row_version_sequence(dataset, frag, RowVersionKind::CreatedAt).await? {
                Some(sequence) => sequence.as_ref().clone(),
                None => RowDatasetVersionSequence::from_uniform_row_count(row_ids.len(), 1),
            };
        let last_updated_at =
            match load_row_version_sequence(dataset, frag, RowVersionKind::LastUpdatedAt).await? {
                Some(sequence) => sequence.as_ref().clone(),
                None => created_at.clone(),
            };
        old_created_at_sequences.push(created_at);
        old_last_updated_sequences.push(last_updated_at);
    }

    // Need to remove deleted rows
    for (index, frag) in old_fragments.iter().enumerate() {
        if let Some(deletion_file) = &frag.deletion_file {
            let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;

            let mut new_seq = old_sequences[index].1.as_ref().clone();
            new_seq.mask(deletions.to_sorted_iter())?;
            old_sequences[index].1 = Arc::new(new_seq);
            old_created_at_sequences[index].mask(deletions.to_sorted_iter())?;
            old_last_updated_sequences[index].mask(deletions.to_sorted_iter())?;
        }
    }

    debug_assert_eq!(
        { old_sequences.iter().map(|(_, seq)| seq.len()).sum::<u64>() },
        { chunk_sizes.iter().sum::<u64>() },
        "{:?}",
        old_sequences
    );

    let new_row_ids = lance_table::rowids::rechunk_sequences(
        old_sequences
            .into_iter()
            .map(|(_, seq)| seq.as_ref().clone()),
        chunk_sizes.iter().copied(),
        false,
    )?;
    let new_created_at = lance_table::rowids::version::rechunk_version_sequences(
        old_created_at_sequences,
        chunk_sizes.iter().copied(),
        false,
    )?;
    let new_last_updated_at = lance_table::rowids::version::rechunk_version_sequences(
        old_last_updated_sequences,
        chunk_sizes.iter().copied(),
        false,
    )?;

    Ok(new_row_ids
        .into_iter()
        .zip(new_created_at)
        .zip(new_last_updated_at)
        .map(|((row_ids, created_at), last_updated_at)| RowLineage {
            row_ids,
            created_at,
            last_updated_at,
        })
        .collect())
}

/// Carry the lineage of `old_fragments` over to the already written
/// `new_fragments` and place each one's sequences inline or in a separate
/// lineage file, as the table's spill policy and their size call for. This is
/// the path for tables that cannot spill, and for binary copy, whose output
/// files cannot take the lineage as extra columns.
///
/// Visible to the crate so tests that assemble an
/// [`crate::dataset::transaction::Operation::Rewrite`] by hand
/// can transfer row lineage the way compaction does, rather than reimplementing it.
pub async fn rechunk_row_lineage(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    let chunk_sizes: Vec<u64> = new_fragments
        .iter()
        .map(|frag| frag.physical_rows.unwrap() as u64)
        .collect();
    let lineages = compute_row_lineage(dataset, old_fragments, &chunk_sizes).await?;
    for (fragment, lineage) in new_fragments.iter_mut().zip(lineages) {
        place_row_lineage(dataset, &lineage).await?.apply(fragment);
    }
    Ok(())
}

/// The row lineage a compaction task computed for its planned output
/// fragments before writing them, and how the task places it.
pub(super) struct PlannedRowLineage {
    /// Each planned output fragment's lineage, in order. The sequences the
    /// plan writes into the data files are empty: their values went into the
    /// written columns.
    pub(super) lineages: Vec<RowLineage>,
    /// Each planned output fragment's row count, which the writer only
    /// follows until a byte limit closes a file early.
    pub(super) planned_rows: Vec<u64>,
    pub(super) plan: RowLineagePlan,
}

/// Place the lineage compaction planned before the write on the fragments it
/// wrote.
pub(super) async fn place_planned_row_lineage(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    planned: PlannedRowLineage,
) -> Result<()> {
    let PlannedRowLineage {
        lineages,
        planned_rows,
        plan,
    } = planned;
    match plan {
        RowLineagePlan::InFile(spill) => {
            place_row_lineage_in_files(new_fragments, lineages, &planned_rows, spill)
        }
        RowLineagePlan::PerFragment => {
            let lineages = fit_row_lineage_to_written(
                new_fragments,
                lineages,
                &planned_rows,
                RowLineageSpill::default(),
            )?;
            for (fragment, lineage) in new_fragments.iter_mut().zip(lineages) {
                place_row_lineage(dataset, &lineage).await?.apply(fragment);
            }
            Ok(())
        }
    }
}

/// Mark each written fragment's spilled sequences as living in its own data
/// file, which was written with them as hidden columns, and place the rest
/// inline.
pub(super) fn place_row_lineage_in_files(
    new_fragments: &mut [Fragment],
    lineages: Vec<RowLineage>,
    planned_rows: &[u64],
    spill: RowLineageSpill,
) -> Result<()> {
    let lineages = fit_row_lineage_to_written(new_fragments, lineages, planned_rows, spill)?;
    for (fragment, lineage) in new_fragments.iter_mut().zip(&lineages) {
        if spill.any() {
            let [file] = fragment.files.as_slice() else {
                return Err(Error::internal(format!(
                    "compaction wrote {} data files for fragment {}; the row lineage columns \
                     are in exactly one",
                    fragment.files.len(),
                    fragment.id
                )));
            };
            let missing = spill
                .field_ids()
                .find(|field_id| !file.fields.contains(field_id));
            if let Some(field_id) = missing {
                return Err(Error::internal(format!(
                    "compaction wrote fragment {}'s data file {} without row lineage field {}",
                    fragment.id, file.path, field_id
                )));
            }
        }
        spill.place_in_file(lineage).apply(fragment);
    }
    Ok(())
}

/// Fit the lineage planned for output fragments of `planned_rows` rows each to
/// the fragments the writer actually produced, one lineage per fragment.
///
/// A byte limit can close a file before its planned row count, after which the
/// writer spreads the remaining rows over files of other sizes, so the
/// sequences kept inline are cut again at the written row counts. The types
/// `in_file` names need nothing: their columns were appended to the written
/// rows one by one, so they follow any split, and their sequences are empty.
///
/// The in-file plan was made for the planned sizes. A written fragment is
/// never larger than the largest planned one, but it can join the tail of one
/// planned fragment to the head of the next, so a sequence that plan keeps
/// inline can end up over the budget, up to about twice it. That is acceptable
/// because the budget only bounds manifest growth and is not a format limit.
/// The per-fragment plan decides again on the cut sequences.
fn fit_row_lineage_to_written(
    new_fragments: &[Fragment],
    lineages: Vec<RowLineage>,
    planned_rows: &[u64],
    in_file: RowLineageSpill,
) -> Result<Vec<RowLineage>> {
    let written_rows = new_fragments
        .iter()
        .map(|fragment| match fragment.physical_rows {
            Some(rows) => Ok(rows as u64),
            None => Err(Error::internal(format!(
                "compaction wrote fragment {} without a physical row count",
                fragment.id
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let planned_total = planned_rows.iter().sum::<u64>();
    let written_total = written_rows.iter().sum::<u64>();
    if written_total != planned_total {
        return Err(Error::internal(format!(
            "compaction wrote {written_total} rows, in fragments of {written_rows:?} rows, \
             but planned row lineage for {planned_total} rows, in fragments of \
             {planned_rows:?} rows"
        )));
    }

    let lineages = if written_rows.as_slice() == planned_rows {
        lineages
    } else {
        let cut_error = |error: Error| {
            Error::internal(format!(
                "compaction could not cut the row lineage planned for fragments of \
                 {planned_rows:?} rows at the written fragments of {written_rows:?} rows: \
                 {error}"
            ))
        };
        let mut row_ids = Vec::with_capacity(lineages.len());
        let mut created_at = Vec::with_capacity(lineages.len());
        let mut last_updated_at = Vec::with_capacity(lineages.len());
        for lineage in lineages {
            row_ids.push(lineage.row_ids);
            created_at.push(lineage.created_at);
            last_updated_at.push(lineage.last_updated_at);
        }
        let fragment_count = written_rows.len();
        let written = || written_rows.iter().copied();
        let row_ids = if in_file.row_ids {
            vec![RowIdSequence::new(); fragment_count]
        } else {
            lance_table::rowids::rechunk_sequences(row_ids, written(), false).map_err(cut_error)?
        };
        let created_at = if in_file.created_at {
            vec![RowDatasetVersionSequence::new(); fragment_count]
        } else {
            lance_table::rowids::version::rechunk_version_sequences(created_at, written(), false)
                .map_err(cut_error)?
        };
        let last_updated_at = if in_file.last_updated_at {
            vec![RowDatasetVersionSequence::new(); fragment_count]
        } else {
            lance_table::rowids::version::rechunk_version_sequences(
                last_updated_at,
                written(),
                false,
            )
            .map_err(cut_error)?
        };
        row_ids
            .into_iter()
            .zip(created_at)
            .zip(last_updated_at)
            .map(|((row_ids, created_at), last_updated_at)| RowLineage {
                row_ids,
                created_at,
                last_updated_at,
            })
            .collect::<Vec<_>>()
    };

    if lineages.len() != new_fragments.len() {
        return Err(Error::internal(format!(
            "compaction has row lineage for {} fragments but wrote {}",
            lineages.len(),
            new_fragments.len()
        )));
    }
    for ((fragment, &rows), lineage) in new_fragments.iter().zip(&written_rows).zip(&lineages) {
        let inline_lengths = [
            (in_file.row_ids, "row ids", lineage.row_ids.len()),
            (
                in_file.created_at,
                "created-at versions",
                lineage.created_at.len(),
            ),
            (
                in_file.last_updated_at,
                "last-updated-at versions",
                lineage.last_updated_at.len(),
            ),
        ];
        let mismatch = inline_lengths
            .into_iter()
            .find(|&(is_in_file, _, len)| !is_in_file && len != rows);
        if let Some((_, sequence, len)) = mismatch {
            return Err(Error::internal(format!(
                "compaction has {len} inline {sequence} for fragment {} of {rows} physical rows",
                fragment.id
            )));
        }
    }
    Ok(lineages)
}

/// Append the hidden row lineage columns, `fields` with the matching entry of
/// `columns` as each one's values, to the batches of `stream`, in row order,
/// so the file writer stores them next to the user columns.
pub(super) fn append_row_lineage_columns(
    stream: SendableRecordBatchStream,
    fields: &[LanceField],
    columns: Vec<Vec<u64>>,
) -> SendableRecordBatchStream {
    let mut arrow_fields = stream.schema().fields().to_vec();
    arrow_fields.extend(fields.iter().map(|field| Arc::new(ArrowField::from(field))));
    let schema = Arc::new(ArrowSchema::new(arrow_fields));
    let total_rows = columns.first().map_or(0, |values| values.len());
    let batch_schema = schema.clone();
    let mut offset = 0usize;
    let batches = stream.map(move |batch| {
        let batch = batch?;
        let rows = batch.num_rows();
        if offset + rows > total_rows {
            return Err(datafusion::error::DataFusionError::External(Box::new(
                Error::internal(format!(
                    "compaction read more rows than its row lineage covers ({total_rows})"
                )),
            )));
        }
        let mut arrays = batch.columns().to_vec();
        // Each batch gets a buffer of its own. A slice would share the whole
        // task's buffer, and the writer, which sizes pages by the memory an
        // array holds, would then cut a tiny page for every batch.
        for values in &columns {
            let batch_values = values[offset..offset + rows].to_vec();
            arrays.push(Arc::new(arrow_array::UInt64Array::from(batch_values)) as ArrayRef);
        }
        offset += rows;
        RecordBatch::try_new(batch_schema.clone(), arrays)
            .map_err(datafusion::error::DataFusionError::from)
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, batches))
}
