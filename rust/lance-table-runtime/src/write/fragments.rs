// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::tracing::{
    AUDIT_MODE_CREATE, AUDIT_MODE_DELETE, AUDIT_TYPE_DATA, TRACE_FILE_AUDIT,
};
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_io::stream::SendableRecordBatchStream;
use lance_io::stream::chunker::{
    break_stream, break_stream_with_sizes, chunk_stream, chunk_stream_with_sizes,
};
use lance_table::format::Fragment;
use object_store::path::Path;
use tracing::info;

use super::bases::blob_v2_external_base_resolver;
use super::dictionary_budget::DictionaryValueBudget;
use super::file::{GenericWriter, WriterOptions, open_writer};
use super::schema::{
    prepare_write_schema, promote_legacy_blob_schema, schema_compare_options,
    split_row_lineage_fields, validate_blob_v2_write_schema, validate_legacy_blob_write_schema,
    validate_row_limits,
};
use super::schema_adapter::SchemaAdapter;
use super::seeds::{SeedCollector, WriteSeedHook};
use super::{
    ExternalBlobMode, PRIMARY_BASE_ID, TargetBaseInfo, WriteMode, WriteParams, WriteStats,
};
use crate::Table;
use crate::blob::{
    BlobPayloadInventory, ExternalBaseResolver, SidecarRewrite, prepared_to_logical_blob_schema,
};
use crate::commit::assign_new_table_field_ids;
use crate::paths::DATA_DIR;

/// Operation-specific controls for writing uncommitted fragments.
#[derive(Default)]
#[doc(hidden)]
pub struct FragmentWriteOptions<'a> {
    pub target_bases_info: Option<Vec<TargetBaseInfo>>,
    pub file_row_counts: Option<Vec<usize>>,
    pub preassigned_data_file_name: Option<Arc<String>>,
    pub inherited_field_ids: bool,
    pub disable_dictionary_split: bool,
    pub created_blob_payloads: Option<BlobPayloadInventory>,
    pub sidecar_rewrite: Option<Arc<SidecarRewrite>>,
    pub seed_hook: Option<&'a dyn WriteSeedHook>,
}

fn take_batch_rows(batches: &mut VecDeque<RecordBatch>, max_rows: usize) -> Vec<RecordBatch> {
    let mut output = Vec::with_capacity(batches.len());
    let mut rows_remaining = max_rows;

    while rows_remaining > 0 {
        let Some(batch) = batches.pop_front() else {
            break;
        };
        let batch_rows = batch.num_rows();
        if batch_rows == 0 {
            continue;
        }
        if batch_rows <= rows_remaining {
            rows_remaining -= batch_rows;
            output.push(batch);
        } else {
            output.push(batch.slice(0, rows_remaining));
            batches.push_front(batch.slice(rows_remaining, batch_rows - rows_remaining));
            rows_remaining = 0;
        }
    }

    output
}

fn balanced_row_counts(total_rows: usize, max_rows_per_file: usize) -> VecDeque<usize> {
    if total_rows == 0 {
        return VecDeque::new();
    }

    let file_count = total_rows.div_ceil(max_rows_per_file);
    let base_rows_per_file = total_rows / file_count;
    let larger_file_count = total_rows % file_count;
    (0..file_count)
        .map(|file_index| base_rows_per_file + usize::from(file_index < larger_file_count))
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn do_write_fragments_impl<OpenWriter, OpenWriterFuture>(
    dataset: Option<&Table>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    mut buffered_reader: futures::stream::BoxStream<'static, Result<Vec<RecordBatch>>>,
    params: WriteParams,
    open_writer: OpenWriter,
    external_base_resolver: Option<Arc<ExternalBaseResolver>>,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    mut seeds: SeedCollector,
    file_row_counts: Option<Vec<usize>>,
    preassigned_data_file_name: Option<Arc<String>>,
    created_blob_payloads: Option<BlobPayloadInventory>,
    mut dictionary_budget: Option<DictionaryValueBudget>,
) -> Result<Vec<Fragment>>
where
    OpenWriter: Fn(Arc<ObjectStore>, Schema, Path, WriterOptions) -> OpenWriterFuture + Send + Sync,
    OpenWriterFuture: Future<Output = Result<Box<dyn GenericWriter>>> + Send,
{
    let options = WriterOptions {
        source_store_registry: dataset
            .map(|ds| ds.session().store_registry())
            .unwrap_or_else(|| params.store_registry()),
        preassigned_data_file_name,
        created_blob_payloads,
        ..WriterOptions::from_params(&params, external_base_resolver)
    };
    let cleanup_bases = target_bases_info.clone();
    let writer_generator = WriterGenerator::new(
        object_store.clone(),
        base_dir,
        schema,
        open_writer,
        target_bases_info,
        options,
    );
    let mut writer: Option<Box<dyn GenericWriter>> = None;
    let mut num_rows_in_current_file = 0;
    let mut fragments: Vec<Fragment> = Vec::new();
    let mut bytes_completed: u64 = 0;
    let mut rows_completed: u64 = 0;
    let mut files_written: u32 = 0;
    let has_file_row_counts = file_row_counts.is_some();
    let max_planned_file_rows = file_row_counts
        .as_ref()
        .and_then(|row_counts| row_counts.iter().copied().max());
    let mut planned_rows_remaining = file_row_counts
        .as_ref()
        .map(|row_counts| {
            row_counts.iter().try_fold(0_usize, |total, &row_count| {
                total
                    .checked_add(row_count)
                    .ok_or_else(|| Error::internal("Planned file row count total overflowed usize"))
            })
        })
        .transpose()?;
    let mut file_row_counts = file_row_counts.map(VecDeque::from);
    let mut rows_remaining_in_planned_file = file_row_counts.as_mut().and_then(VecDeque::pop_front);

    // Wrap the loop in an async block so `?` returns into `loop_result` and we
    // can run cleanup before propagating the error.
    let loop_result: Result<()> = async {
        while let Some(batch_chunk) = buffered_reader.next().await {
            let mut pending_batches = VecDeque::from(batch_chunk?);

            while !pending_batches.is_empty() {
                let mut rows_to_take = if has_file_row_counts {
                    rows_remaining_in_planned_file.ok_or_else(|| {
                        Error::internal(
                            "Writer received rows after all planned file boundaries were consumed",
                        )
                    })?
                } else if dictionary_budget.is_some() {
                    // The input is cut at multiples of `max_rows_per_file` from its start,
                    // which a file finished early for its dictionaries no longer lines up
                    // with, so take no more than the file has room for.
                    params
                        .max_rows_per_file
                        .checked_sub(num_rows_in_current_file)
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "Writer holds {num_rows_in_current_file} rows in a file limited to {} rows",
                                params.max_rows_per_file
                            ))
                        })?
                } else {
                    usize::MAX
                };
                let mut reached_dictionary_limit = false;
                if let Some(budget) = dictionary_budget.as_mut() {
                    let admission = budget.admit(&pending_batches, rows_to_take)?;
                    rows_to_take = admission.rows;
                    reached_dictionary_limit = admission.is_full;
                }
                let mut reached_planned_file_boundary = false;
                let mut current_file_bytes = 0;
                // Nothing is admitted when the next batch would overflow a dictionary of
                // the current file; finish the file without writing.
                if rows_to_take > 0 {
                    let batch_chunk = take_batch_rows(&mut pending_batches, rows_to_take);
                    if batch_chunk.is_empty() {
                        continue;
                    }
                    // Seed observers must see the values the data file stores, so
                    // convert before handing the batches to both. The data file
                    // writer then finds nothing left to convert.
                    let batch_chunk = batch_chunk
                        .into_iter()
                        .map(|batch| SchemaAdapter::new(batch.schema()).to_physical_batch(batch))
                        .collect::<Result<Vec<_>>>()?;

                    if writer.is_none() {
                        let (new_writer, new_fragment) = writer_generator.new_writer().await?;
                        params.progress.begin(&new_fragment).await?;
                        writer = Some(new_writer);
                        fragments.push(new_fragment);
                    }

                    let active_writer = writer.as_mut().ok_or_else(|| {
                        Error::internal("Writer was not initialized before writing a batch")
                    })?;
                    let batch_chunk_rows =
                        batch_chunk.iter().map(RecordBatch::num_rows).sum::<usize>();
                    let reaches_row_limit = if has_file_row_counts {
                        rows_remaining_in_planned_file == Some(batch_chunk_rows)
                    } else {
                        num_rows_in_current_file + batch_chunk_rows >= params.max_rows_per_file
                    };
                    let input_memory = batch_chunk
                        .iter()
                        .map(RecordBatch::get_array_memory_size)
                        .sum::<usize>();
                    let data_cache_bytes = params
                        .file_writer_options
                        .as_ref()
                        .and_then(|options| options.data_cache_bytes)
                        .unwrap_or(8 * 1024 * 1024);
                    let flush_last =
                        reaches_row_limit && input_memory > data_cache_bytes as usize;
                    if flush_last {
                        let (last, prefix) = batch_chunk
                            .split_last()
                            .ok_or_else(|| Error::internal("empty writer batch chunk"))?;
                        if !prefix.is_empty() {
                            active_writer.write(prefix).await?;
                        }
                        active_writer.prepare_for_flush();
                        active_writer.write(std::slice::from_ref(last)).await?;
                    } else {
                        active_writer.write(&batch_chunk).await?;
                    }
                    for batch in &batch_chunk {
                        seeds.observe(batch)?;
                    }
                    num_rows_in_current_file += batch_chunk_rows;
                    // The writer owns the values it needs (either borrowed for an
                    // immediate page or detached into the queue).  Do not keep the
                    // input's parent buffers alive through the file finish below.
                    drop(batch_chunk);

                    reached_planned_file_boundary = if has_file_row_counts {
                        let rows_remaining =
                            rows_remaining_in_planned_file.as_mut().ok_or_else(|| {
                                Error::internal(
                                    "Writer received rows without an active planned file boundary",
                                )
                            })?;
                        *rows_remaining = rows_remaining.checked_sub(batch_chunk_rows).ok_or_else(|| {
                            Error::internal(format!(
                                "Writer chunk of {batch_chunk_rows} rows crossed a planned file boundary with {rows_remaining} rows remaining"
                            ))
                        })?;
                        let total_remaining = planned_rows_remaining.as_mut().ok_or_else(|| {
                            Error::internal("Writer lost the planned row count total")
                        })?;
                        *total_remaining =
                            total_remaining.checked_sub(batch_chunk_rows).ok_or_else(|| {
                                Error::internal(format!(
                                    "Writer consumed {batch_chunk_rows} rows after the planned row count total was exhausted"
                                ))
                            })?;
                        *rows_remaining == 0
                    } else {
                        false
                    };

                    current_file_bytes = writer
                        .as_mut()
                        .ok_or_else(|| Error::internal("Writer disappeared after writing a batch"))?
                        .tell()
                        .await?;
                    if let Some(cb) = &params.write_progress {
                        cb.call(WriteStats {
                            bytes_written: bytes_completed + current_file_bytes,
                            rows_written: rows_completed + num_rows_in_current_file as u64,
                            files_written,
                        });
                    }
                }

                let reached_row_limit = if has_file_row_counts {
                    reached_planned_file_boundary
                } else {
                    num_rows_in_current_file >= params.max_rows_per_file
                };
                let reached_byte_limit = current_file_bytes >= params.max_bytes_per_file as u64;

                if reached_row_limit || reached_byte_limit || reached_dictionary_limit {
                    if has_file_row_counts {
                        if reached_planned_file_boundary {
                            rows_remaining_in_planned_file = file_row_counts
                                .as_mut()
                                .and_then(VecDeque::pop_front);
                        } else {
                            // A byte- or dictionary-driven close is an extra physical boundary. Rebalance all
                            // unwritten rows under the original maximum instead of preserving a
                            // tiny abandoned remainder or rolling it into an oversized tail.
                            let total_remaining = planned_rows_remaining.ok_or_else(|| {
                                Error::internal("Writer lost the planned row count total")
                            })?;
                            let max_rows_per_file = max_planned_file_rows.ok_or_else(|| {
                                Error::internal(
                                    "Writer cannot replan byte-limited files without a maximum planned row count",
                                )
                            })?;
                            let mut replanned_counts =
                                balanced_row_counts(total_remaining, max_rows_per_file);
                            rows_remaining_in_planned_file = replanned_counts.pop_front();
                            file_row_counts = Some(replanned_counts);
                        }
                    }

                    let mut w = writer.take().ok_or_else(|| {
                        Error::internal("Writer disappeared before completing a file")
                    })?;
                    seeds.flush(w.as_mut()).await?;
                    let (num_rows, data_file) = w.finish().await?;
                    info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DATA, path = &data_file.path);
                    debug_assert_eq!(num_rows as usize, num_rows_in_current_file);
                    bytes_completed += data_file.file_size_bytes.get().map_or(0, |s| s.get());
                    rows_completed += num_rows as u64;
                    files_written += 1;
                    let last_fragment = fragments.last_mut().ok_or_else(|| {
                        Error::internal("Writer completed a file without a pending fragment")
                    })?;
                    last_fragment.physical_rows = Some(num_rows as usize);
                    last_fragment.files.push(data_file);
                    // Notify after pushing the data file so it's tracked for cleanup
                    // if the callback fails.
                    let completed_fragment = fragments.last().ok_or_else(|| {
                        Error::internal("Writer completed a file without a fragment")
                    })?;
                    params.progress.complete(completed_fragment).await?;
                    if let Some(cb) = &params.write_progress {
                        cb.call(WriteStats {
                            bytes_written: bytes_completed,
                            rows_written: rows_completed,
                            files_written,
                        });
                    }
                    num_rows_in_current_file = 0;
                    if let Some(budget) = dictionary_budget.as_mut() {
                        budget.reset();
                    }
                }
            }
        }

        if has_file_row_counts && planned_rows_remaining != Some(0) {
            return Err(Error::internal(format!(
                "Writer input ended with {} planned rows remaining",
                planned_rows_remaining.unwrap_or_default()
            )));
        }
        Ok(())
    }
    .await;

    if let Err(e) = loop_result {
        // Drop the writer so its in-progress file is cleaned up (LocalWriter
        // removes its temp file; ObjectWriter aborts the multipart upload).
        drop(writer.take());
        cleanup_data_fragments(
            &object_store,
            base_dir,
            cleanup_bases.as_deref(),
            &fragments,
        )
        .await;
        return Err(e);
    }

    // Complete the final writer
    if let Some(mut writer) = writer.take() {
        if let Err(e) = seeds.flush(writer.as_mut()).await {
            drop(writer);
            cleanup_data_fragments(
                &object_store,
                base_dir,
                cleanup_bases.as_deref(),
                &fragments,
            )
            .await;
            return Err(e);
        }
        match writer.finish().await {
            Ok((num_rows, data_file)) => {
                info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DATA, path = &data_file.path);
                bytes_completed += data_file.file_size_bytes.get().map_or(0, |s| s.get());
                rows_completed += num_rows as u64;
                files_written += 1;
                let last_fragment = fragments.last_mut().unwrap();
                last_fragment.physical_rows = Some(num_rows as usize);
                last_fragment.files.push(data_file);
                // Notify after pushing the data file so it's tracked for cleanup
                // if the callback fails.
                if let Err(e) = params.progress.complete(last_fragment).await {
                    cleanup_data_fragments(
                        &object_store,
                        base_dir,
                        cleanup_bases.as_deref(),
                        &fragments,
                    )
                    .await;
                    return Err(e);
                }
                if let Some(cb) = &params.write_progress {
                    cb.call(WriteStats {
                        bytes_written: bytes_completed,
                        rows_written: rows_completed,
                        files_written,
                    });
                }
            }
            Err(e) => {
                drop(writer);
                cleanup_data_fragments(
                    &object_store,
                    base_dir,
                    cleanup_bases.as_deref(),
                    &fragments,
                )
                .await;
                return Err(e);
            }
        }
    }

    Ok(fragments)
}

/// Best-effort cleanup of data files for fragments that were written but not committed.
///
/// Contract:
/// - Errors from individual `delete` calls are logged and swallowed, never returned —
///   callers should propagate the original write error.
/// - Files in the dataset's default storage (`base_id == None`) are deleted via
///   `object_store`; files whose `base_id` matches an entry in `target_bases` are
///   deleted via that base's object store. Files in bases not listed in
///   `target_bases` are skipped because we don't have their object stores here.
/// - Safe to call with an empty slice.
/// - Must be called before the fragments are committed, otherwise live data may be deleted.
pub async fn cleanup_data_fragments(
    object_store: &ObjectStore,
    base_dir: &Path,
    target_bases: Option<&[TargetBaseInfo]>,
    fragments: &[Fragment],
) {
    let data_dir = base_dir.clone().join(DATA_DIR);
    let mut skipped_external = 0usize;
    for fragment in fragments {
        // Deliberately not `referenced_lance_files()`: callers decide which
        // files belong to the failed write. `schema_evolution` passes a live
        // fragment whose `files` it narrowed to the newly written ones while
        // leaving `overlays` untouched, so including overlays here would delete
        // live data.
        for file in &fragment.files {
            let (store, file_dir) = if let Some(base_id) = file.base_id {
                match target_bases.and_then(|bases| bases.iter().find(|b| b.base_id == base_id)) {
                    Some(base_info) => {
                        let dir = if base_info.is_dataset_root {
                            base_info.base_dir.clone().join(DATA_DIR)
                        } else {
                            base_info.base_dir.clone()
                        };
                        (base_info.object_store.as_ref(), dir)
                    }
                    None => {
                        skipped_external += 1;
                        continue;
                    }
                }
            } else {
                (object_store, data_dir.clone())
            };

            let path = file_dir.clone().join(file.path.as_str());
            match store.delete(&path).await {
                Ok(()) => {
                    info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_DELETE, r#type=AUDIT_TYPE_DATA, path = file.path.as_str());
                }
                Err(e) => {
                    log::warn!("Failed to clean up orphaned data file '{}': {}", path, e);
                }
            }

            // Clean up any blob v2 sidecars that might exist for this data file.
            // Blob v2 sidecars are written to `data/{data_file_key}/{blob_id}.blob`.
            // The `data_file_key` is the file stem of the .lance file.
            if let Some(stem) = std::path::Path::new(file.path.as_str())
                .file_stem()
                .and_then(|s| s.to_str())
            {
                let blob_dir = file_dir.clone().join(stem);
                match store.remove_dir_all(blob_dir.clone()).await {
                    Err(e) if !matches!(e, Error::NotFound { .. }) => {
                        log::warn!("Failed to clean up orphaned blob dir '{}': {}", blob_dir, e);
                    }
                    _ => {}
                }
            }
        }
    }
    if skipped_external > 0 {
        log::warn!(
            "Skipped cleanup of {} orphaned data file(s) in external bases: \
             their object stores are not available here",
            skipped_external
        );
    }
}

struct WriterGenerator<OpenWriter> {
    /// Default object store (used when no target bases specified)
    object_store: Arc<ObjectStore>,
    /// Default base directory (used when no target bases specified)
    base_dir: Path,
    schema: Schema,
    open_writer: OpenWriter,
    /// Target base information (if writing to specific bases)
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    /// Options every writer starts from; `add_data_dir` and `base_id` follow
    /// the selected base.
    options: WriterOptions,
    writers_created: AtomicUsize,
    /// Counter for round-robin selection
    next_base_index: AtomicUsize,
}

impl<OpenWriter, OpenWriterFuture> WriterGenerator<OpenWriter>
where
    OpenWriter: Fn(Arc<ObjectStore>, Schema, Path, WriterOptions) -> OpenWriterFuture + Send + Sync,
    OpenWriterFuture: Future<Output = Result<Box<dyn GenericWriter>>> + Send,
{
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_dir: &Path,
        schema: &Schema,
        open_writer: OpenWriter,
        target_bases_info: Option<Vec<TargetBaseInfo>>,
        options: WriterOptions,
    ) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
            open_writer,
            target_bases_info,
            options,
            writers_created: AtomicUsize::new(0),
            next_base_index: AtomicUsize::new(0),
        }
    }

    /// Select the next target base using round-robin strategy.
    /// TODO: In the future, we can develop different strategies for selecting target bases
    fn select_target_base(&self) -> Option<&TargetBaseInfo> {
        self.target_bases_info.as_ref().map(|bases| {
            let index = self
                .next_base_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            &bases[index % bases.len()]
        })
    }

    pub async fn new_writer(&self) -> Result<(Box<dyn GenericWriter>, Fragment)> {
        let writer_index = self
            .writers_created
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if writer_index > 0 && self.options.preassigned_data_file_name.is_some() {
            return Err(Error::internal(
                "a fixed data file name requires the write to produce exactly one fragment",
            ));
        }
        // Use temporary ID 0; will assign ID later.
        let fragment = Fragment::new(0);

        let writer = if let Some(base_info) = self.select_target_base() {
            (self.open_writer)(
                base_info.object_store.clone(),
                self.schema.clone(),
                base_info.base_dir.clone(),
                WriterOptions {
                    add_data_dir: base_info.is_dataset_root,
                    // Primary-storage slots stamp no base id, like a write
                    // without target bases.
                    base_id: (base_info.base_id != PRIMARY_BASE_ID).then_some(base_info.base_id),
                    ..self.options.clone()
                },
            )
            .await?
        } else {
            (self.open_writer)(
                self.object_store.clone(),
                self.schema.clone(),
                self.base_dir.clone(),
                WriterOptions {
                    add_data_dir: true,
                    base_id: None,
                    ..self.options.clone()
                },
            )
            .await?
        };

        Ok((writer, fragment))
    }
}

/// Write and close fragments without publishing a transaction.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub async fn write_fragments(
    version: ConcreteFileVersion,
    dataset: Option<&Table>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: Schema,
    data: SendableRecordBatchStream,
    mut params: WriteParams,
    options: FragmentWriteOptions<'_>,
) -> Result<(Vec<Fragment>, Schema)> {
    let FragmentWriteOptions {
        target_bases_info,
        file_row_counts,
        preassigned_data_file_name,
        inherited_field_ids,
        disable_dictionary_split,
        created_blob_payloads,
        sidecar_rewrite,
        seed_hook,
    } = options;
    params.max_rows_per_group = params.max_rows_per_group.min(params.max_rows_per_file);
    validate_external_blob_write_params(&params)?;
    let normalized_schema = prepared_to_logical_blob_schema(&schema)?;

    // A writer that spills row lineage into the fragment's own data file
    // carries the hidden columns in its stream. They are not dataset fields:
    // set them aside before the schema is checked against the dataset's and
    // put them back on the schema that is written. This has to come before
    // the blob promotion, which gives every negative field id a new one.
    let (mut normalized_schema, lineage_fields) = split_row_lineage_fields(normalized_schema)?;
    if dataset.is_none() {
        assign_new_table_field_ids(&mut normalized_schema, inherited_field_ids)?;
    }
    let normalized_schema = match version {
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            promote_legacy_blob_schema(&normalized_schema)?
        }
        _ => normalized_schema,
    };
    let version_name = format!("{version:?}");
    let mut schema = prepare_write_schema(
        dataset,
        normalized_schema,
        &params,
        schema_compare_options(version),
        inherited_field_ids,
    )?;
    schema.fields.extend(lineage_fields);
    match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => {
            validate_legacy_blob_write_schema(&schema, &version_name)?;
        }
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            validate_blob_v2_write_schema(&schema)?;
        }
    }
    let seeds = if !matches!(params.mode, WriteMode::Overwrite) {
        SeedCollector::for_write(version, dataset, &schema, seed_hook).await?
    } else {
        SeedCollector::disabled()
    };
    let fragments = Box::pin(write_fragments_direct(
        version,
        dataset,
        object_store,
        base_dir,
        &schema,
        data,
        params,
        target_bases_info,
        seeds,
        file_row_counts,
        preassigned_data_file_name,
        created_blob_payloads,
        disable_dictionary_split,
        sidecar_rewrite,
    ))
    .await?;
    Ok((fragments, schema))
}

#[allow(clippy::too_many_arguments)]
pub async fn write_fragments_direct(
    version: ConcreteFileVersion,
    dataset: Option<&Table>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    seeds: SeedCollector,
    file_row_counts: Option<Vec<usize>>,
    preassigned_data_file_name: Option<Arc<String>>,
    created_blob_payloads: Option<BlobPayloadInventory>,
    disable_dictionary_split: bool,
    sidecar_rewrite: Option<Arc<SidecarRewrite>>,
) -> Result<Vec<Fragment>> {
    validate_row_limits(version, &params)?;
    let buffered_reader = if let Some(file_row_counts) = file_row_counts.as_ref() {
        if let Some(rows) = file_row_counts.iter().find(|rows| **rows == 0) {
            return Err(Error::invalid_input(format!(
                "File row counts must be greater than zero, got {rows}"
            )));
        }
        if let Some(rows) = file_row_counts
            .iter()
            .find(|rows| **rows > u32::MAX as usize)
        {
            return Err(Error::invalid_input(format!(
                "File row counts cannot exceed the fragment row capacity of {} rows, got {rows}",
                u32::MAX
            )));
        }
        match version {
            ConcreteFileVersion::V1 => {
                let max_rows_per_group = params.max_rows_per_group;
                let batch_row_counts =
                    file_row_counts
                        .clone()
                        .into_iter()
                        .flat_map(move |file_rows| {
                            (0..file_rows)
                                .step_by(max_rows_per_group)
                                .map(move |offset| (file_rows - offset).min(max_rows_per_group))
                        });
                chunk_stream_with_sizes(data, batch_row_counts)
            }
            ConcreteFileVersion::V2_0
            | ConcreteFileVersion::V2_1
            | ConcreteFileVersion::V2_2
            | ConcreteFileVersion::V2_3 => break_stream_with_sizes(data, file_row_counts.clone()),
        }
    } else {
        match version {
            ConcreteFileVersion::V1 => chunk_stream(data, params.max_rows_per_group),
            ConcreteFileVersion::V2_0
            | ConcreteFileVersion::V2_1
            | ConcreteFileVersion::V2_2
            | ConcreteFileVersion::V2_3 => break_stream(data, params.max_rows_per_file)
                .map_ok(|batch| vec![batch])
                .boxed(),
        }
    };
    let external_base_resolver = match version {
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            blob_v2_external_base_resolver(dataset, &params, schema).await?
        }
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => None,
    };
    // The 2.0 writer stores a null as one more dictionary value and keeps its own key-type
    // check, so its files are not split here. Neither is a write that must produce one
    // file, the preassigned one or a MemWAL generation's; the encoder's key-type check
    // applies to it instead.
    let dictionary_budget = match version {
        ConcreteFileVersion::V2_1 | ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            if preassigned_data_file_name.is_some() || disable_dictionary_split {
                None
            } else {
                DictionaryValueBudget::try_new(&ArrowSchema::from(schema))?
            }
        }
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 => None,
    };
    do_write_fragments_impl(
        dataset,
        object_store,
        base_dir,
        schema,
        buffered_reader,
        params,
        move |object_store, schema, base_dir, options| {
            let sidecar_rewrite = sidecar_rewrite.clone();
            async move {
                let mut options = options;
                options.sidecar_rewrite = sidecar_rewrite;
                open_writer(version, &object_store, &schema, &base_dir, options).await
            }
        },
        external_base_resolver,
        target_bases_info,
        seeds,
        file_row_counts,
        preassigned_data_file_name,
        created_blob_payloads,
        dictionary_budget,
    )
    .await
}

fn validate_external_blob_write_params(params: &WriteParams) -> Result<()> {
    if params.external_blob_mode == ExternalBlobMode::Ingest
        && params.allow_external_blob_outside_bases
    {
        return Err(Error::invalid_input(
            "allow_external_blob_outside_bases only applies when external_blob_mode=\"reference\"",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::file::open_current_writer;
    use super::*;
    use lance_file::version::LanceFileVersion;
    use lance_table::format::DataFile;

    async fn open_v2_1_test_writer(
        object_store: Arc<ObjectStore>,
        schema: Schema,
        base_dir: Path,
        options: WriterOptions,
    ) -> Result<Box<dyn GenericWriter>> {
        open_current_writer(
            |object_writer, schema, filename, base_id, options| {
                let writer =
                    lance_file::versions::v2_1::create_writer(object_writer, schema, options)?
                        .into();
                let mut data_file = DataFile::new_unstarted(filename, ConcreteFileVersion::V2_1);
                data_file.base_id = base_id;
                Ok((writer, data_file))
            },
            &object_store,
            &schema,
            &base_dir,
            options,
        )
        .await
    }

    #[tokio::test]
    async fn test_explicit_data_file_bases_writer_generator() {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        // Create in-memory object store
        let object_store = Arc::new(ObjectStore::memory());
        let base_dir = Path::from("test/bucket2");

        // Test WriterGenerator with explicit data file bases configuration
        let target_bases = vec![TargetBaseInfo {
            base_id: 2,
            object_store: object_store.clone(),
            base_dir: base_dir.clone(),
            is_dataset_root: false, // Test uses direct data directory
        }];
        let writer_generator = WriterGenerator::new(
            object_store.clone(),
            &base_dir,
            &schema,
            open_v2_1_test_writer,
            Some(target_bases),
            WriterOptions::default(),
        );

        // Create a writer
        let (writer, fragment) = writer_generator.new_writer().await.unwrap();

        // Verify fragment is created
        assert_eq!(fragment.id, 0); // Temporary ID

        // Verify writer is created (we can't test much more without writing data)
        drop(writer); // Clean up
    }

    #[tokio::test]
    async fn test_writer_with_base_id() {
        use arrow_array::Int32Array;
        use arrow_array::RecordBatch;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // Create in-memory object store and writer
        let object_store = Arc::new(ObjectStore::memory());
        let base_dir = Path::from("test/bucket2");

        let mut inner_writer = open_writer(
            LanceFileVersion::Stable.resolve(),
            &object_store,
            &schema,
            &base_dir,
            WriterOptions {
                add_data_dir: false, // Don't add /data
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Write data
        inner_writer.write(&[batch]).await.unwrap();

        // Finish and manually set base_id
        let base_id = 2u32;
        let (_num_rows, mut data_file) = inner_writer.finish().await.unwrap();
        data_file.base_id = Some(base_id);

        assert_eq!(data_file.base_id, Some(base_id));
        assert!(!data_file.path.is_empty());
    }

    #[tokio::test]
    async fn test_round_robin_target_base_selection() {
        use arrow_array::Int32Array;
        use arrow_array::RecordBatch;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        // Create in-memory object stores for different bases
        let store1 = Arc::new(ObjectStore::memory());
        let store2 = Arc::new(ObjectStore::memory());
        let store3 = Arc::new(ObjectStore::memory());

        // Create WriterGenerator with multiple target bases
        let target_bases = vec![
            TargetBaseInfo {
                base_id: 1,
                object_store: store1.clone(),
                base_dir: Path::from("base1"),
                is_dataset_root: false,
            },
            TargetBaseInfo {
                base_id: 2,
                object_store: store2.clone(),
                base_dir: Path::from("base2"),
                is_dataset_root: false,
            },
            TargetBaseInfo {
                base_id: 3,
                object_store: store3.clone(),
                base_dir: Path::from("base3"),
                is_dataset_root: false,
            },
        ];

        let writer_generator = WriterGenerator::new(
            Arc::new(ObjectStore::memory()),
            &Path::from("default"),
            &schema,
            open_v2_1_test_writer,
            Some(target_bases),
            WriterOptions::default(),
        );

        // Create test batch
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // Create multiple writers and verify round-robin selection
        let mut base_ids = Vec::new();
        for _ in 0..6 {
            let (mut writer, _fragment) = writer_generator.new_writer().await.unwrap();
            writer.write(std::slice::from_ref(&batch)).await.unwrap();
            let (_num_rows, data_file) = writer.finish().await.unwrap();
            base_ids.push(data_file.base_id.unwrap());
        }

        // Verify round-robin pattern: 1, 2, 3, 1, 2, 3
        assert_eq!(base_ids, vec![1, 2, 3, 1, 2, 3]);
    }
}
