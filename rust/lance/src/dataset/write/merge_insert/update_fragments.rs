// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::MAX_FRAGMENT_UPDATE_BATCH_BYTES;
use super::job::fragment_update_session_context;
use super::{MergeInsertJob, PatchedFragments};
use crate::dataset::rowids::load_spilled_row_lineage;
use crate::dataset::utils::SchemaAdapter;
use crate::dataset::write::cleanup_data_fragments;
use crate::dataset::write::seeds::SeedCollector;
use crate::dataset::write::{TargetBaseInfo, WriteParams, write_fragments_internal};
use crate::{
    Dataset,
    datafusion::dataframe::SessionContextExt,
    dataset::{fragment::FileFragment, transaction::UpdatedFragmentOffsets, versions},
};
use arrow_array::{RecordBatch, RecordBatchIterator, UInt64Array};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::physical_plan::replace_children_if_necessary;
use datafusion::{
    common::DataFusionError,
    execution::context::SessionContext,
    execution::memory_pool::MemoryConsumer,
    physical_plan::{
        ExecutionPlan, SendableRecordBatchStream, sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter,
    },
    scalar::ScalarValue,
};
use futures::{
    StreamExt, TryStreamExt,
    stream::{self},
};
use lance_arrow::json::convert_json_columns;
use lance_arrow::{RecordBatchExt, SchemaExt, interleave_batches};
use lance_core::utils::address::RowAddress;
use lance_core::{
    Error, ROW_ADDR, ROW_ID, Result,
    datatypes::{OnMissing, OnTypeMismatch},
};
use lance_datafusion::{
    chunker::chunk_stream,
    dataframe::BatchStreamGrouper,
    exec::{HardCapBatchSizeExec, LanceExecutionOptions, provider_to_stream},
    spill::spilling_table_provider_with_job_disk_manager,
    utils::reader_to_stream,
};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::{
    collections::HashSet,
    iter::Peekable,
    sync::{Arc, Mutex},
};
use tokio::task::JoinSet;
use tracing::error;

struct UpdatedRowAddrReconciler<I>
where
    I: Iterator<Item = (u64, (usize, usize))>,
{
    updated_rows: Peekable<I>,
}

impl<I> UpdatedRowAddrReconciler<I>
where
    I: Iterator<Item = (u64, (usize, usize))>,
{
    fn new(updated_rows: I) -> Self {
        Self {
            updated_rows: updated_rows.peekable(),
        }
    }

    fn reconcile_batch(&mut self, original_row_addrs: &[u64]) -> Result<Vec<(usize, usize)>> {
        let mut indices = Vec::with_capacity(original_row_addrs.len());

        for (original_offset, original_row_addr) in original_row_addrs.iter().enumerate() {
            match self.updated_rows.peek().copied() {
                Some((updated_row_addr, updated_row_index))
                    if updated_row_addr == *original_row_addr =>
                {
                    self.updated_rows.next();
                    indices.push(updated_row_index);
                }
                Some((updated_row_addr, _)) if updated_row_addr < *original_row_addr => {
                    return Err(Self::missing_row_error(
                        updated_row_addr,
                        Some(*original_row_addr),
                    ));
                }
                _ => indices.push((0, original_offset)),
            }
        }

        Ok(indices)
    }

    fn finish(mut self) -> Result<()> {
        if let Some((updated_row_addr, _)) = self.updated_rows.next() {
            Err(Self::missing_row_error(updated_row_addr, None))
        } else {
            Ok(())
        }
    }

    fn missing_row_error(updated_row_addr: u64, next_original_row_addr: Option<u64>) -> Error {
        let updated_row_addr = RowAddress::from(updated_row_addr);
        let position = next_original_row_addr.map_or_else(
            || "no target rows remain".to_string(),
            |row_addr| format!("next target row address is {}", RowAddress::from(row_addr)),
        );
        Error::internal(format!(
            "Merge insert update row address {updated_row_addr} is missing from the target fragment; {position}"
        ))
    }
}

/// Where the per-fragment patch tasks in
/// [`MergeInsertJob::update_fragments`] deposit their results.
#[derive(Debug, Default)]
struct PatchSink {
    /// Patched fragments, including key-only matches that only update metadata.
    fragments: Mutex<Vec<Fragment>>,
    /// Physical offsets each fragment had patched. Only populated under stable
    /// row ids, which is the only thing that reads the row-version metadata
    /// these correct.
    offsets: Mutex<HashMap<u64, RoaringBitmap>>,
}

impl MergeInsertJob {
    /// Patches the columns carried by `source` into the fragments that hold the
    /// rows it names, and writes the rows with no target address as new
    /// fragments.
    ///
    /// `source` must carry `_rowaddr` plus the columns to write. A null
    /// `_rowaddr` routes the row to a new fragment. Join columns in `on` are
    /// written for new rows but preserved in existing fragments.
    #[allow(
        clippy::too_many_arguments,
        reason = "Partial updates carry operation-scoped spill resources and caller file-write parameters"
    )]
    pub(super) async fn update_fragments(
        dataset: Arc<Dataset>,
        source: SendableRecordBatchStream,
        target_bases_info: Option<Vec<TargetBaseInfo>>,
        spill_session_context: &SessionContext,
        spill_execution_options: &LanceExecutionOptions,
        versions: (u64, ConcreteFileVersion),
        on: &[String],
        write_params: Arc<WriteParams>,
    ) -> Result<PatchedFragments> {
        let (current_version, write_version) = versions;
        // Fully stage the joined updates before starting the fragment sort. An
        // indexed join can keep the merge stream of its target sort live while
        // its output is pulled; draining a disk-backed replay first releases
        // that upstream reservation before the downstream sort allocates. The
        // zero memory limit is deliberate: the staging buffer holds plain
        // RecordBatches outside DataFusion's memory-pool accounting, so any
        // in-memory allowance would weaken the operation's bounded-memory
        // contract.
        let staged_source = spilling_table_provider_with_job_disk_manager(
            source,
            0,
            spill_session_context.runtime_env().disk_manager.clone(),
        )
        .await?;
        provider_to_stream(staged_source.clone())
            .await?
            .try_for_each(|_| async { Ok::<(), DataFusionError>(()) })
            .await?;
        let source = provider_to_stream(staged_source).await?;
        let source_schema = source.schema();
        let update_projection: Vec<usize> = source_schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(idx, field)| (!on.contains(field.name())).then_some(idx))
            .collect();
        // Every group has the same projected schema. Deletions and fragment
        // coverage change its rows, not whether it carries columns to write.
        let has_updated_columns = update_projection.iter().any(|&idx| {
            let name = source_schema.field(idx).name();
            name != ROW_ADDR && name != ROW_ID
        });
        // Shared across the per-group tasks spawned below; only new fragments
        // are routed to target bases, column patches stay in primary storage.
        let target_bases_info = Arc::new(target_bases_info);
        // Expected source schema: _rowaddr, updated_cols*
        use datafusion::logical_expr::{col, lit};
        let session_ctx =
            fragment_update_session_context(spill_session_context, spill_execution_options);
        // Cap input batches at 25 MiB to leave room for DataFusion's per-batch
        // sort overhead and spill/merge reservation. SortExec must reserve an
        // entire input batch even when spilling is enabled. This cap reduces
        // reservation pressure but cannot guarantee success with a small pool
        // or competing consumers; indivisible oversized rows pass through with
        // a warning.
        let sorted = session_ctx
            .read_one_shot(source)?
            .with_column("_fragment_id", col(ROW_ADDR) >> lit(32))?
            .sort(vec![col(ROW_ADDR).sort(true, true)])?;
        let sorted_plan = sorted.create_physical_plan().await?;
        // Walk the physical plan and insert HardCapBatchSizeExec below every
        // sort node to enforce the input cap (deep-copying oversized slices).
        let capped_plan = sorted_plan
            .transform_down(|node| {
                if node.downcast_ref::<SortExec>().is_some() {
                    let children = node.children();
                    let new_children: Vec<Arc<dyn ExecutionPlan>> = children
                        .into_iter()
                        .map(|c| {
                            Arc::new(HardCapBatchSizeExec::new(
                                c.clone(),
                                MAX_FRAGMENT_UPDATE_BATCH_BYTES,
                            )) as Arc<dyn ExecutionPlan>
                        })
                        .collect();
                    let new_node = replace_children_if_necessary(node, new_children)?;
                    Ok(Transformed::yes(new_node))
                } else {
                    Ok(Transformed::no(node))
                }
            })?
            .data;
        let capped_stream = capped_plan.execute(0, session_ctx.task_ctx())?;
        let mut group_stream = BatchStreamGrouper::new(capped_stream, "_fragment_id".into());

        // Can update the fragments in parallel.
        let patched = Arc::new(PatchSink::default());
        let new_fragments = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = JoinSet::new();
        let task_limit = dataset.object_store.as_ref().io_parallelism();
        let reservation =
            MemoryConsumer::new("MergeInsert").register(session_ctx.task_ctx().memory_pool());

        // Best-effort removal of uncommitted files after a mid-update failure.
        // Aborts in-flight tasks first, then deletes the new fragments written
        // so far (including ones routed to target bases). Column-patch files
        // from completed tasks stay in primary storage where regular dataset
        // cleanup can reclaim them.
        async fn cleanup_on_failure(
            dataset: &Dataset,
            target_bases_info: &Option<Vec<TargetBaseInfo>>,
            new_fragments: &Mutex<Vec<Fragment>>,
            tasks: &mut JoinSet<Result<usize>>,
        ) {
            tasks.shutdown().await;
            let written = new_fragments.lock().unwrap().clone();
            cleanup_data_fragments(
                &dataset.object_store,
                &dataset.base,
                target_bases_info.as_deref(),
                &written,
            )
            .await;
        }

        loop {
            let (frag_id, batches) = match group_stream.next().await.transpose() {
                Ok(Some(group)) => group,
                Ok(None) => break,
                Err(e) => {
                    cleanup_on_failure(&dataset, &target_bases_info, &new_fragments, &mut tasks)
                        .await;
                    return Err(e.into());
                }
            };
            async fn handle_fragment(
                dataset: Arc<Dataset>,
                fragment: FileFragment,
                mut metadata: Fragment,
                mut batches: Vec<RecordBatch>,
                patched: Arc<PatchSink>,
                reservation_size: usize,
                versions: (u64, ConcreteFileVersion),
            ) -> Result<usize> {
                let (current_version, write_version) = versions;
                // batches still have _rowaddr
                let write_schema = batches[0]
                    .schema()
                    .as_ref()
                    .without_column(ROW_ADDR)
                    .without_column(ROW_ID);
                let write_schema = dataset.schema().project_by_schema(
                    &write_schema,
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?;

                let updated_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();

                // This function is here to help rustc with lifetimes.
                fn get_row_addr_iter(
                    batches: &[RecordBatch],
                ) -> impl Iterator<Item = (u64, (usize, usize))> + '_ + Send {
                    batches.iter().enumerate().flat_map(|(batch_idx, batch)| {
                        // The index in source batches will be one more.
                        let batch_idx = batch_idx + 1;
                        let row_addrs = batch
                            .column_by_name(ROW_ADDR)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        row_addrs
                            .values()
                            .iter()
                            .enumerate()
                            .map(move |(offset, row_addr)| (*row_addr, (batch_idx, offset)))
                    })
                }

                let has_full_fragment_coverage = metadata.deletion_file.is_none()
                    && Some(updated_rows) == metadata.physical_rows
                    && get_row_addr_iter(&batches)
                        .map(|(row_addr, _)| row_addr)
                        .eq(RowAddress::address_range(metadata.id as u32).take(updated_rows));

                // Record which offsets this fragment patched before the write
                // paths below consume `_rowaddr`. Only stable row ids read the
                // row-version metadata these drive, so skip the work otherwise.
                if dataset.manifest.uses_stable_row_ids() {
                    let offsets: RoaringBitmap = get_row_addr_iter(&batches)
                        .map(|(row_addr, _)| RowAddress::from(row_addr).row_offset())
                        .collect();
                    patched
                        .offsets
                        .lock()
                        .unwrap()
                        .entry(metadata.id)
                        .or_default()
                        .extend(offsets);
                }

                // A key-only match still updates row-version metadata, but
                // has no column data to write or invalidate.
                if write_schema.fields.is_empty() {
                    if dataset.manifest.uses_stable_row_ids() {
                        let updated_offsets: Vec<usize> = get_row_addr_iter(&batches)
                            .map(|(row_addr, _)| RowAddress::from(row_addr).row_offset() as usize)
                            .collect();
                        let spilled_lineage =
                            load_spilled_row_lineage(&dataset, [&metadata]).await?;
                        lance_table::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                            &mut metadata,
                            &updated_offsets,
                            current_version,
                            dataset.manifest.version,
                            &spilled_lineage,
                        )?;
                    }
                    // Keep the matched fragments in the conflict read set even
                    // without stable row ids: a concurrent rewrite of their
                    // physical addresses must cause the merge to retry.
                    patched
                        .fragments
                        .lock()
                        .map_err(|err| {
                            Error::internal(format!("Failed to lock patched fragments: {err}"))
                        })?
                        .push(metadata);
                    return Ok(reservation_size);
                }

                if has_full_fragment_coverage {
                    // Exact, deletion-free coverage can be written directly because the
                    // batches are sorted by row address.

                    let mut seeds = SeedCollector::for_write(
                        write_version,
                        Some(&dataset),
                        &write_schema,
                        true,
                    )
                    .await?;
                    let mut writer =
                        versions::open_update_writer(write_version, &dataset, &write_schema, false)
                            .await?;

                    // We need to remove rowaddr before writing. The data file
                    // writer and the seed collector must then see the same
                    // representation: a Utf8View or BinaryView source is stored
                    // as its classic offset type, and the writer's own
                    // conversion never reaches these batches, so normalize once
                    // here for both.
                    batches.iter_mut().try_for_each(|batch| {
                        let without_addr = batch.drop_column(ROW_ADDR)?;
                        *batch = SchemaAdapter::new(without_addr.schema())
                            .to_physical_batch(without_addr)?;
                        Ok::<_, Error>(())
                    })?;

                    let source_version = metadata
                        .referenced_lance_files()
                        .next()
                        .map(|file| file.file_version())
                        .transpose()?
                        .unwrap_or_else(|| {
                            dataset.manifest.data_storage_format.lance_file_format()
                        });
                    if let Some(batch_size) =
                        versions::row_group_size_for_rewrite(source_version, &fragment).await?
                    {
                        // Need to match the existing batch size exactly, otherwise
                        // we'll get errors.
                        let stream = stream::iter(batches.into_iter().map(Ok));
                        let stream = Box::pin(RecordBatchStreamAdapter::new(
                            Arc::new((&write_schema).into()),
                            stream,
                        ));
                        let mut stream = chunk_stream(stream, batch_size as usize);
                        while let Some(chunk) = stream.next().await {
                            let chunk = chunk?;
                            writer.write(&chunk).await?;
                            for batch in &chunk {
                                seeds.observe(batch)?;
                            }
                        }
                    } else {
                        writer.write(batches.as_slice()).await?;
                        for batch in &batches {
                            seeds.observe(batch)?;
                        }
                    }

                    seeds.flush(writer.as_mut()).await?;
                    let (_num_rows, data_file) = writer.finish().await?;

                    metadata.files.push(data_file);

                    if dataset.manifest.uses_stable_row_ids() {
                        // in-place frag override: refresh row-level latest update version meta
                        lance_table::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                            &mut metadata,
                            current_version,
                        )?;
                    }

                    patched.fragments.lock().unwrap().push(metadata);
                } else {
                    // TODO: we could skip scanning row addresses we don't need.
                    let update_schema = batches[0].schema();
                    let read_columns = update_schema.field_names();
                    let mut updater = fragment
                        .updater_with_version(
                            Some(&read_columns),
                            Some((write_schema, dataset.schema().clone())),
                            None,
                            None,
                            write_version,
                        )
                        .await?;

                    // We will use interleave to update the rows. The first batch
                    // will be the original source data, and all subsequent batches
                    // will be updates.
                    let mut source_batches = Vec::with_capacity(batches.len() + 1);
                    // Convert Arrow JSON columns (Utf8) to Lance JSON (LargeBinary) so every
                    // batch is in physical format, matching what the updater reads from the
                    // fragment. `convert_json_columns` is a no-op clone when there is nothing
                    // to convert, so it can be applied unconditionally. The first entry is a
                    // placeholder for the source data (overwritten each iteration below); it
                    // must be converted too, otherwise its schema would diverge from the rest.
                    source_batches.push(convert_json_columns(&batches[0]).map_err(Error::from)?);
                    for batch in &batches {
                        let dropped = batch.drop_column(ROW_ADDR)?;
                        source_batches.push(convert_json_columns(&dropped).map_err(Error::from)?);
                    }

                    let mut updated_rows =
                        UpdatedRowAddrReconciler::new(get_row_addr_iter(&batches));

                    while let Some(batch) = updater.next().await? {
                        source_batches[0] =
                            batch.project_by_schema(source_batches[1].schema().as_ref())?;

                        let original_row_addrs = batch
                            .column_by_name(ROW_ADDR)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let indices = updated_rows.reconcile_batch(original_row_addrs.values())?;

                        let updated_batch = interleave_batches(&source_batches, &indices)?;

                        updater.update(updated_batch).await?;
                    }
                    updated_rows.finish()?;

                    let mut updated_fragment = updater.finish().await?;

                    if dataset.manifest.uses_stable_row_ids() {
                        // in-place frag partial rows update, do the in-place refresh the frag's row_latest_update_version_meta
                        // via compute updated local row offsets and write row-level version meta
                        let mut updated_offsets: Vec<usize> = Vec::new();
                        for b in batches.iter() {
                            let row_addrs = b
                                .column_by_name(ROW_ADDR)
                                .unwrap()
                                .as_any()
                                .downcast_ref::<UInt64Array>()
                                .unwrap();
                            updated_offsets.extend(
                                row_addrs
                                    .values()
                                    .iter()
                                    .map(|addr| RowAddress::from(*addr).row_offset() as usize),
                            );
                        }
                        updated_offsets.sort_unstable();
                        updated_offsets.dedup();

                        // The fragment's existing versions may be spilled to a
                        // data file, which the refresh cannot read itself.
                        let spilled_lineage =
                            load_spilled_row_lineage(&dataset, [&updated_fragment]).await?;
                        lance_table::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                            &mut updated_fragment,
                            &updated_offsets,
                            current_version,
                            dataset.manifest.version,
                            &spilled_lineage,
                        )?;
                    }

                    patched.fragments.lock().unwrap().push(updated_fragment);
                }
                Ok(reservation_size)
            }

            async fn handle_new_fragments(
                dataset: Arc<Dataset>,
                batches: Vec<RecordBatch>,
                new_fragments: Arc<Mutex<Vec<Fragment>>>,
                reservation_size: usize,
                target_bases_info: Arc<Option<Vec<TargetBaseInfo>>>,
                write_version: ConcreteFileVersion,
                write_params: Arc<WriteParams>,
            ) -> Result<usize> {
                // Batches still have _rowaddr (used elsewhere to merge with existing data)
                // We need to remove it before writing to Lance files.
                let num_fields = batches[0].schema().fields().len();
                let mut projection = Vec::with_capacity(num_fields - 1);
                for (i, field) in batches[0].schema().fields().iter().enumerate() {
                    if field.name() != ROW_ADDR {
                        projection.push(i);
                    }
                }
                let write_schema = Arc::new(batches[0].schema().project(&projection).unwrap());

                let batches = batches
                    .into_iter()
                    .map(move |batch| batch.project(&projection));
                let reader = RecordBatchIterator::new(batches, write_schema.clone());
                let stream = reader_to_stream(Box::new(reader));

                let write_schema = dataset.schema().project_by_schema(
                    write_schema.as_ref(),
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?;

                let (fragments, _) = write_fragments_internal(
                    write_version,
                    Some(dataset.as_ref()),
                    dataset.object_store.clone(),
                    &dataset.base,
                    write_schema,
                    stream,
                    write_params.as_ref().clone(),
                    (*target_bases_info).clone(),
                )
                .await?;

                new_fragments.lock().unwrap().extend(fragments);
                Ok(reservation_size)
            }
            // We shouldn't need much more memory beyond what is already in the batches.
            let mut memory_size = batches
                .iter()
                .map(|batch| batch.get_array_memory_size())
                .sum();

            loop {
                let have_additional_cpus = tasks.len() < task_limit;
                if have_additional_cpus {
                    if reservation.try_grow(memory_size).is_ok() {
                        break;
                    } else if tasks.is_empty() {
                        // If there are no tasks running, we can bypass the pool limits.
                        // This lets us handle the case where we have a single large batch.
                        memory_size = 0;
                        break;
                    }
                    // If we can't grow the reservation, we will wait for a task to finish
                }

                if let Some(res) = tasks.join_next().await {
                    match res.map_err(Error::from).and_then(|size| size) {
                        Ok(size) => reservation.shrink(size),
                        Err(e) => {
                            cleanup_on_failure(
                                &dataset,
                                &target_bases_info,
                                &new_fragments,
                                &mut tasks,
                            )
                            .await;
                            return Err(e);
                        }
                    }
                }
            }

            match frag_id.first() {
                Some(ScalarValue::UInt64(Some(frag_id))) => {
                    let frag_id = *frag_id;
                    let Some(fragment) = dataset.get_fragment(frag_id as usize) else {
                        error!(
                            fragment_id = frag_id,
                            dataset_uri = %dataset.uri(),
                            manifest_version = dataset.manifest().version,
                            manifest_path = %dataset.manifest_location().path,
                            branch = ?dataset.manifest().branch,
                            "Non-existent fragment id returned from merge result",
                        );
                        cleanup_on_failure(
                            &dataset,
                            &target_bases_info,
                            &new_fragments,
                            &mut tasks,
                        )
                        .await;
                        return Err(Error::internal(format!(
                            "Got non-existent fragment id from merge result: {} (uri={}, version={}, manifest={}, branch={})",
                            frag_id,
                            dataset.uri(),
                            dataset.manifest().version,
                            dataset.manifest_location().path,
                            dataset.manifest().branch.as_deref().unwrap_or("main"),
                        )));
                    };
                    let metadata = fragment.metadata.clone();

                    // Matched join columns are equal by definition. Retain
                    // them in the source only for the new-fragment branch.
                    let batches = batches
                        .into_iter()
                        .map(|batch| batch.project(&update_projection))
                        .collect::<std::result::Result<Vec<_>, _>>()?;

                    let fut = handle_fragment(
                        dataset.clone(),
                        fragment,
                        metadata,
                        batches,
                        patched.clone(),
                        memory_size,
                        (current_version, write_version),
                    );
                    tasks.spawn(fut);
                }
                Some(ScalarValue::Null | ScalarValue::UInt64(None)) => {
                    let fut = handle_new_fragments(
                        dataset.clone(),
                        batches,
                        new_fragments.clone(),
                        memory_size,
                        target_bases_info.clone(),
                        write_version,
                        write_params.clone(),
                    );
                    tasks.spawn(fut);
                }
                _ => {
                    cleanup_on_failure(&dataset, &target_bases_info, &new_fragments, &mut tasks)
                        .await;
                    return Err(Error::internal(format!(
                        "Got non-fragment id from merge result: {:?}",
                        frag_id
                    )));
                }
            };
        }

        while let Some(res) = tasks.join_next().await {
            match res.map_err(Error::from).and_then(|size| size) {
                Ok(size) => reservation.shrink(size),
                Err(e) => {
                    cleanup_on_failure(&dataset, &target_bases_info, &new_fragments, &mut tasks)
                        .await;
                    return Err(e);
                }
            }
        }
        let PatchSink {
            fragments: updated_fragments,
            offsets: matched_offsets,
        } = Arc::try_unwrap(patched).unwrap();
        let mut updated_fragments = updated_fragments.into_inner().unwrap();
        let matched_offsets = matched_offsets.into_inner().unwrap();

        // We keep track of all fields that are updated so we can prune the indices.
        // We could maybe be more precise since some fields are not modified in some
        // fragments (if they were already null) but this is simpler and good enough
        // for now.
        let mut all_fields_updated = HashSet::new();

        // Collect the updated fragments, and map the field ids. Tombstone old ones
        // as needed.
        for fragment in &mut updated_fragments {
            if !has_updated_columns {
                continue;
            }
            let updated_fields = fragment.files.last().unwrap().fields.clone();
            all_fields_updated.extend(updated_fields.iter().map(|&f| f as u32));
            for data_file in &mut fragment.files.iter_mut().rev().skip(1) {
                let new_fields: Arc<[i32]> = data_file
                    .fields
                    .iter()
                    .map(|field| {
                        if updated_fields.contains(field) {
                            -2 // Tombstone
                        } else {
                            *field
                        }
                    })
                    .collect::<Vec<_>>()
                    .into();
                data_file.fields = new_fields;
            }
        }

        // Data files record physical leaf fields, while an index can be attached to
        // a logical parent such as a list. Include every affected ancestor so index
        // coverage is pruned for the complete logical column that was rewritten.
        let directly_updated_fields = all_fields_updated.iter().copied().collect::<Vec<_>>();
        for field_id in directly_updated_fields {
            if let Some(ancestry) = dataset.schema().field_ancestry_by_id(field_id as i32) {
                all_fields_updated.extend(ancestry.into_iter().map(|field| field.id as u32));
            }
        }

        let new_fragments = Arc::try_unwrap(new_fragments)
            .unwrap()
            .into_inner()
            .unwrap();

        Ok(PatchedFragments {
            updated_fragments,
            new_fragments,
            fields_modified: all_fields_updated.into_iter().collect(),
            matched_offsets: UpdatedFragmentOffsets(matched_offsets),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::write::merge_insert::job::merge_insert_execution_options;

    use arrow_array::record_batch;

    use crate::dataset::InsertBuilder;
    use lance_datafusion::exec::new_session_context;

    #[test]
    fn test_updated_row_addr_missing_between_target_rows() {
        let row_addr = |offset| u64::from(RowAddress::new_from_parts(3, offset));
        let mut updated_rows = UpdatedRowAddrReconciler::new([(row_addr(1), (1, 0))].into_iter());

        let error = updated_rows
            .reconcile_batch(&[row_addr(0), row_addr(2)])
            .unwrap_err();

        assert!(matches!(error, Error::Internal { .. }));
        let message = error.to_string();
        assert!(message.contains("update row address (3, 1) is missing"));
        assert!(message.contains("next target row address is (3, 2)"));
    }

    #[test]
    fn test_updated_row_addr_missing_after_target_rows() {
        let row_addr = |offset| u64::from(RowAddress::new_from_parts(7, offset));
        let mut updated_rows = UpdatedRowAddrReconciler::new([(row_addr(2), (1, 0))].into_iter());

        assert_eq!(
            updated_rows
                .reconcile_batch(&[row_addr(0), row_addr(1)])
                .unwrap(),
            vec![(0, 0), (0, 1)]
        );
        let error = updated_rows.finish().unwrap_err();

        assert!(matches!(error, Error::Internal { .. }));
        let message = error.to_string();
        assert!(message.contains("update row address (7, 2) is missing"));
        assert!(message.contains("no target rows remain"));
    }

    #[tokio::test]
    async fn test_updated_row_addr_missing_in_full_fragment_update() {
        let initial = record_batch!(("value", Int32, [10, 20])).unwrap();
        let dataset = Arc::new(
            InsertBuilder::new("memory://")
                .execute(vec![initial])
                .await
                .unwrap(),
        );
        let fragment_id = dataset.get_fragments()[0].id() as u32;
        let row_addr = |offset| u64::from(RowAddress::new_from_parts(fragment_id, offset));
        let updates = record_batch!(
            (ROW_ADDR, UInt64, [row_addr(0), row_addr(2)]),
            ("value", Int32, [100, 200])
        )
        .unwrap();
        let update_stream =
            RecordBatchStreamAdapter::new(updates.schema(), futures::stream::iter([Ok(updates)]));
        let spill_execution_options = merge_insert_execution_options();
        let spill_session_context = new_session_context(&spill_execution_options);

        let error = MergeInsertJob::update_fragments(
            dataset.clone(),
            Box::pin(update_stream),
            None,
            &spill_session_context,
            &spill_execution_options,
            (
                dataset.manifest().version + 1,
                dataset.manifest.data_storage_format.lance_file_format(),
            ),
            &[],
            Arc::new(WriteParams::default()),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, Error::Internal { .. }));
        let message = error.to_string();
        assert!(message.contains("update row address (0, 2) is missing"));
        assert!(message.contains("no target rows remain"));
    }
}
