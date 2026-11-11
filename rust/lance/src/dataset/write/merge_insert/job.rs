// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::assign_action::{
    condition_reads_target, merge_insert_action, parse_delete_condition, parse_update_condition,
};
use super::inserted_rows::{KeyExistenceFilter, KeyExistenceFilterBuilder};

use super::merger::Merger;
use super::row_ids;
use super::{
    DeduplicatingSourcePartitionStream, INDEXED_JOIN_BATCH_ROWS, MAX_FRAGMENT_UPDATE_BATCH_BYTES,
    MAX_INDEXED_JOIN_BATCH_BYTES, MERGE_ACTION_COLUMN, MERGE_SOURCE_SENTINEL, MergeInsertJob,
    MergeInsertWriteMode, MergeStats, PatchedFragments, SchemaComparison, SourceDedupeBehavior,
    UncommittedMergeInsert, WhenMatched, WhenNotMatchedBySource, canonical_source_schema,
    check_source_schema, created_blob_payloads, exec, fill_missing_target_columns, logical_plan,
    one_shot_provider, resolve_target_bases, target_blob_handling, uncommitted_transaction,
};
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::transaction::UpdateMode::{RewriteColumns, RewriteRows};
use crate::dataset::write::cleanup_data_fragments;
use crate::dataset::write::retry::{RetryConfig, RetryExecutor, execute_with_retry};
use crate::dataset::write::{CommitBuilder, write_fragments_internal};
use crate::index::DatasetIndexExt;
use crate::index::visibility::IndexVisibility;
use crate::{
    Dataset,
    datafusion::dataframe::{LanceTableProvider, SessionContextExt},
    dataset::{
        transaction::{Operation, Transaction},
        versions,
        write::merge_insert::logical_plan::{MergeInsertPlanner, WriteSink},
    },
    index::loading::DatasetIndexInternalExt,
    io::exec::{
        RowAddrMaskFilterExec, project,
        scalar_index::{IndexLookup, MapIndexExec},
    },
};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use datafusion::common::NullEquality;
use datafusion::error::DataFusionError;
use datafusion::{
    catalog::{TableProvider, streaming::StreamingTable},
    datasource::MemTable,
    execution::context::{SessionConfig, SessionContext},
    logical_expr::{self, Expr, Extension, JoinType, LogicalPlan},
    physical_plan::{
        ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
        coalesce_partitions::CoalescePartitionsExec, display::DisplayableExecutionPlan,
        joins::SortMergeJoinExec, projection::ProjectionExec, sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter, streaming::PartitionStream, union::UnionExec,
    },
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
    prelude::DataFrame,
};
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column};
use futures::{
    StreamExt, TryStreamExt,
    stream::{self},
};
use lance_arrow::{RecordBatchExt, SchemaExt};
use lance_core::datatypes::NullabilityComparison;
use lance_core::{
    Error, ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, ROW_ID_FIELD, Result,
    datatypes::{OnMissing, SchemaCompareOptions},
    utils::tokio::get_num_compute_intensive_cpus,
};
use lance_datafusion::{
    exec::{
        HardCapBatchSizeExec, LanceExecutionOptions, OneShotExec, analyze_plan,
        execute_plan_with_session_context, new_session_context, provider_to_stream,
    },
    spill::{materialize_stream, spilling_table_provider_with_job_disk_manager},
    utils::StreamingWriteSource,
};
use lance_index::IndexCriteria;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata, RowIdMeta};
use lance_table::io::deletion::deletion_file_path;
use log::info;
use roaring::RoaringTreemap;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
};

/// Whether `field` is a blob or has one anywhere beneath it.
///
/// `Field::children` is populated for structs, lists, and maps alike, so this
/// covers a blob reached through any nesting — not just a direct struct member.
fn subtree_has_blob(field: &lance_core::datatypes::Field) -> bool {
    field.is_blob() || field.children.iter().any(subtree_has_blob)
}

impl MergeInsertJob {
    /// Clone one execution attempt while preserving the spill manager owned by
    /// the surrounding retry loop.
    fn clone_for_retry(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            write_params: self.write_params.clone(),
            spill_session_context: self.spill_session_context.clone(),
            spill_execution_options: self.spill_execution_options.clone(),
        }
    }

    pub async fn execute_reader(
        self,
        source: impl StreamingWriteSource,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let stream = source.into_stream();
        self.execute(stream).await
    }

    fn check_compatible_schema(&self, schema: &Schema) -> Result<SchemaComparison> {
        let lance_schema: lance_core::datatypes::Schema = schema.try_into()?;
        let target_schema = self.dataset.schema();

        let version = self.params.write_version(&self.dataset);
        let mut options = versions::schema_compare_options(version);
        options.compare_nullability = NullabilityComparison::Ignore;
        // Merge columns are matched by name, so a complete source remains a
        // full-schema merge even when the caller orders its fields differently.
        options.ignore_field_order = true;

        // Try full schema match first.
        if check_source_schema(&lance_schema, target_schema, &options).is_ok() {
            return Ok(SchemaComparison::FullCompatible);
        }

        // If full match fails, try subschema match.
        options.allow_subschema = true;

        check_source_schema(&lance_schema, target_schema, &options)
            .map(|_| SchemaComparison::Subschema)
    }

    /// Collect every join column that has a scalar index supporting exact
    /// equality.
    ///
    /// For a single-column join this matches the previous behavior. For a
    /// multi-column (composite key) join, every indexed column contributes
    /// an additional `IsIn` probe inside one [`MapIndexExec`]: their AND
    /// yields the row addresses where every indexed column matches some
    /// source value. The downstream hash join still filters by the full
    /// composite key, so unindexed columns simply do not prune the
    /// candidate set — they are checked by the post-filter.
    ///
    /// Returns an empty vec when no join column has a usable scalar index;
    /// callers should then fall through to the full-scan path.
    async fn indexed_join_keys(&self) -> Result<Vec<(String, IndexMetadata)>> {
        let mut indexed = Vec::with_capacity(self.params.on.len());
        for col in &self.params.on {
            if let Some(idx) = self
                .dataset
                .load_scalar_index(
                    IndexCriteria::default()
                        .for_column(col)
                        // Unclear if this would work if the index does not support exact equality
                        .supports_exact_equality(),
                )
                .await?
            {
                indexed.push((col.clone(), idx));
            }
        }
        Ok(indexed)
    }

    /// Fragments that cannot be reached by every index in `indexed_keys`
    /// and therefore must be scanned separately and unioned in alongside
    /// the indexed take.  A fragment is "reachable" by the composite probe
    /// only if it is in the intersection of all the indices' fragment
    /// bitmaps; everything else falls into the unindexed set.
    async fn unindexed_fragments_for_keys(
        &self,
        indexed_keys: &[(String, IndexMetadata)],
    ) -> Result<Vec<Fragment>> {
        let mut unindexed: HashMap<u64, Fragment> = HashMap::new();
        for (_, index) in indexed_keys {
            for frag in self.dataset.unindexed_fragments(&index.name).await? {
                unindexed.entry(frag.id).or_insert(frag);
            }
        }
        Ok(unindexed.into_values().collect())
    }

    /// Resolve stale join keys using the same segment policy as scalar filtering.
    async fn overlay_stale_rows_for_keys(
        &self,
        indexed_keys: &[(String, IndexMetadata)],
    ) -> Result<IndexVisibility> {
        IndexVisibility::for_scalar_indices(
            self.dataset.clone(),
            &indexed_keys
                .iter()
                .map(|(column, index)| (column.as_str(), index.name.as_str()))
                .collect::<Vec<_>>(),
            self.dataset.fragments(),
            None,
        )
        .await
    }

    async fn create_indexed_scan_joined_stream(
        &self,
        source_provider: Arc<dyn TableProvider>,
        source_mode: SourceProviderMode,
        indexed_keys: Vec<(String, IndexMetadata)>,
        source_projection: Option<Arc<Schema>>,
    ) -> Result<SendableRecordBatchStream> {
        // This relies on a few non-standard physical operators and so we cannot use the
        // datafusion dataframe API and need to construct the plan manually :'(
        debug_assert!(
            !indexed_keys.is_empty(),
            "create_indexed_scan_joined_stream requires at least one indexed key"
        );
        let schema = source_projection
            .clone()
            .unwrap_or_else(|| source_provider.schema());
        let add_row_addr = match self.check_compatible_schema(&schema)? {
            SchemaComparison::FullCompatible => false,
            SchemaComparison::Subschema => true,
        };

        // The sort-merge join groups source rows by key. FirstSeen still needs
        // equal-key rows in their original encounter order, so add a temporary
        // ordinal before the source is replayed or sorted.
        let source_ordinal_column =
            (self.params.source_dedupe_behavior == SourceDedupeBehavior::FirstSeen).then(|| {
                let mut name = "__merge_source_ordinal".to_string();
                while schema.column_with_name(&name).is_some() {
                    name.push('_');
                }
                name
            });
        let execution_options = self.spill_execution_options.clone();
        let max_batch_bytes = ((execution_options.mem_pool_size() as usize) / 8)
            .clamp(1, MAX_INDEXED_JOIN_BATCH_BYTES);
        let session_ctx = &self.spill_session_context;
        let disk_manager = session_ctx.runtime_env().disk_manager.clone();
        let ordinal_field = source_ordinal_column
            .as_ref()
            .map(|name| Field::new(name, DataType::UInt64, false));

        // Materialized providers guarantee that repeated scans observe the same
        // rows. Arbitrary providers do not, so normalize one scan per attempt
        // into a bounded replay before the index probe and final join diverge.
        let (source_input, index_input) = if source_mode.has_stable_snapshot() {
            let source_scan = SequentialSourcePartitionStream::from_provider_scan(
                &source_provider,
                session_ctx,
                ordinal_field.clone(),
                source_projection.clone(),
            )
            .await?;
            let index_scan = SequentialSourcePartitionStream::from_provider_scan(
                &source_provider,
                session_ctx,
                None,
                source_projection.clone(),
            )
            .await?;
            (
                source_scan
                    .scan(&session_ctx.state(), None, &[], None)
                    .await?,
                index_scan
                    .scan(&session_ctx.state(), None, &[], None)
                    .await?,
            )
        } else {
            let normalized_source = SequentialSourcePartitionStream::from_provider_scan(
                &source_provider,
                session_ctx,
                ordinal_field,
                source_projection.clone(),
            )
            .await?;
            let source = provider_to_stream(normalized_source).await?;
            let replayable_source = spilling_table_provider_with_job_disk_manager(
                source,
                max_batch_bytes,
                disk_manager.clone(),
            )
            .await?;
            (
                replayable_source
                    .scan(&session_ctx.state(), None, &[], None)
                    .await?,
                replayable_source
                    .scan(&session_ctx.state(), None, &[], None)
                    .await?,
            )
        };

        // 3 - Probe every indexed join column.  For composite keys this is
        //     the AND of one `IsIn` query per indexed column, which yields
        //     a tighter candidate set than probing a single column.  The
        //     downstream join still filters by the full composite key,
        //     so unindexed `on` columns simply do not prune the candidates.
        let lookup_fields = indexed_keys
            .iter()
            .map(|(col, _)| Ok(schema.field_with_name(col)?.clone()))
            .collect::<Result<Vec<_>>>()?;
        let index_mapper_input = Arc::new(project(index_input, &Schema::new(lookup_fields))?);

        let lookups = indexed_keys
            .iter()
            .map(|(col, idx)| IndexLookup::new(col.clone(), idx.name.clone()))
            .collect::<Vec<_>>();
        let overlay_stale_rows = self.overlay_stale_rows_for_keys(&indexed_keys).await?;
        let mut index_mapper: Arc<dyn ExecutionPlan> = Arc::new(MapIndexExec::new_multi(
            self.dataset.clone(),
            lookups,
            index_mapper_input,
        ));
        if !overlay_stale_rows.is_empty() {
            // The fallback scan below reads these rows using their current keys. Block
            // old index hits so a row cannot enter the target side of the join twice.
            let stale_row_ids = overlay_stale_rows.result_rows().await?.clone();
            index_mapper = Arc::new(RowAddrMaskFilterExec::new(
                index_mapper,
                Arc::new(RowAddrMask::from_block(stale_row_ids)),
            ));
        }

        // Finish the compact index probe before sorting wide source/target
        // rows. Otherwise its retained candidate set competes with both sorts
        // for the same pool. Replaying these row addresses is inexpensive and
        // spills when needed.
        let mapped_rows = execute_plan_with_session_context(
            index_mapper,
            execution_options.clone(),
            session_ctx,
        )?;
        let mapped_rows = spilling_table_provider_with_job_disk_manager(
            mapped_rows,
            max_batch_bytes,
            disk_manager,
        )
        .await?;
        let mut initial_scan = provider_to_stream(mapped_rows.clone()).await?;
        while initial_scan.try_next().await?.is_some() {}
        drop(initial_scan);
        let index_mapper = mapped_rows
            .scan(&session_ctx.state(), None, &[], None)
            .await?;

        // 4 - Take the mapped row ids (TakeExec stays for legacy storage:
        //     the v1 reader cannot serve a FilteredReadExec)
        let projection = self
            .dataset
            .empty_projection()
            .union_arrow_schema(schema.as_ref(), OnMissing::Error)?;
        let mut target = versions::merge_insert_indexed_take(
            self.dataset
                .manifest()
                .data_storage_format
                .lance_file_format(),
            self.dataset.clone(),
            index_mapper,
            projection,
            add_row_addr,
        )?;

        // 5 - Take puts the row id and row addr at the beginning.  A full scan (used when there is
        //     no scalar index) puts the row id and addr at the end.  We need to match these up so
        //     we reorder those columns at the end.
        let schema = target.schema();
        let mut columns = schema
            .fields()
            .iter()
            .filter(|f| f.name() != ROW_ID && f.name() != ROW_ADDR)
            .cloned()
            .collect::<Vec<_>>();
        columns.push(Arc::new(ROW_ID_FIELD.clone()));
        if add_row_addr {
            columns.push(Arc::new(ROW_ADDR_FIELD.clone()));
        }
        target = Arc::new(project(target, &Schema::new(columns))?);

        let column_names = schema
            .field_names()
            .into_iter()
            .filter(|name| name.as_str() != ROW_ID && name.as_str() != ROW_ADDR)
            .collect::<Vec<_>>();

        // 5a - Scan unindexed fragments and overlay-stale rows using current values.
        //      A row can be reached by the composite index probe only if it
        //      lives in a fragment covered by *every* chosen index, so the
        //      "unindexed" set is the union of fragments missing from any
        //      one of them.
        let unindexed_fragments = self.unindexed_fragments_for_keys(&indexed_keys).await?;
        if !unindexed_fragments.is_empty() || !overlay_stale_rows.is_empty() {
            let mut builder = self.dataset.scan();
            if add_row_addr {
                builder.with_row_address();
            }
            if overlay_stale_rows.is_empty() {
                builder.with_fragments(unindexed_fragments);
            } else {
                let mut fallback_rows = overlay_stale_rows.physical_selection();
                for fragment in unindexed_fragments {
                    fallback_rows.insert_fragment(fragment.id as u32);
                }
                builder.with_physical_row_addr_prefilter(fallback_rows);
            }
            let fallback_data = builder
                .with_row_id()
                .project(&column_names)?
                .create_plan()
                .await?;
            let unioned = UnionExec::try_new(vec![target, fallback_data])?;
            // Enforce only 1 partition. RepartitionExec would take a share of the fair pool
            // from the target sort, and until DataFusion 56 spilling from two inputs can
            // deadlock it (apache/datafusion#24883).
            target = Arc::new(CoalescePartitionsExec::new(unioned));
        }

        // We need to prefix the fields in the target with target_ so that we don't have any duplicate
        // field names (DF doesn't support this as of version 44)
        target = Self::prefix_columns_phys(target, "target_");

        // 6 - Join the source against the taken target rows on the full
        //     composite key.  Probing the index produces a super-set of the
        //     actual matches (when not every key column has an index, or
        //     even when they do — the per-column `IsIn` lists do not
        //     correlate values across the tuple), so this join is what
        //     trims candidates down to the exact composite-key matches.
        let on_keys = self
            .params
            .on
            .iter()
            .map(|col| {
                let source_key = Column::new_with_schema(col, source_input.schema().as_ref())?;
                let target_key =
                    Column::new_with_schema(&format!("target_{}", col), target.schema().as_ref())?;
                Ok::<_, Error>((
                    Arc::new(source_key) as Arc<dyn PhysicalExpr>,
                    Arc::new(target_key) as Arc<dyn PhysicalExpr>,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // HashJoinExec cannot spill its build side. Sort both inputs instead,
        // capping oversized batches so the sorts and the join can spill within
        // the configured fair memory pool.
        let sort_options = vec![SortOptions::default(); on_keys.len()];
        let mut source_ordering = on_keys
            .iter()
            .zip(sort_options.iter())
            .map(|((source_key, _), options)| PhysicalSortExpr {
                expr: source_key.clone(),
                options: *options,
            })
            .collect::<Vec<_>>();
        if let Some(source_ordinal_column) = source_ordinal_column.as_ref() {
            source_ordering.push(PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema(
                    source_ordinal_column,
                    source_input.schema().as_ref(),
                )?),
                options: SortOptions::default(),
            });
        }
        let target_ordering = on_keys
            .iter()
            .zip(sort_options.iter())
            .map(|((_, target_key), options)| PhysicalSortExpr {
                expr: target_key.clone(),
                options: *options,
            })
            .collect::<Vec<_>>();
        let source_ordering = LexOrdering::new(source_ordering)
            .ok_or_else(|| Error::internal("merge-insert source join ordering is empty"))?;
        let target_ordering = LexOrdering::new(target_ordering)
            .ok_or_else(|| Error::internal("merge-insert target join ordering is empty"))?;
        let target_partition_count = target.properties().output_partitioning().partition_count();
        if target_partition_count != 1 {
            return Err(Error::internal(format!(
                "indexed merge-insert target must have exactly one partition before sorting, got {target_partition_count}"
            )));
        }
        let source_input = Arc::new(HardCapBatchSizeExec::new(source_input, max_batch_bytes));
        let target = Arc::new(HardCapBatchSizeExec::new(target, max_batch_bytes));
        // A sort spilled to several runs keeps a merge reservation until its output is read,
        // and that reservation grows with the number of runs, up to the whole pool. Finish
        // the source sort first, so that each sort has the pool to itself.
        let sorted_source = execute_plan_with_session_context(
            Arc::new(SortExec::new(source_ordering, source_input)),
            execution_options.clone(),
            session_ctx,
        )?;
        let (sorted_source, _) = materialize_stream(
            sorted_source,
            session_ctx.runtime_env(),
            execution_options.mem_pool_size() as usize / 8,
            "materializing sorted merge insert source",
        )
        .await?;
        let source_input = Arc::new(OneShotExec::new(sorted_source));
        let target = Arc::new(SortExec::new(target_ordering, target));
        let joined: Arc<dyn ExecutionPlan> = Arc::new(SortMergeJoinExec::try_new(
            source_input,
            target,
            on_keys,
            None,
            JoinType::Full,
            sort_options,
            NullEquality::NullEqualsNothing,
        )?);
        let joined = if let Some(source_ordinal_column) = source_ordinal_column {
            let output_schema = joined
                .schema()
                .as_ref()
                .without_column(&source_ordinal_column);
            Arc::new(project(joined, &output_schema)?) as Arc<dyn ExecutionPlan>
        } else {
            joined
        };
        execute_plan_with_session_context(joined, execution_options, session_ctx)
    }

    fn prefix_columns(df: DataFrame, prefix: &str) -> DataFrame {
        let schema = df.schema();
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                // Need to "quote" the column name so it gets interpreted case-sensitively
                logical_expr::col(format!("\"{}\"", f.name())).alias(format!(
                    "{}{}",
                    prefix,
                    f.name()
                ))
            })
            .collect::<Vec<_>>();
        df.select(columns).unwrap()
    }

    fn prefix_columns_phys(inp: Arc<dyn ExecutionPlan>, prefix: &str) -> Arc<dyn ExecutionPlan> {
        let schema = inp.schema();
        let exprs = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, f)| {
                let col = Arc::new(Column::new(f.name(), idx)) as Arc<dyn PhysicalExpr>;
                let new_name = format!("{}{}", prefix, f.name());
                (col, new_name)
            })
            .collect::<Vec<_>>();
        Arc::new(ProjectionExec::try_new(exprs, inp).unwrap())
    }

    // If the join keys are not indexed then we need to do a full scan of the table
    async fn create_full_table_joined_stream(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let session_config = SessionConfig::default().with_target_partitions(1);
        let session_ctx = SessionContext::new_with_config(session_config);
        let schema = source.schema();
        let new_data = session_ctx.read_one_shot(source)?;
        let join_cols = self
            .params
            .on // columns to join on
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>(); // vector of strings of col names to join
        let target_cols = self
            .params
            .on
            .iter()
            .map(|c| format!("target_{}", c))
            .collect::<Vec<_>>();
        let target_cols = target_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>();

        match self.check_compatible_schema(&schema)? {
            SchemaComparison::FullCompatible => {
                let existing = session_ctx.read_lance(self.dataset.clone(), true, false)?;
                // We need to rename the columns from the target table so that they don't conflict with the source table
                let existing = Self::prefix_columns(existing, "target_");
                let joined =
                    new_data.join(existing, JoinType::Full, &join_cols, &target_cols, None)?; // full join
                Ok(joined.execute_stream().await?)
            }
            SchemaComparison::Subschema => {
                let existing = session_ctx.read_lance(self.dataset.clone(), true, true)?;
                let columns = schema
                    .field_names()
                    .iter()
                    .map(|s| s.as_str())
                    .chain([ROW_ID, ROW_ADDR])
                    .collect::<Vec<_>>();
                let projected = existing.select_columns(&columns)?;
                // We need to rename the columns from the target table so that they don't conflict with the source table
                let projected = Self::prefix_columns(projected, "target_");
                // We aren't supporting inserts or deletes right now, so we can use inner join
                let join_type = if self.params.insert_not_matched {
                    JoinType::Left
                } else {
                    JoinType::Inner
                };
                let joined = new_data.join(projected, join_type, &join_cols, &target_cols, None)?;
                Ok(joined.execute_stream().await?)
            }
        }
    }

    /// Join the source and target data streams
    ///
    /// If there is a scalar index on the join key, we can use it to do an indexed join.  Otherwise we need to do
    /// a full outer join.
    ///
    /// Datafusion doesn't allow duplicate column names so during this join we rename the columns from target and
    /// prefix them with _target.
    async fn create_joined_stream(
        &self,
        source_provider: Arc<dyn TableProvider>,
        source_mode: SourceProviderMode,
        source_projection: Option<Arc<Schema>>,
    ) -> Result<SendableRecordBatchStream> {
        if self.params.use_index
            && matches!(
                self.params.delete_not_matched_by_source,
                WhenNotMatchedBySource::Keep
            )
        {
            // keeping unmatched rows, no deletion. Use the indexed-scan path
            // only when EVERY join column is indexed; a partially-indexed
            // composite key under-matches there (the probe can't resolve the
            // full tuple), so fall through to the correct full-table join.
            let indexed_keys = self.indexed_join_keys().await?;
            if indexed_keys.len() == self.params.on.len() {
                return self
                    .create_indexed_scan_joined_stream(
                        source_provider,
                        source_mode,
                        indexed_keys,
                        source_projection,
                    )
                    .await;
            }
        }

        if !matches!(
            self.params.delete_not_matched_by_source,
            WhenNotMatchedBySource::Keep
        ) {
            info!(
                "The merge insert operation is configured to delete rows from the target table, this requires a potentially costly full table scan"
            );
        }

        let source = provider_to_stream(source_provider).await?;
        let source = if let Some(projection_schema) = source_projection {
            let output_schema = projection_schema.clone();
            let projected = source.map(move |batch| {
                batch.and_then(|batch| {
                    batch
                        .project_by_schema(projection_schema.as_ref())
                        .map_err(DataFusionError::from)
                })
            });
            Box::pin(RecordBatchStreamAdapter::new(output_schema, projected))
                as SendableRecordBatchStream
        } else {
            source
        };
        self.create_full_table_joined_stream(source).await
    }

    /// Executes the merge insert job from a one-shot stream source.
    ///
    /// This will take in the source, merge it with the existing target data, and insert new
    /// rows, update existing rows, and delete existing rows.
    ///
    /// A stream can only be read once, so when `conflict_retries > 0` the stream is
    /// spilled (in memory, then to disk) so it can be replayed on each retry. See
    /// [`super::MergeInsertBuilder::spill_for_retry`] to fail fast instead, and
    /// [`Self::execute_batches`] / [`Self::execute_provider`] for sources that can
    /// be scanned again on each retry.
    pub async fn execute(
        self,
        source: SendableRecordBatchStream,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let (provider, source_mode) = self.stream_source_to_provider(source).await?;
        self.execute_inner(provider, source_mode).await
    }

    /// Executes the merge insert job from a re-scannable [`TableProvider`].
    ///
    /// This is the canonical entry point: [`Self::execute`] and
    /// [`Self::execute_batches`] are thin wrappers that build a provider and call
    /// this method. Retries re-read the provider directly. Within each indexed
    /// attempt, one scan is captured in a bounded replay so the index probe and
    /// final join cannot observe different source snapshots. The provider's
    /// reported statistics (e.g. from a [`MemTable`] or file source) also let
    /// DataFusion optimize non-indexed merge joins.
    ///
    /// [`MemTable`]: datafusion::datasource::MemTable
    pub async fn execute_provider(
        self,
        provider: Arc<dyn TableProvider>,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        // TableProvider supports repeated scans, but does not guarantee that
        // separate scans observe the same rows.
        self.execute_inner(provider, SourceProviderMode::Rescannable)
            .await
    }

    fn dataset_source_provider(source: Arc<Dataset>) -> Arc<dyn TableProvider> {
        Arc::new(
            LanceTableProvider::new(source, false, false)
                .with_blob_handling(lance_core::datatypes::BlobHandling::default()),
        )
    }

    /// Executes the merge insert job from a native Lance dataset source.
    ///
    /// The dataset handle is scanned directly through [`LanceTableProvider`],
    /// including on conflict retries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    /// # use std::sync::Arc;
    /// # async fn example(source: Arc<Dataset>, target: Arc<Dataset>) -> Result<()> {
    /// let (merged, _stats) =
    ///     MergeInsertBuilder::try_new(target.clone(), vec!["id".to_string()])?
    ///         .when_matched(WhenMatched::UpdateAll)
    ///         .when_not_matched(WhenNotMatched::InsertAll)
    ///         .try_build()?
    ///         .execute_dataset(source.clone())
    ///         .await?;
    /// let _ = merged;
    ///
    /// let plan = MergeInsertBuilder::try_new(target, vec!["id".to_string()])?
    ///     .when_matched(WhenMatched::UpdateAll)
    ///     .when_not_matched(WhenNotMatched::InsertAll)
    ///     .try_build()?
    ///     .analyze_plan_dataset(source)
    ///     .await?;
    /// let _ = plan;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_dataset(self, source: Arc<Dataset>) -> Result<(Arc<Dataset>, MergeStats)> {
        self.execute_provider(Self::dataset_source_provider(source))
            .await
    }

    /// Executes the merge insert job from materialized record batches.
    ///
    /// The batches are wrapped in an in-memory [`MemTable`], which is re-scannable
    /// (retries replay from memory, never spilling) and reports exact statistics to
    /// the merge join. This is the preferred entry point when the full source is
    /// already in memory.
    pub async fn execute_batches(
        self,
        batches: Vec<RecordBatch>,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let provider = self.batches_to_provider(batches)?;
        self.execute_inner(provider, SourceProviderMode::StableSnapshot)
            .await
    }

    /// Like [`Self::execute_batches`] but returns the uncommitted transaction.
    ///
    /// Use [`CommitBuilder`] to commit the returned transaction.
    pub async fn execute_uncommitted_batches(
        self,
        batches: Vec<RecordBatch>,
    ) -> Result<UncommittedMergeInsert> {
        let provider = self.batches_to_provider(batches)?;
        self.execute_uncommitted_impl(provider, SourceProviderMode::StableSnapshot)
            .await
    }

    /// Like [`Self::execute_uncommitted_batches`] for a native Lance dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// # use datafusion::catalog::TableProvider;
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    /// # use std::sync::Arc;
    /// # async fn example(
    /// #     source: Arc<Dataset>,
    /// #     provider: Arc<dyn TableProvider>,
    /// #     target: Arc<Dataset>,
    /// # ) -> Result<()> {
    /// let dataset_result =
    ///     MergeInsertBuilder::try_new(target.clone(), vec!["id".to_string()])?
    ///         .when_matched(WhenMatched::UpdateAll)
    ///         .when_not_matched(WhenNotMatched::InsertAll)
    ///         .try_build()?
    ///         .execute_uncommitted_dataset(source)
    ///         .await?;
    ///
    /// let provider_result =
    ///     MergeInsertBuilder::try_new(target, vec!["id".to_string()])?
    ///         .when_matched(WhenMatched::UpdateAll)
    ///         .when_not_matched(WhenNotMatched::InsertAll)
    ///         .try_build()?
    ///         .execute_uncommitted_provider(provider)
    ///         .await?;
    /// let _ = (dataset_result, provider_result);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted_dataset(
        self,
        source: Arc<Dataset>,
    ) -> Result<UncommittedMergeInsert> {
        self.execute_uncommitted_impl(
            Self::dataset_source_provider(source),
            SourceProviderMode::Rescannable,
        )
        .await
    }

    /// Like [`Self::execute_uncommitted_batches`] for a re-scannable provider.
    ///
    /// See [`Self::execute_uncommitted_dataset`] for a compiling example.
    pub async fn execute_uncommitted_provider(
        self,
        provider: Arc<dyn TableProvider>,
    ) -> Result<UncommittedMergeInsert> {
        self.execute_uncommitted_impl(provider, SourceProviderMode::Rescannable)
            .await
    }

    /// Wrap materialized batches in a multi-partition in-memory [`MemTable`].
    fn batches_to_provider(&self, batches: Vec<RecordBatch>) -> Result<Arc<dyn TableProvider>> {
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| Arc::new(Schema::from(self.dataset.schema())));
        // FirstSeen needs a defined encounter order. Keep materialized batches in
        // their caller-provided order; other modes retain parallel source scans.
        let partitions = if self.params.source_dedupe_behavior == SourceDedupeBehavior::FirstSeen {
            vec![batches]
        } else {
            Self::batches_into_partitions(batches)
        };
        Ok(Arc::new(MemTable::try_new(schema, partitions)?))
    }

    /// Distribute batches round-robin across up to `num_compute_intensive_cpus`
    /// partitions, so a [`MemTable`] built from them can be scanned in parallel.
    /// Always returns at least one (possibly empty) partition so an empty source
    /// still produces a valid provider.
    fn batches_into_partitions(batches: Vec<RecordBatch>) -> Vec<Vec<RecordBatch>> {
        let num_partitions = batches.len().min(get_num_compute_intensive_cpus()).max(1);
        let mut partitions = vec![Vec::new(); num_partitions];
        for (idx, batch) in batches.into_iter().enumerate() {
            partitions[idx % num_partitions].push(batch);
        }
        partitions
    }

    /// Wrap a one-shot stream source in a provider, returning whether it can be
    /// replayed across retries.
    ///
    /// With retries enabled and spilling allowed, the stream is drained into a
    /// replayable spill (memory up to 100MB, then disk). Otherwise the stream is
    /// wrapped in a non-replayable one-shot provider and any conflict fails fast.
    async fn stream_source_to_provider(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<(Arc<dyn TableProvider>, SourceProviderMode)> {
        if self.params.conflict_retries > 0 && self.params.spill_for_retry {
            // Allow buffering up to 100MB in memory before spilling to disk.
            let disk_manager = self
                .spill_session_context
                .runtime_env()
                .disk_manager
                .clone();
            let provider = spilling_table_provider_with_job_disk_manager(
                source,
                100 * 1024 * 1024,
                disk_manager,
            )
            .await?;
            Ok((provider, SourceProviderMode::StableSnapshot))
        } else {
            Ok((one_shot_provider(source)?, SourceProviderMode::OneShot))
        }
    }

    /// Run the retry loop against a provider, re-scanning it on each attempt.
    ///
    /// `source_mode` indicates whether the provider can be scanned more than once
    /// and whether those scans share one materialized snapshot. One-shot sources
    /// disable retries; arbitrary providers are re-scanned once per attempt.
    async fn execute_inner(
        self,
        provider: Arc<dyn TableProvider>,
        source_mode: SourceProviderMode,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let dataset = self.dataset.clone();
        let config = RetryConfig {
            max_retries: if source_mode.supports_retries() {
                self.params.conflict_retries
            } else {
                0
            },
            retry_timeout: self.params.retry_timeout,
        };

        let wrapper = MergeInsertJobWithProvider {
            job: self,
            provider,
            source_mode,
            attempt_count: Arc::new(AtomicU32::new(0)),
        };

        Box::pin(execute_with_retry(wrapper, dataset, config)).await
    }

    /// Execute the merge insert job without committing the changes.
    ///
    /// Use [`CommitBuilder`] to commit the returned transaction.
    pub async fn execute_uncommitted(
        self,
        source: impl StreamingWriteSource,
    ) -> Result<UncommittedMergeInsert> {
        let stream = source.into_stream();
        self.execute_uncommitted_impl(one_shot_provider(stream)?, SourceProviderMode::OneShot)
            .await
    }

    fn create_plan_join_type(&self) -> JoinType {
        let keep_unmatched_source_rows = self.params.insert_not_matched;
        let keep_unmatched_target_rows = !matches!(
            self.params.delete_not_matched_by_source,
            WhenNotMatchedBySource::Keep
        );

        match (keep_unmatched_target_rows, keep_unmatched_source_rows) {
            (false, false) => JoinType::Inner,
            (false, true) => JoinType::Right,
            (true, false) => JoinType::Left,
            (true, true) => JoinType::Full,
        }
    }

    /// Whether the source provides a dataset column with a blob anywhere in it.
    fn source_carries_blob(&self, source_schema: &Schema) -> bool {
        self.dataset
            .schema()
            .fields
            .iter()
            .filter(|field| source_schema.column_with_name(&field.name).is_some())
            .any(subtree_has_blob)
    }

    /// Resolves the caller's [`MergeInsertWriteMode`] against this operation.
    ///
    /// [`WriteSink::RewriteColumns`] only ever replaces column data within a
    /// fragment, so it cannot express anything that moves or removes rows:
    /// inserting unmatched source rows, deleting matched rows, and deleting
    /// target rows unmatched by the source all need [`WriteSink::RewriteRows`].
    /// Nor is it worth using when there is nothing to save, which is why a
    /// source covering every dataset column or carrying nothing but the join key
    /// is left on the row-rewrite path.
    ///
    /// Asking for it explicitly on an operation it cannot express is an error
    /// rather than a silent fallback: the fallback would write orders of
    /// magnitude more bytes than the caller asked for, with no signal.
    fn select_write_sink(&self, source_schema: &Schema) -> Result<WriteSink> {
        // Every merge insert wrote whole rows before column patching existed, so
        // that is what `Auto` still resolves to.
        if self.params.write_mode != MergeInsertWriteMode::RewriteColumns {
            return Ok(WriteSink::RewriteRows);
        }

        let mut blockers: Vec<&str> = Vec::new();

        if !self
            .dataset
            .schema()
            .fields
            .iter()
            .any(|field| source_schema.column_with_name(&field.name).is_none())
        {
            blockers.push("the source covers every dataset column, so there is nothing to skip");
        }
        // A source of nothing but the join key has no new column values to
        // write, so there is no column patch for this sink to perform.
        if !source_schema
            .fields()
            .iter()
            .any(|field| !self.params.on.iter().any(|key| key == field.name()))
        {
            blockers.push("the source carries no column besides the join key");
        }
        if !matches!(
            self.params.when_matched,
            WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_)
        ) {
            blockers.push("when_matched must update rather than delete or do nothing");
        }
        if self.params.insert_not_matched {
            blockers.push("inserting unmatched source rows adds rows, which patching cannot do");
        }
        if !matches!(
            self.params.delete_not_matched_by_source,
            WhenNotMatchedBySource::Keep
        ) {
            blockers.push("deleting target rows unmatched by the source removes rows");
        }
        // Patching a blob column would have to go through the fragment updater,
        // which reads blobs in their stored descriptor form rather than the
        // logical one the source provides. Leave those on the row-rewrite path,
        // which already converts between the two representations.
        //
        // The source can only name top-level columns, so the check is rooted at
        // the top-level fields it carries and then descends: a blob nested
        // anywhere under one of them (struct member, list item, map value) is
        // still patched by writing that whole top-level column.
        if self.source_carries_blob(source_schema) {
            blockers.push("the source carries a blob column, whose stored form differs from the one it provides");
        }

        if blockers.is_empty() {
            Ok(WriteSink::RewriteColumns)
        } else {
            Err(Error::invalid_input(format!(
                "MergeInsertWriteMode::RewriteColumns cannot express this merge insert: {}. \
                 Use MergeInsertWriteMode::Auto or RewriteRows instead.",
                blockers.join("; ")
            )))
        }
    }

    async fn create_plan(self, provider: Arc<dyn TableProvider>) -> Result<Arc<dyn ExecutionPlan>> {
        // Goal: we shouldn't manually have to specify which columns to scan.
        //       DataFusion's optimizer should be able to automatically perform
        //       projection pushdown for us.
        // Goal: we shouldn't have to add new branches in this code to handle
        //       indexed vs non-indexed cases. That should be handled by optimizer rules.
        let write_sink = self.select_write_sink(provider.schema().as_ref())?;
        let session_ctx = SessionContext::new();
        let mut target_provider =
            crate::datafusion::dataframe::LanceTableProvider::new_with_ordering(
                self.dataset.clone(),
                true,
                true,
                false,
            )
            .with_blob_handling(target_blob_handling(&self.dataset));
        if write_sink == WriteSink::RewriteRows
            && self
                .params
                .carried_lineage_spill_budget(&self.dataset)?
                .is_some()
        {
            target_provider = target_provider.with_row_created_at_version()?;
        }
        let scan = session_ctx.read_table(Arc::new(target_provider))?;
        // Wrap column names in double quotes to preserve case (DataFusion lowercases unquoted identifiers)
        let on_cols = self
            .params
            .on
            .iter()
            .map(|name| format!("\"{}\"", name))
            .collect::<Vec<_>>();
        let on_cols_refs = on_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        // FirstSeen must observe the caller's source order even though the join can
        // reorder batches. Deduplicating source partitions sequentially before
        // the join fixes the winner at that contract boundary. Other modes plan
        // directly against the provider so its statistics reach the optimizer.
        let deduplicate_source =
            self.params.source_dedupe_behavior == SourceDedupeBehavior::FirstSeen;
        let source_skipped_duplicates = Arc::new(AtomicU64::new(0));
        let source_df = if deduplicate_source {
            let source_plan = provider.scan(&session_ctx.state(), None, &[], None).await?;
            let deduplicated_partition = Arc::new(DeduplicatingSourcePartitionStream::new(
                source_plan,
                self.params.on.clone(),
                source_skipped_duplicates.clone(),
            ));
            let deduplicated_provider = Arc::new(StreamingTable::try_new(
                deduplicated_partition.schema().clone(),
                vec![deduplicated_partition],
            )?);
            session_ctx.read_table(deduplicated_provider)?
        } else {
            session_ctx.read_table(provider)?
        };
        // Capture the source field names *before* aliasing / joining so we
        // can tell which dataset columns are missing from the source and
        // need to be filled from the target side of the join below.
        let source_field_names: std::collections::HashSet<String> = source_df
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let source_blob_columns = self
            .dataset
            .schema()
            .fields
            .iter()
            .filter(|field| {
                source_field_names.contains(&field.name)
                    && crate::dataset::blob::field_contains_blob_v2(field)
            })
            .map(|field| field.name.clone())
            .collect();
        // Inject a sentinel literal column so we can reliably determine, after the join,
        // whether the source side contributed a row.  This is NULL-safe: even when every
        // ON column is NULL the sentinel lets us distinguish a source-only row from a
        // target-only row (where the sentinel is filled with NULL by the outer join).
        let source_df = source_df
            .with_column(MERGE_SOURCE_SENTINEL, logical_expr::lit(true))
            .map_err(crate::Error::from)?;
        let source_df_aliased = source_df.alias("source")?;
        let scan_aliased = scan.alias("target")?;
        let join_type = self.create_plan_join_type();
        let dataset_schema: Schema = self.dataset.schema().into();
        // A condition that reads target columns is evaluated after the join
        // on the rows it gates (`MergeConditionExec`), which takes only the
        // columns its input lacks; folded into `__action` it would pull them
        // into the target scan.
        let split = |condition: Option<Expr>| match condition {
            Some(condition) if condition_reads_target(&condition, self.dataset.schema()) => {
                (None, Some(condition))
            }
            condition => (condition, None),
        };
        let (update_condition, deferred_update_if) =
            split(parse_update_condition(&self.params, &dataset_schema)?);
        let (delete_condition, deferred_delete_if) = split(parse_delete_condition(&self.params)?);
        let mut df = scan_aliased
            .join(
                source_df_aliased,
                join_type,
                &on_cols_refs,
                &on_cols_refs,
                None,
            )?
            .with_column(
                MERGE_ACTION_COLUMN,
                merge_insert_action(&self.params, update_condition, delete_condition)?,
            )?;

        // Partial-schema upsert: fill the dataset columns the source does not
        // carry from the target side of the join.
        //
        // `RewriteColumns` patches the source columns into the fragments that
        // already hold the matched rows, so the missing columns keep their
        // stored values and must not be filled here. Skipping the fill is also
        // what keeps them out of the target scan's projection.
        //
        // Only an update writes back a row that already exists, so only it
        // needs the stored values. Every other row written is an insert, whose
        // target side the join leaves NULL, so without an update the columns are
        // filled with NULL and not scanned. A deferred `DeleteIf` reads its
        // columns through the fill (`MergeConditionExec`), so those are still
        // filled from the target.
        if write_sink == WriteSink::RewriteRows {
            // Exhaustive, so a new variant has to decide whether it updates.
            let updates = match self.params.when_matched {
                WhenMatched::UpdateAll
                | WhenMatched::UpdateIf(_)
                | WhenMatched::UpdateIfExpr(_) => true,
                WhenMatched::DoNothing | WhenMatched::Fail | WhenMatched::Delete => false,
            };
            let delete_if_columns = deferred_delete_if
                .iter()
                .flat_map(|condition| condition.column_refs())
                .map(|column| column.name.as_str())
                .collect::<HashSet<_>>();
            df = fill_missing_target_columns(df, &dataset_schema, &source_field_names, |name| {
                updates || delete_if_columns.contains(name)
            })?;
        }

        let (session_state, logical_plan) = df.into_parts();

        let write_node = logical_plan::MergeInsertWriteNode::new(
            logical_plan,
            self.dataset.clone(),
            self.params.clone(),
            source_skipped_duplicates,
            write_sink,
            self.spill_session_context.clone(),
            self.spill_execution_options.clone(),
            source_blob_columns,
            logical_plan::DeferredConditions {
                update_if: deferred_update_if,
                delete_if: deferred_delete_if,
            },
        );
        let logical_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(write_node),
        });

        let logical_plan = session_state.optimize(&logical_plan)?;

        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(MergeInsertPlanner {
                write_params: self.write_params.clone(),
            })]);
        // This method already does the optimization for us.
        let physical_plan = planner
            .create_physical_plan(&logical_plan, &session_state)
            .await?;

        Ok(physical_plan)
    }

    async fn execute_uncommitted_v2(
        self,
        provider: Arc<dyn TableProvider>,
    ) -> Result<(
        Transaction,
        MergeStats,
        Option<RowAddrTreeMap>,
        Option<KeyExistenceFilter>,
    )> {
        let plan = self.create_plan(provider).await?;

        // Execute the plan
        // Assert that we have exactly one partition since we're designed for single-partition execution
        let partition_count = plan.properties().output_partitioning().partition_count();

        if partition_count != 1 {
            return Err(Error::invalid_input(format!(
                "Expected exactly 1 partition, got {}",
                partition_count
            )));
        }

        // Execute partition 0 (the only partition)
        let task_context = Arc::new(datafusion::execution::TaskContext::default());
        let mut stream = plan.execute(0, task_context)?;

        // Assert that the execution produces no output (this is a write operation)
        if let Some(batch) = stream.next().await {
            let batch = batch?;
            if batch.num_rows() > 0 {
                return Err(Error::invalid_input(format!(
                    "Expected no output from write operation, got {} rows",
                    batch.num_rows()
                )));
            }
        }

        // Extract merge stats from the execution plan
        let (stats, transaction, affected_rows, inserted_rows_filter) = if let Some(full_exec) =
            plan.downcast_ref::<exec::FullSchemaMergeInsertExec>()
        {
            let stats = full_exec.merge_stats().ok_or_else(|| {
                Error::internal("Merge stats not available - execution may not have completed")
            })?;
            let transaction = full_exec.transaction().ok_or_else(|| {
                Error::internal("Transaction not available - execution may not have completed")
            })?;
            let affected_rows = full_exec.affected_rows().map(RowAddrTreeMap::from);
            let inserted_rows_filter = full_exec.inserted_rows_filter();
            (stats, transaction, affected_rows, inserted_rows_filter)
        } else if let Some(in_place_exec) = plan.downcast_ref::<exec::InPlaceMergeInsertExec>() {
            let stats = in_place_exec.merge_stats().ok_or_else(|| {
                Error::internal("Merge stats not available - execution may not have completed")
            })?;
            let transaction = in_place_exec.transaction().ok_or_else(|| {
                Error::internal("Transaction not available - execution may not have completed")
            })?;
            // An in-place patch rewrites column data rather than only touching
            // deletion files, so the affected rows cannot be expressed as a row
            // address set (same reasoning as the legacy in-place path).
            (stats, transaction, None, None)
        } else if let Some(delete_exec) = plan.downcast_ref::<exec::DeleteOnlyMergeInsertExec>() {
            let stats = delete_exec.merge_stats().ok_or_else(|| {
                Error::internal("Merge stats not available - execution may not have completed")
            })?;
            let transaction = delete_exec.transaction().ok_or_else(|| {
                Error::internal("Transaction not available - execution may not have completed")
            })?;
            let affected_rows = delete_exec.affected_rows().map(RowAddrTreeMap::from);
            (stats, transaction, affected_rows, None)
        } else {
            return Err(Error::internal(
                "Expected FullSchemaMergeInsertExec, InPlaceMergeInsertExec or DeleteOnlyMergeInsertExec",
            ));
        };

        Ok((transaction, stats, affected_rows, inserted_rows_filter))
    }

    /// Check if the merge insert operation can use the fast path (create_plan).
    ///
    /// The fast path is available when:
    /// - `when_matched` is `UpdateAll`, `UpdateIf`, `Fail`, `Delete`, or `DoNothing`
    /// - Either `use_index` is false OR there's no scalar index on the join key
    /// - The source schema is either (a) the full dataset schema, or (b) a
    ///   subset of it (partial-schema upsert), or (c) just the key columns for
    ///   delete-only operations
    /// - `when_not_matched_by_source` is `Keep`, `Delete`, or `DeleteIf`
    ///
    /// For partial-schema upserts with `insert_not_matched=true`, every missing
    /// target column must be nullable — otherwise this method returns an
    /// `InvalidInput` error, because inserted rows would otherwise attempt to
    /// write a non-nullable NULL downstream.
    async fn can_use_create_plan(&self, source_schema: &Schema) -> Result<bool> {
        // Convert to lance schema for comparison
        let lance_schema = lance_core::datatypes::Schema::try_from(source_schema)?;
        let full_schema = self.dataset.schema();
        let is_full_schema = check_source_schema(
            &lance_schema,
            full_schema,
            &SchemaCompareOptions {
                compare_metadata: false,
                // Allow nullable source fields for non-nullable targets.
                compare_nullability: NullabilityComparison::Ignore,
                // Allow columns to be in a different order; they will be matched by name.
                ignore_field_order: true,
                ..Default::default()
            },
        )
        .is_ok();

        // Partial-schema upsert: every source field must exist in the target
        // and have a compatible data type (legacy blob input matches Blob v2,
        // as in `check_source_schema`). Missing target columns will be
        // filled from the target side of the join in `create_plan`.
        let is_subset_schema = !is_full_schema
            && lance_schema.fields.iter().all(|sf| {
                full_schema
                    .field(&sf.name)
                    .map(|tf| {
                        tf.data_type() == sf.data_type()
                            || (tf.is_blob_v2() && sf.is_blob() && !sf.is_blob_v2())
                    })
                    .unwrap_or(false)
            });

        // If the user is inserting unmatched rows with a partial source, any
        // target column missing from the source would receive NULL for those
        // inserts. Non-nullable targets cannot accept that, so reject early
        // with a descriptive error instead of failing later in the writer.
        if is_subset_schema && self.params.insert_not_matched {
            let non_nullable_missing: Vec<&str> = full_schema
                .fields
                .iter()
                .filter(|tf| lance_schema.field(&tf.name).is_none() && !tf.nullable)
                .map(|tf| tf.name.as_str())
                .collect();
            if !non_nullable_missing.is_empty() {
                return Err(Error::invalid_input(format!(
                    "Cannot insert rows with a partial-schema source: target column(s) \
                     {:?} are non-nullable and not provided by the source. Either add \
                     them to the source or set when_not_matched to DoNothing.",
                    non_nullable_missing
                )));
            }
        }

        // A partial-schema source that both deletes matched rows and inserts
        // unmatched rows cannot be expressed by the indexed-scan delete path
        // (the delete cannot be folded into a partial write). Keep it off the
        // scalar-index route so it falls through to the v2 plan, which handles
        // delete + insert directly.
        let is_partial_delete_with_insert = is_subset_schema
            && self.params.insert_not_matched
            && matches!(self.params.when_matched, WhenMatched::Delete);

        // For a partial-schema update the indexed-scan path patches columns and
        // has no branch that rewrites whole rows, so an explicit `RewriteRows`
        // can only be honored by falling through to the v2 plan, giving up the
        // index probe on the join. Naming the write sink wins over keeping the
        // probe: the sink decides how many bytes are written, the probe only how
        // the matched rows are found. Merges that write nothing (no matched
        // update) or write whole rows on both paths (a full-schema source) are
        // unaffected, so they keep the index. A source carrying a blob column
        // needs whole rows too: patching cannot write blobs (`select_write_sink`).
        let write_mode_needs_v2 = (self.params.write_mode == MergeInsertWriteMode::RewriteRows
            || self.source_carries_blob(source_schema))
            && is_subset_schema
            && matches!(
                self.params.when_matched,
                WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_)
            );

        let would_use_scalar_index = if self.params.use_index
            && !is_partial_delete_with_insert
            && !write_mode_needs_v2
            && matches!(
                self.params.delete_not_matched_by_source,
                WhenNotMatchedBySource::Keep
            ) {
            // Only when EVERY join column is indexed. A partially-indexed
            // composite key cannot be fully resolved by the index probe and the
            // indexed-scan path then under-matches (e.g. a delete silently
            // no-ops), so it must fall through to the correct full path.
            self.indexed_join_keys().await?.len() == self.params.on.len()
        } else {
            false
        };

        // Check if this is a delete-only operation (no update/insert writes needed from source)
        // For delete-only, we don't need the full source schema, just key columns for matching
        let no_upsert = matches!(
            self.params.when_matched,
            WhenMatched::Delete | WhenMatched::DoNothing
        ) && !self.params.insert_not_matched;

        // For delete-only, verify source has all key columns
        let source_has_key_columns = self.params.on.iter().all(|key| {
            source_schema
                .fields()
                .iter()
                .any(|f| f.name() == key.as_str())
        });
        let schema_ok = is_full_schema || is_subset_schema || (no_upsert && source_has_key_columns);

        Ok(matches!(
            self.params.when_matched,
            WhenMatched::UpdateAll
                | WhenMatched::UpdateIf(_)
                | WhenMatched::UpdateIfExpr(_)
                | WhenMatched::Fail
                | WhenMatched::Delete
                | WhenMatched::DoNothing
        ) && !would_use_scalar_index
            && schema_ok
            && matches!(
                self.params.delete_not_matched_by_source,
                WhenNotMatchedBySource::Keep
                    | WhenNotMatchedBySource::Delete
                    | WhenNotMatchedBySource::DeleteIf(_)
            ))
    }

    /// Build the uncommitted merge-insert transaction, using the v2 plan when
    /// it supports the source and the legacy path otherwise.
    async fn execute_uncommitted_impl(
        self,
        provider: Arc<dyn TableProvider>,
        source_mode: SourceProviderMode,
    ) -> Result<UncommittedMergeInsert> {
        // Resolve the write mode before the path fork. The v2 plan resolves it
        // again in `create_plan`, but the legacy path below does not go through
        // that, and its partial-schema branch patches columns whatever the mode
        // says. Resolving here means an unexpressible `RewriteColumns` is
        // rejected on either path rather than only on one.
        self.select_write_sink(provider.schema().as_ref())?;

        // Check if we can use the fast path
        let can_use_fast_path = self.can_use_create_plan(provider.schema().as_ref()).await?;

        if can_use_fast_path {
            let (transaction, stats, affected_rows, inserted_rows_filter) =
                self.execute_uncommitted_v2(provider).await?;
            return Ok(UncommittedMergeInsert {
                transaction,
                affected_rows,
                stats,
                inserted_rows_filter,
            });
        }

        let target_bases_info = resolve_target_bases(&self.dataset, &self.params).await?;

        let source_schema = provider.schema();
        let lance_schema = lance_core::datatypes::Schema::try_from(source_schema.as_ref())?;
        let full_schema = self.dataset.schema();
        let is_full_schema = check_source_schema(
            &lance_schema,
            full_schema,
            &SchemaCompareOptions {
                compare_metadata: false,
                // Allow nullable source fields for non-nullable targets.
                compare_nullability: NullabilityComparison::Ignore,
                // Keep this classification consistent with `can_use_create_plan`
                // and `check_compatible_schema`: merge columns match by name.
                ignore_field_order: true,
                ..Default::default()
            },
        )
        .is_ok();
        let source_projection = if is_full_schema {
            Some(Arc::new(canonical_source_schema(
                source_schema.as_ref(),
                &Schema::from(full_schema),
            )?))
        } else {
            None
        };
        let source_schema = source_projection.clone().unwrap_or(source_schema);
        let joined = self
            .create_joined_stream(provider, source_mode, source_projection)
            .await?;
        let inserted_rows_filter = self
            .params
            .primary_key_field_ids(full_schema)
            .filter(|_| self.params.insert_not_matched)
            .map(KeyExistenceFilterBuilder::new);
        let merger = Merger::try_new(
            self.params.clone(),
            source_schema,
            joined.schema().as_ref(),
            !is_full_schema,
            self.dataset.manifest.uses_stable_row_ids(),
            inserted_rows_filter,
        )?;
        let merge_statistics = merger.merge_stats.clone();
        let deleted_rows = merger.deleted_rows.clone();
        let updating_row_ids = merger.updating_row_ids.clone();
        let inserted_rows_filter = merger.inserted_rows_filter.clone();
        let order_updates_first = merger.with_row_id;
        let merger_schema = merger.output_schema().clone();
        let stream = joined
            .and_then(move |batch| merger.clone().execute_batch(batch))
            .try_flatten();
        let stream: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(merger_schema, stream));
        let stream = if order_updates_first {
            row_ids::updates_before_inserts(
                stream,
                self.spill_session_context
                    .runtime_env()
                    .disk_manager
                    .clone(),
            )
            .await?
        } else {
            stream
        };

        // A partial-schema source can patch columns or delete matched rows, but
        // not both in one commit: the writer rejects subschema rows up front and
        // the delete cannot be folded into the write. Reject that combination.
        if !is_full_schema
            && matches!(self.params.when_matched, WhenMatched::Delete)
            && self.params.insert_not_matched
        {
            return Err(Error::not_supported_source("Combining when_matched(Delete) with inserts from a partial-schema source is not supported; provide the full target schema in the source".into()));
        }

        // The commit strategy follows what the merge does to matched rows. A
        // pure delete (no inserts) writes nothing: the merger emits no batches
        // and only records the matched row ids. This holds for any source schema
        // width, so it is keyed on the operation rather than `is_full_schema`.
        let is_delete_only = matches!(self.params.when_matched, WhenMatched::Delete)
            && !self.params.insert_not_matched;

        let (operation, affected_rows, inserted_rows_filter) = if is_delete_only {
            // Consume the stream so the merger records the matched row ids in
            // `deleted_rows`; it produces no batches.
            let drained: Vec<RecordBatch> = stream.try_collect().await?;
            debug_assert!(drained.is_empty(), "delete-only merge must not emit rows");

            let removed_row_ids = Arc::into_inner(deleted_rows).unwrap().into_inner().unwrap();
            let removed_row_addr_vec =
                if let Some(row_id_index) = get_row_id_index(&self.dataset).await? {
                    let mut addresses = Vec::with_capacity(removed_row_ids.len());
                    for id in &removed_row_ids {
                        if let Some(address) = row_id_index.get(*id)? {
                            addresses.push(address.into());
                        }
                    }
                    addresses
                } else {
                    removed_row_ids
                };
            let removed_row_addrs = RoaringTreemap::from_iter(removed_row_addr_vec);

            let (updated_fragments, removed_fragment_ids) =
                Self::apply_deletions(&self.dataset, &removed_row_addrs).await?;

            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: self.params.compacted_sstables.clone(),
                mem_wal_dir_id: self.params.mem_wal_dir_id.clone(),
                fields_for_preserving_frag_bitmap: full_schema
                    .fields_pre_order()
                    .map(|f| f.id as u32)
                    .collect(),
                update_mode: Some(RewriteRows),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            };

            let affected_rows = Some(RowAddrTreeMap::from(removed_row_addrs));
            (operation, affected_rows, None)
        } else if !is_full_schema {
            // Non-delete partial-schema merge: patch the provided columns into
            // existing fragments in place. (Delete is handled above; a wider
            // full source takes the row-rewrite branch below.)
            if !matches!(
                self.params.delete_not_matched_by_source,
                WhenNotMatchedBySource::Keep
            ) {
                return Err(Error::not_supported_source("Deleting rows from the target table when there is no match in the source table is not supported when the source data has a different schema than the target data".into()));
            }

            // We will have a different commit path here too, as we are modifying
            // fragments rather than writing new ones
            let PatchedFragments {
                updated_fragments,
                new_fragments,
                fields_modified,
                matched_offsets,
            } = Self::update_fragments(
                self.dataset.clone(),
                stream,
                target_bases_info,
                &self.spill_session_context,
                &self.spill_execution_options,
                (
                    self.dataset.manifest.version + 1,
                    self.params.write_version(&self.dataset),
                ),
                &self.params.on,
                self.write_params.clone(),
            )
            .await?;

            let inserted_rows_filter = inserted_rows_filter
                .lock()
                .map_err(|_| Error::internal("Merge insert key filter lock poisoned"))?
                .as_ref()
                .map(KeyExistenceFilterBuilder::build);
            let operation = Operation::Update {
                removed_fragment_ids: Vec::new(),
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables: self.params.compacted_sstables.clone(),
                mem_wal_dir_id: self.params.mem_wal_dir_id.clone(),
                fields_for_preserving_frag_bitmap: vec![], // in-place update do not affect preserving frag bitmap
                update_mode: Some(RewriteColumns),
                inserted_rows_filter: inserted_rows_filter.clone(),
                // The version stamped above is a guess; carry the patched offsets
                // so `build_manifest` can re-stamp them at the real commit
                // version after a rebase.
                updated_fragment_offsets: Some(matched_offsets),
            };
            // We have rewritten the fragments, not just the deletion files, so
            // we can't use affected rows here.
            (operation, None, inserted_rows_filter)
        } else {
            let cleanup_bases = target_bases_info.clone();
            let (mut new_fragments, _) = write_fragments_internal(
                self.params.write_version(&self.dataset),
                Some(&self.dataset),
                self.dataset.object_store.clone(),
                &self.dataset.base,
                self.dataset.schema().clone(),
                stream,
                self.write_params.as_ref().clone(),
                target_bases_info,
            )
            .await?;

            // The new data files exist but are not committed yet; clean them up
            // (including files routed to target bases) if any later step fails.
            let post_write_result: Result<RoaringTreemap> = async {
                if let Some(row_id_sequence) = updating_row_ids.lock().unwrap().row_id_sequence() {
                    let fragment_sizes = new_fragments
                        .iter()
                        .map(|f| f.physical_rows.unwrap() as u64);

                    // Only updated rows carry IDs. The write stream explicitly
                    // stages updates before inserts, preserving capture order,
                    // so commit can assign fresh IDs to each fragment's tail.
                    let sequences = lance_table::rowids::rechunk_sequences(
                        [row_id_sequence.clone()],
                        fragment_sizes,
                        true,
                    )
                    .map_err(|e| {
                        Error::internal(format!(
                            "Captured row ids not equal to number of rows written: {}",
                            e
                        ))
                    })?;

                    for (fragment, sequence) in new_fragments.iter_mut().zip(sequences) {
                        let serialized = lance_table::rowids::write_row_ids(&sequence);
                        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized.into()));
                    }
                }

                // Apply deletions
                let removed_row_ids = Arc::into_inner(deleted_rows).unwrap().into_inner().unwrap();

                let removed_row_addr_vec =
                    if let Some(row_id_index) = get_row_id_index(&self.dataset).await? {
                        let mut addresses: Vec<u64> = Vec::with_capacity(removed_row_ids.len());
                        for id in &removed_row_ids {
                            if let Some(address) = row_id_index.get(*id)? {
                                addresses.push(address.into());
                            }
                        }
                        addresses
                    } else {
                        removed_row_ids
                    };

                Ok(RoaringTreemap::from_iter(removed_row_addr_vec))
            }
            .await;
            let removed_row_addrs = match post_write_result {
                Ok(removed_row_addrs) => removed_row_addrs,
                Err(e) => {
                    cleanup_data_fragments(
                        &self.dataset.object_store,
                        &self.dataset.base,
                        cleanup_bases.as_deref(),
                        &new_fragments,
                    )
                    .await;
                    return Err(e);
                }
            };

            let deletions_result = Self::apply_deletions(&self.dataset, &removed_row_addrs).await;
            let (old_fragments, removed_fragment_ids) = match deletions_result {
                Ok(v) => v,
                Err(e) => {
                    cleanup_data_fragments(
                        &self.dataset.object_store,
                        &self.dataset.base,
                        cleanup_bases.as_deref(),
                        &new_fragments,
                    )
                    .await;
                    return Err(e);
                }
            };

            // Commit updated and new fragments
            let inserted_rows_filter = inserted_rows_filter
                .lock()
                .map_err(|_| Error::internal("Merge insert key filter lock poisoned"))?
                .as_ref()
                .map(KeyExistenceFilterBuilder::build);
            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments: old_fragments,
                new_fragments,
                // On this path we only make deletions against updated_fragments and will not
                // modify any field values.
                fields_modified: vec![],
                compacted_sstables: self.params.compacted_sstables.clone(),
                mem_wal_dir_id: self.params.mem_wal_dir_id.clone(),
                fields_for_preserving_frag_bitmap: full_schema
                    .fields_pre_order()
                    .map(|f| f.id as u32)
                    .collect(),
                update_mode: Some(RewriteRows),
                inserted_rows_filter: inserted_rows_filter.clone(),
                updated_fragment_offsets: None,
            };

            let affected_rows = Some(RowAddrTreeMap::from(removed_row_addrs));
            (operation, affected_rows, inserted_rows_filter)
        };

        let stats = Arc::into_inner(merge_statistics)
            .unwrap()
            .into_inner()
            .unwrap();

        let transaction = Transaction::new(self.dataset.manifest.version, operation, None);

        Ok(UncommittedMergeInsert {
            transaction,
            affected_rows,
            stats,
            inserted_rows_filter,
        })
    }

    // Delete a batch of rows by id, returns the fragments modified and the fragments removed
    async fn apply_deletions(
        dataset: &Dataset,
        removed_row_ids: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_ids.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Box<Fragment>),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.extend_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(dataset.object_store.io_parallelism());

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }

    /// Generate the execution plan and return it as a formatted string for debugging.
    ///
    /// This method takes an optional schema representing the source data and calls `create_plan()`
    /// to generate the execution plan, then formats it for display. If no schema is provided,
    /// defaults to the dataset's schema. The verbose flag controls the level of detail shown.
    ///
    /// # Arguments
    ///
    /// * `schema` - Optional schema of the source data. If None, uses the dataset's schema
    /// * `verbose` - If true, provides more detailed information in the plan output
    ///
    /// A schema says nothing about how the source would be wrapped, so this always
    /// reports the streaming shape: the source is stood in for by an empty one-shot
    /// stream. The wrapping affects the plan, so use [`Self::analyze_plan_batches`]
    /// or [`Self::analyze_plan_provider`] when that matters. Those execute the merge
    /// to collect metrics, writing data files they delete afterwards; this method
    /// writes nothing.
    ///
    /// # Errors
    ///
    /// Returns Error::NotSupported if the merge insert configuration doesn't support
    /// the fast path required for plan generation.
    pub async fn explain_plan(&self, schema: Option<&Schema>, verbose: bool) -> Result<String> {
        // Use provided schema or default to dataset schema
        let schema = match schema {
            Some(s) => s.clone(),
            None => arrow_schema::Schema::from(self.dataset.schema()),
        };

        // Check if we can use create_plan
        if !self.can_use_create_plan(&schema).await? {
            return Err(Error::not_supported_source("This merge insert configuration does not support explain_plan: either the source schema is not one the plan path accepts, or the join takes the scalar-index execution path.".into()));
        }

        // Create an empty batch with the provided schema to pass to create_plan
        let empty_batch = RecordBatch::new_empty(Arc::new(schema.clone()));
        let stream = RecordBatchStreamAdapter::new(
            Arc::new(schema.clone()),
            futures::stream::once(async { Ok(empty_batch) }).boxed(),
        );

        // Clone self since create_plan consumes the job
        let cloned_job = self.clone();
        let plan = cloned_job
            .create_plan(one_shot_provider(Box::pin(stream))?)
            .await?;
        let display = DisplayableExecutionPlan::new(plan.as_ref());

        Ok(format!("{}", display.indent(verbose)))
    }

    /// Generate the execution plan, execute it with the provided data to collect metrics,
    /// and return the analysis.
    ///
    /// This method takes actual source data, calls `create_plan()` to generate the plan,
    /// and executes it to collect performance metrics and analysis.
    ///
    /// **Note:** This method executes the merge insert operation to collect metrics
    /// but **does not commit the changes**. Data and deletion files written during
    /// execution are deleted once the metrics are collected, and the dataset
    /// remains unchanged. This is intended for performance analysis only.
    ///
    /// # Arguments
    ///
    /// * `source` - The source data stream that would be used in the merge insert
    ///
    /// A stream reports no statistics, so the plan this returns is the streaming
    /// one. Callers holding materialized data or a source that reports statistics
    /// should use [`Self::analyze_plan_batches`] or [`Self::analyze_plan_provider`],
    /// which report the plan those sources actually run.
    ///
    /// # Errors
    ///
    /// See [`Self::analyze_plan_provider`], which this delegates to.
    pub async fn analyze_plan(&self, source: SendableRecordBatchStream) -> Result<String> {
        self.analyze_plan_provider(one_shot_provider(source)?).await
    }

    /// [`Self::analyze_plan`] for materialized batches.
    ///
    /// Mirrors [`Self::execute_batches`]: the batches are wrapped in a
    /// [`MemTable`], so the reported plan is the one an in-memory source actually
    /// runs. That plan can differ from the streaming one, because the join picks
    /// its collected side from the statistics each source reports.
    ///
    /// Under [`SourceDedupeBehavior::FirstSeen`] the source is deduplicated ahead
    /// of the join and re-wrapped in a stream, so the reported plan is the
    /// streaming one and the in-memory node does not appear in it.
    ///
    /// An empty `batches` still reports an in-memory source, but it carries no
    /// schema for the provider to use, so the support check runs against the
    /// dataset's; see [`Self::analyze_plan_provider`].
    ///
    /// [`MemTable`]: datafusion::datasource::MemTable
    pub async fn analyze_plan_batches(&self, batches: Vec<RecordBatch>) -> Result<String> {
        self.analyze_plan_provider(self.batches_to_provider(batches)?)
            .await
    }

    /// [`Self::analyze_plan`] from a re-scannable [`TableProvider`].
    ///
    /// Mirrors [`Self::execute_provider`]. Under
    /// [`SourceDedupeBehavior::FirstSeen`] the provider is re-wrapped in a stream
    /// before the join, so its own node does not appear in the reported plan.
    ///
    /// The support check runs against `provider.schema()`, which is the source
    /// schema the caller supplied. For a provider built from an empty batch list
    /// that schema is the dataset's, so a source whose declared schema the dataset
    /// does not have is reported rather than rejected. `execute_batches` builds its
    /// provider the same way, so the two agree.
    ///
    /// # Errors
    ///
    /// * `Error::NotSupported` when the configuration cannot use the plan path.
    ///   `can_use_create_plan` decides that, and its own doc comment lists the
    ///   source shapes it accepts.
    /// * `Error::invalid_input` from the support check, e.g. a non-nullable dataset
    ///   column the source does not supply.
    /// * Any error from building or executing the plan. This method runs the merge
    ///   to collect metrics, so I/O and source-deduplication failures surface here.
    pub async fn analyze_plan_provider(&self, provider: Arc<dyn TableProvider>) -> Result<String> {
        // Check if we can use create_plan
        if !self.can_use_create_plan(provider.schema().as_ref()).await? {
            return Err(Error::not_supported_source("This merge insert configuration does not support plan reporting: either the source schema is not one the plan path accepts, or the join takes the scalar-index execution path.".into()));
        }

        // Clone self since create_plan consumes the job
        let cloned_job = self.clone();
        let plan = cloned_job.create_plan(provider).await?;

        // Use the analyze_plan function from lance_datafusion, but strip out the wrapper lines
        let options = LanceExecutionOptions::default();
        let analysis_result = analyze_plan(plan.clone(), options).await;

        // Executing the plan wrote files for a transaction that is never
        // committed, so nothing will ever reference them. Remove them now
        // rather than leaving them to orphan cleanup, whether or not the
        // analysis itself succeeded.
        if let Some(transaction) = uncommitted_transaction(plan.as_ref()) {
            self.cleanup_uncommitted_files(&transaction).await;
        }
        // Managed blob payloads are separate objects no fragment lists. The
        // write node records each one as it creates it, including in a run
        // that failed before building a transaction.
        if let Some(created_blob_payloads) = created_blob_payloads(plan.as_ref()) {
            created_blob_payloads.delete_all().await;
        }
        let full_analysis = analysis_result?;

        // Remove the AnalyzeExec and TracedExec lines from the output
        let lines: Vec<&str> = full_analysis.lines().collect();
        let filtered_lines: Vec<&str> = lines
            .into_iter()
            .filter(|line| {
                !line.trim_start().starts_with("AnalyzeExec")
                    && !line.trim_start().starts_with("TracedExec")
            })
            .collect();

        Ok(filtered_lines.join("\n"))
    }

    /// [`Self::analyze_plan_provider`] for a native Lance dataset.
    ///
    /// See [`Self::execute_dataset`] for a compiling example.
    pub async fn analyze_plan_dataset(&self, source: Arc<Dataset>) -> Result<String> {
        self.analyze_plan_provider(Self::dataset_source_provider(source))
            .await
    }

    /// Delete the files an executed-but-uncommitted merge wrote.
    ///
    /// New fragments are entirely new. An updated fragment keeps the files it
    /// had in the read version, so only the data files and deletion file that
    /// differ from that version are removed. Failures are logged, not
    /// returned: the files are unreferenced either way, and orphan cleanup
    /// still reclaims anything left behind.
    async fn cleanup_uncommitted_files(&self, transaction: &Transaction) {
        let Operation::Update {
            new_fragments,
            updated_fragments,
            ..
        } = &transaction.operation
        else {
            return;
        };

        let original_fragments: HashMap<u64, &Fragment> = self
            .dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| (fragment.id, fragment))
            .collect();

        let mut written_fragments = new_fragments.clone();
        let mut written_deletion_files = Vec::new();
        for updated in updated_fragments {
            let original = original_fragments.get(&updated.id);
            let mut written = updated.clone();
            written.files.retain(|file| {
                original.is_none_or(|original| {
                    !original
                        .files
                        .iter()
                        .any(|existing| existing.path == file.path)
                })
            });
            if !written.files.is_empty() {
                written_fragments.push(written);
            }
            if let Some(deletion_file) = &updated.deletion_file
                && !deletion_file.is_inline()
                && original
                    .is_none_or(|original| original.deletion_file.as_ref() != Some(deletion_file))
            {
                written_deletion_files.push(deletion_file_path(
                    &self.dataset.base,
                    updated.id,
                    deletion_file,
                ));
            }
        }

        if !written_fragments.is_empty() {
            let target_bases_info = resolve_target_bases(&self.dataset, &self.params)
                .await
                .ok()
                .flatten();
            cleanup_data_fragments(
                &self.dataset.object_store,
                &self.dataset.base,
                target_bases_info.as_deref(),
                &written_fragments,
            )
            .await;
        }
        // Merge insert writes deletion files with `write_deletion_file`, which
        // always places them under the dataset root.
        for path in written_deletion_files {
            if let Err(e) = self.dataset.object_store.delete(&path).await {
                log::warn!(
                    "Failed to clean up orphaned deletion file '{}': {}",
                    path,
                    e
                );
            }
        }
    }
}

/// Wrapper struct that combines MergeInsertJob with the source provider for retry functionality
struct MergeInsertJobWithProvider {
    job: MergeInsertJob,
    provider: Arc<dyn TableProvider>,
    source_mode: SourceProviderMode,
    attempt_count: Arc<AtomicU32>,
}

impl RetryExecutor for MergeInsertJobWithProvider {
    type Data = UncommittedMergeInsert;
    type Result = (Arc<Dataset>, MergeStats);

    async fn execute_impl(&self) -> Result<Self::Data> {
        // Increment attempt counter
        self.attempt_count.fetch_add(1, Ordering::SeqCst);

        // Re-scan the provider on each retry attempt.
        self.job
            .clone_for_retry()
            .execute_uncommitted_impl(self.provider.clone(), self.source_mode)
            .await
    }

    async fn commit(&self, dataset: Arc<Dataset>, mut data: Self::Data) -> Result<Self::Result> {
        // Update stats with the current attempt count
        data.stats.num_attempts = self.attempt_count.load(Ordering::SeqCst);

        // The dataset argument is the refreshed per-attempt dataset (the same
        // manifest execute_impl resolved against); keep a handle so conflict
        // cleanup resolves bases added between attempts.
        let cleanup_dataset = dataset.clone();
        let mut commit_builder =
            CommitBuilder::new(dataset).with_skip_auto_cleanup(self.job.params.skip_auto_cleanup);
        if let Some(commit_retries) = self.job.params.commit_retries {
            commit_builder = commit_builder.with_max_retries(commit_retries);
        }
        if let Some(affected_rows) = data.affected_rows {
            commit_builder = commit_builder.with_affected_rows(affected_rows);
        }

        let new_fragments = match &data.transaction.operation {
            Operation::Update { new_fragments, .. } => new_fragments.clone(),
            _ => Vec::new(),
        };
        match commit_builder.execute(data.transaction).await {
            Ok(new_dataset) => Ok((Arc::new(new_dataset), data.stats)),
            Err(e) => {
                // A retryable conflict discards this attempt and re-executes it,
                // so its data files are provably uncommitted; remove them
                // (including files routed to target bases, which version cleanup
                // never scans). Other commit errors may be ambiguous about
                // whether the manifest was written, so leave the files alone.
                if matches!(e, Error::RetryableCommitConflict { .. }) && !new_fragments.is_empty() {
                    let target_bases_info =
                        resolve_target_bases(&cleanup_dataset, &self.job.params)
                            .await
                            .ok()
                            .flatten();
                    cleanup_data_fragments(
                        &cleanup_dataset.object_store,
                        &cleanup_dataset.base,
                        target_bases_info.as_deref(),
                        &new_fragments,
                    )
                    .await;
                }
                Err(e)
            }
        }
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.job.dataset = dataset;
    }
}

impl Clone for MergeInsertJob {
    fn clone(&self) -> Self {
        let spill_execution_options = self.spill_execution_options.clone();
        Self {
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            write_params: self.write_params.clone(),
            spill_session_context: new_session_context(&spill_execution_options),
            spill_execution_options,
        }
    }
}

pub(super) fn merge_insert_execution_options() -> LanceExecutionOptions {
    let options = LanceExecutionOptions {
        use_spilling: true,
        batch_size: Some(INDEXED_JOIN_BATCH_ROWS),
        ..Default::default()
    };
    // Size the operation-scoped pool with the same CPU ceiling used by the
    // fragment updater, without setting target_partition on the actual job and
    // thereby changing plan parallelism everywhere.
    let pool_sizing_options = LanceExecutionOptions {
        target_partition: Some(get_num_compute_intensive_cpus().min(8)),
        ..options.clone()
    };
    // Resolve environment-backed limits once so the held context and every
    // execution attempt use exactly the same resource configuration.
    LanceExecutionOptions {
        mem_pool_size: Some(pool_sizing_options.mem_pool_size()),
        max_temp_directory_size: Some(options.max_temp_directory_size()),
        ..options
    }
}

fn max_fragment_update_partitions(memory_pool_size: u64) -> usize {
    // The joined update stream is staged before this sort, so no upstream sort
    // reservation remains live. Budget each downstream partition for its merge
    // reservation plus one capped input and output/fragment-grouping batch.
    let sort_spill_reservation = (memory_pool_size / 3).clamp(1, 40 * 1024 * 1024);
    let working_memory = u64::try_from(MAX_FRAGMENT_UPDATE_BATCH_BYTES)
        .unwrap_or(u64::MAX)
        .saturating_mul(2);
    let partition_budget = sort_spill_reservation.saturating_add(working_memory);
    usize::try_from((memory_pool_size / partition_budget).max(1)).unwrap_or(usize::MAX)
}

pub(super) fn fragment_update_session_context(
    spill_session_context: &SessionContext,
    spill_execution_options: &LanceExecutionOptions,
) -> SessionContext {
    let target_partitions =
        get_num_compute_intensive_cpus()
            .min(8)
            .min(max_fragment_update_partitions(
                spill_execution_options.mem_pool_size(),
            ));
    let config = spill_session_context
        .copied_config()
        .with_target_partitions(target_partitions);
    SessionContext::new_with_config_rt(config, spill_session_context.runtime_env())
}

/// How repeated scans of a merge source relate to one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceProviderMode {
    /// The provider can be scanned only once.
    OneShot,
    /// Every scan reads the same materialized rows.
    StableSnapshot,
    /// The provider can be scanned repeatedly, but each scan may observe different rows.
    Rescannable,
}

impl SourceProviderMode {
    fn supports_retries(self) -> bool {
        !matches!(self, Self::OneShot)
    }

    fn has_stable_snapshot(self) -> bool {
        matches!(self, Self::StableSnapshot)
    }
}

/// Presents all partitions from one source scan as a deterministic stream.
///
/// The indexed join snapshots this stream when the provider does not guarantee
/// that repeated scans observe the same rows. FirstSeen sources also receive a
/// temporary encounter ordinal before either consumer can reorder them.
#[derive(Debug)]
struct SequentialSourcePartitionStream {
    input: Arc<dyn ExecutionPlan>,
    schema: Arc<Schema>,
    ordinal_field: Option<Field>,
    projection_schema: Option<Arc<Schema>>,
}

impl SequentialSourcePartitionStream {
    fn try_new(
        input: Arc<dyn ExecutionPlan>,
        ordinal_field: Option<Field>,
        projection_schema: Option<Arc<Schema>>,
    ) -> Result<Self> {
        let input_schema = projection_schema.clone().unwrap_or_else(|| input.schema());
        let schema = if let Some(field) = ordinal_field.as_ref() {
            Arc::new(input_schema.as_ref().try_with_column(field.clone())?)
        } else {
            input_schema
        };
        Ok(Self {
            input,
            schema,
            ordinal_field,
            projection_schema,
        })
    }

    /// Obtain one provider scan and normalize its partitions into a single,
    /// deterministic stream.
    async fn from_provider_scan(
        provider: &Arc<dyn TableProvider>,
        context: &SessionContext,
        ordinal_field: Option<Field>,
        projection_schema: Option<Arc<Schema>>,
    ) -> Result<Arc<dyn TableProvider>> {
        let input = provider.scan(&context.state(), None, &[], None).await?;
        let partition = Arc::new(Self::try_new(input, ordinal_field, projection_schema)?);
        Ok(Arc::new(StreamingTable::try_new(
            partition.schema().clone(),
            vec![partition],
        )?))
    }
}

impl PartitionStream for SequentialSourcePartitionStream {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn execute(
        &self,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> SendableRecordBatchStream {
        let input = self.input.clone();
        let partition_count = input.properties().output_partitioning().partition_count();
        let partition_streams = stream::iter(0..partition_count)
            .map(move |partition| input.execute(partition, context.clone()))
            .try_flatten();
        let projection_schema = self.projection_schema.clone();
        let partition_streams = partition_streams.map(move |batch| {
            let batch = batch?;
            if let Some(projection_schema) = projection_schema.as_ref() {
                batch
                    .project_by_schema(projection_schema.as_ref())
                    .map_err(DataFusionError::from)
            } else {
                Ok(batch)
            }
        });

        if let Some(ordinal_field) = self.ordinal_field.clone() {
            let mut next_ordinal = 0_u64;
            let source_with_ordinals = partition_streams.map(move |batch| {
                let batch = batch?;
                let num_rows = u64::try_from(batch.num_rows()).map_err(|_| {
                    DataFusionError::Execution(format!(
                        "merge-insert source batch row count {} exceeds u64::MAX",
                        batch.num_rows()
                    ))
                })?;
                let end_ordinal = next_ordinal.checked_add(num_rows).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "merge-insert source ordinal overflow at {} with batch row count {}",
                        next_ordinal,
                        batch.num_rows()
                    ))
                })?;
                let ordinals = Arc::new(UInt64Array::from_iter_values(next_ordinal..end_ordinal));
                next_ordinal = end_ordinal;
                batch
                    .try_with_column(ordinal_field.clone(), ordinals)
                    .map_err(|error| DataFusionError::External(Box::new(error)))
            });
            Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                source_with_ordinals,
            ))
        } else {
            Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                partition_streams,
            ))
        }
    }
}

impl Clone for MergeInsertJobWithProvider {
    fn clone(&self) -> Self {
        Self {
            job: self.job.clone_for_retry(),
            provider: self.provider.clone(),
            source_mode: self.source_mode,
            attempt_count: self.attempt_count.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
