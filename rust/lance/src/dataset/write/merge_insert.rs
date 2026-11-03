// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The merge insert operation merges a batch of new data into an existing batch of old data.  This can be
//! used to implement a bulk update-or-insert (upsert), bulk delete or find-or-create operation.  It can also be used to
//! replace a specified region of data with new data (e.g. replace the data for the month of January)
//!
//! The terminology for this operation can be slightly confusing.  We try and stick with the terminology from
//! SQL.  The "target table" is the OLD data that already exists.  The "source table" is the NEW data which is
//! being inserted into the dataset.
//!
//! In order for this operation to work we need to be able to match rows from the source table with rows in the
//! target table.  For example, given a row we need to know if this is a brand-new row or matches an existing row.
//!
//! This match condition is currently limited to a key-match.  This means we consider a row to be a match if the
//! key columns are identical in both the source and the target.  This means that you will need some kind of
//! meaningful key column to be able to perform a merge insert.
//!
//! On stable-row-id V2 tables that opt into `lance.row_lineage.spill`, the V2
//! row-rewrite plan spills carried row ids and creation versions above the
//! table's inline budget. Updated and inserted rows are written to separate
//! fragments so inserts retain commit-assigned lineage through conflict retries.
//! Spilling is disabled by default and does not rewrite existing fragments.

// Internal column name for the merge action. Using "__action" to avoid collisions with user columns.
const MERGE_ACTION_COLUMN: &str = "__action";
// ## NULL-safe source row detection via sentinel column
//
// The merge join uses standard SQL equality for ON columns, which means NULL != NULL.
// After an outer join we need to know for each output row whether it came from the
// source side, the target side, or both.  The naive approach — checking whether an ON
// column IS NOT NULL — is wrong: a source row whose ON column is legitimately NULL is
// indistinguishable from a NULL introduced by the outer join on the target side.
//
// Solution: inject a `lit(true)` sentinel into every source row *before* the join.
// After the join:
//   - source rows (matched or unmatched)  → sentinel = true   (never NULL)
//   - target-only rows                    → sentinel = NULL   (outer-join fill)
//
// `assign_action` then uses `sentinel IS NOT NULL` instead of key-column IS NOT NULL
// to determine which side each row came from.  The sentinel is stripped by
// `prepare_stream_schema` and never written to the dataset.
pub(super) const MERGE_SOURCE_SENTINEL: &str = "__merge_source_sentinel";

pub mod inserted_rows;

use inserted_rows::KeyExistenceFilter;

use super::{
    TargetBaseInfo, WriteMode, WriteParams, validate_and_resolve_target_bases_with_primary,
};
use crate::{
    Dataset,
    dataset::blob::BlobPayloadInventory,
    dataset::rowids::inline_row_lineage_max_bytes,
    dataset::transaction::{Transaction, UpdatedFragmentOffsets},
    io::exec::Planner,
};
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use datafusion::common::{Column as DFColumn, TableReference};
use datafusion::error::DataFusionError;
use datafusion::{
    catalog::{TableProvider, streaming::StreamingTable},
    execution::context::SessionContext,
    logical_expr::{self, Expr},
    physical_plan::{
        ExecutionPlan, SendableRecordBatchStream, stream::RecordBatchStreamAdapter,
        streaming::PartitionStream,
    },
    prelude::DataFrame,
    scalar::ScalarValue,
};
use futures::{
    StreamExt, TryStreamExt,
    stream::{self},
};
use lance_core::{
    Error, Result,
    datatypes::SchemaCompareOptions,
    error::{InvalidInputSnafu, box_error},
};
use lance_datafusion::exec::{LanceExecutionOptions, OneShotPartitionStream};
use lance_file::version::ConcreteFileVersion;
use lance_index::mem_wal::CompactedSsTable;
use lance_select::RowAddrTreeMap;
use lance_table::format::Fragment;
use snafu::ResultExt;
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

mod assign_action;
mod builder;
mod exec;
mod job;
mod logical_plan;
mod merger;
mod row_ids;
mod update_fragments;

/// Check a merge source against the target schema. As on append, a legacy
/// blob input matches a Blob v2 target column; the writer converts it.
fn check_source_schema(
    source: &lance_core::datatypes::Schema,
    target: &lance_core::datatypes::Schema,
    options: &SchemaCompareOptions,
) -> Result<()> {
    source.check_compatible(target, options).or_else(|err| {
        let source = super::promote_legacy_blob_schema(source)?;
        let target = super::promote_legacy_blob_schema(target)?;
        source.check_compatible(&target, options).map_err(|_| err)
    })
}

/// Build a source schema in target field order while preserving the source's
/// logical leaf types. The latter matters for extension columns such as Arrow
/// JSON, whose write input is Utf8 while the dataset's physical type is binary.
pub(crate) fn canonical_source_schema(
    source: &Schema,
    target: &Schema,
) -> std::result::Result<Schema, ArrowError> {
    fn canonical_field(source: &Field, target: &Field) -> std::result::Result<Field, ArrowError> {
        let data_type = match (source.data_type(), target.data_type()) {
            (DataType::Struct(source_fields), DataType::Struct(target_fields)) => {
                let fields = target_fields
                    .iter()
                    .map(|target_field| {
                        let source_field = source_fields
                            .iter()
                            .find(|field| field.name() == target_field.name())
                            .ok_or_else(|| {
                                ArrowError::SchemaError(format!(
                                    "field {} does not exist in source struct {}",
                                    target_field.name(),
                                    source.name()
                                ))
                            })?;
                        canonical_field(source_field, target_field).map(Arc::new)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                DataType::Struct(fields.into())
            }
            (DataType::List(source_item), DataType::List(target_item)) => {
                DataType::List(Arc::new(canonical_field(source_item, target_item)?))
            }
            (DataType::LargeList(source_item), DataType::LargeList(target_item)) => {
                DataType::LargeList(Arc::new(canonical_field(source_item, target_item)?))
            }
            (
                DataType::FixedSizeList(source_item, size),
                DataType::FixedSizeList(target_item, _),
            ) => {
                DataType::FixedSizeList(Arc::new(canonical_field(source_item, target_item)?), *size)
            }
            (DataType::Map(source_entries, sorted), DataType::Map(target_entries, _)) => {
                DataType::Map(
                    Arc::new(canonical_field(source_entries, target_entries)?),
                    *sorted,
                )
            }
            _ => source.data_type().clone(),
        };
        Ok(source.clone().with_data_type(data_type))
    }

    let fields = target
        .fields()
        .iter()
        .map(|target_field| {
            let source_field = source.field_with_name(target_field.name()).map_err(|_| {
                ArrowError::SchemaError(format!(
                    "field {} does not exist in source schema",
                    target_field.name()
                ))
            })?;
            canonical_field(source_field, target_field).map(Arc::new)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Schema::new_with_metadata(fields, source.metadata().clone()))
}

/// Format key values for error messages via extracting "on" column values from the given RecordBatch.
pub fn format_key_values_on_columns(
    batch: &RecordBatch,
    row_idx: usize,
    on_columns: &[String],
) -> String {
    let mut on_values = Vec::new();

    for col_name in on_columns {
        if let Some(col_idx) = batch.schema().column_with_name(col_name) {
            let column = batch.column(col_idx.0);
            let value_str = if column.is_null(row_idx) {
                "NULL".to_string()
            } else {
                // Convert the value to string representation
                match ScalarValue::try_from_array(column, row_idx) {
                    Ok(scalar_value) => match &scalar_value {
                        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => {
                            format!("\"{}\"", s)
                        }
                        _ => scalar_value.to_string(),
                    },
                    Err(_) => format!("<{:?}>", column.data_type()),
                }
            };
            on_values.push(format!("{} = {}", col_name, value_str));
        }
    }

    if on_values.is_empty() {
        "<unable to extract on column values>".to_string()
    } else {
        on_values.join(", ")
    }
}

/// Create duplicate rows error via extracting "on" column values from the given RecordBatch.
pub fn create_duplicate_row_error(
    batch: &RecordBatch,
    row_idx: usize,
    on_columns: &[String],
) -> DataFusionError {
    DataFusionError::External(Box::new(Error::invalid_input(format!(
        "Ambiguous merge inserts are prohibited: multiple source rows match the same target row on ({}). \
                    Please ensure each target row is matched by at most one source row.",
        format_key_values_on_columns(batch, row_idx, on_columns)
    ))))
}

/// Tracks non-null join keys for source rows that will be inserted.
///
/// NULL join keys are deliberately not tracked because merge insert uses SQL
/// equality, where a key containing NULL does not equal another such key.
#[derive(Debug, Default)]
struct InsertedKeyTracker {
    keys: HashSet<Vec<ScalarValue>>,
}

impl InsertedKeyTracker {
    /// Returns true when the row has a new key or a key containing NULL.
    fn insert(
        &mut self,
        batch: &RecordBatch,
        row_idx: usize,
        on_columns: &[String],
    ) -> datafusion::common::Result<bool> {
        let mut key = Vec::with_capacity(on_columns.len());
        for column_name in on_columns {
            let column = batch.column_by_name(column_name).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "merge insert key column '{}' not found in source batch",
                    column_name
                ))
            })?;
            let value = ScalarValue::try_from_array(column, row_idx)?;
            if value.is_null() {
                return Ok(true);
            }
            key.push(value);
        }
        Ok(self.keys.insert(key))
    }
}

/// How a merge insert writes the merged rows to disk.
///
/// A partial-schema update (the source omits some dataset columns) can be
/// written two ways, and which one writes fewer bytes depends on the fraction of
/// each fragment's rows the source matches, which is only known once the join
/// has run. So the mode is chosen by the caller rather than guessed here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MergeInsertWriteMode {
    /// Let the engine choose.
    ///
    /// Currently always rewrites whole rows, except on the one path that
    /// predates this enum: a partial-schema update whose join key carries a
    /// scalar index patches columns, as it has since before there was a way to
    /// ask for either.
    // TODO: choose per fragment inside the write sink, where the matched row
    // count for that fragment is known. That needs a transaction-format change,
    // because `UpdateMode` is per-commit and the conflict resolver treats
    // `RewriteColumns` as proof that no row moved.
    #[default]
    Auto,
    /// Delete the matched rows and write whole rows into new fragments.
    ///
    /// Cost scales with the number of matched rows, so this is the cheaper mode
    /// when few rows match or when most columns are being replaced anyway.
    ///
    /// For a partial-schema update this gives up the scalar-index probe on the
    /// join key if there is one, because the indexed path only ever patches
    /// columns.
    RewriteRows,
    /// Attach new data files holding the source columns to the fragments that
    /// already hold the matched rows, and tombstone the old versions of those
    /// columns.
    ///
    /// The columns the source omits are neither read nor written, so this is the
    /// cheaper mode for a narrow update of a wide table. The replacement column
    /// file covers every row of each fragment it touches, so the bytes written
    /// barely fall as fewer rows match.
    ///
    /// Errors if the merge cannot be expressed this way: it must update matched
    /// rows only (no inserts, no matched deletes, no delete-by-source) with a
    /// source that omits at least one dataset column, carries at least one
    /// column besides the join key, and carries no blob column.
    RewriteColumns,
}

/// Describes how rows should be handled when there is no matching row in the source table
///
/// These are old rows which do not match any new data
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum WhenNotMatchedBySource {
    /// Do not delete rows from the target table
    ///
    /// This can be used for a find-or-create or an upsert operation
    Keep,
    /// Delete all rows from target table that don't match a row in the source table
    Delete,
    /// Delete rows from the target table if there is no match AND the expression evaluates to true
    ///
    /// This can be used to replace a region of data with new data
    DeleteIf(Expr),
}

impl WhenNotMatchedBySource {
    /// Create an instance of WhenNotMatchedBySource::DeleteIf from
    /// an SQL filter string
    ///
    /// This will parse the filter string (using the schema of the provided
    /// dataset) and simplify the resulting expression
    pub fn delete_if(dataset: &Dataset, expr: &str) -> Result<Self> {
        let planner = Planner::new(Arc::new(dataset.schema().into()));
        let expr = planner
            .parse_filter(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {})?;
        Ok(Self::DeleteIf(expr))
    }
}

/// Describes how rows should be handled when there is a match between the target table and source table
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum WhenMatched {
    /// The row is deleted from the target table and a new row is inserted based on the source table
    ///
    /// This can be used to achieve upsert behavior
    UpdateAll,
    /// The row is kept unchanged
    ///
    /// This can be used to achieve find-or-create behavior
    DoNothing,
    /// The row is updated (similar to UpdateAll) only for rows where the expression evaluates to
    /// true
    UpdateIf(String),
    /// The row is updated (similar to UpdateAll) only for rows where the expression evaluates to
    /// true
    UpdateIfExpr(Expr),
    /// Fail the operation if a match is found
    ///
    /// This can be used to ensure that no existing rows are overwritten or modified after inserted.
    Fail,
    /// The matching row is deleted from the target table
    ///
    /// This can be used for bulk deletion by matching on key columns.
    /// Unlike UpdateAll, no new row is inserted - the matched row is simply removed.
    Delete,
}

impl WhenMatched {
    pub fn update_if(_dataset: &Dataset, expr: &str) -> Result<Self> {
        // Store the expression string and defer parsing until we know which path to take
        Ok(Self::UpdateIf(expr.to_string()))
    }

    pub fn update_if_expr(expr: Expr) -> Self {
        Self::UpdateIfExpr(expr)
    }
}

/// Describes how rows should be handled when there is no matching row in the target table
///
/// These are new rows which do not match any old data
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WhenNotMatched {
    /// The new row is inserted into the target table
    ///
    /// This is used in both find-or-create and upsert operations
    InsertAll,
    /// The new row is ignored
    DoNothing,
}

/// Describes how to handle duplicate source rows.
///
/// If the source contains duplicates and `FirstSeen` behavior doesn't match your needs,
/// sort the source data before passing it to the merge insert operation.
/// Rows whose join keys contain NULL are not duplicates because merge insert uses SQL
/// equality, where NULL does not equal NULL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum SourceDedupeBehavior {
    /// Fail if multiple source rows match the same target row (default)
    #[default]
    Fail,
    /// Keep the first row for each join key and skip subsequent rows
    ///
    /// This applies both to rows that match a target row and to unmatched rows that
    /// would otherwise insert the same non-null join key more than once.
    FirstSeen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanFileVersion(ConcreteFileVersion);

impl PartialOrd for PlanFileVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // DataFusion requires extension-node parameters to have a deterministic
        // structural order. This is only plan identity; operation capability
        // decisions always match on ConcreteFileVersion directly.
        self.0
            .to_data_file_numbers()
            .partial_cmp(&other.0.to_data_file_numbers())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
struct MergeInsertParams {
    // The column(s) to join on
    on: Vec<String>,
    // If true, then update all columns of the old data to the new data when there is a match
    when_matched: WhenMatched,
    // If true, then insert all columns of the new data when there is no match in the old data
    insert_not_matched: bool,
    // Controls whether data that is not matched by the source is deleted or not
    delete_not_matched_by_source: WhenNotMatchedBySource,
    conflict_retries: u32,
    // When the source is a one-shot stream and `conflict_retries > 0`, the source
    // is spilled (memory, then disk) so it can be replayed on each retry. Set to
    // false to fail fast on contention instead of buffering the stream. Has no
    // effect on re-scannable sources; arbitrary providers may still use bounded
    // replay within an indexed attempt to preserve one logical snapshot.
    spill_for_retry: bool,
    retry_timeout: Duration,
    // MemWAL SSTables to mark as compacted when this commit succeeds.
    compacted_sstables: Vec<CompactedSsTable>,
    // The `dir_id` of the MemWAL `compacted_sstables` belongs to.
    mem_wal_dir_id: Option<String>,
    // If true, skip auto cleanup during commits. This should be set to true
    // for high frequency writes to improve performance. This is also useful
    // if the writer does not have delete permissions and the clean up would
    // just try and log a failure anyway.
    skip_auto_cleanup: bool,
    // Controls whether to use indices for the merge operation. Default is true.
    // Setting to false forces a full table scan even if an index exists.
    use_index: bool,
    // How the merged rows are written to disk. Default is `Auto`, which today
    // always rewrites whole rows.
    write_mode: MergeInsertWriteMode,
    // Controls how to handle duplicate source rows that match the same target row.
    source_dedupe_behavior: SourceDedupeBehavior,
    // Number of inner commit retries for manifest version conflicts. Default is 20.
    commit_retries: Option<u32>,
    // Target base IDs for routing new fragments, mirroring WriteParams::target_bases.
    target_bases: Option<Vec<u32>>,
    // Target base names or path URIs (unresolved), mirroring
    // WriteParams::target_base_names_or_paths. Resolved at execution time.
    target_base_names_or_paths: Option<Vec<String>>,
    // Target all registered bases, mirroring WriteParams::target_all_bases.
    // Some(include_primary); resolved at execution time.
    target_all_bases: Option<bool>,
    // Exact output data file version. The manifest default is used when absent.
    data_storage_version: Option<PlanFileVersion>,
}

impl MergeInsertParams {
    /// Return ON field IDs only when they match the declared primary key.
    fn primary_key_field_ids(&self, schema: &lance_core::datatypes::Schema) -> Option<Vec<i32>> {
        let field_ids: Vec<i32> = self
            .on
            .iter()
            .filter_map(|name| schema.field(name).map(|field| field.id))
            .collect();
        let pk_field_ids: Vec<i32> = schema
            .unenforced_primary_key()
            .iter()
            .map(|field| field.id)
            .collect();
        (!pk_field_ids.is_empty() && field_ids == pk_field_ids).then_some(field_ids)
    }

    fn write_version(&self, dataset: &Dataset) -> ConcreteFileVersion {
        self.data_storage_version
            .map(|version| version.0)
            .unwrap_or_else(|| dataset.manifest.data_storage_format.lance_file_format())
    }

    fn carried_lineage_spill_budget(&self, dataset: &Dataset) -> Result<Option<usize>> {
        if dataset.manifest.uses_stable_row_ids()
            && self.write_version(dataset) != ConcreteFileVersion::V1
            && matches!(
                self.when_matched,
                WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_)
            )
        {
            inline_row_lineage_max_bytes(dataset)
        } else {
            Ok(None)
        }
    }
}

/// What [`MergeInsertJob::update_fragments`] wrote.
#[derive(Debug)]
pub(super) struct PatchedFragments {
    /// Existing fragments with column patches or updated row-version metadata.
    pub updated_fragments: Vec<Fragment>,
    /// Fragments written for rows that carried no target address.
    pub new_fragments: Vec<Fragment>,
    /// Ids of the fields written, so the caller can prune indices covering them.
    pub fields_modified: Vec<u32>,
    /// Physical row offsets patched in each updated fragment.
    ///
    /// `update_fragments` stamps row-level version metadata with the version it
    /// was given, which is only a guess: a compatible transaction can commit
    /// first, making the real commit version later. Handing these offsets to
    /// `Operation::Update`'s `updated_fragment_offsets` lets `build_manifest`
    /// re-stamp exactly the patched rows with the version the commit actually
    /// got. Empty when the dataset does not use stable row ids, since nothing
    /// reads the metadata then.
    pub matched_offsets: UpdatedFragmentOffsets,
}

/// A MergeInsertJob inserts new rows, deletes old rows, and updates existing rows all as
/// part of a single transaction.
pub struct MergeInsertJob {
    // The column to merge the new data into
    dataset: Arc<Dataset>,
    // The parameters controlling how to merge the two streams
    params: MergeInsertParams,
    // The caller's parameters for the data files this job writes. Kept out of
    // `MergeInsertParams`, which the logical plan compares and hashes.
    write_params: Arc<WriteParams>,
    // A job-local context shared by retry replay, indexed execution, and
    // partial-update sorting, including across commit retries.
    spill_session_context: SessionContext,
    spill_execution_options: LanceExecutionOptions,
}

/// Build a merge insert operation.
///
/// This operation is similar to SQL's MERGE statement. It allows you to merge
/// new data with existing data.
///
/// Use the [MergeInsertBuilder] to construct an merge insert job.
///
/// If the `on` parameter is empty, the builder will fall back to the
/// schema's unenforced primary key (if configured). If neither `on` nor a
/// primary key is available, this constructor returns an error.
/// For example:
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::{MergeInsertBuilder, WhenNotMatched, WhenNotMatchedBySource};
/// # use datafusion::physical_plan::SendableRecordBatchStream;
/// # use datafusion::prelude::Expr;
/// # use std::sync::Arc;
/// # async fn example(dataset: Arc<Dataset>, new_data1: SendableRecordBatchStream, new_data2: SendableRecordBatchStream, new_data3: SendableRecordBatchStream, month_eq_jan: Expr) -> Result<()> {
/// // find-or-create, insert new rows only
/// let (updated_dataset, _stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["my_key".to_string()])?
///     .try_build()?
///     .execute(new_data1)
///     .await?;
///
/// // upsert, insert or update
/// let (updated_dataset, _stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["my_key".to_string()])?
///     .when_not_matched(WhenNotMatched::InsertAll)
///     .try_build()?
///     .execute(new_data2)
///     .await?;
///
/// // replace data for month=january
/// let (updated_dataset, _stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["my_key".to_string()])?
///     .when_not_matched(WhenNotMatched::InsertAll)
///     .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(month_eq_jan))
///     .try_build()?
///     .execute(new_data3)
///     .await?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct MergeInsertBuilder {
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    write_params: Arc<WriteParams>,
}

/// Resolve the merge insert target bases against the base paths registered in
/// the dataset manifest. Returns `None` when no target bases were requested.
/// Base id [`super::PRIMARY_BASE_ID`] and the dataset's URI refer to the
/// dataset's primary storage.
///
/// Resolution runs once per execution attempt so retries validate against the
/// manifest version they are writing to.
async fn resolve_target_bases(
    dataset: &Dataset,
    params: &MergeInsertParams,
) -> Result<Option<Vec<TargetBaseInfo>>> {
    if params.target_bases.is_none()
        && params.target_base_names_or_paths.is_none()
        && params.target_all_bases.is_none()
    {
        return Ok(None);
    }
    // Reuse the normal write path resolution (validation, name/path lookup,
    // and per-base credential handling) through a parameter shim.
    let mut write_params = WriteParams {
        mode: WriteMode::Append,
        target_bases: params.target_bases.clone(),
        target_base_names_or_paths: params.target_base_names_or_paths.clone(),
        target_all_bases: params.target_all_bases,
        session: Some(dataset.session.clone()),
        store_params: dataset.store_params.as_deref().cloned(),
        base_store_params: dataset.base_store_params.as_deref().cloned(),
        ..Default::default()
    };
    validate_and_resolve_target_bases_with_primary(
        &mut write_params,
        Some(&dataset.manifest.base_paths),
        &dataset.object_store,
        &dataset.base,
        dataset.uri(),
    )
    .await
}

enum SchemaComparison {
    FullCompatible,
    Subschema,
}

/// Wrap a one-shot stream in a non-replayable [`StreamingTable`] provider.
///
/// The provider can only be scanned once (its single partition hands out the
/// underlying stream), so it must not be used where retries may re-scan it.
fn one_shot_provider(stream: SendableRecordBatchStream) -> Result<Arc<dyn TableProvider>> {
    let schema = stream.schema();
    let partition = Arc::new(OneShotPartitionStream::new(stream));
    Ok(Arc::new(StreamingTable::try_new(schema, vec![partition])?))
}

/// Scans source partitions sequentially and removes duplicate non-null keys.
///
/// Deduplicating before the join fixes the `FirstSeen` winner at the source
/// boundary, before DataFusion can reorder rows. The tracker retains only keys,
/// so this stays streaming without buffering source batches.
#[derive(Debug)]
struct DeduplicatingSourcePartitionStream {
    input: Arc<dyn ExecutionPlan>,
    schema: Arc<Schema>,
    on_columns: Vec<String>,
    skipped_duplicates: Arc<AtomicU64>,
}

impl DeduplicatingSourcePartitionStream {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        on_columns: Vec<String>,
        skipped_duplicates: Arc<AtomicU64>,
    ) -> Self {
        let schema = input.schema();
        Self {
            input,
            schema,
            on_columns,
            skipped_duplicates,
        }
    }
}

impl PartitionStream for DeduplicatingSourcePartitionStream {
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

        let mut tracker = InsertedKeyTracker::default();
        let on_columns = self.on_columns.clone();
        let skipped_duplicates = self.skipped_duplicates.clone();
        skipped_duplicates.store(0, Ordering::Relaxed);
        let deduplicated = partition_streams.map(move |batch| {
            let batch = batch?;
            let mut keep = Vec::with_capacity(batch.num_rows());
            let mut num_skipped = 0_u64;
            for row_idx in 0..batch.num_rows() {
                let is_first = tracker.insert(&batch, row_idx, &on_columns)?;
                keep.push(is_first);
                if !is_first {
                    num_skipped = num_skipped.checked_add(1).ok_or_else(|| {
                        DataFusionError::Execution(
                            "source duplicate count overflowed u64".to_string(),
                        )
                    })?;
                }
            }

            let mut current = skipped_duplicates.load(Ordering::Relaxed);
            loop {
                let updated = current.checked_add(num_skipped).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "source duplicate count overflow at {} with batch count {}",
                        current, num_skipped
                    ))
                })?;
                match skipped_duplicates.compare_exchange_weak(
                    current,
                    updated,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }

            if num_skipped == 0 {
                Ok(batch)
            } else {
                arrow::compute::filter_record_batch(&batch, &BooleanArray::from(keep))
                    .map_err(DataFusionError::from)
            }
        });

        Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            deduplicated,
        ))
    }
}

/// Re-expose every dataset column the source does not carry under its bare
/// dataset field name, taking the value from the `target` side of the join.
///
/// `df` is the target-joined-source plan, whose target columns are qualified
/// `target` and whose source columns are qualified `source`. For matched rows a
/// filled column carries the existing target value (preserving non-source
/// columns on update); for unmatched source rows the outer join leaves the
/// target side NULL, so inserts get NULL. The unqualified name matches the
/// dataset field and makes it a normal data column from the write exec's
/// perspective.
///
/// A column `from_target` rejects is filled with a NULL of the target column's
/// type instead, which keeps it out of the target scan. That is only correct
/// when no row that reaches the write needs its stored value.
///
/// A filled column replaces `target.X` in place rather than being appended, so
/// the output is the join schema with some columns unqualified, in the same
/// order, and the same width. Appending instead would leave both `target.X` and
/// a bare `X` in scope, which downstream column references read as ambiguous.
fn fill_missing_target_columns(
    df: DataFrame,
    dataset_schema: &Schema,
    source_field_names: &HashSet<String>,
    from_target: impl Fn(&str) -> bool,
) -> Result<DataFrame> {
    let missing_from_source: HashSet<&str> = dataset_schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .filter(|name| !source_field_names.contains(*name))
        .collect();
    if missing_from_source.is_empty() {
        return Ok(df);
    }

    // Keep this a single projection. `DataFrame::with_column` would express the
    // fill one field at a time and read better, but each call re-lists every
    // column in a fresh `Projection`, so a per-field loop nests N of them and
    // the optimizer's recursive walk overflows the stack on wide tables (#9504).
    let target_qualifier = TableReference::bare("target");
    let projection = df
        .schema()
        .iter()
        .map(|(qualifier, field)| {
            let expr = logical_expr::col(DFColumn::from((qualifier, field)));
            if qualifier == Some(&target_qualifier)
                && missing_from_source.contains(field.name().as_str())
            {
                // Same type and metadata as the target column, so the write
                // handles both alike.
                let expr = match from_target(field.name()) {
                    true => expr,
                    false => logical_expr::lit_with_metadata(
                        ScalarValue::try_from(field.data_type())?,
                        Some(field.as_ref().into()),
                    ),
                };
                // An alias is unqualified, which is what drops the `target.`
                // prefix and makes this a plain data column.
                Ok(expr.alias(field.name()))
            } else {
                Ok(expr)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(df.select(projection)?)
}

/// How the merge plan reads the target's blob columns: legacy blobs as binary,
/// Blob v2 as descriptors. The target scan and the deferred take of
/// `MergeConditionExec` must agree, or a condition sees different types.
fn target_blob_handling(dataset: &Dataset) -> lance_core::datatypes::BlobHandling {
    lance_core::datatypes::BlobHandling::SomeBlobsBinary(
        dataset
            .schema()
            .fields_pre_order()
            .filter(|field| field.is_blob() && !field.is_blob_v2())
            .map(|field| field.id as u32)
            .collect(),
    )
}

/// Merger will store these statistics as it runs (for each batch)
#[derive(Debug, Default, Clone)]
pub struct MergeStats {
    /// Number of inserted rows (for user statistics)
    pub num_inserted_rows: u64,
    /// Number of updated rows (for user statistics)
    pub num_updated_rows: u64,
    /// Number of deleted rows (for user statistics)
    /// Note: This is different from internal references to 'deleted_rows', since we technically "delete" updated rows during processing.
    /// However those rows are not shared with the user.
    pub num_deleted_rows: u64,
    /// Number of attempts performed.
    ///
    /// See [`MergeInsertBuilder::conflict_retries`] for more information.
    pub num_attempts: u32,
    /// Total bytes written to storage. This currently only includes data files.
    pub bytes_written: u64,
    /// Number of data files written. This currently only includes data files.
    pub num_files_written: u64,
    /// Number of duplicate source rows skipped (when SourceDedupeBehavior::FirstSeen)
    pub num_skipped_duplicates: u64,
}

pub struct UncommittedMergeInsert {
    pub transaction: Transaction,
    pub affected_rows: Option<RowAddrTreeMap>,
    pub stats: MergeStats,
    pub inserted_rows_filter: Option<KeyExistenceFilter>,
}

#[cfg(test)]
mod tests;

/// Row cap for indexed external-sort output batches; payload rows may be wide.
const INDEXED_JOIN_BATCH_ROWS: usize = 128;

/// Byte cap for indexed sort input batches and in-memory replay buffers.
const MAX_INDEXED_JOIN_BATCH_BYTES: usize = 25 * 1024 * 1024;

/// Byte cap for each batch entering the partial-update fragment sort.
const MAX_FRAGMENT_UPDATE_BATCH_BYTES: usize = 25 * 1024 * 1024;

/// The transaction a merge insert write node built during execution, if the
/// node ran to completion.
fn uncommitted_transaction(plan: &dyn ExecutionPlan) -> Option<Transaction> {
    if let Some(full_exec) = plan.downcast_ref::<exec::FullSchemaMergeInsertExec>() {
        full_exec.transaction()
    } else if let Some(in_place_exec) = plan.downcast_ref::<exec::InPlaceMergeInsertExec>() {
        in_place_exec.transaction()
    } else if let Some(delete_exec) = plan.downcast_ref::<exec::DeleteOnlyMergeInsertExec>() {
        delete_exec.transaction()
    } else {
        None
    }
}

/// The managed blob payloads a merge insert write node created.
///
/// Only the full-schema node can create any. The in-place node refuses a source
/// that carries a blob column and never inserts rows, and the delete-only node
/// writes no data.
fn created_blob_payloads(plan: &dyn ExecutionPlan) -> Option<&BlobPayloadInventory> {
    plan.downcast_ref::<exec::FullSchemaMergeInsertExec>()
        .map(|full_exec| full_exec.created_blob_payloads())
}
