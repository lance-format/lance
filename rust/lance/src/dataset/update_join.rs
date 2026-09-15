// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded-memory implementation of fragment column updates.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StructArray, UInt32Array, UInt64Array, new_null_array,
};
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_schema::{
    ArrowError, DataType, Field as ArrowField, FieldRef, Schema as ArrowSchema, SchemaRef,
    SortOptions,
};
use datafusion::common::{JoinType, NullEquality};
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::joins::SortMergeJoinExec;
use futures::{StreamExt, stream};
use lance_arrow::memory::MemoryAccumulator;
use lance_arrow::{FieldExt, SchemaExt, interleave_batches};
use lance_core::datatypes::{
    BLOB_V2_LOGICAL_FIELDS, BlobHandling, BlobV2Layout, OnMissing, OnTypeMismatch, Schema,
};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_datafusion::exec::{
    ExecutionSummaryCounts, HardCapBatchSizeExec, LanceExecutionOptions, OneShotExec,
    collect_execution_metrics, new_session_context,
};
use lance_datafusion::utils::reader_to_stream;
use lance_table::format::Fragment;
use roaring::RoaringBitmap;

use super::fragment::{FileFragment, FragmentUpdateColumnsResult};
use super::hash_joiner::HashJoiner;
use super::utils::SchemaAdapter;
use super::{WriteParams, blob};

/// Row count used by scans, DataFusion operators, and the physical updater.
const EXECUTION_BATCH_SIZE: usize = 1024;
/// Largest RHS row count retained for the in-memory hash join.
const DEFAULT_MAX_HASH_ROWS: usize = 1_000_000;
/// Largest estimated RHS allocation retained for the in-memory hash join.
const DEFAULT_MAX_HASH_BYTES: usize = 256 * 1024 * 1024;
/// Upper bound for each sort's spill-merge reservation.
const MAX_SORT_MERGE_RESERVATION: usize = 10 * 1024 * 1024;
/// Upper bound for a batch entering an external sort.
const MAX_INPUT_BATCH_BYTES: usize = 25 * 1024 * 1024;

#[derive(Clone)]
pub(super) struct UpdateColumnsOptions {
    pub(super) execution_options: LanceExecutionOptions,
    pub(super) max_hash_rows: usize,
    pub(super) max_hash_bytes: usize,
    #[cfg(test)]
    pub(super) session_context: Option<SessionContext>,
}

impl Default for UpdateColumnsOptions {
    fn default() -> Self {
        Self {
            execution_options: LanceExecutionOptions {
                use_spilling: true,
                target_partition: Some(1),
                batch_size: Some(EXECUTION_BATCH_SIZE),
                ..Default::default()
            },
            max_hash_rows: DEFAULT_MAX_HASH_ROWS,
            max_hash_bytes: DEFAULT_MAX_HASH_BYTES,
            #[cfg(test)]
            session_context: None,
        }
    }
}

enum PreparedRhs {
    InMemory(Vec<RecordBatch>),
    External(SendableRecordBatchStream),
}

/// Normalize valid minimal Blob v2 values to the complete logical shape used
/// when existing descriptors are materialized for update fallback.
fn canonical_blob_field(field: &ArrowField) -> Result<ArrowField> {
    if field.is_blob_v2() {
        let DataType::Struct(children) = field.data_type() else {
            return Err(Error::invalid_input(format!(
                "Blob v2 field '{}' must be a struct, got {}",
                field.name(),
                field.data_type()
            )));
        };
        if BlobV2Layout::classify(children) != Some(BlobV2Layout::Logical) {
            return Err(Error::invalid_input(format!(
                "Blob v2 update field '{}' must use a logical layout, got {}",
                field.name(),
                field.data_type()
            )));
        }
        return Ok(ArrowField::new(
            field.name(),
            DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()),
            field.is_nullable(),
        )
        .with_metadata(field.metadata().clone()));
    }

    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|child| canonical_blob_field(child))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        DataType::List(child) => DataType::List(Arc::new(canonical_blob_field(child)?)),
        DataType::LargeList(child) => DataType::LargeList(Arc::new(canonical_blob_field(child)?)),
        _ => return Ok(field.clone()),
    };
    Ok(
        ArrowField::new(field.name(), data_type, field.is_nullable())
            .with_metadata(field.metadata().clone()),
    )
}

fn canonical_blob_array(
    source_field: &ArrowField,
    target_field: &ArrowField,
    array: ArrayRef,
) -> std::result::Result<ArrayRef, ArrowError> {
    if source_field.data_type() == target_field.data_type() {
        return Ok(array);
    }

    if source_field.is_blob_v2() {
        let source = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "Blob v2 field '{}' has non-struct array type {}",
                    source_field.name(),
                    array.data_type()
                ))
            })?;
        let DataType::Struct(target_children) = target_field.data_type() else {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Canonical Blob v2 field '{}' has non-struct type {}",
                target_field.name(),
                target_field.data_type()
            )));
        };
        if !matches!(source.num_columns(), 2 | 4) || target_children.len() != 4 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Cannot normalize Blob v2 field '{}' from {} to {} children",
                source_field.name(),
                source.num_columns(),
                target_children.len()
            )));
        }
        let mut columns = source.columns().to_vec();
        if source.num_columns() == 2 {
            columns.push(new_null_array(&DataType::UInt64, source.len()));
            columns.push(new_null_array(&DataType::UInt64, source.len()));
        }
        return Ok(Arc::new(StructArray::try_new(
            target_children.clone(),
            columns,
            source.nulls().cloned(),
        )?));
    }

    match (source_field.data_type(), target_field.data_type()) {
        (DataType::Struct(source_children), DataType::Struct(target_children)) => {
            let source = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "Field '{}' has non-struct array type {}",
                        source_field.name(),
                        array.data_type()
                    ))
                })?;
            let columns = source_children
                .iter()
                .zip(target_children.iter())
                .zip(source.columns())
                .map(|((source_child, target_child), column)| {
                    canonical_blob_array(source_child, target_child, column.clone())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(Arc::new(StructArray::try_new(
                target_children.clone(),
                columns,
                source.nulls().cloned(),
            )?))
        }
        (DataType::List(source_child), DataType::List(target_child)) => {
            let source = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "Field '{}' has non-list array type {}",
                    source_field.name(),
                    array.data_type()
                ))
            })?;
            let values = canonical_blob_array(source_child, target_child, source.values().clone())?;
            Ok(Arc::new(ListArray::try_new(
                target_child.clone(),
                source.offsets().clone(),
                values,
                source.nulls().cloned(),
            )?))
        }
        (DataType::LargeList(source_child), DataType::LargeList(target_child)) => {
            let source = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "Field '{}' has non-large-list array type {}",
                        source_field.name(),
                        array.data_type()
                    ))
                })?;
            let values = canonical_blob_array(source_child, target_child, source.values().clone())?;
            Ok(Arc::new(LargeListArray::try_new(
                target_child.clone(),
                source.offsets().clone(),
                values,
                source.nulls().cloned(),
            )?))
        }
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Cannot normalize update field '{}' from {} to {}",
            source_field.name(),
            source_field.data_type(),
            target_field.data_type()
        ))),
    }
}

fn canonicalize_blob_reader(
    reader: Box<dyn RecordBatchReader + Send>,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let source_schema = reader.schema();
    let target_fields = source_schema
        .fields()
        .iter()
        .map(|field| canonical_blob_field(field))
        .collect::<Result<Vec<_>>>()?;
    let target_schema = Arc::new(ArrowSchema::new_with_metadata(
        target_fields,
        source_schema.metadata().clone(),
    ));
    if source_schema.fields() == target_schema.fields() {
        return Ok(reader);
    }

    let output_schema = target_schema.clone();
    let converted = reader.map(move |batch| {
        let batch = batch?;
        let columns = batch
            .schema()
            .fields()
            .iter()
            .zip(output_schema.fields())
            .zip(batch.columns())
            .map(|((source_field, target_field), column)| {
                canonical_blob_array(source_field, target_field, column.clone())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(output_schema.clone(), columns)
    });
    Ok(Box::new(RecordBatchIterator::new(converted, target_schema)))
}

struct BlobUpdateContext {
    dataset: Arc<super::Dataset>,
    has_blob_v2: bool,
    blob_handling: Option<BlobHandling>,
    external_base_resolver: Option<Arc<blob::ExternalBaseResolver>>,
}

impl BlobUpdateContext {
    async fn try_new(
        fragment: &FileFragment,
        read_columns: &[String],
        write_schema: &Schema,
    ) -> Result<Self> {
        let selected_field_ids = read_columns
            .iter()
            .filter_map(|column| fragment.schema().field(column))
            .map(|field| field.id)
            .collect::<Vec<_>>();
        let descriptor_blob_ids = fragment
            .schema()
            .project_by_ids(&selected_field_ids, true)
            .fields_pre_order()
            .filter(|field| field.is_blob_v2())
            .filter_map(|field| u32::try_from(field.id).ok())
            .collect::<HashSet<_>>();
        let has_blob_v2 = !descriptor_blob_ids.is_empty();
        let blob_handling = has_blob_v2.then(|| {
            let materialized_blob_ids = fragment
                .schema()
                .fields_pre_order()
                .filter(|field| field.is_blob())
                .filter_map(|field| u32::try_from(field.id).ok())
                .filter(|field_id| !descriptor_blob_ids.contains(field_id))
                .collect();
            BlobHandling::SomeBlobsBinary(materialized_blob_ids)
        });
        let external_base_resolver = if has_blob_v2 {
            super::write::blob_v2_external_base_resolver(
                Some(fragment.dataset()),
                &WriteParams::default(),
                write_schema,
            )
            .await?
        } else {
            None
        };
        Ok(Self {
            dataset: Arc::new(fragment.dataset().clone()),
            has_blob_v2,
            blob_handling,
            external_base_resolver,
        })
    }

    async fn transform_input(
        &self,
        fragment: &FileFragment,
        batch: &RecordBatch,
    ) -> Result<RecordBatch> {
        if self.has_blob_v2 {
            super::optimize::transform_blob_v2_batch(
                &self.dataset,
                fragment.schema(),
                batch.clone(),
                true,
            )
            .await
        } else {
            Ok(batch.clone())
        }
    }

    async fn validate_updates(&self, batch: &RecordBatch, matched_rows: &[bool]) -> Result<()> {
        if let Some(resolver) = self.external_base_resolver.as_deref() {
            blob::validate_external_blob_references(resolver, batch, matched_rows).await?;
        }
        Ok(())
    }
}

async fn prepare_rhs(
    reader: Box<dyn RecordBatchReader + Send>,
    right_on: &str,
    options: &UpdateColumnsOptions,
) -> Result<PreparedRhs> {
    let schema = reader.schema();
    let key_index = schema.index_of(right_on)?;
    let resolved_pool_size = options.execution_options.mem_pool_size();
    let hash_byte_budget = options.max_hash_bytes;

    let mut unread = reader_to_stream(reader);
    let mut prefetched = Vec::new();
    let mut all_buffers = MemoryAccumulator::default();
    let mut key_buffers = MemoryAccumulator::default();
    let mut row_count = 0usize;

    while let Some(batch) = unread.next().await {
        let batch = batch?;
        all_buffers.record_batch(&batch);
        key_buffers.record_array(batch.column(key_index).as_ref());
        let next_row_count = row_count.checked_add(batch.num_rows());
        let estimated_bytes = next_row_count.and_then(|rows| {
            key_buffers
                .total()
                .checked_mul(2)
                .and_then(|key_bytes| all_buffers.total().checked_add(key_bytes))
                .and_then(|buffer_bytes| {
                    rows.checked_mul(64)
                        .and_then(|row_bytes| buffer_bytes.checked_add(row_bytes))
                })
        });
        prefetched.push(batch);

        let is_within_limits = next_row_count
            .zip(estimated_bytes)
            .map(|(rows, bytes)| rows <= options.max_hash_rows && bytes <= hash_byte_budget)
            .unwrap_or(false);
        row_count = next_row_count.unwrap_or(usize::MAX);

        if !is_within_limits {
            tracing::debug!(
                strategy = "external",
                estimated_bytes = estimated_bytes.unwrap_or(u64::MAX as usize),
                rows = row_count,
                hash_byte_budget,
                resolved_pool_size,
                "selected update-columns join strategy"
            );
            let replay =
                stream::iter(prefetched.into_iter().map(Ok::<_, DataFusionError>)).chain(unread);
            return Ok(PreparedRhs::External(Box::pin(
                RecordBatchStreamAdapter::new(schema, replay),
            )));
        }
    }

    let estimated_bytes = key_buffers
        .total()
        .checked_mul(2)
        .and_then(|key_bytes| all_buffers.total().checked_add(key_bytes))
        .and_then(|buffer_bytes| {
            row_count
                .checked_mul(64)
                .and_then(|row_bytes| buffer_bytes.checked_add(row_bytes))
        })
        .unwrap_or(usize::MAX);
    tracing::debug!(
        strategy = "hash",
        estimated_bytes,
        rows = row_count,
        hash_byte_budget,
        resolved_pool_size,
        "selected update-columns join strategy"
    );
    Ok(PreparedRhs::InMemory(prefetched))
}

pub(super) async fn update_columns_with_options(
    fragment: &mut FileFragment,
    right_reader: Box<dyn RecordBatchReader + Send>,
    left_on: &str,
    right_on: &str,
    options: UpdateColumnsOptions,
) -> Result<FragmentUpdateColumnsResult> {
    if fragment.schema().field(left_on).is_none() && left_on != ROW_ID && left_on != ROW_ADDR {
        return Err(Error::invalid_input(format!(
            "Column {} does not exist in the left side fragment",
            left_on
        )));
    }

    let original_right_schema = right_reader.schema();
    if original_right_schema.field_with_name(right_on).is_err() {
        return Err(Error::invalid_input(format!(
            "Column {} does not exist in the right side fragment",
            right_on
        )));
    }
    let requested_write_schema = original_right_schema.as_ref().without_column(right_on);
    for field in requested_write_schema.fields() {
        if field.name() == ROW_ID || field.name() == ROW_ADDR {
            return Err(Error::invalid_input(format!(
                "Column {} is a reserved metadata column and cannot be updated",
                field.name()
            )));
        }
        if fragment.schema().field(field.name()).is_none() {
            return Err(Error::invalid_input(format!(
                "Column {} in right side fragment does not exist in left side fragment",
                field.name()
            )));
        }
    }
    let write_schema = fragment.schema().project_by_schema(
        &requested_write_schema,
        OnMissing::Error,
        OnTypeMismatch::Error,
    )?;

    // Match the physical representation returned by fragment scans (for example,
    // JSONB and offset-based string/binary arrays).
    let right_reader = SchemaAdapter::new(original_right_schema).to_physical_reader(right_reader);
    let right_reader = canonicalize_blob_reader(right_reader)?;
    let right_schema = right_reader.schema();
    validate_key_types(fragment, left_on, &right_schema, right_on)?;

    match prepare_rhs(right_reader, right_on, &options).await? {
        PreparedRhs::InMemory(batches) => {
            run_hash_update(
                fragment,
                batches,
                right_schema,
                left_on,
                right_on,
                write_schema,
            )
            .await
        }
        PreparedRhs::External(stream) => {
            if !options.execution_options.use_spilling() {
                return Err(Error::not_supported(
                    "A large fragment update requires DataFusion spill support, but spilling is disabled"
                        .to_string(),
                ));
            }
            #[cfg(test)]
            let session_context = options.session_context;
            #[cfg(not(test))]
            let session_context = None;
            run_external_update(
                fragment,
                stream,
                left_on,
                right_on,
                write_schema,
                options.execution_options,
                session_context,
            )
            .await
        }
    }
}

fn validate_key_types(
    fragment: &FileFragment,
    left_on: &str,
    right_schema: &ArrowSchema,
    right_on: &str,
) -> Result<()> {
    let left_type = match left_on {
        ROW_ID | ROW_ADDR => DataType::UInt64,
        _ => fragment
            .schema()
            .field(left_on)
            .ok_or_else(|| Error::invalid_input(format!("Column {} does not exist", left_on)))?
            .data_type(),
    };
    let right_type = right_schema.field_with_name(right_on)?.data_type();
    if &left_type != right_type {
        return Err(Error::invalid_input(format!(
            "Join key type mismatch: left key '{}' has type {}, but right key '{}' has type {}",
            left_on, left_type, right_on, right_type
        )));
    }
    Ok(())
}

async fn run_hash_update(
    fragment: &FileFragment,
    batches: Vec<RecordBatch>,
    right_schema: SchemaRef,
    left_on: &str,
    right_on: &str,
    write_schema: Schema,
) -> Result<FragmentUpdateColumnsResult> {
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        right_schema,
    ));
    let joiner = Arc::new(HashJoiner::try_new(reader, right_on).await?);

    let mut read_columns = write_schema
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    if !read_columns.iter().any(|name| name == left_on) {
        read_columns.push(left_on.to_string());
    }
    if !read_columns.iter().any(|name| name == ROW_ADDR) {
        read_columns.push(ROW_ADDR.to_string());
    }
    let blob_context = BlobUpdateContext::try_new(fragment, &read_columns, &write_schema).await?;
    let mut updater = fragment
        .updater(
            Some(&read_columns),
            Some((write_schema, fragment.schema().clone())),
            None,
            blob_context.blob_handling.clone(),
        )
        .await?;
    if blob_context.has_blob_v2 {
        updater.allow_external_blob_outside_bases();
    }
    let fragment_id = fragment_id(fragment)?;
    let mut matched_offsets = RoaringBitmap::new();

    while let Some(batch) = updater.next().await? {
        let batch = blob_context.transform_input(fragment, batch).await?;
        let index_column = batch
            .column_by_name(left_on)
            .ok_or_else(|| Error::internal(format!("Updater did not return join key '{left_on}'")))?
            .clone();
        let matched_rows = joiner.matched_join_rows(index_column.clone())?;
        if let Some(addresses) = batch
            .column_by_name(ROW_ADDR)
            .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
        {
            for (row_index, is_matched) in matched_rows.iter().copied().enumerate() {
                if !is_matched || addresses.is_null(row_index) {
                    continue;
                }
                let address = RowAddress::from(addresses.value(row_index));
                if address.fragment_id() == fragment_id {
                    matched_offsets.insert(address.row_offset());
                }
            }
        }
        let updated_batch = joiner
            .collect_with_fallback(&batch, index_column, fragment.dataset())
            .await?;
        blob_context
            .validate_updates(&updated_batch, &matched_rows)
            .await?;
        updater.update(updated_batch).await?;
    }

    let updated_fragment = updater.finish().await?;
    finalize_update(updated_fragment, matched_offsets)
}

async fn run_external_update(
    fragment: &FileFragment,
    right_stream: SendableRecordBatchStream,
    left_on: &str,
    right_on: &str,
    write_schema: Schema,
    mut execution_options: LanceExecutionOptions,
    session_context: Option<SessionContext>,
) -> Result<FragmentUpdateColumnsResult> {
    execution_options.batch_size = Some(EXECUTION_BATCH_SIZE);
    execution_options.target_partition = Some(1);
    let pool_size = usize::try_from(execution_options.mem_pool_size()).unwrap_or(usize::MAX);
    let sort_merge_reservation = MAX_SORT_MERGE_RESERVATION.min(pool_size / 8).max(1);
    let input_batch_cap = MAX_INPUT_BATCH_BYTES.min(pool_size / 8).max(1);
    let input_batch_cap_u64 = u64::try_from(input_batch_cap).unwrap_or(u64::MAX);
    let temp_directory_limit = execution_options.max_temp_directory_size();
    if temp_directory_limit < input_batch_cap_u64 {
        return Err(DataFusionError::ResourcesExhausted(format!(
            "Temporary disk limit for a large update is {temp_directory_limit} bytes, but at least \
             one capped input batch ({input_batch_cap_u64} bytes) of spill storage is required"
        ))
        .into());
    }

    let mut scanner = fragment.scan();
    let mut left_projection = vec![left_on];
    if left_on != ROW_ADDR {
        left_projection.push(ROW_ADDR);
    }
    scanner
        .project(&left_projection)?
        .batch_size(EXECUTION_BATCH_SIZE)
        .batch_size_bytes(u64::try_from(input_batch_cap).unwrap_or(u64::MAX));
    let left_stream = scanner.try_into_dfstream(execution_options.clone()).await?;

    let session = session_context.unwrap_or_else(|| new_session_context(&execution_options));
    let mut state = session.state();
    state
        .config_mut()
        .options_mut()
        .execution
        .sort_spill_reservation_bytes = sort_merge_reservation;
    state.config_mut().options_mut().execution.batch_size = EXECUTION_BATCH_SIZE;
    let task_context = state.task_ctx();

    let left_source = Arc::new(OneShotExec::new(left_stream)) as Arc<dyn ExecutionPlan>;
    let right_schema = right_stream.schema();
    let right_source = Arc::new(OneShotExec::new(right_stream)) as Arc<dyn ExecutionPlan>;
    let sorted_left = sort_by_column(left_source, left_on, input_batch_cap)?;
    let sorted_right = sort_by_column(right_source, right_on, input_batch_cap)?;

    let sorted_right_stream = sorted_right.execute(0, task_context.clone())?;
    let deduplicated_right = deduplicate_sorted_stream(sorted_right_stream, right_on)?;
    let deduplicated_right =
        Arc::new(OneShotExec::new(deduplicated_right)) as Arc<dyn ExecutionPlan>;

    let left_key_index = sorted_left.schema().index_of(left_on)?;
    let right_key_index = right_schema.index_of(right_on)?;
    let join = Arc::new(SortMergeJoinExec::try_new(
        sorted_left.clone(),
        deduplicated_right,
        vec![(
            Arc::new(Column::new(left_on, left_key_index)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new(right_on, right_key_index)) as Arc<dyn PhysicalExpr>,
        )],
        None,
        JoinType::Inner,
        vec![SortOptions {
            descending: false,
            nulls_first: true,
        }],
        NullEquality::NullEqualsNull,
    )?);

    let left_column_count = sorted_left.schema().fields().len();
    let left_address_index = sorted_left.schema().index_of(ROW_ADDR)?;
    let mut patch_projection = Vec::with_capacity(write_schema.fields.len() + 1);
    patch_projection.push((
        Arc::new(Column::new(ROW_ADDR, left_address_index)) as Arc<dyn PhysicalExpr>,
        ROW_ADDR.to_string(),
    ));
    for field in &write_schema.fields {
        let right_index = right_schema.index_of(&field.name)?;
        patch_projection.push((
            Arc::new(Column::new(&field.name, left_column_count + right_index))
                as Arc<dyn PhysicalExpr>,
            field.name.clone(),
        ));
    }
    let projected_patches =
        Arc::new(ProjectionExec::try_new(patch_projection, join)?) as Arc<dyn ExecutionPlan>;
    let capped_patches = Arc::new(HardCapBatchSizeExec::new(
        projected_patches,
        input_batch_cap,
    )) as Arc<dyn ExecutionPlan>;
    let sorted_patches = sort_by_column_without_cap(capped_patches, ROW_ADDR)?;
    let patch_stream = sorted_patches.execute(0, task_context)?;

    let fragment_id = fragment_id(fragment)?;
    let mut patch_cursor = SortedPatchCursor::try_new(patch_stream, fragment_id).await?;
    let mut update_projection = write_schema
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    if !update_projection.iter().any(|name| name == ROW_ADDR) {
        update_projection.push(ROW_ADDR.to_string());
    }
    let blob_context =
        BlobUpdateContext::try_new(fragment, &update_projection, &write_schema).await?;
    let mut updater = fragment
        .updater(
            Some(&update_projection),
            Some((write_schema, fragment.schema().clone())),
            Some(EXECUTION_BATCH_SIZE as u32),
            blob_context.blob_handling.clone(),
        )
        .await?;
    if blob_context.has_blob_v2 {
        updater.allow_external_blob_outside_bases();
    }
    let mut matched_offsets = RoaringBitmap::new();

    while let Some(batch) = updater.next().await? {
        let batch = blob_context.transform_input(fragment, batch).await?;
        let (updated_batch, matched_rows) = patch_cursor
            .merge_batch(&batch, &mut matched_offsets)
            .await?;
        for column in updated_batch.columns() {
            HashJoiner::check_lance_support_null(column, fragment.dataset())?;
        }
        blob_context
            .validate_updates(&updated_batch, &matched_rows)
            .await?;
        updater.update(updated_batch).await?;
    }
    patch_cursor.ensure_exhausted()?;

    if let Some(callback) = execution_options.execution_stats_callback.as_ref() {
        let mut counts = ExecutionSummaryCounts::default();
        collect_execution_metrics(sorted_right.as_ref(), &mut counts);
        collect_spill_metrics(sorted_right.as_ref(), &mut counts);
        collect_execution_metrics(sorted_patches.as_ref(), &mut counts);
        collect_spill_metrics(sorted_patches.as_ref(), &mut counts);
        callback(&counts);
    }

    let updated_fragment = updater.finish().await?;
    finalize_update(updated_fragment, matched_offsets)
}

fn sort_by_column(
    input: Arc<dyn ExecutionPlan>,
    column_name: &str,
    input_batch_cap: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let capped = Arc::new(HardCapBatchSizeExec::new(input, input_batch_cap));
    sort_by_column_without_cap(capped, column_name)
}

fn sort_by_column_without_cap(
    input: Arc<dyn ExecutionPlan>,
    column_name: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let column_index = input.schema().index_of(column_name)?;
    let sort_expression = PhysicalSortExpr {
        expr: Arc::new(Column::new(column_name, column_index)),
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    };
    Ok(Arc::new(SortExec::new([sort_expression].into(), input)))
}

fn collect_spill_metrics(plan: &dyn ExecutionPlan, counts: &mut ExecutionSummaryCounts) {
    if let Some(metrics) = plan.metrics() {
        for metric in metrics.iter() {
            let (name, value) = match metric.value() {
                MetricValue::SpillCount(value) => ("spill_count", value.value()),
                MetricValue::SpilledBytes(value) => ("spilled_bytes", value.value()),
                MetricValue::SpilledRows(value) => ("spilled_rows", value.value()),
                _ => continue,
            };
            *counts.all_counts.entry(name.to_string()).or_default() += value;
        }
    }
    for child in plan.children() {
        collect_spill_metrics(child.as_ref(), counts);
    }
}

struct DedupState {
    input: SendableRecordBatchStream,
    converter: RowConverter,
    key_index: usize,
    previous_key: Option<OwnedRow>,
}

fn deduplicate_sorted_stream(
    input: SendableRecordBatchStream,
    key_name: &str,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let key_index = schema.index_of(key_name)?;
    let converter = RowConverter::new(vec![SortField::new(
        schema.field(key_index).data_type().clone(),
    )])?;
    let state = DedupState {
        input,
        converter,
        key_index,
        previous_key: None,
    };
    let output = stream::try_unfold(state, |mut state| async move {
        loop {
            let Some(batch) = state.input.next().await else {
                return Ok(None);
            };
            let batch = batch?;
            let keys = state
                .converter
                .convert_columns(&[batch.column(state.key_index).clone()])?;
            let mut keep = Vec::with_capacity(batch.num_rows());
            for (row_index, key) in keys.iter().enumerate() {
                let owned_key = key.owned();
                if state.previous_key.as_ref() != Some(&owned_key) {
                    keep.push(u32::try_from(row_index).map_err(|_| {
                        DataFusionError::External(Box::new(Error::invalid_input(format!(
                            "RHS update batch row index {row_index} does not fit in UInt32"
                        ))))
                    })?);
                    state.previous_key = Some(owned_key);
                }
            }
            if keep.is_empty() {
                continue;
            }
            let batch = arrow_select::take::take_record_batch(&batch, &UInt32Array::from(keep))?;
            return Ok(Some((batch, state)));
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
}

struct SortedPatchCursor {
    input: SendableRecordBatchStream,
    payload_schema: SchemaRef,
    fragment_id: u32,
    current_batch: Option<RecordBatch>,
    current_position: usize,
    current_batch_id: u64,
    last_patch_address: Option<u64>,
    last_original_address: Option<u64>,
}

impl SortedPatchCursor {
    async fn try_new(input: SendableRecordBatchStream, fragment_id: u32) -> Result<Self> {
        let input_schema = input.schema();
        let address_index = input_schema.index_of(ROW_ADDR)?;
        if address_index != 0 {
            return Err(Error::internal(format!(
                "Sorted patch address column must be at position 0, got {address_index}"
            )));
        }
        let payload_fields: Vec<FieldRef> = input_schema.fields().iter().skip(1).cloned().collect();
        let payload_schema = Arc::new(ArrowSchema::new_with_metadata(
            payload_fields,
            input_schema.metadata().clone(),
        ));
        let mut cursor = Self {
            input,
            payload_schema,
            fragment_id,
            current_batch: None,
            current_position: 0,
            current_batch_id: 0,
            last_patch_address: None,
            last_original_address: None,
        };
        cursor.load_next_batch().await?;
        Ok(cursor)
    }

    async fn load_next_batch(&mut self) -> Result<()> {
        loop {
            let Some(batch) = self.input.next().await else {
                self.current_batch = None;
                self.current_position = 0;
                return Ok(());
            };
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let addresses = batch
                .column_by_name(ROW_ADDR)
                .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Sorted patch column '{ROW_ADDR}' must be a UInt64 array"
                    ))
                })?;
            for row_index in 0..addresses.len() {
                if addresses.is_null(row_index) {
                    return Err(Error::internal(format!(
                        "Sorted patch address is null at row {row_index}"
                    )));
                }
                let value = addresses.value(row_index);
                let address = RowAddress::from(value);
                if address.fragment_id() != self.fragment_id {
                    return Err(Error::internal(format!(
                        "Sorted patch address {address} belongs to fragment {}, expected fragment {}",
                        address.fragment_id(),
                        self.fragment_id
                    )));
                }
                if let Some(previous) = self.last_patch_address {
                    if value == previous {
                        return Err(Error::internal(format!(
                            "Sorted patch stream contains duplicate address {address}"
                        )));
                    }
                    if value < previous {
                        return Err(Error::internal(format!(
                            "Sorted patch addresses are out of order: {value} follows {previous}"
                        )));
                    }
                }
                self.last_patch_address = Some(value);
            }
            self.current_batch_id = self.current_batch_id.checked_add(1).ok_or_else(|| {
                Error::internal("Sorted patch batch identifier overflowed".to_string())
            })?;
            self.current_batch = Some(batch);
            self.current_position = 0;
            return Ok(());
        }
    }

    async fn merge_batch(
        &mut self,
        original: &RecordBatch,
        matched_offsets: &mut RoaringBitmap,
    ) -> Result<(RecordBatch, Vec<bool>)> {
        let original_addresses = original
            .column_by_name(ROW_ADDR)
            .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| {
                Error::internal(format!(
                    "Updater column '{ROW_ADDR}' must be a UInt64 array"
                ))
            })?;
        let original_columns = self
            .payload_schema
            .fields()
            .iter()
            .map(|field| {
                original
                    .column_by_name(field.name())
                    .cloned()
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Updater did not return update column '{}'",
                            field.name()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let original_payload = RecordBatch::try_new(self.payload_schema.clone(), original_columns)?;
        let mut source_batches = vec![original_payload];
        let mut indices = Vec::with_capacity(original.num_rows());
        let mut matched_rows = Vec::with_capacity(original.num_rows());
        let mut local_patch_source = None;

        for row_index in 0..original.num_rows() {
            if original_addresses.is_null(row_index) {
                return Err(Error::internal(format!(
                    "Updater address is null at row {row_index}"
                )));
            }
            let original_value = original_addresses.value(row_index);
            let original_address = RowAddress::from(original_value);
            if original_address.fragment_id() != self.fragment_id {
                return Err(Error::internal(format!(
                    "Updater address {original_address} belongs to fragment {}, expected fragment {}",
                    original_address.fragment_id(),
                    self.fragment_id
                )));
            }
            if let Some(previous) = self.last_original_address
                && original_value <= previous
            {
                return Err(Error::internal(format!(
                    "Updater addresses must be strictly increasing: {original_value} follows {previous}"
                )));
            }
            self.last_original_address = Some(original_value);

            let Some(patch_batch) = self.current_batch.as_ref() else {
                indices.push((0, row_index));
                matched_rows.push(false);
                continue;
            };
            let patch_addresses = patch_batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Sorted patch column '{ROW_ADDR}' must be a UInt64 array"
                    ))
                })?;
            let patch_value = patch_addresses.value(self.current_position);
            if patch_value < original_value {
                return Err(Error::internal(format!(
                    "Sorted patch address {} precedes the current live fragment row {}",
                    RowAddress::from(patch_value),
                    original_address
                )));
            }
            if patch_value > original_value {
                indices.push((0, row_index));
                matched_rows.push(false);
                continue;
            }

            let source_index = match local_patch_source {
                Some((batch_id, source_index)) if batch_id == self.current_batch_id => source_index,
                _ => {
                    let payload_indices = (1..patch_batch.num_columns()).collect::<Vec<_>>();
                    let payload = patch_batch.project(&payload_indices)?;
                    source_batches.push(payload);
                    let source_index = source_batches.len() - 1;
                    local_patch_source = Some((self.current_batch_id, source_index));
                    source_index
                }
            };
            indices.push((source_index, self.current_position));
            matched_rows.push(true);
            matched_offsets.insert(original_address.row_offset());
            self.current_position += 1;
            if self.current_position == patch_batch.num_rows() {
                self.load_next_batch().await?;
                local_patch_source = None;
            }
        }

        Ok((interleave_batches(&source_batches, &indices)?, matched_rows))
    }

    fn ensure_exhausted(&self) -> Result<()> {
        if let Some(batch) = self.current_batch.as_ref() {
            let addresses = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Sorted patch column '{ROW_ADDR}' must be a UInt64 array"
                    ))
                })?;
            let address = RowAddress::from(addresses.value(self.current_position));
            return Err(Error::internal(format!(
                "Sorted patch stream still contains address {address} after the updater was exhausted"
            )));
        }
        Ok(())
    }
}

fn fragment_id(fragment: &FileFragment) -> Result<u32> {
    u32::try_from(fragment.metadata.id).map_err(|_| {
        Error::invalid_input(format!(
            "Fragment id {} does not fit RowAddress fragment id",
            fragment.metadata.id
        ))
    })
}

fn finalize_update(
    mut updated_fragment: Fragment,
    matched_offsets: RoaringBitmap,
) -> Result<FragmentUpdateColumnsResult> {
    let updated_fields = updated_fragment
        .files
        .last()
        .ok_or_else(|| Error::internal("Updated fragment has no data files".to_string()))?
        .fields
        .clone();
    for data_file in updated_fragment.files.iter_mut().rev().skip(1) {
        data_file.fields = data_file
            .fields
            .iter()
            .map(|field| {
                if updated_fields.contains(field) {
                    -2
                } else {
                    *field
                }
            })
            .collect::<Vec<_>>()
            .into();
    }
    updated_fragment
        .files
        .retain(|data_file| data_file.fields.iter().any(|field| *field != -2));
    let fields_modified = updated_fields
        .iter()
        .filter_map(|field| u32::try_from(*field).ok())
        .collect();
    Ok(FragmentUpdateColumnsResult {
        fragment: updated_fragment,
        fields_modified,
        matched_offsets,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;
    use std::sync::Mutex;

    use arrow_array::types::Int32Type;
    use arrow_array::{
        ArrayRef, Int32Array, Int64Array, LargeBinaryArray, ListArray, StringArray, record_batch,
    };
    use arrow_schema::{ArrowError, Field};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::TryStreamExt;
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_table::format::RowIdMeta;
    use lance_table::rowids::{RowIdSequence, write_row_ids};
    use rstest::rstest;

    use super::*;
    use crate::{BlobArrayBuilder, Dataset, blob_field, dataset::WriteParams};

    fn reader(
        schema: SchemaRef,
        batches: Vec<std::result::Result<RecordBatch, ArrowError>>,
    ) -> Box<dyn RecordBatchReader + Send> {
        Box::new(RecordBatchIterator::new(batches, schema))
    }

    fn stream_from_batches(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    fn options(max_hash_rows: usize, max_hash_bytes: usize) -> UpdateColumnsOptions {
        UpdateColumnsOptions {
            max_hash_rows,
            max_hash_bytes,
            ..Default::default()
        }
    }

    fn list_batch(keys: Vec<i32>, is_update: bool) -> RecordBatch {
        const VALUES_PER_ROW: usize = 128;
        let lists = ListArray::from_iter_primitive::<Int32Type, _, _>(keys.iter().map(|key| {
            let value = if is_update { *key } else { -1 };
            Some((0..VALUES_PER_ROW).map(|_| Some(value)).collect::<Vec<_>>())
        }));
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("payload", lists.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(lists),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_prepare_rhs_limits_and_empty_batches() {
        let defaults = UpdateColumnsOptions::default();
        assert_eq!(defaults.max_hash_rows, 1_000_000);
        assert_eq!(defaults.max_hash_bytes, 256 * 1024 * 1024);

        let batch = record_batch!(("key", Int32, [1, 2]), ("value", Int32, [10, 20])).unwrap();
        let schema = batch.schema();

        let prepared = prepare_rhs(
            reader(schema.clone(), vec![Ok(batch.clone())]),
            "key",
            &options(2, usize::MAX),
        )
        .await
        .unwrap();
        assert!(matches!(prepared, PreparedRhs::InMemory(batches) if batches.len() == 1));

        let prepared = prepare_rhs(
            reader(schema.clone(), vec![Ok(batch.clone())]),
            "key",
            &options(1, usize::MAX),
        )
        .await
        .unwrap();
        assert!(matches!(prepared, PreparedRhs::External(_)));

        let prepared = prepare_rhs(
            reader(schema.clone(), vec![Ok(batch)]),
            "key",
            &options(usize::MAX, 1),
        )
        .await
        .unwrap();
        assert!(matches!(prepared, PreparedRhs::External(_)));

        let prepared = prepare_rhs(reader(schema.clone(), vec![]), "key", &options(1, 1))
            .await
            .unwrap();
        assert!(matches!(prepared, PreparedRhs::InMemory(batches) if batches.is_empty()));

        let empty = RecordBatch::new_empty(schema.clone());
        let prepared = prepare_rhs(
            reader(schema, vec![Ok(empty)]),
            "key",
            &options(1, usize::MAX),
        )
        .await
        .unwrap();
        assert!(
            matches!(prepared, PreparedRhs::InMemory(batches) if batches.len() == 1 && batches[0].num_rows() == 0)
        );
    }

    #[tokio::test]
    async fn test_external_rhs_replays_prefetch_before_remainder_error() {
        let batch = record_batch!(("key", Int32, [1]), ("value", Int32, [10])).unwrap();
        let schema = batch.schema();
        let error = ArrowError::ExternalError(Box::new(io::Error::other("remaining input failed")));
        let prepared = prepare_rhs(
            reader(schema, vec![Ok(batch.clone()), Err(error)]),
            "key",
            &options(0, usize::MAX),
        )
        .await
        .unwrap();
        let PreparedRhs::External(mut replayed) = prepared else {
            panic!("row limit should select external preparation");
        };
        assert_eq!(replayed.next().await.unwrap().unwrap(), batch);
        let error = replayed.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("remaining input failed"));
    }

    #[tokio::test]
    async fn test_deduplicate_across_batches_with_null_payloads() {
        let first = record_batch!(
            ("key", Int32, [None, None, Some(1), Some(1)]),
            ("value", Int32, [Some(10), None, None, Some(11)])
        )
        .unwrap();
        let second = record_batch!(
            ("key", Int32, [Some(1), Some(2), Some(2)]),
            ("value", Int32, [Some(12), Some(20), Some(21)])
        )
        .unwrap();
        let schema = first.schema();
        let deduplicated = deduplicate_sorted_stream(
            stream_from_batches(schema.clone(), vec![first, second]),
            "key",
        )
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
        let result = arrow::compute::concat_batches(&schema, &deduplicated).unwrap();
        assert_eq!(
            result.column_by_name("key").unwrap().as_ref(),
            &Int32Array::from(vec![None, Some(1), Some(2)])
        );
        assert_eq!(
            result.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![Some(10), None, Some(20)])
        );
    }

    #[tokio::test]
    async fn test_patch_cursor_merges_across_batch_boundaries() {
        let address = |offset| u64::from(RowAddress::new_from_parts(7, offset));
        let first_patch = record_batch!(
            (ROW_ADDR, UInt64, [address(1), address(3)]),
            ("value", Int32, [101, 103])
        )
        .unwrap();
        let second_patch =
            record_batch!((ROW_ADDR, UInt64, [address(4)]), ("value", Int32, [104])).unwrap();
        let patch_schema = first_patch.schema();
        let mut cursor = SortedPatchCursor::try_new(
            stream_from_batches(patch_schema, vec![first_patch, second_patch]),
            7,
        )
        .await
        .unwrap();
        let first_original = record_batch!(
            ("value", Int32, [10, 11, 12]),
            (ROW_ADDR, UInt64, [address(0), address(1), address(2)])
        )
        .unwrap();
        let second_original = record_batch!(
            ("value", Int32, [13, 14, 15]),
            (ROW_ADDR, UInt64, [address(3), address(4), address(5)])
        )
        .unwrap();
        let mut matched = RoaringBitmap::new();
        let (first, first_matched) = cursor
            .merge_batch(&first_original, &mut matched)
            .await
            .unwrap();
        let (second, second_matched) = cursor
            .merge_batch(&second_original, &mut matched)
            .await
            .unwrap();
        cursor.ensure_exhausted().unwrap();
        assert_eq!(first_matched, vec![false, true, false]);
        assert_eq!(second_matched, vec![true, true, false]);
        assert_eq!(
            first.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![10, 101, 12])
        );
        assert_eq!(
            second.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![103, 104, 15])
        );
        assert_eq!(matched.iter().collect::<Vec<_>>(), vec![1, 3, 4]);
    }

    #[tokio::test]
    async fn test_patch_cursor_rejects_invalid_addresses() {
        let address =
            |fragment_id, offset| u64::from(RowAddress::new_from_parts(fragment_id, offset));
        let duplicate = record_batch!(
            (ROW_ADDR, UInt64, [address(7, 1), address(7, 1)]),
            ("value", Int32, [10, 11])
        )
        .unwrap();
        let Err(error) =
            SortedPatchCursor::try_new(stream_from_batches(duplicate.schema(), vec![duplicate]), 7)
                .await
        else {
            panic!("duplicate patch addresses should fail");
        };
        assert!(error.to_string().contains("duplicate address"));

        let wrong_fragment =
            record_batch!((ROW_ADDR, UInt64, [address(8, 1)]), ("value", Int32, [10])).unwrap();
        let Err(error) = SortedPatchCursor::try_new(
            stream_from_batches(wrong_fragment.schema(), vec![wrong_fragment]),
            7,
        )
        .await
        else {
            panic!("wrong-fragment patch addresses should fail");
        };
        assert!(error.to_string().contains("expected fragment 7"));

        let nullable_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ADDR, DataType::UInt64, true),
            Field::new("value", DataType::Int32, true),
        ]));
        let null_address = RecordBatch::try_new(
            nullable_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![None])),
                Arc::new(Int32Array::from(vec![10])),
            ],
        )
        .unwrap();
        let Err(error) =
            SortedPatchCursor::try_new(stream_from_batches(nullable_schema, vec![null_address]), 7)
                .await
        else {
            panic!("null patch addresses should fail");
        };
        assert!(error.to_string().contains("address is null"));

        let out_of_order = record_batch!(
            (ROW_ADDR, UInt64, [address(7, 2), address(7, 1)]),
            ("value", Int32, [20, 10])
        )
        .unwrap();
        let Err(error) = SortedPatchCursor::try_new(
            stream_from_batches(out_of_order.schema(), vec![out_of_order]),
            7,
        )
        .await
        else {
            panic!("out-of-order patch addresses should fail");
        };
        assert!(error.to_string().contains("out of order"));
    }

    #[tokio::test]
    async fn test_patch_cursor_rejects_preceding_and_leftover_patches() {
        let address = |offset| u64::from(RowAddress::new_from_parts(7, offset));
        let patch = record_batch!(
            (ROW_ADDR, UInt64, [address(0), address(2)]),
            ("value", Int32, [100, 102])
        )
        .unwrap();
        let mut cursor =
            SortedPatchCursor::try_new(stream_from_batches(patch.schema(), vec![patch]), 7)
                .await
                .unwrap();
        let original =
            record_batch!(("value", Int32, [11]), (ROW_ADDR, UInt64, [address(1)])).unwrap();
        let error = cursor
            .merge_batch(&original, &mut RoaringBitmap::new())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("precedes the current live"));

        let patch =
            record_batch!((ROW_ADDR, UInt64, [address(2)]), ("value", Int32, [102])).unwrap();
        let cursor =
            SortedPatchCursor::try_new(stream_from_batches(patch.schema(), vec![patch]), 7)
                .await
                .unwrap();
        let error = cursor.ensure_exhausted().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("after the updater was exhausted")
        );
    }

    #[rstest]
    #[case::legacy(LanceFileVersion::Legacy)]
    #[case::current(LanceFileVersion::Stable)]
    #[tokio::test]
    async fn test_hash_and_external_updates_match_for_unsorted_keys(
        #[case] file_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let input = record_batch!(
            ("key", Int32, [4, 1, 3, 1, 2]),
            ("value", Int32, [40, 10, 30, 11, 20])
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            Some(WriteParams {
                data_storage_version: Some(file_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let first_update =
            record_batch!(("rhs_key", Int32, [3, 1]), ("value", Int32, [300, 100])).unwrap();
        let second_update =
            record_batch!(("rhs_key", Int32, [4, 2]), ("value", Int32, [400, 200])).unwrap();
        let update_schema = first_update.schema();

        let run = |mut fragment: FileFragment, options| {
            let update_schema = update_schema.clone();
            let first_update = first_update.clone();
            let second_update = second_update.clone();
            async move {
                update_columns_with_options(
                    &mut fragment,
                    reader(update_schema, vec![Ok(first_update), Ok(second_update)]),
                    "key",
                    "rhs_key",
                    options,
                )
                .await
                .unwrap()
            }
        };
        let hash_result = run(
            dataset.get_fragment(0).unwrap(),
            options(usize::MAX, usize::MAX),
        )
        .await;
        let external_result = run(dataset.get_fragment(0).unwrap(), options(0, usize::MAX)).await;
        assert_eq!(
            hash_result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(hash_result.matched_offsets, external_result.matched_offsets);

        let hash_fragment = FileFragment::new(Arc::new(dataset.clone()), hash_result.fragment);
        let external_fragment =
            FileFragment::new(Arc::new(dataset.clone()), external_result.fragment);
        let hash_batch = hash_fragment.scan().try_into_batch().await.unwrap();
        let external_batch = external_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(hash_batch, external_batch);
        assert_eq!(
            external_batch.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![400, 100, 300, 100, 200])
        );
    }

    #[rstest]
    #[case::hash(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn test_update_preserves_blob_v2_fallback_values(#[case] is_external: bool) {
        let test_dir = TempStrDir::default();
        let mut original_blobs = BlobArrayBuilder::new(3);
        original_blobs.push_bytes(b"one").unwrap();
        original_blobs.push_bytes(b"two").unwrap();
        original_blobs.push_null().unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Int32, false),
            blob_field("payload", true),
        ]));
        let input = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                original_blobs.finish().unwrap(),
            ],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input)], schema),
            test_dir.as_ref(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut updated_blob = BlobArrayBuilder::new(1);
        updated_blob.push_bytes(b"NEW").unwrap();
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Int32, false),
            blob_field("payload", true),
        ]));
        let update = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                updated_blob.finish().unwrap(),
            ],
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(update_schema, vec![Ok(update)]),
            "key",
            "key",
            if is_external {
                options(0, usize::MAX)
            } else {
                options(usize::MAX, usize::MAX)
            },
        )
        .await
        .unwrap();
        assert_eq!(result.matched_offsets.iter().collect::<Vec<_>>(), vec![1]);
        let fragment_id = result.fragment.id;
        let updated_dataset = Dataset::commit(
            test_dir.as_ref(),
            crate::dataset::transaction::Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![result.fragment],
                new_fragments: vec![],
                fields_modified: result.fields_modified,
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(crate::dataset::transaction::UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(
                    crate::dataset::transaction::UpdatedFragmentOffsets(HashMap::from([(
                        fragment_id,
                        result.matched_offsets,
                    )])),
                ),
            },
            Some(dataset.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();
        let mut scanner = updated_dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let batch = scanner.try_into_batch().await.unwrap();
        let payload = batch
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(payload.value(0), b"one");
        assert_eq!(payload.value(1), b"NEW");
        assert!(payload.is_null(2));
    }

    #[rstest]
    #[case::hash(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn test_duplicate_rhs_keys_are_applied_consistently(#[case] is_external: bool) {
        let test_dir = TempStrDir::default();
        let input = record_batch!(
            ("key", Int32, [2, 1, 2, 1]),
            ("value", Int32, [-1, -1, -1, -1])
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        let first_update =
            record_batch!(("rhs_key", Int32, [1, 2]), ("value", Int32, [100, 200])).unwrap();
        let second_update =
            record_batch!(("rhs_key", Int32, [2, 1]), ("value", Int32, [201, 101])).unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(
                first_update.schema(),
                vec![Ok(first_update), Ok(second_update)],
            ),
            "key",
            "rhs_key",
            if is_external {
                options(0, usize::MAX)
            } else {
                options(usize::MAX, usize::MAX)
            },
        )
        .await
        .unwrap();
        assert_eq!(
            result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        let updated_fragment = FileFragment::new(Arc::new(dataset), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        let values = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.value(0), values.value(2));
        assert_eq!(values.value(1), values.value(3));
        assert!([200, 201].contains(&values.value(0)));
        assert!([100, 101].contains(&values.value(1)));
    }

    #[rstest]
    #[case::hash(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn test_join_key_can_also_be_an_updated_column(#[case] is_external: bool) {
        let test_dir = TempStrDir::default();
        let input = record_batch!(("key", Int32, [1, 2]), ("value", Int32, [10, 20])).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        let update = record_batch!(
            ("rhs_key", Int32, [2, 1]),
            ("key", Int32, [22, 11]),
            ("value", Int32, [200, 100])
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(update.schema(), vec![Ok(update)]),
            "key",
            "rhs_key",
            if is_external {
                options(0, usize::MAX)
            } else {
                options(usize::MAX, usize::MAX)
            },
        )
        .await
        .unwrap();
        let updated_fragment = FileFragment::new(Arc::new(dataset), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch.column_by_name("key").unwrap().as_ref(),
            &Int32Array::from(vec![11, 22])
        );
        assert_eq!(
            batch.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![100, 200])
        );
    }

    #[tokio::test]
    async fn test_external_update_offsets_are_fragment_local() {
        let test_dir = TempStrDir::default();
        let input = record_batch!(
            ("key", Int32, [0, 1, 2, 3, 4, 5]),
            ("value", Int32, [10, 11, 12, 13, 14, 15])
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        let update = record_batch!(
            ("key", Int32, [5, 3, 2, 0]),
            ("value", Int32, [500, 300, 200, 100])
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(1).unwrap();
        let original_fragment_id = fragment.metadata().id;
        let result = update_columns_with_options(
            &mut fragment,
            reader(update.schema(), vec![Ok(update)]),
            "key",
            "key",
            options(0, usize::MAX),
        )
        .await
        .unwrap();
        assert_eq!(result.fragment.id, original_fragment_id);
        assert_eq!(
            result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![0, 1]
        );
        let updated_fragment = FileFragment::new(Arc::new(dataset), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![200, 300])
        );
    }

    #[tokio::test]
    async fn test_external_update_nulls_deletions_and_row_address() {
        let test_dir = TempStrDir::default();
        let input = record_batch!(
            ("key", Int32, [Some(4), None, Some(3), Some(1), Some(2)]),
            (
                "value",
                Int32,
                [Some(40), Some(10), Some(30), Some(11), Some(20)]
            )
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        dataset.delete("key = 3").await.unwrap();

        let update = record_batch!(
            ("rhs_key", Int32, [None, Some(2), Some(3), Some(9)]),
            ("value", Int32, [Some(100), None, Some(300), Some(900)])
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(update.schema(), vec![Ok(update)]),
            "key",
            "rhs_key",
            options(0, usize::MAX),
        )
        .await
        .unwrap();
        assert_eq!(
            result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![1, 4]
        );
        let updated_fragment = FileFragment::new(Arc::new(dataset.clone()), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![Some(40), Some(100), Some(11), None])
        );

        let row_address_update = record_batch!(
            (
                ROW_ADDR,
                UInt64,
                [
                    u64::from(RowAddress::new_from_parts(0, 3)),
                    u64::from(RowAddress::new_from_parts(0, 0))
                ]
            ),
            ("value", Int32, [Some(333), Some(444)])
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(row_address_update.schema(), vec![Ok(row_address_update)]),
            ROW_ADDR,
            ROW_ADDR,
            options(0, usize::MAX),
        )
        .await
        .unwrap();
        assert_eq!(
            result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![0, 3]
        );
    }

    #[tokio::test]
    async fn test_external_update_requires_spilling_and_valid_key_types() {
        let test_dir = TempStrDir::default();
        let input = record_batch!(("key", Int32, [1]), ("value", Int32, [10])).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        let update = record_batch!(("key", Int32, [1]), ("value", Int32, [20])).unwrap();
        let mut no_spill = options(0, usize::MAX);
        no_spill.execution_options.use_spilling = false;
        let error = update_columns_with_options(
            &mut dataset.get_fragment(0).unwrap(),
            reader(update.schema(), vec![Ok(update)]),
            "key",
            "key",
            no_spill,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("requires DataFusion spill support")
        );

        let mismatched = record_batch!(("other_key", UInt64, [1]), ("value", Int32, [20])).unwrap();
        let error = update_columns_with_options(
            &mut dataset.get_fragment(0).unwrap(),
            reader(mismatched.schema(), vec![Ok(mismatched)]),
            "key",
            "other_key",
            options(0, usize::MAX),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("left key 'key' has type Int32"));
        assert!(
            error
                .to_string()
                .contains("right key 'other_key' has type UInt64")
        );
    }

    #[tokio::test]
    async fn test_external_update_non_monotonic_stable_row_ids() {
        let test_dir = TempStrDir::default();
        let input = record_batch!(("value", Int32, [0, 1, 2, 3, 4])).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
            test_dir.as_ref(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let stable_row_ids = [50, 10, 40, 20, 30];
        let sequence = RowIdSequence::from(stable_row_ids.as_slice());
        let mut metadata = dataset.get_fragment(0).unwrap().metadata().clone();
        metadata.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
        let mut fragment = FileFragment::new(Arc::new(dataset.clone()), metadata);

        let update = record_batch!(
            (ROW_ID, UInt64, [20, 50, 30]),
            ("value", Int32, [200, 500, 300])
        )
        .unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(update.schema(), vec![Ok(update)]),
            ROW_ID,
            ROW_ID,
            options(0, usize::MAX),
        )
        .await
        .unwrap();
        assert_eq!(
            result.matched_offsets.iter().collect::<Vec<_>>(),
            vec![0, 3, 4]
        );
        let updated_fragment = FileFragment::new(Arc::new(dataset), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch.column_by_name("value").unwrap().as_ref(),
            &Int32Array::from(vec![500, 1, 2, 200, 300])
        );
    }

    #[tokio::test]
    async fn test_external_update_arrow_json_payload() {
        let test_dir = TempStrDir::default();
        let mut json_metadata = HashMap::new();
        json_metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata.clone()),
        ]));
        let input = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![2, 1, 3])),
                Arc::new(StringArray::from(vec![
                    r#"{"old":2}"#,
                    r#"{"old":1}"#,
                    r#"{"old":3}"#,
                ])),
            ],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(input)], schema),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("rhs_key", DataType::Int64, false),
            Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata),
        ]));
        let update = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![3, 1])),
                Arc::new(StringArray::from(vec![
                    r#"{"updated":3}"#,
                    r#"{"updated":1}"#,
                ])),
            ],
        )
        .unwrap();
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(update_schema, vec![Ok(update)]),
            "key",
            "rhs_key",
            options(0, usize::MAX),
        )
        .await
        .unwrap();
        let updated_fragment = FileFragment::new(Arc::new(dataset), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        let meta = batch
            .column_by_name("meta")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(meta.value(0), r#"{"old":2}"#);
        assert_eq!(meta.value(1), r#"{"updated":1}"#);
        assert_eq!(meta.value(2), r#"{"updated":3}"#);
    }

    #[tokio::test]
    async fn test_external_update_spills_and_cleans_up() {
        const ROWS: i32 = 65_536;
        const MATCHED_ROWS: i32 = 512;
        const ROWS_PER_BATCH: i32 = 1_024;
        const MIB: u64 = 1024 * 1024;

        let input_batches = (0..MATCHED_ROWS)
            .step_by(ROWS_PER_BATCH as usize)
            .map(|start| {
                list_batch(
                    (start..(start + ROWS_PER_BATCH).min(MATCHED_ROWS)).collect(),
                    false,
                )
            })
            .collect::<Vec<_>>();
        let input_schema = input_batches[0].schema();
        let test_dir = TempStrDir::default();
        let dataset = Dataset::write(
            RecordBatchIterator::new(input_batches.into_iter().map(Ok), input_schema),
            test_dir.as_ref(),
            None,
        )
        .await
        .unwrap();
        let update_batches = (0..ROWS)
            .step_by(ROWS_PER_BATCH as usize)
            .map(|start| list_batch((start..start + ROWS_PER_BATCH).rev().collect(), true))
            .collect::<Vec<_>>();
        let update_schema = update_batches[0].schema();

        let reported_spills = Arc::new(Mutex::new((0usize, 0usize)));
        let callback_spills = reported_spills.clone();
        let execution_options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(8 * MIB),
            max_temp_directory_size: Some(1024 * MIB),
            batch_size: Some(EXECUTION_BATCH_SIZE),
            target_partition: Some(1),
            execution_stats_callback: Some(Arc::new(move |counts| {
                let mut reported = callback_spills.lock().unwrap();
                reported.0 += counts.all_counts.get("spill_count").copied().unwrap_or(0);
                reported.1 += counts.all_counts.get("spilled_bytes").copied().unwrap_or(0);
            })),
            skip_logging: true,
        };
        let session = new_session_context(&execution_options);
        let mut fragment = dataset.get_fragment(0).unwrap();
        let result = update_columns_with_options(
            &mut fragment,
            reader(
                update_schema.clone(),
                update_batches.iter().cloned().map(Ok).collect(),
            ),
            "key",
            "key",
            UpdateColumnsOptions {
                execution_options,
                max_hash_rows: 0,
                max_hash_bytes: usize::MAX,
                session_context: Some(session.clone()),
            },
        )
        .await
        .unwrap();
        let reported_spills = *reported_spills.lock().unwrap();
        assert!(reported_spills.0 > 0, "no sort spill files were reported");
        assert!(reported_spills.1 > 0, "no spilled bytes were reported");

        let updated_fragment = FileFragment::new(Arc::new(dataset.clone()), result.fragment);
        let batch = updated_fragment.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), MATCHED_ROWS as usize);
        let payload = batch
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = payload.value(0);
        let first = first.as_any().downcast_ref::<Int32Array>().unwrap();
        let last = payload.value(MATCHED_ROWS as usize - 1);
        let last = last.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first.value(0), 0);
        assert_eq!(last.value(0), MATCHED_ROWS - 1);

        let progress = session.runtime_env().disk_manager.spilling_progress();
        assert_eq!(progress.active_files_count, 0);
        assert_eq!(progress.current_bytes, 0);

        let low_disk_options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(8 * MIB),
            max_temp_directory_size: Some(1),
            batch_size: Some(EXECUTION_BATCH_SIZE),
            target_partition: Some(1),
            ..Default::default()
        };
        let low_disk_session = new_session_context(&low_disk_options);
        let mut fragment = dataset.get_fragment(0).unwrap();
        let original_metadata = fragment.metadata().clone();
        let error = update_columns_with_options(
            &mut fragment,
            reader(update_schema, update_batches.into_iter().map(Ok).collect()),
            "key",
            "key",
            UpdateColumnsOptions {
                execution_options: low_disk_options,
                max_hash_rows: 0,
                max_hash_bytes: usize::MAX,
                session_context: Some(low_disk_session.clone()),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("Temporary disk limit"));
        assert_eq!(fragment.metadata(), &original_metadata);
        let progress = low_disk_session
            .runtime_env()
            .disk_manager
            .spilling_progress();
        assert_eq!(progress.active_files_count, 0);
        assert_eq!(progress.current_bytes, 0);
    }
}
