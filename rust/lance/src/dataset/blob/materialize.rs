// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray,
    builder::LargeBinaryBuilder,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef,
};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_arrow::ARROW_EXT_NAME_KEY;
use tokio::sync::Notify;

use super::locate::{BlobV2DescriptorColumns, BlobV2ReadContext, blob_version_from_descriptions};
use super::read::BlobEntry;
use super::read::{ReadBlobsExecution, execute_blob_entries_with_execution};
use crate::dataset::Dataset;
use lance_core::datatypes::{BlobKind, BlobVersion, Field as LanceField, Schema};
use lance_core::{Error, ROW_ADDR, Result};

#[derive(Debug)]
pub(super) struct BlobMaterializationBudget {
    limit: u64,
    pub(super) state: std::sync::Mutex<BlobMaterializationBudgetState>,
    notify: Notify,
}

#[derive(Debug, Default)]
pub(super) struct BlobMaterializationBudgetState {
    pub(super) reserved: u64,
    next_ticket: u64,
    serving_ticket: u64,
    cancelled_tickets: HashSet<u64>,
    #[cfg(test)]
    peak_reserved: u64,
}

impl BlobMaterializationBudgetState {
    fn skip_cancelled(&mut self) {
        while self.cancelled_tickets.remove(&self.serving_ticket) {
            self.serving_ticket = self.serving_ticket.wrapping_add(1);
        }
    }
}

impl BlobMaterializationBudget {
    fn admission(self: &Arc<Self>) -> BlobMaterializationAdmission {
        let ticket = {
            let mut state = self.state.lock().unwrap();
            let ticket = state.next_ticket;
            state.next_ticket = state.next_ticket.wrapping_add(1);
            ticket
        };
        BlobMaterializationAdmission {
            budget: Some(self.clone()),
            ticket,
            acquired: false,
        }
    }

    #[cfg(test)]
    async fn reserve(self: &Arc<Self>, bytes: u64) -> BlobMaterializationReservation {
        self.admission().reserve(bytes).await.unwrap()
    }
}

pub struct BlobMaterializationAdmission {
    budget: Option<Arc<BlobMaterializationBudget>>,
    ticket: u64,
    acquired: bool,
}

impl BlobMaterializationAdmission {
    async fn reserve(mut self, bytes: u64) -> Option<BlobMaterializationReservation> {
        let Some(budget) = self.budget.clone() else {
            self.acquired = true;
            return None;
        };
        loop {
            let notified = budget.notify.notified();
            {
                let mut state = budget.state.lock().unwrap();
                let fits = bytes <= budget.limit.saturating_sub(state.reserved);
                let oversized_and_idle = state.reserved == 0 && bytes > budget.limit;
                if self.ticket == state.serving_ticket && (fits || oversized_and_idle) {
                    state.reserved = state.reserved.saturating_add(bytes);
                    #[cfg(test)]
                    {
                        state.peak_reserved = state.peak_reserved.max(state.reserved);
                    }
                    state.serving_ticket = state.serving_ticket.wrapping_add(1);
                    state.skip_cancelled();
                    self.acquired = true;
                    let reservation = BlobMaterializationReservation {
                        budget: budget.clone(),
                        bytes,
                    };
                    drop(state);
                    budget.notify.notify_waiters();
                    return Some(reservation);
                }
            }
            notified.await;
        }
    }
}

impl Drop for BlobMaterializationAdmission {
    fn drop(&mut self) {
        let Some(budget) = &self.budget else {
            return;
        };
        if self.acquired {
            return;
        }
        let mut state = budget.state.lock().unwrap();
        if self.ticket >= state.serving_ticket {
            state.cancelled_tickets.insert(self.ticket);
            state.skip_cancelled();
        }
        drop(state);
        budget.notify.notify_waiters();
    }
}

#[derive(Debug)]
struct BlobMaterializationReservation {
    budget: Arc<BlobMaterializationBudget>,
    bytes: u64,
}

impl Drop for BlobMaterializationReservation {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap();
        state.reserved = state.reserved.saturating_sub(self.bytes);
        drop(state);
        self.budget.notify.notify_waiters();
    }
}

/// Shared state for asynchronously materializing blob v2 descriptor batches.
#[derive(Debug)]
pub struct BlobMaterializationContext {
    execution: Arc<ReadBlobsExecution>,
    pub(super) budget: Option<Arc<BlobMaterializationBudget>>,
}

impl BlobMaterializationContext {
    pub(crate) fn new(
        io_buffer_size_bytes: Option<u64>,
        materialization_readahead_bytes: Option<u64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            execution: Arc::new(ReadBlobsExecution::new(io_buffer_size_bytes)),
            budget: materialization_readahead_bytes.map(|limit| {
                Arc::new(BlobMaterializationBudget {
                    limit,
                    state: std::sync::Mutex::new(BlobMaterializationBudgetState::default()),
                    notify: Notify::new(),
                })
            }),
        })
    }

    pub(crate) fn admission(&self) -> BlobMaterializationAdmission {
        match &self.budget {
            Some(budget) => budget.admission(),
            None => BlobMaterializationAdmission {
                budget: None,
                ticket: 0,
                acquired: false,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn peak_reserved_bytes(&self) -> u64 {
        self.budget
            .as_ref()
            .map(|budget| budget.state.lock().unwrap().peak_reserved)
            .unwrap_or(0)
    }
}

/// A materialized batch that retains its byte-budget reservation until yielded.
pub struct MaterializedBlobBatch {
    batch: RecordBatch,
    /// Reservations are reference-counted so that splitting a batch into
    /// multiple chunks keeps the original memory reservation alive until the
    /// last chunk is dropped.
    _reservations: Vec<Arc<BlobMaterializationReservation>>,
}

impl MaterializedBlobBatch {
    pub(crate) fn unreserved(batch: RecordBatch) -> Self {
        Self {
            batch,
            _reservations: Vec::new(),
        }
    }

    #[cfg(test)]
    fn with_reservations(
        batch: RecordBatch,
        reservations: Vec<Arc<BlobMaterializationReservation>>,
    ) -> Self {
        Self {
            batch,
            _reservations: reservations,
        }
    }

    pub(crate) fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub(crate) fn with_batch(self, batch: RecordBatch) -> Self {
        Self {
            batch,
            _reservations: self._reservations,
        }
    }

    pub(crate) fn concat(schema: &SchemaRef, batches: Vec<Self>) -> Result<Self> {
        let mut record_batches = Vec::with_capacity(batches.len());
        let mut reservations = Vec::new();
        for batch in batches {
            record_batches.push(batch.batch);
            reservations.extend(batch._reservations);
        }
        Ok(Self {
            batch: arrow::compute::concat_batches(schema, record_batches.iter())?,
            _reservations: reservations,
        })
    }

    /// Split the batch into chunks of at most `max_bytes` array memory.
    ///
    /// The original materialization reservation is shared across all chunks so
    /// that the memory remains accounted for until every chunk has been
    /// dropped, preserving the `materialization_readahead_bytes` bound while
    /// the split chunks are queued.
    pub(crate) fn split_by_bytes(self, max_bytes: usize) -> Result<Vec<Self>> {
        let Self {
            batch,
            _reservations,
        } = self;
        let chunks = split_batch_by_bytes(batch, max_bytes)?;
        Ok(chunks
            .into_iter()
            .map(|batch| Self {
                batch,
                _reservations: _reservations.clone(),
            })
            .collect())
    }

    pub(crate) fn into_batch(self) -> RecordBatch {
        self.batch
    }
}

fn is_blob_v2_binary_view(field: &LanceField) -> bool {
    field.is_blob_v2() && matches!(field.data_type(), ArrowDataType::LargeBinary)
}

fn public_blob_v2_binary_output_field(mut field: LanceField) -> LanceField {
    if is_blob_v2_binary_view(&field) {
        field.metadata.remove(ARROW_EXT_NAME_KEY);
    }
    field.children = field
        .children
        .into_iter()
        .map(public_blob_v2_binary_output_field)
        .collect();
    field
}

/// Return the public Arrow-facing schema for a blob v2 binary scan.
///
/// Scan planning uses a blob v2 extension marker on `LargeBinary` leaves to
/// identify payloads that need descriptor-based materialization. This helper
/// removes that internal marker before the schema is exposed to callers.
pub fn public_blob_v2_binary_output_schema(schema: &Schema) -> Schema {
    Schema {
        fields: schema
            .fields
            .iter()
            .cloned()
            .map(public_blob_v2_binary_output_field)
            .collect(),
        metadata: schema.metadata.clone(),
    }
}

fn field_has_blob_v2_binary_view(field: &LanceField) -> bool {
    is_blob_v2_binary_view(field) || field.children.iter().any(field_has_blob_v2_binary_view)
}

/// Return true if the schema contains a blob v2 leaf in binary payload view.
///
/// This detects the internal `LargeBinary` view created by
/// [`BlobHandling::AllBinary`](lance_core::datatypes::BlobHandling::AllBinary)
/// or selective binary blob handling.
pub fn schema_has_blob_v2_binary_view(schema: &Schema) -> bool {
    schema.fields.iter().any(field_has_blob_v2_binary_view)
}

fn blob_v2_descriptor_field(mut field: LanceField) -> LanceField {
    if is_blob_v2_binary_view(&field) {
        field.unloaded_mut();
        return field;
    }

    field.children = field
        .children
        .into_iter()
        .map(blob_v2_descriptor_field)
        .collect();
    field
}

/// Convert blob v2 binary-view leaves back to descriptor-view leaves.
///
/// Readers use this schema to fetch stored blob descriptors first. The scan
/// layer then materializes those descriptors into the caller's binary payload
/// view after row addresses are available.
pub fn blob_v2_descriptor_schema(schema: &Schema) -> Schema {
    Schema {
        fields: schema
            .fields
            .iter()
            .cloned()
            .map(blob_v2_descriptor_field)
            .collect(),
        metadata: schema.metadata.clone(),
    }
}

/// Materialize blob v2 descriptor arrays in a decoded batch into binary arrays.
///
/// The input batch must include `_rowaddr`, which is used to resolve packed,
/// dedicated, inline, and external blob payload locations. `output_schema`
/// defines the exact returned columns, including requested system columns, with
/// blob v2 binary leaves exposed as plain `LargeBinary` fields.
pub async fn materialize_blob_v2_binary_batch(
    dataset: &Arc<Dataset>,
    output_schema: &Schema,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    let context = BlobMaterializationContext::new(None, None);
    Ok(
        materialize_blob_v2_binary_batch_with_context(dataset, output_schema, batch, &context)
            .await?
            .into_batch(),
    )
}

/// Split `batch` into row-aligned chunks whose total array memory size is at
/// most `max_bytes`.
///
/// The split estimates chunk boundaries from average bytes per row and then
/// recursively re-checks each deep-copied slice. Because `RecordBatch::slice`
/// shares backing buffers, each slice is deep-copied so that
/// `get_array_memory_size` reflects the true chunk size. Splitting stops when
/// a chunk is within budget or contains a single indivisible row.
pub fn split_batch_by_bytes(batch: RecordBatch, max_bytes: usize) -> Result<Vec<RecordBatch>> {
    split_batch_by_bytes_recursive(batch, max_bytes)
}

fn split_batch_by_bytes_recursive(
    batch: RecordBatch,
    max_bytes: usize,
) -> Result<Vec<RecordBatch>> {
    if batch.num_rows() == 0 {
        return Ok(Vec::new());
    }
    let total_bytes = batch.get_array_memory_size();
    let num_rows = batch.num_rows();
    if total_bytes <= max_bytes || num_rows == 1 {
        return Ok(vec![batch]);
    }
    let bytes_per_row = total_bytes / num_rows;
    let rows_per_chunk = (max_bytes / bytes_per_row.max(1)).max(1);
    // Make sure we make progress: never take all rows in one slice.
    let rows_per_chunk = rows_per_chunk.min(num_rows - 1).max(1);

    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < num_rows {
        let rows = (offset + rows_per_chunk).min(num_rows) - offset;
        let sliced = batch.slice(offset, rows);
        // Deep-copy so that `get_array_memory_size()` reports the slice's own
        // size rather than the parent buffer's size, allowing recursion to
        // catch skewed rows that still exceed the budget.
        let copied = lance_arrow::deepcopy::deep_copy_batch_sliced(&sliced)?;
        chunks.extend(split_batch_by_bytes_recursive(copied, max_bytes)?);
        offset += rows;
    }
    Ok(chunks)
}

pub fn materialize_blob_v2_binary_batch_with_context<'a>(
    dataset: &'a Arc<Dataset>,
    output_schema: &'a Schema,
    batch: RecordBatch,
    context: &'a Arc<BlobMaterializationContext>,
) -> BoxFuture<'a, Result<MaterializedBlobBatch>> {
    let admission = context.admission();
    materialize_blob_v2_binary_batch_with_admission(
        dataset,
        output_schema,
        batch,
        context,
        admission,
    )
}

pub fn materialize_blob_v2_binary_batch_with_admission<'a>(
    dataset: &'a Arc<Dataset>,
    output_schema: &'a Schema,
    batch: RecordBatch,
    context: &'a Arc<BlobMaterializationContext>,
    admission: BlobMaterializationAdmission,
) -> BoxFuture<'a, Result<MaterializedBlobBatch>> {
    async move {
        let materialized_bytes =
            estimate_blob_v2_materialized_batch_bytes(dataset, output_schema, &batch).await?;
        let reservation = admission.reserve(materialized_bytes).await;
        let row_addr_idx = batch
            .schema()
            .column_with_name(ROW_ADDR)
            .ok_or_else(|| {
                Error::internal(format!(
                    "_rowaddr column missing from blob v2 binary scan batch, columns: {:?}",
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| field.name())
                        .collect::<Vec<_>>()
                ))
            })?
            .0;
        let row_addrs = batch
            .column(row_addr_idx)
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let row_addrs: Arc<[u64]> = row_addrs.into();

        let mut columns = Vec::with_capacity(output_schema.fields.len());
        let mut fields = Vec::with_capacity(output_schema.fields.len());

        for field in &output_schema.fields {
            let input = batch
                .column_by_name(&field.name)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "blob v2 binary scan batch missing projected column '{}'",
                        field.name
                    ))
                })?
                .clone();
            let materialized =
                materialize_blob_v2_binary_array(dataset, field, input, row_addrs.clone(), context)
                    .await?;
            columns.push(materialized);
            let output_field = public_blob_v2_binary_output_field(field.clone());
            fields.push(ArrowField::from(&output_field));
        }

        Ok(MaterializedBlobBatch {
            batch: RecordBatch::try_new(
                Arc::new(ArrowSchema::new_with_metadata(
                    fields,
                    batch.schema().metadata().clone(),
                )),
                columns,
            )?,
            _reservations: reservation.into_iter().map(Arc::new).collect(),
        })
    }
    .boxed()
}

async fn estimate_blob_v2_materialized_batch_bytes(
    dataset: &Arc<Dataset>,
    output_schema: &Schema,
    batch: &RecordBatch,
) -> Result<u64> {
    let mut bytes = u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX);
    let row_addr_idx = batch
        .schema()
        .column_with_name(ROW_ADDR)
        .ok_or_else(|| Error::internal("_rowaddr missing while estimating blob materialization"))?
        .0;
    let row_addrs = batch
        .column(row_addr_idx)
        .as_primitive::<UInt64Type>()
        .values();
    for field in &output_schema.fields {
        let input = batch.column_by_name(&field.name).ok_or_else(|| {
            Error::internal(format!(
                "blob v2 binary scan batch missing projected column '{}'",
                field.name
            ))
        })?;
        bytes = bytes.saturating_add(
            estimate_blob_v2_materialized_array_bytes(dataset, field, input, row_addrs.as_ref())
                .await?,
        );
    }
    Ok(bytes)
}

fn estimate_blob_v2_materialized_array_bytes<'a>(
    dataset: &'a Arc<Dataset>,
    field: &'a LanceField,
    array: &'a ArrayRef,
    row_addrs: &'a [u64],
) -> BoxFuture<'a, Result<u64>> {
    async move {
        if is_blob_v2_binary_view(field) {
            let descriptions = array.as_struct();
            match blob_version_from_descriptions(descriptions)? {
                BlobVersion::V1 => {
                    return Err(Error::not_supported(
                        "Blob v2 binary materialization received a legacy blob descriptor"
                            .to_string(),
                    ));
                }
                BlobVersion::V2 => {}
            }
            if descriptions.len() != row_addrs.len() {
                return Err(Error::internal(format!(
                    "blob v2 descriptor count {} did not match row address count {}",
                    descriptions.len(),
                    row_addrs.len()
                )));
            }
            let columns = BlobV2DescriptorColumns::new(descriptions);
            let mut read_context = BlobV2ReadContext::new(dataset, field.id as u32);
            let mut payload_bytes = 0_u64;
            for (idx, row_addr) in row_addrs.iter().copied().enumerate() {
                if columns.is_null_blob(idx) {
                    continue;
                }
                let kind = BlobKind::try_from(columns.kinds.value(idx))?;
                if matches!(kind, BlobKind::Inline)
                    && columns.positions.value(idx) == 0
                    && columns.sizes.value(idx) == 0
                {
                    continue;
                }
                let file = read_context
                    .collect_file(&columns, idx, row_addr)
                    .await?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "blob v2 descriptor at index {idx} unexpectedly resolved to null"
                        ))
                    })?;
                payload_bytes = payload_bytes.saturating_add(file.size);
            }
            let offsets_bytes =
                u64::try_from((descriptions.len() + 1).saturating_mul(std::mem::size_of::<i64>()))
                    .unwrap_or(u64::MAX);
            return Ok(payload_bytes.saturating_add(offsets_bytes));
        }

        match field.data_type() {
            ArrowDataType::Struct(_) => {
                let array = array.as_struct();
                let mut total = 0_u64;
                for (child, array) in field.children.iter().zip(array.columns()) {
                    total = total.saturating_add(
                        estimate_blob_v2_materialized_array_bytes(dataset, child, array, row_addrs)
                            .await?,
                    );
                }
                Ok(total)
            }
            ArrowDataType::List(_) => {
                let array = array.as_list::<i32>();
                let child = field.children.first().ok_or_else(|| {
                    Error::internal(format!(
                        "List field '{}' missing child while estimating blob v2 materialization",
                        field.name
                    ))
                })?;
                let (values_start, child_row_addrs) =
                    list_child_row_addrs(array.value_offsets(), row_addrs)?;
                let values = array.values().slice(values_start, child_row_addrs.len());
                estimate_blob_v2_materialized_array_bytes(dataset, child, &values, &child_row_addrs)
                    .await
            }
            ArrowDataType::LargeList(_) => {
                let array = array.as_list::<i64>();
                let child = field.children.first().ok_or_else(|| {
                    Error::internal(format!(
                        "List field '{}' missing child while estimating blob v2 materialization",
                        field.name
                    ))
                })?;
                let (values_start, child_row_addrs) =
                    list_child_row_addrs(array.value_offsets(), row_addrs)?;
                let values = array.values().slice(values_start, child_row_addrs.len());
                estimate_blob_v2_materialized_array_bytes(dataset, child, &values, &child_row_addrs)
                    .await
            }
            _ => Ok(0),
        }
    }
    .boxed()
}

fn list_child_row_addrs<O: OffsetSizeTrait>(
    offsets: &[O],
    row_addrs: &[u64],
) -> Result<(usize, Vec<u64>)> {
    if offsets.len() != row_addrs.len() + 1 {
        return Err(Error::internal(
            "list offsets did not match row addresses while estimating blob materialization"
                .to_string(),
        ));
    }
    let values_start = offsets[0].as_usize();
    let values_end = offsets[row_addrs.len()].as_usize();
    let mut child_row_addrs = Vec::with_capacity(values_end.saturating_sub(values_start));
    for (row_idx, row_addr) in row_addrs.iter().copied().enumerate() {
        let start = offsets[row_idx].as_usize();
        let end = offsets[row_idx + 1].as_usize();
        if end < start {
            return Err(Error::internal(
                "list offsets decreased while estimating blob materialization".to_string(),
            ));
        }
        child_row_addrs.extend(std::iter::repeat_n(row_addr, end - start));
    }
    Ok((values_start, child_row_addrs))
}

fn materialize_blob_v2_binary_array<'a>(
    dataset: &'a Arc<Dataset>,
    field: &'a LanceField,
    array: ArrayRef,
    row_addrs: Arc<[u64]>,
    context: &'a Arc<BlobMaterializationContext>,
) -> BoxFuture<'a, Result<ArrayRef>> {
    async move {
        if is_blob_v2_binary_view(field) {
            let descriptions = array.as_struct();
            return materialize_blob_v2_descriptors(
                dataset,
                field.id as u32,
                descriptions,
                row_addrs.as_ref(),
                context,
            )
            .await;
        }

        match field.data_type() {
            ArrowDataType::Struct(_) => {
                let struct_array = array.as_struct();
                let mut children = Vec::with_capacity(field.children.len());
                for (child_field, child_array) in
                    field.children.iter().zip(struct_array.columns().iter())
                {
                    children.push(
                        materialize_blob_v2_binary_array(
                            dataset,
                            child_field,
                            child_array.clone(),
                            row_addrs.clone(),
                            context,
                        )
                        .await?,
                    );
                }
                let public_field = public_blob_v2_binary_output_field(field.clone());
                let ArrowDataType::Struct(fields) = public_field.data_type() else {
                    unreachable!("public output field preserved struct type")
                };
                Ok(Arc::new(StructArray::try_new(
                    fields,
                    children,
                    struct_array.nulls().cloned(),
                )?) as ArrayRef)
            }
            ArrowDataType::List(_) => {
                let list_array = array.as_list::<i32>();
                materialize_blob_v2_list_array::<i32>(
                    dataset, field, list_array, row_addrs, context,
                )
                .await
            }
            ArrowDataType::LargeList(_) => {
                let list_array = array.as_list::<i64>();
                materialize_blob_v2_list_array::<i64>(
                    dataset, field, list_array, row_addrs, context,
                )
                .await
            }
            _ => Ok(array),
        }
    }
    .boxed()
}

async fn materialize_blob_v2_list_array<O: OffsetSizeTrait>(
    dataset: &Arc<Dataset>,
    field: &LanceField,
    list_array: &GenericListArray<O>,
    row_addrs: Arc<[u64]>,
    context: &Arc<BlobMaterializationContext>,
) -> Result<ArrayRef> {
    let offsets = list_array.value_offsets();
    let values_start = offsets[0].as_usize();
    let values_end = offsets[list_array.len()].as_usize();
    if values_end < values_start {
        return Err(Error::internal(format!(
            "List field '{}' has invalid offsets while materializing blob v2 binary scan",
            field.name
        )));
    }

    let values_len = values_end - values_start;
    let mut normalized_offsets = Vec::with_capacity(list_array.len() + 1);
    normalized_offsets.push(O::usize_as(0));
    let mut child_row_addrs = Vec::with_capacity(values_len);
    for row_idx in 0..list_array.len() {
        let start = offsets[row_idx].as_usize();
        let end = offsets[row_idx + 1].as_usize();
        if end < start {
            return Err(Error::internal(format!(
                "List field '{}' has decreasing offsets while materializing blob v2 binary scan",
                field.name
            )));
        }
        let row_addr = row_addrs.get(row_idx).copied().ok_or_else(|| {
            Error::internal(format!(
                "List field '{}' row address count {} did not match row count {}",
                field.name,
                row_addrs.len(),
                list_array.len()
            ))
        })?;
        for _ in start..end {
            child_row_addrs.push(row_addr);
        }
        normalized_offsets.push(O::usize_as(end - values_start));
    }
    let child_row_addrs: Arc<[u64]> = child_row_addrs.into();
    let child = field.children.first().ok_or_else(|| {
        Error::internal(format!(
            "List field '{}' missing child while materializing blob v2 binary scan",
            field.name
        ))
    })?;
    let values = list_array.values().slice(values_start, values_len);
    let values =
        materialize_blob_v2_binary_array(dataset, child, values, child_row_addrs, context).await?;
    let child_field = public_blob_v2_binary_output_field(child.clone());
    let list_array = GenericListArray::<O>::try_new(
        Arc::new(ArrowField::from(&child_field)),
        OffsetBuffer::new(ScalarBuffer::from(normalized_offsets)),
        values,
        list_array.nulls().cloned(),
    )?;
    Ok(Arc::new(list_array))
}

async fn materialize_blob_v2_descriptors(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &[u64],
    context: &Arc<BlobMaterializationContext>,
) -> Result<ArrayRef> {
    if descriptions.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "blob v2 descriptor count {} did not match row address count {}",
            descriptions.len(),
            row_addrs.len()
        )));
    }
    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => {
            return Err(Error::not_supported(
                "Blob v2 binary materialization received a legacy blob descriptor".to_string(),
            ));
        }
        BlobVersion::V2 => {}
    }

    let columns = BlobV2DescriptorColumns::new(descriptions);
    let mut read_context = BlobV2ReadContext::new(dataset, blob_field_id);
    let mut entries = Vec::with_capacity(descriptions.len());
    let mut payloads = vec![None; descriptions.len()];

    for (idx, row_addr) in row_addrs.iter().copied().enumerate() {
        if descriptions.is_null(idx) || columns.kinds.is_null(idx) {
            continue;
        }

        let kind = BlobKind::try_from(columns.kinds.value(idx))?;
        if matches!(kind, BlobKind::Inline)
            && columns.positions.value(idx) == 0
            && columns.sizes.value(idx) == 0
        {
            payloads[idx] = Some(Bytes::new());
            continue;
        }

        let file = read_context
            .collect_file(&columns, idx, row_addr)
            .await?
            .ok_or_else(|| {
                Error::internal(format!(
                    "blob v2 descriptor at index {idx} unexpectedly resolved to null"
                ))
            })?;
        entries.push(BlobEntry {
            selection_index: idx,
            row_address: row_addr,
            file,
            requested_range: None,
        });
    }

    let blobs = execute_blob_entries_with_execution(
        entries,
        dataset.object_store.io_parallelism(),
        context.execution.clone(),
    )
    .await?;
    for blob in blobs {
        let payload = payloads.get_mut(blob.selection_index).ok_or_else(|| {
            Error::internal(format!(
                "blob result selection index {} exceeded descriptor count {}",
                blob.selection_index,
                descriptions.len()
            ))
        })?;
        if payload.replace(blob.data).is_some() {
            return Err(Error::internal(format!(
                "blob result selection index {} was produced more than once",
                blob.selection_index
            )));
        }
    }

    let payload_capacity = payloads.iter().flatten().map(Bytes::len).sum::<usize>();
    let mut builder = LargeBinaryBuilder::with_capacity(descriptions.len(), payload_capacity);
    for (idx, payload) in payloads.into_iter().enumerate() {
        if descriptions.is_null(idx) || columns.kinds.is_null(idx) {
            builder.append_null();
        } else {
            let payload = payload.ok_or_else(|| {
                Error::internal(format!(
                    "blob v2 descriptor at index {idx} did not produce a payload"
                ))
            })?;
            builder.append_value(payload);
        }
    }

    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, LargeBinaryArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn test_blob_materialization_budget_blocks_and_admits_oversized_batch() {
        let budget = Arc::new(BlobMaterializationBudget {
            limit: 10,
            state: std::sync::Mutex::new(BlobMaterializationBudgetState::default()),
            notify: Notify::new(),
        });
        let first = budget.reserve(8).await;
        let waiting_budget = budget.clone();
        let waiting = tokio::spawn(async move { waiting_budget.reserve(4).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), waiting_budget_wait(&waiting))
                .await
                .is_err()
        );
        drop(first);
        let second = waiting.await.unwrap();
        drop(second);

        let oversized = budget.reserve(11).await;
        assert_eq!(budget.state.lock().unwrap().reserved, 11);
        drop(oversized);
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
    }

    #[tokio::test]
    async fn test_blob_materialization_admission_cannot_invert_output_order() {
        let budget = Arc::new(BlobMaterializationBudget {
            limit: 100,
            state: std::sync::Mutex::new(BlobMaterializationBudgetState::default()),
            notify: Notify::new(),
        });
        let first = budget.admission();
        let second = budget.admission();
        let later = tokio::spawn(async move { second.reserve(60).await.unwrap() });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), waiting_budget_wait(&later))
                .await
                .is_err()
        );
        assert_eq!(budget.state.lock().unwrap().reserved, 0);

        let earlier = first.reserve(80).await.unwrap();
        assert_eq!(budget.state.lock().unwrap().reserved, 80);
        drop(earlier);
        let later = later.await.unwrap();
        assert_eq!(budget.state.lock().unwrap().reserved, 60);
        drop(later);
        assert_eq!(budget.state.lock().unwrap().reserved, 0);
    }

    async fn waiting_budget_wait(task: &tokio::task::JoinHandle<BlobMaterializationReservation>) {
        while !task.is_finished() {
            tokio::task::yield_now().await;
        }
    }

    #[test]
    fn test_split_batch_by_bytes_skewed_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::LargeBinary,
            false,
        )]));
        let small = vec![0u8; 1];
        let large = vec![0u8; 10_000];
        let values: Vec<Option<&[u8]>> = (0..100)
            .map(|_| Some(small.as_slice()))
            .chain(std::iter::once(Some(large.as_slice())))
            .collect();
        let array = Arc::new(LargeBinaryArray::from_iter(values)) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();

        let max_bytes = 200;
        let chunks = split_batch_by_bytes(batch.clone(), max_bytes).unwrap();
        let total_rows: usize = chunks.iter().map(|c| c.num_rows()).sum();
        assert_eq!(total_rows, batch.num_rows());

        for chunk in &chunks {
            if chunk.num_rows() > 1 {
                assert!(
                    chunk.get_array_memory_size() <= max_bytes,
                    "chunk with {} rows has {} bytes, max is {}",
                    chunk.num_rows(),
                    chunk.get_array_memory_size(),
                    max_bytes
                );
            }
        }
    }

    // Regression: multiple medium rows that individually fit the budget can
    // still be grouped into a chunk above it by the average-based split; the
    // recursive re-check must split them apart (bot reproducer: 10 rows × 900 B
    // plus 90 rows × 1 B, budget 1024 B).
    #[test]
    fn test_split_batch_by_bytes_grouped_oversized_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::LargeBinary,
            false,
        )]));
        let small = vec![0u8; 1];
        let medium = vec![0u8; 900];
        let values: Vec<Option<&[u8]>> = (0..10)
            .map(|_| Some(medium.as_slice()))
            .chain((0..90).map(|_| Some(small.as_slice())))
            .collect();
        let array = Arc::new(LargeBinaryArray::from_iter(values)) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();

        let max_bytes = 1024;
        let chunks = split_batch_by_bytes(batch.clone(), max_bytes).unwrap();
        let total_rows: usize = chunks.iter().map(|c| c.num_rows()).sum();
        assert_eq!(total_rows, batch.num_rows());

        for chunk in &chunks {
            if chunk.num_rows() > 1 {
                assert!(
                    chunk.get_array_memory_size() <= max_bytes,
                    "chunk with {} rows has {} bytes, max is {}",
                    chunk.num_rows(),
                    chunk.get_array_memory_size(),
                    max_bytes
                );
            }
        }
    }

    #[tokio::test]
    async fn test_split_materialized_batch_retains_reservation_until_chunks_drop() {
        let budget = Arc::new(BlobMaterializationBudget {
            limit: 1024,
            state: std::sync::Mutex::new(BlobMaterializationBudgetState::default()),
            notify: Notify::new(),
        });
        let reservation = budget.reserve(800).await;
        assert_eq!(budget.state.lock().unwrap().reserved, 800);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::LargeBinary,
            false,
        )]));
        let value = [0u8; 100];
        let values: Vec<Option<&[u8]>> = (0..10).map(|_| Some(value.as_slice())).collect();
        let array = Arc::new(LargeBinaryArray::from_iter(values)) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();

        let materialized =
            MaterializedBlobBatch::with_reservations(batch, vec![Arc::new(reservation)]);
        let chunks = materialized.split_by_bytes(200).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(
            budget.state.lock().unwrap().reserved,
            800,
            "reservation should be held while chunks are alive"
        );

        drop(chunks);
        assert_eq!(
            budget.state.lock().unwrap().reserved,
            0,
            "reservation should be released after chunks drop"
        );
    }
}
