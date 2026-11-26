// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray, UInt32Array,
    cast::AsArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use lance_core::datatypes::{
    BLOB_DESC_LANCE_FIELD, BLOB_V2_DESC_FIELDS, BlobKind, Field as LanceField, Schema,
};
use lance_core::{Error, Result};
use lance_encoding::decoder::{ColumnInfo, FilterExpression, PageEncoding};
use lance_encoding::format::metadata::{PageLayout, page_layout::Layout};
use lance_file::reader::ProjectedFileReader;
use lance_file::version::ConcreteFileVersion;
use lance_file::{LanceEncodingsIo, versions as file_versions};
use lance_io::ReadBatchParams;
use lance_io::scheduler::FileScheduler;
use lance_table::utils::stream::{ReadBatchTask, ReadBatchTaskStream};

use super::GenericFileReader;

/// Request the legacy descriptor layout for the blobs a file stores that way.
/// Adaptation belongs at the dataset boundary so the released file readers
/// retain their original behavior.
pub(super) fn legacy_blob_read_schema(schema: &Schema, legacy_blobs: &HashSet<i32>) -> Schema {
    fn adapt(field: &mut LanceField, legacy_blobs: &HashSet<i32>) {
        if field.is_blob_v2() && legacy_blobs.contains(&field.id) {
            field.metadata.remove(lance_arrow::ARROW_EXT_NAME_KEY);
            field
                .metadata
                .insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
            field.logical_type = BLOB_DESC_LANCE_FIELD.logical_type.clone();
            field.children = BLOB_DESC_LANCE_FIELD.children.clone();
        } else {
            for child in &mut field.children {
                adapt(child, legacy_blobs);
            }
        }
    }
    let mut schema = schema.clone();
    for field in &mut schema.fields {
        adapt(field, legacy_blobs);
    }
    schema
}

/// A legacy payload is an Inline Blob v2 extent in the same data file. Normalize
/// before batches from different file versions are concatenated or materialized.
fn normalize_legacy_blob_batch(
    batch: RecordBatch,
    schema: &Schema,
    legacy_blobs: &HashSet<i32>,
) -> Result<RecordBatch> {
    fn adapt(
        array: &ArrayRef,
        field: &LanceField,
        legacy_blobs: &HashSet<i32>,
    ) -> Result<ArrayRef> {
        if field.is_blob_v2() {
            if !legacy_blobs.contains(&field.id) {
                return Ok(array.clone());
            }
            let descriptors = array.as_struct();
            let positions = descriptors.column(0).as_primitive::<UInt64Type>();
            let sizes = descriptors.column(1).as_primitive::<UInt64Type>();
            let valid = (0..array.len())
                .map(|i| {
                    descriptors.is_valid(i)
                        && positions.is_valid(i)
                        && sizes.is_valid(i)
                        && !(sizes.value(i) == 0 && positions.value(i) != 0)
                })
                .collect::<Vec<_>>();
            return Ok(Arc::new(StructArray::try_new(
                BLOB_V2_DESC_FIELDS.clone(),
                vec![
                    Arc::new(arrow_array::UInt8Array::from(vec![
                        BlobKind::Inline as u8;
                        array.len()
                    ])),
                    Arc::new(arrow_array::UInt64Array::new(
                        positions.values().clone(),
                        None,
                    )),
                    Arc::new(arrow_array::UInt64Array::new(sizes.values().clone(), None)),
                    Arc::new(arrow_array::UInt32Array::from(vec![0; array.len()])),
                    Arc::new(arrow_array::StringArray::from(vec![""; array.len()])),
                ],
                Some(arrow_buffer::NullBuffer::from(valid)),
            )?));
        }
        match field.data_type() {
            DataType::Struct(_) => {
                let values = array.as_struct();
                let columns = values
                    .columns()
                    .iter()
                    .zip(&field.children)
                    .map(|(array, field)| adapt(array, field, legacy_blobs))
                    .collect::<Result<Vec<_>>>()?;
                let fields = field
                    .children
                    .iter()
                    .map(ArrowField::from)
                    .collect::<Vec<_>>();
                Ok(Arc::new(StructArray::try_new(
                    fields.into(),
                    columns,
                    values.nulls().cloned(),
                )?))
            }
            DataType::List(_) => adapt_list::<i32>(array, field, legacy_blobs),
            DataType::LargeList(_) => adapt_list::<i64>(array, field, legacy_blobs),
            _ => Ok(array.clone()),
        }
    }
    fn adapt_list<O: OffsetSizeTrait>(
        array: &ArrayRef,
        field: &LanceField,
        legacy_blobs: &HashSet<i32>,
    ) -> Result<ArrayRef> {
        let list = array.as_list::<O>();
        let child = &field.children[0];
        Ok(Arc::new(GenericListArray::<O>::try_new(
            Arc::new(ArrowField::from(child)),
            list.offsets().clone(),
            adapt(list.values(), child, legacy_blobs)?,
            list.nulls().cloned(),
        )?))
    }
    let columns = batch
        .columns()
        .iter()
        .zip(&schema.fields)
        .map(|(array, field)| adapt(array, field, legacy_blobs))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(
        Arc::new(ArrowSchema::from(schema)),
        columns,
    )?)
}

/// Find the projected blob fields whose column a 2.2 data file stores in the
/// legacy layout.
///
/// The single-fragment writers skip the Blob v2 promotion, so a 2.2 file can keep
/// a legacy blob column after the table has promoted it. Only the legacy encoder
/// writes blob page layouts, and the decoder dispatches on the same signal.
pub(super) fn legacy_blob_fields(
    projection: &Schema,
    field_id_to_column_idx: &BTreeMap<u32, u32>,
    column_infos: &[Arc<ColumnInfo>],
) -> HashSet<i32> {
    projection
        .fields_pre_order()
        .filter(|field| {
            field.is_blob_v2()
                && field_id_to_column_idx
                    .get(&(field.id as u32))
                    .and_then(|column_index| column_infos.get(*column_index as usize))
                    .is_some_and(|column| {
                        column.page_infos.iter().any(|page| {
                            matches!(
                                page.encoding,
                                PageEncoding::Structural(PageLayout {
                                    layout: Some(Layout::BlobLayout(_))
                                })
                            )
                        })
                    })
        })
        .map(|field| field.id)
        .collect()
}

#[derive(Debug, Clone)]
pub(super) struct Reader {
    reader: Arc<ProjectedFileReader>,
    projection: Arc<Schema>,
    field_id_to_column_idx: Arc<BTreeMap<u32, u32>>,
    /// Column index to the field that owns that column's bytes.
    ///
    /// Keyed by column, not field id, so repeated tombstone slots stay
    /// distinct boundaries. A column missing from the map extends the previous
    /// field.
    column_idx_to_field_id: BTreeMap<u32, u32>,
    /// The blob fields this file stores in the legacy layout.
    legacy_blobs: Arc<HashSet<i32>>,
    default_priority: u32,
    file_scheduler: FileScheduler,
}

impl Reader {
    pub(super) fn new(
        reader: Arc<ProjectedFileReader>,
        projection: Arc<Schema>,
        mut field_id_to_column_idx: Arc<BTreeMap<u32, u32>>,
        field_slots: &[(i32, i32)],
        legacy_blobs: Arc<HashSet<i32>>,
        default_priority: u32,
        file_scheduler: FileScheduler,
    ) -> Self {
        if reader.version() == ConcreteFileVersion::V2_0 {
            // A child rewrite can tombstone its ancestor IDs in the manifest,
            // but V2.0 still needs those physical headers to read other children.
            // Recover only structural headers, leaving superseded values excluded.
            let missing_headers = projection
                .fields_pre_order()
                .filter(|field| {
                    field.logical_type.is_struct()
                        && !field.is_packed_struct()
                        && !field.is_blob()
                        && !field_id_to_column_idx.contains_key(&(field.id as u32))
                })
                .map(|field| field.id)
                .collect::<HashSet<_>>();
            if !missing_headers.is_empty() {
                let (field_ids, column_indices) =
                    file_versions::data_file_columns(reader.version(), reader.schema());
                Arc::make_mut(&mut field_id_to_column_idx).extend(
                    field_ids
                        .into_iter()
                        .zip(column_indices)
                        .filter(|(field_id, _)| missing_headers.contains(field_id))
                        .map(|(field_id, column_index)| (field_id as u32, column_index as u32)),
                );
            }
        }
        let column_idx_to_field_id =
            column_byte_owners(field_slots, field_id_to_column_idx.as_ref());
        Self {
            reader,
            projection,
            field_id_to_column_idx,
            column_idx_to_field_id,
            legacy_blobs,
            default_priority,
            file_scheduler,
        }
    }
    async fn read_tasks(
        &self,
        reader: &ProjectedFileReader,
        params: ReadBatchParams,
        batch_size: u32,
        output_schema: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let has_legacy_blobs = output_schema
            .fields_pre_order()
            .any(|field| field.is_blob_v2() && self.legacy_blobs.contains(&field.id));
        let physical_schema = if has_legacy_blobs {
            legacy_blob_read_schema(&output_schema, &self.legacy_blobs)
        } else {
            output_schema.as_ref().clone()
        };
        let projection = file_versions::reader_projection_from_field_ids(
            reader.version(),
            &physical_schema,
            self.field_id_to_column_idx.as_ref(),
        )?;
        let legacy_blobs = self.legacy_blobs.clone();
        Ok(reader
            .read_tasks(
                params,
                batch_size,
                Some(projection),
                FilterExpression::no_filter(),
            )
            .await?
            .map(move |task| {
                let output_schema = output_schema.clone();
                let legacy_blobs = legacy_blobs.clone();
                ReadBatchTask {
                    task: async move {
                        let batch = task.task.await?;
                        if has_legacy_blobs {
                            normalize_legacy_blob_batch(batch, &output_schema, &legacy_blobs)
                        } else {
                            Ok(batch)
                        }
                    }
                    .boxed(),
                    num_rows: task.num_rows,
                }
            })
            .boxed())
    }
}

/// Column boundaries for byte attribution.
///
/// `slots` is `data_file.fields` zipped with `column_indices`. Every
/// non-negative column index is a boundary, including each repeated tombstone
/// (`-2`). Keying the read map by field id would keep only the last tombstone
/// and treat the earlier column as a continuation of the previous field.
///
/// `field_id_to_column_idx` may also name a v2.0 structural header whose slot
/// was tombstoned. That header is still the field's bytes, so it replaces the
/// tombstone on its column. A replaced value column is not put back, and its
/// tombstone stays a boundary.
fn column_byte_owners(
    slots: &[(i32, i32)],
    field_id_to_column_idx: &BTreeMap<u32, u32>,
) -> BTreeMap<u32, u32> {
    let mut owners = BTreeMap::new();
    let mut slotted_field_ids = HashSet::new();
    for &(field_id, column_index) in slots {
        if column_index < 0 {
            continue;
        }
        owners.insert(column_index as u32, field_id as u32);
        slotted_field_ids.insert(field_id as u32);
    }
    for (&field_id, &column_idx) in field_id_to_column_idx {
        if !slotted_field_ids.contains(&field_id) {
            owners.insert(column_idx, field_id);
        }
    }
    owners
}

/// Bytes of a column with no boundary extend the field that owns the previous
/// column. Columns before the first boundary stay on field 0, matching the
/// historical default.
fn attribute_column_bytes(owners: &BTreeMap<u32, u32>, column_sizes: &[u64]) -> Vec<(u32, u64)> {
    let mut stats = Vec::with_capacity(column_sizes.len());
    let mut current_field_id = 0;
    for (column_idx, size_bytes) in column_sizes.iter().copied().enumerate() {
        if let Some(field_id) = owners.get(&(column_idx as u32)) {
            current_field_id = *field_id;
        }
        stats.push((current_field_id, size_bytes));
    }
    stats
}

impl GenericFileReader for Reader {
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        self.read_tasks(
            &self.reader,
            ReadBatchParams::Range(range.start as usize..range.end as usize),
            batch_size,
            projection,
        )
        .boxed()
    }

    fn read_ranges_tasks(
        &self,
        ranges: Arc<[Range<u64>]>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        self.read_tasks(
            &self.reader,
            ReadBatchParams::Ranges(ranges),
            batch_size,
            projection,
        )
        .boxed()
    }

    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        self.read_tasks(
            &self.reader,
            ReadBatchParams::RangeFull,
            batch_size,
            projection,
        )
        .boxed()
    }

    fn take_all_tasks(
        &self,
        indices: &[u32],
        batch_size: u32,
        projection: Arc<Schema>,
        take_priority: Option<u32>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        let indices = UInt32Array::from(indices.to_vec());
        async move {
            let reader = if let Some(take_priority) = take_priority {
                let op_priority = ((take_priority as u64) << 32) | self.default_priority as u64;
                let scheduler = self.file_scheduler.with_priority(op_priority);
                Arc::new(
                    self.reader
                        .with_scheduler(Arc::new(LanceEncodingsIo::new(scheduler))),
                )
            } else {
                self.reader.clone()
            };
            self.read_tasks(
                &reader,
                ReadBatchParams::Indices(indices),
                batch_size,
                projection,
            )
            .await
        }
        .boxed()
    }

    fn storage_stats(&self) -> Result<Vec<(u32, u64)>> {
        let file_statistics = self.reader.file_statistics().ok_or_else(|| {
            Error::internal("storage_stats requires full file metadata".to_string())
        })?;
        let column_sizes = file_statistics
            .columns
            .iter()
            .map(|col| col.size_bytes)
            .collect::<Vec<_>>();
        Ok(attribute_column_bytes(
            &self.column_idx_to_field_id,
            &column_sizes,
        ))
    }

    fn projection(&self) -> &Arc<Schema> {
        &self.projection
    }

    /// Return the number of rows in the file
    fn len(&self) -> u32 {
        self.reader.num_rows() as u32
    }

    fn clone_box(&self) -> Box<dyn GenericFileReader> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOMBSTONE: u32 = -2i32 as u32;

    #[test]
    fn repeated_tombstone_slots_do_not_extend_the_previous_field() {
        // Aligned replacement of two columns leaves field ids [0, -2, -2].
        // A field-id map keeps only the last -2, so sizes [10, 20, 30] would
        // bill field 0 for 30 bytes. Each tombstone is its own column.
        let slots = [(0, 0), (-2, 1), (-2, 2)];
        let owners = column_byte_owners(&slots, &BTreeMap::from([(0, 0), (TOMBSTONE, 2)]));
        assert_eq!(
            attribute_column_bytes(&owners, &[10, 20, 30]),
            vec![(0, 10), (TOMBSTONE, 20), (TOMBSTONE, 30)]
        );
    }

    #[test]
    fn column_without_a_slot_extends_the_previous_field() {
        // A list can occupy two columns while only the first has a field slot.
        // Children stored with column index -1 are not boundaries.
        let slots = [(0, 0), (1, -1), (2, -1), (3, 1), (4, 3)];
        let owners = column_byte_owners(&slots, &BTreeMap::from([(0, 0), (3, 1), (4, 3)]));
        assert_eq!(
            attribute_column_bytes(&owners, &[10, 20, 30, 40]),
            vec![(0, 10), (3, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    fn recovered_structural_header_keeps_its_column() {
        // V2.0 puts a tombstoned struct header back on its column. The other
        // tombstone is a replaced value column and must not join field 0.
        let slots = [(0, 0), (-2, 1), (-2, 2)];
        let owners = column_byte_owners(&slots, &BTreeMap::from([(0, 0), (7, 1), (TOMBSTONE, 2)]));
        assert_eq!(
            attribute_column_bytes(&owners, &[10, 20, 30]),
            vec![(0, 10), (7, 20), (TOMBSTONE, 30)]
        );
    }
}
