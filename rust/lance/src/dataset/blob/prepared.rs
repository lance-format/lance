// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, GenericListArray, RecordBatch, StringArray, StructArray, UInt8Array,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use super::external::arrow_field_contains_blob_v2;
use crate::blob::{blob_v2_layout, validate_prepared_blob_array};
use lance_arrow::list::ListArrayExt;
use lance_arrow::r#struct::StructArrayExt;
use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BlobKind, BlobV2Layout};
use lance_core::{Error, Result};

fn prepared_blob_to_descriptor(
    array: &ArrayRef,
    field: &ArrowField,
) -> Result<(ArrayRef, Arc<ArrowField>)> {
    validate_prepared_blob_array(field, array)?;
    let values = array.as_struct();
    let kinds = values
        .column_by_name("kind")
        .expect("validated prepared Blob has kind")
        .as_primitive::<UInt8Type>();
    let data = values
        .column_by_name("data")
        .expect("validated prepared Blob has data")
        .as_binary::<i64>();
    let uris = values
        .column_by_name("uri")
        .expect("validated prepared Blob has uri")
        .as_string::<i32>();
    let blob_ids = values
        .column_by_name("blob_id")
        .expect("validated prepared Blob has blob_id")
        .as_primitive::<UInt32Type>();
    let sizes = values
        .column_by_name("blob_size")
        .expect("validated prepared Blob has blob_size")
        .as_primitive::<UInt64Type>();
    let positions = values
        .column_by_name("position")
        .expect("validated prepared Blob has position")
        .as_primitive::<UInt64Type>();

    let mut output_kinds = Vec::with_capacity(values.len());
    let mut output_positions = Vec::with_capacity(values.len());
    let mut output_sizes = Vec::with_capacity(values.len());
    let mut output_ids = Vec::with_capacity(values.len());
    let mut output_uris = Vec::with_capacity(values.len());

    for row in 0..values.len() {
        if values.is_null(row) {
            output_kinds.push(BlobKind::Inline as u8);
            output_positions.push(0);
            output_sizes.push(0);
            output_ids.push(0);
            output_uris.push(String::new());
            continue;
        }
        match BlobKind::try_from(kinds.value(row))? {
            BlobKind::Inline => {
                let inline = data.value(row);
                output_kinds.push(BlobKind::Inline as u8);
                output_positions.push(0);
                output_sizes.push(inline.len() as u64);
                output_ids.push(0);
                output_uris.push(String::new());
            }
            BlobKind::Packed => {
                output_kinds.push(BlobKind::Packed as u8);
                output_positions.push(positions.value(row));
                output_sizes.push(sizes.value(row));
                output_ids.push(blob_ids.value(row));
                output_uris.push(String::new());
            }
            BlobKind::Dedicated => {
                output_kinds.push(BlobKind::Dedicated as u8);
                output_positions.push(0);
                output_sizes.push(sizes.value(row));
                output_ids.push(blob_ids.value(row));
                output_uris.push(String::new());
            }
            BlobKind::Managed | BlobKind::ManagedWithBase => {
                output_kinds.push(kinds.value(row));
                output_positions.push(positions.value(row));
                output_sizes.push(sizes.value(row));
                output_ids.push(if kinds.value(row) == BlobKind::ManagedWithBase as u8 {
                    blob_ids.value(row)
                } else {
                    0
                });
                output_uris.push(uris.value(row).to_string());
            }
            BlobKind::External => {
                output_kinds.push(BlobKind::External as u8);
                output_positions.push(if positions.is_null(row) {
                    0
                } else {
                    positions.value(row)
                });
                output_sizes.push(if sizes.is_null(row) {
                    0
                } else {
                    sizes.value(row)
                });
                output_ids.push(if blob_ids.is_null(row) {
                    0
                } else {
                    blob_ids.value(row)
                });
                output_uris.push(uris.value(row).to_string());
            }
        }
    }

    let descriptor = StructArray::try_new(
        BLOB_V2_DESC_FIELDS.clone(),
        vec![
            Arc::new(UInt8Array::from(output_kinds)),
            Arc::new(UInt64Array::from(output_positions)),
            Arc::new(UInt64Array::from(output_sizes)),
            Arc::new(UInt32Array::from(output_ids)),
            Arc::new(StringArray::from(output_uris)),
        ],
        values.nulls().cloned(),
    )?;
    let field = Arc::new(
        ArrowField::new(
            field.name(),
            descriptor.data_type().clone(),
            field.is_nullable(),
        )
        .with_metadata(field.metadata().clone()),
    );
    Ok((Arc::new(descriptor), field))
}

fn prepared_field_to_descriptor(
    array: &ArrayRef,
    field: &Arc<ArrowField>,
) -> Result<(ArrayRef, Arc<ArrowField>)> {
    if blob_v2_layout(field.as_ref()) == Some(BlobV2Layout::Prepared) {
        return prepared_blob_to_descriptor(array, field.as_ref());
    }

    match field.data_type() {
        ArrowDataType::Struct(children) => {
            let values = array.as_struct();
            let converted = values
                .columns()
                .iter()
                .zip(children.iter())
                .map(|(array, field)| prepared_field_to_descriptor(array, field))
                .collect::<Result<Vec<_>>>()?;
            let (arrays, fields): (Vec<_>, Vec<_>) = converted.into_iter().unzip();
            let output = StructArray::try_new(fields.into(), arrays, values.nulls().cloned())?;
            let field = Arc::new(
                ArrowField::new(
                    field.name(),
                    output.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );
            Ok((Arc::new(output), field))
        }
        ArrowDataType::List(child) => {
            let values = array.as_list::<i32>();
            let (child_array, child_field) = prepared_field_to_descriptor(values.values(), child)?;
            let output = GenericListArray::<i32>::try_new(
                child_field,
                values.offsets().clone(),
                child_array,
                values.nulls().cloned(),
            )?;
            let field = Arc::new(
                ArrowField::new(
                    field.name(),
                    output.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );
            Ok((Arc::new(output), field))
        }
        ArrowDataType::LargeList(child) => {
            let values = array.as_list::<i64>();
            let (child_array, child_field) = prepared_field_to_descriptor(values.values(), child)?;
            let output = GenericListArray::<i64>::try_new(
                child_field,
                values.offsets().clone(),
                child_array,
                values.nulls().cloned(),
            )?;
            let field = Arc::new(
                ArrowField::new(
                    field.name(),
                    output.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );
            Ok((Arc::new(output), field))
        }
        _ => Ok((array.clone(), field.clone())),
    }
}

pub fn prepared_blob_batch_to_descriptors(batch: &RecordBatch) -> Result<RecordBatch> {
    let converted = batch
        .columns()
        .iter()
        .zip(batch.schema().fields().iter())
        .map(|(array, field)| prepared_field_to_descriptor(array, field))
        .collect::<Result<Vec<_>>>()?;
    let (columns, fields): (Vec<_>, Vec<_>) = converted.into_iter().unzip();
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        batch.schema().metadata().clone(),
    ));
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|error| Error::internal(format!("convert prepared MemWAL Blob batch: {error}")))
}

/// Call `visit` with the kind, `blob_id` and `blob_size` of every non-null
/// prepared blob v2 value in `array`, at any depth.
fn visit_prepared_blobs(
    array: &ArrayRef,
    field: &Arc<ArrowField>,
    visit: &mut dyn FnMut(BlobKind, u32, u64),
) -> Result<()> {
    // Normalizing a subtree can copy its values, so skip those without blobs.
    if !arrow_field_contains_blob_v2(field) {
        return Ok(());
    }
    if blob_v2_layout(field.as_ref()) == Some(BlobV2Layout::Prepared) {
        validate_prepared_blob_array(field.as_ref(), array)?;
        let values = array.as_struct();
        let kinds = values
            .column_by_name("kind")
            .expect("validated prepared Blob has kind")
            .as_primitive::<UInt8Type>();
        let ids = values
            .column_by_name("blob_id")
            .expect("validated prepared Blob has blob_id")
            .as_primitive::<UInt32Type>();
        let sizes = values
            .column_by_name("blob_size")
            .expect("validated prepared Blob has blob_size")
            .as_primitive::<UInt64Type>();
        for row in 0..values.len() {
            if values.is_valid(row) {
                visit(
                    BlobKind::try_from(kinds.value(row))?,
                    ids.value(row),
                    sizes.value(row),
                );
            }
        }
        return Ok(());
    }

    match field.data_type() {
        // Only the values the rows hold: a sliced list's offsets reach part of
        // its values, and a null struct masks its children's values.
        ArrowDataType::Struct(children) => {
            let values = array.as_struct().normalize_slicing()?.pushdown_nulls()?;
            for (child_array, child_field) in values.columns().iter().zip(children.iter()) {
                visit_prepared_blobs(child_array, child_field, visit)?;
            }
        }
        ArrowDataType::List(child) => {
            let list = array.as_list::<i32>().filter_garbage_nulls();
            visit_prepared_blobs(&list.trimmed_values(), child, visit)?;
        }
        ArrowDataType::LargeList(child) => {
            let list = array.as_list::<i64>().filter_garbage_nulls();
            visit_prepared_blobs(&list.trimmed_values(), child, visit)?;
        }
        _ => {}
    }
    Ok(())
}

fn visit_batch_prepared_blobs(
    batch: &RecordBatch,
    visit: &mut dyn FnMut(BlobKind, u32, u64),
) -> Result<()> {
    for (array, field) in batch.columns().iter().zip(batch.schema().fields().iter()) {
        visit_prepared_blobs(array, field, visit)?;
    }
    Ok(())
}

pub fn prepared_blob_ids(batch: &RecordBatch) -> Result<HashSet<u32>> {
    let mut blob_ids = HashSet::new();
    visit_batch_prepared_blobs(batch, &mut |kind, blob_id, _| {
        if matches!(kind, BlobKind::Packed | BlobKind::Dedicated) {
            blob_ids.insert(blob_id);
        }
    })?;
    Ok(blob_ids)
}

/// Payload bytes of the prepared blob v2 values in `batch` that sit in packed
/// or dedicated sidecars rather than in the batch.
pub fn prepared_sidecar_bytes(batch: &RecordBatch) -> Result<u64> {
    let mut bytes = 0u64;
    visit_batch_prepared_blobs(batch, &mut |kind, _, size| {
        if matches!(kind, BlobKind::Packed | BlobKind::Dedicated) {
            bytes = bytes.saturating_add(size);
        }
    })?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder};
    use crate::dataset::blob::locate::BlobV2DescriptorColumns;
    use arrow::array::AsArray;
    use arrow_array::{ListArray, RecordBatch, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    #[case::local(None, 7, 16)]
    #[case::range(Some(0), 7, 16)]
    #[case::empty(Some(u32::MAX), 0, 0)]
    fn managed_prepared_batch_preserves_address(
        #[case] base_id: Option<u32>,
        #[case] offset: u64,
        #[case] size: u64,
    ) {
        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .push(BlobDescriptor::Managed {
                base_id,
                uri: "_blobs/payload.blob".to_string(),
                offset,
                size,
            })
            .unwrap();
        let (field, array) = builder.finish().unwrap().into_parts();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).unwrap();
        let batch = prepared_blob_batch_to_descriptors(&batch).unwrap();
        let columns = BlobV2DescriptorColumns::new(batch["blob"].as_struct());
        assert_eq!(columns.managed_base_id(0), base_id);
        assert_eq!(columns.positions.value(0), offset);
        assert_eq!(columns.sizes.value(0), size);
        assert_eq!(columns.blob_uris.value(0), "_blobs/payload.blob");
    }

    /// A prepared blob column holding one value of every kind: null, inline,
    /// packed (20 bytes), dedicated (1000 bytes), managed and External.
    fn every_kind() -> (Arc<Field>, ArrayRef) {
        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        for value in [
            BlobDescriptor::Null,
            BlobDescriptor::Inline { data: "abc".into() },
            BlobDescriptor::Packed {
                blob_id: 1,
                offset: 0,
                size: 20,
            },
            BlobDescriptor::Dedicated {
                blob_id: 2,
                size: 1000,
            },
            BlobDescriptor::Managed {
                base_id: None,
                uri: "_blobs/payload.blob".to_string(),
                offset: 0,
                size: 7,
            },
            BlobDescriptor::External {
                base_id: 0,
                uri: "file:///external.bin".to_string(),
                offset: 0,
                size: 9,
            },
        ] {
            builder.push(value).unwrap();
        }
        let (field, array) = builder.finish().unwrap().into_parts();
        (Arc::new(field), array)
    }

    fn in_struct(
        field: &Arc<Field>,
        values: ArrayRef,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef) {
        let children: Fields = vec![field.clone()].into();
        (
            Field::new("meta", DataType::Struct(children.clone()), true),
            Arc::new(StructArray::new(children, vec![values], nulls)),
        )
    }

    /// `values` as one-element lists.
    fn in_list(field: &Arc<Field>, values: ArrayRef) -> (Field, ArrayRef) {
        let rows = values.len();
        (
            Field::new("items", DataType::List(field.clone()), true),
            Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::from_lengths(vec![1; rows]),
                values,
                None,
            )),
        )
    }

    fn sidecar_bytes_of(columns: Vec<(Field, ArrayRef)>) -> u64 {
        let (fields, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        prepared_sidecar_bytes(&batch).unwrap()
    }

    /// Only packed and dedicated payloads count, at the top level, in a struct
    /// and as list elements.
    #[test]
    fn sidecar_bytes_count_packed_and_dedicated_at_any_depth() {
        let (field, values) = every_kind();
        let bytes = sidecar_bytes_of(vec![
            (field.as_ref().clone(), values.clone()),
            in_struct(&field, values.clone(), None),
            in_list(&field, values),
        ]);
        assert_eq!(bytes, 3 * (20 + 1000));
    }

    /// Values a sliced list no longer reaches, and values under a null struct,
    /// are not counted.
    #[test]
    fn sidecar_bytes_count_only_the_rows_the_batch_holds() {
        let (field, values) = every_kind();
        let (items, list) = in_list(&field, values.clone());
        // The dedicated value alone.
        assert_eq!(sidecar_bytes_of(vec![(items, list.slice(3, 1))]), 1000);
        // Only the packed value's struct is valid.
        let nulls = NullBuffer::from(vec![false, false, true, false, false, false]);
        assert_eq!(
            sidecar_bytes_of(vec![in_struct(&field, values, Some(nulls))]),
            20
        );
    }
}
