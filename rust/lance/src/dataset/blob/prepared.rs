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

use crate::blob::{blob_v2_layout, validate_prepared_blob_array};
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

fn collect_prepared_blob_ids(
    array: &ArrayRef,
    field: &Arc<ArrowField>,
    blob_ids: &mut HashSet<u32>,
) -> Result<()> {
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
        for row in 0..values.len() {
            if values.is_valid(row)
                && matches!(
                    BlobKind::try_from(kinds.value(row))?,
                    BlobKind::Packed | BlobKind::Dedicated
                )
            {
                blob_ids.insert(ids.value(row));
            }
        }
        return Ok(());
    }

    match field.data_type() {
        ArrowDataType::Struct(children) => {
            let values = array.as_struct();
            for (child_array, child_field) in values.columns().iter().zip(children.iter()) {
                collect_prepared_blob_ids(child_array, child_field, blob_ids)?;
            }
        }
        ArrowDataType::List(child) => {
            collect_prepared_blob_ids(array.as_list::<i32>().values(), child, blob_ids)?;
        }
        ArrowDataType::LargeList(child) => {
            collect_prepared_blob_ids(array.as_list::<i64>().values(), child, blob_ids)?;
        }
        _ => {}
    }
    Ok(())
}

pub fn prepared_blob_ids(batch: &RecordBatch) -> Result<HashSet<u32>> {
    let mut blob_ids = HashSet::new();
    for (array, field) in batch.columns().iter().zip(batch.schema().fields().iter()) {
        collect_prepared_blob_ids(array, field, &mut blob_ids)?;
    }
    Ok(blob_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder};
    use crate::dataset::blob::locate::StoredBlob;
    use arrow::array::AsArray;
    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
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
        assert_eq!(
            StoredBlob::parse(batch["blob"].as_struct(), 0).unwrap(),
            Some(StoredBlob::Managed {
                base_id,
                uri: "_blobs/payload.blob",
                position: offset,
                size,
            })
        );
    }
}
