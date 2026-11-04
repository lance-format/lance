// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::builder::{LargeBinaryBuilder, StringBuilder, UInt64Builder};
use arrow_array::{
    Array, ArrayRef, GenericListArray, RecordBatch, StringArray, StructArray, UInt8Array,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;

use super::external::arrow_field_contains_blob_v2;
use crate::blob::{blob_v2_layout, validate_prepared_blob_array};
use lance_arrow::list::ListArrayExt;
use lance_arrow::r#struct::StructArrayExt;
use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BLOB_V2_LOGICAL_FIELDS, BlobKind, BlobV2Layout};
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

/// `field` at `data_type`, keeping its name, nullability and metadata.
fn retyped(field: &ArrowField, data_type: ArrowDataType) -> Arc<ArrowField> {
    Arc::new(
        ArrowField::new(field.name(), data_type, field.is_nullable())
            .with_metadata(field.metadata().clone()),
    )
}

type PreparedBlobMap<'a> =
    &'a dyn Fn(&ArrayRef, &ArrowField) -> Result<(ArrayRef, Arc<ArrowField>)>;

/// `array` with `convert` applied to each prepared blob v2 value at any depth.
///
/// Only the values the rows hold are converted, the same ones
/// [`visit_prepared_blob_values`] visits: a null struct masks its children's
/// values, and a list keeps only the values its offsets reach.
fn map_prepared_blobs(
    array: &ArrayRef,
    field: &Arc<ArrowField>,
    convert: PreparedBlobMap<'_>,
) -> Result<(ArrayRef, Arc<ArrowField>)> {
    if !arrow_field_contains_blob_v2(field) {
        return Ok((array.clone(), field.clone()));
    }
    if blob_v2_layout(field.as_ref()) == Some(BlobV2Layout::Prepared) {
        return convert(array, field.as_ref());
    }

    let output: ArrayRef = match field.data_type() {
        ArrowDataType::Struct(children) => {
            let values = array.as_struct().normalize_slicing()?.pushdown_nulls()?;
            let converted = values
                .columns()
                .iter()
                .zip(children.iter())
                .map(|(array, field)| map_prepared_blobs(array, field, convert))
                .collect::<Result<Vec<_>>>()?;
            let (arrays, fields): (Vec<_>, Vec<_>) = converted.into_iter().unzip();
            Arc::new(StructArray::try_new(
                fields.into(),
                arrays,
                values.nulls().cloned(),
            )?)
        }
        ArrowDataType::List(child) => {
            let values = array.as_list::<i32>().filter_garbage_nulls();
            let (child_array, child_field) =
                map_prepared_blobs(&values.trimmed_values(), child, convert)?;
            Arc::new(GenericListArray::<i32>::try_new(
                child_field,
                values.trimmed_offsets(),
                child_array,
                values.nulls().cloned(),
            )?)
        }
        ArrowDataType::LargeList(child) => {
            let values = array.as_list::<i64>().filter_garbage_nulls();
            let (child_array, child_field) =
                map_prepared_blobs(&values.trimmed_values(), child, convert)?;
            Arc::new(GenericListArray::<i64>::try_new(
                child_field,
                values.trimmed_offsets(),
                child_array,
                values.nulls().cloned(),
            )?)
        }
        _ => return Ok((array.clone(), field.clone())),
    };
    let field = retyped(field, output.data_type().clone());
    Ok((output, field))
}

/// `batch` with `convert` applied to each prepared blob v2 value at any depth.
fn map_batch_prepared_blobs(
    batch: &RecordBatch,
    convert: PreparedBlobMap<'_>,
) -> Result<RecordBatch> {
    let converted = batch
        .columns()
        .iter()
        .zip(batch.schema().fields().iter())
        .map(|(array, field)| map_prepared_blobs(array, field, convert))
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

pub fn prepared_blob_batch_to_descriptors(batch: &RecordBatch) -> Result<RecordBatch> {
    map_batch_prepared_blobs(batch, &prepared_blob_to_descriptor)
}

/// `batch` with each prepared blob v2 value at any depth in the complete
/// logical layout. A packed or dedicated payload comes from `sidecars`, keyed
/// by [`prepared_sidecar_reads`]; an External value keeps its link.
pub fn prepared_blob_batch_to_logical(
    batch: &RecordBatch,
    sidecars: &HashMap<SidecarRead, Bytes>,
) -> Result<RecordBatch> {
    map_batch_prepared_blobs(batch, &|array, field| {
        prepared_blob_to_logical(array, field, sidecars)
    })
}

fn prepared_blob_to_logical(
    array: &ArrayRef,
    field: &ArrowField,
    sidecars: &HashMap<SidecarRead, Bytes>,
) -> Result<(ArrayRef, Arc<ArrowField>)> {
    validate_prepared_blob_array(field, array)?;
    let values = array.as_struct();
    let column = |name: &str| {
        values
            .column_by_name(name)
            .expect("validated prepared Blob")
    };
    let kinds = column("kind").as_primitive::<UInt8Type>();
    let inline = column("data").as_binary::<i64>();
    let uris = column("uri").as_string::<i32>();
    let positions = column("position").as_primitive::<UInt64Type>();
    let sizes = column("blob_size").as_primitive::<UInt64Type>();
    let mut data = LargeBinaryBuilder::new();
    let mut uri = StringBuilder::new();
    let mut position = UInt64Builder::new();
    let mut size = UInt64Builder::new();
    for row in 0..values.len() {
        let kind = match values.is_valid(row) {
            true => Some(BlobKind::try_from(kinds.value(row))?),
            false => None,
        };
        match kind {
            Some(BlobKind::Inline) => data.append_value(inline.value(row)),
            Some(BlobKind::Packed | BlobKind::Dedicated) => {
                let bytes = SidecarRead::of(values, row)?
                    .and_then(|read| sidecars.get(&read))
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Blob v2 field '{}' row {row}: its sidecar payload was not read",
                            field.name()
                        ))
                    })?;
                data.append_value(bytes);
            }
            Some(kind @ (BlobKind::Managed | BlobKind::ManagedWithBase)) => {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {row}: a {kind:?} value has no logical form",
                    field.name()
                )));
            }
            None | Some(BlobKind::External) => data.append_null(),
        }
        // Only an External value carries a link, with its optional range.
        if kind == Some(BlobKind::External) {
            uri.append_value(uris.value(row));
            position.append_option(positions.is_valid(row).then(|| positions.value(row)));
            size.append_option(sizes.is_valid(row).then(|| sizes.value(row)));
        } else {
            uri.append_null();
            position.append_null();
            size.append_null();
        }
    }
    let logical = StructArray::try_new(
        BLOB_V2_LOGICAL_FIELDS.clone(),
        vec![
            Arc::new(data.finish()),
            Arc::new(uri.finish()),
            Arc::new(position.finish()),
            Arc::new(size.finish()),
        ],
        values.nulls().cloned(),
    )?;
    let field = retyped(field, logical.data_type().clone());
    Ok((Arc::new(logical), field))
}

/// A packed or dedicated payload's place in its sidecar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SidecarRead {
    pub blob_id: u32,
    pub position: u64,
    pub size: u64,
}

impl SidecarRead {
    /// The sidecar read row `row` of a validated prepared blob needs, if any.
    fn of(values: &StructArray, row: usize) -> Result<Option<Self>> {
        if values.is_null(row) {
            return Ok(None);
        }
        let column = |name: &str| {
            values
                .column_by_name(name)
                .expect("validated prepared Blob")
        };
        let kind = BlobKind::try_from(column("kind").as_primitive::<UInt8Type>().value(row))?;
        let position = match kind {
            BlobKind::Packed => column("position").as_primitive::<UInt64Type>().value(row),
            BlobKind::Dedicated => 0,
            _ => return Ok(None),
        };
        Ok(Some(Self {
            blob_id: column("blob_id").as_primitive::<UInt32Type>().value(row),
            position,
            size: column("blob_size").as_primitive::<UInt64Type>().value(row),
        }))
    }
}

/// The packed and dedicated sidecar reads the prepared blob v2 values in
/// `batch` need to become logical values.
pub fn prepared_sidecar_reads(batch: &RecordBatch) -> Result<HashSet<SidecarRead>> {
    let mut reads = HashSet::new();
    visit_sidecar_reads(batch, &mut |read| {
        reads.insert(read);
    })?;
    Ok(reads)
}

/// Call `visit` with each non-null prepared blob v2 value in `array`, at any
/// depth, as its struct and row.
fn visit_prepared_blob_values(
    array: &ArrayRef,
    field: &Arc<ArrowField>,
    visit: &mut dyn FnMut(&StructArray, usize) -> Result<()>,
) -> Result<()> {
    // Normalizing a subtree can copy its values, so skip those without blobs.
    if !arrow_field_contains_blob_v2(field) {
        return Ok(());
    }
    if blob_v2_layout(field.as_ref()) == Some(BlobV2Layout::Prepared) {
        validate_prepared_blob_array(field.as_ref(), array)?;
        let values = array.as_struct();
        for row in 0..values.len() {
            if values.is_valid(row) {
                visit(values, row)?;
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
                visit_prepared_blob_values(child_array, child_field, visit)?;
            }
        }
        ArrowDataType::List(child) => {
            let list = array.as_list::<i32>().filter_garbage_nulls();
            visit_prepared_blob_values(&list.trimmed_values(), child, visit)?;
        }
        ArrowDataType::LargeList(child) => {
            let list = array.as_list::<i64>().filter_garbage_nulls();
            visit_prepared_blob_values(&list.trimmed_values(), child, visit)?;
        }
        _ => {}
    }
    Ok(())
}

/// Call `visit` with the sidecar read of each packed or dedicated value in
/// `batch`, once per value.
fn visit_sidecar_reads(batch: &RecordBatch, visit: &mut dyn FnMut(SidecarRead)) -> Result<()> {
    for (array, field) in batch.columns().iter().zip(batch.schema().fields().iter()) {
        visit_prepared_blob_values(array, field, &mut |values, row| {
            if let Some(read) = SidecarRead::of(values, row)? {
                visit(read);
            }
            Ok(())
        })?;
    }
    Ok(())
}

pub fn prepared_blob_ids(batch: &RecordBatch) -> Result<HashSet<u32>> {
    let mut blob_ids = HashSet::new();
    visit_sidecar_reads(batch, &mut |read| {
        blob_ids.insert(read.blob_id);
    })?;
    Ok(blob_ids)
}

/// Payload bytes of the prepared blob v2 values in `batch` that sit in packed
/// or dedicated sidecars rather than in the batch.
pub fn prepared_sidecar_bytes(batch: &RecordBatch) -> Result<u64> {
    let mut bytes = 0u64;
    visit_sidecar_reads(batch, &mut |read| bytes = bytes.saturating_add(read.size))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blob::{BlobDescriptor, BlobDescriptorArrayBuilder};
    use crate::dataset::blob::locate::StoredBlob;
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

    /// Every prepared kind but Managed has a logical value, at any depth: an
    /// inline or sidecar payload as its bytes, an External value as its link.
    #[test]
    fn prepared_values_become_logical() {
        let (field, values) = every_kind();
        let (packed, dedicated) = (
            SidecarRead {
                blob_id: 1,
                position: 0,
                size: 20,
            },
            SidecarRead {
                blob_id: 2,
                position: 0,
                size: 1000,
            },
        );
        let sidecars = HashMap::from([
            (packed, Bytes::from(vec![b'p'; 20])),
            (dedicated, Bytes::from(vec![b'd'; 1000])),
        ]);
        let batch = |values: ArrayRef| {
            let (fields, arrays): (Vec<_>, Vec<_>) = [
                (field.as_ref().clone(), values.clone()),
                in_struct(&field, values.clone(), None),
                in_list(&field, values),
            ]
            .into_iter()
            .unzip();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
        };
        assert_eq!(
            prepared_sidecar_reads(&batch(values.clone())).unwrap(),
            HashSet::from([packed, dedicated])
        );
        let managed = prepared_blob_batch_to_logical(&batch(values.clone()), &sidecars);
        assert!(managed.unwrap_err().to_string().contains("no logical form"));

        let without_managed =
            arrow_select::take::take(&values, &UInt32Array::from(vec![0, 1, 2, 3, 5]), None)
                .unwrap();
        // A value a null struct masks needs no read.
        let (meta, masked) = in_struct(
            &field,
            without_managed.clone(),
            Some(NullBuffer::from(vec![false, true, false, false, false])),
        );
        let masked = RecordBatch::try_new(Arc::new(Schema::new(vec![meta])), vec![masked]).unwrap();
        assert!(prepared_sidecar_reads(&masked).unwrap().is_empty());
        prepared_blob_batch_to_logical(&masked, &HashMap::new()).unwrap();

        let logical = prepared_blob_batch_to_logical(&batch(without_managed), &sidecars).unwrap();
        for blobs in [
            logical.column(0).clone(),
            logical.column(1).as_struct().column(0).clone(),
            logical.column(2).as_list::<i32>().values().clone(),
        ] {
            let blobs = blobs.as_struct();
            let data = blobs.column_by_name("data").unwrap().as_binary::<i64>();
            let uri = blobs.column_by_name("uri").unwrap().as_string::<i32>();
            let value = |row| match (blobs.is_valid(row), uri.is_valid(row)) {
                (false, _) => None,
                (true, false) => Some(data.value(row).to_vec()),
                (true, true) => Some(uri.value(row).as_bytes().to_vec()),
            };
            assert_eq!(
                (0..5).map(value).collect::<Vec<_>>(),
                [
                    None,
                    Some(b"abc".to_vec()),
                    Some(vec![b'p'; 20]),
                    Some(vec![b'd'; 1000]),
                    Some(b"file:///external.bin".to_vec()),
                ]
            );
        }
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
