// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray, UInt64Array,
    cast::AsArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use lance_core::{Error, Result, utils::address::RowAddress};

use crate::Dataset;

pub(super) struct PreparedCapture {
    pub(super) field: Field,
    pub(super) descriptors: StructArray,
    pub(super) addresses: Vec<RowAddress>,
    pub(super) indices: RecordBatch,
}

pub(super) fn prepare(
    dataset: &Dataset,
    path: &[i32],
    field: &Field,
    addresses: &[RowAddress],
    array: &ArrayRef,
) -> Result<PreparedCapture> {
    if addresses.len() != array.len() {
        return Err(Error::invalid_input(format!(
            "Nested blob capture row address count {} does not match root row count {}",
            addresses.len(),
            array.len()
        )));
    }
    let mut leaf = None;
    let (index_field, indices) = descend(
        dataset,
        path,
        field,
        array,
        addresses,
        &vec![true; array.len()],
        &mut leaf,
    )?;
    let (field, descriptors, addresses) =
        leaf.ok_or_else(|| Error::internal("Nested blob capture did not reach its selected leaf"))?;
    Ok(PreparedCapture {
        field,
        descriptors,
        addresses,
        indices: RecordBatch::try_new(Arc::new(Schema::new(vec![index_field])), vec![indices])?,
    })
}

type Leaf = Option<(Field, StructArray, Vec<RowAddress>)>;

fn descend(
    dataset: &Dataset,
    path: &[i32],
    field: &Field,
    array: &ArrayRef,
    addresses: &[RowAddress],
    visible: &[bool],
    leaf: &mut Leaf,
) -> Result<(Field, ArrayRef)> {
    let expected = path
        .first()
        .and_then(|id| dataset.schema().field_by_id(*id))
        .ok_or_else(|| Error::internal("Nested blob capture has an invalid field path"))?;
    if field.name() != &expected.name || field.data_type() != array.data_type() {
        return Err(Error::invalid_input(format!(
            "Nested blob capture expected field '{}' (field_id {}), got {field:?} with array type {}",
            expected.name,
            expected.id,
            array.data_type()
        )));
    }
    let visible = visible
        .iter()
        .enumerate()
        .map(|(row, visible)| *visible && array.is_valid(row))
        .collect::<Vec<_>>();
    if path.len() == 1 {
        let descriptors = array.as_struct_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "Nested blob capture leaf '{}' is not a descriptor struct",
                field.name()
            ))
        })?;
        let nulls = NullBuffer::union(descriptors.nulls(), Some(&NullBuffer::from(visible)));
        // Descriptor readers also treat a null kind as a SQL null blob.
        // Keep the mapping consistent with the flat references for this form.
        let nulls = NullBuffer::union(
            nulls.as_ref(),
            descriptors
                .column_by_name("kind")
                .and_then(|kind| kind.nulls()),
        );
        let descriptors = StructArray::try_new(
            descriptors.fields().clone(),
            descriptors.columns().to_vec(),
            nulls,
        )?;
        let indices: ArrayRef = Arc::new(UInt64Array::new(
            ScalarBuffer::from((0..descriptors.len() as u64).collect::<Vec<_>>()),
            descriptors.nulls().cloned(),
        ));
        *leaf = Some((field.clone(), descriptors, addresses.to_vec()));
        return Ok((Field::new(field.name(), DataType::UInt64, true), indices));
    }
    let (data_type, indices): (DataType, ArrayRef) = match array.data_type() {
        DataType::Struct(fields) => {
            let child = dataset
                .schema()
                .field_by_id(path[1])
                .ok_or_else(|| Error::internal("Nested blob capture child field disappeared"))?;
            let (index, child_field) = fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name() == &child.name)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Nested blob capture field '{}' is missing selected child '{}'",
                        field.name(),
                        child.name
                    ))
                })?;
            let (child_field, child_indices) = descend(
                dataset,
                &path[1..],
                child_field,
                array.as_struct().column(index),
                addresses,
                &visible,
                leaf,
            )?;
            let fields: arrow_schema::Fields = vec![child_field].into();
            let indices =
                StructArray::try_new(fields.clone(), vec![child_indices], array.nulls().cloned())?;
            (DataType::Struct(fields), Arc::new(indices))
        }
        DataType::List(child) => {
            let (field, indices) = descend_list(
                dataset,
                &path[1..],
                child,
                array.as_list::<i32>(),
                addresses,
                &visible,
                leaf,
            )?;
            (DataType::List(Arc::new(field)), indices)
        }
        DataType::LargeList(child) => {
            let (field, indices) = descend_list(
                dataset,
                &path[1..],
                child,
                array.as_list::<i64>(),
                addresses,
                &visible,
                leaf,
            )?;
            (DataType::LargeList(Arc::new(field)), indices)
        }
        data_type => {
            return Err(Error::not_supported(format!(
                "Nested blob capture field '{}' has unsupported container {data_type}; supported containers are Struct, List, and LargeList",
                field.name()
            )));
        }
    };
    // The mapping is not a value of the original container's extension type.
    Ok((Field::new(field.name(), data_type, true), indices))
}

fn descend_list<O: OffsetSizeTrait>(
    dataset: &Dataset,
    path: &[i32],
    field: &Field,
    array: &GenericListArray<O>,
    addresses: &[RowAddress],
    visible: &[bool],
    leaf: &mut Leaf,
) -> Result<(Field, ArrayRef)> {
    let offsets = array.value_offsets();
    let start = offsets[0].as_usize();
    let end = offsets[array.len()].as_usize();
    let mut child_addresses = Vec::with_capacity(end - start);
    let mut child_visible = Vec::with_capacity(end - start);
    let mut normalized = Vec::with_capacity(offsets.len());
    normalized.push(O::usize_as(0));
    for row in 0..array.len() {
        let length = offsets[row + 1].as_usize() - offsets[row].as_usize();
        child_addresses.extend(std::iter::repeat_n(addresses[row], length));
        child_visible.extend(std::iter::repeat_n(visible[row], length));
        normalized.push(O::usize_as(offsets[row + 1].as_usize() - start));
    }
    let (field, indices) = descend(
        dataset,
        path,
        field,
        &array.values().slice(start, end - start),
        &child_addresses,
        &child_visible,
        leaf,
    )?;
    let list = GenericListArray::<O>::try_new(
        Arc::new(field.clone()),
        OffsetBuffer::new(ScalarBuffer::from(normalized)),
        indices,
        array.nulls().cloned(),
    )?;
    Ok((field, Arc::new(list)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use arrow_array::{
        LargeListArray, ListArray, RecordBatchIterator, UInt32Array, types::UInt64Type,
    };
    use futures::TryStreamExt;
    use lance_core::datatypes::BlobHandling;
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;

    use crate::blob::reference::{
        BlobMaterializationContext, BlobRefBinding, BlobRefCapture, BlobRefReader,
        BlobSourceContext, NestedBlobRefBatch,
    };
    use crate::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
    use crate::dataset::WriteParams;

    #[cfg_attr(coverage, coverage(off))]
    async fn nested_dataset(large: bool, deep: bool) -> Arc<Dataset> {
        let field = blob_field_with_options(
            "blob.dot",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(0)
                .with_dedicated_size_threshold(NonZeroUsize::new(1).unwrap()),
        );
        let mut blobs = BlobArrayBuilder::new(7);
        let mut siblings = BlobArrayBuilder::new(7);
        for value in [
            Some(b"first".as_slice()),
            None,
            Some(b""),
            Some(b"hidden-child"),
            Some(b"hidden-parent"),
            Some(b"second"),
            None,
        ] {
            match value {
                Some(value) => blobs.push_bytes(value).unwrap(),
                None => blobs.push_null().unwrap(),
            }
            siblings.push_bytes(b"unrelated").unwrap();
        }
        let fields = vec![field.clone(), field.with_name("sibling")].into();
        let mut values: ArrayRef = Arc::new(
            StructArray::try_new(
                fields,
                vec![blobs.finish().unwrap(), siblings.finish().unwrap()],
                Some(NullBuffer::from(vec![
                    true, true, true, false, true, true, true,
                ])),
            )
            .unwrap(),
        );
        if deep {
            values = Arc::new(
                ListArray::try_new(
                    Arc::new(Field::new("item", values.data_type().clone(), true)),
                    OffsetBuffer::new(ScalarBuffer::from(
                        (0..=values.len() as i32).collect::<Vec<_>>(),
                    )),
                    values,
                    None,
                )
                .unwrap(),
            );
        }
        let item = Arc::new(Field::new("item", values.data_type().clone(), true));
        let offsets = vec![0, 3, 3, 3, 4, 5, 6, 7];
        let nulls = Some(NullBuffer::from(vec![
            true, true, false, true, true, true, true,
        ]));
        let lists: ArrayRef = if large {
            Arc::new(
                LargeListArray::try_new(
                    item,
                    OffsetBuffer::new(ScalarBuffer::from(
                        offsets.into_iter().map(i64::from).collect::<Vec<_>>(),
                    )),
                    values,
                    nulls,
                )
                .unwrap(),
            )
        } else {
            Arc::new(
                ListArray::try_new(
                    item,
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    values,
                    nulls,
                )
                .unwrap(),
            )
        };
        let root: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![Field::new("clips.dot", lists.data_type().clone(), true)].into(),
                vec![lists],
                Some(NullBuffer::from(vec![
                    true, true, true, true, false, true, true,
                ])),
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("media", root.data_type().clone(), true).with_metadata(
                std::collections::HashMap::from([
                    (
                        "ARROW:extension:name".to_string(),
                        "example.media".to_string(),
                    ),
                    (
                        "ARROW:extension:metadata".to_string(),
                        "original-value-schema".to_string(),
                    ),
                ]),
            ),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![root]).unwrap();
        Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 3,
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    #[rstest]
    #[case::list(false, false)]
    #[case::large_list(true, false)]
    #[case::nested_list(false, true)]
    #[case::mixed_lists(true, true)]
    #[tokio::test]
    async fn capture_nested_preserves_nulls_offsets_and_element_identity(
        #[case] large: bool,
        #[case] deep: bool,
    ) {
        let dataset = nested_dataset(large, deep).await;
        assert_eq!(dataset.get_fragments().len(), 3);
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/nested", "workers")
            .await
            .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .with_row_address()
            .blob_handling(BlobHandling::AllDescriptions);
        let scanned = scanner.try_into_batch().await.unwrap();
        let selected = UInt32Array::from(vec![5, 0, 4, 3, 1, 2, 6, 0]);
        let values = arrow::compute::take(&scanned["media"], &selected, None).unwrap();
        let addresses = selected
            .values()
            .iter()
            .map(|index| {
                RowAddress::from(
                    scanned["_rowaddr"]
                        .as_primitive::<UInt64Type>()
                        .value(*index as usize),
                )
            })
            .collect::<Vec<_>>();
        let field_id = dataset
            .schema()
            .fields_pre_order()
            .find(|field| field.name == "blob.dot")
            .unwrap()
            .id;
        let mut capture =
            BlobRefCapture::try_new(dataset.clone(), source.clone(), field_id).unwrap();
        let before = dataset.object_store.io_stats_snapshot().requests.len();
        let nested = capture
            .capture_nested(&source, scanned.schema().field(0), &addresses, &values)
            .await
            .unwrap();
        assert_eq!(
            dataset.object_store.io_stats_snapshot().requests.len(),
            before
        );
        let nested = NestedBlobRefBatch::decode(&nested.encode().unwrap()).unwrap();
        let references = nested.references();
        let roots = nested.indices().column(0).as_struct();
        assert!(roots.is_null(2));
        let lists = roots.column(0);
        let values = if large {
            let lists = lists.as_list::<i64>();
            assert_eq!(lists.value_offsets(), &[0, 1, 4, 4, 5, 5, 5, 6, 9]);
            assert_eq!(lists.value_length(4), 0);
            assert!(lists.is_null(5));
            lists.values()
        } else {
            let lists = lists.as_list::<i32>();
            assert_eq!(lists.value_offsets(), &[0, 1, 4, 4, 5, 5, 5, 6, 9]);
            assert_eq!(lists.value_length(4), 0);
            assert!(lists.is_null(5));
            lists.values()
        };
        let values = if deep {
            values.as_list::<i32>().values()
        } else {
            values
        };
        let indices = values.as_struct().column(0).as_primitive::<UInt64Type>();
        assert_eq!(
            indices.iter().collect::<Vec<_>>(),
            vec![
                Some(0),
                Some(1),
                None,
                Some(3),
                None,
                None,
                Some(6),
                None,
                Some(8)
            ]
        );
        assert!(values.is_null(4));
        for index in [2, 4, 5, 7] {
            assert!(references.reference(index).unwrap().is_none());
        }
        let reader = BlobRefReader::try_new(
            vec![BlobRefBinding::try_new(source, dataset.clone()).unwrap()],
            BlobMaterializationContext::try_new(1024, 1024).unwrap(),
        )
        .unwrap();
        let selected = [0, 1, 3, 4, 5, 6, 8];
        let requests = references
            .read_requests(&selected, &selected, &[None; 7])
            .unwrap();
        let mut stream = reader.read(requests).unwrap();
        let mut results = std::collections::HashMap::new();
        while let Some(result) = stream.try_next().await.unwrap() {
            results.insert(
                result.request_index,
                result.data.map(|bytes| bytes.to_vec()),
            );
        }
        assert_eq!(results[&0].as_deref(), Some(b"second".as_slice()));
        assert_eq!(results[&1].as_deref(), Some(b"first".as_slice()));
        assert_eq!(results[&6], results[&1]);
        assert_eq!(results[&3].as_deref(), Some(b"".as_slice()));
        assert!(results[&4].is_none());
        assert!(results[&5].is_none());
        let metrics = reader.metrics().unwrap();
        assert_eq!(metrics.payload_requests, 2);
        assert_eq!(metrics.payload_bytes, 11);
        assert_eq!(metrics.requested_logical_bytes, 16);
        assert_eq!(metrics.returned_logical_bytes, 16);
        assert_eq!(metrics.duplicate_demands, 1);
    }

    #[rstest]
    #[case::struct_parent(true)]
    #[case::list_parent(false)]
    #[tokio::test]
    async fn capture_nested_masks_hidden_children_before_resolution(#[case] null_struct: bool) {
        let dataset = nested_dataset(false, false).await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/hidden", "workers")
            .await
            .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .with_row_address()
            .blob_handling(BlobHandling::AllDescriptions);
        let scanned = scanner.try_into_batch().await.unwrap().slice(0, 1);
        let root = scanned["media"].as_struct();
        let lists = root.column(0).as_list::<i32>();
        // Engine operators can introduce null parents while retaining their child buffers.
        let DataType::List(item) = lists.data_type() else {
            panic!("Expected List fixture")
        };
        let lists: ArrayRef = Arc::new(
            ListArray::try_new(
                item.clone(),
                lists.offsets().clone(),
                lists.values().clone(),
                (!null_struct).then(|| NullBuffer::from(vec![false])),
            )
            .unwrap(),
        );
        let values: ArrayRef = Arc::new(
            StructArray::try_new(
                root.fields().clone(),
                vec![lists],
                null_struct.then(|| NullBuffer::from(vec![false])),
            )
            .unwrap(),
        );
        let addresses = [RowAddress::from(
            scanned["_rowaddr"].as_primitive::<UInt64Type>().value(0),
        )];
        let field_id = dataset
            .schema()
            .fields_pre_order()
            .find(|field| field.name == "blob.dot")
            .unwrap()
            .id;
        let mut capture =
            BlobRefCapture::try_new(dataset.clone(), source.clone(), field_id).unwrap();
        let nested = capture
            .capture_nested(&source, scanned.schema().field(0), &addresses, &values)
            .await
            .unwrap();
        for row in 0..3 {
            assert!(nested.references().reference(row).unwrap().is_none());
        }
        let reader = BlobRefReader::try_new(
            vec![BlobRefBinding::try_new(source, dataset).unwrap()],
            BlobMaterializationContext::try_new(1024, 1024).unwrap(),
        )
        .unwrap();
        let requests = nested
            .references()
            .read_requests(&[0, 1, 2], &[0, 1, 2], &[None; 3])
            .unwrap();
        let results = reader
            .read(requests)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(results.iter().all(|result| result.data.is_none()));
        assert_eq!(reader.metrics().unwrap().payload_requests, 0);
    }

    #[tokio::test]
    async fn capture_nested_mapping_drops_value_extension_metadata() {
        let dataset = nested_dataset(false, false).await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/mapping", "workers")
            .await
            .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .with_row_address()
            .blob_handling(BlobHandling::AllDescriptions);
        let scanned = scanner.try_into_batch().await.unwrap();
        let field_id = dataset
            .schema()
            .fields_pre_order()
            .find(|field| field.name == "blob.dot")
            .unwrap()
            .id;
        let field = scanned.schema().field(0).clone();
        assert_eq!(
            field
                .metadata()
                .get("ARROW:extension:name")
                .map(String::as_str),
            Some("example.media")
        );
        let addresses = scanned["_rowaddr"]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect::<Vec<_>>();
        let mut capture = BlobRefCapture::try_new(dataset, source.clone(), field_id).unwrap();
        let nested = capture
            .capture_nested(&source, &field, &addresses, &scanned["media"])
            .await
            .unwrap();
        assert!(nested.indices().schema().field(0).metadata().is_empty());
        let restored = NestedBlobRefBatch::decode(&nested.encode().unwrap()).unwrap();
        assert_eq!(restored.indices(), nested.indices());
    }
}
