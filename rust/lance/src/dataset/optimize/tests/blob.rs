// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the blob contract area.

use super::*;

#[tokio::test]
async fn test_compact_blob_columns() {
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("blob", DataType::LargeBinary, false)
            .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
    ]));
    let expected_payload: Vec<Vec<u8>> =
        vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9, 10], vec![11]];
    let id_column: ArrayRef = Arc::new(Int32Array::from_iter_values(
        0..expected_payload.len() as i32,
    ));
    let blob_array: ArrayRef = Arc::new(LargeBinaryArray::from_iter(
        expected_payload.iter().map(|value| Some(value.as_slice())),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![id_column, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.validate().await.unwrap();
    assert!(dataset.get_fragments().len() > 1);

    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);

    let dataset = Arc::new(dataset);
    let row_indices: Vec<u64> = (0..expected_payload.len() as u64).collect();
    let blobs = dataset
        .take_blobs_by_indices(&row_indices, "blob")
        .await
        .unwrap();
    assert_eq!(blobs.len(), expected_payload.len());
    for (blob, expected) in blobs.iter().zip(expected_payload.iter()) {
        let bytes = blob.as_ref().unwrap().read().await.unwrap();
        assert_eq!(bytes.as_ref(), expected.as_slice());
    }
}
#[tokio::test]
async fn test_compact_blob_v1_preserves_null_empty_and_payload_order() {
    let test_dir = TempStrDir::default();
    let expected = mixed_blob_values();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("blob", DataType::LargeBinary, true)
            .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..expected.len() as i32)),
            Arc::new(LargeBinaryArray::from_iter(
                expected.iter().map(|(_, value)| value.as_deref()),
            )),
        ],
    )
    .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    Box::pin(assert_compaction_preserves_blob_values(dataset, &expected)).await;
}
#[tokio::test]
async fn test_compact_blob_v2_preserves_null_empty_and_payload_order() {
    use crate::BlobArrayBuilder;

    let test_dir = TempStrDir::default();
    let expected = mixed_blob_values();
    let mut blob_builder = BlobArrayBuilder::new(expected.len());
    for (_, value) in &expected {
        match value {
            Some(value) => blob_builder.push_bytes(value).unwrap(),
            None => blob_builder.push_null().unwrap(),
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..expected.len() as i32)),
            blob_builder.finish().unwrap(),
        ],
    )
    .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    Box::pin(assert_compaction_preserves_blob_values(dataset, &expected)).await;
}
#[tokio::test]
async fn test_compact_blob_v2_preserves_external_references() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;
    use lance_table::format::BasePath;

    let test_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    std::fs::write(&external_path, b"external-data").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let base_uri = format!("file://{}", external_dir.std_path().display());

    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_uri(external_uri.clone()).unwrap();
    blob_builder.push_bytes(b"inline-data").unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 1,
            initial_bases: Some(vec![BasePath {
                id: 1,
                name: Some("external".to_string()),
                path: base_uri,
                is_dataset_root: false,
            }]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    for frag in dataset.get_fragments() {
        let rows = frag.physical_rows().await.unwrap();
        assert!(rows > 0, "fragment {} should have rows", frag.id());
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 1024 * 1024,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert!(
        !plan.tasks().is_empty(),
        "compaction plan should have tasks, got {} tasks",
        plan.tasks().len()
    );

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let scan_result = dataset
        .scan()
        .project(&["id", "blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(scan_result.num_rows(), 2);

    let ids = scan_result
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let mut id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
    id_values.sort();
    assert_eq!(id_values, vec![0, 1]);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"external-data".to_vec())),
            (1, Some(b"inline-data".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_packed_and_dedicated() {
    use crate::BlobArrayBuilder;
    use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let inline_data = b"small-inline-blob".as_slice();
    let packed_data: Vec<u8> = (0..64 * 1024 + 1024).map(|i| (i % 256) as u8).collect();
    let dedicated_data: Vec<u8> = (0..4 * 1024 * 1024 + 512)
        .map(|i| ((i + 97) % 256) as u8)
        .collect();

    let mut blob_builder = BlobArrayBuilder::new(3);
    blob_builder.push_bytes(inline_data).unwrap();
    blob_builder.push_bytes(&packed_data).unwrap();
    blob_builder.push_bytes(&dedicated_data).unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
    let mut blob_field = crate::blob_field("blob", true);
    {
        let metadata = blob_field.metadata().clone();
        let mut new_metadata = metadata;
        new_metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            (4 * 1024 * 1024).to_string(),
        );
        blob_field = blob_field.with_metadata(new_metadata);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        blob_field,
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 3);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let scan_result = dataset
        .scan()
        .project(&["id", "blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(scan_result.num_rows(), 3);

    let ids = scan_result
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
    assert_eq!(id_values, vec![0, 1, 2]);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(inline_data.to_vec())),
            (1, Some(packed_data)),
            (2, Some(dedicated_data))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_with_null_rows() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let mut blob_builder = BlobArrayBuilder::new(4);
    blob_builder.push_bytes(b"inline-0").unwrap();
    blob_builder.push_null().unwrap();
    blob_builder.push_bytes(b"inline-2").unwrap();
    blob_builder.push_null().unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![Some(0), Some(1), Some(2), Some(3)]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let scan_result = dataset
        .scan()
        .project(&["id", "blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(scan_result.num_rows(), 4);

    let ids = scan_result
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
    assert_eq!(id_values, vec![0, 1, 2, 3]);

    let blob_col = scan_result.column_by_name("blob").unwrap();
    assert!(
        matches!(blob_col.data_type(), DataType::Struct(_)),
        "blob column should be a struct after compaction"
    );

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"inline-0".to_vec())),
            (1, None),
            (2, Some(b"inline-2".to_vec())),
            (3, None)
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_deleted_rows_not_resurrected() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let mut blob_builder = BlobArrayBuilder::new(4);
    blob_builder.push_bytes(b"blob-0").unwrap();
    blob_builder.push_bytes(b"blob-1").unwrap();
    blob_builder.push_bytes(b"blob-2").unwrap();
    blob_builder.push_bytes(b"blob-3").unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    dataset.delete("id = 1").await.unwrap();
    dataset.delete("id = 2").await.unwrap();

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    let scan_result = dataset
        .scan()
        .project(&["id", "blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(scan_result.num_rows(), 2);

    let ids = scan_result
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let mut id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
    id_values.sort();
    assert_eq!(id_values, vec![0, 3]);

    let blob_col = scan_result.column_by_name("blob").unwrap();
    let struct_arr = blob_col.as_any().downcast_ref::<StructArray>().unwrap();
    let kind_col = struct_arr
        .column_by_name("kind")
        .unwrap()
        .as_primitive::<UInt8Type>();

    for i in 0..kind_col.len() {
        assert!(
            !kind_col.is_null(i),
            "row {} should have a non-null kind after compaction of deleted rows",
            i
        );
    }

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![(0, Some(b"blob-0".to_vec())), (3, Some(b"blob-3".to_vec()))]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_external_and_data_blob_mixed() {
    use crate::BlobArrayBuilder;
    use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
    use lance_core::utils::tempfile::TempDir;
    use lance_table::format::BasePath;

    let test_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    std::fs::write(&external_path, b"external-payload").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let base_uri = format!("file://{}", external_dir.std_path().display());

    let packed_data: Vec<u8> = (0..64 * 1024 + 512).map(|i| (i % 256) as u8).collect();

    let mut blob_builder = BlobArrayBuilder::new(4);
    blob_builder.push_uri(external_uri.clone()).unwrap();
    blob_builder.push_bytes(&packed_data).unwrap();
    blob_builder.push_bytes(b"inline-small").unwrap();
    blob_builder.push_uri(external_uri.clone()).unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
    let mut blob_field = crate::blob_field("blob", true);
    {
        let mut new_metadata = blob_field.metadata().clone();
        new_metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            (4 * 1024 * 1024).to_string(),
        );
        blob_field = blob_field.with_metadata(new_metadata);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        blob_field,
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            initial_bases: Some(vec![BasePath {
                id: 1,
                name: Some("external".to_string()),
                path: base_uri,
                is_dataset_root: false,
            }]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"external-payload".to_vec())),
            (1, Some(packed_data)),
            (2, Some(b"inline-small".to_vec())),
            (3, Some(b"external-payload".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_multiple_blob_columns() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let mut image_builder = BlobArrayBuilder::new(3);
    image_builder.push_bytes(b"image-0").unwrap();
    image_builder.push_bytes(b"image-1").unwrap();
    image_builder.push_bytes(b"image-2").unwrap();
    let image_array: ArrayRef = image_builder.finish().unwrap();

    let mut thumb_builder = BlobArrayBuilder::new(3);
    thumb_builder.push_bytes(b"thumb-0").unwrap();
    thumb_builder.push_null().unwrap();
    thumb_builder.push_bytes(b"thumb-2").unwrap();
    let thumb_array: ArrayRef = thumb_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("image", true),
        crate::blob_field("thumbnail", true),
    ]));

    let batch =
        RecordBatch::try_new(schema.clone(), vec![id_array, image_array, thumb_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 3);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut image_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "image").await;
    image_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        image_values,
        vec![
            (0, Some(b"image-0".to_vec())),
            (1, Some(b"image-1".to_vec())),
            (2, Some(b"image-2".to_vec()))
        ]
    );

    let mut thumb_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "thumbnail").await;
    thumb_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        thumb_values,
        vec![
            (0, Some(b"thumb-0".to_vec())),
            (1, None),
            (2, Some(b"thumb-2".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_external_and_null_mixed() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;
    use lance_table::format::BasePath;

    let test_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("mixed-external.bin");
    std::fs::write(&external_path, b"external-mixed-data").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let base_uri = format!("file://{}", external_dir.std_path().display());

    let mut blob_builder = BlobArrayBuilder::new(4);
    blob_builder.push_uri(external_uri.clone()).unwrap();
    blob_builder.push_null().unwrap();
    blob_builder.push_uri(external_uri.clone()).unwrap();
    blob_builder.push_null().unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            initial_bases: Some(vec![BasePath {
                id: 1,
                name: Some("external".to_string()),
                path: base_uri,
                is_dataset_root: false,
            }]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"external-mixed-data".to_vec())),
            (1, None),
            (2, Some(b"external-mixed-data".to_vec())),
            (3, None)
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_all_null_and_all_external_fragments() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;
    use lance_table::format::BasePath;

    let test_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("all-ext.bin");
    std::fs::write(&external_path, b"all-external-data").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let base_uri = format!("file://{}", external_dir.std_path().display());

    let mut null_builder = BlobArrayBuilder::new(2);
    null_builder.push_null().unwrap();
    null_builder.push_null().unwrap();
    let null_array: ArrayRef = null_builder.finish().unwrap();

    let mut ext_builder = BlobArrayBuilder::new(2);
    ext_builder.push_uri(external_uri.clone()).unwrap();
    ext_builder.push_uri(external_uri.clone()).unwrap();
    let ext_array: ArrayRef = ext_builder.finish().unwrap();

    let id_null_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
    let null_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));
    let null_batch =
        RecordBatch::try_new(null_schema.clone(), vec![id_null_array, null_array]).unwrap();

    let id_ext_array: ArrayRef = Arc::new(Int32Array::from(vec![2, 3]));
    let ext_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));
    let ext_batch =
        RecordBatch::try_new(ext_schema.clone(), vec![id_ext_array, ext_array]).unwrap();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![null_batch, ext_batch].into_iter().map(Ok),
            null_schema.clone(),
        ),
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            initial_bases: Some(vec![BasePath {
                id: 1,
                name: Some("external".to_string()),
                path: base_uri,
                is_dataset_root: false,
            }]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, None),
            (1, None),
            (2, Some(b"all-external-data".to_vec())),
            (3, Some(b"all-external-data".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_external_with_multiple_base_ids() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;
    use lance_table::format::BasePath;

    let test_dir = TempDir::default();
    let base_a_dir = TempDir::default();
    let base_b_dir = TempDir::default();

    let path_a = base_a_dir.std_path().join("data-a.bin");
    std::fs::write(&path_a, b"from-base-a").unwrap();
    let uri_a = format!("file://{}", path_a.display());
    let base_uri_a = format!("file://{}", base_a_dir.std_path().display());

    let path_b = base_b_dir.std_path().join("data-b.bin");
    std::fs::write(&path_b, b"from-base-b").unwrap();
    let uri_b = format!("file://{}", path_b.display());
    let base_uri_b = format!("file://{}", base_b_dir.std_path().display());

    let mut blob_builder = BlobArrayBuilder::new(4);
    blob_builder.push_uri(uri_a.clone()).unwrap();
    blob_builder.push_uri(uri_b).unwrap();
    blob_builder.push_bytes(b"inline-data").unwrap();
    blob_builder.push_uri(uri_a).unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            initial_bases: Some(vec![
                BasePath {
                    id: 1,
                    name: Some("base_a".to_string()),
                    path: base_uri_a,
                    is_dataset_root: false,
                },
                BasePath {
                    id: 2,
                    name: Some("base_b".to_string()),
                    path: base_uri_b,
                    is_dataset_root: false,
                },
            ]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"from-base-a".to_vec())),
            (1, Some(b"from-base-b".to_vec())),
            (2, Some(b"inline-data".to_vec())),
            (3, Some(b"from-base-a".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_large_blobs() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let large_blob_a: Vec<u8> = (0..512 * 1024).map(|i| (i % 256) as u8).collect();
    let large_blob_b: Vec<u8> = (0..256 * 1024).map(|i| ((i + 42) % 256) as u8).collect();

    let mut blob_builder = BlobArrayBuilder::new(3);
    blob_builder.push_bytes(&large_blob_a).unwrap();
    blob_builder.push_bytes(&large_blob_b).unwrap();
    blob_builder.push_bytes(b"small-blob").unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 3);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(large_blob_a)),
            (1, Some(large_blob_b)),
            (2, Some(b"small-blob".to_vec()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_blob_kind_reclassification() {
    use crate::BlobArrayBuilder;
    use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let medium_data: Vec<u8> = (0..32 * 1024).map(|i| (i % 256) as u8).collect();

    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(&medium_data).unwrap();
    blob_builder.push_bytes(&medium_data).unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
    let mut blob_field = crate::blob_field("blob", true);
    {
        let mut new_metadata = blob_field.metadata().clone();
        new_metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            (16 * 1024).to_string(),
        );
        blob_field = blob_field.with_metadata(new_metadata);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        blob_field,
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(medium_data.clone())),
            (1, Some(medium_data.clone()))
        ]
    );
}
#[tokio::test]
async fn test_compact_blob_v2_multi_batch() {
    use crate::BlobArrayBuilder;
    use lance_core::utils::tempfile::TempDir;

    let test_dir = TempDir::default();

    let mut blob_builder = BlobArrayBuilder::new(6);
    blob_builder.push_bytes(b"batch-0-row-0").unwrap();
    blob_builder.push_bytes(b"batch-0-row-1").unwrap();
    blob_builder.push_bytes(b"batch-1-row-0").unwrap();
    blob_builder.push_null().unwrap();
    blob_builder.push_bytes(b"batch-1-row-2").unwrap();
    blob_builder.push_bytes(b"batch-1-row-3").unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(
        reader,
        &test_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 3);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            batch_size: Some(2),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    blob_values.sort_by_key(|(id, _)| *id);
    assert_eq!(
        blob_values,
        vec![
            (0, Some(b"batch-0-row-0".to_vec())),
            (1, Some(b"batch-0-row-1".to_vec())),
            (2, Some(b"batch-1-row-0".to_vec())),
            (3, None),
            (4, Some(b"batch-1-row-2".to_vec())),
            (5, Some(b"batch-1-row-3".to_vec()))
        ]
    );
}
