// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
mod tests {
    use crate::dataset::blob::test_util::{
        complete_blob_v2_array, complete_blob_v2_field, nested_blob_v2_batch,
        nested_blob_v2_batch_with_field,
    };
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, blob_field},
        dataset::{ExternalBlobMode, WriteMode, WriteParams, scanner::MaterializationStyle},
    };
    use arrow::{array::AsArray, datatypes::UInt8Type};
    use arrow_array::RecordBatch;
    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatchIterator};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY};
    use lance_core::datatypes::{BlobHandling, BlobKind};
    use lance_core::{
        Error, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION,
        utils::tempfile::{TempDir, TempStrDir},
    };
    use lance_file::version::LanceFileVersion;
    use lance_table_runtime::blob::thresholds::{DEDICATED_THRESHOLD, INLINE_MAX};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    #[tokio::test]
    async fn test_write_and_take_nested_blob_v2(
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let packed_payload = vec![0x4A; INLINE_MAX + 1024];

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_bytes(&packed_payload).unwrap();
        blob_builder.push_null().unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let (schema, batch) = nested_blob_v2_batch(blob_array);
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let info_batch = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let blob_desc = info_batch
            .column(0)
            .as_struct()
            .column_by_name("blob")
            .unwrap()
            .as_struct();
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Inline as u8
        );
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(1),
            BlobKind::Packed as u8
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "info.blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2);
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"hello"
        );
        assert_eq!(
            blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
            packed_payload.as_slice()
        );

        let null_blobs = dataset
            .take_blobs_by_indices(&[2], "info.blob")
            .await
            .unwrap();
        assert_eq!(null_blobs.len(), 1);
        assert!(null_blobs[0].is_none());

        let filtered = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .filter("info.blob IS NOT NULL")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_and_take_nested_complete_blob_v2(
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let packed_payload = vec![0x4A; INLINE_MAX + 1024];

        let blob_array = complete_blob_v2_array(
            vec![Some(b"hello".to_vec()), Some(packed_payload.clone()), None],
            vec![None, None, None],
            vec![None, None, None],
            vec![None, None, None],
            Some(NullBuffer::from(vec![true, true, false])),
        );

        let (schema, batch) =
            nested_blob_v2_batch_with_field(complete_blob_v2_field("blob", true), blob_array);
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let info_batch = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let blob_desc = info_batch
            .column(0)
            .as_struct()
            .column_by_name("blob")
            .unwrap()
            .as_struct();
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Inline as u8
        );
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(1),
            BlobKind::Packed as u8
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "info.blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2);
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"hello"
        );
        assert_eq!(
            blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
            packed_payload.as_slice()
        );

        let null_blobs = dataset
            .take_blobs_by_indices(&[2], "info.blob")
            .await
            .unwrap();
        assert_eq!(null_blobs.len(), 1);
        assert!(null_blobs[0].is_none());

        let filtered = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .filter("info.blob IS NOT NULL")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_and_scan_list_blob_v2_descriptions(
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let packed_payload = vec![0x4B; INLINE_MAX + 1024];

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_null().unwrap();
        blob_builder.push_bytes(&packed_payload).unwrap();
        blob_builder.push_bytes(b"tail").unwrap();
        let blob_values = blob_builder.finish().unwrap();

        let item_field = Arc::new(blob_field("item", true));
        let list_array: ArrayRef = Arc::new(
            arrow_array::ListArray::try_new(
                item_field.clone(),
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![
                    0i32, 3, 3, 3, 4,
                ])),
                blob_values,
                Some(arrow_buffer::NullBuffer::from(vec![
                    true, true, false, true,
                ])),
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("blobs", DataType::List(item_field), true),
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![list_array, Arc::new(Int32Array::from(vec![0, 1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let descriptions = dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let lists = descriptions.column(0).as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 3, 3, 3, 4]);
        assert!(lists.is_valid(0));
        assert!(lists.is_valid(1));
        assert!(lists.is_null(2));
        assert!(lists.is_valid(3));

        let DataType::List(descriptor_field) = lists.data_type() else {
            panic!("unexpected list type: {}", lists.data_type());
        };
        assert!(matches!(descriptor_field.data_type(), DataType::Struct(_)));
        assert!(!descriptor_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let descriptors = lists.values().as_struct();
        assert_eq!(descriptors.fields().len(), 5);
        assert_eq!(descriptors.fields()[0].name(), "kind");
        assert!(descriptors.is_valid(0));
        assert!(descriptors.is_null(1));
        assert!(descriptors.is_valid(2));
        assert!(descriptors.is_valid(3));
        let kinds = descriptors
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();
        assert_eq!(kinds.value(0), BlobKind::Inline as u8);
        assert_eq!(kinds.value(2), BlobKind::Packed as u8);
        assert_eq!(kinds.value(3), BlobKind::Inline as u8);

        let filtered = dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .filter("blobs IS NOT NULL")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 3);

        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let bytes = scanner
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let lists = bytes.column(0).as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 3, 3, 3, 4]);
        assert!(lists.is_valid(0));
        assert!(lists.is_valid(1));
        assert!(lists.is_null(2));
        assert!(lists.is_valid(3));
        let DataType::List(value_field) = lists.data_type() else {
            panic!("unexpected list type: {}", lists.data_type());
        };
        assert_eq!(value_field.data_type(), &DataType::LargeBinary);
        assert!(!value_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let values = lists.values().as_binary::<i64>();
        assert_eq!(values.value(0), b"hello");
        assert!(values.is_null(1));
        assert_eq!(values.value(2), packed_payload.as_slice());
        assert_eq!(values.value(3), b"tail");

        for (filter, materialization_style) in [
            (None, MaterializationStyle::Heuristic),
            (Some("id >= 2"), MaterializationStyle::Heuristic),
            (Some("id >= 2"), MaterializationStyle::AllEarly),
        ] {
            let mut scanner = dataset.scan();
            scanner.blob_handling(BlobHandling::AllBinary);
            scanner.materialization_style(materialization_style);
            scanner
                .project(&["blobs", ROW_LAST_UPDATED_AT_VERSION, ROW_CREATED_AT_VERSION])
                .unwrap()
                .with_row_id()
                .with_row_address();
            if let Some(filter) = filter {
                scanner.filter(filter).unwrap();
            }

            let expected_schema = scanner.schema().await.unwrap();
            let batch = scanner.try_into_batch().await.unwrap();
            assert_eq!(batch.schema().as_ref(), expected_schema.as_ref());
            assert_eq!(batch.num_rows(), if filter.is_some() { 2 } else { 4 });
            for column in [
                ROW_ID,
                ROW_ADDR,
                ROW_LAST_UPDATED_AT_VERSION,
                ROW_CREATED_AT_VERSION,
            ] {
                assert!(
                    batch.column_by_name(column).is_some(),
                    "requested system column {column} was missing"
                );
            }
        }
    }

    #[rstest]
    #[case::both_small(Some(5), true)]
    #[case::both_packed(Some(INLINE_MAX + 1), true)]
    #[case::neither(None, false)]
    #[tokio::test]
    async fn test_complete_blob_v2_rejects_invalid_representation(
        #[case] data_size: Option<usize>,
        #[case] has_uri: bool,
    ) {
        let dataset_dir = TempDir::default();
        let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![complete_blob_v2_array(
                vec![data_size.map(|size| vec![0x41; size])],
                vec![has_uri.then(|| "file:///source.bin".to_string())],
                vec![None],
                vec![None],
                None,
            )],
        )
        .unwrap();

        let error = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                allow_external_blob_outside_bases: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("must set exactly one of `data` and `uri`")
        );
    }

    #[tokio::test]
    async fn test_blob_v2_external_ingest_packed() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        let payload = vec![0x5A; INLINE_MAX + 1024];
        std::fs::write(&external_path, &payload).unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    external_blob_mode: ExternalBlobMode::Ingest,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Packed as u8
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        let blob = blobs[0].as_ref().unwrap();
        assert_eq!(blob.kind(), BlobKind::Packed);
        assert_eq!(blob.read().await.unwrap().as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn test_blob_v2_external_ingest_dedicated() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        let payload = vec![0x7A; DEDICATED_THRESHOLD + 1024];
        std::fs::write(&external_path, &payload).unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    external_blob_mode: ExternalBlobMode::Ingest,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Dedicated as u8
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        let blob = blobs[0].as_ref().unwrap();
        assert_eq!(blob.kind(), BlobKind::Dedicated);
        assert_eq!(blob.read().await.unwrap().as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn test_blob_v2_append_accepts_explicit_default_inline_threshold() {
        let dataset_dir = TempDir::default();
        let payload = vec![0u8; 2048];

        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(payload.clone()).unwrap();
        let initial_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
        let dataset = Dataset::write(
            initial_reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut append_field = blob_field("blob", true);
        let mut append_metadata = append_field.metadata().clone();
        append_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            INLINE_MAX.to_string(),
        );
        append_field = append_field.with_metadata(append_metadata);
        let append_schema = Arc::new(Schema::new(vec![append_field]));
        let mut append_builder = BlobArrayBuilder::new(1);
        append_builder.push_bytes(payload).unwrap();
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

        let dataset = Dataset::write(
            append_reader,
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    }
}
