// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::test_util::managed_blob_schema;
use crate::{
    Dataset,
    blob::{BlobArrayBuilder, BlobDescriptorArrayBuilder, ManagedBlobWriter, blob_field},
    dataset::{BlobRangeRequest, WriteParams, optimize::compact_files},
};
use arrow_array::{LargeBinaryArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::{
    BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_META_KEY,
};
use lance_core::datatypes::BlobKind;
use lance_core::utils::tempfile::TempStrDir;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::BasePath;
use object_store::path::Path;
use rstest::rstest;
use std::collections::HashMap;
use std::sync::Arc;

#[rstest]
#[tokio::test]
async fn prepared_managed_sharing_reuses_physical_ranges(
    #[values(None, Some(7), Some(u32::MAX))] base_id: Option<u32>,
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    let dir = TempStrDir::default();
    let root = Path::from_absolute_path(dir.to_string()).unwrap();
    let mut writer = ManagedBlobWriter::try_new(ObjectStore::local(), root, base_id)
        .await
        .unwrap();
    let payload_path = writer.path().clone();
    writer
        .write_blobs([
            Some(b"shared".as_slice()),
            Some(b"".as_slice()),
            None,
            Some(b"unused".as_slice()),
        ])
        .await
        .unwrap();
    let values = writer.finish().await.unwrap();
    let mut builder = BlobDescriptorArrayBuilder::new("blob");
    builder
        .extend([0, 0, 1, 2, 0].map(|i| values[i].clone()))
        .unwrap();
    let (field, array) = builder.finish().unwrap().into_parts();
    let schema = Arc::new(managed_blob_schema(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            data_storage_version: Some(version),
            max_rows_per_file: 2,
            initial_bases: base_id.map(|id| vec![BasePath::new(id, dir.to_string(), None, false)]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    for compacted in [false, true] {
        if compacted {
            let payload_store = dataset.object_store(base_id).await.unwrap();
            let _ = payload_store.io_stats_incremental();
            compact_files(&mut dataset, Default::default(), None)
                .await
                .unwrap();
            assert_eq!(dataset.get_fragments().len(), 1);
            let stats = payload_store.io_stats_incremental();
            assert!(
                stats.requests.iter().all(|request| {
                    !request.path.as_ref().contains("/_blobs/")
                        || !request.method.starts_with("put")
                }),
                "{stats:?}"
            );
        }
        let dataset = Arc::new(dataset.clone());
        let blobs = dataset
            .take_blobs_by_indices(&[0, 1, 2, 3, 4], "blob")
            .await
            .unwrap();
        assert!(blobs[3].is_none());
        assert!(blobs[2].as_ref().unwrap().read().await.unwrap().is_empty());
        for i in [0, 1, 4] {
            assert_eq!(blobs[i].as_ref().unwrap().data_path(), &payload_path);
            assert_eq!(
                blobs[i].as_ref().unwrap().read().await.unwrap(),
                b"shared"[..]
            );
        }
        // Descriptor lookup is warm, but payload reads are submitted through
        // the real dataset planner. Duplicate and overlapping selections
        // must share the same physical range in this single read plan.
        let payload_store = dataset.object_store(base_id).await.unwrap();
        let _ = payload_store.io_stats_incremental();
        let results = dataset
            .read_blob_ranges("blob")
            .unwrap()
            .with_row_indices(vec![
                BlobRangeRequest::new(4, 1, 3),
                BlobRangeRequest::new(0, 1, 3),
                BlobRangeRequest::new(1, 2, 2),
            ])
            .execute()
            .await
            .unwrap();
        let stats = payload_store.io_stats_incremental();
        assert_eq!(results[0].data.as_deref(), Some(b"har".as_slice()));
        assert_eq!(results[1].data.as_deref(), Some(b"har".as_slice()));
        assert_eq!(results[2].data.as_deref(), Some(b"ar".as_slice()));
        let requests = stats
            .requests
            .iter()
            .filter(|request| request.path == payload_path)
            .collect::<Vec<_>>();
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert_eq!(requests[0].range, Some(1..4));
        let objects = std::fs::read_dir(std::path::Path::new(dir.as_str()).join("_blobs"))
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(objects.len(), 1);
        assert_eq!(
            std::fs::read(objects[0].as_ref().unwrap().path()).unwrap(),
            b"sharedunused"
        );
    }
}

#[tokio::test]
async fn test_blob_v1_files_read_source_base_after_shallow_clone() {
    let source_dir = TempStrDir::default();
    let clone_dir = TempStrDir::default();
    let expected = b"known independent legacy clone payload\x00\xff";
    let blob_metadata = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("blob", DataType::LargeBinary, true).with_metadata(blob_metadata),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![0, 1, 2])),
            Arc::new(LargeBinaryArray::from_iter([
                Some(b"".as_slice()),
                None,
                Some(expected.as_slice()),
            ])),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &source_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut source = Dataset::open(source_dir.as_ref()).await.unwrap();
    assert_eq!(source.get_fragments().len(), 3);
    let source_blobs = Arc::new(source.clone())
        .take_blobs_by_indices(&[2, 1, 0], "blob")
        .await
        .unwrap();
    assert_eq!(
        source_blobs[0]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .as_ref(),
        expected
    );
    assert!(source_blobs[1].is_none());
    assert!(
        source_blobs[2]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .is_empty()
    );

    source
        .shallow_clone(clone_dir.as_ref(), source.version_id(), None)
        .await
        .unwrap();
    let cloned = Arc::new(Dataset::open(clone_dir.as_ref()).await.unwrap());
    assert_eq!(cloned.get_fragments().len(), 3);
    let data_file = cloned.get_fragments()[0].metadata().files[0].clone();
    assert!(
        data_file.base_id.is_some(),
        "shallow clone data file must reference the source base"
    );
    assert!(
        !std::path::Path::new(clone_dir.as_ref())
            .join("data")
            .join(&data_file.path)
            .exists(),
        "shallow clone publication must not copy the legacy blob data file"
    );

    let cloned_blobs = cloned
        .take_blobs_by_indices(&[2, 1, 0], "blob")
        .await
        .unwrap();
    assert_eq!(
        cloned_blobs[0]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .as_ref(),
        expected
    );
    assert!(cloned_blobs[1].is_none());
    assert!(
        cloned_blobs[2]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .is_empty()
    );
}

#[rstest]
#[case::inline(BlobKind::Inline, 1024 * 1024, 1024 * 1024)]
#[case::packed(BlobKind::Packed, 0, 1024 * 1024)]
#[case::dedicated(BlobKind::Dedicated, 0, 1)]
#[tokio::test]
async fn test_read_blob_ranges_avoids_whole_value_read_amplification_end_to_end(
    #[case] kind: BlobKind,
    #[case] inline_threshold: usize,
    #[case] dedicated_threshold: usize,
) {
    const VALUE_SIZE: usize = 256 * 1024;
    const WINDOW_SIZE: u64 = 4 * 1024;
    const OFFSET: u64 = 64 * 1024;

    let test_dir = TempStrDir::default();
    let mut blob_builder = BlobArrayBuilder::new(4);
    for value in 0xA5..0xA9 {
        blob_builder.push_bytes(vec![value; VALUE_SIZE]).unwrap();
    }
    let mut blob_field = blob_field("blob", false);
    let mut metadata = blob_field.metadata().clone();
    metadata.insert(
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
        dedicated_threshold.to_string(),
    );
    metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        inline_threshold.to_string(),
    );
    blob_field = blob_field.with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![blob_field]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(blob_builder.finish().unwrap())],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    assert_eq!(dataset.get_fragments().len(), 2);
    let indices = [3, 0, 2, 1];
    let blobs = dataset
        .take_blobs_by_indices(&indices, "blob")
        .await
        .unwrap();
    for blob in &blobs {
        assert_eq!(blob.as_ref().unwrap().kind(), kind);
    }
    let _ = dataset.table_state.object_store.io_stats_incremental();
    let results = dataset
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_indices(indices.map(|row| BlobRangeRequest::new(row, OFFSET, WINDOW_SIZE)))
        .execute()
        .await
        .unwrap();
    let stats = dataset.table_state.object_store.io_stats_incremental();

    assert_eq!(results.len(), indices.len());
    for ((result, row), blob) in results.iter().zip(indices).zip(blobs) {
        let data = result.data.as_ref().unwrap();
        assert_eq!(data.as_ref(), vec![0xA5 + row as u8; WINDOW_SIZE as usize]);
        let blob = blob.unwrap();
        let range = blob.position() + OFFSET..blob.position() + OFFSET + WINDOW_SIZE;
        assert!(
            stats.requests.iter().any(|request| {
                &request.path == blob.data_path() && request.range.as_ref() == Some(&range)
            }),
            "{stats:?}"
        );
        assert!(
            stats
                .requests
                .iter()
                .filter(|request| &request.path == blob.data_path())
                .all(|request| {
                    request.method != "head"
                        && (request.method != "get_opts" || request.range.is_some())
                }),
            "{stats:?}"
        );
    }
}
