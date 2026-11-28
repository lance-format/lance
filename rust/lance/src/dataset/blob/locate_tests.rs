// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{StringArray, StructArray, UInt8Array, UInt32Array, UInt64Array};
use arrow_buffer::NullBuffer;
use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BlobKind};
use lance_core::utils::blob::blob_path;
use lance_core::utils::tempfile::TempStdDir;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

use super::locate::{BlobLocator, StoredBlob, data_file_key_from_path};
use super::test_util::create_multi_base_blob_v2_fixture;
use crate::Dataset;
use lance_table_runtime::blob::thresholds::INLINE_MAX;

mod policies;

fn descriptor(
    kind: Option<BlobKind>,
    position: u64,
    size: u64,
    blob_id: u32,
    uri: &str,
    valid: bool,
) -> StructArray {
    // Older descriptor schemas mark null blobs with a nullable `kind` child.
    let fields = BLOB_V2_DESC_FIELDS
        .iter()
        .map(|field| field.as_ref().clone().with_nullable(true))
        .collect::<Vec<_>>();
    StructArray::try_new(
        fields.into(),
        vec![
            Arc::new(UInt8Array::from(vec![kind.map(|kind| kind as u8)])),
            Arc::new(UInt64Array::from(vec![position])),
            Arc::new(UInt64Array::from(vec![size])),
            Arc::new(UInt32Array::from(vec![blob_id])),
            Arc::new(StringArray::from(vec![uri])),
        ],
        Some(NullBuffer::from(vec![valid])),
    )
    .unwrap()
}

#[tokio::test]
async fn test_blob_locator_locates_each_kind() {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    let dataset = &fixture.dataset;
    let field_id = dataset.schema().field("blob").unwrap().id as u32;
    let data_file = dataset.get_fragment(0).unwrap().metadata().files[0].clone();
    let data_dir = dataset.data_file_dir(&data_file).unwrap();
    let base_root = dataset.table_view().blob_base_path(Some(1)).unwrap();
    let sidecar = blob_path(&data_dir, data_file_key_from_path(&data_file.path), 7);
    let external_dir = TempStdDir::default();
    let external_file = external_dir.join("payload.bin");
    std::fs::write(&external_file, b"abcd").unwrap();
    let external_uri = format!("file://{}", external_file.display());
    let data_file_path = data_dir.clone().join(data_file.path.as_str());
    let row_addr = 0;

    let cases = [
        (
            StoredBlob::Empty,
            BlobKind::Inline,
            data_file_path.clone(),
            0,
            0,
            None,
        ),
        (
            StoredBlob::Inline {
                position: 3,
                size: 5,
            },
            BlobKind::Inline,
            data_file_path.clone(),
            3,
            5,
            None,
        ),
        (
            StoredBlob::Packed {
                blob_id: 7,
                position: 3,
                size: 5,
            },
            BlobKind::Packed,
            sidecar.clone(),
            3,
            5,
            None,
        ),
        (
            StoredBlob::Dedicated {
                blob_id: 7,
                size: 5,
            },
            BlobKind::Dedicated,
            sidecar.clone(),
            0,
            5,
            None,
        ),
        (
            StoredBlob::External {
                base_id: 1,
                uri: "a/b.bin",
                position: 3,
                size: 5,
            },
            BlobKind::External,
            base_root.clone().join("a").join("b.bin"),
            3,
            5,
            Some("a/b.bin"),
        ),
        (
            StoredBlob::External {
                base_id: 0,
                uri: &external_uri,
                position: 1,
                size: 0,
            },
            BlobKind::External,
            Path::from_absolute_path(&external_file).unwrap(),
            1,
            4,
            Some(external_uri.as_str()),
        ),
        (
            StoredBlob::Managed {
                base_id: None,
                uri: "_blobs/a.blob",
                position: 3,
                size: 5,
            },
            BlobKind::Managed,
            base_root.clone().join("_blobs").join("a.blob"),
            3,
            5,
            Some("_blobs/a.blob"),
        ),
        (
            StoredBlob::Managed {
                base_id: Some(1),
                uri: "_blobs/a.blob",
                position: 3,
                size: 5,
            },
            BlobKind::Managed,
            base_root.clone().join("_blobs").join("a.blob"),
            3,
            5,
            Some("_blobs/a.blob"),
        ),
    ];
    let table = dataset.table_view().to_table();
    let mut locator = BlobLocator::new(&table, field_id);
    for (blob, kind, path, position, size, uri) in cases {
        let file = locator.locate(&blob, row_addr).await.unwrap();
        assert_eq!(file.kind(), kind, "{blob:?}");
        assert_eq!(file.data_path(), &path, "{blob:?}");
        assert_eq!(file.position(), position, "{blob:?}");
        assert_eq!(file.size(), size, "{blob:?}");
        assert_eq!(file.uri(), uri, "{blob:?}");
    }

    assert_eq!(locator.managed_base(None, row_addr).unwrap(), Some(1));
    assert_eq!(locator.managed_base(Some(2), row_addr).unwrap(), Some(2));
    let (sidecar_base, relative) = locator.sidecar(row_addr, 7).unwrap();
    assert_eq!(sidecar_base, Some(1));
    assert_eq!(
        Path::from_iter(base_root.parts().chain(relative.parts())),
        sidecar
    );
}

#[tokio::test]
async fn test_blob_locator_rejects_unknown_base() {
    let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
    let field_id = fixture.dataset.schema().field("blob").unwrap().id as u32;
    let table = fixture.dataset.table_view().to_table();
    let mut locator = BlobLocator::new(&table, field_id);
    for blob in [
        StoredBlob::External {
            base_id: 9,
            uri: "a.bin",
            position: 0,
            size: 1,
        },
        StoredBlob::Managed {
            base_id: Some(9),
            uri: "a.blob",
            position: 0,
            size: 1,
        },
    ] {
        let error = locator.locate(&blob, 0).await.unwrap_err();
        assert!(error.to_string().contains("unknown base_id 9"), "{error}");
    }
}

#[tokio::test]
async fn test_take_blob_v2_from_non_default_base_packed() {
    let fixture =
        create_multi_base_blob_v2_fixture(vec![0x5A; INLINE_MAX + 4096], None, true).await;

    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&[0], "blob")
        .await
        .unwrap();

    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Packed);
    assert_eq!(blob.base_id(), Some(1));
    assert_eq!(
        blob.read().await.unwrap().as_ref(),
        fixture.expected.as_slice()
    );
}

#[tokio::test]
async fn test_take_blob_v2_from_data_only_base() {
    let fixture =
        create_multi_base_blob_v2_fixture(vec![0x6B; INLINE_MAX + 2048], None, false).await;

    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&[0], "blob")
        .await
        .unwrap();

    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Packed);
    assert_eq!(
        blob.read().await.unwrap().as_ref(),
        fixture.expected.as_slice()
    );
}
