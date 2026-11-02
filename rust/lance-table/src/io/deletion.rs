// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc, sync::LazyLock};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_ipc::CompressionType;
use arrow_ipc::reader::FileReader as ArrowFileReader;
use arrow_ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use bytes::Buf;
use lance_core::error::{CorruptFileSnafu, box_error};
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tracing::{AUDIT_MODE_CREATE, AUDIT_TYPE_DELETION, TRACE_FILE_AUDIT};
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::Rng;
use roaring::bitmap::RoaringBitmap;
use snafu::ResultExt;
use tracing::{info, instrument};

use crate::format::{DeletionFile, DeletionFileType};

pub const DELETIONS_DIR: &str = "_deletions";

/// The Arrow schema for an Arrow deletion file.
static DELETION_ARROW_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![Field::new(
        "row_id",
        DataType::UInt32,
        false,
    )]))
});

/// Get the file path for a deletion file. This is relative to the dataset root.
pub fn deletion_file_path(base: &Path, fragment_id: u64, deletion_file: &DeletionFile) -> Path {
    let DeletionFile {
        read_version,
        id,
        file_type,
        ..
    } = deletion_file;
    let suffix = file_type.suffix();
    base.clone()
        .join(DELETIONS_DIR)
        .join(format!("{fragment_id}-{read_version}-{id}.{suffix}"))
}

pub fn relative_deletion_file_path(fragment_id: u64, deletion_file: &DeletionFile) -> String {
    let DeletionFile {
        read_version,
        id,
        file_type,
        ..
    } = deletion_file;
    let suffix = file_type.suffix();
    format!("{DELETIONS_DIR}/{fragment_id}-{read_version}-{id}.{suffix}")
}

/// Write a deletion file for a fragment for a given deletion vector.
///
/// Returns the deletion metadata when the vector was written or stored inline.
/// Returns `Ok(None)` when no deletions were present.
///
/// A deletion vector whose bitmap encoding is at most `inline_max_bytes` is
/// returned inline (see [`DeletionFile::inline`]) and no file is written; `0`
/// always writes a file.
pub async fn write_deletion_file(
    base: &Path,
    fragment_id: u64,
    read_version: u64,
    removed_rows: &DeletionVector,
    object_store: &ObjectStore,
    inline_max_bytes: usize,
) -> Result<Option<DeletionFile>> {
    if inline_max_bytes > 0 && !removed_rows.is_empty() {
        let bitmap = RoaringBitmap::from(removed_rows);
        let encoded_size = bitmap.serialized_size();
        if encoded_size <= inline_max_bytes {
            let mut encoded = Vec::with_capacity(encoded_size);
            bitmap.serialize_into(&mut encoded)?;
            return Ok(Some(DeletionFile {
                read_version,
                id: rand::rng().random::<u64>(),
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(bitmap.len() as usize),
                base_id: None,
                inline: Some(encoded.into()),
            }));
        }
    }
    let deletion_file = match removed_rows {
        DeletionVector::NoDeletions => None,
        DeletionVector::Set(set) => {
            let id = rand::rng().random::<u64>();
            let deletion_file = DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Array,
                num_deleted_rows: Some(set.len()),
                base_id: None,
                inline: None,
            };
            let path = deletion_file_path(base, fragment_id, &deletion_file);

            let array = UInt32Array::from_iter(set.iter().copied());
            let array = Arc::new(array);

            let schema = DELETION_ARROW_SCHEMA.clone();
            let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

            let mut out: Vec<u8> = Vec::new();
            let write_options =
                IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?;
            {
                let mut writer = ArrowFileWriter::try_new_with_options(
                    &mut out,
                    schema.as_ref(),
                    write_options,
                )?;
                writer.write(&batch)?;
                writer.finish()?;
                // Drop writer so out is no longer borrowed.
            }

            object_store.put(&path, &out).await?;

            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DELETION, path = path.to_string());

            Some(deletion_file)
        }
        DeletionVector::Bitmap(bitmap) => {
            let id = rand::rng().random::<u64>();
            let deletion_file = DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(bitmap.len() as usize),
                base_id: None,
                inline: None,
            };
            let path = deletion_file_path(base, fragment_id, &deletion_file);

            let mut out: Vec<u8> = Vec::new();
            bitmap.serialize_into(&mut out)?;

            object_store.put(&path, &out).await?;

            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DELETION, path = path.to_string());

            Some(deletion_file)
        }
    };
    Ok(deletion_file)
}

#[instrument(
    level = "debug",
    skip(base, object_store),
    fields(
        base = base.as_ref(),
        bytes_read = tracing::field::Empty
    )
)]
pub async fn read_deletion_file(
    fragment_id: u64,
    deletion_file: &DeletionFile,
    base: &Path,
    object_store: &ObjectStore,
) -> Result<DeletionVector> {
    let span = tracing::Span::current();
    if let Some(encoded) = &deletion_file.inline {
        let bitmap = RoaringBitmap::deserialize_from(&encoded[..]).map_err(|error| {
            Error::corrupt_file(
                Path::from("manifest"),
                format!(
                    "failed to deserialize the inline deletion vector for fragment {fragment_id}: {error}"
                ),
            )
        })?;
        return Ok(DeletionVector::Bitmap(bitmap));
    }
    match deletion_file.file_type {
        DeletionFileType::Array => {
            let path = deletion_file_path(base, fragment_id, deletion_file);

            let data = object_store.read_one_all(&path).await?;
            span.record("bytes_read", data.len());
            let data = std::io::Cursor::new(data);
            let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(data, None)?
                .collect::<std::result::Result<_, ArrowError>>()
                .map_err(box_error)
                .context(CorruptFileSnafu { path: path.clone() })?;

            if batches.len() != 1 {
                return Err(Error::corrupt_file(
                    path,
                    format!(
                        "Expected exactly one batch in deletion file, got {}",
                        batches.len()
                    ),
                ));
            }

            let batch = batches.pop().unwrap();
            if batch.schema().as_ref() != DELETION_ARROW_SCHEMA.as_ref() {
                return Err(Error::corrupt_file(
                    path,
                    format!(
                        "Expected schema {:?} in deletion file, got {:?}",
                        DELETION_ARROW_SCHEMA.as_ref(),
                        batch.schema()
                    ),
                ));
            }

            let array = batch.columns()[0]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();

            let mut set = HashSet::with_capacity(array.len());
            for val in array.iter() {
                if let Some(val) = val {
                    set.insert(val);
                } else {
                    return Err(Error::corrupt_file(
                        path,
                        "Null values are not allowed in deletion files",
                    ));
                }
            }

            Ok(DeletionVector::Set(set))
        }
        DeletionFileType::Bitmap => {
            let path = deletion_file_path(base, fragment_id, deletion_file);

            let data = object_store.read_one_all(&path).await?;
            span.record("bytes_read", data.len());
            let reader = data.reader();
            let bitmap = RoaringBitmap::deserialize_from(reader)
                .map_err(box_error)
                .context(CorruptFileSnafu { path })?;

            Ok(DeletionVector::Bitmap(bitmap))
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use object_store::ObjectStoreExt;

    #[tokio::test]
    async fn test_write_no_deletions() {
        let dv = DeletionVector::NoDeletions;

        let (object_store, path) = ObjectStore::from_uri("memory:///no_deletion")
            .await
            .unwrap();
        let file = write_deletion_file(&path, 0, 0, &dv, &object_store, 0)
            .await
            .unwrap();
        assert!(file.is_none());
    }

    /// Under the inline budget the deletion vector lives in the returned
    /// metadata and nothing is written; over it a file is written as before.
    #[tokio::test]
    async fn test_write_inline_within_budget() {
        let dv = DeletionVector::Set(HashSet::from_iter([3, 7, 1000]));
        let object_store = ObjectStore::memory();
        let path = Path::from("/inline");

        let file = write_deletion_file(&path, 21, 12, &dv, &object_store, 1024)
            .await
            .unwrap()
            .unwrap();
        assert!(file.is_inline());
        assert_eq!(file.file_type, DeletionFileType::Bitmap);
        assert_eq!(file.num_deleted_rows, Some(3));
        let would_be_path = deletion_file_path(&path, 21, &file);
        assert!(!object_store.exists(&would_be_path).await.unwrap());
        let read_back = read_deletion_file(21, &file, &path, &object_store)
            .await
            .unwrap();
        assert_eq!(
            read_back.into_sorted_iter().collect::<Vec<_>>(),
            vec![3, 7, 1000]
        );

        let corrupt = DeletionFile {
            inline: Some(bytes::Bytes::from_static(b"not a roaring bitmap")),
            ..file.clone()
        };
        let error = read_deletion_file(21, &corrupt, &path, &object_store)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("inline deletion vector for fragment 21"),
            "{error}"
        );

        let file = write_deletion_file(&path, 21, 12, &dv, &object_store, 8)
            .await
            .unwrap()
            .unwrap();
        assert!(!file.is_inline());
        let written_path = deletion_file_path(&path, 21, &file);
        assert!(object_store.exists(&written_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_write_array() {
        let dv = DeletionVector::Set(HashSet::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/write");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store, 0)
            .await
            .unwrap();

        assert!(matches!(
            file,
            Some(DeletionFile {
                file_type: DeletionFileType::Array,
                ..
            })
        ));

        let file = file.unwrap();
        assert_eq!(file.read_version, read_version);
        let path = deletion_file_path(&path, fragment_id, &file);
        assert_eq!(
            path,
            Path::from(format!("/write/_deletions/21-12-{}.arrow", file.id))
        );

        let data = object_store
            .inner
            .get(&path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let data = std::io::Cursor::new(data);
        let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(data, None)
            .unwrap()
            .collect::<std::result::Result<_, ArrowError>>()
            .unwrap();

        assert_eq!(batches.len(), 1);
        let batch = batches.pop().unwrap();
        assert_eq!(batch.schema(), *DELETION_ARROW_SCHEMA);
        let array = batch["row_id"]
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let read_dv = DeletionVector::from_iter(array.iter().map(|v| v.unwrap()));
        assert_eq!(read_dv, dv);
    }

    #[tokio::test]
    async fn test_write_bitmap() {
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/bitmap");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store, 0)
            .await
            .unwrap();

        assert!(matches!(
            file,
            Some(DeletionFile {
                file_type: DeletionFileType::Bitmap,
                ..
            })
        ));

        let file = file.unwrap();
        assert_eq!(file.read_version, read_version);
        let path = deletion_file_path(&path, fragment_id, &file);
        assert_eq!(
            path,
            Path::from(format!("/bitmap/_deletions/21-12-{}.bin", file.id))
        );

        let data = object_store
            .inner
            .get(&path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let reader = data.reader();
        let read_bitmap = RoaringBitmap::deserialize_from(reader).unwrap();
        assert_eq!(read_bitmap, dv.into_iter().collect::<RoaringBitmap>());
    }

    #[tokio::test]
    async fn test_roundtrip_array() {
        let dv = DeletionVector::Set(HashSet::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/roundtrip");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store, 0)
            .await
            .unwrap();

        let read_dv = read_deletion_file(fragment_id, &file.unwrap(), &path, &object_store)
            .await
            .unwrap();
        assert_eq!(read_dv, dv);
    }

    #[tokio::test]
    async fn test_roundtrip_bitmap() {
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/bitmap");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store, 0)
            .await
            .unwrap();

        let read_dv = read_deletion_file(fragment_id, &file.unwrap(), &path, &object_store)
            .await
            .unwrap();
        assert_eq!(read_dv, dv);
    }
}
