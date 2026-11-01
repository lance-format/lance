// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Buf;
use futures::TryStreamExt;
use lance_file::{
    version::ConcreteFileVersion,
    versions::v1::{
        encoding::write_schema_dictionaries, writer::ManifestProvider as V1ManifestProvider,
    },
};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use std::collections::HashMap;
use std::{ops::Range, sync::Arc};
use tracing::instrument;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::{
    object_store::ObjectStore,
    traits::{WriteExt, Writer},
    utils::{ChunkedBuf, METADATA_READ_CHUNK_SIZE, read_message, read_range_in_chunks},
};

use crate::format::{DataStorageFormat, IndexMetadata, MAGIC, Manifest, Transaction, pb};

use super::commit::ManifestLocation;

/// Read Manifest on URI.
///
/// This only reads manifest files. It does not read data files.
#[instrument(level = "debug", skip(object_store))]
pub async fn read_manifest(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<Manifest> {
    let buf = read_manifest_bytes(object_store, path, known_size).await?;
    let message_len = buf.remaining();
    let proto = pb::Manifest::decode(buf)?;
    let mut manifest = Manifest::try_from(proto)?;
    manifest.detach_sparse_inline_row_ids(message_len);
    Ok(manifest)
}

/// Read the original manifest protobuf bytes, excluding the length prefix and footer.
///
/// A known file size avoids a metadata request. Small messages reuse the prefetched
/// tail; larger messages remain in chunks so decoding does not require a full copy.
///
/// ```
/// # use lance_core::Result;
/// # use lance_io::object_store::ObjectStore;
/// # use object_store::path::Path;
/// # use prost::Message;
/// # async fn example(store: &ObjectStore, path: &Path) -> Result<()> {
/// let bytes = lance_table::io::manifest::read_manifest_bytes(store, path, None).await?;
/// let manifest = lance_table::format::pb::Manifest::decode(bytes)?;
/// # Ok(())
/// # }
/// ```
#[instrument(level = "debug", skip(object_store))]
pub async fn read_manifest_bytes(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<ChunkedBuf> {
    let file_size = if let Some(known_size) = known_size {
        known_size
    } else {
        object_store.inner.head(path).await?.size
    };
    if file_size < 16 {
        if known_size.is_some() {
            return Box::pin(read_manifest_bytes(object_store, path, None)).await;
        }
        return Err(Error::corrupt_file(
            path.clone(),
            format!("Invalid format: manifest is only {file_size} bytes (minimum 16)"),
        ));
    }
    const PREFETCH_SIZE: u64 = 64 * 1024;
    let initial_start = file_size.saturating_sub(PREFETCH_SIZE);
    let range = Range {
        start: initial_start,
        end: file_size,
    };
    let expected_tail_len = (file_size - initial_start) as usize;
    let buf = object_store.inner.get_range(path, range).await?;

    // A stale size can truncate the footer or make the backend clamp the tail
    // range while preserving its magic. Retry once using the actual file size.
    if known_size.is_some()
        && (buf.len() < expected_tail_len || buf.len() < 16 || !buf.ends_with(MAGIC))
    {
        return Box::pin(read_manifest_bytes(object_store, path, None)).await;
    }

    if buf.len() < 16 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: file size is smaller than 16 bytes".to_string(),
        ));
    }
    if !buf.ends_with(MAGIC) {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: magic number does not match".to_string(),
        ));
    }
    let manifest_pos = LittleEndian::read_u64(&buf[buf.len() - 16..buf.len() - 8]);
    let manifest_len = file_size
        .checked_sub(manifest_pos)
        .filter(|len| *len >= 20)
        .ok_or_else(|| {
            Error::corrupt_file(
                path.clone(),
                format!("Invalid manifest offset {manifest_pos} for file size {file_size}"),
            )
        })? as usize;

    let mut message = ChunkedBuf::default();
    if manifest_len <= buf.len() {
        // The prefetch captured the entire manifest. We just need to trim the buffer.
        message.push(buf.slice(buf.len() - manifest_len..));
    } else {
        // The prefetch only captured part of the manifest. Fetch the remainder
        // as concurrent chunked range requests: a single GET is limited to one
        // connection's throughput, which dominates load time for manifests of
        // datasets with many fragments.
        let reader = object_store
            .open_with_size(path, file_size as usize)
            .await?;
        let mut chunks = read_range_in_chunks(
            reader.as_ref(),
            manifest_pos as usize..(file_size - PREFETCH_SIZE) as usize,
            METADATA_READ_CHUNK_SIZE,
        );
        // Decode straight from the chunks rather than re-copying ~all of the
        // file into one buffer; see `ChunkedBuf`.
        while let Some(chunk) = chunks.try_next().await? {
            message.push(chunk);
        }
        message.push(buf);
    }
    // A u32 length prefix in front, the 16 byte footer behind.
    if message.remaining() < 20 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: manifest shorter than its length prefix and footer".to_string(),
        ));
    }
    let recorded_length = message.get_u32_le() as usize;
    message.truncate(message.remaining() - 16);
    if message.remaining() != recorded_length {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {}, got {}",
            recorded_length,
            message.remaining()
        )));
    }
    Ok(message)
}

#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_indexes(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Vec<IndexMetadata>> {
    if let Some(pos) = manifest.index_section.as_ref() {
        let result = read_index_section(object_store, &location.path, location.size, *pos).await;
        // A stale cached size makes the index offset fall outside the sized view,
        // so the read fails as "file size is too small". Retry once with the true
        // size; surface any other error unchanged.
        let section = match result {
            Err(e)
                if location.size.is_some() && e.to_string().contains("file size is too small") =>
            {
                read_index_section(object_store, &location.path, None, *pos).await?
            }
            other => other?,
        };

        let indices = section
            .indices
            .into_iter()
            .map(IndexMetadata::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(indices)
    } else {
        Ok(vec![])
    }
}

/// Read the index section message at `pos`, opening the manifest with a known
/// size when one is provided.
async fn read_index_section(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
    pos: usize,
) -> Result<pb::IndexSection> {
    let reader = if let Some(size) = size {
        object_store.open_with_size(path, size as usize).await?
    } else {
        object_store.open(path).await?
    };
    read_message(reader.as_ref(), pos).await
}

async fn do_write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    mut transaction: Option<Transaction>,
) -> Result<usize> {
    // Write indices if presented.
    if let Some(indices) = indices.as_ref() {
        let section = pb::IndexSection {
            indices: indices.iter().map(|i| i.into()).collect(),
        };
        let pos = writer.write_protobuf(&section).await?;
        manifest.index_section = Some(pos);
    } else {
        // No index section is written to this file, so an inherited offset
        // would point at unrelated bytes in the new manifest file.
        manifest.index_section = None;
    }

    // Write inline transaction if presented.
    if let Some(tx) = transaction.take() {
        // Convert to protobuf at the write boundary to persist inline
        let pb_tx: pb::Transaction = tx.into();
        let pos = writer.write_protobuf(&pb_tx).await?;
        manifest.transaction_section = Some(pos);
    } else {
        // No inline copy is written to this file. Clear any offset inherited
        // from a previous manifest (e.g. via restore or clone), which would
        // otherwise point at arbitrary bytes of the file being written.
        manifest.transaction_section = None;
    }

    writer.write_struct(manifest).await
}

/// Write manifest to an open file.
pub async fn write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<usize> {
    match manifest.data_storage_format.version {
        ConcreteFileVersion::V1 => {
            write_schema_dictionaries(writer, &mut manifest.schema).await?;
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {}
    }

    do_write_manifest(writer, manifest, indices, transaction).await
}

/// Implementation of ManifestProvider that describes a Lance file by writing
/// a manifest that contains nothing but default fields and the schema
pub struct ManifestDescribing {}

#[async_trait]
impl V1ManifestProvider for ManifestDescribing {
    async fn store_schema(
        object_writer: &mut dyn Writer,
        schema: &Schema,
    ) -> Result<Option<usize>> {
        let mut manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::new(ConcreteFileVersion::V1),
            HashMap::new(),
        );
        let pos = do_write_manifest(object_writer, &mut manifest, None, None).await?;
        Ok(Some(pos))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatch};
    use bytes::Bytes;
    use std::collections::HashMap;

    use crate::format::SelfDescribingFileReader;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
    use lance_file::versions::v1::{
        reader::FileReader as V1FileReader, writer::FileWriter as V1FileWriter,
    };
    use lance_io::object_store::WrappingObjectStore;
    use rand::{Rng, distr::Alphanumeric};
    use rstest::rstest;
    use tokio::io::AsyncWriteExt;

    use super::*;

    #[rstest]
    #[case::listed_size(Some(0))]
    #[case::head_size(None)]
    #[tokio::test]
    async fn test_read_empty_manifest_reports_path(#[case] known_size: Option<u64>) {
        let store = ObjectStore::memory();
        let path = Path::from("dataset/_versions/1.manifest");
        store.inner.put(&path, Bytes::new().into()).await.unwrap();

        let error = read_manifest(&store, &path, known_size).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error:?}");
        assert!(error.to_string().contains(path.as_ref()), "{error}");
        assert!(error.to_string().contains("0 bytes"), "{error}");
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn write_test_manifest(
        prefix_size: usize,
        manifest_min_size: usize,
    ) -> (ObjectStore, Path, Manifest, usize, u64) {
        let mut store = ObjectStore::memory();
        store.inner = store.io_tracker().wrap("", store.inner.clone());
        let path = Path::from("/read_large_manifest");

        let mut writer = store.create(&path).await.unwrap();

        // Write prefix we should ignore
        let prefix: Vec<u8> = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(prefix_size)
            .collect();
        writer.write_all(&prefix).await.unwrap();

        // A cheap deterministic filler; only the size matters for these tests.
        let long_name: String = "a".repeat(manifest_min_size);

        let arrow_schema =
            ArrowSchema::new(vec![ArrowField::new(long_name, DataType::Int64, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let mut config = HashMap::new();
        config.insert("key".to_string(), "value".to_string());

        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        let pos = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        let file_size = writer.tell().await.unwrap() as u64;
        Writer::shutdown(writer.as_mut()).await.unwrap();

        (store, path, manifest, pos, file_size)
    }

    async fn test_roundtrip_manifest(prefix_size: usize, manifest_min_size: usize) {
        let (store, path, manifest, pos, file_size) =
            write_test_manifest(prefix_size, manifest_min_size).await;
        let expected_bytes = store
            .inner
            .get_range(&path, pos as u64 + 4..file_size - 16)
            .await
            .unwrap();
        store.io_stats_incremental();
        let mut bytes = read_manifest_bytes(&store, &path, Some(file_size))
            .await
            .unwrap();
        let stats = store.io_stats_incremental();
        let remaining_size = (file_size as usize - pos).saturating_sub(64 * 1024);
        assert_eq!(
            stats.read_iops,
            1 + remaining_size.div_ceil(METADATA_READ_CHUNK_SIZE) as u64
        );
        assert_eq!(
            stats.read_bytes,
            file_size.min(64 * 1024) + remaining_size as u64
        );
        assert_eq!(bytes.copy_to_bytes(bytes.remaining()), expected_bytes);

        let roundtripped_manifest = read_manifest(&store, &path, None).await.unwrap();

        assert_eq!(manifest, roundtripped_manifest);

        store.inner.delete(&path).await.unwrap();
    }

    #[tokio::test]
    async fn test_read_large_manifest() {
        test_roundtrip_manifest(0, 100_000).await;
        test_roundtrip_manifest(1000, 100_000).await;
        test_roundtrip_manifest(1000, 1000).await;
    }

    #[tokio::test]
    async fn test_read_manifest_larger_than_read_chunk() {
        // Crosses METADATA_READ_CHUNK_SIZE so the manifest body is fetched as
        // multiple concurrent chunks and reassembled with the prefetched tail.
        test_roundtrip_manifest(1000, METADATA_READ_CHUNK_SIZE + 4 * 1024 * 1024).await;
    }

    #[rstest]
    #[case::too_small(-1)]
    #[case::too_large(1)]
    #[tokio::test]
    async fn test_read_manifest_bytes_retries_stale_size(
        #[case] size_delta: i64,
        #[values(1000, 100_000)] manifest_min_size: usize,
    ) {
        let (store, path, manifest, _, file_size) = write_test_manifest(0, manifest_min_size).await;
        let known_size = file_size.checked_add_signed(size_delta).unwrap();
        let bytes = read_manifest_bytes(&store, &path, Some(known_size))
            .await
            .unwrap();
        let observed = Manifest::try_from(pb::Manifest::decode(bytes).unwrap()).unwrap();
        assert_eq!(observed, manifest);
    }

    #[rstest]
    #[case::invalid_offset(100, 0, "Invalid manifest offset")]
    #[case::invalid_length(0, 1, "manifest length does not match")]
    #[tokio::test]
    async fn test_read_manifest_bytes_rejects_corruption(
        #[case] offset: usize,
        #[case] recorded_length: u32,
        #[case] message: &str,
    ) {
        let store = ObjectStore::memory();
        let path = Path::from("corrupt.manifest");
        let mut writer = store.create(&path).await.unwrap();
        writer
            .write_all(&recorded_length.to_le_bytes())
            .await
            .unwrap();
        writer
            .write_magics(offset, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let error = read_manifest_bytes(&store, &path, Some(20))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            Error::CorruptFile { .. } | Error::InvalidInput { .. }
        ));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[tokio::test]
    async fn test_write_manifest_clears_unwritten_index_section() {
        let store = ObjectStore::memory();
        let path = Path::from("/clear_unwritten_index_section");
        let mut writer = store.create(&path).await.unwrap();
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.index_section = Some(42);

        let pos = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        assert!(manifest.index_section.is_none());
        let roundtripped_manifest = read_manifest(&store, &path, None).await.unwrap();
        assert!(roundtripped_manifest.index_section.is_none());
    }

    #[tokio::test]
    async fn test_update_schema_metadata() {
        let store = ObjectStore::memory();
        let path = Path::from("/update_schema_metadata");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = V1FileWriter::<ManifestDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = Int32Array::from_iter_values(0..10);
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array)]).unwrap();
        file_writer
            .write(std::slice::from_ref(&batch))
            .await
            .unwrap();
        let mut metadata = HashMap::new();
        metadata.insert(String::from("lance:extra"), String::from("for_test"));
        file_writer.finish_with_metadata(&metadata).await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let reader = V1FileReader::try_new_self_described_from_reader(reader.into(), None)
            .await
            .unwrap();
        let schema = ArrowSchema::from(reader.schema());
        assert_eq!(schema.metadata().get("lance:extra").unwrap(), "for_test");
    }
}
