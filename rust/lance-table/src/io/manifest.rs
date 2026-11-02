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
use rayon::prelude::*;
use std::collections::HashMap;
use std::{ops::Range, sync::Arc};
use tokio::io::AsyncWriteExt;
use tracing::instrument;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::{
    object_store::ObjectStore,
    traits::{WriteExt, Writer},
    utils::{ChunkedBuf, METADATA_READ_CHUNK_SIZE, read_message, read_range_in_chunks},
};

use crate::format::{
    DataStorageFormat, Fragment, IndexMetadata, MAGIC, Manifest, Transaction, pb,
    pb_manifest_without_fragments,
};

use super::commit::ManifestLocation;

/// Read the raw Manifest protobuf from a URI.
///
/// This only reads manifest files. It does not read data files or translate the
/// protobuf into the semantic [`Manifest`] type.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> lance_core::Result<()> {
/// use lance_io::object_store::ObjectStore;
/// use lance_table::format::pb;
/// use lance_table::io::manifest::read_manifest_proto;
/// use object_store::path::Path;
///
/// let object_store = ObjectStore::local();
/// let path = Path::from_absolute_path("/data/table/_versions/1.manifest")?;
/// let manifest: pb::Manifest = read_manifest_proto(&object_store, &path, None).await?;
/// println!("manifest version: {}", manifest.version);
/// # Ok(())
/// # }
/// ```
pub async fn read_manifest_proto(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<pb::Manifest> {
    let buf = read_manifest_bytes(object_store, path, known_size).await?;
    Ok(pb::Manifest::decode(buf)?)
}

/// Read the semantic [`Manifest`] from a URI.
///
/// This only reads manifest files. It does not read data files. Use
/// [`Manifest::summary`] to inspect aggregate fragment and row counts after
/// loading.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> lance_core::Result<()> {
/// use lance_io::object_store::ObjectStore;
/// use lance_table::format::Manifest;
/// use lance_table::io::manifest::read_manifest;
/// use object_store::path::Path;
///
/// let object_store = ObjectStore::local();
/// let path = Path::from_absolute_path("/data/table/_versions/1.manifest")?;
/// let manifest: Manifest = read_manifest(&object_store, &path, None).await?;
/// println!("fragments: {}", manifest.summary().total_fragments);
/// # Ok(())
/// # }
/// ```
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

    write_manifest_message(writer, manifest).await
}

/// Field key of `Manifest.fragments` in table.proto: field 2, wire type 2
/// (length-delimited).
const MANIFEST_FRAGMENTS_KEY: u8 = (2 << 3) | 2;

/// When the fragments of a manifest are encoded in parallel.
#[derive(Clone, Copy, Debug)]
struct ParallelEncode {
    /// Fewest fragments without cached bytes for which encoding runs in
    /// parallel.
    min_uncached: usize,
    /// Fragments per chunk when encoding in parallel.
    chunk_fragments: usize,
}

const PARALLEL_ENCODE: ParallelEncode = ParallelEncode {
    min_uncached: 64 * 1024,
    chunk_fragments: 16 * 1024,
};

/// The u32 length prefix of a manifest message of `len` bytes.
///
/// Readers take the prefix as the message length, so a message that does not
/// fit in a u32 cannot be written.
fn manifest_message_len_prefix(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        Error::invalid_input(format!(
            "Manifest message is {len} bytes, which exceeds the {} byte limit of its \
             u32 length prefix",
            u32::MAX
        ))
    })
}

/// Whether any data file of `fragment` has an unknown size.
///
/// File sizes are cached in place once learned, so the encoding of such a
/// fragment can change without the fragment list changing. It is encoded on
/// every write instead of cached.
fn has_unknown_file_size(fragment: &Fragment) -> bool {
    fragment
        .referenced_lance_files()
        .any(|file| file.file_size_bytes.get().is_none())
}

/// Append `manifest`'s protobuf message to `buf`, preceded by its u32
/// little-endian length, the layout [`WriteExt::write_protobuf`] produces.
///
/// Fragments are appended one `fragments` field at a time: bytes cached in
/// [`Manifest::encoded_fragments`] are copied as they are, and only the other
/// fragments are encoded. prost encodes fields in tag order (`fields` = 1,
/// `fragments` = 2, then the rest), so the message decodes to the same
/// `pb::Manifest` as `pb::Manifest::from(manifest)` and its bytes differ at
/// most in the order of map entries.
///
/// Returns, for each fragment, the range of `buf` holding its encoded bytes,
/// or `None` when those bytes must not be reused.
///
/// When many fragments have no cached bytes (the first write after a manifest
/// is loaded), they are encoded in parallel in chunks that are then appended
/// in order.
fn encode_manifest_message(
    manifest: &Manifest,
    buf: &mut Vec<u8>,
    parallel: ParallelEncode,
) -> Result<Vec<Option<Range<usize>>>> {
    let fragments = manifest.fragments.as_slice();
    let cached = manifest.encoded_fragments();
    // Each cached entry also needs a key byte and a length delimiter of at
    // most 5 bytes.
    let cached_len: usize = cached.map_or(0, |cached| {
        cached.iter().flatten().map(|bytes| bytes.len() + 6).sum()
    });
    buf.reserve(cached_len + 64 * 1024);

    let len_pos = buf.len();
    buf.extend_from_slice(&[0; 4]);
    let msg_start = buf.len();

    let mut rest = pb_manifest_without_fragments(manifest);
    let head = pb::Manifest {
        fields: std::mem::take(&mut rest.fields),
        ..Default::default()
    };
    head.encode(buf)?;

    let num_uncached = cached.map_or(fragments.len(), |cached| {
        cached.iter().filter(|bytes| bytes.is_none()).count()
    });
    let mut ranges = Vec::with_capacity(fragments.len());
    if num_uncached < parallel.min_uncached {
        append_fragments(fragments, cached, buf, &mut ranges)?;
    } else {
        let chunks = fragments
            .par_chunks(parallel.chunk_fragments)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let start = chunk_index * parallel.chunk_fragments;
                let cached = cached.map(|cached| &cached[start..start + chunk.len()]);
                let mut chunk_buf = Vec::new();
                let mut chunk_ranges = Vec::with_capacity(chunk.len());
                append_fragments(chunk, cached, &mut chunk_buf, &mut chunk_ranges)?;
                Ok((chunk_buf, chunk_ranges))
            })
            .collect::<Result<Vec<_>>>()?;
        buf.reserve(chunks.iter().map(|(chunk_buf, _)| chunk_buf.len()).sum());
        for (chunk_buf, chunk_ranges) in chunks {
            let offset = buf.len();
            buf.extend_from_slice(&chunk_buf);
            ranges.extend(
                chunk_ranges
                    .into_iter()
                    .map(|range| range.map(|range| range.start + offset..range.end + offset)),
            );
        }
    }

    rest.encode(buf)?;

    let msg_len = manifest_message_len_prefix(buf.len() - msg_start)?;
    buf[len_pos..msg_start].copy_from_slice(&msg_len.to_le_bytes());
    Ok(ranges)
}

/// Append each fragment to `buf` as a `Manifest.fragments` field, copying
/// `cached[i]` when present and encoding `fragments[i]` otherwise, and push the
/// range of its encoded bytes in `buf` (or `None` if they must not be reused)
/// to `ranges`.
fn append_fragments(
    fragments: &[Fragment],
    cached: Option<&[Option<Bytes>]>,
    buf: &mut Vec<u8>,
    ranges: &mut Vec<Option<Range<usize>>>,
) -> Result<()> {
    for (i, fragment) in fragments.iter().enumerate() {
        // Checked before encoding: sizes only ever go from unknown to known,
        // so a fragment whose sizes are all known here is encoded with them.
        // Checking afterwards could cache bytes that missed a size filled in
        // by a concurrent reader.
        let reusable = !has_unknown_file_size(fragment);
        buf.push(MANIFEST_FRAGMENTS_KEY);
        let range = match cached.and_then(|cached| cached[i].as_ref()) {
            Some(bytes) => {
                prost::encode_length_delimiter(bytes.len(), buf)?;
                let start = buf.len();
                buf.extend_from_slice(bytes);
                start..buf.len()
            }
            None => {
                let delimiter_pos = buf.len();
                pb::DataFragment::from(fragment).encode_length_delimited(buf)?;
                let len = prost::decode_length_delimiter(&buf[delimiter_pos..])?;
                buf.len() - len..buf.len()
            }
        };
        ranges.push(reusable.then_some(range));
    }
    Ok(())
}

/// Keep the encoded fragments written in `buf` on `manifest` for the next write.
///
/// The cached slices keep all of `buf` alive, so `buf` should hold the
/// manifest message and nothing else.
fn cache_encoded_fragments(
    manifest: &mut Manifest,
    buf: Vec<u8>,
    ranges: Vec<Option<Range<usize>>>,
) {
    // `Bytes::from(Vec)` keeps the Vec allocation when capacity exceeds len,
    // so cache the capacity before conversion rather than the logical length.
    let retained_bytes = buf.capacity();
    let buf = Bytes::from(buf);
    let encoded = ranges
        .into_iter()
        .map(|range| range.map(|range| buf.slice(range)))
        .collect();
    manifest.set_encoded_fragments(encoded, retained_bytes);
}

/// Write the length-prefixed manifest message and return its position.
async fn write_manifest_message(writer: &mut dyn Writer, manifest: &mut Manifest) -> Result<usize> {
    let pos = writer.tell().await?;
    let mut buf = Vec::new();
    let ranges = encode_manifest_message(manifest, &mut buf, PARALLEL_ENCODE)?;
    writer.write_all(&buf).await?;
    cache_encoded_fragments(manifest, buf, ranges);
    Ok(pos)
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

    use crate::feature_flags::{FLAG_OBJECT_METADATA, FLAG_STABLE_ROW_IDS};
    use crate::format::ClusteringDefinition;
    use crate::format::SelfDescribingFileReader;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::deepsize::DeepSizeOf;
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

    /// A manifest whose fragments exercise most `DataFragment` fields.
    fn manifest_with_fragments(num_fragments: u64) -> Manifest {
        use crate::format::{DeletionFile, DeletionFileType, RowIdMeta};

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, false),
            ArrowField::new("b", DataType::Utf8, true),
        ]);
        let fragments = (0..num_fragments)
            .map(|id| {
                let mut fragment = Fragment::new(id).with_physical_rows(1000 + id as usize);
                for field_id in 0..2 {
                    fragment.add_file(
                        format!("data/{id}-{field_id}.lance"),
                        vec![field_id],
                        vec![field_id],
                        ConcreteFileVersion::V2_0,
                        std::num::NonZero::new(4096 + id),
                    );
                }
                if id % 3 == 0 {
                    fragment.deletion_file = Some(DeletionFile {
                        read_version: id,
                        id,
                        file_type: DeletionFileType::Bitmap,
                        num_deleted_rows: Some(7),
                        base_id: None,
                        inline: Some(Bytes::from_static(b"deleted-rows")),
                    });
                }
                fragment.row_id_meta = Some(RowIdMeta::Inline(vec![id as u8; 5].into()));
                fragment.clustering_id = Some(1);
                fragment
                    .metadata
                    .insert("object-key".into(), format!("fragment-{id}"));
                fragment
            })
            .collect();
        let mut manifest = Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(fragments),
            DataStorageFormat::new(ConcreteFileVersion::V2_0),
            HashMap::new(),
        );
        // One entry per map keeps prost's map encoding order deterministic,
        // so messages can be compared byte for byte.
        manifest
            .config_mut()
            .insert("key".to_string(), "value".to_string());
        manifest
            .table_metadata
            .insert("object-key".into(), "object-value".into());
        manifest.clustering = vec![ClusteringDefinition {
            id: 1,
            columns: vec![0],
            provider: "org.example.cluster".into(),
            version: 1,
        }];
        manifest.max_clustering_id = 1;
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_STABLE_ROW_IDS | FLAG_OBJECT_METADATA;
        manifest.version = 42;
        manifest.update_max_fragment_id();
        manifest
    }

    /// The length-prefixed message `write_protobuf` writes for `manifest`.
    fn prost_manifest_message(manifest: &Manifest) -> Vec<u8> {
        let message = pb::Manifest::from(manifest).encode_to_vec();
        let mut buf = (message.len() as u32).to_le_bytes().to_vec();
        buf.extend_from_slice(&message);
        buf
    }

    fn encode_message(manifest: &mut Manifest) -> Vec<u8> {
        encode_message_with(manifest, PARALLEL_ENCODE)
    }

    fn encode_message_with(manifest: &mut Manifest, parallel: ParallelEncode) -> Vec<u8> {
        let mut buf = Vec::new();
        let ranges = encode_manifest_message(manifest, &mut buf, parallel).unwrap();
        let message = buf.clone();
        cache_encoded_fragments(manifest, buf, ranges);
        message
    }

    #[test]
    fn test_cached_fragments_account_for_vec_allocation_capacity() {
        let mut manifest = manifest_with_fragments(1);
        let uncached_size = manifest.deep_size_of();

        encode_message(&mut manifest);
        let cached_size = manifest.deep_size_of();
        let message_bytes = manifest.serialized().len() + 4;
        let retained_bytes = cached_size - uncached_size;

        assert!(
            retained_bytes > message_bytes + 32 * 1024,
            "cache retained {retained_bytes} bytes for a {message_bytes}-byte message; \
             it must account for the Vec allocation capacity"
        );
    }

    #[test]
    fn test_encode_manifest_message_matches_prost() {
        let mut manifest = manifest_with_fragments(20);
        let expected = prost_manifest_message(&manifest);

        // Cold: every fragment is encoded.
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(encode_message(&mut manifest), expected);
        let cached = manifest.encoded_fragments().unwrap();
        assert!(cached.iter().all(Option::is_some));

        // Warm: every fragment is copied from the cache.
        assert_eq!(encode_message(&mut manifest), expected);

        // Partly warm: fragments without cached bytes are encoded.
        let partial = manifest
            .encoded_fragments()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, bytes)| bytes.clone().filter(|_| i % 4 != 0))
            .collect();
        manifest.set_encoded_fragments(partial, expected.len());
        assert_eq!(encode_message(&mut manifest), expected);

        // No fragments.
        let mut empty = manifest_with_fragments(0);
        assert_eq!(encode_message(&mut empty), prost_manifest_message(&empty));
    }

    /// Parallel encoding, in chunks that do not divide the fragment count,
    /// produces the same message as encoding sequentially.
    #[rstest]
    #[case::cold(0)]
    #[case::partly_warm(3)]
    fn test_encode_manifest_message_parallel_matches_prost(#[case] keep_every: usize) {
        let parallel = ParallelEncode {
            min_uncached: 8,
            chunk_fragments: 7,
        };
        let mut manifest = manifest_with_fragments(50);
        let expected = prost_manifest_message(&manifest);
        if keep_every > 0 {
            encode_message(&mut manifest);
            let partial = manifest
                .encoded_fragments()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, bytes)| bytes.clone().filter(|_| i % keep_every == 0))
                .collect();
            manifest.set_encoded_fragments(partial, expected.len());
        }

        assert_eq!(encode_message_with(&mut manifest, parallel), expected);
        let cached = manifest.encoded_fragments().unwrap();
        assert_eq!(cached.len(), 50);
        assert!(cached.iter().all(Option::is_some));

        // Warm: every fragment is copied from the cache.
        assert_eq!(encode_message_with(&mut manifest, parallel), expected);
    }

    #[test]
    fn test_encode_manifest_message_ignores_cache_of_replaced_fragments() {
        let mut manifest = manifest_with_fragments(10);
        encode_message(&mut manifest);
        assert!(manifest.encoded_fragments().is_some());

        // Replacing the fragment list, or mutating it, makes the cache stale.
        let mut fragments = manifest.fragments.as_ref().clone();
        fragments[3].physical_rows = Some(1);
        manifest.fragments = Arc::new(fragments);
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(
            encode_message(&mut manifest),
            prost_manifest_message(&manifest)
        );

        Arc::make_mut(&mut manifest.fragments)[4].physical_rows = Some(2);
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(
            encode_message(&mut manifest),
            prost_manifest_message(&manifest)
        );
    }

    #[test]
    fn test_encode_manifest_message_does_not_cache_unknown_file_sizes() {
        let mut manifest = manifest_with_fragments(3);
        let mut fragments = manifest.fragments.as_ref().clone();
        fragments[1].add_file(
            "data/unsized.lance",
            vec![1],
            vec![1],
            ConcreteFileVersion::V2_0,
            None,
        );
        manifest.fragments = Arc::new(fragments);

        assert_eq!(
            encode_message(&mut manifest),
            prost_manifest_message(&manifest)
        );
        let cached = manifest.encoded_fragments().unwrap();
        assert_eq!(
            cached.iter().map(Option::is_some).collect::<Vec<_>>(),
            vec![true, false, true]
        );
    }

    #[test]
    fn test_manifest_message_len_prefix_rejects_oversized_messages() {
        assert_eq!(manifest_message_len_prefix(0).unwrap(), 0);
        assert_eq!(
            manifest_message_len_prefix(u32::MAX as usize).unwrap(),
            u32::MAX
        );
        let err = manifest_message_len_prefix(u32::MAX as usize + 1).unwrap_err();
        assert!(
            err.to_string().contains("exceeds"),
            "unexpected error: {err}"
        );
    }

    async fn write_manifest_file(store: &ObjectStore, path: &Path, manifest: &mut Manifest) {
        let mut writer = store.create(path).await.unwrap();
        let pos = write_manifest(writer.as_mut(), manifest, Some(vec![]), None)
            .await
            .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();
    }

    /// A manifest file written from cached fragment bytes is byte-identical to
    /// one written by encoding every fragment, and reads back the same.
    #[tokio::test]
    async fn test_write_manifest_from_cached_fragments_matches_cold_write() {
        let store = ObjectStore::memory();
        let mut manifest = manifest_with_fragments(50);

        let cold = Path::from("cold");
        write_manifest_file(&store, &cold, &mut manifest).await;
        assert!(manifest.encoded_fragments().is_some());

        let warm = Path::from("warm");
        write_manifest_file(&store, &warm, &mut manifest).await;
        assert_eq!(
            store.read_one_all(&warm).await.unwrap(),
            store.read_one_all(&cold).await.unwrap()
        );

        let read_back = read_manifest(&store, &warm, None).await.unwrap();
        assert!(read_back.encoded_fragments().is_none());
        assert_eq!(read_back, manifest);
    }
}
