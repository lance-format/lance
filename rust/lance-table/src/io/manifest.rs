// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes, BytesMut};
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
    traits::{Reader, WriteExt, Writer},
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
    let file_size = if let Some(known_size) = known_size {
        known_size
    } else {
        object_store.inner.head(path).await?.size
    };
    if file_size < 16 {
        if known_size.is_some() {
            return Box::pin(read_manifest(object_store, path, None)).await;
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
    let buf = object_store.inner.get_range(path, range).await?;

    // In case of corruption, the known_size might be wrong. We can retry without
    // the size to be more robust.
    if (buf.len() < 16 || !buf.ends_with(MAGIC)) && known_size.is_some() {
        return Box::pin(read_manifest(object_store, path, None)).await;
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
    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    let manifest_len = file_size as usize - manifest_pos;

    let buf: Bytes = if manifest_len <= buf.len() {
        // The prefetch captured the entire manifest. We just need to trim the buffer.
        buf.slice(buf.len() - manifest_len..buf.len())
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
            manifest_pos..(file_size - PREFETCH_SIZE) as usize,
            METADATA_READ_CHUNK_SIZE,
        );
        // Decode straight from the chunks rather than re-copying ~all of the
        // file into one buffer; see `ChunkedBuf`.
        let mut chunked = ChunkedBuf::default();
        while let Some(chunk) = chunks.try_next().await? {
            chunked.push(chunk);
        }
        chunked.push(buf);
        // A u32 length prefix in front, the 16 byte footer behind.
        if chunked.remaining() < 20 {
            return Err(Error::corrupt_file(
                path.clone(),
                "Invalid format: manifest shorter than its length prefix and footer".to_string(),
            ));
        }
        let recorded_length = chunked.get_u32_le() as usize;
        chunked.truncate(chunked.remaining() - 16);
        return decode_manifest(chunked, recorded_length);
    };

    let recorded_length = LittleEndian::read_u32(&buf[0..4]) as usize;
    // Need to trim the magic number at end and message length at beginning
    let buf = buf.slice(4..buf.len() - 16);
    decode_manifest(buf, recorded_length)
}

fn decode_manifest(buf: impl Buf, recorded_length: usize) -> Result<Manifest> {
    if buf.remaining() != recorded_length {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {}, got {}",
            recorded_length,
            buf.remaining()
        )));
    }
    let proto = pb::Manifest::decode(buf)?;
    let mut manifest = Manifest::try_from(proto)?;
    manifest.detach_sparse_inline_row_ids(recorded_length);
    Ok(manifest)
}

/// Bytes of the little-endian u32 length prefix that precedes every section
/// message of a manifest file (index section, inline transaction, manifest).
const LEN_PREFIX_LEN: usize = 4;

/// Bytes of the footer that follows the manifest message: its i64 position,
/// the u16 major and minor versions and the magic.
const FOOTER_LEN: usize = 16;

/// First byte of a non-empty `pb::IndexSection`: field 1 (`indices`),
/// length-delimited.
const INDEX_SECTION_FIRST_KEY: u8 = (1 << 3) | 2;

/// Possible first bytes of a `pb::Transaction` as written by prost, which
/// encodes fields in field-number order: field 1 (`read_version`, varint), or
/// field 2 (`uuid`, length-delimited) when `read_version` is 0.
const TRANSACTION_FIRST_KEYS: [u8; 2] = [1 << 3, (2 << 3) | 2];

/// What [`read_inline_transaction_by_layout`] learned from a manifest file.
#[derive(Debug, Clone, PartialEq)]
pub enum InlineTransactionLookup {
    /// The file's sections are laid out so that its last section before the
    /// manifest message can only be an inline transaction, which decoded to
    /// this message.
    Found(Box<pb::Transaction>),
    /// The layout does not prove where the inline transaction is, or that the
    /// file has one. The reason is for logs. The manifest must be read to
    /// learn the transaction; this is never evidence that there is none.
    Inconclusive(&'static str),
}

/// Read the inline transaction of a manifest file from its section layout,
/// without reading or decoding the manifest message.
///
/// A manifest file is `[dictionaries (legacy files only)] [index section]
/// [inline transaction] [manifest message] [footer]`, where every section is
/// optional and is a u32 length prefix followed by a protobuf message. The
/// footer gives the manifest message's position. The transaction is found
/// when the sections before the manifest message are exactly one of:
///
/// - one section starting with a transaction's first field key
///   ([`TRANSACTION_FIRST_KEYS`]);
/// - an index section (empty, or starting with [`INDEX_SECTION_FIRST_KEY`])
///   followed by a section starting with a transaction's first field key.
///
/// Every length prefix must equal the distance to the next section exactly.
/// Any other layout, a reader size that does not end at the footer (a stale
/// size) and a message that fails to decode are inconclusive too.
///
/// Costs one round trip (the first block and the footer, read concurrently)
/// when the transaction fits in the first block, and one more otherwise.
///
/// # Example
///
/// ```no_run
/// # use lance_io::object_store::ObjectStore;
/// # use lance_table::io::manifest::{InlineTransactionLookup, read_inline_transaction_by_layout};
/// # use object_store::path::Path;
/// # async fn example(store: &ObjectStore, manifest_path: &Path) -> lance_core::Result<()> {
/// let reader = store.open(manifest_path).await?;
/// match read_inline_transaction_by_layout(reader.as_ref()).await? {
///     InlineTransactionLookup::Found(transaction) => {
///         println!("built on version {}", transaction.read_version);
///     }
///     // Not evidence of a missing transaction: read the manifest, which
///     // locates an inline transaction, an external file or none.
///     InlineTransactionLookup::Inconclusive(reason) => {
///         println!("reading the manifest: {reason}");
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub async fn read_inline_transaction_by_layout(
    reader: &dyn Reader,
) -> Result<InlineTransactionLookup> {
    let file_size = reader.size().await?;
    // One section with a non-empty body, a manifest prefix and the footer.
    if file_size < LEN_PREFIX_LEN + 1 + LEN_PREFIX_LEN + FOOTER_LEN {
        return Ok(InlineTransactionLookup::Inconclusive(
            "file is too small to hold a section",
        ));
    }
    let head_len = file_size.min(reader.block_size().max(LEN_PREFIX_LEN + 1));
    let footer_range = file_size - FOOTER_LEN..file_size;
    let (head, footer) = if head_len == file_size {
        let all = reader.get_range(0..file_size).await?;
        let footer = all.slice(footer_range);
        (all, footer)
    } else {
        futures::try_join!(
            reader.get_range(0..head_len),
            reader.get_range(footer_range)
        )?
    };
    if head.len() != head_len || footer.len() != FOOTER_LEN || !footer.ends_with(MAGIC) {
        return Ok(InlineTransactionLookup::Inconclusive(
            "no footer at the end of the reader",
        ));
    }
    let Ok(manifest_pos) = usize::try_from(LittleEndian::read_i64(&footer[..8])) else {
        return Ok(InlineTransactionLookup::Inconclusive(
            "negative manifest position",
        ));
    };
    if manifest_pos > file_size - LEN_PREFIX_LEN - FOOTER_LEN {
        return Ok(InlineTransactionLookup::Inconclusive(
            "manifest position past the end of the file",
        ));
    }

    let start = match transaction_section_start(&head, manifest_pos) {
        Ok(start) => start,
        Err(reason) => return Ok(InlineTransactionLookup::Inconclusive(reason)),
    };
    let section = if manifest_pos <= head.len() {
        head.slice(start..manifest_pos)
    } else {
        let rest = reader
            .get_range(head.len().max(start)..manifest_pos)
            .await?;
        if start >= head.len() {
            rest
        } else {
            let mut joined = BytesMut::with_capacity(manifest_pos - start);
            joined.extend_from_slice(&head[start..]);
            joined.extend_from_slice(&rest);
            joined.freeze()
        }
    };
    if section.len() != manifest_pos - start {
        return Ok(InlineTransactionLookup::Inconclusive(
            "short read of the transaction section",
        ));
    }

    let body = &section[LEN_PREFIX_LEN..];
    if LittleEndian::read_u32(&section[..LEN_PREFIX_LEN]) as usize != body.len() {
        return Ok(InlineTransactionLookup::Inconclusive(
            "last section's length prefix does not reach the manifest",
        ));
    }
    if !body
        .first()
        .is_some_and(|key| TRANSACTION_FIRST_KEYS.contains(key))
    {
        return Ok(InlineTransactionLookup::Inconclusive(
            "last section is not a transaction",
        ));
    }
    match pb::Transaction::decode(body) {
        Ok(transaction) => Ok(InlineTransactionLookup::Found(Box::new(transaction))),
        Err(_) => Ok(InlineTransactionLookup::Inconclusive(
            "last section does not decode as a transaction",
        )),
    }
}

/// The position of the section that, by the layout rules of
/// [`read_inline_transaction_by_layout`], must be the inline transaction of a
/// manifest file whose manifest message is at `manifest_pos`, given `head`,
/// the first bytes of the file. The section's own prefix and first byte are
/// left for the caller to check.
///
/// `head` must hold at least the first section's prefix and first body byte.
fn transaction_section_start(
    head: &[u8],
    manifest_pos: usize,
) -> std::result::Result<usize, &'static str> {
    let first_len = LittleEndian::read_u32(&head[..LEN_PREFIX_LEN]) as usize;
    let second = LEN_PREFIX_LEN + first_len;
    if second == manifest_pos {
        return Ok(0);
    }
    // The first section must be an index section and a non-empty second
    // section must end at the manifest.
    if second + LEN_PREFIX_LEN + 1 > manifest_pos {
        return Err("sections do not end at the manifest");
    }
    if first_len != 0 && head[LEN_PREFIX_LEN] != INDEX_SECTION_FIRST_KEY {
        return Err("first of two sections is not an index section");
    }
    // The candidate's prefix must equal its length, so a longer candidate
    // can never match. Reject it before its bytes are read: it arises when
    // the first section's prefix wrapped, and would span the rest of that
    // section.
    if manifest_pos - second - LEN_PREFIX_LEN > u32::MAX as usize {
        return Err("second section is longer than a length prefix can hold");
    }
    Ok(second)
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
    use std::collections::HashMap;

    use crate::format::SelfDescribingFileReader;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
    use lance_file::versions::v1::{
        reader::FileReader as V1FileReader, writer::FileWriter as V1FileWriter,
    };
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

    async fn test_roundtrip_manifest(prefix_size: usize, manifest_min_size: usize) {
        let store = ObjectStore::memory();
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
        Writer::shutdown(writer.as_mut()).await.unwrap();

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

    /// Sections to write in front of a manifest message, in file order.
    #[allow(clippy::large_enum_variant)]
    enum TestSection {
        Index(pb::IndexSection),
        Transaction(pb::Transaction),
        Raw(Vec<u8>),
    }

    /// Write a manifest file holding `sections`, the manifest message and the
    /// footer, and return its size.
    async fn write_manifest_file_with_sections(
        store: &ObjectStore,
        path: &Path,
        sections: &[TestSection],
    ) -> usize {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("a", DataType::Int64, false)]);
        let manifest = Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        let mut writer = store.create(path).await.unwrap();
        for section in sections {
            match section {
                TestSection::Index(index) => {
                    writer.write_protobuf(index).await.unwrap();
                }
                TestSection::Transaction(transaction) => {
                    writer.write_protobuf(transaction).await.unwrap();
                }
                TestSection::Raw(bytes) => writer.write_all(bytes).await.unwrap(),
            }
        }
        let pos = writer.write_struct(&manifest).await.unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap().size
    }

    async fn lookup(store: &ObjectStore, path: &Path) -> InlineTransactionLookup {
        let reader = store.open(path).await.unwrap();
        read_inline_transaction_by_layout(reader.as_ref())
            .await
            .unwrap()
    }

    fn test_transaction(read_version: u64, padding: usize) -> pb::Transaction {
        pb::Transaction {
            read_version,
            uuid: "layout transaction".to_string(),
            tag: "x".repeat(padding),
            transaction_properties: HashMap::from([("k".to_string(), "v".to_string())]),
            operation: Some(pb::transaction::Operation::Append(
                pb::transaction::Append { fragments: vec![] },
            )),
        }
    }

    fn non_empty_index_section() -> pb::IndexSection {
        pb::IndexSection {
            indices: vec![pb::IndexMetadata {
                name: "idx".to_string(),
                dataset_version: 3,
                ..Default::default()
            }],
        }
    }

    #[tokio::test]
    async fn test_inline_transaction_by_layout_finds_transaction() {
        let store = ObjectStore::memory();
        let path = Path::from("layout");
        // Read version 0 leaves field 1 out, so the message starts at the uuid.
        // A large tag puts the section past the first block.
        for transaction in [
            test_transaction(7, 0),
            test_transaction(0, 0),
            test_transaction(7, 300 * 1024),
        ] {
            let layouts = [
                vec![TestSection::Transaction(transaction.clone())],
                vec![
                    TestSection::Index(pb::IndexSection::default()),
                    TestSection::Transaction(transaction.clone()),
                ],
                vec![
                    TestSection::Index(non_empty_index_section()),
                    TestSection::Transaction(transaction.clone()),
                ],
            ];
            for sections in layouts {
                write_manifest_file_with_sections(&store, &path, &sections).await;
                assert_eq!(
                    lookup(&store, &path).await,
                    InlineTransactionLookup::Found(Box::new(transaction.clone()))
                );
            }
        }
    }

    #[tokio::test]
    async fn test_inline_transaction_by_layout_is_inconclusive_without_proof() {
        let store = ObjectStore::memory();
        let path = Path::from("layout");
        let transaction = test_transaction(7, 0);
        let encoded = transaction.encode_to_vec();
        let mut wrong_prefix = ((encoded.len() + 2) as u32).to_le_bytes().to_vec();
        wrong_prefix.extend_from_slice(&encoded);
        wrong_prefix.push(0);

        let layouts = [
            // No sections, an index section alone (the transaction may be in
            // an external file), and an empty one.
            (vec![], "sections do not end at the manifest"),
            (
                vec![TestSection::Index(non_empty_index_section())],
                "last section is not a transaction",
            ),
            (
                vec![TestSection::Index(pb::IndexSection::default())],
                "last section is not a transaction",
            ),
            // Three sections, as with legacy dictionaries in front.
            (
                vec![
                    TestSection::Raw(vec![1, 2, 3, 4, 5, 6, 7, 8]),
                    TestSection::Index(non_empty_index_section()),
                    TestSection::Transaction(transaction.clone()),
                ],
                "sections do not end at the manifest",
            ),
            // A prefix that does not match the section's length.
            (
                vec![TestSection::Raw(wrong_prefix)],
                "sections do not end at the manifest",
            ),
            // Two sections whose first is not an index section.
            (
                vec![
                    TestSection::Transaction(transaction.clone()),
                    TestSection::Transaction(transaction.clone()),
                ],
                "first of two sections is not an index section",
            ),
        ];
        for (sections, reason) in layouts {
            write_manifest_file_with_sections(&store, &path, &sections).await;
            assert_eq!(
                lookup(&store, &path).await,
                InlineTransactionLookup::Inconclusive(reason)
            );
        }

        // A section whose first byte is a transaction key but that does not
        // decode.
        let mut garbage = vec![1 << 3];
        garbage.extend_from_slice(&[0xFF; 12]);
        let mut raw = (garbage.len() as u32).to_le_bytes().to_vec();
        raw.extend_from_slice(&garbage);
        write_manifest_file_with_sections(&store, &path, &[TestSection::Raw(raw)]).await;
        assert_eq!(
            lookup(&store, &path).await,
            InlineTransactionLookup::Inconclusive("last section does not decode as a transaction")
        );
    }

    #[tokio::test]
    async fn test_inline_transaction_by_layout_with_stale_size() {
        let store = ObjectStore::memory();
        let path = Path::from("layout");
        let transaction = test_transaction(7, 0);
        let size = write_manifest_file_with_sections(
            &store,
            &path,
            &[TestSection::Transaction(transaction.clone())],
        )
        .await;

        // A size short of the true size does not end at the footer.
        let short = store.open_with_size(&path, size - 1).await.unwrap();
        assert_eq!(
            read_inline_transaction_by_layout(short.as_ref())
                .await
                .unwrap(),
            InlineTransactionLookup::Inconclusive("no footer at the end of the reader")
        );
        // A size past the end fails to read; callers fall back on errors too.
        let long = store.open_with_size(&path, size + 100).await.unwrap();
        assert!(!matches!(
            read_inline_transaction_by_layout(long.as_ref()).await,
            Ok(InlineTransactionLookup::Found(_))
        ));
        let exact = store.open_with_size(&path, size).await.unwrap();
        assert_eq!(
            read_inline_transaction_by_layout(exact.as_ref())
                .await
                .unwrap(),
            InlineTransactionLookup::Found(Box::new(transaction))
        );
    }

    #[test]
    fn test_transaction_section_start() {
        let mut head = vec![0u8; 64];
        // One 20 byte section.
        head[..4].copy_from_slice(&20u32.to_le_bytes());
        assert_eq!(transaction_section_start(&head, 24), Ok(0));
        // An empty index section followed by another section.
        head[..4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(transaction_section_start(&head, 30), Ok(4));
        // The second section must hold a prefix and at least one byte.
        assert!(transaction_section_start(&head, 8).is_err());
        assert_eq!(transaction_section_start(&head, 9), Ok(4));
        // A non-empty first section must look like an index section.
        head[..4].copy_from_slice(&10u32.to_le_bytes());
        head[4] = INDEX_SECTION_FIRST_KEY;
        assert_eq!(transaction_section_start(&head, 40), Ok(14));
        head[4] = TRANSACTION_FIRST_KEYS[0];
        assert!(transaction_section_start(&head, 40).is_err());
        // A first section running past the manifest.
        head[4] = INDEX_SECTION_FIRST_KEY;
        assert!(transaction_section_start(&head, 12).is_err());
        // A first section whose prefix wrapped leaves a candidate longer than
        // any prefix describes; it is rejected without reading it.
        #[cfg(target_pointer_width = "64")]
        {
            let candidate_end = 14 + LEN_PREFIX_LEN + u32::MAX as usize;
            assert_eq!(transaction_section_start(&head, candidate_end), Ok(14));
            assert!(transaction_section_start(&head, candidate_end + 1).is_err());
        }
    }
}
