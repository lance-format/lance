// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Portable, snapshot-bound blob reference transport.
//!
//! [`BlobRefBatch`] carries a nullable Arrow extension array and its source
//! contexts through exchange, spill, and retries. It does not read payloads or
//! retain source snapshots. See the [execution contract](https://lance.org/guide/blob_references/).
//! [`BlobRefCapture`] creates references from scanned Blob V2 descriptors;
//! reference readers are a separate API.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::ops::Range;
use std::sync::{Arc, LazyLock};

use arrow_array::{
    Array, RecordBatch, StructArray,
    builder::{LargeBinaryBuilder, UInt32Builder, UInt64Builder},
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
};
use arrow_buffer::NullBufferBuilder;
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::{Buf, Bytes};
use lance_arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY};
use lance_core::datatypes::BlobKind;
use lance_core::{Error, Result};
use lance_io::object_store::uri_to_url;
use lance_table::{
    format::{Manifest, pb as table_pb},
    io::manifest::read_manifest_bytes,
};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;

use crate::Dataset;
use crate::dataset::refs::check_valid_branch;

pub use crate::dataset::blob::capture::BlobRefCapture;

mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance.blob.reference.rs"));
}

const VERSION: u32 = 1;
const OBJECT_RANGES: u32 = 1;
const IMMUTABLE_EXTERNAL: u32 = 2;
const EXTENSION_NAME: &str = "lance.blob.ref";
const EXTENSION_METADATA: &str = r#"{"version":1}"#;

static REFERENCE_FIELDS: LazyLock<Fields> = LazyLock::new(|| {
    Fields::from(vec![
        Field::new("source_context", DataType::UInt32, false),
        Field::new("locator", DataType::LargeBinary, false),
        Field::new("length", DataType::UInt64, true),
    ])
});

/// An exact source snapshot and a portable key to its authenticated storage context.
///
/// Contexts contain no native handles or storage credentials. Workers must resolve
/// `access_context_key` independently. Holding this value does not protect files
/// from cleanup; the application must retain the source for all outstanding reads.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::blob::reference::BlobSourceContext;
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let context = BlobSourceContext::from_dataset(
///     dataset, "catalog/table-incarnation", "query/storage-binding",
/// ).await?;
/// let transferred = BlobSourceContext::decode(&context.encode())?;
/// assert_eq!(context.source_id(), transferred.source_id());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct BlobSourceContext {
    proto: pb::SourceContext,
    source_id: [u8; 32],
    registered_bases: Arc<RoaringBitmap>,
}

impl BlobSourceContext {
    /// Capture the resolved snapshot, including the original manifest protobuf bytes.
    ///
    /// `table_identity` identifies the table incarnation and its catalog/storage
    /// authority. It must distinguish different tables that can reuse a URI.
    /// `access_context_key` names an authenticated binding available to every worker,
    /// including per-base configuration. Neither value may be empty.
    ///
    /// This reads manifest metadata, never blob payloads. It rejects a manifest
    /// that no longer matches `dataset`, rather than capturing a different snapshot.
    pub async fn from_dataset(
        dataset: &Dataset,
        table_identity: impl Into<String>,
        access_context_key: impl Into<String>,
    ) -> Result<Self> {
        let location = dataset.manifest_location();
        let mut manifest =
            read_manifest_bytes(&dataset.object_store, &location.path, location.size).await?;
        let manifest = manifest.copy_to_bytes(manifest.remaining());
        let mut observed = Manifest::try_from(table_pb::Manifest::decode(manifest.clone())?)?;
        observed.detach_sparse_inline_row_ids(manifest.len());
        if !matches_loaded_manifest(&observed, dataset.manifest()) {
            return Err(Error::invalid_input(format!(
                "Manifest {} no longer matches the loaded snapshot at version {}",
                location.path,
                dataset.manifest().version,
            )));
        }
        let root = dataset.refs.root()?;
        Self::from_proto(pb::SourceContext {
            protocol_version: VERSION,
            required_capabilities: vec![OBJECT_RANGES],
            table_identity: table_identity.into(),
            table_uri: canonical_uri(&root.uri)?,
            branch: dataset.manifest().branch.clone(),
            version: dataset.manifest().version,
            manifest_path: location.path.to_string(),
            manifest,
            manifest_identity: location.identity.clone(),
            access_context_key: access_context_key.into(),
        })
    }

    /// Explicitly accept the application's guarantee that external object contents
    /// remain immutable for the lifetime of these references.
    ///
    /// This does not acquire an object version or enforce conditional reads. A
    /// snapshot alone cannot freeze external contents. Protocol v1 only supports
    /// external references under this assumption.
    pub fn with_external_immutability_assumption(mut self) -> Self {
        self.proto.required_capabilities = vec![OBJECT_RANGES, IMMUTABLE_EXTERNAL];
        self.source_id = *blake3::hash(&self.proto.encode_to_vec()).as_bytes();
        self
    }

    /// Encode the context for another process. Forward the resulting bytes intact.
    pub fn encode(&self) -> Vec<u8> {
        self.proto.encode_to_vec()
    }

    /// Decode and validate a context, rejecting unsupported versions/capabilities
    /// and inconsistent snapshot metadata before it can be used in a batch.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let proto = pb::SourceContext::decode(bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid blob source context protobuf: {error}",))
        })?;
        Self::from_proto(proto)
    }

    /// Fingerprint binding a locator to this snapshot, authority, and access context.
    pub fn source_id(&self) -> &[u8; 32] {
        &self.source_id
    }

    /// Canonical table-incarnation identity supplied by the application.
    pub fn table_identity(&self) -> &str {
        &self.proto.table_identity
    }

    /// Absolute table root URI, without the branch suffix.
    pub fn table_uri(&self) -> &str {
        &self.proto.table_uri
    }

    /// Resolved branch name; `None` identifies the main branch.
    pub fn branch(&self) -> Option<&str> {
        self.proto.branch.as_deref()
    }

    /// Exact version within the resolved branch, never an unresolved latest version.
    pub fn version(&self) -> u64 {
        self.proto.version
    }

    /// Portable key for independently resolving credentials and base routing.
    pub fn access_context_key(&self) -> &str {
        &self.proto.access_context_key
    }

    /// Original protobuf manifest image, including the source's registered bases.
    pub fn serialized_manifest(&self) -> &[u8] {
        &self.proto.manifest
    }

    pub(crate) fn validate_dataset(&self, dataset: &Dataset) -> Result<()> {
        let location = dataset.manifest_location();
        let mut manifest =
            Manifest::try_from(table_pb::Manifest::decode(self.proto.manifest.clone())?)?;
        manifest.detach_sparse_inline_row_ids(self.proto.manifest.len());
        if self.proto.table_uri != canonical_uri(&dataset.refs.root()?.uri)?
            || self.proto.manifest_path != location.path.as_ref()
            || self.proto.manifest_identity != location.identity
            || !matches_loaded_manifest(&manifest, dataset.manifest())
        {
            return Err(Error::invalid_input(format!(
                "Blob source context {:?}/{} at {} does not match dataset snapshot {:?}/{} at {}",
                self.proto.branch,
                self.proto.version,
                self.proto.table_uri,
                dataset.manifest().branch,
                dataset.version().version,
                dataset.uri(),
            )));
        }
        Ok(())
    }

    pub(crate) fn reference(
        &self,
        object_uri: String,
        offset: u64,
        length: Option<u64>,
        kind: BlobKind,
        base_id: Option<u32>,
    ) -> Result<BlobRef> {
        let origin = match kind {
            BlobKind::External => {
                if !self
                    .proto
                    .required_capabilities
                    .contains(&IMMUTABLE_EXTERNAL)
                {
                    return Err(Error::invalid_input(
                        "Capturing an external blob requires the external immutability assumption",
                    ));
                }
                pb::object_locator::Origin::ExternalImmutable
            }
            BlobKind::Inline
            | BlobKind::Packed
            | BlobKind::Dedicated
            | BlobKind::Managed
            | BlobKind::ManagedWithBase => pb::object_locator::Origin::Lance,
        };
        BlobRef::try_new(
            0,
            pb::ObjectLocator {
                protocol_version: VERSION,
                source_id: Bytes::copy_from_slice(&self.source_id),
                object_uri: canonical_uri(&object_uri)?,
                offset,
                length,
                origin: origin as i32,
                base_id,
            }
            .encode_to_vec(),
            length,
        )
    }

    fn from_proto(proto: pb::SourceContext) -> Result<Self> {
        check_version(proto.protocol_version, "source context")?;
        if proto.required_capabilities != [OBJECT_RANGES]
            && proto.required_capabilities != [OBJECT_RANGES, IMMUTABLE_EXTERNAL]
        {
            return Err(Error::invalid_input(format!(
                "Unsupported required blob reference capabilities {:?}",
                proto.required_capabilities,
            )));
        }
        for (name, value) in [
            ("table_identity", &proto.table_identity),
            ("access_context_key", &proto.access_context_key),
            ("manifest_path", &proto.manifest_path),
        ] {
            if value.trim().is_empty() {
                return Err(Error::invalid_input(format!(
                    "Blob source {name} cannot be empty"
                )));
            }
        }
        if proto.table_uri != canonical_uri(&proto.table_uri)? {
            return Err(Error::invalid_input(format!(
                "Noncanonical table URI {}",
                proto.table_uri
            )));
        }
        Path::parse(&proto.manifest_path)?;
        if let Some(branch) = &proto.branch {
            check_valid_branch(branch).map_err(|error| {
                Error::invalid_input(format!("Invalid blob source branch {branch:?}: {error}",))
            })?;
        }
        if proto.version == 0 {
            return Err(Error::invalid_input(
                "Blob source version must be greater than 0, got 0",
            ));
        }
        if proto
            .manifest_identity
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(Error::invalid_input(
                "Blob source manifest identity cannot be empty",
            ));
        }
        let manifest = table_pb::Manifest::decode(proto.manifest.clone()).map_err(|error| {
            Error::invalid_input(format!("Invalid blob source manifest protobuf: {error}",))
        })?;
        if manifest.version != proto.version || manifest.branch != proto.branch {
            return Err(Error::invalid_input(format!(
                "Blob source snapshot {:?}/{} disagrees with manifest {:?}/{}",
                proto.branch, proto.version, manifest.branch, manifest.version,
            )));
        }
        let source_id = *blake3::hash(&proto.encode_to_vec()).as_bytes();
        let registered_bases = Arc::new(manifest.base_paths.iter().map(|base| base.id).collect());
        Ok(Self {
            proto,
            source_id,
            registered_bases,
        })
    }
}

/// One non-null, opaque blob reference. Null values are `None`, not empty locators.
///
/// Reference equality identifies a source and object range; it does not compare
/// payload contents. `source_context` is a batch-local slot and must be remapped
/// when combining independently produced batches. Use [`BlobRefBatch::remap_contexts`].
///
/// ```
/// # use lance::{Result, blob::reference::BlobRef};
/// # fn roundtrip(reference: &BlobRef) -> Result<()> {
/// let restored = BlobRef::try_new(
///     reference.source_context(),
///     reference.locator().to_vec(),
///     reference.length(),
/// )?;
/// assert_eq!(reference, &restored);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRef {
    source_context: u32,
    locator: Bytes,
    length: Option<u64>,
}

impl BlobRef {
    /// Reconstruct a transported reference from its opaque parts.
    ///
    /// `length` is exact when present, including `Some(0)` for an empty blob;
    /// `None` means an external whole object of unknown length, at object offset
    /// zero. The locator must be produced by Lance. Its embedded length must
    /// agree with `length`. Context membership is checked when adding the
    /// reference to a [`BlobRefBatch`].
    pub fn try_new(
        source_context: u32,
        locator: impl Into<Bytes>,
        length: Option<u64>,
    ) -> Result<Self> {
        let reference = Self {
            source_context,
            locator: locator.into(),
            length,
        };
        reference.object_locator()?;
        Ok(reference)
    }

    /// Batch-local context slot. It has no meaning outside its context table.
    pub fn source_context(&self) -> u32 {
        self.source_context
    }

    /// Opaque Lance locator bytes, suitable for lossless transport.
    pub fn locator(&self) -> &[u8] {
        &self.locator
    }

    /// Exact logical byte length, or `None` when the length is unknown.
    pub fn length(&self) -> Option<u64> {
        self.length
    }

    /// Validate a blob-local byte range without reading payloads.
    ///
    /// An empty range is valid at the end of a known-length blob. Unknown lengths
    /// still reject arithmetic overflow; a reader must check the remaining bounds.
    pub fn validate_range(&self, offset: u64, length: u64) -> Result<Range<u64>> {
        let end = offset.checked_add(length).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range offset {offset} + length {length} overflows u64",
            ))
        })?;
        if self.length.is_some_and(|size| end > size) {
            return Err(Error::invalid_input(format!(
                "Blob range {offset}..{end} exceeds blob length {:?}",
                self.length,
            )));
        }
        let object = self.object_locator()?;
        object.offset.checked_add(end).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob object offset {} + logical range end {end} overflows u64",
                object.offset,
            ))
        })?;
        Ok(offset..end)
    }

    fn object_locator(&self) -> Result<pb::ObjectLocator> {
        let object = pb::ObjectLocator::decode(self.locator.clone()).map_err(|error| {
            Error::invalid_input(format!("Invalid blob object locator protobuf: {error}",))
        })?;
        check_version(object.protocol_version, "object locator")?;
        if object.source_id.len() != 32 {
            return Err(Error::invalid_input(format!(
                "Blob locator source_id has {} bytes, expected 32",
                object.source_id.len(),
            )));
        }
        if canonical_uri(&object.object_uri)? != object.object_uri {
            return Err(Error::invalid_input(format!(
                "Noncanonical blob object URI {}",
                object.object_uri
            )));
        }
        if object.length != self.length {
            return Err(Error::invalid_input(format!(
                "Blob reference length {:?} disagrees with locator length {:?}",
                self.length, object.length,
            )));
        }
        match pb::object_locator::Origin::try_from(object.origin) {
            Ok(pb::object_locator::Origin::Lance) if object.length.is_some() => {}
            Ok(pb::object_locator::Origin::ExternalImmutable) => {}
            _ => {
                return Err(Error::invalid_input(format!(
                    "Unsupported blob locator origin {} or unknown length for a Lance object",
                    object.origin,
                )));
            }
        }
        if object.length.is_none() && object.offset != 0 {
            return Err(Error::invalid_input(format!(
                "Blob locator with unknown length must reference a whole external object at offset 0; got offset {}",
                object.offset,
            )));
        }
        if let Some(length) = object.length {
            object.offset.checked_add(length).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob locator offset {} + length {length} overflows u64",
                    object.offset,
                ))
            })?;
        }
        Ok(object)
    }
}

/// Arrow extension field for nullable execution references, distinct from BINARY payloads.
///
/// ```
/// use lance::blob::reference::blob_ref_field;
/// let field = blob_ref_field("image_ref");
/// assert!(field.is_nullable());
/// assert_eq!(field.metadata()["ARROW:extension:name"], "lance.blob.ref");
/// ```
pub fn blob_ref_field(name: impl Into<String>) -> Field {
    Field::new(name, DataType::Struct(REFERENCE_FIELDS.clone()), true).with_metadata(HashMap::from(
        [
            (ARROW_EXT_NAME_KEY.to_string(), EXTENSION_NAME.to_string()),
            (
                ARROW_EXT_META_KEY.to_string(),
                EXTENSION_METADATA.to_string(),
            ),
        ],
    ))
}

/// A validated nullable reference array together with its complete source-context table.
///
/// Rows preserve order, duplicates, and parent validity through serialization and
/// context remapping. The logical request index is the row position. Engines must
/// carry that index if execution later reorders completion.
///
/// ```
/// # use arrow_array::Array;
/// use lance::blob::reference::BlobRefBatch;
/// let nulls = BlobRefBatch::try_new(vec![], [None, None]).unwrap();
/// let restored = BlobRefBatch::decode(&nulls.encode().unwrap()).unwrap();
/// assert_eq!(restored.array().len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct BlobRefBatch {
    contexts: Vec<BlobSourceContext>,
    array: StructArray,
}

impl BlobRefBatch {
    /// Build an aligned reference array, validating every non-null source binding.
    pub fn try_new(
        contexts: Vec<BlobSourceContext>,
        references: impl IntoIterator<Item = Option<BlobRef>>,
    ) -> Result<Self> {
        let mut slots = UInt32Builder::new();
        let mut locators = LargeBinaryBuilder::new();
        let mut lengths = UInt64Builder::new();
        let mut validity = NullBufferBuilder::new(0);
        for reference in references {
            match reference {
                Some(reference) => {
                    slots.append_value(reference.source_context);
                    locators.append_value(&reference.locator);
                    lengths.append_option(reference.length);
                    validity.append_non_null();
                }
                None => {
                    slots.append_value(0);
                    locators.append_value([]);
                    lengths.append_null();
                    validity.append_null();
                }
            }
        }
        let field = blob_ref_field("blob_ref");
        let array = StructArray::try_new(
            REFERENCE_FIELDS.clone(),
            vec![
                Arc::new(slots.finish()),
                Arc::new(locators.finish()),
                Arc::new(lengths.finish()),
            ],
            validity.finish(),
        )?;
        Self::try_from_arrow(contexts, &field, array)
    }

    /// Validate an extension field and its Arrow array with the supplied context table.
    ///
    /// Ordinary binary arrays and untagged descriptor structs are not references.
    /// Null parent slots do not require a context or a locator.
    pub fn try_from_arrow(
        contexts: Vec<BlobSourceContext>,
        field: &Field,
        array: StructArray,
    ) -> Result<Self> {
        let expected = blob_ref_field(field.name());
        if field.data_type() != expected.data_type()
            || array.data_type() != expected.data_type()
            || !field.is_nullable()
            || field.metadata().get(ARROW_EXT_NAME_KEY)
                != expected.metadata().get(ARROW_EXT_NAME_KEY)
            || field.metadata().get(ARROW_EXT_META_KEY)
                != expected.metadata().get(ARROW_EXT_META_KEY)
        {
            return Err(Error::invalid_input(format!(
                "Invalid lance.blob.ref extension field {field:?}"
            )));
        }
        let mut ids = HashSet::with_capacity(contexts.len());
        for context in &contexts {
            if !ids.insert(context.source_id) {
                return Err(Error::invalid_input(
                    "Duplicate blob source contexts; intern contexts before building a batch",
                ));
            }
        }
        let batch = Self { contexts, array };
        for index in 0..batch.array.len() {
            if let Some(reference) = batch.reference(index)? {
                let context = batch.contexts.get(reference.source_context as usize).ok_or_else(|| Error::invalid_input(format!(
                    "Blob reference row {index} uses missing source_context slot {} (context count {})",
                    reference.source_context, batch.contexts.len(),
                )))?;
                let object = reference.object_locator()?;
                if object.source_id.as_ref() != context.source_id {
                    return Err(Error::invalid_input(format!(
                        "Blob reference row {index} does not match source_context slot {}",
                        reference.source_context,
                    )));
                }
                if let Some(base_id) = object.base_id
                    && !context.registered_bases.contains(base_id)
                {
                    return Err(Error::invalid_input(format!(
                        "Blob reference row {index} references unknown base_id {base_id}",
                    )));
                }
                if object.origin == pb::object_locator::Origin::ExternalImmutable as i32
                    && !context
                        .proto
                        .required_capabilities
                        .contains(&IMMUTABLE_EXTERNAL)
                {
                    return Err(Error::invalid_input(format!(
                        "Blob reference row {index} requires the external immutability assumption",
                    )));
                }
            }
        }
        Ok(batch)
    }

    /// Complete context table used by this batch's local slots.
    pub fn contexts(&self) -> &[BlobSourceContext] {
        &self.contexts
    }

    /// Nullable extension storage array. Use [`blob_ref_field`] when transporting it.
    pub fn array(&self) -> &StructArray {
        &self.array
    }

    /// Return one reference or SQL null at a checked row position.
    pub fn reference(&self, index: usize) -> Result<Option<BlobRef>> {
        if index >= self.array.len() {
            return Err(Error::invalid_input(format!(
                "Blob reference index {index} exceeds row count {}",
                self.array.len(),
            )));
        }
        if self.array.is_null(index) {
            return Ok(None);
        }
        let slots = self.array.column(0).as_primitive::<UInt32Type>();
        let locators = self.array.column(1).as_binary::<i64>();
        let lengths = self.array.column(2).as_primitive::<UInt64Type>();
        if slots.is_null(index) || locators.is_null(index) {
            return Err(Error::invalid_input(format!(
                "Non-null blob reference row {index} has a null source_context or locator",
            )));
        }
        Ok(Some(BlobRef {
            source_context: slots.value(index),
            locator: Bytes::copy_from_slice(locators.value(index)),
            length: lengths.is_valid(index).then(|| lengths.value(index)),
        }))
    }

    /// Remap this array into a shared context table when combining batches.
    ///
    /// A missing source fails rather than reusing an unrelated numeric slot.
    pub fn remap_contexts(&self, contexts: Vec<BlobSourceContext>) -> Result<Self> {
        let by_id: HashMap<_, _> = contexts
            .iter()
            .enumerate()
            .map(|(index, context)| (context.source_id, index))
            .collect();
        let references = (0..self.array.len())
            .map(|index| {
                let Some(mut reference) = self.reference(index)? else {
                    return Ok(None);
                };
                let source = &self.contexts[reference.source_context as usize];
                let slot = by_id.get(&source.source_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "New context table is missing the source of blob reference row {index}",
                    ))
                })?;
                reference.source_context = u32::try_from(*slot).map_err(|_| {
                    Error::invalid_input(format!("Blob source context slot {slot} exceeds u32",))
                })?;
                Ok(Some(reference))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new(contexts, references)
    }

    /// Serialize contexts and one Arrow IPC reference batch for another process.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let schema = Arc::new(Schema::new(vec![blob_ref_field("blob_ref")]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(self.array.clone())])?;
        let mut ipc = Vec::new();
        let mut writer = StreamWriter::try_new(&mut ipc, schema.as_ref())?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(pb::ReferenceBatch {
            protocol_version: VERSION,
            contexts: self
                .contexts
                .iter()
                .map(|context| context.proto.clone())
                .collect(),
            arrow_ipc: ipc.into(),
        }
        .encode_to_vec())
    }

    /// Deserialize and validate a complete execution message before any payload I/O.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let proto = pb::ReferenceBatch::decode(bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid blob reference batch protobuf: {error}",))
        })?;
        check_version(proto.protocol_version, "reference batch")?;
        let contexts = proto
            .contexts
            .into_iter()
            .map(BlobSourceContext::from_proto)
            .collect::<Result<Vec<_>>>()?;
        let mut reader = StreamReader::try_new(Cursor::new(proto.arrow_ipc), None)?;
        let batch = reader
            .next()
            .transpose()?
            .ok_or_else(|| Error::invalid_input("Missing blob reference Arrow batch"))?;
        if batch.num_columns() != 1 || reader.next().transpose()?.is_some() {
            return Err(Error::invalid_input(
                "Blob reference transport requires exactly one Arrow batch and one column",
            ));
        }
        let array = batch.column(0).as_struct_opt().ok_or_else(|| {
            Error::invalid_input("Blob reference extension storage must be a struct")
        })?;
        Self::try_from_arrow(contexts, batch.schema().field(0), array.clone())
    }
}

/// Compare a decoded manifest with a loaded snapshot, ignoring hydrated dictionary values.
fn matches_loaded_manifest(observed: &Manifest, loaded: &Manifest) -> bool {
    // Writes and session caches populate dictionary values that are not part of
    // the manifest schema's persisted dictionary metadata.
    let mut expected = loaded.clone();
    let mut fields = expected.schema.fields.iter_mut().collect::<Vec<_>>();
    while let Some(field) = fields.pop() {
        if let Some(dictionary) = &mut field.dictionary {
            dictionary.values = None;
        }
        fields.extend(&mut field.children);
    }
    observed == &expected
}

fn check_version(version: u32, component: &str) -> Result<()> {
    if version != VERSION {
        return Err(Error::invalid_input(format!(
            "Unsupported blob reference {component} protocol version {version}; supported version is {VERSION}",
        )));
    }
    Ok(())
}

fn canonical_uri(uri: &str) -> Result<String> {
    if url::Url::parse(uri).is_err() && !std::path::Path::new(uri).is_absolute() {
        return Err(Error::invalid_input(format!(
            "Blob reference URI must be absolute: {uri}"
        )));
    }
    let url = uri_to_url(uri)?;
    let has_username = !url.username().is_empty() && !matches!(url.scheme(), "az" | "abfss");
    if url.fragment().is_some() || has_username || url.password().is_some() {
        return Err(Error::invalid_input(
            "Blob reference URI must not contain credentials or a fragment",
        ));
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    use arrow_array::{
        DictionaryArray, RecordBatchIterator, StringArray, UInt8Array, UInt32Array, UInt64Array,
    };
    use lance_core::datatypes::{BlobHandling, BlobKind};
    use lance_core::utils::address::RowAddress;
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::BasePath;
    use object_store::ObjectStoreExt;
    use rstest::rstest;

    use crate::blob::{
        BlobArrayBuilder, BlobDescriptor, BlobDescriptorArrayBuilder, BlobFieldOptions, blob_field,
        blob_field_with_options,
    };
    use crate::dataset::WriteParams;
    use crate::dataset::blob::test_util::{create_multi_base_blob_v2_fixture, managed_blob_schema};
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};

    #[cfg_attr(coverage, coverage(off))]
    fn context(version: u64, branch: Option<&str>) -> BlobSourceContext {
        let manifest = table_pb::Manifest {
            version,
            branch: branch.map(str::to_string),
            transaction_file: format!("0-transaction-{version}.txn"),
            ..Default::default()
        };
        BlobSourceContext::from_proto(pb::SourceContext {
            protocol_version: VERSION,
            required_capabilities: vec![OBJECT_RANGES, IMMUTABLE_EXTERNAL],
            table_identity: "catalog/table-incarnation".to_string(),
            table_uri: "s3://source/table.lance".to_string(),
            branch: branch.map(str::to_string),
            version,
            manifest_path: format!("table.lance/_versions/{version}.manifest"),
            manifest: manifest.encode_to_vec().into(),
            manifest_identity: None,
            access_context_key: "query/authenticated-storage".to_string(),
        })
        .unwrap()
    }

    #[cfg_attr(coverage, coverage(off))]
    fn reference(context: &BlobSourceContext, slot: u32, length: Option<u64>) -> BlobRef {
        let locator = pb::ObjectLocator {
            protocol_version: VERSION,
            source_id: Bytes::copy_from_slice(context.source_id()),
            object_uri: "s3://source/table.lance/_blobs/payload.blob".to_string(),
            offset: if length.is_some() { 1024 } else { 0 },
            length,
            origin: if length.is_some() {
                pb::object_locator::Origin::Lance
            } else {
                pb::object_locator::Origin::ExternalImmutable
            } as i32,
            base_id: None,
        };
        BlobRef::try_new(slot, locator.encode_to_vec(), length).unwrap()
    }

    #[cfg_attr(coverage, coverage(off))]
    fn mixed_batch() -> BlobRefBatch {
        let main = context(1, None);
        let later = context(2, None);
        let branch = context(1, Some("experiment"));
        let first = reference(&main, 0, Some(3));
        let empty = reference(&later, 1, Some(0));
        let unknown = reference(&branch, 2, None);
        BlobRefBatch::try_new(
            vec![main, later, branch],
            [
                Some(first.clone()),
                None,
                Some(empty.clone()),
                Some(unknown),
                Some(first),
                Some(empty),
                None,
            ],
        )
        .unwrap()
    }

    #[test]
    fn roundtrip_and_remap_contexts() {
        let batch = mixed_batch();
        let restored = BlobRefBatch::decode(&batch.encode().unwrap()).unwrap();
        assert_eq!(batch.contexts(), restored.contexts());
        assert_eq!(batch.array(), restored.array());
        assert_eq!(restored.reference(1).unwrap(), None);
        assert_eq!(restored.reference(2).unwrap().unwrap().length(), Some(0));
        assert_eq!(restored.reference(3).unwrap().unwrap().length(), None);
        assert_eq!(
            restored.reference(0).unwrap(),
            restored.reference(4).unwrap()
        );
        assert_eq!(
            restored.reference(2).unwrap(),
            restored.reference(5).unwrap()
        );

        let contexts = vec![
            batch.contexts[2].clone(),
            batch.contexts[0].clone(),
            batch.contexts[1].clone(),
        ];
        let remapped = batch.remap_contexts(contexts).unwrap();
        for (index, slot) in [(0, 1), (2, 2), (3, 0), (4, 1), (5, 2)] {
            let old = batch.reference(index).unwrap().unwrap();
            let new = remapped.reference(index).unwrap().unwrap();
            assert_eq!(new.source_context(), slot);
            assert_eq!(new.locator(), old.locator());
            assert_eq!(new.length(), old.length());
        }
        assert!(remapped.array().is_null(1));
        let error = batch.remap_contexts(vec![]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("missing the source"));
    }

    #[rstest]
    #[case::empty(0)]
    #[case::all_null(3)]
    fn null_only_batches(#[case] count: usize) {
        let batch = BlobRefBatch::try_new(vec![], (0..count).map(|_| None)).unwrap();
        let restored = BlobRefBatch::decode(&batch.encode().unwrap()).unwrap();
        assert_eq!(restored.array().len(), count);
        assert_eq!(restored.array().null_count(), count);
        assert!(restored.contexts().is_empty());
    }

    #[rstest]
    #[case::missing(vec![], 0, "missing source_context")]
    #[case::wrong_snapshot(vec![context(2, None)], 0, "does not match")]
    #[case::wrong_branch(vec![context(1, Some("experiment"))], 0, "does not match")]
    #[case::out_of_range(vec![context(1, None)], 1, "missing source_context")]
    #[case::duplicate(vec![context(1, None), context(1, None)], 0, "Duplicate blob source")]
    fn rejects_wrong_contexts(
        #[case] contexts: Vec<BlobSourceContext>,
        #[case] slot: u32,
        #[case] message: &str,
    ) {
        let reference = reference(&context(1, None), slot, Some(3));
        let error = BlobRefBatch::try_new(contexts, [Some(reference)]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::table_identity(|p: &mut pb::SourceContext| p.table_identity = "catalog/other-incarnation".to_string())]
    #[case::storage_authority(|p: &mut pb::SourceContext| p.access_context_key = "other-storage-binding".to_string())]
    fn rejects_different_table_or_storage_authority(#[case] mutate: fn(&mut pb::SourceContext)) {
        let original = context(1, None);
        let reference = reference(&original, 0, Some(3));
        let mut proto = original.proto.clone();
        mutate(&mut proto);
        let other = BlobSourceContext::from_proto(proto).unwrap();
        assert_eq!(original.table_uri(), other.table_uri());
        assert_eq!(original.version(), other.version());
        assert_eq!(original.serialized_manifest(), other.serialized_manifest());
        let error = BlobRefBatch::try_new(vec![other], [Some(reference)]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("does not match source_context"));
    }

    #[rstest]
    #[case::version(|p: &mut pb::SourceContext| p.protocol_version = 2, "protocol version 2")]
    #[case::capability(|p: &mut pb::SourceContext| p.required_capabilities.push(999), "capabilities")]
    #[case::missing_capability(|p: &mut pb::SourceContext| p.required_capabilities.clear(), "capabilities")]
    #[case::identity(|p: &mut pb::SourceContext| p.table_identity.clear(), "table_identity")]
    #[case::access(|p: &mut pb::SourceContext| p.access_context_key.clear(), "access_context_key")]
    #[case::uri(|p: &mut pb::SourceContext| p.table_uri = "relative/table".to_string(), "must be absolute")]
    #[case::main_alias(|p: &mut pb::SourceContext| p.branch = Some("main".to_string()), "branch")]
    #[case::version_mismatch(|p: &mut pb::SourceContext| p.version = 9, "disagrees with manifest")]
    #[case::branch_mismatch(|p: &mut pb::SourceContext| p.branch = Some("other".to_string()), "disagrees with manifest")]
    #[case::empty_manifest_identity(|p: &mut pb::SourceContext| p.manifest_identity = Some(String::new()), "manifest identity")]
    fn rejects_invalid_source_context(
        #[case] mutate: fn(&mut pb::SourceContext),
        #[case] message: &str,
    ) {
        let mut proto = context(1, None).proto;
        mutate(&mut proto);
        let error = BlobSourceContext::decode(&proto.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::version(|p: &mut pb::ObjectLocator| p.protocol_version = 3, "protocol version 3")]
    #[case::source_id(|p: &mut pb::ObjectLocator| p.source_id = Bytes::new(), "source_id")]
    #[case::uri(|p: &mut pb::ObjectLocator| p.object_uri = "relative/object".to_string(), "must be absolute")]
    #[case::origin(|p: &mut pb::ObjectLocator| p.origin = 99, "origin 99")]
    #[case::length(|p: &mut pb::ObjectLocator| p.length = Some(9), "disagrees with locator length")]
    #[case::overflow(|p: &mut pb::ObjectLocator| p.offset = u64::MAX, "overflows u64")]
    fn rejects_invalid_locator(#[case] mutate: fn(&mut pb::ObjectLocator), #[case] message: &str) {
        let reference = reference(&context(1, None), 0, Some(3));
        let mut locator = reference.object_locator().unwrap();
        mutate(&mut locator);
        let error = BlobRef::try_new(0, locator.encode_to_vec(), Some(3)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn rejects_unknown_length_with_nonzero_offset() {
        let reference = reference(&context(1, None), 0, None);
        let mut locator = reference.object_locator().unwrap();
        locator.offset = 1024;
        let error = BlobRef::try_new(0, locator.encode_to_vec(), None).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("unknown length"), "{error}");
        assert!(error.to_string().contains("got offset 1024"), "{error}");
    }

    #[test]
    fn external_immutability_is_explicit() {
        let mut proto = context(1, None).proto;
        proto.required_capabilities = vec![OBJECT_RANGES];
        let context = BlobSourceContext::from_proto(proto).unwrap();
        let external = reference(&context, 0, None);
        let error = BlobRefBatch::try_new(vec![context.clone()], [Some(external)]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("external immutability assumption")
        );
        let explicit = context.with_external_immutability_assumption();
        let external = reference(&explicit, 0, None);
        BlobRefBatch::try_new(vec![explicit], [Some(external)]).unwrap();
    }

    #[test]
    fn validates_logical_ranges() {
        let context = context(1, None);
        let known = reference(&context, 0, Some(3));
        assert_eq!(known.validate_range(1, 2).unwrap(), 1..3);
        assert_eq!(known.validate_range(3, 0).unwrap(), 3..3);
        let error = known.validate_range(2, 2).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("exceeds blob length"));
        let unknown = reference(&context, 0, None);
        assert_eq!(unknown.validate_range(500, 2).unwrap(), 500..502);
        let error = unknown.validate_range(u64::MAX, 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("overflows u64"));
        assert_eq!(
            unknown.validate_range(u64::MAX, 0).unwrap(),
            u64::MAX..u64::MAX
        );
    }

    #[test]
    fn rejects_untagged_arrays_and_unknown_envelopes() {
        let batch = mixed_batch();
        let field = blob_ref_field("blob_ref").with_metadata(HashMap::new());
        let error =
            BlobRefBatch::try_from_arrow(batch.contexts.clone(), &field, batch.array.clone())
                .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("extension field"));
        let mut proto = pb::ReferenceBatch::decode(batch.encode().unwrap().as_slice()).unwrap();
        proto.protocol_version = 99;
        proto.arrow_ipc = Bytes::new();
        let error = BlobRefBatch::decode(&proto.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("protocol version 99"));
    }

    #[rstest]
    #[case::object_key("s3://source/path/with%20space/")]
    #[case::azure("abfss://filesystem@account.dfs.core.windows.net/path/blob")]
    #[case::azure_az("az://container@account/path/blob")]
    fn preserves_object_uri(#[case] uri: &str) {
        assert_eq!(canonical_uri(uri).unwrap(), uri);
    }

    #[test]
    fn transport_in_another_process() {
        const INPUT: &str = "LANCE_BLOB_REFERENCE_TEST_INPUT";
        if let Ok(path) = std::env::var(INPUT) {
            let bytes = std::fs::read(path).unwrap();
            let restored = BlobRefBatch::decode(&bytes).unwrap();
            let expected = mixed_batch();
            assert_eq!(expected.array(), restored.array());
            assert_eq!(expected.contexts(), restored.contexts());
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("references.bin");
        std::fs::write(&path, mixed_batch().encode().unwrap()).unwrap();
        let this_test = std::thread::current().name().unwrap().to_string();
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &this_test, "--nocapture"])
            .env(INPUT, &path)
            .output()
            .unwrap();
        assert!(output.status.success(), "Child failed: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "Child did not run {this_test}: {output:?}"
        );
    }

    #[tokio::test]
    async fn captures_versions_branches_and_recreated_tables() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("table.lance");
        let uri = path.to_str().unwrap();
        let mut blobs = BlobArrayBuilder::new(4);
        blobs.push_bytes(b"abc").unwrap();
        blobs.push_null().unwrap();
        blobs.push_empty().unwrap();
        blobs.push_bytes(b"xyz").unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
            Field::new(
                "category",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
                blobs.finish().unwrap(),
                Arc::new(DictionaryArray::<UInt32Type>::new(
                    UInt32Array::from(vec![0, 1, 0, 1]),
                    Arc::new(StringArray::from(vec!["first", "second"])),
                )),
            ],
        )
        .unwrap();
        let params = WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: true,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let original = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
            uri,
            Some(params.clone()),
        )
        .await
        .unwrap();
        assert_eq!(original.get_fragments().len(), 2);
        assert!(
            original
                .schema()
                .field("category")
                .unwrap()
                .dictionary
                .as_ref()
                .unwrap()
                .values
                .is_some()
        );
        let first = BlobSourceContext::from_dataset(&original, "catalog/first", "workers")
            .await
            .unwrap();
        let repeated = BlobSourceContext::from_dataset(&original, "catalog/first", "workers")
            .await
            .unwrap();
        assert_eq!(first, repeated);
        let cached = DatasetBuilder::from_uri(uri)
            .with_session(original.session.clone())
            .load()
            .await
            .unwrap();
        let reopened = Dataset::open(uri).await.unwrap();
        for dataset in [&original, &cached, &reopened] {
            let context = BlobSourceContext::from_dataset(dataset, "catalog/first", "workers")
                .await
                .unwrap();
            assert_eq!(first, context);
            BlobRefCapture::try_new(
                Arc::new(dataset.clone()),
                context,
                dataset.schema().field("blob").unwrap().id,
            )
            .unwrap();
        }
        let mut latest = original.clone();
        latest
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
                Some(params.clone()),
            )
            .await
            .unwrap();
        let later = BlobSourceContext::from_dataset(&latest, "catalog/first", "workers")
            .await
            .unwrap();
        assert_eq!(first.table_uri(), later.table_uri());
        assert_ne!(first.version(), later.version());
        assert_ne!(first.source_id(), later.source_id());
        let branch = latest
            .create_branch("experiment", original.manifest().version, None)
            .await
            .unwrap();
        let branched = BlobSourceContext::from_dataset(&branch, "catalog/first", "workers")
            .await
            .unwrap();
        assert_eq!(branched.branch(), Some("experiment"));
        assert_eq!(first.table_uri(), branched.table_uri());
        assert_eq!(first.version(), branched.version());
        assert_ne!(first.source_id(), branched.source_id());

        std::fs::remove_dir_all(&path).unwrap();
        let recreated = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            uri,
            Some(params),
        )
        .await
        .unwrap();
        let replacement = BlobSourceContext::from_dataset(&recreated, "catalog/second", "workers")
            .await
            .unwrap();
        assert_eq!(first.table_uri(), replacement.table_uri());
        assert_eq!(first.version(), replacement.version());
        assert_ne!(first.source_id(), replacement.source_id());
        let error = BlobSourceContext::from_dataset(&original, "catalog/first", "workers")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("no longer matches"));
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn capture_dataset(uri: &str) -> Arc<Dataset> {
        let mut blobs = BlobArrayBuilder::new(6);
        for bytes in [
            Some(b"abc".as_slice()),
            Some(b"abcde"),
            Some(b"abcdefghij"),
            None,
            Some(b""),
            Some(b"uv"),
        ] {
            if let Some(bytes) = bytes {
                blobs.push_bytes(bytes).unwrap();
            } else {
                blobs.push_null().unwrap();
            }
        }
        let schema = Arc::new(Schema::new(vec![blob_field_with_options(
            "blob",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(4)
                .with_dedicated_size_threshold(std::num::NonZeroUsize::new(8).unwrap()),
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                uri,
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn capture_external_dataset(
        directory: &std::path::Path,
        range: Option<crate::blob::BlobRange>,
        is_registered: bool,
    ) -> (Arc<Dataset>, std::path::PathBuf) {
        let external = directory.join("external with space.bin");
        std::fs::write(&external, b"abcdefgh").unwrap();
        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .push(BlobDescriptor::External {
                base_id: if is_registered { 1 } else { 0 },
                uri: if is_registered {
                    "external with space.bin".to_string()
                } else {
                    url::Url::from_file_path(&external).unwrap().to_string()
                },
                offset: range.map_or(0, |range| range.offset),
                size: range.map_or(0, |range| range.size),
            })
            .unwrap();
        let (field, array) = builder.finish().unwrap().into_parts();
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                directory.join("table").to_str().unwrap(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    allow_external_blob_outside_bases: true,
                    initial_bases: is_registered.then(|| {
                        vec![BasePath::new(
                            1,
                            url::Url::from_directory_path(directory).unwrap().into(),
                            None,
                            false,
                        )]
                    }),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        (dataset, external)
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn capture_managed_dataset(
        directory: &std::path::Path,
        base_id: Option<u32>,
    ) -> (Arc<Dataset>, std::path::PathBuf) {
        let base = directory.join("base with space");
        let object_path = base.join("_blobs/managed.bin");
        std::fs::create_dir_all(base.join("_blobs")).unwrap();
        std::fs::write(&object_path, b"abcdefgh").unwrap();
        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .push(BlobDescriptor::Managed {
                base_id,
                uri: "_blobs/managed.bin".to_string(),
                offset: 2,
                size: 3,
            })
            .unwrap();
        let (field, array) = builder.finish().unwrap().into_parts();
        let schema = Arc::new(managed_blob_schema(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                directory.join("table").to_str().unwrap(),
                Some(WriteParams {
                    initial_bases: Some(vec![BasePath::new(
                        base_id.unwrap_or(1),
                        url::Url::from_directory_path(&base).unwrap().into(),
                        None,
                        true,
                    )]),
                    target_bases: base_id.is_none().then_some(vec![1]),
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        (dataset, object_path)
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn capture_inputs(dataset: &Dataset) -> (Field, Vec<RowAddress>, StructArray) {
        let batch = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .blob_handling(BlobHandling::BlobsDescriptions)
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let field = batch.schema().field_with_name("blob").unwrap().clone();
        let addresses = batch[lance_core::ROW_ADDR]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect();
        (field, addresses, batch["blob"].as_struct().clone())
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn read_captured(dataset: &Dataset, reference: &BlobRef) -> Bytes {
        let object = reference.object_locator().unwrap();
        let params = dataset.store_params.as_deref().cloned().unwrap_or_default();
        let (store, path) = if let Some(base_id) = object.base_id {
            (
                dataset.object_store(Some(base_id)).await.unwrap(),
                ObjectStore::extract_path_from_uri(
                    dataset.session.store_registry(),
                    &object.object_uri,
                )
                .unwrap(),
            )
        } else if object.origin == pb::object_locator::Origin::Lance as i32 {
            // Resolve the primary source through its authenticated context.
            // Reopening memory:// by URI would create a different storage authority.
            (
                dataset.object_store.clone(),
                ObjectStore::extract_path_from_uri(
                    dataset.session.store_registry(),
                    &object.object_uri,
                )
                .unwrap(),
            )
        } else {
            ObjectStore::from_uri_and_params(
                dataset.session.store_registry(),
                &object.object_uri,
                &params,
            )
            .await
            .unwrap()
        };
        let length = match object.length {
            Some(length) => length,
            None => store.size(&path).await.unwrap(),
        };
        if length == 0 {
            return Bytes::new();
        }
        let before = store.io_stats_snapshot().requests.len();
        let bytes = store
            .inner
            .get_range(&path, object.offset..object.offset + length)
            .await
            .unwrap();
        let stats = store.io_stats_snapshot();
        let requests = &stats.requests[before..];
        assert_eq!(
            requests.len(),
            1,
            "payload reads must not reread descriptors: {requests:?}"
        );
        assert_eq!(requests[0].path, path);
        assert_eq!(
            requests[0].range,
            Some(object.offset..object.offset + length)
        );
        bytes
    }

    #[tokio::test]
    async fn capture_roundtrip_without_payload_or_descriptor_reads() {
        let dataset = capture_dataset("memory://").await;
        assert_eq!(dataset.get_fragments().len(), 2);
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&dataset).await;
        let indices = UInt32Array::from(vec![2, 0, 4, 3, 1, 2, 5]);
        let selected = arrow::compute::take(&descriptors, &indices, None).unwrap();
        let addresses = indices
            .values()
            .iter()
            .map(|index| addresses[*index as usize])
            .collect::<Vec<_>>();
        let mut capture = BlobRefCapture::try_new(
            dataset.clone(),
            source.clone(),
            dataset.schema().field("blob").unwrap().id,
        )
        .unwrap();
        let before = dataset.object_store.io_stats_snapshot();
        let batch = capture
            .capture(&source, &field, &addresses, selected.as_struct())
            .await
            .unwrap();
        // Reusing the session must not introduce descriptor or payload lookups.
        let repeated = capture
            .capture(&source, &field, &addresses, selected.as_struct())
            .await
            .unwrap();
        let restored = BlobRefBatch::decode(&batch.encode().unwrap()).unwrap();
        assert_eq!(batch.array(), repeated.array());
        assert_eq!(batch.array(), restored.array());
        assert_eq!(
            dataset.object_store.io_stats_snapshot().requests.len(),
            before.requests.len()
        );
        for (row, expected) in [
            Some(b"abcdefghij".as_slice()),
            Some(b"abc"),
            Some(b""),
            None,
            Some(b"abcde"),
            Some(b"abcdefghij"),
            Some(b"uv"),
        ]
        .into_iter()
        .enumerate()
        {
            match (restored.reference(row).unwrap(), expected) {
                (Some(reference), Some(expected)) => {
                    assert_eq!(reference.length(), Some(expected.len() as u64));
                    assert_eq!(read_captured(&dataset, &reference).await.as_ref(), expected);
                }
                (None, None) => {}
                other => panic!("Unexpected captured value {other:?}"),
            }
        }
    }

    #[rstest]
    #[case::inline(3, None, true)]
    #[case::packed(65_537, None, true)]
    #[case::dedicated(65_537, Some(65_536), true)]
    #[case::data_only(65_537, None, false)]
    #[tokio::test]
    async fn capture_non_default_bases(
        #[case] size: usize,
        #[case] dedicated: Option<usize>,
        #[case] is_dataset_root: bool,
    ) {
        let fixture =
            create_multi_base_blob_v2_fixture(vec![7; size], dedicated, is_dataset_root).await;
        let source = BlobSourceContext::from_dataset(&fixture.dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&fixture.dataset).await;
        let id = fixture.dataset.schema().field("blob").unwrap().id;
        let mut capture =
            BlobRefCapture::try_new(fixture.dataset.clone(), source.clone(), id).unwrap();
        let captured = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        let captured = BlobRefBatch::decode(&captured.encode().unwrap()).unwrap();
        let reference = captured.reference(0).unwrap().unwrap();
        let object = reference.object_locator().unwrap();
        assert_eq!(object.base_id, Some(1));
        assert!(object.object_uri.contains("/blob_base/"));
        assert_eq!(
            read_captured(&fixture.dataset, &reference).await.as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[rstest]
    #[case::whole_object(None, false)]
    #[case::range(Some(crate::blob::BlobRange { offset: 2, size: 3 }), false)]
    #[case::registered_whole_object(None, true)]
    #[case::registered_range(Some(crate::blob::BlobRange { offset: 2, size: 3 }), true)]
    #[tokio::test]
    async fn capture_external_preserves_unknown_lengths(
        #[case] range: Option<crate::blob::BlobRange>,
        #[case] is_registered: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let (dataset, external) =
            capture_external_dataset(directory.path(), range, is_registered).await;
        let uri = url::Url::from_file_path(&external).unwrap().to_string();
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&dataset).await;
        let id = dataset.schema().field("blob").unwrap().id;
        // Capture must also work without access to external object metadata.
        std::fs::remove_file(&external).unwrap();
        let mut capture = BlobRefCapture::try_new(dataset.clone(), source.clone(), id).unwrap();
        let error = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("external immutability assumption")
        );
        let source = source.with_external_immutability_assumption();
        let mut capture = BlobRefCapture::try_new(dataset.clone(), source.clone(), id).unwrap();
        let before = dataset.object_store.io_stats_snapshot();
        let references = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        assert_eq!(
            dataset.object_store.io_stats_snapshot().requests.len(),
            before.requests.len()
        );
        let reference = references.reference(0).unwrap().unwrap();
        let object = reference.object_locator().unwrap();
        assert_eq!(object.object_uri, uri);
        assert_eq!(object.base_id, is_registered.then_some(1));
        assert_eq!(object.offset, range.map_or(0, |range| range.offset));
        assert_eq!(reference.length(), range.map(|range| range.size));
        std::fs::write(&external, b"abcdefgh").unwrap();
        if range.is_some() {
            assert_eq!(read_captured(&dataset, &reference).await.as_ref(), b"cde");
        } else {
            assert_eq!(
                read_captured(&dataset, &reference).await.as_ref(),
                b"abcdefgh"
            );
        }
    }

    #[rstest]
    #[case::inherited(None)]
    #[case::explicit(Some(1))]
    #[tokio::test]
    async fn capture_managed_bases(#[case] base_id: Option<u32>) {
        let directory = tempfile::tempdir().unwrap();
        let (dataset, object_path) = capture_managed_dataset(directory.path(), base_id).await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&dataset).await;
        let mut capture = BlobRefCapture::try_new(
            dataset.clone(),
            source.clone(),
            dataset.schema().field("blob").unwrap().id,
        )
        .unwrap();
        let references = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        let references = BlobRefBatch::decode(&references.encode().unwrap()).unwrap();
        let reference = references.reference(0).unwrap().unwrap();
        assert_eq!(read_captured(&dataset, &reference).await.as_ref(), b"cde");
        let object = reference.object_locator().unwrap();
        assert_eq!(object.base_id, Some(base_id.unwrap_or(1)));
        assert_eq!(
            object.object_uri,
            url::Url::from_file_path(object_path).unwrap().as_str()
        );
    }

    #[tokio::test]
    async fn capture_refs_survive_compaction_and_shallow_clone() {
        let directory = tempfile::tempdir().unwrap();
        let dataset = capture_dataset(directory.path().join("source").to_str().unwrap()).await;
        dataset
            .tags()
            .create("captured", dataset.manifest().version)
            .await
            .unwrap();
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/source", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&dataset).await;
        let field_id = dataset.schema().field("blob").unwrap().id;
        let mut capture =
            BlobRefCapture::try_new(dataset.clone(), source.clone(), field_id).unwrap();
        let captured = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        let mut latest = dataset.as_ref().clone();
        let cloned = Arc::new(
            latest
                .shallow_clone(
                    directory.path().join("clone").to_str().unwrap(),
                    source.version(),
                    None,
                )
                .await
                .unwrap(),
        );
        let cloned_source = BlobSourceContext::from_dataset(&cloned, "catalog/clone", "workers")
            .await
            .unwrap();
        let (clone_field, clone_addresses, clone_descriptors) = capture_inputs(&cloned).await;
        let mut clone_capture =
            BlobRefCapture::try_new(cloned.clone(), cloned_source.clone(), field_id).unwrap();
        let clone_refs = clone_capture
            .capture(
                &cloned_source,
                &clone_field,
                &clone_addresses,
                &clone_descriptors,
            )
            .await
            .unwrap();
        for row in [0, 1, 2, 4, 5] {
            assert_eq!(
                captured
                    .reference(row)
                    .unwrap()
                    .unwrap()
                    .object_locator()
                    .unwrap()
                    .object_uri,
                clone_refs
                    .reference(row)
                    .unwrap()
                    .unwrap()
                    .object_locator()
                    .unwrap()
                    .object_uri
            );
        }
        compact_files(
            &mut latest,
            CompactionOptions {
                target_rows_per_fragment: 100,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(latest.manifest().version > source.version());
        let error = BlobRefCapture::try_new(Arc::new(latest), source.clone(), field_id)
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot")
        );
        let restored = BlobRefBatch::decode(&captured.encode().unwrap()).unwrap();
        for (row, expected) in [(0, b"abc".as_slice()), (1, b"abcde"), (2, b"abcdefghij")] {
            assert_eq!(
                read_captured(&dataset, &restored.reference(row).unwrap().unwrap())
                    .await
                    .as_ref(),
                expected
            );
            assert_eq!(
                read_captured(&cloned, &clone_refs.reference(row).unwrap().unwrap())
                    .await
                    .as_ref(),
                expected
            );
        }
    }

    #[rstest]
    #[case::wrong_source("source", "scan source does not match")]
    #[case::wrong_field("field", "expected descriptor field")]
    #[case::wrong_layout("layout", "expected descriptor field")]
    #[case::wrong_descriptor_shape("shape", "expected descriptor field")]
    #[case::cardinality("cardinality", "descriptor count")]
    #[case::missing_fragment("fragment", "missing fragment")]
    #[case::bad_offset("offset", "invalid address")]
    #[case::null_child("null_child", "null children")]
    #[case::unknown_kind("kind", "Unknown blob kind")]
    #[case::overflow("overflow", "overflows u64")]
    #[case::dedicated_offset("dedicated", "nonzero position")]
    #[tokio::test]
    async fn capture_rejects_invalid_inputs(#[case] invalid: &str, #[case] message: &str) {
        let dataset = capture_dataset("memory://").await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (mut field, addresses, descriptors) = capture_inputs(&dataset).await;
        let mut addresses = addresses[..1].to_vec();
        let mut descriptors = descriptors.slice(0, 1);
        let mut scan_source = source.clone();
        let mut arrays = descriptors.columns().to_vec();
        match invalid {
            "source" => scan_source = source.clone().with_external_immutability_assumption(),
            "field" => field = field.with_name("other"),
            "layout" => field = field.with_metadata(HashMap::new()),
            "shape" => {}
            "cardinality" => addresses.clear(),
            "fragment" => addresses[0] = RowAddress::new_from_parts(999, 0),
            "offset" => addresses[0] = RowAddress::new_from_parts(0, 999),
            "null_child" => arrays[1] = Arc::new(UInt64Array::from(vec![None])),
            "kind" => arrays[0] = Arc::new(UInt8Array::from(vec![255])),
            "overflow" => arrays[1] = Arc::new(UInt64Array::from(vec![u64::MAX])),
            "dedicated" => {
                arrays[0] = Arc::new(UInt8Array::from(vec![BlobKind::Dedicated as u8]));
                arrays[1] = Arc::new(UInt64Array::from(vec![1]));
            }
            _ => unreachable!(),
        }
        let mut fields = descriptors
            .fields()
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>();
        if invalid == "shape" {
            fields[1] = fields[1].clone().with_name("not_position");
        }
        descriptors = StructArray::try_new(fields.into(), arrays, None).unwrap();
        field = field.with_data_type(descriptors.data_type().clone());
        let mut capture = BlobRefCapture::try_new(
            dataset.clone(),
            source,
            dataset.schema().field("blob").unwrap().id,
        )
        .unwrap();
        let error = capture
            .capture(&scan_source, &field, &addresses, &descriptors)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::empty(0)]
    #[case::all_null(3)]
    #[tokio::test]
    async fn capture_empty_and_null_batches(#[case] count: usize) {
        let dataset = capture_dataset("memory://").await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, _, descriptors) = capture_inputs(&dataset).await;
        let descriptors = StructArray::new_null(descriptors.fields().clone(), count);
        let addresses = vec![RowAddress::from(RowAddress::TOMBSTONE_ROW); count];
        let mut capture = BlobRefCapture::try_new(
            dataset.clone(),
            source.clone(),
            dataset.schema().field("blob").unwrap().id,
        )
        .unwrap();
        let before = dataset.object_store.io_stats_snapshot().requests.len();
        let captured = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        let restored = BlobRefBatch::decode(&captured.encode().unwrap()).unwrap();
        assert_eq!(restored.array().len(), count);
        assert_eq!(restored.array().null_count(), count);
        assert_eq!(
            dataset.object_store.io_stats_snapshot().requests.len(),
            before
        );
    }

    #[tokio::test]
    async fn capture_branch_snapshot() {
        let directory = tempfile::tempdir().unwrap();
        let source_dataset =
            capture_dataset(directory.path().join("table").to_str().unwrap()).await;
        let field_id = source_dataset.schema().field("blob").unwrap().id;
        let main = BlobSourceContext::from_dataset(&source_dataset, "catalog/table", "workers")
            .await
            .unwrap();
        let mut source_dataset = source_dataset.as_ref().clone();
        let branch = Arc::new(
            source_dataset
                .create_branch("experiment", main.version(), None)
                .await
                .unwrap(),
        );
        let error = BlobRefCapture::try_new(branch.clone(), main, field_id)
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match dataset snapshot")
        );
        let source = BlobSourceContext::from_dataset(&branch, "catalog/table", "workers")
            .await
            .unwrap();
        let (field, addresses, descriptors) = capture_inputs(&branch).await;
        let mut capture =
            BlobRefCapture::try_new(branch.clone(), source.clone(), field_id).unwrap();
        let captured = capture
            .capture(&source, &field, &addresses, &descriptors)
            .await
            .unwrap();
        assert_eq!(
            read_captured(&branch, &captured.reference(0).unwrap().unwrap())
                .await
                .as_ref(),
            b"abc"
        );
    }

    #[test]
    fn rejects_unregistered_locator_base() {
        let context = context(1, None);
        let mut locator = reference(&context, 0, Some(3)).object_locator().unwrap();
        locator.base_id = Some(999);
        let reference = BlobRef::try_new(0, locator.encode_to_vec(), Some(3)).unwrap();
        let error = BlobRefBatch::try_new(vec![context], [Some(reference)]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("unknown base_id 999"));
    }

    #[test]
    fn registered_zero_base_survives_transport() {
        let mut proto = context(1, None).proto;
        let mut manifest = table_pb::Manifest::decode(proto.manifest.clone()).unwrap();
        manifest.base_paths.push(table_pb::BasePath {
            id: 0,
            name: None,
            path: "s3://source/table.lance".to_string(),
            is_dataset_root: true,
        });
        proto.manifest = manifest.encode_to_vec().into();
        let context = BlobSourceContext::from_proto(proto).unwrap();
        let reference = context
            .reference(
                "s3://source/table.lance/data/blob".to_string(),
                0,
                Some(3),
                BlobKind::Managed,
                Some(0),
            )
            .unwrap();
        let batch = BlobRefBatch::try_new(vec![context], [Some(reference)]).unwrap();
        let restored = BlobRefBatch::decode(&batch.encode().unwrap()).unwrap();
        assert_eq!(
            restored
                .reference(0)
                .unwrap()
                .unwrap()
                .object_locator()
                .unwrap()
                .base_id,
            Some(0)
        );
    }
}
