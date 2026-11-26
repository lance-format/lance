// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::{Arc, Mutex},
};

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_arrow::{FieldExt, list::ListArrayExt, r#struct::StructArrayExt};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use object_store::path::Path;

use super::BlobWriteSources;
use super::external::{
    ExternalBaseResolver, collect_external_blob_uris, normalize_external_absolute_uri,
};
use super::types::{BlobObjectWriter, ManagedBlobLocation};
use super::write_sources::BlobWriteObject;
use crate::blob::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobIdAllocator, BlobRange, blob_v2_layout,
    blob_v2_shape_error, sidecar_path_for_data_file, validate_prepared_blob_array,
};
use crate::write::ExternalBlobMode;
use lance_core::datatypes::{BlobKind, BlobV2Layout};
use lance_core::{Error, Result};
use lance_io::traits::Reader;
use lance_table::feature_flags::MANAGED_BLOBS_META_KEY;

#[cfg(test)]
use crate::blob::thresholds::DEDICATED_THRESHOLD;
pub(super) use crate::blob::thresholds::INLINE_MAX;
use crate::blob::thresholds::{
    blob_dedicated_threshold_from_metadata, blob_inline_threshold_from_metadata,
    blob_pack_file_threshold_from_metadata,
};

// Some Reader implementations buffer a whole requested range.
const BLOB_SOURCE_MAX_REQUEST_SIZE: usize = 8 * 1024 * 1024;

/// The `_blobs/<uuid>.blob` payload objects a write created.
///
/// Nothing else records them: they are not part of any fragment's file list,
/// and a descriptor scan over the written fragments cannot tell a payload this
/// write created from a reused Managed descriptor that points at a live one.
/// A caller that discards the write's result uses this inventory to delete
/// exactly what the write created. Clones share the same inventory.
#[derive(Debug, Clone, Default)]
pub struct BlobPayloadInventory(Arc<Mutex<Vec<(ObjectStore, Path)>>>);

impl BlobPayloadInventory {
    fn record(&self, object_store: &ObjectStore, path: &Path) {
        self.0
            .lock()
            .unwrap()
            .push((object_store.clone(), path.clone()));
    }

    /// Delete every recorded payload and clear the inventory.
    ///
    /// Failures are logged rather than returned: the payloads are unreferenced
    /// either way, so ordinary garbage collection still reclaims them.
    pub async fn delete_all(&self) {
        let payloads = std::mem::take(&mut *self.0.lock().unwrap());
        for (object_store, path) in payloads {
            match object_store.delete(&path).await {
                // A path is recorded before its writer opens, so a writer that
                // failed to open leaves nothing behind.
                Ok(()) | Err(Error::NotFound { .. }) => {}
                Err(e) => {
                    log::warn!("Failed to clean up orphaned blob payload '{}': {}", path, e);
                }
            }
        }
    }
}

/// Allocate a fresh managed payload under `root`, recording it before any bytes
/// are written so a partially uploaded payload is still accounted for.
fn new_managed_payload(
    base_id: Option<u32>,
    root: &Path,
    object_store: &ObjectStore,
    created_payloads: Option<&BlobPayloadInventory>,
) -> (ManagedBlobLocation, Path) {
    let location = ManagedBlobLocation::new(base_id);
    let path = location.path(root);
    if let Some(created_payloads) = created_payloads {
        created_payloads.record(object_store, &path);
    }
    (location, path)
}

enum BlobPackAddress {
    Sidecar(u32),
    Managed(ManagedBlobLocation),
}

struct BlobPack {
    writer: BlobObjectWriter,
    address: BlobPackAddress,
    max_size: usize,
}

impl BlobPack {
    async fn write(&mut self, source: BlobWriteSource<'_>) -> Result<BlobDescriptor> {
        let range = match source {
            BlobWriteSource::Bytes(data) => self.writer.write_blob(data).await?,
            BlobWriteSource::Reader(source) => {
                self.writer
                    .write_blob_from_reader(
                        source.reader.as_ref(),
                        source.reader_range()?,
                        source.max_request_size,
                    )
                    .await?
            }
        };
        Ok(match &self.address {
            BlobPackAddress::Sidecar(blob_id) => BlobDescriptor::Packed {
                blob_id: *blob_id,
                offset: range.offset,
                size: range.size,
            },
            BlobPackAddress::Managed(location) => location.descriptor(range),
        })
    }
}

/// Preprocesses blob v2 columns on the write path so the encoder sees bounded inline values and
/// lightweight descriptors for larger payloads:
///
/// - Spills large blobs to sidecar files before encoding, reducing memory/CPU and avoiding copying huge payloads through page builders.
/// - Emits relative Managed addresses for dataset writes; low-level sidecar writers retain their existing descriptors.
/// - Leaves small inline blobs and URI rows unchanged for compatibility.
pub struct BlobPreprocessor {
    sidecar_rewrite: Option<Arc<super::SidecarRewrite>>,
    copied_dedicated: HashMap<u32, BlobDescriptor>,
    managed_base: Option<(Option<u32>, Path)>,
    created_payloads: Option<BlobPayloadInventory>,
    object_store: ObjectStore,
    data_dir: Path,
    data_file_key: String,
    blob_id_allocator: BlobIdAllocator,
    is_part: bool,
    current_pack: Option<BlobPack>,
    /// Write-param override for the pack-file roll size. When set, it takes
    /// precedence over each field's `blob-pack-file-size-threshold` metadata for
    /// this write job only; it is not persisted into the dataset schema.
    pack_file_size_override: Option<usize>,
    field_processors: Vec<BlobPreprocessField>,
    external_base_resolver: Option<Arc<ExternalBaseResolver>>,
    allow_external_blob_outside_bases: bool,
    external_blob_mode: ExternalBlobMode,
    source_store_registry: Arc<ObjectStoreRegistry>,
    source_store_params: ObjectStoreParams,
}

/// A resolved payload range that can be inlined or streamed into destination storage.
struct BlobReaderSource {
    reader: Box<dyn Reader>,
    start: u64,
    size: u64,
    max_request_size: usize,
}

/// A blob payload source used by packed and dedicated writers.
///
enum BlobWriteSource<'a> {
    Bytes(&'a [u8]),
    Reader(&'a BlobReaderSource),
}

#[derive(Clone, Debug)]
struct BlobPreprocessField {
    kind: BlobPreprocessFieldKind,
}

#[derive(Clone, Debug)]
enum BlobPreprocessFieldKind {
    BlobV2 {
        inline_threshold: usize,
        dedicated_threshold: usize,
        pack_file_threshold: usize,
        writer_metadata: HashMap<String, String>,
    },
    Struct {
        children: Vec<BlobPreprocessField>,
    },
    List {
        child: Box<BlobPreprocessField>,
    },
    Passthrough,
}

impl BlobPreprocessField {
    fn new(field: &ArrowField) -> Result<Self> {
        if field.is_blob_v2() {
            return match blob_v2_layout(field) {
                Some(BlobV2Layout::Prepared | BlobV2Layout::Logical) => Ok(Self {
                    kind: BlobPreprocessFieldKind::BlobV2 {
                        inline_threshold: blob_inline_threshold_from_metadata(
                            field.metadata(),
                            field.name(),
                        )?,
                        dedicated_threshold: blob_dedicated_threshold_from_metadata(
                            field.metadata(),
                            field.name(),
                        )?,
                        pack_file_threshold: blob_pack_file_threshold_from_metadata(
                            field.metadata(),
                            field.name(),
                        )?,
                        writer_metadata: field.metadata().clone(),
                    },
                }),
                _ => Err(blob_v2_shape_error(
                    field,
                    &[BlobV2Layout::Logical, BlobV2Layout::Prepared],
                )),
            };
        }

        if let ArrowDataType::Struct(children) = field.data_type() {
            let children = children
                .iter()
                .map(|child| Self::new(child.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            if children.iter().any(|child| child.requires_preprocessing()) {
                return Ok(Self {
                    kind: BlobPreprocessFieldKind::Struct { children },
                });
            }
        }

        if let ArrowDataType::List(child) | ArrowDataType::LargeList(child) = field.data_type() {
            let child = Self::new(child.as_ref())?;
            if child.requires_preprocessing() {
                return Ok(Self {
                    kind: BlobPreprocessFieldKind::List {
                        child: Box::new(child),
                    },
                });
            }
        }

        Ok(Self {
            kind: BlobPreprocessFieldKind::Passthrough,
        })
    }

    fn requires_preprocessing(&self) -> bool {
        !matches!(self.kind, BlobPreprocessFieldKind::Passthrough)
    }

    fn force_non_empty_inline_to_sidecar(&mut self) {
        match &mut self.kind {
            BlobPreprocessFieldKind::BlobV2 {
                inline_threshold, ..
            } => *inline_threshold = 0,
            BlobPreprocessFieldKind::Struct { children } => {
                for child in children {
                    child.force_non_empty_inline_to_sidecar();
                }
            }
            BlobPreprocessFieldKind::List { child } => {
                child.force_non_empty_inline_to_sidecar();
            }
            BlobPreprocessFieldKind::Passthrough => {}
        }
    }
}

impl BlobReaderSource {
    /// Return the logical payload size after applying the source range.
    fn size(&self) -> u64 {
        self.size
    }

    /// Convert the logical slice into the current reader API's usize-based range.
    fn reader_range(&self) -> Result<Range<usize>> {
        let start = usize::try_from(self.start).map_err(|_| {
            Error::invalid_input(format!(
                "Blob source position {} does not fit into usize",
                self.start
            ))
        })?;
        let size = usize::try_from(self.size).map_err(|_| {
            Error::invalid_input(format!(
                "Blob source size {} does not fit into usize",
                self.size
            ))
        })?;
        let end = start.checked_add(size).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob source range overflows usize: position={}, size={}",
                self.start, self.size
            ))
        })?;
        Ok(start..end)
    }

    /// Materialize the slice into memory for the inline blob path.
    async fn read_all(&self) -> Result<bytes::Bytes> {
        if self.size == 0 {
            return Ok(bytes::Bytes::new());
        }
        let range = self.reader_range()?;
        let data = self.reader.get_range(range).await?;
        if data.len() as u64 != self.size {
            return Err(Error::io(format!(
                "Blob source '{}' returned {} bytes, expected {}",
                self.reader.path(),
                data.len(),
                self.size
            )));
        }
        Ok(data)
    }
}

impl BlobWriteSource<'_> {
    /// Return the payload size for a value that can be packed.
    fn size(&self) -> u64 {
        match self {
            Self::Bytes(data) => data.len() as u64,
            Self::Reader(source) => source.size(),
        }
    }
}

impl BlobPreprocessor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        schema: &lance_core::datatypes::Schema,
        external_base_resolver: Option<Arc<ExternalBaseResolver>>,
        allow_external_blob_outside_bases: bool,
        external_blob_mode: ExternalBlobMode,
        source_store_registry: Arc<ObjectStoreRegistry>,
        source_store_params: ObjectStoreParams,
        pack_file_size_override: Option<usize>,
    ) -> Result<Self> {
        schema.check_blob_nesting(lance_core::datatypes::Field::is_blob_v2)?;
        let arrow_schema = arrow_schema::Schema::from(schema);
        let field_processors = arrow_schema
            .fields()
            .iter()
            .map(|field| BlobPreprocessField::new(field.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sidecar_rewrite: None,
            copied_dedicated: HashMap::new(),
            managed_base: None,
            created_payloads: None,
            object_store,
            data_dir,
            data_file_key,
            blob_id_allocator: BlobIdAllocator::new(1),
            is_part: false,
            current_pack: None,
            pack_file_size_override,
            field_processors,
            external_base_resolver,
            allow_external_blob_outside_bases,
            external_blob_mode,
            source_store_registry,
            source_store_params,
        })
    }

    pub(crate) fn with_sidecar_rewrite(
        mut self,
        context: Option<Arc<super::SidecarRewrite>>,
    ) -> Self {
        self.sidecar_rewrite = context;
        self
    }

    pub(crate) fn with_managed_base(mut self, base_id: Option<u32>, root: Path) -> Self {
        self.managed_base = Some((base_id, root));
        self
    }

    /// Record every managed payload object this preprocessor creates into
    /// `created_payloads`.
    pub(crate) fn with_created_payloads(
        mut self,
        created_payloads: Option<BlobPayloadInventory>,
    ) -> Self {
        self.created_payloads = created_payloads;
        self
    }

    pub(crate) fn with_part_blob_ids(mut self, blob_ids: Option<Range<u32>>) -> Result<Self> {
        if let Some(blob_ids) = blob_ids {
            self.blob_id_allocator = BlobIdAllocator::from_range(blob_ids)?;
        }
        self.is_part = true;
        for processor in &mut self.field_processors {
            processor.force_non_empty_inline_to_sidecar();
        }
        Ok(self)
    }

    /// Create a preprocessor using a MemWAL generation's shared blob IDs.
    #[doc(hidden)]
    pub fn for_mem_wal(
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        schema: &lance_core::datatypes::Schema,
        source_store_registry: Arc<ObjectStoreRegistry>,
        source_store_params: ObjectStoreParams,
        blob_id_allocator: BlobIdAllocator,
    ) -> Result<Self> {
        let mut preprocessor = Self::new(
            object_store,
            data_dir,
            data_file_key,
            schema,
            None,
            true,
            ExternalBlobMode::Reference,
            source_store_registry,
            source_store_params,
            None,
        )?;
        preprocessor.blob_id_allocator = blob_id_allocator;
        Ok(preprocessor)
    }

    fn blob_writer_with_metadata(
        &self,
        field: &ArrowField,
        metadata: HashMap<String, String>,
    ) -> Result<BlobDescriptorArrayBuilder> {
        BlobDescriptorArrayBuilder::new_with_metadata(field.name(), field.is_nullable(), metadata)
            .with_field_ids_of(field)
    }

    async fn write_dedicated(&mut self, source: BlobWriteSource<'_>) -> Result<BlobDescriptor> {
        if let Some((base_id, root)) = &self.managed_base {
            let (location, path) = new_managed_payload(
                *base_id,
                root,
                &self.object_store,
                self.created_payloads.as_ref(),
            );
            let mut writer = BlobObjectWriter::try_new(self.object_store.clone(), path).await?;
            let range = match source {
                BlobWriteSource::Bytes(data) => writer.write_blob(data).await?,
                BlobWriteSource::Reader(source) => {
                    writer
                        .write_blob_from_reader(
                            source.reader.as_ref(),
                            source.reader_range()?,
                            source.max_request_size,
                        )
                        .await?
                }
            };
            writer.finish().await?;
            return Ok(location.descriptor(range));
        }

        let data_file_path = self
            .data_dir
            .clone()
            .join(format!("{}.lance", self.data_file_key));
        let mut writer = crate::blob::DedicatedBlobWriter::try_new(
            self.object_store.clone(),
            data_file_path,
            self.blob_id_allocator.next()?,
        )
        .await?;
        match source {
            BlobWriteSource::Bytes(data) => writer.write(data).await?,
            BlobWriteSource::Reader(source) => {
                writer
                    .write_from_reader(
                        source.reader.as_ref(),
                        source.reader_range()?,
                        source.max_request_size,
                    )
                    .await?;
            }
        }
        writer.finish().await
    }

    async fn write_packed(
        &mut self,
        field_pack_file_threshold: usize,
        source: BlobWriteSource<'_>,
    ) -> Result<BlobDescriptor> {
        let max_pack_size = self
            .pack_file_size_override
            .unwrap_or(field_pack_file_threshold);
        let len = source.size();
        let needs_new_pack = self.current_pack.as_ref().is_none_or(|pack| {
            pack.max_size != max_pack_size
                || pack
                    .writer
                    .size()
                    .checked_add(len)
                    .is_none_or(|size| size > max_pack_size as u64)
        });
        if needs_new_pack {
            if let Some(pack) = self.current_pack.take() {
                pack.writer.finish().await?;
            }
            let (address, path) = if let Some((base_id, root)) = &self.managed_base {
                let (location, path) = new_managed_payload(
                    *base_id,
                    root,
                    &self.object_store,
                    self.created_payloads.as_ref(),
                );
                (BlobPackAddress::Managed(location), path)
            } else {
                let blob_id = self.blob_id_allocator.next()?;
                let data_file_path = self
                    .data_dir
                    .clone()
                    .join(format!("{}.lance", self.data_file_key));
                (
                    BlobPackAddress::Sidecar(blob_id),
                    sidecar_path_for_data_file(&data_file_path, blob_id)?,
                )
            };
            self.current_pack = Some(BlobPack {
                writer: BlobObjectWriter::try_new(self.object_store.clone(), path).await?,
                address,
                max_size: max_pack_size,
            });
        }
        self.current_pack
            .as_mut()
            .ok_or_else(|| Error::internal("Blob pack was not initialized"))?
            .write(source)
            .await
    }

    async fn prepare_blob_references(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        processor: &BlobPreprocessFieldKind,
        write_sources: Option<(&Path, &BlobWriteSources)>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let BlobPreprocessFieldKind::BlobV2 {
            inline_threshold,
            dedicated_threshold,
            pack_file_threshold,
            writer_metadata,
        } = processor
        else {
            return Err(Error::internal(
                "Prepared Blob is missing its write configuration",
            ));
        };
        let values = array.as_struct();
        let kinds = values
            .column_by_name("kind")
            .expect("validated prepared Blob has kind")
            .as_primitive::<UInt8Type>();
        let data = values
            .column_by_name("data")
            .expect("validated prepared Blob has data")
            .as_binary::<i64>();
        let uris = values
            .column_by_name("uri")
            .expect("validated prepared Blob has uri")
            .as_string::<i32>();
        let blob_ids = values
            .column_by_name("blob_id")
            .expect("validated prepared Blob has blob_id")
            .as_primitive::<UInt32Type>();
        let sizes = values
            .column_by_name("blob_size")
            .expect("validated prepared Blob has blob_size")
            .as_primitive::<UInt64Type>();
        let positions = values
            .column_by_name("position")
            .expect("validated prepared Blob has position")
            .as_primitive::<UInt64Type>();
        let mut output = self.blob_writer_with_metadata(field, writer_metadata.clone())?;

        if self.is_part && write_sources.is_none() {
            for row in 0..values.len() {
                if !values.is_null(row)
                    && matches!(
                        BlobKind::try_from(kinds.value(row))?,
                        BlobKind::Packed | BlobKind::Dedicated
                    )
                {
                    let blob_id = blob_ids.value(row);
                    self.blob_id_allocator.reserve(blob_id).map_err(|error| {
                        Error::invalid_input(format!(
                            "Prepared Blob v2 field '{}' row {row} uses invalid managed Blob ID {blob_id}: {error}",
                            field.name()
                        ))
                    })?;
                }
            }
        }

        for row in 0..values.len() {
            if values.is_null(row) {
                output.push_null()?;
                continue;
            }
            match BlobKind::try_from(kinds.value(row))? {
                BlobKind::Inline => {
                    let value = data.value(row);
                    if self.is_part && !value.is_empty() {
                        let descriptor = self
                            .write_packed(*pack_file_threshold, BlobWriteSource::Bytes(value))
                            .await?;
                        output.push(descriptor)?;
                    } else {
                        output.push_inline(Bytes::copy_from_slice(value))?;
                    }
                }
                BlobKind::Packed => {
                    let id = blob_ids.value(row);
                    let offset = positions.value(row);
                    let size = sizes.value(row);
                    if let Some((root, sources)) = write_sources {
                        let source = sources.get(id, offset, size)?;
                        let descriptor = self
                            .write_registered_blob(
                                source,
                                root,
                                *inline_threshold,
                                *dedicated_threshold,
                                *pack_file_threshold,
                            )
                            .await?;
                        output.push(descriptor)?;
                    } else {
                        output.push_packed(id, BlobRange { offset, size })?;
                    }
                }
                BlobKind::Dedicated => {
                    if write_sources.is_some() {
                        return Err(Error::invalid_input(format!(
                            "Blob field '{}' row {row} must register existing Dedicated payloads as a write source",
                            field.name()
                        )));
                    }
                    output.push_dedicated(blob_ids.value(row), sizes.value(row))?;
                }
                BlobKind::Managed | BlobKind::ManagedWithBase => {
                    output.push(BlobDescriptor::Managed {
                        base_id: (kinds.value(row) == BlobKind::ManagedWithBase as u8)
                            .then(|| blob_ids.value(row)),
                        uri: uris.value(row).to_string(),
                        offset: positions.value(row),
                        size: sizes.value(row),
                    })?;
                }
                BlobKind::External => {
                    output.push(BlobDescriptor::External {
                        base_id: blob_ids.value(row),
                        uri: uris.value(row).to_string(),
                        offset: positions.value(row),
                        size: sizes.value(row),
                    })?;
                }
            }
        }
        let (field, array) = output.finish()?.into_parts();
        Ok((array, Arc::new(field)))
    }

    async fn write_registered_blob(
        &mut self,
        source: &BlobWriteObject,
        root: &Path,
        inline_threshold: usize,
        dedicated_threshold: usize,
        pack_file_threshold: usize,
    ) -> Result<BlobDescriptor> {
        // A whole uploaded object that fits in one copy request can have its
        // length checked by the GET. Keep the metadata check for adoption,
        // empty blobs, and copies that need a bounded range reader.
        let check_length_with_get = self.managed_base.is_none()
            && self.object_store.is_cloud()
            && matches!(source, BlobWriteObject::Upload { size, .. }
                if (1..=BLOB_SOURCE_MAX_REQUEST_SIZE as u64).contains(size));
        let (store, path, start, size) = match source {
            BlobWriteObject::Upload {
                relative_path,
                size,
            } => {
                let path = root.parts().chain(relative_path.parts()).collect();
                if !check_length_with_get {
                    let actual_size = self.object_store.size(&path).await?;
                    if actual_size != *size {
                        return Err(Error::invalid_input(format!(
                            "Blob upload '{relative_path}' has size {actual_size}, expected {size}"
                        )));
                    }
                }
                if let Some((base_id, _)) = &self.managed_base {
                    return Ok(BlobDescriptor::Managed {
                        base_id: *base_id,
                        uri: relative_path.to_string(),
                        offset: 0,
                        size: *size,
                    });
                }
                (&self.object_store, path, 0, *size)
            }
            BlobWriteObject::Stored(file) => (
                file.object_store().as_ref(),
                file.data_path().clone(),
                file.position(),
                file.size(),
            ),
        };
        if size == 0 {
            return Ok(BlobDescriptor::Inline { data: Bytes::new() });
        }
        let reader = if check_length_with_get {
            store
                .with_block_size(BLOB_SOURCE_MAX_REQUEST_SIZE)
                .open_with_expected_size(&path, size)
                .await?
        } else {
            store.open(&path).await?
        };
        let source = BlobReaderSource {
            reader,
            start,
            size,
            max_request_size: BLOB_SOURCE_MAX_REQUEST_SIZE,
        };
        if size > dedicated_threshold as u64 {
            self.write_dedicated(BlobWriteSource::Reader(&source)).await
        } else if size > inline_threshold.min(INLINE_MAX) as u64 {
            // Uploaded values may be much larger than the write's memory budget,
            // even when the field allows an unusually high inline threshold.
            self.write_packed(pack_file_threshold, BlobWriteSource::Reader(&source))
                .await
        } else {
            Ok(BlobDescriptor::Inline {
                data: source.read_all().await?,
            })
        }
    }

    /// Resolve task-local source IDs before descriptors reach the encoder.
    /// Packed values stream through the rolling pack writer, so each output
    /// retains only its live ranges. Dedicated objects can be copied whole.
    async fn rewrite_sidecars(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        pack_file_threshold: usize,
        writer_metadata: &HashMap<String, String>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let context = self
            .sidecar_rewrite
            .clone()
            .ok_or_else(|| Error::internal("Missing compaction sidecar context"))?;
        let values = array.as_struct();
        let kinds = values
            .column_by_name("kind")
            .expect("validated kind")
            .as_primitive::<UInt8Type>();
        let ids = values
            .column_by_name("blob_id")
            .expect("validated blob_id")
            .as_primitive::<UInt32Type>();
        let positions = values
            .column_by_name("position")
            .expect("validated position")
            .as_primitive::<UInt64Type>();
        let sizes = values
            .column_by_name("blob_size")
            .expect("validated blob_size")
            .as_primitive::<UInt64Type>();
        let data = values
            .column_by_name("data")
            .expect("validated data")
            .as_binary::<i64>();
        let uris = values
            .column_by_name("uri")
            .expect("validated uri")
            .as_string::<i32>();
        let mut output = self.blob_writer_with_metadata(field, writer_metadata.clone())?;
        for row in 0..values.len() {
            if values.is_null(row) {
                output.push_null()?;
                continue;
            }
            let kind = BlobKind::try_from(kinds.value(row))?;
            match kind {
                BlobKind::Packed | BlobKind::Dedicated => {
                    if kind == BlobKind::Packed && sizes.value(row) == 0 {
                        // Empty ranges need no object access, including readers
                        // that reject empty range requests.
                        output.push_inline(Bytes::new())?;
                        continue;
                    }
                    if kind == BlobKind::Dedicated
                        && let Some(descriptor) = self.copied_dedicated.get(&ids.value(row))
                    {
                        output.push(descriptor.clone())?;
                        continue;
                    }
                    let source = context.source(ids.value(row))?;
                    let descriptor = if kind == BlobKind::Packed {
                        let source = BlobReaderSource {
                            reader: source.store.open(&source.path).await?,
                            start: positions.value(row),
                            size: sizes.value(row),
                            max_request_size: usize::MAX,
                        };
                        self.write_packed(pack_file_threshold, BlobWriteSource::Reader(&source))
                            .await?
                    } else {
                        let id = self.blob_id_allocator.next()?;
                        let destination = lance_core::utils::blob::blob_path(
                            &self.data_dir,
                            &self.data_file_key,
                            id,
                        );
                        source
                            .store
                            .copy_bulk(&source.path, &self.object_store, &destination)
                            .await?;
                        BlobDescriptor::Dedicated {
                            blob_id: id,
                            size: sizes.value(row),
                        }
                    };
                    output.push(descriptor.clone())?;
                    if kind == BlobKind::Dedicated {
                        self.copied_dedicated.insert(ids.value(row), descriptor);
                    }
                }
                BlobKind::Inline => output.push_inline(Bytes::copy_from_slice(data.value(row)))?,
                BlobKind::External => output.push(BlobDescriptor::External {
                    base_id: ids.value(row),
                    uri: uris.value(row).to_string(),
                    offset: positions.value(row),
                    size: sizes.value(row),
                })?,
                BlobKind::Managed | BlobKind::ManagedWithBase => {
                    return Err(Error::internal(
                        "Managed reference in ordinary sidecar rewrite",
                    ));
                }
            }
        }
        let (field, array) = output.finish()?.into_parts();
        Ok((array, Arc::new(field)))
    }

    async fn resolve_external_reference(&mut self, uri: &str) -> Result<(u32, String)> {
        let mapped = if let Some(resolver) = &self.external_base_resolver {
            resolver.resolve_external_uri(uri).await?
        } else {
            None
        };
        if let Some(mapped) = mapped {
            return Ok((mapped.base_id, mapped.relative_path));
        }

        if self
            .external_base_resolver
            .as_ref()
            .is_some_and(|resolver| {
                resolver.unregistered_external_blob_policy
                    == crate::session::UnregisteredExternalBlobPolicy::Deny
            })
        {
            return Err(Error::invalid_input(
                "Remote tables only support external blobs under registered storage bases",
            ));
        }

        if self.allow_external_blob_outside_bases {
            let normalized = normalize_external_absolute_uri(uri)?;
            return Ok((0, normalized));
        }

        Err(Error::invalid_input(format!(
            "External blob URI '{}' is outside registered external bases (dataset root is not allowed). Set allow_external_blob_outside_bases=true to store it as absolute external URI.",
            uri
        )))
    }

    #[doc(hidden)]
    pub async fn validate_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let selected_rows = vec![true; batch.num_rows()];
        let mut external_uris = Vec::new();
        for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
            collect_external_blob_uris(
                field,
                array,
                &selected_rows,
                field.name(),
                &mut external_uris,
            )?;
        }

        if self.external_blob_mode == ExternalBlobMode::Reference {
            let mut validated_uris = HashSet::new();
            for (_, uri) in external_uris {
                if validated_uris.insert(uri.clone()) {
                    self.resolve_external_reference(&uri).await?;
                }
            }
        }
        Ok(())
    }

    async fn open_external_source(
        &mut self,
        uri: &str,
        position: Option<u64>,
        size: Option<u64>,
    ) -> Result<BlobReaderSource> {
        let (object_store, path) = ObjectStore::from_uri_and_params(
            self.source_store_registry.clone(),
            uri,
            &self.source_store_params,
        )
        .await?;
        let reader = object_store.open(&path).await?;
        match (position, size) {
            (Some(position), Some(size)) => {
                position.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "External blob range overflows u64: position={}, size={}",
                        position, size
                    ))
                })?;
                Ok(BlobReaderSource {
                    reader,
                    start: position,
                    size,
                    max_request_size: usize::MAX,
                })
            }
            (None, None) => {
                let size = reader.size().await? as u64;
                Ok(BlobReaderSource {
                    reader,
                    start: 0,
                    size,
                    max_request_size: usize::MAX,
                })
            }
            _ => Err(Error::invalid_input(format!(
                "External blob URI '{}' must set both position and size when slicing for ingest",
                uri
            ))),
        }
    }

    #[doc(hidden)]
    pub async fn preprocess_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch> {
        self.preprocess_batch_with_sources(batch, None).await
    }

    pub(crate) async fn preprocess_batch_with_sources(
        &mut self,
        batch: &RecordBatch,
        write_sources: Option<(&Path, &BlobWriteSources)>,
    ) -> Result<RecordBatch> {
        self.validate_batch(batch).await?;
        let expected_columns = self.field_processors.len();
        if batch.num_columns() != expected_columns {
            return Err(Error::invalid_input(format!(
                "Unexpected number of columns: expected {}, got {}",
                expected_columns,
                batch.num_columns()
            )));
        }

        let batch_schema = batch.schema();
        let batch_fields = batch_schema.fields();
        let field_processors = self.field_processors.clone();

        let mut new_columns = Vec::with_capacity(batch.num_columns());
        let mut new_fields = Vec::with_capacity(batch.num_columns());

        for ((processor, array), field) in field_processors
            .iter()
            .zip(batch.columns().iter())
            .zip(batch_fields.iter())
        {
            let (new_column, new_field) = self
                .preprocess_field(processor, array.clone(), field, write_sources)
                .await?;
            new_columns.push(new_column);
            new_fields.push(new_field);
        }

        let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            new_fields
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>(),
            batch_schema.metadata().clone(),
        ));

        RecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| Error::invalid_input(e.to_string()))
    }

    fn preprocess_field<'a>(
        &'a mut self,
        processor: &'a BlobPreprocessField,
        array: ArrayRef,
        field: &'a Arc<ArrowField>,
        write_sources: Option<(&'a Path, &'a BlobWriteSources)>,
    ) -> BoxFuture<'a, Result<(ArrayRef, Arc<ArrowField>)>> {
        async move {
            if blob_v2_layout(field.as_ref()) == Some(BlobV2Layout::Prepared) {
                validate_prepared_blob_array(field.as_ref(), &array)?;
                let values = array.as_struct();
                let kinds = values
                    .column_by_name("kind")
                    .expect("validated prepared kind")
                    .as_primitive::<UInt8Type>();
                let ids = values
                    .column_by_name("blob_id")
                    .expect("validated prepared base")
                    .as_primitive::<UInt32Type>();
                for row in 0..values.len() {
                    if !values.is_null(row)
                        && matches!(
                            BlobKind::try_from(kinds.value(row))?,
                            BlobKind::Managed | BlobKind::ManagedWithBase
                        )
                        && self.managed_base.is_none()
                    {
                        return Err(Error::invalid_input(format!(
                            "Managed blob field '{}' row {row} requires schema metadata {}=true",
                            field.name(),
                            MANAGED_BLOBS_META_KEY,
                        )));
                    }
                    if !values.is_null(row)
                        && kinds.is_valid(row)
                        && kinds.value(row) == BlobKind::ManagedWithBase as u8
                    {
                        let id = ids.value(row);
                        if !self
                            .external_base_resolver
                            .as_ref()
                            .is_some_and(|resolver| resolver.registered_base_ids.contains(id))
                        {
                            return Err(Error::invalid_input(format!(
                                "Managed blob references unknown base_id {id}"
                            )));
                        }
                    }
                }

                if self.sidecar_rewrite.is_some()
                    && let BlobPreprocessFieldKind::BlobV2 {
                        pack_file_threshold,
                        writer_metadata,
                        ..
                    } = &processor.kind
                {
                    return self
                        .rewrite_sidecars(
                            array,
                            field.as_ref(),
                            *pack_file_threshold,
                            writer_metadata,
                        )
                        .await;
                }

                if self.is_part || write_sources.is_some() {
                    return self
                        .prepare_blob_references(
                            array,
                            field.as_ref(),
                            &processor.kind,
                            write_sources,
                        )
                        .await;
                }
                return Ok((array, field.clone()));
            }

            match &processor.kind {
                BlobPreprocessFieldKind::Passthrough => Ok((array, field.clone())),
                BlobPreprocessFieldKind::BlobV2 {
                    inline_threshold,
                    dedicated_threshold,
                    pack_file_threshold,
                    writer_metadata,
                } => {
                    self.logical_to_prepared_blob_array(
                        array,
                        field.as_ref(),
                        *inline_threshold,
                        *dedicated_threshold,
                        *pack_file_threshold,
                        writer_metadata,
                    )
                    .await
                }
                BlobPreprocessFieldKind::Struct { children } => {
                    self.preprocess_struct_array(array, field.as_ref(), children, write_sources)
                        .await
                }
                BlobPreprocessFieldKind::List { child } => match field.data_type() {
                    ArrowDataType::List(_) => {
                        self.preprocess_list_array::<i32>(
                            array,
                            field.as_ref(),
                            child,
                            write_sources,
                        )
                        .await
                    }
                    ArrowDataType::LargeList(_) => {
                        self.preprocess_list_array::<i64>(
                            array,
                            field.as_ref(),
                            child,
                            write_sources,
                        )
                        .await
                    }
                    _ => Err(Error::internal(format!(
                        "Blob list preprocessor received non-list field '{}'",
                        field.name()
                    ))),
                },
            }
        }
        .boxed()
    }

    async fn preprocess_struct_array(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        children: &[BlobPreprocessField],
        write_sources: Option<(&Path, &BlobWriteSources)>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let struct_arr = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input("Struct field was not a struct array"))?;
        if struct_arr.num_columns() != children.len() {
            return Err(Error::invalid_input(format!(
                "Struct field '{}' expected {} children, got {}",
                field.name(),
                children.len(),
                struct_arr.num_columns()
            )));
        }

        let struct_arr = struct_arr.normalize_slicing()?;
        let parent_nulls = struct_arr.nulls().cloned();
        let pushed_down = struct_arr.pushdown_nulls()?;
        let child_fields = pushed_down.fields().clone();
        let child_columns = pushed_down.columns().to_vec();

        let mut new_columns = Vec::with_capacity(children.len());
        let mut new_fields = Vec::with_capacity(children.len());
        for ((child_processor, child_array), child_field) in
            children.iter().zip(child_columns).zip(child_fields.iter())
        {
            let (new_column, new_field) = self
                .preprocess_field(child_processor, child_array, child_field, write_sources)
                .await?;
            new_columns.push(new_column);
            new_fields.push(new_field);
        }

        let struct_array =
            StructArray::try_new(new_fields.clone().into(), new_columns, parent_nulls)?;
        let field = Arc::new(
            ArrowField::new(
                field.name(),
                ArrowDataType::Struct(new_fields.into()),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone()),
        );
        Ok((Arc::new(struct_array), field))
    }

    async fn preprocess_list_array<O: OffsetSizeTrait>(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        child: &BlobPreprocessField,
        write_sources: Option<(&Path, &BlobWriteSources)>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let list_arr = array.as_list::<O>();
        let list_arr = if list_arr.null_count() > 0 {
            list_arr.filter_garbage_nulls()
        } else {
            list_arr.clone()
        };

        let first_offset = *list_arr
            .offsets()
            .first()
            .ok_or_else(|| Error::invalid_input("List offsets cannot be empty"))?;
        let last_offset = *list_arr
            .offsets()
            .last()
            .ok_or_else(|| Error::invalid_input("List offsets cannot be empty"))?;
        let values_len = list_arr.values().len();
        let needs_trim = first_offset != O::zero()
            || last_offset.to_usize().ok_or_else(|| {
                Error::invalid_input(format!(
                    "List field '{}' offset does not fit into usize",
                    field.name()
                ))
            })? != values_len;

        let (offsets, values) = if needs_trim {
            let values = list_arr.trimmed_values();
            let offsets = list_arr
                .offsets()
                .iter()
                .map(|offset| *offset - first_offset)
                .collect::<Vec<_>>();
            (OffsetBuffer::new(ScalarBuffer::from(offsets)), values)
        } else {
            (list_arr.offsets().clone(), list_arr.values().clone())
        };

        let child_field = match field.data_type() {
            ArrowDataType::List(child_field) | ArrowDataType::LargeList(child_field) => {
                child_field.clone()
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Blob list preprocessor expected list field '{}', got {other}",
                    field.name()
                )));
            }
        };
        let (new_values, new_child_field) = self
            .preprocess_field(child, values, &child_field, write_sources)
            .await?;

        let list_array = GenericListArray::<O>::try_new(
            new_child_field,
            offsets,
            new_values,
            list_arr.nulls().cloned(),
        )?;
        let field = Arc::new(
            ArrowField::new(
                field.name(),
                list_array.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone()),
        );
        Ok((Arc::new(list_array), field))
    }

    async fn logical_to_prepared_blob_array(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        inline_threshold: usize,
        dedicated_threshold: usize,
        pack_file_threshold: usize,
        writer_metadata: &HashMap<String, String>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        // Legacy byte input follows the same per-leaf preparation as logical v2 input.
        let array = if matches!(
            array.data_type(),
            ArrowDataType::Binary | ArrowDataType::LargeBinary
        ) {
            Arc::new(StructArray::try_new(
                lance_core::datatypes::BLOB_V2_LOGICAL_MINIMAL_FIELDS.clone(),
                vec![
                    arrow_cast::cast(&array, &ArrowDataType::LargeBinary)?,
                    arrow_array::new_null_array(&ArrowDataType::Utf8, array.len()),
                ],
                array.nulls().cloned(),
            )?) as ArrayRef
        } else {
            array
        };
        let struct_arr = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input("Blob column was not a struct array"))?;
        if BlobV2Layout::classify(struct_arr.fields()) != Some(BlobV2Layout::Logical) {
            let actual = BlobV2Layout::classify(struct_arr.fields())
                .map(|layout| layout.to_string())
                .unwrap_or_else(|| format!("unrecognized ({:?})", struct_arr.fields()));
            return Err(Error::invalid_input(format!(
                "Blob v2 field '{}' has {actual} array layout; expected logical layout before preparation",
                field.name()
            )));
        }

        let data_col = struct_arr
            .column_by_name("data")
            .ok_or_else(|| Error::invalid_input("Blob struct missing `data` field"))?
            .as_binary::<i64>();
        let uri_col = struct_arr
            .column_by_name("uri")
            .ok_or_else(|| Error::invalid_input("Blob struct missing `uri` field"))?
            .as_string::<i32>();
        let position_col = struct_arr
            .column_by_name("position")
            .map(|col| col.as_primitive::<UInt64Type>());
        let size_col = struct_arr
            .column_by_name("size")
            .map(|col| col.as_primitive::<UInt64Type>());

        let mut blob_writer = self.blob_writer_with_metadata(field, writer_metadata.clone())?;

        for i in 0..struct_arr.len() {
            if struct_arr.is_null(i) {
                blob_writer.push_null()?;
                continue;
            }

            let has_data = !data_col.is_null(i);
            let has_uri = !uri_col.is_null(i);
            let has_position = position_col
                .as_ref()
                .map(|col| !col.is_null(i))
                .unwrap_or(false);
            let has_size = size_col
                .as_ref()
                .map(|col| !col.is_null(i))
                .unwrap_or(false);

            if has_position != has_size {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {i} must set both `position` and `size`, or neither",
                    field.name()
                )));
            }
            if has_position && !has_uri {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {i} sets `position` and `size` but `uri` is null",
                    field.name()
                )));
            }
            if has_data == has_uri {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {i} must set exactly one of `data` and `uri`",
                    field.name()
                )));
            }
            if has_size && size_col.as_ref().is_some_and(|col| col.value(i) == 0) {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {i} external range `size` must be greater than zero",
                    field.name()
                )));
            }

            let data_len = if has_data { data_col.value(i).len() } else { 0 };

            if has_data && data_len > dedicated_threshold {
                let value = self
                    .write_dedicated(BlobWriteSource::Bytes(data_col.value(i)))
                    .await?;
                blob_writer.push(value)?;
                continue;
            }

            if has_data && data_len > inline_threshold {
                let value = self
                    .write_packed(
                        pack_file_threshold,
                        BlobWriteSource::Bytes(data_col.value(i)),
                    )
                    .await?;
                blob_writer.push(value)?;
                continue;
            }

            if has_uri {
                let uri_val = uri_col.value(i);
                if self.external_blob_mode == ExternalBlobMode::Ingest {
                    let position = if has_position {
                        Some(
                            position_col
                                .as_ref()
                                .expect("position column must exist")
                                .value(i),
                        )
                    } else {
                        None
                    };
                    let size = if has_size {
                        Some(size_col.as_ref().expect("size column must exist").value(i))
                    } else {
                        None
                    };
                    let source = self.open_external_source(uri_val, position, size).await?;
                    let data_len = source.size();

                    if data_len > dedicated_threshold as u64 {
                        let value = self
                            .write_dedicated(BlobWriteSource::Reader(&source))
                            .await?;
                        blob_writer.push(value)?;
                        continue;
                    }

                    if data_len > inline_threshold as u64 {
                        let value = self
                            .write_packed(pack_file_threshold, BlobWriteSource::Reader(&source))
                            .await?;
                        blob_writer.push(value)?;
                        continue;
                    }

                    let data = source.read_all().await?;
                    blob_writer.push_inline(data)?;
                    continue;
                }

                let (external_base_id, external_uri_or_path) =
                    self.resolve_external_reference(uri_val).await?;
                let range = if has_position && has_size {
                    BlobRange {
                        offset: position_col
                            .as_ref()
                            .expect("position column must exist")
                            .value(i),
                        size: size_col.as_ref().expect("size column must exist").value(i),
                    }
                } else {
                    BlobRange { offset: 0, size: 0 }
                };
                blob_writer.push(BlobDescriptor::External {
                    base_id: external_base_id,
                    uri: external_uri_or_path,
                    offset: range.offset,
                    size: range.size,
                })?;
                continue;
            }

            if has_data {
                blob_writer.push_inline(Bytes::copy_from_slice(data_col.value(i)))?;
            } else {
                blob_writer.push_null()?;
            }
        }

        let column = blob_writer.finish()?;
        let (field, array) = column.into_parts();
        Ok((array, Arc::new(field)))
    }

    #[doc(hidden)]
    pub async fn finish(&mut self) -> Result<()> {
        if let Some(pack) = self.current_pack.take() {
            pack.writer.finish().await?;
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn abort(&mut self) {
        self.current_pack.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::test_util::reject_empty_range_store;
    use crate::blob::{BlobArrayBuilder, blob_field};
    use arrow_schema::{Field, Schema};
    use lance_arrow::{
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
        BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
    };
    use lance_io::testing::MockObjectStore;
    use object_store::{ObjectStore as _, ObjectStoreExt, memory::InMemory};
    use rstest::rstest;

    #[rstest]
    #[case::exact(4096)]
    #[case::short(4095)]
    #[case::long(4097)]
    #[tokio::test]
    async fn cloud_blob_upload_validates_size_with_one_get(#[case] actual_size: usize) {
        let expected_size = 4096;
        let body = Bytes::from(vec![7; actual_size]);
        let root = Path::from("table");
        let relative_path = Path::from("_blobs/ingest/attempt/value.payload");
        let path = root.parts().chain(relative_path.parts()).collect::<Path>();
        let backend = Arc::new(InMemory::new());
        backend.put(&path, body.clone().into()).await.unwrap();
        let mut cloud = MockObjectStore::new();
        cloud
            .expect_get_opts()
            .once()
            .withf(move |location, options| {
                location == &path && !options.head && options.range.is_none()
            })
            .returning(move |location, options| {
                let backend = backend.clone();
                let location = location.clone();
                Box::pin(async move { backend.get_opts(&location, options).await })
            });
        let store = ObjectStore::new(
            Arc::new(cloud),
            url::Url::parse(concat!("s3", "://", "fixture")).unwrap(),
            Some(1024),
            None,
            false,
            true,
            1,
            0,
            None,
        );
        let schema =
            lance_core::datatypes::Schema::try_from(&Schema::new(vec![blob_field("blob", true)]))
                .unwrap();
        let mut writer = BlobPreprocessor::new(
            store,
            root.child("data"),
            "output".into(),
            &schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            None,
        )
        .unwrap();
        let source = BlobWriteObject::Upload {
            relative_path,
            size: expected_size as u64,
        };
        let result = writer
            .write_registered_blob(&source, &root, expected_size, usize::MAX, usize::MAX)
            .await;
        if actual_size == expected_size {
            let BlobDescriptor::Inline { data } = result.unwrap() else {
                panic!("the fixture should be inlined");
            };
            assert_eq!(data, body);
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error.find_root(), Error::InvalidInput { .. }));
            assert!(
                error.to_string().contains("expected complete file"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn compaction_rewrites_empty_packed_blob_without_source_io() {
        let field = crate::blob::blob_field("blob", true);
        let schema =
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![field]))
                .unwrap();
        let mut preprocessor = BlobPreprocessor::new(
            ObjectStore::memory(),
            Path::from("data"),
            "output".into(),
            &schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            None,
        )
        .unwrap()
        .with_sidecar_rewrite(Some(Arc::new(super::super::SidecarRewrite::default())));
        let mut values = BlobDescriptorArrayBuilder::new("blob");
        values
            .push_packed(
                1,
                BlobRange {
                    offset: 123,
                    size: 0,
                },
            )
            .unwrap();
        let (field, array) = values.finish().unwrap().into_parts();
        let (array, _) = preprocessor
            .rewrite_sidecars(array, &field, 1024, field.metadata())
            .await
            .unwrap();
        let array = array.as_struct();
        assert_eq!(
            array
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Inline as u8
        );
        assert_eq!(
            array
                .column_by_name("data")
                .unwrap()
                .as_binary::<i64>()
                .value(0),
            b""
        );
    }

    #[tokio::test]
    async fn test_blob_reader_source_read_all_empty_range_returns_empty_bytes() {
        let store = reject_empty_range_store();
        let reader = store.open(&Path::from("blobs/test.bin")).await.unwrap();
        let source = BlobReaderSource {
            reader,
            start: 0,
            size: 0,
            max_request_size: usize::MAX,
        };

        assert!(source.read_all().await.unwrap().is_empty());
    }

    async fn try_preprocess_blobs_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        pack_file_size_override: Option<usize>,
        blob_len: usize,
        num_blobs: usize,
    ) -> Result<arrow_array::StructArray> {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        for (key, value) in metadata_entries {
            metadata.insert(key.to_string(), value);
        }
        field = field.with_metadata(metadata);

        let writer_arrow_schema = Schema::new(vec![field.clone()]);
        let writer_schema = lance_core::datatypes::Schema::try_from(&writer_arrow_schema).unwrap();

        let mut preprocessor = BlobPreprocessor::new(
            object_store.clone(),
            data_dir,
            "data_file_key".to_string(),
            &writer_schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            pack_file_size_override,
        )?;

        let mut blob_builder = BlobArrayBuilder::new(num_blobs);
        for _ in 0..num_blobs {
            blob_builder.push_bytes(vec![0u8; blob_len]).unwrap();
        }
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let field_without_metadata =
            Field::new("blob", field.data_type().clone(), field.is_nullable());
        let batch_schema = Arc::new(Schema::new(vec![field_without_metadata]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await?;
        Ok(out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .clone())
    }

    async fn try_preprocess_kind_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        data_len: usize,
    ) -> Result<u8> {
        let struct_arr =
            try_preprocess_blobs_with_blob_metadata(metadata_entries, None, data_len, 1).await?;
        Ok(struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .value(0))
    }

    async fn preprocess_kind_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        data_len: usize,
    ) -> u8 {
        try_preprocess_kind_with_blob_metadata(metadata_entries, data_len)
            .await
            .unwrap()
    }

    async fn packed_blobs_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        pack_file_size_override: Option<usize>,
        blob_len: usize,
        num_blobs: usize,
    ) -> Vec<u32> {
        let struct_arr = try_preprocess_blobs_with_blob_metadata(
            metadata_entries,
            pack_file_size_override,
            blob_len,
            num_blobs,
        )
        .await
        .unwrap();
        let blob_ids = struct_arr
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt32Type>();
        (0..struct_arr.len()).map(|i| blob_ids.value(i)).collect()
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_rejects_non_positive_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "0".to_string())],
            256 * 1024,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("expected a positive integer"));
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_rejects_invalid_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
                "not-a-number".to_string(),
            )],
            256 * 1024,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected a non-negative integer that fits in usize")
        );
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "131072".to_string())],
            256 * 1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_larger_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
                "8388608".to_string(),
            )],
            DEDICATED_THRESHOLD + 1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_rolls_at_metadata_value() {
        // Blobs must exceed the inline cutoff to be packed at all. With a pack-file
        // threshold equal to a single blob's size, each of the three blobs rolls to its
        // own pack file (a distinct blob_id).
        let blob_len = INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, blob_len.to_string())],
            None,
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            3,
            "expected one pack file per blob: {blob_ids:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_write_param_overrides_metadata() {
        let blob_len = INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(
                BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
                (blob_len * 3).to_string(),
            )],
            Some(blob_len),
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            3,
            "write-param override should force one pack file per blob: {blob_ids:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_packs_within_metadata_value() {
        // A pack-file threshold large enough for all three blobs keeps them in a single
        // pack file (one shared blob_id).
        let blob_len = INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(
                BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
                (blob_len * 3).to_string(),
            )],
            None,
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            1,
            "expected a single shared pack file: {blob_ids:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_rejects_non_positive_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, "0".to_string())],
            1024,
        )
        .await
        .expect_err("zero pack-file threshold should be rejected");
        assert!(err.to_string().contains("expected a positive integer"));
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "1024".to_string())],
            2048,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_respects_larger_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
                (INLINE_MAX + 8192).to_string(),
            )],
            INLINE_MAX + 4096,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_uses_strict_greater_than() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "1024".to_string())],
            1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_uses_strict_greater_than() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![
                (BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "2048".to_string()),
                (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "1024".to_string()),
            ],
            1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_does_not_override_dedicated_threshold() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![
                (BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "8192".to_string()),
                (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "4096".to_string()),
            ],
            6144,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_is_per_column() {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut inline_field = blob_field("inline_blob", true);
        let mut inline_metadata = inline_field.metadata().clone();
        inline_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "4096".to_string(),
        );
        inline_field = inline_field.with_metadata(inline_metadata);

        let mut packed_field = blob_field("packed_blob", true);
        let mut packed_metadata = packed_field.metadata().clone();
        packed_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        packed_field = packed_field.with_metadata(packed_metadata);

        let writer_arrow_schema = Schema::new(vec![inline_field.clone(), packed_field.clone()]);
        let writer_schema = lance_core::datatypes::Schema::try_from(&writer_arrow_schema).unwrap();

        let mut preprocessor = BlobPreprocessor::new(
            object_store.clone(),
            data_dir,
            "data_file_key".to_string(),
            &writer_schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            None,
        )
        .unwrap();

        let mut inline_builder = BlobArrayBuilder::new(1);
        inline_builder.push_bytes(vec![0u8; 2048]).unwrap();
        let inline_array: arrow_array::ArrayRef = inline_builder.finish().unwrap();

        let mut packed_builder = BlobArrayBuilder::new(1);
        packed_builder.push_bytes(vec![0u8; 2048]).unwrap();
        let packed_array: arrow_array::ArrayRef = packed_builder.finish().unwrap();

        let batch_schema = Arc::new(Schema::new(vec![
            Field::new(
                "inline_blob",
                inline_field.data_type().clone(),
                inline_field.is_nullable(),
            ),
            Field::new(
                "packed_blob",
                packed_field.data_type().clone(),
                packed_field.is_nullable(),
            ),
        ]));
        let batch = RecordBatch::try_new(batch_schema, vec![inline_array, packed_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        let inline_kind = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .value(0);
        let packed_kind = out
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .value(0);

        assert_eq!(inline_kind, lance_core::datatypes::BlobKind::Inline as u8);
        assert_eq!(packed_kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_starts_new_packed_blob() {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "0".to_string(),
        );
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        field = field.with_metadata(metadata);

        let writer_arrow_schema = Schema::new(vec![field.clone()]);
        let writer_schema = lance_core::datatypes::Schema::try_from(&writer_arrow_schema).unwrap();

        let mut preprocessor = BlobPreprocessor::new(
            object_store.clone(),
            data_dir,
            "data_file_key".to_string(),
            &writer_schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            Some(3),
        )
        .unwrap();

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(vec![0x11; 2]).unwrap();
        blob_builder.push_bytes(vec![0x22; 2]).unwrap();
        blob_builder.push_bytes(vec![0x33; 1]).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let batch_schema = Arc::new(Schema::new(vec![Field::new(
            "blob",
            field.data_type().clone(),
            field.is_nullable(),
        )]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        preprocessor.finish().await.unwrap();

        let struct_arr = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        let kinds = struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();
        let blob_ids = struct_arr
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<UInt32Type>();
        let sizes = struct_arr
            .column_by_name("blob_size")
            .unwrap()
            .as_primitive::<UInt64Type>();
        let positions = struct_arr
            .column_by_name("position")
            .unwrap()
            .as_primitive::<UInt64Type>();

        assert_eq!(kinds.values(), &[BlobKind::Packed as u8; 3]);
        assert_eq!(blob_ids.values(), &[1, 2, 2]);
        assert_eq!(sizes.values(), &[2, 2, 1]);
        assert_eq!(positions.values(), &[0, 0, 2]);
    }
}
