// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builders and file-level writer helpers for Lance blob v2 columns.
//!
//! Logical blob input uses either `Struct<data: LargeBinary?, uri: Utf8?>` or the complete
//! `Struct<data: LargeBinary?, uri: Utf8?, position: UInt64?, size: UInt64?>` shape. In the
//! complete shape, `position` and `size` select a non-empty range within an external `uri` and
//! must be set together. Every non-null row must set exactly one of `data` and `uri`. File-level
//! blob preparation produces a kind-aware writer intermediate with `blob_id` and range fields.
//!
//! Managed blob writes and compaction reuse are opt-in. Set Arrow schema metadata
//! `lance.blob.managed` to `true` to enable them. Committing this opt-in
//! permanently requires Managed-aware readers and maintenance clients. Without
//! this metadata, logical writes use Inline, Packed, and Dedicated descriptors;
//! prepared Managed descriptors are rejected.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};

use arrow_array::{
    Array, ArrayRef, StructArray,
    builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder},
    cast::AsArray,
    types::{UInt8Type, UInt32Type, UInt64Type},
};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME, FieldExt,
};
use lance_core::{
    datatypes::{
        BLOB_V2_DESC_TYPE, BLOB_V2_LOGICAL_MINIMAL_FIELDS, BLOB_V2_PREPARED_TYPE, BlobKind,
        BlobV2Layout, Field as LanceField, Schema as LanceSchema,
    },
    utils::blob::blob_path,
};
use lance_io::{
    object_store::ObjectStore,
    traits::{Reader, WriteExt, Writer},
};
use object_store::path::Path;
use tokio::io::AsyncWriteExt;

use crate::{Error, Result};

/// Construct the Arrow field for a blob v2 column.
///
/// This helper constructs the minimal logical shape
/// `Struct<data: LargeBinary?, uri: Utf8?>`, tagged with
/// `ARROW:extension:name = "lance.blob.v2"`. Writers also accept the complete logical shape
/// with trailing `position: UInt64?` and `size: UInt64?` fields for external URI ranges.
pub fn blob_field(name: &str, nullable: bool) -> Field {
    blob_field_with_options(name, nullable, BlobFieldOptions::default())
}

/// Options for constructing a blob v2 field.
#[derive(Clone, Debug, Default)]
pub struct BlobFieldOptions {
    /// Maximum payload size to keep inline in the data file before using packed blob storage.
    pub inline_size_threshold: Option<usize>,
    /// Maximum payload size to store in packed blob storage before using dedicated blob storage.
    ///
    /// A zero threshold is invalid because dedicated blob storage is selected when
    /// the payload size is greater than this value.
    pub dedicated_size_threshold: Option<NonZeroUsize>,
}

impl BlobFieldOptions {
    /// Set the maximum payload size to keep inline in the data file.
    pub fn with_inline_size_threshold(mut self, threshold: usize) -> Self {
        self.inline_size_threshold = Some(threshold);
        self
    }

    /// Set the maximum payload size to store in packed blob storage.
    pub fn with_dedicated_size_threshold(mut self, threshold: NonZeroUsize) -> Self {
        self.dedicated_size_threshold = Some(threshold);
        self
    }
}

/// Construct the Arrow field for a blob v2 column with storage layout options.
///
/// This helper constructs the minimal logical shape
/// `Struct<data: LargeBinary?, uri: Utf8?>`, tagged with
/// `ARROW:extension:name = "lance.blob.v2"`. Writers also accept the complete logical shape
/// with trailing `position: UInt64?` and `size: UInt64?` fields for external URI ranges.
///
/// ```
/// # use lance::{BlobFieldOptions, blob_field_with_options};
/// let field = blob_field_with_options(
///     "blob",
///     true,
///     BlobFieldOptions::default().with_inline_size_threshold(16 * 1024),
/// );
/// assert_eq!(
///     field
///         .metadata()
///         .get("lance-encoding:blob-inline-size-threshold")
///         .map(String::as_str),
///     Some("16384"),
/// );
/// ```
pub fn blob_field_with_options(name: &str, nullable: bool, options: BlobFieldOptions) -> Field {
    let mut metadata = [(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    if let Some(threshold) = options.inline_size_threshold {
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.to_string(),
        );
    }
    if let Some(threshold) = options.dedicated_size_threshold {
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.get().to_string(),
        );
    }
    Field::new(
        name,
        DataType::Struct(BLOB_V2_LOGICAL_MINIMAL_FIELDS.clone()),
        nullable,
    )
    .with_metadata(metadata)
}

pub(crate) fn prepared_blob_field_with_metadata(
    name: &str,
    nullable: bool,
    metadata: HashMap<String, String>,
) -> Field {
    Field::new(name, BLOB_V2_PREPARED_TYPE.clone(), nullable).with_metadata(metadata)
}

/// `prepared` under the field ids `source` carries: its own, and each
/// same-named child's. A stored blob keeps only `data` and `uri`, under the ids
/// of its prepared children, so this is what keeps a generation's blob children
/// under the table's ids.
pub(crate) fn with_field_ids_of(prepared: Field, source: &Field) -> Result<Field> {
    let source = LanceField::try_from(source)?;
    Ok(crate::dataset::mem_wal::stamp_field_id(
        &prepared,
        std::slice::from_ref(&source),
    ))
}

pub(crate) fn logical_to_prepared_blob_field(field: &Field) -> Result<Field> {
    if field.is_blob_v2() {
        return match blob_v2_layout(field) {
            Some(BlobV2Layout::Logical) | Some(BlobV2Layout::Prepared) => with_field_ids_of(
                prepared_blob_field_with_metadata(
                    field.name(),
                    field.is_nullable(),
                    field.metadata().clone(),
                ),
                field,
            ),
            _ => Err(blob_v2_shape_error(
                field,
                &[BlobV2Layout::Logical, BlobV2Layout::Prepared],
            )),
        };
    }

    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|child| logical_to_prepared_blob_field(child.as_ref()).map(Arc::new))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        DataType::List(child) => DataType::List(Arc::new(logical_to_prepared_blob_field(child)?)),
        DataType::LargeList(child) => {
            DataType::LargeList(Arc::new(logical_to_prepared_blob_field(child)?))
        }
        _ => field.data_type().clone(),
    };
    Ok(field.clone().with_data_type(data_type))
}

pub(crate) fn logical_to_prepared_blob_schema(schema: &ArrowSchema) -> Result<ArrowSchema> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| logical_to_prepared_blob_field(field.as_ref()).map(Arc::new))
        .collect::<Result<Vec<_>>>()?;
    Ok(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ))
}

fn prepared_to_descriptor_blob_field(field: &Field) -> Result<Field> {
    if field.is_blob_v2() {
        return match blob_v2_layout(field) {
            Some(BlobV2Layout::Prepared) | Some(BlobV2Layout::Descriptor) => {
                Ok(
                    Field::new(field.name(), BLOB_V2_DESC_TYPE.clone(), field.is_nullable())
                        .with_metadata(field.metadata().clone()),
                )
            }
            Some(BlobV2Layout::Logical) => Ok(field.clone()),
            _ => Err(blob_v2_shape_error(
                field,
                &[BlobV2Layout::Prepared, BlobV2Layout::Descriptor],
            )),
        };
    }

    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|child| prepared_to_descriptor_blob_field(child.as_ref()).map(Arc::new))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        DataType::List(child) => {
            DataType::List(Arc::new(prepared_to_descriptor_blob_field(child)?))
        }
        DataType::LargeList(child) => {
            DataType::LargeList(Arc::new(prepared_to_descriptor_blob_field(child)?))
        }
        _ => field.data_type().clone(),
    };
    Ok(field.clone().with_data_type(data_type))
}

pub(crate) fn prepared_to_descriptor_blob_schema(schema: &ArrowSchema) -> Result<ArrowSchema> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| prepared_to_descriptor_blob_field(field.as_ref()).map(Arc::new))
        .collect::<Result<Vec<_>>>()?;
    Ok(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ))
}

fn logical_blob_lance_children() -> Result<Vec<LanceField>> {
    BLOB_V2_LOGICAL_MINIMAL_FIELDS
        .iter()
        .map(|field| LanceField::try_from(field.as_ref()))
        .collect()
}

pub(crate) fn blob_v2_layout(field: &Field) -> Option<BlobV2Layout> {
    if !field.is_blob_v2() {
        return None;
    }
    let DataType::Struct(fields) = field.data_type() else {
        return None;
    };
    BlobV2Layout::classify(fields)
}

pub(crate) fn blob_v2_shape_error(field: &Field, expected: &[BlobV2Layout]) -> Error {
    let expected = expected
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" or ");
    let actual = match field.data_type() {
        DataType::Struct(fields) => BlobV2Layout::classify(fields)
            .map(|layout| format!("{layout} layout"))
            .unwrap_or_else(|| format!("unrecognized layout {fields:?}")),
        data_type => format!("non-struct type {data_type}"),
    };
    Error::invalid_input(format!(
        "Blob v2 field '{}' has {actual}; expected {expected} layout",
        field.name(),
    ))
}

fn prepared_to_logical_blob_lance_field(field: &LanceField) -> Result<LanceField> {
    if field.is_blob_v2() {
        let arrow_field = Field::from(field);
        match blob_v2_layout(&arrow_field) {
            Some(BlobV2Layout::Prepared) => {
                let mut normalized = field.clone();
                let mut logical_children = logical_blob_lance_children()?;
                for logical_child in &mut logical_children {
                    let prepared_child = field
                        .children
                        .iter()
                        .find(|prepared_child| prepared_child.name == logical_child.name)
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "Prepared blob v2 field '{}' is missing logical child '{}'",
                                field.name, logical_child.name
                            ))
                        })?;
                    logical_child.id = prepared_child.id;
                    logical_child.parent_id = field.id;
                }
                normalized.children = logical_children;
                return Ok(normalized);
            }
            Some(BlobV2Layout::Logical) => return Ok(field.clone()),
            _ => {
                return Err(blob_v2_shape_error(
                    &arrow_field,
                    &[BlobV2Layout::Logical, BlobV2Layout::Prepared],
                ));
            }
        }
    }

    if field.children.is_empty() {
        return Ok(field.clone());
    }

    let normalized_children = field
        .children
        .iter()
        .map(prepared_to_logical_blob_lance_field)
        .collect::<Result<Vec<_>>>()?;

    Ok(LanceField {
        children: normalized_children,
        ..field.clone()
    })
}

pub(crate) fn prepared_to_logical_blob_schema(schema: &LanceSchema) -> Result<LanceSchema> {
    let fields = schema
        .fields
        .iter()
        .map(prepared_to_logical_blob_lance_field)
        .collect::<Result<Vec<_>>>()?;
    Ok(LanceSchema {
        fields,
        metadata: schema.metadata.clone(),
    })
}

#[derive(Clone, Debug)]
pub(crate) struct BlobIdAllocator {
    inner: Arc<BlobIdAllocatorInner>,
}

#[derive(Debug)]
struct BlobIdAllocatorInner {
    start_inclusive: u32,
    next: AtomicU32,
    end_exclusive: Option<u32>,
    state: Mutex<BlobIdAllocatorState>,
}

#[derive(Debug, Default)]
struct BlobIdAllocatorState {
    used: HashSet<u32>,
    allocated: HashSet<u32>,
}

impl BlobIdAllocator {
    pub(crate) fn new(start: u32) -> Self {
        Self {
            inner: Arc::new(BlobIdAllocatorInner {
                start_inclusive: start,
                next: AtomicU32::new(start),
                end_exclusive: None,
                state: Mutex::new(BlobIdAllocatorState::default()),
            }),
        }
    }

    pub(crate) fn from_range(range: Range<u32>) -> Result<Self> {
        if range.start == 0 || range.start >= range.end {
            return Err(Error::invalid_input(format!(
                "Blob ID range must be non-empty and start at 1 or greater, got {}..{}",
                range.start, range.end
            )));
        }
        Ok(Self {
            inner: Arc::new(BlobIdAllocatorInner {
                start_inclusive: range.start,
                next: AtomicU32::new(range.start),
                end_exclusive: Some(range.end),
                state: Mutex::new(BlobIdAllocatorState::default()),
            }),
        })
    }

    pub(crate) fn next(&self) -> Result<u32> {
        loop {
            let id = self.inner.next.load(Ordering::Relaxed);
            if id == u32::MAX || self.inner.end_exclusive.is_some_and(|end| id >= end) {
                return Err(Error::invalid_input(match self.inner.end_exclusive {
                    Some(end) => format!(
                        "Blob ID range exhausted before allocating another sidecar; range ends at {end}"
                    ),
                    None => "Blob id allocator exhausted u32 id space".to_string(),
                }));
            }
            if self
                .inner
                .next
                .compare_exchange(id, id + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let mut state =
                self.inner.state.lock().map_err(|_| {
                    Error::internal("Blob id allocator mutex was poisoned".to_string())
                })?;
            if state.used.insert(id) {
                state.allocated.insert(id);
                return Ok(id);
            }
        }
    }

    pub(crate) fn reserve(&self, id: u32) -> Result<()> {
        if id < self.inner.start_inclusive || self.inner.end_exclusive.is_some_and(|end| id >= end)
        {
            return Err(Error::invalid_input(match self.inner.end_exclusive {
                Some(end) => format!(
                    "Blob ID {id} is outside allocator range {}..{end}",
                    self.inner.start_inclusive
                ),
                None => format!(
                    "Blob ID {id} is below allocator start {}",
                    self.inner.start_inclusive
                ),
            }));
        }
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| Error::internal("Blob id allocator mutex was poisoned".to_string()))?;
        if state.allocated.contains(&id) {
            return Err(Error::invalid_input(format!(
                "Blob ID {id} was already allocated for a generated sidecar"
            )));
        }
        state.used.insert(id);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn allocated_ids(&self) -> Result<Vec<u32>> {
        let mut ids = self
            .inner
            .state
            .lock()
            .map_err(|_| Error::internal("Blob id allocator mutex was poisoned".to_string()))?
            .allocated
            .iter()
            .copied()
            .collect::<Vec<_>>();
        ids.sort_unstable();
        Ok(ids)
    }
}

fn validate_blob_id(blob_id: u32) -> Result<()> {
    if blob_id == 0 {
        return Err(Error::invalid_input("Blob id 0 is reserved"));
    }
    Ok(())
}

fn validate_prepared_blob_value_array(field: &Field, array: &ArrayRef) -> Result<()> {
    if blob_v2_layout(field) != Some(BlobV2Layout::Prepared) {
        return Err(blob_v2_shape_error(field, &[BlobV2Layout::Prepared]));
    }

    let struct_arr = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::invalid_input("Prepared blob column was not a struct array"))?;
    if BlobV2Layout::classify(struct_arr.fields()) != Some(BlobV2Layout::Prepared) {
        let actual = BlobV2Layout::classify(struct_arr.fields())
            .map(|layout| layout.to_string())
            .unwrap_or_else(|| format!("unrecognized ({:?})", struct_arr.fields()));
        return Err(Error::invalid_input(format!(
            "Prepared blob column '{}' has {actual} array layout; expected prepared layout",
            field.name()
        )));
    }
    let kind_col = struct_arr
        .column_by_name("kind")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `kind` field"))?
        .as_primitive::<UInt8Type>();
    let data_col = struct_arr
        .column_by_name("data")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `data` field"))?
        .as_binary::<i64>();
    let uri_col = struct_arr
        .column_by_name("uri")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `uri` field"))?
        .as_string::<i32>();
    let blob_id_col = struct_arr
        .column_by_name("blob_id")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `blob_id` field"))?
        .as_primitive::<UInt32Type>();
    let blob_size_col = struct_arr
        .column_by_name("blob_size")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `blob_size` field"))?
        .as_primitive::<UInt64Type>();
    let position_col = struct_arr
        .column_by_name("position")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `position` field"))?
        .as_primitive::<UInt64Type>();

    for row in 0..struct_arr.len() {
        if struct_arr.is_null(row) {
            continue;
        }
        if kind_col.is_null(row) {
            return Err(Error::invalid_input(format!(
                "Prepared blob row {row} is non-null but `kind` is null"
            )));
        }

        match BlobKind::try_from(kind_col.value(row))? {
            BlobKind::Inline => {
                if data_col.is_null(row) {
                    return Err(Error::invalid_input(format!(
                        "Prepared inline blob row {row} must set `data`"
                    )));
                }
            }
            BlobKind::Packed => {
                if blob_id_col.is_null(row)
                    || blob_size_col.is_null(row)
                    || position_col.is_null(row)
                {
                    return Err(Error::invalid_input(format!(
                        "Prepared packed blob row {row} must set `blob_id`, `blob_size`, and `position`"
                    )));
                }
                validate_blob_id(blob_id_col.value(row))?;
                let offset = position_col.value(row);
                let size = blob_size_col.value(row);
                offset.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Prepared packed blob row {row} range overflows u64: offset={offset}, size={size}"
                    ))
                })?;
            }
            BlobKind::Dedicated => {
                if blob_id_col.is_null(row) || blob_size_col.is_null(row) {
                    return Err(Error::invalid_input(format!(
                        "Prepared dedicated blob row {row} must set `blob_id` and `blob_size`"
                    )));
                }
                validate_blob_id(blob_id_col.value(row))?;
            }
            BlobKind::Managed | BlobKind::ManagedWithBase => {
                if uri_col.is_null(row)
                    || ((kind_col.value(row) == BlobKind::ManagedWithBase as u8)
                        != blob_id_col.is_valid(row))
                    || blob_size_col.is_null(row)
                    || position_col.is_null(row)
                {
                    return Err(Error::invalid_input(format!(
                        "Prepared Managed blob row {row} requires `uri`, `blob_size`, and `position`, plus `blob_id` for an explicit base"
                    )));
                }
                lance_core::utils::blob::validate_managed_reference(
                    uri_col.value(row),
                    position_col.value(row),
                    blob_size_col.value(row),
                )?;
            }
            BlobKind::External => {
                if uri_col.is_null(row) || uri_col.value(row).is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Prepared external blob row {row} must set a non-empty `uri`"
                    )));
                }
                let offset = if position_col.is_null(row) {
                    0
                } else {
                    position_col.value(row)
                };
                let size = if blob_size_col.is_null(row) {
                    0
                } else {
                    blob_size_col.value(row)
                };
                offset.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Prepared external blob row {row} range overflows u64: offset={offset}, size={size}"
                    ))
                })?;
            }
        }
    }

    Ok(())
}

/// Validate a writer-side prepared blob v2 array before it reaches the encoder.
pub(crate) fn validate_prepared_blob_array(field: &Field, array: &ArrayRef) -> Result<()> {
    validate_prepared_blob_value_array(field, array)
}

pub(crate) fn sidecar_path_for_data_file(data_file_path: &Path, blob_id: u32) -> Result<Path> {
    validate_blob_id(blob_id)?;
    let file_name = data_file_path.filename().ok_or_else(|| {
        Error::invalid_input("Data file path must include a file name".to_string())
    })?;
    let data_file_key = file_name.strip_suffix(".lance").ok_or_else(|| {
        Error::invalid_input(format!(
            "Data file path '{}' must end with '.lance'",
            data_file_path
        ))
    })?;
    let data_dir = data_file_path.parent().unwrap_or_default();
    Ok(blob_path(&data_dir, data_file_key, blob_id))
}

/// Byte range inside a packed or external blob object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobRange {
    /// Byte offset relative to the beginning of the blob object.
    pub offset: u64,
    /// Number of bytes in this range.
    pub size: u64,
}

/// A kind-aware row used to build the writer-prepared blob representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlobDescriptor {
    /// A null blob row.
    Null,
    /// Payload bytes embedded into the data file by the blob encoder.
    Inline { data: Bytes },
    /// Payload bytes stored as a range in a packed sidecar blob.
    Packed {
        blob_id: u32,
        offset: u64,
        size: u64,
    },
    /// Payload bytes stored as the full contents of a dedicated sidecar blob.
    Dedicated { blob_id: u32, size: u64 },
    /// A known range in an immutable Lance-owned object. `None` uses the
    /// writer's table base; `Some(id)` requires an already registered base.
    Managed {
        base_id: Option<u32>,
        uri: String,
        offset: u64,
        size: u64,
    },
    /// Payload bytes referenced from an external object or registered base.
    External {
        base_id: u32,
        uri: String,
        offset: u64,
        size: u64,
    },
}

/// A writer-prepared blob column ready to be included in a [`RecordBatch`](arrow_array::RecordBatch).
///
/// Despite the legacy type name, its Arrow representation is
/// [`BlobV2Layout::Prepared`], not the descriptor stored in a Lance file.
pub struct BlobDescriptorColumn {
    field: Field,
    array: ArrayRef,
}

impl BlobDescriptorColumn {
    /// Return the Arrow field for the prepared column.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Return the Arrow array for the prepared column.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Consume this column into `(field, array)` parts.
    pub fn into_parts(self) -> (Field, ArrayRef) {
        (self.field, self.array)
    }
}

/// Builds the writer-prepared representation for one blob v2 column.
///
/// This builder only produces the writer-prepared struct array. It does not allocate blob ids,
/// choose sidecar paths, write blob objects, or commit data files.
pub struct BlobDescriptorArrayBuilder {
    field: Field,
    values: Vec<BlobDescriptor>,
}

impl BlobDescriptorArrayBuilder {
    /// Create a descriptor array builder for one blob column.
    pub fn new(column: impl Into<String>) -> Self {
        let mut metadata = HashMap::with_capacity(1);
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        Self::new_with_metadata(column, true, metadata)
    }

    pub(crate) fn new_with_metadata(
        column: impl Into<String>,
        nullable: bool,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            field: prepared_blob_field_with_metadata(&column.into(), nullable, metadata),
            values: Vec::new(),
        }
    }

    /// Prepare the column under the field ids `source` carries.
    pub(crate) fn with_field_ids_of(mut self, source: &Field) -> Result<Self> {
        self.field = with_field_ids_of(self.field, source)?;
        Ok(self)
    }

    /// Append one packed blob descriptor.
    pub fn push_packed(&mut self, blob_id: u32, range: BlobRange) -> Result<()> {
        self.push(BlobDescriptor::Packed {
            blob_id,
            offset: range.offset,
            size: range.size,
        })
    }

    /// Append multiple packed blob descriptors for the same blob object.
    pub fn extend_packed(
        &mut self,
        blob_id: u32,
        ranges: impl IntoIterator<Item = BlobRange>,
    ) -> Result<()> {
        for range in ranges {
            self.push_packed(blob_id, range)?;
        }
        Ok(())
    }

    /// Append one dedicated blob descriptor.
    pub fn push_dedicated(&mut self, blob_id: u32, size: u64) -> Result<()> {
        self.push(BlobDescriptor::Dedicated { blob_id, size })
    }

    /// Append one blob descriptor to this column.
    pub fn push(&mut self, value: BlobDescriptor) -> Result<()> {
        validate_blob_descriptor(&value)?;
        self.values.push(value);
        Ok(())
    }

    /// Append multiple blob descriptors to this column.
    pub fn extend(&mut self, values: impl IntoIterator<Item = BlobDescriptor>) -> Result<()> {
        for value in values {
            self.push(value)?;
        }
        Ok(())
    }

    /// Append an inline blob value.
    pub fn push_inline(&mut self, data: impl Into<Bytes>) -> Result<()> {
        self.push(BlobDescriptor::Inline { data: data.into() })
    }

    /// Append an external blob reference.
    pub fn push_external(
        &mut self,
        uri: impl Into<String>,
        range: Option<BlobRange>,
    ) -> Result<()> {
        let range = range.unwrap_or(BlobRange { offset: 0, size: 0 });
        self.push(BlobDescriptor::External {
            base_id: 0,
            uri: uri.into(),
            offset: range.offset,
            size: range.size,
        })
    }

    /// Append a null blob row.
    pub fn push_null(&mut self) -> Result<()> {
        self.push(BlobDescriptor::Null)
    }

    /// Return the prepared Arrow field for this blob column.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Finish this column and return the writer-prepared struct array.
    pub fn finish(self) -> Result<BlobDescriptorColumn> {
        let mut kind_builder = PrimitiveBuilder::<UInt8Type>::with_capacity(self.values.len());
        let mut data_builder = LargeBinaryBuilder::with_capacity(self.values.len(), 0);
        let mut uri_builder = StringBuilder::with_capacity(self.values.len(), 0);
        let mut blob_id_builder = PrimitiveBuilder::<UInt32Type>::with_capacity(self.values.len());
        let mut blob_size_builder =
            PrimitiveBuilder::<UInt64Type>::with_capacity(self.values.len());
        let mut position_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(self.values.len());
        let mut validity = NullBufferBuilder::new(self.values.len());

        for value in self.values {
            match value {
                BlobDescriptor::Null => {
                    validity.append_null();
                    kind_builder.append_null();
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                }
                BlobDescriptor::Inline { data } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Inline as u8);
                    data_builder.append_value(data.as_ref());
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                }
                BlobDescriptor::Packed {
                    blob_id,
                    offset,
                    size,
                } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Packed as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_value(offset);
                }
                BlobDescriptor::Dedicated { blob_id, size } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Dedicated as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_null();
                }
                BlobDescriptor::Managed {
                    base_id,
                    uri,
                    offset,
                    size,
                } => {
                    validity.append_non_null();
                    kind_builder.append_value(if base_id.is_some() {
                        BlobKind::ManagedWithBase as u8
                    } else {
                        BlobKind::Managed as u8
                    });
                    data_builder.append_null();
                    uri_builder.append_value(uri);
                    blob_id_builder.append_option(base_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_value(offset);
                }
                BlobDescriptor::External {
                    base_id,
                    uri,
                    offset,
                    size,
                } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::External as u8);
                    data_builder.append_null();
                    uri_builder.append_value(uri);
                    blob_id_builder.append_value(base_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_value(offset);
                }
            }
        }

        let DataType::Struct(children) = self.field.data_type() else {
            return Err(Error::internal(format!(
                "prepared blob field '{}' is not a struct",
                self.field.name()
            )));
        };
        let array = Arc::new(StructArray::try_new(
            children.clone(),
            vec![
                Arc::new(kind_builder.finish()),
                Arc::new(data_builder.finish()),
                Arc::new(uri_builder.finish()),
                Arc::new(blob_id_builder.finish()),
                Arc::new(blob_size_builder.finish()),
                Arc::new(position_builder.finish()),
            ],
            validity.finish(),
        )?) as ArrayRef;
        validate_prepared_blob_array(&self.field, &array)?;

        Ok(BlobDescriptorColumn {
            field: self.field,
            array,
        })
    }
}

fn validate_blob_descriptor(value: &BlobDescriptor) -> Result<()> {
    match value {
        BlobDescriptor::Null => Ok(()),
        BlobDescriptor::Inline { .. } => Ok(()),
        BlobDescriptor::Packed {
            blob_id,
            offset,
            size,
        } => {
            validate_blob_id(*blob_id)?;
            offset.checked_add(*size).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Packed blob range overflows u64: offset={offset}, size={size}"
                ))
            })?;
            Ok(())
        }
        BlobDescriptor::Dedicated { blob_id, .. } => validate_blob_id(*blob_id),
        BlobDescriptor::Managed {
            uri, offset, size, ..
        } => {
            lance_core::utils::blob::validate_managed_reference(uri, *offset, *size)?;
            Ok(())
        }
        BlobDescriptor::External {
            uri, offset, size, ..
        } => {
            if uri.is_empty() {
                return Err(Error::invalid_input("External blob URI cannot be empty"));
            }
            offset.checked_add(*size).ok_or_else(|| {
                Error::invalid_input(format!(
                    "External blob range overflows u64: offset={offset}, size={size}"
                ))
            })?;
            Ok(())
        }
    }
}

fn blob_range(offset: u64, size: u64) -> Result<(BlobRange, u64)> {
    let next_offset = offset.checked_add(size).ok_or_else(|| {
        Error::invalid_input(format!(
            "Blob object writer offset overflowed: offset={offset}, size={size}"
        ))
    })?;
    Ok((BlobRange { offset, size }, next_offset))
}

/// Writes payload ranges into one object without choosing their addressing scheme.
///
/// Each write takes ownership of the active upload so a partial failure or
/// cancellation drops it and prevents reusing incomplete payloads.
pub(crate) struct BlobObjectWriter {
    path: Path,
    writer: Option<Box<dyn Writer>>,
    offset: u64,
}

impl BlobObjectWriter {
    pub(crate) async fn try_new(object_store: ObjectStore, path: Path) -> Result<Self> {
        let writer = object_store.create(&path).await?;
        Ok(Self {
            path,
            writer: Some(writer),
            offset: 0,
        })
    }

    pub(crate) fn size(&self) -> u64 {
        self.offset
    }

    async fn write_blobs<'a>(
        &mut self,
        blobs: impl IntoIterator<Item = Option<&'a [u8]>>,
    ) -> Result<Vec<Option<BlobRange>>> {
        let mut writer = self.take_writer()?;
        let mut ranges = Vec::new();
        let mut next_offset = self.offset;
        for blob in blobs {
            let Some(blob) = blob else {
                ranges.push(None);
                continue;
            };
            let (range, following_offset) = blob_range(next_offset, blob.len() as u64)?;
            if !blob.is_empty() {
                writer.write_all(blob).await?;
            }
            ranges.push(Some(range));
            next_offset = following_offset;
        }
        self.writer = Some(writer);
        self.offset = next_offset;
        Ok(ranges)
    }

    pub(crate) async fn write_blob(&mut self, bytes: &[u8]) -> Result<BlobRange> {
        let (range, next_offset) = blob_range(self.offset, bytes.len() as u64)?;
        let mut writer = self.take_writer()?;
        writer.write_all(bytes).await?;
        self.writer = Some(writer);
        self.offset = next_offset;
        Ok(range)
    }

    pub(crate) async fn write_blob_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<BlobRange> {
        let (value, next_offset) = blob_range(self.offset, range.len() as u64)?;
        let mut writer = self.take_writer()?;
        writer.copy_range_from_reader(reader, range).await?;
        self.writer = Some(writer);
        self.offset = next_offset;
        Ok(value)
    }

    fn take_writer(&mut self) -> Result<Box<dyn Writer>> {
        self.writer.take().ok_or_else(|| {
            Error::io(format!(
                "Blob object writer for '{}' has no active upload",
                self.path
            ))
        })
    }

    pub(crate) async fn finish(mut self) -> Result<u64> {
        let mut writer = self.take_writer()?;
        let result = Writer::shutdown(writer.as_mut()).await?;
        let object_size = result.size as u64;
        if object_size != self.offset {
            return Err(Error::io(format!(
                "Blob object '{}' has size {object_size}, expected {}",
                self.path, self.offset
            )));
        }
        Ok(object_size)
    }
}

/// An independent object address, shared by explicit uploads and dataset writes.
pub(crate) struct ManagedBlobLocation {
    base_id: Option<u32>,
    relative_path: Path,
}

impl ManagedBlobLocation {
    pub(crate) fn new(base_id: Option<u32>) -> Self {
        Self {
            base_id,
            relative_path: Path::from("_blobs").join(format!("{}.blob", uuid::Uuid::new_v4())),
        }
    }

    pub(crate) fn path(&self, base_path: &Path) -> Path {
        base_path
            .parts()
            .chain(self.relative_path.parts())
            .collect()
    }

    pub(crate) fn descriptor(&self, range: BlobRange) -> BlobDescriptor {
        BlobDescriptor::Managed {
            base_id: self.base_id,
            uri: self.relative_path.to_string(),
            offset: range.offset,
            size: range.size,
        }
    }
}

/// Writes a Lance-owned packed sidecar blob for one data file and returns descriptors.
///
/// Use [`ManagedBlobWriter`] to share payload references across data files.
pub struct PackedBlobWriter {
    upload: BlobObjectWriter,
    blob_id: u32,
    ranges: Vec<Option<BlobRange>>,
}

impl PackedBlobWriter {
    /// Create a packed blob writer for `data_file_path` and `blob_id`.
    pub async fn try_new(
        object_store: ObjectStore,
        data_file_path: Path,
        blob_id: u32,
    ) -> Result<Self> {
        let path = sidecar_path_for_data_file(&data_file_path, blob_id)?;
        Ok(Self {
            upload: BlobObjectWriter::try_new(object_store, path).await?,
            blob_id,
            ranges: Vec::new(),
        })
    }

    /// Return the blob id for this packed sidecar.
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    /// Return the derived sidecar path for this packed blob.
    pub fn path(&self) -> &Path {
        &self.upload.path
    }

    /// Append one logical blob payload to the packed sidecar.
    pub async fn write_blob(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        let range = self.upload.write_blob(bytes.as_ref()).await?;
        self.ranges.push(Some(range));
        Ok(())
    }

    /// Append multiple logical blobs, one per iterator item.
    ///
    /// Each `Some(bytes)` is appended to the sidecar and records a packed
    /// descriptor; an empty slice records a valid zero-length blob. Each `None`
    /// records a [`BlobDescriptor::Null`] without writing any bytes, so the
    /// descriptors returned by [`Self::finish`] stay row-aligned with the input.
    ///
    /// If writing fails or the future is cancelled after a partial write, no
    /// descriptors from this call are recorded, the active writer is dropped,
    /// and this instance cannot be reused.
    ///
    /// ```
    /// # use lance::{PackedBlobWriter, Result};
    /// # async fn write(mut writer: PackedBlobWriter) -> Result<()> {
    /// writer
    ///     .write_packed_blobs([Some(b"first".as_slice()), None, Some(b"second".as_slice())])
    ///     .await?;
    /// let descriptors = writer.finish().await?;
    /// assert_eq!(descriptors.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_packed_blobs<'a>(
        &mut self,
        blobs: impl IntoIterator<Item = Option<&'a [u8]>>,
    ) -> Result<()> {
        self.ranges.extend(self.upload.write_blobs(blobs).await?);
        Ok(())
    }

    /// Finish the packed sidecar and return descriptors in write order.
    pub async fn finish(self) -> Result<Vec<BlobDescriptor>> {
        self.upload.finish().await?;
        Ok(self
            .ranges
            .into_iter()
            .map(|range| {
                range.map_or(BlobDescriptor::Null, |range| BlobDescriptor::Packed {
                    blob_id: self.blob_id,
                    offset: range.offset,
                    size: range.size,
                })
            })
            .collect())
    }
}

/// Writes immutable payloads into an independently addressed Lance-owned object.
///
/// Its Managed descriptors can be reused across rows, fragments and ordinary
/// compaction. This writer does not publish a dataset version or register bases.
/// Use [`BlobDescriptorArrayBuilder`] and the normal dataset write APIs to commit
/// references to the completed object.
pub struct ManagedBlobWriter {
    upload: BlobObjectWriter,
    location: ManagedBlobLocation,
    ranges: Vec<Option<BlobRange>>,
}

impl ManagedBlobWriter {
    /// Create an independently addressed object under `base_path/_blobs`.
    ///
    /// `base_path` must be the table's primary root when `base_id` is `None`, or
    /// the root of the already registered base identified by `Some(id)`. The
    /// caller supplies an object store for that root. This does not register bases.
    /// Without an explicit base id, write the referencing data files in primary
    /// storage. Supply a registered payload base id when targeting other bases.
    ///
    /// [`Self::finish`] returns [`BlobDescriptor::Managed`] values. Reuse each
    /// descriptor in a [`BlobDescriptorArrayBuilder`] for as many rows as needed,
    /// including across file splits. Commit with schema metadata
    /// `lance.blob.managed=true`. Finish the upload before committing references;
    /// retain a referencing snapshot if concurrent cleanup can retire them.
    /// Uncommitted objects follow the normal unverified-object cleanup policy.
    ///
    /// ```
    /// # use std::{collections::HashMap, sync::Arc};
    /// # use arrow_array::{RecordBatch, RecordBatchIterator};
    /// # use arrow_schema::Schema;
    /// # use lance::{BlobDescriptorArrayBuilder, Dataset, ManagedBlobWriter, Result};
    /// # use lance::dataset::{WriteParams, optimize::compact_files};
    /// # use lance::io::ObjectStore;
    /// # use lance_file::version::LanceFileVersion;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let dir = tempfile::tempdir()?;
    /// # let location = dir.path().to_string_lossy();
    /// # let uri = location.as_ref();
    /// let (store, root) = ObjectStore::from_uri(uri).await?;
    /// let mut writer = ManagedBlobWriter::try_new((*store).clone(), root, None).await?;
    /// writer.write_blob(b"shared payload").await?;
    /// let references = writer.finish().await?;
    /// let mut builder = BlobDescriptorArrayBuilder::new("blob");
    /// builder.extend(vec![references[0].clone(); 4])?;
    /// let (field, array) = builder.finish()?.into_parts();
    /// let schema = Arc::new(Schema::new_with_metadata(
    ///     vec![field], HashMap::from([("lance.blob.managed".into(), "true".into())]),
    /// ));
    /// let batch = RecordBatch::try_new(schema.clone(), vec![array])?;
    /// let mut dataset = Dataset::write(
    ///     RecordBatchIterator::new(vec![Ok(batch)], schema), uri,
    ///     Some(WriteParams {
    ///         max_rows_per_file: 2,
    ///         data_storage_version: Some(LanceFileVersion::V2_2),
    ///         ..Default::default()
    ///     }),
    /// ).await?;
    /// assert_eq!(dataset.get_fragments().len(), 2);
    /// compact_files(&mut dataset, Default::default(), None).await?;
    /// let rows = Arc::new(dataset).read_blobs("blob")?
    ///     .with_row_indices(vec![0, 1, 2, 3]).execute().await?;
    /// assert!(rows.iter().all(|row| row.data.as_deref() == Some(b"shared payload")));
    /// # Ok(()) }
    /// ```
    pub async fn try_new(
        object_store: ObjectStore,
        base_path: Path,
        base_id: Option<u32>,
    ) -> Result<Self> {
        let location = ManagedBlobLocation::new(base_id);
        let path = location.path(&base_path);
        Ok(Self {
            upload: BlobObjectWriter::try_new(object_store, path).await?,
            location,
            ranges: Vec::new(),
        })
    }

    /// Return the object path in the writer's object store.
    pub fn path(&self) -> &Path {
        &self.upload.path
    }

    /// Append one payload. Each call records one descriptor returned by [`Self::finish`].
    pub async fn write_blob(&mut self, data: impl AsRef<[u8]>) -> Result<()> {
        let range = self.upload.write_blob(data.as_ref()).await?;
        self.ranges.push(Some(range));
        Ok(())
    }

    /// Append payloads in order, preserving null and empty values.
    ///
    /// `None` records a null descriptor without writing bytes. Empty bytes record
    /// a zero-length Managed range. A failed or cancelled write drops the active
    /// upload, making this writer unusable. Reuse completed descriptors rather
    /// than passing the same payload repeatedly to this method.
    pub async fn write_blobs<'a>(
        &mut self,
        payloads: impl IntoIterator<Item = Option<&'a [u8]>>,
    ) -> Result<()> {
        self.ranges.extend(self.upload.write_blobs(payloads).await?);
        Ok(())
    }

    /// Complete the upload and return Managed or null descriptors in write order.
    ///
    /// Finish before committing references. Without a committed reference, this
    /// object is subject to the existing unverified-object cleanup policy.
    pub async fn finish(self) -> Result<Vec<BlobDescriptor>> {
        self.upload.finish().await?;
        Ok(self
            .ranges
            .into_iter()
            .map(|range| {
                range.map_or(BlobDescriptor::Null, |range| {
                    self.location.descriptor(range)
                })
            })
            .collect())
    }
}

/// Writes a Lance-owned dedicated sidecar blob for one data file and returns its descriptor.
pub struct DedicatedBlobWriter {
    upload: BlobObjectWriter,
    blob_id: u32,
}

impl DedicatedBlobWriter {
    /// Create a dedicated blob writer for `data_file_path` and `blob_id`.
    pub async fn try_new(
        object_store: ObjectStore,
        data_file_path: Path,
        blob_id: u32,
    ) -> Result<Self> {
        let path = sidecar_path_for_data_file(&data_file_path, blob_id)?;
        Ok(Self {
            upload: BlobObjectWriter::try_new(object_store, path).await?,
            blob_id,
        })
    }

    /// Return the blob id for this dedicated sidecar.
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    /// Return the derived sidecar path for this dedicated blob.
    pub fn path(&self) -> &Path {
        &self.upload.path
    }

    /// Append bytes to the dedicated sidecar.
    ///
    /// A failed or cancelled write drops the active upload, making this writer
    /// unusable. All successful appends form the single payload returned by
    /// [`Self::finish`].
    pub async fn write(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        self.upload.write_blob(bytes.as_ref()).await?;
        Ok(())
    }

    pub(crate) async fn write_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<()> {
        self.upload.write_blob_from_reader(reader, range).await?;
        Ok(())
    }

    /// Finish the dedicated sidecar and return its descriptor.
    pub async fn finish(self) -> Result<BlobDescriptor> {
        let size = self.upload.finish().await?;
        Ok(BlobDescriptor::Dedicated {
            blob_id: self.blob_id,
            size,
        })
    }
}

/// Builder for blob v2 input struct columns.
///
/// The builder enforces that each row contains exactly one of `data` or `uri` (or is null).
pub struct BlobArrayBuilder {
    data_builder: LargeBinaryBuilder,
    uri_builder: StringBuilder,
    validity: NullBufferBuilder,
    expected_len: usize,
    len: usize,
}

impl BlobArrayBuilder {
    /// Create a new builder with the given row capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data_builder: LargeBinaryBuilder::with_capacity(capacity, 0),
            uri_builder: StringBuilder::with_capacity(capacity, 0),
            validity: NullBufferBuilder::new(capacity),
            expected_len: capacity,
            len: 0,
        }
    }

    /// Append a blob backed by raw bytes.
    pub fn push_bytes(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_non_null();
        self.data_builder.append_value(bytes);
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Append a blob referenced by URI.
    pub fn push_uri(&mut self, uri: impl Into<String>) -> Result<()> {
        self.ensure_capacity()?;
        let uri = uri.into();
        if uri.is_empty() {
            return Err(Error::invalid_input("URI cannot be empty"));
        }
        self.validity.append_non_null();
        self.data_builder.append_null();
        self.uri_builder.append_value(uri);
        self.len += 1;
        Ok(())
    }

    /// Append an empty blob (inline, zero-length payload).
    pub fn push_empty(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_non_null();
        self.data_builder.append_value([]);
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Append a null row.
    pub fn push_null(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_null();
        self.data_builder.append_null();
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Finish building and return an Arrow struct array.
    pub fn finish(mut self) -> Result<ArrayRef> {
        if self.len != self.expected_len {
            return Err(Error::invalid_input(format!(
                "Expected {} rows but received {}",
                self.expected_len, self.len
            )));
        }

        let data = Arc::new(self.data_builder.finish());
        let uri = Arc::new(self.uri_builder.finish());
        let validity = self.validity.finish();

        let struct_array = StructArray::try_new(
            BLOB_V2_LOGICAL_MINIMAL_FIELDS.clone(),
            vec![data as ArrayRef, uri as ArrayRef],
            validity,
        )?;

        Ok(Arc::new(struct_array))
    }

    fn ensure_capacity(&self) -> Result<()> {
        if self.len >= self.expected_len {
            Err(Error::invalid_input("BlobArrayBuilder capacity exceeded"))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::io;
    use std::num::NonZeroUsize;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, StringArray};
    use arrow_schema::{Fields, Schema as ArrowSchema};
    use async_trait::async_trait;
    use futures::task::noop_waker;
    use lance_core::datatypes::BLOB_V2_PREPARED_FIELDS;
    use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BLOB_V2_LOGICAL_FIELDS};
    use lance_core::utils::tempfile::TempDir;
    use lance_io::object_writer::WriteResult;
    use rstest::rstest;
    use tokio::io::AsyncWrite;

    #[derive(Clone, Copy)]
    enum WriteTerminal {
        Error,
        Pending,
    }

    struct PartialWriter {
        bytes_before_terminal: usize,
        terminal: WriteTerminal,
        bytes_written: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
    }

    impl AsyncWrite for PartialWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            bytes: &[u8],
        ) -> Poll<io::Result<usize>> {
            if self.bytes_before_terminal > 0 {
                let written = self.bytes_before_terminal.min(bytes.len());
                self.bytes_before_terminal -= written;
                self.bytes_written.fetch_add(written, Ordering::SeqCst);
                return Poll::Ready(Ok(written));
            }
            match self.terminal {
                WriteTerminal::Error => {
                    Poll::Ready(Err(io::Error::other("injected write failure")))
                }
                WriteTerminal::Pending => Poll::Pending,
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[async_trait]
    impl Writer for PartialWriter {
        async fn tell(&mut self) -> Result<usize> {
            Ok(self.bytes_written.load(Ordering::SeqCst))
        }

        async fn shutdown(&mut self) -> Result<WriteResult> {
            Ok(WriteResult {
                size: self.bytes_written.load(Ordering::SeqCst),
                ..Default::default()
            })
        }
    }

    impl Drop for PartialWriter {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    fn partial_writer(
        terminal: WriteTerminal,
    ) -> (Box<dyn Writer>, Arc<AtomicUsize>, Arc<AtomicBool>) {
        let bytes_written = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        (
            Box::new(PartialWriter {
                bytes_before_terminal: 2,
                terminal,
                bytes_written: bytes_written.clone(),
                dropped: dropped.clone(),
            }),
            bytes_written,
            dropped,
        )
    }

    fn cancel_pending<F: Future>(future: F) {
        let mut future = Box::pin(future);
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
    }

    #[test]
    fn part_blob_id_allocator_stops_at_lease_end() {
        let allocator = BlobIdAllocator::from_range(7..8).unwrap();
        assert_eq!(allocator.next().unwrap(), 7);
        let error = allocator.next().unwrap_err();
        assert!(error.to_string().contains("range ends at 8"), "{error}");
        assert_eq!(allocator.allocated_ids().unwrap(), vec![7]);

        let allocator = BlobIdAllocator::from_range(7..9).unwrap();
        allocator.reserve(7).unwrap();
        assert_eq!(allocator.next().unwrap(), 8);
        assert_eq!(allocator.allocated_ids().unwrap(), vec![8]);
    }

    #[test]
    fn part_blob_id_allocator_rejects_generated_id_reservations() {
        let allocator = BlobIdAllocator::from_range(7..9).unwrap();
        assert_eq!(allocator.next().unwrap(), 7);
        let error = allocator.reserve(7).unwrap_err();
        assert!(error.to_string().contains("already allocated"), "{error}");

        allocator.reserve(8).unwrap();
        allocator.reserve(8).unwrap();
    }

    #[test]
    fn test_field_metadata() {
        let field = blob_field("blob", true);
        assert!(field.metadata().get(ARROW_EXT_NAME_KEY).is_some());
        assert_eq!(
            field.metadata().get(ARROW_EXT_NAME_KEY).unwrap(),
            BLOB_V2_EXT_NAME
        );
    }

    #[test]
    fn test_field_metadata_with_options() {
        let field = blob_field_with_options(
            "blob",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(16 * 1024)
                .with_dedicated_size_threshold(NonZeroUsize::new(2 * 1024 * 1024).unwrap()),
        );
        assert_eq!(
            field
                .metadata()
                .get(BLOB_INLINE_SIZE_THRESHOLD_META_KEY)
                .unwrap(),
            "16384"
        );
        assert_eq!(
            field
                .metadata()
                .get(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY)
                .unwrap(),
            "2097152"
        );
    }

    #[test]
    fn test_mem_wal_blob_schema_transforms_nested_fields() {
        let schema = ArrowSchema::new(vec![
            Field::new(
                "record",
                DataType::Struct(vec![blob_field("payload", true)].into()),
                true,
            ),
            Field::new(
                "items",
                DataType::List(Arc::new(blob_field("item", false))),
                true,
            ),
        ]);

        let prepared = logical_to_prepared_blob_schema(&schema).unwrap();
        let DataType::Struct(record_fields) = prepared.field(0).data_type() else {
            panic!("record must remain a struct");
        };
        assert_eq!(
            blob_v2_layout(record_fields[0].as_ref()),
            Some(BlobV2Layout::Prepared)
        );
        let DataType::List(item) = prepared.field(1).data_type() else {
            panic!("items must remain a list");
        };
        assert_eq!(blob_v2_layout(item), Some(BlobV2Layout::Prepared));

        let descriptor = prepared_to_descriptor_blob_schema(&prepared).unwrap();
        let DataType::Struct(record_fields) = descriptor.field(0).data_type() else {
            panic!("record must remain a struct");
        };
        assert_eq!(
            blob_v2_layout(record_fields[0].as_ref()),
            Some(BlobV2Layout::Descriptor)
        );
        let DataType::List(item) = descriptor.field(1).data_type() else {
            panic!("items must remain a list");
        };
        assert_eq!(blob_v2_layout(item), Some(BlobV2Layout::Descriptor));
    }

    #[test]
    fn test_builder_basic() {
        let mut b = BlobArrayBuilder::new(4);
        b.push_bytes(b"hi").unwrap();
        b.push_uri("s3://bucket/key").unwrap();
        b.push_empty().unwrap();
        b.push_null().unwrap();

        let arr = b.finish().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 1);

        let struct_arr = arr.as_struct();
        let data = struct_arr.column(0).as_binary::<i64>();
        let uri = struct_arr.column(1).as_string::<i32>();

        assert_eq!(data.value(0), b"hi");
        assert!(uri.is_null(0));
        assert!(data.is_null(1));
        assert_eq!(uri.value(1), "s3://bucket/key");
        assert_eq!(data.value(2).len(), 0);
        assert!(uri.is_null(2));
    }

    #[test]
    fn test_capacity_error() {
        let mut b = BlobArrayBuilder::new(1);
        b.push_bytes(b"a").unwrap();
        let err = b.push_bytes(b"b").unwrap_err();
        assert!(err.to_string().contains("capacity exceeded"));
    }

    #[test]
    fn test_empty_uri_rejected() {
        let mut b = BlobArrayBuilder::new(1);
        let err = b.push_uri("").unwrap_err();
        assert!(err.to_string().contains("URI cannot be empty"));
    }

    #[test]
    fn test_prepared_blob_column_finish() {
        let mut writer = BlobDescriptorArrayBuilder::new("blob");
        writer.push_inline(Bytes::from_static(b"hello")).unwrap();
        writer
            .push_packed(7, BlobRange { offset: 3, size: 5 })
            .unwrap();
        writer.push_dedicated(8, 9).unwrap();
        writer.push_null().unwrap();

        let column = writer.finish().unwrap();
        assert_eq!(blob_v2_layout(column.field()), Some(BlobV2Layout::Prepared));
        let struct_arr = column.array().as_struct();
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

        assert_eq!(kinds.value(0), BlobKind::Inline as u8);
        assert_eq!(kinds.value(1), BlobKind::Packed as u8);
        assert_eq!(blob_ids.value(1), 7);
        assert_eq!(positions.value(1), 3);
        assert_eq!(sizes.value(1), 5);
        assert_eq!(kinds.value(2), BlobKind::Dedicated as u8);
        assert_eq!(blob_ids.value(2), 8);
        assert_eq!(sizes.value(2), 9);
        assert!(struct_arr.is_null(3));
    }

    #[test]
    fn test_prepared_blob_array_rejects_range_overflow() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        let field = prepared_blob_field_with_metadata("blob", true, metadata);

        for (kind, uri) in [
            (BlobKind::Packed, None),
            (BlobKind::External, Some("file:///external.bin")),
        ] {
            let array = Arc::new(
                StructArray::try_new(
                    BLOB_V2_PREPARED_FIELDS.clone(),
                    vec![
                        Arc::new(arrow_array::UInt8Array::from(vec![kind as u8])) as ArrayRef,
                        Arc::new(arrow_array::LargeBinaryArray::from_iter([None::<&[u8]>])),
                        Arc::new(arrow_array::StringArray::from_iter([uri])),
                        Arc::new(arrow_array::UInt32Array::from(vec![1])),
                        Arc::new(arrow_array::UInt64Array::from(vec![4])),
                        Arc::new(arrow_array::UInt64Array::from(vec![u64::MAX - 1])),
                    ],
                    None,
                )
                .unwrap(),
            ) as ArrayRef;

            let err = validate_prepared_blob_array(&field, &array).unwrap_err();
            assert!(err.to_string().contains("range overflows u64"));
        }
    }

    #[derive(Clone, Copy)]
    enum LogicalBlobShape {
        Minimal,
        CompleteNullableRange,
        CompleteRequiredRange,
    }

    fn blob_v2_field_with_children(name: &str, children: Fields, nullable: bool) -> Field {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        metadata.insert("blob-root-metadata".to_string(), "preserved".to_string());
        Field::new(name, DataType::Struct(children), nullable).with_metadata(metadata)
    }

    fn logical_blob_fields(shape: LogicalBlobShape) -> Fields {
        let source = match shape {
            LogicalBlobShape::Minimal => &*BLOB_V2_LOGICAL_MINIMAL_FIELDS,
            LogicalBlobShape::CompleteNullableRange | LogicalBlobShape::CompleteRequiredRange => {
                &*BLOB_V2_LOGICAL_FIELDS
            }
        };
        source
            .iter()
            .enumerate()
            .map(|(index, field)| {
                let nullable = !matches!(shape, LogicalBlobShape::CompleteRequiredRange)
                    || index < BLOB_V2_LOGICAL_MINIMAL_FIELDS.len();
                Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_nullable(nullable)
                        .with_metadata(HashMap::from([(
                            "blob-child-metadata".to_string(),
                            field.name().to_string(),
                        )])),
                )
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn lance_schema_with_metadata(fields: Vec<Field>) -> LanceSchema {
        let arrow_schema = ArrowSchema::new_with_metadata(
            fields,
            HashMap::from([("schema-metadata".to_string(), "preserved".to_string())]),
        );
        let mut schema = LanceSchema::try_from(&arrow_schema).unwrap();
        schema.set_field_id(None);
        schema
    }

    #[rstest]
    #[case::minimal(LogicalBlobShape::Minimal)]
    #[case::complete_nullable_range(LogicalBlobShape::CompleteNullableRange)]
    #[case::complete_required_range(LogicalBlobShape::CompleteRequiredRange)]
    fn test_logical_blob_schema_normalization_is_identity(#[case] shape: LogicalBlobShape) {
        let blob_field = blob_v2_field_with_children("blob", logical_blob_fields(shape), false);
        let schema = lance_schema_with_metadata(vec![blob_field]);

        let normalized = prepared_to_logical_blob_schema(&schema).unwrap();

        assert_eq!(normalized.fields, schema.fields);
        assert_eq!(normalized.metadata, schema.metadata);
    }

    #[test]
    fn test_nested_logical_blob_schema_normalization_is_identity() {
        let struct_blob = blob_v2_field_with_children(
            "struct_blob",
            logical_blob_fields(LogicalBlobShape::CompleteNullableRange),
            true,
        );
        let list_blob = blob_v2_field_with_children(
            "item",
            logical_blob_fields(LogicalBlobShape::CompleteRequiredRange),
            false,
        );
        let schema = lance_schema_with_metadata(vec![
            Field::new("payload", DataType::Struct(vec![struct_blob].into()), true),
            Field::new("items", DataType::List(Arc::new(list_blob)), false),
        ]);

        let normalized = prepared_to_logical_blob_schema(&schema).unwrap();

        assert_eq!(normalized.fields, schema.fields);
        assert_eq!(normalized.metadata, schema.metadata);
    }

    #[test]
    fn test_prepared_to_logical_blob_schema_preserves_non_blob_fields() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        let prepared_field = prepared_blob_field_with_metadata("blob", true, metadata);
        let dict_field = Field::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        );
        let mut schema =
            LanceSchema::try_from(&ArrowSchema::new(vec![dict_field, prepared_field])).unwrap();
        schema.fields[0].id = 42;
        schema.fields[1].id = 7;

        let dictionary_values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        schema.fields[0].set_dictionary_values(&dictionary_values);

        let normalized = prepared_to_logical_blob_schema(&schema).unwrap();

        assert_eq!(normalized.fields[0].id, 42);
        assert_eq!(
            normalized.fields[0]
                .dictionary
                .as_ref()
                .and_then(|dict| dict.values.as_ref())
                .map(|values| values.len()),
            Some(2)
        );
        assert_eq!(normalized.fields[1].id, 7);
        assert_eq!(normalized.fields[1].children.len(), 2);
        assert_eq!(normalized.fields[1].children[0].name, "data");
        assert_eq!(normalized.fields[1].children[1].name, "uri");
        assert!(normalized.fields[1].children[0].id >= 0);
        assert!(normalized.fields[1].children[1].id >= 0);
    }

    #[test]
    fn blob_runtime_and_descriptor_fields_do_not_enter_logical_field_id_space() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        let prepared_field = prepared_blob_field_with_metadata("blob", true, metadata);
        let prepared = LanceSchema::try_from(&ArrowSchema::new(vec![prepared_field])).unwrap();

        // Prepared-only children such as `blob_id` and `blob_size` are writer
        // representation details. Normalization retains IDs only for the
        // persistent logical identities (`blob`, `data`, and `uri`).
        let logical = prepared_to_logical_blob_schema(&prepared).unwrap();
        assert_eq!(logical.fields[0].id, 0);
        assert_eq!(logical.fields[0].children[0].id, 2);
        assert_eq!(logical.fields[0].children[1].id, 3);
        assert_eq!(logical.max_field_id(), Some(3));

        // Descriptor projection creates a file/read-local representation. Its
        // children deliberately remain synthetic and cannot advance a manifest
        // field-ID high-water mark.
        let mut descriptor = logical;
        descriptor.fields[0].unloaded_mut();
        assert_eq!(descriptor.fields[0].id, 0);
        assert!(
            descriptor.fields[0]
                .children
                .iter()
                .all(|child| child.id == -1)
        );
        assert_eq!(descriptor.max_field_id(), Some(0));
    }

    #[test]
    fn test_prepared_blob_schema_normalizes_by_semantic_child_name() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        metadata.insert("blob-root-metadata".to_string(), "preserved".to_string());
        let prepared_field = prepared_blob_field_with_metadata("blob", false, metadata);
        let dict_field = Field::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        );
        let mut schema = lance_schema_with_metadata(vec![dict_field, prepared_field]);
        schema.fields[0].id = 42;
        schema.fields[1].id = 7;
        for (child, id) in schema.fields[1].children.iter_mut().zip(70..76) {
            child.id = id;
            child.parent_id = 7;
        }

        let dictionary_values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        schema.fields[0].set_dictionary_values(&dictionary_values);

        let normalized = prepared_to_logical_blob_schema(&schema).unwrap();

        assert_eq!(normalized.metadata, schema.metadata);
        assert_eq!(normalized.fields[0], schema.fields[0]);
        assert_eq!(normalized.fields[1].id, 7);
        assert!(!normalized.fields[1].nullable);
        assert_eq!(normalized.fields[1].metadata, schema.fields[1].metadata);
        assert_eq!(normalized.fields[1].children.len(), 2);
        assert_eq!(normalized.fields[1].children[0].name, "data");
        assert_eq!(normalized.fields[1].children[0].id, 71);
        assert_eq!(normalized.fields[1].children[0].parent_id, 7);
        assert_eq!(normalized.fields[1].children[1].name, "uri");
        assert_eq!(normalized.fields[1].children[1].id, 72);
        assert_eq!(normalized.fields[1].children[1].parent_id, 7);
    }

    #[rstest]
    #[case::descriptor(BLOB_V2_DESC_FIELDS.clone(), "descriptor layout")]
    #[case::malformed(
        vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
            Field::new("size", DataType::UInt64, true),
        ].into(),
        "unrecognized layout"
    )]
    fn test_non_logical_blob_schema_normalization_is_rejected(
        #[case] fields: Fields,
        #[case] actual_layout: &str,
    ) {
        let field = blob_v2_field_with_children("blob", fields, true);
        let schema = lance_schema_with_metadata(vec![field]);

        let error = prepared_to_logical_blob_schema(&schema).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(actual_layout));
        assert!(
            error
                .to_string()
                .contains("expected logical or prepared layout")
        );
    }

    #[tokio::test]
    async fn test_sidecar_writers_return_prepared_values() {
        let temp_dir = TempDir::default();
        let data_dir = Path::from_absolute_path(temp_dir.std_path().join("data")).unwrap();
        let data_file_key = "data-file".to_string();
        let data_file_path = data_dir.clone().join(format!("{data_file_key}.lance"));
        let object_store = ObjectStore::local();

        let packed_id = 7;
        let packed_path = blob_path(&data_dir, &data_file_key, packed_id);
        let mut packed =
            PackedBlobWriter::try_new(object_store.clone(), data_file_path.clone(), packed_id)
                .await
                .unwrap();
        assert_eq!(packed.path(), &packed_path);
        packed.write_blob(b"abc").await.unwrap();
        packed.write_blob(b"de").await.unwrap();
        let packed_values = packed.finish().await.unwrap();
        assert_eq!(
            packed_values,
            vec![
                BlobDescriptor::Packed {
                    blob_id: packed_id,
                    offset: 0,
                    size: 3,
                },
                BlobDescriptor::Packed {
                    blob_id: packed_id,
                    offset: 3,
                    size: 2,
                },
            ]
        );

        let dedicated_id = 8;
        let dedicated_path = blob_path(&data_dir, &data_file_key, dedicated_id);
        let mut dedicated =
            DedicatedBlobWriter::try_new(object_store.clone(), data_file_path, dedicated_id)
                .await
                .unwrap();
        assert_eq!(dedicated.path(), &dedicated_path);
        dedicated.write(b"abc").await.unwrap();
        dedicated.write(b"def").await.unwrap();
        assert_eq!(
            dedicated.finish().await.unwrap(),
            BlobDescriptor::Dedicated {
                blob_id: dedicated_id,
                size: 6,
            }
        );

        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .extend_packed(
                42,
                vec![
                    BlobRange { offset: 1, size: 2 },
                    BlobRange { offset: 4, size: 1 },
                ],
            )
            .unwrap();
        builder.push_dedicated(43, 3).unwrap();
        let column = builder.finish().unwrap();
        assert_eq!(column.array().len(), 3);
    }

    #[tokio::test]
    async fn test_packed_blob_writer_bulk_bytes() {
        let temp_dir = TempDir::default();
        let data_dir = Path::from_absolute_path(temp_dir.std_path().join("data")).unwrap();
        let data_file_path = data_dir.join("data-file.lance");
        let mut writer = PackedBlobWriter::try_new(ObjectStore::local(), data_file_path, 7)
            .await
            .unwrap();

        writer
            .write_packed_blobs([
                Some(b"a".as_slice()),
                Some(b"".as_slice()),
                None,
                Some(b"bc".as_slice()),
            ])
            .await
            .unwrap();

        assert_eq!(
            writer.finish().await.unwrap(),
            vec![
                BlobDescriptor::Packed {
                    blob_id: 7,
                    offset: 0,
                    size: 1,
                },
                BlobDescriptor::Packed {
                    blob_id: 7,
                    offset: 1,
                    size: 0,
                },
                BlobDescriptor::Null,
                BlobDescriptor::Packed {
                    blob_id: 7,
                    offset: 1,
                    size: 2,
                },
            ]
        );
    }

    #[tokio::test]
    async fn test_packed_blob_writer_bulk_drops_after_partial_write_error() {
        let (partial_writer, bytes_written, dropped) = partial_writer(WriteTerminal::Error);
        let previous_range = BlobRange { offset: 0, size: 3 };
        let mut writer = PackedBlobWriter {
            upload: BlobObjectWriter {
                path: Path::from("packed.blob"),
                writer: Some(partial_writer),
                offset: 3,
            },
            blob_id: 7,
            ranges: vec![Some(previous_range)],
        };

        let error = writer
            .write_packed_blobs([Some(b"abcdef".as_slice())])
            .await
            .unwrap_err();

        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected write failure"));
        assert_eq!(bytes_written.load(Ordering::SeqCst), 2);
        assert!(dropped.load(Ordering::SeqCst));
        assert!(writer.upload.writer.is_none());
        assert_eq!(writer.upload.offset, 3);
        assert_eq!(writer.ranges, vec![Some(previous_range)]);
        let retry_error = writer.write_blob(b"retry").await.unwrap_err();
        assert!(matches!(retry_error, Error::IO { .. }));
        assert!(retry_error.to_string().contains("no active upload"));
    }

    #[test]
    fn test_packed_blob_writer_bulk_drops_if_cancelled() {
        let (partial_writer, bytes_written, dropped) = partial_writer(WriteTerminal::Pending);
        let previous_range = BlobRange { offset: 0, size: 3 };
        let mut writer = PackedBlobWriter {
            upload: BlobObjectWriter {
                path: Path::from("packed.blob"),
                writer: Some(partial_writer),
                offset: 3,
            },
            blob_id: 7,
            ranges: vec![Some(previous_range)],
        };

        cancel_pending(writer.write_packed_blobs([Some(b"abcdef".as_slice())]));

        assert_eq!(bytes_written.load(Ordering::SeqCst), 2);
        assert!(dropped.load(Ordering::SeqCst));
        assert!(writer.upload.writer.is_none());
        assert_eq!(writer.upload.offset, 3);
        assert_eq!(writer.ranges, vec![Some(previous_range)]);
    }

    #[rstest]
    #[case::short(2)]
    #[case::exact(3)]
    #[case::long(4)]
    #[tokio::test]
    async fn test_blob_object_writer_validates_completion_size(#[case] completed_size: usize) {
        let (writer, bytes_written, dropped) = partial_writer(WriteTerminal::Error);
        bytes_written.store(completed_size, Ordering::SeqCst);
        let upload = BlobObjectWriter {
            path: Path::from("completion.blob"),
            writer: Some(writer),
            offset: 3,
        };

        let result = upload.finish().await;
        assert!(dropped.load(Ordering::SeqCst));
        if completed_size == 3 {
            assert_eq!(result.unwrap(), 3);
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains(&format!(
                "Blob object 'completion.blob' has size {completed_size}, expected 3"
            )));
        }
    }

    #[rstest]
    #[case::error(WriteTerminal::Error)]
    #[case::cancelled(WriteTerminal::Pending)]
    #[tokio::test]
    async fn test_dedicated_blob_writer_drops_partial_upload(#[case] terminal: WriteTerminal) {
        let (partial_writer, bytes_written, dropped) = partial_writer(terminal);
        let mut writer = DedicatedBlobWriter {
            upload: BlobObjectWriter {
                path: Path::from("dedicated.blob"),
                writer: Some(partial_writer),
                offset: 0,
            },
            blob_id: 7,
        };

        match terminal {
            WriteTerminal::Error => {
                let error = writer.write(b"abcdef").await.unwrap_err();
                assert!(matches!(error, Error::IO { .. }));
                assert!(error.to_string().contains("injected write failure"));
            }
            WriteTerminal::Pending => cancel_pending(writer.write(b"abcdef")),
        }

        assert_eq!(bytes_written.load(Ordering::SeqCst), 2);
        assert!(dropped.load(Ordering::SeqCst));
        let error = writer.write(b"retry").await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("no active upload"));
        let error = writer.finish().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("no active upload"));
    }
}
