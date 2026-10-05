// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Concatenation of complete encoded Lance files.
//!
//! This module owns compatibility checks and metadata relocation for copying
//! already-encoded pages into a new ordinary Lance file. Callers retain
//! responsibility for dataset-level grouping, transactions, and fallbacks.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    future::Future,
    ops::Range,
    sync::Arc,
};

use arrow_array::{Array, ArrayRef, cast::AsArray, types::UInt8Type};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use futures::TryStreamExt;
use lance_arrow::FieldExt;
use lance_core::{
    Error, Result,
    cache::LanceCache,
    datatypes::{BLOB_V2_DESC_LANCE_FIELD, BlobHandling, BlobKind, Field, Schema},
};
use lance_encoding::decoder::{ColumnInfo, DecoderPlugins, FilterExpression, PageInfo};
use lance_io::{ReadBatchParams, scheduler::FileScheduler, traits::Writer as ObjectWriter};
use prost::Message;
use prost_types::Any;

use crate::{
    reader::{CachedFileMetadata, FileReader, RawFileMetadataOpen},
    version::ConcreteFileVersion,
    versions,
    writer::{FileWriteSummary, FileWriterOptions},
};

/// Caller-defined runtime identity of the final target for Blob-bearing parts.
///
/// Lance only compares this value when assembling data-file parts. It does not
/// interpret it as a dataset, base, or object-store identity, persist it, or
/// define a recovery protocol for it. The caller must provide the same identity
/// for every part and target that belong to one assembly operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobTargetId(Arc<str>);

impl BlobTargetId {
    /// Create an opaque Blob target identity.
    pub fn new(identity: impl Into<Arc<str>>) -> Self {
        Self(identity.into())
    }

    /// Return the caller-defined target identity.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

/// One complete immutable Lance file supplied to [`concat_files`].
#[derive(Clone)]
pub struct EncodedFileInput {
    scheduler: FileScheduler,
    expected_num_rows: Option<u64>,
}

impl EncodedFileInput {
    /// Create an input from an already-open file scheduler.
    pub fn new(scheduler: FileScheduler) -> Self {
        Self {
            scheduler,
            expected_num_rows: None,
        }
    }

    /// Require the file metadata to report this physical row count.
    ///
    /// A mismatch is an input error, not a compatibility result.
    pub fn with_expected_num_rows(mut self, expected_num_rows: u64) -> Self {
        self.expected_num_rows = Some(expected_num_rows);
        self
    }

    /// The path used to read this input.
    pub fn path(&self) -> &object_store::path::Path {
        self.scheduler.reader().path()
    }

    fn scheduler(&self) -> FileScheduler {
        self.scheduler.clone()
    }
}

/// A complete ordinary Lance file validated for data-file concatenation.
///
/// A part is independently readable and is not an incomplete file-format
/// fragment or a persisted Manifest entity. Opening one reads its real footer,
/// verifies that its physical columns form a complete rectangular file, and
/// checks Blob v2 descriptors against the caller-provided ID lease. Parts with
/// Blob v2 columns are also bound to the caller-provided [`BlobTargetId`].
///
/// This runtime value has no fragment, source-row, range, dataset, or storage
/// identity. Lance defines no serialization or recovery contract for it. The
/// caller owns the input storage and must keep every Blob-bearing part associated
/// with the dataset and base where its managed payloads were written. The order
/// passed to [`concat_data_file_parts`] determines final physical row order.
///
/// # Example
///
/// ```
/// use lance_file::concat::{DataFilePart, EncodedFileInput};
/// use lance_io::scheduler::FileScheduler;
///
/// # async fn open_part(file: FileScheduler) -> lance_core::Result<()> {
/// let part = DataFilePart::open(EncodedFileInput::new(file), None, None).await?;
/// println!("part rows: {}", part.num_rows());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct DataFilePart {
    input: EncodedFileInput,
    metadata: Arc<CachedFileMetadata>,
    schema: Arc<Schema>,
    blob_ids: Option<Range<u32>>,
    blob_target_id: Option<BlobTargetId>,
}

impl fmt::Debug for DataFilePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataFilePart")
            .field("path", &self.input.path())
            .field("version", &self.metadata.version)
            .field("num_rows", &self.metadata.num_rows)
            .field("blob_ids", &self.blob_ids)
            .field("blob_target_id", &self.blob_target_id)
            .finish()
    }
}

impl DataFilePart {
    /// Open and validate one complete encoded file.
    ///
    /// `blob_ids` is the half-open ID range leased to this part. Managed
    /// Packed and Dedicated descriptors must fall inside it. `blob_target_id`
    /// associates the part with one final target for runtime equality checks and
    /// is required whenever the part contains Blob v2 columns. It does not prove
    /// that the target belongs to a particular dataset or storage namespace.
    /// Non-empty Inline Blob v2 descriptors and legacy Blob v1 columns are
    /// rejected because their payload locations cannot be reused in a different
    /// data file.
    pub async fn open(
        input: EncodedFileInput,
        blob_ids: Option<Range<u32>>,
        blob_target_id: Option<BlobTargetId>,
    ) -> Result<Self> {
        validate_blob_id_range(blob_ids.as_ref())?;
        let metadata = Arc::new(FileReader::read_all_metadata(&input.scheduler()).await?);
        let schema = Arc::new(normalize_blob_footer_schema(metadata.file_schema.as_ref()));
        if let Some(expected_num_rows) = input.expected_num_rows
            && metadata.num_rows != expected_num_rows
        {
            return Err(Error::invalid_input(format!(
                "part at '{}' has {} physical rows but {} were expected",
                input.path(),
                metadata.num_rows,
                expected_num_rows
            )));
        }
        let has_blob_v1 = schema
            .fields_pre_order()
            .any(|field| field.is_blob() && !field.is_blob_v2());
        if has_blob_v1 {
            return Err(Error::not_supported(format!(
                "part at '{}' contains legacy Blob v1 columns",
                input.path()
            )));
        }
        let validation_schema = descriptor_projection_schema(schema.as_ref());
        let normalized_rows = versions::validate_external_metadata(
            metadata.version,
            &validation_schema,
            metadata.as_ref(),
        )
        .map_err(|error| {
            Error::corrupt_file(
                input.path().clone(),
                format!("part has incomplete file metadata: {error}"),
            )
        })?;
        if normalized_rows != metadata.num_rows {
            return Err(Error::corrupt_file(
                input.path().clone(),
                format!(
                    "part descriptor reports {} physical rows but its columns normalize to {normalized_rows}",
                    metadata.num_rows
                ),
            ));
        }

        let has_blob_v2 = schema.fields_pre_order().any(|field| field.is_blob_v2());
        if has_blob_v2 {
            validate_blob_descriptors(
                &input,
                metadata.as_ref(),
                schema.as_ref(),
                blob_ids.as_ref(),
            )
            .await?;
            if blob_target_id.is_none() {
                return Err(Error::invalid_input(format!(
                    "part at '{}' contains Blob v2 columns but no Blob target ID was provided",
                    input.path()
                )));
            }
        }

        Ok(Self {
            input,
            metadata,
            schema,
            blob_ids,
            blob_target_id,
        })
    }

    /// Number of physical rows described by the part footer.
    pub fn num_rows(&self) -> u64 {
        self.metadata.num_rows
    }
}

fn descriptor_projection_schema(schema: &Schema) -> Schema {
    let mut projected = schema.clone();
    projected.fields = projected
        .fields
        .into_iter()
        .map(|field| BlobHandling::BlobsDescriptions.unload_if_needed(field))
        .collect();
    projected
}

fn descriptor_child_matches(field: &Field, expected: &Field) -> bool {
    field.id == -1
        && field.parent_id == -1
        && field.name == expected.name
        && field.logical_type == expected.logical_type
        && field.children.is_empty()
}

fn attach_blob_descriptor_children(
    fields: &mut [Field],
    descriptor_children: &mut VecDeque<Vec<Field>>,
) {
    for field in fields {
        if field.is_blob() && field.children.is_empty() {
            if let Some(children) = descriptor_children.pop_front() {
                field.children = children;
            }
        } else {
            attach_blob_descriptor_children(&mut field.children, descriptor_children);
        }
    }
}

/// Blob descriptor children historically use anonymous field IDs in the file
/// descriptor. Reconstruct their tree shape before applying ordinary schema
/// projection rules. This interprets the existing footer representation and
/// does not add persisted metadata or alter the file grammar.
fn normalize_blob_footer_schema(schema: &Schema) -> Schema {
    let expected = &BLOB_V2_DESC_LANCE_FIELD.children;
    let missing_descriptor_count = schema
        .fields_pre_order()
        .filter(|field| field.is_blob() && field.children.is_empty())
        .count();
    if missing_descriptor_count == 0 {
        return schema.clone();
    }
    let mut normalized = schema.clone();
    let mut descriptor_children = VecDeque::new();
    let mut field_index = 0;
    while descriptor_children.len() < missing_descriptor_count
        && field_index + expected.len() <= normalized.fields.len()
    {
        if normalized.fields[field_index..field_index + expected.len()]
            .iter()
            .zip(expected)
            .all(|(field, expected)| descriptor_child_matches(field, expected))
        {
            descriptor_children.push_back(
                normalized
                    .fields
                    .drain(field_index..field_index + expected.len())
                    .collect(),
            );
        } else {
            field_index += 1;
        }
    }
    attach_blob_descriptor_children(&mut normalized.fields, &mut descriptor_children);
    normalized
}

fn validate_blob_id_range(blob_ids: Option<&Range<u32>>) -> Result<()> {
    if let Some(blob_ids) = blob_ids
        && (blob_ids.start == 0 || blob_ids.start >= blob_ids.end)
    {
        return Err(Error::invalid_input(format!(
            "part Blob ID range must be non-empty and start at 1 or greater, got {}..{}",
            blob_ids.start, blob_ids.end
        )));
    }
    Ok(())
}

async fn validate_blob_descriptors(
    input: &EncodedFileInput,
    metadata: &CachedFileMetadata,
    schema: &Schema,
    blob_ids: Option<&Range<u32>>,
) -> Result<()> {
    let projected_schema = descriptor_projection_schema(schema);
    let blob_field_ids = projected_schema
        .fields_pre_order()
        .filter(|field| field.is_blob_v2())
        .map(|field| field.id)
        .collect::<Vec<_>>();
    let unique_blob_field_ids = blob_field_ids.iter().copied().collect::<BTreeSet<_>>();
    if unique_blob_field_ids.len() != blob_field_ids.len()
        || unique_blob_field_ids
            .first()
            .is_some_and(|field_id| *field_id < 0)
    {
        return Err(Error::corrupt_file(
            input.path().clone(),
            "Blob v2 fields in a data-file part must have unique non-negative field IDs",
        ));
    }
    let blob_schema = projected_schema.project_by_ids(&blob_field_ids, true);
    let (field_ids, column_indices) =
        versions::data_file_columns(metadata.version, &projected_schema);
    let field_id_to_column_index = field_ids
        .into_iter()
        .zip(column_indices)
        .filter_map(|(field_id, column_index)| {
            (field_id >= 0 && column_index >= 0).then_some((field_id as u32, column_index as u32))
        })
        .collect::<BTreeMap<_, _>>();
    let projection = versions::reader_projection_from_field_ids(
        metadata.version,
        &blob_schema,
        &field_id_to_column_index,
    )?;
    let reader = FileReader::try_open(
        input.scheduler(),
        Some(projection),
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        Default::default(),
    )
    .await?;
    let mut batches = reader
        .read_stream(
            ReadBatchParams::RangeFull,
            8192,
            4,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = batches.try_next().await? {
        let selected = vec![true; batch.num_rows()];
        for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
            validate_blob_field(field.as_ref(), array, &selected, blob_ids, input.path())?;
        }
    }
    Ok(())
}

fn validate_blob_field(
    field: &ArrowField,
    array: &ArrayRef,
    selected: &[bool],
    blob_ids: Option<&Range<u32>>,
    path: &object_store::path::Path,
) -> Result<()> {
    if field.is_blob() {
        let descriptors = array.as_struct();
        let kinds = descriptors
            .column_by_name("kind")
            .ok_or_else(|| Error::corrupt_file(path.clone(), "Blob v2 descriptor has no kind"))?
            .as_primitive::<UInt8Type>();
        let positions = descriptors
            .column_by_name("position")
            .ok_or_else(|| Error::corrupt_file(path.clone(), "Blob v2 descriptor has no position"))?
            .as_primitive::<arrow_array::types::UInt64Type>();
        let sizes = descriptors
            .column_by_name("size")
            .ok_or_else(|| Error::corrupt_file(path.clone(), "Blob v2 descriptor has no size"))?
            .as_primitive::<arrow_array::types::UInt64Type>();
        let ids = descriptors
            .column_by_name("blob_id")
            .ok_or_else(|| Error::corrupt_file(path.clone(), "Blob v2 descriptor has no blob_id"))?
            .as_primitive::<arrow_array::types::UInt32Type>();
        for (row, is_selected) in selected.iter().copied().enumerate() {
            if !is_selected || descriptors.is_null(row) {
                continue;
            }
            let kind = BlobKind::try_from(kinds.value(row))?;
            match kind {
                BlobKind::Inline if sizes.value(row) > 0 => {
                    return Err(Error::invalid_input(format!(
                        "part at '{}' contains a non-empty Inline Blob v2 descriptor at row {row}; data-file part concatenation requires Packed or Dedicated storage",
                        path
                    )));
                }
                BlobKind::Packed | BlobKind::Dedicated => {
                    let blob_id = ids.value(row);
                    let Some(blob_ids) = blob_ids else {
                        return Err(Error::invalid_input(format!(
                            "part at '{}' contains managed Blob ID {blob_id} at row {row} but no Blob ID range was provided",
                            path
                        )));
                    };
                    if !blob_ids.contains(&blob_id) {
                        return Err(Error::invalid_input(format!(
                            "part at '{}' contains managed Blob ID {blob_id} at row {row}, outside declared range {}..{}",
                            path, blob_ids.start, blob_ids.end
                        )));
                    }
                    if kind == BlobKind::Dedicated && positions.value(row) != 0 {
                        return Err(Error::corrupt_file(
                            path.clone(),
                            format!(
                                "Dedicated Blob descriptor at row {row} has non-zero position {}",
                                positions.value(row)
                            ),
                        ));
                    }
                }
                BlobKind::Inline | BlobKind::External => {}
            }
        }
        return Ok(());
    }

    match field.data_type() {
        ArrowDataType::Struct(children) => {
            let struct_array = array.as_struct();
            let child_selected = selected
                .iter()
                .copied()
                .enumerate()
                .map(|(row, is_selected)| is_selected && struct_array.is_valid(row))
                .collect::<Vec<_>>();
            for (child, child_array) in children.iter().zip(struct_array.columns()) {
                validate_blob_field(child.as_ref(), child_array, &child_selected, blob_ids, path)?;
            }
        }
        ArrowDataType::List(child) => {
            let list = array.as_list::<i32>();
            let mut child_selected = vec![false; list.values().len()];
            for (row, is_selected) in selected.iter().copied().enumerate() {
                if is_selected && list.is_valid(row) {
                    let start = list.value_offsets()[row] as usize;
                    let end = list.value_offsets()[row + 1] as usize;
                    child_selected[start..end].fill(true);
                }
            }
            validate_blob_field(
                child.as_ref(),
                list.values(),
                &child_selected,
                blob_ids,
                path,
            )?;
        }
        ArrowDataType::LargeList(child) => {
            let list = array.as_list::<i64>();
            let mut child_selected = vec![false; list.values().len()];
            for (row, is_selected) in selected.iter().copied().enumerate() {
                if is_selected && list.is_valid(row) {
                    let start = list.value_offsets()[row] as usize;
                    let end = list.value_offsets()[row + 1] as usize;
                    child_selected[start..end].fill(true);
                }
            }
            validate_blob_field(
                child.as_ref(),
                list.values(),
                &child_selected,
                blob_ids,
                path,
            )?;
        }
        _ => {}
    }
    Ok(())
}

/// The exact file grammar and schema required for concatenated output.
#[derive(Debug, Clone)]
pub struct FileConcatTarget {
    /// Exact output grammar. Release aliases are resolved before this boundary.
    pub version: ConcreteFileVersion,
    /// Complete schema stored in every input and regenerated in the output.
    pub schema: Arc<Schema>,
    blob_target_id: Option<BlobTargetId>,
}

impl FileConcatTarget {
    /// Create a concatenation target.
    pub fn new(version: ConcreteFileVersion, schema: Arc<Schema>) -> Self {
        Self {
            version,
            schema,
            blob_target_id: None,
        }
    }

    /// Bind Blob-bearing parts to one caller-defined final target.
    pub fn with_blob_target_id(mut self, blob_target_id: BlobTargetId) -> Self {
        self.blob_target_id = Some(blob_target_id);
        self
    }
}

/// Runtime controls for encoded-file concatenation.
#[derive(Debug, Clone)]
pub struct FileConcatOptions {
    /// Maximum page-buffer bytes requested in one read batch.
    pub read_batch_bytes: usize,
    /// Options passed to the exact-version footer writer.
    pub writer_options: FileWriterOptions,
}

impl Default for FileConcatOptions {
    fn default() -> Self {
        Self {
            read_batch_bytes: 16 * 1024 * 1024,
            writer_options: FileWriterOptions::default(),
        }
    }
}

/// Metadata describing the complete file represented by a concat result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileConcatOutput {
    /// Exact grammar of the completed or reused file.
    pub version: ConcreteFileVersion,
    /// Total physical rows in input order.
    pub num_rows: u64,
    /// Size of the completed or reused object.
    pub size_bytes: u64,
}

/// A compatibility reason that requires a caller-controlled decode/re-encode fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileConcatReason {
    /// Lance v1 does not support encoded-file concatenation.
    LegacyVersion,
    /// An input uses a different exact grammar than the target.
    VersionMismatch {
        /// Zero-based input position.
        input_index: usize,
        /// Version found in the file footer.
        actual: ConcreteFileVersion,
        /// Version requested by the target.
        expected: ConcreteFileVersion,
    },
    /// An input's persisted schema differs from the target schema.
    SchemaMismatch {
        /// Zero-based input position.
        input_index: usize,
    },
    /// Inputs do not describe the same physical columns.
    ColumnLayoutMismatch {
        /// Zero-based input position.
        input_index: usize,
        /// Zero-based physical column when one could be identified.
        column_index: Option<usize>,
    },
    /// A column-level encoding cannot safely combine its buffers.
    ColumnEncodingMismatch {
        /// Zero-based input position.
        input_index: usize,
        /// Zero-based physical column.
        column_index: usize,
    },
    /// A column uses file-level buffers whose page references cannot be relocated.
    ColumnBuffers {
        /// Zero-based input position.
        input_index: usize,
        /// Zero-based physical column.
        column_index: usize,
        /// Number of column buffers referenced by the column metadata.
        count: usize,
    },
    /// A file contains global buffers whose relocation semantics are not defined.
    ExtraGlobalBuffers {
        /// Zero-based input position.
        input_index: usize,
        /// Number of global buffers, including the schema descriptor.
        count: usize,
    },
    /// The schema contains offsets into external blob storage.
    BlobColumns,
}

impl fmt::Display for FileConcatReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyVersion => f.write_str("Lance v1 files cannot be concatenated"),
            Self::VersionMismatch {
                input_index,
                actual,
                expected,
            } => write!(
                f,
                "input {input_index} has file version {actual}, expected {expected}"
            ),
            Self::SchemaMismatch { input_index } => {
                write!(f, "input {input_index} has a different file schema")
            }
            Self::ColumnLayoutMismatch {
                input_index,
                column_index,
            } => match column_index {
                Some(column_index) => write!(
                    f,
                    "input {input_index} has a different layout for physical column {column_index}"
                ),
                None => write!(
                    f,
                    "input {input_index} has a different physical column count"
                ),
            },
            Self::ColumnEncodingMismatch {
                input_index,
                column_index,
            } => write!(
                f,
                "input {input_index} has an incompatible encoding for physical column {column_index}"
            ),
            Self::ColumnBuffers {
                input_index,
                column_index,
                count,
            } => write!(
                f,
                "input {input_index} physical column {column_index} has {count} column buffers whose references cannot be relocated"
            ),
            Self::ExtraGlobalBuffers { input_index, count } => write!(
                f,
                "input {input_index} has {count} global buffers; only the schema descriptor is supported"
            ),
            Self::BlobColumns => {
                f.write_str("schemas containing blob columns cannot be concatenated")
            }
        }
    }
}

/// Result of one encoded-file concatenation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileConcatResult {
    /// A new ordinary Lance file was written.
    Written(FileConcatOutput),
    /// One compatible complete input already is the requested output.
    Reused(usize, FileConcatOutput),
    /// Compatibility was rejected before the output factory was called.
    Unsupported(FileConcatReason),
}

struct PreparedInput<'a> {
    input: &'a EncodedFileInput,
    metadata: &'a CachedFileMetadata,
    schema: &'a Schema,
}

fn encoded_column_encoding(column: &ColumnInfo) -> Result<Vec<u8>> {
    Ok(Any::from_msg(&column.encoding)?.encode_to_vec())
}

fn check_compatibility(
    target: &FileConcatTarget,
    inputs: &[PreparedInput<'_>],
    allow_blob_columns: bool,
) -> Result<Option<FileConcatReason>> {
    if !allow_blob_columns
        && target
            .schema
            .fields_pre_order()
            .any(|field| field.is_blob())
    {
        return Ok(Some(FileConcatReason::BlobColumns));
    }

    let Some(first) = inputs.first() else {
        return Err(Error::invalid_input(
            "concat_files requires at least one complete input file",
        ));
    };
    let baseline_columns = &first.metadata.column_infos;
    let expected_schema = if allow_blob_columns {
        descriptor_projection_schema(target.schema.as_ref())
    } else {
        target.schema.as_ref().clone()
    };
    let baseline_encodings = baseline_columns
        .iter()
        .map(|column| encoded_column_encoding(column))
        .collect::<Result<Vec<_>>>()?;

    for (input_index, prepared) in inputs.iter().enumerate() {
        let metadata = &prepared.metadata;
        if let Some(expected_num_rows) = prepared.input.expected_num_rows
            && metadata.num_rows != expected_num_rows
        {
            return Err(Error::invalid_input(format!(
                "input {input_index} at '{}' has {} physical rows but {} were expected",
                prepared.input.path(),
                metadata.num_rows,
                expected_num_rows
            )));
        }
        if metadata.version != target.version {
            return Ok(Some(FileConcatReason::VersionMismatch {
                input_index,
                actual: metadata.version,
                expected: target.version,
            }));
        }
        if prepared.schema != &expected_schema {
            return Ok(Some(FileConcatReason::SchemaMismatch { input_index }));
        }
        let normalized_rows =
            versions::validate_external_metadata(metadata.version, prepared.schema, metadata)
                .map_err(|error| {
                    Error::corrupt_file(
                        prepared.input.path().clone(),
                        format!("input {input_index} has incomplete file metadata: {error}"),
                    )
                })?;
        if normalized_rows != metadata.num_rows {
            return Err(Error::corrupt_file(
                prepared.input.path().clone(),
                format!(
                    "input {input_index} descriptor reports {} physical rows but its columns normalize to {normalized_rows}",
                    metadata.num_rows
                ),
            ));
        }
        if metadata.file_buffers.len() > 1 {
            return Ok(Some(FileConcatReason::ExtraGlobalBuffers {
                input_index,
                count: metadata.file_buffers.len(),
            }));
        }
        if metadata.column_infos.len() != baseline_columns.len() {
            return Ok(Some(FileConcatReason::ColumnLayoutMismatch {
                input_index,
                column_index: None,
            }));
        }
        for (column_index, (column, baseline)) in metadata
            .column_infos
            .iter()
            .zip(baseline_columns)
            .enumerate()
        {
            if !column.buffer_offsets_and_sizes.is_empty() {
                return Ok(Some(FileConcatReason::ColumnBuffers {
                    input_index,
                    column_index,
                    count: column.buffer_offsets_and_sizes.len(),
                }));
            }
            if column.index != baseline.index {
                return Ok(Some(FileConcatReason::ColumnLayoutMismatch {
                    input_index,
                    column_index: Some(column_index),
                }));
            }
            if encoded_column_encoding(column)? != baseline_encodings[column_index] {
                return Ok(Some(FileConcatReason::ColumnEncodingMismatch {
                    input_index,
                    column_index,
                }));
            }
        }
    }
    Ok(None)
}

async fn copy_page_buffers(
    writer: &mut crate::writer::FileWriter,
    scheduler: &FileScheduler,
    pages: &[PageInfo],
    read_batch_bytes: u64,
    input_index: usize,
    column_index: usize,
    row_offset: u64,
) -> Result<Vec<PageInfo>> {
    let mut copied = Vec::with_capacity(pages.len());
    let mut page_index = 0;
    while page_index < pages.len() {
        let batch_start = page_index;
        let mut batch_bytes = 0u64;
        let mut batch_ranges = Vec::new();
        let mut batch_buffer_counts = Vec::new();
        while page_index < pages.len() {
            let page = &pages[page_index];
            let page_bytes = page.buffer_offsets_and_sizes.iter().try_fold(
                0u64,
                |total, (offset, size)| {
                    offset.checked_add(*size).ok_or_else(|| {
                        Error::corrupt_file(
                            scheduler.reader().path().clone(),
                            format!(
                                "input {input_index} column {column_index} page {page_index} buffer range overflows"
                            ),
                        )
                    })?;
                    total.checked_add(*size).ok_or_else(|| {
                        Error::corrupt_file(
                            scheduler.reader().path().clone(),
                            format!(
                                "input {input_index} column {column_index} page {page_index} buffer sizes overflow"
                            ),
                        )
                    })
                },
            )?;
            if page_index > batch_start
                && batch_bytes
                    .checked_add(page_bytes)
                    .is_none_or(|total| total > read_batch_bytes)
            {
                break;
            }
            batch_bytes = batch_bytes.checked_add(page_bytes).ok_or_else(|| {
                Error::corrupt_file(
                    scheduler.reader().path().clone(),
                    format!("input {input_index} column {column_index} read batch size overflows"),
                )
            })?;
            batch_buffer_counts.push(page.buffer_offsets_and_sizes.len());
            batch_ranges.extend(
                page.buffer_offsets_and_sizes
                    .iter()
                    .filter(|(_, size)| *size > 0)
                    .map(|(offset, size)| *offset..(*offset + *size)),
            );
            page_index += 1;
        }

        let batch_data = if batch_ranges.is_empty() {
            Vec::new()
        } else {
            scheduler.submit_request(batch_ranges, 0).await?
        };
        let mut batch_data = batch_data.into_iter();
        for (relative_page_index, (page, buffer_count)) in pages[batch_start..page_index]
            .iter()
            .zip(batch_buffer_counts)
            .enumerate()
        {
            let source_page_index = batch_start + relative_page_index;
            let mut relocated_buffers = Vec::with_capacity(buffer_count);
            for (buffer_index, (_, size)) in page.buffer_offsets_and_sizes.iter().enumerate() {
                let data = if *size == 0 {
                    None
                } else {
                    let data = batch_data.next().ok_or_else(|| {
                        Error::io(format!(
                            "short read for input {input_index} column {column_index} page {source_page_index} buffer {buffer_index}: expected {size} bytes"
                        ))
                    })?;
                    if data.len() as u64 != *size {
                        return Err(Error::io(format!(
                            "short read for input {input_index} column {column_index} page {source_page_index} buffer {buffer_index}: expected {size} bytes, got {}",
                            data.len()
                        )));
                    }
                    Some(data)
                };
                relocated_buffers.push(
                    writer
                        .write_external_buffer(data.as_deref().unwrap_or_default())
                        .await?,
                );
            }
            copied.push(PageInfo {
                num_rows: page.num_rows,
                priority: page.priority.checked_add(row_offset).ok_or_else(|| {
                    Error::invalid_input_source(
                        format!(
                            "input {input_index} column {column_index} page {source_page_index} priority overflows after row relocation"
                        )
                        .into(),
                    )
                })?,
                encoding: page.encoding.clone(),
                buffer_offsets_and_sizes: Arc::from(relocated_buffers),
            });
        }
        if batch_data.next().is_some() {
            return Err(Error::io(format!(
                "read for input {input_index} column {column_index} returned more buffers than requested"
            )));
        }
    }
    Ok(copied)
}

async fn concat_prepared<Factory, FactoryFuture>(
    target: &FileConcatTarget,
    prepared: &[PreparedInput<'_>],
    allow_blob_columns: bool,
    reuse_single_input: bool,
    output_factory: Factory,
    options: FileConcatOptions,
) -> Result<FileConcatResult>
where
    Factory: FnOnce() -> FactoryFuture,
    FactoryFuture: Future<Output = Result<Box<dyn ObjectWriter>>>,
{
    if options.read_batch_bytes == 0 {
        return Err(Error::invalid_input(
            "FileConcatOptions.read_batch_bytes must be greater than zero",
        ));
    }
    if let Some(reason) = check_compatibility(target, prepared, allow_blob_columns)? {
        return Ok(FileConcatResult::Unsupported(reason));
    }

    let total_rows = prepared.iter().try_fold(0u64, |total, input| {
        total.checked_add(input.metadata.num_rows).ok_or_else(|| {
            Error::invalid_input_source("concat_files total physical row count overflows".into())
        })
    })?;
    if prepared.len() == 1 && reuse_single_input {
        return Ok(FileConcatResult::Reused(
            0,
            FileConcatOutput {
                version: target.version,
                num_rows: total_rows,
                size_bytes: prepared[0].metadata.file_size_bytes,
            },
        ));
    }

    let object_writer = output_factory().await?;
    let mut writer =
        versions::create_lazy_writer(target.version, object_writer, options.writer_options)?;
    let write_result: Result<FileWriteSummary> = async {
        let column_count = prepared[0].metadata.column_infos.len();
        let mut output_pages = std::iter::repeat_with(Vec::new)
            .take(column_count)
            .collect::<Vec<Vec<PageInfo>>>();
        let mut row_offset = 0u64;

        for (input_index, prepared_input) in prepared.iter().enumerate() {
            for (column_index, column) in prepared_input.metadata.column_infos.iter().enumerate() {
                let has_existing_pages = !output_pages[column_index].is_empty();
                versions::copy_external_metadata_column(
                    target.version,
                    target.schema.as_ref(),
                    column_index,
                    has_existing_pages,
                    || async {
                        let pages = copy_page_buffers(
                            &mut writer,
                            &prepared_input.input.scheduler,
                            &column.page_infos,
                            options.read_batch_bytes as u64,
                            input_index,
                            column_index,
                            row_offset,
                        )
                        .await?;
                        output_pages[column_index].extend(pages);

                        Ok(())
                    },
                )
                .await?;
            }
            row_offset = row_offset
                .checked_add(prepared_input.metadata.num_rows)
                .ok_or_else(|| {
                    Error::invalid_input_source("concat_files physical row offset overflows".into())
                })?;
        }

        let mut columns = Vec::with_capacity(column_count);
        for (column_index, pages) in output_pages.iter_mut().enumerate() {
            versions::finalize_external_metadata_column(
                target.version,
                target.schema.as_ref(),
                column_index,
                pages,
                total_rows,
            )?;
            let baseline = &prepared[0].metadata.column_infos[column_index];
            columns.push(Arc::new(ColumnInfo::new(
                baseline.index,
                Arc::from(std::mem::take(pages)),
                Vec::new(),
                baseline.encoding.clone(),
            )));
        }
        // The schema descriptor is the first global buffer and must start at
        // the page-buffer alignment required by the reader.
        writer.write_external_buffer(&[]).await?;
        writer.initialize_with_external_columns(
            target.schema.as_ref().clone(),
            &columns,
            total_rows,
        )?;
        writer.finish().await
    }
    .await;

    match write_result {
        Ok(summary) => Ok(FileConcatResult::Written(FileConcatOutput {
            version: target.version,
            num_rows: summary.num_rows,
            size_bytes: summary.size_bytes,
        })),
        Err(error) => {
            writer.abort().await;
            Err(error)
        }
    }
}

/// Concatenate complete compatible encoded files in the supplied order.
///
/// Metadata is read exactly once per input. The factory is invoked only after
/// all compatibility checks succeed and is never invoked for [`FileConcatResult::Reused`]
/// or [`FileConcatResult::Unsupported`]. Page payloads are copied without Arrow
/// decoding; offsets, priorities, exact-version structural metadata, and the
/// footer are regenerated.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::Result;
/// # use lance_file::concat::{concat_files, EncodedFileInput, FileConcatOptions, FileConcatResult, FileConcatTarget};
/// # use lance_io::object_store::ObjectStore;
/// # use object_store::path::Path;
/// # async fn stitch(
/// #     target: &FileConcatTarget,
/// #     inputs: &[EncodedFileInput],
/// #     output_store: Arc<ObjectStore>,
/// #     output_path: Path,
/// # ) -> Result<FileConcatResult> {
/// let store = output_store.clone();
/// concat_files(
///     target,
///     inputs,
///     move || async move { store.create(&output_path).await },
///     FileConcatOptions::default(),
/// )
/// .await
/// # }
/// ```
pub async fn concat_files<Factory, FactoryFuture>(
    target: &FileConcatTarget,
    ordered_inputs: &[EncodedFileInput],
    output_factory: Factory,
    options: FileConcatOptions,
) -> Result<FileConcatResult>
where
    Factory: FnOnce() -> FactoryFuture,
    FactoryFuture: Future<Output = Result<Box<dyn ObjectWriter>>>,
{
    if ordered_inputs.is_empty() {
        return Err(Error::invalid_input(
            "concat_files requires at least one complete input file",
        ));
    }
    let raw_metadata = futures::future::try_join_all(
        ordered_inputs
            .iter()
            .map(|input| FileReader::read_raw_metadata_for_dispatch(&input.scheduler)),
    )
    .await?;
    if target.version == ConcreteFileVersion::V1
        || raw_metadata
            .iter()
            .any(|metadata| matches!(metadata, RawFileMetadataOpen::Legacy { .. }))
    {
        return Ok(FileConcatResult::Unsupported(
            FileConcatReason::LegacyVersion,
        ));
    }
    let metadata = raw_metadata
        .into_iter()
        .map(|metadata| match metadata {
            RawFileMetadataOpen::Current { version, metadata } => {
                versions::finish_metadata(version, metadata)
            }
            RawFileMetadataOpen::Legacy { .. } => Err(Error::internal(
                "legacy concat input reached current metadata finalization".to_string(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let prepared = ordered_inputs
        .iter()
        .zip(metadata.iter())
        .map(|(input, metadata)| PreparedInput {
            input,
            metadata,
            schema: metadata.file_schema.as_ref(),
        })
        .collect::<Vec<_>>();
    concat_prepared(target, &prepared, false, true, output_factory, options).await
}

/// Concatenate validated data-file parts in caller-supplied row order.
///
/// Unlike [`concat_files`], this entry point accepts Blob v2 columns because
/// every [`DataFilePart`] has already rejected file-relative Inline payloads
/// and validated managed descriptors against an explicit ID lease. Leases from
/// different parts must not overlap. No logical values or Blob payloads are
/// decoded and re-encoded during concatenation.
pub async fn concat_data_file_parts<Factory, FactoryFuture>(
    target: &FileConcatTarget,
    ordered_parts: &[DataFilePart],
    output_factory: Factory,
    options: FileConcatOptions,
) -> Result<FileConcatResult>
where
    Factory: FnOnce() -> FactoryFuture,
    FactoryFuture: Future<Output = Result<Box<dyn ObjectWriter>>>,
{
    if ordered_parts.is_empty() {
        return Err(Error::invalid_input(
            "concat_data_file_parts requires at least one data-file part",
        ));
    }
    for (part_index, part) in ordered_parts.iter().enumerate() {
        if part.blob_target_id != target.blob_target_id {
            return Err(Error::invalid_input(format!(
                "part {part_index} Blob target ID {:?} does not match target ID {:?}",
                part.blob_target_id.as_ref().map(BlobTargetId::as_str),
                target.blob_target_id.as_ref().map(BlobTargetId::as_str)
            )));
        }
    }
    let mut ranges = ordered_parts
        .iter()
        .enumerate()
        .filter_map(|(part_index, part)| {
            part.blob_ids
                .clone()
                .map(|range| (range.start, range.end, part_index))
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        let (left_start, left_end, left_index) = pair[0];
        let (right_start, right_end, right_index) = pair[1];
        if right_start < left_end {
            return Err(Error::invalid_input(format!(
                "part Blob ID ranges overlap: part {left_index} uses {left_start}..{left_end}, part {right_index} uses {right_start}..{right_end}"
            )));
        }
    }

    let prepared = ordered_parts
        .iter()
        .map(|part| PreparedInput {
            input: &part.input,
            metadata: part.metadata.as_ref(),
            schema: part.schema.as_ref(),
        })
        .collect::<Vec<_>>();
    concat_prepared(target, &prepared, true, false, output_factory, options).await
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lance_core::utils::tempfile::TempObjFile;
    use lance_io::{
        object_store::ObjectStore,
        scheduler::{ScanScheduler, SchedulerConfig},
        traits::Writer,
        utils::CachedFileSize,
    };
    use tokio::io::AsyncWriteExt;

    use super::*;

    async fn write_file(
        store: &Arc<ObjectStore>,
        path: &object_store::path::Path,
        version: ConcreteFileVersion,
        values: &[i32],
    ) -> Arc<Schema> {
        let batch = arrow_array::record_batch!(("value", Int32, values.to_vec())).unwrap();
        let schema = Arc::new(Schema::try_from(batch.schema_ref().as_ref()).unwrap());
        let mut writer = versions::create_writer(
            version,
            store.create(path).await.unwrap(),
            schema.as_ref().clone(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();
        schema
    }

    async fn input(
        store: Arc<ObjectStore>,
        path: &object_store::path::Path,
        expected_num_rows: u64,
    ) -> EncodedFileInput {
        let scheduler = ScanScheduler::new(store, SchedulerConfig::default_for_testing());
        let file = scheduler
            .open_file(path, &CachedFileSize::unknown())
            .await
            .unwrap();
        EncodedFileInput::new(file).with_expected_num_rows(expected_num_rows)
    }

    #[tokio::test]
    async fn concat_writes_relocated_metadata_and_reuses_single_input() {
        let store = Arc::new(ObjectStore::local());
        let first_path = TempObjFile::default();
        let second_path = TempObjFile::default();
        let output_path = TempObjFile::default();
        let schema = write_file(&store, &first_path, ConcreteFileVersion::V2_1, &[1, 2, 3]).await;
        write_file(&store, &second_path, ConcreteFileVersion::V2_1, &[4, 5]).await;
        let inputs = vec![
            input(store.clone(), &first_path, 3).await,
            input(store.clone(), &second_path, 2).await,
        ];
        let target = FileConcatTarget::new(ConcreteFileVersion::V2_1, schema);
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let result = concat_files(
            &target,
            &inputs,
            {
                let store = store.clone();
                let output_path = output_path.clone();
                let factory_calls = factory_calls.clone();
                move || async move {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    store.create(&output_path).await
                }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(
            result,
            FileConcatResult::Written(FileConcatOutput { num_rows: 5, .. })
        ));
        assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
        let output = input(store.clone(), &output_path, 5).await;
        let metadata = FileReader::read_all_metadata(&output.scheduler)
            .await
            .unwrap();
        assert_eq!(metadata.num_rows, 5);
        assert_eq!(metadata.column_infos[0].page_infos.len(), 2);
        assert!(
            metadata.column_infos[0].page_infos[0].priority
                < metadata.column_infos[0].page_infos[1].priority
        );

        let reuse_calls = Arc::new(AtomicUsize::new(0));
        let result = concat_files(
            &target,
            &inputs[..1],
            {
                let reuse_calls = reuse_calls.clone();
                move || async move {
                    reuse_calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::internal("reuse factory must not be called"))
                }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(result, FileConcatResult::Reused(0, _)));
        assert_eq!(reuse_calls.load(Ordering::SeqCst), 0);
    }

    #[rstest::rstest]
    #[case(ConcreteFileVersion::V2_0)]
    #[case(ConcreteFileVersion::V2_1)]
    #[case(ConcreteFileVersion::V2_2)]
    #[case(ConcreteFileVersion::V2_3)]
    #[tokio::test]
    async fn concat_preserves_schema_metadata(#[case] version: ConcreteFileVersion) {
        let store = Arc::new(ObjectStore::local());
        let first_path = TempObjFile::default();
        let second_path = TempObjFile::default();
        let output_path = TempObjFile::default();
        let batch = arrow_array::record_batch!(("value", Int32, [1, 2])).unwrap();
        let mut schema = Schema::try_from(batch.schema_ref().as_ref()).unwrap();
        schema
            .metadata
            .insert("review-key".into(), "review-value".into());
        let schema = Arc::new(schema);

        for path in [&first_path, &second_path] {
            let mut writer = versions::create_writer(
                version,
                store.create(path).await.unwrap(),
                schema.as_ref().clone(),
                FileWriterOptions::default(),
            )
            .unwrap();
            writer.write_batch(&batch).await.unwrap();
            writer.finish().await.unwrap();
        }
        let inputs = vec![
            input(store.clone(), &first_path, 2).await,
            input(store.clone(), &second_path, 2).await,
        ];
        let result = concat_files(
            &FileConcatTarget::new(version, schema),
            &inputs,
            {
                let store = store.clone();
                let output_path = output_path.clone();
                move || async move { store.create(&output_path).await }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(result, FileConcatResult::Written(_)));

        let output = input(store, &output_path, 4).await;
        let metadata = FileReader::read_all_metadata(&output.scheduler)
            .await
            .unwrap();
        assert_eq!(
            metadata.file_schema.metadata.get("review-key"),
            Some(&"review-value".to_string())
        );
    }

    #[tokio::test]
    async fn unsupported_does_not_create_output() {
        let store = Arc::new(ObjectStore::local());
        let first_path = TempObjFile::default();
        let second_path = TempObjFile::default();
        let schema = write_file(&store, &first_path, ConcreteFileVersion::V2_1, &[1]).await;
        write_file(&store, &second_path, ConcreteFileVersion::V2_2, &[2]).await;
        let inputs = vec![
            input(store.clone(), &first_path, 1).await,
            input(store, &second_path, 1).await,
        ];
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let result = concat_files(
            &FileConcatTarget::new(ConcreteFileVersion::V2_1, schema),
            &inputs,
            {
                let factory_calls = factory_calls.clone();
                move || async move {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::internal("unsupported factory must not be called"))
                }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap();
        assert!(matches!(
            result,
            FileConcatResult::Unsupported(FileConcatReason::VersionMismatch { input_index: 1, .. })
        ));
        assert_eq!(factory_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn legacy_input_is_unsupported_without_creating_output() {
        let store = Arc::new(ObjectStore::local());
        let current_path = TempObjFile::default();
        let legacy_path = TempObjFile::default();
        let schema = write_file(&store, &current_path, ConcreteFileVersion::V2_1, &[1]).await;
        let mut legacy_writer = store.create(&legacy_path).await.unwrap();
        legacy_writer
            .write_all(include_bytes!("../test_data/exact_versions/v1.lance"))
            .await
            .unwrap();
        Writer::shutdown(&mut legacy_writer).await.unwrap();
        let factory_calls = Arc::new(AtomicUsize::new(0));

        let result = concat_files(
            &FileConcatTarget::new(ConcreteFileVersion::V2_1, schema),
            &[input(store, &legacy_path, 0).await],
            {
                let factory_calls = factory_calls.clone();
                move || async move {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::internal("legacy factory must not be called"))
                }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap();

        assert!(matches!(
            result,
            FileConcatResult::Unsupported(FileConcatReason::LegacyVersion)
        ));
        assert_eq!(factory_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn incompatible_column_buffers_and_incomplete_metadata_are_rejected() {
        let store = Arc::new(ObjectStore::local());
        let path = TempObjFile::default();
        let schema = write_file(&store, &path, ConcreteFileVersion::V2_1, &[1, 2]).await;
        let encoded_input = input(store, &path, 2).await;
        let target = FileConcatTarget::new(ConcreteFileVersion::V2_1, schema);

        let mut with_column_buffer = FileReader::read_all_metadata(&encoded_input.scheduler)
            .await
            .unwrap();
        let column = with_column_buffer.column_infos[0].as_ref();
        with_column_buffer.column_infos[0] = Arc::new(ColumnInfo::new(
            column.index,
            column.page_infos.clone(),
            vec![(0, 1)],
            column.encoding.clone(),
        ));
        let prepared = [PreparedInput {
            input: &encoded_input,
            metadata: &with_column_buffer,
            schema: with_column_buffer.file_schema.as_ref(),
        }];
        assert!(matches!(
            check_compatibility(&target, &prepared, false).unwrap(),
            Some(FileConcatReason::ColumnBuffers {
                input_index: 0,
                column_index: 0,
                count: 1
            })
        ));

        let mut missing_column = FileReader::read_all_metadata(&encoded_input.scheduler)
            .await
            .unwrap();
        missing_column.column_infos.clear();
        let prepared = [PreparedInput {
            input: &encoded_input,
            metadata: &missing_column,
            schema: missing_column.file_schema.as_ref(),
        }];
        let error = check_compatibility(&target, &prepared, false).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("schema requires 1 physical columns")
        );

        let mut wrong_rows = FileReader::read_all_metadata(&encoded_input.scheduler)
            .await
            .unwrap();
        let column = wrong_rows.column_infos[0].as_ref();
        let mut pages = column
            .page_infos
            .iter()
            .map(|page| PageInfo {
                num_rows: page.num_rows,
                priority: page.priority,
                encoding: page.encoding.clone(),
                buffer_offsets_and_sizes: page.buffer_offsets_and_sizes.clone(),
            })
            .collect::<Vec<_>>();
        pages[0].num_rows -= 1;
        wrong_rows.column_infos[0] = Arc::new(ColumnInfo::new(
            column.index,
            Arc::from(pages),
            Vec::new(),
            column.encoding.clone(),
        ));
        let prepared = [PreparedInput {
            input: &encoded_input,
            metadata: &wrong_rows,
            schema: wrong_rows.file_schema.as_ref(),
        }];
        let error = check_compatibility(&target, &prepared, false).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("descriptor reports 2 physical rows")
        );
    }

    #[tokio::test]
    async fn data_file_parts_reject_overlapping_blob_leases_before_output() {
        let store = Arc::new(ObjectStore::local());
        let first_path = TempObjFile::default();
        let second_path = TempObjFile::default();
        let schema = write_file(&store, &first_path, ConcreteFileVersion::V2_1, &[1]).await;
        write_file(&store, &second_path, ConcreteFileVersion::V2_1, &[2]).await;
        let first = DataFilePart::open(
            input(store.clone(), &first_path, 1).await,
            Some(1..10),
            None,
        )
        .await
        .unwrap();
        let second = DataFilePart::open(input(store, &second_path, 1).await, Some(5..20), None)
            .await
            .unwrap();
        let factory_calls = Arc::new(AtomicUsize::new(0));

        let error = concat_data_file_parts(
            &FileConcatTarget::new(ConcreteFileVersion::V2_1, schema),
            &[first, second],
            {
                let factory_calls = factory_calls.clone();
                move || async move {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::internal("overlap factory must not be called"))
                }
            },
            FileConcatOptions::default(),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("part 0 uses 1..10"), "{error}");
        assert!(error.to_string().contains("part 1 uses 5..20"), "{error}");
        assert_eq!(factory_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn missing_and_corrupt_inputs_are_errors_without_output() {
        let store = Arc::new(ObjectStore::local());
        let valid_path = TempObjFile::default();
        let missing_path = TempObjFile::default();
        let corrupt_path = TempObjFile::default();
        let schema = write_file(&store, &valid_path, ConcreteFileVersion::V2_1, &[1, 2]).await;
        write_file(&store, &missing_path, ConcreteFileVersion::V2_1, &[3, 4]).await;
        let missing_input = input(store.clone(), &missing_path, 2).await;
        store.delete(&missing_path).await.unwrap();

        let target = FileConcatTarget::new(ConcreteFileVersion::V2_1, schema.clone());
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let result = concat_files(
            &target,
            &[input(store.clone(), &valid_path, 2).await, missing_input],
            {
                let factory_calls = factory_calls.clone();
                move || async move {
                    factory_calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::internal("error factory must not be called"))
                }
            },
            FileConcatOptions::default(),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(factory_calls.load(Ordering::SeqCst), 0);

        let mut corrupt_writer = store.create(&corrupt_path).await.unwrap();
        corrupt_writer.write_all(b"not a Lance file").await.unwrap();
        Writer::shutdown(&mut corrupt_writer).await.unwrap();
        let corrupt_input = input(store.clone(), &corrupt_path, 2).await;
        let result = concat_files(
            &target,
            &[input(store, &valid_path, 2).await, corrupt_input],
            || async { Err(Error::internal("error factory must not be called")) },
            FileConcatOptions::default(),
        )
        .await;
        assert!(result.is_err());
    }
}
