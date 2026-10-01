// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Converting an IVF_RQ index between the column and plane-row layouts.
//!
//! [`rewrite_rq_row_layout`] rewrites the auxiliary file of a dataset's
//! single-segment IVF_RQ index in the other layout and commits it as a new
//! segment. Both layouts hold the same values, so nothing is re-encoded: the
//! index file, the partition offsets and lengths, the global buffers and their
//! positions, and every other schema metadata value stay as they are. Only the
//! internal columns change, and the storage metadata gains or loses its
//! `row_layout` member, as text, so every other byte of it is kept.
//! [`verify_rq_row_layout`] checks a converted index against its source.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_schema::{Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions as file_versions;
use lance_file::writer::{FileWriteSummary, FileWriterOptions};
use lance_index::pb::VectorIndexDetails;
use lance_index::pb::vector_index_details::{Compression, rabit_quantization};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::layered::{EntryColumns, SIGN_BOUNDS_PLANE, SignBounds};
use lance_index::vector::bq::plane_rows::{PlaneRowsSpec, requests_fullzip};
use lance_index::vector::bq::storage::{
    RABIT_METADATA_KEY, RQRowLayout, RabitQuantizationMetadata, RabitQuantizationStorage,
};
use lance_index::vector::storage::{IvfQuantizationStorage, STORAGE_METADATA_KEY, VectorStore};
use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, ivf_rq_index_version};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{IndexFile, IndexMetadata};
use object_store::path::Path;
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::read_partition_window_batches;
use crate::Dataset;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;

/// The storage metadata key of the row layout.
const ROW_LAYOUT_KEY: &str = "row_layout";
/// The storage metadata member the plane-row layout adds, as JSON text.
const PLANE_ROWS_MEMBER: &str = r#""row_layout":"plane_rows""#;

/// Partition bytes above which one partition's read of a plane-row file is a
/// large GET, reported by [`verify_rq_row_layout`].
const LARGE_PARTITION_BYTES: u64 = 16 * 1024 * 1024;

/// One schema metadata value a conversion rewrote.
#[derive(Debug, Clone, Serialize)]
pub struct MetadataPatch {
    pub key: String,
    pub source: String,
    pub converted: String,
}

/// What [`rewrite_rq_row_layout`] did.
#[derive(Debug, Clone, Serialize)]
pub struct RowLayoutRewrite {
    pub index_name: String,
    pub old_uuid: String,
    pub new_uuid: String,
    pub source_layout: String,
    pub target_layout: String,
    /// The index version the new segment was committed with.
    pub index_version: i32,
    /// The Lance file version of both auxiliary files.
    pub file_version: String,
    pub num_partitions: usize,
    pub num_rows: u64,
    pub source_aux_bytes: u64,
    pub aux_bytes: u64,
    pub index_file_bytes: u64,
    /// User global buffers copied, in order, at their positions.
    pub global_buffers: usize,
    pub metadata: Vec<MetadataPatch>,
    /// The dataset version the conversion committed.
    pub dataset_version: u64,
}

/// Rewrite the auxiliary file of `dataset`'s IVF_RQ index `index_name` in
/// `target` and commit it as a new segment with the matching index version,
/// in place of the old one. The index must be one segment, in the other
/// layout, whose `_indices/<uuid>/` directory `dataset` holds. The old
/// directory is left for the caller to remove once the conversion is
/// verified, so earlier dataset versions stay readable until then.
pub async fn rewrite_rq_row_layout(
    dataset: &mut Dataset,
    index_name: &str,
    target: RQRowLayout,
) -> Result<RowLayoutRewrite> {
    rewrite_rq_row_layout_with_options(dataset, index_name, target, FileWriterOptions::default())
        .await
}

/// [`rewrite_rq_row_layout`], writing the new auxiliary file with
/// `writer_options`, which only change where its pages end.
pub async fn rewrite_rq_row_layout_with_options(
    dataset: &mut Dataset,
    index_name: &str,
    target: RQRowLayout,
    writer_options: FileWriterOptions,
) -> Result<RowLayoutRewrite> {
    rewrite_rq_row_layout_impl(dataset, index_name, target, writer_options, None).await
}

/// The conversion, writing each partition in batches of at most
/// `batch_rows` rows when set (one batch otherwise), so tests can end pages
/// inside partitions.
async fn rewrite_rq_row_layout_impl(
    dataset: &mut Dataset,
    index_name: &str,
    target: RQRowLayout,
    writer_options: FileWriterOptions,
    batch_rows: Option<usize>,
) -> Result<RowLayoutRewrite> {
    let old = single_rq_segment(dataset, index_name).await?;
    let store = dataset.object_store.clone();
    let old_dir = dataset.indices_dir().join(old.uuid.to_string());
    let old_aux = old_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
    let old_index = old_dir.clone().join(INDEX_FILE_NAME);
    for path in [&old_aux, &old_index] {
        if !store.exists(path).await? {
            return Err(Error::invalid_input(format!(
                "converting index {index_name}: {path} is missing; convert a dataset that holds the index's directory"
            )));
        }
    }
    let source = open_rq_storage(&store, &old_aux).await?;
    let metadata = source.metadata();
    let source_layout = metadata.row_layout;
    if source_layout == target {
        return Err(Error::invalid_input(format!(
            "index {index_name} is already in the {target} layout"
        )));
    }
    let reader = source.reader();
    let file_schema = ArrowSchema::from(reader.schema().as_ref());
    let spec = match target {
        RQRowLayout::PlaneRows => PlaneRowsSpec::for_column_file(metadata, &file_schema)?,
        RQRowLayout::Columns => PlaneRowsSpec::for_file(metadata, &file_schema)?,
    };
    let patches = patch_schema_metadata(file_schema.metadata(), target)?;
    let mut schema_metadata = file_schema.metadata().clone();
    for patch in &patches {
        schema_metadata.insert(patch.key.clone(), patch.converted.clone());
    }
    let version = reader.metadata().version();
    let fields: Vec<Field> = match target {
        RQRowLayout::PlaneRows => spec.file_fields(requests_fullzip(version)),
        RQRowLayout::Columns => spec
            .logical_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect(),
    };
    let schema = ArrowSchema::new_with_metadata(fields, schema_metadata);

    let new_uuid = Uuid::new_v4();
    let new_dir = dataset.indices_dir().join(new_uuid.to_string());
    let new_aux = new_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
    let mut writer = file_versions::create_writer(
        version,
        store.create(&new_aux).await?,
        lance_core::datatypes::Schema::try_from(&schema)?,
        writer_options,
    )?;
    // The IVF buffer and the rotation matrix are referenced by position, so
    // they are copied in order to the same positions.
    let global_buffers = reader.metadata().file_buffers.len().saturating_sub(1);
    for position in 1..=global_buffers as u32 {
        let bytes = reader.read_global_buffer(position).await?;
        let copied = writer.add_global_buffer(bytes).await?;
        if copied != position {
            return Err(Error::internal(format!(
                "converting index {index_name}: global buffer {position} was written at {copied}"
            )));
        }
    }
    let ivf = source.ivf();
    let file_schema_ref = Arc::new(file_schema.clone());
    for partition in 0..ivf.num_partitions() {
        let range = ivf.row_range(partition);
        if range.is_empty() {
            continue;
        }
        let batches = reader
            .read_stream(
                ReadBatchParams::Range(range),
                u32::MAX,
                1,
                FilterExpression::no_filter(),
            )
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let batch = match target {
            RQRowLayout::PlaneRows => spec.pack(&concat_batches(&file_schema_ref, &batches)?)?,
            RQRowLayout::Columns => spec.unpack(&batches, spec.logical_schema())?,
        };
        let step = batch_rows.unwrap_or(batch.num_rows()).max(1);
        for start in (0..batch.num_rows()).step_by(step) {
            let rows = step.min(batch.num_rows() - start);
            writer.write_batch(&batch.slice(start, rows)).await?;
        }
    }
    let FileWriteSummary {
        num_rows,
        size_bytes: aux_bytes,
    } = writer.finish().await?;
    if num_rows != reader.num_rows() {
        return Err(Error::internal(format!(
            "converting index {index_name}: wrote {num_rows} of the source's {} rows",
            reader.num_rows()
        )));
    }
    let new_index = new_dir.join(INDEX_FILE_NAME);
    store.copy(&old_index, &new_index).await?;
    let index_file_bytes = store.size(&new_index).await?;
    let source_aux_bytes = store.size(&old_aux).await?;

    let index_version = ivf_rq_index_version(target);
    let new = IndexMetadata {
        uuid: new_uuid,
        index_details: Some(Arc::new(details_with_row_layout(&old, target)?)),
        index_version,
        files: Some(vec![
            IndexFile {
                path: INDEX_AUXILIARY_FILE_NAME.to_string(),
                size_bytes: aux_bytes,
            },
            IndexFile {
                path: INDEX_FILE_NAME.to_string(),
                size_bytes: index_file_bytes,
            },
        ]),
        ..old.clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new],
            removed_indices: vec![old.clone()],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(RowLayoutRewrite {
        index_name: index_name.to_string(),
        old_uuid: old.uuid.to_string(),
        new_uuid: new_uuid.to_string(),
        source_layout: source_layout.to_string(),
        target_layout: target.to_string(),
        index_version,
        file_version: version.to_manifest_string().to_string(),
        num_partitions: ivf.num_partitions(),
        num_rows,
        source_aux_bytes,
        aux_bytes,
        index_file_bytes,
        global_buffers,
        metadata: patches,
        dataset_version: dataset.manifest.version,
    })
}

/// The one segment of IVF_RQ index `index_name`.
async fn single_rq_segment(dataset: &Dataset, index_name: &str) -> Result<IndexMetadata> {
    let mut segments = dataset.load_indices_by_name(index_name).await?;
    if segments.len() != 1 {
        return Err(Error::invalid_input(format!(
            "converting index {index_name}: it has {} segments; only a single-segment index converts",
            segments.len()
        )));
    }
    let segment = segments.remove(0);
    rq_details(&segment)?;
    Ok(segment)
}

/// The segment's vector details, which must describe IVF_RQ.
fn rq_details(segment: &IndexMetadata) -> Result<VectorIndexDetails> {
    let details = segment
        .index_details
        .as_ref()
        .filter(|details| !details.value.is_empty())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "converting index {}: segment {} has no vector index details",
                segment.name, segment.uuid
            ))
        })?
        .to_msg::<VectorIndexDetails>()
        .map_err(|error| {
            Error::invalid_input(format!(
                "converting index {}: segment {} is not a vector index: {error}",
                segment.name, segment.uuid
            ))
        })?;
    if !matches!(details.compression, Some(Compression::Rq(_)))
        || details.hnsw_index_config.is_some()
    {
        return Err(Error::invalid_input(format!(
            "converting index {}: segment {} is not an IVF_RQ index",
            segment.name, segment.uuid
        )));
    }
    Ok(details)
}

/// `segment`'s details with the row layout `target`.
fn details_with_row_layout(
    segment: &IndexMetadata,
    target: RQRowLayout,
) -> Result<prost_types::Any> {
    let mut details = rq_details(segment)?;
    if let Some(Compression::Rq(rq)) = details.compression.as_mut() {
        rq.set_row_layout(match target {
            RQRowLayout::Columns => rabit_quantization::RowLayout::Columns,
            RQRowLayout::PlaneRows => rabit_quantization::RowLayout::PlaneRows,
        });
    }
    prost_types::Any::from_msg(&details)
        .map_err(|error| Error::internal(format!("encoding vector index details: {error}")))
}

/// The IVF_RQ storage of the auxiliary file at `path`.
async fn open_rq_storage(
    store: &Arc<ObjectStore>,
    path: &Path,
) -> Result<IvfQuantizationStorage<RabitQuantizer>> {
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(store));
    let reader = FileReader::try_open(
        scheduler
            .open_file(path, &CachedFileSize::unknown())
            .await?,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await?;
    IvfQuantizationStorage::try_new(reader, None).await
}

/// The schema metadata values a conversion to `target` rewrites: the
/// storage metadata, and the quantizer metadata files written by the
/// distributed merger also carry, each with the plane-row member added as
/// its last member or removed, as text.
fn patch_schema_metadata(
    metadata: &HashMap<String, String>,
    target: RQRowLayout,
) -> Result<Vec<MetadataPatch>> {
    let insert = target == RQRowLayout::PlaneRows;
    let storage = metadata.get(STORAGE_METADATA_KEY).ok_or_else(|| {
        Error::invalid_input(format!(
            "the auxiliary file has no {STORAGE_METADATA_KEY} metadata"
        ))
    })?;
    // The storage metadata is a JSON list of one JSON string, so the member
    // goes in escaped, before the closing brace of that string's object.
    let escaped = serde_json::to_string(PLANE_ROWS_MEMBER)?;
    let escaped = &escaped[1..escaped.len() - 1];
    let converted = patch_object_text(storage, escaped, insert)?;
    check_storage_metadata(storage, &converted, target)?;
    let mut patches = vec![MetadataPatch {
        key: STORAGE_METADATA_KEY.to_string(),
        source: storage.clone(),
        converted,
    }];
    if let Some(rabit) = metadata.get(RABIT_METADATA_KEY) {
        let converted = patch_object_text(rabit, PLANE_ROWS_MEMBER, insert)?;
        check_metadata_object(rabit, &converted, target)?;
        patches.push(MetadataPatch {
            key: RABIT_METADATA_KEY.to_string(),
            source: rabit.clone(),
            converted,
        });
    }
    Ok(patches)
}

/// `text`, which ends with a JSON object's closing brace and possibly
/// closing list and string delimiters, with `member` added as the object's
/// last member or, when `!insert`, removed with its separator; every other
/// byte is kept.
fn patch_object_text(text: &str, member: &str, insert: bool) -> Result<String> {
    if insert {
        let (open, close) = text
            .find('{')
            .zip(text.rfind('}'))
            .ok_or_else(|| Error::invalid_input(format!("{text:?} holds no JSON object")))?;
        let separator = if text[open + 1..close].trim().is_empty() {
            ""
        } else {
            ","
        };
        return Ok(format!(
            "{}{separator}{member}{}",
            &text[..close],
            &text[close..]
        ));
    }
    for candidate in [
        format!(",{member}"),
        format!("{member},"),
        member.to_string(),
    ] {
        if let Some(start) = text.find(&candidate) {
            return Ok(format!(
                "{}{}",
                &text[..start],
                &text[start + candidate.len()..]
            ));
        }
    }
    Err(Error::invalid_input(format!(
        "{text:?} has no {member} member to remove"
    )))
}

/// Check that `converted`, the storage metadata after a conversion to
/// `target`, is `source`'s single entry with the row layout member alone
/// added or removed.
fn check_storage_metadata(source: &str, converted: &str, target: RQRowLayout) -> Result<()> {
    let entry = |text: &str| -> Result<String> {
        let mut entries: Vec<String> = serde_json::from_str(text)?;
        if entries.len() != 1 {
            return Err(Error::invalid_input(format!(
                "the storage metadata holds {} entries, expected one",
                entries.len()
            )));
        }
        Ok(entries.remove(0))
    };
    check_metadata_object(&entry(source)?, &entry(converted)?, target)
}

/// Check that `converted`, an IVF_RQ metadata object after a conversion to
/// `target`, is `source` with the row layout member alone added or removed,
/// and reads back as `target`.
fn check_metadata_object(source: &str, converted: &str, target: RQRowLayout) -> Result<()> {
    let object = |text: &str| -> Result<serde_json::Map<String, serde_json::Value>> {
        match serde_json::from_str(text)? {
            serde_json::Value::Object(object) => Ok(object),
            other => Err(Error::invalid_input(format!(
                "the IVF_RQ metadata {other} is not a JSON object"
            ))),
        }
    };
    let mut expected = object(source)?;
    match target {
        RQRowLayout::PlaneRows => {
            if expected.contains_key(ROW_LAYOUT_KEY) {
                return Err(Error::invalid_input(format!(
                    "the IVF_RQ metadata {source} already declares a row layout"
                )));
            }
            expected.insert(
                ROW_LAYOUT_KEY.to_string(),
                serde_json::Value::String(RQRowLayout::PlaneRows.to_string()),
            );
        }
        RQRowLayout::Columns => {
            expected.remove(ROW_LAYOUT_KEY);
        }
    }
    let read: RabitQuantizationMetadata = serde_json::from_str(converted)?;
    if object(converted)? != expected || read.row_layout != target {
        return Err(Error::internal(format!(
            "rewriting the IVF_RQ metadata {source} for {target} gave {converted}"
        )));
    }
    Ok(())
}

/// How a plane-row file's pages fall on its partitions.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PageStats {
    /// Pages of the packed columns, summed over them.
    pub pages: usize,
    /// Partition reads of a packed column whose rows span more than one of
    /// its pages, summed over the packed columns.
    pub straddling_partitions: usize,
    /// The largest number of bytes one partition takes, over all its
    /// packed columns.
    pub max_partition_bytes: u64,
    /// Partitions taking more than 16 MiB.
    pub partitions_over_16mib: usize,
}

/// Shapes of reads and cache entries whose memory [`verify_rq_row_layout`]
/// compares, as [`RowLayoutVerification::deep_sizes`] names them.
pub mod deep_size_shapes {
    /// Every row of a partition, which partition entries are built from.
    pub const PARTITION_BATCH: &str = "partition_batch";
    /// Every row of a plane of a layered index, which a plane entry holding
    /// every column holds.
    pub const PLANE: &str = "plane";
    /// Every third row of a plane of a layered index: a sparse read, which
    /// no cache entry holds.
    pub const PLANE_SPARSE: &str = "plane_sparse";
    /// The storage a partition entry holds.
    pub const PARTITION_STORAGE: &str = "partition_storage";
    /// The storage a partition entry holds when a prewarm built it from a
    /// read of several partitions.
    pub const PREWARM_PARTITION_STORAGE: &str = "prewarm_partition_storage";
    /// A code-only entry: a native partition's or a layered plane's code
    /// (and bounds) columns.
    pub const CODE_ONLY_ENTRY: &str = "code_only_entry";
    /// A native partition's code-only entry as a prewarm cuts it from a read
    /// of several partitions.
    pub const PREWARM_CODE_ONLY_ENTRY: &str = "prewarm_code_only_entry";
}

/// The memory one shape of read or cache entry takes on both sides.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DeepSizes {
    /// Reads or entries compared.
    pub count: usize,
    /// Those whose two sides take different memory.
    pub mismatches: usize,
    pub source_bytes: u64,
    pub converted_bytes: u64,
}

impl DeepSizes {
    fn record(&mut self, source: u64, converted: u64) {
        self.count += 1;
        self.mismatches += usize::from(source != converted);
        self.source_bytes += source;
        self.converted_bytes += converted;
    }
}

/// What [`verify_rq_row_layout`] found.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RowLayoutVerification {
    pub source_uuid: String,
    pub converted_uuid: String,
    pub source_layout: String,
    pub converted_layout: String,
    /// The Lance file version of each side's auxiliary file.
    pub source_file_version: String,
    pub converted_file_version: String,
    pub partitions: usize,
    pub rows: u64,
    /// Reads compared: every partition whole, and for a layered index every
    /// plane, whole and at every third row.
    pub reads: usize,
    /// Cache entries compared: every partition's storage, read alone and
    /// cut from a prewarm's read, and its code-only entries.
    pub entries: usize,
    /// Reads and entries whose batches differ in a field or a value.
    pub batch_mismatches: usize,
    /// Whole reads and cache entries whose two sides take different memory,
    /// so would be charged differently in a cache, and their memory.
    pub deep_size_mismatches: usize,
    pub source_deep_bytes: u64,
    pub converted_deep_bytes: u64,
    /// Sparse reads whose two sides take different memory, and their
    /// memory. No cache entry holds a sparse read, so this is reported
    /// apart: a column-layout read of selected rows of a Lance 2.0 file can
    /// keep more buffer than its values.
    pub sparse_deep_size_mismatches: usize,
    pub sparse_source_deep_bytes: u64,
    pub sparse_converted_deep_bytes: u64,
    /// The memory of each shape of read and entry ([`deep_size_shapes`]).
    pub deep_sizes: BTreeMap<String, DeepSizes>,
    /// SHA-256 over every partition's values, rows packed as the plane-row
    /// layout packs them, for each side.
    pub source_digest: String,
    pub converted_digest: String,
    /// Whether the column layout's fields read from both sides are the same.
    pub schema_matches: bool,
    /// The storage metadata of both sides, and whether the converted one is
    /// the source's with the row layout alone added or removed, as text.
    pub source_storage_metadata: String,
    pub converted_storage_metadata: String,
    pub metadata_patch_ok: bool,
    /// Whether every other schema metadata value is byte for byte the same.
    pub other_metadata_equal: bool,
    pub rotate_mat_position: Option<u32>,
    pub rotate_mat_position_equal: bool,
    /// SHA-256 of each user global buffer of the source, and whether the
    /// converted file's are the same, in the same positions.
    pub global_buffer_digests: Vec<String>,
    pub global_buffers_equal: bool,
    /// Whether both index files hold the same bytes.
    pub index_file_equal: bool,
    /// Page stats of whichever side stores plane rows.
    pub pages: PageStats,
}

impl RowLayoutVerification {
    /// Whether the converted index holds the source's values, files and
    /// metadata, but for the layout, and every whole read and cache entry of
    /// it takes the memory the source's does, so a cache charges both the
    /// same. Sparse reads, which no entry holds, are reported apart.
    pub fn is_ok(&self) -> bool {
        self.batch_mismatches == 0
            && self.deep_size_mismatches == 0
            && self.source_digest == self.converted_digest
            && self.schema_matches
            && self.metadata_patch_ok
            && self.other_metadata_equal
            && self.rotate_mat_position_equal
            && self.global_buffers_equal
            && self.index_file_equal
    }

    /// Count a read or entry of `shape` whose values are `same` and whose
    /// sides take `source` and `converted` bytes.
    fn record(&mut self, shape: &str, same: bool, source: u64, converted: u64) {
        self.batch_mismatches += usize::from(!same);
        self.deep_sizes
            .entry(shape.to_string())
            .or_default()
            .record(source, converted);
        let mismatch = usize::from(source != converted);
        if shape == deep_size_shapes::PLANE_SPARSE {
            self.sparse_deep_size_mismatches += mismatch;
            self.sparse_source_deep_bytes += source;
            self.sparse_converted_deep_bytes += converted;
        } else {
            self.deep_size_mismatches += mismatch;
            self.source_deep_bytes += source;
            self.converted_deep_bytes += converted;
        }
    }

    /// Count read `a` of the source against read `b` of the converted index.
    fn compare(&mut self, shape: &str, a: &RecordBatch, b: &RecordBatch) {
        self.reads += usize::from(!is_entry_shape(shape));
        self.entries += usize::from(is_entry_shape(shape));
        let same = a.schema().fields() == b.schema().fields() && a.columns() == b.columns();
        self.record(shape, same, deep_bytes(a), deep_bytes(b));
    }

    /// Count the storage of a partition entry of each side.
    fn compare_storage(
        &mut self,
        shape: &str,
        a: &RabitQuantizationStorage,
        b: &RabitQuantizationStorage,
    ) {
        self.entries += 1;
        let same = a.len() == b.len() && a.row_ids().eq(b.row_ids());
        self.record(
            shape,
            same,
            a.deep_size_of() as u64,
            b.deep_size_of() as u64,
        );
    }
}

/// `storage`'s partitions as a prewarm cuts them from one read of every
/// partition, which it builds their entries from.
async fn prewarm_window(
    storage: &IvfQuantizationStorage<RabitQuantizer>,
) -> Result<Vec<Vec<RecordBatch>>> {
    let schema = Arc::new(ArrowSchema::from(storage.reader().schema().as_ref()));
    read_partition_window_batches(
        storage.reader(),
        None,
        &schema,
        storage.ivf(),
        0..storage.num_partitions(),
        None,
    )
    .await
}

/// Whether `shape` is a cache entry rather than a read.
fn is_entry_shape(shape: &str) -> bool {
    [
        deep_size_shapes::CODE_ONLY_ENTRY,
        deep_size_shapes::PREWARM_CODE_ONLY_ENTRY,
    ]
    .contains(&shape)
}

/// The memory `batch`'s arrays take, as a cache charges it.
fn deep_bytes(batch: &RecordBatch) -> u64 {
    batch.deep_size_of_children(&mut Context::new()) as u64
}

/// Check that index `index_name` of `converted`, a single segment converted
/// from `source`'s, holds the same values: every partition and, for a
/// layered index, every plane under both bounds placements, whole and at
/// every third row; the same global buffers in the same positions, the same
/// index file, and the same schema metadata but for the row layout member.
pub async fn verify_rq_row_layout(
    source: &Dataset,
    converted: &Dataset,
    index_name: &str,
) -> Result<RowLayoutVerification> {
    let source_segment = single_rq_segment(source, index_name).await?;
    let converted_segment = single_rq_segment(converted, index_name).await?;
    let segment_file = |dataset: &Dataset, segment: &IndexMetadata, file: &str| {
        dataset
            .indices_dir()
            .join(segment.uuid.to_string())
            .join(file)
    };
    let source_aux = segment_file(source, &source_segment, INDEX_AUXILIARY_FILE_NAME);
    let converted_aux = segment_file(converted, &converted_segment, INDEX_AUXILIARY_FILE_NAME);
    let open = |dataset: &Dataset, path: &Path, sign_bounds: SignBounds| {
        let store = dataset.object_store.clone();
        let path = path.clone();
        async move {
            Ok::<_, Error>(
                open_rq_storage(&store, &path)
                    .await?
                    .with_sign_bounds(sign_bounds),
            )
        }
    };
    let lazy = [
        open(source, &source_aux, SignBounds::Lazy).await?,
        open(converted, &converted_aux, SignBounds::Lazy).await?,
    ];
    let eager = [
        open(source, &source_aux, SignBounds::Eager).await?,
        open(converted, &converted_aux, SignBounds::Eager).await?,
    ];
    let [src, conv] = &lazy;
    let file_version = |storage: &IvfQuantizationStorage<RabitQuantizer>| -> String {
        storage
            .reader()
            .metadata()
            .version()
            .to_manifest_string()
            .to_string()
    };
    let mut report = RowLayoutVerification {
        source_uuid: source_segment.uuid.to_string(),
        converted_uuid: converted_segment.uuid.to_string(),
        source_layout: src.row_layout().to_string(),
        converted_layout: conv.row_layout().to_string(),
        source_file_version: file_version(src),
        converted_file_version: file_version(conv),
        partitions: src.num_partitions(),
        rows: src.num_rows(),
        schema_matches: src.logical_schema()?.fields() == conv.logical_schema()?.fields(),
        ..Default::default()
    };
    if src.row_layout() == conv.row_layout() {
        return Err(Error::invalid_input(format!(
            "index {index_name} is in the {} layout on both sides",
            src.row_layout()
        )));
    }
    if src.ivf().lengths != conv.ivf().lengths || src.num_rows() != conv.num_rows() {
        return Err(Error::invalid_input(format!(
            "index {index_name}: the converted partitions differ from the source's"
        )));
    }

    // The packed rows of both sides digest the values the same way.
    let plane_rows = |storage: &IvfQuantizationStorage<RabitQuantizer>| -> Result<PlaneRowsSpec> {
        let schema = ArrowSchema::from(storage.reader().schema().as_ref());
        match storage.row_layout() {
            RQRowLayout::PlaneRows => PlaneRowsSpec::for_file(storage.metadata(), &schema),
            RQRowLayout::Columns => PlaneRowsSpec::for_column_file(storage.metadata(), &schema),
        }
    };
    let specs = [plane_rows(src)?, plane_rows(conv)?];
    let mut digests = [Sha256::new(), Sha256::new()];
    let layered = src.is_layered_rq();
    let [mut src_prewarm, mut conv_prewarm] =
        [prewarm_window(src).await?, prewarm_window(conv).await?].map(Vec::into_iter);
    // A native partition's code-only entry as a prewarm of the column side
    // cuts it; the plane-row layout caches none, so its twin is the read of
    // the same columns.
    let column_side = if src.row_layout() == RQRowLayout::Columns {
        src
    } else {
        conv
    };
    let mut prewarm_codes = if layered {
        Vec::new()
    } else {
        let projection = column_side.partition_codes_projection()?;
        let schema = Arc::new(ArrowSchema::from(projection.schema.as_ref()));
        read_partition_window_batches(
            column_side.reader(),
            Some(&projection),
            &schema,
            column_side.ivf(),
            0..column_side.num_partitions(),
            None,
        )
        .await?
    }
    .into_iter();
    for partition in 0..report.partitions {
        let batches = [
            src.read_partition_batch(partition, None).await?,
            conv.read_partition_batch(partition, None).await?,
        ];
        report.compare(deep_size_shapes::PARTITION_BATCH, &batches[0], &batches[1]);
        for ((digest, spec), batch) in digests.iter_mut().zip(&specs).zip(&batches) {
            let packed = spec.pack(batch)?;
            for column in spec.packed_columns() {
                let rows = packed
                    .column_by_name(column.name())
                    .and_then(|rows| rows.as_fixed_size_list_opt())
                    .and_then(|rows| rows.values().as_primitive_opt::<UInt8Type>())
                    .ok_or_else(|| {
                        Error::internal(format!("packed column {} is missing", column.name()))
                    })?;
                digest.update(rows.values());
            }
        }
        report.compare_storage(
            deep_size_shapes::PARTITION_STORAGE,
            &src.load_partition(partition, None).await?,
            &conv.load_partition(partition, None).await?,
        );
        let prewarmed = [src_prewarm.next(), conv_prewarm.next()].map(|batches| {
            batches.ok_or_else(|| Error::internal("a prewarm read missed a partition"))
        });
        let [src_prewarmed, conv_prewarmed] = prewarmed;
        report.compare_storage(
            deep_size_shapes::PREWARM_PARTITION_STORAGE,
            &src.materialize_partition_for_prewarm(src_prewarmed?)
                .await?,
            &conv
                .materialize_partition_for_prewarm(conv_prewarmed?)
                .await?,
        );
        if !layered {
            let codes = [
                src.read_partition_codes(partition, None).await?,
                conv.read_partition_codes(partition, None).await?,
            ];
            report.compare(deep_size_shapes::CODE_ONLY_ENTRY, &codes[0].0, &codes[1].0);
            let prewarmed = column_side
                .partition_codes_from_batches(
                    prewarm_codes
                        .next()
                        .ok_or_else(|| Error::internal("a prewarm read missed a partition"))?,
                )?
                .0;
            let [a, b] = if column_side.row_layout() == src.row_layout() {
                [&prewarmed, &codes[1].0]
            } else {
                [&codes[0].0, &prewarmed]
            };
            report.compare(deep_size_shapes::PREWARM_CODE_ONLY_ENTRY, a, b);
            continue;
        }
        let rows = src.partition_size(partition);
        let every_third: Vec<u32> = (0..rows as u32).step_by(3).collect();
        for (storages, planes) in [
            (&lazy, &[0, 1, 2, SIGN_BOUNDS_PLANE][..]),
            (&eager, &[0][..]),
        ] {
            let [a, b] = storages;
            for &plane in planes {
                for selected in [None, Some(every_third.clone())] {
                    let shape = if selected.is_some() {
                        deep_size_shapes::PLANE_SPARSE
                    } else {
                        deep_size_shapes::PLANE
                    };
                    report.compare(
                        shape,
                        &a.read_plane(partition, plane, selected.clone(), None)
                            .await?,
                        &b.read_plane(partition, plane, selected.clone(), None)
                            .await?,
                    );
                }
                report.compare(
                    deep_size_shapes::CODE_ONLY_ENTRY,
                    &a.read_plane_entry_with(partition, plane, EntryColumns::Codes, None)
                        .await?,
                    &b.read_plane_entry_with(partition, plane, EntryColumns::Codes, None)
                        .await?,
                );
            }
        }
    }
    let [source_digest, converted_digest] = digests.map(|digest| hex(&digest.finalize()));
    report.source_digest = source_digest;
    report.converted_digest = converted_digest;

    // Metadata: the storage metadata differs by the row layout member alone,
    // and every other value is the same.
    let metadata = |storage: &IvfQuantizationStorage<RabitQuantizer>| {
        storage.reader().schema().metadata.clone()
    };
    let [source_metadata, converted_metadata] = [metadata(src), metadata(conv)];
    let patched = patch_schema_metadata(&source_metadata, conv.row_layout());
    report.source_storage_metadata = source_metadata
        .get(STORAGE_METADATA_KEY)
        .cloned()
        .unwrap_or_default();
    report.converted_storage_metadata = converted_metadata
        .get(STORAGE_METADATA_KEY)
        .cloned()
        .unwrap_or_default();
    report.metadata_patch_ok = patched.as_ref().is_ok_and(|patches| {
        patches
            .iter()
            .all(|patch| converted_metadata.get(&patch.key) == Some(&patch.converted))
    });
    let patched_keys = [STORAGE_METADATA_KEY, RABIT_METADATA_KEY];
    let others = |metadata: &HashMap<String, String>| {
        metadata
            .iter()
            .filter(|(key, _)| !patched_keys.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<HashMap<_, _>>()
    };
    report.other_metadata_equal = others(&source_metadata) == others(&converted_metadata)
        && source_metadata.contains_key(RABIT_METADATA_KEY)
            == converted_metadata.contains_key(RABIT_METADATA_KEY);
    report.rotate_mat_position = src.metadata().rotate_mat_position;
    report.rotate_mat_position_equal =
        src.metadata().rotate_mat_position == conv.metadata().rotate_mat_position;

    // Global buffers, in their positions, and the index file.
    let buffers = |storage: &IvfQuantizationStorage<RabitQuantizer>| {
        storage
            .reader()
            .metadata()
            .file_buffers
            .len()
            .saturating_sub(1)
    };
    report.global_buffers_equal = buffers(src) == buffers(conv);
    for position in 1..=buffers(src) as u32 {
        let source_bytes = src.reader().read_global_buffer(position).await?;
        report
            .global_buffer_digests
            .push(hex(&Sha256::digest(&source_bytes)));
        if report.global_buffers_equal {
            let converted_bytes = conv.reader().read_global_buffer(position).await?;
            report.global_buffers_equal &= source_bytes == converted_bytes;
        }
    }
    let index_bytes = |dataset: &Dataset, segment: &IndexMetadata| {
        let store = dataset.object_store.clone();
        let path = segment_file(dataset, segment, INDEX_FILE_NAME);
        async move { store.read_one_all(&path).await }
    };
    report.index_file_equal = index_bytes(source, &source_segment).await?
        == index_bytes(converted, &converted_segment).await?;

    let plane_row_side = if src.row_layout() == RQRowLayout::PlaneRows {
        src
    } else {
        conv
    };
    report.pages = page_stats(plane_row_side)?;
    Ok(report)
}

/// How the pages of `storage`'s plane-row file fall on its partitions.
fn page_stats(storage: &IvfQuantizationStorage<RabitQuantizer>) -> Result<PageStats> {
    let spec = storage
        .plane_rows()?
        .ok_or_else(|| Error::internal("page stats need a plane-row file"))?;
    let reader = storage.reader();
    let mut stats = PageStats::default();
    let ivf = storage.ivf();
    let row_bytes: usize = spec
        .packed_columns()
        .iter()
        .map(|column| column.stride())
        .sum();
    for partition in 0..ivf.num_partitions() {
        let bytes = (ivf.partition_size(partition) * row_bytes) as u64;
        stats.max_partition_bytes = stats.max_partition_bytes.max(bytes);
        stats.partitions_over_16mib += usize::from(bytes > LARGE_PARTITION_BYTES);
    }
    for column in spec.packed_columns() {
        let projection = lance_file::versions::reader_projection_from_column_names(
            reader.metadata().version(),
            reader.schema(),
            &[column.name()],
        )?;
        let index = projection.column_indices[0] as usize;
        let info =
            reader.metadata().column_infos.get(index).ok_or_else(|| {
                Error::internal(format!("column {} has no page info", column.name()))
            })?;
        stats.pages += info.page_infos.len();
        // The first row of every page after the first.
        let mut starts = Vec::with_capacity(info.page_infos.len());
        let mut next = 0u64;
        for page in info.page_infos.iter() {
            next += page.num_rows;
            starts.push(next);
        }
        starts.pop();
        for partition in 0..ivf.num_partitions() {
            let range = ivf.row_range(partition);
            if range.is_empty() {
                continue;
            }
            let (start, end) = (range.start as u64, range.end as u64);
            stats.straddling_partitions += usize::from(
                starts
                    .iter()
                    .any(|&page_start| page_start > start && page_start < end),
            );
        }
    }
    Ok(stats)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::{Float32Type, UInt64Type};
    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatchIterator, UInt64Array,
    };
    use arrow_schema::DataType;
    use bytes::Bytes;
    use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
    use lance_core::ROW_ID;
    use lance_core::cache::CacheCodec;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_encoding::decoder::PageEncoding;
    use lance_encoding::format::pb21::page_layout::Layout;
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::vector::DIST_COL;
    use lance_index::vector::bq::layered::{PlaneBatch, RQPrecision, SignBounds};
    use lance_index::vector::bq::storage::{RABIT_CODE_COLUMN, unpack_codes};
    use lance_index::vector::bq::{RQBuildParams, RQRotationType};
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::storage::VectorStore;
    use lance_io::scheduler::IoStats;
    use lance_linalg::distance::DistanceType;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::index::vector::ivf::v2::IVFIndex;
    use crate::index::{DatasetIndexInternalExt, vector::VectorIndex};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::vector::flat::index::FlatIndex;

    const INDEX_NAME: &str = "vector_idx";
    const DIM: usize = 64;
    /// Partitions of the test index; the data clusters around all but the
    /// last centroid, which is left with an empty partition.
    const PARTITIONS: usize = 6;
    const ROWS: usize = 3_000;
    /// A page budget well below a partition's bytes, and batches of a number
    /// of rows that divides no partition, so pages end inside partitions.
    const SMALL_PAGE_BYTES: u64 = 4096;
    const SMALL_BATCH_ROWS: usize = 97;

    type IvfRq = IVFIndex<FlatIndex, RabitQuantizer>;

    fn centroids() -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(0x9470);
        (0..PARTITIONS * DIM)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect()
    }

    /// `rows` vectors clustered around the first `PARTITIONS - 1` centroids,
    /// with ids from `first_id`.
    fn test_batch(rows: usize, first_id: u64, seed: u64) -> RecordBatch {
        let centroids = centroids();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = Vec::with_capacity(rows * DIM);
        for _ in 0..rows {
            let cluster = rng.random_range(0..PARTITIONS - 1);
            for dim in 0..DIM {
                values.push(centroids[cluster * DIM + dim] + rng.random_range(-0.05f32..0.05));
            }
        }
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from_iter_values(
                    first_id..first_id + rows as u64,
                )),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    async fn write_dataset(
        uri: &str,
        version: LanceFileVersion,
        max_rows_per_file: usize,
    ) -> Dataset {
        let batch = test_batch(ROWS, 0, 1);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let params = WriteParams {
            data_storage_version: Some(version),
            max_rows_per_file,
            mode: WriteMode::Create,
            ..Default::default()
        };
        Dataset::write(reader, uri, Some(params)).await.unwrap()
    }

    fn ivf_params() -> IvfBuildParams {
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(centroids()), DIM as i32)
                .unwrap();
        IvfBuildParams::try_with_centroids(PARTITIONS, Arc::new(centroids)).unwrap()
    }

    fn rq_params(layered: bool, rotation_type: RQRotationType) -> RQBuildParams {
        RQBuildParams::with_rotation_type(7, rotation_type).with_layered(layered)
    }

    async fn create_index(dataset: &mut Dataset, rq: RQBuildParams) {
        let params = VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf_params(), rq);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
    }

    /// A column-layout dataset at `uri` with a native or layered index.
    async fn column_dataset(
        uri: &str,
        layered: bool,
        rotation_type: RQRotationType,
        version: LanceFileVersion,
    ) -> Dataset {
        let mut dataset = write_dataset(uri, version, ROWS).await;
        create_index(&mut dataset, rq_params(layered, rotation_type)).await;
        dataset
    }

    /// Every file under `dir` with its size, sorted.
    fn list_files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, u64)> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                files.extend(list_files(&entry.path()));
            } else {
                files.push((entry.path(), entry.metadata().unwrap().len()));
            }
        }
        files.sort();
        files
    }

    fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    /// A full copy of the dataset at `from`, index directories included, at
    /// `to`, converted in place to plane rows.
    async fn converted_copy(
        from: &str,
        to: &str,
        options: FileWriterOptions,
    ) -> (Dataset, RowLayoutRewrite) {
        converted_copy_in_batches(from, to, options, None).await
    }

    /// [`converted_copy`], writing at most `batch_rows` rows per batch.
    async fn converted_copy_in_batches(
        from: &str,
        to: &str,
        options: FileWriterOptions,
        batch_rows: Option<usize>,
    ) -> (Dataset, RowLayoutRewrite) {
        copy_dir(std::path::Path::new(from), std::path::Path::new(to));
        let mut dataset = Dataset::open(to).await.unwrap();
        let rewrite = rewrite_rq_row_layout_impl(
            &mut dataset,
            INDEX_NAME,
            RQRowLayout::PlaneRows,
            options,
            batch_rows,
        )
        .await
        .unwrap();
        (dataset, rewrite)
    }

    async fn index_storage(dataset: &Dataset) -> IvfQuantizationStorage<RabitQuantizer> {
        let segment = single_rq_segment(dataset, INDEX_NAME).await.unwrap();
        let path = dataset
            .indices_dir()
            .join(segment.uuid.to_string())
            .join(INDEX_AUXILIARY_FILE_NAME);
        open_rq_storage(&dataset.object_store, &path).await.unwrap()
    }

    async fn open_index(dataset: &Dataset) -> Arc<dyn VectorIndex> {
        let segment = single_rq_segment(dataset, INDEX_NAME).await.unwrap();
        dataset
            .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap()
    }

    fn query(seed: u64) -> ArrayRef {
        test_batch(1, 0, seed)["vector"]
            .as_fixed_size_list()
            .value(0)
    }

    /// Row ids and distance bits of a query, in result order.
    async fn search(
        dataset: &Dataset,
        query: &dyn Array,
        k: usize,
        nprobes: usize,
        precision: RQPrecision,
    ) -> (Vec<u64>, Vec<u32>) {
        let result = dataset
            .scan()
            .nearest("vector", query, k)
            .unwrap()
            .nprobes(nprobes)
            .rq_precision(precision)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        (
            result[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec(),
            result[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|dist| dist.to_bits())
                .collect(),
        )
    }

    async fn assert_same_results(a: &Dataset, b: &Dataset, layered: bool) {
        let precisions: &[RQPrecision] = if layered {
            &[RQPrecision::Full, RQPrecision::High, RQPrecision::Sign]
        } else {
            &[RQPrecision::Full]
        };
        for seed in [11, 12, 13] {
            let query = query(seed);
            for (k, nprobes) in [(1, 1), (10, 2), (100, PARTITIONS), (ROWS + 1, PARTITIONS)] {
                for &precision in precisions {
                    assert_eq!(
                        search(a, query.as_ref(), k, nprobes, precision).await,
                        search(b, query.as_ref(), k, nprobes, precision).await,
                        "seed={seed} k={k} nprobes={nprobes} precision={precision:?}"
                    );
                }
            }
        }
    }

    fn assert_same_batch(a: &RecordBatch, b: &RecordBatch, context: &str) {
        assert_eq!(a.schema().fields(), b.schema().fields(), "{context}");
        assert_eq!(a.columns(), b.columns(), "{context}");
    }

    /// The conversion runs in place on a copy that still holds the old index
    /// directory: it commits a new segment at version 3 whose files it
    /// records, every query answers as before, earlier versions stay readable
    /// until the caller removes the old directory, and the source is not
    /// written. Only `row_layout` is added to the metadata, as text.
    #[rstest]
    #[case::native_fast(false, RQRotationType::Fast)]
    #[case::native_matrix(false, RQRotationType::Matrix)]
    #[case::layered_fast(true, RQRotationType::Fast)]
    #[case::layered_matrix(true, RQRotationType::Matrix)]
    #[tokio::test]
    async fn test_rewrite_rq_row_layout_in_place(
        #[case] layered: bool,
        #[case] rotation_type: RQRotationType,
    ) {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let source = column_dataset(
            source_dir.as_str(),
            layered,
            rotation_type,
            LanceFileVersion::V2_2,
        )
        .await;
        let source_segment = single_rq_segment(&source, INDEX_NAME).await.unwrap();
        let source_files = list_files(std::path::Path::new(source_dir.as_str()));
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (converted, rewrite) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;

        assert_eq!(rewrite.old_uuid, source_segment.uuid.to_string());
        assert_ne!(rewrite.new_uuid, rewrite.old_uuid);
        assert_eq!(rewrite.source_layout, "columns");
        assert_eq!(rewrite.target_layout, "plane_rows");
        assert_eq!(rewrite.index_version, 3);
        assert_eq!(rewrite.num_rows, ROWS as u64);
        assert_eq!(rewrite.num_partitions, PARTITIONS);
        let expected_buffers = match rotation_type {
            RQRotationType::Fast => 1,
            RQRotationType::Matrix => 2,
        };
        assert_eq!(rewrite.global_buffers, expected_buffers);
        assert_eq!(rewrite.metadata.len(), 1);
        let patch = &rewrite.metadata[0];
        assert_eq!(patch.key, STORAGE_METADATA_KEY);
        let close = patch.source.rfind('}').unwrap();
        assert_eq!(
            patch.converted,
            format!(
                r#"{},\"row_layout\":\"plane_rows\"{}"#,
                &patch.source[..close],
                &patch.source[close..]
            )
        );

        // The committed segment: the new uuid alone, at version 3, with the
        // layout in its details and its files' sizes recorded.
        let segments = converted.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(segments.len(), 1);
        let segment = &segments[0];
        assert_eq!(segment.uuid.to_string(), rewrite.new_uuid);
        assert_eq!(segment.index_version, 3);
        assert_eq!(segment.fragment_bitmap, source_segment.fragment_bitmap);
        assert_eq!(segment.dataset_version, source_segment.dataset_version);
        let details = rq_details(segment).unwrap();
        let Some(Compression::Rq(rq)) = details.compression else {
            unreachable!()
        };
        assert_eq!(rq.row_layout(), rabit_quantization::RowLayout::PlaneRows);
        let new_dir = converted.indices_dir().join(rewrite.new_uuid.as_str());
        for file in segment.files.as_ref().unwrap() {
            let size = converted
                .object_store
                .size(&new_dir.clone().join(file.path.as_str()))
                .await
                .unwrap();
            assert_eq!(size, file.size_bytes, "{}", file.path);
        }
        let index = open_index(&converted).await;
        let index = index.as_any().downcast_ref::<IvfRq>().unwrap();
        assert_eq!(index.row_layout(), RQRowLayout::PlaneRows);
        assert!(!index.resident_columns_enabled());
        assert_eq!(index.resident_columns_bytes(), 0);

        assert_same_results(&source, &converted, layered).await;
        let verification = verify_rq_row_layout(&source, &converted, INDEX_NAME)
            .await
            .unwrap();
        assert!(verification.is_ok(), "{verification:#?}");
        assert_eq!(verification.deep_size_mismatches, 0, "{verification:#?}");
        assert!(verification.reads >= PARTITIONS);
        assert_eq!(verification.source_layout, "columns");
        assert_eq!(verification.converted_layout, "plane_rows");
        assert_eq!(verification.global_buffer_digests.len(), expected_buffers);
        assert_eq!(
            verification.rotate_mat_position,
            match rotation_type {
                RQRotationType::Fast => None,
                RQRotationType::Matrix => Some(2),
            }
        );
        assert_eq!(verification.pages.straddling_partitions, 0);
        assert!(verification.pages.max_partition_bytes > 0);
        assert_eq!(verification.pages.partitions_over_16mib, 0);

        // The version before the conversion still reads the old directory.
        let previous = converted
            .checkout_version(rewrite.dataset_version - 1)
            .await
            .unwrap();
        assert_eq!(
            single_rq_segment(&previous, INDEX_NAME).await.unwrap().uuid,
            source_segment.uuid
        );
        assert_same_results(&source, &previous, layered).await;
        // Removing it leaves the converted index whole.
        converted
            .object_store
            .remove_dir_all(converted.indices_dir().join(rewrite.old_uuid.as_str()))
            .await
            .unwrap();
        let reopened = Dataset::open(&copy_path).await.unwrap();
        assert_same_results(&source, &reopened, layered).await;

        // The source was not written.
        assert_eq!(
            list_files(std::path::Path::new(source_dir.as_str())),
            source_files
        );
        assert_eq!(
            single_rq_segment(&source, INDEX_NAME).await.unwrap().uuid,
            source_segment.uuid
        );
    }

    /// Converting back to columns restores the source's auxiliary file:
    /// its fields, every value, its metadata byte for byte, its global
    /// buffers, and version 2.
    #[rstest]
    #[tokio::test]
    async fn test_rewrite_rq_row_layout_round_trips(#[values(false, true)] layered: bool) {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let source = column_dataset(
            source_dir.as_str(),
            layered,
            RQRotationType::Matrix,
            LanceFileVersion::V2_2,
        )
        .await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (mut converted, _) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;
        let back = rewrite_rq_row_layout(&mut converted, INDEX_NAME, RQRowLayout::Columns)
            .await
            .unwrap();
        assert_eq!(back.index_version, 2);
        assert_eq!(back.target_layout, "columns");
        assert_eq!(
            converted.load_indices_by_name(INDEX_NAME).await.unwrap()[0].index_version,
            2
        );

        let original = index_storage(&source).await;
        let round_trip = index_storage(&converted).await;
        assert_eq!(round_trip.row_layout(), RQRowLayout::Columns);
        let schema = |storage: &IvfQuantizationStorage<RabitQuantizer>| {
            ArrowSchema::from(storage.reader().schema().as_ref())
        };
        assert_eq!(schema(&original), schema(&round_trip));
        assert_eq!(
            original.reader().metadata().file_buffers.len(),
            round_trip.reader().metadata().file_buffers.len()
        );
        for position in 1..original.reader().metadata().file_buffers.len() as u32 {
            assert_eq!(
                original
                    .reader()
                    .read_global_buffer(position)
                    .await
                    .unwrap(),
                round_trip
                    .reader()
                    .read_global_buffer(position)
                    .await
                    .unwrap()
            );
        }
        for partition in 0..PARTITIONS {
            assert_same_batch(
                &original
                    .read_partition_batch(partition, None)
                    .await
                    .unwrap(),
                &round_trip
                    .read_partition_batch(partition, None)
                    .await
                    .unwrap(),
                &format!("partition {partition}"),
            );
        }
        assert_same_results(&source, &converted, layered).await;
    }

    #[tokio::test]
    async fn test_rewrite_rq_row_layout_rejects_what_it_cannot_convert() {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let mut source = column_dataset(
            source_dir.as_str(),
            true,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (mut converted, rewrite) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;

        // Already in the target layout.
        let err = rewrite_rq_row_layout(&mut converted, INDEX_NAME, RQRowLayout::PlaneRows)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("already in the plane_rows layout"),
            "{err}"
        );
        let err = rewrite_rq_row_layout(&mut source, INDEX_NAME, RQRowLayout::Columns)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("already in the columns layout"),
            "{err}"
        );

        // A dataset copied without the index's directory.
        converted
            .object_store
            .remove_dir_all(converted.indices_dir().join(rewrite.new_uuid.as_str()))
            .await
            .unwrap();
        let err = rewrite_rq_row_layout(&mut converted, INDEX_NAME, RQRowLayout::Columns)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("is missing"), "{err}");

        // An index of several segments.
        let batch = test_batch(200, ROWS as u64, 2);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        source.append(reader, None).await.unwrap();
        source
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(
            source.load_indices_by_name(INDEX_NAME).await.unwrap().len(),
            2
        );
        let err = rewrite_rq_row_layout(&mut source, INDEX_NAME, RQRowLayout::PlaneRows)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("2 segments"), "{err}");

        // Not an IVF_RQ index.
        let other_dir = TempStrDir::default();
        let mut other = write_dataset(other_dir.as_str(), LanceFileVersion::V2_2, ROWS).await;
        other
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params()),
                true,
            )
            .await
            .unwrap();
        let err = rewrite_rq_row_layout(&mut other, INDEX_NAME, RQRowLayout::PlaneRows)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not an IVF_RQ index"), "{err}");
    }

    /// Text patches add the member last, or remove it with its separator,
    /// and keep every other byte.
    #[test]
    fn test_patch_object_text() {
        assert_eq!(
            patch_object_text(r#"{"a":1}"#, PLANE_ROWS_MEMBER, true).unwrap(),
            r#"{"a":1,"row_layout":"plane_rows"}"#
        );
        assert_eq!(
            patch_object_text("{ }", PLANE_ROWS_MEMBER, true).unwrap(),
            r#"{ "row_layout":"plane_rows"}"#
        );
        for (text, expected) in [
            (r#"{"a":1,"row_layout":"plane_rows"}"#, r#"{"a":1}"#),
            (r#"{"row_layout":"plane_rows","a":1}"#, r#"{"a":1}"#),
            (
                r#"{"a":1,"row_layout":"plane_rows","b":2}"#,
                r#"{"a":1,"b":2}"#,
            ),
            (r#"{"row_layout":"plane_rows"}"#, "{}"),
        ] {
            assert_eq!(
                patch_object_text(text, PLANE_ROWS_MEMBER, false).unwrap(),
                expected
            );
        }
        assert!(patch_object_text(r#"{"a":1}"#, PLANE_ROWS_MEMBER, false).is_err());
        assert!(patch_object_text("[1]", PLANE_ROWS_MEMBER, true).is_err());

        let storage = serde_json::to_string(&vec![
            r#"{"code_dim":64,"num_bits":7,"packed":true}"#.to_string(),
        ])
        .unwrap();
        let metadata = HashMap::from([(STORAGE_METADATA_KEY.to_string(), storage.clone())]);
        let patches = patch_schema_metadata(&metadata, RQRowLayout::PlaneRows).unwrap();
        let entries: Vec<String> = serde_json::from_str(&patches[0].converted).unwrap();
        let read: RabitQuantizationMetadata = serde_json::from_str(&entries[0]).unwrap();
        assert_eq!(read.row_layout, RQRowLayout::PlaneRows);
        // Metadata that already declares a layout is not patched again.
        let declared = HashMap::from([(
            STORAGE_METADATA_KEY.to_string(),
            patches[0].converted.clone(),
        )]);
        assert!(patch_schema_metadata(&declared, RQRowLayout::PlaneRows).is_err());
        let back = patch_schema_metadata(&declared, RQRowLayout::Columns).unwrap();
        assert_eq!(back[0].converted, storage);
    }

    /// Every read of a plane-row file returns what the same read of its
    /// column-layout source returns: whole partitions, whole planes under
    /// both bounds placements, selected rows at every coalesce gap, and no
    /// rows; on 2.0 and 2.2 files, with pages that hold whole partitions or
    /// end inside them.
    #[rstest]
    #[tokio::test]
    async fn test_plane_rows_reads_match_column_reads(
        #[values(false, true)] layered: bool,
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_2)] version: LanceFileVersion,
        #[values(false, true)] small_pages: bool,
    ) {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let source =
            column_dataset(source_dir.as_str(), layered, RQRotationType::Fast, version).await;
        let options = if small_pages {
            FileWriterOptions {
                data_cache_bytes: Some(SMALL_PAGE_BYTES),
                max_page_bytes: Some(SMALL_PAGE_BYTES),
                ..Default::default()
            }
        } else {
            FileWriterOptions::default()
        };
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (converted, _) = converted_copy_in_batches(
            source_dir.as_str(),
            &copy_path,
            options,
            small_pages.then_some(SMALL_BATCH_ROWS),
        )
        .await;
        let verification = verify_rq_row_layout(&source, &converted, INDEX_NAME)
            .await
            .unwrap();
        // Whole reads and cache entries weigh the same in a cache; a sparse
        // column read of a 2.0 file can keep more buffer than its values.
        assert!(verification.is_ok(), "{verification:#?}");
        assert_eq!(verification.deep_size_mismatches, 0, "{verification:#?}");
        for file_version in [
            &verification.source_file_version,
            &verification.converted_file_version,
        ] {
            assert_eq!(file_version, version.resolve().to_manifest_string());
        }
        assert!(
            verification.sparse_converted_deep_bytes <= verification.sparse_source_deep_bytes,
            "{verification:#?}"
        );
        if version != LanceFileVersion::V2_0 {
            assert_eq!(
                verification.sparse_deep_size_mismatches, 0,
                "{verification:#?}"
            );
        }
        if small_pages {
            assert!(
                verification.pages.straddling_partitions > 0,
                "{verification:#?}"
            );
        }

        let mut rng = StdRng::seed_from_u64(u64::from(layered) + 2 * u64::from(small_pages));
        for sign_bounds in [SignBounds::Lazy, SignBounds::Eager] {
            let columns = index_storage(&source).await.with_sign_bounds(sign_bounds);
            let rows = index_storage(&converted)
                .await
                .with_sign_bounds(sign_bounds);
            assert_eq!(rows.row_layout(), RQRowLayout::PlaneRows);
            assert_eq!(
                rows.logical_schema().unwrap().fields(),
                columns.logical_schema().unwrap().fields()
            );
            for partition in 0..PARTITIONS {
                let context = format!("{sign_bounds:?} partition {partition}");
                assert_same_batch(
                    &columns.read_partition_batch(partition, None).await.unwrap(),
                    &rows.read_partition_batch(partition, None).await.unwrap(),
                    &context,
                );
                let a = columns.load_partition(partition, None).await.unwrap();
                let b = rows.load_partition(partition, None).await.unwrap();
                assert_eq!(a.len(), b.len(), "{context}");
                assert_eq!(
                    a.row_ids().collect::<Vec<_>>(),
                    b.row_ids().collect::<Vec<_>>(),
                    "{context}"
                );
                if !layered {
                    continue;
                }
                let size = columns.partition_size(partition) as u32;
                let some: Vec<u32> = (0..size).filter(|_| rng.random_bool(0.2)).collect();
                let planes: &[u8] = match sign_bounds {
                    SignBounds::Lazy => &[0, 1, 2, SIGN_BOUNDS_PLANE],
                    SignBounds::Eager => &[0, 1, 2],
                };
                for &plane in planes {
                    for selected in [None, Some(Vec::new()), Some(some.clone())] {
                        let context = format!(
                            "{context} plane {plane} rows {:?}",
                            selected.as_ref().map(Vec::len)
                        );
                        assert_same_batch(
                            &columns
                                .read_plane(partition, plane, selected.clone(), None)
                                .await
                                .unwrap(),
                            &rows
                                .read_plane(partition, plane, selected.clone(), None)
                                .await
                                .unwrap(),
                            &context,
                        );
                    }
                    for gap in [0, 64 * 1024, 1024 * 1024, u64::MAX] {
                        assert_same_batch(
                            &columns
                                .read_plane_with_coalesce_gap(
                                    partition,
                                    plane,
                                    Some(some.clone()),
                                    None,
                                    Some(gap),
                                )
                                .await
                                .unwrap(),
                            &rows
                                .read_plane_with_coalesce_gap(
                                    partition,
                                    plane,
                                    Some(some.clone()),
                                    None,
                                    Some(gap),
                                )
                                .await
                                .unwrap(),
                            &format!("{context} plane {plane} gap {gap}"),
                        );
                    }
                }
            }
        }

        // On 2.2 every packed page is full-zip: rows times the stride, flat.
        if version == LanceFileVersion::V2_2 {
            let storage = index_storage(&converted).await;
            let spec = storage.plane_rows().unwrap().unwrap().clone();
            let reader = storage.reader();
            for column in spec.packed_columns() {
                let projection = lance_file::versions::reader_projection_from_column_names(
                    reader.metadata().version(),
                    reader.schema(),
                    &[column.name()],
                )
                .unwrap();
                let info = &reader.metadata().column_infos[projection.column_indices[0] as usize];
                assert!(!info.page_infos.is_empty());
                for page in info.page_infos.iter() {
                    let PageEncoding::Structural(layout) = &page.encoding else {
                        panic!("{} page is not structural", column.name());
                    };
                    assert!(
                        matches!(layout.layout, Some(Layout::FullZipLayout(_))),
                        "{} page is {layout:?}",
                        column.name()
                    );
                    let bytes: u64 = page
                        .buffer_offsets_and_sizes
                        .iter()
                        .map(|(_, size)| size)
                        .sum();
                    assert_eq!(
                        bytes,
                        page.num_rows * column.stride() as u64,
                        "{}",
                        column.name()
                    );
                }
            }
        }
    }

    /// A copy of the plane-row dataset at `from` at `to`, converted in place
    /// back to the column layout with `options`, at most `batch_rows` rows
    /// per batch.
    async fn column_copy_in_batches(
        from: &str,
        to: &str,
        options: FileWriterOptions,
        batch_rows: Option<usize>,
    ) -> Dataset {
        copy_dir(std::path::Path::new(from), std::path::Path::new(to));
        let mut dataset = Dataset::open(to).await.unwrap();
        rewrite_rq_row_layout_impl(
            &mut dataset,
            INDEX_NAME,
            RQRowLayout::Columns,
            options,
            batch_rows,
        )
        .await
        .unwrap();
        dataset
    }

    /// Partitions whose rows of `column` span more than one of its pages.
    fn straddling_partitions(
        storage: &IvfQuantizationStorage<RabitQuantizer>,
        column: &str,
    ) -> usize {
        let reader = storage.reader();
        let projection = lance_file::versions::reader_projection_from_column_names(
            reader.metadata().version(),
            reader.schema(),
            &[column],
        )
        .unwrap();
        let info = &reader.metadata().column_infos[projection.column_indices[0] as usize];
        let mut page_starts = Vec::new();
        let mut next = 0;
        for page in info.page_infos.iter() {
            next += page.num_rows;
            page_starts.push(next);
        }
        page_starts.pop();
        (0..storage.num_partitions())
            .filter(|&partition| {
                let range = storage.ivf().row_range(partition);
                page_starts
                    .iter()
                    .any(|&start| start > range.start as u64 && start < range.end as u64)
            })
            .count()
    }

    /// Twins weigh the same in a cache even where the column layout's reads
    /// span pages: every whole read, partition storage (read alone or cut
    /// from a prewarm's read) and code-only entry of a column file whose
    /// pages end inside partitions takes the memory the plane-row twin's
    /// does, on 2.0 and 2.2 files and in either direction of a conversion.
    #[rstest]
    #[tokio::test]
    async fn test_column_entries_weigh_as_plane_rows(
        #[values(false, true)] layered: bool,
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_2)] version: LanceFileVersion,
    ) {
        let source_dir = TempStrDir::default();
        let rows_dir = TempStrDir::default();
        let columns_dir = TempStrDir::default();
        column_dataset(source_dir.as_str(), layered, RQRotationType::Fast, version).await;
        let rows_path = format!("{}/rows", rows_dir.as_str());
        let (rows, _) = converted_copy(
            source_dir.as_str(),
            &rows_path,
            FileWriterOptions::default(),
        )
        .await;
        let columns = column_copy_in_batches(
            &rows_path,
            &format!("{}/columns", columns_dir.as_str()),
            FileWriterOptions {
                data_cache_bytes: Some(SMALL_PAGE_BYTES),
                max_page_bytes: Some(SMALL_PAGE_BYTES),
                ..Default::default()
            },
            Some(SMALL_BATCH_ROWS),
        )
        .await;
        let storage = index_storage(&columns).await;
        assert_eq!(storage.row_layout(), RQRowLayout::Columns);
        assert!(straddling_partitions(&storage, RABIT_CODE_COLUMN) > 0);

        let entry_shapes: &[&str] = if layered {
            &[
                deep_size_shapes::PARTITION_BATCH,
                deep_size_shapes::PLANE,
                deep_size_shapes::PARTITION_STORAGE,
                deep_size_shapes::PREWARM_PARTITION_STORAGE,
                deep_size_shapes::CODE_ONLY_ENTRY,
            ]
        } else {
            &[
                deep_size_shapes::PARTITION_BATCH,
                deep_size_shapes::PARTITION_STORAGE,
                deep_size_shapes::PREWARM_PARTITION_STORAGE,
                deep_size_shapes::CODE_ONLY_ENTRY,
                deep_size_shapes::PREWARM_CODE_ONLY_ENTRY,
            ]
        };
        for (source, converted) in [(&columns, &rows), (&rows, &columns)] {
            let verification = verify_rq_row_layout(source, converted, INDEX_NAME)
                .await
                .unwrap();
            assert!(verification.is_ok(), "{verification:#?}");
            assert_eq!(verification.deep_size_mismatches, 0, "{verification:#?}");
            assert_eq!(
                verification.source_deep_bytes, verification.converted_deep_bytes,
                "{verification:#?}"
            );
            for shape in entry_shapes {
                let sizes = &verification.deep_sizes[*shape];
                assert!(sizes.count >= PARTITIONS, "{shape}: {sizes:?}");
                assert_eq!(sizes.mismatches, 0, "{shape}: {sizes:?}");
                assert_eq!(
                    sizes.source_bytes, sizes.converted_bytes,
                    "{shape}: {sizes:?}"
                );
            }
            assert_eq!(
                verification
                    .deep_sizes
                    .contains_key(deep_size_shapes::PLANE_SPARSE),
                layered
            );
        }
    }

    /// A plane-row file reads a native partition, or a whole plane of a
    /// layered one, in one request, an eager sign plane (sign and bounds) in
    /// two, and selected rows in one request per run of adjacent rows; no
    /// read loads a resident store, whatever residency asks.
    #[rstest]
    #[tokio::test]
    async fn test_plane_rows_cut_origin_requests(#[values(false, true)] layered: bool) {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        column_dataset(
            source_dir.as_str(),
            layered,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (converted, _) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;
        let requests = |stats: &IoStats| stats.snapshot().iops;

        let lazy = index_storage(&converted)
            .await
            .with_resident_columns_enabled(true);
        assert!(!lazy.resident_columns_enabled());
        assert_eq!(lazy.resident_columns_bytes(), 0);
        let eager = index_storage(&converted)
            .await
            .with_sign_bounds(SignBounds::Eager);
        let lazy = lazy.with_sign_bounds(SignBounds::Lazy);
        let mut rng = StdRng::seed_from_u64(7);
        for partition in 0..PARTITIONS {
            let size = lazy.partition_size(partition) as u32;
            let stats = IoStats::new();
            lazy.read_partition_batch(partition, Some(&stats))
                .await
                .unwrap();
            // One request per packed column: one for a native partition,
            // at most four for a layered one, whose plane pages can be close
            // enough in the file to share a request.
            let columns = if layered { 4 } else { 1 };
            if size == 0 {
                assert_eq!(requests(&stats), 0, "partition {partition}");
            } else if layered {
                assert!(
                    (1..=columns).contains(&requests(&stats)),
                    "partition {partition}: {} requests",
                    requests(&stats)
                );
            } else {
                assert_eq!(requests(&stats), 1, "partition {partition}");
            }
            if !layered {
                continue;
            }
            for (storage, plane, expected) in [
                (&lazy, 0, 1),
                (&eager, 0, 2),
                (&lazy, 1, 1),
                (&lazy, 2, 1),
                (&lazy, SIGN_BOUNDS_PLANE, 1),
            ] {
                let stats = IoStats::new();
                storage
                    .read_plane(partition, plane, None, Some(stats.clone()))
                    .await
                    .unwrap();
                let expected = if size == 0 { 0 } else { expected };
                assert_eq!(
                    requests(&stats),
                    expected,
                    "partition {partition} plane {plane}"
                );
            }
            let selected: Vec<u32> = (0..size).filter(|_| rng.random_bool(0.3)).collect();
            let runs = selected
                .iter()
                .enumerate()
                .filter(|(index, row)| *index == 0 || selected[index - 1] + 1 != **row)
                .count() as u64;
            for plane in [1, 2] {
                let stats = IoStats::new();
                lazy.read_plane_with_coalesce_gap(
                    partition,
                    plane,
                    Some(selected.clone()),
                    Some(stats.clone()),
                    Some(0),
                )
                .await
                .unwrap();
                assert_eq!(
                    requests(&stats),
                    runs,
                    "partition {partition} plane {plane}"
                );
            }
        }
    }

    /// Building with plane-row params writes the values a column build
    /// writes, packed: the same rows of every partition, at version 3.
    #[rstest]
    #[tokio::test]
    async fn test_plane_rows_build_matches_conversion(#[values(false, true)] layered: bool) {
        let columns_dir = TempStrDir::default();
        let rows_dir = TempStrDir::default();
        let columns = column_dataset(
            columns_dir.as_str(),
            layered,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let column_storage = index_storage(&columns).await;
        let mut rows = write_dataset(rows_dir.as_str(), LanceFileVersion::V2_2, ROWS).await;
        let mut rq =
            rq_params(layered, RQRotationType::Fast).with_row_layout(RQRowLayout::PlaneRows);
        rq.rotation = Some(column_storage.metadata().clone());
        create_index(&mut rows, rq).await;
        assert_eq!(
            rows.load_indices_by_name(INDEX_NAME).await.unwrap()[0].index_version,
            3
        );
        let row_storage = index_storage(&rows).await;
        assert_eq!(row_storage.row_layout(), RQRowLayout::PlaneRows);
        assert!(row_storage.metadata().packed);
        assert_eq!(row_storage.ivf().lengths, column_storage.ivf().lengths);

        // Rows in id order, sign codes as each row's own: the partitions'
        // row order can differ between builds.
        let spec = PlaneRowsSpec::try_new(row_storage.metadata()).unwrap();
        let rows_by_id = |batch: RecordBatch| -> Vec<(u64, Vec<u8>)> {
            let codes = Arc::new(unpack_codes(batch[RABIT_CODE_COLUMN].as_fixed_size_list()));
            let batch = batch
                .replace_column_by_name(RABIT_CODE_COLUMN, codes)
                .unwrap();
            let packed = spec.pack(&batch).unwrap();
            let mut rows: Vec<(u64, Vec<u8>)> = Vec::new();
            let ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            for row in 0..batch.num_rows() {
                let mut bytes = Vec::new();
                for column in spec.packed_columns() {
                    let list = packed[column.name()].as_fixed_size_list();
                    bytes.extend_from_slice(list.value(row).as_primitive::<UInt8Type>().values());
                }
                rows.push((ids.value(row), bytes));
            }
            rows.sort();
            rows
        };
        for partition in 0..PARTITIONS {
            assert_eq!(
                rows_by_id(
                    column_storage
                        .read_partition_batch(partition, None)
                        .await
                        .unwrap()
                ),
                rows_by_id(
                    row_storage
                        .read_partition_batch(partition, None)
                        .await
                        .unwrap()
                ),
                "partition {partition}"
            );
        }
    }

    /// The ex planes a plane-row file reads serialize to the same cache
    /// entries as the column layout's, so both share one NVMe format.
    #[tokio::test]
    async fn test_plane_rows_ex_plane_entries_match() {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let source = column_dataset(
            source_dir.as_str(),
            true,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (converted, _) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        let columns = index_storage(&source).await;
        let rows = index_storage(&converted).await;
        let entry = |batch: RecordBatch| {
            let mut bytes = Vec::new();
            codec
                .serialize(
                    &(Arc::new(PlaneBatch(batch)) as Arc<dyn std::any::Any + Send + Sync>),
                    &mut bytes,
                )
                .unwrap();
            Bytes::from(bytes)
        };
        for partition in 0..PARTITIONS {
            for plane in [1, 2] {
                assert_eq!(
                    entry(
                        columns
                            .read_plane(partition, plane, None, None)
                            .await
                            .unwrap()
                    ),
                    entry(rows.read_plane(partition, plane, None, None).await.unwrap()),
                    "partition {partition} plane {plane}"
                );
            }
        }
    }

    /// A file whose schema does not match the layout its metadata declares
    /// is an invalid index when it opens.
    #[tokio::test]
    async fn test_plane_rows_open_rejects_layout_mismatch() {
        let source_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let source = column_dataset(
            source_dir.as_str(),
            true,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (converted, _) = converted_copy(
            source_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;
        let columns = index_storage(&source).await;
        let rows = index_storage(&converted).await;
        let store = converted.object_store.clone();
        let storage_metadata = |storage: &IvfQuantizationStorage<RabitQuantizer>| {
            storage
                .reader()
                .schema()
                .metadata
                .get(STORAGE_METADATA_KEY)
                .unwrap()
                .clone()
        };
        // Each file's columns under the other's storage metadata.
        for (data, metadata, case) in [
            (
                &rows,
                storage_metadata(&columns),
                "plane rows declared as columns",
            ),
            (
                &columns,
                storage_metadata(&rows),
                "columns declared as plane rows",
            ),
        ] {
            let reader = data.reader();
            let mut schema_metadata = reader.schema().metadata.clone();
            schema_metadata.insert(STORAGE_METADATA_KEY.to_string(), metadata);
            let schema = ArrowSchema::new_with_metadata(
                ArrowSchema::from(reader.schema().as_ref()).fields().clone(),
                schema_metadata,
            );
            let path = Path::from(format!("{copy_path}/mismatch/{}", case.replace(' ', "_")));
            let mut writer = file_versions::create_writer(
                reader.metadata().version(),
                store.create(&path).await.unwrap(),
                lance_core::datatypes::Schema::try_from(&schema).unwrap(),
                FileWriterOptions::default(),
            )
            .unwrap();
            for position in 1..reader.metadata().file_buffers.len() as u32 {
                writer
                    .add_global_buffer(reader.read_global_buffer(position).await.unwrap())
                    .await
                    .unwrap();
            }
            let batches = reader
                .read_stream(
                    ReadBatchParams::RangeFull,
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            for batch in &batches {
                writer.write_batch(batch).await.unwrap();
            }
            writer.finish().await.unwrap();
            let err = open_rq_storage(&store, &path).await.unwrap_err();
            assert!(matches!(err, Error::Index { .. }), "{case}: {err:?}");
        }
    }

    /// Remapping a plane-row index for compaction, at once or through a
    /// fragment reuse index, keeps plane rows and version 3, and answers as
    /// its column-layout twin remapped the same way.
    #[rstest]
    #[tokio::test]
    async fn test_plane_rows_survive_remap(#[values(false, true)] defer_index_remap: bool) {
        let columns_dir = TempStrDir::default();
        let copy_dir_ = TempStrDir::default();
        let mut columns =
            write_dataset(columns_dir.as_str(), LanceFileVersion::V2_2, ROWS / 4).await;
        create_index(&mut columns, rq_params(true, RQRotationType::Fast)).await;
        let copy_path = format!("{}/copy", copy_dir_.as_str());
        let (mut rows, _) = converted_copy(
            columns_dir.as_str(),
            &copy_path,
            FileWriterOptions::default(),
        )
        .await;
        let options = CompactionOptions {
            defer_index_remap,
            ..Default::default()
        };
        for dataset in [&mut columns, &mut rows] {
            dataset.delete("id % 3 = 0").await.unwrap();
            compact_files(dataset, options.clone(), None).await.unwrap();
        }
        let segment = single_rq_segment(&rows, INDEX_NAME).await.unwrap();
        assert_eq!(segment.index_version, 3);
        let index = open_index(&rows).await;
        let index = index.as_any().downcast_ref::<IvfRq>().unwrap();
        assert_eq!(index.row_layout(), RQRowLayout::PlaneRows);
        assert_eq!(
            single_rq_segment(&columns, INDEX_NAME)
                .await
                .unwrap()
                .index_version,
            2
        );
        assert_same_results(&columns, &rows, true).await;
    }

    /// Appending to an index keeps its layout: a delta segment of a
    /// plane-row index is plane rows at version 3, of a column-layout index
    /// columns at version 2, and a merge through optimize writes the
    /// reference segment's layout and version. A plane-row index answers as
    /// its column-layout twin, built from the same model.
    #[rstest]
    #[tokio::test]
    async fn test_optimize_append_keeps_row_layout_and_version(
        #[values(false, true)] layered: bool,
    ) {
        let columns_dir = TempStrDir::default();
        let rows_dir = TempStrDir::default();
        let mut columns = column_dataset(
            columns_dir.as_str(),
            layered,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let mut rq =
            rq_params(layered, RQRotationType::Fast).with_row_layout(RQRowLayout::PlaneRows);
        rq.rotation = Some(index_storage(&columns).await.metadata().clone());
        let mut rows = write_dataset(rows_dir.as_str(), LanceFileVersion::V2_2, ROWS).await;
        create_index(&mut rows, rq).await;

        let batch = test_batch(400, ROWS as u64, 3);
        for (dataset, layout, version) in [
            (&mut columns, RQRowLayout::Columns, 2),
            (&mut rows, RQRowLayout::PlaneRows, 3),
        ] {
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            dataset.append(reader, None).await.unwrap();
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
            let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
            assert_eq!(segments.len(), 2, "{layout}");
            for segment in &segments {
                assert_eq!(segment.index_version, version, "{layout}");
                let index = dataset
                    .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                let index = index.as_any().downcast_ref::<IvfRq>().unwrap();
                assert_eq!(index.row_layout(), layout);
            }
        }
        assert_same_results(&columns, &rows, layered).await;

        for (dataset, layout, version) in [
            (&mut columns, RQRowLayout::Columns, 2),
            (&mut rows, RQRowLayout::PlaneRows, 3),
        ] {
            dataset
                .optimize_indices(&OptimizeOptions::merge(2))
                .await
                .unwrap();
            let segment = single_rq_segment(dataset, INDEX_NAME).await.unwrap();
            assert_eq!(segment.index_version, version, "{layout}");
            assert_eq!(index_storage(dataset).await.row_layout(), layout);
        }
        assert_same_results(&columns, &rows, layered).await;
    }

    /// A steady-state optimize that rewrites one oversized segment keeps its
    /// layout and version.
    #[tokio::test]
    async fn test_segment_rebalance_keeps_layout_and_version() {
        let dir = TempStrDir::default();
        let mut dataset = write_dataset(dir.as_str(), LanceFileVersion::V2_2, ROWS).await;
        let mut ivf = ivf_params();
        ivf.target_partition_size = Some(256);
        let params = VectorIndexParams::with_ivf_rq_params(
            DistanceType::L2,
            ivf,
            rq_params(true, RQRotationType::Fast).with_row_layout(RQRowLayout::PlaneRows),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        // A delta segment with more than 4 x 256 rows in one partition.
        let centroids = centroids();
        let mut values = Vec::with_capacity(1_500 * DIM);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..1_500 {
            values.extend(
                centroids[..DIM]
                    .iter()
                    .map(|value| value + rng.random_range(-0.05f32..0.05)),
            );
        }
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap();
        let schema = test_batch(1, 0, 0).schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(
                    ROWS as u64..ROWS as u64 + 1_500,
                )),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        dataset
            .append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let before = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(before.len(), 2);
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let after = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(after.len(), 2, "one segment was rewritten in place");
        assert_ne!(
            after.iter().map(|segment| segment.uuid).collect::<Vec<_>>(),
            before
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>(),
            "the steady-state optimize rewrote a segment"
        );
        for segment in &after {
            assert_eq!(segment.index_version, 3);
            let index = dataset
                .open_vector_index("vector", &segment.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let index = index.as_any().downcast_ref::<IvfRq>().unwrap();
            assert_eq!(index.row_layout(), RQRowLayout::PlaneRows);
        }
    }

    /// Distributed segments, built from one model on fragment groups, merge
    /// into one segment at their version in the column layout; the merger
    /// does not support plane-row segments.
    #[rstest]
    #[tokio::test]
    async fn test_merge_segments_by_row_layout(
        #[values(RQRowLayout::Columns, RQRowLayout::PlaneRows)] row_layout: RQRowLayout,
    ) {
        let model_dir = TempStrDir::default();
        let dir = TempStrDir::default();
        let model = column_dataset(
            model_dir.as_str(),
            false,
            RQRotationType::Fast,
            LanceFileVersion::V2_2,
        )
        .await;
        let mut rq = rq_params(false, RQRotationType::Fast).with_row_layout(row_layout);
        rq.rotation = Some(index_storage(&model).await.metadata().clone());
        let params = VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf_params(), rq);
        let mut dataset = write_dataset(dir.as_str(), LanceFileVersion::V2_2, ROWS / 2).await;
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        let mut segments = Vec::new();
        for fragment in &fragments {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name(INDEX_NAME.to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            assert_eq!(segment.index_version, ivf_rq_index_version(row_layout));
            segments.push(segment);
        }
        let merged = dataset.merge_existing_index_segments(segments).await;
        match row_layout {
            RQRowLayout::Columns => {
                let merged = merged.unwrap();
                assert_eq!(merged.index_version, 2);
            }
            RQRowLayout::PlaneRows => {
                let err = merged.unwrap_err();
                assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
            }
        }
    }
}
