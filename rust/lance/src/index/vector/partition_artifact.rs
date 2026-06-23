// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::mem;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::{FixedSizeListArray, RecordBatch, UInt8Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_index::vector::v3::shuffler::ShuffleReader;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_io::ReadBatchParams;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_io::traits::Writer;
use lance_io::utils::CachedFileSize;
use object_store::path::{Path, PathPart};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

const PARTITION_ARTIFACT_MANIFEST_VERSION: u32 = 1;
const PARTITION_ARTIFACT_MANIFEST_FILE_NAME: &str = "manifest.json";
const PARTITION_ARTIFACT_PARTITIONS_DIR: &str = "partitions";
const PARTITION_ARTIFACT_DEFAULT_BUCKETS: usize = 256;
const PARTITION_ARTIFACT_BUCKET_PREFIX: &str = "bucket-";
const PARTITION_ARTIFACT_FILE_VERSION: &str = "2.2";
const PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS: usize = 32 * 1024;

/// Top-level manifest for a precomputed partition artifact.
///
/// The manifest is intentionally small and JSON-encoded so an external backend
/// can materialize partition data once and Lance can reopen it later without
/// understanding any backend-specific details.
#[derive(Debug, Serialize, Deserialize)]
struct PartitionArtifactManifest {
    version: u32,
    num_partitions: usize,
    #[serde(default)]
    metadata_file: Option<String>,
    #[serde(default)]
    total_loss: Option<f64>,
    partitions: Vec<PartitionArtifactPartition>,
}

/// Describes where one logical IVF partition lives inside the artifact.
///
/// Multiple logical partitions can share the same physical file when they hash
/// to the same bucket. `ranges` records the row spans within that file that
/// belong to this partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionArtifactPartition {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    num_rows: usize,
    #[serde(default)]
    ranges: Vec<PartitionArtifactRange>,
}

/// A contiguous row range for a partition inside one bucket file.
///
/// The builder sorts each finalized bucket by partition id, so a partition is
/// usually represented by a single range. The type still allows multiple runs
/// so the reader does not depend on that implementation detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionArtifactRange {
    offset: u64,
    num_rows: u64,
}

/// In-memory staging buffer for one bucket before it is flushed to disk.
///
/// Batches arrive grouped arbitrarily by the backend. The builder first
/// appends rows into per-bucket buffers so it can write larger sequential runs
/// to temporary files instead of issuing tiny file writes.
#[derive(Default, Debug)]
struct BucketBuffer {
    row_ids: Vec<u64>,
    partition_ids: Vec<u32>,
    pq_values: Vec<u8>,
}

impl BucketBuffer {
    /// Number of staged rows currently buffered for this bucket.
    fn len(&self) -> usize {
        self.row_ids.len()
    }

    /// Whether the bucket currently has any staged rows.
    fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }
}

/// Writes partition-addressable encoded rows for a later Lance finalization.
///
/// The builder uses bucket-local buffering to keep append-time memory bounded.
/// Each flush sorts only the current in-memory bucket and appends it directly to
/// the finalized bucket file, while the manifest accumulates per-partition row
/// ranges. This keeps the writer streaming and avoids a full read/sort/rewrite
/// pass at `finish()` time.
pub struct PartitionArtifactBuilder {
    object_store: Arc<ObjectStore>,
    root_dir: Path,
    num_partitions: usize,
    num_buckets: usize,
    pq_code_width: usize,
    final_schema: Arc<ArrowSchema>,
    final_writers: Vec<Option<FileWriter>>,
    buffers: Vec<BucketBuffer>,
    partitions: Vec<PartitionArtifactPartition>,
    bucket_row_counts: Vec<u64>,
}

impl PartitionArtifactBuilder {
    /// Create a builder from a URI and optional storage options.
    ///
    /// This is the external entry point used by backends that only know an
    /// artifact URI. It resolves the object store and then delegates to the
    /// store-aware constructor.
    pub async fn try_new(
        uri: &str,
        num_partitions: usize,
        pq_code_width: usize,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = if let Some(storage_options) = storage_options {
            ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        storage_options.clone(),
                    ),
                )),
                ..Default::default()
            }
        } else {
            ObjectStoreParams::default()
        };
        let (object_store, root_dir) =
            ObjectStore::from_uri_and_params(registry, uri, &params).await?;
        Self::try_new_with_store(object_store, root_dir, num_partitions, pq_code_width)
    }

    /// Create a builder against an already-resolved object store.
    ///
    /// The builder precomputes the final schema and allocates one staging
    /// buffer per bucket. Buckets are a write-time sharding scheme: they are
    /// not visible to readers, but they keep memory usage bounded and avoid one
    /// file per partition.
    pub fn try_new_with_store(
        object_store: Arc<ObjectStore>,
        root_dir: Path,
        num_partitions: usize,
        pq_code_width: usize,
    ) -> Result<Self> {
        if num_partitions == 0 {
            return Err(Error::invalid_input(
                "partition artifact builder requires num_partitions > 0".to_string(),
            ));
        }
        if pq_code_width == 0 {
            return Err(Error::invalid_input(
                "partition artifact builder requires pq_code_width > 0".to_string(),
            ));
        }

        let num_buckets = num_partitions.clamp(1, PARTITION_ARTIFACT_DEFAULT_BUCKETS);
        let final_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    pq_code_width as i32,
                ),
                true,
            ),
        ]));

        Ok(Self {
            object_store,
            root_dir,
            num_partitions,
            num_buckets,
            pq_code_width,
            final_schema,
            final_writers: (0..num_buckets).map(|_| None).collect(),
            buffers: (0..num_buckets).map(|_| BucketBuffer::default()).collect(),
            partitions: vec![
                PartitionArtifactPartition {
                    path: None,
                    num_rows: 0,
                    ranges: Vec::new(),
                };
                num_partitions
            ],
            bucket_row_counts: vec![0; num_buckets],
        })
    }

    /// Append one encoded batch into the artifact staging area.
    ///
    /// Input batches must already contain row ids, partition ids, and PQ codes.
    /// Rows are redistributed into bucket-local in-memory buffers and flushed to
    /// temporary files once they become large enough.
    pub async fn append_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        validate_input_batch(batch, self.pq_code_width)?;

        let row_ids = batch[ROW_ID].as_primitive::<arrow::datatypes::UInt64Type>();
        let part_ids = batch[PART_ID_COLUMN].as_primitive::<arrow::datatypes::UInt32Type>();
        let pq_codes = batch[PQ_CODE_COLUMN].as_fixed_size_list();
        let pq_values = pq_codes
            .values()
            .as_primitive::<arrow::datatypes::UInt8Type>();
        let pq_values = pq_values.values().as_ref();

        for row_idx in 0..batch.num_rows() {
            let partition_id = part_ids.value(row_idx) as usize;
            if partition_id >= self.num_partitions {
                return Err(Error::invalid_input(format!(
                    "partition artifact batch contains partition id {} but num_partitions is {}",
                    partition_id, self.num_partitions
                )));
            }
            let bucket_id = partition_id % self.num_buckets;
            let buffer = &mut self.buffers[bucket_id];
            buffer.row_ids.push(row_ids.value(row_idx));
            buffer.partition_ids.push(partition_id as u32);
            let start = row_idx * self.pq_code_width;
            let end = start + self.pq_code_width;
            buffer.pq_values.extend_from_slice(&pq_values[start..end]);
            if buffer.len() >= PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS {
                self.flush_bucket(bucket_id).await?;
            }
        }
        Ok(())
    }

    /// Finalize the artifact and return the relative files that were created.
    ///
    /// Finalization only needs to flush the remaining in-memory buffers and
    /// persist the manifest because bucket files are already in their final
    /// layout.
    pub async fn finish(
        &mut self,
        metadata_file: &str,
        total_loss: Option<f64>,
    ) -> Result<Vec<String>> {
        for bucket_id in 0..self.num_buckets {
            self.flush_bucket(bucket_id).await?;
        }
        for writer in self.final_writers.iter_mut() {
            if let Some(writer) = writer.as_mut() {
                writer.finish().await?;
            }
        }

        let mut artifact_files = Vec::with_capacity(self.num_buckets + 1);
        for bucket_id in 0..self.num_buckets {
            if self.final_writers[bucket_id].is_some() {
                artifact_files.push(self.final_bucket_relative_path(bucket_id));
            }
        }

        let manifest = PartitionArtifactManifest {
            version: PARTITION_ARTIFACT_MANIFEST_VERSION,
            num_partitions: self.num_partitions,
            metadata_file: Some(metadata_file.to_string()),
            total_loss,
            partitions: self.partitions.clone(),
        };
        write_json(
            self.object_store.as_ref(),
            &path_child(&self.root_dir, PARTITION_ARTIFACT_MANIFEST_FILE_NAME),
            &manifest,
        )
        .await?;

        let mut files = vec![PARTITION_ARTIFACT_MANIFEST_FILE_NAME.to_string()];
        files.extend(artifact_files);
        Ok(files)
    }

    /// Flush the current in-memory buffer for one bucket into its finalized
    /// bucket file.
    ///
    /// Each flush sorts only the buffered rows for this bucket and appends them
    /// to the final file while recording new manifest ranges for the affected
    /// partitions.
    async fn flush_bucket(&mut self, bucket_id: usize) -> Result<()> {
        if self.buffers[bucket_id].is_empty() {
            return Ok(());
        }

        let buffer = &mut self.buffers[bucket_id];
        let row_ids = UInt64Array::from(mem::take(&mut buffer.row_ids));
        let part_ids = mem::take(&mut buffer.partition_ids);
        let pq_values = UInt8Array::from(mem::take(&mut buffer.pq_values));
        let total_rows = row_ids.len();

        let mut permutation = (0..total_rows).collect::<Vec<_>>();
        permutation.sort_unstable_by_key(|&idx| part_ids[idx]);

        let mut sorted_row_ids = Vec::with_capacity(total_rows);
        let mut sorted_partition_ids = Vec::with_capacity(total_rows);
        let mut sorted_pq_values = Vec::with_capacity(total_rows * self.pq_code_width);
        for idx in permutation {
            sorted_row_ids.push(row_ids.value(idx));
            sorted_partition_ids.push(part_ids[idx]);
            let start = idx * self.pq_code_width;
            let end = start + self.pq_code_width;
            sorted_pq_values.extend_from_slice(&pq_values.values()[start..end]);
        }

        let file_offset = self.bucket_row_counts[bucket_id];
        let final_relative_path = self.final_bucket_relative_path(bucket_id);
        let mut offset = 0usize;
        while offset < sorted_partition_ids.len() {
            let partition_id = sorted_partition_ids[offset] as usize;
            let mut end = offset + 1;
            while end < sorted_partition_ids.len()
                && sorted_partition_ids[end] == sorted_partition_ids[offset]
            {
                end += 1;
            }
            let partition = &mut self.partitions[partition_id];
            match &partition.path {
                Some(existing) if existing != &final_relative_path => {
                    return Err(Error::io(format!(
                        "partition {} is split across multiple bucket files: '{}' vs '{}'",
                        partition_id, existing, final_relative_path
                    )));
                }
                None => partition.path = Some(final_relative_path.clone()),
                _ => {}
            }
            partition.num_rows += end - offset;
            partition.ranges.push(PartitionArtifactRange {
                offset: file_offset + offset as u64,
                num_rows: (end - offset) as u64,
            });
            offset = end;
        }

        let pq_codes = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(sorted_pq_values),
            self.pq_code_width as i32,
        )?;
        let final_batch = RecordBatch::try_new(
            self.final_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(sorted_row_ids)),
                Arc::new(pq_codes),
            ],
        )?;
        let writer = self.ensure_final_writer(bucket_id).await?;
        writer.write_batch(&final_batch).await?;
        self.bucket_row_counts[bucket_id] += total_rows as u64;
        Ok(())
    }

    /// Lazily create the finalized writer for a bucket.
    ///
    /// Buckets that never receive rows never create a file, which keeps sparse
    /// artifacts compact.
    async fn ensure_final_writer(&mut self, bucket_id: usize) -> Result<&mut FileWriter> {
        if self.final_writers[bucket_id].is_none() {
            let path = self.final_bucket_path(bucket_id);
            let writer = FileWriter::try_new(
                self.object_store.create(&path).await?,
                Schema::try_from(self.final_schema.as_ref())?,
                file_writer_options()?,
            )?;
            self.final_writers[bucket_id] = Some(writer);
        }
        Ok(self.final_writers[bucket_id]
            .as_mut()
            .expect("final writer initialized"))
    }

    /// Path of the finalized file for one bucket.
    fn final_bucket_path(&self, bucket_id: usize) -> Path {
        path_child(
            &path_child(&self.root_dir, PARTITION_ARTIFACT_PARTITIONS_DIR),
            format!("{PARTITION_ARTIFACT_BUCKET_PREFIX}{bucket_id:05}.lance"),
        )
    }

    /// Relative path recorded in the manifest for one finalized bucket.
    fn final_bucket_relative_path(&self, bucket_id: usize) -> String {
        format!(
            "{PARTITION_ARTIFACT_PARTITIONS_DIR}/{PARTITION_ARTIFACT_BUCKET_PREFIX}{bucket_id:05}.lance"
        )
    }
}

/// Reopens a partition artifact as a `ShuffleReader`.
///
/// The final Lance builder consumes artifacts through the generic
/// [`ShuffleReader`] interface, so this adapter hides the manifest parsing and
/// file caching needed to expose partition-local record batch streams.
#[derive(Debug)]
pub struct PartitionArtifactShuffleReader {
    scheduler: Arc<ScanScheduler>,
    root_dir: Path,
    partitions: Vec<PartitionArtifactPartition>,
    total_loss: Option<f64>,
    file_readers: Mutex<HashMap<String, Arc<FileReader>>>,
}

/// Writer options for all files stored inside a partition artifact.
///
/// The artifact uses a fixed file version so external backends and Lance
/// finalization agree on the on-disk layout.
fn file_writer_options() -> Result<FileWriterOptions> {
    Ok(FileWriterOptions {
        format_version: Some(
            PARTITION_ARTIFACT_FILE_VERSION
                .parse::<LanceFileVersion>()
                .map_err(|error| {
                    Error::invalid_input(format!(
                        "invalid partition artifact file version '{}': {}",
                        PARTITION_ARTIFACT_FILE_VERSION, error
                    ))
                })?,
        ),
        ..Default::default()
    })
}

/// Validate that a backend-produced batch matches the artifact contract.
///
/// The builder is intentionally strict here because any schema drift would only
/// surface much later during finalization.
fn validate_input_batch(batch: &RecordBatch, pq_code_width: usize) -> Result<()> {
    let Some(row_ids) = batch.column_by_name(ROW_ID) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {ROW_ID}"
        )));
    };
    if row_ids.data_type() != &DataType::UInt64 {
        return Err(Error::invalid_input(format!(
            "partition artifact batch column {ROW_ID} must be uint64, got {}",
            row_ids.data_type()
        )));
    }
    let Some(part_ids) = batch.column_by_name(PART_ID_COLUMN) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {PART_ID_COLUMN}"
        )));
    };
    if part_ids.data_type() != &DataType::UInt32 {
        return Err(Error::invalid_input(format!(
            "partition artifact batch column {PART_ID_COLUMN} must be uint32, got {}",
            part_ids.data_type()
        )));
    }
    let Some(pq_codes) = batch.column_by_name(PQ_CODE_COLUMN) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {PQ_CODE_COLUMN}"
        )));
    };
    match pq_codes.data_type() {
        DataType::FixedSizeList(_, width) if *width as usize == pq_code_width => Ok(()),
        other => Err(Error::invalid_input(format!(
            "partition artifact batch column {PQ_CODE_COLUMN} must be fixed_size_list<uint8>[{}], got {}",
            pq_code_width, other
        ))),
    }
}

/// Serialize a small JSON sidecar directly into the object store.
async fn write_json<T: Serialize>(
    object_store: &ObjectStore,
    path: &Path,
    value: &T,
) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        Error::invalid_input(format!(
            "failed to serialize partition artifact manifest '{}': {}",
            path, error
        ))
    })?;
    let mut writer = object_store.create(path).await?;
    writer.write_all(&bytes).await?;
    Writer::shutdown(writer.as_mut()).await?;
    Ok(())
}

impl PartitionArtifactShuffleReader {
    /// Open an artifact reader from a URI and optional storage options.
    pub async fn try_open(
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = if let Some(storage_options) = storage_options {
            ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        storage_options.clone(),
                    ),
                )),
                ..Default::default()
            }
        } else {
            ObjectStoreParams::default()
        };
        let (object_store, root_dir) =
            ObjectStore::from_uri_and_params(registry, uri, &params).await?;
        Self::try_open_with_store(object_store, root_dir).await
    }

    /// Open an artifact reader once the object store has already been resolved.
    ///
    /// This reads the manifest once, validates it, and initializes the shared
    /// scheduler and reader cache used by partition reads.
    async fn try_open_with_store(object_store: Arc<ObjectStore>, root_dir: Path) -> Result<Self> {
        let manifest_path = path_child(&root_dir, "manifest.json");
        let manifest_bytes = object_store.read_one_all(&manifest_path).await?;
        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                Error::invalid_input(format!(
                    "failed to parse partition artifact manifest '{}': {}",
                    manifest_path, error
                ))
            })?;
        if manifest.version != 1 {
            return Err(Error::invalid_input(format!(
                "unsupported partition artifact manifest version {}",
                manifest.version
            )));
        }
        if manifest.partitions.len() != manifest.num_partitions {
            return Err(Error::invalid_input(format!(
                "partition artifact manifest has {} partitions but num_partitions is {}",
                manifest.partitions.len(),
                manifest.num_partitions
            )));
        }

        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        Ok(Self {
            scheduler,
            root_dir,
            partitions: manifest.partitions,
            total_loss: manifest.total_loss,
            file_readers: Mutex::new(HashMap::new()),
        })
    }

    /// Open and cache a file reader for a finalized bucket file.
    ///
    /// Multiple logical partitions can point at the same bucket file, so the
    /// reader cache prevents redundant file opens during finalization.
    async fn open_file_reader(&self, relative_path: &str) -> Result<Arc<FileReader>> {
        if let Some(reader) = self
            .file_readers
            .lock()
            .expect("partition artifact reader mutex poisoned")
            .get(relative_path)
            .cloned()
        {
            return Ok(reader);
        }

        let path = join_relative_path(&self.root_dir, relative_path);
        let reader = Arc::new(
            FileReader::try_open(
                self.scheduler
                    .open_file(&path, &CachedFileSize::unknown())
                    .await?,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await?,
        );
        self.file_readers
            .lock()
            .expect("partition artifact reader mutex poisoned")
            .insert(relative_path.to_string(), reader.clone());
        Ok(reader)
    }
}

/// Join a manifest-relative path onto the artifact root.
fn join_relative_path(root_dir: &Path, relative_path: &str) -> Path {
    relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .fold(root_dir.clone(), |path, segment| path_child(&path, segment))
}

#[allow(deprecated)]
fn path_child<'a>(path: &Path, child: impl Into<PathPart<'a>>) -> Path {
    path.child(child)
}

#[async_trait::async_trait]
impl ShuffleReader for PartitionArtifactShuffleReader {
    /// Return a stream over all rows belonging to one logical partition.
    ///
    /// The manifest already records the precise row ranges for each partition,
    /// so the reader can issue targeted range reads without scanning unrelated
    /// partitions.
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        let Some(partition) = self.partitions.get(partition_id) else {
            return Ok(None);
        };
        if partition.num_rows == 0 {
            return Ok(None);
        }
        let path = partition.path.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "partition artifact partition {} has {} rows but no path",
                partition_id, partition.num_rows
            ))
        })?;
        if partition.ranges.is_empty() {
            return Err(Error::invalid_input(format!(
                "partition artifact partition {} has {} rows but no ranges",
                partition_id, partition.num_rows
            )));
        }

        let reader = self.open_file_reader(path).await?;
        let ranges = partition
            .ranges
            .iter()
            .map(|range| Range {
                start: range.offset,
                end: range.offset + range.num_rows,
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(reader.schema().as_ref().into());
        Ok(Some(Box::new(RecordBatchStreamAdapter::new(
            schema,
            reader
                .read_stream(
                    ReadBatchParams::Ranges(ranges.into()),
                    u32::MAX,
                    16,
                    FilterExpression::no_filter(),
                )
                .await?,
        ))))
    }

    /// Number of encoded rows available for one logical partition.
    fn partition_size(&self, partition_id: usize) -> Result<usize> {
        Ok(self
            .partitions
            .get(partition_id)
            .map(|partition| partition.num_rows)
            .unwrap_or(0))
    }

    /// Optional training loss propagated from the backend into the artifact.
    fn total_loss(&self) -> Option<f64> {
        self.total_loss
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatch, UInt8Array, UInt32Array, UInt64Array};
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use lance_core::datatypes::Schema;
    use lance_file::writer::{FileWriter, FileWriterOptions};
    use lance_io::object_store::ObjectStore;

    use crate::Error;

    use super::*;

    #[tokio::test]
    async fn partition_artifact_builder_compacts_runs_into_single_partition_range() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();

        let mut builder = PartitionArtifactBuilder::try_new_with_store(
            object_store.clone(),
            root_path.clone(),
            300,
            2,
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(PART_ID_COLUMN, DataType::UInt32, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 2),
                true,
            ),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 11, 12, 13])),
                Arc::new(UInt32Array::from(vec![0_u32, 256, 0, 256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![14_u64, 15])),
                Arc::new(UInt32Array::from(vec![1_u32, 256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![9, 10, 11, 12]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        builder.append_batch(&batch1).await.unwrap();
        builder.append_batch(&batch2).await.unwrap();
        let artifact_files = builder.finish("metadata.lance", Some(2.5)).await.unwrap();
        assert_eq!(artifact_files[0], "manifest.json");
        assert!(
            artifact_files
                .iter()
                .any(|path| path.ends_with("bucket-00000.lance"))
        );

        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&fs::read(root_dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.metadata_file.as_deref(), Some("metadata.lance"));
        assert_eq!(manifest.total_loss, Some(2.5));
        assert_eq!(manifest.partitions[0].num_rows, 2);
        assert_eq!(manifest.partitions[0].ranges.len(), 1);
        assert_eq!(manifest.partitions[1].num_rows, 1);
        assert_eq!(manifest.partitions[1].ranges.len(), 1);
        assert_eq!(manifest.partitions[256].num_rows, 3);
        assert_eq!(manifest.partitions[256].ranges.len(), 1);
        assert_eq!(
            manifest.partitions[0].path, manifest.partitions[256].path,
            "partitions sharing a bucket should share one final file"
        );

        let reader = PartitionArtifactShuffleReader::try_open_with_store(object_store, root_path)
            .await
            .unwrap();
        let partition_0 = reader
            .read_partition(0)
            .await
            .unwrap()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let partition_0_row_ids = partition_0
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(partition_0_row_ids, vec![10, 12]);

        let partition_256 = reader
            .read_partition(256)
            .await
            .unwrap()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let partition_256_row_ids = partition_256
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(partition_256_row_ids, vec![11, 13, 15]);
    }

    #[tokio::test]
    async fn partition_artifact_reader_reads_partition_ranges() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(root_dir.join("partitions")).unwrap();

        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();
        let partition_path =
            path_child(&path_child(&root_path, "partitions"), "bucket-00000.lance");
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(
                lance_index::vector::PQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::UInt8,
                        true,
                    )),
                    2,
                ),
                true,
            ),
        ]));
        let mut writer = FileWriter::try_new(
            object_store.create(&partition_path).await.unwrap(),
            Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 11, 12])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![1, 2, 3, 4, 5, 6]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![13_u64, 14])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![7, 8, 9, 10]), 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        writer.write_batch(&batch1).await.unwrap();
        writer.write_batch(&batch2).await.unwrap();
        writer.finish().await.unwrap();

        let manifest = serde_json::json!({
            "version": 1,
            "num_partitions": 3,
            "total_loss": 1.5,
            "partitions": [
                {
                    "path": "partitions/bucket-00000.lance",
                    "num_rows": 2,
                    "ranges": [
                        {"offset": 0, "num_rows": 1},
                        {"offset": 3, "num_rows": 1},
                    ],
                },
                {
                    "path": "partitions/bucket-00000.lance",
                    "num_rows": 2,
                    "ranges": [
                        {"offset": 1, "num_rows": 2},
                    ],
                },
                {
                    "num_rows": 0,
                    "ranges": [],
                },
            ],
        });
        fs::write(
            root_dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let reader = PartitionArtifactShuffleReader::try_open_with_store(object_store, root_path)
            .await
            .unwrap();
        assert_eq!(reader.partition_size(0).unwrap(), 2);
        assert_eq!(reader.partition_size(1).unwrap(), 2);
        assert_eq!(reader.partition_size(2).unwrap(), 0);
        assert_eq!(reader.total_loss(), Some(1.5));

        let stream = reader.read_partition(0).await.unwrap().unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let row_ids = batches
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![10, 13]);
        assert!(reader.read_partition(2).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn partition_artifact_reader_rejects_missing_partition_entry() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "num_partitions": 2,
            "partitions": [{"num_rows": 0, "ranges": []}],
        });
        fs::write(
            root_dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = PartitionArtifactShuffleReader::try_open_with_store(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(&root_dir).unwrap(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn partition_artifact_builder_records_multiple_ranges_for_repeated_flushes() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();

        let mut builder =
            PartitionArtifactBuilder::try_new_with_store(object_store, root_path, 4, 2).unwrap();
        let num_rows = PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS + 1024;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(PART_ID_COLUMN, DataType::UInt32, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 2),
                true,
            ),
        ]));
        let row_ids = UInt64Array::from_iter_values(0..num_rows as u64);
        let part_ids = UInt32Array::from_iter_values((0..num_rows).map(|_| 0_u32));
        let pq_values = UInt8Array::from_iter_values((0..num_rows * 2).map(|v| (v % 251) as u8));
        let pq_codes = FixedSizeListArray::try_new_from_values(pq_values, 2).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(row_ids), Arc::new(part_ids), Arc::new(pq_codes)],
        )
        .unwrap();

        builder.append_batch(&batch).await.unwrap();
        builder.finish("metadata.lance", None).await.unwrap();

        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&fs::read(root_dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.partitions[0].num_rows, num_rows);
        assert_eq!(manifest.partitions[0].ranges.len(), 2);
        assert_eq!(
            manifest.partitions[0].ranges[0].num_rows,
            PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS as u64
        );
        assert_eq!(
            manifest.partitions[0].ranges[1].offset,
            PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS as u64
        );
    }
}
