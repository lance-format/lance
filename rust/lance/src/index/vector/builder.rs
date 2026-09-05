// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, pin::Pin};

use arrow::array::{AsArray as _, PrimitiveBuilder, UInt32Builder, UInt64Builder};
use arrow::compute::sort_to_indices;
use arrow::datatypes::{self};
use arrow::datatypes::{Float16Type, Float64Type, UInt8Type, UInt64Type};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, PrimitiveArray,
    RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Fields};
use futures::{FutureExt, stream};
use futures::{
    Stream,
    prelude::stream::{StreamExt, TryStreamExt},
};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::ROW_ID;
use lance_core::datatypes::Schema;
use lance_core::utils::tempfile::TempStdDir;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, ROW_ID_FIELD, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions as file_versions;
use lance_file::writer::FileWriterOptions;
use lance_index::frag_reuse::{CompactFragReuseIndex, CompactFragReuseIndexHandle};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};
use lance_index::scalar::RowIdRemapper;
use lance_index::vector::bq::storage::{RABIT_CODE_COLUMN, unpack_codes};
use lance_index::vector::kmeans::KMeansParams;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::quantizer::{
    QuantizationMetadata, QuantizationType, QuantizerBuildParams,
};
use lance_index::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use lance_index::vector::shared::{SupportedIvfIndexType, write_unified_ivf_and_index_metadata};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::vector::transform::Flatten;
use lance_index::vector::v3::shuffler::{
    DEFAULT_PARTITION_WINDOW_BYTES, EmptyReader, IvfShufflerReader, create_ivf_shuffler,
};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::{LOSS_METADATA_KEY, PART_ID_COLUMN, PQ_CODE_COLUMN, VectorIndex};
use lance_index::vector::{PART_ID_FIELD, ivf::storage::IvfModel};
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, pb,
    vector::{
        DISTANCE_TYPE_KEY,
        ivf::{IvfBuildParams, storage::IVF_METADATA_KEY},
        quantizer::Quantization,
        storage::{StorageBuilder, VectorStore},
        transform::Transformer,
        v3::{
            shuffler::{ShuffleReader, Shuffler},
            subindex::IvfSubIndex,
        },
    },
};
use lance_index::{
    INDEX_METADATA_SCHEMA_KEY, IndexMetadata, IndexType, MAX_PARTITION_SIZE_FACTOR,
    MIN_PARTITION_SIZE_PERCENT, scalar::OldIndexDataFilter,
};
use lance_io::local::to_local_path;
use lance_io::stream::RecordBatchStream;
use lance_io::{object_store::ObjectStore, stream::RecordBatchStreamAdapter};
use lance_linalg::distance::{DistanceType, Dot, L2, Normalize};
use lance_linalg::kernels::normalize_fsl;
use lance_table::format::IndexFile;
use log::info;
use object_store::path::Path;
use prost::Message;
use rand::{SeedableRng, rngs::SmallRng};
use roaring::RoaringBitmap;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};
use tracing::{Level, instrument, span};

use crate::Dataset;
use crate::dataset::ProjectionRequest;
use crate::dataset::index::dataset_format_version;
use crate::index::append::build_old_data_filter;
use crate::index::vector::bounded_partition_stream::{
    BoundedPartitionStream, Budgeted, OrderedPartitionResults, WeightedJob,
};
use crate::index::vector::utils::infer_vector_dim;

use super::v2::IVFIndex;
use super::{
    ivf::load_precomputed_partitions_if_available,
    utils::{self, get_vector_type},
};

// the number of partitions to evaluate for reassigning
const REASSIGN_RANGE: usize = 64;
/// Training vectors sampled per centroid when a partition is split.
const SPLIT_SAMPLE_RATE: usize = 256;
/// Upper bound on the number of partitions one oversized partition is split into
/// in a single optimize pass; anything still oversized waits for the next pass.
const MAX_SPLIT_WAYS: usize = 1024;
/// Rows sampled from a neighbor partition to decide whether a split can pull any
/// of its rows away.
const REASSIGN_SAMPLE_SIZE: usize = 512;
/// A sampled row counts as movable when a new centroid is within this relative
/// margin of its own centroid's distance, so near-ties do not prune a neighbor
/// that unsampled rows would still leave.
const REASSIGN_MARGIN: f32 = 0.05;

/// Number of partitions the split partitions' rows now belong to.
fn new_partition_ids_len(split_partitions: &[(usize, Vec<usize>)]) -> usize {
    split_partitions.iter().map(|(_, ids)| ids.len()).sum()
}

/// One oversized partition and the number of partitions it is split into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PartitionSplit {
    partition: usize,
    ways: usize,
}
/// Maximum decoded input bytes admitted across active builds and completed
/// partitions waiting for their turn to be written.
const PARTITION_BUILD_BUDGET_BYTES: usize = 512 * 1024 * 1024;
/// Bound ready-map overhead even when many consecutive partitions are empty.
const PARTITION_BUILD_ENTRIES_PER_WORKER: usize = 2;

#[derive(Debug, Clone, Copy)]
struct FreshPartitionBuildLimits {
    window_bytes: usize,
    decoded_budget_bytes: usize,
}

impl Default for FreshPartitionBuildLimits {
    fn default() -> Self {
        Self {
            window_bytes: DEFAULT_PARTITION_WINDOW_BYTES,
            decoded_budget_bytes: PARTITION_BUILD_BUDGET_BYTES,
        }
    }
}

/// Build a new centroid array that incorporates the results of partition splits.
///
/// For each `(part_idx, centroids)` in `splits`, `original[part_idx]` is replaced
/// by the first new centroid and the remaining ones are appended after all
/// existing centroids, in split order. Unchanged centroids keep their original
/// indices.
fn apply_centroid_splits(
    original: &FixedSizeListArray,
    splits: &[(usize, Vec<ArrayRef>)],
) -> Result<FixedSizeListArray> {
    let mut new_centroids: Vec<ArrayRef> = original.iter().map(|v| v.unwrap()).collect();
    for (part_idx, centroids) in splits {
        let (first, rest) = centroids.split_first().ok_or_else(|| {
            Error::invalid_input(format!(
                "split of partition {part_idx} produced no centroids"
            ))
        })?;
        new_centroids[*part_idx] = first.clone();
        new_centroids.extend(rest.iter().cloned());
    }
    let refs: Vec<&dyn Array> = new_centroids.iter().map(|a| a.as_ref()).collect();
    let concatenated = arrow::compute::concat(&refs)?;
    Ok(FixedSizeListArray::try_new_from_values(
        concatenated,
        original.value_length(),
    )?)
}

/// An index segment an optimize pass reads existing rows from, paired with the rows
/// that segment is still allowed to contribute.
///
/// A segment's index file keeps every row it was built with, but the rows it may still
/// contribute shrink afterwards: an update rewrites a row and either prunes its
/// fragment from the segment's bitmap (in-place column rewrite) or deletion-marks the
/// old physical row while the rewritten copy reuses its stable row id. Copying such a
/// row into a merged segment would duplicate it, because the merged coverage also spans
/// the fresh copy and nothing downstream can tell the two apart.
///
/// The filter is resolved on first use rather than up front: under stable row ids
/// building it loads every covered fragment's row-id sequence, and the common optimize
/// pass appends a delta without reading a single existing row.
#[derive(Clone)]
pub struct ExistingIndex {
    pub index: Arc<dyn VectorIndex>,
    coverage: Option<Arc<SegmentCoverage>>,
}

/// The inputs [`ExistingIndex::old_data_filter`] needs, plus the filter once built.
struct SegmentCoverage {
    dataset: Dataset,
    effective_frags: RoaringBitmap,
    deleted_frags: RoaringBitmap,
    filter: OnceCell<Option<OldIndexDataFilter>>,
}

impl ExistingIndex {
    /// An existing index whose rows are all still valid.
    pub fn unfiltered(index: Arc<dyn VectorIndex>) -> Self {
        Self {
            index,
            coverage: None,
        }
    }

    /// An existing index that may only contribute rows still live in `effective_frags`.
    pub fn with_coverage(
        index: Arc<dyn VectorIndex>,
        dataset: Dataset,
        effective_frags: RoaringBitmap,
        deleted_frags: RoaringBitmap,
    ) -> Self {
        Self {
            index,
            coverage: Some(Arc::new(SegmentCoverage {
                dataset,
                effective_frags,
                deleted_frags,
                filter: OnceCell::new(),
            })),
        }
    }

    /// Whether the filter has already been built. A pass that reads no existing rows
    /// must never pay for one.
    #[cfg(test)]
    pub(crate) fn filter_is_built(&self) -> bool {
        self.coverage
            .as_deref()
            .is_some_and(|coverage| coverage.filter.initialized())
    }

    /// The filter to apply to this segment's stored rows, or `None` when every row it
    /// holds is still valid. Built once and shared by all partitions.
    pub(crate) async fn old_data_filter(&self) -> Result<Option<&OldIndexDataFilter>> {
        let Some(coverage) = self.coverage.as_deref() else {
            return Ok(None);
        };
        let filter = coverage
            .filter
            .get_or_try_init(|| {
                build_old_data_filter(
                    &coverage.dataset,
                    &coverage.effective_frags,
                    &coverage.deleted_frags,
                )
            })
            .await?;
        Ok(filter.as_ref())
    }
}

// Builder for IVF index
// The builder will train the IVF model and quantizer, shuffle the dataset, and build the sub index
// for each partition.
// To build the index for the whole dataset, call `build` method.
// To build the index for given IVF, quantizer, data stream,
// call `with_ivf`, `with_quantizer`, `shuffle_data_input`, and `build` in order.
pub struct IvfIndexBuilder<S: IvfSubIndex, Q: Quantization> {
    store: ObjectStore,
    column: String,
    index_dir: Path,
    distance_type: DistanceType,
    // build params, only needed for building new IVF, quantizer
    dataset: Option<Dataset>,
    shuffler: Option<Arc<dyn Shuffler>>,
    ivf_params: Option<IvfBuildParams>,
    quantizer_params: Option<Q::BuildParams>,
    sub_index_params: Option<S::BuildParams>,
    _temp_dir: TempStdDir, // store this for keeping the temp dir alive and clean up after build
    temp_dir: Path,

    // fields will be set during build
    ivf: Option<IvfModel>,
    quantizer: Option<Q>,
    shuffle_reader: Option<Arc<dyn ShuffleReader>>,
    // unindexed input stream attached by callers; consumed during `build`'s
    // shuffle stage so progress is reported. Wrapped in Mutex so the builder
    // remains `Sync` (the boxed dyn Stream is not Sync on its own).
    shuffle_data_input: Mutex<Option<UnindexedStream>>,

    // fields for merging indices / remapping
    existing_indices: Vec<ExistingIndex>,

    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,

    // fragments for distributed indexing
    fragment_filter: Option<Vec<u32>>,

    // optimize options for only incremental build
    optimize_options: Option<OptimizeOptions>,
    // number of indices merged
    merged_num: usize,
    // rows per partition the index was created for; `None` falls back to the
    // index type's default when deciding splits and joins
    target_partition_size: Option<usize>,
    // whether to transpose codes when building storage
    transpose_codes: bool,

    // lance file version for writing index files
    format_version: ConcreteFileVersion,

    progress: Arc<dyn IndexBuildProgress>,
}

type BuildStream<S, Q> =
    Pin<Box<dyn Stream<Item = Result<Budgeted<PartitionBuildResult<S, Q>>>> + Send>>;

type FreshWindowBuildStream<S, Q> =
    Pin<Box<dyn Stream<Item = Result<(PartitionBuildResult<S, Q>, OwnedSemaphorePermit)>> + Send>>;
type PartitionInputAdmissionStream<T> =
    Pin<Box<dyn Stream<Item = Result<(T, OwnedSemaphorePermit)>> + Send>>;

fn admit_partition_inputs<T: Send + 'static>(
    inputs: Vec<T>,
    entry_permits: Arc<Semaphore>,
) -> PartitionInputAdmissionStream<T> {
    stream::iter(inputs)
        .then(move |input| {
            let entry_permits = entry_permits.clone();
            async move {
                let entry_permit = entry_permits
                    .acquire_owned()
                    .await
                    .map_err(|_| Error::internal("partition build entry semaphore was closed"))?;
                Ok((input, entry_permit))
            }
        })
        .boxed()
}

fn partition_window_entry_limit(
    partition_range: &std::ops::Range<usize>,
    num_partitions: usize,
    max_entries: usize,
    concurrency: usize,
) -> usize {
    if partition_range.start == 0 && partition_range.end == num_partitions {
        max_entries
    } else {
        max_entries.div_ceil(concurrency)
    }
}

struct PartitionBuildResult<S: IvfSubIndex, Q: Quantization> {
    partition_id: usize,
    built: Option<(Q::Storage, S, f64)>,
}

struct FreshPartitionInput {
    partition_id: usize,
    batches: Vec<RecordBatch>,
    loss: f64,
}

type UnindexedStream = Box<dyn Stream<Item = Result<RecordBatch>> + Send + Unpin + 'static>;

pub struct VectorIndexBuildSummary {
    pub indices_merged: usize,
    pub files: Vec<IndexFile>,
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> IvfIndexBuilder<S, Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        ivf_params: Option<IvfBuildParams>,
        quantizer_params: Option<Q::BuildParams>,
        sub_index_params: S::BuildParams,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    ) -> Result<Self> {
        let temp_dir = TempStdDir::default();
        let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
        let format_version = dataset_format_version(&dataset);
        Ok(Self {
            store: dataset.object_store.as_ref().clone(),
            column,
            index_dir,
            distance_type,
            dataset: Some(dataset),
            shuffler: Some(shuffler.into()),
            ivf_params,
            quantizer_params,
            sub_index_params: Some(sub_index_params),
            _temp_dir: temp_dir,
            temp_dir: temp_dir_path,
            // fields will be set during build
            ivf: None,
            quantizer: None,
            shuffle_reader: None,
            shuffle_data_input: Mutex::new(None),
            existing_indices: Vec::new(),
            frag_reuse_index,
            fragment_filter: None,
            optimize_options: None,
            merged_num: 0,
            target_partition_size: None,
            transpose_codes: true,
            format_version,
            progress: Arc::new(NoopIndexBuildProgress),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_incremental(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        sub_index_params: S::BuildParams,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
        optimize_options: OptimizeOptions,
    ) -> Result<Self> {
        let mut builder = Self::new(
            dataset,
            column,
            index_dir,
            distance_type,
            shuffler,
            None,
            None,
            sub_index_params,
            frag_reuse_index,
        )?;
        builder.optimize_options = Some(optimize_options);
        Ok(builder)
    }

    pub fn new_remapper(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        index: Arc<dyn VectorIndex>,
    ) -> Result<Self> {
        let ivf_index = index
            .as_any()
            .downcast_ref::<IVFIndex<S, Q>>()
            .ok_or(Error::invalid_input("existing index is not IVF index"))?;

        let temp_dir = TempStdDir::default();
        let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
        let format_version = dataset_format_version(&dataset);
        Ok(Self {
            store: dataset.object_store.as_ref().clone(),
            column,
            index_dir,
            distance_type: ivf_index.metric_type(),
            dataset: Some(dataset),
            shuffler: None,
            ivf_params: None,
            quantizer_params: None,
            sub_index_params: None,
            _temp_dir: temp_dir,
            temp_dir: temp_dir_path,
            ivf: Some(ivf_index.ivf_model().clone()),
            quantizer: Some(ivf_index.quantizer().try_into()?),
            shuffle_reader: None,
            shuffle_data_input: Mutex::new(None),
            existing_indices: vec![ExistingIndex::unfiltered(index)],
            frag_reuse_index: None,
            fragment_filter: None,
            optimize_options: None,
            merged_num: 0,
            target_partition_size: None,
            transpose_codes: true,
            format_version,
            progress: Arc::new(NoopIndexBuildProgress),
        })
    }

    // build the index and return the files created by the writer.
    pub async fn build(&mut self) -> Result<VectorIndexBuildSummary> {
        let progress = self.progress.clone();

        // step 1. train IVF & quantizer
        let max_iters = self.ivf_params.as_ref().map(|p| p.max_iters as u64);
        progress
            .stage_start("train_ivf", max_iters, "iterations")
            .await?;
        self.with_ivf(self.load_or_build_ivf().boxed().await?);
        progress.stage_complete("train_ivf").await?;

        progress.stage_start("train_quantizer", None, "").await?;
        self.with_quantizer(self.load_or_build_quantizer().await?);
        progress.stage_complete("train_quantizer").await?;

        // step 2. shuffle the dataset
        if self.shuffle_reader.is_none() {
            let num_rows = self.num_rows_to_shuffle().await?;
            progress.stage_start("shuffle", num_rows, "rows").await?;
            let input = self.shuffle_data_input.lock().unwrap().take();
            if let Some(input) = input {
                self.shuffle_data(Some(input)).boxed().await?;
            } else {
                self.shuffle_dataset().boxed().await?;
            }
            progress.stage_complete("shuffle").await?;
        }

        // step 3. build and merge partitions
        let num_partitions = self.ivf.as_ref().map(|ivf| ivf.num_partitions() as u64);
        progress
            .stage_start("merge_partitions", num_partitions, "partitions")
            .await?;
        let build_idx_stream = self.build_partitions().boxed().await?;
        let files = self.merge_partitions(build_idx_stream).await?;
        progress.stage_complete("merge_partitions").await?;

        Ok(VectorIndexBuildSummary {
            indices_merged: self.merged_num,
            files,
        })
    }

    pub async fn remap(&mut self, mapping: &RowAddrRemap) -> Result<Vec<IndexFile>> {
        if self.existing_indices.is_empty() {
            return Err(Error::invalid_input(
                "No existing indices available for remapping",
            ));
        }
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input("IVF model not set before remapping"));
        };

        log::info!("remap {} partitions", ivf.num_partitions());
        let existing_index = self.existing_indices[0].index.clone();
        let mapping = Arc::new(mapping.clone());
        let build_iter = (0..ivf.num_partitions()).map(move |part_id| {
            let existing_index = existing_index.clone();
            let mapping = mapping.clone();
            async move {
                let ivf_index = existing_index
                    .as_any()
                    .downcast_ref::<IVFIndex<S, Q>>()
                    .ok_or(Error::invalid_input("existing index is not IVF index"))?;
                let part = ivf_index
                    .load_partition(part_id, false, &NoOpMetricsCollector)
                    .await?;

                let storage = part.storage.remap(&mapping)?;
                let index = part.index.remap(&mapping, &storage)?;
                Result::Ok(Budgeted::untracked(PartitionBuildResult {
                    partition_id: part_id,
                    built: Some((storage, index, 0.0)),
                }))
            }
        });

        let files = self
            .merge_partitions(
                stream::iter(build_iter)
                    .buffered(get_num_compute_intensive_cpus())
                    .boxed(),
            )
            .await?;
        Ok(files)
    }

    pub fn with_ivf(&mut self, ivf: IvfModel) -> &mut Self {
        self.ivf = Some(ivf);
        self
    }

    pub fn with_quantizer(&mut self, quantizer: Q) -> &mut Self {
        self.quantizer = Some(quantizer);
        self
    }

    /// Read existing rows from `indices`, keeping every row they hold.
    pub fn with_existing_indices(&mut self, indices: Vec<Arc<dyn VectorIndex>>) -> &mut Self {
        self.existing_indices = indices.into_iter().map(ExistingIndex::unfiltered).collect();
        self
    }

    /// Read existing rows from `sources`, keeping only the rows each segment is still
    /// allowed to contribute. See [`ExistingIndex`].
    pub fn with_existing_index_sources(&mut self, sources: Vec<ExistingIndex>) -> &mut Self {
        self.existing_indices = sources;
        self
    }

    /// Set fragment filter for distributed indexing
    pub fn with_fragment_filter(&mut self, fragment_ids: Vec<u32>) -> &mut Self {
        self.fragment_filter = Some(Dataset::normalize_fragment_ids(&fragment_ids));
        self
    }

    pub fn with_optional_fragment_filter(&mut self, fragment_ids: Option<&[u32]>) -> &mut Self {
        if let Some(fragment_ids) = fragment_ids {
            self.fragment_filter = Some(Dataset::normalize_fragment_ids(fragment_ids));
        }
        self
    }

    /// Control whether codes are transposed when building storage.
    /// This mainly affects intermediate PQ/RQ storage when building distributed indices.
    pub fn with_transpose(&mut self, transpose: bool) -> &mut Self {
        self.transpose_codes = transpose;
        self
    }

    /// Set progress callback for index building
    /// Rows per partition the index was created for. Split and join thresholds
    /// are derived from it; without it the index type's default is used.
    pub fn with_target_partition_size(
        &mut self,
        target_partition_size: Option<usize>,
    ) -> &mut Self {
        self.target_partition_size = target_partition_size;
        self
    }

    pub fn with_progress(&mut self, progress: Arc<dyn IndexBuildProgress>) -> &mut Self {
        self.progress = progress;
        self
    }

    #[instrument(name = "load_or_build_ivf", level = "debug", skip_all)]
    async fn load_or_build_ivf(&self) -> Result<IvfModel> {
        match &self.ivf {
            Some(ivf) => Ok(ivf.clone()),
            None => {
                let Some(dataset) = self.dataset.as_ref() else {
                    return Err(Error::invalid_input(
                        "dataset not set before loading or building IVF",
                    ));
                };
                let dim = utils::get_vector_dim(dataset.schema(), &self.column)?;
                let ivf_params = self
                    .ivf_params
                    .as_ref()
                    .ok_or(Error::invalid_input("IVF build params not set"))?;
                super::build_ivf_model(
                    dataset,
                    &self.column,
                    dim,
                    self.distance_type,
                    ivf_params,
                    self.fragment_filter.as_deref(),
                    self.progress.clone(),
                )
                .await
            }
        }
    }

    #[instrument(name = "load_or_build_quantizer", level = "debug", skip_all)]
    async fn load_or_build_quantizer(&self) -> Result<Q> {
        if self.quantizer.is_some() {
            return Ok(self.quantizer.clone().unwrap());
        }

        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before loading or building quantizer",
            ));
        };
        let sample_size_hint = match &self.quantizer_params {
            Some(params) => params.try_sample_size()?,
            None => 256 * 256, // here it must be retrain, let's just set sample size to the default value
        };

        let start = std::time::Instant::now();
        info!(
            "loading training data for quantizer. sample size: {}",
            sample_size_hint
        );
        let training_data = utils::maybe_sample_training_data(
            dataset,
            &self.column,
            sample_size_hint,
            self.fragment_filter.as_deref(),
        )
        .await?;
        info!(
            "Finished loading training data in {:02} seconds",
            start.elapsed().as_secs_f32()
        );

        // If metric type is cosine, normalize the training data, and after this point,
        // treat the metric type as L2.
        let training_data = if self.distance_type == DistanceType::Cosine {
            lance_linalg::kernels::normalize_fsl_owned(training_data)?
        } else {
            training_data
        };

        // we filtered out nulls when sampling, but we still need to filter out NaNs and INFs here
        let training_data = utils::filter_finite_training_data(training_data)?;

        let training_data = match (self.ivf.as_ref(), Q::use_residual(self.distance_type)) {
            (Some(ivf), true) => {
                let ivf_transformer = lance_index::vector::ivf::new_ivf_transformer(
                    ivf.centroids.clone().unwrap(),
                    DistanceType::L2,
                    vec![],
                );
                span!(Level::INFO, "compute residual for PQ training")
                    .in_scope(|| ivf_transformer.compute_residual(&training_data))?
            }
            _ => training_data,
        };

        info!("Start to train quantizer");
        let start = std::time::Instant::now();
        let quantizer = match &self.quantizer {
            Some(q) => q.clone(),
            None => {
                let quantizer_params = self
                    .quantizer_params
                    .as_ref()
                    .ok_or(Error::invalid_input("quantizer build params not set"))?;
                Q::build(&training_data, DistanceType::L2, quantizer_params)?
            }
        };
        info!(
            "Trained quantizer in {:02} seconds",
            start.elapsed().as_secs_f32()
        );
        Ok(quantizer)
    }

    fn rename_row_id(
        stream: impl RecordBatchStream + Unpin + 'static,
        row_id_idx: usize,
    ) -> impl RecordBatchStream + Unpin + 'static {
        let new_schema = Arc::new(arrow_schema::Schema::new(
            stream
                .schema()
                .fields
                .iter()
                .enumerate()
                .map(|(field_idx, field)| {
                    if field_idx == row_id_idx {
                        arrow_schema::Field::new(
                            ROW_ID,
                            field.data_type().clone(),
                            field.is_nullable(),
                        )
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Fields>(),
        ));
        RecordBatchStreamAdapter::new(
            new_schema.clone(),
            stream.map_ok(move |batch| {
                RecordBatch::try_new(new_schema.clone(), batch.columns().to_vec()).unwrap()
            }),
        )
    }

    async fn num_rows_to_shuffle(&self) -> Result<Option<u64>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Ok(None);
        };
        match &self.fragment_filter {
            Some(fragment_ids) => Ok(Some(
                dataset
                    .count_rows_in_existing_fragments(fragment_ids)
                    .await? as u64,
            )),
            None => Ok(Some(dataset.count_rows(None).await? as u64)),
        }
    }

    async fn shuffle_dataset(&mut self) -> Result<()> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input("dataset not set before shuffling"));
        };

        let stream = match self
            .ivf_params
            .as_ref()
            .and_then(|p| p.precomputed_shuffle_buffers.as_ref())
        {
            Some((uri, _)) => {
                let uri = to_local_path(uri);
                // the uri points to data directory,
                // so need to trim the "data" suffix for reading the dataset
                let uri = uri.trim_end_matches("data");
                log::info!("shuffle with precomputed shuffle buffers from {}", uri);
                let ds = Dataset::open(uri).await?;
                ds.scan().try_into_stream().await?
            }
            _ => {
                log::info!("shuffle column {} over dataset", self.column);
                let mut builder = dataset.scan();
                builder
                    .batch_readahead(get_num_compute_intensive_cpus())
                    .project(&[self.column.as_str()])?
                    .with_row_id();

                // Apply fragment filter for distributed indexing
                if let Some(fragment_ids) = &self.fragment_filter {
                    log::info!(
                        "applying fragment filter for distributed indexing: {:?}",
                        fragment_ids
                    );
                    builder.with_fragments(
                        dataset.get_existing_fragment_metadata_from_ids(fragment_ids),
                    );
                }

                let (vector_type, _) = get_vector_type(dataset.schema(), &self.column)?;
                let is_multivector = matches!(vector_type, datatypes::DataType::List(_));
                if is_multivector {
                    builder.batch_size(64);
                }
                builder.try_into_stream().await?
            }
        };

        if let Some((row_id_idx, _)) = stream.schema().column_with_name("row_id") {
            // When using precomputed shuffle buffers we can't use the column name _rowid
            // since it is reserved.  So we tolerate `row_id` as well here (and rename it
            // to _rowid to match the non-precomputed path)
            self.shuffle_data(Some(Self::rename_row_id(stream, row_id_idx)))
                .await?;
        } else {
            self.shuffle_data(Some(stream)).await?;
        }
        Ok(())
    }

    /// Attach an unindexed input stream. The shuffle is deferred until
    /// `build()` so progress reporting wraps the actual shuffle work.
    /// Data must have schema | ROW_ID | vector_column |.
    ///
    /// Passing `None` records "no unindexed data" by installing an empty
    /// shuffle reader directly, so `build()` won't fall back to re-scanning
    /// the dataset.
    pub fn shuffle_data_input(
        &mut self,
        data: Option<impl RecordBatchStream + Unpin + 'static>,
    ) -> &mut Self {
        match data {
            Some(d) => {
                *self.shuffle_data_input.lock().unwrap() = Some(Box::new(d) as UnindexedStream);
            }
            None => {
                self.shuffle_reader = Some(Arc::new(EmptyReader));
            }
        }
        self
    }

    // shuffle the unindexed data and existing indices
    // data must be with schema | ROW_ID | vector_column |
    // the shuffled data will be with schema | ROW_ID | PART_ID | code_column |
    pub async fn shuffle_data(
        &mut self,
        data: Option<impl Stream<Item = Result<RecordBatch>> + Unpin + Send + 'static>,
    ) -> Result<&mut Self> {
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input("IVF not set before shuffle data"));
        };

        let Some(data) = data else {
            // If we don't specify the shuffle reader, it's going to re-read the
            // dataset and duplicate the data.
            self.shuffle_reader = Some(Arc::new(EmptyReader));

            return Ok(self);
        };

        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before shuffle data",
            ));
        };
        let Some(shuffler) = self.shuffler.as_ref() else {
            return Err(Error::invalid_input("shuffler not set before shuffle data"));
        };

        let code_column = quantizer.column();

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                ivf.centroids.clone().unwrap(),
                self.distance_type,
                &self.column,
                quantizer.into(),
                None,
            )?,
        );

        let precomputed_partitions = if let Some(params) = self.ivf_params.as_ref() {
            load_precomputed_partitions_if_available(params)
                .await?
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        let partition_map = Arc::new(precomputed_partitions);
        let mut transformed_stream = Box::pin(
            data.map(move |batch| {
                let partition_map = partition_map.clone();
                let ivf_transformer = transformer.clone();
                tokio::spawn(async move {
                    let mut batch = batch?;
                    if !partition_map.is_empty() {
                        let row_ids = &batch[ROW_ID];
                        let part_ids = UInt32Array::from_iter(
                            row_ids
                                .as_primitive::<UInt64Type>()
                                .values()
                                .iter()
                                .map(|row_id| partition_map.get(row_id).copied()),
                        );
                        let part_ids = UInt32Array::from(part_ids);
                        batch = batch
                            .try_with_column(PART_ID_FIELD.clone(), Arc::new(part_ids.clone()))
                            .expect("failed to add part id column");

                        if part_ids.null_count() > 0 {
                            log::info!(
                                "Filter out rows without valid partition IDs: null_count={}",
                                part_ids.null_count()
                            );
                            let indices = UInt32Array::from_iter(
                                part_ids
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(idx, v)| v.map(|_| idx as u32)),
                            );
                            assert_eq!(indices.len(), batch.num_rows() - part_ids.null_count());
                            batch = batch.take(&indices)?;
                        }
                    }

                    match batch.schema().column_with_name(code_column) {
                        Some(_) => {
                            // this batch is already transformed (in case of GPU training)
                            Ok(batch)
                        }
                        None => ivf_transformer.transform(&batch),
                    }
                })
            })
            .buffered(get_num_compute_intensive_cpus())
            .map(|x| x.unwrap())
            .peekable(),
        );

        let batch = transformed_stream.as_mut().peek_mut().await;
        let schema = match batch {
            Some(Ok(b)) => b.schema(),
            Some(Err(e)) => return Err(std::mem::replace(e, Error::Stop)),
            None => {
                log::info!("no data to shuffle");
                self.shuffle_reader = Some(Arc::new(IvfShufflerReader::new(
                    Arc::new(self.store.clone()),
                    self.temp_dir.clone(),
                    vec![0; ivf.num_partitions()],
                    0.0,
                )));
                return Ok(self);
            }
        };

        self.shuffle_reader = Some(
            shuffler
                .shuffle(Box::new(RecordBatchStreamAdapter::new(
                    schema,
                    transformed_stream,
                )))
                .await?
                .into(),
        );

        Ok(self)
    }

    #[instrument(name = "build_partitions", level = "debug", skip_all)]
    async fn build_partitions(&mut self) -> Result<BuildStream<S, Q>> {
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF not set before building partitions",
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before building partition",
            ));
        };
        let Some(sub_index_params) = self.sub_index_params.clone() else {
            return Err(Error::invalid_input(
                "sub index params not set before building partition",
            ));
        };
        let Some(reader) = self.shuffle_reader.as_ref() else {
            return Err(Error::invalid_input(
                "shuffle reader not set before building partitions",
            ));
        };

        // if no partitions to split, we just create a new delta index,
        // otherwise, we need to merge all existing indices and split large partitions.
        let reader = reader.clone();
        let num_indices_to_merge = self
            .optimize_options
            .as_ref()
            .and_then(|opt| opt.num_indices_to_merge);
        let no_partition_adjustment = || {
            let is_retrain = self
                .optimize_options
                .as_ref()
                .map(|opt| opt.retrain)
                .unwrap_or(false);
            let num_to_merge = match is_retrain {
                true => self.existing_indices.len(), // retrain, merge all indices
                false => num_indices_to_merge.unwrap_or(0),
            };

            let indices_to_merge = self.existing_indices
                [self.existing_indices.len().saturating_sub(num_to_merge)..]
                .to_vec();

            (
                vec![None; ivf.num_partitions()],
                Arc::new(indices_to_merge),
                None,
            )
        };

        let (assign_batches, merge_indices, partition_adjustment) = if num_indices_to_merge
            .is_some()
            || self.optimize_options.is_none()
        {
            no_partition_adjustment()
        } else {
            let target_partition_size = self.effective_target_partition_size()?;
            let (splits, joins) = Self::check_partition_adjustment(
                ivf,
                reader.as_ref(),
                &self.existing_indices,
                target_partition_size,
            )?;
            let split_result = if splits.is_empty() {
                None
            } else {
                log::info!(
                    "split partitions {:?} (target partition size {}), will merge all {} delta indices",
                    splits,
                    target_partition_size,
                    self.existing_indices.len()
                );
                self.split_partitions_streaming(&splits, ivf)
                    .boxed()
                    .await?
            };
            if let Some(split_result) = split_result {
                let Some(ivf) = self.ivf.as_mut() else {
                    return Err(Error::invalid_input(
                        "IVF not set before building partitions",
                    ));
                };
                ivf.centroids = Some(split_result.new_centroids);
                (
                    vec![None; ivf.num_partitions()],
                    Arc::new(self.existing_indices.clone()),
                    Some(PartitionAdjustment::Split {
                        affected_partitions: split_result.affected_partitions,
                        split_shuffle_reader: split_result.shuffle_reader,
                    }),
                )
            } else if !joins.is_empty() {
                log::info!(
                    "join partitions {:?} (target partition size {}), will merge all {} delta indices",
                    joins,
                    target_partition_size,
                    self.existing_indices.len()
                );
                let results = self.join_partitions(&joins, ivf).boxed().await?;
                let Some(ivf) = self.ivf.as_mut() else {
                    return Err(Error::invalid_input(
                        "IVF model not set before joining partitions",
                    ));
                };
                ivf.centroids = Some(results.new_centroids);
                (
                    results.assign_batches,
                    Arc::new(self.existing_indices.clone()),
                    Some(PartitionAdjustment::Join {
                        kept_partitions: results.kept_partitions,
                    }),
                )
            } else {
                no_partition_adjustment()
            }
        };
        self.merged_num = merge_indices.len();
        log::info!(
            "merge {}/{} delta indices",
            self.merged_num,
            self.existing_indices.len()
        );

        let distance_type = self.distance_type;
        let column = self.column.clone();
        let frag_reuse_index = self.frag_reuse_index.clone();
        if self.optimize_options.is_none()
            && self.existing_indices.is_empty()
            && partition_adjustment.is_none()
        {
            let num_partitions = assign_batches.len();
            return Self::build_fresh_partitions_windowed(
                reader,
                num_partitions,
                distance_type,
                quantizer,
                sub_index_params,
                column,
                frag_reuse_index,
                FreshPartitionBuildLimits::default(),
            );
        }
        let partition_adjustment = Arc::new(partition_adjustment);
        let build_iter =
            assign_batches
                .into_iter()
                .enumerate()
                .map(move |(partition, assign_batch)| {
                    let output_partition_id = partition;
                    let reader = reader.clone();
                    let indices = merge_indices.clone();
                    let distance_type = distance_type;
                    let quantizer = quantizer.clone();
                    let sub_index_params = sub_index_params.clone();
                    let column = column.clone();
                    let frag_reuse_index = frag_reuse_index.clone();
                    let partition_adjustment = partition_adjustment.clone();
                    async move {
                        let (is_affected, split_reader) = match partition_adjustment.as_ref() {
                            Some(PartitionAdjustment::Split {
                                affected_partitions,
                                split_shuffle_reader,
                            }) => (
                                affected_partitions.contains(&partition),
                                Some(split_shuffle_reader.clone()),
                            ),
                            _ => (false, None),
                        };
                        let partition = match partition_adjustment.as_ref() {
                            Some(PartitionAdjustment::Join { kept_partitions }) => {
                                kept_partitions[partition]
                            }
                            _ => partition,
                        };

                        // For affected partitions, the split shuffle reader has
                        // all data (existing + new), re-assigned with updated
                        // centroids. For other partitions, read from existing
                        // indices + original shuffle reader as normal.
                        let (mut batches, mut loss) = if is_affected {
                            Self::take_partition_batches(
                                partition,
                                &[],
                                Some(split_reader.as_ref().unwrap().as_ref()),
                            )
                            .await?
                        } else {
                            Self::take_partition_batches(
                                partition,
                                indices.as_ref(),
                                Some(reader.as_ref()),
                            )
                            .await?
                        };

                        // For unaffected partitions during a split, vectors from
                        // affected partitions may have been reassigned here.
                        if !is_affected && let Some(sr) = split_reader.as_ref() {
                            let (extra, extra_loss) =
                                Self::take_partition_batches(partition, &[], Some(sr.as_ref()))
                                    .await?;
                            batches.extend(extra);
                            loss += extra_loss;
                        }

                        spawn_cpu(move || {
                            // Apply assign_batch for join operations (splits no
                            // longer use assign_batches)
                            if let Some((assign_batch, deleted_row_ids)) = assign_batch {
                                if !deleted_row_ids.is_empty() {
                                    let deleted_row_ids = HashSet::<u64>::from_iter(
                                        deleted_row_ids.values().iter().copied(),
                                    );
                                    for batch in batches.iter_mut() {
                                        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                                        let mask =
                                            BooleanArray::from_iter(row_ids.iter().map(|row_id| {
                                                row_id.map(|row_id| {
                                                    !deleted_row_ids.contains(&row_id)
                                                })
                                            }));
                                        *batch = arrow::compute::filter_record_batch(batch, &mask)?;
                                    }
                                }

                                if assign_batch.num_rows() > 0 {
                                    // Drop PART_ID column from assign_batch to match schema of existing batches
                                    let assign_batch = assign_batch.drop_column(PART_ID_COLUMN)?;
                                    batches.push(assign_batch);
                                }
                            }

                            let num_rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
                            if num_rows == 0 {
                                return Ok(Budgeted::untracked(PartitionBuildResult {
                                    partition_id: output_partition_id,
                                    built: None,
                                }));
                            }

                            let (storage, sub_index) = Self::build_index(
                                distance_type,
                                quantizer,
                                sub_index_params,
                                batches,
                                column,
                                frag_reuse_index,
                            )?;
                            Ok(Budgeted::untracked(PartitionBuildResult {
                                partition_id: output_partition_id,
                                built: Some((storage, sub_index, loss)),
                            }))
                        })
                        .await
                    }
                });
        Ok(stream::iter(build_iter)
            .buffered(get_num_compute_intensive_cpus())
            .boxed())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_fresh_partitions_windowed(
        reader: Arc<dyn ShuffleReader>,
        num_partitions: usize,
        distance_type: DistanceType,
        quantizer: Q,
        sub_index_params: S::BuildParams,
        column: String,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
        limits: FreshPartitionBuildLimits,
    ) -> Result<BuildStream<S, Q>> {
        let concurrency = get_num_compute_intensive_cpus().max(1);
        let max_entries = concurrency.saturating_mul(PARTITION_BUILD_ENTRIES_PER_WORKER);
        let cpu_permits = Arc::new(Semaphore::new(concurrency));
        let jobs = stream::try_unfold(0usize, move |next_partition_id| {
            let reader = reader.clone();
            let quantizer = quantizer.clone();
            let sub_index_params = sub_index_params.clone();
            let column = column.clone();
            let frag_reuse_index = frag_reuse_index.clone();
            let cpu_permits = cpu_permits.clone();
            async move {
                if next_partition_id == num_partitions {
                    return Ok(None);
                }
                let plan = reader.plan_partition_window(
                    next_partition_id,
                    limits.window_bytes,
                )?;
                if plan.partition_range.start != next_partition_id
                    || plan.partition_range.end <= plan.partition_range.start
                    || plan.partition_range.end > num_partitions
                {
                    return Err(Error::internal(format!(
                        "shuffle reader planned invalid partition window {:?}; expected a non-empty window starting at {} within {} partitions",
                        plan.partition_range, next_partition_id, num_partitions
                    )));
                }
                let next_partition_id = plan.partition_range.end;
                let planned_range = plan.partition_range;
                let window_entry_limit = partition_window_entry_limit(
                    &planned_range,
                    num_partitions,
                    max_entries,
                    concurrency,
                );
                let job = WeightedJob::with_permit(
                    plan.estimated_decoded_bytes,
                    move |mut admission| async move {
                        let mut window = reader
                            .read_partition_window(
                                planned_range.start,
                                limits.window_bytes,
                            )
                            .await?;
                        if window.partition_range != planned_range
                            || window.partitions.len() != planned_range.len()
                        {
                            return Err(Error::internal(format!(
                                "shuffle reader returned partition window {:?} with {} entries after planning {:?}",
                                window.partition_range,
                                window.partitions.len(),
                                planned_range
                            )));
                        }
                        for (expected_partition_id, partition) in
                            planned_range.clone().zip(&window.partitions)
                        {
                            if partition.partition_id != expected_partition_id {
                                return Err(Error::internal(format!(
                                    "shuffle reader window {:?} returned partition id {} at position {}",
                                    planned_range,
                                    partition.partition_id,
                                    expected_partition_id - planned_range.start
                                )));
                            }
                        }

                        let count_stream_bytes = window.materialized_decoded_bytes.is_none();
                        let mut decoded_bytes =
                            window.materialized_decoded_bytes.unwrap_or_default();
                        let mut inputs = Vec::with_capacity(window.partitions.len());
                        for mut partition in window.partitions.drain(..) {
                            let mut batches = Vec::new();
                            let mut loss = 0.0;
                            if let Some(mut data) = partition.data.take() {
                                while let Some(batch) = data.try_next().await? {
                                    loss += batch
                                        .metadata()
                                        .get(LOSS_METADATA_KEY)
                                        .map(|value| value.parse::<f64>().unwrap_or(0.0))
                                        .unwrap_or(0.0);
                                    if count_stream_bytes {
                                        decoded_bytes = batch.columns().iter().try_fold(
                                            decoded_bytes,
                                            |total, array| {
                                                total
                                                    .checked_add(array.get_array_memory_size())
                                                    .ok_or_else(|| {
                                                        Error::internal(format!(
                                                            "decoded byte count overflow for partition {}",
                                                            partition.partition_id
                                                        ))
                                                    })
                                            },
                                        )?;
                                    }
                                    batches.push(batch.drop_column(PART_ID_COLUMN)?);
                                }
                            }
                            inputs.push(FreshPartitionInput {
                                partition_id: partition.partition_id,
                                batches,
                                loss,
                            });
                        }
                        admission.reconcile(decoded_bytes);

                        // Multiple windows each own a small FIFO entry budget so a
                        // later window cannot consume every slot needed for the
                        // oldest window to make ordered progress. If the whole
                        // shuffle fits in one window, that window owns the complete
                        // entry budget and can use the full CPU concurrency.
                        let entry_permits = Arc::new(Semaphore::new(window_entry_limit));
                        let builds = admit_partition_inputs(inputs, entry_permits)
                            .map_ok(move |(input, entry_permit)| {
                                let quantizer = quantizer.clone();
                                let sub_index_params = sub_index_params.clone();
                                let column = column.clone();
                                let frag_reuse_index = frag_reuse_index.clone();
                                let cpu_permits = cpu_permits.clone();
                                async move {
                                    let partition_id = input.partition_id;
                                    let loss = input.loss;
                                    let _cpu_permit =
                                        cpu_permits.acquire_owned().await.map_err(|_| {
                                            Error::internal(
                                                "partition build CPU semaphore was closed",
                                            )
                                        })?;
                                    let built = spawn_cpu(move || -> Result<_> {
                                        let num_rows = input
                                            .batches
                                            .iter()
                                            .map(|batch| batch.num_rows())
                                            .sum::<usize>();
                                        if num_rows == 0 {
                                            return Ok(None);
                                        }
                                        let (storage, sub_index) = Self::build_index(
                                            distance_type,
                                            quantizer,
                                            sub_index_params,
                                            input.batches,
                                            column,
                                            frag_reuse_index,
                                        )?;
                                        Ok(Some((storage, sub_index, loss)))
                                    })
                                    .await?;
                                    Ok::<_, Error>((
                                        PartitionBuildResult {
                                            partition_id,
                                            built,
                                        },
                                        entry_permit,
                                    ))
                                }
                            })
                            .try_buffer_unordered(concurrency)
                            .boxed();
                        Ok::<(FreshWindowBuildStream<S, Q>, _), Error>((builds, admission))
                    },
                );
                Ok(Some((job, next_partition_id)))
            }
        })
        .boxed();

        let windows = BoundedPartitionStream::try_new(
            jobs,
            concurrency,
            limits.decoded_budget_bytes,
            // One admission entry per outstanding window. Together with each
            // window's entry limit above, this bounds partition results by
            // `max_entries` even after an inner stream has finished.
            concurrency,
        )?;
        Ok(windows
            .map_ok(|window| {
                let Budgeted {
                    value: builds,
                    permit,
                    entry_permit,
                } = window;
                debug_assert!(entry_permit.is_none());
                builds.map_ok(move |(value, entry_permit)| Budgeted {
                    value,
                    permit: permit.clone(),
                    entry_permit: Some(entry_permit),
                })
            })
            .try_flatten_unordered(Some(concurrency))
            .boxed())
    }

    #[instrument(name = "build_index", level = "debug", skip_all)]
    #[allow(clippy::too_many_arguments)]
    fn build_index(
        distance_type: DistanceType,
        quantizer: Q,
        sub_index_params: S::BuildParams,
        batches: Vec<RecordBatch>,
        column: String,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    ) -> Result<(Q::Storage, S)> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(CompactFragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        let storage =
            StorageBuilder::new_with_remapper(column, distance_type, quantizer, frag_reuse_index)?
                .build(batches)?;
        let sub_index = S::index_vectors(&storage, sub_index_params)?;

        Ok((storage, sub_index))
    }

    #[instrument(name = "take_partition_batches", level = "debug", skip_all)]
    async fn take_partition_batches(
        part_id: usize,
        existing_indices: &[ExistingIndex],
        reader: Option<&dyn ShuffleReader>,
    ) -> Result<(Vec<RecordBatch>, f64)> {
        let mut batches = Vec::new();
        for source in existing_indices.iter() {
            let existing_index = source
                .index
                .as_any()
                .downcast_ref::<IVFIndex<S, Q>>()
                .ok_or(Error::invalid_input("existing index is not IVF index"))?;

            // Skip if this partition doesn't exist in the existing index
            // This can happen after a split creates a new partition
            if part_id >= existing_index.ivf_model().num_partitions() {
                continue;
            }

            // Resolved before the partition is decoded: partitions are built
            // concurrently, so whichever one builds the filter would otherwise hold a
            // decoded partition per in-flight task while the rest wait on it.
            let old_data_filter = source.old_data_filter().await?;

            let part_storage = existing_index.load_partition_storage(part_id, None).await?;
            let mut part_batches = part_storage.to_batches()?.collect::<Vec<_>>();
            // for PQ, the PQ codes are transposed, so we need to transpose them back
            match Q::quantization_type() {
                QuantizationType::Product => {
                    for batch in part_batches.iter_mut() {
                        if batch.num_rows() == 0 {
                            continue;
                        }

                        let codes = batch[PQ_CODE_COLUMN]
                            .as_fixed_size_list()
                            .values()
                            .as_primitive::<datatypes::UInt8Type>();
                        let codes_num_bytes = codes.len() / batch.num_rows();
                        let original_codes = transpose(codes, codes_num_bytes, batch.num_rows());
                        let original_codes = FixedSizeListArray::try_new_from_values(
                            original_codes,
                            codes_num_bytes as i32,
                        )?;
                        *batch = batch
                            .replace_column_by_name(PQ_CODE_COLUMN, Arc::new(original_codes))?
                            .drop_column(PART_ID_COLUMN)?;
                    }
                }
                QuantizationType::Rabit => {
                    for batch in part_batches.iter_mut() {
                        if batch.num_rows() == 0 {
                            continue;
                        }

                        let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list();
                        let original_codes = unpack_codes(codes);
                        *batch = batch
                            .replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(original_codes))?
                            .drop_column(PART_ID_COLUMN)?;
                    }
                }
                _ => {}
            }

            // Drop rows this segment may no longer contribute. They are physically
            // still in its index file, and the merged segment covers their live copies
            // again, so keeping them would emit the same row twice.
            if let Some(filter) = old_data_filter {
                for batch in part_batches.iter_mut() {
                    let keep = filter.filter_row_ids(batch[ROW_ID].as_primitive::<UInt64Type>());
                    if keep.true_count() < batch.num_rows() {
                        *batch = arrow::compute::filter_record_batch(batch, &keep)?;
                    }
                }
            }

            batches.extend(part_batches);
        }

        let mut loss = 0.0;
        // Skip if this partition doesn't exist in the reader
        // This can happen after a split creates a new partition
        if let Some(reader) = reader
            && reader.partition_size(part_id)? > 0
        {
            let mut partition_data =
                reader
                    .read_partition(part_id)
                    .await?
                    .ok_or(Error::invalid_input(format!(
                        "partition {} is empty",
                        part_id
                    )))?;
            while let Some(batch) = partition_data.try_next().await? {
                loss += batch
                    .metadata()
                    .get(LOSS_METADATA_KEY)
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0);
                batches.push(batch.drop_column(PART_ID_COLUMN)?);
            }
        }

        Ok((batches, loss))
    }

    #[instrument(name = "merge_partitions", level = "debug", skip_all)]
    async fn merge_partitions(
        &mut self,
        mut build_stream: BuildStream<S, Q>,
    ) -> Result<Vec<IndexFile>> {
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input("IVF not set before merge partitions"));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before merge partitions",
            ));
        };

        let quantization_type = Q::quantization_type();
        let is_pq = quantization_type == QuantizationType::Product;
        let is_rq = quantization_type == QuantizationType::Rabit;
        let is_flat = quantization_type == QuantizationType::Flat;

        // prepare the final writers
        let storage_path = self.index_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        let index_path = self.index_dir.clone().join(INDEX_FILE_NAME);

        let writer_options = FileWriterOptions::default();
        let mut storage_writer = if is_flat {
            None
        } else {
            let mut fields = vec![ROW_ID_FIELD.clone(), quantizer.field()];
            fields.extend(quantizer.extra_fields());
            let storage_schema: Schema = (&arrow_schema::Schema::new(fields)).try_into()?;
            Some(file_versions::create_writer(
                self.format_version,
                self.store.create(&storage_path).await?,
                storage_schema,
                writer_options.clone(),
            )?)
        };
        let mut index_writer = file_versions::create_writer(
            self.format_version,
            self.store.create(&index_path).await?,
            S::schema().as_ref().try_into()?,
            writer_options.clone(),
        )?;

        // maintain the IVF partitions
        let mut storage_ivf = IvfModel::empty();
        let mut index_ivf = IvfModel::new(ivf.centroids.clone().unwrap(), ivf.loss);
        let mut partition_index_metadata = Vec::with_capacity(ivf.num_partitions());

        let num_partitions = ivf.num_partitions();
        let mut ordered_results = OrderedPartitionResults::new(num_partitions);
        let mut total_loss = 0.0;
        let progress = self.progress.clone();
        log::info!("merging {} partitions", num_partitions);
        while let Some(result) = build_stream.try_next().await? {
            let partition_id = result.value.partition_id;
            ordered_results.push(partition_id, result)?;

            while let Some((partition_id, result)) = ordered_results.pop_next() {
                let Budgeted {
                    value: PartitionBuildResult { built: part, .. },
                    permit: _permit,
                    entry_permit: _entry_permit,
                } = result;
                let completed_partitions = partition_id + 1;
                progress
                    .stage_progress("merge_partitions", completed_partitions as u64)
                    .await?;
                let Some((storage, index, loss)) = part else {
                    log::warn!("partition {} is empty, skipping", partition_id);

                    storage_ivf.add_partition(0);
                    index_ivf.add_partition(0);
                    partition_index_metadata.push(String::new());

                    continue;
                };
                total_loss += loss;

                if storage.len() == 0 {
                    storage_ivf.add_partition(0);
                } else {
                    for mut batch in storage.to_batches()? {
                        if is_pq
                            && !self.transpose_codes
                            && batch.num_rows() > 0
                            && batch.column_by_name(PQ_CODE_COLUMN).is_some()
                        {
                            let codes_fsl = batch
                                .column_by_name(PQ_CODE_COLUMN)
                                .unwrap()
                                .as_fixed_size_list();
                            let num_rows = batch.num_rows();
                            let bytes_per_code = codes_fsl.value_length() as usize;
                            let codes = codes_fsl.values().as_primitive::<datatypes::UInt8Type>();
                            let original_codes = transpose(codes, bytes_per_code, num_rows);
                            let original_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                                original_codes,
                                bytes_per_code as i32,
                            )?);
                            batch = batch.replace_column_by_name(PQ_CODE_COLUMN, original_fsl)?;
                        }

                        if is_rq
                            && !self.transpose_codes
                            && batch.num_rows() > 0
                            && batch.column_by_name(RABIT_CODE_COLUMN).is_some()
                        {
                            let codes_fsl = batch
                                .column_by_name(RABIT_CODE_COLUMN)
                                .unwrap()
                                .as_fixed_size_list();
                            let unpacked = Arc::new(unpack_codes(codes_fsl));
                            batch = batch.replace_column_by_name(RABIT_CODE_COLUMN, unpacked)?;
                        }

                        if storage_writer.is_none() {
                            let storage_schema: Schema = batch.schema_ref().as_ref().try_into()?;
                            storage_writer = Some(file_versions::create_writer(
                                self.format_version,
                                self.store.create(&storage_path).await?,
                                storage_schema,
                                writer_options.clone(),
                            )?);
                        }
                        storage_writer
                            .as_mut()
                            .expect("storage writer must be initialized before write")
                            .write_batch(&batch)
                            .await?;
                        storage_ivf.add_partition(batch.num_rows() as u32);
                    }
                }

                let index_batch = index.to_batch()?;
                if index_batch.num_rows() == 0 {
                    index_ivf.add_partition(0);
                    partition_index_metadata.push(String::new());
                } else {
                    index_writer.write_batch(&index_batch).await?;
                    index_ivf.add_partition(index_batch.num_rows() as u32);
                    partition_index_metadata.push(
                        index_batch
                            .schema()
                            .metadata
                            .get(S::metadata_key())
                            .cloned()
                            .unwrap_or_default(),
                    );
                }
            }
        }

        ordered_results.finish()?;

        match self.shuffle_reader.as_ref() {
            Some(reader) => {
                // it's building index, the loss is already calculated in the shuffle reader
                if let Some(loss) = reader.total_loss() {
                    total_loss += loss;
                }
                index_ivf.loss = Some(total_loss);
            }
            None => {
                // it's remapping, we don't need to change the loss
            }
        }

        if storage_writer.is_none() {
            let Some(centroids) = ivf.centroids.as_ref() else {
                return Err(Error::invalid_input(
                    "flat storage writer could not infer schema from empty partitions without IVF centroids",
                ));
            };
            let flat_schema = arrow_schema::Schema::new(vec![
                ROW_ID_FIELD.as_ref().clone(),
                arrow_schema::Field::new(
                    lance_index::vector::flat::storage::FLAT_COLUMN,
                    DataType::FixedSizeList(
                        Arc::new(arrow_schema::Field::new(
                            "item",
                            centroids.value_type(),
                            true,
                        )),
                        centroids.value_length(),
                    ),
                    true,
                ),
            ]);
            let storage_schema: Schema = (&flat_schema).try_into()?;
            storage_writer = Some(file_versions::create_writer(
                self.format_version,
                self.store.create(&storage_path).await?,
                storage_schema,
                writer_options.clone(),
            )?);
        }

        let storage_writer = storage_writer
            .as_mut()
            .expect("storage writer must be initialized before final metadata write");
        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, self.distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        let transposed = match quantization_type {
            QuantizationType::Product | QuantizationType::Rabit => self.transpose_codes,
            _ => false,
        };
        // For now, each partition's metadata is just the quantizer,
        // it's all the same for now, so we just take the first one
        let mut metadata = quantizer.metadata(Some(QuantizationMetadata {
            codebook_position: Some(0),
            codebook: None,
            transposed,
        }));
        if let Some(extra_metadata) = metadata.extra_metadata()? {
            let idx = storage_writer.add_global_buffer(extra_metadata).await?;
            metadata.set_buffer_index(idx);
        }
        let metadata = serde_json::to_string(&metadata)?;
        let storage_partition_metadata = vec![metadata];
        storage_writer.add_schema_metadata(
            STORAGE_METADATA_KEY,
            serde_json::to_string(&storage_partition_metadata)?,
        );

        let index_type_str = index_type_string(S::name().try_into()?, Q::quantization_type());
        if let Some(idx_type) = SupportedIvfIndexType::from_index_type_str(&index_type_str) {
            write_unified_ivf_and_index_metadata(
                &mut index_writer,
                &index_ivf,
                self.distance_type,
                idx_type,
            )
            .await?;
        } else {
            // Fallback for index types not covered by SupportedIndexType (e.g. IVF_RQ).
            let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
            let index_metadata = IndexMetadata {
                index_type: index_type_str,
                distance_type: self.distance_type.to_string(),
            };
            index_writer.add_schema_metadata(
                INDEX_METADATA_SCHEMA_KEY,
                serde_json::to_string(&index_metadata)?,
            );
            let ivf_buffer_pos = index_writer
                .add_global_buffer(index_ivf_pb.encode_to_vec().into())
                .await?;
            index_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        }
        index_writer.add_schema_metadata(
            S::metadata_key(),
            serde_json::to_string(&partition_index_metadata)?,
        );

        let storage_summary = storage_writer.finish().await?;
        let index_summary = index_writer.finish().await?;

        log::info!("merging {} partitions done", ivf.num_partitions());

        Ok(vec![
            IndexFile {
                path: INDEX_AUXILIARY_FILE_NAME.to_string(),
                size_bytes: storage_summary.size_bytes,
            },
            IndexFile {
                path: INDEX_FILE_NAME.to_string(),
                size_bytes: index_summary.size_bytes,
            },
        ])
    }

    // take raw vectors from the dataset
    //
    // returns batches of schema | row_id | vector |
    async fn take_vectors(
        dataset: &Dataset,
        column: &str,
        store: &ObjectStore,
        row_ids: &[u64],
    ) -> Result<Vec<RecordBatch>> {
        let projection = Arc::new(dataset.schema().project(&[column])?);
        // arrow uses i32 for index, so we chunk the row ids to avoid large batch causing overflow
        let row_ids = dataset.filter_deleted_ids(row_ids).await?;
        let chunks: Vec<Vec<u64>> = row_ids
            .chunks(store.block_size())
            .map(|chunk| chunk.to_vec())
            .collect();
        let batches = stream::iter(chunks)
            .map(|chunk| {
                let dataset = dataset.clone();
                let projection = projection.clone();
                async move {
                    let batch = dataset
                        .take_rows(&chunk, ProjectionRequest::Schema(projection))
                        .await?;
                    if batch.num_rows() != chunk.len() {
                        return Err(Error::invalid_input(format!(
                            "batch.num_rows() != chunk.len() ({} != {})",
                            batch.num_rows(),
                            chunk.len()
                        )));
                    }
                    Ok(batch.try_with_column(
                        ROW_ID_FIELD.clone(),
                        Arc::new(UInt64Array::from(chunk)),
                    )?)
                }
            })
            .buffered(store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;
        Ok(batches)
    }

    // helper to load row ids and vectors for a partition
    async fn load_partition_raw_vectors(
        &self,
        part_idx: usize,
    ) -> Result<Option<(UInt64Array, FixedSizeListArray)>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before split partition",
            ));
        };

        let mut row_ids = self.partition_row_ids(part_idx).await?;
        if !row_ids.is_sorted() {
            row_ids.sort();
        }
        // dedup is needed if it's multivector
        row_ids.dedup();

        let batches = Self::take_vectors(dataset, &self.column, &self.store, &row_ids).await?;
        if batches.is_empty() {
            return Ok(None);
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
        // for multivector, we need to flatten the vectors
        let batch = Flatten::new(&self.column).transform(&batch)?;
        // need to retrieve the row ids from the batch because some rows may have been deleted
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
        let vectors = batch
            .column_by_qualified_name(&self.column)
            .ok_or(Error::invalid_input(format!(
                "vector column {} not found in batch {}",
                self.column,
                batch.schema()
            )))?
            .as_fixed_size_list()
            .clone();
        Ok(Some((row_ids, vectors)))
    }

    /// Rows per partition that split and join thresholds are derived from: the
    /// value the index was created with when known, else the type default.
    fn effective_target_partition_size(&self) -> Result<usize> {
        if let Some(size) = self.target_partition_size {
            return Ok(size);
        }
        let index_type = IndexType::try_from(
            index_type_string(S::name().try_into()?, Q::quantization_type()).as_str(),
        )?;
        Ok(index_type.target_partition_size())
    }

    /// Decide which partitions to split and which to join away.
    ///
    /// A partition above `MAX_PARTITION_SIZE_FACTOR` times the target is split
    /// straight to the target size (`ceil(rows / target)` ways, capped at
    /// `MAX_SPLIT_WAYS`), so one pass leaves nothing above the threshold. Every
    /// partition below `MIN_PARTITION_SIZE_PERCENT` of the target is joined away,
    /// except that the largest of them stays when all partitions are undersized.
    fn check_partition_adjustment(
        ivf: &IvfModel,
        reader: &dyn ShuffleReader,
        existing_indices: &[ExistingIndex],
        target_partition_size: usize,
    ) -> Result<(Vec<PartitionSplit>, Vec<usize>)> {
        let split_threshold = MAX_PARTITION_SIZE_FACTOR * target_partition_size;
        let join_threshold = MIN_PARTITION_SIZE_PERCENT * target_partition_size / 100;

        let mut splits = Vec::new();
        let mut joins = Vec::new();
        for partition in 0..ivf.num_partitions() {
            let mut num_rows = reader.partition_size(partition)?;
            for source in existing_indices.iter() {
                num_rows += source.index.partition_size(partition);
            }
            if num_rows > split_threshold {
                splits.push(PartitionSplit {
                    partition,
                    ways: num_rows
                        .div_ceil(target_partition_size)
                        .clamp(2, MAX_SPLIT_WAYS),
                });
            } else if ivf.num_partitions() > 1 && num_rows < join_threshold {
                joins.push((num_rows, partition));
            }
        }
        if joins.len() == ivf.num_partitions() {
            let largest = joins
                .iter()
                .enumerate()
                .max_by_key(|(_, (num_rows, _))| *num_rows)
                .map(|(i, _)| i)
                .expect("joins is not empty");
            joins.remove(largest);
        }
        let joins = joins.into_iter().map(|(_, partition)| partition).collect();

        Ok((splits, joins))
    }

    /// Split oversized partitions using a streaming approach.
    ///
    /// 1. Train new centroids by sampling vectors (low memory).
    /// 2. Compute the set of affected partitions (split targets + their neighbors).
    /// 3. Stream raw vectors for affected partitions through the IVF+quantizer
    ///    transform pipeline, writing to temp files via a shuffler.
    ///
    /// Returns `None` when no partition could be split (none had live rows), so
    /// the caller does not pay for a merge that changes nothing.
    async fn split_partitions_streaming(
        &self,
        splits: &[PartitionSplit],
        ivf: &IvfModel,
    ) -> Result<Option<SplitResult>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before split partition",
            ));
        };

        let (_, element_type) = get_vector_type(dataset.schema(), &self.column)?;
        let (new_centroids, split_partitions) = match element_type {
            DataType::Float16 => {
                self.compute_split_centroids::<Float16Type>(splits, ivf)
                    .await?
            }
            DataType::Float32 => {
                self.compute_split_centroids::<Float32Type>(splits, ivf)
                    .await?
            }
            DataType::Float64 => {
                self.compute_split_centroids::<Float64Type>(splits, ivf)
                    .await?
            }
            DataType::UInt8 => {
                self.compute_split_centroids::<UInt8Type>(splits, ivf)
                    .await?
            }
            dt => {
                return Err(Error::invalid_input(format!(
                    "vectors must be float16, float32, float64 or uint8, but got {:?}",
                    dt
                )));
            }
        };

        if split_partitions.is_empty() {
            return Ok(None);
        }

        // Affected partitions: every partition that received a new centroid plus
        // those neighbors of the split partitions' old centroids that can lose a
        // row to a new centroid.
        let mut affected_partitions = HashSet::new();
        let mut new_partition_ids = Vec::new();
        let mut candidates = HashSet::new();
        for (part_idx, new_partitions) in &split_partitions {
            affected_partitions.extend(new_partitions.iter().copied());
            new_partition_ids.extend(new_partitions.iter().map(|id| *id as u32));
            let c0 = ivf.centroid(*part_idx).ok_or(Error::invalid_input(format!(
                "centroid not found for partition {part_idx}",
            )))?;
            let (neighbor_ids, _) = select_reassign_candidates_impl(
                self.distance_type,
                ivf,
                *part_idx,
                &c0,
                &HashSet::new(),
            )?;
            candidates.extend(neighbor_ids.values().iter().map(|id| *id as usize));
        }
        let mut candidates: Vec<usize> = candidates
            .difference(&affected_partitions)
            .copied()
            .collect();
        candidates.sort_unstable();
        let split_centroids =
            arrow::compute::take(&new_centroids, &UInt32Array::from(new_partition_ids), None)?
                .as_fixed_size_list()
                .clone();
        let kept = self
            .prune_reassign_candidates(&candidates, &new_centroids, &split_centroids)
            .await?;
        log::info!(
            "split {} partitions into {} partitions; {} of {} neighbor partitions can lose rows to the new centroids and are reassigned, {} total affected partitions",
            split_partitions.len(),
            new_partition_ids_len(&split_partitions),
            kept.len(),
            candidates.len(),
            affected_partitions.len() + kept.len(),
        );
        affected_partitions.extend(kept);

        // Stream raw vectors for affected partitions through the IVF+quantizer
        // transform, writing to temp files via a second shuffler.
        let split_shuffle_reader = self
            .reshuffle_partitions(&affected_partitions, &new_centroids)
            .await?;

        Ok(Some(SplitResult {
            new_centroids,
            affected_partitions,
            shuffle_reader: split_shuffle_reader.into(),
        }))
    }

    /// Of the neighbor partitions of a split, keep those where a sampled row is
    /// now closer to one of the new centroids than to its own centroid, within
    /// `REASSIGN_MARGIN` (SPFresh's necessary condition for a row to move). The
    /// others keep their rows and codes untouched, which is what saves the
    /// raw-vector reads: most neighbors of a split cannot lose a row.
    async fn prune_reassign_candidates(
        &self,
        candidates: &[usize],
        centroids: &FixedSizeListArray,
        split_centroids: &FixedSizeListArray,
    ) -> Result<Vec<usize>> {
        let checks = stream::iter(candidates.iter().copied())
            .map(|candidate| async move {
                let Some(sample) = self
                    .sample_partition_raw_vectors(candidate, REASSIGN_SAMPLE_SIZE)
                    .await?
                else {
                    return Ok::<_, Error>(None);
                };
                let dist_fn = self.distance_type.arrow_batch_func();
                let own_centroid = centroids.slice(candidate, 1);
                for i in 0..sample.len() {
                    let vector = sample.value(i);
                    let own_dist = dist_fn(&vector, &own_centroid)?.value(0);
                    let new_dist = dist_fn(&vector, split_centroids)?
                        .values()
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, f32::min);
                    if new_dist < own_dist + REASSIGN_MARGIN * own_dist.abs() {
                        return Ok(Some(candidate));
                    }
                }
                Ok(None)
            })
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;
        Ok(checks.into_iter().flatten().collect())
    }

    /// Train new centroids for partitions that need splitting.
    ///
    /// Returns the full updated centroids array and, for every partition that was
    /// actually split (partitions without live rows are skipped), the ids of the
    /// partitions its rows now belong to: the original id followed by the appended
    /// ones.
    async fn compute_split_centroids<T: ArrowPrimitiveType>(
        &self,
        splits: &[PartitionSplit],
        ivf: &IvfModel,
    ) -> Result<(FixedSizeListArray, Vec<(usize, Vec<usize>)>)>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let centroids = ivf.centroids_array().unwrap();

        // Train split centroids in parallel (low memory: only samples).
        let trained_centroids = stream::iter(splits.iter().copied())
            .map(|split| async move { self.train_split_centroids::<T>(split).await })
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;

        let mut applied = Vec::new();
        let mut split_partitions = Vec::new();
        let mut next_partition = centroids.len();
        for (split, trained) in splits.iter().zip(trained_centroids) {
            let Some(trained) = trained else {
                continue;
            };
            let appended = next_partition..next_partition + trained.len() - 1;
            next_partition = appended.end;
            let mut partitions = vec![split.partition];
            partitions.extend(appended);
            split_partitions.push((split.partition, partitions));
            applied.push((split.partition, trained));
        }

        if applied.is_empty() {
            return Ok((centroids.clone(), vec![]));
        }

        let new_centroids = apply_centroid_splits(centroids, &applied)?;
        Ok((new_centroids, split_partitions))
    }

    /// Stream raw vectors for the given partitions through the IVF+quantizer
    /// transform, writing to temp files. Returns a ShuffleReader for the results.
    async fn reshuffle_partitions(
        &self,
        affected_partitions: &HashSet<usize>,
        new_centroids: &FixedSizeListArray,
    ) -> Result<Box<dyn ShuffleReader>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input("dataset not set before reshuffle"));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input("quantizer not set before reshuffle"));
        };

        // Collect all row IDs for affected partitions, dedup, sort.
        let mut all_row_ids = Vec::new();
        for &part_idx in affected_partitions {
            let mut row_ids = self.partition_row_ids(part_idx).await?;
            all_row_ids.append(&mut row_ids);
        }
        all_row_ids.sort();
        all_row_ids.dedup();

        // Stream raw vectors in chunks
        let projection = Arc::new(dataset.schema().project(&[self.column.as_str()])?);
        let row_ids = dataset.filter_deleted_ids(&all_row_ids).await?;
        let block_size = self.store.block_size();
        let io_parallelism = self.store.io_parallelism();
        let column = self.column.clone();

        let dataset_clone = dataset.clone();
        let projection_clone = projection.clone();
        let raw_stream = stream::iter(
            row_ids
                .chunks(block_size)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map(move |chunk| {
            let dataset = dataset_clone.clone();
            let projection = projection_clone.clone();
            let column = column.clone();
            async move {
                let batch = dataset
                    .take_rows(&chunk, ProjectionRequest::Schema(projection))
                    .await?;
                let batch = batch
                    .try_with_column(ROW_ID_FIELD.clone(), Arc::new(UInt64Array::from(chunk)))?;
                // For multivector, flatten
                Flatten::new(&column).transform(&batch)
            }
        })
        .buffered(io_parallelism)
        .boxed();

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                new_centroids.clone(),
                self.distance_type,
                &self.column,
                quantizer.into(),
                None,
            )?,
        );

        let mut transformed_stream = Box::pin(
            raw_stream
                .map(move |batch| {
                    let ivf_transformer = transformer.clone();
                    tokio::spawn(async move { ivf_transformer.transform(&batch?) })
                })
                .buffered(get_num_compute_intensive_cpus())
                .map(|x| x.unwrap())
                .peekable(),
        );

        // Peek transformed stream to get schema (includes PART_ID + PQ codes)
        let schema = match transformed_stream.as_mut().peek_mut().await {
            Some(Ok(b)) => b.schema(),
            Some(Err(e)) => return Err(std::mem::replace(e, Error::Stop)),
            None => {
                log::info!("no vectors to reshuffle");
                let empty_reader: Box<dyn ShuffleReader> = Box::new(IvfShufflerReader::new(
                    Arc::new(self.store.clone()),
                    self.temp_dir.clone().join("split_shuffle"),
                    vec![0; new_centroids.len()],
                    0.0,
                ));
                return Ok(empty_reader);
            }
        };

        let transformed_stream =
            Box::new(RecordBatchStreamAdapter::new(schema, transformed_stream));

        let split_shuffle_dir = self.temp_dir.clone().join("split_shuffle");
        let shuffler = create_ivf_shuffler(
            split_shuffle_dir,
            new_centroids.len(),
            self.format_version,
            None,
        );
        shuffler.shuffle(transformed_stream).await
    }

    /// Sample raw vectors from a partition for kmeans training.
    ///
    /// Samples row IDs first, then only loads `sample_size` vectors from disk.
    async fn sample_partition_raw_vectors(
        &self,
        part_idx: usize,
        sample_size: usize,
    ) -> Result<Option<FixedSizeListArray>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before sample partition",
            ));
        };

        let mut row_ids = self.partition_row_ids(part_idx).await?;
        if !row_ids.is_sorted() {
            row_ids.sort();
        }
        row_ids.dedup();

        if row_ids.is_empty() {
            return Ok(None);
        }

        // Sample row ids before loading any vectors. Uniform rather than strided:
        // row ids follow insertion order, so a stride would favour old rows.
        // Seeded per partition so a rerun trains the same split.
        if row_ids.len() > sample_size {
            let mut rng = SmallRng::seed_from_u64(part_idx as u64);
            let mut chosen =
                rand::seq::index::sample(&mut rng, row_ids.len(), sample_size).into_vec();
            chosen.sort_unstable();
            row_ids = chosen.into_iter().map(|i| row_ids[i]).collect();
        }

        let batches = Self::take_vectors(dataset, &self.column, &self.store, &row_ids).await?;
        if batches.is_empty() {
            return Ok(None);
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
        let batch = Flatten::new(&self.column).transform(&batch)?;
        let vectors = batch
            .column_by_qualified_name(&self.column)
            .ok_or(Error::invalid_input(format!(
                "vector column {} not found in batch",
                self.column,
            )))?
            .as_fixed_size_list()
            .clone();
        Ok(Some(vectors))
    }

    /// Train the centroids that split one partition `split.ways` ways. Only a
    /// sample of `SPLIT_SAMPLE_RATE` vectors per centroid is read, so this is
    /// cheap enough to run in parallel.
    ///
    /// Returns `None` when the partition has no live rows, and fewer centroids
    /// than requested when it has fewer distinct rows than that.
    async fn train_split_centroids<T: ArrowPrimitiveType>(
        &self,
        split: PartitionSplit,
    ) -> Result<Option<Vec<ArrayRef>>>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let Some(vectors) = self
            .sample_partition_raw_vectors(split.partition, SPLIT_SAMPLE_RATE * split.ways)
            .await?
        else {
            return Ok(None);
        };
        let ways = split.ways.min(vectors.len());
        if ways < 2 {
            return Ok(None);
        }

        let dimension = infer_vector_dim(vectors.data_type())?;
        let (normalized_dist_type, normalized_vectors) = match self.distance_type {
            DistanceType::Cosine => {
                let vectors = normalize_fsl(&vectors)?;
                (DistanceType::L2, vectors)
            }
            _ => (self.distance_type, vectors),
        };
        // Balanced like IVF training itself, so the pieces come out even.
        let params = KMeansParams::new(None, 50, 1, normalized_dist_type).with_balance_factor(1.0);
        let kmeans = lance_index::vector::kmeans::train_kmeans::<T>(
            normalized_vectors.values().as_primitive::<T>(),
            params,
            dimension,
            ways,
            SPLIT_SAMPLE_RATE,
        )?;

        // A centroid no sampled row is nearest to would only produce an empty
        // partition (near-duplicate rows collapse onto one centroid). Keep the
        // centroids that attract rows, and leave the partition alone when fewer
        // than two do: it cannot be split by clustering.
        let (membership, _) = kmeans.compute_membership_and_distances(&normalized_vectors)?;
        let mut attracts_rows = vec![false; ways];
        for cluster in membership.into_iter().flatten() {
            attracts_rows[cluster as usize] = true;
        }
        let centroids: Vec<ArrayRef> = (0..ways)
            .filter(|&i| attracts_rows[i])
            .map(|i| kmeans.centroids.slice(i * dimension, dimension))
            .collect();
        if centroids.len() < 2 {
            log::warn!(
                "partition {} is oversized but its sampled rows are too alike to split; leaving it as is",
                split.partition
            );
            return Ok(None);
        }
        Ok(Some(centroids))
    }

    /// Join undersized partitions away in one pass: their centroids are removed
    /// and every one of their vectors is reassigned to the nearest remaining
    /// partition among the `REASSIGN_RANGE` nearest neighbors of its old centroid.
    async fn join_partitions(&self, partitions: &[usize], ivf: &IvfModel) -> Result<JoinResult> {
        let removed: HashSet<usize> = partitions.iter().copied().collect();
        let kept_partitions: Vec<usize> = (0..ivf.num_partitions())
            .filter(|partition| !removed.contains(partition))
            .collect();
        let centroids = ivf.centroids_array().unwrap();
        let kept_ids = UInt32Array::from(
            kept_partitions
                .iter()
                .map(|partition| *partition as u32)
                .collect::<Vec<_>>(),
        );
        let new_centroids = arrow::compute::take(centroids, &kept_ids, None)?
            .as_fixed_size_list()
            .clone();

        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before joining partitions",
            ));
        };
        let (_, element_type) = get_vector_type(dataset.schema(), &self.column)?;
        let assign_batches = match element_type {
            DataType::Float16 => {
                self.join_partitions_impl::<Float16Type>(
                    partitions,
                    &kept_partitions,
                    ivf,
                    &new_centroids,
                )
                .await?
            }
            DataType::Float32 => {
                self.join_partitions_impl::<Float32Type>(
                    partitions,
                    &kept_partitions,
                    ivf,
                    &new_centroids,
                )
                .await?
            }
            DataType::Float64 => {
                self.join_partitions_impl::<Float64Type>(
                    partitions,
                    &kept_partitions,
                    ivf,
                    &new_centroids,
                )
                .await?
            }
            DataType::UInt8 => {
                self.join_partitions_impl::<UInt8Type>(
                    partitions,
                    &kept_partitions,
                    ivf,
                    &new_centroids,
                )
                .await?
            }
            dt => {
                return Err(Error::invalid_input(format!(
                    "vectors must be float16, float32, float64 or uint8, but got {:?}",
                    dt
                )));
            }
        };

        Ok(JoinResult {
            assign_batches,
            new_centroids,
            kept_partitions,
        })
    }

    async fn join_partitions_impl<T: ArrowPrimitiveType>(
        &self,
        partitions: &[usize],
        kept_partitions: &[usize],
        ivf: &IvfModel,
        new_centroids: &FixedSizeListArray,
    ) -> Result<Vec<Option<(RecordBatch, UInt64Array)>>>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let removed: HashSet<usize> = partitions.iter().copied().collect();
        // old partition id -> output partition id
        let mut output_partition = vec![None; ivf.num_partitions()];
        for (output, &old) in kept_partitions.iter().enumerate() {
            output_partition[old] = Some(output);
        }

        let mut assign_ops = vec![Vec::new(); kept_partitions.len()];
        for &part_idx in partitions {
            let Some((row_ids, vectors)) = self.load_partition_raw_vectors(part_idx).await? else {
                continue;
            };
            assert_eq!(row_ids.len(), vectors.len());
            let c0 = ivf
                .centroid(part_idx)
                .ok_or(Error::invalid_input("original centroid not found"))?;
            let (reassign_part_ids, reassign_part_centroids) =
                self.select_reassign_candidates(ivf, part_idx, &c0, &removed)?;

            for (i, &row_id) in row_ids.values().iter().enumerate() {
                let ReassignPartition::ReassignCandidate(target) = self.reassign_vectors(
                    vectors.value(i).as_primitive::<T>(),
                    None,
                    &reassign_part_ids,
                    &reassign_part_centroids,
                )?
                else {
                    log::warn!("this is a bug, the vector is not reassigned");
                    continue;
                };
                let output = output_partition[target as usize].ok_or_else(|| {
                    Error::internal(format!(
                        "partition {target} is being joined away but was selected to receive vectors of partition {part_idx}"
                    ))
                })?;
                assign_ops[output].push(AssignOp::Add((row_id, vectors.value(i))));
            }
        }

        self.build_assign_batch::<T>(new_centroids, &assign_ops)
    }

    // Build the assign batch form assign ops for each partition
    // returns the assign batch and the deleted row ids
    fn build_assign_batch<T: ArrowPrimitiveType>(
        &self,
        centroids: &FixedSizeListArray,
        assign_ops: &[Vec<AssignOp>],
    ) -> Result<Vec<Option<(RecordBatch, UInt64Array)>>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before building assign batch",
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before building assign batch",
            ));
        };

        let Some(vector_field) =
            dataset
                .schema()
                .field(&self.column)
                .map(|f| match f.data_type() {
                    DataType::List(inner) | DataType::LargeList(inner) => {
                        Field::new(self.column.as_str(), inner.data_type().clone(), true)
                    }
                    _ => f.into(),
                })
        else {
            return Err(Error::invalid_input(
                "vector field not found in dataset schema",
            ));
        };

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                centroids.clone(),
                self.distance_type,
                vector_field.name().as_str(),
                quantizer.into(),
                None,
            )?,
        );

        let num_rows: usize = assign_ops.iter().map(|ops| ops.len()).sum();

        // build the input batch with schema | row_id | vector | part_id |
        let mut row_ids_builder = UInt64Builder::with_capacity(num_rows);
        let mut vector_builder =
            PrimitiveBuilder::<T>::with_capacity(num_rows * centroids.value_length() as usize);
        let mut part_ids_builder = UInt32Builder::with_capacity(num_rows);

        let mut counts = Vec::with_capacity(assign_ops.len());
        for (part_idx, ops) in assign_ops.iter().enumerate() {
            for AssignOp::Add((row_id, vector)) in ops {
                row_ids_builder.append_value(*row_id);
                vector_builder.append_array(vector.as_primitive::<T>());
                part_ids_builder.append_value(part_idx as u32);
            }
            counts.push(ops.len());
        }

        let row_ids = row_ids_builder.finish();
        let vector = FixedSizeListArray::try_new_from_values(
            vector_builder.finish(),
            centroids.value_length(),
        )?;
        let part_ids = part_ids_builder.finish();
        let schema = arrow_schema::Schema::new(vec![
            ROW_ID_FIELD.clone(),
            vector_field,
            PART_ID_FIELD.clone(),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(row_ids), Arc::new(vector), Arc::new(part_ids)],
        )?;
        let batch = transformer.transform(&batch)?;

        let empty_deleted = UInt64Array::from(Vec::<u64>::new());
        let mut results = Vec::with_capacity(assign_ops.len());
        let mut offset = 0;
        for count in counts {
            if count == 0 {
                results.push(None);
            } else {
                results.push(Some((batch.slice(offset, count), empty_deleted.clone())));
                offset += count;
            }
        }
        Ok(results)
    }

    async fn partition_row_ids(&self, part_idx: usize) -> Result<Vec<u64>> {
        // existing part: read from the existing indices
        let mut row_ids = Vec::new();
        for source in self.existing_indices.iter() {
            let index = &source.index;
            if part_idx >= index.ivf_model().num_partitions() {
                // there was a bug that may cause delta indices have different number of partitions,
                // it's safe to skip loading the extra partition, and split/join the existing partitions,
                // split/join would merge all delta indices into one so it would fix the issue
                // see https://github.com/lance-format/lance/issues/5312
                log::warn!(
                    "partition index is {} but the number of partitions is {}, skip loading it",
                    part_idx,
                    index.ivf_model().num_partitions()
                );
                continue;
            }
            let mut reader = index
                .partition_reader(part_idx, false, &NoOpMetricsCollector)
                .await?;
            let old_data_filter = source.old_data_filter().await?;
            while let Some(batch) = reader.try_next().await? {
                let batch_row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                match old_data_filter {
                    // Rows the segment may no longer contribute must not be reassigned
                    // to a partition; their live copy comes from another source.
                    Some(filter) => row_ids.extend(
                        batch_row_ids
                            .values()
                            .iter()
                            .zip(filter.filter_row_ids(batch_row_ids).values().iter())
                            .filter_map(|(row_id, keep)| keep.then_some(row_id)),
                    ),
                    None => row_ids.extend(batch_row_ids.values()),
                }
            }
        }

        // incremental part: read from the shuffler reader
        if let Some(reader) = self.shuffle_reader.as_ref() {
            // TODO: don't read vectors here, just read row ids
            if let Some(mut reader) = reader.read_partition(part_idx).await? {
                while let Some(batch) = reader.try_next().await? {
                    row_ids.extend(batch[ROW_ID].as_primitive::<UInt64Type>().values());
                }
            }
        }
        Ok(row_ids)
    }

    // returns the closest REASSIGN_RANGE partitions (indices and centroids) from c0
    fn select_reassign_candidates(
        &self,
        ivf: &IvfModel,
        part_idx: usize,
        c0: &ArrayRef,
        excluded: &HashSet<usize>,
    ) -> Result<(UInt32Array, FixedSizeListArray)> {
        select_reassign_candidates_impl(self.distance_type, ivf, part_idx, c0, excluded)
    }
    // assign a vector to the closest partition among:
    // 1. the 2 new centroids
    // 2. the closest REASSIGN_RANGE partitions from the original centroid
    fn reassign_vectors<T: ArrowPrimitiveType>(
        &self,
        vector: &PrimitiveArray<T>,
        // the dists to the 2 new centroids
        split_centroids_dists: Option<(f32, f32)>,
        reassign_candidate_ids: &UInt32Array,
        reassign_candidate_centroids: &FixedSizeListArray,
    ) -> Result<ReassignPartition> {
        Self::reassign_vectors_impl(
            self.distance_type,
            vector,
            split_centroids_dists,
            reassign_candidate_ids,
            reassign_candidate_centroids,
        )
    }

    fn reassign_vectors_impl<T: ArrowPrimitiveType>(
        distance_type: DistanceType,
        vector: &PrimitiveArray<T>,
        split_centroids_dists: Option<(f32, f32)>,
        reassign_candidate_ids: &UInt32Array,
        reassign_candidate_centroids: &FixedSizeListArray,
    ) -> Result<ReassignPartition> {
        let dists = distance_type.arrow_batch_func()(vector, reassign_candidate_centroids)?;
        let min_dist_idx = dists
            .values()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);
        let min_dist = min_dist_idx
            .map(|idx| dists.value(idx))
            .unwrap_or(f32::INFINITY);
        match split_centroids_dists {
            Some((d1, d2)) => {
                if min_dist <= d1 && min_dist <= d2 {
                    Ok(ReassignPartition::ReassignCandidate(
                        reassign_candidate_ids.value(min_dist_idx.unwrap()),
                    ))
                } else if d1 <= d2 {
                    Ok(ReassignPartition::NewCentroid1)
                } else {
                    Ok(ReassignPartition::NewCentroid2)
                }
            }
            None => Ok(ReassignPartition::ReassignCandidate(
                reassign_candidate_ids.value(min_dist_idx.unwrap()),
            )),
        }
    }
}

/// The nearest `REASSIGN_RANGE` partitions to `c0`, skipping `part_idx` itself and
/// every partition in `excluded` (partitions being joined away cannot receive
/// vectors). Non-empty whenever a partition outside `excluded` exists.
fn select_reassign_candidates_impl(
    distance_type: DistanceType,
    ivf: &IvfModel,
    part_idx: usize,
    c0: &ArrayRef,
    excluded: &HashSet<usize>,
) -> Result<(UInt32Array, FixedSizeListArray)> {
    let centroids = ivf.centroids_array().unwrap();
    let centroid_dists = distance_type.arrow_batch_func()(c0, centroids)?;
    // Fetch enough neighbors that the exclusions cannot empty the candidate set.
    let fetch = (REASSIGN_RANGE + excluded.len() + 1).min(ivf.num_partitions());
    let nearest = sort_to_indices(centroid_dists.as_ref(), None, Some(fetch))?;
    let filtered_ids = nearest
        .values()
        .iter()
        .copied()
        .filter(|&idx| idx as usize != part_idx && !excluded.contains(&(idx as usize)))
        .take(REASSIGN_RANGE)
        .collect::<Vec<_>>();
    let reassign_candidate_ids = UInt32Array::from(filtered_ids);
    let reassign_candidate_centroids =
        arrow::compute::take(centroids, &reassign_candidate_ids, None)?;
    Ok((
        reassign_candidate_ids,
        reassign_candidate_centroids.as_fixed_size_list().clone(),
    ))
}

struct JoinResult {
    assign_batches: Vec<Option<(RecordBatch, UInt64Array)>>,
    new_centroids: FixedSizeListArray,
    /// Old ids of the partitions that survive, in output order.
    kept_partitions: Vec<usize>,
}

struct SplitResult {
    new_centroids: FixedSizeListArray,
    affected_partitions: HashSet<usize>,
    shuffle_reader: Arc<dyn ShuffleReader>,
}

#[derive(Debug, Clone)]
enum AssignOp {
    Add((u64, ArrayRef)),
}

#[derive(Debug, Copy, Clone)]
enum ReassignPartition {
    NewCentroid1,
    NewCentroid2,
    ReassignCandidate(u32),
}

enum PartitionAdjustment {
    /// Split partitions. Carries the set of all affected partitions (split
    /// targets, the partitions appended for them and their neighbors) and a
    /// shuffle reader with re-quantized data.
    Split {
        affected_partitions: HashSet<usize>,
        split_shuffle_reader: Arc<dyn ShuffleReader>,
    },
    /// Join partitions away; `kept_partitions[i]` is the old id of output
    /// partition `i`.
    Join { kept_partitions: Vec<usize> },
}

impl std::fmt::Debug for PartitionAdjustment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Split {
                affected_partitions,
                ..
            } => f
                .debug_struct("Split")
                .field("affected_partitions", affected_partitions)
                .finish(),
            Self::Join { kept_partitions } => f
                .debug_struct("Join")
                .field("kept_partitions", &kept_partitions.len())
                .finish(),
        }
    }
}

pub(crate) fn index_type_string(sub_index: SubIndexType, quantizer: QuantizationType) -> String {
    // FlatBin is a QuantizationType variant used internally for reconstruction,
    // but the persisted index type string uses "FLAT" (differentiated by DataType).
    let quantizer = match quantizer {
        QuantizationType::FlatBin => QuantizationType::Flat,
        other => other,
    };
    match (sub_index, quantizer) {
        // ignore FLAT sub index,
        // IVF_FLAT_FLAT => IVF_FLAT
        // IVF_FLAT_PQ => IVF_PQ
        (SubIndexType::Flat, quantization_type) => format!("IVF_{}", quantization_type),
        (sub_index_type, quantization_type) => {
            if sub_index_type.to_string() == quantization_type.to_string() {
                // ignore redundant quantization type
                // e.g. IVF_PQ_PQ should be IVF_PQ
                format!("IVF_{}", sub_index_type)
            } else {
                format!("IVF_{}_{}", sub_index_type, quantization_type)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use arrow_array::{Array, Float32Array, NullArray};
    use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
    use lance_index::vector::v3::shuffler::{
        ShufflePartition, ShufflePartitionWindow, ShufflePartitionWindowPlan,
    };

    struct SingleBatchReader {
        batch: RecordBatch,
        partition_id: usize,
    }

    #[async_trait::async_trait]
    impl ShuffleReader for SingleBatchReader {
        async fn read_partition(
            &self,
            partition_id: usize,
        ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
            if partition_id != self.partition_id || self.batch.num_rows() == 0 {
                return Ok(None);
            }

            let schema = self.batch.schema();
            let stream = stream::iter(vec![Ok(self.batch.clone())]);
            Ok(Some(Box::new(RecordBatchStreamAdapter::new(
                schema, stream,
            ))))
        }

        fn partition_size(&self, partition_id: usize) -> Result<usize> {
            Ok(if partition_id == self.partition_id {
                self.batch.num_rows()
            } else {
                0
            })
        }

        fn total_loss(&self) -> Option<f64> {
            None
        }
    }

    struct WindowedBatchReader {
        batches: Vec<RecordBatch>,
        windows_read: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ShuffleReader for WindowedBatchReader {
        async fn read_partition(
            &self,
            partition_id: usize,
        ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
            let Some(batch) = self.batches.get(partition_id) else {
                return Ok(None);
            };
            Ok(Some(Box::new(RecordBatchStreamAdapter::new(
                batch.schema(),
                stream::iter(vec![Ok(batch.clone())]),
            ))))
        }

        fn plan_partition_window(
            &self,
            start_partition_id: usize,
            max_decoded_bytes: usize,
        ) -> Result<ShufflePartitionWindowPlan> {
            if max_decoded_bytes == 0 {
                return Err(Error::invalid_input(
                    "max_decoded_bytes must be greater than 0",
                ));
            }
            if start_partition_id >= self.batches.len() {
                return Err(Error::invalid_input(format!(
                    "start_partition_id={} is out of range [0, {})",
                    start_partition_id,
                    self.batches.len()
                )));
            }
            let end_partition_id = start_partition_id
                .saturating_add(max_decoded_bytes)
                .min(self.batches.len());
            Ok(ShufflePartitionWindowPlan {
                partition_range: start_partition_id..end_partition_id,
                estimated_decoded_bytes: end_partition_id - start_partition_id,
            })
        }

        async fn read_partition_window(
            &self,
            start_partition_id: usize,
            max_decoded_bytes: usize,
        ) -> Result<ShufflePartitionWindow> {
            let plan = self.plan_partition_window(start_partition_id, max_decoded_bytes)?;
            self.windows_read.fetch_add(1, Ordering::Relaxed);
            if start_partition_id == 0 {
                tokio::time::timeout(std::time::Duration::from_secs(1), async {
                    while self.windows_read.load(Ordering::Relaxed) < 2 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .map_err(|_| Error::internal("second partition window was not admitted"))?;
            }
            let partitions = plan
                .partition_range
                .clone()
                .map(|partition_id| {
                    let batch = self.batches[partition_id].clone();
                    ShufflePartition {
                        partition_id,
                        data: Some(Box::new(RecordBatchStreamAdapter::new(
                            batch.schema(),
                            stream::iter(vec![Ok(batch)]),
                        ))),
                    }
                })
                .collect();
            Ok(ShufflePartitionWindow {
                materialized_decoded_bytes: Some(plan.partition_range.len()),
                partition_range: plan.partition_range,
                partitions,
            })
        }

        fn partition_size(&self, partition_id: usize) -> Result<usize> {
            Ok(self
                .batches
                .get(partition_id)
                .map(RecordBatch::num_rows)
                .unwrap_or(0))
        }

        fn total_loss(&self) -> Option<f64> {
            None
        }
    }

    fn flat_partition_batch(partition_id: usize) -> RecordBatch {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![partition_id as f32, partition_id as f32 + 0.5]),
            2,
        )
        .unwrap();
        RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                ROW_ID_FIELD.clone(),
                Field::new("vector", vectors.data_type().clone(), false),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![partition_id as u64])),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    // Helper to read centroid i from a FixedSizeListArray as a Vec<f32>
    fn centroid_values(arr: &FixedSizeListArray, i: usize) -> Vec<f32> {
        arr.value(i).as_primitive::<Float32Type>().values().to_vec()
    }

    #[tokio::test]
    async fn partition_entry_admission_preserves_input_order() {
        let entry_permits = Arc::new(Semaphore::new(1));
        let held_permit = entry_permits.clone().acquire_owned().await.unwrap();
        let mut admitted = admit_partition_inputs(vec![0, 1, 2], entry_permits);

        let first = admitted.next();
        tokio::pin!(first);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut first)
                .await
                .is_err()
        );

        drop(held_permit);
        let (partition_id, first_permit) =
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut first)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(partition_id, 0);

        let second = admitted.next();
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );
        drop(first_permit);
        let (partition_id, _second_permit) =
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut second)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(partition_id, 1);
    }

    #[test]
    fn single_partition_window_uses_full_entry_budget() {
        assert_eq!(partition_window_entry_limit(&(0..64), 64, 32, 16), 32);
        assert_eq!(partition_window_entry_limit(&(0..32), 64, 32, 16), 2);
        assert_eq!(partition_window_entry_limit(&(32..64), 64, 32, 16), 2);
    }

    #[tokio::test]
    async fn fresh_partition_build_runs_multiple_windows_end_to_end() {
        let num_partitions = 6;
        let windows_read = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(WindowedBatchReader {
            batches: (0..num_partitions).map(flat_partition_batch).collect(),
            windows_read: windows_read.clone(),
        });
        let mut build_stream =
            IvfIndexBuilder::<FlatIndex, FlatQuantizer>::build_fresh_partitions_windowed(
                reader,
                num_partitions,
                DistanceType::L2,
                FlatQuantizer::new(2, DistanceType::L2),
                (),
                "vector".to_string(),
                None,
                FreshPartitionBuildLimits {
                    window_bytes: 2,
                    decoded_budget_bytes: 4,
                },
            )
            .unwrap();

        let mut ordered_results = OrderedPartitionResults::new(num_partitions);
        let mut merged_partition_ids = Vec::with_capacity(num_partitions);
        while let Some(result) = build_stream.try_next().await.unwrap() {
            ordered_results
                .push(result.value.partition_id, result)
                .unwrap();
            while let Some((partition_id, result)) = ordered_results.pop_next() {
                assert!(result.value.built.is_some());
                merged_partition_ids.push(partition_id);
            }
        }
        ordered_results.finish().unwrap();

        assert_eq!(windows_read.load(Ordering::Relaxed), 3);
        assert_eq!(
            merged_partition_ids,
            (0..num_partitions).collect::<Vec<_>>()
        );
    }

    #[test]
    fn apply_centroid_splits_correct_count_and_ordering() {
        // 4 original centroids at [0,0], [1,1], [2,2], [3,3].
        // Split partitions 1 and 3; verify that:
        //   - result has 6 centroids (4 original + 2 splits)
        //   - unchanged partition indices 0 and 2 keep their original values
        //   - split partitions 1 and 3 have centroid1 at their original index
        //   - centroid2 for each split is appended at the end (indices 4, 5)
        let original = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
            2,
        )
        .unwrap();

        let c1_for_1: ArrayRef = Arc::new(Float32Array::from(vec![1.1_f32, 1.1]));
        let c2_for_1: ArrayRef = Arc::new(Float32Array::from(vec![0.9_f32, 0.9]));
        let c3_for_1: ArrayRef = Arc::new(Float32Array::from(vec![1.2_f32, 1.2]));
        let c1_for_3: ArrayRef = Arc::new(Float32Array::from(vec![3.1_f32, 3.1]));
        let c2_for_3: ArrayRef = Arc::new(Float32Array::from(vec![2.9_f32, 2.9]));

        // Partition 1 splits three ways, partition 3 two ways.
        let splits = vec![
            (1_usize, vec![c1_for_1, c2_for_1, c3_for_1]),
            (3_usize, vec![c1_for_3, c2_for_3]),
        ];
        let result = apply_centroid_splits(&original, &splits).unwrap();

        assert_eq!(result.len(), 7);
        // Unchanged partitions
        assert_eq!(centroid_values(&result, 0), [0.0, 0.0]);
        assert_eq!(centroid_values(&result, 2), [2.0, 2.0]);
        // Replaced centroids (first new centroid of each split partition)
        assert_eq!(centroid_values(&result, 1), [1.1, 1.1]);
        assert_eq!(centroid_values(&result, 3), [3.1, 3.1]);
        // Appended centroids, in split order
        assert_eq!(centroid_values(&result, 4), [0.9, 0.9]);
        assert_eq!(centroid_values(&result, 5), [1.2, 1.2]);
        assert_eq!(centroid_values(&result, 6), [2.9, 2.9]);
    }

    #[test]
    fn select_reassign_candidates_skips_deleted_partition() {
        let dim = 4;
        let centroid_values = Float32Array::from(vec![0.0_f32; dim * 2]);
        let centroids =
            FixedSizeListArray::try_new_from_values(centroid_values, dim as i32).unwrap();
        let mut ivf = IvfModel::new(centroids, None);
        ivf.lengths = vec![10, 20];
        ivf.offsets = vec![0, 10];

        let c0 = ivf.centroid(1).unwrap();
        let (reassign_ids, reassign_centroids) =
            select_reassign_candidates_impl(DistanceType::L2, &ivf, 1, &c0, &HashSet::new())
                .unwrap();

        assert_eq!(reassign_ids.len(), 1);
        assert_eq!(reassign_ids.value(0), 0);
        assert_eq!(reassign_centroids.len(), 1);

        let expected_centroid = ivf.centroid(0).unwrap();
        assert_eq!(
            reassign_centroids
                .value(0)
                .as_primitive::<Float32Type>()
                .values(),
            expected_centroid.as_primitive::<Float32Type>().values()
        );
    }

    #[tokio::test]
    async fn optimize_split_after_append_pushes_partition_over_threshold() {
        use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
        use crate::index::vector::VectorIndexParams;
        use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
        use arrow_array::RecordBatchIterator;
        use arrow_schema::Schema as ArrowSchema;
        use lance_index::optimize::OptimizeOptions;
        use lance_linalg::distance::MetricType;

        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(item_field, 4),
            false,
        )]));

        // Tiny per-row perturbation gives kmeans something to fit while keeping
        // within-cluster variance negligible relative to the 1000-unit cluster
        // separation, so kmeans (k=2) reliably places one centroid per cluster.
        let make_batch = |num_rows: usize, center: f32| -> RecordBatch {
            let mut values = Vec::with_capacity(num_rows * 4);
            for i in 0..num_rows {
                let p = center + (i as f32) * 0.0001;
                values.extend_from_slice(&[p, p, p, p]);
            }
            let fsl =
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), 4).unwrap();
            RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap()
        };

        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();

        // 15k near origin (just under split threshold of 4 * 4096 = 16384) plus
        // 1.5k far away in cluster B.
        let initial = vec![Ok(make_batch(15_000, 0.0)), Ok(make_batch(1_500, 1000.0))];
        let reader = RecordBatchIterator::new(initial, schema.clone());
        let mut dataset = crate::Dataset::write(reader, uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("idx").await.unwrap();
        let initial_index = dataset
            .open_vector_index("vec", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let initial_ivf = initial_index.ivf_model();
        assert_eq!(initial_ivf.num_partitions(), 2);
        let max_initial = (0..2).map(|p| initial_ivf.partition_size(p)).max().unwrap();
        assert!(
            max_initial <= 16_384,
            "initial max partition size {max_initial} should be at or under split threshold",
        );

        // Append 3k more cluster-A rows so partition 0 grows past the threshold.
        let append = make_batch(3_000, 0.0);
        let mut dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![append])
            .await
            .unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("idx").await.unwrap();
        assert_eq!(indices.len(), 1, "expected merge-all on split");
        let optimized = dataset
            .open_vector_index("vec", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let ivf = optimized.ivf_model();

        // 18k rows split ceil(18_000 / 4096) = 5 ways: 2 original + 4 new.
        assert_eq!(
            ivf.num_partitions(),
            6,
            "expected one 5-way split: 2 original + 4 new = 6 partitions",
        );
        let total_rows: usize = (0..ivf.num_partitions())
            .map(|p| optimized.partition_size(p))
            .sum();
        assert_eq!(total_rows, 19_500, "all vectors preserved across split");
    }

    /// Rows of the vector column of the `k` nearest rows to `[center; 4]`,
    /// probing one partition, so a wrongly placed row shows up as a miss.
    async fn nearest_first_components(dataset: &crate::Dataset, center: f32, k: usize) -> Vec<f32> {
        let query = Float32Array::from(vec![center; 4]);
        let batch = dataset
            .scan()
            .nearest("vec", &query, k)
            .unwrap()
            .minimum_nprobes(1)
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch["vec"].as_fixed_size_list();
        (0..vectors.len())
            .map(|i| vectors.value(i).as_primitive::<Float32Type>().value(0))
            .collect()
    }

    fn cluster_batch(
        schema: &Arc<arrow_schema::Schema>,
        num_rows: usize,
        center: f32,
    ) -> RecordBatch {
        // Tiny per-row perturbation gives kmeans something to fit while keeping
        // within-cluster variance negligible relative to the 1000-unit cluster
        // separation.
        let mut values = Vec::with_capacity(num_rows * 4);
        for i in 0..num_rows {
            let p = center + (i as f32) * 0.0001;
            values.extend_from_slice(&[p, p, p, p]);
        }
        let fsl = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 4).unwrap();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap()
    }

    fn cluster_schema() -> Arc<arrow_schema::Schema> {
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(item_field, 4),
            false,
        )]))
    }

    async fn write_clusters(uri: &str, clusters: &[(usize, f32)]) -> crate::Dataset {
        use arrow_array::RecordBatchIterator;
        let schema = cluster_schema();
        let batches = clusters
            .iter()
            .map(|(rows, center)| Ok(cluster_batch(&schema, *rows, *center)))
            .collect::<Vec<_>>();
        let reader = RecordBatchIterator::new(batches, schema);
        crate::Dataset::write(reader, uri, None).await.unwrap()
    }

    async fn append_cluster(dataset: crate::Dataset, rows: usize, center: f32) -> crate::Dataset {
        use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
        let batch = cluster_batch(&cluster_schema(), rows, center);
        InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap()
    }

    async fn open_single_segment(dataset: &crate::Dataset) -> Arc<dyn VectorIndex> {
        use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
        let indices = dataset.load_indices_by_name("idx").await.unwrap();
        assert_eq!(
            indices.len(),
            1,
            "expected the rebalance to merge into one segment"
        );
        dataset
            .open_vector_index("vec", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn optimize_splits_oversized_partition_to_target_in_one_pass() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use lance_index::optimize::OptimizeOptions;
        use lance_linalg::distance::MetricType;

        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let mut dataset = write_clusters(uri, &[(15_000, 0.0), (1_500, 1000.0)]).await;
        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        // 36k rows in partition 0: ceil(36_000 / 4096) = 9 pieces of the target
        // size, produced by one optimize instead of one halving per call.
        let mut dataset = append_cluster(dataset, 21_000, 0.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let optimized = open_single_segment(&dataset).await;
        let ivf = optimized.ivf_model();
        assert_eq!(ivf.num_partitions(), 2 + 8, "one 9-way split");
        let sizes: Vec<usize> = (0..ivf.num_partitions())
            .map(|p| optimized.partition_size(p))
            .collect();
        assert_eq!(
            sizes.iter().sum::<usize>(),
            37_500,
            "all vectors preserved across split"
        );
        let max_size = *sizes.iter().max().unwrap();
        assert!(
            max_size <= 4 * 4096,
            "no partition may stay above the split threshold after one pass, got {max_size}"
        );
        // Every piece is findable through its own centroid.
        let found = nearest_first_components(&dataset, 3.0, 5).await;
        assert_eq!(found.len(), 5);
        assert!(found.iter().all(|v| (0.0..=3.6).contains(v)), "{found:?}");
    }

    #[tokio::test]
    async fn optimize_joins_all_undersized_partitions_in_one_pass() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use lance_index::optimize::OptimizeOptions;
        use lance_linalg::distance::MetricType;

        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        // Three partitions under the join threshold (25% of 4096 = 1024 rows).
        let mut dataset = write_clusters(
            uri,
            &[(2_000, 0.0), (100, 1000.0), (100, 2000.0), (100, 3000.0)],
        )
        .await;
        let params = VectorIndexParams::ivf_flat(4, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        let mut dataset = append_cluster(dataset, 10, 0.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let optimized = open_single_segment(&dataset).await;
        let ivf = optimized.ivf_model();
        assert_eq!(
            ivf.num_partitions(),
            1,
            "all three undersized partitions joined in one pass"
        );
        assert_eq!(optimized.partition_size(0), 2_310);
        // Rows of a joined partition are still indexed.
        let found = nearest_first_components(&dataset, 3000.0, 3).await;
        assert!(
            found.iter().all(|v| (3000.0..=3000.1).contains(v)),
            "{found:?}"
        );
    }

    #[tokio::test]
    async fn optimize_steady_state_keeps_small_delta_partitions() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use lance_index::optimize::OptimizeOptions;
        use lance_linalg::distance::MetricType;

        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        // Two partitions of 1_500 rows, above the join threshold (25% of 4096).
        let mut dataset = write_clusters(uri, &[(1_500, 0.0), (1_500, 1000.0)]).await;
        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        // A small append becomes a delta segment whose partitions hold ~100 rows.
        let dataset = append_cluster(dataset, 100, 0.0).await;
        let mut dataset = append_cluster(dataset, 100, 1000.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let indices = dataset.load_indices_by_name("idx").await.unwrap();
        assert_eq!(indices.len(), 2, "the append is a delta segment");

        // Summed over both segments every partition is normal-sized, so a
        // steady-state optimize must leave the delta's small partitions alone.
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let indices = dataset.load_indices_by_name("idx").await.unwrap();
        assert_eq!(indices.len(), 2, "steady state is a no-op");
        for index in &indices {
            use crate::index::DatasetIndexInternalExt;
            let opened = dataset
                .open_vector_index("vec", &index.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(
                opened.ivf_model().num_partitions(),
                2,
                "no partition was joined away"
            );
        }
    }

    #[tokio::test]
    async fn optimize_split_threshold_honors_persisted_target_partition_size() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::{StageParams, VectorIndexParams};
        use lance_index::optimize::OptimizeOptions;
        use lance_linalg::distance::MetricType;

        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let mut dataset = write_clusters(uri, &[(5_000, 0.0), (500, 1000.0)]).await;
        // 5_000 rows exceed 4 x 1024 but not 4 x 4096 (the IVF_FLAT default), so
        // the split only happens when the recorded target is used.
        let mut params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        let StageParams::Ivf(ivf_params) = &mut params.stages[0] else {
            panic!("ivf_flat params start with the IVF stage");
        };
        ivf_params.target_partition_size = Some(1024);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        let mut dataset = append_cluster(dataset, 10, 0.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let optimized = open_single_segment(&dataset).await;
        let ivf = optimized.ivf_model();
        assert_eq!(
            ivf.num_partitions(),
            2 + 4,
            "5_010 rows split ceil(5010 / 1024) = 5 ways"
        );
        let total: usize = (0..ivf.num_partitions())
            .map(|p| optimized.partition_size(p))
            .sum();
        assert_eq!(total, 5_510);
    }

    #[tokio::test]
    async fn take_partition_batches_preserves_partition_order_for_large_fixed_size_list() {
        let value_length = 1_073_741_824i32;
        let num_rows = 5usize;
        let row_ids = UInt64Array::from(vec![4_u64, 3, 2, 1, 0]);
        let part_ids = UInt32Array::from(vec![0_u32; num_rows]);
        let values = Arc::new(NullArray::new(num_rows * value_length as usize));
        let item_field = Arc::new(Field::new("item", DataType::Null, true));
        let codes = FixedSizeListArray::try_new(item_field, value_length, values, None).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                ROW_ID_FIELD.clone(),
                PART_ID_FIELD.clone(),
                Field::new(PQ_CODE_COLUMN, codes.data_type().clone(), true),
            ])),
            vec![Arc::new(row_ids), Arc::new(part_ids), Arc::new(codes)],
        )
        .unwrap();
        let reader = SingleBatchReader {
            batch,
            partition_id: 0,
        };

        let (batches, loss) = IvfIndexBuilder::<FlatIndex, FlatQuantizer>::take_partition_batches(
            0,
            &[],
            Some(&reader),
        )
        .await
        .unwrap();

        assert_eq!(loss, 0.0);
        assert_eq!(batches.len(), 1);
        assert!(batches[0].column_by_name(PART_ID_COLUMN).is_none());
        let row_ids = batches[0][ROW_ID].as_primitive::<UInt64Type>();
        assert_eq!(row_ids.values(), &[4, 3, 2, 1, 0]);
    }
}
