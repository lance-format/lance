// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::iter::repeat_n;
use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use all_asserts::{assert_ge, assert_lt};
use arrow::datatypes::{Float64Type, UInt8Type, UInt32Type, UInt64Type};
use arrow::{array::AsArray, datatypes::Float32Type};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Int64Array, ListArray,
    PrimitiveArray, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use itertools::Itertools;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::bq::{
    RQBuildParams, RQRotationType,
    ex_dot::{blocked_ex_code_bytes, padded_query_len},
    storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RabitQuantizationMetadata, RabitQueryEstimator},
    transform::{EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN},
};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::IvfSubIndex;

use crate::dataset::{InsertBuilder, UpdateBuilder, WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::DatasetIndexInternalExt;
use crate::index::vector::details::target_partition_size_from_details;
use crate::index::vector::ivf::v2::{
    IVFPartitionKey, IvfFlatIndex, IvfHnswSqIndex, IvfPq, IvfStateEntryBox, PartitionEntry,
};
use crate::utils::test::copy_test_data_to_tmp;
use crate::{
    Dataset,
    index::vector::{VectorIndex, VectorIndexParams},
};
use crate::{
    dataset::optimize::{CompactionOptions, compact_files},
    index::vector::IndexFileVersion,
};
use futures::TryStreamExt;
use lance_core::cache::{CacheBackend, CacheCodecImpl, LanceCache, WeakLanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ID, Result};
use lance_datagen::{Dimension, RowCount, Seed, array, gen_batch};
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::prefilter::{NoFilter, PreFilter};
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::DIST_COL;
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::hnsw::builder::{HNSW_METADATA_KEY, HnswBuildParams};
use lance_index::vector::hnsw::{HNSW, HnswMetadata};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_index::vector::quantizer::QuantizerMetadata;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};
use lance_index::vector::{
    pq::storage::ProductQuantizationMetadata,
    sq::storage::{SQ_METADATA_KEY, ScalarQuantizationMetadata},
    storage::STORAGE_METADATA_KEY,
};
use lance_index::{INDEX_AUXILIARY_FILE_NAME, metrics::NoOpMetricsCollector};
use lance_io::{
    object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor},
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::IndexMetadata;
use lance_testing::datagen::{generate_random_array, generate_random_array_with_range};
use rand::distr::{Distribution, StandardUniform, uniform::SampleUniform};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rstest::rstest;
use uuid::Uuid;

const NUM_ROWS: usize = 512;
const DIM: usize = 32;
// 8-bit PQ needs at least 256 training vectors; 320 leaves a stable margin
// while 20 neighbors provide a useful recall oracle.
const PQ_MATRIX_NUM_ROWS: usize = 320;
const PQ_MATRIX_K: usize = 20;
const LIGHTWEIGHT_PQ_PARTITIONS: usize = 2;
const LIGHTWEIGHT_IVF_SAMPLE_RATE: usize = 16;
// An 8-bit PQ codebook has 256 codes and needs a training vector for each,
// so this is the smallest valid fixture, shared by the 8-bit and 4-bit
// runtime cases.
const LIGHTWEIGHT_PQ_ROWS: usize = 256;
const LIGHTWEIGHT_PQ_SUB_VECTORS: usize = 4;

/// How many partitions `rows` supports when each one is trained on
/// `sample_rate` vectors. At least 1, and never more than `requested`.
fn supported_partitions(rows: usize, sample_rate: usize, requested: usize) -> usize {
    requested.min(rows / sample_rate).max(1)
}

lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

#[test]
fn test_prewarm_parallelism_is_bounded_by_io_and_cpu() {
    assert_eq!(super::prewarm_parallelism(8, 4), 4);
    assert_eq!(super::prewarm_parallelism(2, 4), 2);
    assert_eq!(super::prewarm_parallelism(0, 0), 1);
}

#[test]
fn test_prewarm_window_size_config() {
    assert_eq!(
        super::parse_prewarm_window_size_bytes(None).unwrap(),
        64 * 1024 * 1024
    );
    assert_eq!(
        super::parse_prewarm_window_size_bytes(Some("1048576")).unwrap(),
        1_048_576
    );
    for value in ["0", "not-a-byte-count"] {
        let error = super::parse_prewarm_window_size_bytes(Some(value)).unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(message.contains("LANCE_IVF_PREWARM_WINDOW_SIZE_BYTES"));
        assert!(message.contains(value));
    }
}

fn ivf_with_lengths(lengths: &[u32]) -> IvfModel {
    let mut ivf = IvfModel::empty();
    for &length in lengths {
        ivf.add_partition(length);
    }
    ivf
}

fn prewarm_layout(
    ivf: &IvfModel,
    encoded_bytes: u64,
    num_rows: u64,
) -> super::PrewarmFileLayout<'_> {
    super::PrewarmFileLayout {
        ivf,
        encoded_bytes,
        num_rows,
    }
}

#[test]
fn test_plan_prewarm_windows_uses_combined_encoded_bytes() {
    let ivf = ivf_with_lengths(&[2, 0, 4, 20, 0, 2]);
    let windows = super::plan_partition_windows(
        prewarm_layout(&ivf, 112, 28),
        prewarm_layout(&ivf, 168, 28),
        50,
        100,
    )
    .unwrap();

    assert_eq!(
        windows,
        vec![
            super::PartitionWindow {
                partitions: 0..2,
                estimated_encoded_bytes: 20,
            },
            super::PartitionWindow {
                partitions: 2..3,
                estimated_encoded_bytes: 40,
            },
            super::PartitionWindow {
                partitions: 3..4,
                estimated_encoded_bytes: 200,
            },
            super::PartitionWindow {
                partitions: 4..6,
                estimated_encoded_bytes: 20,
            },
        ]
    );
    assert_eq!(windows.first().unwrap().partitions.start, 0);
    assert_eq!(windows.last().unwrap().partitions.end, ivf.num_partitions());
    for pair in windows.windows(2) {
        assert_eq!(pair[0].partitions.end, pair[1].partitions.start);
    }
}

#[test]
fn test_plan_prewarm_windows_caps_empty_partitions_and_splits_gaps() {
    let empty_ivf = ivf_with_lengths(&[0, 0, 0, 0, 0]);
    let windows = super::plan_partition_windows(
        prewarm_layout(&empty_ivf, 0, 0),
        prewarm_layout(&empty_ivf, 0, 0),
        1024,
        2,
    )
    .unwrap();
    assert_eq!(
        windows
            .iter()
            .map(|window| window.partitions.clone())
            .collect::<Vec<_>>(),
        vec![0..2, 2..4, 4..5]
    );

    let index_ivf = ivf_with_lengths(&[2, 2, 2]);
    let mut storage_ivf = IvfModel::empty();
    storage_ivf.add_partition_with_offset(0, 2);
    storage_ivf.add_partition_with_offset(20, 2);
    storage_ivf.add_partition_with_offset(22, 2);
    let windows = super::plan_partition_windows(
        prewarm_layout(&index_ivf, 6, 6),
        prewarm_layout(&storage_ivf, 6, 6),
        1024,
        100,
    )
    .unwrap();
    assert_eq!(
        windows
            .iter()
            .map(|window| window.partitions.clone())
            .collect::<Vec<_>>(),
        vec![0..1, 1..3]
    );
}

#[test]
fn test_split_prewarm_window_compacts_partition_buffers() {
    let parent = RecordBatch::try_from_iter([(
        "value",
        Arc::new(UInt64Array::from_iter_values(0..100)) as ArrayRef,
    )])
    .unwrap();
    let parent_ptr = parent["value"]
        .as_primitive::<UInt64Type>()
        .values()
        .as_ptr();
    let parent_size = parent["value"].get_array_memory_size();
    let mut partitions =
        super::split_window_batches(&parent.schema(), &[10, 0, 90], vec![parent]).unwrap();

    let shared_ptr = partitions[0][0]["value"]
        .as_primitive::<UInt64Type>()
        .values()
        .as_ptr();
    assert_eq!(shared_ptr, parent_ptr);
    let compact = super::compact_partition_batches(partitions.remove(0)).unwrap();
    let compact_ptr = compact["value"]
        .as_primitive::<UInt64Type>()
        .values()
        .as_ptr();
    assert_ne!(compact_ptr, parent_ptr);
    assert_lt!(compact["value"].get_array_memory_size(), parent_size);
    assert_eq!(compact.num_rows(), 10);
    assert_eq!(partitions[0][0].num_rows(), 0);
    assert_eq!(partitions[1][0].num_rows(), 90);
}

#[derive(Default)]
struct PartitionCoverageTestFilter {
    needs_partition_rows: bool,
    selected_rows: Arc<RowAddrMask>,
    coverage_checks: AtomicUsize,
    mask_reads: AtomicUsize,
}

#[async_trait::async_trait]
impl PreFilter for PartitionCoverageTestFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn needs_partition_row_ids(&self) -> bool {
        self.needs_partition_rows
    }

    fn is_empty_for(&self, rows: &RowAddrTreeMap) -> bool {
        self.coverage_checks.fetch_add(1, Ordering::Relaxed);
        self.selected_rows.selects_all(rows)
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.mask_reads.fetch_add(1, Ordering::Relaxed);
        self.selected_rows.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask().selected_indices(row_ids)
    }
}

#[tokio::test]
async fn test_partition_coverage_is_only_built_for_capable_filters() {
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0_f32; 16]), 4).unwrap();
    let entry = Arc::new(PartitionEntry::<FlatIndex, FlatQuantizer>::new(
        FlatIndex::default(),
        FlatFloatStorage::new(vectors, DistanceType::L2),
    ));
    let cache = LanceCache::with_capacity(1 << 20);
    cache
        .insert_with_key(
            &IVFPartitionKey::<FlatIndex, FlatQuantizer>::new(0),
            entry.clone(),
        )
        .await;
    let weak_cache = WeakLanceCache::from(&cache);
    let size_without_coverage = entry.deep_size_of();
    // Memoize the chunk-budget size before the coverage exists; it must
    // still pick the coverage up once a later filter builds it.
    assert_eq!(entry.size_bytes(), entry.as_ref().deep_size_of());
    let cache_weight_without_coverage = cache.size_bytes().await;

    let ordinary_filter: Arc<dyn PreFilter> = Arc::new(PartitionCoverageTestFilter {
        needs_partition_rows: false,
        ..Default::default()
    });
    let returned = super::IVFIndex::<FlatIndex, FlatQuantizer>::prefilter_for_partition(
        &weak_cache,
        0,
        &entry,
        ordinary_filter.clone(),
    )
    .await
    .unwrap();
    assert!(Arc::ptr_eq(&returned, &ordinary_filter));
    assert!(entry.partition_rows.get().is_none());
    assert_eq!(cache.size_bytes().await, cache_weight_without_coverage);

    let segment_filter: Arc<dyn PreFilter> = Arc::new(PartitionCoverageTestFilter {
        needs_partition_rows: true,
        ..Default::default()
    });
    let returned = super::IVFIndex::<FlatIndex, FlatQuantizer>::prefilter_for_partition(
        &weak_cache,
        0,
        &entry,
        segment_filter,
    )
    .await
    .unwrap();
    assert!(returned.is_empty());

    let first_rows = entry.partition_rows();
    let second_rows = entry.partition_rows();
    assert!(Arc::ptr_eq(&first_rows, &second_rows));
    assert!(entry.deep_size_of() > size_without_coverage);
    assert_eq!(entry.size_bytes(), entry.as_ref().deep_size_of());
    let cache_weight_with_coverage = cache.size_bytes().await;
    assert!(cache_weight_with_coverage > cache_weight_without_coverage);
    assert!(cache_weight_with_coverage >= entry.deep_size_of());
    assert!(entry.partition_rows_accounted.load(Ordering::Acquire));
}

#[rstest]
#[case::flat(None)]
#[case::pq4(Some(4))]
#[case::pq8(Some(8))]
#[tokio::test]
async fn test_batch_partition_prefilter_preserves_bulk_scoring(#[case] bits: Option<usize>) {
    const NUM_QUERIES: usize = 4;
    const K: usize = 10;
    let (batch, schema) = make_seeded_vector_batch(PQ_MATRIX_NUM_ROWS);
    let vectors = batch["vector"].as_fixed_size_list();
    let query = Query {
        column: "vector".to_owned(),
        key: vectors.values().slice(0, NUM_QUERIES * DIM),
        k: PQ_MATRIX_NUM_ROWS,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: LIGHTWEIGHT_PQ_PARTITIONS,
        maximum_nprobes: Some(LIGHTWEIGHT_PQ_PARTITIONS),
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: Some(DistanceType::L2),
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let mut ivf = IvfBuildParams::new(LIGHTWEIGHT_PQ_PARTITIONS);
    ivf.max_iters = 2;
    ivf.sample_rate = LIGHTWEIGHT_IVF_SAMPLE_RATE;
    let params = match bits {
        None => VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
        Some(bits) => VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf,
            lightweight_pq_params_with_bits(bits),
        ),
    };
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();
    let metadata = dataset.load_indices_by_name("vector_idx").await.unwrap();
    let index = dataset
        .open_vector_index("vector", &metadata[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let mut partitions = Vec::with_capacity(NUM_QUERIES);
    let mut centroid_dists = Vec::with_capacity(NUM_QUERIES);
    for query_index in 0..NUM_QUERIES {
        let mut single_query = query.clone();
        single_query.key = query.key.slice(query_index * DIM, DIM);
        let (parts, dists) = index.find_partitions(&single_query).unwrap();
        assert_eq!(parts.len(), LIGHTWEIGHT_PQ_PARTITIONS);
        partitions.push(Arc::new(parts));
        centroid_dists.push(Arc::new(dists));
    }
    let all_results = index
        .clone()
        .search_partitions_batch(
            query.clone(),
            partitions.clone(),
            centroid_dists.clone(),
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
        )
        .await
        .unwrap()
        .into_iter()
        .map(|batch| {
            let order = arrow::compute::sort_to_indices(&batch[DIST_COL], None, None).unwrap();
            arrow::compute::take_record_batch(&batch, &order).unwrap()
        })
        .collect::<Vec<_>>();
    assert!(
        all_results
            .iter()
            .all(|batch| batch.num_rows() == PQ_MATRIX_NUM_ROWS)
    );
    for (query_index, result) in all_results.iter().enumerate() {
        let key = query.key.slice(query_index * DIM, DIM);
        let truth = dataset
            .scan()
            .with_row_id()
            .project(&[ROW_ID])
            .unwrap()
            .nearest("vector", key.as_ref(), K)
            .unwrap()
            .use_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let truth_ids = truth[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .take(K)
            .filter(|row_id| truth_ids.contains(row_id))
            .count() as f32
            / K as f32;
        assert_ge!(recall, 0.5);
    }
    let row_ids = all_results[0][ROW_ID].as_primitive::<UInt64Type>();
    let stale_row = row_ids.value(0);
    let owned_rows = row_ids.values().iter().copied().collect::<RowAddrTreeMap>();
    let mixed_rows = row_ids
        .values()
        .iter()
        .copied()
        .filter(|row_id| *row_id != stale_row)
        .collect::<RowAddrTreeMap>();
    let mut batch_query = query;
    batch_query.k = K;
    for (mask, needs_partition_rows, expect_mask_reads) in [
        (RowAddrMask::from_allowed(owned_rows), true, false),
        (RowAddrMask::from_allowed(mixed_rows.clone()), true, true),
        (RowAddrMask::allow_nothing(), true, true),
        (RowAddrMask::from_allowed(mixed_rows), false, true),
    ] {
        let filter = Arc::new(PartitionCoverageTestFilter {
            needs_partition_rows,
            selected_rows: Arc::new(mask),
            ..Default::default()
        });
        let results = index
            .clone()
            .search_partitions_batch(
                batch_query.clone(),
                partitions.clone(),
                centroid_dists.clone(),
                filter.clone(),
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap();
        assert_eq!(
            filter.coverage_checks.load(Ordering::Relaxed),
            if needs_partition_rows {
                LIGHTWEIGHT_PQ_PARTITIONS
            } else {
                0
            },
            "coverage must be checked once per distinct partition, not once per query"
        );
        assert_eq!(
            filter.mask_reads.load(Ordering::Relaxed) > 0,
            expect_mask_reads,
            "fully owned partitions must use the bulk scorer without reading the mask"
        );
        for (result, all_result) in results.iter().zip(&all_results) {
            let selected = all_result[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .enumerate()
                .filter(|(_, row_id)| filter.selected_rows.selected(**row_id))
                .take(K)
                .map(|(position, _)| position as u32)
                .collect::<Vec<_>>();
            let expected =
                arrow::compute::take_record_batch(all_result, &UInt32Array::from(selected))
                    .unwrap();
            let order = arrow::compute::sort_to_indices(&result[DIST_COL], None, None).unwrap();
            let result = arrow::compute::take_record_batch(result, &order).unwrap();
            assert_eq!(
                result[ROW_ID].as_primitive::<UInt64Type>().values(),
                expected[ROW_ID].as_primitive::<UInt64Type>().values(),
                "ownership must be applied before truncating the top-k heap"
            );
            for (actual, expected) in result[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .zip(expected[DIST_COL].as_primitive::<Float32Type>().values())
            {
                assert_lt!((actual - expected).abs(), 1e-4);
            }
        }
    }
}

#[test]
fn test_rotated_partition_centroid_slice_borrows_cache() {
    let cache = super::RabitSearchCache {
        rotated_centroids: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        code_dim: 2,
    };

    let centroid = super::rotated_partition_centroid_slice(Some(&cache), 1).unwrap();

    assert_eq!(centroid, &[3.0, 4.0]);
    assert_eq!(centroid.as_ptr(), cache.rotated_centroids[2..].as_ptr());
    assert!(super::rotated_partition_centroid_slice(Some(&cache), 3).is_none());
    assert!(super::rotated_partition_centroid_slice(None, 0).is_none());
}

#[test]
fn test_rabit_ex_scratch_len_uses_num_bits() {
    // Block-aligned dims read the rotated query in place.
    let dim = 960;
    for num_bits in [1, 3, 5, 7, 9] {
        assert_eq!(super::rabit_ex_scratch_len(dim, num_bits), 0);
    }

    // Unaligned multi-bit queries add one padded query copy.
    let dim = 968;
    assert_eq!(super::rabit_ex_scratch_len(dim, 1), 0);
    assert_eq!(super::rabit_ex_scratch_len(dim, 7), padded_query_len(dim));
}

#[test]
fn test_rabit_u8_scratch_len_includes_ex_fastscan_tables() {
    let dim = 960;

    assert_eq!(super::rabit_u8_scratch_len(dim, 1), dim * 4);
    assert_eq!(super::rabit_u8_scratch_len(dim, 3), dim * 8);
    assert_eq!(super::rabit_u8_scratch_len(dim, 5), dim * 16);
    assert_eq!(super::rabit_u8_scratch_len(dim, 7), dim * 4);
    assert_eq!(super::rabit_u8_scratch_len(dim, 9), dim * 32);
}

#[test]
fn test_rabit_query_scratch_capacity_does_not_preallocate_u32() {
    let dim = 960;
    let max_partition_len = 4096;

    let capacity = super::rabit_query_scratch_capacity(dim, max_partition_len, 5);

    assert_eq!(capacity.distances, max_partition_len);
    assert_eq!(capacity.query_f32, dim + dim * 4);
    assert_eq!(capacity.u16, max_partition_len);
    assert_eq!(capacity.u8, dim * 16);
    assert_eq!(capacity.u32, 0);
}

async fn generate_test_dataset<T: ArrowPrimitiveType>(
    test_uri: &str,
    range: Range<T::Native>,
) -> (Dataset, Arc<FixedSizeListArray>)
where
    T::Native: SampleUniform,
{
    let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, false);
    let vectors = batch.column_by_name("vector").unwrap().clone();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            mode: crate::dataset::WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    (dataset, Arc::new(vectors.as_fixed_size_list().clone()))
}

async fn generate_multivec_test_dataset<T: ArrowPrimitiveType>(
    test_uri: &str,
    range: Range<T::Native>,
) -> (Dataset, Arc<ListArray>)
where
    T::Native: SampleUniform,
{
    let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, true);
    let vectors = batch.column_by_name("vector").unwrap().clone();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
    (dataset, Arc::new(vectors.as_list::<i32>().clone()))
}

async fn append_dataset<T: ArrowPrimitiveType>(
    dataset: &mut Dataset,
    num_rows: usize,
    range: Range<T::Native>,
) -> ArrayRef
where
    T::Native: SampleUniform,
{
    let is_multivector = matches!(
        dataset.schema().field("vector").unwrap().data_type(),
        DataType::List(_)
    );
    let row_count = dataset.count_all_rows().await.unwrap();
    let (batch, schema) =
        generate_batch::<T>(num_rows, Some(row_count as u64), range, is_multivector);
    let vectors = batch["vector"].clone();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    dataset.append(batches, None).await.unwrap();
    vectors
}

async fn open_rq_aux_reader(
    dataset: &Dataset,
    scheduler: Arc<ScanScheduler>,
    index_uuid: &str,
) -> FileReader {
    let index_path = dataset
        .indices_dir()
        .join(index_uuid)
        .join(INDEX_AUXILIARY_FILE_NAME);
    let file_scheduler = scheduler
        .open_file(&index_path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

async fn get_rq_metadata(
    dataset: &Dataset,
    scheduler: Arc<ScanScheduler>,
    index_uuid: &str,
) -> RabitQuantizationMetadata {
    let reader = open_rq_aux_reader(dataset, scheduler, index_uuid).await;
    let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
    let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
    serde_json::from_str(&metadata_entries[0]).unwrap()
}

async fn get_sq_metadata(
    dataset: &Dataset,
    scheduler: Arc<ScanScheduler>,
    index_uuid: &str,
) -> ScalarQuantizationMetadata {
    let index_path = dataset
        .indices_dir()
        .join(index_uuid)
        .join(INDEX_AUXILIARY_FILE_NAME);
    let file_scheduler = scheduler
        .open_file(&index_path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    if let Some(metadata) = reader.schema().metadata.get(SQ_METADATA_KEY) {
        serde_json::from_str(metadata).unwrap()
    } else {
        let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
        let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
        serde_json::from_str(&metadata_entries[0]).unwrap()
    }
}

async fn assert_rq_rotation_type(dataset: &Dataset, expected: RQRotationType) {
    let obj_store = Arc::new(ObjectStore::local());
    let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty(), "Expected at least one vector index");
    for index in indices.iter() {
        let rq_meta = get_rq_metadata(dataset, scheduler.clone(), &index.uuid.to_string()).await;
        assert_eq!(
            rq_meta.rotation_type, expected,
            "RQ rotation type mismatch for index {}",
            index.uuid
        );
    }
}

fn generate_batch<T: ArrowPrimitiveType>(
    num_rows: usize,
    start_id: Option<u64>,
    range: Range<T::Native>,
    is_multivector: bool,
) -> (RecordBatch, SchemaRef)
where
    T::Native: SampleUniform,
{
    const VECTOR_NUM_PER_ROW: usize = 3;
    let start_id = start_id.unwrap_or(0);
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_id..start_id + num_rows as u64,
    ));
    let total_floats = match is_multivector {
        true => num_rows * VECTOR_NUM_PER_ROW * DIM,
        false => num_rows * DIM,
    };
    let vectors = generate_random_array_with_range::<T>(total_floats, range);
    let data_type = vectors.data_type().clone();
    let mut fields = vec![Field::new("id", DataType::UInt64, false)];
    let mut arrays: Vec<ArrayRef> = vec![ids];
    let mut fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
    if fsl.value_type() != DataType::UInt8 {
        fsl = normalize_fsl(&fsl).unwrap();
    }
    if is_multivector {
        let vector_field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
            true,
        ));
        fields.push(Field::new(
            "vector",
            DataType::List(vector_field.clone()),
            true,
        ));
        let array = Arc::new(ListArray::new(
            vector_field,
            OffsetBuffer::from_lengths(std::iter::repeat_n(VECTOR_NUM_PER_ROW, num_rows)),
            Arc::new(fsl),
            None,
        ));
        arrays.push(array);
    } else {
        fields.push(Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
            true,
        ));
        let array = Arc::new(fsl);
        arrays.push(array);
    }
    let schema: Arc<_> = Schema::new(fields).into();
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    (batch, schema)
}

fn generate_clustered_batch(
    rows_per_partition: usize,
    offsets: [f32; 2],
) -> (RecordBatch, SchemaRef) {
    let num_partitions = offsets.len();
    let total_rows = rows_per_partition * num_partitions;
    let mut ids = Vec::with_capacity(total_rows);
    let mut values = Vec::with_capacity(total_rows * DIM);
    let mut rng = StdRng::seed_from_u64(42);
    for (cluster_idx, offset) in offsets.iter().enumerate() {
        for row in 0..rows_per_partition {
            ids.push((cluster_idx * rows_per_partition + row) as u64);
            for dim in 0..DIM {
                let base = if dim == 0 { *offset } else { 0.0 };
                let noise = (rng.random::<f32>() - 0.5) * 0.02;
                values.push(base + noise);
            }
        }
    }
    let ids = Arc::new(UInt64Array::from(ids));
    let vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap(),
    );
    let schema: Arc<_> = Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ])
    .into();
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
    (batch, schema)
}

/// Rows of `vectors_per_row` vectors around their cluster's centroid. The
/// `straddling_row`, if any, spreads its vectors over the centroids instead,
/// one per vector, so it belongs to several partitions at once.
fn generate_clustered_multivec_batch(
    cluster_sizes: &[usize],
    centroids: &[(f32, f32)],
    vectors_per_row: usize,
    start_id: u64,
    straddling_row: Option<u64>,
) -> (RecordBatch, SchemaRef) {
    assert_eq!(
        cluster_sizes.len(),
        centroids.len(),
        "cluster sizes and centroids must match"
    );
    const ITEM_FIELD_NAME: &str = "item";
    let total_rows: usize = cluster_sizes.iter().sum();
    let mut ids = Vec::with_capacity(total_rows);
    let mut values = Vec::with_capacity(total_rows * vectors_per_row * DIM);
    let mut rng = StdRng::seed_from_u64(12345);
    let mut current_id = start_id;
    for (&rows, &(x, y)) in cluster_sizes.iter().zip(centroids.iter()) {
        for _ in 0..rows {
            let row_id = current_id;
            ids.push(row_id);
            current_id += 1;
            for vector_idx in 0..vectors_per_row {
                let (x, y) = match straddling_row {
                    Some(id) if id == row_id => centroids[vector_idx % centroids.len()],
                    _ => (x, y),
                };
                for dim in 0..DIM {
                    let base = match dim {
                        0 => x,
                        1 => y,
                        _ => 0.0,
                    };
                    let noise = (rng.random::<f32>() - 0.5) * 0.02;
                    values.push(base + noise);
                }
            }
        }
    }
    let ids_array = Arc::new(UInt64Array::from(ids));
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap();
    let vector_field = Arc::new(Field::new(
        ITEM_FIELD_NAME,
        DataType::FixedSizeList(
            Arc::new(Field::new(ITEM_FIELD_NAME, DataType::Float32, true)),
            DIM as i32,
        ),
        true,
    ));
    let offsets_buffer =
        OffsetBuffer::from_lengths(std::iter::repeat_n(vectors_per_row, total_rows));
    let list_array = Arc::new(ListArray::new(
        vector_field.clone(),
        offsets_buffer,
        Arc::new(vectors),
        None,
    ));
    let schema: Arc<_> = Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", DataType::List(vector_field), false),
    ])
    .into();
    let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, list_array]).unwrap();
    (batch, schema)
}

fn build_centroids_for_offsets(offsets: &[f32]) -> Arc<FixedSizeListArray> {
    let mut centroid_values = Vec::with_capacity(offsets.len() * DIM);
    for &offset in offsets {
        for dim in 0..DIM {
            centroid_values.push(if dim == 0 { offset } else { 0.0 });
        }
    }
    Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroid_values), DIM as i32)
            .unwrap(),
    )
}

fn build_centroids_2d(centroids: &[(f32, f32)]) -> Arc<FixedSizeListArray> {
    let mut values = Vec::with_capacity(centroids.len() * DIM);
    for &(x, y) in centroids {
        for dim in 0..DIM {
            values.push(match dim {
                0 => x,
                1 => y,
                _ => 0.0,
            });
        }
    }
    Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap(),
    )
}

fn make_fragment_offset_batches(
    rows_per_fragment: usize,
    offsets: &[f32],
) -> (Arc<Schema>, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));

    let mut next_id = 0_u64;
    let batches = offsets
        .iter()
        .map(|offset| {
            let ids = Arc::new(UInt64Array::from_iter_values(
                next_id..next_id + rows_per_fragment as u64,
            ));
            next_id += rows_per_fragment as u64;

            let mut values = Vec::with_capacity(rows_per_fragment * DIM);
            for _ in 0..rows_per_fragment {
                for dim in 0..DIM {
                    values.push(*offset + dim as f32);
                }
            }

            let vectors = Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                    .unwrap(),
            );
            RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap()
        })
        .collect();

    (schema, batches)
}

struct VectorIndexTestContext {
    stats_json: String,
    stats: serde_json::Value,
    index: Arc<dyn VectorIndex>,
}

impl VectorIndexTestContext {
    fn stats(&self) -> &serde_json::Value {
        &self.stats
    }

    fn stats_json(&self) -> &str {
        &self.stats_json
    }

    fn num_partitions(&self) -> usize {
        self.stats()["indices"][0]["num_partitions"]
            .as_u64()
            .expect("num_partitions should be present") as usize
    }

    fn ivf(&self) -> &IvfPq {
        self.index
            .as_any()
            .downcast_ref::<IvfPq>()
            .expect("expected IvfPq index")
    }

    fn ivf_flat(&self) -> &IvfFlatIndex {
        self.index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlat index")
    }
}

fn lightweight_pq_params() -> PQBuildParams {
    PQBuildParams {
        num_sub_vectors: LIGHTWEIGHT_PQ_SUB_VECTORS,
        num_bits: 4,
        max_iters: 2,
        sample_rate: 16,
        ..Default::default()
    }
}

fn lightweight_pq_params_with_bits(num_bits: usize) -> PQBuildParams {
    let num_sub_vectors = if num_bits == 4 {
        // M4 is only a 2-byte code, so random KMeans/HNSW can leave recall near
        // the threshold. M32 restores the original 4-bit test capacity.
        DIM
    } else {
        LIGHTWEIGHT_PQ_SUB_VECTORS
    };
    PQBuildParams {
        num_sub_vectors,
        num_bits,
        max_iters: 2,
        sample_rate: 16,
        ..Default::default()
    }
}

fn lightweight_hnsw_params() -> HnswBuildParams {
    HnswBuildParams::default()
        .max_level(2)
        .num_edges(4)
        .ef_construction(16)
}

fn make_seeded_vector_batch(num_rows: usize) -> (RecordBatch, SchemaRef) {
    let batch = lance_datagen::gen_batch()
        .with_seed(lance_datagen::Seed::from(42))
        .col("id", lance_datagen::array::step::<UInt64Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<Float32Type>((DIM as u32).into()),
        )
        .into_batch_rows(lance_datagen::RowCount::from(num_rows as u64))
        .unwrap();
    let schema = batch.schema();
    (batch, schema)
}

#[rstest]
#[case::flat(None)]
#[case::pq4(Some(4))]
#[case::pq8(Some(8))]
#[case::rq5(Some(5))]
#[tokio::test]
async fn test_pairwise_partition_replay_has_no_source_io(#[case] bits: Option<usize>) {
    // Cross the 8K source-read boundary with a packed RQ tail. Force both
    // memory and spill paths, replaying the same index in 1K vector batches.
    const NUM_ROWS: usize = 8193;
    const BATCH_SIZE: usize = 1024;
    let (batch, schema) = make_seeded_vector_batch(NUM_ROWS);
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            // Keep this tiny fixture out of the whole-file read cache so
            // the source-I/O assertion also exercises PQ4 preparation.
            store_params: Some(ObjectStoreParams {
                block_size: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut ivf = IvfBuildParams::new(1);
    ivf.centroids = Some(Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; DIM]), DIM as i32)
            .unwrap(),
    ));
    let params = match bits {
        None => VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
        Some(5) => VectorIndexParams::with_ivf_rq_params(
            DistanceType::L2,
            ivf,
            RQBuildParams::with_rotation_type(5, RQRotationType::Fast),
        ),
        Some(bits) => VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf,
            PQBuildParams::with_codebook(
                16,
                bits,
                Arc::new(generate_random_array(DIM * (1 << bits))),
            ),
        ),
    };
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("pairs".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let meta = dataset.load_indices_by_name("pairs").await.unwrap();
    let index = dataset
        .open_vector_index("vector", &meta[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let session = dataset.session();
    let store = session.spill_store();
    dataset.object_store.as_ref().io_stats_incremental();
    let memory = index
        .prepare_pairwise_partition(0, BATCH_SIZE, 16 * 1024 * 1024, store)
        .await
        .unwrap();
    let memory_bytes = dataset
        .object_store
        .as_ref()
        .io_stats_incremental()
        .read_bytes;
    let spilled = index
        .prepare_pairwise_partition(0, BATCH_SIZE, 0, store)
        .await
        .unwrap();
    let spilled_bytes = dataset
        .object_store
        .as_ref()
        .io_stats_incremental()
        .read_bytes;
    assert!(memory_bytes > 0);
    // Spilled source reads fetch each column's bytes about once; small
    // tail ranges round up to encoding chunks. Rereading the PQ code
    // column for every source window used to cost ~2.6x.
    assert!(
        spilled_bytes < 2 * memory_bytes,
        "spilled preparation read {spilled_bytes} bytes, in-memory {memory_bytes}"
    );
    dataset.object_store.as_ref().io_stats_incremental();
    let mut seen = HashSet::new();
    for _ in 0..3 {
        for batch_id in 0..NUM_ROWS.div_ceil(BATCH_SIZE) {
            let expected = memory.read_vectors(batch_id).await.unwrap();
            let actual = spilled.read_vectors(batch_id).await.unwrap();
            assert_eq!(actual.row_ids, expected.row_ids);
            assert_eq!(actual.codes, expected.codes);
            assert_eq!(
                actual.row_ids.len(),
                BATCH_SIZE.min(NUM_ROWS - batch_id * BATCH_SIZE)
            );
            seen.extend(actual.row_ids.values().iter().copied());
        }
    }
    assert_eq!(seen, (0..NUM_ROWS as u64).collect::<HashSet<u64>>());
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_eq!(
        stats.read_iops, 0,
        "replay must not reread the source index"
    );
    assert_eq!(stats.read_bytes, 0);

    // Spilled and in-memory staging score bit-identically, in every tile.
    let anchor = memory.read_vectors(0).await.unwrap();
    let spilled_anchor = spilled.read_vectors(0).await.unwrap();
    for batch_id in 0..NUM_ROWS.div_ceil(BATCH_SIZE) {
        let expected = memory
            .score_block(
                &anchor,
                0..32,
                &memory.read_vectors(batch_id).await.unwrap(),
                f32::MAX,
            )
            .unwrap();
        let actual = spilled
            .score_block(
                &spilled_anchor,
                0..32,
                &spilled.read_vectors(batch_id).await.unwrap(),
                f32::MAX,
            )
            .unwrap();
        let candidates = BATCH_SIZE.min(NUM_ROWS - batch_id * BATCH_SIZE);
        let pairs = if batch_id == 0 {
            (0..32).map(|row| candidates - row - 1).sum()
        } else {
            32 * candidates
        };
        assert_eq!(expected.distances.len(), pairs);
        assert_eq!(actual, expected);
    }
}

async fn search_lightweight_pq_index(
    dataset: &Dataset,
    query: &dyn Array,
    k: usize,
    num_partitions: usize,
    refine_factor: u32,
    ef: usize,
    distance_type: DistanceType,
) -> RecordBatch {
    dataset
        .scan()
        .nearest("vector", query, k)
        .unwrap()
        .distance_metric(distance_type)
        .minimum_nprobes(num_partitions)
        .ef(ef)
        .refine(refine_factor)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap()
}

async fn assert_lightweight_pq_index(distance_type: DistanceType, num_bits: usize, use_hnsw: bool) {
    const INDEX_NAME: &str = "test_index";
    const K: usize = 10;

    let test_dir = TempStrDir::default();
    let (batch, schema) = make_seeded_vector_batch(LIGHTWEIGHT_PQ_ROWS);
    let vectors = batch["vector"].as_fixed_size_list().clone();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
        .await
        .unwrap();

    let mut ivf_params = IvfBuildParams::new(LIGHTWEIGHT_PQ_PARTITIONS);
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = LIGHTWEIGHT_IVF_SAMPLE_RATE;
    let pq_params = lightweight_pq_params_with_bits(num_bits);
    let expected_num_sub_vectors = pq_params.num_sub_vectors;
    let params = if use_hnsw {
        VectorIndexParams::with_ivf_hnsw_pq_params(
            distance_type,
            ivf_params,
            lightweight_hnsw_params(),
            pq_params,
        )
    } else {
        VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params)
    };
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let stats_json = dataset.index_statistics(INDEX_NAME).await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
    let expected_index_type = if use_hnsw { "IVF_HNSW_PQ" } else { "IVF_PQ" };
    let expected_sub_index = if use_hnsw { "HNSW" } else { "PQ" };
    assert_eq!(stats["index_type"], expected_index_type);
    let expected_partitions = supported_partitions(
        LIGHTWEIGHT_PQ_ROWS,
        LIGHTWEIGHT_IVF_SAMPLE_RATE,
        LIGHTWEIGHT_PQ_PARTITIONS,
    );
    assert_eq!(stats["indices"][0]["num_partitions"], expected_partitions);
    assert_eq!(
        stats["indices"][0]["sub_index"]["index_type"],
        expected_sub_index
    );
    assert_eq!(stats["indices"][0]["sub_index"]["nbits"], num_bits);
    assert_eq!(
        stats["indices"][0]["sub_index"]["num_sub_vectors"],
        expected_num_sub_vectors
    );
    if use_hnsw {
        let hnsw_params = &stats["indices"][0]["sub_index"]["params"];
        assert_eq!(hnsw_params["max_level"], 2);
        assert_eq!(hnsw_params["m"], 4);
        assert_eq!(hnsw_params["ef_construction"], 16);
    }

    // A single k=10 query of 4-bit PQ + Dot sits near the 0.5 bar: Windows
    // CI has landed at 0.4 (4/10). Average over several queries so one
    // platform-dependent ranking does not fail the contract.
    const NUM_QUERIES: usize = 10;
    // 4-bit PQ ranking is coarser; rescoring more candidates is cheap on
    // this 256-row fixture and keeps mean recall clear of 0.5.
    let refine_factor = if num_bits == 4 { 8 } else { 4 };
    // HNSW requires ef >= k * refine. IVF_PQ ignores ef.
    let ef = 64.max(K * refine_factor as usize);
    let mut hits = 0usize;
    let query = vectors.value(0);
    let before_reopen = search_lightweight_pq_index(
        &dataset,
        query.as_ref(),
        K,
        LIGHTWEIGHT_PQ_PARTITIONS,
        refine_factor,
        ef,
        distance_type,
    )
    .await;
    for query_idx in 0..NUM_QUERIES {
        let query = vectors.value(query_idx);
        let ground_truth = ground_truth(&dataset, "vector", query.as_ref(), K, distance_type).await;
        let result = if query_idx == 0 {
            before_reopen.clone()
        } else {
            search_lightweight_pq_index(
                &dataset,
                query.as_ref(),
                K,
                LIGHTWEIGHT_PQ_PARTITIONS,
                refine_factor,
                ef,
                distance_type,
            )
            .await
        };
        assert_eq!(result.num_rows(), K);
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), K);
        let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
        assert!(distances.iter().all(|distance| distance.is_finite()));
        assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
        hits += row_ids
            .iter()
            .filter(|row_id| ground_truth.contains(row_id))
            .count();
    }
    let recall = hits as f32 / (NUM_QUERIES * K) as f32;
    assert_ge!(recall, 0.5, "recall: {recall}");

    drop(dataset);
    let reopened = Dataset::open(test_dir.as_str()).await.unwrap();
    let reopened_stats: serde_json::Value =
        serde_json::from_str(&reopened.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
    assert_eq!(reopened_stats, stats);
    assert_eq!(
        search_lightweight_pq_index(
            &reopened,
            query.as_ref(),
            K,
            LIGHTWEIGHT_PQ_PARTITIONS,
            refine_factor,
            ef,
            distance_type,
        )
        .await,
        before_reopen
    );
}

async fn load_vector_index_context(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
) -> VectorIndexTestContext {
    let stats_json = dataset.index_statistics(index_name).await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
    let uuid_str = stats["indices"][0]["uuid"]
        .as_str()
        .expect("Index uuid should be present");
    let uuid = Uuid::parse_str(uuid_str).expect("uuid in stats should be a valid UUID");
    let index = dataset
        .open_vector_index(column, &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();

    VectorIndexTestContext {
        stats_json,
        stats,
        index,
    }
}

/// An empty HNSW partition must carry the segment's build parameters so that
/// `statistics()` never reports the sub-index defaults for a partition without
/// rows (regression: append-optimize segments with a leading empty partition
/// surfaced `{max_level: 7, m: 20, ef_construction: 150}` instead of the
/// reference segment's parameters).
#[tokio::test]
async fn test_empty_partition_metadata_preserves_hnsw_build_params() {
    const INDEX_NAME: &str = "test_index";

    let test_dir = TempStrDir::default();
    let (batch, schema) = make_seeded_vector_batch(LIGHTWEIGHT_PQ_ROWS);
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
        .await
        .unwrap();

    // A centroid at 100000.0 puts every row closer to the zero centroid, so
    // partition 0 is deterministically empty without retraining.
    let mut centroid_values = vec![0.0_f32; DIM * 2];
    for value in centroid_values.iter_mut().take(DIM) {
        *value = 100000.0;
    }
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroid_values), DIM as i32)
            .unwrap(),
    );
    let mut ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = LIGHTWEIGHT_IVF_SAMPLE_RATE;

    let params = VectorIndexParams::with_ivf_hnsw_pq_params(
        DistanceType::L2,
        ivf_params,
        lightweight_hnsw_params(),
        lightweight_pq_params(),
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let stats_json = dataset.index_statistics(INDEX_NAME).await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
    let hnsw_params = &stats["indices"][0]["sub_index"]["params"];
    assert_eq!(hnsw_params["max_level"], 2, "stats: {stats_json}");
    assert_eq!(hnsw_params["m"], 4, "stats: {stats_json}");
    assert_eq!(hnsw_params["ef_construction"], 16, "stats: {stats_json}");
}

async fn shrink_smallest_partition(
    dataset: &mut Dataset,
    index_name: &str,
    expected_after_join: usize,
    next_id: &mut u64,
) -> (usize, usize, usize) {
    const ROWS_TO_APPEND_FOR_JOIN: usize = 32;
    let row_count_before = dataset.count_all_rows().await.unwrap();
    let index_ctx = load_vector_index_context(dataset, "vector", index_name).await;
    let partitions = index_ctx.stats()["indices"][0]["partitions"]
        .as_array()
        .expect("partitions should be present");
    let (partition_idx, _size) = partitions
        .iter()
        .enumerate()
        .filter_map(|(idx, part)| part["size"].as_u64().map(|size| (idx, size)))
        .filter(|(_, size)| *size > 1)
        .min_by_key(|(_, size)| *size)
        .expect("should have at least one partition with joinable rows");

    let row_ids = load_partition_row_ids(index_ctx.ivf(), partition_idx).await;
    assert!(
        row_ids.len() > 1,
        "Partition {} should have removable rows",
        partition_idx
    );

    let rows = dataset
        .take_rows(&row_ids, dataset.schema().clone())
        .await
        .unwrap();
    let ids = rows["id"].as_primitive::<UInt64Type>().values();
    let template_values = rows["vector"]
        .as_fixed_size_list()
        .value(0)
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();

    delete_ids(dataset, &ids[1..]).await;
    compact_after_deletions(dataset).await;

    append_template_vector_with_start_id(
        dataset,
        ROWS_TO_APPEND_FOR_JOIN,
        &template_values,
        next_id,
    )
    .await;
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let post_ctx = load_vector_index_context(dataset, "vector", index_name).await;
    let post_partitions = post_ctx.num_partitions();
    assert_eq!(
        post_partitions,
        expected_after_join,
        "Expected partitions to be at most {} after join, got stats: {}",
        expected_after_join,
        post_ctx.stats_json()
    );

    let row_count_after = dataset.count_all_rows().await.unwrap();
    debug_assert!(
        row_count_before + ROWS_TO_APPEND_FOR_JOIN >= row_count_after,
        "row count should not increase after delete + append"
    );
    let deleted_rows = row_count_before + ROWS_TO_APPEND_FOR_JOIN - row_count_after;

    (deleted_rows, ROWS_TO_APPEND_FOR_JOIN, post_partitions)
}

async fn append_template_vector_with_start_id(
    dataset: &mut Dataset,
    rows: usize,
    template: &[f32],
    next_id: &mut u64,
) {
    append_template_vector_batch(dataset, rows, template, *next_id, None).await;
    *next_id += rows as u64;
}

async fn append_partition_templates(
    dataset: &mut Dataset,
    rows_per_template: usize,
    templates: &[Vec<f32>],
) {
    assert!(
        !templates.is_empty(),
        "at least one template is required for append"
    );
    for template in templates {
        assert_eq!(
            template.len(),
            DIM,
            "Template vector should have {} dimensions",
            DIM
        );
    }

    let start_id = dataset.count_all_rows().await.unwrap() as u64;
    let total_rows = rows_per_template * templates.len();
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_id..start_id + total_rows as u64,
    ));
    // A tiny per-row drift keeps the appended rows distinct (identical rows
    // cannot be split by clustering) without moving them off their template's
    // partition.
    let mut appended_values = Vec::with_capacity(total_rows * DIM);
    for template in templates {
        for row in 0..rows_per_template {
            let mut values = template.clone();
            values[0] += row as f32 * 0.0001;
            appended_values.extend_from_slice(&values);
        }
    }
    let vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(appended_values), DIM as i32)
            .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    dataset.append(batches, None).await.unwrap();
}

async fn append_template_vector_with_params(
    dataset: &mut Dataset,
    rows: usize,
    template: &[f32],
    write_params: Option<WriteParams>,
) {
    let start_id = dataset.count_all_rows().await.unwrap() as u64;
    append_template_vector_batch(dataset, rows, template, start_id, write_params).await;
}

async fn append_template_vector_batch(
    dataset: &mut Dataset,
    rows: usize,
    template: &[f32],
    start_id: u64,
    write_params: Option<WriteParams>,
) {
    assert_eq!(
        template.len(),
        DIM,
        "Template vector should have {} dimensions",
        DIM
    );

    let ids = Arc::new(UInt64Array::from_iter_values(
        start_id..start_id + rows as u64,
    ));
    // The same tiny per-row drift as `append_partition_templates`: a split
    // into `ceil(rows / target)` pieces needs that many distinct rows, and
    // identical rows would leave some pieces empty at random.
    let mut appended_values = Vec::with_capacity(rows * DIM);
    for row in 0..rows {
        let mut values = template.to_vec();
        values[0] += row as f32 * 0.0001;
        appended_values.extend_from_slice(&values);
    }
    let vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(appended_values), DIM as i32)
            .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let params = write_params.map(|mut params| {
        params.mode = WriteMode::Append;
        params
    });
    dataset.append(batches, params).await.unwrap();
}

#[allow(clippy::too_many_arguments)]
async fn append_and_verify_append_phase(
    dataset: &mut Dataset,
    index_name: &str,
    template: &[f32],
    next_id: &mut u64,
    rows_to_append: usize,
    expected_partitions: usize,
    expected_total_rows: usize,
    expected_index_count: usize,
    expect_split: bool,
) {
    append_template_vector_with_start_id(dataset, rows_to_append, template, next_id).await;
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let stats_json = dataset.index_statistics(index_name).await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();

    let indices = stats["indices"]
        .as_array()
        .expect("indices array should exist");
    if expect_split {
        assert_eq!(
            indices.len(),
            expected_index_count,
            "Expected {} index entries after split, got {}, stats: {}",
            expected_index_count,
            indices.len(),
            stats
        );
    } else {
        assert!(
            indices.len() >= expected_index_count,
            "Expected at least {} index entries after append, got {}, stats: {}",
            expected_index_count,
            indices.len(),
            stats
        );
    }
    assert!(
        stats["num_indices"].as_u64().unwrap() as usize >= expected_index_count,
        "num_indices should be at least {}, stats: {}",
        expected_index_count,
        stats
    );
    assert_eq!(
        stats["num_indexed_rows"].as_u64().unwrap() as usize,
        expected_total_rows,
        "Total indexed rows mismatch after append"
    );

    let base_index = indices
        .iter()
        .max_by_key(|entry| entry["num_partitions"].as_u64().unwrap_or(0))
        .expect("at least one index entry should exist");
    assert_eq!(
        base_index["num_partitions"].as_u64().unwrap() as usize,
        expected_partitions,
        "Partition count mismatch after append"
    );

    if expected_index_count == 1 {
        let partitions = base_index["partitions"]
            .as_array()
            .expect("partitions should exist");
        assert_eq!(
            partitions.len(),
            expected_partitions,
            "Expected {} partitions, found {}",
            expected_partitions,
            partitions.len()
        );
        let partition_sizes: Vec<usize> = partitions
            .iter()
            .map(|part| part["size"].as_u64().unwrap() as usize)
            .collect();
        let total_partition_rows: usize = partition_sizes.iter().sum();
        assert_eq!(
            total_partition_rows, expected_total_rows,
            "Partition sizes should sum to total rows: {:?}",
            partition_sizes
        );
    } else {
        assert!(
            !expect_split,
            "Split should result in a single merged index"
        );
    }

    assert_eq!(
        dataset.count_all_rows().await.unwrap(),
        expected_total_rows,
        "Dataset row count mismatch after append"
    );
}

async fn load_partition_row_ids(index: &IvfPq, partition_idx: usize) -> Vec<u64> {
    index
        .storage
        .load_partition(partition_idx, None)
        .await
        .unwrap()
        .row_ids()
        .copied()
        .collect()
}

async fn load_flat_partition_row_ids(index: &IvfFlatIndex, partition_idx: usize) -> Vec<u64> {
    index
        .storage
        .load_partition(partition_idx, None)
        .await
        .unwrap()
        .row_ids()
        .copied()
        .collect()
}

async fn delete_ids(dataset: &mut Dataset, ids: &[u64]) {
    if ids.is_empty() {
        return;
    }
    let predicate = ids
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    dataset
        .delete(&format!("id in ({})", predicate))
        .await
        .unwrap();
}

async fn compact_after_deletions(dataset: &mut Dataset) {
    compact_files(
        dataset,
        CompactionOptions {
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
}

async fn ground_truth(
    dataset: &Dataset,
    column: &str,
    query: &dyn Array,
    k: usize,
    distance_type: DistanceType,
) -> HashSet<u64> {
    let batch = dataset
        .scan()
        .with_row_id()
        .nearest(column, query, k)
        .unwrap()
        .distance_metric(distance_type)
        .use_index(false)
        .try_into_batch()
        .await
        .unwrap();
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect()
}

fn multivec_ground_truth(
    vectors: &ListArray,
    query: &dyn Array,
    k: usize,
    distance_type: DistanceType,
) -> Vec<(f32, u64)> {
    let query = if let Some(list_array) = query.as_list_opt::<i32>() {
        list_array.values().clone()
    } else {
        query.as_fixed_size_list().values().clone()
    };
    distance_type
        .multivec(&query, vectors)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, dist)| (dist, i as u64))
        .sorted_by(|a, b| a.0.total_cmp(&b.0))
        .take(k)
        .collect()
}

const TWO_FRAG_NUM_ROWS: usize = 2000;
const TWO_FRAG_DIM: usize = 128;
const TWO_FRAG_NUM_PARTITIONS: usize = 4;
const TWO_FRAG_NUM_SUBVECTORS: usize = 16;
const TWO_FRAG_NUM_BITS: usize = 8;
const TWO_FRAG_SAMPLE_RATE: usize = 7;
const TWO_FRAG_MAX_ITERS: u32 = 20;

fn make_two_fragment_batches() -> (Arc<Schema>, Vec<RecordBatch>) {
    let ids = Arc::new(UInt64Array::from_iter_values(0..TWO_FRAG_NUM_ROWS as u64));

    let values = generate_random_array_with_range(TWO_FRAG_NUM_ROWS * TWO_FRAG_DIM, 0.0..1.0);
    let vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), TWO_FRAG_DIM as i32)
            .unwrap(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();

    (schema, vec![batch])
}

async fn write_dataset_from_batches(
    test_uri: &str,
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
) -> Dataset {
    write_dataset_from_batches_with_max_rows(test_uri, schema, batches, 500).await
}

async fn write_dataset_from_batches_with_max_rows(
    test_uri: &str,
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    max_rows_per_file: usize,
) -> Dataset {
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let write_params = WriteParams {
        max_rows_per_file,
        mode: WriteMode::Overwrite,
        ..Default::default()
    };

    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap()
}

async fn prepare_global_ivf_pq(
    dataset: &Dataset,
    vector_column: &str,
) -> (IvfBuildParams, PQBuildParams) {
    prepare_ivf_pq(
        dataset,
        vector_column,
        TWO_FRAG_DIM,
        TWO_FRAG_NUM_PARTITIONS,
        TWO_FRAG_NUM_SUBVECTORS,
        TWO_FRAG_NUM_BITS,
        TWO_FRAG_MAX_ITERS,
        TWO_FRAG_SAMPLE_RATE,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn prepare_ivf_pq(
    dataset: &Dataset,
    vector_column: &str,
    expected_dimension: usize,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits: usize,
    max_iters: u32,
    sample_rate: usize,
) -> (IvfBuildParams, PQBuildParams) {
    let batch = dataset
        .scan()
        .project(&[vector_column.to_string()])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch
        .column_by_name(vector_column)
        .expect("vector column should exist")
        .as_fixed_size_list();

    let dim = vectors.value_length() as usize;
    assert_eq!(dim, expected_dimension, "unexpected vector dimension");

    let values = vectors.values().as_primitive::<Float32Type>();

    let kmeans_params = KMeansParams::new(None, max_iters, 1, DistanceType::L2);
    let kmeans =
        train_kmeans::<Float32Type>(values, kmeans_params, dim, num_partitions, sample_rate)
            .unwrap();

    let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
    let centroids_fsl =
        Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
    let mut ivf_params = IvfBuildParams::try_with_centroids(num_partitions, centroids_fsl).unwrap();
    ivf_params.max_iters = max_iters as usize;
    ivf_params.sample_rate = sample_rate;

    let mut pq_train_params = PQBuildParams::new(num_sub_vectors, num_bits);
    pq_train_params.max_iters = max_iters as usize;
    pq_train_params.sample_rate = sample_rate;

    let pq = pq_train_params.build(vectors, DistanceType::L2).unwrap();
    let codebook_flat = pq.codebook.values().as_primitive::<Float32Type>().clone();
    let pq_codebook: ArrayRef = Arc::new(codebook_flat);
    let mut pq_params = PQBuildParams::with_codebook(num_sub_vectors, num_bits, pq_codebook);
    pq_params.max_iters = max_iters as usize;
    pq_params.sample_rate = sample_rate;

    (ivf_params, pq_params)
}

async fn prepare_global_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
    let batch = dataset
        .scan()
        .project(&[vector_column.to_string()])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch
        .column_by_name(vector_column)
        .expect("vector column should exist")
        .as_fixed_size_list();

    let dim = vectors.value_length() as usize;
    assert_eq!(dim, TWO_FRAG_DIM, "unexpected vector dimension");

    let values = vectors.values().as_primitive::<Float32Type>();
    let kmeans_params = KMeansParams::new(None, TWO_FRAG_MAX_ITERS, 1, DistanceType::L2);
    let kmeans = train_kmeans::<Float32Type>(
        values,
        kmeans_params,
        dim,
        TWO_FRAG_NUM_PARTITIONS,
        TWO_FRAG_SAMPLE_RATE,
    )
    .unwrap();

    let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
    let centroids_fsl =
        Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
    let mut ivf_params =
        IvfBuildParams::try_with_centroids(TWO_FRAG_NUM_PARTITIONS, centroids_fsl).unwrap();
    ivf_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
    ivf_params.sample_rate = TWO_FRAG_SAMPLE_RATE;
    ivf_params
}

async fn build_segments_for_fragment_groups(
    dataset: &mut Dataset,
    fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
    params: &VectorIndexParams,
    index_name: &str,
) -> Vec<IndexMetadata> {
    let mut segments = Vec::new();

    for fragments in fragment_groups {
        let mut builder = dataset.create_index_builder(&["vector"], IndexType::Vector, params);
        builder = builder.name(index_name.to_string()).fragments(fragments);
        segments.push(builder.execute_uncommitted().await.unwrap());
    }

    segments
}

async fn build_ivfpq_for_fragment_groups(
    dataset: &mut Dataset,
    fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
    index_name: &str,
) {
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        ivf_params.clone(),
        pq_params.clone(),
    );

    let segments =
        build_segments_for_fragment_groups(dataset, fragment_groups, &params, index_name).await;
    let committed_segments =
        build_distributed_segments(dataset, segments, params.index_type(), index_name).await;
    assert!(!committed_segments.is_empty());
}

fn assert_centroids_equal(reference: &serde_json::Value, candidate: &serde_json::Value) {
    let centroids_a = reference["centroids"]
        .as_array()
        .expect("centroids should be an array");
    let centroids_b = candidate["centroids"]
        .as_array()
        .expect("centroids should be an array");
    assert_eq!(
        centroids_a.len(),
        centroids_b.len(),
        "num centroids mismatch",
    );
    for (row_a, row_b) in centroids_a.iter().zip(centroids_b.iter()) {
        let row_a = row_a
            .as_array()
            .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_a));
        let row_b = row_b
            .as_array()
            .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_b));
        assert_eq!(row_a.len(), row_b.len(), "centroid dim mismatch");
        for (va, vb) in row_a.iter().zip(row_b.iter()) {
            let fa = va.as_f64().expect("centroid must be numeric") as f32;
            let fb = vb.as_f64().expect("centroid must be numeric") as f32;
            assert!(
                (fa - fb).abs() <= 1e-4,
                "centroid mismatch: {} vs {}",
                fa,
                fb
            );
        }
    }
}

fn sum_partition_sizes(indices: &[serde_json::Value]) -> Vec<u64> {
    let mut totals = Vec::new();
    for index in indices {
        let partitions = index["partitions"]
            .as_array()
            .expect("partitions should be an array");
        if totals.is_empty() {
            totals.resize(partitions.len(), 0);
        } else {
            assert_eq!(totals.len(), partitions.len(), "num partitions mismatch");
        }
        for (total, partition) in totals.iter_mut().zip(partitions.iter()) {
            *total += partition["size"].as_u64().expect("partition size");
        }
    }
    totals
}

fn assert_ivf_layout_compatible(stats_a: &serde_json::Value, stats_b: &serde_json::Value) {
    let indices_a = stats_a["indices"]
        .as_array()
        .expect("indices should be an array");
    let indices_b = stats_b["indices"]
        .as_array()
        .expect("indices should be an array");
    assert!(
        !indices_a.is_empty() && !indices_b.is_empty(),
        "indices should not be empty",
    );

    let reference = &indices_a[0];
    for index in indices_a.iter().skip(1).chain(indices_b.iter()) {
        assert_centroids_equal(reference, index);
    }

    let sizes_a = sum_partition_sizes(indices_a);
    let sizes_b = sum_partition_sizes(indices_b);
    assert_eq!(sizes_a, sizes_b, "aggregated partition sizes mismatch");
}

/// Commit caller-defined segment groups as one logical index.
async fn build_distributed_segments(
    dataset: &mut Dataset,
    segments: Vec<IndexMetadata>,
    _index_type: IndexType,
    index_name: &str,
) -> Vec<IndexMetadata> {
    dataset
        .commit_existing_index_segments(index_name, "vector", segments.clone())
        .await
        .unwrap();
    segments
}

#[tokio::test]
async fn test_ivfpq_recall_performance_on_two_frags_single_vs_split() {
    const INDEX_NAME: &str = "vector_idx";

    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();

    let (schema, batches) = make_two_fragment_batches();

    let ds_single_uri = format!("{}/single", base_uri);
    let ds_split_uri = format!("{}/split", base_uri);

    let mut ds_single =
        write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
    let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

    let fragments_single = ds_single.get_fragments();
    assert!(
        fragments_single.len() >= 2,
        "expected at least 2 fragments in ds_single, got {}",
        fragments_single.len()
    );
    let fragments_split = ds_split.get_fragments();
    assert!(
        fragments_split.len() >= 2,
        "expected at least 2 fragments in ds_split, got {}",
        fragments_split.len()
    );

    let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;

    let group_single = vec![
        fragments_single[0].id() as u32,
        fragments_single[1].id() as u32,
    ];
    build_ivfpq_for_fragment_groups(
        &mut ds_single,
        vec![group_single],
        &ivf_params,
        &pq_params,
        INDEX_NAME,
    )
    .await;

    let group0 = vec![fragments_split[0].id() as u32];
    let group1 = vec![fragments_split[1].id() as u32];
    build_ivfpq_for_fragment_groups(
        &mut ds_split,
        vec![group0, group1],
        &ivf_params,
        &pq_params,
        INDEX_NAME,
    )
    .await;

    let stats_single_json = ds_single.index_statistics(INDEX_NAME).await.unwrap();
    let stats_split_json = ds_split.index_statistics(INDEX_NAME).await.unwrap();
    let stats_single: serde_json::Value = serde_json::from_str(&stats_single_json).unwrap();
    let stats_split: serde_json::Value = serde_json::from_str(&stats_split_json).unwrap();
    assert_ivf_layout_compatible(&stats_single, &stats_split);
    assert_eq!(
        stats_single["num_indexed_rows"],
        stats_split["num_indexed_rows"]
    );

    const K: usize = 10;
    const NUM_QUERIES: usize = 10;

    async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
        let mut ids_per_query = Vec::with_capacity(queries.len());
        for q in queries {
            let result = ds
                .scan()
                .with_row_id()
                .project(&["_rowid"] as &[&str])
                .unwrap()
                .nearest("vector", q.as_ref(), K)
                .unwrap()
                .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                .try_into_batch()
                .await
                .unwrap();

            let row_ids = result[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied()
                .collect::<Vec<u64>>();
            ids_per_query.push(row_ids);
        }
        ids_per_query
    }

    let query_batch = ds_single
        .scan()
        .project(&["vector"] as &[&str])
        .unwrap()
        .limit(Some(NUM_QUERIES as i64), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = query_batch["vector"].as_fixed_size_list();
    let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
        .map(|i| vectors.value(i) as Arc<dyn Array>)
        .collect();

    let ids_single = collect_row_ids(&ds_single, &queries).await;
    let ids_split = collect_row_ids(&ds_split, &queries).await;

    assert_eq!(
        ids_single, ids_split,
        "single vs split index returned different Top-K row ids",
    );
}

#[rstest]
#[case::ivf_flat(IndexType::IvfFlat)]
#[case::ivf_pq(IndexType::IvfPq)]
#[case::ivf_sq(IndexType::IvfSq)]
#[case::ivf_rq(IndexType::IvfRq)]
#[tokio::test]
async fn test_distributed_vector_build_commits_multiple_segments_and_preserves_query_results(
    #[case] index_type: IndexType,
) {
    const INDEX_NAME: &str = "vector_idx";
    const K: usize = 10;
    const NUM_QUERIES: usize = 10;

    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();

    // Generate the data once, then write it twice to two independent dataset URIs.
    let (schema, batches) = make_two_fragment_batches();

    let ds_single_uri = format!("{}/single", base_uri);
    let ds_split_uri = format!("{}/split", base_uri);

    let mut ds_single =
        write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
    let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

    // Ensure we have at least 2 fragments.
    let fragments_single = ds_single.get_fragments();
    assert!(
        fragments_single.len() >= 2,
        "expected at least 2 fragments in ds_single, got {}",
        fragments_single.len()
    );
    let fragments_split = ds_split.get_fragments();
    assert!(
        fragments_split.len() >= 2,
        "expected at least 2 fragments in ds_split, got {}",
        fragments_split.len()
    );

    let distributed_params = match index_type {
        IndexType::IvfFlat => {
            let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
        }
        IndexType::IvfPq => {
            let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
        }
        IndexType::IvfSq => {
            let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_sq_params(
                DistanceType::L2,
                ivf_params,
                SQBuildParams::default(),
            )
        }
        IndexType::IvfRq => {
            let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                ivf_params,
                RQBuildParams::with_rotation_type(1, RQRotationType::Fast),
            )
        }
        other => panic!("unsupported test index type: {}", other),
    };

    ds_single
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_string()),
            &distributed_params,
            true,
        )
        .await
        .unwrap();

    let fragment_groups = fragments_split
        .iter()
        .map(|fragment| vec![fragment.id() as u32])
        .collect::<Vec<_>>();
    let expected_segment_count = fragment_groups.len();
    let segments = build_segments_for_fragment_groups(
        &mut ds_split,
        fragment_groups,
        &distributed_params,
        INDEX_NAME,
    )
    .await;
    let segments =
        build_distributed_segments(&mut ds_split, segments, index_type, INDEX_NAME).await;
    assert_eq!(segments.len(), expected_segment_count);
    for segment in &segments {
        let segment_index = ds_split
            .indices_dir()
            .clone()
            .join(segment.uuid.to_string())
            .join(crate::index::INDEX_FILE_NAME);
        assert!(
            ds_split
                .object_store
                .as_ref()
                .exists(&segment_index)
                .await
                .unwrap(),
            "segment file should exist at {}",
            segment_index
        );
    }

    let committed_segments = ds_split.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(committed_segments.len(), expected_segment_count);
    for committed in committed_segments {
        let covered_fragments = committed
            .fragment_bitmap
            .as_ref()
            .expect("distributed segment should have fragment coverage");
        assert_eq!(covered_fragments.len(), 1);
    }

    async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
        let mut ids_per_query = Vec::with_capacity(queries.len());
        for q in queries {
            let result = ds
                .scan()
                .with_row_id()
                .project(&["_rowid"] as &[&str])
                .unwrap()
                .nearest("vector", q.as_ref(), K)
                .unwrap()
                .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                .try_into_batch()
                .await
                .unwrap();

            let row_ids = result[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied()
                .collect::<Vec<u64>>();
            ids_per_query.push(row_ids);
        }
        ids_per_query
    }

    // Collect a deterministic query set from ds_single.
    let query_batch = ds_single
        .scan()
        .project(&["vector"] as &[&str])
        .unwrap()
        .limit(Some(NUM_QUERIES as i64), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = query_batch["vector"].as_fixed_size_list();
    let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
        .map(|i| vectors.value(i) as Arc<dyn Array>)
        .collect();

    let ids_single = collect_row_ids(&ds_single, &queries).await;
    let ids_split = collect_row_ids(&ds_split, &queries).await;

    if index_type == IndexType::IvfRq {
        for row_ids in &ids_split {
            assert_eq!(
                row_ids.len(),
                K,
                "distributed IVF_RQ query should still return exactly {K} row ids",
            );
        }
    } else {
        assert_eq!(
            ids_single, ids_split,
            "single vs segmented distributed index returned different Top-K row ids",
        );
    }
}

#[rstest]
#[case::ivf_flat(IndexType::IvfFlat)]
#[case::ivf_pq(IndexType::IvfPq)]
#[case::ivf_sq(IndexType::IvfSq)]
#[tokio::test]
async fn test_distributed_vector_grouped_build_allows_concurrent_group_execution(
    #[case] index_type: IndexType,
) {
    const INDEX_NAME: &str = "grouped_idx";
    const K: usize = 10;
    const NUM_QUERIES: usize = 10;

    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();

    let (schema, batches) = make_two_fragment_batches();
    let ds_single_uri = format!("{}/grouped_single", base_uri);
    let ds_split_uri = format!("{}/grouped_split", base_uri);

    let mut ds_single =
        write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
    let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

    let distributed_params = match index_type {
        IndexType::IvfFlat => {
            let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
        }
        IndexType::IvfPq => {
            let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
        }
        IndexType::IvfSq => {
            let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
            VectorIndexParams::with_ivf_sq_params(
                DistanceType::L2,
                ivf_params,
                SQBuildParams::default(),
            )
        }
        other => panic!("unsupported test index type: {}", other),
    };

    ds_single
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_string()),
            &distributed_params,
            true,
        )
        .await
        .unwrap();

    let fragment_groups = ds_split
        .get_fragments()
        .into_iter()
        .map(|fragment| vec![fragment.id() as u32])
        .collect::<Vec<_>>();
    let segments = build_segments_for_fragment_groups(
        &mut ds_split,
        fragment_groups,
        &distributed_params,
        INDEX_NAME,
    )
    .await;

    assert!(segments.len() >= 4);
    let grouped_inputs = segments
        .chunks(2)
        .map(|group| group.to_vec())
        .collect::<Vec<_>>();
    let mut expected_fragment_coverage = grouped_inputs
        .iter()
        .map(|group| {
            group
                .iter()
                .flat_map(|partial| {
                    partial
                        .fragment_bitmap
                        .as_ref()
                        .expect("partial shard should have fragment coverage")
                        .iter()
                })
                .sorted()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    expected_fragment_coverage.sort();

    let grouped_segments = futures::future::try_join_all(
        grouped_inputs
            .into_iter()
            .map(|group| ds_split.merge_existing_index_segments(group)),
    )
    .await
    .unwrap();
    let grouped_segments =
        build_distributed_segments(&mut ds_split, grouped_segments, index_type, INDEX_NAME).await;
    assert_eq!(grouped_segments.len(), expected_fragment_coverage.len());
    let mut actual_fragment_coverage = grouped_segments
        .iter()
        .map(|segment| {
            segment
                .fragment_bitmap
                .as_ref()
                .expect("segment should have fragment coverage")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    actual_fragment_coverage.sort();
    assert_eq!(
        actual_fragment_coverage, expected_fragment_coverage,
        "built segment coverage should equal the union of its source partial shards",
    );

    async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
        let mut ids_per_query = Vec::with_capacity(queries.len());
        for q in queries {
            let result = ds
                .scan()
                .with_row_id()
                .project(&["_rowid"] as &[&str])
                .unwrap()
                .nearest("vector", q.as_ref(), K)
                .unwrap()
                .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                .try_into_batch()
                .await
                .unwrap();

            ids_per_query.push(
                result[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
            );
        }
        ids_per_query
    }

    let query_batch = ds_single
        .scan()
        .project(&["vector"] as &[&str])
        .unwrap()
        .limit(Some(NUM_QUERIES as i64), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = query_batch["vector"].as_fixed_size_list();
    let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
        .map(|i| vectors.value(i) as Arc<dyn Array>)
        .collect();

    let ids_single = collect_row_ids(&ds_single, &queries).await;
    let ids_split = collect_row_ids(&ds_split, &queries).await;
    if matches!(index_type, IndexType::IvfSq) {
        for (single, split) in ids_single.iter().zip(ids_split.iter()) {
            assert_eq!(single.len(), split.len());
            let overlap = single
                .iter()
                .filter(|row_id| split.contains(row_id))
                .count();
            assert!(
                overlap >= K / 3,
                "single vs segmented distributed SQ index returned too little top-k overlap",
            );
        }
    } else {
        assert_eq!(ids_single, ids_split);
    }
}

#[tokio::test]
async fn test_distributed_vector_plan_rejects_overlapping_fragment_coverage() {
    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/overlap_fragments", base_uri);
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragment = dataset.get_fragments()[0].id() as u32;
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        prepare_global_ivf(&dataset, "vector").await,
    );
    let mut segments = Vec::new();

    for _ in 0..2 {
        let segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .fragments(vec![fragment])
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(segment);
    }

    let err = dataset
        .merge_existing_index_segments(segments)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("overlapping fragment coverage"));
}

async fn build_ivf_flat_segment(
    dataset: &mut Dataset,
    metric: DistanceType,
    fragment_ids: Vec<u32>,
) -> IndexMetadata {
    // Each segment trains its own IVF model, as distributed workers do.
    // The build name differs from the committed index name on purpose:
    // builders may not reuse the name of an existing index.
    let params = VectorIndexParams::ivf_flat(TWO_FRAG_NUM_PARTITIONS, metric);
    dataset
        .create_index_builder(&["vector"], IndexType::Vector, &params)
        .name("worker_idx".to_string())
        .fragments(fragment_ids)
        .execute_uncommitted()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_commit_index_segments_rejects_mixed_vector_metrics() {
    let test_dir = TempStrDir::default();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/mixed_metric_segments", test_dir.as_str());
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);
    let l2_segment = build_ivf_flat_segment(
        &mut dataset,
        DistanceType::L2,
        vec![fragments[0].id() as u32],
    )
    .await;
    let cosine_segment = build_ivf_flat_segment(
        &mut dataset,
        DistanceType::Cosine,
        vec![fragments[1].id() as u32],
    )
    .await;

    let version_before = dataset.manifest.version;
    let err = dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![l2_segment, cosine_segment])
        .await
        .unwrap_err();
    assert!(
        matches!(err, lance_core::Error::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );
    assert!(
        err.to_string().contains("metric"),
        "error should name the metric mismatch: {err}"
    );
    assert_eq!(dataset.manifest.version, version_before);
    assert!(
        dataset
            .load_indices_by_name("vector_idx")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn test_commit_index_segments_rejects_metric_mismatch_with_retained_segment() {
    let test_dir = TempStrDir::default();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/retained_mixed_metric_segments", test_dir.as_str());
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);
    let l2_segment = build_ivf_flat_segment(
        &mut dataset,
        DistanceType::L2,
        vec![fragments[0].id() as u32],
    )
    .await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![l2_segment])
        .await
        .unwrap();

    let version_before = dataset.manifest.version;
    let cosine_segment = build_ivf_flat_segment(
        &mut dataset,
        DistanceType::Cosine,
        vec![fragments[1].id() as u32],
    )
    .await;
    let err = dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![cosine_segment])
        .await
        .unwrap_err();
    assert!(
        matches!(err, lance_core::Error::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );
    assert!(
        err.to_string().contains("metric"),
        "error should name the metric mismatch: {err}"
    );
    assert_eq!(dataset.manifest.version, version_before);
    // The retained L2 segment still serves queries, unmodified.
    let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(indices.len(), 1);
}

#[tokio::test]
async fn test_commit_index_segments_allows_metric_change_on_full_replacement() {
    let test_dir = TempStrDir::default();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/full_replacement_metric_change", test_dir.as_str());
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);
    let all_fragment_ids = fragments.iter().map(|f| f.id() as u32).collect::<Vec<_>>();
    let l2_segment = build_ivf_flat_segment(
        &mut dataset,
        DistanceType::L2,
        vec![fragments[0].id() as u32],
    )
    .await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![l2_segment])
        .await
        .unwrap();

    // A new metric is valid when no old segment remains in the index.
    let cosine_segment =
        build_ivf_flat_segment(&mut dataset, DistanceType::Cosine, all_fragment_ids).await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![cosine_segment])
        .await
        .unwrap();
    let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(indices.len(), 1);
}

#[tokio::test]
async fn test_distributed_vector_build_supports_hnsw_variants() {
    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/distributed_hnsw_supported", base_uri);
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);
    let params = VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        prepare_global_ivf(&dataset, "vector").await,
        HnswBuildParams::default(),
    );
    let mut segments = Vec::new();

    for fragment in fragments.iter().take(2) {
        let segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .fragments(vec![fragment.id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(segment);
    }

    dataset
        .commit_existing_index_segments("vector_idx", "vector", segments)
        .await
        .unwrap();

    let query_batch = dataset
        .scan()
        .project(&["vector"] as &[&str])
        .unwrap()
        .limit(Some(4), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let q = query_batch["vector"].as_fixed_size_list().value(0);
    let result = dataset
        .scan()
        .project(&["_rowid"] as &[&str])
        .unwrap()
        .nearest("vector", q.as_ref(), 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(result.num_rows() > 0);
}

#[rstest]
#[case::flat("IVF_HNSW_FLAT")]
#[case::pq("IVF_HNSW_PQ")]
#[case::sq("IVF_HNSW_SQ")]
#[tokio::test]
async fn test_merge_existing_hnsw_segments_rebuilds_graph(#[case] expected_index_type: &str) {
    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();
    let (schema, batches, max_rows_per_file) = if expected_index_type == "IVF_HNSW_PQ" {
        let (batch, schema) = make_seeded_vector_batch(LIGHTWEIGHT_PQ_ROWS * 2);
        (schema, vec![batch], LIGHTWEIGHT_PQ_ROWS)
    } else {
        let (schema, batches) = make_two_fragment_batches();
        (schema, batches, 500)
    };
    let dataset_uri = format!("{}/merge_hnsw_rebuilds_graph", base_uri);
    let mut dataset =
        write_dataset_from_batches_with_max_rows(&dataset_uri, schema, batches, max_rows_per_file)
            .await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);
    let params = match expected_index_type {
        "IVF_HNSW_FLAT" => VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
        ),
        "IVF_HNSW_PQ" => {
            let (ivf_params, pq_params) = prepare_ivf_pq(
                &dataset,
                "vector",
                DIM,
                LIGHTWEIGHT_PQ_PARTITIONS,
                LIGHTWEIGHT_PQ_SUB_VECTORS,
                8,
                2,
                16,
            )
            .await;
            VectorIndexParams::with_ivf_hnsw_pq_params(
                DistanceType::L2,
                ivf_params,
                lightweight_hnsw_params(),
                pq_params,
            )
        }
        "IVF_HNSW_SQ" => VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
            SQBuildParams::default(),
        ),
        other => panic!("unexpected HNSW index type {other}"),
    };
    let mut segments = Vec::new();

    for fragment in fragments.iter().take(2) {
        let segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .fragments(vec![fragment.id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(segment);
    }

    let merged = dataset
        .merge_existing_index_segments(segments)
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![merged])
        .await
        .unwrap();

    let stats = dataset.index_statistics("vector_idx").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["index_type"].as_str().unwrap(), expected_index_type);
    assert_eq!(
        stats["indices"][0]["sub_index"]["index_type"]
            .as_str()
            .unwrap(),
        "HNSW"
    );

    let query_batch = dataset
        .scan()
        .project(&["vector"] as &[&str])
        .unwrap()
        .limit(Some(4), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let q = query_batch["vector"].as_fixed_size_list().value(0);
    let result = dataset
        .scan()
        .project(&["_rowid"] as &[&str])
        .unwrap()
        .nearest("vector", q.as_ref(), 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(result.num_rows() > 0);
}

#[tokio::test]
async fn test_merge_existing_hnsw_segments_rejects_mismatched_build_params() {
    let test_dir = TempStrDir::default();
    let base_uri = test_dir.as_str();
    let (schema, batches) = make_two_fragment_batches();
    let dataset_uri = format!("{}/merge_hnsw_rejects_mismatched_params", base_uri);
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(fragments.len() >= 2);

    let ivf_params = prepare_global_ivf(&dataset, "vector").await;
    let default_params = VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        ivf_params.clone(),
        HnswBuildParams::default(),
    );
    let custom_params = VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        ivf_params,
        HnswBuildParams::default().num_edges(16),
    );

    let first_segment = dataset
        .create_index_builder(&["vector"], IndexType::Vector, &default_params)
        .name("vector_idx".to_string())
        .fragments(vec![fragments[0].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();
    let second_segment = dataset
        .create_index_builder(&["vector"], IndexType::Vector, &custom_params)
        .name("vector_idx".to_string())
        .fragments(vec![fragments[1].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();

    let error = dataset
        .merge_existing_index_segments(vec![first_segment, second_segment])
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("HNSW build parameters mismatch while merging index segments"),
        "{error}"
    );
}

#[tokio::test]
async fn test_merge_index_metadata_reports_progress() {
    const INDEX_NAME: &str = "vector_idx";

    let test_dir = TempStrDir::default();
    let dataset_uri = format!("{}/progress", test_dir.as_str());
    let (schema, batches) = make_two_fragment_batches();
    let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

    let fragments = dataset.get_fragments();
    assert!(
        fragments.len() >= 2,
        "expected at least 2 fragments, got {}",
        fragments.len()
    );
    let expected_rows = fragments[0].physical_rows().await.unwrap() as u64
        + fragments[1].physical_rows().await.unwrap() as u64;
    let (ivf_params, pq_params) = prepare_global_ivf_pq(&dataset, "vector").await;
    let params = VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
    let mut segments = Vec::new();
    for fragment in fragments.iter().take(2) {
        segments.push(
            dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name(INDEX_NAME.to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }

    let progress = Arc::new(RecordingProgress::default());
    let merged_segment = crate::index::vector::ivf::merge_segments_with_progress(
        dataset.object_store.as_ref(),
        &dataset.indices_dir(),
        segments,
        progress.clone(),
    )
    .await
    .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, "vector", vec![merged_segment])
        .await
        .unwrap();

    let events = progress.recorded_events();
    let tags = events
        .iter()
        .map(|(kind, stage, _)| format!("{kind}:{stage}"))
        .collect::<Vec<_>>();
    let merge_total = events
        .iter()
        .find_map(|(kind, stage, value)| {
            if kind == "start" && stage == "merge_partitions" {
                Some(*value)
            } else {
                None
            }
        })
        .expect("missing merge_partitions start total");
    let merged_rows = events
        .iter()
        .filter_map(|(kind, stage, value)| {
            if kind == "progress" && stage == "merge_partitions" {
                Some(*value)
            } else {
                None
            }
        })
        .next_back()
        .unwrap_or_default();
    let read_start = tags
        .iter()
        .position(|e| e == "start:read_shard_metadata")
        .expect("missing read_shard_metadata start");
    let read_complete = tags
        .iter()
        .position(|e| e == "complete:read_shard_metadata")
        .expect("missing read_shard_metadata complete");
    let merge_start = tags
        .iter()
        .position(|e| e == "start:merge_partitions")
        .expect("missing merge_partitions start");
    let merge_complete = tags
        .iter()
        .position(|e| e == "complete:merge_partitions")
        .expect("missing merge_partitions complete");
    let aux_start = tags
        .iter()
        .position(|e| e == "start:write_auxiliary_index")
        .expect("missing write_auxiliary_index start");
    let aux_complete = tags
        .iter()
        .position(|e| e == "complete:write_auxiliary_index")
        .expect("missing write_auxiliary_index complete");
    let root_start = tags
        .iter()
        .position(|e| e == "start:write_root_index")
        .expect("missing write_root_index start");
    let root_complete = tags
        .iter()
        .position(|e| e == "complete:write_root_index")
        .expect("missing write_root_index complete");

    assert!(read_start < read_complete);
    assert!(read_complete < merge_start);
    assert!(merge_start < merge_complete);
    assert!(merge_complete < aux_start);
    assert!(aux_start < aux_complete);
    assert!(aux_complete < root_start);
    assert!(root_start < root_complete);
    assert_eq!(
        merge_total, expected_rows,
        "expected merge_partitions total rows to match dataset rows"
    );
    assert_eq!(
        merged_rows, expected_rows,
        "expected merge_partitions completed rows to match dataset rows"
    );
    assert!(
        tags.iter().any(|e| e == "progress:write_root_index"),
        "expected write_root_index progress callbacks"
    );
}

#[tokio::test]
async fn test_distributed_ivf_sq_worker_training_respects_fragment_filter() {
    const ROWS_PER_FRAGMENT: usize = 64;
    const FRAGMENT_OFFSETS: [f32; 2] = [0.0, 1000.0];

    let test_dir = TempStrDir::default();
    let dataset_uri = format!("{}/distributed_sq_fragment_filter", test_dir.as_str());
    let (schema, batches) = make_fragment_offset_batches(ROWS_PER_FRAGMENT, &FRAGMENT_OFFSETS);
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        batches,
        &dataset_uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            mode: WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), FRAGMENT_OFFSETS.len());

    let ivf_params =
        IvfBuildParams::try_with_centroids(2, build_centroids_for_offsets(&FRAGMENT_OFFSETS))
            .unwrap();
    let params = VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        ivf_params,
        SQBuildParams::default(),
    );

    let segment = dataset
        .create_index_builder(&["vector"], IndexType::Vector, &params)
        .name("sq_fragment_filter".to_string())
        .fragments(vec![fragments[0].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();

    let scheduler = ScanScheduler::new(
        Arc::new(dataset.object_store.as_ref().clone()),
        SchedulerConfig::default_for_testing(),
    );
    let sq_meta = get_sq_metadata(&dataset, scheduler, &segment.uuid.to_string()).await;

    assert_eq!(sq_meta.bounds.start, 0.0);
    assert_eq!(sq_meta.bounds.end, (DIM - 1) as f64);
    assert_lt!(sq_meta.bounds.end, FRAGMENT_OFFSETS[1] as f64);
}

async fn test_index(
    params: VectorIndexParams,
    nlist: usize,
    recall_requirement: f32,
    dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
) {
    match params.metric_type {
        DistanceType::Hamming => {
            test_index_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4, dataset).await;
        }
        _ => {
            test_index_impl::<Float32Type>(
                params.clone(),
                nlist,
                recall_requirement,
                0.0..1.0,
                dataset.clone(),
            )
            .await;

            if dataset.is_none() {
                test_index_impl::<Float64Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                    dataset,
                )
                .await;
            }
        }
    }
}

fn pq_matrix_batch<T>(rows: usize) -> RecordBatch
where
    T: ArrowPrimitiveType + 'static,
    T::Native: Copy + 'static,
    PrimitiveArray<T>: From<Vec<T::Native>> + 'static,
    StandardUniform: Distribution<T::Native>,
{
    gen_batch()
        .with_seed(Seed(42))
        .col("id", array::step::<UInt64Type>())
        .col("vector", array::rand_vec::<T>(Dimension::from(DIM as u32)))
        .into_batch_rows(RowCount::from(rows as u64))
        .unwrap()
}

fn pq_matrix_params(
    nlist: usize,
    distance_type: DistanceType,
    version: IndexFileVersion,
) -> VectorIndexParams {
    let mut ivf_params = IvfBuildParams::new(nlist);
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = PQ_MATRIX_NUM_ROWS;
    let pq_params = PQBuildParams {
        num_sub_vectors: 4,
        num_bits: 8,
        max_iters: 2,
        sample_rate: 1,
        ..Default::default()
    };
    let mut params = VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
    params.version(version);
    params
}

async fn test_pq_matrix_case(nlist: usize, distance_type: DistanceType, version: IndexFileVersion) {
    const INDEX_NAME: &str = "pq_matrix";

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let params = pq_matrix_params(nlist, distance_type, version.clone());
    let batch = pq_matrix_batch::<Float32Type>(PQ_MATRIX_NUM_ROWS);
    // `pq_matrix_params` fits each centroid on the whole fixture.
    let expected_partitions = supported_partitions(PQ_MATRIX_NUM_ROWS, PQ_MATRIX_NUM_ROWS, nlist);
    let schema = batch.schema();
    let query = batch["vector"].as_fixed_size_list().value(0);
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();
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

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
    assert_eq!(stats["index_type"], "IVF_PQ");
    let indices = stats["indices"].as_array().unwrap();
    assert_eq!(indices.len(), 1);
    let index = &indices[0];
    assert_eq!(index["index_type"], "IVF_PQ");
    assert_eq!(index["metric_type"], distance_type.to_string());
    assert_eq!(index["num_partitions"], expected_partitions);
    assert_eq!(index["sub_index"]["index_type"], "PQ");
    assert_eq!(
        index["index_file_version"],
        match version {
            IndexFileVersion::Legacy => "Legacy",
            IndexFileVersion::V3 => "V3",
        }
    );

    drop(dataset);
    let dataset = Dataset::open(test_uri).await.unwrap();
    let ground_truth = ground_truth(
        &dataset,
        "vector",
        query.as_ref(),
        PQ_MATRIX_K,
        distance_type,
    )
    .await;
    let result = dataset
        .scan()
        .nearest("vector", query.as_primitive::<Float32Type>(), PQ_MATRIX_K)
        .unwrap()
        .minimum_nprobes(nlist)
        .maximum_nprobes(nlist)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(result.num_rows(), PQ_MATRIX_K);
    let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
    assert_eq!(
        row_ids.iter().copied().collect::<HashSet<_>>().len(),
        PQ_MATRIX_K
    );
    let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
    assert!(distances.iter().all(|distance| distance.is_finite()));
    assert!(
        distances.windows(2).all(|pair| pair[0] <= pair[1]),
        "distances are not sorted: {distances:?}"
    );
    let recall = row_ids
        .iter()
        .filter(|row_id| ground_truth.contains(row_id))
        .count() as f32
        / PQ_MATRIX_K as f32;
    assert_ge!(recall, 0.5, "recall: {recall}, row_ids: {row_ids:?}");
}

async fn test_index_impl<T: ArrowPrimitiveType>(
    params: VectorIndexParams,
    nlist: usize,
    recall_requirement: f32,
    range: Range<T::Native>,
    dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
) where
    T::Native: SampleUniform,
{
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = match dataset {
        Some((dataset, vectors)) => (dataset, vectors),
        None => generate_test_dataset::<T>(test_uri, range).await,
    };

    let vector_column = "vector";
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    test_recall::<T>(
        params.clone(),
        nlist,
        recall_requirement,
        vector_column,
        &dataset,
        vectors.clone(),
    )
    .await;

    if params.stages.len() > 1
        && matches!(params.version, IndexFileVersion::V3)
        && params.index_type() == IndexType::IvfPq
    {
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let old_meta = indices[0].clone();
        rewrite_pq_storage(&mut dataset, &old_meta).await.unwrap();
        // do the test again
        test_recall::<T>(
            params,
            nlist,
            recall_requirement,
            vector_column,
            &dataset,
            vectors.clone(),
        )
        .await;
    }
}

async fn test_remap(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
    match params.metric_type {
        DistanceType::Hamming => {
            Box::pin(test_remap_impl::<UInt8Type>(
                params,
                nlist,
                recall_requirement,
                0..4,
            ))
            .await;
        }
        _ => {
            let index_type = params.index_type();
            Box::pin(test_remap_impl::<Float32Type>(
                params.clone(),
                nlist,
                recall_requirement,
                0.0..1.0,
            ))
            .await;
            if matches!(index_type, IndexType::IvfFlat | IndexType::IvfHnswFlat) {
                Box::pin(test_remap_impl::<Float64Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                ))
                .await;
            }
        }
    }
}

async fn test_remap_impl<T: ArrowPrimitiveType>(
    params: VectorIndexParams,
    nlist: usize,
    recall_requirement: f32,
    range: Range<T::Native>,
) where
    T::Native: SampleUniform,
{
    // let recall_requirement = recall_requirement * 0.99;
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

    let vector_column = "vector";
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let query = vectors.value(0);
    // delete half rows to trigger compact
    let half_rows = NUM_ROWS / 2;
    dataset
        .delete(&format!("id < {}", half_rows))
        .await
        .unwrap();
    // update the other half rows
    let update_result = UpdateBuilder::new(Arc::new(dataset))
        .update_where(&format!("id >= {} and id<{}", half_rows, half_rows + 50))
        .unwrap()
        .set("id", &format!("{}+id", NUM_ROWS))
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    let mut dataset = Dataset::open(update_result.new_dataset.uri())
        .await
        .unwrap();
    let num_rows = dataset.count_rows(None).await.unwrap();
    assert_eq!(num_rows, half_rows);
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    // query again, the result should not include the deleted row
    let result = dataset.scan().try_into_batch().await.unwrap();
    let ids = result["id"].as_primitive::<UInt64Type>();
    assert_eq!(ids.len(), half_rows);
    ids.values().iter().for_each(|id| {
        assert!(*id >= half_rows as u64 + 50);
    });

    // make sure we can still hit the recall
    let gt = ground_truth(&dataset, vector_column, &query, 100, params.metric_type).await;
    let results = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), 100)
        .unwrap()
        .minimum_nprobes(nlist)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let recall = row_ids.intersection(&gt).count() as f32 / 100.0;
    // 100 can't be exactly expressed as a float, so we need to use a tolerance
    assert_ge!(
        recall,
        recall_requirement - f32::EPSILON,
        "num_rows: {}, intersection: {}, recall: {}",
        row_ids.len(),
        row_ids.intersection(&gt).count(),
        recall
    );

    // delete so that only one row left, to trigger remap and there must be some empty partitions
    let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    dataset.delete("id > 0").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    let results = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), 100)
        .unwrap()
        .minimum_nprobes(nlist)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);
}

async fn test_delete_all_rows(params: VectorIndexParams) {
    // Boxed for CI clippy `large_futures`: each typed delete-all future grew past 16 KiB.
    match params.metric_type {
        DistanceType::Hamming => {
            Box::pin(test_delete_all_rows_impl::<UInt8Type>(params, 0..4)).await;
        }
        _ => {
            Box::pin(test_delete_all_rows_impl::<Float32Type>(params, 0.0..1.0)).await;
        }
    }
}

async fn test_delete_all_rows_impl<T: ArrowPrimitiveType>(
    params: VectorIndexParams,
    range: Range<T::Native>,
) where
    T::Native: SampleUniform,
{
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

    let vector_column = "vector";
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset.delete("id >= 0").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

    // optimize after delete all rows
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let query = vectors.value(0);
    let results = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), 100)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 0);

    // compact after delete all rows
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;

    let vector_column = "vector";
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset.delete("id >= 0").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    let results = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), 100)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 0);
}

#[tokio::test]
async fn test_flat_knn() {
    test_distance_range(None, 4).await;
}

#[rstest]
#[case(4, DistanceType::L2, 1.0)]
#[case(4, DistanceType::Cosine, 1.0)]
#[case(4, DistanceType::Dot, 1.0)]
#[case(4, DistanceType::Hamming, 0.9)]
#[tokio::test]
async fn test_build_ivf_flat(
    #[case] nlist: usize,
    #[case] distance_type: DistanceType,
    #[case] recall_requirement: f32,
) {
    let params = VectorIndexParams::ivf_flat(nlist, distance_type);
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(params.clone(), nlist, recall_requirement, None)).await;
    if distance_type == DistanceType::Cosine {
        test_index_multivec(params.clone(), nlist, recall_requirement).await;
    }
    test_distance_range(Some(params.clone()), nlist).await;
    test_remap(params.clone(), nlist, recall_requirement).await;
    // Boxed for CI clippy `large_futures`: the delete-all future grew past 16 KiB.
    Box::pin(test_delete_all_rows(params)).await;
}

#[rstest]
#[case::l2(4, DistanceType::L2)]
#[case::cosine(4, DistanceType::Cosine)]
#[case::dot(4, DistanceType::Dot)]
#[tokio::test]
async fn test_build_ivf_pq(#[case] nlist: usize, #[case] distance_type: DistanceType) {
    test_pq_matrix_case(nlist, distance_type, IndexFileVersion::Legacy).await;
}

#[rstest]
#[case::l2_nlist1(1, DistanceType::L2)]
#[case::cosine_nlist1(1, DistanceType::Cosine)]
#[case::dot_nlist1(1, DistanceType::Dot)]
#[case::l2_nlist4(4, DistanceType::L2)]
#[case::cosine_nlist4(4, DistanceType::Cosine)]
#[case::dot_nlist4(4, DistanceType::Dot)]
#[tokio::test]
async fn test_build_ivf_pq_v3(#[case] nlist: usize, #[case] distance_type: DistanceType) {
    test_pq_matrix_case(nlist, distance_type, IndexFileVersion::V3).await;
}

#[rstest]
#[case::legacy(IndexFileVersion::Legacy)]
#[case::v3(IndexFileVersion::V3)]
#[tokio::test]
async fn test_ivf_pq_distance_range(#[case] version: IndexFileVersion) {
    let params = pq_matrix_params(1, DistanceType::L2, version);
    test_distance_range(Some(params), 1).await;
}

#[rstest]
#[case::legacy(IndexFileVersion::Legacy)]
#[case::v3(IndexFileVersion::V3)]
#[tokio::test]
async fn test_ivf_pq_f64_smoke(#[case] version: IndexFileVersion) {
    let test_dir = TempStrDir::default();
    let batch = pq_matrix_batch::<Float64Type>(PQ_MATRIX_NUM_ROWS);
    let schema = batch.schema();
    let vectors = Arc::new(batch["vector"].as_fixed_size_list().clone());
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
        .await
        .unwrap();
    let params = pq_matrix_params(1, DistanceType::L2, version);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    test_recall::<Float64Type>(params, 1, 0.5, "vector", &dataset, vectors).await;
}

#[tokio::test]
async fn test_legacy_ivf_pq_cosine_multivec_smoke() {
    let mut ivf_params = IvfBuildParams::new(1);
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = PQ_MATRIX_NUM_ROWS;
    // Multi-vector scoring compounds quantization error. Four-dimensional
    // sub-vectors keep recall clear of the threshold with this small codebook
    // training sample; the single-vector matrix's eight-dimensional ones do not.
    let pq_params = PQBuildParams {
        num_sub_vectors: 8,
        num_bits: 8,
        max_iters: 2,
        sample_rate: 1,
        ..Default::default()
    };
    let mut params =
        VectorIndexParams::with_ivf_pq_params(DistanceType::Cosine, ivf_params, pq_params);
    params.version(IndexFileVersion::Legacy);
    test_index_multivec_impl::<Float32Type>(params, 1, 0.5, 0.0..1.0).await;
}

#[tokio::test]
async fn test_ivf_pq_delete_all_rows_lifecycle() {
    let params = pq_matrix_params(1, DistanceType::L2, IndexFileVersion::V3);
    // Boxed for CI clippy `large_futures`: the delete-all future grew past 16 KiB.
    Box::pin(test_delete_all_rows(params)).await;
}

#[rstest]
#[case::l2(DistanceType::L2)]
#[case::cosine(DistanceType::Cosine)]
#[case::dot(DistanceType::Dot)]
#[tokio::test]
async fn test_build_ivf_pq_4bit(#[case] distance_type: DistanceType) {
    assert_lightweight_pq_index(distance_type, 4, false).await;
}

#[rstest]
#[case(4, DistanceType::L2, 0.85)]
#[case(4, DistanceType::Cosine, 0.85)]
#[case(4, DistanceType::Dot, 0.75)]
#[tokio::test]
async fn test_build_ivf_sq(
    #[case] nlist: usize,
    #[case] distance_type: DistanceType,
    #[case] recall_requirement: f32,
) {
    let ivf_params = IvfBuildParams::new(nlist);
    let sq_params = SQBuildParams::default();
    let params = VectorIndexParams::with_ivf_sq_params(distance_type, ivf_params, sq_params);
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(params.clone(), nlist, recall_requirement, None)).await;
    if distance_type == DistanceType::Cosine {
        test_index_multivec(params.clone(), nlist, recall_requirement).await;
    }
    test_remap(params, nlist, recall_requirement).await;
}

#[tokio::test]
async fn test_build_ivf_sq_dot_with_negative_values() {
    let nlist = 4;
    let ivf_params = IvfBuildParams::new(nlist);
    let sq_params = SQBuildParams::default();
    let params = VectorIndexParams::with_ivf_sq_params(DistanceType::Dot, ivf_params, sq_params);

    test_index_impl::<Float32Type>(params, nlist, 0.75, -1.0..1.0, None).await;
}

// These queries probe every partition, so recall here measures RaBitQ quantization
// error alone. At 1 bit per dimension it averages ~0.67 on this uniformly random,
// L2-normalized data, and each build draws a fresh random rotation, so no bar worth
// asserting sits clear of the spread. 5 bits lifts recall to ~0.97; its `ex_bits = 4`
// also covers a FastScan ex-code kernel that the multi-bit test below never reaches.
#[rstest]
#[case(1, DistanceType::L2, 0.9)]
#[case(1, DistanceType::Cosine, 0.9)]
#[case(1, DistanceType::Dot, 0.9)]
#[case(4, DistanceType::L2, 0.9)]
#[case(4, DistanceType::Cosine, 0.9)]
#[case(4, DistanceType::Dot, 0.9)]
#[tokio::test]
async fn test_build_ivf_rq(
    #[case] nlist: usize,
    #[case] distance_type: DistanceType,
    #[case] recall_requirement: f32,
    #[values(RQRotationType::Fast, RQRotationType::Matrix)] rotation_type: RQRotationType,
) {
    let _ = env_logger::try_init();
    let ivf_params = IvfBuildParams::new(nlist);
    let rq_params = RQBuildParams::with_rotation_type(5, rotation_type);
    let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(params.clone(), nlist, recall_requirement, None)).await;
    if distance_type == DistanceType::Cosine {
        test_index_multivec(params.clone(), nlist, recall_requirement).await;
    }
    test_remap(params.clone(), nlist, recall_requirement).await;
}

#[rstest]
#[case::l2(DistanceType::L2, 9)]
#[case::cosine(DistanceType::Cosine, 9)]
// ex_bits=3 and ex_bits=5 have no FastScan support and use the bit-plane
// repack, so these searches go through the exact ex-dot rerank kernels
// end to end.
#[case::l2_plane_repack_3bit(DistanceType::L2, 4)]
#[case::l2_plane_repack_5bit(DistanceType::L2, 6)]
#[tokio::test]
async fn test_build_ivf_rq_multi_bit_persists_split_codes_and_searches(
    #[case] distance_type: DistanceType,
    #[case] num_bits: u8,
) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

    let ivf_params = IvfBuildParams::new(4);
    let rq_params = RQBuildParams::with_rotation_type(num_bits, RQRotationType::Fast);
    let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let obj_store = Arc::new(ObjectStore::local());
    let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
    let index_uuid = indices[0].uuid.to_string();
    let rq_meta = get_rq_metadata(&dataset, scheduler.clone(), &index_uuid).await;
    assert_eq!(rq_meta.num_bits, num_bits);
    assert_eq!(rq_meta.query_estimator, RabitQueryEstimator::RawQuery);

    let reader = open_rq_aux_reader(&dataset, scheduler, &index_uuid).await;
    let schema = reader.schema();
    let ex_field = schema.field(RABIT_BLOCKED_EX_CODE_COLUMN).unwrap();
    let DataType::FixedSizeList(_, ex_code_bytes) = ex_field.data_type() else {
        panic!("RQ ex-code field should be FixedSizeList");
    };
    let expected_ex_code_bytes = blocked_ex_code_bytes(rq_meta.rotated_dim(), num_bits - 1) as i32;
    assert_eq!(ex_code_bytes, expected_ex_code_bytes);
    assert!(schema.field(EX_ADD_FACTORS_COLUMN).is_some());
    assert!(schema.field(EX_SCALE_FACTORS_COLUMN).is_some());

    test_recall::<Float32Type>(params, 4, 0.5, "vector", &dataset, vectors).await;
}

#[rstest]
#[case::fast(RQRotationType::Fast)]
#[case::matrix(RQRotationType::Matrix)]
#[tokio::test]
async fn test_ivf_rq_rotation_type_after_optimize(#[case] rotation_type: RQRotationType) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

    let ivf_params = IvfBuildParams::new(4);
    let rq_params = RQBuildParams::with_rotation_type(1, rotation_type);
    let params = VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf_params, rq_params);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    assert_rq_rotation_type(&dataset, rotation_type).await;

    append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let indices_after_append = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_append.len(),
        2,
        "Expected append optimize to create one delta index"
    );
    assert_rq_rotation_type(&dataset, rotation_type).await;

    dataset
        .optimize_indices(&OptimizeOptions::merge(10))
        .await
        .unwrap();
    let indices_after_merge = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_merge.len(),
        1,
        "Expected merge optimize to merge indices into one"
    );
    assert_rq_rotation_type(&dataset, rotation_type).await;
}

#[rstest]
#[case(4, DistanceType::L2, 0.9)]
#[case(4, DistanceType::Cosine, 0.9)]
#[case(4, DistanceType::Dot, 0.85)]
#[case(4, DistanceType::Hamming, 0.9)]
#[tokio::test]
async fn test_create_ivf_hnsw_flat(
    #[case] nlist: usize,
    #[case] distance_type: DistanceType,
    #[case] recall_requirement: f32,
) {
    let ivf_params = IvfBuildParams::new(nlist);
    let hnsw_params = HnswBuildParams::default();
    let params = VectorIndexParams::ivf_hnsw(distance_type, ivf_params, hnsw_params);
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(params.clone(), nlist, recall_requirement, None)).await;
    if distance_type == DistanceType::Cosine {
        test_index_multivec(params.clone(), nlist, recall_requirement).await;
    }
    test_remap(params, nlist, recall_requirement).await;
}

#[rstest]
#[case(4, DistanceType::L2, 0.9)]
#[case(4, DistanceType::Cosine, 0.9)]
#[case(4, DistanceType::Dot, 0.85)]
#[tokio::test]
async fn test_create_ivf_hnsw_sq(
    #[case] nlist: usize,
    #[case] distance_type: DistanceType,
    #[case] recall_requirement: f32,
) {
    let ivf_params = IvfBuildParams::new(nlist);
    let sq_params = SQBuildParams::default();
    let hnsw_params = HnswBuildParams::default();
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        distance_type,
        ivf_params,
        hnsw_params,
        sq_params,
    );
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(params.clone(), nlist, recall_requirement, None)).await;
    if distance_type == DistanceType::Cosine {
        test_index_multivec(params.clone(), nlist, recall_requirement).await;
    }
    test_distance_range(Some(params.clone()), nlist).await;
    // Boxed for CI clippy `large_futures`: the delete-all future grew past 16 KiB.
    Box::pin(test_delete_all_rows(params.clone())).await;
    test_remap(params, nlist, recall_requirement).await;
}

#[tokio::test]
async fn test_create_ivf_hnsw_sq_dot_with_negative_values() {
    let nlist = 4;
    let ivf_params = IvfBuildParams::new(nlist);
    let sq_params = SQBuildParams::default();
    let hnsw_params = HnswBuildParams::default();
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::Dot,
        ivf_params,
        hnsw_params,
        sq_params,
    );

    test_index_impl::<Float32Type>(params, nlist, 0.75, -1.0..1.0, None).await;
}

#[rstest]
#[case::l2(DistanceType::L2)]
#[case::cosine(DistanceType::Cosine)]
#[case::dot(DistanceType::Dot)]
#[tokio::test]
async fn test_create_ivf_hnsw_pq(#[case] distance_type: DistanceType) {
    assert_lightweight_pq_index(distance_type, 8, true).await;
}

#[rstest]
#[case::l2(DistanceType::L2)]
#[case::cosine(DistanceType::Cosine)]
#[case::dot(DistanceType::Dot)]
#[tokio::test]
async fn test_create_ivf_hnsw_pq_4bit(#[case] distance_type: DistanceType) {
    assert_lightweight_pq_index(distance_type, 4, true).await;
}

#[tokio::test]
async fn test_create_ivf_hnsw_pq_multivec() {
    const NUM_ROWS: usize = 64;
    const K: usize = 10;

    let test_dir = TempStrDir::default();
    let batch = lance_datagen::gen_batch()
        .with_seed(lance_datagen::Seed::from(42))
        .col("id", lance_datagen::array::step::<UInt64Type>())
        .col(
            "vector",
            lance_datagen::array::cycle_vec_var(
                lance_datagen::array::rand_vec::<Float32Type>((DIM as u32).into()),
                3_u32.into(),
                4_u32.into(),
            ),
        )
        .into_batch_rows(lance_datagen::RowCount::from(NUM_ROWS as u64))
        .unwrap();
    let vectors = batch["vector"].as_list::<i32>().clone();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
        .await
        .unwrap();

    let mut ivf_params = IvfBuildParams::new(1);
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = 16;
    // Multivector search ranks the final candidates by PQ-approximated
    // scores before refining, so the M4 code of `lightweight_pq_params`
    // (unseeded KMeans) drops recall below the threshold in a few percent
    // of runs. The 4-bit M32 code keeps recall at 1.0.
    let params = VectorIndexParams::with_ivf_hnsw_pq_params(
        DistanceType::Cosine,
        ivf_params,
        lightweight_hnsw_params(),
        lightweight_pq_params_with_bits(4),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let query = vectors.value(0);
    // Three vectors per query amplify the internal candidate k. This
    // bounded budget covers all 64 * 3 vector entries in the fixture.
    let result =
        search_lightweight_pq_index(&dataset, query.as_ref(), K, 1, 2, 256, DistanceType::Cosine)
            .await;
    assert_eq!(result.num_rows(), K);
    let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
    assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), K);
    let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
    assert!(distances.iter().all(|distance| distance.is_finite()));
    assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));

    let ground_truth = multivec_ground_truth(&vectors, query.as_ref(), K, DistanceType::Cosine)
        .into_iter()
        .map(|(_, row_id)| row_id)
        .collect::<HashSet<_>>();
    let recall = row_ids
        .iter()
        .filter(|row_id| ground_truth.contains(row_id))
        .count() as f32
        / K as f32;
    assert_ge!(recall, 0.5, "recall: {recall}");
}

// `lance-index` keeps these crate-private; spelling them out here also pins
// the on-disk names, which are part of the index file contract.
const HNSW_VECTOR_ID_COL: &str = "__vector_id";
const HNSW_NEIGHBORS_COL: &str = "__neighbors";

/// Store wrapper that holds every read open for a measurable window and
/// records how many were in flight at once. Instantaneous reads never
/// overlap, so a delay is what makes concurrency observable at all.
#[derive(Debug)]
struct ConcurrencyProbeStore {
    target: Arc<dyn object_store::ObjectStore>,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: Arc<AtomicUsize>,
}

impl std::fmt::Display for ConcurrencyProbeStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConcurrencyProbeStore({})", self.target)
    }
}

impl ConcurrencyProbeStore {
    async fn enter(&self) {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for ConcurrencyProbeStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.target.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.enter().await;
        self.target.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        self.enter().await;
        self.target.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.target.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.target.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.target.copy_opts(from, to, opts).await
    }
}

/// Opening the storage reads the IVF protobuf and the quantizer buffer,
/// two independent global buffers whose positions both come from the schema
/// metadata. Reading them one after the other costs an extra round trip on
/// every cold open, so pin that they go out together: with the reads
/// serialized this sees one in flight at a time.
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_open_fetches_ivf_and_quantizer_buffers_together() {
    let (mut dataset, _) = generate_test_dataset::<Float32Type>("memory://", 0.0..1.0).await;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(16),
        PQBuildParams::new(4, 8),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();

    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let mut probed = dataset.object_store.as_ref().clone();
    probed.inner = Arc::new(ConcurrencyProbeStore {
        target: probed.inner.clone(),
        in_flight: in_flight.clone(),
        max_in_flight: max_in_flight.clone(),
    });
    let probed = Arc::new(probed);
    let scheduler = ScanScheduler::new(probed, SchedulerConfig::default_for_testing());
    let reader = open_rq_aux_reader(&dataset, scheduler, &indices[0].uuid.to_string()).await;
    max_in_flight.store(0, Ordering::SeqCst);
    let _storage = lance_index::vector::storage::IvfQuantizationStorage::<
        lance_index::vector::pq::ProductQuantizer,
    >::try_new(reader, None)
    .await
    .unwrap();
    assert_eq!(
        max_in_flight.load(Ordering::SeqCst),
        2,
        "both global buffer reads should be in flight at once"
    );
}

async fn build_ivf_hnsw_sq(test_uri: &str, nlist: usize) -> Dataset {
    let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(nlist),
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
}

async fn open_ivf_hnsw_sq(dataset: &Dataset) -> Arc<dyn VectorIndex> {
    let indices = dataset.load_indices().await.unwrap();
    dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap()
}

async fn assert_hnsw_columns(dataset: &Dataset, context: &str) {
    let index = open_ivf_hnsw_sq(dataset).await;
    let hnsw = index
        .as_any()
        .downcast_ref::<IvfHnswSqIndex>()
        .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");

    let written = hnsw
        .reader
        .schema()
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        written,
        vec![HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL, DIST_COL],
        "{context}: the written index file must keep every column"
    );

    // Every partition, not just the first: a projection that applied
    // unevenly would leave the partition cache holding mixed schemas.
    for partition_id in 0..hnsw.ivf.num_partitions() {
        let entry = hnsw.load_partition_entry(partition_id, None).await.unwrap();
        let loaded = entry.index.to_batch().unwrap();
        let loaded_schema = loaded.schema();
        let read = loaded_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            read,
            vec![HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL],
            "{context}: partition {partition_id} materialized the write-only distance column"
        );
    }
}

/// The index file keeps all three HNSW columns while a loaded partition
/// carries only the two the graph reads. Both halves matter: shrinking the
/// written schema would panic readers older than v8.0.0, and widening the
/// read back would undo the saving.
///
/// Re-checked after a delta merge, because that is the one path that writes
/// a new index file while an already-projected index is open.
#[tokio::test]
async fn test_hnsw_partition_load_reads_only_graph_columns() {
    let test_dir = TempStrDir::default();
    let mut dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
    assert_hnsw_columns(&dataset, "fresh index").await;

    append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(10))
        .await
        .unwrap();
    assert_hnsw_columns(&dataset, "after delta merge").await;
}

/// The saving is the point of the change, so pin it: reading a real
/// partition range through the declared projection must move strictly fewer
/// bytes than the full-schema read the index used to perform.
#[tokio::test]
async fn test_hnsw_read_projection_moves_fewer_bytes() {
    use futures::TryStreamExt as _;

    let test_dir = TempStrDir::default();
    let dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
    let index = open_ivf_hnsw_sq(&dataset).await;
    let hnsw = index
        .as_any()
        .downcast_ref::<IvfHnswSqIndex>()
        .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");

    let projection = hnsw
        .read_projection
        .as_ref()
        .expect("HNSW declares a read projection");
    assert_eq!(projection.schema.fields.len(), 2);

    let row_range = hnsw.ivf.row_range(0);
    assert!(!row_range.is_empty(), "partition 0 should hold rows");
    let store = dataset.object_store.as_ref();

    let read_bytes_for = async |projection: lance_file::reader::ReaderProjection| {
        let _ = store.io_stats_incremental();
        hnsw.reader
            .read_stream_projected(
                lance_io::ReadBatchParams::Range(row_range.clone()),
                u32::MAX,
                1,
                projection,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        store.io_stats_incremental().read_bytes
    };

    // Projected first on purpose: it then pays any first-touch metadata
    // cost, so the comparison understates rather than flatters the saving.
    let projected_bytes = read_bytes_for(projection.clone()).await;
    let full_bytes = read_bytes_for(lance_file::versions::reader_projection_from_whole_schema(
        hnsw.reader.schema(),
        hnsw.reader.metadata().version(),
    ))
    .await;

    assert!(
        projected_bytes > 0,
        "the projected read still has to fetch the graph"
    );
    assert_lt!(projected_bytes, full_bytes);
}

/// The projection selects columns by field id, and `__neighbors` and
/// `_distance` are both 4-byte-item lists whose child fields share a name,
/// so a wrong column index would reinterpret distances as neighbor ids with
/// no type error to catch it. Compare the columns themselves, not just names.
#[tokio::test]
async fn test_hnsw_projected_read_matches_full_read() {
    use futures::TryStreamExt as _;

    let test_dir = TempStrDir::default();
    let dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
    let index = open_ivf_hnsw_sq(&dataset).await;
    let hnsw = index
        .as_any()
        .downcast_ref::<IvfHnswSqIndex>()
        .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");
    let projection = hnsw
        .read_projection
        .as_ref()
        .expect("HNSW declares a read projection");

    let read_range = async |proj: lance_file::reader::ReaderProjection,
                            range: std::ops::Range<usize>| {
        let batches = hnsw
            .reader
            .read_stream_projected(
                lance_io::ReadBatchParams::Range(range),
                u32::MAX,
                1,
                proj,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
    };

    let mut compared = 0;
    for partition_id in 0..hnsw.ivf.num_partitions() {
        let range = hnsw.ivf.row_range(partition_id);
        if range.is_empty() {
            continue;
        }
        let full = read_range(
            lance_file::versions::reader_projection_from_whole_schema(
                hnsw.reader.schema(),
                hnsw.reader.metadata().version(),
            ),
            range.clone(),
        )
        .await;
        let projected = read_range(projection.clone(), range).await;

        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.num_rows(), full.num_rows());
        for name in [HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL] {
            assert_eq!(
                projected.column_by_name(name).unwrap(),
                full.column_by_name(name).unwrap(),
                "partition {partition_id}: {name} differs between the projected and full read"
            );
        }
        compared += 1;
    }
    assert!(compared > 0, "no non-empty partition was compared");
}

/// Each partition's graph as written, with the row ids of its storage.
async fn read_hnsw_graphs(dataset: &Dataset) -> Vec<(RecordBatch, Vec<u64>)> {
    let index = open_ivf_hnsw_sq(dataset).await;
    let hnsw = index
        .as_any()
        .downcast_ref::<IvfHnswSqIndex>()
        .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");
    let mut graphs = Vec::with_capacity(hnsw.ivf.num_partitions());
    for partition_id in 0..hnsw.ivf.num_partitions() {
        let graph = hnsw
            .read_sub_index_batch(partition_id, None, None)
            .await
            .unwrap();
        let storage = hnsw
            .load_partition_storage(partition_id, None)
            .await
            .unwrap();
        graphs.push((graph, storage.row_ids().copied().collect()));
    }
    graphs
}

fn level0_edges(graph: &RecordBatch) -> HashSet<(u32, u32)> {
    let metadata: HnswMetadata =
        serde_json::from_str(&graph.schema_ref().metadata()[HNSW_METADATA_KEY]).unwrap();
    let neighbors = graph[HNSW_NEIGHBORS_COL].as_list::<i32>();
    (0..metadata.level_offsets[1])
        .flat_map(|node| {
            let neighbors = neighbors.value(node);
            neighbors
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .map(|neighbor| (node as u32, *neighbor))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Compaction copies each HNSW graph when no row is deleted. When rows are
/// deleted the surviving edges are kept and the nodes that lost a neighbor
/// are reconnected, so the level-0 edges are not just the old edges relabeled.
#[rstest]
#[case::no_deletions(None, true)]
#[case::few_deletions(Some(20), false)]
#[case::many_deletions(Some(2), false)]
#[tokio::test]
async fn test_compaction_keeps_hnsw_graph(
    #[case] delete_every: Option<u64>,
    #[case] is_graph_kept: bool,
) {
    const NLIST: usize = 2;
    const K: usize = 10;
    const NUM_QUERIES: usize = 20;
    let test_dir = TempStrDir::default();
    let (batch, schema) = generate_batch::<Float32Type>(NUM_ROWS, None, 0.0..1.0, false);
    let vectors = batch["vector"].as_fixed_size_list().clone();
    let mut dataset = write_dataset_from_batches_with_max_rows(
        test_dir.as_str(),
        schema,
        vec![batch],
        NUM_ROWS / 4,
    )
    .await;
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(NLIST),
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let search = async |dataset: &Dataset, query: &dyn Array| {
        dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .with_row_id()
            .nearest("vector", query, K)
            .unwrap()
            .minimum_nprobes(NLIST)
            .try_into_batch()
            .await
            .unwrap()
    };
    let old_graphs = read_hnsw_graphs(&dataset).await;
    let mut old_results = Vec::with_capacity(NUM_QUERIES);
    for i in 0..NUM_QUERIES {
        old_results.push(search(&dataset, &vectors.value(i)).await);
    }
    let old_dataset = dataset.clone();
    let old_index_id = dataset.load_indices().await.unwrap()[0].uuid;

    let is_deleted = |id: u64| delete_every.is_some_and(|every| id.is_multiple_of(every));
    if let Some(every) = delete_every {
        dataset.delete(&format!("id % {every} = 0")).await.unwrap();
    }
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_ne!(dataset.load_indices().await.unwrap()[0].uuid, old_index_id);
    let new_graphs = read_hnsw_graphs(&dataset).await;
    assert_eq!(new_graphs.len(), old_graphs.len());

    if delete_every.is_none() {
        for ((old, old_rows), (new, new_rows)) in old_graphs.iter().zip(&new_graphs) {
            assert_eq!(old_rows.len(), new_rows.len());
            assert_eq!(old.columns(), new.columns());
            assert_eq!(
                old.schema_ref().metadata()[HNSW_METADATA_KEY],
                new.schema_ref().metadata()[HNSW_METADATA_KEY]
            );
        }
        // Same graph, same vectors: same neighbors at the same distances,
        // reported under the rewritten row addresses.
        for (i, old_result) in old_results.iter().enumerate() {
            let new_result = search(&dataset, &vectors.value(i)).await;
            assert_eq!(&old_result["id"], &new_result["id"], "query {i}");
            assert_eq!(&old_result[DIST_COL], &new_result[DIST_COL], "query {i}");
            assert_ne!(&old_result[ROW_ID], &new_result[ROW_ID], "query {i}");
        }
        return;
    }

    let id_projection = dataset.schema().project(&["id"]).unwrap();
    let mut num_rebuilt = 0;
    for ((old, old_rows), (new, new_rows)) in old_graphs.iter().zip(&new_graphs) {
        let old_ids = old_dataset
            .take_rows(old_rows, id_projection.clone())
            .await
            .unwrap();
        let mut new_of_old = Vec::with_capacity(old_rows.len());
        let mut num_kept = 0;
        for id in old_ids["id"].as_primitive::<UInt64Type>().values() {
            if is_deleted(*id) {
                new_of_old.push(None);
            } else {
                new_of_old.push(Some(num_kept));
                num_kept += 1;
            }
        }
        assert_eq!(num_kept as usize, new_rows.len());

        let new_edges = level0_edges(new);
        assert!(
            new_edges
                .iter()
                .all(|(from, to)| (*from as usize) < new_rows.len()
                    && (*to as usize) < new_rows.len())
        );
        let relabeled_old_edges = level0_edges(old)
            .into_iter()
            .filter_map(|(from, to)| Some((new_of_old[from as usize]?, new_of_old[to as usize]?)))
            .collect::<HashSet<_>>();
        if is_graph_kept {
            assert_eq!(new_edges, relabeled_old_edges);
        } else if !new_edges.is_subset(&relabeled_old_edges) {
            num_rebuilt += 1;
        }
    }
    // Repair adds links the old graph did not have. A tiny partition can
    // be fully linked either way, so require that of any one partition.
    assert_eq!(num_rebuilt > 0, !is_graph_kept);

    let mut hits = 0;
    for i in 0..NUM_QUERIES {
        let query = vectors.value(i);
        let result = search(&dataset, &query).await;
        let ids = result["id"].as_primitive::<UInt64Type>();
        assert!(ids.values().iter().all(|id| !is_deleted(*id)), "query {i}");
        // The row addresses search reports must point at the rows it matched.
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        let taken = dataset
            .take_rows(row_ids, id_projection.clone())
            .await
            .unwrap();
        assert_eq!(&taken["id"], &result["id"], "query {i}");

        let gt = ground_truth(&dataset, "vector", &query, K, DistanceType::L2).await;
        hits += row_ids.iter().filter(|id| gt.contains(id)).count();
    }
    let recall = hits as f32 / (NUM_QUERIES * K) as f32;
    assert_ge!(recall, 0.8, "recall after compaction with deletions");
}

async fn test_index_multivec(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
    // we introduce XTR for performance, which would reduce the recall a little bit
    let recall_requirement = recall_requirement * 0.9;
    match params.metric_type {
        DistanceType::Hamming => {
            test_index_multivec_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4).await;
        }
        _ => {
            test_index_multivec_impl::<Float32Type>(params, nlist, recall_requirement, 0.0..1.0)
                .await;
        }
    }
}

async fn test_index_multivec_impl<T: ArrowPrimitiveType>(
    params: VectorIndexParams,
    nlist: usize,
    recall_requirement: f32,
    range: Range<T::Native>,
) where
    T::Native: SampleUniform,
{
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let (mut dataset, vectors) = generate_multivec_test_dataset::<T>(test_uri, range).await;

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_index".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let query = vectors.value(0);
    let k = 100;

    let result = dataset
        .scan()
        .nearest("vector", &query, k)
        .unwrap()
        .minimum_nprobes(nlist)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = result[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    assert_eq!(row_ids.len(), k);
    assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), k);
    let dists = result[DIST_COL]
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let results = dists.into_iter().zip(row_ids.clone()).collect::<Vec<_>>();
    let row_ids = row_ids.into_iter().collect::<HashSet<_>>();

    let gt = multivec_ground_truth(&vectors, &query, k, params.metric_type);
    let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

    let recall = row_ids.intersection(&gt_set).count() as f32 / 100.0;
    assert!(
        recall >= recall_requirement,
        "recall: {}\n results: {:?}\n\ngt: {:?}",
        recall,
        results,
        gt
    );
}

#[rstest]
#[tokio::test]
async fn test_migrate_v1_to_v3() {
    // only test the case of IVF_PQ
    // because only IVF_PQ is supported in v1
    let nlist = 4;
    let recall_requirement = 0.9;
    let ivf_params = IvfBuildParams::new(nlist);
    let pq_params = PQBuildParams::default();
    let v1_params =
        VectorIndexParams::with_ivf_pq_params(DistanceType::Cosine, ivf_params, pq_params)
            .version(crate::index::vector::IndexFileVersion::Legacy)
            .clone();

    let v3_params = v1_params
        .clone()
        .version(crate::index::vector::IndexFileVersion::V3)
        .clone();

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(
        v1_params,
        nlist,
        recall_requirement,
        Some((dataset.clone(), vectors.clone())),
    ))
    .await;
    dataset.checkout_latest().await.unwrap();
    // retest with v3 params on the same dataset
    // Boxed for CI clippy `large_futures`: the build-and-search future grew past 16 KiB.
    Box::pin(test_index(
        v3_params,
        nlist,
        recall_requirement,
        Some((dataset.clone(), vectors)),
    ))
    .await;

    dataset.checkout_latest().await.unwrap();
    let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(indices.len(), 1); // v1 index should be replaced by v3 index
    let index = dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let v3_index = index.as_any().downcast_ref::<super::IvfPq>();
    assert!(v3_index.is_some());
}

/// The global-top-k path (flat sub-index, no early-stop control) must not
/// pin every probed partition before scoring: a prepared partition holds the
/// partition's whole quantized storage, so with a high explicit `nprobes`
/// that made peak memory scale with `nprobes` instead of with the prepare
/// window. Probe every partition of an index with more partitions than the
/// window and check the in-flight high-water mark stays within it.
#[tokio::test]
async fn test_global_topk_search_bounds_in_flight_prepared_partitions() {
    const INDEX_NAME: &str = "vector_idx";
    const K: usize = 10;
    const ROWS_PER_PARTITION: usize = 16;

    // Resident partitions are bounded by the prepare window plus two scoring
    // chunks (one being scored, one being assembled). The partitions here are
    // far smaller than the chunk byte budget, so the partition cap is what
    // ends a chunk. Size the index so that the old collect-everything
    // behavior would clearly exceed the bound.
    let in_flight_bound =
        get_num_compute_intensive_cpus().max(1) + 2 * super::GLOBAL_TOPK_CHUNK_MAX_PARTITIONS;
    let num_partitions = 2 * in_flight_bound;

    let test_dir = TempStrDir::default();
    let (batch, schema) = make_seeded_vector_batch(num_partitions * ROWS_PER_PARTITION);
    let query_vector = batch["vector"].as_fixed_size_list().value(0);
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
        .await
        .unwrap();
    let mut ivf_params = IvfBuildParams::new(num_partitions);
    ivf_params.max_iters = 2;
    ivf_params.sample_rate = 16;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        ivf_params,
        lightweight_pq_params(),
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    let index = dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let tracker = index
        .as_any()
        .downcast_ref::<super::IvfPq>()
        .expect("IVF_PQ index")
        .prepared_partitions();

    let query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: K,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: num_partitions,
        maximum_nprobes: Some(num_partitions),
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: Some(DistanceType::L2),
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
    };
    let (partitions, q_c_dists) = index.find_partitions(&query).unwrap();
    assert_eq!(partitions.len(), num_partitions);
    let results = index
        .clone()
        .search_partitions(
            query,
            Arc::new(partitions),
            Arc::new(q_c_dists),
            0,
            num_partitions,
            Arc::new(NoFilter),
            None,
            Arc::new(NoOpMetricsCollector),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let num_results: usize = results.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(num_results, K);
    assert_eq!(
        tracker.in_flight(),
        0,
        "every prepared partition must be released once the search completes"
    );
    assert!(
        tracker.peak() >= 1,
        "the search must have gone through prepared partitions"
    );
    assert!(
        tracker.peak() <= in_flight_bound,
        "peak in-flight prepared partitions {} exceeds the bound {} (probed {} partitions)",
        tracker.peak(),
        in_flight_bound,
        num_partitions
    );
}

#[rstest]
#[tokio::test]
async fn test_index_stats(
    #[values(
            (VectorIndexParams::ivf_flat(2, DistanceType::Hamming), IndexType::IvfFlat),
            (VectorIndexParams::ivf_pq(2, 8, 8, DistanceType::L2, 10), IndexType::IvfPq),
            (VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::Cosine,
                IvfBuildParams::new(2),
                Default::default(),
                Default::default()
            ), IndexType::IvfHnswSq),
        )]
    index: (VectorIndexParams, IndexType),
) {
    let (params, index_type) = index;
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let nlist = 2;
    let (mut dataset, _) = match params.metric_type {
        DistanceType::Hamming => generate_test_dataset::<UInt8Type>(test_uri, 0..2).await,
        _ => generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await,
    };
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_index".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let stats = dataset.index_statistics("test_index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

    assert_eq!(
        stats["index_type"].as_str().unwrap(),
        index_type.to_string()
    );
    for index in stats["indices"].as_array().unwrap() {
        assert_eq!(
            index["index_type"].as_str().unwrap(),
            index_type.to_string()
        );
        assert_eq!(
            index["num_partitions"].as_number().unwrap(),
            &serde_json::Number::from(nlist)
        );

        let sub_index = match index_type {
            IndexType::IvfHnswPq | IndexType::IvfHnswSq => "HNSW",
            IndexType::IvfPq => "PQ",
            _ => "FLAT",
        };
        assert_eq!(
            index["sub_index"]["index_type"].as_str().unwrap(),
            sub_index
        );
    }
}

#[tokio::test]
async fn test_index_stats_empty_partition() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let num_rows = 32;
    let num_partitions = num_rows + 2;
    let mut vector_values = vec![0.0; num_rows * DIM];
    for row in 0..num_rows {
        vector_values[row * DIM + row] = 1.0;
    }
    let one_hot_vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vector_values.clone()),
            DIM as i32,
        )
        .unwrap(),
    );
    let batch = gen_batch()
        .col("id", array::step::<UInt64Type>())
        .col("vector", array::jitter_centroids(one_hot_vectors, 0.0))
        .into_batch_rows(RowCount::from(num_rows as u64))
        .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

    // Keep partition 0 empty: stats previously failed when the first partition was empty.
    let mut centroid_values = Vec::with_capacity(num_partitions * DIM);
    centroid_values.extend(std::iter::repeat_n(2.0, DIM));
    centroid_values.extend(vector_values);
    centroid_values.extend(std::iter::repeat_n(-2.0, DIM));
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroid_values), DIM as i32)
            .unwrap(),
    );
    let ivf_params = IvfBuildParams::try_with_centroids(num_partitions, centroids).unwrap();
    let sq_params = SQBuildParams::default();
    let hnsw_params = HnswBuildParams::default()
        .max_level(1)
        .num_edges(4)
        .ef_construction(4);
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        ivf_params,
        hnsw_params,
        sq_params,
    );

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_index".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let stats = dataset.index_statistics("test_index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

    assert_eq!(stats["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
    let indices = stats["indices"].as_array().unwrap();
    assert_eq!(indices.len(), 1);
    let index = &indices[0];
    assert_eq!(index["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
    assert_eq!(
        index["num_partitions"].as_number().unwrap(),
        &serde_json::Number::from(num_partitions)
    );
    assert_eq!(index["sub_index"]["index_type"].as_str().unwrap(), "HNSW");
    let partition_sizes = index["partitions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|partition| partition["size"].as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(partition_sizes.len(), num_partitions);
    assert_eq!(partition_sizes.iter().sum::<u64>(), num_rows as u64);
    assert_eq!(partition_sizes[0], 0);
    assert!(partition_sizes.contains(&0));
}

async fn test_distance_range(params: Option<VectorIndexParams>, nlist: usize) {
    match params.as_ref().map_or(DistanceType::L2, |p| p.metric_type) {
        DistanceType::Hamming => {
            test_distance_range_impl::<UInt8Type>(params, nlist, 0..255).await;
        }
        _ => {
            test_distance_range_impl::<Float32Type>(params, nlist, 0.0..1.0).await;
        }
    }
}

async fn test_distance_range_impl<T: ArrowPrimitiveType>(
    params: Option<VectorIndexParams>,
    nlist: usize,
    range: Range<T::Native>,
) where
    T::Native: SampleUniform,
{
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range).await;

    let vector_column = "vector";
    let dist_type = params.as_ref().map_or(DistanceType::L2, |p| p.metric_type);
    if let Some(params) = params {
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
    }

    let query = vectors.value(0);
    let k = 10;
    let result = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), k)
        .unwrap()
        .minimum_nprobes(nlist)
        .ef(100)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), k);
    let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
    let dists = result[DIST_COL].as_primitive::<Float32Type>().values();

    let part_idx = k / 2;
    let part_dist = dists[part_idx];

    let left_res = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), part_idx)
        .unwrap()
        .minimum_nprobes(nlist)
        .ef(100)
        .with_row_id()
        .distance_range(None, Some(part_dist))
        .try_into_batch()
        .await
        .unwrap();
    let right_res = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), k - part_idx)
        .unwrap()
        .minimum_nprobes(nlist)
        .ef(100)
        .with_row_id()
        .distance_range(Some(part_dist), None)
        .try_into_batch()
        .await
        .unwrap();
    // don't verify the number of results and row ids for hamming distance,
    // because there are many vectors with the same distance
    if dist_type != DistanceType::Hamming {
        // Tolerate a single tied pair at the partition boundary. When
        // dists[part_idx - 1] == part_dist, the strict-less left filter
        // excludes both tied vectors and the inclusive right filter
        // includes both, shifting one row from left to right and dropping
        // row_ids[k - 1] off right_res's limit. Observed for Dot distance
        // on ARM where SIMD FMA yields tied float32 dot products that x86
        // does not. The distance-value assertions below still cover
        // partition correctness in both cases.
        let boundary_tie = part_idx > 0 && dists[part_idx - 1] == part_dist;
        let left_row_ids = left_res[ROW_ID].as_primitive::<UInt64Type>().values();
        let right_row_ids = right_res[ROW_ID].as_primitive::<UInt64Type>().values();
        if boundary_tie {
            assert_eq!(left_res.num_rows(), part_idx - 1);
            for i in 0..(part_idx - 1) {
                assert_eq!(left_row_ids[i], row_ids[i]);
            }
            assert_eq!(right_res.num_rows(), k - part_idx);
            // right_row_ids[0..2] are the two tied vectors in tiebreaker
            // order; their identity is not pinned. right_row_ids[i] for
            // i >= 2 aligns with row_ids[part_idx + i - 1] because the
            // tie shifts one vector from left to right.
            for i in 2..(k - part_idx) {
                assert_eq!(right_row_ids[i], row_ids[part_idx + i - 1]);
            }
        } else {
            assert_eq!(left_res.num_rows(), part_idx);
            assert_eq!(right_res.num_rows(), k - part_idx);
            row_ids.iter().enumerate().for_each(|(i, id)| {
                if i < part_idx {
                    assert_eq!(left_row_ids[i], *id,);
                } else {
                    assert_eq!(right_row_ids[i - part_idx], *id,);
                }
            });
        }
    }
    let left_dists = left_res[DIST_COL].as_primitive::<Float32Type>().values();
    let right_dists = right_res[DIST_COL].as_primitive::<Float32Type>().values();
    left_dists.iter().for_each(|d| {
        assert!(d < &part_dist);
    });
    right_dists.iter().for_each(|d| {
        assert!(d >= &part_dist);
    });

    let exclude_last_res = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), k)
        .unwrap()
        .minimum_nprobes(nlist)
        .ef(100)
        .with_row_id()
        .distance_range(dists.first().copied(), dists.last().copied())
        .try_into_batch()
        .await
        .unwrap();
    if dist_type != DistanceType::Hamming {
        let excluded_count = dists.iter().filter(|d| *d == dists.last().unwrap()).count();
        assert_eq!(exclude_last_res.num_rows(), k - excluded_count);
        let res_row_ids = exclude_last_res[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values();
        row_ids.iter().enumerate().for_each(|(i, id)| {
            if i < k - excluded_count {
                assert_eq!(res_row_ids[i], *id);
            }
        });
    }
    let res_dists = exclude_last_res[DIST_COL]
        .as_primitive::<Float32Type>()
        .values();
    res_dists.iter().for_each(|d| {
        assert_ge!(*d, dists[0]);
        assert_lt!(*d, dists[k - 1]);
    });
}

#[tokio::test]
async fn test_index_with_zero_vectors() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (batch, schema) = generate_batch::<Float32Type>(256, None, 0.0..1.0, false);
    let vector_field = schema.field(1).clone();
    let zero_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![256])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![0.0; DIM]),
                    DIM as i32,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![batch, zero_batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            mode: crate::dataset::WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let vector_column = vector_field.name();
    let params = VectorIndexParams::ivf_pq(4, 8, DIM / 8, DistanceType::Cosine, 50);
    dataset
        .create_index(&[vector_column], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
}

async fn test_recall<T: ArrowPrimitiveType>(
    params: VectorIndexParams,
    nlist: usize,
    recall_requirement: f32,
    vector_column: &str,
    dataset: &Dataset,
    vectors: Arc<FixedSizeListArray>,
) {
    let query = vectors.value(0);
    let k = 100;
    let result = dataset
        .scan()
        .nearest(vector_column, query.as_primitive::<T>(), k)
        .unwrap()
        .minimum_nprobes(nlist)
        .maximum_nprobes(nlist)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let row_ids = result[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let dists = result[DIST_COL]
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let results = dists.into_iter().zip(row_ids).collect::<Vec<_>>();
    let row_ids = results.iter().map(|(_, id)| *id).collect::<HashSet<_>>();
    assert!(row_ids.len() == k);

    let gt = ground_truth(dataset, vector_column, &query, k, params.metric_type).await;

    let recall = row_ids.intersection(&gt).count() as f32 / k as f32;
    assert!(
        recall >= recall_requirement,
        "recall: {}\n results: {:?}\n\ngt: {:?}",
        recall,
        results,
        gt,
    );
}

/// Rewrite the auxiliary storage file to the legacy PQ format (codebook
/// embedded in schema metadata rather than stored as a global buffer), then
/// commit a `CreateIndex` transaction so the manifest records the correct
/// new file size.
/// Rewrite the auxiliary PQ storage file with the codebook inlined into
/// schema metadata (legacy format). Uses a new UUID to avoid cache key
/// collisions with the original index.
async fn rewrite_pq_storage(dataset: &mut Dataset, old_meta: &IndexMetadata) -> Result<()> {
    use crate::dataset::transaction::{Operation, Transaction};

    let obj_store = Arc::new(ObjectStore::local());
    let old_dir = dataset.indices_dir().join(old_meta.uuid.to_string());
    let new_uuid = uuid::Uuid::new_v4();
    let new_dir = dataset.indices_dir().join(new_uuid.to_string());

    // Copy the main index file to the new directory unchanged.
    obj_store
        .copy(
            &old_dir.clone().join(super::INDEX_FILE_NAME),
            &new_dir.clone().join(super::INDEX_FILE_NAME),
        )
        .await?;

    // Read the original auxiliary file.
    let old_aux_path = old_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
    let scheduler = ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());
    let reader = FileReader::try_open(
        scheduler
            .open_file(&old_aux_path, &CachedFileSize::unknown())
            .await?,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await?;

    // Rewrite auxiliary file with PQ codebook inlined into schema metadata.
    let mut metadata = reader.schema().metadata.clone();
    let projection = lance_file::versions::reader_projection_from_whole_schema(
        reader.schema(),
        reader.metadata().version(),
    );
    let batches = reader
        .read_stream_projected(
            lance_io::ReadBatchParams::RangeFull,
            u32::MAX,
            u32::MAX,
            projection,
            lance_encoding::decoder::FilterExpression::no_filter(),
        )
        .await?;
    use futures::TryStreamExt as _;
    let batches = batches.try_collect::<Vec<_>>().await?;
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
    let new_aux_path = new_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
    let mut writer = lance_file::versions::create_writer(
        reader.metadata().version(),
        obj_store.create(&new_aux_path).await?,
        batch.schema_ref().as_ref().try_into()?,
        Default::default(),
    )?;
    writer.write_batch(&batch).await?;
    writer
        .add_global_buffer(reader.read_global_buffer(1).await?)
        .await?;
    let codebook = reader.read_global_buffer(2).await?;
    let pq_metadata: Vec<String> = serde_json::from_str(&metadata[STORAGE_METADATA_KEY])?;
    let mut pq_metadata: ProductQuantizationMetadata = serde_json::from_str(&pq_metadata[0])?;
    pq_metadata.codebook_position = 0;
    pq_metadata.codebook_tensor = codebook.to_vec();
    let pq_metadata = serde_json::to_string(&pq_metadata)?;
    metadata.insert(
        STORAGE_METADATA_KEY.to_owned(),
        serde_json::to_string(&vec![pq_metadata])?,
    );
    for (key, value) in metadata {
        writer.add_schema_metadata(key, value);
    }
    writer.finish().await?;

    // Build new IndexMetadata with the new UUID and file sizes.
    let new_files = lance_table::format::list_index_files_with_sizes(&obj_store, &new_dir).await?;
    let mut new_meta = old_meta.clone();
    new_meta.uuid = new_uuid;
    new_meta.files = Some(new_files);

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_meta],
            removed_indices: vec![old_meta.clone()],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

#[tokio::test]
async fn test_legacy_non_divisible_pq_search() {
    const DIM: usize = 64;
    const PERSISTED_DIM: usize = 56;

    let test_dir = copy_test_data_to_tmp("v0.10.15/non_divisible_pq").unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let query = Float32Array::from(
        (1..=DIM)
            .map(|value| value as f32 + if value <= PERSISTED_DIM { 1.0 } else { 1_000.0 })
            .collect::<Vec<_>>(),
    );

    let result = dataset
        .scan()
        .nearest("vector", &query, 1)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(result.num_rows(), 1);
    assert_eq!(
        result[DIST_COL].as_primitive::<Float32Type>().values(),
        &[PERSISTED_DIM as f32]
    );
}

#[tokio::test]
async fn test_pq_storage_backwards_compat() {
    let test_dir = copy_test_data_to_tmp("v0.27.1/pq_in_schema").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    // Just make sure we can query the index.
    let dataset = Dataset::open(test_uri).await.unwrap();
    let query_vec = Float32Array::from(vec![0_f32; 32]);
    let search_result = dataset
        .scan()
        .nearest("vec", &query_vec, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(search_result.num_rows(), 5);

    let obj_store = Arc::new(ObjectStore::local());
    let scheduler = ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());

    async fn get_pq_metadata(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
    ) -> ProductQuantizationMetadata {
        let index = dataset.load_indices().await.unwrap();
        let index_path = dataset.indices_dir().join(index[0].uuid.to_string());
        let file_scheduler = scheduler
            .open_file(
                &index_path.clone().join(INDEX_AUXILIARY_FILE_NAME),
                &CachedFileSize::unknown(),
            )
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
        serde_json::from_str(&serde_json::from_str::<Vec<String>>(metadata).unwrap()[0]).unwrap()
    }
    let pq_meta: ProductQuantizationMetadata = get_pq_metadata(&dataset, scheduler.clone()).await;
    assert!(pq_meta.buffer_index().is_none());

    // If we add data and optimize indices, then we start using the global
    // buffer for the PQ index.
    let new_data = RecordBatch::try_new(
        Arc::new(Schema::from(dataset.schema())),
        vec![
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 32]), 32)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![new_data])
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();

    let pq_meta: ProductQuantizationMetadata = get_pq_metadata(&dataset, scheduler.clone()).await;
    assert!(pq_meta.buffer_index().is_some());
}

#[tokio::test]
async fn test_optimize_with_empty_partition() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

    let num_rows = dataset.count_all_rows().await.unwrap();
    let nlist = num_rows + 2;
    let centroids = generate_random_array(nlist * DIM);
    let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
    let ivf_params = IvfBuildParams::try_with_centroids(nlist, Arc::new(ivf_centroids)).unwrap();
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::Cosine,
        ivf_params,
        PQBuildParams::default(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    append_dataset::<Float32Type>(&mut dataset, 1, 0.0..1.0).await;
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();
}

#[tokio::test]
async fn test_compaction_remaps_second_delta_with_shared_partition_topology() {
    const INDEX_NAME: &str = "vector_idx";
    const BASE_ROWS_PER_PARTITION: usize = 2_200;
    const SMALL_APPEND_ROWS: usize = 64;
    let offsets = [-50.0, 50.0];

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let centroids = build_centroids_for_offsets(&offsets);
    let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        ivf_params,
        lightweight_pq_params(),
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

    let template_batch = dataset
        .take_rows(&[0], dataset.schema().clone())
        .await
        .unwrap();
    let template_values = template_batch["vector"]
        .as_fixed_size_list()
        .value(0)
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let mut append_params = WriteParams {
        max_rows_per_file: 32,
        max_rows_per_group: 32,
        ..Default::default()
    };
    append_params.mode = WriteMode::Append;
    append_template_vector_with_params(
        &mut dataset,
        SMALL_APPEND_ROWS,
        &template_values,
        Some(append_params),
    )
    .await;

    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let stats_before: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
    assert_eq!(stats_before["num_indices"].as_u64().unwrap(), 2);
    let partitions_before: Vec<usize> = stats_before["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
        .collect();
    assert_eq!(partitions_before.len(), 2);
    let base_partition_count = partitions_before
        .iter()
        .copied()
        .max()
        .expect("expected at least one partition count");
    assert!(base_partition_count >= 2);
    assert!(
        partitions_before
            .iter()
            .all(|count| *count == base_partition_count)
    );

    let indices_meta = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(indices_meta.len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 5_000,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    let stats_after_compaction: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
    assert_eq!(stats_after_compaction["num_indices"].as_u64().unwrap(), 2);
    let mut partitions_after: Vec<usize> = stats_after_compaction["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
        .collect();
    partitions_after.sort_unstable();
    assert_eq!(
        partitions_after,
        vec![base_partition_count, base_partition_count]
    );
}

#[tokio::test]
async fn test_spfresh_join_split() {
    const INDEX_NAME: &str = "vector_idx";
    const NLIST: usize = 2;
    const NO_SPLIT_APPEND_ROWS: usize = 32;
    // The joined base and no-split delta contain 2,265 rows. This append
    // takes the single IVF-PQ partition one row past its 32,768-row limit.
    const SPLIT_APPEND_ROWS: usize = 30_504;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let cluster_sizes = [100, 2_200];
    let total_rows: usize = cluster_sizes.iter().sum();

    let mut centroid_values = Vec::new();
    for i in 0..NLIST {
        for j in 0..DIM {
            centroid_values.push(if j == 0 { (i as f32) * 10.0 } else { 0.0 });
        }
    }
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroid_values), DIM as i32)
            .unwrap(),
    );

    let mut ids = Vec::new();
    let mut vector_values = Vec::new();
    let mut current_id = 0u64;
    for (cluster_idx, &size) in cluster_sizes.iter().enumerate() {
        let centroid_base = (cluster_idx as f32) * 10.0;
        for _ in 0..size {
            ids.push(current_id);
            current_id += 1;
            for j in 0..DIM {
                vector_values.push(if j == 0 {
                    centroid_base + (current_id % 100) as f32 * 0.005
                } else {
                    (current_id % 50) as f32 * 0.01
                });
            }
        }
    }

    let ids_array = Arc::new(UInt64Array::from(ids.clone()));
    let vectors = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(vector_values), DIM as i32)
            .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, vectors]).unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let mut dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            mode: crate::dataset::WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Pin the type default target so the limits above are the ones this test
    // reasons about; a build from a partition count would record its own.
    let ivf_params = IvfBuildParams {
        target_partition_size: Some(IndexType::IvfPq.target_partition_size()),
        ..IvfBuildParams::try_with_centroids(NLIST, centroids).unwrap()
    };
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        ivf_params,
        lightweight_pq_params(),
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

    let template_id = cluster_sizes[0] as u64;
    let template_batch = dataset
        .take_rows(&[template_id], dataset.schema().clone())
        .await
        .unwrap();
    let template_values = template_batch["vector"]
        .as_fixed_size_list()
        .value(0)
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    assert_eq!(
        template_values.len(),
        DIM,
        "Template vector should match DIM"
    );

    let mut next_id = total_rows as u64;
    let mut expected_rows = total_rows;

    let (deleted_rows, appended_rows, actual_partitions) =
        shrink_smallest_partition(&mut dataset, INDEX_NAME, 1, &mut next_id).await;
    expected_rows = expected_rows - deleted_rows + appended_rows;
    assert_eq!(actual_partitions, 1);
    assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

    append_and_verify_append_phase(
        &mut dataset,
        INDEX_NAME,
        &template_values,
        &mut next_id,
        NO_SPLIT_APPEND_ROWS,
        1,
        expected_rows + NO_SPLIT_APPEND_ROWS,
        2,
        false,
    )
    .await;
    expected_rows += NO_SPLIT_APPEND_ROWS;

    // The oversized partition is split straight to the target size in one
    // optimize: ceil(rows / target) pieces instead of a single halving.
    let split_rows = expected_rows + SPLIT_APPEND_ROWS;
    append_and_verify_append_phase(
        &mut dataset,
        INDEX_NAME,
        &template_values,
        &mut next_id,
        SPLIT_APPEND_ROWS,
        split_rows.div_ceil(IndexType::IvfPq.target_partition_size()),
        split_rows,
        1,
        true,
    )
    .await;
}

/// A multivector index built from a partition count records no target
/// partition size: a retrain divides the table's rows by the target, so a
/// target in vectors would train a fraction of the partitions.
#[tokio::test]
async fn test_multivec_build_from_partition_count_records_no_target() {
    let test_dir = TempStrDir::default();
    let (mut dataset, _) =
        generate_multivec_test_dataset::<Float32Type>(test_dir.as_str(), 0.0..1.0).await;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::Cosine,
        IvfBuildParams::new(1),
        lightweight_pq_params(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let details = indices[0]
        .index_details
        .as_ref()
        .expect("a vector index records its details");
    assert_eq!(target_partition_size_from_details(details), None);
}

#[tokio::test]
async fn test_partition_split_on_append_multivec() {
    const INDEX_NAME: &str = "vector_idx";
    const VECTORS_PER_ROW: usize = 3;
    // 512 base rows and this append flatten to 33,036 vectors, just over
    // the 32,768-vector IVF-PQ split threshold.
    const APPEND_ROWS: usize = 10_500;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let (mut dataset, _) = generate_multivec_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::Cosine,
        IvfBuildParams::new(1),
        lightweight_pq_params(),
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

    let initial_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(initial_ctx.num_partitions(), 1);

    append_dataset::<Float32Type>(&mut dataset, APPEND_ROWS, 0.0..0.05).await;
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let expected_rows = NUM_ROWS + APPEND_ROWS;
    // Every vector of a row counts towards the partition size, and the split
    // goes straight to the target size: ceil(vectors / target) pieces.
    let expected_partitions =
        (expected_rows * VECTORS_PER_ROW).div_ceil(IndexType::IvfPq.target_partition_size());
    let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(
        final_ctx.num_partitions(),
        expected_partitions,
        "Expected the oversized multivector partition to split into {expected_partitions}, stats: {}",
        final_ctx.stats_json()
    );
    let partitions = final_ctx.stats()["indices"][0]["partitions"]
        .as_array()
        .expect("partitions should be present");
    assert_eq!(partitions.len(), expected_partitions);
    assert_eq!(
        partitions
            .iter()
            .map(|partition| partition["size"].as_u64().unwrap() as usize)
            .sum::<usize>(),
        expected_rows * VECTORS_PER_ROW
    );
    assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

    let query_batch = dataset
        .scan()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = query_batch["vector"].as_list::<i32>().value(0);
    let results = dataset
        .scan()
        .with_row_id()
        .nearest("vector", &query, 10)
        .unwrap()
        .distance_metric(DistanceType::Cosine)
        .try_into_batch()
        .await
        .unwrap();
    let mut row_ids = HashSet::new();
    for row_id in results[ROW_ID].as_primitive::<UInt64Type>().values() {
        assert!(row_ids.insert(*row_id), "duplicate row id {row_id}");
    }
}

#[tokio::test]
async fn test_split_multiple_partitions_in_one_optimize() {
    const INDEX_NAME: &str = "vector_idx";
    const BASE_ROWS_PER_PARTITION: usize = 512;
    // Each IVF-FLAT partition reaches 16,512 rows, just over its 16,384-row
    // split threshold.
    const APPEND_ROWS_PER_PARTITION: usize = 16_000;
    let offsets = [-50.0, 50.0];

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let centroids = build_centroids_for_offsets(&offsets);
    // Pin the type default target so the threshold above is the one this test
    // reasons about; a build from a partition count would record its own.
    let ivf_params = IvfBuildParams {
        target_partition_size: Some(IndexType::IvfFlat.target_partition_size()),
        ..IvfBuildParams::try_with_centroids(2, centroids).unwrap()
    };
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params);
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

    let initial_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(initial_ctx.num_partitions(), 2);
    let templates = offsets
        .iter()
        .map(|offset| {
            let mut template = vec![0.0; DIM];
            template[0] = *offset;
            template
        })
        .collect::<Vec<_>>();

    append_partition_templates(&mut dataset, APPEND_ROWS_PER_PARTITION, &templates).await;

    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    // Both partitions hold BASE + APPEND rows and are split straight to the
    // target size in one optimize: ceil(rows / target) pieces each.
    let pieces_per_partition = (BASE_ROWS_PER_PARTITION + APPEND_ROWS_PER_PARTITION)
        .div_ceil(IndexType::IvfFlat.target_partition_size());
    let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(
        final_ctx.num_partitions(),
        2 * pieces_per_partition,
        "Expected both original partitions to split in one optimize, stats: {}",
        final_ctx.stats_json()
    );

    let indices = final_ctx.stats()["indices"]
        .as_array()
        .expect("indices should be present");
    assert_eq!(
        indices.len(),
        1,
        "Expected split optimize to merge into one index, stats: {}",
        final_ctx.stats_json()
    );

    let partitions = indices[0]["partitions"]
        .as_array()
        .expect("partitions should be present");
    assert_eq!(partitions.len(), 2 * pieces_per_partition);
    let expected_rows = 2 * BASE_ROWS_PER_PARTITION + 2 * APPEND_ROWS_PER_PARTITION;
    let total_partition_rows = partitions
        .iter()
        .map(|part| part["size"].as_u64().unwrap() as usize)
        .sum::<usize>();
    assert_eq!(total_partition_rows, expected_rows);
    assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

    let mut indexed_row_ids = HashSet::with_capacity(expected_rows);
    for partition_idx in 0..final_ctx.num_partitions() {
        for row_id in load_flat_partition_row_ids(final_ctx.ivf_flat(), partition_idx).await {
            assert!(
                indexed_row_ids.insert(row_id),
                "row id {row_id} appeared in multiple partitions"
            );
        }
    }
    assert_eq!(indexed_row_ids.len(), expected_rows);
    let live_row_ids = dataset.scan().with_row_id().try_into_batch().await.unwrap()[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(indexed_row_ids, live_row_ids);

    let nearest = dataset
        .scan()
        .with_row_id()
        .nearest("vector", &Float32Array::from(templates[0].clone()), 10)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = nearest[ROW_ID].as_primitive::<UInt64Type>();
    let mut seen = HashSet::new();
    for row_id in ids.values() {
        assert!(seen.insert(*row_id), "Duplicate row id found: {}", row_id);
    }
}

#[tokio::test]
async fn test_join_partition_on_delete_multivec() {
    const INDEX_NAME: &str = "vector_idx";
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    const MULTIVEC_PER_ROW: usize = 3;
    const APPEND_ROWS: usize = 32;
    let cluster_sizes = [800, 800, 400];
    // Multivector indices require cosine distance. Unit centroids in three
    // distinct directions avoid the collinear assignment in the old fixture.
    let centroids = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)];
    let total_rows = cluster_sizes.iter().sum::<usize>();
    // Row 1600, the one retained below, has one vector in each partition.
    // Joining the partition that holds one of them must reassign only that
    // vector, not re-add the two the other partitions keep.
    let mut dataset = {
        let (batch, schema) = generate_clustered_multivec_batch(
            &cluster_sizes,
            &centroids,
            MULTIVEC_PER_ROW,
            0,
            Some(1600),
        );
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    };

    let ivf_params =
        IvfBuildParams::try_with_centroids(centroids.len(), build_centroids_2d(&centroids))
            .unwrap();
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::Cosine,
        ivf_params,
        lightweight_pq_params(),
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

    let index_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(index_ctx.num_partitions(), 3);

    let mut logical_row_ids = {
        let ivf = index_ctx.ivf();
        let mut smallest: Option<HashSet<u64>> = None;
        for i in 0..ivf.ivf.num_partitions() {
            let partition_row_ids = load_partition_row_ids(ivf, i)
                .await
                .into_iter()
                .collect::<HashSet<_>>();
            if partition_row_ids.is_empty() {
                continue;
            }

            let is_better = smallest
                .as_ref()
                .map(|existing| partition_row_ids.len() < existing.len())
                .unwrap_or(true);
            if is_better {
                smallest = Some(partition_row_ids);
            }
        }
        smallest
            .expect("expected a non-empty partition")
            .into_iter()
            .collect::<Vec<_>>()
    };
    logical_row_ids.sort_unstable();
    assert_eq!(logical_row_ids.len(), cluster_sizes[2]);
    let retained_id = logical_row_ids[0];
    delete_ids(&mut dataset, &logical_row_ids[1..]).await;
    compact_after_deletions(&mut dataset).await;

    let (append_batch, append_schema) = generate_clustered_multivec_batch(
        &[APPEND_ROWS],
        &centroids[2..],
        MULTIVEC_PER_ROW,
        total_rows as u64,
        None,
    );
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(append_batch)], append_schema),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    assert_eq!(
        final_ctx.num_partitions(),
        2,
        "Expected the reduced multivector partition to join, stats: {}",
        final_ctx.stats_json()
    );
    assert_eq!(final_ctx.stats()["num_indices"].as_u64().unwrap(), 1);
    let expected_rows = total_rows - cluster_sizes[2] + 1 + APPEND_ROWS;
    assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

    let sample_row = dataset
        .scan()
        .with_row_id()
        .filter(&format!("id = {retained_id}"))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(sample_row.num_rows(), 1);
    let retained_row_id = sample_row[ROW_ID].as_primitive::<UInt64Type>().value(0);
    let mut indexed_row_id_counts = HashMap::new();
    for partition_idx in 0..final_ctx.num_partitions() {
        for row_id in load_partition_row_ids(final_ctx.ivf(), partition_idx).await {
            *indexed_row_id_counts.entry(row_id).or_insert(0usize) += 1;
        }
    }
    assert_eq!(
        indexed_row_id_counts.values().sum::<usize>(),
        expected_rows * MULTIVEC_PER_ROW
    );
    assert_eq!(
        indexed_row_id_counts.get(&retained_row_id),
        Some(&MULTIVEC_PER_ROW),
        "all vectors for the retained logical row should survive the join"
    );
    assert!(
        indexed_row_id_counts
            .values()
            .all(|count| *count == MULTIVEC_PER_ROW),
        "each logical row should have exactly {MULTIVEC_PER_ROW} indexed vectors"
    );
    let live_row_ids = dataset.scan().with_row_id().try_into_batch().await.unwrap()[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(live_row_ids.len(), expected_rows);
    assert_eq!(
        indexed_row_id_counts
            .keys()
            .copied()
            .collect::<HashSet<_>>(),
        live_row_ids
    );
}

async fn row_ids_matching(dataset: &Dataset, predicate: &str) -> HashSet<u64> {
    let mut scan = dataset.scan();
    scan.with_row_id();
    scan.filter(predicate).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect()
}

struct OptimizeAfterDelete {
    deleted_row_ids: HashSet<u64>,
    index_row_ids: HashSet<u64>,
    num_partitions_after: usize,
    stats_json: String,
}

/// Shared scenario for the issue-7701 regressions: stable-row-id dataset,
/// IVF_FLAT index, scattered delete, optimize. Asserts the invariants both
/// partition adjustments must hold -- no live row lost, no id that never
/// existed -- and returns the state for the mode-specific assertions.
async fn optimize_after_delete(
    total_rows: usize,
    nlist: usize,
    delete_predicate: &str,
    keep_predicate: &str,
) -> OptimizeAfterDelete {
    const INDEX_NAME: &str = "vector_idx";
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let (batch, schema) = generate_batch::<Float32Type>(total_rows, None, 0.0..1.0, false);
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        batches,
        test_uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Pin the type default target: both callers size their partitions against
    // its split and join thresholds, and a build from a partition count would
    // record its own.
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams {
            target_partition_size: Some(IndexType::IvfFlat.target_partition_size()),
            ..IvfBuildParams::new(nlist)
        },
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

    let deleted_row_ids = row_ids_matching(&dataset, delete_predicate).await;
    let live_row_ids = row_ids_matching(&dataset, keep_predicate).await;
    dataset.delete(delete_predicate).await.unwrap();

    dataset
        .optimize_indices(&OptimizeOptions::new())
        .await
        .unwrap();

    let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
    let num_partitions_after = final_ctx.num_partitions();
    let stats_json = final_ctx.stats_json().to_string();
    let flat = final_ctx
        .index
        .as_any()
        .downcast_ref::<IvfFlatIndex>()
        .expect("expected IvfFlat index");
    let mut index_row_ids = HashSet::new();
    for part in 0..flat.ivf.num_partitions() {
        index_row_ids.extend(load_flat_partition_row_ids(flat, part).await);
    }

    for row_id in &live_row_ids {
        assert!(
            index_row_ids.contains(row_id),
            "live row id {} missing from index after optimize",
            row_id
        );
    }
    for row_id in &index_row_ids {
        assert!(
            live_row_ids.contains(row_id) || deleted_row_ids.contains(row_id),
            "unexpected row id {} in index after optimize",
            row_id
        );
    }

    OptimizeAfterDelete {
        deleted_row_ids,
        index_row_ids,
        num_partitions_after,
        stats_json,
    }
}

#[tokio::test]
async fn test_optimize_join_after_delete_with_stable_row_ids() {
    // Regression test for https://github.com/lance-format/lance/issues/7701:
    // every partition (400 rows / 4) is under the IVF_FLAT join threshold,
    // so one optimize joins all of them but the largest after a scattered delete.
    let run = optimize_after_delete(400, 4, "id % 3 = 0", "id % 3 != 0").await;

    assert_eq!(
        run.num_partitions_after, 1,
        "optimize should have joined every undersized partition but one, got stats: {}",
        run.stats_json
    );

    // The join reads every partition's stored rows through the merge filter, so
    // deleted ids are dropped index-wide and not just from the joined partition.
    for row_id in &run.deleted_row_ids {
        assert!(
            !run.index_row_ids.contains(row_id),
            "deleted row id {} still in index after join",
            row_id
        );
    }
}

#[tokio::test]
async fn test_optimize_split_after_delete_with_stable_row_ids() {
    // Regression test for https://github.com/lance-format/lance/issues/7701:
    // one partition holds more than 4x the IVF_FLAT target, so optimize
    // splits it after a scattered delete. This path reaches
    // filter_deleted_ids through reshuffle_partitions, unlike the join
    // path's take_vectors.
    let run = optimize_after_delete(20_000, 1, "id % 5 = 0", "id % 5 != 0").await;

    assert!(
        run.num_partitions_after > 1,
        "optimize should have split the oversized partition, got stats: {}",
        run.stats_json
    );

    // The split rebuilds the whole partition from live rows: no deleted
    // ids remain.
    for row_id in &run.deleted_row_ids {
        assert!(
            !run.index_row_ids.contains(row_id),
            "deleted row id {} still in index after split",
            row_id
        );
    }
}

#[rstest]
#[case::ivf_pq(VectorIndexParams::with_ivf_pq_params(
    DistanceType::L2,
    IvfBuildParams::new(4),
    PQBuildParams::new(4, 4),
))]
#[case::ivf_rq(VectorIndexParams::with_ivf_rq_params(
    DistanceType::L2,
    IvfBuildParams::new(4),
    RQBuildParams::with_rotation_type(5, RQRotationType::Fast),
))]
#[tokio::test]
async fn test_prewarm_vector_index(#[case] params: VectorIndexParams) {
    use lance_io::assert_io_eq;

    const INDEX_NAME: &str = "my_idx";
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(INDEX_NAME.to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    append_dataset::<Float32Type>(&mut dataset, 8, 0.0..1.0).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    // Reopen to avoid carrying index state in memory from index creation.
    let dataset = Dataset::open(test_uri).await.unwrap();
    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(indices.len(), 2, "expected two index deltas");
    let unique_uuids: HashSet<_> = indices.iter().map(|meta| meta.uuid).collect();
    assert_eq!(unique_uuids.len(), 2, "expected two unique index UUIDs");

    // Reset IO stats after index creation.
    dataset.object_store.as_ref().io_stats_incremental();

    // Concurrent prewarms should single-flight each partition through the
    // cache loader and leave a complete warm cache for both callers.
    let (first, second) = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        tokio::join!(
            dataset.prewarm_index(INDEX_NAME),
            dataset.prewarm_index(INDEX_NAME)
        )
    })
    .await
    .expect("concurrent prewarms deadlocked");
    first.unwrap();
    second.unwrap();
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert!(
        stats.read_iops > 0,
        "prewarm should have read from disk, but read_iops was 0"
    );

    // Query should not perform IO after prewarming all deltas.
    let q = vectors.value(0);
    dataset
        .scan()
        .nearest("vector", q.as_primitive::<Float32Type>(), 10)
        .unwrap()
        .project(&["_rowid"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(
        stats,
        read_iops,
        0,
        "query should not perform IO after prewarm"
    );

    // Second prewarm should not need IO (already cached).
    dataset.prewarm_index(INDEX_NAME).await.unwrap();
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
}

/// Index-cache backend that can drop partition entries on demand.
///
/// Used to simulate cache invalidation after a credential rotation:
/// partition keys are opaque digests, so the test supplies the exact
/// [`InternalCacheKey`] set to bypass once the index identity is known
/// (see [`ivf_partition_cache_keys`]).
#[derive(Debug)]
struct PartitionBypassCacheBackend {
    inner: lance_core::cache::MokaCacheBackend,
    partition_keys: std::sync::Mutex<HashSet<lance_core::cache::InternalCacheKey>>,
    bypass_partitions: AtomicBool,
    partition_hits: AtomicUsize,
}

impl PartitionBypassCacheBackend {
    fn new() -> Self {
        Self {
            inner: lance_core::cache::MokaCacheBackend::with_capacity(256 * 1024 * 1024),
            partition_keys: std::sync::Mutex::new(HashSet::new()),
            bypass_partitions: AtomicBool::new(false),
            partition_hits: AtomicUsize::new(0),
        }
    }

    fn set_partition_keys(&self, partition_keys: HashSet<lance_core::cache::InternalCacheKey>) {
        *self.partition_keys.lock().unwrap() = partition_keys;
    }

    fn is_partition(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
        self.partition_keys.lock().unwrap().contains(key)
    }

    fn set_bypass_partitions(&self, bypass_partitions: bool) {
        self.bypass_partitions
            .store(bypass_partitions, Ordering::Relaxed);
    }

    fn should_bypass(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
        self.bypass_partitions.load(Ordering::Relaxed) && self.is_partition(key)
    }

    /// Whether the backend currently holds an entry for `key`.
    async fn contains(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
        self.inner.get(key, None).await.is_some()
    }

    fn partition_hits(&self) -> usize {
        self.partition_hits.load(Ordering::Relaxed)
    }
}

/// Derive the internal cache keys of the IVF partition entries for an
/// index, replicating the namespace path
/// `dataset URI -> index UUID -> frag-reuse UUID` used when opening the
/// index. V3 partitions use [`IVFPartitionKey`]; legacy (v0.1/v0.2)
/// indices use `LegacyIVFPartitionKey`.
fn ivf_partition_cache_keys(
    dataset_uri: &str,
    uuid: &uuid::Uuid,
    fri_uuid: Option<&uuid::Uuid>,
    num_partitions: usize,
    index_version: &IndexFileVersion,
) -> HashSet<lance_core::cache::InternalCacheKey> {
    use lance_core::cache::{CacheKey, CacheNamespace, KeyBuilder, UnsizedCacheKey};

    let mut namespace = CacheNamespace::root().child(dataset_uri);
    namespace = namespace.child(uuid.as_hyphenated().to_string().as_str());
    if let Some(fri_uuid) = fri_uuid {
        namespace = namespace.child(fri_uuid.as_hyphenated().to_string().as_str());
    }

    (0..num_partitions)
        .map(|partition_id| {
            if matches!(index_version, IndexFileVersion::V3) {
                let cache_key = IVFPartitionKey::<FlatIndex, ProductQuantizer>::new(partition_id);
                let mut builder = KeyBuilder::new(
                    namespace,
                    IVFPartitionKey::<FlatIndex, ProductQuantizer>::stable_type_id(),
                    IVFPartitionKey::<FlatIndex, ProductQuantizer>::schema(),
                );
                cache_key.write_key(&mut builder);
                builder.finish()
            } else {
                let cache_key = crate::index::vector::ivf::LegacyIVFPartitionKey::new(partition_id);
                let mut builder = KeyBuilder::new(
                    namespace,
                    crate::index::vector::ivf::LegacyIVFPartitionKey::stable_type_id(),
                    crate::index::vector::ivf::LegacyIVFPartitionKey::schema(),
                );
                cache_key.write_key(&mut builder);
                builder.finish()
            }
        })
        .collect()
}

#[async_trait::async_trait]
impl lance_core::cache::CacheBackend for PartitionBypassCacheBackend {
    async fn get(
        &self,
        key: &lance_core::cache::InternalCacheKey,
        codec: Option<lance_core::cache::CacheCodec>,
    ) -> Option<lance_core::cache::CacheEntry> {
        if self.should_bypass(key) {
            None
        } else {
            let entry = self.inner.get(key, codec).await;
            if entry.is_some() && self.is_partition(key) {
                self.partition_hits.fetch_add(1, Ordering::Relaxed);
            }
            entry
        }
    }

    async fn insert(
        &self,
        key: &lance_core::cache::InternalCacheKey,
        entry: lance_core::cache::CacheEntry,
        size_bytes: usize,
        codec: Option<lance_core::cache::CacheCodec>,
    ) {
        if !self.should_bypass(key) {
            self.inner.insert(key, entry, size_bytes, codec).await;
        }
    }

    async fn get_or_insert<'a>(
        &self,
        key: &lance_core::cache::InternalCacheKey,
        loader: std::pin::Pin<
            Box<
                dyn futures::Future<Output = Result<(lance_core::cache::CacheEntry, usize)>>
                    + Send
                    + 'a,
            >,
        >,
        codec: Option<lance_core::cache::CacheCodec>,
    ) -> Result<(lance_core::cache::CacheEntry, bool)> {
        if self.should_bypass(key) {
            let (entry, _) = loader.await?;
            Ok((entry, false))
        } else {
            let result = self.inner.get_or_insert(key, loader, codec).await;
            if result.as_ref().is_ok_and(|(_, is_cache_hit)| *is_cache_hit)
                && self.is_partition(key)
            {
                self.partition_hits.fetch_add(1, Ordering::Relaxed);
            }
            result
        }
    }

    async fn clear(&self) {
        self.inner.clear().await;
    }

    async fn num_entries(&self) -> usize {
        self.inner.num_entries().await
    }

    async fn size_bytes(&self) -> usize {
        self.inner.size_bytes().await
    }

    fn approx_num_entries(&self) -> usize {
        self.inner.approx_num_entries()
    }

    fn approx_size_bytes(&self) -> usize {
        self.inner.approx_size_bytes()
    }
}

/// Integration test: create a vector index, prewarm it through a
/// serializing cache backend, then query. Verifies that entries are
/// serialized to bytes and that queries produce correct results after
/// deserialization.
#[rstest]
#[case::ivf_pq(
        VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        ),
        <PartitionEntry<FlatIndex, ProductQuantizer> as CacheCodecImpl>::TYPE_ID
    )]
#[case::ivf_hnsw_sq(
        VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        ),
        <PartitionEntry<HNSW, ScalarQuantizer> as CacheCodecImpl>::TYPE_ID
    )]
#[tokio::test]
async fn test_prewarm_and_query_with_serializing_backend(
    #[case] params: VectorIndexParams,
    #[case] partition_type_id: &'static str,
) {
    use crate::utils::test::serializing_cache::SerializingCacheBackend;
    use lance_io::assert_io_eq;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    // Create dataset with vector index using default cache
    let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("serde_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let q = Float32Array::from_iter_values(repeat_n(0.5, DIM));
    let expected = ground_truth(&dataset, "vector", &q, 10, DistanceType::L2).await;

    // Re-open with the serializing backend
    let backend = Arc::new(SerializingCacheBackend::new());
    let session = Arc::new(crate::session::Session::with_index_cache_backend(
        backend.clone(),
        128 * 1024 * 1024,
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
    ));
    let dataset = crate::DatasetBuilder::from_uri(test_uri)
        .with_session(session)
        .load()
        .await
        .unwrap();

    // Prewarm — this should serialize entries into the backend
    dataset.prewarm_index("serde_idx").await.unwrap();
    let serialized = backend.serialized_entry_count().await;
    let state_type_id = IvfStateEntryBox::TYPE_ID;
    let state_inserts = backend.serialized_insert_count(state_type_id).await;
    let partition_inserts = backend.serialized_insert_count(partition_type_id).await;
    let passthrough = backend.l1_entry_count().await;
    assert!(
        serialized > 0,
        "prewarm should have serialized entries into the backend"
    );
    assert_eq!(
        passthrough, 0,
        "all index cache entries should have codecs (nothing in passthrough), \
             but found {passthrough} passthrough entries"
    );

    drop(dataset);
    let backend = Arc::new(backend.restart());
    assert_eq!(
        backend.l1_entry_count().await,
        0,
        "restarting must discard the in-memory L1"
    );
    assert_eq!(
        backend.serialized_entry_count().await,
        serialized,
        "restarting must retain the serialized IVF state and partitions"
    );
    let session = Arc::new(crate::session::Session::with_index_cache_backend(
        backend.clone(),
        128 * 1024 * 1024,
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
    ));
    let dataset = crate::DatasetBuilder::from_uri(test_uri)
        .with_session(session)
        .load()
        .await
        .unwrap();

    // Query — the recreated backend will deserialize entries from bytes.
    // All index entries are in serialized form, so every cache hit involves
    // a deserialization round-trip.
    let results = dataset
        .scan()
        .with_row_id()
        .nearest("vector", &q, 10)
        .unwrap()
        .minimum_nprobes(4)
        .maximum_nprobes(4)
        .project(&["_rowid"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        backend.serialized_insert_count(state_type_id).await,
        state_inserts,
        "the first restarted query must reuse the serialized IVF state"
    );
    assert_eq!(
        backend.serialized_insert_count(partition_type_id).await,
        partition_inserts,
        "the first restarted query must reuse every serialized IVF partition"
    );
    assert_eq!(results.num_rows(), 10, "should return 10 nearest neighbors");

    // Verify distances are sorted (ascending for L2)
    let distances: Vec<f32> = results
        .column_by_name("_distance")
        .unwrap()
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    for w in distances.windows(2) {
        assert!(w[1] >= w[0], "distances should be sorted ascending");
    }

    let row_ids = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let recall = row_ids.intersection(&expected).count() as f32 / expected.len() as f32;
    assert_ge!(
        recall,
        0.5,
        "serialized IVF query recall is below threshold: {recall}"
    );

    dataset.object_store.as_ref().io_stats_incremental();
    dataset
        .scan()
        .nearest("vector", &q, 10)
        .unwrap()
        .minimum_nprobes(4)
        .maximum_nprobes(4)
        .project(&["_rowid"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(
        stats,
        read_iops,
        0,
        "warmed IVF query should not perform IO after backend restart"
    );
}

#[rstest]
#[case::v3(IndexFileVersion::V3)]
#[case::legacy(IndexFileVersion::Legacy)]
#[tokio::test]
async fn test_vector_cache_uses_current_object_store(#[case] index_version: IndexFileVersion) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    append_dataset::<Float32Type>(&mut dataset, NUM_ROWS, 0.0..1.0).await;
    assert_eq!(dataset.get_fragments().len(), 2);

    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        PQBuildParams::default(),
    )
    .version(index_version.clone())
    .clone();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("credential_rotation_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();
    let index_meta = dataset
        .load_indices_by_name("credential_rotation_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let query = vectors.value(0);
    let ground_truth = ground_truth(&dataset, "vector", &query, 20, DistanceType::L2).await;

    let cache_backend = Arc::new(PartitionBypassCacheBackend::new());
    let session = Arc::new(crate::session::Session::with_index_cache_backend(
        cache_backend.clone(),
        128 * 1024 * 1024,
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
    ));
    let dataset = crate::DatasetBuilder::from_uri(test_uri)
        .with_session(session)
        .load()
        .await
        .unwrap();

    let store_params_a = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            HashMap::from([(
                "credential_generation".to_owned(),
                "secret-generation-a".to_owned(),
            )]),
        ))),
        ..Default::default()
    };
    let (store_a, _) = ObjectStore::from_uri_and_params(
        dataset.session().store_registry(),
        dataset.uri(),
        &store_params_a,
    )
    .await
    .unwrap();
    let dataset_a = dataset.with_object_store(store_a.clone(), Some(store_params_a));

    let store_params_b = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            HashMap::from([(
                "credential_generation".to_owned(),
                "secret-generation-b".to_owned(),
            )]),
        ))),
        ..Default::default()
    };
    let (store_b, _) = ObjectStore::from_uri_and_params(
        dataset.session().store_registry(),
        dataset.uri(),
        &store_params_b,
    )
    .await
    .unwrap();
    assert!(!Arc::ptr_eq(&store_a, &store_b));
    let dataset_b = dataset.with_object_store(store_b.clone(), Some(store_params_b));

    let _ = store_a.io_stats_incremental();
    let _ = store_b.io_stats_incremental();

    dataset_a
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let frag_reuse_uuid = dataset_a.frag_reuse_index_uuid().await;
    let state_cache_key =
        crate::index::IvfIndexStateCacheKey::new(&index_meta.uuid, frag_reuse_uuid.as_ref());
    let cached_state = if matches!(index_version, IndexFileVersion::V3) {
        Some(
            dataset_a
                .index_cache
                .get_with_key(&state_cache_key)
                .await
                .expect("V3 IVF state should be cached"),
        )
    } else {
        None
    };
    let index_path_fragment = format!("_indices/{}", index_meta.uuid);
    let first_store_stats = store_a.io_stats_incremental();
    assert!(
        first_store_stats
            .requests
            .iter()
            .any(|request| request.path.as_ref().contains(&index_path_fragment)),
        "the first query should read the index through the first object store: {first_store_stats:#?}"
    );
    let partition_keys = ivf_partition_cache_keys(
        dataset.uri(),
        &index_meta.uuid,
        frag_reuse_uuid.as_ref(),
        4,
        &index_version,
    );
    cache_backend.set_partition_keys(partition_keys.clone());
    for partition_key in &partition_keys {
        assert!(
            cache_backend.contains(partition_key).await,
            "the first query should populate portable partition entries"
        );
    }
    let index_entries_after_a = dataset.session().index_cache_stats().await.num_entries;
    let metadata_entries_after_a = dataset.session().metadata_cache_stats().await.num_entries;
    let _ = store_b.io_stats_incremental();

    cache_backend.set_bypass_partitions(true);
    let results = dataset_b
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let recall = row_ids.intersection(&ground_truth).count() as f32 / 20.0;
    assert_ge!(recall, 0.5);

    let old_store_stats = store_a.io_stats_incremental();
    let old_store_index_reads = old_store_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    let new_store_stats = store_b.io_stats_incremental();
    let new_store_index_reads = new_store_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    if matches!(index_version, IndexFileVersion::V3) {
        assert_eq!(
            old_store_index_reads, 0,
            "the new dataset query must not use readers bound to the old object store: {old_store_stats:#?}"
        );
        assert!(
            new_store_index_reads > 0,
            "the new dataset query should read the index through the new object store: {new_store_stats:#?}"
        );
    } else {
        // Legacy live indices are shared across dataset opens: their
        // readers stay bound to the object store that first populated the
        // cache, so the second dataset keeps reading through the old store.
        assert!(
            old_store_index_reads > 0,
            "the cached legacy index should keep reading through the original object store: {old_store_stats:#?}"
        );
        assert_eq!(
            new_store_index_reads, 0,
            "the cached legacy index must not reopen through the new object store: {new_store_stats:#?}"
        );
    }
    if let Some(cached_state) = cached_state {
        let state_after_rotation = dataset_b
            .index_cache
            .get_with_key(&state_cache_key)
            .await
            .expect("V3 IVF state should remain cached after rotation");
        assert!(
            Arc::ptr_eq(&cached_state, &state_after_rotation),
            "store-free IVF state should be reused across object-store generations"
        );
    }

    // Re-query through the first dataset: V3 portable state is rebound to
    // the store supplied by each reconstruction, while the cached legacy
    // index keeps reading through its original store. Either way the second
    // store must not be touched.
    let _ = store_a.io_stats_incremental();
    let _ = store_b.io_stats_incremental();
    dataset_a
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let store_a_stats = store_a.io_stats_incremental();
    let store_a_index_reads = store_a_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    let store_b_stats = store_b.io_stats_incremental();
    let store_b_index_reads = store_b_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    assert!(
        store_a_index_reads > 0,
        "re-querying the first dataset should read the index through its object store: {store_a_stats:#?}"
    );
    assert_eq!(
        store_b_index_reads, 0,
        "re-querying the first dataset must not use the second object store: {store_b_stats:#?}"
    );

    // Cache keys are opaque digests, so they cannot embed credential
    // material by construction. What rotation must not do is mint new
    // entries: the same portable state, partitions, and file metadata
    // serve both object-store generations.
    let index_entries_after_rotation = dataset.session().index_cache_stats().await.num_entries;
    let metadata_entries_after_rotation =
        dataset.session().metadata_cache_stats().await.num_entries;
    assert_eq!(
        index_entries_after_rotation, index_entries_after_a,
        "credential rotation must not create new index cache entries"
    );
    assert_eq!(
        metadata_entries_after_rotation, metadata_entries_after_a,
        "credential rotation must not create new metadata cache entries"
    );

    cache_backend.set_bypass_partitions(false);
    let partition_hits_before = cache_backend.partition_hits();
    dataset_b
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    assert!(
        cache_backend.partition_hits() > partition_hits_before,
        "the second store should reuse portable partitions populated by the first"
    );
}

#[tokio::test]
async fn test_shallow_clone_ivf_rq_uses_resolved_index_directory() {
    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/source", test_dir.as_str());
    let clone_uri = format!("{}/clone", test_dir.as_str());
    let (mut source, vectors) = generate_test_dataset::<Float32Type>(&source_uri, 0.0..1.0).await;
    append_dataset::<Float32Type>(&mut source, NUM_ROWS, 0.0..1.0).await;
    assert_eq!(source.get_fragments().len(), 2);

    let params = VectorIndexParams::ivf_rq(4, 5, DistanceType::L2);
    source
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("ivf_rq_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let query = vectors.value(0);
    let ground_truth = ground_truth(&source, "vector", &query, 20, DistanceType::L2).await;
    source
        .tags()
        .create("with_ivf_rq", source.version().version)
        .await
        .unwrap();
    let cloned = source
        .shallow_clone(&clone_uri, "with_ivf_rq", None)
        .await
        .unwrap();

    let index_meta = cloned
        .load_indices_by_name("ivf_rq_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert!(
        index_meta.base_id.is_some(),
        "a shallow-cloned index should reference its source base"
    );
    assert_eq!(
        cloned.indice_files_dir(&index_meta).unwrap(),
        source.indices_dir(),
        "the cloned index should resolve its path through the source base"
    );
    assert_ne!(
        cloned.indice_files_dir(&index_meta).unwrap(),
        cloned.indices_dir(),
        "the cloned index should not use the clone's primary index directory"
    );

    let cloned = crate::DatasetBuilder::from_uri(&clone_uri)
        .with_session(Arc::new(crate::session::Session::default()))
        .load()
        .await
        .unwrap();

    let results = cloned
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let recall = row_ids.intersection(&ground_truth).count() as f32 / 20.0;
    assert_ge!(recall, 0.5);
}

mod native_resident_tests;
mod native_scan_tests;
mod native_share_tests;
mod native_storage_tests;
