// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF_VAMANA_* through Lance: created, described, searched and maintained
//! like IVF_HNSW_*.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type, UInt8Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, RecordBatchIterator,
    UInt8Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::exec::ExecutionSummaryCounts;
use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
use lance_datagen::{ArrayGeneratorExt, BatchCount, RowCount, Seed, array, gen_batch};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::optimize::OptimizeOptions;
use lance_index::pb::vector_index_details::{Compression, ScalarQuantization};
use lance_index::pb::{HnswParameters, VamanaParameters, VectorIndexDetails};
use lance_index::prefilter::NoFilter;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::flat::index::FlatQuantizer;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_index::vector::quantizer::{Quantization, QuantizationType};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::{IvfSubIndex, SubIndexType};
use lance_index::vector::vamana::graph::Adjacency;
use lance_index::vector::vamana::{AdaptiveStop, Vamana, VamanaBuildParams};
use lance_index::vector::{
    DEFAULT_QUERY_PARALLELISM, DIST_COL, PQ_CODE_COLUMN, PartitionSearchControl, Query,
    SQ_CODE_COLUMN, VectorIndex,
};
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, IVF_RQ_INDEX_VERSION, IVF_VAMANA_INDEX_VERSION,
    IndexType,
};
use lance_io::scheduler::IoStats;
use lance_io::utils::tracking_store::IoRequestRecord;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use lance_table::io::manifest::read_manifest_indexes;
use lance_testing::datagen::generate_random_array_with_seed;
use roaring::RoaringBitmap;
use rstest::rstest;
use uuid::Uuid;

use super::merge::merge_base;
use super::vectors::read_vectors;
use crate::Error;
use crate::dataset::UpdateBuilder;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::mem_wal::index::{HnswMemIndexPlugin, MemIndexPlugin, ResolveContext};
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::scanner::{QueryFilter, Scanner};
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, Dataset, WriteMode, WriteParams};
use crate::index::vector::builder::{ExistingIndex, index_type_string};
use crate::index::vector::ivf::v2::IVFIndex;
use crate::index::vector::utils::get_vector_dim;
use crate::index::vector::{IndexFileVersion, StageParams, VectorIndexParams};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::index::{unsupported_index_version, vector_index_details_default};

const DIM: u32 = 32;
/// The vectors of product-code cases. In a debug build a graph over product
/// codes draws several times as slowly as one over scalar codes, as every
/// prune copies a table of each sub-vector's codewords, so these cases index
/// fewer dimensions, in four sub-vectors of two dimensions each.
const PQ_DIM: u32 = 8;
/// Vectors wide enough that Lance stores them flat, so that they are scored
/// again from their byte offsets.
const WIDE: u32 = 128;
const ROWS: u64 = 600;
const INDEX: &str = "vec_idx";
/// The search list of indexed queries: wide enough that recall measures the
/// wiring and the codes rather than the walk, whose recall `lance-index`
/// tests.
const EF: usize = 60;

fn vamana() -> VamanaBuildParams {
    VamanaBuildParams {
        max_degree: 16,
        search_list_size: 32,
        alpha: 1.2,
    }
}

fn ivf() -> IvfBuildParams {
    IvfBuildParams {
        num_partitions: Some(2),
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum Codes {
    Flat,
    Sq,
    Pq,
}

impl Codes {
    /// The dimension of the vectors a case of these codes indexes.
    fn dimension(self) -> u32 {
        match self {
            Self::Flat | Self::Sq => DIM,
            Self::Pq => PQ_DIM,
        }
    }
}

fn params(codes: Codes, distance_type: DistanceType) -> VectorIndexParams {
    match codes {
        Codes::Flat => VectorIndexParams::ivf_vamana(distance_type, ivf(), vamana()),
        Codes::Sq => VectorIndexParams::with_ivf_vamana_sq_params(
            distance_type,
            ivf(),
            vamana(),
            SQBuildParams::default(),
        ),
        // Centroids and codebook both given, so that the codes are the same on
        // every run.
        Codes::Pq => VectorIndexParams::with_ivf_vamana_pq_params(
            distance_type,
            IvfBuildParams::try_with_centroids(
                2,
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(splitting_centroids(distance_type, PQ_DIM).concat()),
                        PQ_DIM as i32,
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
            vamana(),
            PQBuildParams {
                num_sub_vectors: PQ_DIM as usize / 2,
                num_bits: 8,
                codebook: Some(pq_codebook(PQ_DIM, distance_type)),
                ..Default::default()
            },
        ),
    }
}

/// Two IVF centroids of `dimension` values that split the vectors of the cube
/// under `distance_type`: under L2 the [`DIAGONAL`] ones; under cosine, where
/// the diagonal is one direction, unit vectors along the even axes and along
/// the odd ones.
fn splitting_centroids(distance_type: DistanceType, dimension: u32) -> [Vec<f32>; 2] {
    match distance_type {
        DistanceType::Cosine => [0, 1].map(|parity| {
            let unit = (dimension as f32 / 2.0).sqrt().recip();
            (0..dimension)
                .map(|axis| if axis % 2 == parity { unit } else { 0.0 })
                .collect()
        }),
        _ => DIAGONAL.map(|value| vec![value; dimension as usize]),
    }
}

/// A codebook of product codes no build trains, the same on every run: the
/// residuals of 256 vectors of the cube drawn from a seed against the nearer
/// [`splitting_centroids`], as k-means would start from, the vectors
/// normalized under cosine as the codes measure them.
fn pq_codebook(dimension: u32, distance_type: DistanceType) -> ArrayRef {
    let centroids = splitting_centroids(distance_type, dimension);
    let vectors = generate_random_array_with_seed::<Float32Type>(256 * dimension as usize, [5; 32]);
    let codewords = vectors
        .values()
        .chunks(dimension as usize)
        .flat_map(|vector| {
            let scale = match distance_type {
                DistanceType::Cosine => {
                    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
                }
                _ => 1.0,
            };
            let residual = |centroid: &Vec<f32>| -> Vec<f32> {
                vector
                    .iter()
                    .zip(centroid)
                    .map(|(value, center)| value / scale - center)
                    .collect()
            };
            let [near, far] = centroids.each_ref().map(residual);
            let length =
                |residual: &Vec<f32>| residual.iter().map(|value| value * value).sum::<f32>();
            if length(&near) <= length(&far) {
                near
            } else {
                far
            }
        })
        .collect::<Vec<_>>();
    Arc::new(Float32Array::from(codewords))
}

/// Centroids a quarter and three quarters of the way along the diagonal of
/// the cube the vectors fill. Training often splits the rows three to one;
/// these split the rows of [`two_fragments`] 307 to 293 on every build.
const DIAGONAL: [f32; 2] = [0.25, 0.75];

/// [`params`] under L2 with the IVF centroids at [`DIAGONAL`].
fn diagonal_params(codes: Codes) -> VectorIndexParams {
    let dimension = codes.dimension();
    let centroids = Float32Array::from_iter_values(
        DIAGONAL
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, dimension as usize)),
    );
    let centroids = FixedSizeListArray::try_new_from_values(centroids, dimension as i32).unwrap();
    let mut params = params(codes, DistanceType::L2);
    params.stages[0] =
        StageParams::Ivf(IvfBuildParams::try_with_centroids(2, Arc::new(centroids)).unwrap());
    params
}

/// `ROWS` random vectors of `DIM` values, numbered by `id`, in two fragments.
async fn two_fragments(uri: &str) -> Dataset {
    two_fragments_of(uri, DIM, false).await
}

/// [`two_fragments`] of vectors of `dimension` values, with stable row ids
/// when `enable_stable_row_ids`.
async fn two_fragments_of(uri: &str, dimension: u32, enable_stable_row_ids: bool) -> Dataset {
    two_fragments_of_rows(uri, ROWS, dimension, enable_stable_row_ids).await
}

/// [`two_fragments_of`] with `rows` vectors in all.
async fn two_fragments_of_rows(
    uri: &str,
    rows: u64,
    dimension: u32,
    enable_stable_row_ids: bool,
) -> Dataset {
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(dimension.into()))
        .into_reader_rows(RowCount::from(rows / 2), BatchCount::from(2));
    let params = WriteParams {
        max_rows_per_file: (rows / 2) as usize,
        enable_stable_row_ids,
        ..Default::default()
    };
    let dataset = Dataset::write(data, uri, Some(params)).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
}

/// Ten fixed queries in the unit cube the vectors fill.
fn queries() -> Vec<Float32Array> {
    queries_of(DIM)
}

/// [`queries`] of `dimension` values.
fn queries_of(dimension: u32) -> Vec<Float32Array> {
    let mut state = 7u32;
    (0..10)
        .map(|_| {
            Float32Array::from_iter_values((0..dimension).map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1 << 24) as f32
            }))
        })
        .collect()
}

/// One query's `k` nearest rows and their distances, nearest first, with the
/// counts of the search: through the index, under the metric it was built
/// with, or, given `exact`, by brute force under that metric; `configure`
/// sets the scan up further.
async fn search(
    dataset: &Dataset,
    query: &Float32Array,
    k: usize,
    exact: Option<DistanceType>,
    configure: impl FnOnce(&mut Scanner),
) -> (Vec<(u64, f32)>, ExecutionSummaryCounts) {
    let mut scan = dataset.scan();
    scan.nearest("vec", query, k)
        .unwrap()
        .minimum_nprobes(2)
        .ef(EF)
        .with_row_id();
    // Brute force never reads the index's metric: left unset, it ranks by L2.
    if let Some(distance_type) = exact {
        scan.use_index(false).distance_metric(distance_type);
    }
    configure(&mut scan);
    let counts = Arc::new(Mutex::new(None));
    let record = counts.clone();
    scan.scan_stats_callback(Arc::new(move |stats| {
        *record.lock().unwrap() = Some(stats.clone());
    }));
    let rows = rows(&scan.try_into_batch().await.unwrap());
    let counts = counts.lock().unwrap().take().unwrap();
    (rows, counts)
}

/// The `(row id, distance)` pairs of a batch of search results.
fn rows(batch: &RecordBatch) -> Vec<(u64, f32)> {
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .zip(
            batch[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .copied(),
        )
        .collect()
}

/// [`search`] as configured for every query.
async fn nearest(
    dataset: &Dataset,
    query: &Float32Array,
    k: usize,
    exact: Option<DistanceType>,
) -> Vec<(u64, f32)> {
    search(dataset, query, k, exact, |_| {}).await.0
}

/// Every row of `found` is ranked by `exact`, which brute force measured,
/// and stands at the distance it measured there, nearest first.
fn assert_exact(found: &[(u64, f32)], exact: &[(u64, f32)]) {
    assert!(
        found.windows(2).all(|pair| pair[0].1 <= pair[1].1),
        "{found:?}"
    );
    let exact = exact.iter().copied().collect::<HashMap<_, _>>();
    for (row_id, distance) in found {
        let expected = exact
            .get(row_id)
            .unwrap_or_else(|| panic!("row {row_id} at {distance} is not one brute force ranks"));
        assert!(
            (distance - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "row {row_id} at {distance}, not at {expected}"
        );
    }
}

/// The share of each query's exact `k` nearest rows under `distance_type`
/// that the index finds, every row it finds at its exact distance.
async fn recall(dataset: &Dataset, k: usize, distance_type: DistanceType) -> f32 {
    recall_over(
        dataset,
        &queries_of(dimension_of(dataset)),
        k,
        distance_type,
    )
    .await
}

/// The dimension of `dataset`'s vectors.
fn dimension_of(dataset: &Dataset) -> u32 {
    get_vector_dim(dataset.schema(), "vec").unwrap() as u32
}

/// [`recall`] over `queries`.
async fn recall_over(
    dataset: &Dataset,
    queries: &[Float32Array],
    k: usize,
    distance_type: DistanceType,
) -> f32 {
    let rows = dataset.count_rows(None).await.unwrap();
    let mut found = 0;
    for query in queries {
        let indexed = nearest(dataset, query, k, None).await;
        let exact = nearest(dataset, query, rows, Some(distance_type)).await;
        assert_exact(&indexed, &exact);
        let indexed = indexed
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect::<HashSet<_>>();
        found += exact
            .iter()
            .take(k)
            .filter(|(row_id, _)| indexed.contains(row_id))
            .count();
    }
    found as f32 / (k * queries.len()) as f32
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap()
}

/// Java passes these codes to `IndexType::try_from`, so they are pinned here.
#[test]
fn the_vamana_types_name_and_number_themselves() {
    for (index_type, name, code, quantization) in [
        (
            IndexType::IvfVamanaFlat,
            "IVF_VAMANA_FLAT",
            108,
            QuantizationType::Flat,
        ),
        (
            IndexType::IvfVamanaSq,
            "IVF_VAMANA_SQ",
            109,
            QuantizationType::Scalar,
        ),
        (
            IndexType::IvfVamanaPq,
            "IVF_VAMANA_PQ",
            110,
            QuantizationType::Product,
        ),
    ] {
        assert_eq!(index_type.to_string(), name);
        assert_eq!(IndexType::try_from(code).unwrap(), index_type);
        assert!(index_type.is_vector());
        assert_eq!(index_type.target_partition_size(), 1 << 20);
        assert_eq!(super::index_type(quantization), index_type);
        assert_eq!(index_type_string(SubIndexType::Vamana, quantization), name);
    }
    assert!(matches!(
        SubIndexType::try_from("VAMANA").unwrap(),
        SubIndexType::Vamana
    ));
    assert_eq!(SubIndexType::Vamana.to_string(), "VAMANA");
}

fn segment(index_version: i32, details: VectorIndexDetails) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        name: INDEX.into(),
        fields: vec![0],
        covering_fields: vec![],
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
        index_version,
        created_at: None,
        base_id: None,
        files: None,
    }
}

fn vamana_details() -> VectorIndexDetails {
    VectorIndexDetails {
        vamana_index_config: Some(VamanaParameters {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
        }),
        ..Default::default()
    }
}

/// A build reads IVF_VAMANA_* segments, also with the empty details a failed
/// inference leaves, and hides the next vector version. A segment is an
/// IVF_VAMANA_* one by its version and details together: neither the version
/// with another index's details nor empty details at another version is.
#[test]
fn a_build_reads_vamana_segments_and_tells_them_apart() {
    let version = IVF_VAMANA_INDEX_VERSION as i32;
    let hnsw = VectorIndexDetails {
        hnsw_index_config: Some(HnswParameters::default()),
        ..Default::default()
    };
    for (index, is_vamana) in [
        (segment(version, vamana_details()), true),
        (segment(version, VectorIndexDetails::default()), true),
        (segment(version, hnsw), false),
        (
            segment(IVF_RQ_INDEX_VERSION as i32, VectorIndexDetails::default()),
            false,
        ),
    ] {
        assert_eq!(unsupported_index_version(&index), None, "{index:?}");
        assert_eq!(super::is_vamana_segment(&index), is_vamana, "{index:?}");
    }
    let next = segment(version + 1, vamana_details());
    assert_eq!(
        unsupported_index_version(&next),
        Some(IndexType::max_vector_version())
    );
    assert!(!super::is_vamana_segment(&segment(
        version + 1,
        VectorIndexDetails::default()
    )));
}

#[rstest]
#[tokio::test]
async fn an_index_is_built_described_and_searched(
    #[values(Codes::Flat, Codes::Sq, Codes::Pq)] codes: Codes,
    #[values(DistanceType::L2, DistanceType::Cosine)] distance_type: DistanceType,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of(dir.as_str(), codes.dimension(), false).await;
    let params = params(codes, distance_type);
    let type_name = params.index_type().to_string();
    let index = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    assert_eq!(index.index_version, IVF_VAMANA_INDEX_VERSION as i32);

    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].index_type(), type_name);
    let details = json(&descriptions[0].details().unwrap());
    assert_eq!(
        details["vamana"],
        serde_json::json!({"max_degree": 16, "search_list_size": 32, "alpha": 1.2})
    );
    assert!(details.get("hnsw").is_none(), "{details}");

    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    assert_eq!(statistics["index_type"], type_name.as_str());
    for segment in statistics["indices"].as_array().unwrap() {
        assert_eq!(segment["index_type"], type_name.as_str());
        assert_eq!(segment["metric_type"], distance_type.to_string());
        assert_eq!(segment["num_partitions"], 2);
        assert_eq!(segment["sub_index"]["index_type"], "VAMANA");
        assert_eq!(segment["sub_index"]["params"]["max_degree"], 16);
    }

    let recall = recall(&dataset, 10, distance_type).await;
    assert!(recall >= 0.95, "recall {recall}");

    // As over HNSW, a list shorter than the results is refused; one shorter
    // than the shortlist is widened to it.
    let query = &queries_of(codes.dimension())[0];
    let (found, _) = search(&dataset, query, 10, None, |scan| {
        scan.ef(10);
    })
    .await;
    assert_eq!(found.len(), 10);
    for (ef, refine_factor) in [(9, None), (15, Some(2))] {
        let mut scan = dataset.scan();
        scan.nearest("vec", query, 10).unwrap().ef(ef);
        if let Some(refine_factor) = refine_factor {
            scan.refine(refine_factor);
        }
        let error = scan.try_into_batch().await.unwrap_err();
        // Behind a refine the error reaches the caller shared, as an external one.
        assert!(
            matches!(error, Error::Execution { .. } | Error::External { .. }),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("ef must be greater than or equal to k"),
            "{error}"
        );
    }
}

async fn indexed(uri: &str, codes: Codes, distance_type: DistanceType) -> Dataset {
    indexed_of(uri, codes, distance_type, codes.dimension()).await
}

/// [`indexed`] over vectors of `dimension` values.
async fn indexed_of(
    uri: &str,
    codes: Codes,
    distance_type: DistanceType,
    dimension: u32,
) -> Dataset {
    let mut dataset = two_fragments_of(uri, dimension, false).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(codes, distance_type),
            true,
        )
        .await
        .unwrap();
    dataset
}

/// The index a query searches answers at exact distances whether it was
/// opened from its files or rebuilt from the state the cache keeps, under
/// either distance, over vectors narrow enough that Lance stores them in
/// mini-blocks, which are scored again through take, or wide enough that it
/// stores them flat, which are scored again from their byte offsets. A query
/// opens the index twice, to rank the partitions and then to search them:
/// with an index cache that keeps nothing both read the files, and with one
/// that keeps the state only the first does.
#[rstest]
#[tokio::test]
async fn the_searched_index_answers_at_exact_distances_cold_or_warm(
    #[values((0, 2), (DEFAULT_INDEX_CACHE_SIZE, 1))] cache: (usize, usize),
    #[values(DIM, WIDE)] dimension: u32,
    #[values(DistanceType::L2, DistanceType::Cosine)] distance_type: DistanceType,
) {
    let (index_cache_size_bytes, indices_loaded) = cache;
    let dir = TempStrDir::default();
    // Half the rows keep a case of wide vectors within a second.
    let rows = if dimension == WIDE { ROWS / 2 } else { ROWS };
    let mut built = two_fragments_of_rows(dir.as_str(), rows, dimension, false).await;
    built
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, distance_type),
            true,
        )
        .await
        .unwrap();
    let dataset = DatasetBuilder::from_uri(dir.as_str())
        .with_index_cache_size_bytes(index_cache_size_bytes)
        .load()
        .await
        .unwrap();
    let queries = queries_of(dimension);
    let exact = nearest(&dataset, &queries[0], rows as usize, Some(distance_type)).await;
    let (found, counts) = search(&dataset, &queries[0], 10, None, |_| {}).await;
    assert_eq!(counts.indices_loaded, indices_loaded);
    assert_eq!(found.len(), 10);
    assert_exact(&found, &exact);
    let recall = recall_over(&dataset, &queries, 10, distance_type).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Vectors scored again at their byte offsets are read as the query's own
/// reads, where take would count none: once the partitions are cached, a
/// query that projects no column, so that nothing is taken after the index,
/// still reads at least the vectors of its shortlist.
#[tokio::test]
async fn vectors_scored_again_at_their_offsets_are_read_as_part_of_the_query() {
    let dir = TempStrDir::default();
    let dataset = indexed_of(dir.as_str(), Codes::Sq, DistanceType::L2, WIDE).await;
    let query = &queries_of(WIDE)[0];
    let project_no_column = |scan: &mut Scanner| {
        scan.project::<&str>(&[]).unwrap().with_row_id();
    };
    search(&dataset, query, 10, None, project_no_column).await;
    let (found, counts) = search(&dataset, query, 10, None, project_no_column).await;
    assert_eq!(found.len(), 10);
    assert_eq!(counts.parts_loaded, 0, "{counts:?}");
    let shortlist_bytes = 2 * 10 * WIDE as usize * 4;
    assert!(counts.bytes_read >= shortlist_bytes, "{counts:?}");
}

/// The ways a query reaches an index's partitions.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// With the filter applied before the walks.
    Prefiltered,
    /// With the filter applied to what the walks return.
    Postfiltered,
    /// Through one partition, then, as it holds too few rows that pass the
    /// filter, through the others one at a time.
    PartitionByPartition,
    /// Through every partition at once.
    InParallel,
    /// With twice the results asked of the index, which refine ranks again
    /// from the dataset's vectors.
    Refined,
}

/// Every way a query reaches the partitions answers at exact distances, and
/// with only the rows that pass its filter.
#[rstest]
#[case::prefiltered(Path::Prefiltered)]
#[case::postfiltered(Path::Postfiltered)]
#[case::partition_by_partition(Path::PartitionByPartition)]
#[case::in_parallel(Path::InParallel)]
#[case::refined(Path::Refined)]
#[tokio::test]
async fn every_way_to_the_partitions_answers_at_exact_distances(#[case] path: Path) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    // A trained split of three to one can leave the larger partition `k` of
    // the rows that pass a filter.
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &diagonal_params(Codes::Sq),
            true,
        )
        .await
        .unwrap();
    // Partition by partition, a twentieth of the rows pass, 12 and 18 of
    // them in the two partitions, and more are asked for than either holds.
    let (filter, k) = match path {
        Path::Prefiltered | Path::Postfiltered => (Some("id % 3 = 0"), 10),
        Path::PartitionByPartition => (Some("id % 20 = 0"), 25),
        Path::InParallel | Path::Refined => (None, 10),
    };
    let configure = |scan: &mut Scanner| {
        if let Some(filter) = filter {
            scan.filter(filter).unwrap();
        }
        match path {
            Path::Prefiltered => {
                scan.prefilter(true);
            }
            Path::Postfiltered => {
                scan.prefilter(false);
            }
            Path::PartitionByPartition => {
                scan.prefilter(true)
                    .minimum_nprobes(1)
                    .search_effort(0.0)
                    .unwrap();
            }
            Path::InParallel => {
                scan.query_parallelism(2);
            }
            Path::Refined => {
                scan.refine(2);
            }
        }
    };
    let mut found_nearest = 0;
    for query in queries() {
        let (found, counts) = search(&dataset, &query, k, None, configure).await;
        let (exact, _) = search(
            &dataset,
            &query,
            ROWS as usize,
            Some(DistanceType::L2),
            configure,
        )
        .await;
        assert_exact(&found, &exact);
        // A postfilter keeps what passes of the unfiltered answer, in order.
        if matches!(path, Path::Postfiltered) {
            let passing = exact
                .iter()
                .map(|(row_id, _)| *row_id)
                .collect::<HashSet<_>>();
            let unfiltered = nearest(&dataset, &query, k, None).await;
            let kept = unfiltered
                .into_iter()
                .filter(|(row_id, _)| passing.contains(row_id))
                .collect::<Vec<_>>();
            assert_eq!(found, kept);
        }
        let found = found
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect::<HashSet<_>>();
        found_nearest += exact
            .iter()
            .take(k)
            .filter(|(row_id, _)| found.contains(row_id))
            .count();
        if matches!(path, Path::PartitionByPartition) {
            assert_eq!(counts.all_counts[PARTITIONS_SEARCHED_METRIC], 2);
        }
    }
    if !matches!(path, Path::Postfiltered) {
        let recall = found_nearest as f32 / (k * queries().len()) as f32;
        assert!(recall >= 0.95, "recall {recall}");
    }
}

/// Stops a search once it has recorded `batches` partitions' results, and
/// keeps what it records.
struct StopAfter {
    batches: usize,
    recorded: Mutex<Vec<RecordBatch>>,
}

impl StopAfter {
    fn new(batches: usize) -> Self {
        Self {
            batches,
            recorded: Mutex::default(),
        }
    }
}

impl PartitionSearchControl for StopAfter {
    fn should_stop(&self) -> bool {
        self.recorded.lock().unwrap().len() >= self.batches
    }

    fn record_batch(&self, batch: &RecordBatch) {
        self.recorded.lock().unwrap().push(batch.clone());
    }
}

/// Counts the I/O of the searches it is handed to.
struct IoCounter(IoStats);

impl MetricsCollector for IoCounter {
    fn record_parts_loaded(&self, _num_parts: usize) {}
    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
    fn io_stats(&self) -> Option<IoStats> {
        Some(self.0.clone())
    }
}

/// Partitions searched together answer with one batch, the query's results
/// at exact distances. Under a control the partitions are searched one after
/// another from the first of the range, each answering as it does alone, and
/// the control records each answer before it decides whether the next
/// partition is searched. Either way, once the partitions are cached, the
/// search reads at least the vectors of each shortlist it scores again, at
/// their byte offsets, as its own reads. An empty range answers with nothing,
/// refusing no `ef`, as HNSW refuses one only in a partition it searches. The
/// index is opened by its name, as prewarming opens it, and still reads its
/// column.
#[tokio::test]
async fn partitions_answer_together_unless_a_control_steps_through_them() {
    let shortlist_bytes = 2 * 10 * u64::from(WIDE) * 4;
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of(dir.as_str(), WIDE, false).await;
    let mut params = params(Codes::Sq, DistanceType::L2);
    params.stages[0] = StageParams::Ivf(IvfBuildParams {
        num_partitions: Some(4),
        ..Default::default()
    });
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let uuid = dataset.load_indices_by_name(INDEX).await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index(INDEX, &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let key = queries_of(WIDE).swap_remove(0);
    let exact = nearest(&dataset, &key, ROWS as usize, Some(DistanceType::L2)).await;
    let query = Query {
        column: "vec".to_string(),
        key: Arc::new(key),
        k: 10,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 4,
        maximum_nprobes: Some(4),
        search_effort: 0.5,
        ef: Some(EF),
        refine_factor: None,
        metric_type: Some(DistanceType::L2),
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
        adaptive_stop: None,
    };
    let (partitions, q_c_dists) = index.find_partitions(&query).unwrap();
    assert_eq!(partitions.len(), 4);
    let mut alone = Vec::new();
    for (partition, dist_q_c) in partitions.values().iter().zip(q_c_dists.values()) {
        let mut query = query.clone();
        query.dist_q_c = *dist_q_c;
        let batch = index
            .search_in_partition(
                *partition as usize,
                &query,
                Arc::new(NoFilter),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 10);
        assert_exact(&rows(&batch), &exact);
        alone.push(batch);
    }
    let (partitions, q_c_dists) = (Arc::new(partitions), Arc::new(q_c_dists));
    let search = |start: usize, control: Option<Arc<dyn PartitionSearchControl>>| {
        let index = index.clone();
        let (query, partitions, q_c_dists) = (query.clone(), partitions.clone(), q_c_dists.clone());
        let io_stats = IoStats::new();
        let metrics = Arc::new(IoCounter(io_stats.clone()));
        async move {
            let batches = index
                .search_partitions(
                    query,
                    partitions,
                    q_c_dists,
                    start,
                    4,
                    Arc::new(NoFilter),
                    control,
                    metrics,
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            (batches, io_stats.snapshot().bytes_read)
        }
    };

    let (together, bytes_read) = search(0, None).await;
    assert_eq!(together.len(), 1);
    assert_eq!(together[0].num_rows(), 10);
    assert_exact(&rows(&together[0]), &exact);
    assert!(bytes_read >= shortlist_bytes, "{bytes_read} bytes read");

    for (batches, searched) in [(2, 2), (usize::MAX, 3)] {
        let control = Arc::new(StopAfter::new(batches));
        let (stepped, bytes_read) = search(1, Some(control.clone())).await;
        assert_eq!(stepped, alone[1..1 + searched]);
        assert_eq!(*control.recorded.lock().unwrap(), stepped);
        assert!(
            bytes_read >= searched as u64 * shortlist_bytes,
            "{bytes_read} bytes read"
        );
    }

    let short_list = Query {
        ef: Some(5),
        ..query.clone()
    };
    let none = index
        .clone()
        .search_partitions(
            short_list,
            partitions.clone(),
            q_c_dists.clone(),
            4,
            4,
            Arc::new(NoFilter),
            None,
            Arc::new(NoOpMetricsCollector),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert!(none.iter().all(|batch| batch.num_rows() == 0), "{none:?}");
}

/// Without a refine factor a query scores twice `k` of the walk's candidates
/// again, so it answers exactly as a refine of factor 2 does, which reads back
/// the vectors of the `2k` nearest the walks find and keeps the `k` nearest:
/// with one partition both score the same candidates of the same walk, and
/// with two the same `2k` of both walks together. The codes are coarse enough
/// that refines of factor 1 and 3, which score the `k` and the `3k` nearest,
/// each answer otherwise for some query, as a query scoring that many
/// candidates again would.
#[rstest]
#[tokio::test]
async fn a_query_scores_twice_k_of_the_walk_again(#[values(1, 2)] num_partitions: usize) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of(dir.as_str(), PQ_DIM, false).await;
    // Given centroids, so that the codes are the same on every run.
    let centroids = match num_partitions {
        1 => vec![0.5; PQ_DIM as usize],
        _ => splitting_centroids(DistanceType::L2, PQ_DIM).concat(),
    };
    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroids), PQ_DIM as i32)
            .unwrap();
    let params = VectorIndexParams::with_ivf_vamana_pq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(num_partitions, Arc::new(centroids)).unwrap(),
        vamana(),
        // Eight dimensions to a sub-vector: coarse codes.
        PQBuildParams {
            num_sub_vectors: PQ_DIM as usize / 8,
            num_bits: 8,
            codebook: Some(pq_codebook(PQ_DIM, DistanceType::L2)),
            ..Default::default()
        },
    );
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let mut refined_otherwise = [0; 2];
    for query in queries_of(PQ_DIM) {
        let (plain, _) = search(&dataset, &query, 10, None, |_| {}).await;
        let mut refined = Vec::new();
        for factor in 1..=3 {
            let (found, _) = search(&dataset, &query, 10, None, |scan| {
                scan.refine(factor);
            })
            .await;
            refined.push(found);
        }
        assert_eq!(plain, refined[1]);
        refined_otherwise[0] += usize::from(refined[0] != refined[1]);
        refined_otherwise[1] += usize::from(refined[2] != refined[1]);
    }
    assert!(
        refined_otherwise.iter().all(|queries| *queries > 0),
        "queries refines of factor 1 and 3 answer otherwise for: {refined_otherwise:?}"
    );
}

/// A margin hangs its bar off the query's own `k`-th candidate, however many
/// candidates each partition is asked for: at zero, over scalar codes scored
/// again (each partition answering with `2k`), over flat codes, and under a
/// refine of three, a walk measures exactly what the walk at `L = k` does.
/// That walk is a refine of one with `ef` at `k`: without a refine a walk
/// keeps at least `2k`. The same cap walks further without the margin.
#[rstest]
#[case::scalar_codes_scored_again(Codes::Sq, None)]
#[case::flat_codes(Codes::Flat, None)]
#[case::refined(Codes::Sq, Some(3))]
#[tokio::test]
async fn a_margin_of_zero_measures_what_the_walk_at_k_does(
    #[case] codes: Codes,
    #[case] refine_factor: Option<u32>,
) {
    let dir = TempStrDir::default();
    let dataset = indexed(dir.as_str(), codes, DistanceType::L2).await;
    let capped = |margin: Option<f32>| {
        move |scan: &mut Scanner| {
            scan.ef(200);
            if let Some(factor) = refine_factor {
                scan.refine(factor);
            }
            if let Some(margin) = margin {
                scan.stop_margin(margin).unwrap();
            }
        }
    };
    let mut further = 0;
    for query in queries_of(codes.dimension()) {
        let (at_k, at_k_counts) = search(&dataset, &query, 10, None, |scan| {
            scan.refine(1).ef(10);
        })
        .await;
        let (ruled, ruled_counts) = search(&dataset, &query, 10, None, capped(Some(0.0))).await;
        assert_eq!(
            ruled_counts.index_comparisons,
            at_k_counts.index_comparisons
        );
        // Flat codes are exact, so their walks answer with the same rows.
        if matches!(codes, Codes::Flat) {
            let rows = |found: &[(u64, f32)]| {
                found
                    .iter()
                    .map(|(row_id, _)| *row_id)
                    .collect::<HashSet<_>>()
            };
            assert_eq!(rows(&ruled), rows(&at_k));
        }
        let (_, unruled_counts) = search(&dataset, &query, 10, None, capped(None)).await;
        further += usize::from(unruled_counts.index_comparisons > at_k_counts.index_comparisons);
    }
    assert!(
        further > 0,
        "the cap never let a walk without the margin go further than L = k, so nothing showed that \
         the margin was what stopped it"
    );
}

/// A wider margin walks further, which is what makes the margin a knob: over
/// the queries together the distances never fall from one margin to the
/// next, and some query measures more at the widest than at zero. Query by
/// query this need not hold, as a hop of four can take vertices a narrower
/// margin left and part from its walk (`lance-index` checks it a vertex a
/// hop). Partitions along the diagonal, so that the index is the same on
/// every build.
#[tokio::test]
async fn a_wider_margin_walks_further_over_the_queries() {
    let dir = TempStrDir::default();
    let dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Sq, ROWS, false).await;
    let margins = [0.0f32, 0.05, 0.15, 0.4];
    let mut totals = [0; 4];
    let mut grew = 0;
    for query in queries() {
        let mut counts = [0; 4];
        for (count, margin) in counts.iter_mut().zip(margins) {
            let (_, found) = search(&dataset, &query, 10, None, |scan| {
                scan.ef(400).stop_margin(margin).unwrap();
            })
            .await;
            *count = found.index_comparisons;
        }
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += count;
        }
        grew += usize::from(counts[3] > counts[0]);
    }
    assert!(
        totals.windows(2).all(|pair| pair[0] <= pair[1]),
        "margins {margins:?} measured {totals:?} distances, and a wider one fewer"
    );
    assert!(
        grew > 0,
        "no query measured more at a margin of 0.4 than at 0"
    );
}

/// Deleted rows reach the partitions as a prefilter, and a margin still stops
/// the walks through it: fewer distances than the same cap without one, and
/// only live rows among the answers.
#[tokio::test]
async fn a_margin_stops_the_walks_through_deleted_rows() {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Sq, ROWS, false).await;
    dataset.delete("id % 7 = 0").await.unwrap();
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().deletion_file.is_some())
    );
    let live = ids_by_row_id(&dataset).await;
    for query in queries() {
        let capped = |margin: Option<f32>| {
            move |scan: &mut Scanner| {
                scan.ef(200);
                if let Some(margin) = margin {
                    scan.stop_margin(margin).unwrap();
                }
            }
        };
        let (found, ruled) = search(&dataset, &query, 10, None, capped(Some(0.05))).await;
        let (_, unruled) = search(&dataset, &query, 10, None, capped(None)).await;
        assert!(
            ruled.index_comparisons < unruled.index_comparisons,
            "a margin of 0.05 measured {} distances, the cap without it {}",
            ruled.index_comparisons,
            unruled.index_comparisons
        );
        assert_eq!(found.len(), 10);
        assert!(
            found.iter().all(|(row_id, _)| live.contains_key(row_id)),
            "{found:?}"
        );
    }
}

/// Queries in flight together answer exactly as the same queries one at a
/// time: the same rows at the same distances, to the bit, from 24, 12 and 4
/// tasks on a runtime of four threads. Each round opens the dataset afresh,
/// so that its first queries race to load the index into an empty cache, and
/// the most tasks go first, onto data files no query has opened yet. The ten
/// queries run three times over, so that one query is often in flight twice.
/// Over full vectors, scalar codes scored again from their byte offsets,
/// product codes scored again through take, and walks a margin stops, with
/// and without deleted rows, which reach every walk as a prefilter. Half the
/// rows keep each case within a second.
#[rstest]
#[case::full_vectors(Codes::Flat, DIM, None)]
#[case::scalar_codes_scored_again_from_their_offsets(Codes::Sq, WIDE, None)]
#[case::product_codes_scored_again_through_take(Codes::Pq, PQ_DIM, None)]
#[case::walks_a_margin_stops(Codes::Sq, WIDE, Some(0.05))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queries_in_flight_together_answer_as_one_at_a_time(
    #[case] codes: Codes,
    #[case] dimension: u32,
    #[case] stop_margin: Option<f32>,
    #[values(false, true)] deleted: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), ROWS / 2, dimension, false).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(codes, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    if deleted {
        dataset.delete("id % 7 = 0").await.unwrap();
    }
    let queries = Arc::new(
        queries_of(dimension)
            .iter()
            .cycle()
            .take(30)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let mut rounds = Vec::new();
    for tasks in [24, 12, 4, 1] {
        let dataset = DatasetBuilder::from_uri(dir.as_str()).load().await.unwrap();
        let (answers, most_in_flight) =
            answers_together(Arc::new(dataset), queries.clone(), tasks, stop_margin).await;
        rounds.push((tasks, answers, most_in_flight));
    }
    let (_, one_at_a_time, _) = rounds.pop().unwrap();
    assert!(
        one_at_a_time.iter().all(|found| found.len() == 10),
        "{one_at_a_time:?}"
    );
    for (tasks, answers, most_in_flight) in rounds {
        assert!(
            most_in_flight > 1,
            "{tasks} tasks never had two queries in flight"
        );
        for (query, (found, expected)) in answers.iter().zip(&one_at_a_time).enumerate() {
            assert_eq!(found, expected, "query {query} from {tasks} tasks");
        }
    }
}

/// The answers to `queries` from `tasks` tasks, started together, each taking
/// the next query as it finishes its last, with the distances as their bits;
/// and the most queries that were in flight at once.
async fn answers_together(
    dataset: Arc<Dataset>,
    queries: Arc<Vec<Float32Array>>,
    tasks: usize,
    stop_margin: Option<f32>,
) -> (Vec<Vec<(u64, u32)>>, usize) {
    let next = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let most_in_flight = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(tokio::sync::Barrier::new(tasks));
    let handles = (0..tasks)
        .map(|_| {
            let dataset = dataset.clone();
            let queries = queries.clone();
            let next = next.clone();
            let in_flight = in_flight.clone();
            let most_in_flight = most_in_flight.clone();
            let start = start.clone();
            tokio::spawn(async move {
                start.wait().await;
                let mut answers = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(query) = queries.get(index) else {
                        break answers;
                    };
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    most_in_flight.fetch_max(now, Ordering::SeqCst);
                    let (found, _) = search(&dataset, query, 10, None, |scan| {
                        if let Some(margin) = stop_margin {
                            scan.ef(400).stop_margin(margin).unwrap();
                        }
                    })
                    .await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    let found = found
                        .into_iter()
                        .map(|(row_id, distance)| (row_id, distance.to_bits()))
                        .collect::<Vec<_>>();
                    answers.push((index, found));
                }
            })
        })
        .collect::<Vec<_>>();
    let mut answers = vec![Vec::new(); queries.len()];
    for handle in handles {
        for (index, found) in handle.await.unwrap() {
            answers[index] = found;
        }
    }
    (answers, most_in_flight.load(Ordering::SeqCst))
}

/// A margin no walk can stop by is refused as invalid input when it is set,
/// and so is one set before the query it would stop; zero is a margin. A
/// vector query given as a filter is the caller's own, so its margin, and
/// the rank that must be its k, are checked when the scan is planned: first,
/// before this filter-only scan is refused for wanting a full-text query.
#[tokio::test]
async fn a_margin_no_walk_can_stop_by_is_refused() {
    let dir = TempStrDir::default();
    let dataset = two_fragments(dir.as_str()).await;
    let key = queries().swap_remove(0);
    let refused = |error: Error, reason: &str| {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains(reason), "{error}");
    };
    let mut scan = dataset.scan();
    refused(
        scan.stop_margin(0.1).err().unwrap(),
        "stop_margin requires nearest to be called first",
    );
    scan.nearest("vec", &key, 10).unwrap();
    for margin in [-0.1, f32::NAN, f32::INFINITY, 1e20] {
        refused(scan.stop_margin(margin).err().unwrap(), "finite fraction");
    }
    scan.stop_margin(0.0).unwrap();

    let mut filter = Query {
        column: "vec".to_string(),
        key: Arc::new(key),
        k: 10,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 1,
        maximum_nprobes: None,
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: None,
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
        adaptive_stop: None,
    };
    for (margin, rank, reason) in [
        (-0.1, 10, "finite fraction"),
        (0.05, 7, "carries rank 7 for a query of k 10"),
    ] {
        filter.adaptive_stop = Some(AdaptiveStop { margin, rank });
        let error = dataset
            .scan()
            .filter_query(QueryFilter::Vector(filter.clone()))
            .unwrap()
            .create_plan()
            .await
            .unwrap_err();
        refused(error, reason);
    }
}

/// A range query keeps the rows whose exact distances lie within its bounds,
/// as brute force does, whatever the units the codes measure in: a range
/// from the fifth nearest row up to the twenty-fifth, which the lower bound
/// limits, and one up to the ninth, which holds fewer rows than are asked
/// for and so is limited by its upper bound. Each bound lies halfway between
/// two neighbouring rows, so that only the codes, and not the rounding of one
/// distance against another, put a row on the wrong side of it. A refined
/// query, here as wide as the index's own re-score, hands the index its bounds
/// on the codes' scale already, an unrefined one at exact distances.
#[rstest]
#[case::scalar_codes(Codes::Sq, DistanceType::L2)]
#[case::scalar_codes_under_cosine(Codes::Sq, DistanceType::Cosine)]
#[case::product_codes(Codes::Pq, DistanceType::L2)]
#[case::product_codes_under_cosine(Codes::Pq, DistanceType::Cosine)]
#[case::flat_codes_under_cosine(Codes::Flat, DistanceType::Cosine)]
#[tokio::test]
async fn a_range_query_keeps_the_rows_within_its_bounds(
    #[case] codes: Codes,
    #[case] distance_type: DistanceType,
    #[values(None, Some(2))] refine_factor: Option<u32>,
) {
    let dir = TempStrDir::default();
    let dataset = indexed(dir.as_str(), codes, distance_type).await;
    let (mut found_in_range, mut in_range) = (0, 0);
    for query in queries_of(codes.dimension()) {
        let exact = nearest(&dataset, &query, ROWS as usize, Some(distance_type)).await;
        let between = |rank: usize| (exact[rank - 1].1 + exact[rank].1) / 2.0;
        for (lower, upper) in [
            (Some(between(4)), Some(between(24))),
            (None, Some(between(8))),
        ] {
            let within = |distance: f32| {
                lower.is_none_or(|lower| lower <= distance)
                    && upper.is_none_or(|upper| distance < upper)
            };
            let (found, _) = search(&dataset, &query, 10, None, |scan| {
                scan.distance_range(lower, upper);
                if let Some(factor) = refine_factor {
                    scan.refine(factor);
                }
            })
            .await;
            assert_exact(&found, &exact);
            assert!(
                found.iter().all(|(_, distance)| within(*distance)),
                "{found:?} not within [{lower:?}, {upper:?})"
            );
            let found = found
                .into_iter()
                .map(|(row_id, _)| row_id)
                .collect::<HashSet<_>>();
            let expected = exact
                .iter()
                .filter(|(_, distance)| within(*distance))
                .take(10)
                .collect::<Vec<_>>();
            in_range += expected.len();
            found_in_range += expected
                .iter()
                .filter(|(row_id, _)| found.contains(row_id))
                .count();
        }
    }
    // The walk sees the codes' distances, and product codes misplace rows
    // across a bound even halfway to their neighbours.
    let bar = match codes {
        Codes::Flat | Codes::Sq => 0.95,
        Codes::Pq => 0.85,
    };
    let recall = found_in_range as f32 / in_range as f32;
    assert!(recall >= bar, "recall {recall}");
}

/// Deleted rows stay out of the results, and rows appended after the build,
/// which brute force searches beside the index, rank among the indexed ones
/// at exact distances, with row ids stable or not, and whether the deletions
/// are still files or a compaction has rewritten them away. A compaction does
/// not rewrite an index of stable row ids, so that index still holds the ids
/// of the rows it deleted, and while the index covers every fragment no
/// deletion mask is built to block them.
#[rstest]
#[tokio::test]
async fn deleted_and_appended_rows_rank_with_the_indexed_ones(
    #[values(false, true)] enable_stable_row_ids: bool,
    #[values(false, true)] compacted: bool,
    #[values(DIM, WIDE)] dimension: u32,
) {
    let dir = TempStrDir::default();
    // Half the rows keep a case of wide vectors within a second.
    let rows = if dimension == WIDE { ROWS / 2 } else { ROWS };
    let mut dataset =
        two_fragments_of_rows(dir.as_str(), rows, dimension, enable_stable_row_ids).await;
    let queries = queries_of(dimension);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    dataset.delete("id % 7 = 0").await.unwrap();
    if compacted {
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert!(
            dataset
                .get_fragments()
                .iter()
                .all(|fragment| fragment.metadata().deletion_file.is_none())
        );
    }
    let indexed_only = recall_over(&dataset, &queries, 10, DistanceType::L2).await;
    assert!(indexed_only >= 0.95, "recall {indexed_only}");
    let appended = gen_batch()
        .with_seed(Seed::from(9))
        .col("id", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(dimension.into()))
        .into_reader_rows(RowCount::from(rows / 4), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), if compacted { 2 } else { 3 });
    let recall = recall_over(&dataset, &queries, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A compaction that leaves the index to be remapped later leaves each
/// partition fewer rows than its graph has vertices, so that its rows are
/// scored directly, at exact distances still.
#[tokio::test]
async fn a_deferred_remap_scores_the_partitions_directly() {
    let dir = TempStrDir::default();
    let mut dataset = indexed(dir.as_str(), Codes::Sq, DistanceType::L2).await;
    dataset.delete("id % 7 = 0").await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        indices
            .iter()
            .any(|index| index.name == FRAG_REUSE_INDEX_NAME),
        "{indices:?}"
    );
    // Scored directly, each partition compares every row it holds.
    let live_rows = dataset.count_rows(None).await.unwrap();
    let (_, counts) = search(&dataset, &queries()[0], 10, None, |_| {}).await;
    assert_eq!(counts.index_comparisons, live_rows);
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Vectors of half and double precision are walked and scored again as
/// single precision ones are, at the distances brute force measures for them.
#[rstest]
#[case::half(DataType::Float16)]
#[case::double(DataType::Float64)]
#[tokio::test]
async fn vectors_of_every_float_width_answer_at_exact_distances(#[case] element_type: DataType) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_cast(dir.as_str(), element_type).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A query over two segments of the index, the second built over appended
/// rows, opens and searches each as an IVF_VAMANA_* index.
#[tokio::test]
async fn a_query_over_two_segments_answers_at_exact_distances() {
    let dir = TempStrDir::default();
    let mut dataset = indexed(dir.as_str(), Codes::Sq, DistanceType::L2).await;
    let appended = gen_batch()
        .with_seed(Seed::from(9))
        .col("id", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(DIM.into()))
        .into_reader_rows(RowCount::from(ROWS / 4), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(dataset.load_indices_by_name(INDEX).await.unwrap().len(), 2);
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// The definition recorded without training trains as IVF_VAMANA_* once
/// `optimize_indices` finds rows for it.
#[tokio::test]
async fn a_definition_trains_when_optimized() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let params = params(Codes::Sq, DistanceType::L2);
    dataset
        .create_index_builder(&["vec"], IndexType::Vector, &params)
        .name(INDEX.into())
        .train(false)
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices[0].index_version, IVF_VAMANA_INDEX_VERSION as i32);
    assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].index_version, IVF_VAMANA_INDEX_VERSION as i32);
    assert_eq!(indices[0].fragment_bitmap.as_ref().unwrap().len(), 2);
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    assert_eq!(statistics["index_type"], "IVF_VAMANA_SQ");
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Writes `ROWS / 2` vectors of `dimension` values drawn from `seed` as one
/// fragment, each null with probability `null_share`; returns the dataset and
/// how many are not null.
async fn write_with_nulls(
    uri: &str,
    mode: WriteMode,
    seed: u64,
    null_share: f64,
    dimension: u32,
) -> (Dataset, usize) {
    let batch = gen_batch()
        .with_seed(Seed::from(seed))
        .col(
            "vec",
            array::rand_vec::<Float32Type>(dimension.into()).with_random_nulls(null_share),
        )
        .into_batch_rows(RowCount::from(ROWS / 2))
        .unwrap();
    let non_null = batch.num_rows() - batch["vec"].logical_null_count();
    let schema = batch.schema();
    let params = WriteParams {
        mode,
        ..Default::default()
    };
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(params),
    )
    .await
    .unwrap();
    (dataset, non_null)
}

/// The index's partitions, read through the quantizer `codes` name, hold
/// every non-null vector and nothing else, and the index finds the nearest of
/// them.
async fn assert_indexes_the_non_null_vectors(dataset: &Dataset, non_null: usize, codes: Codes) {
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    let indexed = statistics["indices"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|segment| segment["partitions"].as_array().unwrap())
        .map(|partition| partition["size"].as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(indexed, non_null as u64);
    // Counted in the graphs too: a search scores a partition whose graph and
    // storage differ in length directly, so recall would not tell.
    let vertices = partitions(dataset, codes)
        .await
        .iter()
        .map(|partition| partition.graph.graph().row_ids().len())
        .sum::<usize>();
    assert_eq!(vertices, non_null);
    // Among no vectors at all there is nothing to find.
    if non_null > 0 {
        let recall = recall(dataset, 10, DistanceType::L2).await;
        assert!(recall >= 0.95, "recall {recall}");
    }
}

/// Rows without a vector stay out of the graph: in a build over a half-null
/// fragment and an all-null one, and in the merges of `optimize_indices`
/// that bring more of them, among vectors or alone.
#[tokio::test]
async fn null_vectors_stay_out_of_the_graph() {
    let dir = TempStrDir::default();
    let (_, mut non_null) = write_with_nulls(dir.as_str(), WriteMode::Create, 1, 0.5, DIM).await;
    assert!(0 < non_null && non_null < ROWS as usize / 2, "{non_null}");
    let (mut dataset, none) = write_with_nulls(dir.as_str(), WriteMode::Append, 2, 1.0, DIM).await;
    assert_eq!(none, 0);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    assert_indexes_the_non_null_vectors(&dataset, non_null, Codes::Sq).await;

    for (seed, null_share) in [(3, 0.5), (4, 1.0)] {
        let (appended, added) =
            write_with_nulls(dir.as_str(), WriteMode::Append, seed, null_share, DIM).await;
        dataset = appended;
        non_null += added;
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
        assert_indexes_the_non_null_vectors(&dataset, non_null, Codes::Sq).await;
    }
}

/// A page that holds a null vector puts a control word in front of every
/// vector, and a re-score reads the vectors beside the nulls at their byte
/// offsets: of wide vectors, half of them null, only the null ones are read
/// through take, and the index finds the nearest of the others at their
/// exact distances.
#[tokio::test]
async fn vectors_beside_nulls_are_rescored_from_their_offsets() {
    let dir = TempStrDir::default();
    let (mut dataset, non_null) =
        write_with_nulls(dir.as_str(), WriteMode::Create, 1, 0.5, WIDE).await;
    assert!(0 < non_null && non_null < ROWS as usize / 2, "{non_null}");
    let every_row = (0..ROWS / 2).collect::<Vec<_>>();
    let read = read_vectors(&Arc::new(dataset.clone()), "vec", &every_row, None)
        .await
        .unwrap();
    assert_eq!(read.through_take, ROWS as usize / 2 - non_null);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    assert_indexes_the_non_null_vectors(&dataset, non_null, Codes::Sq).await;
}

/// [`two_fragments_of_rows`] indexed under [`diagonal_params`].
async fn indexed_along_the_diagonal(
    uri: &str,
    codes: Codes,
    rows: u64,
    enable_stable_row_ids: bool,
) -> Dataset {
    let mut dataset =
        two_fragments_of_rows(uri, rows, codes.dimension(), enable_stable_row_ids).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &diagonal_params(codes),
            true,
        )
        .await
        .unwrap();
    dataset
}

/// Appends `rows` random vectors drawn from `seed` as one fragment.
async fn append_vectors(dataset: &mut Dataset, rows: u64, seed: u64) {
    let appended = gen_batch()
        .with_seed(Seed::from(seed))
        .col("id", array::step::<Int32Type>())
        .col(
            "vec",
            array::rand_vec::<Float32Type>(dimension_of(dataset).into()),
        )
        .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
}

/// A partition of a segment of the index, as its files hold it.
struct Partition {
    graph: Vamana,
    /// The row ids of the partition's storage, in order.
    row_ids: Vec<u64>,
}

impl Partition {
    /// Each vertex's out-edges, by row id.
    fn edges(&self) -> HashMap<u64, HashSet<u64>> {
        let graph = self.graph.graph();
        let row_ids = graph.row_ids();
        (0..row_ids.len() as u32)
            .map(|vertex| {
                let neighbors = graph
                    .neighbors(vertex)
                    .iter()
                    .map(|neighbor| row_ids[*neighbor as usize])
                    .collect();
                (row_ids[vertex as usize], neighbors)
            })
            .collect()
    }

    /// Whether the graph is `other`: the same vertices, out-edges, medoid and
    /// entry points.
    fn is(&self, other: &Vamana) -> bool {
        let (graph, other_graph) = (self.graph.graph(), other.graph());
        self.graph.metadata() == other.metadata()
            && graph.row_ids() == other_graph.row_ids()
            && (0..graph.row_ids().len() as u32)
                .all(|vertex| graph.neighbors(vertex) == other_graph.neighbors(vertex))
    }

    /// Whether the graph extends `old`'s, as inserting rows does and a
    /// rebuild does not: a vertex of `old` gains out-edges only to rows `old`
    /// does not hold, though the back-edges of new rows may prune some of its
    /// own. It holds while no vertex is cut off, since one linked again draws
    /// its list anew. The `replaced` rows left their vertices and came back as
    /// new rows, so that their lists and the lists that pointed at them are
    /// drawn again.
    fn extends(&self, old: &Self, replaced: &HashSet<u64>) -> bool {
        let (old_edges, edges) = (old.edges(), self.edges());
        let is_new = |row_id: &u64| replaced.contains(row_id) || !old_edges.contains_key(row_id);
        old_edges
            .iter()
            .filter(|(row_id, neighbors)| {
                !replaced.contains(*row_id) && neighbors.is_disjoint(replaced)
            })
            .all(|(row_id, neighbors)| {
                edges.get(row_id).is_some_and(|merged| {
                    merged
                        .iter()
                        .all(|neighbor| neighbors.contains(neighbor) || is_new(neighbor))
                })
            })
    }
}

/// The partitions of the index's one segment, read through the quantizer
/// `codes` name.
async fn partitions(dataset: &Dataset, codes: Codes) -> Vec<Partition> {
    partitions_in(open_segment(dataset).await.as_ref(), codes).await
}

/// The partitions of the segment `index`, read through the quantizer `codes`
/// name.
async fn partitions_in(index: &dyn VectorIndex, codes: Codes) -> Vec<Partition> {
    match codes {
        Codes::Flat => partitions_of::<FlatQuantizer>(index).await,
        Codes::Sq => partitions_of::<ScalarQuantizer>(index).await,
        Codes::Pq => partitions_of::<ProductQuantizer>(index).await,
    }
}

/// The graphs builds from scratch draw over the storage of each partition of
/// the index's one segment, read through the quantizer `codes` name, with
/// the parameters [`vamana`] gives.
async fn drawn_afresh(dataset: &Dataset, codes: Codes) -> Vec<Vamana> {
    drawn_afresh_in(open_segment(dataset).await.as_ref(), codes, &vamana()).await
}

/// [`drawn_afresh`] over the partitions of the segment `index`, with the
/// build parameters `params`.
async fn drawn_afresh_in(
    index: &dyn VectorIndex,
    codes: Codes,
    params: &VamanaBuildParams,
) -> Vec<Vamana> {
    match codes {
        Codes::Flat => drawn_afresh_over::<FlatQuantizer>(index, params).await,
        Codes::Sq => drawn_afresh_over::<ScalarQuantizer>(index, params).await,
        Codes::Pq => drawn_afresh_over::<ProductQuantizer>(index, params).await,
    }
}

/// The index's one segment, opened.
async fn open_segment(dataset: &Dataset) -> Arc<dyn VectorIndex> {
    let segments = dataset.load_indices_by_name(INDEX).await.unwrap();
    assert_eq!(segments.len(), 1);
    dataset
        .open_vector_index("vec", &segments[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap()
}

async fn partitions_of<Q: Quantization + 'static>(index: &dyn VectorIndex) -> Vec<Partition> {
    let ivf = index
        .as_any()
        .downcast_ref::<IVFIndex<Vamana, Q>>()
        .unwrap();
    let num_partitions = ivf.ivf_model().num_partitions();
    let mut partitions = Vec::with_capacity(num_partitions);
    for partition in 0..num_partitions {
        let batch = ivf
            .read_sub_index_batch(partition, None, None)
            .await
            .unwrap();
        let storage = ivf.load_partition_storage(partition, None).await.unwrap();
        let graph = Vamana::load(batch).unwrap();
        let row_ids = storage.row_ids().copied().collect::<Vec<_>>();
        // A search walks vertex i over the row at storage position i.
        assert_eq!(graph.graph().row_ids(), row_ids.as_slice());
        partitions.push(Partition { graph, row_ids });
    }
    partitions
}

async fn drawn_afresh_over<Q: Quantization + 'static>(
    index: &dyn VectorIndex,
    params: &VamanaBuildParams,
) -> Vec<Vamana> {
    let ivf = index
        .as_any()
        .downcast_ref::<IVFIndex<Vamana, Q>>()
        .unwrap();
    let mut storages = Vec::with_capacity(ivf.ivf_model().num_partitions());
    for partition in 0..ivf.ivf_model().num_partitions() {
        storages.push(ivf.load_partition_storage(partition, None).await.unwrap());
    }
    // One partition a thread, as an index build draws them.
    std::thread::scope(|scope| {
        let builds = storages
            .iter()
            .map(|storage| scope.spawn(|| Vamana::index_vectors(storage, params.clone()).unwrap()))
            .collect::<Vec<_>>();
        builds
            .into_iter()
            .map(|build| build.join().unwrap())
            .collect()
    })
}

/// Each row's `id` by its row id.
async fn ids_by_row_id(dataset: &Dataset) -> HashMap<u64, i32> {
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
    let ids = batch["id"].as_primitive::<Int32Type>().values();
    row_ids.iter().copied().zip(ids.iter().copied()).collect()
}

/// A merge of new rows into the index extends each partition's graph rather
/// than drawing it anew: every old vertex keeps its out-edges, gaining only
/// edges to the new rows, under every kind of codes.
#[rstest]
#[case::flat(Codes::Flat, ROWS)]
#[case::scalar(Codes::Sq, ROWS)]
// A graph over product codes draws more slowly in a debug build: half the
// rows.
#[case::product(Codes::Pq, ROWS / 2)]
#[tokio::test]
async fn a_merge_extends_each_partitions_graph(#[case] codes: Codes, #[case] rows: u64) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), codes, rows, false).await;
    let old = partitions(&dataset, codes).await;
    append_vectors(&mut dataset, rows / 4, 9).await;
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();

    let merged = partitions(&dataset, codes).await;
    let fresh = drawn_afresh(&dataset, codes).await;
    assert_eq!(merged.len(), old.len());
    for ((old, merged), fresh) in old.iter().zip(&merged).zip(&fresh) {
        // Each partition opens with its old rows and gained new ones.
        assert!(!old.row_ids.is_empty());
        assert!(merged.row_ids.starts_with(&old.row_ids));
        assert!(merged.row_ids.len() > old.row_ids.len());
        assert!(merged.extends(old, &HashSet::new()));
        assert!(!merged.is(fresh));
    }
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A merge of two segments extends the graphs of the first, whose rows open
/// each merged partition; the second segment's rows go in with the new ones.
#[tokio::test]
async fn a_merge_of_two_segments_extends_the_first_ones_graphs() {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, ROWS, false).await;
    let old = partitions(&dataset, Codes::Flat).await;
    append_vectors(&mut dataset, ROWS / 8, 9).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(dataset.load_indices_by_name(INDEX).await.unwrap().len(), 2);
    append_vectors(&mut dataset, ROWS / 8, 10).await;
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();

    let merged = partitions(&dataset, Codes::Flat).await;
    assert_eq!(merged.len(), old.len());
    for (old, merged) in old.iter().zip(&merged) {
        assert!(!old.row_ids.is_empty());
        assert!(merged.row_ids.starts_with(&old.row_ids));
        assert!(merged.row_ids.len() > old.row_ids.len());
        assert!(merged.extends(old, &HashSet::new()));
    }
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// The vertices of a partition's graph a test rewrites the rows of.
#[derive(Clone, Copy, Debug)]
enum Ends {
    First,
    Last,
}

/// Rows rewritten under stable row ids come back as new rows under their
/// old ids, with new vectors. The merge inserts them anew rather than give
/// them the edges of their old vertices, whether those opened the
/// partition's graph or closed it, and extends the graph over the rest.
#[rstest]
#[case::first(Ends::First)]
#[case::last(Ends::Last)]
#[tokio::test]
async fn rows_rewritten_under_stable_ids_are_inserted_anew(#[case] ends: Ends) {
    const REWRITTEN: usize = 5;
    let dir = TempStrDir::default();
    let dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, ROWS, true).await;
    // Stable row ids number the rows as `id` does, so `id` names them.
    let ids = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    assert!(
        ids["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .zip(ids[ROW_ID].as_primitive::<UInt64Type>().values())
            .all(|(id, row_id)| *id as u64 == *row_id)
    );
    let old = partitions(&dataset, Codes::Flat).await;
    let rows = &old[1].row_ids;
    let rewritten = match ends {
        Ends::First => rows[..REWRITTEN].to_vec(),
        Ends::Last => rows[rows.len() - REWRITTEN..].to_vec(),
    };
    // Further out along the diagonal than the second centroid: nearer it
    // than the first, and far from every other row.
    let far = DIAGONAL[1] + 10.0 * (DIAGONAL[1] - DIAGONAL[0]);
    let literal = format!("array[{}]", vec![far.to_string(); DIM as usize].join(", "));
    let names = rewritten
        .iter()
        .map(|row_id| row_id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut dataset = UpdateBuilder::new(Arc::new(dataset))
        .update_where(&format!("id IN ({names})"))
        .unwrap()
        .set("vec", &literal)
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset
        .as_ref()
        .clone();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();

    let merged = partitions(&dataset, Codes::Flat).await;
    let replaced = rewritten.iter().copied().collect::<HashSet<_>>();
    // The rewritten rows come back after the rest of the partition, in the
    // order of their old vertices: a vertex matched by row id past the old
    // rows would be matched with its own rewritten row.
    let expected = rows
        .iter()
        .filter(|row_id| !replaced.contains(row_id))
        .chain(&rewritten)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(merged[1].row_ids, expected);
    assert_eq!(merged[0].row_ids, old[0].row_ids);
    assert!(merged[1].extends(&old[1], &replaced));
    let (old_edges, edges) = (old[1].edges(), merged[1].edges());
    for row_id in &rewritten {
        assert_ne!(edges[row_id], old_edges[row_id], "row {row_id}");
    }
    let query = Float32Array::from(vec![far; DIM as usize]);
    let found = nearest(&dataset, &query, REWRITTEN, None).await;
    assert_eq!(
        found
            .iter()
            .map(|(row_id, _)| *row_id)
            .collect::<HashSet<_>>(),
        replaced
    );
    assert!(
        found.iter().all(|(_, distance)| *distance == 0.0),
        "{found:?}"
    );
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A merge after a compaction that left the index to be remapped later finds
/// every row of the old graphs renamed, the compaction having rewritten both
/// fragments, and builds each partition anew, over every live row once.
#[tokio::test]
async fn a_merge_after_a_deferred_remap_builds_each_partition_anew() {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, ROWS, false).await;
    dataset.delete("id % 7 = 0").await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.id() > 1)
    );
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        indices
            .iter()
            .any(|index| index.name == FRAG_REUSE_INDEX_NAME),
        "{indices:?}"
    );
    append_vectors(&mut dataset, ROWS / 4, 9).await;
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();

    let merged = partitions(&dataset, Codes::Flat).await;
    let fresh = drawn_afresh(&dataset, Codes::Flat).await;
    assert!(
        merged
            .iter()
            .zip(&fresh)
            .all(|(merged, fresh)| merged.is(fresh))
    );
    let indexed = merged
        .iter()
        .map(|partition| partition.row_ids.len())
        .sum::<usize>();
    assert_eq!(indexed, dataset.count_rows(None).await.unwrap());
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// How many of `requests` went to the file `name` of the segment `segment`.
fn requests_to(requests: &[IoRequestRecord], segment: &Uuid, name: &str) -> usize {
    let path = format!("{segment}/{name}");
    requests
        .iter()
        .filter(|request| request.path.as_ref().ends_with(&path))
        .count()
}

/// An optimize that grows each partition more than three times over builds it
/// anew without reading its old graph, while one that extends the graphs
/// reads them. The segment is opened first, so that all the merge reads of its
/// index file is graphs.
#[rstest]
#[case::extended(ROWS / 32, true)]
#[case::grown(ROWS / 2, false)]
#[tokio::test]
async fn a_merge_reads_the_old_graphs_only_to_extend_them(
    #[case] appended: u64,
    #[case] extended: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, ROWS / 8, false).await;
    let old = partitions(&dataset, Codes::Flat).await;
    let segment = dataset.load_indices_by_name(INDEX).await.unwrap()[0].uuid;
    append_vectors(&mut dataset, appended, 9).await;
    let _ = dataset.object_store.io_stats_incremental();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();
    let requests = dataset.object_store.io_stats_incremental().requests;
    // The old rows came out of the segment's storage, through the count.
    assert!(requests_to(&requests, &segment, INDEX_AUXILIARY_FILE_NAME) > 0);
    let graph_reads = requests_to(&requests, &segment, INDEX_FILE_NAME);

    let merged = partitions(&dataset, Codes::Flat).await;
    assert_eq!(merged.len(), old.len());
    for (old, merged) in old.iter().zip(&merged) {
        assert_eq!(
            merged.row_ids.len() <= 3 * old.row_ids.len(),
            extended,
            "{} rows merged into {}",
            old.row_ids.len(),
            merged.row_ids.len()
        );
    }
    assert_eq!(
        graph_reads > 0,
        extended,
        "{graph_reads} reads of the old graphs"
    );
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A merge reads no old graph it would build the partition anew over. That
/// the partition grew more than three times over, or shrank below half its
/// graph, tells it before the graph is read. That the merge keeps fewer than
/// half the graph's rows only the graph's row ids tell, and the graph then
/// stays out of the build rather than held through it.
#[tokio::test]
async fn a_merge_reads_no_graph_it_would_build_anew() {
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, ROWS, true).await;
    let index = open_segment(&dataset).await;
    let row_ids = partitions_in(index.as_ref(), Codes::Flat)
        .await
        .swap_remove(0)
        .row_ids;
    let vertices = row_ids.len();
    assert_eq!(index.ivf_model().partition_size(0), vertices);
    let store = dataset.object_store.clone();
    // Whether the merge hands the graph of partition 0 on, and the reads it
    // took.
    let base = async |source: &ExistingIndex, storage_rows: usize| {
        let _ = store.io_stats_incremental();
        let base =
            merge_base::<Vamana, FlatQuantizer>(std::slice::from_ref(source), 0, storage_rows)
                .await
                .unwrap();
        (base.is_some(), store.io_stats_incremental().read_iops)
    };

    let unfiltered = ExistingIndex::unfiltered(index.clone());
    // Extended, with at most twice as many new rows as old ones and at least
    // half the graph's rows: the graph is read, and the count sees it.
    for rows in [vertices + 1, 3 * vertices, vertices.div_ceil(2)] {
        let (handed_on, reads) = base(&unfiltered, rows).await;
        assert!(
            handed_on && reads > 0,
            "{rows} rows: {handed_on}, {reads} reads"
        );
    }
    for rows in [3 * vertices + 1, vertices.div_ceil(2) - 1] {
        assert_eq!(base(&unfiltered, rows).await, (false, 0), "{rows} rows");
    }

    // Three rows in four deleted under stable row ids, which the filter of
    // the segment's rows then drops.
    dataset.delete("id % 4 != 0").await.unwrap();
    let live = ids_by_row_id(&dataset).await;
    let surviving = row_ids
        .iter()
        .filter(|row_id| live.contains_key(row_id))
        .count();
    assert!(
        surviving > 0 && 2 * surviving < vertices,
        "{surviving} of {vertices}"
    );
    let filtered = ExistingIndex::with_coverage(
        index.clone(),
        dataset.clone(),
        RoaringBitmap::from_iter([0, 1]),
        RoaringBitmap::new(),
    );
    // With no new rows the storage alone is below half the graph; with as
    // many new rows as deleted ones only the row ids the filter keeps tell.
    assert_eq!(base(&filtered, surviving).await, (false, 0));
    let (handed_on, reads) = base(&filtered, vertices).await;
    assert!(!handed_on && reads > 0, "{handed_on}, {reads} reads");
}

/// Other merges stay as they were: an IVF_HNSW_* merge builds every
/// partition anew from the old segment's rows and reads none of its graphs.
#[tokio::test]
async fn a_merge_of_hnsw_graphs_reads_none_of_them() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), ROWS / 8, DIM, false).await;
    let mut params =
        VectorIndexParams::ivf_hnsw(DistanceType::L2, ivf(), HnswBuildParams::default());
    params.stages[0] = diagonal_params(Codes::Flat).stages.swap_remove(0);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    // Opened first, as in the Vamana cases, so that the merge has no reason
    // to read the index file but its graphs.
    open_segment(&dataset).await;
    let segment = dataset.load_indices_by_name(INDEX).await.unwrap()[0].uuid;
    append_vectors(&mut dataset, ROWS / 32, 9).await;
    let _ = dataset.object_store.io_stats_incremental();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();
    let requests = dataset.object_store.io_stats_incremental().requests;

    let merged = dataset.load_indices_by_name(INDEX).await.unwrap();
    assert_eq!(merged.len(), 1);
    assert_ne!(merged[0].uuid, segment);
    assert!(requests_to(&requests, &segment, INDEX_AUXILIARY_FILE_NAME) > 0);
    assert_eq!(requests_to(&requests, &segment, INDEX_FILE_NAME), 0);
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// `optimize_indices` with no merge asked for joins partitions far below the
/// target size, the smallest first. The partition it keeps, the second here,
/// extends its graph with the other's rows and the new ones, though the merged
/// index numbers it first.
#[tokio::test]
async fn a_join_extends_the_graph_of_the_partition_it_keeps() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let mut params = diagonal_params(Codes::Flat);
    let StageParams::Ivf(ivf) = &mut params.stages[0] else {
        unreachable!()
    };
    // Centroids alone would record the size they imply, 300 rows a partition,
    // which neither partition falls far below.
    ivf.target_partition_size = Some(1 << 20);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let old = partitions(&dataset, Codes::Flat).await;
    // New rows from the half of the cube nearer the second centroid all go to
    // its partition, which leaves the first the smaller.
    let batch = gen_batch()
        .with_seed(Seed::from(9))
        .col("id", array::step_custom::<Int32Type>(ROWS as i32, 1))
        .col("vec", array::rand_vec::<Float32Type>(DIM.into()))
        .into_batch_rows(RowCount::from(ROWS / 4))
        .unwrap();
    let values = batch["vec"].as_fixed_size_list().values();
    let upper = values
        .as_primitive::<Float32Type>()
        .values()
        .iter()
        .map(|value| 0.5 + value / 2.0);
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from_iter_values(upper), DIM as i32)
            .unwrap();
    let schema = batch.schema();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![batch["id"].clone(), Arc::new(vectors)]).unwrap();
    dataset
        .append(RecordBatchIterator::new([Ok(batch)], schema), None)
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let merged = partitions(&dataset, Codes::Flat).await;
    assert_eq!(merged.len(), 1);
    assert!(!old[1].row_ids.is_empty());
    assert!(merged[0].row_ids.starts_with(&old[1].row_ids));
    assert!(merged[0].extends(&old[1], &HashSet::new()));
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// `optimize_indices` with no merge asked for splits partitions above four
/// times the target size. It reads the rows of every partition the split
/// touches again, from the dataset, and builds their graphs anew.
#[tokio::test]
async fn a_split_builds_the_partitions_it_touches_anew() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let mut params = diagonal_params(Codes::Flat);
    let StageParams::Ivf(ivf) = &mut params.stages[0] else {
        unreachable!()
    };
    ivf.target_partition_size = Some(64);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    append_vectors(&mut dataset, ROWS / 4, 9).await;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let merged = partitions(&dataset, Codes::Flat).await;
    let fresh = drawn_afresh(&dataset, Codes::Flat).await;
    assert!(merged.len() > 2, "{} partitions", merged.len());
    assert!(
        merged
            .iter()
            .zip(&fresh)
            .all(|(merged, fresh)| merged.is(fresh))
    );
    let probe_every_partition = |scan: &mut Scanner| {
        scan.minimum_nprobes(merged.len());
    };
    let rows = dataset.count_rows(None).await.unwrap();
    let mut found = 0;
    for query in &queries() {
        let (indexed, _) = search(&dataset, query, 10, None, probe_every_partition).await;
        let exact = nearest(&dataset, query, rows, Some(DistanceType::L2)).await;
        assert_exact(&indexed, &exact);
        let indexed = indexed
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect::<HashSet<_>>();
        found += exact
            .iter()
            .take(10)
            .filter(|(row_id, _)| indexed.contains(row_id))
            .count();
    }
    let recall = found as f32 / (10 * queries().len()) as f32;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Compaction carries each partition's graph over to the rows it moved
/// rather than drawing it anew. With no row deleted, a graph a merge
/// extended, which no fresh build draws, comes through edge for edge; with
/// rows deleted, every survivor none of whose neighbors died keeps its
/// out-edges.
#[rstest]
#[case::nothing_deleted(None)]
// Few enough deaths that most survivors keep every neighbor.
#[case::rows_deleted(Some("id % 13 = 0"))]
#[tokio::test]
async fn compaction_carries_each_partitions_graph_over(#[case] deleted: Option<&str>) {
    const INDEXED: u64 = ROWS / 2;
    let dir = TempStrDir::default();
    let mut dataset = indexed_along_the_diagonal(dir.as_str(), Codes::Flat, INDEXED, false).await;
    // Numbered on from the indexed rows, so that `id` names every row.
    let appended = gen_batch()
        .with_seed(Seed::from(9))
        .col("id", array::step_custom::<Int32Type>(INDEXED as i32, 1))
        .col("vec", array::rand_vec::<Float32Type>(DIM.into()))
        .into_reader_rows(RowCount::from(INDEXED / 4), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();
    let old = partitions(&dataset, Codes::Flat).await;
    if deleted.is_none() {
        // Were its lists the sets a fresh build draws, a rebuild would draw
        // them again.
        let fresh = drawn_afresh(&dataset, Codes::Flat).await;
        let list = |graph: &Vamana, vertex: u32| {
            let neighbors = graph.graph().neighbors(vertex).iter();
            neighbors.copied().collect::<HashSet<_>>()
        };
        for (old, fresh) in old.iter().zip(&fresh) {
            let mut vertices = 0..old.row_ids.len() as u32;
            assert!(vertices.any(|vertex| list(&old.graph, vertex) != list(fresh, vertex)));
        }
    }
    let old_ids = ids_by_row_id(&dataset).await;

    if let Some(predicate) = deleted {
        dataset.delete(predicate).await.unwrap();
    }
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    let ids = ids_by_row_id(&dataset).await;
    // Every row moved, so that only the remap can have named the vertices.
    assert!(ids.keys().all(|row_id| !old_ids.contains_key(row_id)));
    let live = ids.values().copied().collect::<HashSet<_>>();
    let dead = old_ids
        .values()
        .filter(|id| !live.contains(id))
        .copied()
        .collect::<HashSet<_>>();
    assert_eq!(dead.is_empty(), deleted.is_none());

    let compacted = partitions(&dataset, Codes::Flat).await;
    assert_eq!(compacted.len(), old.len());
    let vertices = |partition: &Partition, ids: &HashMap<u64, i32>| {
        let row_ids = partition.graph.graph().row_ids();
        row_ids.iter().map(|row_id| ids[row_id]).collect::<Vec<_>>()
    };
    let edges = |partition: &Partition, ids: &HashMap<u64, i32>| {
        let edges = partition.edges().into_iter();
        edges
            .map(|(row_id, neighbors)| {
                let neighbors = neighbors.iter().map(|neighbor| ids[neighbor]).collect();
                (ids[&row_id], neighbors)
            })
            .collect::<HashMap<i32, HashSet<i32>>>()
    };
    for (old, compacted) in old.iter().zip(&compacted) {
        let mut survivors = vertices(old, &old_ids);
        survivors.retain(|id| !dead.contains(id));
        assert_eq!(vertices(compacted, &ids), survivors);
        let (old_edges, compacted_edges) = (edges(old, &old_ids), edges(compacted, &ids));
        let (intact, touched): (Vec<_>, Vec<_>) = old_edges
            .iter()
            .filter(|(id, _)| !dead.contains(*id))
            .partition(|(_, neighbors)| neighbors.is_disjoint(&dead));
        assert!(!intact.is_empty());
        assert_eq!(touched.is_empty(), deleted.is_none());
        // Deaths this sparse cut no survivor off. One linked again would draw
        // its own list anew and push edges into the lists of others.
        for (id, neighbors) in intact {
            assert_eq!(&compacted_edges[id], neighbors, "vertex {id}");
        }
        if deleted.is_none() {
            assert_eq!(compacted.graph.metadata(), old.graph.metadata());
        }
    }
    if deleted.is_some() {
        let fresh = drawn_afresh(&dataset, Codes::Flat).await;
        assert!(
            compacted
                .iter()
                .zip(&fresh)
                .all(|(compacted, fresh)| !compacted.is(fresh))
        );
        let recall = recall(&dataset, 10, DistanceType::L2).await;
        assert!(recall >= 0.95, "recall {recall}");
    }
}

/// The graph parameters of an index built apart: [`vamana`] with an `alpha`
/// other than the default, so that every parameter a merge carries over
/// shows in the graphs it draws.
fn shard_vamana() -> VamanaBuildParams {
    vamana().with_alpha(1.3)
}

/// Three IVF centroids for an index built apart under `distance_type`: the
/// [`splitting_centroids`] and a third that no vector is nearest to, so that
/// its partition holds no row: under L2 at 4.0 along the diagonal of the cube
/// the vectors fill, under cosine against the diagonal.
fn shard_centroids(distance_type: DistanceType, dimension: u32) -> FixedSizeListArray {
    let empty = match distance_type {
        DistanceType::Cosine => vec![-(dimension as f32).sqrt().recip(); dimension as usize],
        _ => vec![4.0; dimension as usize],
    };
    let [first, second] = splitting_centroids(distance_type, dimension);
    FixedSizeListArray::try_new_from_values(
        Float32Array::from([first, second, empty].concat()),
        dimension as i32,
    )
    .unwrap()
}

/// [`params`] for an index built apart, whose shards train neither their IVF
/// centroids, [`shard_centroids`], nor, over product codes, their codebook,
/// [`pq_codebook`].
fn shard_params(codes: Codes, distance_type: DistanceType) -> VectorIndexParams {
    let dimension = codes.dimension();
    let mut params = params(codes, distance_type);
    params.stages[0] = StageParams::Ivf(
        IvfBuildParams::try_with_centroids(3, Arc::new(shard_centroids(distance_type, dimension)))
            .unwrap(),
    );
    params.stages[1] = StageParams::Vamana(shard_vamana());
    params
}

/// The shard of the index built apart over `fragment` under `params`, left
/// uncommitted.
async fn shard(dataset: &mut Dataset, fragment: u32, params: &VectorIndexParams) -> IndexMetadata {
    let shard = dataset
        .create_index_builder(&["vec"], IndexType::Vector, params)
        .name(INDEX.into())
        .fragments(vec![fragment])
        .execute_uncommitted()
        .await
        .unwrap();
    assert_eq!(shard.index_version, IVF_VAMANA_INDEX_VERSION as i32);
    assert_eq!(
        shard.fragment_bitmap,
        Some(RoaringBitmap::from_iter([fragment]))
    );
    shard
}

/// A [`shard`] of the index per fragment of `dataset`, in fragment order.
async fn shards(dataset: &mut Dataset, params: &VectorIndexParams) -> Vec<IndexMetadata> {
    let mut shards = Vec::new();
    for fragment in dataset.get_fragments() {
        shards.push(shard(dataset, fragment.id() as u32, params).await);
    }
    shards
}

/// Shards of the index built apart merge into one segment whose partitions
/// each hold a graph drawn over all of their merged rows, with the build
/// parameters the shards share, under every kind of codes and either metric.
#[rstest]
#[case::flat(Codes::Flat, DistanceType::L2, 400)]
#[case::scalar(Codes::Sq, DistanceType::L2, 400)]
// The given codebook needs no rows to train on, so the case keeps few.
#[case::product(Codes::Pq, DistanceType::L2, ROWS / 5)]
#[case::flat_cosine(Codes::Flat, DistanceType::Cosine, 400)]
#[tokio::test]
async fn shards_built_apart_merge_into_graphs_drawn_over_their_rows(
    #[case] codes: Codes,
    #[case] distance_type: DistanceType,
    #[case] rows: u64,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), rows, codes.dimension(), false).await;
    let params = shard_params(codes, distance_type);
    let shards = shards(&mut dataset, &params).await;
    let merged = dataset.merge_existing_index_segments(shards).await.unwrap();
    assert_eq!(merged.index_version, IVF_VAMANA_INDEX_VERSION as i32);
    assert_eq!(
        merged.fragment_bitmap,
        Some(RoaringBitmap::from_iter([0, 1]))
    );
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();

    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    assert_eq!(statistics["index_type"], params.index_type().to_string());
    let segment = &statistics["indices"][0];
    assert_eq!(segment["metric_type"], distance_type.to_string());
    assert_eq!(segment["sub_index"]["index_type"], "VAMANA");
    let built_with: VamanaBuildParams =
        serde_json::from_value(segment["sub_index"]["params"].clone()).unwrap();
    assert_eq!(built_with, shard_vamana());

    let index = open_segment(&dataset).await;
    let merged = partitions_in(index.as_ref(), codes).await;
    let fresh = drawn_afresh_in(index.as_ref(), codes, &shard_vamana()).await;
    assert_eq!(merged.len(), 3);
    // A graph read back empty records the default parameters, so the empty
    // partition is checked by its size.
    assert!(merged[2].row_ids.is_empty());
    for (partition, fresh) in merged.iter().zip(&fresh).take(2) {
        // Both shards brought rows to the partition.
        let fragments = partition
            .row_ids
            .iter()
            .map(|row_id| RowAddress::from(*row_id).fragment_id())
            .collect::<HashSet<_>>();
        assert_eq!(fragments, HashSet::from([0, 1]));
        assert!(partition.is(fresh));
    }
    assert_eq!(
        merged
            .iter()
            .map(|partition| partition.row_ids.len())
            .sum::<usize>(),
        rows as usize
    );
    let recall = recall(&dataset, 10, distance_type).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Shards of the index built apart answer side by side, unmerged, each a
/// segment over exactly its own fragment with graphs drawn over its rows. A
/// shard keeps the IVF centroids it is given, though `retrain` asks to train
/// them again; given none, each shard trains its own.
#[rstest]
#[case::flat(Codes::Flat, ROWS, true)]
#[case::product(Codes::Pq, ROWS / 5, true)]
#[case::flat_trained_apart(Codes::Flat, ROWS, false)]
#[tokio::test]
async fn shards_built_apart_answer_side_by_side(
    #[case] codes: Codes,
    #[case] rows: u64,
    #[case] centroids_given: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), rows, codes.dimension(), false).await;
    let params = if centroids_given {
        let mut params = shard_params(codes, DistanceType::L2);
        if let StageParams::Ivf(ivf) = &mut params.stages[0] {
            ivf.retrain = true;
        }
        params
    } else {
        params(codes, DistanceType::L2)
    };
    let graph_params = super::vamana_stage(&params).unwrap().clone();
    let shards = shards(&mut dataset, &params).await;
    dataset
        .commit_existing_index_segments(INDEX, "vec", shards)
        .await
        .unwrap();

    let segments = dataset.load_indices_by_name(INDEX).await.unwrap();
    assert_eq!(segments.len(), 2);
    let row_ids = ids_by_row_id(&dataset)
        .await
        .into_keys()
        .collect::<Vec<_>>();
    let mut trained = Vec::new();
    for segment in &segments {
        let fragments = segment.fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragments.len(), 1);
        let fragment = fragments.min().unwrap();
        let index = dataset
            .open_vector_index("vec", &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let partitions = partitions_in(index.as_ref(), codes).await;
        let fresh = drawn_afresh_in(index.as_ref(), codes, &graph_params).await;
        let indexed = partitions
            .iter()
            .flat_map(|partition| partition.row_ids.iter().copied())
            .collect::<HashSet<_>>();
        let expected = row_ids
            .iter()
            .copied()
            .filter(|row_id| RowAddress::from(*row_id).fragment_id() == fragment)
            .collect::<HashSet<_>>();
        assert_eq!(indexed, expected);
        for (partition, fresh) in partitions.iter().zip(&fresh) {
            if !partition.row_ids.is_empty() {
                assert!(partition.is(fresh));
            }
        }
        let centroids = index.ivf_model().centroids.clone().unwrap();
        if centroids_given {
            assert_eq!(
                centroids,
                shard_centroids(DistanceType::L2, codes.dimension())
            );
            // The given centroids split the rows between the first two
            // partitions and leave the third empty.
            assert!(!partitions[0].row_ids.is_empty());
            assert!(!partitions[1].row_ids.is_empty());
            assert!(partitions[2].row_ids.is_empty());
        }
        trained.push(centroids);
    }
    if !centroids_given {
        assert_ne!(trained[0], trained[1]);
    }
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Shards whose graphs were built under different parameters are not merged:
/// one merged graph keeps one set.
#[rstest]
#[case::max_degree(shard_vamana().with_max_degree(12))]
#[case::search_list_size(shard_vamana().with_search_list_size(24))]
#[case::alpha(shard_vamana().with_alpha(1.1))]
#[tokio::test]
async fn shards_of_unlike_graphs_are_not_merged(#[case] unlike: VamanaBuildParams) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let params = shard_params(Codes::Flat, DistanceType::L2);
    let mut unlike_params = params.clone();
    unlike_params.stages[1] = StageParams::Vamana(unlike.clone());
    let shards = vec![
        shard(&mut dataset, 0, &params).await,
        shard(&mut dataset, 1, &unlike_params).await,
    ];
    let error = dataset
        .merge_existing_index_segments(shards)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    let message = error.to_string();
    assert!(
        message.contains("Vamana build parameters mismatch while merging index segments"),
        "{message}"
    );
    assert!(message.contains(&format!("got {unlike:?}")), "{message}");
}

/// A shard over product codes is refused without the codebook every shard
/// shares, as a shard of IVF_HNSW_PQ is, also over every fragment of a table
/// of fewer rows than eight-bit codes train on, where a whole build would
/// record its index untrained.
#[rstest]
#[case::over_one_fragment(ROWS, vec![0])]
#[case::over_every_fragment(ROWS / 5, vec![0, 1])]
#[tokio::test]
async fn a_product_code_shard_needs_its_codebook(#[case] rows: u64, #[case] fragments: Vec<u32>) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), rows, PQ_DIM, false).await;
    let mut params = shard_params(Codes::Pq, DistanceType::L2);
    if let Some(StageParams::PQ(pq)) = params.stages.last_mut() {
        pq.codebook = None;
    }
    let error = dataset
        .create_index_builder(&["vec"], IndexType::Vector, &params)
        .name(INDEX.into())
        .fragments(fragments)
        .execute_uncommitted()
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("missing precomputed PQ codebook"),
        "{error}"
    );
}

/// Rows without a vector stay out of a merged graph: when a shard holds some
/// of them, when one holds only them and leads the merge, and when every
/// shard does. The graphs keep the parameters of the shards that record any,
/// and scalar codes the bounds of the shards that hold vectors: a shard
/// without vectors trained none, and its codes decode nothing.
#[rstest]
#[case::half_null(Codes::Sq, [0.5, 0.5])]
#[case::all_null_first(Codes::Flat, [1.0, 0.5])]
#[case::all_null_first_under_scalar_codes(Codes::Sq, [1.0, 0.5])]
#[case::all_null(Codes::Sq, [1.0, 1.0])]
#[tokio::test]
async fn null_vectors_stay_out_of_a_merged_graph(
    #[case] codes: Codes,
    #[case] null_shares: [f64; 2],
) {
    let dir = TempStrDir::default();
    let (_, first) =
        write_with_nulls(dir.as_str(), WriteMode::Create, 1, null_shares[0], DIM).await;
    let (mut dataset, second) =
        write_with_nulls(dir.as_str(), WriteMode::Append, 2, null_shares[1], DIM).await;
    for (non_null, null_share) in [first, second].into_iter().zip(null_shares) {
        assert_eq!(non_null == 0, null_share == 1.0, "{non_null}");
        assert!(non_null < ROWS as usize / 2, "{non_null}");
    }
    let shards = shards(&mut dataset, &shard_params(codes, DistanceType::L2)).await;
    let merged = dataset.merge_existing_index_segments(shards).await.unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();

    let index = open_segment(&dataset).await;
    let partitions = partitions_in(index.as_ref(), codes).await;
    let fresh = drawn_afresh_in(index.as_ref(), codes, &shard_vamana()).await;
    for (partition, fresh) in partitions.iter().zip(&fresh) {
        if !partition.row_ids.is_empty() {
            assert!(partition.is(fresh));
        }
    }
    if matches!(codes, Codes::Sq) && first + second > 0 {
        assert_eq!(scalar_bounds(&dataset).await, value_range(&dataset).await);
    }
    assert_indexes_the_non_null_vectors(&dataset, first + second, codes).await;
}

/// The bounds that decode the scalar codes of the index's one segment, at the
/// precision of the vectors they were trained on: the statistics read them
/// back from JSON within an ulp of the stored ones.
async fn scalar_bounds(dataset: &Dataset) -> Range<f32> {
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    let bounds = &statistics["indices"][0]["sub_index"]["bounds"];
    bounds["start"].as_f64().unwrap() as f32..bounds["end"].as_f64().unwrap() as f32
}

/// The smallest and the largest value of `dataset`'s non-null vectors.
async fn value_range(dataset: &Dataset) -> Range<f32> {
    let batch = dataset
        .scan()
        .project(&["vec"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch["vec"].as_fixed_size_list();
    (0..vectors.len())
        .filter(|row| vectors.is_valid(*row))
        .flat_map(|row| {
            vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec()
        })
        .fold(f32::MAX..f32::MIN, |range, value| {
            range.start.min(value)..range.end.max(value)
        })
}

/// Each row's vector by its row id.
async fn vectors_by_row_id(dataset: &Dataset) -> HashMap<u64, Vec<f32>> {
    let batch = dataset
        .scan()
        .project(&["vec"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch["vec"].as_fixed_size_list();
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .enumerate()
        .map(|(row, row_id)| {
            let vector = vectors.value(row);
            (
                *row_id,
                vector.as_primitive::<Float32Type>().values().to_vec(),
            )
        })
        .collect()
}

/// `rows` random vectors numbered by `id`, in one fragment per scale of
/// `scales`, each fragment's values its scale times those drawn for it, so
/// that shards over them train unlike scalar bounds.
async fn fragments_over_ranges(
    uri: &str,
    rows: u64,
    scales: &[f32],
    enable_stable_row_ids: bool,
) -> Dataset {
    let per_fragment = rows as usize / scales.len();
    let batches = scales
        .iter()
        .enumerate()
        .map(|(fragment, scale)| {
            let values = generate_random_array_with_seed::<Float32Type>(
                per_fragment * DIM as usize,
                [fragment as u8 + 1; 32],
            );
            let values = Float32Array::from_iter_values(values.values().iter().map(|v| v * scale));
            let vectors = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();
            let first = (fragment * per_fragment) as i32;
            let ids = arrow_array::Int32Array::from_iter_values(first..first + per_fragment as i32);
            RecordBatch::try_from_iter([
                ("id", Arc::new(ids) as ArrayRef),
                ("vec", Arc::new(vectors) as ArrayRef),
            ])
            .unwrap()
        })
        .collect::<Vec<_>>();
    let schema = batches[0].schema();
    let dataset = Dataset::write(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: per_fragment,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), scales.len());
    dataset
}

/// The scalar codes of every live row of the index's one segment decode
/// within `steps` steps of its values, and the segment holds no other row.
async fn assert_scalar_codes_decode_within(dataset: &Dataset, steps: f64) {
    let bounds = scalar_bounds(dataset).await;
    let bounds = f64::from(bounds.start)..f64::from(bounds.end);
    let step = (bounds.end - bounds.start) / 255.0;
    let vectors = vectors_by_row_id(dataset).await;
    let index = open_segment(dataset).await;
    let ivf = index
        .as_any()
        .downcast_ref::<IVFIndex<Vamana, ScalarQuantizer>>()
        .unwrap();
    let mut decoded_rows = 0;
    for partition in 0..ivf.ivf_model().num_partitions() {
        let storage = ivf.load_partition_storage(partition, None).await.unwrap();
        for batch in storage.to_batches().unwrap() {
            let codes = batch[SQ_CODE_COLUMN].as_fixed_size_list();
            for (row, row_id) in batch[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .enumerate()
            {
                let vector = vectors
                    .get(row_id)
                    .unwrap_or_else(|| panic!("row {row_id} is not a live row"));
                let row_codes = codes.value(row);
                for (value, code) in vector
                    .iter()
                    .zip(row_codes.as_primitive::<UInt8Type>().values())
                {
                    let decoded = bounds.start + f64::from(*code) * step;
                    assert!(
                        (f64::from(*value) - decoded).abs() <= steps * step + 1e-9,
                        "row {row_id}: {value} decodes as {decoded}"
                    );
                }
                decoded_rows += 1;
            }
        }
    }
    assert_eq!(decoded_rows, vectors.len());
}

/// The rows deleted between the build of shards and their merge.
#[derive(Clone, Copy, Debug)]
enum Deleted {
    Nothing,
    Rows,
    Shard,
}

/// Shards of scalar codes over unlike value ranges merge under bounds they
/// share, the union of the bounds of the shards whose rows the merge keeps:
/// every row's codes decode within a step and a half of its values, as codes
/// re-encoded from a shard's own bounds do, where the first shard's bounds
/// alone would clamp the second's values. On a table of stable row ids a
/// merge leaves out the rows deleted after the shards were built, and a
/// shard whose rows are all gone adds no bounds.
#[rstest]
#[case::nothing_deleted(Deleted::Nothing)]
#[case::rows_deleted(Deleted::Rows)]
#[case::shard_deleted(Deleted::Shard)]
#[tokio::test]
async fn sq_shards_over_unlike_ranges_merge_under_shared_bounds(#[case] deleted: Deleted) {
    let dir = TempStrDir::default();
    let stable_row_ids = !matches!(deleted, Deleted::Nothing);
    let mut dataset =
        fragments_over_ranges(dir.as_str(), ROWS / 2, &[1.0, 4.0], stable_row_ids).await;
    let shards = shards(&mut dataset, &shard_params(Codes::Sq, DistanceType::L2)).await;
    let union = value_range(&dataset).await;
    // The second shard's values reach far past the first's.
    assert!(union.end > 3.0, "{union:?}");
    let predicate = match deleted {
        Deleted::Nothing => None,
        Deleted::Rows => Some("id % 7 = 0".to_string()),
        Deleted::Shard => Some(format!("id >= {}", ROWS / 4)),
    };
    if let Some(predicate) = predicate {
        dataset.delete(&predicate).await.unwrap();
    }
    let merged = dataset.merge_existing_index_segments(shards).await.unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();

    let bounds = scalar_bounds(&dataset).await;
    match deleted {
        Deleted::Shard => {
            assert_eq!(bounds, value_range(&dataset).await);
            assert!(bounds.end < 1.0, "{bounds:?}");
        }
        _ => assert_eq!(bounds, union),
    }
    assert_scalar_codes_decode_within(&dataset, 1.5).await;
    // Queries over both ranges, but over the first alone once the second's
    // rows are gone: a query past every row's bounds is clamped to them.
    let queries = queries()
        .into_iter()
        .flat_map(|query| {
            let wide = Float32Array::from_iter_values(query.values().iter().map(|v| v * 4.0));
            match deleted {
                Deleted::Shard => vec![query],
                _ => vec![query, wide],
            }
        })
        .collect::<Vec<_>>();
    let recall = recall_over(&dataset, &queries, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Every segment of `dataset`'s index keeps its product codes row-major.
async fn assert_row_major(dataset: &Dataset) {
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    for segment in statistics["indices"].as_array().unwrap() {
        assert_eq!(segment["sub_index"]["transposed"], false, "{segment}");
    }
}

/// Product codes stay row-major through a whole build, an index initialized
/// from it on another table, an append, a merge of segments and a
/// compaction, as in a shard.
#[tokio::test]
async fn product_codes_stay_row_major_through_maintenance() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), ROWS / 2, PQ_DIM, false).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Pq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    assert_row_major(&dataset).await;
    let copy_dir = TempStrDir::default();
    let mut copy = two_fragments_of_rows(copy_dir.as_str(), ROWS / 2, PQ_DIM, false).await;
    copy.initialize_index(&dataset, INDEX).await.unwrap();
    assert_row_major(&copy).await;
    let copy_recall = recall(&copy, 10, DistanceType::L2).await;
    assert!(copy_recall >= 0.95, "recall {copy_recall}");
    append_vectors(&mut dataset, ROWS / 8, 9).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(dataset.load_indices_by_name(INDEX).await.unwrap().len(), 2);
    assert_row_major(&dataset).await;
    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    assert_row_major(&dataset).await;
    dataset.delete("id % 7 = 0").await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_row_major(&dataset).await;
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// Each row's product codes in the segment `index`, by row id. A partition
/// holds its codes transposed once loaded, whatever their layout on disk.
async fn product_codes_by_row_id(index: &dyn VectorIndex) -> HashMap<u64, Vec<u8>> {
    let ivf = index
        .as_any()
        .downcast_ref::<IVFIndex<Vamana, ProductQuantizer>>()
        .unwrap();
    let mut codes_by_row_id = HashMap::new();
    for partition in 0..ivf.ivf_model().num_partitions() {
        let storage = ivf.load_partition_storage(partition, None).await.unwrap();
        for batch in storage.to_batches().unwrap() {
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
            let codes = batch[PQ_CODE_COLUMN].as_fixed_size_list();
            let width = codes.value_length() as usize;
            let rows = transpose(
                codes.values().as_primitive::<UInt8Type>(),
                width,
                row_ids.len(),
            );
            for (row_id, codes) in row_ids.iter().zip(rows.values().chunks_exact(width)) {
                codes_by_row_id.insert(*row_id, codes.to_vec());
            }
        }
    }
    codes_by_row_id
}

/// A segment merged from shards merges again: of three shards, the first two
/// merge, and their merge with the third. Product codes come through both
/// merges as the shards wrote them. Scalar codes over ranges each wider than
/// the last, which the second merge re-encodes again, decode within two
/// steps of the bounds the three shards share.
#[rstest]
#[case::product(Codes::Pq)]
#[case::scalar(Codes::Sq)]
#[tokio::test]
async fn merges_merge_again(#[case] codes: Codes) {
    let dir = TempStrDir::default();
    let mut dataset = match codes {
        Codes::Pq => {
            let mut dataset = two_fragments_of_rows(dir.as_str(), ROWS / 10, PQ_DIM, false).await;
            append_vectors(&mut dataset, ROWS / 20, 9).await;
            dataset
        }
        _ => fragments_over_ranges(dir.as_str(), ROWS / 4, &[1.0, 2.0, 4.0], false).await,
    };
    let mut shards = shards(&mut dataset, &shard_params(codes, DistanceType::L2)).await;
    assert_eq!(shards.len(), 3);
    let mut shard_codes = HashMap::new();
    if matches!(codes, Codes::Pq) {
        for shard in &shards {
            let index = dataset
                .open_vector_index_from_metadata("vec", shard, &NoOpMetricsCollector)
                .await
                .unwrap();
            shard_codes.extend(product_codes_by_row_id(index.as_ref()).await);
        }
    }
    let last = shards.pop().unwrap();
    let first = dataset.merge_existing_index_segments(shards).await.unwrap();
    let merged = dataset
        .merge_existing_index_segments(vec![first, last])
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();
    if matches!(codes, Codes::Pq) {
        assert_row_major(&dataset).await;
        let merged = product_codes_by_row_id(open_segment(&dataset).await.as_ref()).await;
        assert_eq!(merged.len(), dataset.count_rows(None).await.unwrap());
        assert_eq!(merged, shard_codes);
    } else {
        assert_eq!(scalar_bounds(&dataset).await, value_range(&dataset).await);
        assert_scalar_codes_decode_within(&dataset, 2.0).await;
    }
    let merged = partitions(&dataset, codes).await;
    let fresh = drawn_afresh_in(
        open_segment(&dataset).await.as_ref(),
        codes,
        &shard_vamana(),
    )
    .await;
    for (partition, fresh) in merged.iter().zip(&fresh) {
        if !partition.row_ids.is_empty() {
            assert!(partition.is(fresh));
        }
    }
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A shard asked for every fragment there is stays a shard: it keeps its
/// centroids though `retrain` asks to train them again, and merges with the
/// shards built over the fragments appended after it.
#[tokio::test]
async fn a_shard_over_every_fragment_merges_with_later_shards() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), ROWS / 2, DIM, false).await;
    let mut params = shard_params(Codes::Flat, DistanceType::L2);
    if let StageParams::Ivf(ivf) = &mut params.stages[0] {
        ivf.retrain = true;
    }
    let every = dataset
        .create_index_builder(&["vec"], IndexType::Vector, &params)
        .name(INDEX.into())
        .fragments(vec![0, 1])
        .execute_uncommitted()
        .await
        .unwrap();
    append_vectors(&mut dataset, ROWS / 8, 9).await;
    let later = shard(&mut dataset, 2, &params).await;
    let merged = dataset
        .merge_existing_index_segments(vec![every, later])
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();
    let index = open_segment(&dataset).await;
    assert_eq!(
        index.ivf_model().centroids.as_ref(),
        Some(&shard_centroids(DistanceType::L2, DIM))
    );
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A request for every fragment without given centroids builds the whole
/// index, as it does for any index type: fewer vectors than the partitions
/// it asks for cap their count, where a shard keeps the count it was given.
#[tokio::test]
async fn a_request_for_every_fragment_without_centroids_builds_the_whole_index() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of_rows(dir.as_str(), 8, DIM, false).await;
    assert_eq!(dataset.count_rows(None).await.unwrap(), 8);
    let ivf = IvfBuildParams {
        num_partitions: Some(9),
        ..Default::default()
    };
    let params = VectorIndexParams::ivf_vamana(DistanceType::L2, ivf, vamana());
    let segment = dataset
        .create_index_builder(&["vec"], IndexType::Vector, &params)
        .name(INDEX.into())
        .fragments(vec![0, 1])
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![segment])
        .await
        .unwrap();
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    assert_eq!(statistics["indices"][0]["num_partitions"], 8);
}

/// The way a merge of shards that cannot become one segment fails.
#[derive(Clone, Copy, Debug)]
enum Unmergeable {
    UnlikeCodes,
    UnlikeMetrics,
    LostRoot,
    LostStorage,
    UnlikeEmptyShard,
    UnlikeGraphsWithoutDetails,
    UnlikeDetailsAndGraphs,
    OtherIndexFirst,
    OtherIndexSecond,
}

/// A merge of shards that cannot become one segment is refused before it
/// writes anything: shards over unlike codes or metrics, a shard whose root
/// or storage is gone, graphs built under unlike parameters, as the shards'
/// details or, where a shard has none, their root files tell, a shard
/// without vectors whose details hold other parameters than the graphs of a
/// shard without details, and a shard of another index in either place.
#[rstest]
#[case::unlike_codes(
    Unmergeable::UnlikeCodes,
    "Cannot merge an IVF_VAMANA_FLAT segment under l2 with an IVF_VAMANA_SQ segment under l2"
)]
#[case::unlike_metrics(
    Unmergeable::UnlikeMetrics,
    "Cannot merge an IVF_VAMANA_FLAT segment under l2 with an IVF_VAMANA_FLAT segment under \
     cosine"
)]
#[case::lost_root(Unmergeable::LostRoot, "index.idx")]
#[case::lost_storage(Unmergeable::LostStorage, "auxiliary.idx")]
#[case::unlike_empty_shard(
    Unmergeable::UnlikeEmptyShard,
    "Vamana build parameters mismatch while merging index segments"
)]
#[case::unlike_graphs_without_details(
    Unmergeable::UnlikeGraphsWithoutDetails,
    "Vamana build parameters mismatch while merging index segments"
)]
#[case::unlike_details_and_graphs(
    Unmergeable::UnlikeDetailsAndGraphs,
    "in the graphs their root files record"
)]
#[case::other_index_first(Unmergeable::OtherIndexFirst, "which is not an IVF_VAMANA segment")]
#[case::other_index_second(Unmergeable::OtherIndexSecond, "which is not an IVF_VAMANA segment")]
#[tokio::test]
async fn merges_that_cannot_make_one_segment_are_refused_before_writing(
    #[case] unmergeable: Unmergeable,
    #[case] reason: &str,
) {
    let dir = TempStrDir::default();
    let flat = shard_params(Codes::Flat, DistanceType::L2);
    let mut unlike = flat.clone();
    unlike.stages[1] = StageParams::Vamana(shard_vamana().with_max_degree(12));
    let (dataset, mut shards) = match unmergeable {
        Unmergeable::UnlikeEmptyShard | Unmergeable::UnlikeDetailsAndGraphs => {
            write_with_nulls(dir.as_str(), WriteMode::Create, 1, 1.0, DIM).await;
            let (mut dataset, _) =
                write_with_nulls(dir.as_str(), WriteMode::Append, 2, 0.5, DIM).await;
            let shards = vec![
                shard(&mut dataset, 0, &unlike).await,
                shard(&mut dataset, 1, &flat).await,
            ];
            (dataset, shards)
        }
        Unmergeable::OtherIndexFirst | Unmergeable::OtherIndexSecond => {
            let mut dataset = two_fragments(dir.as_str()).await;
            let vamana = shard(&mut dataset, 0, &flat).await;
            let ivf = IvfBuildParams::try_with_centroids(
                3,
                Arc::new(shard_centroids(DistanceType::L2, DIM)),
            )
            .unwrap();
            let other = dataset
                .create_index_builder(
                    &["vec"],
                    IndexType::Vector,
                    &VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
                )
                .name(INDEX.into())
                .fragments(vec![1])
                .execute_uncommitted()
                .await
                .unwrap();
            assert!(!super::is_vamana_segment(&other));
            let shards = match unmergeable {
                Unmergeable::OtherIndexFirst => vec![other, vamana],
                _ => vec![vamana, other],
            };
            (dataset, shards)
        }
        _ => {
            let mut dataset = two_fragments(dir.as_str()).await;
            let second = match unmergeable {
                Unmergeable::UnlikeCodes => shard_params(Codes::Sq, DistanceType::L2),
                Unmergeable::UnlikeMetrics => shard_params(Codes::Flat, DistanceType::Cosine),
                Unmergeable::UnlikeGraphsWithoutDetails => unlike.clone(),
                _ => flat.clone(),
            };
            let shards = vec![
                shard(&mut dataset, 0, &flat).await,
                shard(&mut dataset, 1, &second).await,
            ];
            (dataset, shards)
        }
    };
    match unmergeable {
        Unmergeable::LostRoot | Unmergeable::LostStorage => {
            let (shard, file) = match unmergeable {
                Unmergeable::LostRoot => (0, INDEX_FILE_NAME),
                _ => (1, INDEX_AUXILIARY_FILE_NAME),
            };
            let path = dataset
                .indices_dir()
                .join(shards[shard].uuid.to_string())
                .join(file);
            dataset.object_store.delete(&path).await.unwrap();
        }
        // Only the root files tell the second shard's parameters.
        Unmergeable::UnlikeGraphsWithoutDetails | Unmergeable::UnlikeDetailsAndGraphs => {
            shards[1].index_details = Some(Arc::new(vector_index_details_default()));
        }
        _ => {}
    }
    let before = dataset
        .object_store
        .read_dir(dataset.indices_dir())
        .await
        .unwrap();
    let error = dataset
        .merge_existing_index_segments(shards)
        .await
        .unwrap_err();
    match unmergeable {
        Unmergeable::LostRoot | Unmergeable::LostStorage => {
            assert!(matches!(error, Error::NotFound { .. }), "{error}")
        }
        _ => assert!(matches!(error, Error::InvalidInput { .. }), "{error}"),
    }
    assert!(error.to_string().contains(reason), "{error}");
    let after = dataset
        .object_store
        .read_dir(dataset.indices_dir())
        .await
        .unwrap();
    assert_eq!(after, before);
}

/// [`two_fragments`] of vectors of `element_type` values.
async fn two_fragments_cast(uri: &str, element_type: DataType) -> Dataset {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(DIM.into()))
        .into_batch_rows(RowCount::from(ROWS))
        .unwrap();
    let DataType::FixedSizeList(item, dimension) = batch["vec"].data_type() else {
        unreachable!()
    };
    let item = Field::new(item.name(), element_type, item.is_nullable());
    let vectors = arrow::compute::cast(
        &batch["vec"],
        &DataType::FixedSizeList(Arc::new(item), *dimension),
    )
    .unwrap();
    let batch =
        RecordBatch::try_from_iter([("id", batch["id"].clone()), ("vec", vectors)]).unwrap();
    let schema = batch.schema();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: (ROWS / 2) as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
}

/// The parameter of a shard that does not fit the indexed column.
#[derive(Clone, Copy, Debug)]
enum Misfit {
    CentroidWidth,
    CentroidType,
    CodebookType,
    CodebookOfCodewords,
}

/// Centroids or a codebook that do not fit the indexed column are refused:
/// centroids of another width, centroids or a codebook of another float type
/// than the vectors', or a codebook of codewords rather than one flat array
/// of the vectors' values, as both builds take it.
#[rstest]
#[case::centroids_of_another_width(Misfit::CentroidWidth, DataType::Float32, "centroids of 16")]
#[case::centroids_of_another_type(
    Misfit::CentroidType,
    DataType::Float16,
    "centroids of 32 Float32"
)]
#[case::a_codebook_of_another_type(Misfit::CodebookType, DataType::Float16, "codebook of Float32")]
#[case::a_codebook_of_codewords(
    Misfit::CodebookOfCodewords,
    DataType::Float32,
    "codebook of FixedSizeList"
)]
#[tokio::test]
async fn parameters_that_do_not_fit_the_column_are_refused(
    #[case] misfit: Misfit,
    #[case] element_type: DataType,
    #[case] reason: &str,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_cast(dir.as_str(), element_type.clone()).await;
    let codebook = match misfit {
        Misfit::CentroidWidth | Misfit::CentroidType => None,
        Misfit::CodebookType => Some(pq_codebook(DIM, DistanceType::L2)),
        Misfit::CodebookOfCodewords => Some(Arc::new(
            FixedSizeListArray::try_new_from_values(
                pq_codebook(DIM, DistanceType::L2)
                    .as_primitive::<Float32Type>()
                    .clone(),
                DIM as i32,
            )
            .unwrap(),
        ) as ArrayRef),
    };
    let codes = if codebook.is_some() {
        Codes::Pq
    } else {
        Codes::Flat
    };
    let mut params = shard_params(codes, DistanceType::L2);
    let width = match misfit {
        Misfit::CentroidWidth => 16,
        _ => DIM,
    };
    let mut centroids = shard_centroids(DistanceType::L2, width);
    if let Some(codebook) = codebook {
        // Centroids of the vectors' own type, so that only the codebook is off.
        centroids = arrow::compute::cast(
            &centroids,
            &DataType::FixedSizeList(Arc::new(Field::new("item", element_type, true)), DIM as i32),
        )
        .unwrap()
        .as_fixed_size_list()
        .clone();
        if let Some(StageParams::PQ(pq)) = params.stages.last_mut() {
            pq.num_sub_vectors = DIM as usize / 2;
            pq.codebook = Some(codebook);
        }
    }
    params.stages[0] =
        StageParams::Ivf(IvfBuildParams::try_with_centroids(3, Arc::new(centroids)).unwrap());
    let error = dataset
        .create_index_builder(&["vec"], IndexType::Vector, &params)
        .name(INDEX.into())
        .fragments(vec![0])
        .execute_uncommitted()
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains(reason), "{error}");
}

/// The root of a merged segment records the k-means loss of its shards
/// together, as one build over all of their rows would.
#[tokio::test]
async fn a_merged_segment_records_the_loss_of_its_shards() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let shards = shards(&mut dataset, &shard_params(Codes::Flat, DistanceType::L2)).await;
    dataset
        .commit_existing_index_segments(INDEX, "vec", shards.clone())
        .await
        .unwrap();
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    let losses = statistics["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| segment["loss"].as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(losses.len(), 2);
    let merged = dataset.merge_existing_index_segments(shards).await.unwrap();
    dataset
        .commit_existing_index_segments(INDEX, "vec", vec![merged])
        .await
        .unwrap();
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    assert_eq!(statistics["indices"].as_array().unwrap().len(), 1);
    let loss = statistics["indices"][0]["loss"].as_f64().unwrap();
    let total = losses.iter().sum::<f64>();
    assert!(
        (loss - total).abs() <= 1e-9 * total.abs().max(1.0),
        "{loss} != {total}"
    );
}

/// Over product codes without a given codebook a build trains one, as an
/// index users build without one does, under either metric.
#[rstest]
#[case::l2(DistanceType::L2)]
#[case::cosine(DistanceType::Cosine)]
#[tokio::test]
async fn a_build_trains_the_codebook_none_gives(#[case] distance_type: DistanceType) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of(dir.as_str(), PQ_DIM, false).await;
    let mut params = params(Codes::Pq, distance_type);
    if let Some(StageParams::PQ(pq)) = params.stages.last_mut() {
        pq.codebook = None;
    }
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let recall = recall(&dataset, 10, distance_type).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// An index whose graphs are all empty, over vectors that are all null,
/// records its graph parameters in its details only. Its statistics report
/// them, and the segment an optimize builds after it keeps them.
#[tokio::test]
async fn an_index_of_empty_graphs_keeps_its_parameters() {
    let dir = TempStrDir::default();
    let (mut dataset, none) = write_with_nulls(dir.as_str(), WriteMode::Create, 1, 1.0, DIM).await;
    assert_eq!(none, 0);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &shard_params(Codes::Flat, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    let (mut dataset, non_null) =
        write_with_nulls(dir.as_str(), WriteMode::Append, 2, 0.5, DIM).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let statistics = json(&dataset.index_statistics(INDEX).await.unwrap());
    let segments = statistics["indices"].as_array().unwrap();
    assert_eq!(segments.len(), 2);
    assert_ne!(shard_vamana(), VamanaBuildParams::default());
    for segment in segments {
        let params = &segment["sub_index"]["params"];
        assert_eq!(
            serde_json::from_value::<VamanaBuildParams>(params.clone()).unwrap(),
            shard_vamana(),
            "{segment}"
        );
    }
    // The details' parameters read as the graphs' do.
    assert_eq!(
        segments[0]["sub_index"]["params"],
        segments[1]["sub_index"]["params"]
    );
    let indexed = segments
        .iter()
        .flat_map(|segment| segment["partitions"].as_array().unwrap())
        .map(|partition| partition["size"].as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(indexed, non_null as u64);
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

#[rstest]
#[case::dot(params(Codes::Flat, DistanceType::Dot), "not dot")]
#[case::hamming(params(Codes::Flat, DistanceType::Hamming), "not hamming")]
#[case::rabitq(
    VectorIndexParams {
        stages: vec![
            StageParams::Ivf(ivf()),
            StageParams::Vamana(vamana()),
            StageParams::RQ(RQBuildParams::new(1)),
        ],
        ..params(Codes::Flat, DistanceType::L2)
    },
    "at most one PQ or SQ stage"
)]
#[case::legacy(
    {
        let mut params = params(Codes::Pq, DistanceType::L2);
        params.version(IndexFileVersion::Legacy);
        params
    },
    "V3 index file format only"
)]
#[case::four_bit_scalar_codes(
    {
        let mut params = params(Codes::Sq, DistanceType::L2);
        if let Some(StageParams::SQ(sq)) = params.stages.last_mut() {
            sq.num_bits = 4;
        }
        params
    },
    "8-bit scalar codes only"
)]
#[tokio::test]
async fn parameters_no_index_can_be_built_from_are_refused(
    #[case] params: VectorIndexParams,
    #[case] reason: &str,
) {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let error = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains(reason), "{error}");
    assert!(dataset.load_indices().await.unwrap().is_empty());
}

#[tokio::test]
async fn graph_parameters_out_of_range_are_refused() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let params = VectorIndexParams::ivf_vamana(DistanceType::L2, ivf(), vamana().with_alpha(0.9));
    let error = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("alpha"), "{error}");
}

async fn dataset_of(uri: &str, column: ArrayRef) -> Dataset {
    let batch = RecordBatch::try_from_iter(vec![("vec", column)]).unwrap();
    let schema = batch.schema();
    Dataset::write(RecordBatchIterator::new(vec![Ok(batch)], schema), uri, None)
        .await
        .unwrap()
}

/// A column of several vectors a row, or of bytes, is refused before any
/// row is read.
#[tokio::test]
async fn columns_no_graph_is_built_over_are_refused() {
    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values((0..64).map(|v| v as f32)),
        4,
    )
    .unwrap();
    let multivector = ListArray::try_new(
        Arc::new(Field::new("item", vectors.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 4, 8, 12, 16])),
        Arc::new(vectors),
        None,
    )
    .unwrap();
    let bytes = FixedSizeListArray::try_new_from_values(
        UInt8Array::from_iter_values((0..64).map(|v| v as u8)),
        8,
    )
    .unwrap();
    for (column, reason) in [
        (Arc::new(multivector) as ArrayRef, "one vector per row"),
        (Arc::new(bytes) as ArrayRef, "float vectors"),
    ] {
        let dir = TempStrDir::default();
        let mut dataset = dataset_of(dir.as_str(), column).await;
        let error = dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some(INDEX.into()),
                &params(Codes::Flat, DistanceType::Cosine),
                true,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains(reason), "{error}");
    }
}

/// Several queries at once go through the per-query loop: the graph keeps
/// no top-k heap shared across partitions, which the batch node needs.
#[tokio::test]
async fn a_batch_of_queries_is_searched_one_query_at_a_time() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    let queries = queries();
    let both = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(
            queries[0]
                .values()
                .iter()
                .chain(queries[1].values())
                .copied(),
        ),
        DIM as i32,
    )
    .unwrap();
    let mut scan = dataset.scan();
    scan.nearest("vec", &both, 5)
        .unwrap()
        .minimum_nprobes(2)
        .maximum_nprobes(2);
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(!plan.contains("ANNIvfBatch"), "{plan}");
    assert!(plan.contains("ANNSubIndex"), "{plan}");
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 10);
}

#[tokio::test]
async fn an_index_is_prewarmed_and_dropped() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments_of(dir.as_str(), PQ_DIM, false).await;
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Pq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    dataset.prewarm_index(INDEX).await.unwrap();
    dataset.drop_index(INDEX).await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
}

/// Compaction rewrites the index through its remap, which carries each
/// partition's graph over by consolidation; the manifest entry keeps its
/// details and version.
#[tokio::test]
async fn compaction_remaps_the_index_and_keeps_its_details() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let before = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    dataset.delete("id % 7 = 0").await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    let raw = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap();
    assert_eq!(raw.len(), 1);
    assert_ne!(raw[0].uuid, before.uuid);
    assert_eq!(raw[0].index_version, IVF_VAMANA_INDEX_VERSION as i32);
    assert_eq!(raw[0].index_details, before.index_details);
    assert_eq!(
        dataset.describe_indices(None).await.unwrap()[0].index_type(),
        "IVF_VAMANA_SQ"
    );
    let recall = recall(&dataset, 10, DistanceType::L2).await;
    assert!(recall >= 0.95, "recall {recall}");
}

/// A Vamana segment committed with empty vector details, as a commit of
/// index metadata that does not carry them leaves it, has them inferred
/// from its files before the manifest is written: the graph parameters from
/// the partitions' metadata and the codes from the auxiliary file.
#[tokio::test]
async fn details_committed_empty_are_inferred_from_the_files() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let index = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    let cleared = IndexMetadata {
        index_details: Some(Arc::new(vector_index_details_default())),
        ..index.clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![cleared],
            removed_indices: vec![index.clone()],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let raw = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0].uuid, index.uuid);
    let details = raw[0]
        .index_details
        .as_ref()
        .unwrap()
        .to_msg::<VectorIndexDetails>()
        .unwrap();
    assert_eq!(
        details.vamana_index_config,
        vamana_details().vamana_index_config
    );
    assert_eq!(
        details.compression,
        Some(Compression::Sq(ScalarQuantization { num_bits: 8 }))
    );
    assert_eq!(
        dataset.describe_indices(None).await.unwrap()[0].index_type(),
        "IVF_VAMANA_SQ"
    );
}

/// A MemWAL shard keeps HNSW memtable indices and would flush them as
/// IVF_HNSW_SQ under an IVF_VAMANA_* index; it is refused instead.
#[tokio::test]
async fn mem_wal_refuses_to_maintain_a_vamana_index() {
    let dir = TempStrDir::default();
    let mut dataset = two_fragments(dir.as_str()).await;
    let index = dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX.into()),
            &params(Codes::Sq, DistanceType::L2),
            true,
        )
        .await
        .unwrap();
    let columns = vec!["vec".to_string()];
    let ctx = ResolveContext::new(INDEX, &dataset, &index, dataset.schema(), &columns, None);
    let error = HnswMemIndexPlugin.resolve(&ctx).await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("IVF_VAMANA"), "{error}");
}
