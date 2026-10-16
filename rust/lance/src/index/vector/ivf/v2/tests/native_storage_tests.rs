// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IvfOpenContext;
use super::*;
use arrow::compute::concat_batches;
use arrow_array::UInt32Array;
use lance_arrow::RecordBatchExt;
use lance_core::Error;
use lance_core::cache::CacheCodec;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::planes::PlaneBatch;
use lance_index::vector::bq::planes::{EntryColumns, plane_columns};
use lance_index::vector::storage::{
    ExIndex, GatherPlan, GatheredRabitEx, OriginLatencyClass, PlaneSource, ResidentAttach,
};
use lance_index::vector::{ApproxMode, VECTOR_RESULT_SCHEMA};
use lance_io::scheduler::IoStats;

type IvfRq = super::super::IVFIndex<FlatIndex, RabitQuantizer>;

pub(super) fn lazy_test_query(key: ArrayRef, k: usize, nprobes: usize) -> Query {
    Query {
        search_effort: 0.5,
        column: "vector".to_string(),
        key,
        k,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: nprobes,
        maximum_nprobes: Some(nprobes),
        ef: None,
        refine_factor: None,
        metric_type: None,
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: ApproxMode::Normal,
        rq_precision: Default::default(),
        rq_cascade_factor: None,
    }
}

pub(super) async fn search_global(
    index: &Arc<dyn VectorIndex>,
    query: &Query,
    filter: Arc<dyn PreFilter>,
) -> Result<RecordBatch> {
    let (partitions, dists) = index.find_partitions(query)?;
    let probes = partitions.len();
    let batches = index
        .clone()
        .search_partitions(
            query.clone(),
            Arc::new(partitions),
            Arc::new(dists),
            0,
            probes,
            filter,
            None,
            Arc::new(NoOpMetricsCollector),
        )
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(concat_batches(&VECTOR_RESULT_SCHEMA, batches.iter())?)
}

pub(super) fn result_bits(batch: &RecordBatch) -> (Vec<u64>, Vec<u32>) {
    (
        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec(),
        batch[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|dist| dist.to_bits())
            .collect(),
    )
}

/// A real current native index in four fragments, with explicit centroids so
/// the fixture spends no time training IVF clusters.
pub(super) async fn native_fixture(uri: &str) -> (Dataset, RecordBatch) {
    let batch = gen_batch()
        .with_seed(Seed::from(42))
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .into_batch_rows(RowCount::from(256))
        .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let vectors = batch["vector"].as_fixed_size_list();
    let params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(4, Arc::new(vectors.slice(0, 4))).unwrap(),
        RQBuildParams::new(7),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);
    (dataset, batch)
}

async fn native_index(
    dataset: &Dataset,
    resident: bool,
    attach: ResidentAttach,
    columns: EntryColumns,
) -> IvfRq {
    let index = dataset.load_indices().await.unwrap()[0].clone();
    IvfRq::try_new(
        dataset.object_store.clone(),
        dataset.indice_files_dir(&index).unwrap(),
        index.uuid,
        None,
        &dataset.metadata_cache,
        dataset.index_cache.for_index(&index.uuid, None),
        index.file_size_map(),
        IvfOpenContext::default(),
    )
    .await
    .unwrap()
    .with_resident_columns_for_test(resident)
    .await
    .unwrap()
    .with_resident_attach_for_test(attach)
    .with_entry_columns_for_test(columns)
    .with_origin_latency_for_test(OriginLatencyClass::Low)
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn native_planes_and_gathers_match_full_reads(
    #[values(EntryColumns::All, EntryColumns::Codes)] columns: EntryColumns,
    #[values(ResidentAttach::Copy, ResidentAttach::Share)] attach: ResidentAttach,
) {
    let dir = TempStrDir::default();
    let (dataset, batch) = native_fixture(dir.as_str()).await;
    let file = native_index(&dataset, false, ResidentAttach::Copy, EntryColumns::All).await;
    let resident = native_index(&dataset, true, attach, columns).await;
    let strong_cache = LanceCache::with_capacity(4 * 1024 * 1024);
    let cache = WeakLanceCache::from(&strong_cache);
    for part in 0..file.storage.num_partitions() {
        let expected = file.storage.load_partition(part, None).await.unwrap();
        let expected = expected.to_batches().unwrap().next().unwrap();
        let full = resident
            .storage
            .load_rabit_partition_planes(part, Default::default(), &cache, None)
            .await
            .unwrap();
        assert_eq!(full.to_batches().unwrap().next().unwrap(), expected);
        for plane in [0, 1] {
            let indices = plane_columns(plane)
                .iter()
                .map(|name| expected.schema().index_of(name).unwrap())
                .collect::<Vec<_>>();
            let projected = expected.project(&indices).unwrap();
            let actual = resident
                .storage
                .load_plane(part, plane, &cache, None)
                .await
                .unwrap();
            assert_eq!(actual, projected);
            let entry = resident
                .storage
                .read_plane_entry(part, plane, None)
                .await
                .unwrap();
            let codec = CacheCodec::from_impl::<PlaneBatch>();
            let mut bytes = Vec::new();
            codec
                .serialize(
                    &(Arc::new(PlaneBatch(entry)) as Arc<dyn std::any::Any + Send + Sync>),
                    &mut bytes,
                )
                .unwrap();
            let decoded = codec
                .deserialize(&bytes::Bytes::from(bytes))
                .hit()
                .unwrap()
                .downcast::<PlaneBatch>()
                .unwrap();
            let restored = resident
                .storage
                .attach_resident(part, plane, &decoded.0, None, None)
                .await
                .unwrap();
            assert_eq!(restored, projected);
        }
        let rows: Vec<u32> = (0..expected.num_rows() as u32).step_by(3).collect();
        let ex_columns = plane_columns(1)
            .iter()
            .map(|name| expected.schema().index_of(name).unwrap())
            .collect::<Vec<_>>();
        let ex = expected.project(&ex_columns).unwrap();
        let gathered = resident
            .storage
            .gather_rabit_ex_rows(
                part,
                &rows,
                GatherPlan {
                    planes: [PlaneSource::Sparse, PlaneSource::Resident],
                    origin_whole: [false; 2],
                },
                Some(0),
                &cache,
                None,
            )
            .await
            .unwrap();
        let GatheredRabitEx::Native { batch, index, .. } = &gathered else {
            panic!("native storage returned layered ex rows")
        };
        assert_eq!(*index, ExIndex::Compact);
        assert_eq!(batch, &ex.take(&UInt32Array::from(rows)).unwrap());
        assert!(!lance_index::vector::storage::shares_resident_store(batch));
        assert!(gathered.sources()[1].is_none());
        let empty_io = IoStats::new();
        let empty = resident
            .storage
            .gather_rabit_ex_rows(
                part,
                &[],
                GatherPlan {
                    planes: [PlaneSource::Sparse; 2],
                    origin_whole: [false; 2],
                },
                None,
                &cache,
                Some(empty_io.clone()),
            )
            .await
            .unwrap();
        assert_eq!(empty.origin_row_reads(), 0);
        assert_eq!(empty_io.snapshot().iops, 0);
    }
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 8, 4);
    let truth =
        super::ground_truth(&dataset, "vector", query.key.as_ref(), 8, DistanceType::L2).await;
    let file: Arc<dyn VectorIndex> = Arc::new(file);
    let resident: Arc<dyn VectorIndex> = Arc::new(resident);
    let expected = result_bits(
        &search_global(&file, &query, Arc::new(NoFilter))
            .await
            .unwrap(),
    );
    let actual = result_bits(
        &search_global(&resident, &query, Arc::new(NoFilter))
            .await
            .unwrap(),
    );
    assert_eq!(actual, expected);
    assert!(
        actual.0.iter().filter(|row| truth.contains(row)).count() >= 4,
        "recall must be >= 0.5"
    );
}

#[tokio::test]
async fn native_gather_rejects_invalid_offsets_without_io() {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let index = native_index(&dataset, false, ResidentAttach::Copy, EntryColumns::All).await;
    let part = (0..index.storage.num_partitions())
        .find(|&part| index.storage.partition_size(part) > 2)
        .unwrap();
    for rows in [
        vec![1, 1],
        vec![2, 0],
        vec![index.storage.partition_size(part) as u32],
    ] {
        let io = IoStats::new();
        let error = index
            .storage
            .gather_rabit_ex_rows(
                part,
                &rows,
                GatherPlan {
                    planes: [PlaneSource::Sparse; 2],
                    origin_whole: [false; 2],
                },
                None,
                &index.index_cache,
                Some(io.clone()),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("sorted, unique and below"),
            "{error}"
        );
        assert_eq!(io.snapshot().iops, 0);
    }
}
