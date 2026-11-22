// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for optimizing IVF indices.

use super::*;
use lance_io::utils::{read_last_block, read_metadata_offset, read_version};

/// Building a merge filter loads a row-id sequence per covered fragment under
/// stable row ids, and an optimize pass that only appends a delta reads no existing
/// row at all. Such a pass must therefore build no filter, and a pass that does
/// merge must build one and reuse it across partitions.
#[tokio::test]
async fn test_optimize_builds_merge_filters_only_when_merging() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    let make_batch = || {
        gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(DIM as u32)),
            )
            .into_batch_rows(RowCount::from(256))
            .unwrap()
    };
    let batch = make_batch();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
        test_uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // A single partition keeps `check_partition_adjustment` from selecting a split
    // or a join, which would legitimately merge every segment.
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(make_batch())], schema),
            None,
        )
        .await
        .unwrap();

    let logical_index = dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap();
    let ivf_view = logical_index.as_ivf().unwrap();
    let new_data = |fragments| async {
        let mut scanner = dataset.scan();
        scanner
            .with_fragments(fragments)
            .with_row_id()
            .project(&["vector"])
            .unwrap();
        scanner.try_into_stream().await.unwrap()
    };
    let unindexed = dataset.unindexed_fragments("vector_idx").await.unwrap();
    assert_eq!(
        unindexed.len(),
        1,
        "the appended fragment must be unindexed"
    );

    // `ExistingIndex` shares its coverage behind an `Arc`, so the sources handed to
    // the builder report what the builder actually did with them.
    let sources = existing_index_sources(&dataset, &ivf_view);
    optimize_vector_indices_v2(
        &dataset,
        Some(new_data(unindexed.clone()).await),
        "vector",
        &sources,
        &OptimizeOptions::new(),
        None,
    )
    .await
    .unwrap();
    assert!(
        sources.iter().all(|source| !source.filter_is_built()),
        "a delta append reads no existing row, so it must build no filter"
    );

    let sources = existing_index_sources(&dataset, &ivf_view);
    optimize_vector_indices_v2(
        &dataset,
        Some(new_data(unindexed).await),
        "vector",
        &sources,
        &OptimizeOptions::merge(1),
        None,
    )
    .await
    .unwrap();
    assert!(
        sources.iter().all(|source| source.filter_is_built()),
        "a merge reads existing rows, so it must build a filter per merged segment"
    );
    assert!(
        matches!(
            sources[0].old_data_filter().await.unwrap(),
            Some(OldIndexDataFilter::RowIds(_))
        ),
        "a stable-row-id segment must filter on exact row-id membership"
    );
}

#[tokio::test]
async fn test_optimize_ivf_flat_binary_vectors() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    const BIN_DIM: usize = 16;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "bin_vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            BIN_DIM as i32,
        ),
        true,
    )]));

    let arr = arrow_array::UInt8Array::from_iter_values((0..1000 * BIN_DIM).map(|i| i as u8));
    let fsl = FixedSizeListArray::try_new_from_values(arr, BIN_DIM as i32).unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

    let params = VectorIndexParams::ivf_flat(2, MetricType::Hamming);
    dataset
        .create_index(&["bin_vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Append more data so optimize_indices has unindexed fragments to merge
    let arr2 = arrow_array::UInt8Array::from_iter_values((0..500 * BIN_DIM).map(|i| (i + 7) as u8));
    let fsl2 = FixedSizeListArray::try_new_from_values(arr2, BIN_DIM as i32).unwrap();
    let batch2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl2)]).unwrap();
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch2])
        .await
        .unwrap();

    // This used to panic with "unsupported index type: FLAT, FLATBIN"
    dataset.optimize_indices(&Default::default()).await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty(), "should have at least one index");
}

#[tokio::test]
async fn test_optimize_ivf_hnsw_records_actual_file_sizes() {
    // Regression test: the current-format IVF_HNSW_SQ builder must record the
    // on-disk file sizes, which are only known after `finish()` writes the
    // footer. `IndexFileVersion::Legacy` does not apply to HNSW, so the index
    // must stay on the v2 path this covers.
    const DIM: usize = 16;
    let data = gen_batch()
        .col(
            "vec",
            array::rand_vec::<Float32Type>(Dimension::from(DIM as u32)),
        )
        .into_batch_rows(RowCount::from(1_000))
        .unwrap();
    let schema = data.schema();

    let mut dataset = InsertBuilder::new("memory://")
        .execute(vec![data])
        .await
        .unwrap();

    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        HnswBuildParams::default().num_edges(50),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Append unindexed data so `optimize_indices` has something to merge.
    let more = gen_batch()
        .col(
            "vec",
            array::rand_vec::<Float32Type>(Dimension::from(DIM as u32)),
        )
        .into_batch_rows(RowCount::from(500))
        .unwrap();
    let more = RecordBatch::try_new(schema.clone(), more.columns().to_vec()).unwrap();
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![more])
        .await
        .unwrap();

    dataset.optimize_indices(&Default::default()).await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let index = indices.first().expect("should have an index");
    let files = index
        .files
        .as_ref()
        .expect("optimized index should record file sizes");
    assert!(!files.is_empty(), "index should record at least one file");

    let indices_dir = dataset.indices_dir().join(index.uuid.to_string());
    for file in files {
        let actual = dataset
            .object_store
            .size(&indices_dir.clone().join(file.path.as_str()))
            .await
            .unwrap();
        assert_eq!(
            file.size_bytes, actual,
            "recorded size for {} must match the on-disk size",
            file.path
        );
    }
}

/// A released 0.11.0 IVF_HNSW_SQ index is a legacy (0, 2) file. There is no
/// incremental legacy HNSW writer, so optimize must refuse it before creating
/// any output or publishing a version. Retraining is not a recovery here, so
/// the error directs users to drop the segment and rebuild from the data.
#[tokio::test]
async fn test_optimize_legacy_hnsw_sq_refuses_before_writing() {
    let test_dir = copy_test_data_to_tmp("v0.11.0/legacy_hnsw_sq").unwrap();
    let uri = test_dir.path_str();
    let mut dataset = Dataset::open(&uri).await.unwrap();
    let segments = dataset.load_indices().await.unwrap();
    assert_eq!(segments.len(), 1);
    let legacy_uuid = segments[0].uuid;
    assert_eq!(
        index_file_version(&dataset, &legacy_uuid).await,
        (0, 2),
        "the fixture must remain a legacy v1 index file"
    );
    let legacy = dataset
        .open_vector_index("vector", &legacy_uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        legacy.as_any().is::<IVFIndex>(),
        "the fixture is a v1 IVFIndex"
    );
    assert!(matches!(
        legacy.sub_index_type(),
        (SubIndexType::Hnsw, QuantizationType::Scalar)
    ));
    assert_eq!(
        dataset
            .unindexed_fragments("vector_idx")
            .await
            .unwrap()
            .len(),
        1,
        "the fixture must leave an unindexed fragment for optimize"
    );

    // The legacy file must still serve its indexed rows before any optimize.
    let old_vector = vector_for_id(&dataset, 0).await;
    assert!(
        nearest_ids(&dataset, &old_vector, 10).await.contains(&0),
        "the legacy index must still read its old rows"
    );

    let version = dataset.version().version;
    let index_dir_count = || {
        std::fs::read_dir(test_dir.std_path().join("_indices"))
            .unwrap()
            .count()
    };
    let index_dirs_before = index_dir_count();
    let error = dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::NotSupported { .. }),
        "expected NotSupported, got {error:?}"
    );
    let message = error.to_string();
    assert!(message.contains("IVF_HNSW_SQ"), "{message}");
    assert!(
        message.contains("drop_index") && message.contains("create_index"),
        "the error must say how to rebuild the index: {message}"
    );
    assert!(
        !message.contains("retrain"),
        "retraining this released fixture fails, so the error must not suggest it: {message}"
    );
    assert_eq!(
        dataset.version().version,
        version,
        "the rejected optimize must not publish a version"
    );
    assert_eq!(dataset.load_indices().await.unwrap()[0].uuid, legacy_uuid);
    assert_eq!(
        index_file_version(&dataset, &legacy_uuid).await,
        (0, 2),
        "the rejected optimize must leave the legacy file untouched"
    );
    assert_eq!(
        index_dir_count(),
        index_dirs_before,
        "the rejected optimize must not allocate an index output directory"
    );

    // Dropping the legacy segment removes it from the rebuild's source scan.
    // Recreating from the data then includes the appended fragment in a
    // current-format segment.
    dataset.drop_index("vector_idx").await.unwrap();
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let rebuilt = dataset.load_indices().await.unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert_ne!(rebuilt[0].uuid, legacy_uuid);
    assert!(
        matches!(
            index_file_version(&dataset, &rebuilt[0].uuid).await,
            (0, 3) | (2, _)
        ),
        "the rebuilt index must use the current file format"
    );
    assert!(
        dataset
            .unindexed_fragments("vector_idx")
            .await
            .unwrap()
            .is_empty(),
        "the rebuilt index must cover the appended fragment"
    );
    let appended_vector = vector_for_id(&dataset, 319).await;
    assert!(
        nearest_ids(&dataset, &appended_vector, 10)
            .await
            .contains(&319),
        "the rebuilt index must serve an appended row"
    );

    // The rebuilt format and the appended row must survive reopen.
    let reopened = Dataset::open(&uri).await.unwrap();
    let reopened_segments = reopened.load_indices().await.unwrap();
    assert_eq!(reopened_segments.len(), 1);
    assert!(matches!(
        index_file_version(&reopened, &reopened_segments[0].uuid).await,
        (0, 3) | (2, _)
    ));
    let appended_vector = vector_for_id(&reopened, 319).await;
    assert!(
        nearest_ids(&reopened, &appended_vector, 10)
            .await
            .contains(&319),
        "the reopened index must serve an appended row"
    );
}

#[tokio::test]
async fn test_optimize_ivf_hnsw_binary_vectors() {
    const BIN_DIM: usize = 16;
    const BASE_ROWS: usize = 1000;
    const K: usize = 10;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "bin_vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            BIN_DIM as i32,
        ),
        true,
    )]));

    // Distinct pseudo-random bytes (splitmix64): repeated or regular rows
    // would tie the Hamming distances the recall check compares.
    let write_batch = |rows: Range<usize>| {
        let arr = arrow_array::UInt8Array::from_iter_values(
            (rows.start * BIN_DIM..rows.end * BIN_DIM).map(|i| {
                let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 56) as u8
            }),
        );
        let fsl = FixedSizeListArray::try_new_from_values(arr, BIN_DIM as i32).unwrap();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap()
    };

    let batch = write_batch(0..BASE_ROWS);
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, "memory://", None).await.unwrap();

    let params = VectorIndexParams::ivf_hnsw(
        MetricType::Hamming,
        IvfBuildParams::new(1),
        HnswBuildParams::default(),
    );
    dataset
        .create_index(&["bin_vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Append more data so optimize_indices has unindexed fragments to merge
    let appended = write_batch(BASE_ROWS..BASE_ROWS + 500);
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended.clone()])
        .await
        .unwrap();

    // This used to panic with "unsupported index type: Hnsw, FlatBin"
    dataset.optimize_indices(&Default::default()).await.unwrap();

    let index_name = dataset.load_indices().await.unwrap()[0].name.clone();
    assert!(
        dataset
            .unindexed_fragments(&index_name)
            .await
            .unwrap()
            .is_empty(),
        "the appended rows must be merged into the index"
    );

    // An appended row as the query, so the merged rows must answer it.
    let query = appended["bin_vector"].as_fixed_size_list().value(0);
    let search = |use_index: bool| {
        let dataset = dataset.clone();
        let query = query.clone();
        async move {
            let mut scanner = dataset.scan();
            scanner
                .nearest("bin_vector", query.as_ref(), K)
                .unwrap()
                .use_index(use_index)
                .with_row_id();
            let plan = scanner.explain_plan(false).await.unwrap();
            let result = scanner.try_into_batch().await.unwrap();
            (plan, result[ROW_ID].as_primitive::<UInt64Type>().clone())
        }
    };
    let (plan, ids) = search(true).await;
    assert!(plan.contains("ANNIvfPartition"), "{plan}");
    assert_eq!(ids.len(), K);
    // The query's own row, offset 0 of the appended fragment 1, is served
    // by the merged index.
    assert!(ids.values().contains(&(1u64 << 32)), "row ids: {ids:?}");
    let (_, expected_ids) = search(false).await;
    let recall = ids
        .values()
        .iter()
        .filter(|id| expected_ids.values().contains(id))
        .count() as f32
        / K as f32;
    assert!(recall >= 0.5, "recall: {recall}, row ids: {ids:?}");
}

#[rstest::rstest]
#[case::pq8(8)]
#[case::pq4(4)]
#[tokio::test]
async fn test_optimize_ivf_pq_legacy_segments(#[case] num_bits: u8) {
    // The shared-model validation in `optimize_vector_indices` runs before
    // the v1/v2 dispatch and inspects every segment, so a legacy
    // (v1-format) IVF_PQ segment used to panic in `sub_index_type()` /
    // `quantizer()` instead of being merged like any other segment.
    const PQ_DIM: usize = 8;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            PQ_DIM as i32,
        ),
        true,
    )]));

    let write_batch = |rows: usize, seed: u8| {
        let fsl = FixedSizeListArray::try_new_from_values(
            generate_random_array_with_seed::<Float32Type>(rows * PQ_DIM, [seed; 32]),
            PQ_DIM as i32,
        )
        .unwrap();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap()
    };

    let batch = write_batch(512, 1);
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, "memory://", None).await.unwrap();

    // A genuine legacy (0, 1) index file, as shipped Lance versions wrote.
    let mut params = VectorIndexParams::ivf_pq(2, num_bits, 4, MetricType::L2, 10);
    params.version(IndexFileVersion::Legacy);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Append unindexed data so optimize_indices has a delta to merge.
    let appended = write_batch(256, 2);
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![appended.clone()])
        .await
        .unwrap();

    // This used to panic with "not implemented: only for v2 IVFIndex".
    dataset.optimize_indices(&Default::default()).await.unwrap();

    // The merged legacy index covers the appended rows and serves ANN.
    let indices = dataset.load_indices().await.unwrap();
    let path = dataset
        .indices_dir()
        .join(indices[0].uuid.to_string())
        .join(INDEX_FILE_NAME);
    let reader = dataset.object_store.open(&path).await.unwrap();
    let footer = read_last_block(reader.as_ref()).await.unwrap();
    assert_eq!(
        read_metadata_offset(&footer).unwrap(),
        (512 + 256) * (4 * num_bits as usize / 8 + 8),
        "the merged file stores every row at the packed width"
    );
    let index = dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        index.as_any().is::<IVFIndex>(),
        "the merge keeps the format"
    );
    assert!(
        dataset
            .unindexed_fragments(&indices[0].name)
            .await
            .unwrap()
            .is_empty(),
        "the appended rows must be merged into the legacy index"
    );
    const K: usize = 10;
    // An appended row as the query, so the merged rows must answer it.
    let query = appended["vector"].as_fixed_size_list().value(0);
    let search = |use_index: bool| {
        let dataset = dataset.clone();
        let query = query.clone();
        async move {
            let mut scanner = dataset.scan();
            scanner
                .nearest("vector", query.as_ref(), K)
                .unwrap()
                // Both partitions, so recall measures the merged codes.
                .minimum_nprobes(2)
                .use_index(use_index)
                .with_row_id();
            let plan = scanner.explain_plan(false).await.unwrap();
            let result = scanner.try_into_batch().await.unwrap();
            (plan, result[ROW_ID].as_primitive::<UInt64Type>().clone())
        }
    };
    let (plan, ids) = search(true).await;
    assert!(plan.contains("ANNIvfPartition"), "{plan}");
    assert_eq!(ids.len(), K);
    // Offset 0 of the appended fragment 1.
    assert!(ids.values().contains(&(1u64 << 32)), "row ids: {ids:?}");
    let (_, expected_ids) = search(false).await;
    let recall = ids
        .values()
        .iter()
        .filter(|id| expected_ids.values().contains(id))
        .count() as f32
        / K as f32;
    assert!(recall >= 0.5, "recall: {recall}, row ids: {ids:?}");

    // Steady state (no unindexed fragments) must skip gracefully.
    let version = dataset.version().version;
    dataset.optimize_indices(&Default::default()).await.unwrap();
    assert_eq!(dataset.version().version, version);

    // Retrain is the documented way to move a legacy index to the current
    // format; it derives the rebuild parameters from the legacy segment.
    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let index = dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        !index.as_any().is::<IVFIndex>(),
        "retrain must rebuild the legacy index in the current format"
    );
}

/// The legacy PQ writer merges by copying every existing posting. With
/// stable row ids an updated row keeps its id in its new fragment, so a
/// merge kept the row's old codes: a query for its old vector found the
/// updated row at distance ~0.
#[tokio::test]
async fn test_optimize_legacy_ivf_pq_with_stable_row_ids_skips_merge() {
    const PQ_DIM: usize = 8;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                PQ_DIM as i32,
            ),
            true,
        ),
    ]));
    let vectors = generate_random_array_with_seed::<Float32Type>(512 * PQ_DIM, [7; 32]);
    let vectors = FixedSizeListArray::try_new_from_values(vectors, PQ_DIM as i32).unwrap();
    let old_vector = vectors.value(0);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int32Array::from_iter_values(0..512)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
        "memory://",
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);
    params.version(IndexFileVersion::Legacy);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Move row 0 to a new fragment with a far-away vector.
    let far = Float32Array::from(vec![50.0f32; PQ_DIM]);
    let update = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int32Array::from(vec![0])),
            Arc::new(FixedSizeListArray::try_new_from_values(far, PQ_DIM as i32).unwrap()),
        ],
    )
    .unwrap();
    let (dataset, _) =
        crate::dataset::MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .try_build()
            .unwrap()
            .execute(lance_datafusion::utils::reader_to_stream(Box::new(
                RecordBatchIterator::new(vec![Ok(update)], schema.clone()),
            )))
            .await
            .unwrap();
    let mut dataset = dataset.as_ref().clone();

    dataset.optimize_indices(&Default::default()).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        dataset
            .unindexed_fragments(&indices[0].name)
            .await
            .unwrap()
            .is_empty()
    );
    let mut scanner = dataset.scan();
    scanner
        .nearest("vector", old_vector.as_ref(), 1)
        .unwrap()
        .project(&["id"])
        .unwrap();
    let nearest = scanner.try_into_batch().await.unwrap();
    assert_ne!(
        nearest["id"]
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        0,
        "the updated row must not be served by its old vector"
    );
    // The default optimize appended a segment instead of merging.
    assert_eq!(indices.len(), 2);

    let error = dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
    assert!(error.to_string().contains("stable row ids"), "{error}");

    // Retrain, which the error suggests, rebuilds one current-format segment.
    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let index = dataset
        .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!index.as_any().is::<IVFIndex>());
    let mut scanner = dataset.scan();
    scanner
        .nearest("vector", old_vector.as_ref(), 1)
        .unwrap()
        .project(&["id"])
        .unwrap();
    let nearest = scanner.try_into_batch().await.unwrap();
    assert_ne!(
        nearest["id"]
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        0
    );
}

/// A legacy segment is never a rebalance candidate, but it must not stop
/// the current-format segments of the same index from being split.
#[tokio::test]
async fn test_steady_state_splits_v3_segment_beside_legacy_segment() {
    const PQ_DIM: usize = 8;
    const ROWS: usize = 512;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            PQ_DIM as i32,
        ),
        true,
    )]));
    let batch = |seed: u8| {
        let values = generate_random_array_with_seed::<Float32Type>(ROWS * PQ_DIM, [seed; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(values, PQ_DIM as i32).unwrap();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch(1))], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let mut params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);
    params.version(IndexFileVersion::Legacy);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let legacy_uuid = dataset.load_indices().await.unwrap()[0].uuid;

    // A current-format segment over a new fragment, as a per-fragment
    // build commits it. Values lie in [0, 1), so all 512 rows land in the
    // partition at 0.5, far above four times the target size of 4.
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch(2)])
        .await
        .unwrap();
    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(
            std::iter::repeat_n(0.5, PQ_DIM).chain(std::iter::repeat_n(100.0, PQ_DIM)),
        ),
        PQ_DIM as i32,
    )
    .unwrap();
    let mut ivf = IvfBuildParams::try_with_centroids(2, Arc::new(centroids)).unwrap();
    ivf.target_partition_size = Some(4);
    // A per-fragment build needs a shared codebook; reuse the legacy one.
    let legacy = dataset
        .open_vector_index("vector", &legacy_uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let Quantizer::Product(pq) = legacy.quantizer() else {
        panic!("a legacy IVF_PQ segment reports a product quantizer");
    };
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        ivf,
        PQBuildParams::with_codebook(2, 8, Arc::new(pq.codebook.values().clone())),
    );
    let segment = crate::index::CreateIndexBuilder::new(
        &mut dataset,
        &["vector"],
        IndexType::Vector,
        &params,
    )
    .name("vector_idx".into())
    .replace(true)
    .fragments(vec![1])
    .execute_uncommitted()
    .await
    .unwrap();
    let v3_uuid = segment.uuid;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![segment])
        .await
        .unwrap();
    let segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(
        segments.iter().map(|s| s.uuid).collect::<HashSet<_>>(),
        HashSet::from([legacy_uuid, v3_uuid])
    );

    let logical = dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap();
    let view = logical.as_ivf().unwrap();
    let compatibility = vector_segment_compatibility(&view, "test").unwrap();
    assert_eq!(
        select_steady_state_rebalance(&view, compatibility).unwrap(),
        Some(SteadyStateRebalance::Segment(v3_uuid))
    );

    dataset.optimize_indices(&Default::default()).await.unwrap();
    let segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(segments.len(), 2);
    assert!(segments.iter().any(|s| s.uuid == legacy_uuid));
    let rebuilt = segments.iter().find(|s| s.uuid != legacy_uuid).unwrap();
    assert_ne!(rebuilt.uuid, v3_uuid);
    let rebuilt = dataset
        .open_vector_index("vector", &rebuilt.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        rebuilt.ivf_model().num_partitions() > 2,
        "the oversized partition must be split"
    );
}

async fn index_file_version(dataset: &Dataset, uuid: &Uuid) -> (u16, u16) {
    let path = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join(INDEX_FILE_NAME);
    let reader = dataset.object_store.open(&path).await.unwrap();
    let footer = read_last_block(reader.as_ref()).await.unwrap();
    read_version(&footer).unwrap()
}

async fn vector_for_id(dataset: &Dataset, id: i64) -> Float32Array {
    let batch = dataset
        .scan()
        .project(&["id", "vector"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["id"].as_primitive::<arrow_array::types::Int64Type>();
    let position = ids
        .values()
        .iter()
        .position(|value| *value == id)
        .unwrap_or_else(|| panic!("row id {id} is missing"));
    batch["vector"]
        .as_fixed_size_list()
        .value(position)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .clone()
}

async fn nearest_ids(dataset: &Dataset, query: &Float32Array, k: usize) -> Vec<i64> {
    let mut scanner = dataset.scan();
    scanner
        .nearest("vector", query, k)
        .unwrap()
        .minimum_nprobes(2)
        .use_index(true)
        .project(&["id"])
        .unwrap();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNIvfPartition"), "{plan}");
    let result = scanner.try_into_batch().await.unwrap();
    result["id"]
        .as_primitive::<arrow_array::types::Int64Type>()
        .values()
        .to_vec()
}
