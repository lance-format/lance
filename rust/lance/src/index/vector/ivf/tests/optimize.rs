// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for optimizing IVF indices.

use super::*;

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
    // Regression test: `optimize_ivf_hnsw_indices` must record the on-disk
    // file sizes, which are only known after `finish()` writes the footer.
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

    // Legacy file version keeps the index on the v1 path that goes through
    // `optimize_ivf_hnsw_indices`.
    let mut params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        HnswBuildParams::default().num_edges(50),
        SQBuildParams::default(),
    );
    params.version(IndexFileVersion::Legacy);
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
