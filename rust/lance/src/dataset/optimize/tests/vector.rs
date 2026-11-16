// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the vector contract area.

use super::*;

/// A merge translates its sources through the fragment-reuse index, so the merged segment
/// is caught up with the manifest it read and must be stamped with it. Stamped with the
/// oldest source instead, the trim keeps every reuse version that source predates, forever.
#[tokio::test]
async fn test_segment_merge_is_stamped_with_the_version_it_was_built_from() {
    use lance_index::optimize::OptimizeOptions;

    let test_dir = TempStrDir::default();
    let uri = test_dir.as_str().to_string();
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(4_000),
        &uri,
        Some(WriteParams {
            max_rows_per_file: 1_000,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // A deferred-remap compaction leaves a reuse version that the base segment predates.
    dataset.delete("id % 10 == 0").await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let reuse_meta = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("a deferred-remap compaction leaves a reuse index");
    let reuse_details = load_frag_reuse_index_details(&dataset, &reuse_meta)
        .await
        .unwrap();
    assert_eq!(reuse_details.versions.len(), 1);
    let base_version = dataset.load_indices_by_name("id_idx").await.unwrap()[0].dataset_version;
    assert!(
        base_version < reuse_details.versions[0].dataset_version,
        "the base segment must predate the reuse version for the stamp to matter"
    );

    // New rows for the merge to fold into the base segment.
    Dataset::write(
        data_gen.batch(1_000),
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Some(WriteParams {
            max_rows_per_file: 1_000,
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();
    let filter = "id >= 1500 and id < 4500".to_owned();
    let mut oracle = dataset.scan();
    oracle.filter(&filter).unwrap();
    oracle.use_scalar_index(false);
    let expected = oracle.count_rows().await.unwrap();

    let built_from = dataset.manifest.version;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let merged = dataset.load_indices_by_name("id_idx").await.unwrap();
    assert_eq!(
        merged.len(),
        1,
        "the new rows must merge into the base segment"
    );
    assert_eq!(
        merged[0].dataset_version, built_from,
        "a merged segment is caught up with the manifest it was built from"
    );

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    dataset.checkout_latest().await.unwrap();
    let reuse_meta = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("the reuse index survives cleanup, trimmed to zero versions");
    let details = load_frag_reuse_index_details(&dataset, &reuse_meta)
        .await
        .unwrap();
    assert_eq!(
        details.versions.len(),
        0,
        "the trim must drop the reuse version the merged segment is past"
    );

    // Nothing translates addresses any more, so only a correct merge answers right, and the
    // plan must show the merged index answering for every fragment rather than a scan.
    let fresh = Dataset::open(&uri).await.unwrap();
    let mut scanner = fresh.scan();
    scanner.filter(&filter).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery"),
        "the merged index must answer the query: {plan}"
    );
    assert!(
        !plan.contains("LanceScan"),
        "no fragment may fall back to a scan: {plan}"
    );
    assert_eq!(scanner.count_rows().await.unwrap(), expected);
}
#[rstest]
#[case(IndexRemapMode::Compact)]
#[case(IndexRemapMode::Direct)]
#[tokio::test]
async fn test_ivf_pq_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
    use arrow_array::cast::AsArray;
    use lance_index::vector::pq::PQBuildParams;

    const DIM: u32 = 32;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        small_ivf(),
        PQBuildParams {
            max_iters: 2,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let original_uuid = dataset
        .load_index_by_name("vec_idx")
        .await
        .unwrap()
        .unwrap()
        .uuid;

    // Delete rows scattered across fragments so the remap must drop some old
    // addresses and shift the survivors.
    dataset.delete("id % 10 == 0").await.unwrap();

    // Sample queries from surviving vectors and capture the pre-compaction
    // KNN answer and the surviving id set.
    let mut survivors: Vec<(i32, Vec<f32>)> = Vec::new();
    {
        let mut scanner = dataset.scan();
        scanner.project(&["id", "vec"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for batch in &batches {
            let ids = batch["id"].as_primitive::<Int32Type>();
            let vecs = batch["vec"].as_fixed_size_list();
            for i in 0..batch.num_rows() {
                let v = vecs.value(i);
                survivors.push((
                    ids.value(i),
                    v.as_primitive::<Float32Type>().values().to_vec(),
                ));
            }
        }
    }
    let surviving_ids: std::collections::HashSet<i32> =
        survivors.iter().map(|(id, _)| *id).collect();
    let step = (survivors.len() / 16).max(1);
    let queries: Vec<Vec<f32>> = survivors
        .iter()
        .step_by(step)
        .map(|(_, v)| v.clone())
        .collect();
    let k = 10;
    let mut baseline: Vec<Vec<i32>> = Vec::new();
    for q in &queries {
        baseline.push(vector_knn_ids(&dataset, q, k).await);
    }

    // Inline remap (defer_index_remap = false): compaction physically
    // rebuilds the vector index through the configured remap mode.
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 50_000,
            index_remap_mode,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // The index was physically remapped inline, so its uuid must change.
    assert_ne!(
        dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid,
        original_uuid,
        "vector index must be physically remapped inline"
    );

    // The remap only relabels row addresses; it must not resurrect deleted
    // rows, and KNN must stay close to the pre-compaction answer in both
    // remap modes.
    for (i, q) in queries.iter().enumerate() {
        let after = vector_knn_ids(&dataset, q, k).await;
        for id in &after {
            assert!(
                surviving_ids.contains(id),
                "KNN returned id {id} that is not a surviving row (query #{i}, mode {index_remap_mode:?})"
            );
        }
        let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
        assert!(
            overlap >= 8,
            "KNN top-{k} diverged after compaction: overlap {overlap} < 8 (query #{i}, mode {index_remap_mode:?})"
        );
    }
}
#[tokio::test]
async fn test_read_ivf_pq_index_v3_with_defer_index_remap() {
    // Create a dataset with vector data
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Get some query vectors for KNN search
    let query_vec1: PrimitiveArray<Float32Type> =
        PrimitiveArray::from_iter_values(std::iter::repeat_n(0.0, 128));
    let query_vec2: PrimitiveArray<Float32Type> =
        PrimitiveArray::from_iter_values(std::iter::repeat_n(1.1, 128));
    let query_vec3: PrimitiveArray<Float32Type> =
        PrimitiveArray::from_iter_values(std::iter::repeat_n(2.2, 128));

    // Get initial KNN search results
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &query_vec1, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let results1 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let count1 = results1.len();

    scanner = dataset.scan();
    scanner.nearest("vec", &query_vec2, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let results2 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let count2 = results2.len();

    scanner = dataset.scan();
    scanner.nearest("vec", &query_vec3, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let results3 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let count3 = results3.len();

    // Create an IVF-PQ index on the vec column
    let index_name = Some("vec_idx".into());
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            index_name.clone(),
            &VectorIndexParams {
                metric_type: DistanceType::L2,
                stages: vec![
                    StageParams::Ivf(IvfBuildParams {
                        max_iters: 2,
                        num_partitions: Some(2),
                        sample_rate: 2,
                        ..Default::default()
                    }),
                    StageParams::PQ(PQBuildParams {
                        max_iters: 2,
                        num_sub_vectors: 2,
                        ..Default::default()
                    }),
                ],
                version: crate::index::vector::IndexFileVersion::V3,
                skip_transpose: false,
                runtime_hints: Default::default(),
            },
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify the index UUID is unchanged (it should not be remapped yet)
    let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
        panic!("vec index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    // Verify that KNN searches still work correctly and return the same counts
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &query_vec1, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let new_results1 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(new_results1.len(), count1);

    scanner = dataset.scan();
    scanner.nearest("vec", &query_vec2, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let new_results2 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(new_results2.len(), count2);

    scanner = dataset.scan();
    scanner.nearest("vec", &query_vec3, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let new_results3 = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(new_results3.len(), count3);

    // Verify that after index creation and compaction, scan uses vector index scan
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &query_vec1, 10).unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ANNSubIndex"),
        "Expected vector index scan in plan: {}",
        plan
    );
    assert!(
        !plan.contains("LanceScan"),
        "Expected no fragment scan in plan: {}",
        plan
    );
}
#[rstest]
#[case::builtin(0, 1)]
#[case::ignore(1, 2)]
#[case::custom(2, 2)]
#[tokio::test]
async fn test_compaction_selected_remapper_boundaries(
    #[case] remapper_kind: usize,
    #[case] expected_segments: usize,
) {
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
        )
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(8))
        .await
        .unwrap();
    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 4]), 4).unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let mut segments = Vec::new();
    for pair in [[0, 1], [2, 3]] {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(pair.to_vec())
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();
    let remapper: Arc<dyn IndexRemapperOptions> = match remapper_kind {
        0 => Arc::new(DatasetIndexRemapperOptions::default()),
        1 => Arc::new(IgnoreRemap::default()),
        _ => Arc::new(IndependentRemapperOptions),
    };
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    assert_eq!(
        plan_compaction(&dataset, &options)
            .await
            .unwrap()
            .num_tasks(),
        2
    );
    if remapper_kind != 0 {
        let crossing = DefaultCompactionPlanner::new(options.clone())
            .unwrap()
            .with_index_remapper(&DatasetIndexRemapperOptions::default());
        let version = dataset.version().version;
        let error = compact_files_with_planner(&mut dataset, Some(remapper.clone()), &crossing)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("remapper/publication path"));
        assert_eq!(dataset.version().version, version);
    }
    compact_files(&mut dataset, options, Some(remapper))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), expected_segments);
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        expected_segments
    );
    if remapper_kind != 1 {
        assert_vector_index_coverage(&dataset).await;
    }
}
#[rstest]
#[case::noop(1)]
#[case::rewrite(10_000)]
#[tokio::test]
async fn test_compaction_legacy_segments(#[case] target_rows_per_fragment: usize) {
    let dir =
        crate::utils::test::copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    dataset.index_statistics("vector_idx").await.unwrap();
    dataset.checkout_latest().await.unwrap();
    let rows = dataset.count_rows(None).await.unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment,
        ..Default::default()
    };
    DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(&dataset)
        .await
        .unwrap();
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
    assert_eq!(
        dataset
            .load_indices_by_name("vector_idx")
            .await
            .unwrap()
            .len(),
        2
    );
    dataset.validate().await.unwrap();
}
#[rstest]
#[case::independent(vec![vec![0, 1], vec![2, 3]], 4)]
#[case::crossing(vec![vec![1, 2]], 3)]
#[case::transitive(vec![vec![1, 2], vec![3, 4]], 2)]
#[tokio::test]
async fn test_compaction_merges_only_connected_segments(
    #[case] groups: Vec<Vec<usize>>,
    #[case] expected_segments: usize,
) {
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
        )
        .into_ram_dataset(FragmentCount::from(8), FragmentRowCount::from(8))
        .await
        .unwrap();
    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 4]), 4).unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let fragments = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.metadata.clone())
        .collect::<Vec<_>>();
    let mut segments = Vec::new();
    for pair in fragments.chunks(2) {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(pair.iter().map(|fragment| fragment.id as u32).collect())
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let mut completed = Vec::new();
    for group in groups {
        let task = TaskData {
            fragments: group.iter().map(|idx| fragments[*idx].clone()).collect(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        };
        completed.push(
            rewrite_files(Cow::Borrowed(&dataset), task, &options)
                .await
                .unwrap(),
        );
    }
    commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        expected_segments
    );
    assert_vector_index_coverage(&dataset).await;
}
#[rstest]
#[case::flat_one(false, 1, None, false)]
#[case::flat_multi(false, 2, None, false)]
#[case::flat_deletions(false, 2, Some("id % 3 = 0"), false)]
#[case::flat_deleted_source(false, 2, Some("id < 128"), false)]
#[case::sq_one(true, 1, None, false)]
#[case::sq_multi(true, 2, Some("id % 3 = 0"), false)]
#[case::flat_nulls(false, 2, None, true)]
#[case::sq_nulls(true, 2, None, true)]
#[tokio::test]
async fn test_compaction_merges_compatible_vector_segments(
    #[case] is_sq: bool,
    #[case] partitions: usize,
    #[case] deletion: Option<&str>,
    #[case] has_nulls: bool,
) {
    const DIM: u32 = 4;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)).with_nulls(
                if has_nulls {
                    &[false, false, true]
                } else {
                    &[false]
                },
            ),
        )
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(128))
        .await
        .unwrap();

    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from(
            (0..partitions)
                .flat_map(|i| std::iter::repeat_n(i as f32, DIM as usize))
                .collect::<Vec<_>>(),
        ),
        DIM as i32,
    )
    .unwrap();
    let ivf = IvfBuildParams::try_with_centroids(partitions, Arc::new(centroids)).unwrap();
    let params = if is_sq {
        VectorIndexParams::with_ivf_sq_params(DistanceType::L2, ivf, SQBuildParams::default())
    } else {
        VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf)
    };
    // Copy one stored model to ensure exact SQ compatibility. Each source
    // deliberately contains stale rows owned by the other two segments.
    let template = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
        .name("vec_idx".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    let mut segments = Vec::new();
    for fragment in dataset.get_fragments() {
        let mut segment = template.clone();
        segment.uuid = Uuid::new_v4();
        segment.fragment_bitmap = Some(RoaringBitmap::from_iter([fragment.id() as u32]));
        for file in ["index.idx", "auxiliary.idx"] {
            dataset
                .object_store
                .copy(
                    &dataset
                        .indices_dir()
                        .join(template.uuid.to_string())
                        .join(file),
                    &dataset
                        .indices_dir()
                        .join(segment.uuid.to_string())
                        .join(file),
                )
                .await
                .unwrap();
        }
        segments.push(segment);
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        3
    );

    if let Some(predicate) = deletion {
        dataset.delete(predicate).await.unwrap();
    }
    let query = vec![0.5; DIM as usize];
    let before = vector_knn_ids(&dataset, &query, 10).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let conservative = plan_compaction(&dataset, &options).await.unwrap();
    assert!(
        conservative
            .tasks()
            .iter()
            .all(|task| task.fragments.len() == 1)
    );
    let plan = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(&dataset)
        .await
        .unwrap();
    assert_eq!(plan.num_tasks(), 1);
    assert_eq!(
        plan.tasks()[0].fragments.len(),
        dataset.get_fragments().len()
    );

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);
    let indices = dataset.load_indices_by_name("vec_idx").await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0].fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([dataset.get_fragments()[0].id() as u32])
    );
    dataset.validate().await.unwrap();

    let after = vector_knn_ids(&dataset, &query, 10).await;
    let query_array = Float32Array::from(query);
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &query_array, 10).unwrap();
    scanner.nprobes(partitions).project(&["id"]).unwrap();
    let execution = scanner.explain_plan(false).await.unwrap();
    assert!(execution.contains("ANNSubIndex"), "{execution}");
    assert!(!execution.contains("LanceScan"), "{execution}");
    scanner.use_index(false);
    let truth = scanner.try_into_batch().await.unwrap();
    let truth = truth["id"].as_primitive::<Int32Type>().values();
    if is_sq {
        assert!(after.iter().filter(|id| truth.contains(id)).count() >= 5);
    } else {
        assert_eq!(after, before);
        assert_eq!(after.as_slice(), truth.as_ref());
    }
    assert_vector_index_coverage(&dataset).await;
}
#[tokio::test]
async fn test_stale_vector_segment_merge_requires_replan() {
    const DIM: u32 = 4;
    let test_dir = TempStrDir::default();
    let reader = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_reader_rows(RowCount::from(64), BatchCount::from(4));
    let mut dataset = Dataset::write(
        reader,
        test_dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; DIM as usize]), 4)
            .unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let template = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
        .name("vec_idx".to_string())
        .execute_uncommitted()
        .await
        .unwrap();

    let fragments = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.metadata.clone())
        .collect::<Vec<_>>();
    let mut segments = Vec::new();
    for fragment_ids in [[0, 1], [2, 3]] {
        let mut segment = template.clone();
        segment.uuid = Uuid::new_v4();
        segment.fragment_bitmap = Some(RoaringBitmap::from_iter(fragment_ids));
        for file in template.files.as_ref().unwrap() {
            dataset
                .object_store
                .copy(
                    &dataset
                        .indices_dir()
                        .join(template.uuid.to_string())
                        .join(file.path.as_str()),
                    &dataset
                        .indices_dir()
                        .join(segment.uuid.to_string())
                        .join(file.path.as_str()),
                )
                .await
                .unwrap();
        }
        segments.push(segment);
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();

    let stale = dataset.clone();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let stale_merged_result = rewrite_files(
        Cow::Borrowed(&stale),
        TaskData {
            fragments: vec![fragments[1].clone(), fragments[2].clone()],
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        &options,
    )
    .await
    .unwrap();
    let disjoint_result = rewrite_files(
        Cow::Borrowed(&stale),
        TaskData {
            fragments: vec![fragments[0].clone()],
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        &options,
    )
    .await
    .unwrap();

    let mut concurrent = dataset.clone();
    commit_compaction(
        &mut concurrent,
        vec![disjoint_result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    let after_concurrent = Dataset::open(test_dir.as_str()).await.unwrap();
    let expected_fragments = after_concurrent.manifest.fragments.clone();
    let expected_indices = load_all_indices(&after_concurrent).await.unwrap();

    let mut stale = stale;
    let error = commit_compaction(
        &mut stale,
        vec![stale_merged_result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("a stale merged segment replacement must not be retried unchanged");
    assert!(matches!(error, Error::IncompatibleTransaction { .. }));
    assert!(
        error
            .to_string()
            .contains("replan compaction from the latest dataset"),
        "unexpected error: {error}"
    );

    let latest = Dataset::open(test_dir.as_str()).await.unwrap();
    assert_eq!(latest.manifest.fragments, expected_fragments);
    assert_eq!(load_all_indices(&latest).await.unwrap(), expected_indices);

    let mut latest = latest;
    compact_files(&mut latest, options, None).await.unwrap();
    assert_eq!(latest.get_fragments().len(), 1);
    latest.validate().await.unwrap();
    assert_vector_index_coverage(&latest).await;
}
#[rstest]
#[case::stable_row_ids(true, false, false)]
#[case::deferred_remap(false, true, false)]
#[case::immediate_remap(false, false, false)]
#[case::different_versions(false, false, true)]
#[tokio::test]
async fn test_compaction_preserves_vector_boundaries_without_immediate_remap(
    #[case] enable_stable_row_ids: bool,
    #[case] defer_index_remap: bool,
    #[case] has_different_versions: bool,
) {
    const DIM: u32 = 16;
    let test_dir = TempStrDir::default();
    let reader = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_reader_rows(RowCount::from(128), BatchCount::from(1));
    let mut dataset = Dataset::write(
        reader,
        test_dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 64,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(std::iter::repeat_n(0.0, DIM as usize)),
        DIM as i32,
    )
    .unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let mut segments = Vec::new();
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    for fragment_id in fragment_ids {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    if has_different_versions {
        assert_ne!(segments[0].index_version, 0);
        segments[0].index_version = 0;
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap,
        ..Default::default()
    };
    assert_eq!(
        plan_compaction(&dataset, &options)
            .await
            .unwrap()
            .num_tasks(),
        0
    );
    let plan = DefaultCompactionPlanner::new(options)
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(&dataset)
        .await
        .unwrap();
    assert_eq!(
        plan.num_tasks(),
        usize::from(!enable_stable_row_ids && !defer_index_remap && !has_different_versions)
    );
    if plan.num_tasks() == 1 {
        assert_eq!(plan.tasks[0].fragments.len(), 2);
    }
}
#[tokio::test]
async fn test_commit_rejects_vector_boundary_crossing_when_remap_mode_changes() {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(64))
        .await
        .unwrap();

    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(std::iter::repeat_n(0.0, DIM as usize)),
        DIM as i32,
    )
    .unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut segments = Vec::new();
    for fragment_id in fragment_ids {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();

    let plan_options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let completed_tasks = execute_compaction_plan(&dataset, &plan_options).await;
    assert_eq!(completed_tasks.len(), 1);
    assert_eq!(completed_tasks[0].original_fragments.len(), 2);

    let deferred_options = CompactionOptions {
        defer_index_remap: true,
        ..plan_options.clone()
    };
    let version_before_commit = dataset.manifest.version;
    let error = commit_compaction(
        &mut dataset,
        completed_tasks.clone(),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &deferred_options,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("physical coverage boundary"));
    assert_eq!(dataset.manifest.version, version_before_commit);

    commit_compaction(
        &mut dataset,
        completed_tasks,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan_options,
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        1
    );
    dataset.validate().await.unwrap();
}
#[tokio::test]
async fn test_compaction_preserves_pq_segment_boundaries() {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(64))
        .await
        .unwrap();

    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(std::iter::repeat_n(0.0, DIM as usize)),
        DIM as i32,
    )
    .unwrap();
    let codebook = Arc::new(Float32Array::from_iter_values(std::iter::repeat_n(
        0.0,
        DIM as usize * 16,
    )));
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
        PQBuildParams::with_codebook(4, 4, codebook),
    );
    let mut segments = Vec::new();
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    for fragment_id in fragment_ids {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();

    let plan = plan_compaction(
        &dataset,
        &CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(plan.num_tasks(), 0);
}
#[tokio::test]
async fn test_compaction_merges_compatible_vector_segment_subset() {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(64))
        .await
        .unwrap();

    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(fragment_ids.len());
    for (fragment_id, centroid) in fragment_ids.into_iter().zip([0.0, 0.0, 1.0]) {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values(std::iter::repeat_n(centroid, DIM as usize)),
            DIM as i32,
        )
        .unwrap();
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
        );
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("vec_idx", "vec", segments)
        .await
        .unwrap();

    let query = vec![0.5; DIM as usize];
    let before = vector_knn_ids(&dataset, &query, 10).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let plan = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(&dataset)
        .await
        .unwrap();
    assert_eq!(plan.num_tasks(), 1);
    assert_eq!(plan.tasks()[0].fragments.len(), 2);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        2
    );
    dataset.validate().await.unwrap();
    let after = vector_knn_ids(&dataset, &query, 10).await;
    assert_eq!(after, before);
    assert_vector_index_coverage(&dataset).await;
}
#[tokio::test]
async fn test_compaction_keeps_incompatible_vector_segments_separate() {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(64))
        .await
        .unwrap();

    let params = |centroid: f32| {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values(std::iter::repeat_n(centroid, DIM as usize)),
            DIM as i32,
        )
        .unwrap();
        VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
        )
    };
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let first = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params(0.0))
        .name("vec_idx".to_string())
        .fragments(fragment_ids[..2].to_vec())
        .execute_uncommitted()
        .await
        .unwrap();
    let second = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params(1.0))
        .name("vec_idx".to_string())
        .fragments(fragment_ids[2..].to_vec())
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("vec_idx", "vec", vec![first, second])
        .await
        .unwrap();

    let query = vec![0.5; DIM as usize];
    let before = vector_knn_ids(&dataset, &query, 10).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.num_tasks(), 2);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.get_fragments().len(), 2);
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        2
    );
    dataset.validate().await.unwrap();
    let after = vector_knn_ids(&dataset, &query, 10).await;
    assert_eq!(after, before);
    assert_vector_index_coverage(&dataset).await;
}
#[rstest]
#[case::flat_same(false, false)]
#[case::flat_different(false, true)]
#[case::sq_same(true, false)]
#[tokio::test]
async fn test_compaction_keeps_hnsw_parameter_boundaries(
    #[case] is_sq: bool,
    #[case] different_params: bool,
) {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(64))
        .await
        .unwrap();

    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(std::iter::repeat_n(0.0, DIM as usize)),
        DIM as i32,
    )
    .unwrap();
    let ivf_params = IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap();
    let params = |num_edges| {
        let hnsw = HnswBuildParams::default().num_edges(num_edges);
        if is_sq {
            VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::L2,
                ivf_params.clone(),
                hnsw,
                SQBuildParams::default(),
            )
        } else {
            VectorIndexParams::ivf_hnsw(DistanceType::L2, ivf_params.clone(), hnsw)
        }
    };
    let default_params = params(8);
    let custom_params = params(if different_params { 16 } else { 8 });
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let first = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &default_params)
        .name("vec_idx".to_string())
        .fragments(fragment_ids[..2].to_vec())
        .execute_uncommitted()
        .await
        .unwrap();
    let second = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &custom_params)
        .name("vec_idx".to_string())
        .fragments(fragment_ids[2..].to_vec())
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("vec_idx", "vec", vec![first, second])
        .await
        .unwrap();

    dataset.delete("id % 7 = 0").await.unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let plan = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(&dataset)
        .await
        .unwrap();
    assert_eq!(plan.num_tasks(), 2);
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    assert_eq!(
        dataset.load_indices_by_name("vec_idx").await.unwrap().len(),
        2
    );
    assert_vector_index_coverage(&dataset).await;
    let query = Float32Array::from(vec![0.5; DIM as usize]);
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &query, 10).unwrap();
    scanner.project(&["id"]).unwrap();
    let execution = scanner.explain_plan(false).await.unwrap();
    assert!(execution.contains("ANNSubIndex"), "{execution}");
    assert!(!execution.contains("LanceScan"), "{execution}");
    let ann = scanner.try_into_batch().await.unwrap();
    scanner.use_index(false);
    let truth = scanner.try_into_batch().await.unwrap();
    let truth = truth["id"].as_primitive::<Int32Type>().values();
    assert!(
        ann["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .filter(|id| truth.contains(id))
            .count()
            >= 5
    );
}
#[tokio::test]
async fn test_read_ivf_rq_index_v3_with_defer_index_remap() {
    use arrow_array::cast::AsArray;
    use lance_index::vector::bq::RQBuildParams;

    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    let stored: Vec<Vec<f32>> = {
        let mut scanner = dataset.scan();
        scanner.project(&["vec"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let vecs = batch["vec"].as_fixed_size_list();
            for i in 0..batch.num_rows() {
                let values = vecs.value(i);
                let values = values.as_primitive::<Float32Type>();
                out.push(values.values().to_vec());
            }
        }
        out
    };

    let index_name = Some("vec_idx".into());
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            index_name.clone(),
            &VectorIndexParams {
                metric_type: DistanceType::L2,
                stages: vec![
                    StageParams::Ivf(IvfBuildParams {
                        max_iters: 2,
                        num_partitions: Some(2),
                        sample_rate: 2,
                        ..Default::default()
                    }),
                    StageParams::RQ(RQBuildParams::new(1)),
                ],
                version: crate::index::vector::IndexFileVersion::V3,
                skip_transpose: false,
                runtime_hints: Default::default(),
            },
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
        panic!("vec index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    let frag_reuse_present = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
    assert!(
        frag_reuse_present,
        "defer_index_remap must record a {} index",
        FRAG_REUSE_INDEX_NAME
    );

    let sample_step = (stored.len() / 8).max(1);
    let mut checked = 0;
    for query in stored.iter().step_by(sample_step) {
        let query_vec = PrimitiveArray::<Float32Type>::from_iter_values(query.iter().copied());
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec, 5).unwrap();
        scanner.project(&["vec"]).unwrap().with_row_id();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(!batches.is_empty(), "query returned no batches");
        let top = &batches[0];
        assert!(top.num_rows() > 0, "query returned empty top batch");
        let top_vec = top["vec"].as_fixed_size_list().value(0);
        let top_vec = top_vec.as_primitive::<Float32Type>();
        assert_eq!(
            top_vec.values(),
            query.as_slice(),
            "top-1 self-recall returned a different vector than the query"
        );
        checked += 1;
    }
    assert!(checked > 0, "expected to check at least one stored vector");
}
#[tokio::test]
async fn test_ivf_flat_defer_compaction_with_deletions() {
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, small_ivf());
    // Flat storage is scanned linearly; dropping deleted rows is exact.
    // Boxed for CI clippy `large_futures`: the compaction future grew past 16 KiB.
    Box::pin(check_vector_defer_compaction(
        params,
        Some("id < 1500"),
        10,
        10,
    ))
    .await;
}
#[tokio::test]
async fn test_ivf_hnsw_sq_defer_compaction_merge_only() {
    use lance_index::vector::{hnsw::builder::HnswBuildParams, sq::builder::SQBuildParams};
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        small_ivf(),
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    // No deletions: storage positions are stable, so the graph stays aligned.
    // Boxed for CI clippy `large_futures`: the compaction future grew past 16 KiB.
    Box::pin(check_vector_defer_compaction(params, None, 10, 9)).await;
}
#[tokio::test]
async fn test_ivf_pq_defer_compaction_with_deletions() {
    use lance_index::vector::pq::PQBuildParams;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        small_ivf(),
        PQBuildParams {
            max_iters: 2,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    // Boxed for CI clippy `large_futures`: the compaction future grew past 16 KiB.
    Box::pin(check_vector_defer_compaction(
        params,
        Some("id < 1500"),
        10,
        8,
    ))
    .await;
}
#[tokio::test]
async fn test_ivf_sq_defer_compaction_with_deletions() {
    use lance_index::vector::sq::builder::SQBuildParams;
    let params = VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        small_ivf(),
        SQBuildParams::default(),
    );
    // Boxed for CI clippy `large_futures`: the compaction future grew past 16 KiB.
    Box::pin(check_vector_defer_compaction(
        params,
        Some("id < 1500"),
        10,
        8,
    ))
    .await;
}
#[tokio::test]
async fn test_ivf_rq_defer_compaction_with_deletions() {
    use lance_index::vector::bq::RQBuildParams;
    let params =
        VectorIndexParams::with_ivf_rq_params(DistanceType::L2, small_ivf(), RQBuildParams::new(1));
    // Boxed for CI clippy `large_futures`: the compaction future grew past 16 KiB.
    Box::pin(check_vector_defer_compaction(
        params,
        Some("id < 1500"),
        10,
        8,
    ))
    .await;
}
#[tokio::test]
async fn test_ivf_flat_remap_and_trim() {
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, small_ivf());
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 8, Some(8))).await;
}
/// Decoded vector partitions embed the address mappings for history with
/// `index_version = 0`, so changing that history under the same UUID must
/// invalidate them. Cover both vector index load paths.
#[rstest]
#[case::ivf_flat(false)]
#[case::legacy_ivf_pq(true)]
#[tokio::test]
async fn test_vector_caches_follow_a_history_that_keeps_its_uuid(#[case] legacy: bool) {
    use lance_index::frag_reuse::{FragDigest, FragReuseIndexDetails, FragReuseVersion};
    use lance_table::format::pb::FragmentReuseIndexDetails as DetailsProto;
    use lance_table::format::pb::fragment_reuse_index_details::Content;
    use prost::Message;

    const K: usize = 20;
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(8)),
        )
        .into_dataset(uri, FragmentCount::from(6), FragmentRowCount::from(64))
        .await
        .unwrap();
    let params = if legacy {
        VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            small_ivf(),
            PQBuildParams {
                max_iters: 2,
                num_sub_vectors: 2,
                ..Default::default()
            },
        )
        .version(crate::index::vector::IndexFileVersion::Legacy)
        .clone()
    } else {
        VectorIndexParams::with_ivf_flat_params(DistanceType::L2, small_ivf())
    };
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..deferred_compaction_excluding(&[0, 1, 2, 3])
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    let batch = dataset
        .scan()
        .project(&["vec"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch["vec"].as_fixed_size_list();
    let queries = (0..batch.num_rows())
        .step_by(37)
        .map(|row| {
            vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    for query in &queries {
        // Warms the partitions, decoded through the history.
        exact_knn_ids(&dataset, query, K, true).await;
    }

    // A raw rewrite of the remaining suffix that keeps the history's UUID.
    let current = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .cloned()
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..deferred_compaction_excluding(&[0, 1])
    };
    let task = plan_one_task(&dataset, &options).await;
    let mut new_fragments = task.new_fragments.clone();
    let reservation = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: new_fragments.len() as u32,
        },
        None,
    );
    dataset
        .apply_commit(reservation, &Default::default(), &Default::default())
        .await
        .unwrap();
    let first_id =
        dataset.manifest.max_fragment_id.unwrap() as u64 + 1 - new_fragments.len() as u64;
    for (fragment, id) in new_fragments.iter_mut().zip(first_id..) {
        fragment.id = id;
    }
    let mut versions = load_frag_reuse_index_details(&dataset, &current)
        .await
        .unwrap()
        .versions
        .clone();
    versions.push(FragReuseVersion {
        dataset_version: dataset.manifest.version,
        groups: vec![FragReuseGroup {
            changed_row_addrs: task.row_addrs.clone().unwrap(),
            old_frags: task
                .original_fragments
                .iter()
                .map(FragDigest::from)
                .collect(),
            new_frags: new_fragments.iter().map(FragDigest::from).collect(),
        }],
    });
    let details = FragReuseIndexDetails { versions };
    let proto = DetailsProto {
        content: Some(Content::Inline((&details).into())),
    };
    let mut entry = current.clone();
    entry.index_details = Some(Arc::new(prost_types::Any {
        type_url: current.index_details.as_ref().unwrap().type_url.clone(),
        value: proto.encode_to_vec(),
    }));
    entry.fragment_bitmap = Some(details.new_frag_bitmap());
    entry.dataset_version = dataset.manifest.version;
    let rewrite = Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: task.original_fragments,
                new_fragments,
            }],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        },
        None,
    );
    let published = crate::dataset::CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(rewrite)
        .await
        .unwrap();
    let version = published.manifest.version;
    let kept = published.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(kept.uuid, current.uuid);

    let open = async |session: Option<Arc<crate::session::Session>>| {
        let mut builder =
            crate::dataset::builder::DatasetBuilder::from_uri(uri).with_version(version);
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        builder.load().await.unwrap()
    };
    let later = open(Some(dataset.session())).await;
    let cold = open(None).await;
    let mut expected = Vec::new();
    for query in &queries {
        let found = exact_knn_ids(&cold, query, K, true).await;
        if !legacy {
            assert_eq!(found, exact_knn_ids(&cold, query, K, false).await);
        }
        expected.push(found);
    }
    for handle in [&published, &later] {
        assert_eq!(
            handle
                .frag_reuse_index()
                .await
                .unwrap()
                .unwrap()
                .details
                .versions
                .len(),
            2
        );
        for (query, expected) in queries.iter().zip(&expected) {
            assert_eq!(&exact_knn_ids(handle, query, K, true).await, expected);
        }
    }
} // Regression: PQ storage used to remap its codes through the frag-reuse
// index but keep the pre-remap `row_ids` field, so search returned stale
// (compacted-away) addresses and the take failed with "fragment ... does
// not exist" — even merge-only, and only observable when the query fetches
// row content (the existing `test_read_ivf_pq_index_v3_with_defer_index_remap`
// projects no columns, so it never takes and missed this).
#[tokio::test]
async fn test_ivf_pq_remap_and_trim() {
    use lance_index::vector::pq::PQBuildParams;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        small_ivf(),
        PQBuildParams {
            max_iters: 2,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 8, Some(8))).await;
}
/// Details a remap must carry over: a metric other than the default and a
/// partition size the build was given.
#[tokio::test]
async fn test_ivf_pq_cosine_with_target_partition_size_remap_and_trim() {
    use lance_index::vector::pq::PQBuildParams;
    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::Cosine,
        lance_index::vector::ivf::IvfBuildParams {
            num_partitions: None,
            target_partition_size: Some(128),
            ..small_ivf()
        },
        PQBuildParams {
            max_iters: 2,
            num_sub_vectors: 2,
            ..Default::default()
        },
    );
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 8, Some(8))).await;
}
#[tokio::test]
async fn test_ivf_sq_remap_and_trim() {
    use lance_index::vector::sq::builder::SQBuildParams;
    let params = VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        small_ivf(),
        SQBuildParams::default(),
    );
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 8, Some(8))).await;
}
#[tokio::test]
async fn test_ivf_rq_remap_and_trim() {
    use lance_index::vector::bq::RQBuildParams;
    let params =
        VectorIndexParams::with_ivf_rq_params(DistanceType::L2, small_ivf(), RQBuildParams::new(1));
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 8, Some(8))).await;
}
#[tokio::test]
async fn test_ivf_hnsw_sq_remap_and_trim() {
    use lance_index::vector::{hnsw::builder::HnswBuildParams, sq::builder::SQBuildParams};
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        small_ivf(),
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    // Physical remap rebuilds the HNSW graph, so use a recall-tolerant overlap.
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 7, None)).await;
}
#[tokio::test]
async fn test_ivf_hnsw_pq_remap_and_trim() {
    use lance_index::vector::{hnsw::builder::HnswBuildParams, pq::PQBuildParams};
    let params = VectorIndexParams::with_ivf_hnsw_pq_params(
        DistanceType::L2,
        small_ivf(),
        HnswBuildParams::default()
            .max_level(2)
            .num_edges(4)
            .ef_construction(16),
        PQBuildParams {
            max_iters: 2,
            num_sub_vectors: 2,
            num_bits: 4,
            sample_rate: 2,
            ..Default::default()
        },
    );
    // Boxed for CI clippy `large_futures`: the remap future grew past 16 KiB.
    Box::pin(check_vector_remap_and_trim(params, 10, 7, None)).await;
}
